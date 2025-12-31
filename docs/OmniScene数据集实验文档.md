# OmniScene 数据集实验计划

本说明文档基于 `depthsplat/docs/OmniScene数据集实验文档.md` 与其对应的数据加载实现整理，阐述在 **DropGaussian（逐场景优化 3DGS）** 中适配 OmniScene（nuScenes 派生）数据集的整体方案。重点描述与 `depthsplat` 前馈式高斯重建的差异、需要新增的代码目录以及数据/流程规划。

---

## 1. DepthSplat 现有实现回顾
- **配置入口**：`config/dataset/omniscene.yaml` 注册 `dataset.name=omniscene`，Hydra 通过 `src/dataset/__init__.py` 将其映射到 `DatasetOmniScene`。
- **数据划分**：`DatasetOmniScene` 根据 `stage` 读取 `bins_*_3.2m.json`。`val` 模式取前 30000 个 bin 中每隔 3000 个的 10 个样本；`test` 使用 `bins_val` 的 mini-test 子集；`demo` 使用 `bins_dynamic_demo`。
- **样本结构**：`context`（6 个环视 key-frame 视角）+ `target`（12 个非关键帧 + 6 个关键帧，共 18 个），均包含 RGB、内参、`c2w`、`near/far`、索引，`target` 额外有动态掩码。
- **图像与内参处理**：`load_conditions` 读取 `samples(_small)`、`samples_param_small` 等文件，resize 到 112×200 或 224×400，并对内参做尺度归一化；输出视图会加载 `samples_mask_small` 作为掩码。
- **主流程调用**：DepthSplat 的 `DataModule` 随机取 batch 传给前馈模型，训练/推理均在单阶段完成。模型内部生成射线、渲染结果并与 `target` 对比。

---

## 2. 与 DropGaussian（逐场景优化）的关键差异
| 维度 | DepthSplat（前馈） | DropGaussian（逐场景优化） |
| --- | --- | --- |
| 数据粒度 | 单次运行遍历大量 bin，随机组 batch | 每次执行针对单个场景路径（LLFF/MipNeRF/Blender 等） |
| 加载接口 | `DataModule` 返回 `context/target` 张量 | `scene.Scene` 预期读取磁盘上 COLMAP/Blender 结构（`images/`+`transforms_*.json` 或 `sparse/0`） |
| 初始点云 | 模型推理生成（不依赖外部 PLY） | 需要输入点云（Colmap PLY 或随机初始化）作为优化起点 |
| 运行脚本 | `python -m src.main` 一次性处理 train/val/test | shell/python 脚本逐场景循环调用 `train.py` + `render.py` |

因此，OmniScene 在本仓库中需要先转换成 **逐场景的 Blender 格式**，再由 `train.py` 与 `render.py` 逐场景运行，并通过脚本整合集体评估。

---

## 3. 新增目录与文件规划
1. **`comp_svfgs/`**：存放与 OmniScene 适配相关的 Python 工具。计划包含：
   - `dataset_omniscene.py`：负责解析原始 `bins_*_3.2m.json`、`bin_infos_3.2m/*.pkl`，并提供 `load_conditions` 等工具（直接复用 depthsplat 的路径/掩码逻辑）。
   - `prepare_omniscene.py`（后续实现）：利用上述 loader，在一次运行中完成数据预处理与磁盘重组。
2. **`output/`**：集中保存：
   - 预处理生成的逐场景数据（例如 `output/omniscene_prepared/01_sceneXXXX/`）。
   - `train.py` 与 `render.py` 运行结果（沿用现有 `args.model_path`，推荐放在 `output/omniscene_experiments/` 下）。
3. **`docs/OmniScene数据集实验文档.md`**（本文档）：描述方案与差异。

---

## 4. 数据预处理与场景组织
### 4.1 数据加载模块 (`comp_svfgs/dataset_omniscene.py`)
- **API 设计**：提供 `OmniSceneBinLoader(mode="val")`，默认 `val`，支持 `train/test/demo` four modes 以保持与 depthsplat 一致。
- **取样策略**：
  - `val`：读取 `bins_val_3.2m.json` 并执行 `self.bin_tokens = bins[:30000:3000][:10]`，得到 10 个 bin，每个 bin 在 DropGaussian 中视作单独场景。
  - `train`：可完整遍历全量训练 bin，后续如需扩展可复用。
  - `test`：保持 depthsplat 中 `0::14` 的 mini-test 抽样。
  - `demo`：沿用 `bins_dynamic_demo`。
- **返回数据**：与 depthsplat 相同的 `context/target` 结构，`context` 6 视角、`target` 18 视角。`load_conditions` 直接复制 depthsplat 版本，包括路径替换与掩码读取。
- **绝对尺度深度与置信度**：参考 `~/Projects/SVF-GS/data/transforms/loading.py` 将 `Metric3D-v2` 生成的 `_dpt.npy`（绝对深度）与 `_conf.npy`（置信度）加载到 `OmniSceneView` 中。读取流程与 SVF-GS 保持一致：  
  1. 将 `samples_small/*.jpg`（或 `sweeps_small/*.jpg`）映射到 `samples_dptm_small/*_dpt.npy`、`samples_dptm_small/*_conf.npy`；  
  2. 在图像重采样时同步双线性缩放深度与置信度；  
  3. 将置信度 > 0.3 的像素记为有效掩码 `depth_mask`，用于后续点云构建；  
  4. 同时保留可选的相对深度（来自 `_dpt.npy`）以兼容 evaluate 需求。  
  数据集对外暴露 `view.depth_metric`（米）、`view.depth_confidence`（0~1）和 `view.depth_valid_mask`（bool）字段，以便 `prepare_scene_directory` 直接消费。

### 4.2 逐场景格式转换（Blender 风格）
- **目的**：让 `scene.Scene` 走 `readNerfSyntheticInfo` 分支，不依赖 COLMAP。
- **转换步骤**（单个 bin）：
  1. 使用 loader 读取 `context/target`。
  2. 将 6 张输入图像保存到 `images/train/`，18 张评估图像保存到 `images/test/`。路径命名统一为 `000.png` 开始的 3 位编号。
  3. 组装 `transforms_train.json`、`transforms_test.json`：仿照 Blender 数据结构写入 `file_path`（相对路径）、`transform_matrix`（使用 `c2w`）、`camera_angle_x`（可由内参 fx/宽换算），并在 `frames` 字段中列出所有视图。
  4. **深度驱动的点云生成**：不再使用随机点。基于每张输入图像的绝对深度与置信度，执行以下流程得到真实尺度点云：  
     - 对每个像素 `(u,v)`，若 `depth_confidence(u,v) > 0.3`，将度量深度 `d` 配合未归一化的内参 `(fx, fy, cx, cy)` 反投影为相机坐标 `(x = (u-cx)*d/fx, y = (v-cy)*d/fy, z=d)`；  
     - 使用对应视图的 `c2w` 将坐标转换到世界坐标系，并将 RGB 取自图像同一像素；  
     - 若需要给点云附带掩码，可直接重用 `depth_valid_mask`，只写入有效点；  
     - 将结果保存为 `points3d.ply`，格式与 `scene.dataset_readers.fetchPly` 兼容，从而用真实场景几何初始化高斯。  
     同时保留 `--rand_pcd` 选项以便在深度文件缺失时退化为随机点。
  5. 将 `near/far`（若设置）记录在额外的元数据 JSON 中，以供尝试不同渲染范围。
- **场景命名**：`output/omniscene_prepared/01_<bin_token>/`、`02_<bin_token>/` … 其中 `01_` 前缀对应排序编号，满足“命名由 `01_` + bin_token 拼接”的要求。
- **缓存策略**：预处理流程会在运行时判断目标场景目录是否已存在。若已存在则直接复用；若不存在则即时生成并保存，确保单次脚本即可完成“检测→补齐→训练”闭环。

---

## 5. 训练 / 渲染 / 评估脚本设计
- 在 `scripts/` 下新增 `run_omniscene.py`（使用 Python，便于管理流程和日志），整体 **单阶段** 完成“场景预检查→必要的预处理→训练→渲染→评估”。流程：
  1. 调用 `comp_svfgs.dataset_omniscene` 获取指定模式的 bin 列表；遍历每个 bin 时先检查 `output/omniscene_prepared/XX_<token>/` 是否存在，若不存在则即时执行预处理并写入磁盘，若存在则直接使用。
  2. 对于成功准备好的场景，立即执行：
     - `python train.py -s <scene_dir> -m output/omniscene_experiments/<scene_name> --eval -r 1 --n_views 6`；由于 `points3d.ply` 已包含绝对尺度点云，默认 **不再加 `--rand_pcd`**，若检测到深度文件缺失可通过 `scripts/run_omniscene.py --force-rand-pcd` 手动退化使用随机点云。
       - `-r 1`：保持 112×200 的低分辨率；若用户切换至 224×400，可在脚本参数中传入。
       - `--n_views 6`：训练集即 6 张输入。
     - `python render.py -m <model_path> --eval -r 1`：使用与训练阶段一致的配置，生成 `metrics_*.txt` 与渲染图像。
  3. 循环结束后，脚本自动调用 `python metric.py --path output/omniscene_experiments` 汇总指标。
- 由于流程已覆盖预处理与训练，**不再设计拆分阶段或 `--only-prepare` 等选项**；脚本会在内部自动处理“已有缓存则跳过生成、否则即时生成”的逻辑，确保一次命令即可完成完整实验。
- CLI 选项（全程单阶段）：
  - `--omniscene-root`：原始数据根目录。
  - `--mode`：`val/train/test/demo`（默认 `val`）。
  - `--resolution`：`112x200` 或 `224x400`（默认 112x200）。
  - `--iterations`：可选覆盖 `OptimizationParams.iterations`。

---

## 6. 参数与运行策略
- **分辨率**：默认 112×200，可通过脚本参数提升至 224×400；预处理会在首次运行时根据选择的分辨率输出图像，后续继续沿用同一尺寸。
- **n_views**：固定为 6，意味着 `Scene.getTrainCameras()` 只包含 6 张输入。
- **迭代次数**：沿用 `OptimizationParams.iterations=10000`（可通过 CLI 覆盖）。由于数据分辨率低，可增加 `test_iterations` 频率便于监控。
- **单阶段运行要求**：所有操作必须在同一脚本执行周期内完成。对于每个场景，流程为“检查缓存 → 若无则预处理并保存 → 训练/优化 → 渲染与指标评估”，不提供拆分运行模式。

---

## 7. 后续实现要点
1. **编写 `comp_svfgs/dataset_omniscene.py`**：直接引用 depthsplat 的 `load_conditions`、`load_info` 路径处理逻辑，确保图像/掩码和相机参数一致。
2. **实现预处理脚本**：读取 loader 输出，生成 Blender 结构与随机点云，结果写入 `output/omniscene_prepared/XX_<token>/`。
3. **新增运行脚本**：遍历场景并调用现有 `train.py`/`render.py`/`metric.py`，同时在 log 中记录每个阶段的命令行参数。
4. **文档 & README 补充**：在主 README 的“Training”部分添加 OmniScene 入口说明，或在 docs 中保持更新。

通过以上规划，可在不破坏现有 3DGS 训练流程的前提下，将 OmniScene 数据集纳入逐场景优化实验，并与用户自研的 SVF-GS 方法进行统一对比。后续实现阶段将严格依照本文档完成目录创建、代码迁移及脚本接入。
