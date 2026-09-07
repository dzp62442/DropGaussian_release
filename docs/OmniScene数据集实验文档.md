# OmniScene 数据集实验说明

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
- **API 设计**：提供 `OmniSceneDataset(mode="val")`，默认 `val`，支持 `train/val/test/demo/center150` 五种模式。
- **取样策略**：
  - `val`：读取 `bins_val_3.2m.json` 并执行 `self.bin_tokens = bins[:30000:3000][:10]`，得到 10 个 bin，每个 bin 在 DropGaussian 中视作单独场景。
  - `train`：可完整遍历全量训练 bin，后续如需扩展可复用。
  - `test`：保持 depthsplat 中 `0::14` 的 mini-test 抽样。
  - `demo`：沿用 `bins_dynamic_demo`。
  - `center150`：只负责读取由 SVF-GS 主项目统一生成的 `bins_center150_v1.json`，并保持其中的样本顺序；该清单从 nuScenes 官方 val 的 150 个场景中各选一个 lower median 中央 bin。本项目不提供生成或修改清单的功能。
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
  3. 组装 `transforms_train.json`、`transforms_test.json`：仿照 Blender 数据结构写入 `file_path`（相对路径）、`transform_matrix`（将 OpenCV `c2w` 右乘 `diag(1,-1,-1,1)` 转成 Blender/OpenGL 相机轴）、`camera_angle_x`（可由内参 fx/宽换算），并在 `frames` 字段中列出所有视图。
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

## 5. 训练 / 渲染 / 评估流程
- `scripts/run_omniscene.py` 整体 **单阶段** 完成“场景预检查→必要的预处理→训练→渲染→评估”。流程：
  1. 调用 `comp_svfgs.dataset_omniscene` 获取指定模式的 bin 列表；遍历每个 bin 时先检查 `output/omniscene_prepared/XX_<token>/` 是否存在，若不存在则即时执行预处理并写入磁盘，若存在则直接使用。
  2. 对于成功准备好的场景，立即执行：
     - `python train.py -s <scene_dir> -m output/omniscene_experiments/<scene_name> --eval -r 1 --n_views 6`；由于 `points3d.ply` 已包含绝对尺度点云，默认 **不再加 `--rand_pcd`**，若检测到深度文件缺失可通过 `scripts/run_omniscene.py --force-rand-pcd` 手动退化使用随机点云。
       - `-r 1`：保持 112×200 的低分辨率；若用户切换至 224×400，可在脚本参数中传入。
       - `--n_views 6`：训练集即 6 张输入。
     - `python render.py -m <model_path> --eval -r 1`：使用与训练阶段一致的配置，生成 `metrics_*.txt` 与渲染图像。
  3. 普通模式循环结束后，脚本调用 `metric.py` 汇总最终迭代指标。
- `center150` 使用独立协议：
  1. 默认优化到 10000 次，在 1000、5000、10000 次迭代分别评估 18 个 target 视角，并同时报告全部 18 路（12 路相邻时刻新视角 + 6 路输入视角）和前 12 路新视角的 PSNR、SSIM、LPIPS 均值，以及累计训练耗时。
  2. 每个评估点保存同名的 `renders/` 与 `gt/` 图像；原有 `metrics_<iteration>.txt` 继续记录 18 路均值，新增 `metrics_novel_12_<iteration>.txt` 记录按 `transforms_test.json` 顺序选取的前 12 路新视角均值，同时保存 `training_time_<iteration>.txt`、点云和 checkpoint。
  3. 对于新训练，评估循环还会写入 `metrics_per_view_<iteration>.json`，逐项记录 18 个视角的顺序、图像名、`novel/context` 身份及 PSNR、SSIM、LPIPS。后续如需按其他视角范围统计，可以直接从该文件重新聚合，无需重新训练或渲染。
  4. 训练耗时按纯优化时间累计，不包含评估、点云保存和 checkpoint I/O；累计值写入 checkpoint，断点续跑后继续累加。
  5. 每个样本完成后写入 `center150_complete.json`。再次运行时会按当前参数核验最终点云、各评估点指标、训练耗时以及 render/GT 文件名集合；训练已完成但缺少 12 路指标的旧实验不会重新训练或渲染，只从既有图像补算新视角指标，且不会改写 `training_time_<iteration>.txt`。
  6. 150 个样本全部完成后生成 `center150_metrics_summary.json` 和 `center150_metrics_summary.txt`。逐样本完成文件和最终汇总均同时包含 `all_18_views` 与 `novel_12_views`；为兼容已有分析代码，原有扁平 PSNR、SSIM、LPIPS 字段仍表示 18 路均值。
- 由于流程已覆盖预处理与训练，**不再设计拆分阶段或 `--only-prepare` 等选项**；脚本会在内部自动处理“已有缓存则跳过生成、否则即时生成”的逻辑，确保一次命令即可完成完整实验。
- CLI 选项（全程单阶段）：
  - `--omniscene-root`：原始数据根目录。
  - `--mode`：`val/train/test/demo/center150`（默认 `center150`）。
  - `--resolution`：`112x200` 或 `224x400`（默认 112x200）。
  - `--iterations`：总优化次数，默认 10000。
  - `--eval-iterations`：`center150` 的评估迭代点，默认 `1000 5000 10000`。
  - `--metrics-device`：从既有 PNG 补算前 12 路指标的设备，默认 `cpu`，可显式覆写为 `cuda` 或 `auto`；该统计阶段不计入训练耗时。

---

## 6. 参数与运行策略
- **分辨率**：默认 112×200，可通过脚本参数提升至 224×400；预处理会在首次运行时根据选择的分辨率输出图像，后续继续沿用同一尺寸。
- **n_views**：固定为 6，意味着 `Scene.getTrainCameras()` 只包含 6 张输入。
- **默认协议**：不传参数时直接运行 `center150`，分辨率为 112×200，总优化次数为 10000，并在 1k/5k/10k 三个预算点评估。上述参数均可通过 CLI 显式覆写，续跑完成态和最终汇总会同步使用覆写后的协议。
- **单阶段运行要求**：所有操作必须在同一脚本执行周期内完成。对于每个场景，流程为“检查缓存 → 若无则预处理并保存 → 训练/优化 → 渲染与指标评估”，不提供拆分运行模式。

---

## 7. Center150 运行命令

```bash
conda run -n DropGaussian python scripts/run_omniscene.py
```

默认实验目录为 `output/omniscene_center150_112x200/`。命令可安全重复执行：预处理缓存会复用，已经完整的样本会跳过，未完成样本优先从最近 checkpoint 继续；只有 150 个样本都通过完整性校验后才会生成最终汇总。

如需覆写协议，可显式传参，例如：

```bash
conda run -n DropGaussian python scripts/run_omniscene.py \
  --resolution 224x400 \
  --iterations 20000 \
  --eval-iterations 1000 5000 10000 20000
```

非默认的总迭代数或评估点会自动写入实验目录名，避免与默认协议的 checkpoint 和指标混用。
