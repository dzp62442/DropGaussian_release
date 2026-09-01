import json
import math
import os
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement


StageLiteral = Literal["train", "val", "test", "demo", "center150"]
DEPTH_CONFIDENCE_THRESHOLD = 0.3
OMNISCENE_PREPARED_FORMAT_VERSION = 2
CENTER150_FILENAME = "bins_center150_v1.json"
CENTER150_SAMPLE_COUNT = 150


def _ensure_image_tensor(img: torch.Tensor) -> torch.Tensor:
    if img.ndim != 3:
        raise ValueError(f"Expect image tensor with shape (C,H,W), got {img.shape}")
    return img.detach().cpu().clamp(0.0, 1.0)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = _ensure_image_tensor(tensor)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _replace_suffix(path: str, new_suffix: str) -> str:
    root, _ = os.path.splitext(path)
    return root + new_suffix


def _resolve_relative_depth_path(img_path: str) -> str:
    rel_path = img_path.replace("samples_small", "samples_dpt_small")
    rel_path = rel_path.replace("sweeps_small", "sweeps_dpt_small")
    return _replace_suffix(rel_path, ".npy")


def _resolve_metric_depth_paths(img_path: str) -> Tuple[str, str]:
    metric_path = img_path.replace("samples_small", "samples_dptm_small")
    metric_path = metric_path.replace("sweeps_small", "sweeps_dptm_small")
    depth_path = _replace_suffix(metric_path, "_dpt.npy")
    conf_path = _replace_suffix(metric_path, "_conf.npy")
    return depth_path, conf_path


def _load_npy_with_resize(path: str, target_reso: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
    target_h, target_w = target_reso
    if not os.path.exists(path):
        print(f"[OmniScene] 缺失文件：{path}")
        return np.zeros((target_h, target_w), dtype=np.float32), False
    data = np.load(path).astype(np.float32)
    if data.shape[0] != target_h or data.shape[1] != target_w:
        data_img = Image.fromarray(data)
        data_img = data_img.resize((target_w, target_h), resample=Image.BILINEAR)
        data = np.array(data_img, dtype=np.float32)
    return data, True


def _load_relative_depth_map(path: str, target_reso: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
    disp, ok = _load_npy_with_resize(path, target_reso)
    if not ok or not np.any(disp):
        return np.zeros(target_reso, dtype=np.float32), False
    max_disp = float(np.max(disp))
    min_disp = float(np.min(disp))
    denom = max(min_disp + 1e-3, 1e-3)
    value_range = min(max_disp / denom, 50.0)
    min_clamped = max_disp / max(value_range, 1e-3)
    depth = 1.0 / np.maximum(disp, min_clamped)
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 1e-6:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)
    return depth.astype(np.float32), True


def _load_metric_depth_and_confidence(
    depth_path: str,
    conf_path: str,
    target_reso: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    depth_map, depth_ok = _load_npy_with_resize(depth_path, target_reso)
    conf_map, conf_ok = _load_npy_with_resize(conf_path, target_reso)
    conf_map = np.clip(conf_map, 0.0, 1.0)
    return depth_map, conf_map, depth_ok and conf_ok


@dataclass
class ViewMetadata:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class OmniSceneView:
    image: torch.Tensor
    c2w: torch.Tensor
    intrinsics: torch.Tensor
    metadata: ViewMetadata
    depth_metric: Optional[torch.Tensor] = None
    depth_confidence: Optional[torch.Tensor] = None
    depth_valid_mask: Optional[torch.Tensor] = None
    depth_relative: Optional[torch.Tensor] = None


@dataclass
class OmniSceneSample:
    token: str
    context: List[OmniSceneView]
    target: List[OmniSceneView]


def load_info(info: Dict) -> Tuple[str, np.ndarray, np.ndarray]:
    img_path = info["data_path"]
    c2w = np.array(info["sensor2lidar_transform"], dtype=np.float32)

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t
    return img_path, c2w, w2c


def _maybe_resize_image(img: Image.Image, target_reso: Tuple[int, int], ck: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    target_h, target_w = target_reso
    if img.height == target_h and img.width == target_w:
        return np.array(img), ck
    fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
    scale_h, scale_w = target_h / img.height, target_w / img.width
    fx_scaled, fy_scaled = fx * scale_w, fy * scale_h
    cx_scaled, cy_scaled = cx * scale_w, cy * scale_h
    ck_scaled = np.array(
        [
            [fx_scaled, 0.0, cx_scaled],
            [0.0, fy_scaled, cy_scaled],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    resized = img.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(resized), ck_scaled


def load_conditions(
    img_paths: Sequence[str],
    reso: Tuple[int, int],
    *,
    is_input: bool,
    load_relative_depth: bool = False,
    confidence_threshold: float = DEPTH_CONFIDENCE_THRESHOLD,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[ViewMetadata],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:

    imgs, masks, intrinsics, metadata = [], [], [], []
    metric_depths, depth_confs, depth_valids = [], [], []
    relative_depths: Optional[List[np.ndarray]]
    relative_depths = [] if load_relative_depth else None
    target_h, target_w = reso

    for path in img_paths:
        param_path = path.replace("samples", "samples_param_small")
        param_path = param_path.replace("sweeps", "sweeps_param_small")
        param_path = param_path.replace(".jpg", ".json")
        param = json.load(open(param_path))
        ck = np.array(param["camera_intrinsic"], dtype=np.float32)

        img_path = path.replace("samples", "samples_small")
        img_path = img_path.replace("sweeps", "sweeps_small")
        img = Image.open(img_path)
        arr, ck_scaled = _maybe_resize_image(img, reso, ck)
        ck_norm = ck_scaled.copy()
        ck_norm[0, :] /= target_w
        ck_norm[1, :] /= target_h

        metric_depth_path, conf_path = _resolve_metric_depth_paths(img_path)
        metric_depth, conf_map, _ = _load_metric_depth_and_confidence(metric_depth_path, conf_path, reso)
        valid_mask = (conf_map >= confidence_threshold) & np.isfinite(metric_depth) & (metric_depth > 0.0)

        metric_depths.append(metric_depth)
        depth_confs.append(conf_map)
        depth_valids.append(valid_mask.astype(np.float32))

        if relative_depths is not None:
            rel_path = _resolve_relative_depth_path(img_path)
            rel_depth, _ = _load_relative_depth_map(rel_path, reso)
            relative_depths.append(rel_depth)

        imgs.append(arr)
        intrinsics.append(ck_norm)

        if is_input:
            mask = np.ones((target_h, target_w), dtype=np.float32)
        else:
            mask_path = img_path.replace("sweeps_small", "sweeps_mask_small")
            mask_path = mask_path.replace("samples_small", "samples_mask_small")
            mask_path = mask_path.replace(".jpg", ".png")
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != (target_w, target_h):
                mask_img = mask_img.resize((target_w, target_h), resample=Image.BILINEAR)
            mask = np.array(mask_img, dtype=np.float32) / 255.0
        masks.append(mask)

        metadata.append(
            ViewMetadata(
                fx=float(ck_scaled[0, 0]),
                fy=float(ck_scaled[1, 1]),
                cx=float(ck_scaled[0, 2]),
                cy=float(ck_scaled[1, 2]),
                width=target_w,
                height=target_h,
            )
        )

    imgs_tensor = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    masks_tensor = torch.from_numpy(np.stack(masks, axis=0)).bool()
    intr_tensor = torch.from_numpy(np.stack(intrinsics, axis=0))
    depth_metric_tensor = torch.from_numpy(np.stack(metric_depths, axis=0))
    depth_conf_tensor = torch.from_numpy(np.stack(depth_confs, axis=0))
    depth_valid_tensor = torch.from_numpy(np.stack(depth_valids, axis=0)).bool()
    relative_depth_tensor = (
        torch.from_numpy(np.stack(relative_depths, axis=0)).float() if relative_depths is not None else None
    )
    return (
        imgs_tensor,
        masks_tensor,
        intr_tensor,
        metadata,
        depth_metric_tensor,
        depth_conf_tensor,
        depth_valid_tensor,
        relative_depth_tensor,
    )


class OmniSceneDataset:
    data_version = "interp_12Hz_trainval"
    dataset_prefix = "/datasets/nuScenes"
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        root: Path,
        mode: StageLiteral = "val",
        resolution: Tuple[int, int] = (112, 200),
    ) -> None:
        self.root = Path(root)
        self.mode = mode
        self.resolution = resolution
        self.bin_tokens = self._load_bin_tokens()

    def _json_path(self, name: str) -> Path:
        return self.root / self.data_version / name

    def _load_bin_tokens(self) -> List[str]:
        if self.mode == "train":
            path = self._json_path("bins_train_3.2m.json")
            tokens = json.load(open(path))["bins"]
        elif self.mode == "val":
            path = self._json_path("bins_val_3.2m.json")
            tokens = json.load(open(path))["bins"]
            tokens = tokens[:30000:3000][:10]
        elif self.mode == "test":
            path = self._json_path("bins_val_3.2m.json")
            tokens = json.load(open(path))["bins"]
            tokens = tokens[0::14][:2048]
        elif self.mode == "center150":
            path = self._json_path(CENTER150_FILENAME)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Center150 split file not found: {path}. "
                    "Generate it from the SVF-GS project first."
                )
            tokens = json.load(open(path))["bins"]
            if len(tokens) != CENTER150_SAMPLE_COUNT or len(set(tokens)) != CENTER150_SAMPLE_COUNT:
                raise ValueError(
                    f"Center150 split must contain {CENTER150_SAMPLE_COUNT} unique bins, "
                    f"got {len(tokens)} entries and {len(set(tokens))} unique entries."
                )
        elif self.mode == "demo":
            tokens = [
                "scenee7ef871f77f44331aefdebc24ec034b7_bin010",
                "scenee7ef871f77f44331aefdebc24ec034b7_bin200",
                "scene30ae9c1092f6404a9e6aa0589e809780_bin100",
                "scene84e056bd8e994362a37cba45c0f75558_bin100",
                "scene717053dec2ef4baa913ba1e24c09edff_bin000",
                "scene82240fd6d5ba4375815f8a7fa1561361_bin050",
                "scene724957e51f464a9aa64a16458443786d_bin000",
                "scened3c39710e9da42f48b605824ce2a1927_bin050",
                "scene034256c9639044f98da7562ef3de3646_bin000",
                "scenee0b14a8e11994763acba690bbcc3f56a_bin080",
                "scene7e2d9f38f8eb409ea57b3864bb4ed098_bin150",
                "scene50ff554b3ecb4d208849d042b7643715_bin000",
            ]
        else:
            raise ValueError(f"Unsupported OmniScene mode: {self.mode}")
        return tokens

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def _replace_prefix(self, path: str) -> str:
        return path.replace(self.dataset_prefix, str(self.root))

    def __getitem__(self, index: int) -> OmniSceneSample:
        token = self.bin_tokens[index]
        info_path = self.root / self.data_version / "bin_infos_3.2m" / f"{token}.pkl"
        with open(info_path, "rb") as f:
            bin_info = pkl.load(f)

        input_paths, input_c2w = [], []
        for cam in self.camera_types:
            img_path, c2w, _ = load_info(bin_info["sensor_info"][cam][0])
            input_paths.append(self._replace_prefix(img_path))
            input_c2w.append(c2w)

        (
            input_imgs,
            _input_masks,
            input_intr,
            input_meta,
            input_depth_metric,
            input_depth_conf,
            input_depth_valid,
            input_depth_relative,
        ) = load_conditions(input_paths, self.resolution, is_input=True)
        context_views = self._pack_views(
            input_imgs,
            input_c2w,
            input_intr,
            input_meta,
            depth_metric=input_depth_metric,
            depth_confidence=input_depth_conf,
            depth_valid_mask=input_depth_valid,
            depth_relative=input_depth_relative,
        )

        output_paths, output_c2w = [], []
        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"])
        if frame_num < 3:
            raise RuntimeError(f"Bin {token} only has {frame_num} frames.")
        for cam_idx, cam in enumerate(self.camera_types):
            for offset in (1, 2):
                img_path, c2w, _ = load_info(bin_info["sensor_info"][cam][offset])
                output_paths.append(self._replace_prefix(img_path))
                output_c2w.append(c2w)

        (
            out_imgs,
            _out_masks,
            out_intr,
            out_meta,
            out_depth_metric,
            out_depth_conf,
            out_depth_valid,
            out_depth_relative,
        ) = load_conditions(output_paths, self.resolution, is_input=False)
        output_views = self._pack_views(
            out_imgs,
            output_c2w,
            out_intr,
            out_meta,
            depth_metric=out_depth_metric,
            depth_confidence=out_depth_conf,
            depth_valid_mask=out_depth_valid,
            depth_relative=out_depth_relative,
        )

        # Append context views to targets for evaluation parity
        output_views.extend(context_views)
        return OmniSceneSample(token=token, context=context_views, target=output_views)

    def _pack_views(
        self,
        images: torch.Tensor,
        c2ws: List[np.ndarray],
        intrinsics: torch.Tensor,
        metadata: List[ViewMetadata],
        depth_metric: Optional[torch.Tensor] = None,
        depth_confidence: Optional[torch.Tensor] = None,
        depth_valid_mask: Optional[torch.Tensor] = None,
        depth_relative: Optional[torch.Tensor] = None,
    ) -> List[OmniSceneView]:
        views: List[OmniSceneView] = []
        for idx in range(images.shape[0]):
            views.append(
                OmniSceneView(
                    image=images[idx],
                    c2w=torch.from_numpy(c2ws[idx].astype(np.float32)),
                    intrinsics=intrinsics[idx],
                    metadata=metadata[idx],
                    depth_metric=depth_metric[idx] if depth_metric is not None else None,
                    depth_confidence=depth_confidence[idx] if depth_confidence is not None else None,
                    depth_valid_mask=depth_valid_mask[idx] if depth_valid_mask is not None else None,
                    depth_relative=depth_relative[idx] if depth_relative is not None else None,
                )
            )
        return views


def compute_camera_angle_x(meta: ViewMetadata) -> float:
    return 2.0 * math.atan((meta.width / 2.0) / max(meta.fx, 1e-6))


def opencv_c2w_to_blender(c2w: torch.Tensor) -> torch.Tensor:
    """Convert an OpenCV C2W pose to Blender/OpenGL camera axes."""
    if c2w.shape != (4, 4):
        raise ValueError(f"Expect c2w with shape (4,4), got {tuple(c2w.shape)}")
    flip_yz = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return c2w @ flip_yz


def _extract_points_from_view(
    view: OmniSceneView,
    confidence_threshold: float = DEPTH_CONFIDENCE_THRESHOLD,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if view.depth_metric is None or view.depth_confidence is None:
        return None
    depth = view.depth_metric.detach().cpu().numpy()
    conf = view.depth_confidence.detach().cpu().numpy()
    valid = (conf >= confidence_threshold) & np.isfinite(depth) & (depth > 0.0)
    if view.depth_valid_mask is not None:
        valid &= view.depth_valid_mask.detach().cpu().numpy()
    if not np.any(valid):
        return None
    ys, xs = np.nonzero(valid)
    depths = depth[ys, xs]
    fx, fy, cx, cy = view.metadata.fx, view.metadata.fy, view.metadata.cx, view.metadata.cy
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    depths = depths.astype(np.float32)
    px = (xs - cx) * depths / max(fx, 1e-6)
    py = (ys - cy) * depths / max(fy, 1e-6)
    cam_points = np.stack([px, py, depths], axis=-1)
    ones = np.ones((cam_points.shape[0], 1), dtype=np.float32)
    cam_points_h = np.concatenate([cam_points, ones], axis=1)
    c2w = view.c2w.detach().cpu().numpy().astype(np.float32)
    world_points = (c2w @ cam_points_h.T).T[:, :3]
    colors = (view.image.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.float32)
    rgb = colors[ys.astype(int), xs.astype(int)]
    return world_points, rgb


def generate_point_cloud_from_views(
    views: Sequence[OmniSceneView],
    confidence_threshold: float = DEPTH_CONFIDENCE_THRESHOLD,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    points_list, colors_list = [], []
    for view in views:
        result = _extract_points_from_view(view, confidence_threshold)
        if result is None:
            continue
        pts, cols = result
        points_list.append(pts)
        colors_list.append(cols)
    if not points_list:
        return None
    points = np.concatenate(points_list, axis=0)
    colors = np.concatenate(colors_list, axis=0)
    return points.astype(np.float32), colors.astype(np.float32)


def save_point_cloud(path: Path, xyz: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = xyz.astype(np.float32)
    normals = np.zeros_like(xyz, dtype=np.float32)
    color_arr = colors.astype(np.float32)
    if color_arr.max() <= 1.0:
        color_arr = color_arr * 255.0
    color_arr = np.clip(color_arr, 0.0, 255.0)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    attributes = np.concatenate((xyz, normals, color_arr), axis=1)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    PlyData([vertex_element]).write(path)


def save_random_point_cloud(path: Path, num_points: int = 10000) -> None:
    xyz = np.random.rand(num_points, 3).astype(np.float32) * 2.0 - 1.0
    colors = np.random.rand(num_points, 3).astype(np.float32)
    save_point_cloud(path, xyz, colors * 255.0)


def prepare_scene_directory(sample: OmniSceneSample, scene_dir: Path) -> None:
    train_dir = scene_dir / "train"
    test_dir = scene_dir / "test"
    transforms_train = {
        "omniscene_prepared_format_version": OMNISCENE_PREPARED_FORMAT_VERSION,
        "camera_angle_x": compute_camera_angle_x(sample.context[0].metadata),
        "frames": [],
    }
    transforms_test = {
        "omniscene_prepared_format_version": OMNISCENE_PREPARED_FORMAT_VERSION,
        "camera_angle_x": compute_camera_angle_x(sample.target[0].metadata),
        "frames": [],
    }

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for idx, view in enumerate(sample.context):
        stem = f"{idx:03d}"
        img_path = train_dir / f"{stem}.png"
        _save_image(view.image, img_path)
        transforms_train["frames"].append(
            {
                "file_path": f"train/{stem}",
                "transform_matrix": opencv_c2w_to_blender(view.c2w).tolist(),
            }
        )

    for idx, view in enumerate(sample.target):
        stem = f"{idx:03d}"
        img_path = test_dir / f"{stem}.png"
        _save_image(view.image, img_path)
        transforms_test["frames"].append(
            {
                "file_path": f"test/{stem}",
                "transform_matrix": opencv_c2w_to_blender(view.c2w).tolist(),
            }
        )

    with open(scene_dir / "transforms_train.json", "w") as f:
        json.dump(transforms_train, f, indent=2)
    with open(scene_dir / "transforms_test.json", "w") as f:
        json.dump(transforms_test, f, indent=2)

    ply_path = scene_dir / "points3d.ply"
    if not ply_path.exists():
        generated = generate_point_cloud_from_views(sample.context)
        if generated is None:
            print(f"[Prepare] 未从深度生成点云，回退为随机初始化：{scene_dir}")
            save_random_point_cloud(ply_path)
        else:
            points, colors = generated
            save_point_cloud(ply_path, points, colors)
