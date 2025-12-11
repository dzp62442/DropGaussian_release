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


StageLiteral = Literal["train", "val", "test", "demo"]


def _ensure_image_tensor(img: torch.Tensor) -> torch.Tensor:
    if img.ndim != 3:
        raise ValueError(f"Expect image tensor with shape (C,H,W), got {img.shape}")
    return img.detach().cpu().clamp(0.0, 1.0)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = _ensure_image_tensor(tensor)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[ViewMetadata]]:

    imgs, masks, intrinsics, metadata = [], [], [], []
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
    return imgs_tensor, masks_tensor, intr_tensor, metadata


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
        else:
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

        input_imgs, input_masks, input_intr, input_meta = load_conditions(
            input_paths, self.resolution, is_input=True
        )
        context_views = self._pack_views(input_imgs, input_c2w, input_intr, input_meta)

        output_paths, output_c2w = [], []
        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"])
        if frame_num < 3:
            raise RuntimeError(f"Bin {token} only has {frame_num} frames.")
        for cam_idx, cam in enumerate(self.camera_types):
            for offset in (1, 2):
                img_path, c2w, _ = load_info(bin_info["sensor_info"][cam][offset])
                output_paths.append(self._replace_prefix(img_path))
                output_c2w.append(c2w)

        out_imgs, out_masks, out_intr, out_meta = load_conditions(
            output_paths, self.resolution, is_input=False
        )
        output_views = self._pack_views(out_imgs, output_c2w, out_intr, out_meta)

        # Append context views to targets for evaluation parity
        output_views.extend(context_views)
        return OmniSceneSample(token=token, context=context_views, target=output_views)

    def _pack_views(
        self,
        images: torch.Tensor,
        c2ws: List[np.ndarray],
        intrinsics: torch.Tensor,
        metadata: List[ViewMetadata],
    ) -> List[OmniSceneView]:
        views: List[OmniSceneView] = []
        for idx in range(images.shape[0]):
            views.append(
                OmniSceneView(
                    image=images[idx],
                    c2w=torch.from_numpy(c2ws[idx].astype(np.float32)),
                    intrinsics=intrinsics[idx],
                    metadata=metadata[idx],
                )
            )
        return views


def compute_camera_angle_x(meta: ViewMetadata) -> float:
    return 2.0 * math.atan((meta.width / 2.0) / max(meta.fx, 1e-6))


def save_random_point_cloud(path: Path, num_points: int = 10000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.random.rand(num_points, 3).astype(np.float32) * 2.0 - 1.0
    normals = np.zeros_like(xyz)
    colors = np.random.rand(num_points, 3).astype(np.float32)
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
    attributes = np.concatenate((xyz, normals, colors * 255.0), axis=1)
    elements = np.empty(num_points, dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    PlyData([vertex_element]).write(path)


def prepare_scene_directory(sample: OmniSceneSample, scene_dir: Path) -> None:
    train_dir = scene_dir / "train"
    test_dir = scene_dir / "test"
    transforms_train = {"camera_angle_x": compute_camera_angle_x(sample.context[0].metadata), "frames": []}
    transforms_test = {"camera_angle_x": compute_camera_angle_x(sample.target[0].metadata), "frames": []}

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for idx, view in enumerate(sample.context):
        stem = f"{idx:03d}"
        img_path = train_dir / f"{stem}.png"
        _save_image(view.image, img_path)
        transforms_train["frames"].append(
            {
                "file_path": f"train/{stem}",
                "transform_matrix": view.c2w.tolist(),
            }
        )

    for idx, view in enumerate(sample.target):
        stem = f"{idx:03d}"
        img_path = test_dir / f"{stem}.png"
        _save_image(view.image, img_path)
        transforms_test["frames"].append(
            {
                "file_path": f"test/{stem}",
                "transform_matrix": view.c2w.tolist(),
            }
        )

    with open(scene_dir / "transforms_train.json", "w") as f:
        json.dump(transforms_train, f, indent=2)
    with open(scene_dir / "transforms_test.json", "w") as f:
        json.dump(transforms_test, f, indent=2)

    ply_path = scene_dir / "points3d.ply"
    if not ply_path.exists():
        save_random_point_cloud(ply_path)
