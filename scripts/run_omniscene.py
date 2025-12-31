import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comp_svfgs.dataset_omniscene import OmniSceneDataset, OmniSceneSample, prepare_scene_directory


OUTPUT_ROOT = REPO_ROOT / "output"
PREPARED_ROOT = OUTPUT_ROOT / "omniscene_prepared"
EXPERIMENT_ROOT = OUTPUT_ROOT / "omniscene_experiments"


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        h_str, w_str = value.lower().split("x")
        return int(h_str), int(w_str)
    except Exception as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(f"Resolution must be formatted as HxW, got '{value}'") from exc


def needs_preparation(scene_dir: Path) -> bool:
    required = ["transforms_train.json", "transforms_test.json", "points3d.ply"]
    return any(not (scene_dir / name).exists() for name in required)


def run_command(cmd: list[str]) -> None:
    print(f"[Run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def prepare_if_needed(dataset: OmniSceneDataset, index: int, scene_dir: Path) -> None:
    if not needs_preparation(scene_dir):
        print(f"[Prepare] Reusing cached scene at {scene_dir}")
        return
    print(f"[Prepare] Generating Blender assets for {scene_dir.name}")
    sample: OmniSceneSample = dataset[index]
    prepare_scene_directory(sample, scene_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DropGaussian on OmniScene dataset")
    parser.add_argument("--omniscene-root", type=Path, default="datasets/omniscene", help="Path to original OmniScene root")
    parser.add_argument("--mode", choices=["train", "val", "test", "demo"], default="val", help="Dataset split")
    parser.add_argument("--resolution", type=parse_resolution, default="112x200", help="Image resolution HxW")
    parser.add_argument("--iterations", type=int, default=10000, help="Override optimization iterations")
    parser.add_argument("--experiment-name", type=str, default="omniscene", help="Experiment folder under output/")
    parser.add_argument(
        "--force-rand-pcd",
        action="store_true",
        help="强制在 train.py 中启用 --rand_pcd，用于调试或当深度文件缺失时手动退化。",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    resolution_tag = f"{args.resolution[0]}x{args.resolution[1]}"
    prepared_root = PREPARED_ROOT / resolution_tag
    args.experiment_name = f"{args.experiment_name}_{resolution_tag}"
    experiment_root = OUTPUT_ROOT / args.experiment_name
    prepared_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)

    dataset = OmniSceneDataset(root=args.omniscene_root, mode=args.mode, resolution=args.resolution)
    python_bin = sys.executable
    iteration_str = str(args.iterations) if args.iterations is not None else None

    for idx, token in enumerate(dataset.bin_tokens):
        scene_name = f"{idx + 1:02d}_{token}"
        scene_dir = prepared_root / scene_name
        prepare_if_needed(dataset, idx, scene_dir)

        model_path = experiment_root / scene_name
        train_cmd = [
            python_bin,
            "train.py",
            "-s",
            str(scene_dir),
            "-m",
            str(model_path),
            "--eval",
            "-r",
            "1",
            "--n_views",
            "6",
        ]
        if args.force_rand_pcd:
            train_cmd.append("--rand_pcd")
        if iteration_str:
            train_cmd.extend(["--iterations", iteration_str])
        run_command(train_cmd)

        render_cmd = [
            python_bin,
            "render.py",
            "-m",
            str(model_path),
            "--eval",
            "-r",
            "1",
        ]
        run_command(render_cmd)

    metric_cmd = [
        python_bin,
        "metric.py",
        "--path",
        str(experiment_root),
    ]
    if iteration_str:
        metric_cmd.extend(["--iteration", iteration_str])
    run_command(metric_cmd)


if __name__ == "__main__":
    main()
