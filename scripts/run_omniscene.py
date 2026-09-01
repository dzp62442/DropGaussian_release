import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comp_svfgs.dataset_omniscene import (
    CENTER150_SAMPLE_COUNT,
    OMNISCENE_PREPARED_FORMAT_VERSION,
    OmniSceneDataset,
    OmniSceneSample,
    prepare_scene_directory,
)


OUTPUT_ROOT = REPO_ROOT / "output"
PREPARED_ROOT = OUTPUT_ROOT / "omniscene_prepared"
DEFAULT_EVAL_ITERATIONS = (1000, 5000, 10000)
DEFAULT_TOTAL_ITERATIONS = 10000
CENTER150_STATE_VERSION = 3
METRIC_NAMES = ("PSNR", "SSIM", "LPIPS")
TRAINING_TIME_KEY = "training_time_seconds"


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        h_str, w_str = value.lower().split("x")
        return int(h_str), int(w_str)
    except Exception as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(f"Resolution must be formatted as HxW, got '{value}'") from exc


def needs_preparation(scene_dir: Path) -> bool:
    required = ["transforms_train.json", "transforms_test.json", "points3d.ply"]
    if any(not (scene_dir / name).exists() for name in required):
        return True
    try:
        for name in ("transforms_train.json", "transforms_test.json"):
            transforms = json.loads((scene_dir / name).read_text())
            if transforms.get("omniscene_prepared_format_version") != OMNISCENE_PREPARED_FORMAT_VERSION:
                return True
    except (OSError, json.JSONDecodeError):
        return True
    return False


def run_command(cmd: list[str]) -> None:
    print(f"[Run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def prepare_if_needed(dataset: OmniSceneDataset, index: int, scene_dir: Path) -> None:
    if not needs_preparation(scene_dir):
        print(f"[Prepare] Reusing cached scene at {scene_dir}")
        return
    print(f"[Prepare] Generating Blender assets for {scene_dir.name}")
    sample: OmniSceneSample = dataset[index]
    prepare_scene_directory(sample, scene_dir)


def parse_metrics(metrics_path: Path) -> dict[str, float]:
    values = {}
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        for line in metrics_file:
            name, separator, value = line.partition(":")
            name = name.strip()
            if separator and name in METRIC_NAMES:
                values[name] = float(value.strip())
    if set(values) != set(METRIC_NAMES):
        raise ValueError(f"Incomplete metric file: {metrics_path}")
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite metric in: {metrics_path}")
    return values


def parse_training_time(training_time_path: Path) -> float:
    line = training_time_path.read_text(encoding="utf-8").strip()
    name, separator, value = line.partition(":")
    if not separator or name.strip() != "TRAINING_TIME_SECONDS":
        raise ValueError(f"Invalid training-time file: {training_time_path}")
    training_time_seconds = float(value.strip())
    if not math.isfinite(training_time_seconds) or training_time_seconds < 0.0:
        raise ValueError(f"Invalid training time in: {training_time_path}")
    return training_time_seconds


def expected_test_image_names(scene_dir: Path) -> set[str]:
    transforms_path = scene_dir / "transforms_test.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    names = [Path(frame["file_path"]).stem + ".png" for frame in transforms["frames"]]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate test image names in {transforms_path}")
    return set(names)


def iteration_complete(model_path: Path, scene_dir: Path, iteration: int) -> bool:
    try:
        parse_metrics(model_path / f"metrics_{iteration}.txt")
        parse_training_time(model_path / f"training_time_{iteration}.txt")
        expected_names = expected_test_image_names(scene_dir)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

    iteration_dir = model_path / "test" / f"ours_{iteration}"
    render_paths = list((iteration_dir / "renders").glob("*.png"))
    gt_paths = list((iteration_dir / "gt").glob("*.png"))
    render_names = {path.name for path in render_paths}
    gt_names = {path.name for path in gt_paths}
    return (
        render_names == expected_names
        and gt_names == expected_names
        and all(path.stat().st_size > 0 for path in render_paths + gt_paths)
    )


def scene_complete(
    model_path: Path,
    scene_dir: Path,
    total_iterations: int,
    evaluation_iterations: tuple[int, ...],
) -> bool:
    final_ply = model_path / "point_cloud" / f"iteration_{total_iterations}" / "point_cloud.ply"
    if not final_ply.is_file() or final_ply.stat().st_size == 0:
        return False
    if not all(
        iteration_complete(model_path, scene_dir, iteration)
        for iteration in evaluation_iterations
    ):
        return False
    training_times = [
        parse_training_time(model_path / f"training_time_{iteration}.txt")
        for iteration in evaluation_iterations
    ]
    return training_times == sorted(training_times)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def atomic_write_text(path: Path, content: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def normalized_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name.lower(): metrics[name] for name in METRIC_NAMES}


def evaluation_record(model_path: Path, iteration: int) -> dict[str, float]:
    record = normalized_metrics(parse_metrics(model_path / f"metrics_{iteration}.txt"))
    record[TRAINING_TIME_KEY] = parse_training_time(model_path / f"training_time_{iteration}.txt")
    return record


def write_scene_completion(
    model_path: Path,
    scene_name: str,
    bin_token: str,
    resolution: tuple[int, int],
    total_iterations: int,
    evaluation_iterations: tuple[int, ...],
) -> None:
    metrics = {
        str(iteration): evaluation_record(model_path, iteration)
        for iteration in evaluation_iterations
    }
    atomic_write_json(
        model_path / "center150_complete.json",
        {
            "state_version": CENTER150_STATE_VERSION,
            "split": "center150",
            "scene_name": scene_name,
            "bin_token": bin_token,
            "resolution": list(resolution),
            "total_iterations": total_iterations,
            "evaluation_iterations": list(evaluation_iterations),
            "metrics": metrics,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def find_latest_checkpoint(model_path: Path, total_iterations: int) -> tuple[int, Path] | None:
    candidates = []
    for checkpoint_path in model_path.glob("chkpnt*.pth"):
        suffix = checkpoint_path.stem.removeprefix("chkpnt")
        if suffix.isdigit() and checkpoint_path.stat().st_size > 0:
            iteration = int(suffix)
            if iteration <= total_iterations:
                candidates.append((iteration, checkpoint_path))
    return max(candidates, default=None, key=lambda item: item[0])


def build_train_command(
    python_bin: str,
    scene_dir: Path,
    model_path: Path,
    iterations: int,
    force_rand_pcd: bool,
) -> list[str]:
    command = [
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
        "--iterations",
        str(iterations),
    ]
    if force_rand_pcd:
        command.append("--rand_pcd")
    return command


def run_center150_scene(
    python_bin: str,
    scene_name: str,
    bin_token: str,
    scene_dir: Path,
    model_path: Path,
    resolution: tuple[int, int],
    total_iterations: int,
    evaluation_iterations: tuple[int, ...],
    force_rand_pcd: bool,
) -> None:
    if scene_complete(model_path, scene_dir, total_iterations, evaluation_iterations):
        write_scene_completion(
            model_path,
            scene_name,
            bin_token,
            resolution,
            total_iterations,
            evaluation_iterations,
        )
        print(f"[Skip] Completed center150 scene: {scene_name}")
        return

    latest_checkpoint = find_latest_checkpoint(model_path, total_iterations)
    if latest_checkpoint is None or latest_checkpoint[0] < total_iterations:
        train_cmd = build_train_command(
            python_bin,
            scene_dir,
            model_path,
            total_iterations,
            force_rand_pcd,
        )
        save_iterations = tuple(sorted(set(evaluation_iterations + (total_iterations,))))
        train_cmd.extend(["--test_iterations", *map(str, evaluation_iterations)])
        train_cmd.extend(["--save_iterations", *map(str, save_iterations)])
        train_cmd.extend(["--checkpoint_iterations", *map(str, save_iterations)])
        train_cmd.append("--full_eval_metrics")
        if latest_checkpoint is not None:
            print(f"[Resume] {scene_name} from iteration {latest_checkpoint[0]}")
            train_cmd.extend(["--start_checkpoint", str(latest_checkpoint[1])])
        run_command(train_cmd)
    else:
        print(f"[Resume] Training already reached iteration {latest_checkpoint[0]}: {scene_name}")

    for iteration in evaluation_iterations:
        if iteration_complete(model_path, scene_dir, iteration):
            continue
        try:
            parse_training_time(model_path / f"training_time_{iteration}.txt")
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot recover training time at iteration {iteration}: {model_path}"
            ) from exc
        point_cloud = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not point_cloud.is_file() or point_cloud.stat().st_size == 0:
            raise RuntimeError(
                f"Cannot recover evaluation at iteration {iteration}; missing point cloud: {point_cloud}"
            )
        run_command(
            [
                python_bin,
                "render.py",
                "-m",
                str(model_path),
                "--iteration",
                str(iteration),
                "--eval",
                "-r",
                "1",
                "--skip_train",
            ]
        )

    if not scene_complete(model_path, scene_dir, total_iterations, evaluation_iterations):
        raise RuntimeError(f"Center150 scene did not produce a complete result: {scene_name}")
    write_scene_completion(
        model_path,
        scene_name,
        bin_token,
        resolution,
        total_iterations,
        evaluation_iterations,
    )
    print(f"[Complete] {scene_name}")


def run_standard_scene(
    python_bin: str,
    scene_dir: Path,
    model_path: Path,
    iterations: int,
    force_rand_pcd: bool,
) -> None:
    train_cmd = build_train_command(
        python_bin,
        scene_dir,
        model_path,
        iterations,
        force_rand_pcd,
    )
    run_command(train_cmd)
    run_command(
        [
            python_bin,
            "render.py",
            "-m",
            str(model_path),
            "--eval",
            "-r",
            "1",
        ]
    )


def aggregate_center150_results(
    experiment_root: Path,
    scene_records: list[tuple[str, str, Path, Path]],
    resolution: tuple[int, int],
    total_iterations: int,
    evaluation_iterations: tuple[int, ...],
) -> dict:
    if len(scene_records) != CENTER150_SAMPLE_COUNT:
        raise RuntimeError(
            f"Center150 aggregation requires {CENTER150_SAMPLE_COUNT} scenes, got {len(scene_records)}."
        )

    samples = []
    accumulators = {
        iteration: {name: [] for name in (*METRIC_NAMES, TRAINING_TIME_KEY)}
        for iteration in evaluation_iterations
    }
    for scene_name, bin_token, scene_dir, model_path in scene_records:
        if not scene_complete(model_path, scene_dir, total_iterations, evaluation_iterations):
            raise RuntimeError(f"Cannot aggregate incomplete center150 scene: {scene_name}")
        sample_metrics = {}
        for iteration in evaluation_iterations:
            metrics = parse_metrics(model_path / f"metrics_{iteration}.txt")
            training_time_seconds = parse_training_time(model_path / f"training_time_{iteration}.txt")
            sample_metrics[str(iteration)] = {
                **normalized_metrics(metrics),
                TRAINING_TIME_KEY: training_time_seconds,
            }
            for name in METRIC_NAMES:
                accumulators[iteration][name].append(metrics[name])
            accumulators[iteration][TRAINING_TIME_KEY].append(training_time_seconds)
        samples.append(
            {
                "scene_name": scene_name,
                "bin_token": bin_token,
                "metrics": sample_metrics,
            }
        )

    averages = {}
    for iteration in evaluation_iterations:
        averages[str(iteration)] = {
            "num_samples": CENTER150_SAMPLE_COUNT,
            **{
                name.lower(): sum(accumulators[iteration][name]) / CENTER150_SAMPLE_COUNT
                for name in METRIC_NAMES
            },
            TRAINING_TIME_KEY: (
                sum(accumulators[iteration][TRAINING_TIME_KEY]) / CENTER150_SAMPLE_COUNT
            ),
        }

    summary = {
        "state_version": CENTER150_STATE_VERSION,
        "split": "center150",
        "sample_count": CENTER150_SAMPLE_COUNT,
        "resolution": list(resolution),
        "total_iterations": total_iterations,
        "evaluation_iterations": list(evaluation_iterations),
        "averages": averages,
        "samples": samples,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(experiment_root / "center150_metrics_summary.json", summary)

    lines = [f"Center150 samples: {CENTER150_SAMPLE_COUNT}"]
    for iteration in evaluation_iterations:
        result = averages[str(iteration)]
        lines.extend(
            [
                f"Iteration {iteration}",
                f"PSNR : {result['psnr']:.7f}",
                f"SSIM : {result['ssim']:.7f}",
                f"LPIPS : {result['lpips']:.7f}",
                f"TRAINING_TIME_SECONDS : {result[TRAINING_TIME_KEY]:.7f}",
            ]
        )
    atomic_write_text(experiment_root / "center150_metrics_summary.txt", "\n".join(lines) + "\n")

    print(f"[Summary] {CENTER150_SAMPLE_COUNT} center150 scenes")
    for iteration in evaluation_iterations:
        result = averages[str(iteration)]
        print(
            f"  {iteration}: PSNR={result['psnr']:.7f}, "
            f"SSIM={result['ssim']:.7f}, LPIPS={result['lpips']:.7f}, "
            f"TRAIN_TIME={result[TRAINING_TIME_KEY]:.7f}s"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DropGaussian on OmniScene dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--omniscene-root", type=Path, default="datasets/omniscene", help="Path to original OmniScene root")
    parser.add_argument("--mode", choices=["train", "val", "test", "demo", "center150"], default="center150", help="Dataset split")
    parser.add_argument("--resolution", type=parse_resolution, default="112x200", help="Image resolution HxW")
    parser.add_argument("--iterations", type=int, default=DEFAULT_TOTAL_ITERATIONS, help="Total optimization iterations")
    parser.add_argument(
        "--eval-iterations",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVAL_ITERATIONS),
        help="Evaluation iterations used by the center150 protocol",
    )
    parser.add_argument("--experiment-name", type=str, default="omniscene", help="Experiment folder under output/")
    parser.add_argument(
        "--force-rand-pcd",
        action="store_true",
        help="强制在 train.py 中启用 --rand_pcd，用于调试或当深度文件缺失时手动退化。",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    evaluation_iterations = tuple(args.eval_iterations)
    if args.mode == "center150":
        if any(iteration <= 0 for iteration in evaluation_iterations):
            parser.error("--eval-iterations must contain positive integers.")
        if tuple(sorted(set(evaluation_iterations))) != evaluation_iterations:
            parser.error("--eval-iterations must be unique and strictly increasing.")
        if evaluation_iterations[-1] > args.iterations:
            parser.error("--eval-iterations cannot exceed --iterations.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    resolution_tag = f"{args.resolution[0]}x{args.resolution[1]}"
    prepared_root = PREPARED_ROOT / resolution_tag
    experiment_name = args.experiment_name
    if args.mode == "center150":
        experiment_name += "_center150"
        if (
            args.iterations != DEFAULT_TOTAL_ITERATIONS
            or evaluation_iterations != DEFAULT_EVAL_ITERATIONS
        ):
            eval_tag = "-".join(map(str, evaluation_iterations))
            experiment_name += f"_iter{args.iterations}_eval{eval_tag}"
    experiment_root = OUTPUT_ROOT / f"{experiment_name}_{resolution_tag}"
    prepared_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)

    dataset = OmniSceneDataset(root=args.omniscene_root, mode=args.mode, resolution=args.resolution)
    python_bin = sys.executable
    scene_index_width = 3 if args.mode == "center150" else 2
    scene_records = []

    for idx, token in enumerate(dataset.bin_tokens):
        scene_name = f"{idx + 1:0{scene_index_width}d}_{token}"
        scene_dir = prepared_root / scene_name
        prepare_if_needed(dataset, idx, scene_dir)

        model_path = experiment_root / scene_name
        scene_records.append((scene_name, token, scene_dir, model_path))
        if args.mode == "center150":
            run_center150_scene(
                python_bin,
                scene_name,
                token,
                scene_dir,
                model_path,
                args.resolution,
                args.iterations,
                evaluation_iterations,
                args.force_rand_pcd,
            )
        else:
            run_standard_scene(
                python_bin,
                scene_dir,
                model_path,
                args.iterations,
                args.force_rand_pcd,
            )

    if args.mode == "center150":
        aggregate_center150_results(
            experiment_root,
            scene_records,
            args.resolution,
            args.iterations,
            evaluation_iterations,
        )
    else:
        run_command(
            [
                python_bin,
                "metric.py",
                "--path",
                str(experiment_root),
                "--iteration",
                str(args.iterations),
            ]
        )


if __name__ == "__main__":
    main()
