import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_omniscene


class FakeMetricEvaluator:
    def __init__(self) -> None:
        self.calls = []

    def evaluate(self, render_paths, gt_paths):
        self.calls.append(
            ([path.name for path in render_paths], [path.name for path in gt_paths])
        )
        return {"PSNR": 21.0, "SSIM": 0.81, "LPIPS": 0.19}


class OmniSceneMetricScopeTest(unittest.TestCase):
    def _make_completed_scene(self, root: Path):
        scene_name = "001_scene"
        scene_dir = root / "prepared" / scene_name
        model_path = root / "experiment" / scene_name
        scene_dir.mkdir(parents=True)
        model_path.mkdir(parents=True)

        image_names = [f"view_{index:02d}.png" for index in range(18)]
        transforms = {
            "frames": [
                {"file_path": f"test/{Path(name).stem}"} for name in image_names
            ]
        }
        (scene_dir / "transforms_test.json").write_text(json.dumps(transforms))
        run_omniscene.write_metrics(
            model_path / "metrics_1000.txt",
            {"PSNR": 20.0, "SSIM": 0.8, "LPIPS": 0.2},
        )
        training_time_path = model_path / "training_time_1000.txt"
        training_time_path.write_text("TRAINING_TIME_SECONDS : 2.5000000\n")
        point_cloud = model_path / "point_cloud" / "iteration_1000" / "point_cloud.ply"
        point_cloud.parent.mkdir(parents=True)
        point_cloud.write_bytes(b"ply")

        for folder in ("renders", "gt"):
            image_dir = model_path / "test" / "ours_1000" / folder
            image_dir.mkdir(parents=True)
            for name in image_names:
                (image_dir / name).write_bytes(b"png")
        return scene_name, scene_dir, model_path, training_time_path, image_names

    def test_completed_scene_backfills_first_twelve_without_touching_time(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (
                scene_name,
                scene_dir,
                model_path,
                training_time_path,
                image_names,
            ) = self._make_completed_scene(root)
            evaluator = FakeMetricEvaluator()
            original_time_bytes = training_time_path.read_bytes()
            original_time_mtime = training_time_path.stat().st_mtime_ns

            self.assertTrue(
                run_omniscene.scene_training_complete(
                    model_path, scene_dir, 1000, (1000,)
                )
            )
            self.assertFalse(
                run_omniscene.scene_complete(model_path, scene_dir, 1000, (1000,))
            )

            with mock.patch.object(run_omniscene, "run_command") as run_command:
                run_omniscene.run_center150_scene(
                    "python",
                    scene_name,
                    "scene",
                    scene_dir,
                    model_path,
                    (112, 200),
                    1000,
                    (1000,),
                    False,
                    evaluator,
                )
            run_command.assert_not_called()

            self.assertEqual(evaluator.calls[0][0], image_names[:12])
            self.assertEqual(evaluator.calls[0][1], image_names[:12])
            self.assertEqual(training_time_path.read_bytes(), original_time_bytes)
            self.assertEqual(training_time_path.stat().st_mtime_ns, original_time_mtime)
            self.assertEqual(
                run_omniscene.parse_metrics(
                    run_omniscene.novel_metrics_path(model_path, 1000)
                ),
                {"PSNR": 21.0, "SSIM": 0.81, "LPIPS": 0.19},
            )
            self.assertTrue(
                run_omniscene.scene_complete(model_path, scene_dir, 1000, (1000,))
            )

            completion = json.loads(
                (model_path / "center150_complete.json").read_text()
            )
            metrics = completion["metrics"]["1000"]
            self.assertEqual(metrics["psnr"], 20.0)
            self.assertEqual(metrics["all_18_views"]["num_views"], 18)
            self.assertEqual(metrics["novel_12_views"]["num_views"], 12)
            self.assertEqual(metrics["novel_12_views"]["psnr"], 21.0)

            records = [(scene_name, "scene", scene_dir, model_path)]
            with mock.patch.object(run_omniscene, "CENTER150_SAMPLE_COUNT", 1):
                summary = run_omniscene.aggregate_center150_results(
                    root / "experiment", records, (112, 200), 1000, (1000,)
                )
            average = summary["averages"]["1000"]
            self.assertEqual(average["psnr"], 20.0)
            self.assertEqual(average["all_18_views"]["psnr"], 20.0)
            self.assertEqual(average["novel_12_views"]["psnr"], 21.0)
            self.assertEqual(average["training_time_seconds"], 2.5)

            with mock.patch.object(run_omniscene, "run_command") as run_command:
                run_omniscene.run_center150_scene(
                    "python",
                    scene_name,
                    "scene",
                    scene_dir,
                    model_path,
                    (112, 200),
                    1000,
                    (1000,),
                    False,
                    evaluator,
                )
            run_command.assert_not_called()
            self.assertEqual(len(evaluator.calls), 1)

    def test_new_scene_trains_then_writes_both_metric_scopes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (
                scene_name,
                scene_dir,
                model_path,
                _training_time_path,
                _image_names,
            ) = self._make_completed_scene(root)
            final_ply = model_path / "point_cloud" / "iteration_1000" / "point_cloud.ply"
            final_ply.unlink()
            evaluator = FakeMetricEvaluator()

            def finish_training(command):
                self.assertIn("train.py", command)
                final_ply.write_bytes(b"ply")

            with mock.patch.object(
                run_omniscene, "run_command", side_effect=finish_training
            ) as run_command:
                run_omniscene.run_center150_scene(
                    "python",
                    scene_name,
                    "scene",
                    scene_dir,
                    model_path,
                    (112, 200),
                    1000,
                    (1000,),
                    False,
                    evaluator,
                )

            run_command.assert_called_once()
            self.assertEqual(len(evaluator.calls), 1)
            self.assertTrue(
                run_omniscene.scene_complete(model_path, scene_dir, 1000, (1000,))
            )
            completion = json.loads(
                (model_path / "center150_complete.json").read_text()
            )
            self.assertIn(
                "novel_12_views", completion["metrics"]["1000"]
            )


if __name__ == "__main__":
    unittest.main()
