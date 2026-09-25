import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.predictions_dataset import build_predictions_dataset as builder
from analysis.predictions_dataset import load_predictions_dataset as loader


class ComponentFalsePositiveCountTests(unittest.TestCase):
    def evaluate(self, prediction, ground_truth):
        case = {
            "image_id": "example",
            "reference_shape": ground_truth.shape,
            "ground_truth_mass": ground_truth,
            "ground_truth_liver": np.zeros_like(ground_truth),
            "mass_present": bool(ground_truth.any()),
            "reference_path": "reference.png",
            "pathology": "normal" if not ground_truth.any() else "mass",
            "original_data_path": "original.png",
        }
        config = {
            "prediction_directory": Path("."),
            "configuration_id": "example_configuration",
            "experiment_name": "example",
            "evaluation_split": "test",
            "training_set_size": 5,
            "seed": 42,
            "epoch": None,
            "checkpoint": "best",
            "encoder": "example",
        }
        with patch.object(builder, "read_mask", return_value=prediction), patch.object(
            builder, "relative_path", return_value="example.png"
        ):
            return builder.evaluate_case(config, case)

    def assert_fp_counts(self, row, overlap, centroid):
        self.assertEqual(
            [row[f"overlap_detection_iou_{threshold}_fp_count"] for threshold in ("00", "02", "05")],
            overlap,
        )
        self.assertEqual(
            [row[f"centroid_detection_deq_{threshold}_fp_count"] for threshold in ("025", "050", "100")],
            centroid,
        )

    def test_normal_components_are_all_false_positives_and_load_as_ints(self):
        prediction = np.zeros((30, 30), dtype=np.uint8)
        prediction[2, 2] = 2
        prediction[20, 20] = 2
        row = self.evaluate(prediction, np.zeros_like(prediction, dtype=bool))
        self.assertEqual(row["retained_component_count"], 2)
        self.assertEqual(row["unmatched_retained_component_count"], 2)
        self.assertIsNone(row["overlap_detection_iou_00_flag"])
        self.assert_fp_counts(row, [2, 2, 2], [2, 2, 2])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=builder.FIELDNAMES)
                writer.writeheader()
                writer.writerow({field: "" if value is None else value for field, value in row.items()})
            loaded = loader.load_predictions_dataset(path).rows[0]
            for field in builder.FIELDNAMES:
                if field.endswith("_fp_count"):
                    self.assertEqual(getattr(loaded, field), 2)
                    self.assertIsInstance(getattr(loaded, field), int)
            self.assertIsNone(loaded.overlap_detection_iou_00_flag)

    def test_one_to_one_match_even_with_two_qualifying_components(self):
        ground_truth = np.zeros((30, 30), dtype=bool)
        ground_truth[10:15, 10:15] = True
        prediction = np.zeros(ground_truth.shape, dtype=np.uint8)
        prediction[10:12, 10:12] = 2
        prediction[13:15, 13:15] = 2
        row = self.evaluate(prediction, ground_truth)
        self.assertEqual(row["retained_component_count"], 2)
        self.assertEqual(row["unmatched_retained_component_count"], 0)
        self.assert_fp_counts(row, [1, 2, 2], [2, 1, 1])

    def test_strict_iou_threshold_and_centroid_distance(self):
        ground_truth = np.zeros((30, 30), dtype=bool)
        ground_truth[10:15, 10:15] = True
        prediction = np.zeros(ground_truth.shape, dtype=np.uint8)
        prediction[10, 10:15] = 2  # IoU exactly 0.2.
        prediction[25, 25] = 2
        row = self.evaluate(prediction, ground_truth)
        self.assert_fp_counts(row, [1, 2, 2], [2, 1, 1])

    def test_centroid_match_without_overlap(self):
        ground_truth = np.zeros((30, 30), dtype=bool)
        ground_truth[10:15, 10:15] = True
        prediction = np.zeros(ground_truth.shape, dtype=np.uint8)
        prediction[12, 16] = 2
        row = self.evaluate(prediction, ground_truth)
        self.assertEqual(row["unmatched_retained_component_count"], 1)
        self.assert_fp_counts(row, [1, 1, 1], [1, 1, 0])

    def test_high_iou_match(self):
        ground_truth = np.zeros((30, 30), dtype=bool)
        ground_truth[10:15, 10:15] = True
        prediction = np.zeros(ground_truth.shape, dtype=np.uint8)
        prediction[10:15, 10:15] = 2
        prediction[25, 25] = 2
        row = self.evaluate(prediction, ground_truth)
        self.assert_fp_counts(row, [1, 1, 1], [1, 1, 1])

    def test_no_predictions_are_not_false_positives(self):
        ground_truth = np.zeros((30, 30), dtype=bool)
        ground_truth[10:15, 10:15] = True
        row = self.evaluate(np.zeros(ground_truth.shape, dtype=np.uint8), ground_truth)
        self.assert_fp_counts(row, [0, 0, 0], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
