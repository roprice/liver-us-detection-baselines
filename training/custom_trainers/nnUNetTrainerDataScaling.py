"""Seeded 150-epoch trainer for the data-scaling experiment.

Each subclass fixes the initialization seed while retaining nnU-Net's
standard training loop and checkpoint behavior. The experiment evaluates
``checkpoint_final.pth`` after the fixed 150-epoch budget selected by the
milestones pilot.

Instrumentation logs model parameter counts and GPU memory after
initialization, then logs current and peak GPU memory after epoch 150.

Usage:
    nnUNetv2_train DATASET_ID 2d 0 --npz \
        -tr nnUNetTrainerDataScaling_seed42 --c
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDataScaling(nnUNetTrainer):
    training_seed = None
    SPLIT_POLICY = "all_pathologies_in_training_v1"
    PATHOLOGIES = ("Malignant", "Benign", "Normal")

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150

    def do_split(self):
        training_cases, validation_cases = super().do_split()
        training_cases = sorted(training_cases)
        validation_cases = sorted(validation_cases)
        if set(training_cases) & set(validation_cases):
            raise RuntimeError("Data scaling requires disjoint training and validation cases")

        from nnunetv2.paths import nnUNet_raw

        mapping_path = (
            Path(nnUNet_raw) / self.plans_manager.dataset_name / "case_mapping.json")
        records = json.loads(mapping_path.read_text())
        categories = {}
        for record in records:
            if record["split"] == "train":
                case = record["case_name"]
                if case in categories:
                    raise RuntimeError(f"Duplicate training case in {mapping_path}: {case}")
                categories[case] = record["category"]
        for case in training_cases + validation_cases:
            if categories.get(case) not in self.PATHOLOGIES:
                raise RuntimeError(f"Missing or unsupported pathology for {case}")

        original_split = {"train": training_cases.copy(), "val": validation_cases.copy()}
        counts = Counter(categories[case] for case in training_cases)
        for pathology in self.PATHOLOGIES:
            if counts[pathology]:
                continue
            incoming = next(
                (case for case in validation_cases if categories[case] == pathology), None)
            outgoing = next(
                (case for case in training_cases if counts[categories[case]] > 1), None)
            if incoming is None or outgoing is None:
                raise RuntimeError(
                    f"Cannot retain all three pathologies in training: missing {pathology}")
            training_cases.remove(outgoing)
            validation_cases.remove(incoming)
            training_cases.append(incoming)
            validation_cases.append(outgoing)
            training_cases.sort()
            validation_cases.sort()
            counts[pathology] += 1
            counts[categories[outgoing]] -= 1
            self.print_to_log_file(
                f"Split repair: {incoming} ({pathology}) -> train; "
                f"{outgoing} ({categories[outgoing]}) -> validation")

        effective_split = {"train": training_cases, "val": validation_cases}
        evidence = {
            "policy": self.SPLIT_POLICY,
            "fold": self.fold,
            "original_split": original_split,
            "effective_split": effective_split,
            "pathology_counts": {
                partition: dict(Counter(categories[case] for case in cases))
                for partition, cases in effective_split.items()
            },
        }
        if self.local_rank == 0:
            output = Path(self.output_folder)
            output.mkdir(parents=True, exist_ok=True)
            (output / "data_scaling_split.json").write_text(
                json.dumps(evidence, indent=2) + "\n")
        self.print_to_log_file("Effective split pathology counts:", evidence["pathology_counts"])
        return training_cases, validation_cases

    def save_checkpoint(self, filename):
        super().save_checkpoint(filename)
        if self.local_rank == 0 and not self.disable_checkpointing:
            checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
            checkpoint["data_scaling_split_policy"] = self.SPLIT_POLICY
            torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        checkpoint = (
            torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False)
            if isinstance(filename_or_checkpoint, (str, Path))
            else filename_or_checkpoint)
        if checkpoint.get("data_scaling_split_policy") != self.SPLIT_POLICY:
            raise RuntimeError(
                "This checkpoint predates the all-pathologies training split policy. "
                "Archive the old run's result folder and restart from scratch; "
                "do not resume it with a different split.")
        super().load_checkpoint(checkpoint)

    def initialize(self):
        first_init = not self.was_initialized
        if first_init:
            assert self.training_seed is not None, (
                "Subclass must set training_seed")
            self.print_to_log_file(
                f"Setting training seed: {self.training_seed}")
            random.seed(self.training_seed)
            np.random.seed(self.training_seed)
            torch.manual_seed(self.training_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.training_seed)

        super().initialize()

        if first_init and self.local_rank == 0:
            num_parameters = sum(
                parameter.numel() for parameter in self.network.parameters())
            num_trainable_parameters = sum(
                parameter.numel() for parameter in self.network.parameters()
                if parameter.requires_grad)
            self.print_to_log_file(
                f"Model parameters: {num_parameters:,} total, "
                f"{num_trainable_parameters:,} trainable")

            if torch.cuda.is_available():
                self.print_to_log_file(
                    "GPU memory post-init: "
                    f"allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB, "
                    f"reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB")
                torch.cuda.reset_peak_memory_stats()
                self.print_to_log_file(
                    "Peak GPU memory stats reset for training measurement")

    def _log_gpu_memory(self, label):
        if not torch.cuda.is_available() or self.local_rank != 0:
            return

        learning_rate = self.optimizer.param_groups[0]["lr"]
        self.print_to_log_file(
            f"{label} GPU memory: "
            f"learning_rate={learning_rate:.3e}, "
            f"allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB, "
            f"reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB, "
            f"peak_allocated="
            f"{torch.cuda.max_memory_allocated() / 1e9:.3f} GB, "
            f"peak_reserved="
            f"{torch.cuda.max_memory_reserved() / 1e9:.3f} GB")

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.current_epoch == self.num_epochs:
            self._log_gpu_memory("Training complete")


class nnUNetTrainerDataScaling_seed42(nnUNetTrainerDataScaling):
    training_seed = 42


class nnUNetTrainerDataScaling_seed43(nnUNetTrainerDataScaling):
    training_seed = 43


class nnUNetTrainerDataScaling_seed44(nnUNetTrainerDataScaling):
    training_seed = 44
