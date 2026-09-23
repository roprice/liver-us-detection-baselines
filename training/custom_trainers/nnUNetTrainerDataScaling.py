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

import random

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDataScaling(nnUNetTrainer):
    training_seed = None

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150

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
