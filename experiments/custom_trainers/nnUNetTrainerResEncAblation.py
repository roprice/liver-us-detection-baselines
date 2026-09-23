"""Seeded 150-epoch trainer for the ResEnc architecture ablation."""

from pathlib import Path
import tempfile

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnUNetTrainerDataScaling import nnUNetTrainerDataScaling


class nnUNetTrainerResEncAblation(nnUNetTrainerDataScaling):
    def initialize(self):
        super().initialize()

        network = self.network
        while True:
            if hasattr(network, "_orig_mod"):
                network = network._orig_mod
            elif hasattr(network, "module"):
                network = network.module
            else:
                break

        network_class = type(network).__name__
        if "ResidualEncoderUNet" not in network_class:
            raise RuntimeError(
                "ResEnc ablation requires ResidualEncoderUNet, "
                f"but the initialized network is {network_class}. "
                "Use -p nnUNetResEncUNetMPlans."
            )

    def load_checkpoint(self, filename_or_checkpoint):
        checkpoint = (
            torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False)
            if isinstance(filename_or_checkpoint, (str, Path))
            else filename_or_checkpoint
        )
        if checkpoint.get("data_scaling_split_policy") != self.SPLIT_POLICY:
            raise RuntimeError(
                "This checkpoint does not use the required data-scaling split policy."
            )

        if isinstance(filename_or_checkpoint, (str, Path)):
            nnUNetTrainer.load_checkpoint(self, str(filename_or_checkpoint))
            return

        with tempfile.NamedTemporaryFile(suffix=".pth") as temporary_checkpoint:
            torch.save(checkpoint, temporary_checkpoint.name)
            nnUNetTrainer.load_checkpoint(self, temporary_checkpoint.name)


class nnUNetTrainerResEncAblation_seed42(nnUNetTrainerResEncAblation):
    training_seed = 42


class nnUNetTrainerResEncAblation_seed43(nnUNetTrainerResEncAblation):
    training_seed = 43


class nnUNetTrainerResEncAblation_seed44(nnUNetTrainerResEncAblation):
    training_seed = 44
