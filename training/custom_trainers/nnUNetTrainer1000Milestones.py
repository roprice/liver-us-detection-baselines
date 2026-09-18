"""
1000-epoch trainer with milestone checkpoints and dual best-checkpoint
selection, for the preliminary epoch-budget experiment.

Uses a seeded dual-checkpoint approach:
  - Runs to nnU-Net's default 1000 epochs
  - Saves dedicated checkpoints at epochs 50, 100, 150, 300, 500, 750
    (in addition to the automatic best_mass, best_joint, and final)
  - Milestone checkpoints are named checkpoint_ep{N}.pth

Checkpoint selection (mass and joint EMA) is identical to
nnUNetTrainer150Seeded so results stay comparable.

Instrumentation:
  - Logs model parameter count after initialization
  - Logs GPU memory (allocated, reserved) after initialization
  - Resets PyTorch peak memory stats after init so training peaks
    are measured separately from model creation
  - Logs GPU memory (current + peak) at each milestone epoch and
    at training completion

Usage (seeds 42, 43, 44):
  nnUNetv2_train 1 2d 0 --npz -tr nnUNetTrainer1000Milestones_s42
  nnUNetv2_train 1 2d 0 --npz -tr nnUNetTrainer1000Milestones_s43
  nnUNetv2_train 1 2d 0 --npz -tr nnUNetTrainer1000Milestones_s44
"""

import random
from os.path import join
from time import time

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer1000Milestones(nnUNetTrainer):
    training_seed = None
    MASS_CLASS_INDEX = 1
    EMA_ALPHA = 0.9
    MILESTONE_EPOCHS = {50, 100, 150, 300, 500, 750}

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self._best_ema_mass = None
        self._ema_mass = None
        self._best_ema_joint = None
        self._ema_joint = None

    def initialize(self):
        first_init = not self.was_initialized
        if first_init:
            assert self.training_seed is not None, \
                "Subclass must set training_seed"
            self.print_to_log_file(
                f"Setting training seed: {self.training_seed}")
            random.seed(self.training_seed)
            np.random.seed(self.training_seed)
            torch.manual_seed(self.training_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.training_seed)
        super().initialize()
        if first_init and self.local_rank == 0:
            # Model footprint
            num_params = sum(p.numel() for p in self.network.parameters())
            num_trainable = sum(
                p.numel() for p in self.network.parameters()
                if p.requires_grad)
            self.print_to_log_file(
                f"Model parameters: {num_params:,} total, "
                f"{num_trainable:,} trainable")
            # GPU memory baseline after init (before training allocations)
            if torch.cuda.is_available():
                self.print_to_log_file(
                    f"GPU memory post-init: "
                    f"allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB, "
                    f"reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB")
                torch.cuda.reset_peak_memory_stats()
                self.print_to_log_file(
                    "Peak GPU memory stats reset for training measurement")

    def _log_gpu_memory(self, label):
        """Log current and peak GPU memory with a descriptive label."""
        if not torch.cuda.is_available() or self.local_rank != 0:
            return
        self.print_to_log_file(
            f"{label} GPU memory: "
            f"allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB, "
            f"reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB, "
            f"peak_allocated="
            f"{torch.cuda.max_memory_allocated() / 1e9:.3f} GB, "
            f"peak_reserved="
            f"{torch.cuda.max_memory_reserved() / 1e9:.3f} GB")

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file(
            'train_loss',
            np.round(self.logger.get_value('train_losses', step=-1), decimals=4))
        self.print_to_log_file(
            'val_loss',
            np.round(self.logger.get_value('val_losses', step=-1), decimals=4))

        dice_per_class = self.logger.get_value('dice_per_class_or_region', step=-1)
        self.print_to_log_file(
            'Pseudo dice', [np.round(i, decimals=4) for i in dice_per_class])
        self.print_to_log_file(
            f"Epoch time: "
            f"{np.round(self.logger.get_value('epoch_end_timestamps', step=-1) - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)} s")

        current_mass_dice = dice_per_class[self.MASS_CLASS_INDEX]
        current_joint_dice = np.mean(dice_per_class)

        if self._ema_mass is None:
            self._ema_mass = current_mass_dice
        else:
            self._ema_mass = (self.EMA_ALPHA * self._ema_mass
                              + (1 - self.EMA_ALPHA) * current_mass_dice)

        if self._ema_joint is None:
            self._ema_joint = current_joint_dice
        else:
            self._ema_joint = (self.EMA_ALPHA * self._ema_joint
                               + (1 - self.EMA_ALPHA) * current_joint_dice)

        self.print_to_log_file(
            f"Mass pseudo Dice: {np.round(current_mass_dice, decimals=4)}, "
            f"EMA: {np.round(self._ema_mass, decimals=4)}")

        # Best mass checkpoint
        if self._best_ema_mass is None or self._ema_mass > self._best_ema_mass:
            self._best_ema_mass = self._ema_mass
            self.print_to_log_file(
                f"New best EMA mass pseudo Dice: "
                f"{np.round(self._best_ema_mass, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_mass.pth'))

        # Best joint checkpoint
        if self._best_ema_joint is None or self._ema_joint > self._best_ema_joint:
            self._best_ema_joint = self._ema_joint
            self.print_to_log_file(
                f"New best EMA joint pseudo Dice: "
                f"{np.round(self._best_ema_joint, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_joint.pth'))

        self._best_ema = self._best_ema_mass

        # Milestone checkpoints (epoch index is 0-based; +1 for human epoch count)
        completed_epoch = self.current_epoch + 1
        if completed_epoch in self.MILESTONE_EPOCHS:
            fname = join(self.output_folder, f'checkpoint_ep{completed_epoch}.pth')
            self.print_to_log_file(f"Saving milestone checkpoint: epoch {completed_epoch}")
            self.save_checkpoint(fname)
            self._log_gpu_memory(f"Milestone {completed_epoch}")

        # Final epoch memory snapshot
        if completed_epoch == self.num_epochs:
            self._log_gpu_memory("Training complete")

        # Periodic checkpoint
        if (self.current_epoch + 1) % self.save_every == 0 \
                and not self.disable_checkpointing:
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank == 0 and not self.disable_checkpointing:
            try:
                checkpoint = torch.load(filename, weights_only=False)
                checkpoint['_best_ema_mass'] = self._best_ema_mass
                checkpoint['_ema_mass'] = self._ema_mass
                checkpoint['_best_ema_joint'] = self._best_ema_joint
                checkpoint['_ema_joint'] = self._ema_joint
                torch.save(checkpoint, filename)
            except Exception:
                pass

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
        self._best_ema_mass = checkpoint.get('_best_ema_mass', None)
        self._ema_mass = checkpoint.get('_ema_mass', None)
        self._best_ema_joint = checkpoint.get('_best_ema_joint', None)
        self._ema_joint = checkpoint.get('_ema_joint', None)


class nnUNetTrainer1000Milestones_s42(nnUNetTrainer1000Milestones):
    training_seed = 42


class nnUNetTrainer1000Milestones_s43(nnUNetTrainer1000Milestones):
    training_seed = 43


class nnUNetTrainer1000Milestones_s44(nnUNetTrainer1000Milestones):
    training_seed = 44
