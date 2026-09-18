#!/usr/bin/env python
"""
Per-image inference benchmark for trained nnU-Net checkpoints.

Measures:
  - Model loading time (reading the checkpoint and building the network),
    timed once per seed, kept separate from inference timing.
  - Per-image inference time: the sliding-window forward pass only
    (predict_logits_from_preprocessed_data), bracketed with
    torch.cuda.synchronize() before and after each image when running on
    GPU. Preprocessing (file I/O, resampling, normalization) and export
    are excluded, since they run on CPU regardless of --device and would
    otherwise dilute the measurement.
  - A warm-up pass over the first N images (discarded) before timing, so
    the one-time host-to-device weight transfer and cuDNN autotuning are
    not counted as inference latency. Those same images are re-timed
    afterward as part of the full measured set.

This does not modify or duplicate the per-checkpoint batch prediction
already performed by nnUNetv2_predict in run_preliminary_milestones_test.sh.
That measures end-to-end throughput (including preprocessing and export)
per checkpoint. This script isolates GPU (or CPU) compute latency for a
single representative checkpoint per seed, since forward-pass latency
depends on architecture and input size, not on which weights are loaded.

Usage (on the rented GPU instance, before it is released):
  python training/benchmark_gpu_inference.py \\
    --nnunet-raw "$nnUNet_raw" \\
    --dataset-name Dataset001_AUL \
    --dataset-id 1 \
    --seeds 42 43 44 \
    --trainer-prefix nnUNetTrainerMilestones_seed \
    --checkpoint checkpoint_final.pth \
    --device cpu \
    --output-dir logs/inference

To repeat later on a different machine with the same checkpoints, images,
and inference settings (only --device and --output-dir change):
  python training/benchmark_gpu_inference.py \\
    --nnunet-raw /path/to/nnUNet_raw \\
    --dataset-name Dataset001_AUL \\
    --dataset-id 1 \\
    --seeds 42 43 44 \\
    --trainer-prefix nnUNetTrainerMilestones_seed \\
    --checkpoint checkpoint_final.pth \\
    --device cpu \\
    --output-dir logs/inference_cpu
"""

import argparse
import csv
import json
import time
import statistics
from pathlib import Path

import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def spread_stats(values):
    n = len(values)
    result = {
        "n_images": n,
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "stdev_seconds": statistics.stdev(values) if n > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }
    if n >= 4:
        q1, _, q3 = statistics.quantiles(values, n=4)
        result["p25_seconds"] = q1
        result["p75_seconds"] = q3
    else:
        result["p25_seconds"] = result["median_seconds"]
        result["p75_seconds"] = result["median_seconds"]
    return result


def format_row(d):
    return {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in d.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Per-image GPU/CPU inference benchmark for trained "
                    "nnU-Net checkpoints.")
    parser.add_argument("--nnunet-raw", required=True,
                        help="Path to nnUNet_raw (contains the dataset's imagesTs)")
    parser.add_argument("--dataset-name", default="Dataset001_AUL")
    parser.add_argument("--dataset-id", default="1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--trainer-prefix", default="nnUNetTrainerMilestones_seed",
                        help="Trainer class name prefix; seed is appended")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoint_final.pth",
                        help="Checkpoint filename benchmarked for every seed. "
                             "Forward-pass latency does not depend on which "
                             "weights are loaded, only on architecture and "
                             "input size, so one checkpoint per seed is "
                             "sufficient to characterize inference time.")
    parser.add_argument("--step-size", type=float, default=0.5,
                        help="Sliding-window step size (matches nnUNetv2_predict default)")
    parser.add_argument("--disable-mirroring", action="store_true",
                        help="Disable test-time mirroring augmentation "
                             "(nnUNetv2_predict default is enabled)")
    parser.add_argument("--warmup-images", type=int, default=3,
                        help="Number of images run before timing starts, to "
                             "absorb one-time device transfer and cuDNN "
                             "autotuning cost. These images are re-timed "
                             "afterward as part of the full measured set.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Use 'cuda' on the rented GPU instance; use "
                             "'cpu' to repeat the identical benchmark later "
                             "on a different machine.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for per-image CSV, summary CSV, and settings JSON")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA device requested but not available"

    # Match nnU-Net's own CLI thread configuration per device, so the
    # benchmark reflects the same settings nnUNetv2_predict would use.
    try:
        if device.type == "cpu":
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
        else:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images_dir = Path(args.nnunet_raw) / args.dataset_name / "imagesTs"
    case_lists = create_lists_from_splitted_dataset_folder(str(test_images_dir), ".png")
    case_ids = [Path(c[0]).name[:-len("_0000.png")] for c in case_lists]
    print(f"Found {len(case_lists)} test images in {test_images_dir}")

    per_image_rows = []
    summary_rows = []
    all_gpu_seconds = []

    for seed in args.seeds:
        trainer_name = f"{args.trainer_prefix}{seed}"
        print(f"\n=== Seed {seed} (trainer {trainer_name}, checkpoint {args.checkpoint}) ===")

        predictor = nnUNetPredictor(
            tile_step_size=args.step_size,
            use_gaussian=True,
            use_mirroring=not args.disable_mirroring,
            perform_everything_on_device=(device.type == "cuda"),
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )

        model_folder = get_output_folder(
            args.dataset_id, trainer_name, args.plans, args.configuration)

        sync(device)
        load_start = time.perf_counter()
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=(args.fold,), checkpoint_name=args.checkpoint)
        sync(device)
        load_seconds = time.perf_counter() - load_start
        print(f"Model load time (checkpoint read + state dict load): {load_seconds:.4f}s")

        # Preprocess every case up front. This runs on CPU regardless of
        # --device and is not part of the timed inference measurement; it
        # mirrors what nnUNetv2_predict does per case before the forward pass.
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
        preprocessed = []
        for file_list, case_id in zip(case_lists, case_ids):
            data, _, _ = preprocessor.run_case(
                file_list, None, predictor.plans_manager,
                predictor.configuration_manager, predictor.dataset_json)
            preprocessed.append((case_id, torch.from_numpy(data)))

        # Warm-up: absorb one-time host-to-device weight transfer and
        # cuDNN autotuning. Discarded, not written to the CSV.
        n_warmup = min(args.warmup_images, len(preprocessed))
        print(f"Warm-up: {n_warmup} image(s) (discarded)...")
        for _, data in preprocessed[:n_warmup]:
            sync(device)
            _ = predictor.predict_logits_from_preprocessed_data(data)
            sync(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Timed inference: every image in the test set, including the
        # ones used for warm-up (now measured under steady-state conditions).
        seed_seconds = []
        for case_id, data in preprocessed:
            sync(device)
            t0 = time.perf_counter()
            _ = predictor.predict_logits_from_preprocessed_data(data)
            sync(device)
            elapsed = time.perf_counter() - t0
            seed_seconds.append(elapsed)
            all_gpu_seconds.append(elapsed)
            per_image_rows.append({
                "seed": seed,
                "checkpoint": args.checkpoint,
                "device": args.device,
                "image_id": case_id,
                "inference_seconds": f"{elapsed:.6f}",
            })

        stats = spread_stats(seed_seconds)
        print(f"Seed {seed}: n={stats['n_images']}, "
              f"median={stats['median_seconds']:.4f}s, "
              f"mean={stats['mean_seconds']:.4f}s, "
              f"stdev={stats['stdev_seconds']:.4f}s, "
              f"IQR=[{stats['p25_seconds']:.4f}, {stats['p75_seconds']:.4f}]s, "
              f"range=[{stats['min_seconds']:.4f}, {stats['max_seconds']:.4f}]s")

        summary_rows.append(format_row({
            "seed": seed,
            "checkpoint": args.checkpoint,
            "device": args.device,
            "model_load_seconds": load_seconds,
            **stats,
        }))

        del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_gpu_seconds:
        overall = spread_stats(all_gpu_seconds)
        print(f"\n=== Overall (all seeds, n={overall['n_images']}) ===")
        print(f"median={overall['median_seconds']:.4f}s, mean={overall['mean_seconds']:.4f}s, "
              f"stdev={overall['stdev_seconds']:.4f}s, "
              f"IQR=[{overall['p25_seconds']:.4f}, {overall['p75_seconds']:.4f}]s")
        summary_rows.append(format_row({
            "seed": "all",
            "checkpoint": args.checkpoint,
            "device": args.device,
            "model_load_seconds": "",
            **overall,
        }))

    # --- Write per-image CSV ---
    per_image_csv = output_dir / f"inference_per_image_{args.device}.csv"
    with open(per_image_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "checkpoint", "device", "image_id", "inference_seconds"])
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"\nPer-image CSV: {per_image_csv}")

    # --- Write summary CSV ---
    summary_csv = output_dir / f"inference_summary_{args.device}.csv"
    summary_fields = ["seed", "checkpoint", "device", "model_load_seconds", "n_images",
                      "mean_seconds", "median_seconds", "stdev_seconds",
                      "p25_seconds", "p75_seconds", "min_seconds", "max_seconds"]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV: {summary_csv}")

    # --- Write settings for exact replication on another device ---
    import nnunetv2
    settings = {
        "nnunet_version": getattr(nnunetv2, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "dataset_name": args.dataset_name,
        "dataset_id": args.dataset_id,
        "trainer_prefix": args.trainer_prefix,
        "seeds": args.seeds,
        "plans": args.plans,
        "configuration": args.configuration,
        "fold": args.fold,
        "checkpoint": args.checkpoint,
        "step_size": args.step_size,
        "use_mirroring": not args.disable_mirroring,
        "use_gaussian": True,
        "warmup_images": args.warmup_images,
        "device": args.device,
        "test_image_count": len(case_lists),
        "notes": (
            "inference_seconds covers only predict_logits_from_preprocessed_data "
            "(the sliding-window forward pass), bracketed with "
            "torch.cuda.synchronize() when device=cuda. Preprocessing "
            "(file I/O, resampling, normalization) and export are excluded. "
            "Model loading time is measured separately and excludes the "
            "one-time host-to-device weight transfer, which occurs during "
            "the discarded warm-up pass. To replicate on another device, "
            "rerun this script with the same --checkpoint, --dataset-name, "
            "--seeds, --step-size, and --disable-mirroring settings recorded "
            "here, changing only --device and --output-dir."
        ),
    }
    settings_path = output_dir / f"inference_settings_{args.device}.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Settings: {settings_path}")


if __name__ == "__main__":
    main()
