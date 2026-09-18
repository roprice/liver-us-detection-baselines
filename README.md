# Liver ultrasound detection baselines

Resource-efficiency baselines for liver and malignant-mass detection on
B-mode ultrasound using nnU-Net.

This study trains PlainConvUNet 2D on the [Annotated Ultrasound Liver
(AUL) dataset](https://doi.org/10.5281/zenodo.7272660) and evaluates
segmentation, overlap-based detection, and triage-level flagging across
nested training-set sizes. The preliminary experiment (see
`reproduce_study/`) determines the epoch budget and primary detection
metric before the full sweep.

## Repository structure

```
training/                  Training pipeline
  convert_aul.py             Convert AUL to nnU-Net format
  benchmark_gpu_inference.py Per-image GPU/CPU inference benchmark
  run_preliminary_1000_epochs.sh  1,000-epoch preliminary experiment
  custom_trainers/           Custom nnU-Net trainers
reproduce_study/            Setup and reproduction guides
```

## License

Apache 2.0. See `LICENSE`.
