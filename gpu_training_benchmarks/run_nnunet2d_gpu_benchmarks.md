# Study GPU Benchmarks on Verda 

We implemented nnU-Net's 2D benchmark using the study's AUL dataset with our 85/15 split yielding 625 training images (static greyscale images, mean dimensions of ~512x700px), `nnUNetPlans` configuration, fold 0, and PyTorch version on each instance.

## Benchmarked GPU cost and performance per 1000 epochs

| GPU | Cost | Spot cost | Training time | Cost per epoch |
|---|---|---|---|---|
|  RTX 6000 Ada | $10.12 | $5.06 | 9.20 hours | $0.01012 |
|  RTX PRO 6000 | $10.64 | $5.32 | 5.72 hours | $0.01064 |
|  L40S | $13.87 | $6.94 | 9.54 hours | $0.01387 |

These figures don't account for setup, predictions or total wall clock - they apply only to training compute.

## 1.  RTX 6000 Ada

1,000-epoch cost:
```text
$0.01012 × 1000 = $10.12
```

1,000-epoch training time:
```text
33.1199 × 1000 ÷ 3600 = 9.20 hours
```

Cost per epoch:
```text
33.1199 × 1.10 ÷ 3600 = $0.01012
```

GPU: NVIDIA RTX 6000 Ada Generation
VRAM: 48 GiB
PyTorch: 2.14.0+cu130
Hourly price: $1.10 USD
Fastest epoch: 33.1199 seconds




## 2.  L40S

1,000-epoch cost:
```text
$0.01387 × 1000 = $13.87
```

1,000-epoch training time:
```text
34.3568 × 1000 ÷ 3600 = 9.54 hours
```

Cost per  epoch:
```text
34.3568 × 1.453 ÷ 3600 = $0.01387
```

GPU: NVIDIA L40S
VRAM: 48 GiB
PyTorch: 2.14.0+cu130
Hourly price: $1.453 USD
Fastest epoch: 34.3568 seconds

## 3.  RTX PRO 6000

1,000-epoch cost:
```text
$0.01064 × 1000 = $10.64
```

1,000-epoch training time:
```text
20.6011 × 1000 ÷ 3600 = 5.72 hours
```

Cost per epoch:
```text
20.6011 × 1.86 ÷ 3600 = $0.01064
```

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM: 96 GiB
PyTorch: 2.14.0+cu130
Hourly price: $1.86 USD
Fastest epoch: 20.6011 seconds
