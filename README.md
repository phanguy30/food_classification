# Parameter-Efficient Fine-Tuning and Robustness on Food-101

A benchmark of five fine-tuning methods for image classification on Food-101, measuring both clean accuracy and robustness to common image corruptions, with Grad-CAM to check what the models actually look at.

## Highlights

- **EfficientNetV2-S:** BatchNorm tuning reached 71.0% mean robustness accuracy while training only **283K parameters**.
- **ResNet18:** A custom CBAM adapter (spatial and channel attention) gave the largest robustness gains.
- Grad-CAM confirmed that gains came from relevant image regions, not spurious features.

## Methods

| Method | What is trained |
|--------|-----------------|
| Linear probing | Final classification head only (baseline) |
| BatchNorm tuning | Scale and shift parameters of BatchNorm layers |
| LoRA | Low-rank adapters on convolutional layers (`loralib`) |
| Task adapters | Custom 1x1 bottleneck convolution layers in the backbone |
| Custom CBAM | Custom adapter combining spatial and channel attention |

**Backbones:** ResNet18 and EfficientNetV2-S.
**Robustness tests:** Clean, Masked, Noise/Rotation, Blur (Little), Blur (Medium), Downsampling.

## Results

Mean accuracy is averaged across all robustness test sets.

| Backbone | Best method | Mean acc. | Clean acc. | Trainable params |
|----------|-------------|-----------|------------|------------------|
| EfficientNetV2-S | BatchNorm tuning | 71.0% | 86.5% | 283K |
| EfficientNetV2-S | LoRA | 66.8% | 84.9% | 1.51M |
| ResNet18 | Custom CBAM | 57.1% | 69.6% | 194K |
| ResNet18 | BatchNorm tuning | 52.1% | 67.9% | 51K |

BatchNorm tuning and LoRA gave the best balance of accuracy and parameter efficiency. Full results for every method are in `combined_results.csv`, and the write-up is in `Project_Report.pdf`.

## Repository structure

```
efficientnetv2_training/      training notebooks for EfficientNetV2-S (one per method)
resnet18_training/            training notebooks for ResNet18 (one per method)
analysis.ipynb                comparison analysis and plots
grad_cam_visualization.ipynb  interpretability analysis
helpers.py                    shared model and data-loading utilities
combined_results.csv          raw results for all experiments
Project_Report.pdf            full write-up
model_weights/                saved checkpoints (not tracked in git)
```

## Run it

1. Clone the repository and install `torch`, `torchvision`, `matplotlib`, `pandas`, `seaborn` and `loralib`.
2. Run the notebook for each method in `resnet18_training/` or `efficientnetv2_training/`.
3. Run `analysis.ipynb` to generate charts and tables from `combined_results.csv`.
4. Run `grad_cam_visualization.ipynb` for the Grad-CAM figures.
