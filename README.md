# Food-101 Classification: Parameter-Efficient Fine-Tuning & Robustness Analysis

This project explores various **Parameter-Efficient Fine-Tuning (PEFT)** techniques for food classification using the **Food-101** dataset. We compare several adaptation methods across two architectures—**ResNet18** and **EfficientNetV2-S**—and evaluate their performance not only on clean data but also their **robustness** to common image corruptions.

## 🚀 Project Highlights

- **Comprehensive PEFT Comparison**: Evaluated LoRA, Task-Specific Adapters, BatchNorm Tuning, and Linear Probing.
- **Custom Architecture**: Developed a **Custom CBAM Adapter** (Convolutional Block Attention Module) that integrates spatial and channel attention for improved feature extraction.
- **Robustness Benchmarking**: Tested models against 6 different scenarios: Clean, Masked, Noise/Rotation, Blur (Little/Medium), and Downsampling.
- **Interpretability**: Leveraged **Grad-CAM** visualizations to understand where models focus during classification.
- **Efficiency Metrics**: Detailed analysis of accuracy vs. trainable parameters and inference throughput.

## 🛠️ Implemented Methods

| Method | Description |
| :--- | :--- |
| **Linear Probing** | Fine-tuning only the final classification head (Baseline). |
| **BatchNorm Tuning** | Tuning only the affine parameters (scale/shift) of BatchNorm layers. |
| **LoRA** | Low-Rank Adaptation of convolutional layers (using `loralib`). |
| **Task Adapters** | Custom 1x1 bottleneck convolution layers inserted into the backbone. |
| **Custom CBAM** | A specialized adapter combining Spatial and Channel attention mechanisms. |

## 📊 Key Results

Our analysis reveals that **BatchNorm Tuning** and **LoRA** offer the best trade-off between accuracy and parameter efficiency.

| Backbone | Best Method | Mean Accuracy | Clean Accuracy | Trainable Params |
| :--- | :--- | :--- | :--- | :--- |
| **EfficientNetV2-S** | BatchNorm Tuning | **71.0%** | **86.5%** | 283k |
| **EfficientNetV2-S** | LoRA | 66.8% | 84.9% | 1.51M |
| **ResNet18** | Custom CBAM | **57.1%** | **69.6%** | 194k |
| **ResNet18** | BatchNorm Tuning | 52.1% | 67.9% | 51k |

> **Note**: Mean accuracy is averaged across all robustness test sets. The **Custom CBAM** model showed the highest robustness gains for the ResNet18 backbone.

## 📂 Project Structure

```text
├── efficientnetv2_training/ # Training notebooks for EfficientNetV2-S
├── resnet18_training/      # Training notebooks for ResNet18
├── model_weights/           # Saved checkpoints (.pt)
├── analysis.ipynb           # Comparative analysis and plotting
├── grad_cam_visualization.ipynb # Interpretability analysis
├── helpers.py               # Shared utilities for models and data loading
├── combined_results.csv     # Raw data for all experiments
└── README.md                # Project documentation
```

## ⚙️ Setup & Usage

### Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib pandas seaborn loralib
   ```

### Training
Each method has a dedicated notebook in the `resnet18_training` or `efficientnetv2_training` folders. Simply open and run the cells to reproduce the results.

### Evaluation & Analysis
Run `analysis.ipynb` to generate comparison charts and summary tables from `combined_results.csv`.

## 🖼️ Visualizations (Grad-CAM)
The `grad_cam_visualization.ipynb` notebook provides visual proof of the models' learning, highlighting relevant food features (e.g., textures of pizza toppings, structure of burgers) that drive the classification decisions.

---
*Developed as part of the MDS project at UBC.*
