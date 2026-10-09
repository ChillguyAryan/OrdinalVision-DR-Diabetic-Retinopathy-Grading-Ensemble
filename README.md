# OrdinalVision-DR: Diabetic Retinopathy Grading Ensemble

A deep learning system for **5-class diabetic retinopathy grading** from retinal fundus photographs. The project combines **ResNet50, EfficientNet-B0, and ViT-Tiny** and evaluates the ensemble on **EyePACS** and **APTOS 2019**.

## Overview

The model predicts diabetic retinopathy across five ordered grades:

| Grade | Severity |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

Since the grades are ordered, **Quadratic Weighted Kappa (QWK)** is used as the primary metric. The system also evaluates **referable diabetic retinopathy**, defined as grade ≥ 2.

### Pipeline

```text
Retinal Fundus Image
        │
        ▼
    Preprocessing
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
    ResNet50      EfficientNet-B0   ViT-Tiny
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                    Ensemble
                       │
                       ▼
              Final DR Prediction
                       │
                       ▼
                    Grade 0–4
```

## Datasets

### EyePACS

The project uses the **Kaggle Diabetic Retinopathy Detection** dataset for model development and evaluation.

A reproducible subset is used with **patient-level splitting** to prevent images from the same patient from appearing across different splits.

EyePACS is used for:

- Training
- Validation
- Model selection
- Held-out testing

### APTOS 2019

The **APTOS 2019 Blindness Detection** dataset is used for external evaluation.

APTOS is kept separate from the training and model-selection process and is used to evaluate the trained models on a different dataset.

## Models

Three ImageNet-pretrained vision models are used:

| Model | Architecture |
|---|---|
| ResNet50 | Convolutional Neural Network |
| EfficientNet-B0 | Convolutional Neural Network |
| ViT-Tiny | Vision Transformer |

The models are partially fine-tuned rather than trained from scratch.

### Fine-tuned Layers

| Model | Fine-tuned layers |
|---|---|
| ResNet50 | `layer2`–`layer4`, `fc` |
| EfficientNet-B0 | `blocks[3:7]`, `conv_head`, `bn2`, `classifier` |
| ViT-Tiny | Blocks 4–11, `norm`, `head` |

## Preprocessing

The fundus images undergo:

- Fundus cropping
- Square padding
- Resize to **384 × 384**
- Ben Graham contrast normalization

Training images additionally use:

- Random rotation
- Flipping
- Brightness variation
- Contrast variation
- Random cropping

## Training

Each model starts from ImageNet-pretrained weights with a new 5-class classification head.

| Setting | Value |
|---|---|
| Image size | 384 × 384 |
| Optimizer | AdamW |
| Loss | Cross-Entropy |
| Batch size | 64 |
| Mixed precision | FP16 |
| Backbone learning rate | 2e-4 |
| Head learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | Warm-up + cosine decay |
| Class weighting | 1 / √class frequency |

Class weighting is used to reduce the effect of class imbalance during training.

The checkpoint with the highest validation QWK is selected for each model.

## Ensemble

The predictions from **ResNet50, EfficientNet-B0, and ViT-Tiny** are combined to produce the final 5-class prediction.

### Ensemble Performance

| Metric | EyePACS | APTOS 2019 |
|---|---:|---:|
| **QWK** | **0.7667** | **0.8418** |
| **Referable DR Sensitivity** | **0.7143** | **0.9960** |
| **Referable DR AUC** | **0.9220** | **0.9728** |
| **ECE** | **0.0349** | **0.1228** |
| **NLL** | **0.6043** | **0.9625** |
| **Brier Score** | **0.3055** | **0.4330** |

## Evaluation Metrics

### QWK

**Quadratic Weighted Kappa (QWK)** is the primary metric for the 5-class grading task. It accounts for the ordered nature of diabetic retinopathy grades and penalizes larger disagreements more heavily.

### Referable DR

Referable diabetic retinopathy is defined as:

```text
Grade ≥ 2
```

The binary evaluation includes:

- Sensitivity
- Specificity
- ROC-AUC

### Additional Metrics

The notebook evaluates:

- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix
- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- Brier score
- Reliability diagrams

## Tech Stack

- Python
- PyTorch
- timm
- Scikit-learn
- OpenCV
- NumPy
- Pandas
- Matplotlib