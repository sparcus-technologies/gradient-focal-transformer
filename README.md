# Gradient Focal Transformer
## Abstract 

Fine-Grained Image Classification (FGIC) remains a complex task in computer vision, as it requires models to distinguish between categories with subtle localized visual differences. Well-studied CNN-based models, while strong in local feature extraction, often fail to capture the global context required for fine-grained recognition, while more recent ViT-backboned models address FGIC with attention-driven mechanisms but lack the ability to adaptively focus on truly discriminative regions. TransFG and other ViT-based extensions introduced part-aware token selection to enhance attention localization, yet they still struggle with computational efficiency, attention region selection flexibility, and detail-focus narrative in complex environments. This paper introduces GFT (Gradient Focal Transformer), a new ViT-derived framework created for FGIC tasks. GFT integrates the Gradient Attention Learning Alignment (GALA) mechanism to dynamically prioritize class-discriminative features by analyzing attention gradient flow. Coupled with a Progressive Patch Selection (PPS) strategy, the model progressively filters out less informative regions, reducing computational overhead while enhancing sensitivity to fine details. GFT achieves SOTA accuracy on FGVC Aircraft, Food-101, and COCO datasets with 93M parameters, outperforming ViT-based advanced FGIC models in efficiency. By bridging global context and localized detail extraction, GFT sets a new benchmark in fine-grained recognition, offering interpretable solutions for real-world deployment scenarios. 

## Preprint Link: https://arxiv.org/abs/2504.09852 

## Overview

This repository contains a comprehensive collection of deep learning models used as code accompaniment for GFT research work, evaluated across three popular datasets:

- **COCO**: Common Objects in Context - a large-scale object detection dataset
- **FGVC-Aircraft**: Fine-Grained Visual Classification of Aircraft
- **Food101**: 101 food categories with 101,000 images

Each model implementation is provided as a standalone Python file.

## Models

| Model Architecture | COCO | FGVC-Aircraft | Food101 |
|-------------------|------|---------------|---------|
| DeiT              | ✅    | ✅             | ✅       |
| DenseNet169       | ✅    | ✅             | ✅       |
| GFT               | ✅    | ✅             | ✅       |
| ResNet18          | ✅    | ✅             | ✅       |
| ResNetV2-101x3-BiTM | ✅  | ✅             | ✅       |
| TransFG           | ✅    | ✅             | ✅       |
| ViT               | ✅    | ✅             | ✅       |

## Model Descriptions

### DeiT (Data-efficient image Transformers)
- Transformers optimized for image classification with less data
- Files: `deit_coco.py`, `deit_fgvcaircraft.py`, `deit_food101.py`

### DenseNet169
- CNN with dense connectivity pattern to strengthen feature propagation
- Files: `densenet169_coco.py`, `densenet169_fgvcaircraft.py`, `densenet169_food101.py`

### GFT (Gradient Focal Transformer)
- Transformer architecture with global and local feature extraction capabilities
- Files: `gft_coco.py`, `gft_fgvcaircraft.py`, `gft_food101.py`

### ResNet18
- Residual Network with 18 layers for efficient image classification
- Files: `resnet18_COCO.py`, `resnet18_FGVCAircraft.py`, `resnet18_Food101.py`

### ResNetV2-101x3-BiTM
- Big Transfer Model based on ResNetV2 architecture
- Files: `resnetv2_101x3_bitm_coco.py`, `resnetv2_101x3_bitm_fgvcaircraft.py`, `resnetv2_101x3_bitm_food101.py`

### TransFG (Transformers for Fine-Grained Recognition)
- Specialized transformer architecture for fine-grained visual recognition
- Files: `transfg_coco.py`, `transfg_fgvcaircraft.py`, `transfg_food101.py`

### ViT (Vision Transformer)
- Pure transformer architecture for image classification
- Files: `vit_coco.py`, `vit_fgvcaircraft.py`, `vit_food101.py`

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- torchvision
- Other dependencies based on specific model requirements

## Recent Updates
The repository has been actively maintained.

## Contributions
* Kriuk Boris, Hong Kong University of Science and Technology
* Simranjit Kaur Gill, University of Westminster
* Shoaib Aslam, University of Engineering and Technology Lahore Pakistan
* Fakhrutdinov Amir, Shanghai Jiao Tong University

