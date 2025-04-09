# Gradient Focal Transformer
## Abstract 

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

### Usage Example

```python
# For using DeiT with COCO dataset
from deit_coco import DeiTCOCO

# Initialize the model
model = DeiTCOCO(pretrained=True)

# Process an image
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open('path_to_image.jpg')
img_tensor = transform(img).unsqueeze(0)

# Get predictions
with torch.no_grad():
    predictions = model(img_tensor)

```

## Recent Updates
The repository has been actively maintained with recent updates to several models:

## Contributions
* Kriuk Boris, Hong Kong University of Science and Technology
* Simranjit Kaur Gill, University of Westminster
* Shoaib Aslam, University of Engineering and Technology Lahore Pakistan
* Fakhrutdinov Amir, Shanghai Jiao Tong University



## Acknowledgements
The model implementations are based on research from various papers and repositories.

