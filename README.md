# Public Analysis

Deep Learning / Machine Learning practice projects using PyTorch and public datasets.

## Setup

```bash
pip install -r requirements.txt
```

## Structure

```
├── notebooks/     # Jupyter notebooks (main workspace)
├── src/           # Reusable utility modules
├── data/          # Local data (not tracked by git)
└── configs/       # Training configurations
```

## Projects

### 1. Medical MNIST Classification
6-class medical image classification using a CNN built from scratch.

| | |
|---|---|
| **Dataset** | [Medical MNIST](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) (58,954 images, 64x64 grayscale) |
| **Classes** | AbdomenCT, BreastMRI, ChestCT, CXR, Hand, HeadCT |
| **Model** | 3-layer CNN (Conv→ReLU→MaxPool) + FC classifier |
| **Test Accuracy** | **99.9%** |
| **Notebook** | [notebooks/01_medical_mnist_classification.ipynb](notebooks/01_medical_mnist_classification.ipynb) |
| **Training Script** | [src/medical_mnist_train.py](src/medical_mnist_train.py) |
| **Kaggle** | [Kaggle Notebook](https://www.kaggle.com/code/txh2120/notebook-medical-mnist) |

## Environment

- **Framework:** PyTorch
- **Datasets:** Kaggle, public medical/healthcare datasets
- **Training:** Kaggle Notebooks (GPU)
