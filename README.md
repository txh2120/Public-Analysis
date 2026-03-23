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

### 2. Breast Cancer Histopathology (IDC) Classification
Binary classification of Invasive Ductal Carcinoma using Transfer Learning.

| | |
|---|---|
| **Dataset** | [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) (277,524 patches, 50x50 RGB) |
| **Classes** | IDC Negative vs IDC Positive (imbalanced ~1:2.5) |
| **Model** | ResNet18 (ImageNet pretrained, fine-tuned layer4) |
| **Techniques** | Transfer Learning, Data Augmentation, WeightedRandomSampler |
| **Test Accuracy** | **86.4%** |
| **AUC-ROC** | **0.939** |
| **Notebook** | [notebooks/02_breast_cancer_classification.ipynb](notebooks/02_breast_cancer_classification.ipynb) |
| **Training Script** | [src/breast_cancer_train.py](src/breast_cancer_train.py) |

### 3. RL Graph Conjecture Counterexample Search
Using Reinforcement Learning to search for counterexamples to open mathematical conjectures.

| | |
|---|---|
| **Method** | Deep Cross-Entropy Method (Wagner 2021) ported to PyTorch |
| **Target** | Laplacian spectral radius upper bounds (Brankov-Hansen-Stevanovic 2006) |
| **Architecture** | MLP (128→64→4→1), SGD, BCELoss |
| **Benchmark** | Bound 2 (disproved) — reproduced counterexample discovery |
| **Attack** | 38 open Laplacian bounds (Bound 1, 4, 9, 33) |
| **Status** | Kaggle execution pending |
| **Notebook** | [notebooks/rl_graph_conjecture.ipynb](notebooks/rl_graph_conjecture.ipynb) |
| **References** | [Wagner 2021](https://arxiv.org/abs/2104.14516), [Ghebleh et al. 2024](https://arxiv.org/abs/2403.18429) |

## Environment

- **Framework:** PyTorch
- **Datasets:** Kaggle, public medical/healthcare datasets
- **Training:** Kaggle Notebooks (GPU/CPU)
