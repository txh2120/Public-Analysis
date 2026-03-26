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
Using Reinforcement Learning to search for counterexamples to open mathematical conjectures on Laplacian spectral radius bounds.

| | |
|---|---|
| **Method** | Deep Cross-Entropy Method (CEM), Ghebleh 2024 settings (Adam lr=0.003, CrossEntropyLoss, adaptive randomness, batch_size 200) |
| **Target** | Laplacian spectral radius upper bounds (Brankov-Hansen-Stevanovic 2006) — 36 open bounds |
| **Architecture** | MLP (128→64→4→2), Adam optimizer |
| **Benchmark** | Bound 3 (disproved) — **5/7 counterexamples found** (n=12,14,16,18,20) |
| **Open Bounds** | Bounds 1, 4, 9, 33 — **0/28 counterexamples** (empirical support for bounds at n≤20) |
| **Best Result** | Bound 3, n=16: reward +0.256 (μ exceeded bound by 0.256) |
| **Status** | **Completed** — 35 runs, ~10.4 hours on CPU |
| **Script** | [src/rl_graph_conjecture_v2.py](src/rl_graph_conjecture_v2.py) |
| **Notebooks** | [v1](notebooks/rl_graph_conjecture.ipynb), [v2](notebooks/rl_graph_conjecture_v2.ipynb) |
| **References** | [Wagner 2021](https://arxiv.org/abs/2104.14516), [Ghebleh et al. 2024](https://arxiv.org/abs/2403.18429), Brankov-Hansen-Stevanovic 2006 |

<details>
<summary>Benchmark Results (Bound 3, n=8-20)</summary>

| n | Best Reward | Counterexample | Iterations | Time |
|---|-------------|----------------|------------|------|
| 8 | 0.000 | No | 5000 | 525s |
| 10 | 0.000 | No | 5000 | 745s |
| 12 | +0.026 | **Yes** | 325 | 26s |
| 14 | +0.003 | **Yes** | 1552 | 408s |
| 16 | +0.256 | **Yes** | 815 | 115s |
| 18 | +0.030 | **Yes** | 2807 | 1236s |
| 20 | +0.179 | **Yes** | 2267 | 550s |

</details>

## Environment

- **Framework:** PyTorch
- **Datasets:** Kaggle, public medical/healthcare datasets
- **Training:** Kaggle Notebooks (GPU/CPU)
