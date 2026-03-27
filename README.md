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

### 3. Graph Conjecture Counterexample Search
Computational search for counterexamples to 38 open BHS (Brankov-Hansen-Stevanovic 2006) upper bounds on the Laplacian spectral radius μ(G).

| | |
|---|---|
| **Target** | Laplacian spectral radius upper bounds — 38 open bounds |
| **Methods** | CEM (Deep Cross-Entropy), Exhaustive enumeration (nauty-geng), AMCS (Adaptive Monte Carlo Search), Tabu Search (edge-flip neighborhood) |
| **Key Finding** | All 38 open bounds resist computational attack — **0 counterexamples found** |
| **Closest Bound** | Bound 44: gap = +0.0096 (exhaustive n≤13, 68.5M graphs), gap = +0.053 (Tabu Search n=15) |
| **Benchmark** | Bound 3 (known disproved) — **5/7 counterexamples found** via CEM (n=12,14,16,18,20) |
| **Status** | **Completed** — 4 methods, exhaustive to n=13, heuristic to n=50 |
| **Scripts** | [CEM](src/rl_graph_conjecture_v2.py), [Exhaustive](src/exhaustive_bound_search.py), [AMCS](src/amcs_bound_search.py), [Tabu](src/tabu_bound_search.py) |
| **References** | [Wagner 2021](https://arxiv.org/abs/2104.14516), [Ghebleh et al. 2024](https://arxiv.org/abs/2403.18429), Brankov-Hansen-Stevanovic 2006 |

<details>
<summary>Methods and Results</summary>

**1. Deep Cross-Entropy Method (CEM)** — MLP (128→64→4→2), Ghebleh 2024 settings
- Benchmark Bound 3: 5/7 counterexamples at n=12,14,16,18,20
- Open bounds: 0 counterexamples at n≤20 (35 runs, ~10.4 hours)
- Limitation: converges to regular graph attractors

**2. Exhaustive Enumeration** — nauty-geng via WSL, subquartic graphs
- n=5 to n=13: 68.5M connected graphs tested
- All 38 bounds hold with positive gap
- Bound 44 closest: gap = +0.0096

**3. Adaptive Monte Carlo Search (AMCS)** — Nested rollout + shrink/grow
- Multi-restart search with adaptive depth escalation
- Trapped at score=0 on small regular graphs (tight case attractor)

**4. Tabu Search** — Edge-flip neighborhood, multi-start with diverse densities
- Avoids scratch-build trap by modifying existing graphs
- Best result: Bound 44 gap = +0.053 at n=15 (30 min, 5 restarts, 8920 iterations)
- Structural analysis: edge-max bounds create robust barrier against violation

</details>

## Environment

- **Framework:** PyTorch
- **Datasets:** Kaggle, public medical/healthcare datasets
- **Training:** Kaggle Notebooks (GPU/CPU)
