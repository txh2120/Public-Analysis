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
| **Methods** | CEM (Deep Cross-Entropy), Exhaustive enumeration (nauty-geng), AMCS (Adaptive Monte Carlo Search), Tabu Search, SDP relaxation, Structural Construction |
| **Key Finding** | **5 of 38 bounds violated** via structural construction — first counterexamples after 7 methods found none |
| **Violated Bounds** | Bound 11 (StarOfCliques), Bound 13 (StarOfCliques), Bound 40 (Windmill), Bound 45 (DoubleStar/Caterpillar), Bound 48 (P3, degenerate) |
| **Remaining 33** | 0 violations across 50,000+ test graphs — classified: 8 SUPER_SAFE, 20 SAFE_LINEAR, 2 MARGINAL, 3 TIGHT |
| **Benchmark** | Bound 3 (known disproved) — **5/7 counterexamples found** via CEM (n=12,14,16,18,20) |
| **Status** | **Completed** — 7 methods, exhaustive to n=13, heuristic to n=200, structural families to n=172 |
| **Scripts** | [CEM](src/rl_graph_conjecture_v2.py), [Exhaustive](src/exhaustive_bound_search.py), [AMCS](src/amcs_bound_search.py), [Tabu](src/tabu_bound_search.py), [Structural](src/structural_counterexample_search.py), [SDP](src/sdp_bound_analysis.py) |
| **References** | [Wagner 2021](https://arxiv.org/abs/2104.14516), [Ghebleh et al. 2024](https://arxiv.org/abs/2403.18429), Brankov-Hansen-Stevanovic 2006 |

<details>
<summary>Methods and Results</summary>

#### Phase 1: Computational Search (0 counterexamples)

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

**5. SDP Relaxation** — Weighted spectral optimization + rounding analysis
- SDP weighted optimization finds near-violations in continuous domain
- Rounding to discrete graphs eliminates all violations (rounding slack mechanism)
- Key insight: continuous optimum lies in non-graphical region of the polytope

#### Phase 2: Structural Construction (5 bounds violated)

**6. Structural Counterexample Search** — Parameterized graph families targeting bound weaknesses
- Insight: high-degree hub + low-degree periphery creates extreme d/m imbalance that exposes formula weaknesses
- 1,100+ structured graphs tested across 6+ families (StarOfCliques, DoubleStar, Caterpillar, Windmill, MultiHub, etc.)
- **111 total counterexamples found** across 5 bounds:

| Bound | Graph Family | Smallest n | Best Gap | Violations |
|-------|-------------|-----------|----------|------------|
| 11 (2m³/d²) | StarOfCliques(K_m, t) | 29 | -1.366 | 30 |
| 13 (2m⁴/d³) | StarOfCliques(K_m, t) | 46 | -0.840 | 18 |
| 40 (2+√...) | Windmill(k, t) | 91 | -1.790 | — |
| 45 (2+√...) | DoubleStar / Caterpillar | 14 | -0.165 | 62 |
| 48 (rational) | P3 only (degenerate) | 3 | -0.071 | 1 |

- Why previous methods missed these: DoubleStar(6,6) at n=14 has max degree 7, outside subquartic exhaustive range (n≤13); CEM/AMCS/Tabu converge to regular graph attractors where all bounds are tight

#### Remaining 33 Bounds: Classification

Mathematical analysis of all 33 non-violated bounds under hub-periphery evaluation:

| Category | Count | Bounds | Description |
|----------|-------|--------|-------------|
| SUPER_SAFE | 8 | 1,4,5,7,14,16,21,30 | Growth O(k^1.5 to k^4) at hub — always far above μ |
| SAFE_LINEAR | 20 | 6,8,9,12,18-20,22,24-27,34,35,37-39,44,46,56 | Growth c·k with c > 1 — robust linear margin |
| MARGINAL | 2 | 10,23 | Gap converges to +0.5 on Star(k) — narrow but positive |
| TIGHT | 3 | 33,42,47 | Exact equality (gap=0) on all complete bipartite K_{a,b} — no violations found on any tested graph |

- 50,000+ graphs tested across 18+ structural families: zero violations in remaining 33 bounds
- Bounds 33, 42, 47 achieve exact equality on semiregular bipartite graphs via algebraic identity

</details>

## Environment

- **Framework:** PyTorch
- **Datasets:** Kaggle, public medical/healthcare datasets
- **Training:** Kaggle Notebooks (GPU/CPU)
