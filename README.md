# Permutation-Invariant Deep Learning for Fraud Detection in Insurance Claims

> Benchmarking permutation-invariant deep architectures (Deep Sets, FT-Transformer, Tabular GCN) against gradient-boosted tree baselines on the Vehicle Insurance Fraud dataset.

---

## Overview

Tabular data is the natural habitat of gradient-boosted trees, and fraud detection is one of its hardest variants — heterogeneous features, severe class imbalance, and asymmetric error costs. This project asks a simple question:

> *On tabular fraud data, can permutation-invariant deep models match or beat the entrenched **XGBoost** baseline?*

To answer it, I built an end-to-end pipeline that benchmarks **seven models** — three classical baselines and four deep-learning architectures — on a real, imbalanced vehicle-insurance-fraud dataset, under a single shared training and evaluation protocol.

## Motivation

Insurance fraud costs roughly **\$308 B** per year in the United States alone, and the cost of mistakes is asymmetric: missing fraud is expensive, but flagging an honest customer is just as costly. The data is **tabular**, **heterogeneous**, and **imbalanced** (≈ 6 % fraud), which is precisely the regime where tree ensembles dominate. The aim here is not to beat the world record but to *quantify* how far modern permutation-invariant deep models are from XGBoost on a representative tabular fraud problem.

## Dataset

**Vehicle Claim Fraud Detection** — Kaggle, [`shivamb/vehicle-claim-fraud-detection`](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection).

| Property            | Value          |
| ------------------- | -------------- |
| Rows (claims)       | 15,420         |
| Raw features        | 33             |
| Positive class      | 6.0 % (923)    |
| Negative class      | 94.0 % (14,497)|
| Task                | Binary classification |

Features cover policy information (month, policy type, deductible…), vehicle information (make, category, price, age…), and accident details (fault, accident area, police report filed, witness present…).

## Methodology

```
raw claims  →  clean & encode  →  stratified split  →  preprocessing  →  oversample (train only)  →  train  →  evaluate
```

**Preprocessing.** ID columns dropped; ordinal mapping for ranged strings (`VehiclePrice`, `AgeOfVehicle`, `AgeOfPolicyHolder`); binary mapping for yes/no fields. Two parallel preprocessing pipelines: *one-hot + StandardScaler* for the classical models, *OrdinalEncoder + learnable embeddings* for the deep models.

**Split.** Stratified 70 / 10 / 20 train / val / test, preserving the 6 % fraud rate in every split.

**Imbalance handling.** `RandomOverSampler` applied to the training set only, plus **Focal Loss** (α = 0.5, γ = 2.0) for the deep models.

**Training (deep models — identical for all four).** AdamW (lr = 1e-3, weight_decay = 1e-4), batch size 64, up to 20 epochs, early stopping with patience 5 on validation PR-AUC, decision threshold tuned on the validation set to maximize F1.

## Models

### Classical baselines
| Model                | Notes                                       |
| -------------------- | ------------------------------------------- |
| Logistic Regression  | Linear baseline, OneHot + L2 regularization |
| Random Forest        | 200 estimators                              |
| **XGBoost**          | Gradient-boosted trees — main baseline      |

### Deep-learning models
All four share a common tokenization frontend: every numerical feature is projected by a small `Linear(1, d)`, every categorical feature by an `Embedding(card, d)`, producing one *d*-dimensional token per feature.

| Model                  | Core mechanism                                   | Layers          |
| ---------------------- | ------------------------------------------------ | --------------- |
| Deep Sets              | mean-pool over feature tokens → MLP              | 0 (just pooling) + 2-layer MLP |
| FT-Transformer         | self-attention across feature tokens + `[CLS]`   | 2 encoder layers, 4 heads      |
| Tabular GCN (features) | learned soft adjacency over feature nodes        | 1 graph conv + 2-layer MLP     |
| Row GCN (samples)      | in-batch sample similarity as adjacency          | 1 graph conv + 2-layer MLP     |

## Results

Sorted by **PR-AUC** on the held-out test set (higher is better):

| Rank | Model                         | PR-AUC    | ROC-AUC   | F1      |
| ---: | ----------------------------- | :-------: | :-------: | :-----: |
| 1    | **XGBoost**                   | **0.197** | 0.806     | **0.270** |
| 2    | Random Forest                 | 0.186     | 0.795     | 0.254   |
| 3    | FT-Transformer                | 0.185     | **0.807** | 0.239   |
| 4    | Deep Sets                     | 0.165     | 0.797     | 0.237   |
| 5    | Tabular GCN (features)        | 0.150     | 0.783     | 0.207   |
| 6    | Row GCN (samples)             | 0.129     | 0.770     | 0.144   |
| 7    | Logistic Regression           | 0.125     | 0.774     | 0.144   |
| —    | Random Guess (prevalence)     | 0.060     | 0.500     | —       |

### Key findings

1. **XGBoost remains the strongest baseline** on tabular fraud — by a small but real margin.
2. **FT-Transformer is a credible deep alternative**, matching XGBoost on ROC-AUC (0.807 vs. 0.806) and lying within 0.012 PR-AUC.
3. **Both GCN variants underperform.** A learned dense adjacency over 37 features (or in-batch row similarity) provides weak structural priors when there is no real graph in the data.
4. **PR-AUC and threshold tuning are essential** when the positive class is only 6 % — ROC-AUC stays optimistic and accuracy is meaningless.

> **Bottom line.** On tabular fraud data, permutation-invariant deep learning is *viable, not yet superior*. Tree ensembles still set the bar; Transformer-style attention is the closest deep contender.

## Reproducibility

### Requirements

- Python ≥ 3.10
- A CUDA-capable GPU is helpful but not required (the deep models train comfortably on CPU at this scale).

### Setup

```bash
git clone https://github.com/<your-username>/fraud-detection-deep-learning.git
cd fraud-detection-deep-learning

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Data

The notebook fetches the dataset automatically via `kagglehub`. You will need a [Kaggle API token](https://www.kaggle.com/docs/api) configured (`~/.kaggle/kaggle.json`).

### Run

```bash
jupyter lab Fraud_Detection_in_Car_Insurance.ipynb
```

Run the cells top-to-bottom. The full pipeline (data download → preprocessing → training of all seven models → evaluation) takes roughly 5–10 minutes on a single GPU and 15–25 minutes on CPU.

Random seed is fixed to 42 in the first cell.

## Project structure

```
.
├── Fraud_Detection_in_Car_Insurance.ipynb   # End-to-end pipeline
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
└── LICENSE                                   # MIT
```

## Author

**Atieh Barati Nia** — PhD Student, Department of Data Science
New Jersey Institute of Technology

## License

This project is released under the [MIT License](LICENSE). The underlying dataset is distributed by its original author under its own Kaggle license; please consult the [dataset page](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) before redistribution.

## Acknowledgements

- [Khusheekapoor / shivamb on Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) for the Vehicle Insurance Fraud dataset.
- *Gorishniy et al.,* **Revisiting Deep Learning Models for Tabular Data** (NeurIPS 2021) — FT-Transformer.
- *Zaheer et al.,* **Deep Sets** (NeurIPS 2017).
- *Kipf & Welling,* **Semi-Supervised Classification with Graph Convolutional Networks** (ICLR 2017).
- *Lin et al.,* **Focal Loss for Dense Object Detection** (ICCV 2017).
