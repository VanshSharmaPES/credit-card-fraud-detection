# Credit Card Fraud Detection

## Project

Detects fraudulent card transactions in a highly imbalanced dataset (about 0.17% fraud) and compares five frameworks on the same data split:

- XGBoost
- Keras
- PyTorch with fastai
- PyTorch with TabNet
- PyTorch with Lightning Flash (or plain Lightning if Flash does not install; see below)

Every model outputs a fraud probability. The main metrics are precision, recall, F1 and PR-AUC at a threshold chosen on the validation split. RMSE, MAE and R2 are also computed between the probability and the 0/1 label. The frameworks are scored on Simplicity, Transparency and Efficacy using rules written before any results existed (`docs/scoring_rules.md`).

Mini project for Machine Learning with Business Data.

## Team and roles

| Person | Name | Role |
|---|---|---|
| 1 | _name_ ([VanshSharmaPES](https://github.com/VanshSharmaPES)) | Data pipeline, EDA, XGBoost, imbalance experiment, README, decisions log |
| 2 | _name_ | `src/evaluate.py`, Keras, Lightning / Flash, combined model, PPT |
| 3 | _name_ | Scoring rules, `src/cost.py`, fastai, TabNet, money-based threshold, comparison |

## Dataset

- Credit card transactions by European cardholders over two days: 284,807 rows, 31 columns, 492 frauds.
- `V1`–`V28` are PCA components with no published meaning. Only `Time`, `Amount` and `Class` are original columns.
- Source: ULB Machine Learning Group, also on Kaggle as `mlg-ulb/creditcardfraud`. The course writeup links a copy at `media.geeksforgeeks.org`. `src/data.py` downloads an identical mirror and checks the row, column and fraud counts.
- The raw file (about 144 MB) is never committed. `src/data.py` downloads it into `data/`, which is in `.gitignore`.

**Preparation (`src/data.py`)**

1. Drop 1,081 exact duplicate rows, leaving 283,726 rows and 473 frauds.
2. Add `Hour = (Time // 3600) % 24`, then drop `Time`.
3. Keep the raw amount as `Amount_raw` (used only for the money analysis), and use `log1p(Amount)` as the feature.
4. Do a stratified 60 / 20 / 20 split with seed 42.
5. Fit a `StandardScaler` on the 30 features of the train split only.

| Split | Rows | Frauds |
|---|---|---|
| Train | 170,235 | 284 |
| Validation | 56,745 | 94 |
| Test | 56,746 | 95 |

Class weight (train normal ÷ train fraud): 169,951 ÷ 284 = **598.4190**.

Every decision and the reason for it is logged in `docs/decisions.md`.

## How to run

### In Google Colab (how the results were produced)

1. Runtime → Change runtime type → **T4 GPU**.
2. The repo is private, so add a GitHub token (classic, `repo` scope) to Colab Secrets (key icon) as `GH_TOKEN`, with notebook access on.
3. Open any notebook from `notebooks/` and run all. The first cell clones the repo, installs `requirements/base.txt` plus that framework's file, and runs `src/data.py`.

### On your own machine

```bash
git clone https://github.com/VanshSharmaPES/credit-card-fraud-detection.git
```

```bash
cd credit-card-fraud-detection
```

```bash
pip install -r requirements/base.txt -r requirements/xgboost.txt
```

```bash
python src/data.py
```

`python src/data.py` should print exactly the split table above. Then open a notebook from `notebooks/`. Install one framework's requirements file per environment, because the frameworks' packages can clash.

### Notebook order

| Notebook | What it does | Needs |
|---|---|---|
| `00_checks` | Sanity checks for `src/evaluate.py` | base |
| `01_eda` | Exploratory data analysis (train split only) | base |
| `02_xgboost` | XGBoost, SHAP, permutation importance | `xgboost.txt` |
| `03_keras` | Keras model | `keras.txt` |
| `04_lightning` | Lightning (or Flash) model | `lightning.txt` |
| `05_fastai` | fastai model | `fastai.txt` |
| `06_tabnet` | TabNet model and feature masks | `tabnet.txt` |
| `07_imbalance` | Class weight vs SMOTE vs undersampling (XGBoost) | `xgboost.txt` |
| `08_ensemble` | Average of the best models | base |
| `09_money` | Money-based threshold | base |
| `10_comparison` | Final tables and charts (reads `results/` and `preds/` only) | base |

Notebooks `02`–`07` write `results/NAME.json`, `preds/NAME_val.csv`, `preds/NAME_test.csv` and a model in `models/`. `08`–`10` only read those files.

## Results

_Filled in from `results/` after the model freeze. Every number is copied from the results files._

| Framework | Precision | Recall | F1 | PR-AUC | ROC-AUC | RMSE | MAE | R2 | Train (s) |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | | | | | | | | | |
| Keras | | | | | | | | | |
| fastai | | | | | | | | | |
| TabNet | | | | | | | | | |
| Lightning | | | | | | | | | |
| Combined | | | | | | | | | |

Simplicity / Transparency / Efficacy scores: _see `notebooks/10_comparison.ipynb`_.

**Lightning Flash:** _outcome of the install test (Step 11), and which route was used._

Key figures: _to be added from `figures/`._

## Repo structure

```
credit-card-fraud-detection/
├── src/
│   ├── data.py          # download, clean, split, scale; load_splits()
│   ├── evaluate.py      # threshold, metrics, results/preds files, permutation importance
│   └── cost.py          # money-based cost functions
├── notebooks/           # one notebook per task (see the table above)
├── results/             # one metrics JSON per model; imbalance/ for the experiment; perm_*.csv
├── preds/               # validation and test probabilities per model (row_id, y_true, y_prob)
├── models/              # saved trained models
├── figures/             # every chart used in the report and PPT
├── docs/                # decisions log, scoring rules, EDA findings, simplicity notes, viva questions
├── requirements/        # base.txt plus one file per framework
└── data/                # created by src/data.py, never committed
```
