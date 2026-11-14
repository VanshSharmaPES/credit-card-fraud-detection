# Simplicity notes (Person 2)

- Keras: compact dense 64-32 network, class-weighted binary cross-entropy, early stopping on validation PR-AUC.
- Lightning: same dense shape and validation protocol, using weighted BCE-with-logits and early stopping.
- Flash was attempted locally but installation stopped with `BackendUnavailable: Cannot import setuptools.build_meta`; plain PyTorch Lightning is the completed route.
- Local runtime measurements are ARM64 CPU measurements. Colab and GPU execution were not verified.
- The combined model is an arithmetic mean of XGBoost, Lightning, and Keras probabilities. Member choice is based on validation PR-AUC only.

## XGBoost (Person 1)

- **Notebook:** `notebooks/02_xgboost.ipynb`
- **Install:** `pip install -r requirements/xgboost.txt` (one package). Install problems: none (CPU, Python 3.13, `xgboost==3.4.1`, one wheel, no build step).
- **Lines of code** (train and predict cells, no comments or blank lines): train 8, predict 5, total 13.
- **Settings changed from the default (6):**

  | Setting | Default | Used |
  |---|---|---|
  | `n_estimators` | 100 | 2000 (early stopping picks the real number) |
  | `learning_rate` | 0.3 | 0.05 |
  | `scale_pos_weight` | 1 | 598.4190 (class weight from `data.py`) |
  | `eval_metric` | `logloss` | `aucpr` |
  | `early_stopping_rounds` | none | 50 |
  | `random_state` | 0 | 42 |

  `max_depth=6` is set in the plan but equals the default, so it is not counted. `device` is picked automatically (`cuda` if a GPU is present); the official run used the default `cpu`, so it is not counted either.
- **Training time (CPU, AMD Ryzen 5 5600H, 12 threads):** 6.68 s (265 trees after early stopping); test prediction 0.047 s for 56,746 rows. From `results/xgboost.json`.
- **Data format:** takes the pandas DataFrames from `load_splits()` directly. No conversion needed.
