# Decisions log

Format: date, decision, reason. Newest at the bottom. Persons 2 and 3 post their entries in chat and Person 1 adds them here.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-30 | Treat the task as classification, and also report RMSE, MAE and R2 computed on the predicted fraud probabilities (Decision 1). | The target is a yes/no label, but the brief asks for regression metrics too. |
| 2026-10-30 | Use Lightning Flash if it installs and runs. If not, use plain PyTorch Lightning and write down why (Decision 2). | Flash is no longer maintained and may not install on current Colab. |
| 2026-10-30 | State the dataset's limits in the report (Decision 3). | V1–V28 are PCA outputs with no known meaning, the data covers only two days in September 2013, and it comes from European cardholders only. |
| 2026-10-30 | Drop exact duplicate rows: 1,081 dropped (284,807 → 283,726 rows, 492 → 473 frauds). | Duplicates could land in both train and test, which would inflate test scores. |
| 2026-10-30 | Freeze the 30 features: V1–V28, `Amount` (log1p, then scaled) and `Hour` (scaled). Drop raw `Time`. | `Time` is seconds since the first transaction, so it only describes when in the two-day window a row was recorded. `Hour = (Time // 3600) % 24` keeps the daily pattern. |
| 2026-10-30 | Keep `Amount_raw` (amount in euros) in the split files, but never as a model input. | Needed only for the money-based threshold analysis. |
| 2026-10-30 | Fit `StandardScaler` on the train split only, then apply it to val and test. | Fitting on all data would leak information from val and test into training. |
| 2026-10-30 | Stratified 60 / 20 / 20 split. Class weight = train normal ÷ train fraud = 169,951 ÷ 284 = 598.42. | Stratifying keeps the fraud share at about 0.17% in every split. |
| 2026-10-30 | Seed 42 everywhere (numpy, random, sklearn, xgboost and the other frameworks). | Makes results repeatable across team members. |
| 2026-11-04 | `src/data.py` downloads `creditcard.csv` from the TensorFlow mirror (`storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`), not the writeup's `media.geeksforgeeks.org` link. Kaggle `mlg-ulb/creditcardfraud` stays the fallback. | It is the same ULB dataset: `data.py` asserts 284,807 rows, 31 columns and 492 frauds before doing anything else. The report's dataset section must name the original source and say a mirror was used. |
| 2026-11-12 | Flash outcome: plain PyTorch Lightning is used instead of Lightning Flash (Decision 2 fallback). Logged for Person 2. | Installing Flash failed with `BackendUnavailable: Cannot import setuptools.build_meta` (see `docs/person2_results.md`). |
| 2026-11-12 | The combined model averages the XGBoost, Lightning and Keras probabilities, with members chosen by validation PR-AUC only (0.793, 0.713, 0.669). Logged for Person 2. | Choosing on validation keeps the test split untouched (`results/ensemble_members.json`). |
| 2026-11-12 | `02_xgboost` and `07_imbalance` are run on CPU, not a Colab T4, and the hardware is written in each results file. After the `02_xgboost` run, `08_ensemble` is re-run. | No GPU run was available. The plan's fallback allows CPU if the hardware is recorded. The combined model must use the official XGBoost preds. |
