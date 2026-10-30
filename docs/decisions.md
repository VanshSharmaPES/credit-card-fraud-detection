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
