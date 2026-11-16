# Demo rehearsal — 16 November 2026

This records the technical rehearsal for the tentative 17 November 2026 demo. It verifies the live command and saved fallback. A spoken three-minute rehearsal by the presenter is still a personal preparation step.

## Timed live check

From the repository root, `python src/data.py` completed successfully in **18.808 seconds** on the local CPU environment. Its output matched the expected split exactly:

```text
Raw rows: 284807, raw frauds: 492
Duplicates dropped: 1081
Rows after cleaning: 283726, frauds: 473
train rows:  170235  frauds:  284  fraud share: 0.1668%
val   rows:   56745  frauds:   94  fraud share: 0.1657%
test  rows:   56746  frauds:   95  fraud share: 0.1674%
Class weight (train normal / train fraud): 169951 / 284 = 598.4190
```

The two planned EDA charts are saved at `figures/eda_class_balance.png` and `figures/eda_top_features.png`. The executed `01_eda.ipynb` provides the fallback if a live command fails.

## Three viva checks

1. **Why does 99.83% accuracy fail here?** Predicting every transaction as normal is almost always correct because fraud is rare, but it catches zero fraud. Precision, recall, F1 and PR-AUC describe the useful fraud-detection trade-off.
2. **Why is SMOTE applied only to training data?** Synthetic points are part of model fitting. Applying SMOTE before the split, or to validation or test data, can leak related information into evaluation and makes the evaluation distribution artificial. Validation and test remain untouched.
3. **What do the SHAP contributions sum to?** For the XGBoost explanation, the baseline value plus the feature contributions equals the model output in log-odds. Applying the logistic function converts that value to probability.

## Demo fallback

If the live run is slow or fails, open the saved output in `01_eda.ipynb` or its PDF, state that it is a saved run, and continue with the two EDA charts. Do not troubleshoot during the presentation.
