# Person 1 slide text

Source wording integrated into the final 18-slide team deck. Numbers come from the frozen published results. The final deck uses slides 2, 3, 4, 5, 13 and 14 for the sections below.

## Slide 2 — Problem and challenges

- Flag fraudulent card transactions while limiting false alerts.
- Fraud is only about 0.17% of the training data; predicting every row as normal gives about 99.83% accuracy but catches no fraud.
- Compare precision, recall, F1 and PR-AUC on the same split.
- Evaluate XGBoost, Keras, fastai, TabNet and PyTorch Lightning.

Speaker note: Precision describes the quality of alerts; recall describes how many frauds are caught. We select each operating threshold on validation before evaluating test data.

## Slide 3 — Dataset and preparation

- Raw ULB data: 284,807 transactions, 31 columns and 492 frauds.
- Remove 1,081 duplicate rows, leaving 283,726 rows and 473 frauds.
- Inputs: V1–V28, log1p(Amount) and elapsed Hour. V1–V28 are anonymized PCA components.
- Stratified 60/20/20 split; fit the scaler on training rows only.
- Train / validation / test fraud counts: 284 / 94 / 95.

Speaker note: Hour wraps every 24 hours from the first transaction and is not a known clock time. Raw amounts are retained for cost analysis but excluded from the model inputs.

## Slide 4A — Class balance and amounts

Charts: figures/eda_class_balance.png and figures/eda_amount_by_class.png.

- Training data: 169,951 normal transactions and 284 frauds.
- Median amount: fraud 9.17; normal 22.00.
- Mean amount: fraud 111.42; normal 88.27.
- The amount chart excludes zero values because it uses a log axis; summary statistics include them.

Speaker note: A higher fraud mean does not mean the typical fraud transaction is larger. The median shows the opposite ordering.

## Slide 4B — Features and elapsed hours

Charts: figures/eda_top_features.png and figures/eda_hour_pattern.png.

- V17 and V14 have the strongest absolute linear correlations with fraud, at -0.3178 and -0.3066.
- Fraud and normal distributions differ but still overlap.
- Highest observed hourly rates: bin 2, 1.30% (26/2,002); bin 4, 0.98% (13/1,329).
- Small fraud counts and elapsed-hour bins limit the interpretation.

Speaker note: Pearson correlation captures linear association. These plots do not identify business meanings for the anonymized features or establish causation.

## Slide 11 — SHAP half

Chart: figures/xgb_shap_case.png. Person 3 supplies the TabNet half.

- Explain validation fraud row 541: predicted probability 0.9999 at threshold 0.9394.
- Positive SHAP contributions raise the fraud score; negative contributions lower it.
- Contributions plus the baseline add to log-odds, not probability.
- The summary sample has 2,000 validation rows and only three frauds.

Speaker note: Row 541 is the caught validation fraud with the lowest row_id, rather than a manually selected dramatic example. SHAP explains model behavior, not the cause of fraud.

## Slide 12 — Imbalance experiment

Chart: figures/imbalance_comparison.png.

| Method | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| No handling | 0.9390 | 0.8105 | 0.8701 | 0.8451 |
| Class weight | 0.9740 | 0.7895 | 0.8721 | 0.8382 |
| SMOTE | 0.9610 | 0.7789 | 0.8605 | 0.8572 |
| Undersampling | 0.7917 | 0.8000 | 0.7958 | 0.7922 |

- Class weighting has the highest F1; SMOTE has the highest PR-AUC in this experiment.
- No handling has the highest recall; undersampling retains only 568 training rows.
- Resample training only; keep validation and test unchanged.

Speaker note: There is no single winner across every metric. The F1 gap between weighting and no handling is small, and these results come from one split.
