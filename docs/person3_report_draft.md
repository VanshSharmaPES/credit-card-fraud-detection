# Results and conclusion

## Evaluation setup

The cleaned dataset was split into 170,235 training, 56,745 validation and 56,746 test transactions using a stratified 60/20/20 split. The held-out test split has 95 frauds. Each classifier's probability threshold was selected using validation data; all metrics below are from the untouched test split. Fraud is rare, so PR-AUC and confusion counts are more informative than accuracy alone.

## Test-set comparison

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | RMSE | MAE | R² | Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.9740 | 0.7895 | 0.8721 | 0.8382 | 0.9719 | 0.0203 | 0.0007 | 0.7529 | 6.676 |
| Keras | 0.7957 | 0.7789 | 0.7872 | 0.7103 | 0.9628 | 0.1708 | 0.1186 | -16.4565 | 1.207 |
| fastai | 0.8837 | 0.8000 | 0.8398 | 0.7377 | 0.9612 | 0.0869 | 0.0269 | -3.5168 | 6.025 |
| TabNet | 0.7976 | 0.7053 | 0.7486 | 0.6833 | 0.9764 | 0.1769 | 0.0899 | -17.7280 | 45.549 |
| Lightning | 0.8824 | 0.7895 | 0.8333 | 0.7576 | 0.9629 | 0.1072 | 0.0266 | -5.8776 | 9.991 |
| Combined | 0.9494 | 0.7895 | 0.8621 | 0.8488 | 0.9662 | 0.0631 | 0.0181 | -1.3815 | — |

XGBoost had the strongest individual thresholded result by F1 and precision: 75 detected frauds and 2 false alarms. fastai detected 76 frauds, the most among the individual models at their selected thresholds, with 10 false alarms. TabNet had the largest ROC-AUC (0.9764), but the lowest PR-AUC (0.6833) and recall (0.7053). The combined model averages the three models ranked highest by validation PR-AUC: XGBoost, Lightning and fastai. It achieved PR-AUC 0.8488 and detected 75 frauds with 4 false alarms. The ensemble outperformed its members on PR-AUC but did not beat XGBoost's individual F1.

RMSE, MAE and R² compare probability outputs to binary labels. XGBoost has the lowest values in this run; negative R² values for the other probability models mean the constant-mean reference would have lower squared error on this test set. These regression metrics should be read alongside classification and ranking measures, not as a replacement for them.

## Money-based analysis

The cost model charges the assumed review fee for every alert and treats the raw amount of a missed fraud as a loss. It assumes a flagged fraud is prevented in full. At fee 10 (same amount units as the transaction amounts), no model has a test fraud loss of 12,973.06.

| Model and validation rule | Test frauds caught | False alarms | Test review cost | Test total cost | Test savings |
|---|---:|---:|---:|---:|---:|
| XGBoost, F1 threshold | 75 | 2 | 770.00 | 3,685.70 | 9,287.36 |
| XGBoost, cost threshold | 70 | 2 | 720.00 | 7,180.05 | 5,793.01 |
| fastai, F1 / cost threshold | 76 | 10 | 860.00 | 3,346.72 | 9,626.34 |
| TabNet, F1 / cost threshold | 67 | 17 | 840.00 | 4,659.55 | 8,313.51 |
| Lightning, F1 / cost threshold | 75 | 10 | 850.00 | 3,348.11 | 9,624.95 |
| Combined, F1 threshold | 75 | 4 | 790.00 | 3,705.70 | 9,267.36 |
| Combined, cost threshold | 71 | 2 | 730.00 | 5,776.58 | 7,196.48 |

The thresholds in both rows were selected on validation data. On this test sample, the cost-minimizing validation threshold for XGBoost and the combined model produced lower savings than their F1 thresholds. This illustrates that optimizing a validation objective does not guarantee lower realized cost on a separate sample. The fee sensitivity analysis at 5, 10 and 25 and the full cost breakdown are in results/money_thresholds.csv; figures are figures/money_cost_curve.png and figures/money_saved_all.png.

## Framework score rubric

The comparison notebook records draft Simplicity, Transparency and Efficacy scores with the supporting facts in results/model_scores_draft.csv. The rubric was written after several model results were already present, so the scores are not preregistered. The team should review the rubric and agree on any subjective ratings before presenting them as final. TabNet's feature importance and per-row masks are model explanations, not evidence of causal effects.

## Conclusion and limitations

For a thresholded workflow that prioritizes high precision and fewer false alerts, XGBoost is the strongest individual model in this run. For a goal of catching the most test frauds among individual models, fastai leads by one fraud over XGBoost, at the cost of eight additional false alarms. The appropriate choice depends on the bank's relative cost of missed fraud and review workload. The assumed fee-10 results support comparing both thresholds rather than assuming a validation cost optimum will always win on new transactions.

The evaluation uses a two-day 2013 dataset from European cardholders, with V1–V28 anonymized PCA features. It does not establish performance on later transactions, other populations or a real bank review operation. Review fees are assumptions, and the cost calculation assumes full prevention of a flagged fraud's transaction amount. A next experiment should use a later time period as a time-based holdout and estimates of review costs from the target operation. The saved training times come from different CPU environments and are not a controlled speed benchmark.
