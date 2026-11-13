# Person 3 presentation notes

These notes use the refreshed outputs in results/ and figures/. The existing Person 2 PowerPoint remains their original presentation file; replace its result slides with the current evidence before presenting.

## Slide 8: Main results

- Show figures/table_main.png.
- State that thresholds were selected on validation and metrics are measured on the held-out test split with 95 frauds.
- XGBoost leads the individual models on precision (0.9740) and F1 (0.8721), with 75 caught and 2 false alarms.
- fastai catches 76 frauds (recall 0.8000), with 10 false alarms.
- The refreshed combined model is XGBoost + Lightning + fastai, selected by validation PR-AUC. It has test PR-AUC 0.8488, precision 0.9494 and recall 0.7895.

## Slide 9: Ranking and errors

- Show figures/pr_curves_all.png and figures/confusion_matrices.png.
- Explain that PR-AUC summarizes ranking quality across thresholds while the confusion matrix shows the selected operating point.
- TabNet has ROC-AUC 0.9764 but PR-AUC 0.6833; its selected threshold catches 67 of 95 frauds.
- PR-AUC differences below 0.02 are treated as close in the written discussion.

## Slide 10: Framework scores

- Show results/model_scores_draft.csv.
- Scores are drafts under a rubric written after results were already available. Obtain team agreement before presenting them as final.
- Explain the evidence for each dimension rather than treating the ordinal scores as measurements.

## Slide 11: Model explanations

- Show the XGBoost SHAP figures and figures/tabnet_importance.png / figures/tabnet_masks.png.
- TabNet masks describe features emphasized by the model at decision steps; they do not establish causal effects.

## Slide 13: Money-based threshold

- Show figures/money_cost_curve.png and figures/money_saved_all.png.
- At fee 10, XGBoost's validation F1 threshold catches 75 frauds, triggers 2 false alarms, and saves 9,287.36 units on test.
- Its validation cost threshold catches 70 with 2 false alarms and saves 5,793.01 units on test.
- The refreshed ensemble's F1 and cost thresholds save 9,267.36 and 7,196.48 units respectively.
- Disclose that the fee is assumed and missed fraud is valued at full raw transaction amount.

## Slide 15: Conclusion and limits

- Choose between fewer false alerts and more fraud recovery using the bank's measured costs.
- Name the two-day 2013 dataset, anonymized PCA features and assumed fees as limitations.
- Recommend a later time-based holdout and bank-specific review-cost estimates.
- Training times came from different CPU environments and are not a controlled benchmark.
