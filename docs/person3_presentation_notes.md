# Person 3 presentation notes

## Slide 8: Main results

- Show `figures/table_main.png`.
- Emphasize that thresholds were chosen on validation and reported metrics are from the held-out test split.
- XGBoost: precision 0.9740, recall 0.7895, F1 0.8721, PR-AUC 0.8382.
- Fastai detected 76 of 95 frauds; XGBoost detected 75 with two false alarms.
- Note that the combined row is the earlier XGBoost/Lightning/Keras ensemble and must be refreshed before final comparison.

## Slide 9: Ranking and errors

- Show `figures/pr_curves_all.png` and `figures/confusion_matrices.png`.
- Explain the precision-recall trade-off using the held-out fraud count of 95.
- PR-AUC gaps under 0.02 are treated as practical ties.

## Slide 10: Framework scores

- Show the score table from `results/model_scores_draft.csv`.
- State that scores are drafts under the documented rubric and require team agreement.
- Use the method evidence fields; write the final spoken interpretation in your own voice.

## Slide 11: Transparency

- Show the XGBoost SHAP figure already in the repository and `figures/tabnet_importance.png` / `figures/tabnet_masks.png`.
- Describe TabNet masks as model attention across decision steps, not causal effects.

## Slide 13: Money-based threshold

- Show `figures/money_cost_curve.png` and `figures/money_saved_all.png`.
- Fee-10 XGBoost F1 threshold: 75 frauds caught, 2 false alarms, 9,287.36 saved units on test.
- Fee-10 XGBoost cost threshold: 70 caught, 2 false alarms, 5,793.01 saved units on test.
- Remind the audience that thresholds were selected on validation and fees are assumptions.

## Slide 15: Conclusion and limits

- Choose the model based on the bank’s cost of missed fraud versus review workload.
- Name the two-day, 2013 dataset, anonymized PCA features, and assumed fees as limits.
- Recommend a later time-based holdout and bank-specific review-cost estimates as next steps.
