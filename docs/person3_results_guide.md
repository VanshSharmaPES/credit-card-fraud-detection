# Person 3 results and conclusion worksheet

This is a structure for the student to complete, not report prose. Write all interpretation in your own words, and copy every number from the current `results/` files and generated figures.

## Results evidence to assemble

1. **Main comparison:** use `figures/table_main.png` and the saved result JSON files. Include precision, recall, F1, PR-AUC, ROC-AUC, threshold, training time and prediction time. State that classification thresholds were chosen on validation data.
2. **Probability errors:** use `figures/table_regression.png` for RMSE, MAE and R². Explain their role in this classification project in your own words.
3. **Ranking and errors:** use `figures/pr_curves_all.png` and `figures/confusion_matrices.png`. Note the test fraud count and treat PR-AUC gaps below 0.02 as ties.
4. **Runtime:** use `figures/time_comparison.png`. State the hardware recorded per model; avoid implying timings from different machines are directly comparable.
5. **Framework scores:** use the draft score table from `notebooks/10_comparison.ipynb` only after the team has approved the rubric and evidence.
6. **Money-based threshold:** use `results/money_thresholds.csv`, `figures/money_cost_curve.png` and `figures/money_saved_all.png`. Compare validation-selected cost thresholds for fees 5, 10 and 25 with the F1 threshold at fee 10. Report caught frauds, false alarms, review cost, total cost and saved amount.
7. **Combined model freshness:** ask Person 2 to rerun `08_ensemble` before finalizing. The currently committed `results/ensemble_members.json` predates the fastai and TabNet predictions; this Person 3 work leaves the existing ensemble files untouched.

## Student-authored Results prompts

- Which models lead on ranking quality and on thresholded precision/recall?
- Which pair of PR-AUC results is close enough to call a tie?
- What do the confusion matrices add beyond the summary metrics?
- How does the cost threshold change across the three assumed fees?
- Does the cost-based choice agree with the F1-based choice at fee 10? Use the saved numbers.
- Which framework scores are sensitive to subjective judgments, and what evidence supports the draft scores?
- Does the combined model still use the top three available validation PR-AUC models after the new predictions were added?

## Student-authored Conclusion prompts

- State a recommendation for a clearly named objective, such as minimizing review workload or maximizing fraud recovery.
- Tie that recommendation to one classification result and one cost result from the saved outputs.
- State what the project does not establish: temporal generalization beyond two days, behavior beyond the European cardholder sample, or production operating costs beyond the assumed review fees.
- Name one next experiment that would reduce a limitation, such as a later time-based holdout or a bank-specific review-cost estimate.

## Evidence checks

- [ ] Every table value matches a JSON result or generated cost table.
- [ ] Test data was not used to select either the F1 or cost threshold.
- [ ] Amounts and review fees use the same currency unit.
- [ ] The assumed full prevention of a flagged fraud principal and review-fee values are disclosed.
- [ ] Missing model runs are labeled pending; no values are filled by estimation.
