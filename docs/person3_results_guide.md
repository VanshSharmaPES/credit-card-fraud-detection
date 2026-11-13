# Results and conclusion evidence guide

The project result write-up is drafted in docs/person3_report_draft.md. This guide identifies the evidence behind it and the checks to do before submitting or presenting.

## Source files

- Main metrics and test confusion counts: results/*.json and figures/table_main.png.
- Probability errors: figures/table_regression.png and results/model_comparison_regression.csv.
- Ranking and thresholded errors: figures/pr_curves_all.png and figures/confusion_matrices.png.
- Recorded runtime: result JSON files and figures/time_comparison.png. Runs used different CPU environments, so do not describe timing differences as a controlled benchmark.
- Draft rubric values and evidence: results/model_scores_draft.csv and docs/scoring_rules.md.
- Cost thresholds and fee sensitivity: results/money_thresholds.csv, results/money_f1_vs_cost.csv, figures/money_cost_curve.png and figures/money_saved_all.png.
- Current ensemble membership: results/ensemble_members.json. It selects XGBoost, Lightning and fastai based on validation PR-AUC.

## Interpretation checkpoints

1. Report that the 56,746-row test split contains 95 frauds, and thresholds were selected on validation.
2. Distinguish PR-AUC/ROC-AUC ranking quality from precision, recall and confusion counts at one threshold.
3. Describe RMSE, MAE and R² as secondary probability-error metrics; avoid conflating them with classification quality.
4. Treat framework rubric scores as drafts requiring team agreement; they are ordinal judgments.
5. Disclose assumed review fees, full prevention of flagged fraud amounts, and that test savings are an evaluation result rather than a guaranteed business outcome.
6. Explain that training times were recorded on different CPUs and are not directly comparable.
7. Describe the dataset's two-day 2013 scope and anonymized PCA features; recommend a later time-based holdout.

## Current headline values

XGBoost is the top individual model by F1 (0.8721) and precision (0.9740). fastai catches the most individual test frauds (76 of 95). The refreshed ensemble's test PR-AUC is 0.8488 and its members are XGBoost, Lightning and fastai. At fee 10, the XGBoost F1 threshold saves 9,287.36 amount units versus 5,793.01 for its validation-cost threshold. These values come from saved project outputs and should be rechecked if any model is rerun.
