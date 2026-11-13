# Viva practice prompts

Write a short answer to each prompt in your own words. Use the current `results/*.json`, notebooks and figures for project-specific evidence. Do not copy generated notes into the report or presentation.

## Evaluation and data

- Explain why accuracy can hide poor fraud detection on this dataset.
- Define precision, recall, F1 and PR-AUC using the project’s fraud class.
- Explain why the probability threshold is selected on validation data and then held fixed for test evaluation.
- Compare class weighting, SMOTE and random undersampling. What did the saved imbalance results show?
- Explain why resampling belongs inside the training split only.
- Describe what `V1`–`V28` represent and what their anonymization limits.
- Explain why the scaler is fitted on training rows only.
- Explain how duplicates could distort the evaluation if the same transaction crossed splits.
- State how RMSE, MAE and R² were calculated here, and why they are secondary to classification metrics.

## Models and explanations

- Describe the main idea behind XGBoost and the purpose of `scale_pos_weight`.
- Explain how TabNet creates feature masks across its decision steps.
- Compare fastai’s tabular workflow with the Keras model used by the team.
- Explain what Lightning adds to a plain PyTorch training loop and why Flash was not used.
- Interpret a SHAP value for one transaction without claiming that a feature causes fraud.
- Explain the three score dimensions, the evidence for each score, and the timing of team approval.
- Explain why the team kept model settings limited and how additional tuning might change the ranking.

## Cost analysis and conclusions

- Walk through how a review fee, a missed fraud amount and a threshold affect total cost.
- Explain how the cost-minimizing threshold changes when the assumed fee changes.
- Compare the F1-selected threshold with the cost-selected threshold at the same fee.
- Describe how averaging model probabilities could help and why it might not.
- Name one limitation of the two-day dataset and one way to test generalization better.
- Recommend a model for a stated bank objective, citing both detection metrics and operational costs.
- Identify the project’s contribution and distinguish it from a production-ready fraud system.

## Your answers

Add your own concise answers below each prompt before the viva. For any numerical statement, copy the number from the current result file and name the split and threshold policy.
