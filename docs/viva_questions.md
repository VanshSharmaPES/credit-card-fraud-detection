# Viva questions and answer guide

Use these as preparation notes. Be ready to explain each answer in your own words and point to the matching project output.

## Evaluation and data

### Why can accuracy be misleading here?
Fraud is only about 0.17% of the cleaned data. A classifier that labels every transaction as legitimate would appear about 99.8% accurate while catching no fraud. Precision, recall, F1, PR-AUC and the confusion counts expose that failure.

### What do precision, recall, F1 and PR-AUC mean in this project?
Precision is the share of flagged transactions that are fraud. Recall is the share of actual frauds caught. F1 is their harmonic mean at a chosen threshold. PR-AUC summarizes the precision-recall trade-off across thresholds and is useful when the positive class is rare.

### Why choose the threshold on validation data?
Validation data is used to choose the operating point without using the held-out test labels. The selected threshold is then fixed while test metrics are calculated, which keeps the test set from influencing the decision being evaluated.

### How do class weights, SMOTE and undersampling differ, and what did the experiment show?
Class weighting increases the training loss for fraud errors without changing the rows. SMOTE creates synthetic minority examples in the training split. Random undersampling removes some normal examples. In the saved XGBoost experiment, class weighting had the highest F1 (0.8721), SMOTE the highest PR-AUC (0.8572), and no handling had PR-AUC 0.8451; undersampling used only 568 training rows and had F1 0.7958.

### Why must SMOTE be applied only to training data?
Applying it before splitting can create synthetic points related to rows that later appear in validation or test. That leaks information across splits and makes evaluation optimistic. Validation and test should retain their natural class distribution.

### What are V1–V28?
They are anonymized PCA components supplied in the dataset. Their original transaction fields and component meanings are not published, so the project can analyze their model associations but cannot give them straightforward business labels.

### Why fit the scaler on training rows only?
The scaler learns means and standard deviations. Fitting it on validation or test rows would let information from evaluation data affect preprocessing. Fitting on train and applying the same transform to the other splits avoids that leakage.

### Why remove exact duplicates?
If duplicate transactions land in both train and test, the model may effectively see a test example during training. Dropping the 1,081 exact duplicates before splitting reduces this leakage risk.

### How were RMSE, MAE and R² computed, and why are they secondary?
They compare each predicted fraud probability with the 0/1 label on the test split. They are included because the brief requests them, but this is a classification problem, so precision, recall, F1, PR-AUC, ROC-AUC and confusion counts are the primary measures.

## Models and explanations

### How does XGBoost work, and what does `scale_pos_weight` do?
XGBoost builds an additive sequence of decision trees, with later trees correcting errors from earlier ones. `scale_pos_weight` increases the loss contribution of fraud examples; this project used the train normal-to-fraud ratio, 598.419.

### How does TabNet use feature masks?
At successive decision steps, TabNet forms sparse masks that select or weight input features for that step. `feature_importances_` summarizes global importance; `explain()` returns row-level masks. A mask describes the model’s allocation of attention, not a causal reason for fraud.

### What does fastai provide compared with the Keras model?
fastai supplies a tabular data and learner workflow, including `TabularPandas`, dataloaders, learning-rate search, one-cycle training and callbacks. Keras uses a dense neural network and its own training API. Both used the shared split and weighted loss approach for this comparison.

### What does Lightning add, and why was Flash not used?
Lightning organizes the PyTorch training loop, validation, logging and callbacks into a module and trainer. The repository records that the Flash install attempt failed, so the team used plain PyTorch Lightning as the planned fallback.

### What does a SHAP value mean?
A SHAP value attributes part of one model prediction to a feature relative to a reference prediction. A positive value pushes the model output toward fraud and a negative value pushes it away, for the chosen output scale. It describes model behavior, not causation.

### How were the three framework scores made?
Simplicity uses install effort, training/prediction code size and configuration; Transparency uses available explanation methods; Efficacy uses the test PR-AUC gap from the best saved model. The current scores are drafts because the scoring file was added after baseline results already existed and still needs team agreement.

### Why limit tuning, and what could more tuning change?
The project aimed for a manageable framework comparison under time limits rather than an exhaustive search. Learning rates, architecture, class weighting, early stopping and threshold choices can change both metrics and ranking. A fair follow-up would define the search budget in advance and use validation data only for model selection.

## Cost analysis and conclusions

### How does the money threshold work?
Every flagged row incurs the assumed review fee; every missed fraud incurs its `Amount_raw`. Total cost is missed-fraud amount plus review fees for all alerts. Savings are the no-model fraud loss minus that total cost.

### How does the best cost threshold respond to a higher fee?
A higher fee makes each alert more expensive, so the validation cost calculation generally favors fewer alerts and a higher threshold. It can change the number of frauds caught as well as false alarms.

### How did the F1 and cost thresholds compare at fee 10?
On test, the cost-selected threshold matched the F1 threshold for fastai, Keras, Lightning and TabNet. For XGBoost, F1 saved 9,287.36 amount units while the validation-cost threshold saved 5,793.01. The old ensemble F1 threshold saved 9,198.09 versus 7,184.17 for its validation-cost threshold. This is a test comparison, not a reason to tune on test.

### How could an ensemble help or hurt?
Averaging probabilities can reduce model-specific errors when the members make different mistakes. It can also smooth away useful high-confidence signals, and its value depends on the members and their calibration. The saved ensemble has not yet been refreshed after adding fastai and TabNet.

### What are key limitations and how could generalization be tested better?
The dataset covers two days in 2013, comes from European cardholders, and uses anonymized PCA features. A later time-based holdout would better measure performance under temporal drift; evaluation on a bank-specific sample would test population and operating-cost fit.

### Which model would you recommend to a bank?
For the current saved individual results, XGBoost is a defensible choice when high precision and low false-alarm workload matter: it had precision 0.9740 and two false alarms while catching 75 test frauds. If catching the maximum number at the selected threshold matters more, fastai caught 76 with 10 false alarms. The final choice depends on the bank’s measured missed-fraud and review costs; the combined model needs refreshing first.

### What is the project contribution, and what is not established?
The project compares five frameworks on one shared split and adds a validation-selected threshold based on an explicit review-fee and missed-amount cost model. It does not establish production readiness, performance on future transaction periods, or bank-specific economics.
