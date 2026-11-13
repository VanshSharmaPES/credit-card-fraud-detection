# Framework score rubric

## Status

This is a **draft for team approval**. Results for XGBoost, Keras, Lightning and the ensemble were already present in the repository when this file was added, so the scores cannot honestly be described as preregistered. Keep the original rubric fixed after the team agrees; record any disagreement and its reason.

Scores run from 1 (lowest) to 5 (highest). Report the numeric score with the evidence fields below. The comparison notebook calculates draft scores from saved artifacts; the team must review and agree them before using them in the report or presentation.

## Simplicity

Record these observations for each implementation:

- Whether the documented requirements install and the notebook runs without a workaround.
- Number of non-blank lines in the training and prediction code cells (exclude setup, evaluation, plotting and comments).
- Number of material settings changed from framework defaults.
- Measured training and prediction seconds from the corresponding result JSON.

| Score | Rule |
|---:|---|
| 5 | One documented install command succeeds, at most 30 training/prediction lines, and default settings are used. |
| 4 | Runs with a documented install and only a small change from the score-5 conditions. |
| 3 | Runs after a minor install fix, or has 31–80 training/prediction lines with several settings specified. |
| 2 | Requires a larger workaround or falls between the score-3 and score-1 conditions. |
| 1 | The installation or run fails, requires a major workaround, or has more than 80 training/prediction lines. |

Scores 2 and 4 are intermediate judgments. Training and prediction times are shown as evidence; the project plan does not define time cutoffs, so do not invent a timing penalty.

## Transparency

| Score | Rule |
|---:|---|
| 5 | The framework provides a useful explanation for an individual prediction. |
| 4 | It provides overall feature importance and individual explanations are straightforward with a standard method such as SHAP. |
| 3 | It provides overall feature importance, but not a practical individual explanation. |
| 2 | Explanation relies on general post-hoc methods such as permutation importance. |
| 1 | No practical explanation method is available in this project. |

For TabNet, record both `feature_importances_` and the per-row feature masks from `explain()`. For models using permutation importance, say that it describes aggregate validation PR-AUC sensitivity, not a causal effect.

## Efficacy

Use held-out test PR-AUC from `results/*.json`, including the combined model when it is available. Let `gap = best_test_PR_AUC - model_test_PR_AUC`:

| Score | Rule |
|---:|---|
| 5 | `gap <= 0.02` |
| 4 | `0.02 < gap <= 0.05` |
| 3 | `0.05 < gap <= 0.10` |
| 2 | `0.10 < gap <= 0.20` |
| 1 | `gap > 0.20` |

Show precision and recall at the validation-selected threshold beside the efficacy score. Treat PR-AUC gaps below 0.02 as practical ties in the written discussion, even though the rubric assigns a score of 5 to that band.

## Evidence record

For every framework, save the numeric score and the supporting measurements for all three dimensions. The comparison notebook stores fact fields, not a report-ready sentence. The student must write any report or presentation explanation in their own words.
