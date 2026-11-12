# Person 2: evaluation, neural models, ensemble, and presentation

The Keras and PyTorch Lightning models use the repository's shared stratified train/validation/test splits, class weighting, and validation-only threshold selection. `src/evaluate.py` writes the agreed metrics and prediction schema. The combined model averages the saved XGBoost, Lightning, and Keras probabilities; its membership is selected using validation PR-AUC.

## Recorded results

These results were recorded on an Apple Silicon ARM64 CPU using the same split files. They are not Colab or GPU timings.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (ensemble input) | 0.9620 | 0.8000 | 0.8736 | 0.8342 | 0.9754 | 3.469 |
| Keras | 0.7957 | 0.7789 | 0.7872 | 0.7103 | 0.9628 | 1.207 |
| Lightning | 0.8824 | 0.7895 | 0.8333 | 0.7576 | 0.9629 | 9.991 |
| Combined | 0.9615 | 0.7895 | 0.8671 | 0.8501 | 0.9685 | — |

The ensemble had the highest PR-AUC among these rows, while XGBoost had the highest F1 and recall. Thresholds were selected on validation data; the reported metrics are from the held-out test split. Full confusion counts, thresholds, regression metrics, hardware, and notes are in the corresponding `results/*.json` files.

## Framework note

The local Flash installation attempt failed because the environment could not import `setuptools.build_meta`. The submitted notebook therefore uses plain PyTorch Lightning. Flash and Colab execution have not been verified.

## Reproduction

From the repository root, run `python src/data.py`, then install `requirements/base.txt` and the relevant framework requirements. Run `notebooks/03_keras.ipynb` and `notebooks/04_lightning.ipynb` in separate environments. `notebooks/08_ensemble.ipynb` reads saved model predictions and does not retrain the members. The raw dataset and processed splits are intentionally not committed.
