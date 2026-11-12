# Simplicity notes (Person 2)

- Keras: compact dense 64-32 network, class-weighted binary cross-entropy, early stopping on validation PR-AUC.
- Lightning: same dense shape and validation protocol, using weighted BCE-with-logits and early stopping.
- Flash was attempted locally but installation stopped with `BackendUnavailable: Cannot import setuptools.build_meta`; plain PyTorch Lightning is the completed route.
- Local runtime measurements are ARM64 CPU measurements. Colab and GPU execution were not verified.
- The combined model is an arithmetic mean of XGBoost, Lightning, and Keras probabilities. Member choice is based on validation PR-AUC only.
