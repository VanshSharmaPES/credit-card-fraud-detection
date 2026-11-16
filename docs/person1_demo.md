# Person 1 demo and offline fallback

Use this as a rehearsal script, then explain it naturally. The planned duration is three minutes. The technical path was timed on 16 November 2026; see `demo_rehearsal.md`.

## Before starting

Open the README, a terminal at the repository root with the project environment active, and these two charts:

- `figures/eda_class_balance.png`
- `figures/eda_top_features.png`

Run `python src/data.py` once before the session so the raw CSV is cached. Keep the report PDF, notebook PDFs, slide text, and the approved team presentation available offline. Do not depend on a network download during the demonstration.

## 0:00–0:35 — Repository and question

“We compare five machine learning frameworks on the same fraud dataset and split. Fraud is rare, so we report precision, recall, F1 and PR-AUC rather than relying on accuracy. The README links the code, saved metrics and charts.”

Point to the README's split table and results table. Say that thresholds were selected on validation, then fixed for test evaluation.

## 0:35–1:25 — Live data pipeline

Run from the repository root:

```bash
python src/data.py
```

“The pipeline checks the raw dataset, removes exact duplicates, creates the amount and elapsed-hour features, and makes a stratified split. The scaler is fitted on training data only.”

Confirm the following summary, allowing for whitespace:

```text
Raw rows: 284807, raw frauds: 492
Duplicates dropped: 1081
Rows after cleaning: 283726, frauds: 473
train rows: 170235  frauds: 284  fraud share: 0.1668%
val   rows: 56745   frauds: 94   fraud share: 0.1657%
test  rows: 56746   frauds: 95   fraud share: 0.1674%
Class weight (train normal / train fraud): 169951 / 284 = 598.4190
```

If the command fails or takes more than about 20 seconds, switch to the saved setup output in `01_eda.ipynb` or its PDF. Say that you are showing the saved run, and continue. Do not debug during the demo.

## 1:25–2:05 — Class balance

Show the class-balance chart. “Training contains 169,951 normal transactions and 284 frauds. Predicting every transaction as normal gives about 99.83% accuracy but catches no fraud. That is why recall and alert precision matter.”

## 2:05–2:45 — Feature distributions

Show the top-feature chart. “The fraud and normal distributions differ for these components, but they overlap. These are anonymized PCA components, so we cannot give them business meanings. Pearson correlation measures linear association and does not describe all the separation visible here.”

## 2:45–3:00 — Handoff

“The same split is used for each framework. XGBoost caught 75 of the 95 test frauds with two false alerts. The following results compare that trade-off with the other frameworks.”

## Rehearsal and day-before checklist

- Rehearse with a timer and answer three randomly selected questions from `viva_questions.md`.
- Two days before the demo, reread the report and everyone's viva answers.
- The day before, run the setup cell and `src/data.py` in a fresh Colab runtime and compare the split summary. Person 1 notebooks use the public clone URL; no GitHub token is needed.
- Open every notebook or its PDF and inspect the saved outputs. All eleven repository notebooks now contain saved execution counts and no saved error output.
- Copy the report PDF, approved team presentation, notebook PDFs and the two charts to the laptop and a pen drive; open the copies to verify them.
- Confirm the public repository opens while signed out. Charge the laptop and check the projector.
- If faculty scope changes, use a branch, rerun affected notebooks, update report and slides, and prepare a new patch release. No scope change has been reported.
