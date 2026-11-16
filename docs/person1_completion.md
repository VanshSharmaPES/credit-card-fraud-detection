# Person 1 completion and verification record

Checked 16 November 2026. This evidence record supersedes the stale task checkboxes; it does not certify team meetings or personal rehearsal.

## Deliverables prepared

- EDA findings: `docs/eda_findings.md`, merged through PR #8. The six findings were checked against recomputed training-split statistics. Hour means elapsed-hour bin, hourly rates have small fraud counts, and correlation is linear association only.
- Report: native Google Doc created with Introduction, Methodology, EDA, Results and Conclusion. Repository copies are `output/credit_card_fraud_report_draft.pdf` and `.docx`; `docs/person1_report_draft.md` mirrors the text. The report has a title page, the corrected team names, six EDA figures, a SHAP case, six captioned tables, references and the repository link. It has no page numbers. No faculty template was requested.
- Results and Conclusion review: three comments in the Google Doc verify the imbalance metrics, SHAP output scale/sample, and limitations. No Person 2 review comments were present when checked; team review has not been certified.
- Slides: `docs/person1_slide_text.md` contains slides 2, 3, 4A, 4B, the SHAP half of 11 and slide 12, including chart choices and speaker notes. It is ready for Person 2 to integrate into the team deck.
- Viva: `docs/viva_questions.md` answers all eight Person 1 topics and includes the other frameworks' preparation notes. It now explains this project's SHAP log-odds scale and sample limitation explicitly.
- Demo: `docs/person1_demo.md` provides a three-minute script, expected data summary, two chart choices and an honest saved-output fallback.

## Execution and numeric checks

The README dependency commands were exercised in a clean clone and a new Python 3.14.10 environment. The raw download was 150,828,752 bytes and passed the dataset checks. The data pipeline consistently produced 170,235 / 56,745 / 56,746 rows with 284 / 94 / 95 frauds.

Notebooks `01_eda`, `02_xgboost` and `07_imbalance` ran top to bottom in separate fresh kernels, with 11, 14 and 7 executed code cells respectively. Their saved repository outputs contain no errors. Local GPU reproduction matched every displayed metric to four decimal places. The largest measured difference from the official CPU result was about 5.7e-7; prediction files were not byte-identical. Official CPU metrics, predictions and models were retained.

The public Colab clone path no longer requires `GH_TOKEN` in these three notebooks. A fresh browser-based Colab run is still a day-before-demo action, not something the local reproduction establishes.

`00_checks.ipynb` also received a repository-relative import path and was executed from the `notebooks/` directory. Its existing metric sanity assertions passed, and the success output is now saved.

The report and Person 1 slide metric tables were checked against `results/*.json`, `results/imbalance/*.json`, `results/model_scores_draft.csv` and `results/money_f1_vs_cost.csv`. EDA values were independently recomputed from the training split. The twelve-page report export was rendered and visually checked.

## XGBoost score review (P1-36)

| Dimension | Draft score | Evidence against `scoring_rules.md` |
| --- | --- | --- |
| Simplicity | 4 | Documented installation succeeds; train/predict cells contain 8 + 5 = 13 nonblank, noncomment lines; six settings differ from defaults. Score 4 is an intermediate judgment because default settings are not used. |
| Transparency | 4 | Built-in gain importance plus an individual SHAP explanation; the explanation is post-hoc, rather than a native per-row explanation method. |
| Efficacy | 5 | Best framework-comparison PR-AUC 0.8488 minus XGBoost 0.8382 gives a gap about 0.0106, within 0.02. Precision 0.9740 and recall 0.7895 are shown alongside it. |

The comparison notebook's score-generation section was executed separately and regenerated the checked-in evidence CSV exactly. The previous count of 14 lines was corrected to 13. The numeric scores were not changed. Team agreement remains necessary; the rubric was written after some model results existed and was not preregistered. Timings from different CPU environments are not a controlled speed ranking.

## Repository and setup

- Base and XGBoost dependencies are pinned and their installation was exercised in the reproduction environment.
- Vidhwath S and Yakshith Naidu have repository collaborator access under their recorded GitHub accounts.
- The repository remains public, as requested. The user will share the Google Doc link directly; report edit access for teammates is not claimed.
- PR #17 carries the report, slide text, score correction, setup fixes and this handoff. The `v1.0` tag and draft release identify the repository handoff; a draft release is not final team approval.

## Actions still requiring a person or teammate

- Share the Google Doc link and ensure both teammates can edit. Review and adapt the prepared wording, integrate Person 1's slide text, and agree the subjective scores as a team.
- Ask Person 2 to save the executed Keras and Lightning notebooks, and the ensemble notebook. Their repository notebooks currently lack saved execution outputs. fastai, TabNet and money notebooks have saved result outputs but unexecuted setup cells. Their model result files are present; that does not prove those notebook files were run top to bottom.
- The Lightning dependency installation, imports and data-loading source had been stored in a Markdown cell (cell index 3). It is now a code cell, and all its code cells pass Python compilation after IPython syntax transformation. Full execution in the isolated Lightning environment is still unverified; the cell-type repair is not evidence of a completed training run.
- Read the full report and all viva answers, resolve any new review comments, and rehearse the actual three-minute segment with a timer.
- Perform the fresh Colab and physical backup/projector checks near the demo date. The demo date has not been provided, so no date-specific rehearsal is claimed or scheduled.
- Confirm the assembled team deck contains regression metrics, score evidence and cost analysis, then copy the approved deck into the offline pack. Person 1's text is prepared; the existing Person 2 deck is not certified as the final integrated presentation.

The offline notebook PDFs preserve the outputs actually saved in the repository; they do not manufacture execution results for unexecuted cells. The separate notebook audit makes these gaps visible.
