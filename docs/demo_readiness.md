# Demo readiness — 16 November 2026

The repository is prepared for the tentative 17 November 2026 demo. The exact time is still unconfirmed.

## Completed

- Corrected 18-slide presentation, including current metrics, team names, EDA wording, RMSE/MAE/R2, current ensemble members, amount units, hardware, and a valid demo command.
- Twelve-page report with a title page, references, repository link, captions, and dynamic page numbers beginning after the title page.
- Saved outputs in all eleven notebooks. Every code cell has an execution count and no notebook contains a saved error output. Cells that normally produce no visible output remain valid.
- Fresh isolated CPU runs of Keras, Lightning, fastai, TabNet, ensemble, money, and comparison. Details and rerun metrics are in `results/verification/2026-11-16/summary.json`.
- Team-approved Simplicity, Transparency and Efficacy scores in `results/model_scores.csv`.
- Timed local `src/data.py` check and viva answers in `docs/demo_rehearsal.md`.

The report and slides use the frozen published results in top-level `results/*.json`. Fresh neural-model reruns vary because their software versions, hardware and stochastic training differ; those verification results are kept separately and do not silently replace the published experiment.

## Presenter actions

- Confirm the exact demo time.
- Share the Google Doc link with Vidhwath S and Yakshith Naidu, as planned.
- Practise the spoken three-minute delivery once.
- Copy the final offline pack to the laptop and pen drive, open the copies, charge the laptop, and check the projector.
- Run or inspect a fresh Colab setup before the demo if time permits.
