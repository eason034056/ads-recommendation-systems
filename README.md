# Ads CTR/CVR Ranking on Criteo Attribution Data

Conversion-optimized ad ranking needs each ad's expected conversions per impression, pCTR × pCVR. This project builds pCTR and pCVR models on public display-ad logs and tests one question: under leakage-safe evaluation, does multi-task learning (ESMM) beat single-task models (DeepFM and logistic regression)?

## Results

The data are 600,000 time-ordered impressions from Criteo's Attribution Modeling for Bidding Dataset. Scores are on the final 15% of time, as the mean ± standard deviation over seeds 7, 8, and 9; logistic regression is deterministic.

| Model | CTR AUC | Post-click CVR AUC | CTCVR AUC |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.6682 | 0.8485 | 0.8337 |
| DeepFM | 0.6717 ± 0.0013 | 0.8564 ± 0.0010 | 0.8395 ± 0.0007 |
| ESMM | 0.6713 ± 0.0005 | 0.8474 ± 0.0004 | 0.8331 ± 0.0008 |

The single-task models train separate CTR and CVR models, the CVR model on clicked impressions only, and score CTCVR as pCTR × pCVR. ESMM trains one model with shared embeddings on all impressions. Log loss, PR-AUC, and calibration are in [metrics.json](experiments/criteo_attribution/metrics.json).

## Findings

- **DeepFM is the strongest model**, at 0.67 CTR AUC and 0.86 post-click CVR AUC.
- **ESMM does not beat it.** ESMM ties DeepFM on CTR but trails it on post-click CVR (0.847 vs. 0.856) and CTCVR, by about ten times the across-seed standard deviation.
- **Logistic regression stays within 0.01 AUC** of DeepFM on every task, so extra model capacity buys little on these 11 fields and 1.24 days of data.
- **ESMM's published gain did not reproduce.** The ESMM paper reports +0.026 CVR AUC on Taobao data in which about 0.53% of clicks convert. Here 14.3% of training clicks convert, about 27 times as many, which weakens the sparsity problem ESMM targets. This explanation fits the result but has not been tested directly.

## Evaluation protocol

- The first 600,000 impressions (about 1.24 days) are split 70/15/15 by time.
- Features are 11 categorical fields known at impression time (`uid`, `campaign`, `cat1`–`cat9`). Vocabularies are fit on the training partition only; values seen fewer than two times, or first seen later, share an out-of-vocabulary index.
- Each neural model keeps the epoch with the lowest validation log loss, and logistic regression's L2 strength is chosen on validation log loss.
- Post-click CVR is scored on clicked test impressions, the only ones with that label. CTCVR is scored on all test impressions.

[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) records the full history, including defects fixed in an earlier version (which scored 0.53 CTR AUC) and a recency-feature ablation that left results unchanged.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_criteo_attribution.py --output data/raw  # data are not committed
PYTHONPATH=src .venv/bin/python scripts/run_criteo_ranking.py              # defaults reproduce the table
PYTHONPATH=src .venv/bin/pytest -q
```

## Scope

- This is an offline study, with no auction simulator, bid optimizer, online A/B test, or production deployment.
- The conversion label marks a conversion within 30 days of an impression, which may be credited to another impression. It is a prediction target, not causal incrementality.
- The dataset is licensed CC BY-NC-SA 4.0 and is downloaded locally, not redistributed; see [NOTICE.md](NOTICE.md).
- The repository also holds a separate generative-retrieval experiment on KuaiRand-Pure (`scripts/run_experiment.py`), documented in the experiment log. It is not part of this study.
