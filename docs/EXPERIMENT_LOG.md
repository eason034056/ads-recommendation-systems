# Experiment log

All commands below were executed on 2026-10-09 with Python 3.14, PyTorch 2.14.1, and seed 7. Metrics are saved in the adjacent JSON artifacts.

| ID | Dataset | Split | Models | Result summary |
| --- | --- | --- | --- | --- |
| `criteo-600k-s7` | Criteo Attribution | First 600,000 impressions, chronological 70/15/15 | DeepFM, ESMM | DeepFM CTR AUC 0.5326 vs ESMM 0.5318; clicked-only DeepFM CVR AUC 0.6967 vs ESMM 0.6887. |
| `kuairand-120k-s7` | KuaiRand-Pure | First 120,000 standard-log rows, chronological 70/15/15 | popularity, item GRU, RQ-VAE semantic-ID GRU | Recall@50: 0.1322, 0.1027, and 0.0134 respectively. |

The KuaiRand run initially used a residual quantizer whose encoder lacked a straight-through reconstruction path. After adding straight-through estimation and retraining, semantic-ID Recall@50 rose from 0.0072 to 0.0134, but remained below both baselines. The repository retains the final code and records the non-winning result because it is material to an honest technical discussion.
