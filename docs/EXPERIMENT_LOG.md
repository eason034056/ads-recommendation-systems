# Experiment log

All commands below were executed on 2026-10-09 with Python 3.14, PyTorch 2.14.1, and the seeds listed per run. Metrics are saved in the adjacent JSON artifacts.

| ID | Dataset | Split | Models | Result summary |
| --- | --- | --- | --- | --- |
| `criteo-600k-s7` (superseded) | Criteo Attribution | First 600,000 impressions, chronological 70/15/15 | DeepFM, ESMM | DeepFM CTR AUC 0.5326 vs ESMM 0.5318; clicked-only DeepFM CVR AUC 0.6967 vs ESMM 0.6887. Invalidated by the defects below. |
| `criteo-600k-v2-s789` | Criteo Attribution | Same rows and split; training-only vocabularies, validation early stopping | DeepFM, shared-embedding ESMM | Means over seeds 7/8/9. CTR AUC 0.6717 vs 0.6713; post-click CVR AUC 0.8564 vs 0.8474; CTCVR AUC 0.8395 vs 0.8331 (DeepFM vs ESMM). |
| `criteo-600k-v2-recency` | Criteo Attribution | As `criteo-600k-v2-s789`, plus a `time_since_last_click` bucket | DeepFM, shared-embedding ESMM | No measurable change: CTR AUC 0.6714 vs 0.6715, post-click CVR AUC 0.8576 vs 0.8470. Mean validation loss of CTR DeepFM 0.6066 with the feature vs 0.6065 without, so it stays opt-in. |
| `kuairand-120k-s7` | KuaiRand-Pure | First 120,000 standard-log rows, chronological 70/15/15 | popularity, item GRU, RQ-VAE semantic-ID GRU | Recall@50: 0.1322, 0.1027, and 0.0134 respectively. |

The KuaiRand run initially used a residual quantizer whose encoder lacked a straight-through reconstruction path. After adding straight-through estimation and retraining, semantic-ID Recall@50 rose from 0.0072 to 0.0134, but remained below both baselines. The repository retains the final code and records the non-winning result because it is material to an honest technical discussion.

## Why `criteo-600k-s7` was superseded

Each defect below was confirmed on the data or the code before it was fixed.

1. **Saturated initialization.** `nn.Embedding` initializes rows from N(0, 1). With 12 fields, the FM term sums 66 field-pair inner products, so untrained DeepFM logits had a standard deviation of 32.8 and 87% of initial predictions were beyond 0.993 or below 0.007. Rows that training never updated kept that noise at test time. Embeddings now start from N(0, 0.01), and first-order weights start at zero.
2. **A time index used as a category.** The 600,000 rows span 1.24 days. The training partition contains only day 0, while 31% of test rows fall on day 1, so the `time_bucket = 1` embedding was never trained. The field was removed: with barely more than one day of data, a daily cycle cannot be learned.
3. **Modulo hashing of high-cardinality IDs.** The training partition has 341,074 distinct `uid` values in 420,000 rows, and 87% of test users never appear in training. `uid mod 8192` mixed about 42 unrelated users per bucket, and `cat7` lost 7,603 of its 14,395 values to collisions. Vocabularies are now fit on the training partition only, and rare or unseen values share index 0, which training updates as a "new value" embedding.
4. **Dropout at inference.** No code called `model.eval()`, so dropout stayed active when test predictions were made.
5. **No model selection.** The validation partition was built but never used, and every model trained for a fixed five epochs. With early stopping on validation log loss, CTR models select epoch 1 and the clicked-only CVR model selects epoch 2 in every seed, which is consistent with the one-epoch overfitting commonly reported for CTR models with large ID embeddings.
6. **Unshared ESMM embeddings.** The two ESMM towers were independent DeepFMs, which removes ESMM's mechanism for letting click labels improve the conversion representation. The towers now share field embeddings, and each tower keeps its own first-order weights, FM dimension weights, and MLP.

ESMM still trails the clicked-only baseline on post-click CVR. Two factors are consistent with that and were not tuned away. First, the joint validation loss selects epoch 1 for ESMM, while the clicked-only CVR model improves until epoch 2, so the shared checkpoint is a compromise between tasks. Second, clicked test impressions are exactly the distribution the clicked-only model was trained on, which favors it on this metric by construction.
