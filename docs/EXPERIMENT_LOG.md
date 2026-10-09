# Experiment log

All commands below were executed on 2026-10-09 with Python 3.14, PyTorch 2.14.1, and the seeds listed per run. Metrics are saved in the adjacent JSON artifacts.

| ID | Dataset | Split | Models | Result summary |
| --- | --- | --- | --- | --- |
| `criteo-600k-s7` (superseded) | Criteo Attribution | First 600,000 impressions, chronological 70/15/15 | DeepFM, ESMM | DeepFM CTR AUC 0.5326 vs ESMM 0.5318; clicked-only DeepFM CVR AUC 0.6967 vs ESMM 0.6887. Invalidated by the defects below. |
| `criteo-600k-v2-s789` | Criteo Attribution | Same rows and split; training-only vocabularies, validation early stopping | DeepFM, shared-embedding ESMM | Means over seeds 7/8/9. CTR AUC 0.6717 vs 0.6713; post-click CVR AUC 0.8564 vs 0.8474; CTCVR AUC 0.8395 vs 0.8331 (DeepFM vs ESMM). |
| `criteo-600k-v2-recency` | Criteo Attribution | As `criteo-600k-v2-s789`, plus a `time_since_last_click` bucket | DeepFM, shared-embedding ESMM | No measurable change: CTR AUC 0.6714 vs 0.6715, post-click CVR AUC 0.8576 vs 0.8470. Mean validation loss of CTR DeepFM 0.6066 with the feature vs 0.6065 without, so it stays opt-in. |
| `criteo-600k-v2-baselines` | Criteo Attribution | Same rows, split, and fields as `criteo-600k-v2-s789` | Constant training rate; L2 logistic regression on one-hot fields, C chosen on validation log loss from {0.01, 0.1, 1} | Deterministic. Logistic regression AUC: CTR 0.6682, post-click CVR 0.8485, CTCVR 0.8337. DeepFM leads it by 0.0035, 0.0079, and 0.0057; ESMM leads on CTR by 0.0031 and trails by 0.0010 and 0.0006. Constant log loss: 0.657, 0.399, 0.198. |
| `kuairand-120k-s7` (superseded) | KuaiRand-Pure | First 120,000 rows of the user-sorted standard logs, chronological 70/15/15 | popularity, item GRU, RQ-VAE semantic-ID GRU | Recall@50: 0.1322, 0.1027, and 0.0134 respectively. Invalidated by the defects below. |
| `kuairand-2500u-original-code` (control) | KuaiRand-Pure | Every standard-log interaction of users with id < 2,500, chronological 70/15/15 | Code at commit `8e3d125`, changed only to load this sample | Means over seeds 7/8/9. Recall@50: popularity 0.1199, item GRU 0.0991, semantic-ID GRU 0.0143. Ranking side study: DeepFM CTR AUC 0.6308, ESMM long-view proxy AUC 0.5200. Separates the effect of the code fixes from the sampling change. |
| `kuairand-2500u-v2-s789` | KuaiRand-Pure | Same sample and split | popularity, item GRU, semantic-ID GRU (co-occurrence vectors, deduplication token, GRU decoder) | Means over seeds 7/8/9. Recall@50: 0.1199, 0.1187, 0.1247; Recall@10: 0.0349, 0.0355, 0.0377. Ranking side study: DeepFM CTR AUC 0.7415, ESMM long-view proxy AUC 0.6090. |

The first KuaiRand run initially used a residual quantizer whose encoder lacked a straight-through reconstruction path. After adding straight-through estimation and retraining, semantic-ID Recall@50 rose from 0.0072 to 0.0134, but remained below both baselines.

## Why `criteo-600k-s7` was superseded

Each defect below was confirmed on the data or the code before it was fixed.

1. **Saturated initialization.** `nn.Embedding` initializes rows from N(0, 1). With 12 fields, the FM term sums 66 field-pair inner products, so untrained DeepFM logits had a standard deviation of 32.8 and 87% of initial predictions were beyond 0.993 or below 0.007. Rows that training never updated kept that noise at test time. Embeddings now start from N(0, 0.01), and first-order weights start at zero.
2. **A time index used as a category.** The 600,000 rows span 1.24 days. The training partition contains only day 0, while 31% of test rows fall on day 1, so the `time_bucket = 1` embedding was never trained. The field was removed: with barely more than one day of data, a daily cycle cannot be learned.
3. **Modulo hashing of high-cardinality IDs.** The training partition has 341,074 distinct `uid` values in 420,000 rows, and 87% of test users never appear in training. `uid mod 8192` mixed about 42 unrelated users per bucket, and `cat7` lost 7,603 of its 14,395 values to collisions. Vocabularies are now fit on the training partition only, and rare or unseen values share index 0, which training updates as a "new value" embedding.
4. **Dropout at inference.** No code called `model.eval()`, so dropout stayed active when test predictions were made.
5. **No model selection.** The validation partition was built but never used, and every model trained for a fixed five epochs. With early stopping on validation log loss, CTR models select epoch 1 and the clicked-only CVR model selects epoch 2 in every seed, which is consistent with the one-epoch overfitting commonly reported for CTR models with large ID embeddings.
6. **Unshared ESMM embeddings.** The two ESMM towers were independent DeepFMs, which removes ESMM's mechanism for letting click labels improve the conversion representation. The towers now share field embeddings, and each tower keeps its own first-order weights, FM dimension weights, and MLP.

ESMM still trails the clicked-only baseline on post-click CVR. Two factors are consistent with that and were not tuned away. First, the joint validation loss selects epoch 1 for ESMM, while the clicked-only CVR model improves until epoch 2, so the shared checkpoint is a compromise between tasks. Second, clicked test impressions are exactly the distribution the clicked-only model was trained on, which favors it on this metric by construction.

For reference, the ESMM paper (Ma et al., SIGIR 2018) reports on its public Taobao dataset (84M impressions, 3.4M clicks, 18k conversions, so about 0.53% of clicks convert) a CVR AUC of 0.6856 against 0.6600 for a single-task model trained on clicks (+0.0256), and a CTCVR AUC of 0.6532 against 0.6207. In this sample, 14.3% of training clicks convert, about 27 times the paper's rate, so the conversion sparsity that ESMM targets is much milder here. That explanation is consistent with the result but has not been tested directly, for example by downsampling training conversions.

## Why `kuairand-120k-s7` was superseded

Each defect below was confirmed on the data or the code before it was fixed.

1. **Row-based sampling of user-sorted files.** Both standard-log CSVs are sorted by `user_id`, not by time. The first 60,000 rows of the 4/8–4/21 file held users 0–2,542 (2,391 users, about 5% of that file), the first 60,000 rows of the 4/22–5/8 file held users 0–10,711 (9,428 users, about 20% of that file), and each cut ended inside one user's history. The sample now keeps every interaction of the users with an id below 2,500 in both periods.
2. **Item GRU overfitting.** On the old sample, validation Recall@50 peaked at epoch 3 (0.138) and fell to 0.093 by epoch 15, while the script trained a fixed 10 epochs. Training now stops on validation NLL and selects epoch 1 or 2.
3. **Under-trained quantizer.** Forty full-batch RQ-VAE steps produced 521 distinct IDs for 4,573 items, with up to 109 items per ID. Validation targets sat in ID groups of 27.9 items on average, which the scorer could only order arbitrarily. The quantizer now trains for 2,000 steps on standardized vectors and uses all 32 codes at both stages, so k-means codebook initialization was not needed.
4. **Non-unique IDs.** Two stages of 32 codes give at most 1,024 IDs for about 4,600 items, so collisions were unavoidable. A deduplication token, numbered by training popularity, now makes every ID unique.
5. **Quantized vectors that were mostly initialization noise.** The quantizer read the item GRU's input embeddings. Their norms (5.6 for items never seen as a training target, 5.8 for items seen ten or more times) stayed near the √32 ≈ 5.66 of N(0, 1) initialization. The vectors are now chosen on validation, as described below.
6. **Additive decoder.** The second code's logits were a linear function of [user state, first-code embedding], so the user-dependent part of the second-code distribution was identical under every first code. A GRU-cell decoder now conditions each token on the user and the prefix jointly.
7. **Ranking side study.** It had the same defects as the Criteo experiment: N(0, 1) embeddings, dropout at inference, and no use of the validation partition.

## Semantic-ID design selection

The design was chosen on validation Recall@50, the headline metric of the retrieval experiment, with seed 7. Validation NLL is shown for reference. The test partition was not scored during selection.

| Item vectors | Decoder | Validation Recall@50 | Validation NLL | Distinct IDs / largest group |
| --- | --- | ---: | ---: | --- |
| Item GRU input embeddings | GRU cell | 0.1272 | 7.923 | 1,013 / 12 |
| Item GRU input embeddings | additive | 0.1244 | 7.933 | 1,013 / 12 |
| Item GRU output weights | GRU cell | 0.1236 | 7.925 | 980 / 19 |
| Item GRU output weights | additive | 0.1240 | 7.963 | 980 / 19 |
| Co-occurrence (PPMI + SVD) | GRU cell | **0.1409** | 7.940 | 627 / 850 |
| Co-occurrence (PPMI + SVD) | additive | 0.1383 | 7.968 | 627 / 850 |
| Content (tags, upload/video/music type, duration) | GRU cell | 0.1282 | **7.823** | 230 / 263 |
| Content | additive | 0.1257 | 7.851 | 230 / 263 |
| Co-occurrence and content, concatenated | GRU cell | 0.1324 | 7.877 | 367 / 168 |

On the same validation partition, popularity reached 0.1325 and the item GRU 0.1314 (NLL 7.711).

The GRU-cell decoder had lower NLL than the additive decoder for every vector source. Co-occurrence vectors gave the best Recall@50 under both decoders, and content vectors gave the best NLL. Concatenating the two landed between them on both metrics, so it was not adopted. Items with no co-occurrence get zero vectors and share a single prefix (595 to 804 items across the final seeds). That costs likelihood on rare targets but leaves the top of the ranking intact.

In the ranking side study, early stopping on the joint ESMM loss selects epoch 1. A run of the code at commit `722f685` (initialization fixed, a fixed 10 epochs, no early stopping) on the same sample reached long-view proxy AUC 0.620, against 0.609 here. As in the Criteo experiment, the joint checkpoint favors the click task. That comparison was observed on the test partition and was not used to change the configuration.
