# Resume bullets grounded in measured results

Use these bullets only with a link to this repository and retain their stated scope.

- Built a reproducible ads-ranking study on 600K chronologically split Criteo display-ad impressions; diagnosed saturated embedding initialization, collision-prone ID hashing, an untrained time feature, and dropout at inference, raising held-out CTR AUC from 0.533 to 0.672 and post-click CVR AUC from 0.697 to 0.856 (mean of 3 seeds) using the same input fields.
- Implemented shared-embedding entire-space multi-task learning (ESMM) for `P(click)` and `P(click ∧ conversion)`, with training-only vocabularies and validation early stopping; documented that a clicked-only DeepFM still outperformed ESMM on post-click CVR (AUC 0.856 vs. 0.847) and CTCVR (0.840 vs. 0.833) rather than claiming an unsupported gain.
- Prototyped generative candidate retrieval with residual-quantized semantic IDs and autoregressive code generation on KuaiRand-Pure; benchmarked against popularity and item-ID GRU baselines, reporting the semantic model's lower Recall@50 (0.013 vs. 0.132) rather than claiming an unsupported gain.
