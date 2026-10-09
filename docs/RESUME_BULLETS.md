# Resume bullets grounded in measured results

Use these bullets only with a link to this repository and retain their stated scope.

- Built a reproducible ads-ranking study on 600K chronologically split Criteo display-ad impressions, comparing DeepFM and ESMM for CTR and post-click conversion prediction; reported both ROC-AUC and PR-AUC under a fixed, leakage-safe protocol.
- Implemented entire-space multi-task learning for `P(click)` and `P(click ∧ conversion)` and evaluated post-click CVR only on clicked impressions; documented that clicked-only DeepFM outperformed ESMM in this run (AUC 0.697 vs. 0.689) rather than claiming an unsupported gain.
- Prototyped generative candidate retrieval with residual-quantized semantic IDs and autoregressive code generation on KuaiRand-Pure; benchmarked against popularity and item-ID GRU baselines, reporting the semantic model's lower Recall@50 (0.013 vs. 0.132) rather than claiming an unsupported gain.
