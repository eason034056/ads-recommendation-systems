# Resume bullets grounded in measured results

Use these bullets only with a link to this repository and retain their stated scope. Every number below appears in `experiments/criteo_attribution/metrics.json` or `docs/EXPERIMENT_LOG.md`.

**Ads CTR/CVR ranking on Criteo Attribution data** (PyTorch, scikit-learn)

- Built a leakage-safe offline CTR/CVR evaluation on 600K chronologically split Criteo display-ad impressions, with training-only vocabularies, validation-only model selection, 3-seed mean ± std, and logistic-regression baselines.
- Diagnosed silent failures in an initial DeepFM/ESMM pipeline (N(0,1) embeddings saturating FM logits at an initial std of 32.8, modulo-hash collisions, a time feature never seen in training, dropout left on at inference), raising CTR/CVR AUC from 0.53/0.70 to 0.67/0.86.
- Showed that DeepFM beat a one-hot logistic regression by only 0.0035 CTR AUC and 0.0079 CVR AUC, and that shared-embedding ESMM did not beat it on post-click CVR (0.8474 vs. 0.8485), indicating that extra model capacity buys little on these 11 fields and 1.24 days of data.
