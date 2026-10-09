# Resume bullets grounded in measured results

Use these bullets only with a link to this repository and retain their stated scope. Every number below appears in `experiments/criteo_attribution/metrics.json` or `docs/EXPERIMENT_LOG.md`.

**Ads CTR/CVR ranking on Criteo Attribution data** (PyTorch, scikit-learn)

- Built pCTR/pCVR models for conversion-optimized ad ranking on 600K time-ordered Criteo impressions, testing whether multi-task ESMM beats single-task DeepFM and logistic regression under leakage-safe evaluation.
- Achieved 0.67 CTR / 0.86 CVR AUC with DeepFM (3 seeds); ESMM tied on CTR but trailed on CVR (0.847 vs. 0.856), and logistic regression stayed within 0.01 AUC.
- Analyzed why ESMM's published gain (+0.026 CVR AUC) did not reproduce: conversions here are ~27× less sparse than in its paper's data, weakening the problem ESMM targets.
