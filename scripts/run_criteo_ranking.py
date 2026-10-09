"""Leakage-safe CTR/CVR experiment on Criteo Attribution Modeling data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from adsrec.criteo import CriteoData, load_criteo_sample, make_criteo_data
from adsrec.evaluation import binary_metrics, summarize
from adsrec.linear import fit_logistic, one_hot
from adsrec.models import DeepFM, ESMM
from adsrec.training import fit


@torch.no_grad()
def probability(model: DeepFM, features: np.ndarray) -> np.ndarray:
    return torch.sigmoid(model(torch.tensor(features))).numpy()


@torch.no_grad()
def esmm_probability(model: ESMM, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pctr, pcvr, pctcvr = model(torch.tensor(features))
    return pctr.numpy(), pcvr.numpy(), pctcvr.numpy()


def run_seed(data: CriteoData, seed: int, max_epochs: int, patience: int) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    features, clicks, click_conversion = data.train
    val_features, val_clicks, val_click_conversion = data.validation
    clicked, val_clicked = clicks == 1, val_clicks == 1
    logits_loss, probability_loss = nn.functional.binary_cross_entropy_with_logits, nn.functional.binary_cross_entropy

    ctr_baseline = DeepFM(data.cardinalities)
    ctr_fit = fit(ctr_baseline, (features, clicks), lambda model, batch: logits_loss(model(batch[0]), batch[1]),
                  lambda model: binary_metrics(val_clicks, probability(model, val_features))["logloss"], max_epochs, patience)
    # The classic post-click baseline: train only where the conversion label is defined (clicked impressions).
    post_click_baseline = DeepFM(data.cardinalities)
    post_click_fit = fit(post_click_baseline, (features[clicked], click_conversion[clicked]), lambda model, batch: logits_loss(model(batch[0]), batch[1]),
                         lambda model: binary_metrics(val_click_conversion[val_clicked], probability(model, val_features[val_clicked]))["logloss"], max_epochs, patience)

    def esmm_loss(model: ESMM, batch: list[torch.Tensor]) -> torch.Tensor:
        pctr, _, pctcvr = model(batch[0])
        return probability_loss(pctr, batch[1]) + probability_loss(pctcvr, batch[2])

    def esmm_validation(model: ESMM) -> float:
        pctr, _, pctcvr = esmm_probability(model, val_features)
        return binary_metrics(val_clicks, pctr)["logloss"] + binary_metrics(val_click_conversion, pctcvr)["logloss"]

    esmm = ESMM(data.cardinalities)
    esmm_fit = fit(esmm, data.train, esmm_loss, esmm_validation, max_epochs, patience)

    test_features, test_clicks, test_click_conversion = data.test
    test_clicked = test_clicks == 1
    ctr_deepfm, cvr_deepfm = probability(ctr_baseline, test_features), probability(post_click_baseline, test_features)
    ctr_esmm, cvr_esmm, ctcvr_esmm = esmm_probability(esmm, test_features)
    return {
        "seed": seed,
        "training": {"deepfm_ctr": ctr_fit, "deepfm_clicked_only": post_click_fit, "esmm": esmm_fit},
        "ctr": {"deepfm": binary_metrics(test_clicks, ctr_deepfm), "esmm": binary_metrics(test_clicks, ctr_esmm)},
        "post_click_cvr": {"deepfm_clicked_only": binary_metrics(test_click_conversion[test_clicked], cvr_deepfm[test_clicked]),
                           "esmm_entire_space": binary_metrics(test_click_conversion[test_clicked], cvr_esmm[test_clicked])},
        # Ranking ads by expected conversions per impression needs pCTR * pCVR on every impression, clicked or not.
        "ctcvr": {"deepfm_ctr_times_clicked_only_cvr": binary_metrics(test_click_conversion, ctr_deepfm * cvr_deepfm),
                  "esmm": binary_metrics(test_click_conversion, ctcvr_esmm)},
    }


def run_baselines(data: CriteoData) -> dict:
    """Seed-free references on the same fields: the training base rate, and one-hot logistic regression."""
    (features, clicks, click_conversion), validation, (test_features, test_clicks, test_click_conversion) = data.train, data.validation, data.test
    clicked, val_clicked, test_clicked = clicks == 1, validation[1] == 1, test_clicks == 1
    ctr_model, ctr_strength = fit_logistic((features, clicks), (validation[0], validation[1]), data.cardinalities)
    cvr_model, cvr_strength = fit_logistic((features[clicked], click_conversion[clicked]), (validation[0][val_clicked], validation[2][val_clicked]), data.cardinalities)
    test_x = one_hot(test_features, data.cardinalities)
    pctr, pcvr = ctr_model.predict_proba(test_x)[:, 1], cvr_model.predict_proba(test_x)[:, 1]
    base_ctr, base_cvr = float(clicks.mean()), float(click_conversion[clicked].mean())
    constant = lambda rate, labels: binary_metrics(labels, np.full(len(labels), rate))
    return {
        "logistic_regression_C": {"ctr": ctr_strength, "post_click_cvr": cvr_strength},
        "ctr": {"constant": constant(base_ctr, test_clicks), "logistic_regression": binary_metrics(test_clicks, pctr)},
        "post_click_cvr": {"constant": constant(base_cvr, test_click_conversion[test_clicked]),
                           "logistic_regression_clicked_only": binary_metrics(test_click_conversion[test_clicked], pcvr[test_clicked])},
        "ctcvr": {"constant": constant(base_ctr * base_cvr, test_click_conversion),
                  "logistic_regression_ctr_times_cvr": binary_metrics(test_click_conversion, pctr * pcvr)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/criteo_attribution_dataset.tsv.gz"))
    parser.add_argument("--output", type=Path, default=Path("experiments/criteo_attribution"))
    parser.add_argument("--rows", type=int, default=600_000)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--recency", action="store_true", help="ablation: add the time-since-last-click bucket")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = make_criteo_data(load_criteo_sample(args.data, args.rows), args.min_count, args.recency)
    runs = [run_seed(data, seed, args.max_epochs, args.patience) for seed in args.seeds]
    results = {
        "dataset": "Criteo Attribution Modeling for Bidding Dataset",
        "license": "CC-BY-NC-SA-4.0",
        "rows": args.rows,
        "seeds": args.seeds,
        "split": "chronological 70/15/15 by impression timestamp; vocabularies are fit on the training partition only",
        "features": data.feature_names,
        "config": {"min_count": args.min_count, "max_epochs": args.max_epochs, "patience": args.patience, "selection": "lowest validation logloss per model"},
        "summary": summarize([{task: run[task] for task in ("ctr", "post_click_cvr", "ctcvr")} for run in runs]),
        "baselines": run_baselines(data),
        "runs": runs,
        "label_definition": "CVR uses conversion within 30 days after an impression and is evaluated on clicked test impressions; CTCVR (click AND conversion) is evaluated on all test impressions.",
        "limitations": ["The Criteo conversion label can include a conversion attributed to another impression; this is an offline prediction study, not causal incrementality.", "Data are an anonymized, sub-sampled 30-day traffic sample; do not infer production lift, bidding utility, or TikTok performance."],
    }
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"summary": results["summary"], "baselines": results["baselines"]}, indent=2))


if __name__ == "__main__":
    main()
