"""Leakage-safe CTR/CVR experiment on Criteo Attribution Modeling data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adsrec.criteo import load_criteo_sample, make_criteo_data
from adsrec.evaluation import binary_metrics
from adsrec.models import DeepFM, ESMM


def loader(*arrays: np.ndarray, shuffle: bool = False) -> DataLoader:
    return DataLoader(TensorDataset(*[torch.tensor(value) for value in arrays]), batch_size=1024, shuffle=shuffle)


def train_deepfm(model: DeepFM, features: np.ndarray, labels: np.ndarray, epochs: int) -> DeepFM:
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        for batch_features, batch_labels in loader(features, labels, shuffle=True):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(batch_features), batch_labels)
            loss.backward(); optimizer.step()
    return model


def train_esmm(model: ESMM, training: tuple[np.ndarray, np.ndarray, np.ndarray], epochs: int) -> ESMM:
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        for features, click, click_conversion in loader(*training, shuffle=True):
            optimizer.zero_grad()
            pctr, pctcvr = model(features)
            loss = nn.functional.binary_cross_entropy(pctr, click) + nn.functional.binary_cross_entropy(pctcvr, click_conversion)
            loss.backward(); optimizer.step()
    return model


@torch.no_grad()
def probability(model: DeepFM, features: np.ndarray) -> np.ndarray:
    return torch.sigmoid(model(torch.tensor(features))).numpy()


@torch.no_grad()
def esmm_probability(model: ESMM, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pctr, pctcvr = model(torch.tensor(features))
    return pctr.numpy(), (pctcvr / pctr.clamp_min(1e-6)).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/criteo_attribution_dataset.tsv.gz"))
    parser.add_argument("--output", type=Path, default=Path("experiments/criteo_attribution"))
    parser.add_argument("--rows", type=int, default=600_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    data = make_criteo_data(load_criteo_sample(args.data, args.rows))
    ctr_baseline = train_deepfm(DeepFM(data.cardinalities), data.train[0], data.train[1], args.epochs)
    clicked = data.train[1] == 1
    post_click_baseline = train_deepfm(DeepFM(data.cardinalities), data.train[0][clicked], data.train[2][clicked], args.epochs)
    esmm = train_esmm(ESMM(data.cardinalities), data.train, args.epochs)
    ctr_deepfm = probability(ctr_baseline, data.test[0])
    cvr_post_click = probability(post_click_baseline, data.test[0])
    ctr_esmm, cvr_esmm = esmm_probability(esmm, data.test[0])
    clicked_test = data.test[1] == 1
    results = {
        "dataset": "Criteo Attribution Modeling for Bidding Dataset",
        "license": "CC-BY-NC-SA-4.0",
        "rows": args.rows,
        "seed": args.seed,
        "split": "chronological 70/15/15 by impression timestamp; mappings are fixed hashing, not fit on future partitions",
        "ctr": {"deepfm": binary_metrics(data.test[1], ctr_deepfm), "esmm": binary_metrics(data.test[1], ctr_esmm)},
        "post_click_cvr": {"deepfm_clicked_only": binary_metrics(data.test[2][clicked_test], cvr_post_click[clicked_test]), "esmm_entire_space": binary_metrics(data.test[2][clicked_test], cvr_esmm[clicked_test])},
        "label_definition": "CVR uses conversion within 30 days after an impression and is evaluated on clicked test impressions; ESMM's CTCVR target is click AND conversion.",
        "limitations": ["The Criteo conversion label can include a conversion attributed to another impression; this is an offline prediction study, not causal incrementality.", "Data are an anonymized, sub-sampled 30-day traffic sample; do not infer production lift, bidding utility, or TikTok performance."]
    }
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
