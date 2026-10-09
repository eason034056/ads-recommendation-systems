"""KuaiRand-Pure retrieval study (popularity, item-ID GRU, semantic-ID GRU) with a click / long-view ranking side study."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from adsrec.datasets import RankingData, RetrievalData, load_kuairand_logs, make_ranking_data, make_retrieval_data
from adsrec.evaluation import binary_metrics, ranking_metrics, summarize
from adsrec.models import DeepFM, ESMM, ItemGRU, RQVAE, SemanticGRU
from adsrec.semantic_ids import collision_stats, cooccurrence_vectors, deduplicate
from adsrec.training import fit

CODEBOOK_SIZE = 32


@torch.no_grad()
def item_scores(model: ItemGRU, history: np.ndarray) -> np.ndarray:
    return model(torch.tensor(history)).numpy()[:, 1:]


@torch.no_grad()
def item_nll(model: ItemGRU, history: np.ndarray, target: np.ndarray) -> float:
    return nn.functional.cross_entropy(model(torch.tensor(history)), torch.tensor(target)).item()


def train_rqvae(item_vectors: torch.Tensor, steps: int, codebook_size: int) -> RQVAE:
    model = RQVAE(item_vectors.shape[1], codebook_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        reconstruction, _, commitment = model(item_vectors)
        loss = nn.functional.mse_loss(reconstruction, item_vectors) + commitment
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def semantic_nll(model: SemanticGRU, history: np.ndarray, target: np.ndarray, codes: np.ndarray) -> float:
    return -model(torch.tensor(history), torch.tensor(codes[target])).mean().item()


@torch.no_grad()
def semantic_scores(model: SemanticGRU, history: np.ndarray, codes: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """log P(item | history) for every item, by teacher-forcing each item's full semantic ID."""
    item_codes, scores = torch.tensor(codes[1:]), []
    for start in range(0, len(history), batch_size):
        state = model.encode(torch.tensor(history[start:start + batch_size]))
        # Row r pairs history r // items with item r % items, so every history scores every item.
        pairs = model.log_likelihood(state.repeat_interleave(len(item_codes), dim=0), item_codes.repeat(len(state), 1))
        scores.append(pairs.view(len(state), -1).numpy())
    return np.vstack(scores)


def run_retrieval(data: RetrievalData, seed: int, max_epochs: int, patience: int, rqvae_steps: int) -> tuple[dict, dict]:
    (val_history, val_target), (test_history, test_target) = data.validation, data.test
    popularity = np.bincount(data.train[1], minlength=data.item_count)[1:]
    item_model = ItemGRU(data.item_count)
    item_fit = fit(item_model, data.train, lambda model, batch: nn.functional.cross_entropy(model(batch[0]), batch[1]),
                   lambda model: item_nll(model, val_history, val_target), max_epochs, patience, 3e-3, 512)

    # Quantize standardized co-occurrence vectors, then append a token that makes every semantic ID unique.
    vectors = torch.tensor(cooccurrence_vectors(*data.train, data.item_count, 32, seed), dtype=torch.float32)
    vectors = (vectors - vectors.mean(0)) / vectors.std(0).clamp_min(1e-6)
    residual_codes = train_rqvae(vectors, rqvae_steps, CODEBOOK_SIZE).semantic_ids(vectors).numpy()
    codes = np.vstack([np.zeros(3, dtype=np.int64), deduplicate(residual_codes, popularity)])  # row 0 is the padding item
    semantic_model = SemanticGRU(data.item_count, [CODEBOOK_SIZE, CODEBOOK_SIZE, int(codes[:, 2].max()) + 1])
    semantic_fit = fit(semantic_model, data.train, lambda model, batch: -model(batch[0], torch.tensor(codes[batch[1].numpy()])).mean(),
                       lambda model: semantic_nll(model, val_history, val_target, codes), max_epochs, patience, 3e-3, 512)

    metrics = {
        "popularity": ranking_metrics(np.broadcast_to(popularity, (len(test_target), len(popularity))), test_target - 1),
        "item_gru": ranking_metrics(item_scores(item_model, test_history), test_target - 1),
        "rqvae_semantic_gru": ranking_metrics(semantic_scores(semantic_model, test_history, codes), test_target - 1),
    }
    details = {"training": {"item_gru": item_fit, "rqvae_semantic_gru": semantic_fit},
               "semantic_ids": {**collision_stats(residual_codes), "deduplication_tokens": int(codes[:, 2].max()) + 1}}
    return metrics, details


@torch.no_grad()
def probability(model: DeepFM, features: np.ndarray) -> np.ndarray:
    return torch.sigmoid(model(torch.tensor(features))).numpy()


@torch.no_grad()
def esmm_probability(model: ESMM, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pctr, pcvr, pctcvr = model(torch.tensor(features))
    return pctr.numpy(), pcvr.numpy(), pctcvr.numpy()


def run_ranking(data: RankingData, max_epochs: int, patience: int) -> tuple[dict, dict]:
    features, click, _ = data.train
    val_features, val_click, val_post_click = data.validation
    deepfm = DeepFM(data.cardinalities)
    deepfm_fit = fit(deepfm, (features, click), lambda model, batch: nn.functional.binary_cross_entropy_with_logits(model(batch[0]), batch[1]),
                     lambda model: binary_metrics(val_click, probability(model, val_features))["logloss"], max_epochs, patience, 2e-3, 512)

    def esmm_loss(model: ESMM, batch: list[torch.Tensor]) -> torch.Tensor:
        pctr, _, pctcvr = model(batch[0])
        return nn.functional.binary_cross_entropy(pctr, batch[1]) + nn.functional.binary_cross_entropy(pctcvr, batch[2])

    def esmm_validation(model: ESMM) -> float:
        pctr, _, pctcvr = esmm_probability(model, val_features)
        return binary_metrics(val_click, pctr)["logloss"] + binary_metrics(val_post_click, pctcvr)["logloss"]

    esmm = ESMM(data.cardinalities)
    esmm_fit = fit(esmm, data.train, esmm_loss, esmm_validation, max_epochs, patience, 2e-3, 512)
    test_features, test_click, test_post_click = data.test
    clicked = test_click == 1
    esmm_ctr, esmm_pcvr, _ = esmm_probability(esmm, test_features)
    metrics = {
        "deepfm_ctr": binary_metrics(test_click, probability(deepfm, test_features)),
        "esmm_ctr": binary_metrics(test_click, esmm_ctr),
        "esmm_post_click_long_view_proxy": binary_metrics(test_post_click[clicked], esmm_pcvr[clicked]),
    }
    return metrics, {"deepfm_ctr": deepfm_fit, "esmm": esmm_fit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("experiments/kuairand_pure"))
    parser.add_argument("--users", type=int, default=2500, help="keep every interaction of users with an id below this")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--rqvae-steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logs = load_kuairand_logs(args.data, args.users)
    retrieval, ranking = make_retrieval_data(logs), make_ranking_data(logs)
    runs = []
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        retrieval_results, retrieval_details = run_retrieval(retrieval, seed, args.max_epochs, args.patience, args.rqvae_steps)
        ranking_results, ranking_training = run_ranking(ranking, args.max_epochs, args.patience)
        runs.append({"seed": seed, "retrieval": retrieval_results, "ranking": ranking_results, **retrieval_details, "ranking_training": ranking_training})
    results = {
        "dataset": "KuaiRand-Pure",
        "license": "CC-BY-SA-4.0",
        "users": args.users,
        "rows": len(logs),
        "seeds": args.seeds,
        "split": "chronological 70/15/15 over every standard-log interaction of the selected users",
        "retrieval_examples": {name: len(partition[1]) for name, partition in (("train", retrieval.train), ("validation", retrieval.validation), ("test", retrieval.test))},
        "config": {"max_epochs": args.max_epochs, "patience": args.patience, "rqvae_steps": args.rqvae_steps, "codebook_size": CODEBOOK_SIZE,
                   "item_vectors": "standardized PPMI-SVD co-occurrence vectors from training examples",
                   "selection": "lowest validation NLL (retrieval) or log loss (ranking) per model"},
        "summary": summarize([{"retrieval": run["retrieval"], "ranking": run["ranking"]} for run in runs]),
        "runs": runs,
        "limitations": ["KuaiRand contains no purchase/conversion label. long_view is a post-click engagement proxy and is not CVR.",
                        "Retrieval and ranking use the same dataset and time split, but this offline study does not estimate auction, bidding, or online lift."],
    }
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    retrieval_summary = results["summary"]["retrieval"]
    names = list(retrieval_summary)
    plt.figure(figsize=(7, 4))
    plt.bar(names, [retrieval_summary[name]["recall@50"]["mean"] for name in names], yerr=[retrieval_summary[name]["recall@50"]["std"] for name in names], capsize=4)
    plt.ylabel("Recall@50 (mean ± std over seeds)"); plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(args.output / "retrieval_recall_at_50.png", dpi=160); plt.close()
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
