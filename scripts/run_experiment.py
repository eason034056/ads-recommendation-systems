from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adsrec.datasets import load_kuairand_logs, make_ranking_data, make_retrieval_data
from adsrec.evaluation import binary_metrics, ranking_metrics
from adsrec.models import DeepFM, ESMM, ItemGRU, RQVAE, SemanticGRU


def batches(*arrays: np.ndarray, batch_size: int = 512, shuffle: bool = False) -> DataLoader:
    values = [torch.tensor(array) for array in arrays]
    return DataLoader(TensorDataset(*values), batch_size=batch_size, shuffle=shuffle)


def train_item_gru(model: ItemGRU, train: tuple[np.ndarray, np.ndarray], epochs: int) -> ItemGRU:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        for history, target in batches(*train, shuffle=True):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(history), target)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def item_scores(model: ItemGRU, history: np.ndarray) -> np.ndarray:
    return model(torch.tensor(history)).numpy()[:, 1:]


def train_rqvae(item_vectors: torch.Tensor, epochs: int, codebook_size: int) -> RQVAE:
    model = RQVAE(item_vectors.shape[1], codebook_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        reconstruction, _, commitment = model(item_vectors)
        loss = nn.functional.mse_loss(reconstruction, item_vectors) + commitment
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def train_semantic(model: SemanticGRU, train: tuple[np.ndarray, np.ndarray], codes: np.ndarray, epochs: int) -> SemanticGRU:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        for history, target in batches(*train, shuffle=True):
            target_codes = torch.tensor(codes[target.numpy()])
            first, second = model(history, target_codes[:, 0])
            loss = nn.functional.cross_entropy(first, target_codes[:, 0]) + nn.functional.cross_entropy(second, target_codes[:, 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def semantic_scores(model: SemanticGRU, history: np.ndarray, codes: np.ndarray) -> np.ndarray:
    history_t = torch.tensor(history)
    encoded, _ = model.encoder(model.item_embedding(history_t))
    state = encoded[:, -1]
    first = torch.log_softmax(model.first(state), dim=1)
    all_codes = torch.arange(first.shape[1])
    code_vectors = model.code_embedding(all_codes)
    repeated_state = state[:, None, :].expand(-1, len(all_codes), -1)
    second = torch.log_softmax(model.second(torch.cat([repeated_state, code_vectors[None].expand(len(state), -1, -1)], dim=2)), dim=2)
    item_codes = torch.tensor(codes[1:])
    batch = torch.arange(len(history))[:, None]
    first_scores = first[batch, item_codes[:, 0][None, :]]
    second_scores = second[batch, item_codes[:, 0][None, :], item_codes[:, 1][None, :]]
    return (first_scores + second_scores).numpy()


def train_ranking(model: nn.Module, train: tuple[np.ndarray, np.ndarray, np.ndarray], esmm: bool, epochs: int) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        for features, click, post_click in batches(*train, shuffle=True):
            optimizer.zero_grad()
            if esmm:
                ctr, _, ctcvr = model(features)
                loss = nn.functional.binary_cross_entropy(ctr, click) + nn.functional.binary_cross_entropy(ctcvr, post_click)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(model(features), click)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def ranking_predictions(model: nn.Module, features: np.ndarray, esmm: bool) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.tensor(features)
    if esmm:
        ctr, pcvr, _ = model(tensor)
        return ctr.numpy(), pcvr.numpy()
    return torch.sigmoid(model(tensor)).numpy(), np.zeros(len(features))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("experiments/kuairand_pure"))
    parser.add_argument("--max-rows", type=int, default=120_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    logs = load_kuairand_logs(args.data, args.max_rows)
    retrieval = make_retrieval_data(logs)
    popularity = np.bincount(retrieval.train[1], minlength=retrieval.item_count)[1:]
    popularity_scores = np.broadcast_to(popularity, (len(retrieval.test[1]), len(popularity)))
    item_model = train_item_gru(ItemGRU(retrieval.item_count), retrieval.train, args.epochs)
    vectors = item_model.embedding.weight.detach()[1:]
    rqvae = train_rqvae(vectors, args.epochs * 4, 32)
    codes = np.vstack([np.zeros(2, dtype=np.int64), rqvae.semantic_ids(vectors).numpy()])
    semantic_model = train_semantic(SemanticGRU(retrieval.item_count, 32), retrieval.train, codes, args.epochs)
    retrieval_results = {
        "popularity": ranking_metrics(popularity_scores, retrieval.test[1] - 1),
        "item_gru": ranking_metrics(item_scores(item_model, retrieval.test[0]), retrieval.test[1] - 1),
        "rqvae_semantic_gru": ranking_metrics(semantic_scores(semantic_model, retrieval.test[0], codes), retrieval.test[1] - 1),
    }
    rank = make_ranking_data(logs)
    deepfm = train_ranking(DeepFM(rank.cardinalities), rank.train, False, args.epochs)
    esmm = train_ranking(ESMM(rank.cardinalities), rank.train, True, args.epochs)
    deepfm_ctr, _ = ranking_predictions(deepfm, rank.test[0], False)
    esmm_ctr, esmm_pcvr = ranking_predictions(esmm, rank.test[0], True)
    click_mask = rank.test[1] == 1
    ranking_results = {
        "deepfm_ctr": binary_metrics(rank.test[1], deepfm_ctr),
        "esmm_ctr": binary_metrics(rank.test[1], esmm_ctr),
        "esmm_post_click_long_view_proxy": binary_metrics(rank.test[2][click_mask], esmm_pcvr[click_mask]),
    }
    results = {"dataset": "KuaiRand-Pure", "license": "CC-BY-SA-4.0", "seed": args.seed, "max_rows": args.max_rows, "retrieval": retrieval_results, "ranking": ranking_results, "limitations": ["KuaiRand contains no purchase/conversion label. long_view is a post-click engagement proxy and is not CVR.", "Retrieval and ranking use the same dataset and time split, but this offline study does not estimate auction, bidding, or online lift."]}
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    names = list(retrieval_results)
    plt.figure(figsize=(7, 4)); plt.bar(names, [retrieval_results[name]["recall@50"] for name in names]); plt.ylabel("Recall@50"); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(args.output / "retrieval_recall_at_50.png", dpi=160); plt.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
