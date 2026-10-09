from __future__ import annotations

import torch
from torch import nn


class ItemGRU(nn.Module):
    def __init__(self, item_count: int, dimension: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(item_count, dimension, padding_idx=0)
        self.encoder = nn.GRU(dimension, dimension, batch_first=True)
        self.output = nn.Linear(dimension, item_count)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(self.embedding(history))
        return self.output(encoded[:, -1])


class RQVAE(nn.Module):
    """Two-stage residual vector quantizer used to assign semantic item IDs."""
    def __init__(self, dimension: int, codebook_size: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, dimension))
        self.codebooks = nn.Parameter(torch.randn(2, codebook_size, dimension) * .1)
        self.decoder = nn.Linear(dimension, dimension)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(values)
        residual = latent
        codes, quantized = [], torch.zeros_like(latent)
        for codebook in self.codebooks:
            distances = ((residual[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
            code = distances.argmin(-1)
            selected = codebook[code]
            codes.append(code)
            quantized = quantized + selected
            residual = residual - selected
        # Straight-through estimation lets the reconstruction objective train the encoder;
        # the codebook and commitment losses keep residual codes close to its latent space.
        quantized_st = latent + (quantized - latent).detach()
        reconstruction = self.decoder(quantized_st)
        codebook_loss = ((latent.detach() - quantized) ** 2).mean()
        commitment = ((latent - quantized.detach()) ** 2).mean()
        return reconstruction, torch.stack(codes, dim=1), codebook_loss + .25 * commitment

    @torch.no_grad()
    def semantic_ids(self, values: torch.Tensor) -> torch.Tensor:
        _, codes, _ = self(values)
        return codes


class SemanticGRU(nn.Module):
    """Encodes the item history, then decodes an item's semantic-ID tokens one at a time."""
    def __init__(self, item_count: int, vocabulary_sizes: list[int], dimension: int = 32):
        super().__init__()
        self.item_embedding = nn.Embedding(item_count, dimension, padding_idx=0)
        self.encoder = nn.GRU(dimension, dimension, batch_first=True)
        self.token_embeddings = nn.ModuleList([nn.Embedding(size, dimension) for size in vocabulary_sizes[:-1]])
        # A recurrent decoder lets each token's distribution depend jointly on the user state and the prefix.
        # A linear layer over [state, prefix] only adds the two, so a user's preference among second-level
        # codes would be the same under every first-level code.
        self.decoder = nn.GRUCell(dimension, dimension)
        self.heads = nn.ModuleList([nn.Linear(dimension, size) for size in vocabulary_sizes])

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.item_embedding(history))[0][:, -1]

    def log_likelihood(self, state: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced log P(codes | state), summed over tokens; row i of `state` decodes row i of `codes`."""
        total = torch.zeros(len(codes))
        for level, head in enumerate(self.heads):
            total = total + torch.log_softmax(head(state), dim=1).gather(1, codes[:, level:level + 1]).squeeze(1)
            if level < len(self.token_embeddings):
                state = self.decoder(self.token_embeddings[level](codes[:, level]), state)
        return total

    def forward(self, history: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood(self.encode(history), codes)


class FieldEmbedding(nn.Module):
    """One vector table per categorical field; row 0 is the field's out-of-vocabulary bucket."""
    def __init__(self, cardinalities: list[int], dimension: int):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(size, dimension) for size in cardinalities])
        for table in self.tables:
            # nn.Embedding defaults to N(0, 1). Summed over every field pair, the FM term then starts
            # near +/-30 logits and saturates the sigmoid, and rows never seen in training keep that noise.
            nn.init.normal_(table.weight, std=.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([table(features[:, i]) for i, table in enumerate(self.tables)], dim=1)


class DeepFMHead(nn.Module):
    """First-order, factorization-machine and MLP terms computed from (possibly shared) field vectors."""
    def __init__(self, cardinalities: list[int], dimension: int):
        super().__init__()
        self.first_order = nn.ModuleList([nn.Embedding(size, 1) for size in cardinalities])
        for table in self.first_order:
            nn.init.zeros_(table.weight)
        # Per-dimension interaction weights let two heads read different pairwise signals from shared vectors.
        self.interaction = nn.Parameter(torch.ones(dimension))
        self.deep = nn.Sequential(nn.Linear(len(cardinalities) * dimension, 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        linear = sum(table(features[:, i]) for i, table in enumerate(self.first_order)).squeeze(1) + self.bias
        fm = .5 * ((vectors.sum(1) ** 2 - (vectors ** 2).sum(1)) * self.interaction).sum(1)
        return linear + fm + self.deep(vectors.flatten(1)).squeeze(1)


class DeepFM(nn.Module):
    def __init__(self, cardinalities: list[int], dimension: int = 16):
        super().__init__()
        self.embedding = FieldEmbedding(cardinalities, dimension)
        self.head = DeepFMHead(cardinalities, dimension)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features, self.embedding(features))


class ESMM(nn.Module):
    """Entire-space multi-task model: pCTCVR = pCTR * pCVR, with both towers reading shared field vectors."""
    def __init__(self, cardinalities: list[int], dimension: int = 16):
        super().__init__()
        # Sharing the vectors is what lets abundant click labels shape the representation of sparse conversions.
        self.embedding = FieldEmbedding(cardinalities, dimension)
        self.ctr = DeepFMHead(cardinalities, dimension)
        self.cvr = DeepFMHead(cardinalities, dimension)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vectors = self.embedding(features)
        pctr = torch.sigmoid(self.ctr(features, vectors))
        pcvr = torch.sigmoid(self.cvr(features, vectors))
        return pctr, pcvr, pctr * pcvr
