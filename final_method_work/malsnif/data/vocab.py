from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PAD = "<pad>"
UNK = "<unk>"


@dataclass
class Vocabulary:
    token_to_idx: dict[str, int]
    idx_to_token: list[str]
    embeddings: np.ndarray

    @classmethod
    def build(cls, corpus: Iterable[Sequence[str]], min_freq: int = 1, dim: int = 64, seed: int = 42) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for sent in corpus:
            counter.update(sent)
        tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
        tokens = sorted(tokens)
        idx_to_token = [PAD, UNK] + tokens
        token_to_idx = {tok: i for i, tok in enumerate(idx_to_token)}
        rng = np.random.default_rng(seed)
        emb = rng.normal(0.0, 0.05, size=(len(idx_to_token), dim)).astype("float32")
        emb[0] = 0.0
        return cls(token_to_idx, idx_to_token, emb)

    def encode(self, tokens: Sequence[str], max_len: int | None = None) -> list[int]:
        ids = [self.token_to_idx.get(t, 1) for t in tokens]
        if max_len is not None:
            ids = ids[:max_len]
        return ids or [1]


class SkipGramPairs(Dataset):
    def __init__(
        self,
        encoded_sentences: list[list[int]],
        window: int = 2,
        negative: int = 5,
        vocab_size: int = 0,
        max_pairs: int | None = None,
        show_progress: bool = True,
    ):
        self.pairs: list[tuple[int, int]] = []
        pbar = tqdm(encoded_sentences, desc="word2vec sample pairs", unit="sent", dynamic_ncols=True, disable=not show_progress)
        stop = False
        for sent in pbar:
            for i, center in enumerate(sent):
                lo = max(0, i - window)
                hi = min(len(sent), i + window + 1)
                for j in range(lo, hi):
                    if i != j:
                        self.pairs.append((center, sent[j]))
                        if max_pairs is not None and len(self.pairs) >= int(max_pairs):
                            stop = True
                            break
                if stop:
                    break
            if stop:
                break
            if len(self.pairs) and len(self.pairs) % 200000 == 0:
                pbar.set_postfix(pairs=len(self.pairs), refresh=False)
        self.negative = negative
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        c, o = self.pairs[idx]
        neg = torch.randint(2, max(self.vocab_size, 3), (self.negative,), dtype=torch.long)
        return torch.tensor(c, dtype=torch.long), torch.tensor(o, dtype=torch.long), neg


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, initial: np.ndarray | None = None):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.out_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        if initial is not None:
            self.in_embed.weight.data.copy_(torch.from_numpy(initial))
            self.out_embed.weight.data.copy_(torch.from_numpy(initial))

    def forward(self, center, pos, neg):
        c = self.in_embed(center)  # [B,D]
        p = self.out_embed(pos)   # [B,D]
        pos_loss = F.logsigmoid((c * p).sum(-1)).neg()
        n = self.out_embed(neg)   # [B,K,D]
        neg_loss = F.logsigmoid(-(n * c.unsqueeze(1)).sum(-1)).neg().sum(-1)
        return (pos_loss + neg_loss).mean()


def train_skipgram(
    vocab: Vocabulary,
    corpus: list[list[str]],
    epochs: int = 1,
    window: int = 2,
    negative: int = 5,
    lr: float = 0.01,
    batch_size: int = 2048,
    seed: int = 42,
    max_sentences: int | None = None,
    max_pairs: int | None = None,
    show_progress: bool = True,
) -> Vocabulary:
    if epochs <= 0:
        return vocab
    torch.manual_seed(seed)
    if max_sentences is not None and int(max_sentences) > 0 and len(corpus) > int(max_sentences):
        # Deterministic chronological budget: keeps the method (Skip-Gram) but
        # avoids exploding pair materialization on tens of millions of events.
        corpus = corpus[: int(max_sentences)]
    encoded = [vocab.encode(sent) for sent in tqdm(corpus, desc="word2vec encode sentences", unit="sent", dynamic_ncols=True, disable=not show_progress) if len(sent) > 1]
    ds = SkipGramPairs(
        encoded, window=window, negative=negative, vocab_size=len(vocab.idx_to_token),
        max_pairs=max_pairs, show_progress=show_progress,
    )
    if len(ds) == 0:
        return vocab
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = SkipGramModel(len(vocab.idx_to_token), vocab.embeddings.shape[1], vocab.embeddings)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for center, pos, neg in tqdm(loader, desc=f"word2vec epoch {ep+1}/{epochs}", unit="batch", dynamic_ncols=True, disable=not show_progress):
            opt.zero_grad(set_to_none=True)
            loss = model(center, pos, neg)
            loss.backward()
            opt.step()
    vocab.embeddings = model.in_embed.weight.detach().cpu().numpy().astype("float32")
    vocab.embeddings[0] = 0.0
    return vocab
