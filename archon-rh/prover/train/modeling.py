from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn


class SimpleTokenizer:
    def __init__(self, vocab: str):
        self.vocab = vocab
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(vocab)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(char, 0) for char in text]

    def decode(self, tokens: Iterable[int]) -> str:
        return "".join(self.itos.get(tok, "?") for tok in tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int = 128, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Embedding(512, emb_size)
        encoder = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=nlayers)
        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        embeddings = self.token_emb(tokens) + self.pos_emb(positions)
        hidden = self.transformer(embeddings)
        return self.lm_head(hidden)
