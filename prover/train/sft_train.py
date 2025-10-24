from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import yaml

from prover.datasets import DatasetConfig, build_dataset
from prover.train.modeling import SimpleTokenizer, TinyTransformer


@dataclass
class SFTConfig:
    dataset_jsonl: str
    output_dir: str = "artifacts/checkpoints/sft"
    seq_len: int = 128
    batch_size: int = 4
    lr: float = 1e-3
    seed: int = 42
    vocab: str = "".join([chr(i) for i in range(32, 127)])


class LeanTacticDataset(Dataset):
    def __init__(self, path: Path, tokenizer: SimpleTokenizer, seq_len: int):
        self.rows: List[str] = []
        with path.open("r", encoding="utf8") as handle:
            for line in handle:
                payload = json.loads(line)
                prompt = payload["goal"] + "\n<SEP>\n" + payload["tactic"]
                self.rows.append(prompt)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        tokens = self.tokenizer.encode(self.rows[idx])
        tokens = tokens[: self.seq_len]
        if len(tokens) < self.seq_len:
            tokens += [0] * (self.seq_len - len(tokens))
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int = 128, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Embedding(512, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        embeddings = self.token_emb(tokens) + self.pos_emb(positions)
        hidden = self.transformer(embeddings)
        return self.lm_head(hidden)


def load_config(path: str) -> SFTConfig:
    with open(path, "r", encoding="utf8") as handle:
        payload = yaml.safe_load(handle)
    dataset_cfg = payload.get("dataset", {})
    dataset_path = Path(dataset_cfg.get("source", "artifacts/datasets/lean_tactics.jsonl"))
    artifacts = build_dataset(
        DatasetConfig(
            source=str(dataset_path),
            output_dir=dataset_cfg.get("output_dir", "artifacts/datasets"),
            limit=dataset_cfg.get("limit"),
        )
    )
    return SFTConfig(
        dataset_jsonl=str(artifacts.jsonl),
        output_dir=payload.get("output_dir", "artifacts/checkpoints/sft"),
        seq_len=payload.get("seq_len", 128),
        batch_size=payload.get("batch_size", 4),
        lr=payload.get("lr", 1e-3),
        seed=payload.get("seed", 42),
        vocab=payload.get("vocab", "".join(chr(i) for i in range(32, 127))),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: SFTConfig, max_steps: int) -> None:
    set_seed(cfg.seed)
    tokenizer = SimpleTokenizer(cfg.vocab)
    dataset = LeanTacticDataset(Path(cfg.dataset_jsonl), tokenizer, cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformer(tokenizer.vocab_size)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    for epoch in range(1000):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                ppl = math.exp(loss.item())
                print(f"[step={step}] loss={loss.item():.4f} ppl={ppl:.2f}")

            if step and step % 50 == 0:
                ckpt_path = output_dir / f"model_step{step}.pt"
                torch.save({"model": model.state_dict(), "step": step}, ckpt_path)

            step += 1
            if step >= max_steps:
                torch.save({"model": model.state_dict(), "step": step}, output_dir / "final.pt")
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny SFT trainer for Lean tactics")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
