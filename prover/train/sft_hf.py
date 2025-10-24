"""Supervised fine-tuning using a minimal HuggingFace GPT-2 stack."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import yaml
from transformers import GPT2Config, GPT2LMHeadModel

from libs.rh_common import get_logger, ensure_dir

logger = get_logger("sft_hf")


@dataclass
class HFConfig:
    seed: int
    model: dict
    trainer: dict


class CharTokenizer:
    """Character-level tokenizer with JSON serialization."""

    def __init__(self, texts: List[str]):
        charset = sorted({ch for text in texts for ch in text})
        self.pad_token_id = 0
        self.eos_token_id = 1
        vocab = ["<pad>", "<eos>"] + charset
        self.stoi = {ch: idx for idx, ch in enumerate(vocab)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_length: int = 64) -> torch.Tensor:
        tokens = [self.stoi.get(ch, self.eos_token_id) for ch in text][: max_length - 1]
        tokens.append(self.eos_token_id)
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: List[int]) -> str:
        chars = []
        for token in tokens:
            if token == self.eos_token_id:
                break
            chars.append(self.itos.get(token, "?"))
        return "".join(chars)

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        path.write_text(json.dumps({"stoi": self.stoi, "pad": self.pad_token_id, "eos": self.eos_token_id}))

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        payload = json.loads(path.read_text())
        tokenizer = cls(texts=[""])
        tokenizer.stoi = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in payload["stoi"].items()}
        tokenizer.itos = {idx: ch for ch, idx in tokenizer.stoi.items()}
        tokenizer.pad_token_id = payload["pad"]
        tokenizer.eos_token_id = payload["eos"]
        return tokenizer


def load_config(path: str) -> HFConfig:
    payload = yaml.safe_load(Path(path).read_text())
    return HFConfig(
        seed=payload["seed"],
        model=payload["model"],
        trainer=payload["trainer"],
    )


def load_dataset(dataset_path: Path) -> List[str]:
    if not dataset_path.exists():
        logger.warning("Dataset %s missing; falling back to sample_data.", dataset_path)
        dataset_path = Path("prover/datasets/sample_data/mini_mathlib.jsonl")
    texts: List[str] = []
    for line in dataset_path.read_text().splitlines():
        item = json.loads(line)
        texts.append(item["goal"] + "\n" + item["tactic"])
    return texts


def train(cfg: HFConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    texts = load_dataset(Path(cfg.trainer["dataset"]))
    tokenizer = CharTokenizer(texts)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_layer=cfg.model.get("n_layer", 2),
        n_head=cfg.model.get("n_head", 2),
        n_embd=cfg.model.get("n_embd", 64),
        n_positions=128,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)

    output_tokens = 64
    steps = cfg.trainer["steps"]
    batch_size = cfg.trainer["batch_size"]
    lr = cfg.trainer["learning_rate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for step in range(steps):
        batch = random.sample(texts, k=min(batch_size, len(texts)))
        inputs = torch.stack([tokenizer.encode(text, max_length=output_tokens) for text in batch]).to(device)
        labels = inputs.clone()
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            logger.info("step=%d loss=%.4f", step, loss.item())

    output_dir = Path(cfg.trainer["output_dir"])
    ensure_dir(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save(output_dir / "char_tokenizer.json")
    logger.info("Saved HF model to %s", output_dir)


def main(cfg_path: str = "orchestration/configs/sft_hf_tiny.yaml") -> None:
    cfg = load_config(cfg_path)
    train(cfg)


if __name__ == "__main__":
    main()
