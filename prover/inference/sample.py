from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from prover.train.modeling import SimpleTokenizer, TinyTransformer


def load_model(checkpoint: Path, vocab: str) -> TinyTransformer:
    tokenizer = SimpleTokenizer(vocab)
    model = TinyTransformer(tokenizer.vocab_size)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def sample(model: TinyTransformer, tokenizer: SimpleTokenizer, prompt: str, max_tokens: int = 32) -> str:
    context = tokenizer.encode(prompt)
    if not context:
        context = [0]
    tokens = torch.tensor([context], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(tokens)
            next_token = logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            tokens = torch.cat([tokens, next_token], dim=1)
    decoded = tokenizer.decode(tokens[0].tolist())
    return decoded[len(prompt) :]


def generate_tactics(checkpoint: str, prompts: List[str], vocab: str) -> List[str]:
    tokenizer = SimpleTokenizer(vocab)
    model = load_model(Path(checkpoint), vocab)
    outputs = []
    for prompt in prompts:
        outputs.append(sample(model, tokenizer, prompt))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample tactics from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True, help="JSON file with list of prompts.")
    parser.add_argument("--vocab", default="".join(chr(i) for i in range(32, 127)))
    parser.add_argument("--output", default="artifacts/inference/samples.json")
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf8"))
    outputs = generate_tactics(args.checkpoint, prompts, args.vocab)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(outputs, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
