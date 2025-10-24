"""Serve tactics using either vLLM (if available) or plain HF generate."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from libs.rh_common import get_logger

logger = get_logger("inference")


def _load_tokenizer(model_dir: Path):
    from prover.train.sft_hf import CharTokenizer

    return CharTokenizer.load(model_dir / "char_tokenizer.json")


def _generate_hf(model_dir: Path, prompts: List[str], max_new_tokens: int = 16) -> List[str]:
    import torch
    from transformers import GPT2LMHeadModel

    tokenizer = _load_tokenizer(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outputs = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_ids = input_ids.unsqueeze(0).to(device)
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        tokens = generated[0].tolist()
        outputs.append(tokenizer.decode(tokens[len(input_ids[0]) :]))
    return outputs


def _generate_vllm(model_dir: Path, prompts: List[str], max_new_tokens: int = 16) -> List[str]:
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception:  # pragma: no cover
        logger.warning("vLLM not available; falling back to HF generate.")
        return _generate_hf(model_dir, prompts, max_new_tokens)

    llm = LLM(model=str(model_dir))
    params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outputs = llm.generate(prompts, params)
    return [out.outputs[0].text for out in outputs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/checkpoints/sft_hf_tiny")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompts", nargs="+", default=["⊢ True"])
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if args.backend == "vllm":
        outputs = _generate_vllm(model_dir, args.prompts, args.max_new_tokens)
    else:
        outputs = _generate_hf(model_dir, args.prompts, args.max_new_tokens)
    for prompt, output in zip(args.prompts, outputs):
        print(f"Prompt: {prompt}\nTactic: {output}\n")


if __name__ == "__main__":
    main()
