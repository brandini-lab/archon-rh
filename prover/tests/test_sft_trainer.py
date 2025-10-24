from pathlib import Path

import yaml

from prover.train import sft_train


def test_sft_tiny(tmp_path):
    config = {
        "dataset": {
            "source": "prover/tests/data/sample_tactics.jsonl",
            "output_dir": str(tmp_path / "dataset"),
            "limit": 2,
        },
        "output_dir": str(tmp_path / "checkpoints"),
        "seq_len": 64,
        "batch_size": 2,
        "lr": 1e-3,
        "seed": 7,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf8")
    cfg = sft_train.load_config(str(cfg_path))
    sft_train.train(cfg, max_steps=2)
    assert (Path(cfg.output_dir) / "final.pt").exists()
