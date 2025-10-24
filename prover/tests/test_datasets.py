from pathlib import Path

from pathlib import Path

from prover.datasets import DatasetConfig, build_dataset
from prover.datasets import build_mathlib_dataset


def test_dataset_builder(tmp_path):
    source = Path("prover/tests/data/sample_tactics.jsonl")
    artifacts = build_dataset(
        DatasetConfig(source=str(source), output_dir=str(tmp_path), limit=2)
    )
    assert artifacts.jsonl.exists()
    assert artifacts.parquet.exists()


def test_mathlib_builder_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shard = build_mathlib_dataset.build(limit=2)
    assert shard.examples
    assert Path("artifacts/data/mini_mathlib.jsonl").exists()
