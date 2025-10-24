from pathlib import Path

from prover.datasets import DatasetConfig, build_dataset


def test_dataset_builder(tmp_path):
    source = Path("prover/tests/data/sample_tactics.jsonl")
    artifacts = build_dataset(
        DatasetConfig(source=str(source), output_dir=str(tmp_path), limit=2)
    )
    assert artifacts.jsonl.exists()
    assert artifacts.parquet.exists()
