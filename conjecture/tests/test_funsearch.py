from pathlib import Path

import yaml

from conjecture.funsearch.funsearch_loop import load_config, run_loop


def test_funsearch_objective(tmp_path):
    cfg_data = {
        "iterations": 12,
        "objectives": ["li_criterion"],
        "export_dir": str(tmp_path / "artifacts"),
        "lean_export": str(tmp_path / "Generated.lean"),
        "seed": 0,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf8")
    cfg = load_config(str(cfg_path))
    summary = run_loop(cfg)
    assert summary["li_criterion"] > -len([0.1, 0.5, 0.9, 1.3])
    assert Path(cfg.lean_export).exists()
