from prover.rl.run_selfplay import load_config, run_selfplay


def test_selfplay_runs(tmp_path):
    cfg = load_config("orchestration/configs/rl_tiny.yaml")
    cfg.log_dir = str(tmp_path / "rl")
    run_selfplay(cfg, walltime=1)
    assert (tmp_path / "rl" / "summary.json").exists()
