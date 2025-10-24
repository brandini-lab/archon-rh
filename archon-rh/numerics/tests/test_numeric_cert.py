from numerics.zeros.run_zero_checks import load_config, run_zero_checks


def test_numeric_cert(tmp_path):
    cfg = load_config("numerics/tests/data/numeric_tiny.yaml")
    cfg.output = str(tmp_path / "certs.json")
    certificates = run_zero_checks(cfg)
    assert certificates
    assert certificates[0]["valid"]
