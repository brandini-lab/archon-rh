from prover.metrics import length_normalized_score, pass_at_k, proof_accept_rate


def test_pass_at_k():
    assert 0.0 <= pass_at_k(1, 10, 5) <= 1.0


def test_proof_accept_rate():
    assert proof_accept_rate([True, False, True]) == 2 / 3


def test_length_normalized_score():
    score = length_normalized_score([10, 20, 30], [1.0, 0.5, 0.25])
    assert score > 0.0
