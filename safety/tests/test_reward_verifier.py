import json
from pathlib import Path

from safety.audits.sign_rewards import sign_events
from safety.policies.reward_verifier import RewardEvent, RewardVerifier


def test_reward_verifier(tmp_path):
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps({"reward": 1.0}), encoding="utf8")
    log_path = tmp_path / "log.jsonl"
    sign_events(str(queue), str(log_path))
    events = [
        RewardEvent.from_dict(json.loads(line))
        for line in log_path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    verifier = RewardVerifier()
    assert verifier.verify(events)
