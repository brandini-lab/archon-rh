from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from safety.policies.reward_verifier import RewardEvent


def sign_events(queue_path: str, output_path: str, signer: str = "archon") -> None:
    queue = Path(queue_path)
    if not queue.exists():
        raise FileNotFoundError(f"No pending rewards at {queue}")

    allowlist = json.loads(Path("safety/policies/allowlist.json").read_text(encoding="utf8"))
    key = allowlist[signer]["public_key"]

    events: Iterable[RewardEvent] = []
    lines = queue.read_text(encoding="utf8").strip().splitlines()
    prev_hash = None
    with Path(output_path).open("a", encoding="utf8") as writer:
        for line in lines:
            payload = json.loads(line)
            digest = hashlib.sha256()
            digest.update(json.dumps(payload, sort_keys=True).encode("utf8"))
            digest.update(key.encode("utf8"))
            signature = digest.hexdigest()
            entry = {
                "payload": payload,
                "signer": signer,
                "signature": signature,
                "prev_hash": prev_hash,
            }
            writer.write(json.dumps(entry) + "\n")
            prev_hash = hashlib.sha256((signature + (prev_hash or "GENESIS")).encode("utf8")).hexdigest()
    queue.unlink()
