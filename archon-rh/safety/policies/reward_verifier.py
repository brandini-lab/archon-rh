from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


def _load_allowlist(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Allowlist missing: {path}")
    return json.loads(path.read_text(encoding="utf8"))


@dataclass
class RewardEvent:
    payload: Dict[str, float]
    signer: str
    signature: str
    prev_hash: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "RewardEvent":
        return cls(
            payload=data["payload"],
            signer=data["signer"],
            signature=data["signature"],
            prev_hash=data.get("prev_hash"),
        )


class RewardVerifier:
    def __init__(self, allowlist_path: str = "safety/policies/allowlist.json"):
        self.allowlist = _load_allowlist(Path(allowlist_path))

    def _expected_signature(self, payload: Dict[str, float], signer: str) -> str:
        key = self.allowlist[signer]["public_key"]
        digest = hashlib.sha256()
        digest.update(json.dumps(payload, sort_keys=True).encode("utf8"))
        digest.update(key.encode("utf8"))
        return digest.hexdigest()

    def verify(self, events: Iterable[RewardEvent]) -> bool:
        prev_chain: Optional[str] = None
        for event in events:
            if event.signer not in self.allowlist:
                raise PermissionError(f"Signer {event.signer} not in allowlist.")
            expected = self._expected_signature(event.payload, event.signer)
            if expected != event.signature:
                raise ValueError("Signature mismatch.")
            chain_hash = hashlib.sha256(
                (event.signature + (prev_chain or "GENESIS")).encode("utf8")
            ).hexdigest()
            if event.prev_hash and event.prev_hash != prev_chain:
                raise ValueError("Broken hash chain.")
            prev_chain = chain_hash
        return True


def verify_file(path: str) -> bool:
    events = []
    with open(path, "r", encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                events.append(RewardEvent.from_dict(json.loads(line)))
    verifier = RewardVerifier()
    return verifier.verify(events)
