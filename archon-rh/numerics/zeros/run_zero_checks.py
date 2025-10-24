from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml
from mpmath import zeta

from numerics.arb_bindings import interval, verify_certified_zero_count


@dataclass
class IntervalConfig:
    lo: float
    hi: float
    samples: int = 64


@dataclass
class ZeroConfig:
    intervals: List[IntervalConfig]
    output: str = "artifacts/numerics/certificates.json"


def load_config(path: str) -> ZeroConfig:
    with open(path, "r", encoding="utf8") as handle:
        payload = yaml.safe_load(handle)
    intervals = [
        IntervalConfig(**item)
        for item in payload.get("intervals", [{"lo": 0.0, "hi": 10.0, "samples": 64}])
    ]
    return ZeroConfig(intervals=intervals, output=payload.get("output", "artifacts/numerics/certificates.json"))


def count_sign_changes(lo: float, hi: float, samples: int) -> int:
    grid = [lo + i * (hi - lo) / samples for i in range(samples + 1)]
    values = [zeta(0.5 + 1j * t).real for t in grid]
    sign_changes = 0
    prev_sign = 0
    for value in values:
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if prev_sign != 0 and sign != 0 and sign != prev_sign:
            sign_changes += 1
        if sign != 0:
            prev_sign = sign
    return sign_changes


def run_zero_checks(cfg: ZeroConfig) -> List[dict]:
    certificates: List[dict] = []
    for item in cfg.intervals:
        count = count_sign_changes(item.lo, item.hi, item.samples)
        target = interval(item.lo, item.hi)
        valid, message = verify_certified_zero_count(target, count)
        certificates.append(
            {
                "interval": [item.lo, item.hi],
                "samples": item.samples,
                "count": count,
                "valid": valid,
                "message": message,
            }
        )
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(certificates, indent=2), encoding="utf8")
    return certificates


def main() -> None:
    parser = argparse.ArgumentParser(description="Rigorous zero counting via crude sampling.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    certs = run_zero_checks(cfg)
    print("Generated certificates:", certs)


if __name__ == "__main__":
    main()
