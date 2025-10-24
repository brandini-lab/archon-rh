from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

from formal.leandojo_bridge import LeanBridge, LeanBridgeConfig


@dataclass
class PPOConfig:
    theorem_path: str
    tactics: List[str]
    log_dir: str = "artifacts/rl"
    seed: int = 1234
    temperature: float = 1.0
    lr: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5


@dataclass
class PolicyState:
    params: Dict[str, float] = field(default_factory=dict)
    values: Dict[str, float] = field(default_factory=dict)

    def normalise(self) -> Dict[str, float]:
        max_logit = max(self.params.values()) if self.params else 0.0
        exp_scores = {k: math.exp(v - max_logit) for k, v in self.params.items()}
        total = sum(exp_scores.values())
        return {k: s / max(total, 1e-6) for k, s in exp_scores.items()}

    def sample(self) -> str:
        probs = self.normalise()
        r = random.random()
        cumulative = 0.0
        for tactic, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return tactic
        return next(iter(probs))  # fallback

    def update(self, tactic: str, reward: float, lr: float, entropy_coef: float) -> None:
        probs = self.normalise()
        baseline = probs[tactic]
        advantage = reward - self.values.get(tactic, 0.0)
        grad = advantage * (1 - baseline)
        self.params[tactic] += lr * grad
        self.values[tactic] = self.values.get(tactic, 0.0) + lr * advantage * 0.5
        # entropy bonus
        for key, prob in probs.items():
            self.params[key] += entropy_coef * (-prob * math.log(max(prob, 1e-6)))


def load_config(path: str) -> PPOConfig:
    with open(path, "r", encoding="utf8") as handle:
        payload = yaml.safe_load(handle)
    return PPOConfig(
        theorem_path=payload["theorem_path"],
        tactics=payload.get("tactics", ["simp", "rfl", "omega"]),
        log_dir=payload.get("log_dir", "artifacts/rl"),
        seed=payload.get("seed", 1234),
        temperature=payload.get("temperature", 1.0),
        lr=payload.get("lr", 0.2),
        entropy_coef=payload.get("entropy_coef", 0.01),
        value_coef=payload.get("value_coef", 0.5),
    )


def run_selfplay(cfg: PPOConfig, walltime: int) -> None:
    random.seed(cfg.seed)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    traces_path = Path(cfg.log_dir) / "proof_traces.jsonl"
    policy_path = Path(cfg.log_dir) / "policy_snapshot.json"
    state = PolicyState(params={t: 0.0 for t in cfg.tactics})
    accepted = 0
    total = 0

    start = time.time()
    while time.time() - start < walltime:
        bridge_cfg = LeanBridgeConfig(
            theorem_path=cfg.theorem_path,
            log_dir=cfg.log_dir,
            allow_toy_fallback=True,
        )
        episode_trace: List[Dict[str, str]] = []
        with LeanBridge(bridge_cfg) as bridge:
            solved = False
            for _ in range(8):
                tactic = state.sample()
                next_state = bridge.apply_tactic(tactic)
                reward = 1.0 if next_state.solved else -0.01
                state.update(tactic, reward, cfg.lr, cfg.entropy_coef)
                episode_trace.append(
                    {
                        "tactic": tactic,
                        "reward": reward,
                        "goal": next_state.goal_text,
                        "solved": next_state.solved,
                    }
                )
                total += 1
                if next_state.solved:
                    solved = True
                    accepted += 1
                    break
            with traces_path.open("a", encoding="utf8") as writer:
                writer.write(json.dumps({"trace": episode_trace, "solved": solved}) + "\n")
        Path(policy_path).write_text(json.dumps(state.params, indent=2), encoding="utf8")

    summary = {
        "accepted": accepted,
        "total": total,
        "accept_rate": accepted / total if total else 0.0,
        "walltime": walltime,
    }
    Path(cfg.log_dir, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(f"Self-play completed. Accepted steps: {accepted}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO self-play over Lean tactics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--walltime", type=int, default=60)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_selfplay(cfg, walltime=args.walltime)


if __name__ == "__main__":
    main()
