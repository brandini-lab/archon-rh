"""Minimal PPO loop over Lean tactics."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from libs.rh_common import ensure_dir, get_logger
from .env_lean import LeanEnv

logger = get_logger("ppo")


@dataclass
class PPOConfig:
    seed: int
    episodes: int
    horizon: int
    backend: str
    learning_rate: float
    gamma: float
    clip: float
    output_dir: Path


def load_config(path: str) -> PPOConfig:
    payload = yaml.safe_load(Path(path).read_text())
    return PPOConfig(
        seed=payload.get("seed", 123),
        episodes=payload.get("episodes", 50),
        horizon=payload.get("horizon", 16),
        backend=payload.get("backend", "mock"),
        learning_rate=payload.get("learning_rate", 3e-4),
        gamma=payload.get("gamma", 0.99),
        clip=payload.get("clip", 0.2),
        output_dir=Path(payload.get("output_dir", "artifacts/checkpoints/rl_ppo")),
    )


class GoalEncoder(nn.Module):
    def __init__(self, vocab_size: int = 256, embed: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)

    def forward(self, goal: str) -> torch.Tensor:
        ids = torch.tensor([ord(ch) % 256 for ch in goal], dtype=torch.long)
        if ids.numel() == 0:
            ids = torch.tensor([0], dtype=torch.long)
        emb = self.embed(ids)
        return emb.mean(dim=0)


class PolicyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GoalEncoder()
        self.policy_head = nn.Linear(32, 1)
        self.value_head = nn.Linear(32, 1)

    def forward(self, goal: str) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(goal)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits.squeeze(-1), value.squeeze(-1)


def select_action(logit: torch.Tensor) -> tuple[int, float]:
    probs = torch.sigmoid(logit)
    action = 1 if random.random() < probs.item() else 0
    log_prob = math.log(probs.item() if action == 1 else 1 - probs.item())
    return action, log_prob


def decode_action(action: int) -> str:
    return "simp" if action else "trivial"


def run_episode(env: LeanEnv, net: PolicyValue, horizon: int) -> dict:
    goal = env.reset()
    steps = []
    for t in range(horizon):
        logit, value = net(goal)
        action, log_prob = select_action(logit)
        tactic = decode_action(action)
        step = env.step(tactic)
        steps.append(
            {
                "goal": goal,
                "action": action,
                "log_prob": log_prob,
                "value": value.item(),
                "reward": step.reward,
            }
        )
        goal = step.goal
        if step.done:
            break
    return {"trajectory": steps, "solved": env.goal_satisfied()}


def compute_returns(trajectory: List[dict], gamma: float) -> List[float]:
    returns = []
    G = 0.0
    for step in reversed(trajectory):
        G = step["reward"] + gamma * G
        returns.append(G)
    return list(reversed(returns))


def ppo_update(net: PolicyValue, optimizer: optim.Optimizer, trajectory: List[dict], returns: List[float], clip: float) -> float:
    losses = []
    for step, ret in zip(trajectory, returns):
        logit, value = net(step["goal"])
        probs = torch.sigmoid(logit)
        action_prob = probs if step["action"] == 1 else 1 - probs
        new_log_prob = torch.log(action_prob + 1e-8)
        ratio = torch.exp(new_log_prob - torch.tensor(step["log_prob"]))
        advantage = ret - step["value"]
        clip_loss = torch.min(
            ratio * advantage,
            torch.clamp(ratio, 1 - clip, 1 + clip) * advantage,
        )
        value_loss = (ret - value) ** 2
        loss = -clip_loss + 0.5 * value_loss
        losses.append(loss)
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train(cfg_path: str = "orchestration/configs/rl_ppo_tiny.yaml") -> None:
    cfg = load_config(cfg_path)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = LeanEnv(backend=cfg.backend)
    net = PolicyValue().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = optim.AdamW(net.parameters(), lr=cfg.learning_rate)
    solved = 0
    for episode in range(cfg.episodes):
        ep = run_episode(env, net, cfg.horizon)
        trajectory = ep["trajectory"]
        returns = compute_returns(trajectory, cfg.gamma)
        loss = ppo_update(net, optimizer, trajectory, returns, cfg.clip)
        solved += int(ep["solved"])
        if episode % 5 == 0:
            logger.info("episode=%d loss=%.4f solved=%d", episode, loss, solved)

    ensure_dir(cfg.output_dir)
    torch.save(net.state_dict(), cfg.output_dir / "ppo_policy.pt")
    logger.info("Saved PPO policy to %s", cfg.output_dir)


if __name__ == "__main__":
    train()
