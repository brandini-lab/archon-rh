# ARCHON-RH

ARCHON-RH is an offline reasoning laboratory for formal mathematics research. The system focuses on Lean theorem proving, reinforcement learning over tactic policies, FunSearch-style conjecture discovery, and interval-arithmetic numeric certification. Everything runs inside an air-gapped environment with signed reward channels and zero network egress.

## Quick Start

```bash
make dev          # create venv with pinned dependencies
make test         # run CPU-only acceptance tests
make run-sft      # launch the tiny supervised finetune loop
make run-rl       # run a 60s self-play PPO session
make run-funsearch
make run-numerics
```

These commands assume a POSIX shell. On Windows, activate the `.venv` and replace `python` with `python.exe`.

## Offline Operation

* All data, checkpoints, and binaries live under the repository.
* Docker images have no package manager updates inside build steps; the base images are pinned via digest.
* `infra/k8s` applies a default deny egress NetworkPolicy and only allows pod-to-pod traffic inside the namespace.
* `safety/policies/reward_verifier.py` loads signed reward messages from the orchestrated jobs. Pipelines fail if a signature is missing or the signer is not on the allowlist.
* Seccomp profiles in `safety/sandbox` disable `clone`, `unshare`, and networking syscalls for model execution containers.

## Reward Signing Flow

1. Task runners write reward events to `artifacts/rewards/queue.jsonl`.
2. `safety/audits/sign_rewards.py` signs and appends the event hash into a Merkle chain (`safety/audits/log.jsonl`).
3. `safety/policies/reward_verifier.py` is invoked before optimizers consume rewards; it rejects unsigned or stale entries.

## Adding New Objectives

1. Implement an evaluator in `conjecture/objectives/`. Follow the template in `li_criterion.py`.
2. Register it inside `conjecture/funsearch/objective_registry.py`.
3. Add acceptance coverage via `conjecture/tests/test_objectives.py`.
4. Update `orchestration/configs/funsearch_loop.yaml` to include the objective in the job.

## Reproducibility Checklist

* Deterministic seeds live in `orchestration/pipelines/common.py`.
* PyTorch, Ray, and Lean toolchains are version pinned in `pyproject.toml` and Dockerfiles.
* `make docker` builds four images and exports SBOMs under `artifacts/sbom/`.
* `infra/terraform` provisions an isolated VPC with no internet gateway; only bastion hosts receive human SSH access.

## Acceptance Tests

* `pytest -k toy_lemma` proves a toy lemma with the Lean bridge via LeanDojo IPC.
* `pytest -k funsearch_objective` verifies that the Pareto archive improves after ten evaluations with random seeds.
* `pytest -k numeric_cert` emits a single zero-count certificate and checks it with the Lean side verifier.

The tests run entirely on CPU and operate on toy datasets or truncated numeric problems.

## Architecture Overview

* **Formal:** Lean library with RH-adjacent lemmas plus LeanDojo RPC bridge.
* **Prover:** Dataset builders, supervised training, RL self-play, inference runners, metrics.
* **Conjecture:** FunSearch loop generating new analytic kernels and exporting conjectures.
* **Numerics:** Interval-arithmetic bindings and certificate emitters.
* **Orchestration:** Ray + CLI pipelines, plus job configuration.
* **Safety:** Reward policy enforcement, sandboxing, signed audit logs.

Each subsystem has focused unit tests along with end-to-end acceptance coverage.

## Re-running Acceptance Tests

```bash
git clone <offline mirror>
cd archon-rh
make dev
make test
pytest -k toy_lemma
pytest -k funsearch_objective
pytest -k numeric_cert
make run-sft
make run-rl
make run-numerics
```

These commands leave artifacts in `artifacts/` with reproducible hashes derived from inputs.

## Lean Integration & Advanced Pipelines
- LeanDojo integration via `formal/leandojo_bridge/client.py` with real Lean project scaffolding under `formal/lean_project/`.
- HuggingFace SFT trainer (`prover/train/sft_hf.py`) and PPO loop (`prover/rl/ppo_loop.py`) run against Mathlib-derived data built by `prover/datasets/build_mathlib_dataset.py`.


## Colab Notebooks
- `colab/archon_setup.ipynb`: environment prep (GPU runtime, install, tests).
- `colab/archon_pipeline.ipynb`: SFT ? PPO ? FunSearch ? numerics ? reward signing.
- `colab/archon_lean.ipynb`: optional LeanDojo proof attempt and conjecture export demo.

