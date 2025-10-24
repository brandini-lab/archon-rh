# Google Colab GPU Quickstart for ARCHON-RH

This guide runs the pipeline on a Colab GPU with minimal changes while preserving the same safety and offline-by-default patterns for the model code (pip installs use network, but training/inference do not make external calls).

## 1) Runtime setup (GPU)
- Runtime ? Change runtime type ? T4/L4/A100 GPU
- New cell:

```bash
!nvidia-smi
!python -V
```

## 2) Clone and install

```bash
!git clone https://github.com/<your-offline-mirror>/archon-rh /content/archon-rh
%cd /content/archon-rh
!python -m pip install -U pip
!pip install -e .[dev]
# Optional extras for larger models
!pip install bitsandbytes accelerate
```

## 3) Build dataset shard

```bash
!python - <<'PY'
from prover.datasets.build_mathlib_dataset import build
build(limit=32)
print('dataset ok')
PY
```

## 4) Train tiny HF model (GPU)

```bash
!python prover/train/sft_hf.py --config orchestration/configs/sft_hf_tiny.yaml
```

Results: `artifacts/checkpoints/sft_hf_tiny/`

## 5) PPO loop on mock Lean (GPU-backed net)

```bash
!python - <<'PY'
from prover.rl.ppo_loop import train
train('orchestration/configs/rl_ppo_tiny.yaml')
print('ppo ok')
PY
```

Results: `artifacts/checkpoints/rl_ppo_tiny/ppo_policy.pt`

## 6) Inference (HF backend)

```bash
!python prover/inference/serve_vllm.py --backend hf --prompts "? True"
```

## 7) Safety + numerics quick check

```bash
!python numerics/zeros/run_zero_checks.py --config orchestration/configs/numeric_verify.yaml
!python - <<'PY'
from safety.reward_verifier import sign_event, verify_event
sig = sign_event('NUMERIC_CERT', {'path':'artifacts/numerics/certificates.json'})
assert verify_event(sig)
print('safety ok')
PY
```

Notes
- LeanDojo can be added if the environment allows (external dependencies + Lean toolchain). The code will gracefully skip LeanDojo-specific tests when not present.
- For bigger runs, copy and edit `orchestration/configs/sft_hf_prod.yaml` and `orchestration/configs/rl_ppo_prod.yaml`.

## Available notebooks
- `colab/archon_setup.ipynb`
- `colab/archon_pipeline.ipynb`
- `colab/archon_lean.ipynb`

