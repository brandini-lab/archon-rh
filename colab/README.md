# ARCHON-RH Colab Notebooks

This directory contains Google Colab notebooks for running ARCHON-RH in the cloud.

## Available Notebooks

### 1. **ARCHON_RH_Complete_Setup.ipynb** (Recommended)
Complete end-to-end pipeline with all components:
- GPU verification
- Dataset building
- SFT training
- PPO reinforcement learning
- FunSearch conjecture discovery
- Numerical verification
- Safety validation

**Time:** ~45-60 minutes on T4 GPU  
**Use case:** Full research pipeline, production runs

### 2. **Quick_Start.ipynb**
Fast testing and validation:
- Minimal setup
- Quick dataset build
- Tiny model training
- Basic verification

**Time:** ~15 minutes on T4 GPU  
**Use case:** Testing, debugging, quick iterations

## Setup Instructions

### Step 1: Open in Colab
1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. File → Upload notebook
3. Select one of the `.ipynb` files from this directory

### Step 2: Enable GPU
1. Runtime → Change runtime type
2. Hardware accelerator: **GPU** (T4/V100/A100)
3. Runtime shape: **High-RAM** (if available)

### Step 3: Update Repository URL
In the clone cell, replace `<YOUR_USERNAME>` with your actual GitHub username or organization:
```python
!git clone https://github.com/YOUR_USERNAME/archon-rh /content/archon-rh
```

### Step 4: Run All Cells
- Runtime → Run all (Ctrl+F9)
- Or run cells sequentially (Shift+Enter)

## GPU Requirements

| Notebook | Minimum GPU | Recommended GPU | RAM |
|----------|------------|-----------------|-----|
| Complete Setup | T4 (16GB) | V100 (32GB) | Standard |
| Quick Start | T4 (16GB) | T4 (16GB) | Standard |

## Dataset Information

**Default Dataset:** Mathlib tactic examples via LeanDojo
- **Fallback:** If LeanDojo unavailable, uses trivial examples
- **Location:** `artifacts/data/mini_mathlib.jsonl`
- **Size:** Configurable (default: 32-64 examples for testing)

**Dataset Structure:**
```json
{
  "goal": "⊢ True",
  "tactic": "trivial",
  "hypotheses": {},
  "metadata": {"source": "fallback"}
}
```

## Configuration Files

All training/testing uses configs from `orchestration/configs/`:

| Config File | Purpose | Model Size | Training Steps |
|-------------|---------|------------|----------------|
| `colab_sft_gpu.yaml` | SFT training (GPU) | 8 layers, 512 embd | 5000 |
| `colab_rl_gpu.yaml` | RL training (GPU) | PPO policy | 2000 episodes |
| `sft_tiny.yaml` | Quick testing | 4 layers, 128 embd | 200 |
| `gpu_production.yaml` | Production runs | Large model | 20000+ |

## Troubleshooting

### GPU Not Available
```python
import torch
print(torch.cuda.is_available())  # Should be True
```
If False: Runtime → Change runtime type → GPU

### Out of Memory
- Use smaller batch size in config files
- Reduce model size (n_layer, n_embd)
- Enable gradient checkpointing
- Switch to `sft_tiny.yaml` config

### Import Errors
```bash
!pip install --upgrade pip
!pip install -e .[dev] --force-reinstall
```

### Dataset Build Fails
The system automatically falls back to trivial examples if LeanDojo is unavailable. This is expected in some Colab environments.

## Downloading Results

All artifacts are saved to `/content/archon-rh/artifacts/`:
```
artifacts/
├── checkpoints/       # Trained models
├── data/             # Datasets
├── funsearch/        # Conjectures
├── numerics/         # Certificates
└── rewards/          # Reward logs
```

The complete setup notebook includes an automatic download cell that zips everything.

## Next Steps After Colab Run

1. **Analyze Results:**
   - Review generated conjectures in `artifacts/funsearch/`
   - Check numerical certificates in `artifacts/numerics/`
   - Examine training logs in checkpoint directories

2. **Scale Up:**
   - Use `gpu_production.yaml` configs
   - Increase dataset size
   - Train for more steps

3. **Local Development:**
   - Download artifacts
   - Continue training locally
   - Run formal verification with full Lean setup

## Safety Notes

- All operations are offline after initial setup
- No external API calls during training/inference
- Reward signatures validated
- Safety policies enforced

## Support

For issues or questions:
1. Check error logs in notebook output
2. Verify GPU is enabled
3. Review configuration files
4. Check `README.md` in project root

