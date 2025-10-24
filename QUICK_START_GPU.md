# Quick Start: Run ARCHON-RH on GPU in 5 Minutes

## Option 1: Google Colab (Easiest - Free GPU) ⭐ RECOMMENDED

1. **Open Colab**: Go to https://colab.research.google.com/
2. **Upload Notebook**: 
   - File → Upload notebook
   - Choose `colab_setup.ipynb` from this repo
3. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator: **GPU**
   - Choose T4 (free) or A100 (if you have Colab Pro)
4. **Run All Cells**: Runtime → Run all
5. **Download Results**: Artifacts will auto-download at the end

**Total Time:** 5-10 minutes  
**Cost:** FREE (T4) or $10/month (Colab Pro with V100/A100)

---

## Option 2: Lambda Labs (Best for Production)

```bash
# 1. Sign up at https://lambdalabs.com/ and create an A100 instance

# 2. SSH into your instance
ssh ubuntu@<your-instance-ip>

# 3. Clone and setup
git clone <your-repo-url> archon-rh
cd archon-rh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 4. Run GPU training
python prover/train/sft_hf.py orchestration/configs/gpu_production.yaml
```

**Cost:** ~$1.10/hour for A100 40GB  
**Best for:** Serious training runs (4-8 hours)

---

## Option 3: RunPod (Cheapest Spot GPUs)

```bash
# 1. Create account at https://runpod.io/
# 2. Deploy: Pods → GPU Instances → PyTorch template
# 3. Open JupyterLab terminal and run:

git clone <your-repo-url> archon-rh
cd archon-rh
pip install -e .
python prover/train/sft_hf.py orchestration/configs/colab_sft_gpu.yaml
```

**Cost:** $0.39/hour (RTX 4090) to $1-2/hour (A100)  
**Best for:** Budget-conscious training

---

## What Gets Trained?

- **SFT Model**: GPT-2 style transformer that predicts Lean tactics from proof goals
- **Dataset**: Mathematical theorem proving tactics from Mathlib
- **Output**: Trained model checkpoint in `artifacts/checkpoints/`

---

## GPU Configurations

| GPU | VRAM | Batch Size | Model Size | Training Time |
|-----|------|------------|------------|---------------|
| T4 (Colab Free) | 16GB | 16 | 6L-384d (~50M params) | 30 min |
| V100 (Colab Pro) | 32GB | 32 | 12L-768d (~300M params) | 1 hour |
| A100 (Lambda/RunPod) | 40GB | 48 | 16L-1024d (~500M params) | 2 hours |
| A100 80GB | 80GB | 96 | 24L-1536d (~1.5B params) | 4 hours |

---

## Verify It's Using GPU

After starting training, you should see:
```
PyTorch version: 2.2.2+cu118
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
Training on device: cuda
```

If you see `device: cpu`, the GPU isn't being detected. Check your CUDA installation.

---

## Monitoring

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or install nvtop for a better interface
pip install nvitop
nvitop
```

**Target GPU Utilization:** 95%+ during training

---

## Troubleshooting

### "CUDA out of memory"
**Solution:** Reduce batch size in the config file:
```yaml
trainer:
  batch_size: 8  # Reduce from 32
  gradient_accumulation_steps: 8  # Keeps effective batch size same
```

### "RuntimeError: CUDA error: no kernel image is available"
**Solution:** Your CUDA version doesn't match PyTorch. Reinstall:
```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

### Training is slow (< 50% GPU utilization)
**Solution:** Increase batch size or model size in config

---

## Next Steps

1. ✅ Train model on GPU using one of the options above
2. ✅ Download checkpoints from `artifacts/checkpoints/`
3. ✅ Evaluate on test theorems
4. ✅ Fine-tune on custom Lean proofs
5. ✅ Scale up to larger models (see `GPU_SETUP.md`)

**Questions?** See the full `GPU_SETUP.md` guide for advanced options.

