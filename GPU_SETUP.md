# Running ARCHON-RH on High-Performance GPUs

## Quick Options

### 1. Google Colab (Easiest - Free GPU)
- **GPU Options:** T4 (free), V100/A100 (Pro/Pro+)
- **Setup Time:** 5 minutes
- **Cost:** Free - $50/month

**Steps:**
1. Open `colab_setup.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU (T4)
3. Run all cells
4. Download artifacts when done

**Recommended for:** Quick experiments, prototyping, demo

---

### 2. Lambda Labs (Best Price/Performance)
- **GPU Options:** A100 (40GB/80GB), H100
- **Cost:** ~$1.10/hr (A100) to $2.50/hr (H100)
- **Setup:** SSH + Docker

**Setup:**
```bash
# SSH into Lambda instance
ssh ubuntu@<instance-ip>

# Clone repo
git clone https://github.com/your-repo/archon-rh.git
cd archon-rh

# Use Docker (recommended)
docker build -f docker/trainer.Dockerfile -t archon-trainer .
docker run --gpus all -v $(pwd):/workspace archon-trainer \
  python prover/train/sft_hf.py orchestration/configs/gpu_production.yaml

# OR install directly
python -m venv .venv
source .venv/bin/activate
pip install -e .
python prover/train/sft_hf.py orchestration/configs/gpu_production.yaml
```

**Recommended for:** Serious training, production runs

---

### 3. RunPod.io (Flexible Spot Instances)
- **GPU Options:** RTX 4090, A100, H100
- **Cost:** $0.39/hr (4090) to $2/hr (H100) on spot
- **Setup:** Web interface + Jupyter

**Setup:**
1. Create account at runpod.io
2. Deploy: "PyTorch" template with GPU
3. Open Jupyter terminal:
```bash
git clone https://github.com/your-repo/archon-rh.git
cd archon-rh
pip install -e .
python prover/train/sft_hf.py orchestration/configs/gpu_production.yaml
```

**Recommended for:** Cost-conscious training, experimentation

---

### 4. Vast.ai (Cheapest Spot GPUs)
- **GPU Options:** Any GPU (P100, V100, 4090, A100, etc.)
- **Cost:** $0.10/hr - $1/hr for most GPUs
- **Setup:** SSH + Docker preferred

**Setup:**
```bash
# SSH into Vast instance
ssh -p <port> root@<ip>

# Run via Docker
cd /workspace
git clone https://github.com/your-repo/archon-rh.git
cd archon-rh
pip install -e .
python prover/train/sft_hf.py orchestration/configs/gpu_production.yaml
```

**Recommended for:** Budget training, large-scale sweeps

---

### 5. Kaggle (Free Alternative to Colab)
- **GPU Options:** P100 (free), T4 (free)
- **Limits:** 30hrs/week GPU quota
- **Cost:** Free

**Setup:**
1. Create Kaggle account
2. New Notebook → Settings → Accelerator: GPU
3. Add Dataset or clone repo:
```python
!git clone https://github.com/your-repo/archon-rh.git
%cd archon-rh
!pip install -e .
!python prover/train/sft_hf.py orchestration/configs/colab_sft_gpu.yaml
```

**Recommended for:** Free alternative to Colab

---

### 6. AWS SageMaker / Azure ML / GCP Vertex AI
- **GPU Options:** Any (A100, V100, etc.)
- **Cost:** $3-8/hr depending on instance
- **Setup:** Platform-specific

**General approach:**
- Use managed Jupyter notebooks
- Upload repo as compressed file
- Install dependencies in notebook
- Run training scripts

**Recommended for:** Enterprise deployments, compliance needs

---

## GPU Configuration Comparison

| Platform | Free Tier | Cheapest GPU | Best GPU | Setup | Persistence |
|----------|-----------|--------------|----------|-------|-------------|
| Colab | T4 | T4 (Free) | A100 ($50/mo) | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Lambda Labs | ❌ | A10 ($0.60/hr) | H100 ($2.50/hr) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| RunPod | ❌ | RTX 4090 ($0.39/hr) | H100 ($2/hr) | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Vast.ai | ❌ | GTX 1080 ($0.10/hr) | A100 ($1/hr) | ⭐⭐⭐ | ⭐⭐⭐ |
| Kaggle | P100/T4 | Free | Free | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## Optimized Configs for Different GPUs

### T4 (16GB VRAM) - Colab Free
```yaml
model:
  n_layer: 6
  n_head: 6
  n_embd: 384
trainer:
  batch_size: 16
  gradient_accumulation_steps: 4
  mixed_precision: fp16
```

### V100 (32GB VRAM) - Colab Pro
```yaml
model:
  n_layer: 12
  n_head: 12
  n_embd: 768
trainer:
  batch_size: 32
  gradient_accumulation_steps: 2
  mixed_precision: fp16
```

### A100 (40GB VRAM) - Lambda/RunPod
```yaml
model:
  n_layer: 24
  n_head: 16
  n_embd: 1024
trainer:
  batch_size: 64
  gradient_accumulation_steps: 2
  mixed_precision: fp16
```

### A100 (80GB VRAM) - Production
```yaml
model:
  n_layer: 32
  n_head: 32
  n_embd: 2048
trainer:
  batch_size: 128
  gradient_accumulation_steps: 1
  mixed_precision: bf16  # Better for A100
```

---

## Adding Mixed Precision Training

The current code supports automatic GPU usage. To add mixed precision (2-4x speedup):

**For PyTorch AMP:**
```python
# In sft_hf.py, add:
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for step in range(steps):
    with autocast():
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**For HuggingFace Accelerate (multi-GPU):**
```python
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision='fp16')
model, optimizer = accelerator.prepare(model, optimizer)
```

---

## Monitoring GPU Usage

**During Training:**
```bash
# Terminal 1: Run training
python prover/train/sft_hf.py orchestration/configs/gpu_production.yaml

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Or use nvtop (better interface)
pip install nvitop
nvitop
```

**Check GPU utilization:**
- **Target:** 95%+ GPU utilization during training
- **If low:** Increase batch size or model size
- **If OOM:** Decrease batch size or enable gradient checkpointing

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
# In config YAML:
batch_size: 8  # Reduce from 32
gradient_accumulation_steps: 8  # Keep effective batch size

# Enable gradient checkpointing (trades compute for memory)
# Add to model config:
gradient_checkpointing: true
```

### Slow Training
```bash
# Enable mixed precision (should be ~2x faster)
mixed_precision: fp16

# Use larger batch sizes
batch_size: 64  # As large as GPU allows

# Check if CPU is bottleneck
# Monitor with htop - if CPU at 100%, increase num_workers for data loading
```

### CUDA Version Mismatch
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.8:
pip install torch==2.2.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 12.1:
pip install torch==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

---

## Recommended Workflow

**For experimentation (free):**
1. Start with Google Colab T4
2. Prototype with `colab_sft_gpu.yaml`
3. Train for 500-1000 steps (~15 minutes)
4. Download checkpoints

**For serious training:**
1. Use Lambda Labs A100 ($1/hr)
2. Use `gpu_production.yaml` config
3. Train for 10K-50K steps (~2-4 hours)
4. Save to cloud storage (S3/GCS)
5. Shut down instance

**For large-scale experiments:**
1. Use Vast.ai spot instances
2. Run multiple parallel jobs
3. Use Ray Tune for hyperparameter sweeps
4. Aggregate results

---

## Cost Estimates

**Free Tier (Colab T4):**
- Cost: $0
- Time: 1-2 hours training
- Model: Small (~100M params)

**Production (Lambda A100):**
- Cost: $1-2/hr × 4 hrs = $4-8
- Model: Medium (~500M params)
- Quality: Good for research

**Large-Scale (Vast.ai multi-GPU):**
- Cost: $1/hr × 10 GPUs × 8 hrs = $80
- Model: Large (1-3B params)
- Quality: SOTA for theorem proving

---

## Next Steps

1. ✅ Start with Colab notebook (`colab_setup.ipynb`)
2. ✅ Train small model to verify setup
3. ✅ Scale up to Lambda/RunPod for production
4. ✅ Monitor training metrics
5. ✅ Save checkpoints regularly
6. ✅ Evaluate on held-out theorems

**Need help?** Check the main README or open an issue.

