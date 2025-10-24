# ARCHON-RH Colab Setup Summary

## ✅ What We've Set Up

### 1. **Organized Colab Directory Structure**
```
colab/
├── ARCHON_RH_Complete_Setup.ipynb    # Full end-to-end pipeline (45-60 min)
├── Quick_Start.ipynb                  # Fast testing setup (15 min)
├── README.md                          # Main documentation
├── dataset_setup_guide.md             # Dataset building instructions
└── SETUP_SUMMARY.md                   # This file
```

### 2. **Dataset Setup - VERIFIED ✓**

**Available Datasets:**

1. **Automatic Mini Mathlib** (Recommended)
   - Location: `artifacts/data/mini_mathlib.jsonl` (auto-generated)
   - Source: `prover/datasets/build_mathlib_dataset.py`
   - Size: Configurable (default: 32-64 examples)
   - **Fallback:** If LeanDojo unavailable, uses trivial examples
   
2. **Sample Data** (Always Available)
   - Location: `archon-rh/prover/datasets/sample_data/mini_mathlib.jsonl`
   - Size: 2 examples (for testing)
   - Format:
     ```json
     {"goal": "? True", "tactic": "trivial", "hypotheses": {}, "metadata": {"source": "sample"}}
     {"goal": "? Nat.zero + 1 = 1", "tactic": "simp", "hypotheses": {}, "metadata": {"source": "sample"}}
     ```

**Dataset Build Command:**
```python
from prover.datasets.build_mathlib_dataset import build
shard = build(limit=64)  # Builds 64 examples
```

### 3. **GPU Configuration Files**

**Available Configs:**
- `orchestration/configs/colab_sft_gpu.yaml` - SFT training optimized for Colab GPUs
- `orchestration/configs/colab_rl_gpu.yaml` - RL training optimized for Colab GPUs
- `orchestration/configs/sft_tiny.yaml` - Quick testing config
- `orchestration/configs/gpu_production.yaml` - Full production runs

### 4. **Riemann Hypothesis Research Plan**

See `RIEMANN_HYPOTHESIS_CHECKLIST.md` (root directory) for:
- 10-phase research roadmap
- Detailed success criteria
- Timeline estimates (2-18 months depending on approach)
- Technical requirements and resources

---

## 🚀 Quick Start Guide

### For First-Time Users:
1. Open `colab/Quick_Start.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU
3. Replace `<YOUR_USERNAME>` with your GitHub username
4. Run all cells (Ctrl+F9)
5. Wait ~15 minutes
6. Download artifacts

### For Full Pipeline:
1. Open `colab/ARCHON_RH_Complete_Setup.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU (High-RAM if available)
3. Replace `<YOUR_USERNAME>` with your GitHub username
4. Run all cells sequentially
5. Wait ~45-60 minutes
6. Review and download all artifacts

---

## 📊 Pipeline Components

### Phase 1: Dataset Building
- **Tool:** `build_mathlib_dataset.py`
- **Output:** `artifacts/data/mini_mathlib.jsonl`
- **Time:** 30 seconds - 2 minutes
- **GPU:** Not required

### Phase 2: SFT Training
- **Tool:** `prover/train/sft_hf.py`
- **Config:** `colab_sft_gpu.yaml`
- **Output:** `artifacts/checkpoints/sft_colab_gpu/`
- **Time:** 10-20 minutes (T4 GPU)
- **GPU:** Required (16GB+ VRAM)

### Phase 3: PPO Reinforcement Learning
- **Tool:** `prover/rl/ppo_loop.py`
- **Config:** `colab_rl_gpu.yaml`
- **Output:** `artifacts/checkpoints/rl_colab_gpu/`
- **Time:** 5-15 minutes
- **GPU:** Required

### Phase 4: FunSearch Conjecture Discovery
- **Tool:** `conjecture/funsearch/funsearch_loop.py`
- **Config:** `funsearch_loop.yaml`
- **Output:** `artifacts/funsearch/`
- **Time:** 10-20 minutes
- **GPU:** Recommended

### Phase 5: Numerical Verification
- **Tool:** `numerics/zeros/run_zero_checks.py`
- **Config:** `numeric_verify.yaml`
- **Output:** `artifacts/numerics/`
- **Time:** 1-5 minutes
- **GPU:** Optional

---

## 🔧 Troubleshooting

### Issue: GPU Not Available
**Solution:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
```
If False: Runtime → Change runtime type → Hardware accelerator: GPU

### Issue: Out of Memory
**Solution:**
- Reduce batch size in config files
- Use `sft_tiny.yaml` instead of `colab_sft_gpu.yaml`
- Restart runtime and clear outputs

### Issue: Dataset Build Fails
**Solution:**
- Check if fallback examples are being used (this is normal)
- Verify sample data exists: `!ls archon-rh/prover/datasets/sample_data/`
- Manually check dataset: `!cat artifacts/data/mini_mathlib.jsonl`

### Issue: Import Errors
**Solution:**
```bash
!pip install --upgrade pip
!pip install -e .[dev] --force-reinstall
```

### Issue: Training Takes Too Long
**Solution:**
- Use Quick_Start.ipynb instead
- Reduce training steps in config files
- Use T4/V100/A100 GPU (not CPU)

---

## 📁 File Organization

### Original Structure (archon-rh/)
```
archon-rh/
├── colab/                    # Old notebooks (can be removed)
│   ├── archon_lean.ipynb
│   ├── archon_pipeline.ipynb
│   └── archon_setup.ipynb
├── prover/                   # ML theorem prover
├── conjecture/               # FunSearch loop
├── numerics/                 # Interval arithmetic
├── formal/                   # Lean formalization
└── orchestration/            # Configs & pipelines
```

### New Structure (root/colab/)
```
colab/
├── ARCHON_RH_Complete_Setup.ipynb   # Main notebook (NEW)
├── Quick_Start.ipynb                # Fast testing (NEW)
├── README.md                        # Documentation (NEW)
├── dataset_setup_guide.md           # Dataset help (NEW)
└── SETUP_SUMMARY.md                 # This file (NEW)
```

**Recommendation:** Use the new `colab/` directory at root level. Old notebooks in `archon-rh/colab/` can be archived or removed.

---

## 🎯 Riemann Hypothesis Checklist Summary

From `RIEMANN_HYPOTHESIS_CHECKLIST.md`:

**Phase 1-2:** Environment & Dataset (1-4 weeks)
- ✅ Colab setup complete
- ✅ Dataset infrastructure ready
- ⏳ RH-specific dataset curation needed

**Phase 3-4:** Training (2-6 weeks)
- ✅ SFT pipeline ready
- ✅ RL pipeline ready
- ⏳ Train on large dataset
- ⏳ Evaluate on Mathlib gaps

**Phase 5-6:** Verification & Discovery (2-8 weeks)
- ✅ Numerics module ready
- ✅ FunSearch loop ready
- ⏳ Verify 10^5+ zeros
- ⏳ Generate conjectures

**Phase 7-10:** Formalization & Publication (2-12 months)
- ⏳ Formalize in Lean
- ⏳ Automated proof attempts
- ⏳ Human-in-the-loop refinement
- ⏳ Peer review & publication

---

## 📚 Key Resources

### Documentation
- `colab/README.md` - Main Colab documentation
- `colab/dataset_setup_guide.md` - Dataset instructions
- `RIEMANN_HYPOTHESIS_CHECKLIST.md` - Full research roadmap
- `README.md` (root) - Project overview
- `docs/COLAB.md` - Original Colab guide

### Code Modules
- `prover/` - ML-based theorem prover
- `formal/leanlib/ArchonRiemann/` - RH Lean formalization
- `numerics/` - Interval arithmetic & zero verification
- `conjecture/` - FunSearch conjecture discovery
- `orchestration/configs/` - All config files

### Key Scripts
- `prover/datasets/build_mathlib_dataset.py` - Dataset builder
- `prover/train/sft_hf.py` - SFT training
- `prover/rl/ppo_loop.py` - RL training
- `conjecture/funsearch/funsearch_loop.py` - Conjecture discovery
- `numerics/zeros/run_zero_checks.py` - Zero verification

---

## ✅ Pre-Flight Checklist

Before starting any Colab notebook:

- [ ] GPU runtime enabled (T4 minimum, V100/A100 recommended)
- [ ] High-RAM runtime (if available)
- [ ] GitHub repository URL updated in clone cells
- [ ] Expected runtime: 15 min (Quick) or 60 min (Complete)
- [ ] Stable internet connection
- [ ] Google Drive space for artifacts (~500MB-2GB)

---

## 🎓 Learning Path

### Beginner (Week 1)
1. Run `Quick_Start.ipynb`
2. Read `colab/README.md`
3. Inspect generated artifacts
4. Understand dataset format

### Intermediate (Week 2-4)
1. Run `ARCHON_RH_Complete_Setup.ipynb`
2. Modify config files (batch size, steps)
3. Read `RIEMANN_HYPOTHESIS_CHECKLIST.md`
4. Explore Lean formalization basics

### Advanced (Month 2+)
1. Create custom datasets for RH
2. Train larger models (production configs)
3. Contribute to Lean library
4. Run full numerical verification
5. Generate and validate conjectures

---

## 🤝 Next Steps

1. **Immediate:** Open a notebook and run the pipeline
2. **This Week:** Review all generated artifacts
3. **This Month:** Scale to production configs
4. **Long-term:** Follow the 10-phase RH research checklist

---

## 📞 Support

- **Issues:** Check Troubleshooting section above
- **Documentation:** See Key Resources section
- **Code:** All Python modules have docstrings
- **Community:** Check project README for contribution guidelines

---

**Status:** ✅ Colab setup complete and ready for Riemann Hypothesis research!

**Last Updated:** 2025-10-24

