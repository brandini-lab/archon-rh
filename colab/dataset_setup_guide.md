# Dataset Setup Guide for Colab

## Overview

ARCHON-RH supports multiple dataset sources for training the theorem prover. For Colab, we provide automatic fallback mechanisms to ensure the pipeline works even without full LeanDojo setup.

## Dataset Types

### 1. **Mini Mathlib Dataset** (Automatic, Recommended for Colab)

Built from LeanDojo's prebuilt Mathlib dataset with graceful fallback.

**Location:** `artifacts/data/mini_mathlib.jsonl`

**How to build:**
```python
from prover.datasets.build_mathlib_dataset import build

# Build small dataset (fallback if LeanDojo unavailable)
shard = build(limit=32)
print(f"Dataset ready with {len(shard.examples)} examples")
```

**What happens:**
- If LeanDojo is available: Extracts real tactic examples from Mathlib
- If LeanDojo unavailable: Uses fallback trivial examples (still functional)

**Example output:**
```json
{"goal": "⊢ True", "tactic": "trivial", "hypotheses": {}, "metadata": {"source": "fallback"}}
{"goal": "⊢ Nat.zero + 1 = 1", "tactic": "simp", "hypotheses": {}, "metadata": {"source": "fallback"}}
```

### 2. **Sample Data** (Prebuilt, Always Available)

Minimal dataset for testing.

**Location:** `prover/datasets/sample_data/mini_mathlib.jsonl`

**Contents:** ~10 basic tactic examples

**Use case:** Quick testing, CI/CD, debugging

### 3. **Custom RH Dataset** (Advanced, Manual Setup)

Focused dataset for Riemann Hypothesis research.

**How to create:**
```python
from prover.datasets.builders import build_custom_dataset

# Build RH-specific dataset
build_custom_dataset(
    filter_keywords=["Riemann", "Zeta", "Prime", "Dirichlet"],
    output="artifacts/data/rh_tactics.jsonl",
    limit=1000
)
```

**Requires:** Full LeanDojo + Mathlib installation (not ideal for Colab)

## Dataset Structure

All datasets follow the same schema:

```python
{
    "goal": str,           # Lean goal to prove (e.g., "⊢ 1 + 1 = 2")
    "tactic": str,         # Tactic to apply (e.g., "rfl")
    "hypotheses": dict,    # Context hypotheses (optional)
    "metadata": dict       # Additional info (source, file, etc.)
}
```

## Setup in Colab

### Option A: Automatic (Recommended)

```python
# In notebook cell
from prover.datasets.build_mathlib_dataset import build

# Builds dataset automatically (32 examples)
shard = build(limit=32)

# Verify
!ls -lh artifacts/data/
!head -n 3 artifacts/data/mini_mathlib.jsonl
```

**Pros:**
- No manual setup
- Works in any Colab environment
- Automatic fallback

**Cons:**
- Smaller dataset
- May use trivial examples if LeanDojo unavailable

### Option B: Use Sample Data

```python
# Point your config to the prebuilt sample
# In orchestration/configs/sft_tiny.yaml:
trainer:
  dataset: prover/datasets/sample_data/mini_mathlib.jsonl
```

**Pros:**
- Always available
- No build step
- Fast

**Cons:**
- Very small (10 examples)
- Only for testing

### Option C: Upload Your Own

```python
# Upload custom dataset via Colab UI
from google.colab import files
uploaded = files.upload()  # Select your .jsonl file

# Move to expected location
!mv uploaded_file.jsonl artifacts/data/custom_dataset.jsonl

# Update config to point to it
# In your YAML config:
# trainer:
#   dataset: artifacts/data/custom_dataset.jsonl
```

**Pros:**
- Full control
- Can use large, curated datasets

**Cons:**
- Manual upload each session
- Large files slow to upload

## Scaling Dataset Size

### For Testing (Fast)
```python
build(limit=32)  # ~32 examples, builds in seconds
```

### For Development (Medium)
```python
build(limit=500)  # ~500 examples, builds in 1-2 minutes
```

### For Production (Large)
```python
build(limit=10000)  # 10K examples, builds in 10-20 minutes
# Note: May timeout or OOM in free Colab tier
```

## Troubleshooting

### "LeanDojo not found" Warning
**Normal!** The system automatically falls back to trivial examples.

**Fix (optional, advanced):**
```bash
# Install LeanDojo in Colab (may be complex)
!pip install lean-dojo
!lean4 --version  # Verify Lean installation
```

### Empty Dataset
```python
# Check if file exists
!ls -lh artifacts/data/mini_mathlib.jsonl

# Rebuild
from prover.datasets.build_mathlib_dataset import build
build(limit=32)
```

### Training Fails on Dataset
Check format:
```python
import json
with open("artifacts/data/mini_mathlib.jsonl") as f:
    for i, line in enumerate(f):
        print(f"Line {i}: {json.loads(line)}")
        if i > 5:
            break
```

Each line must be valid JSON with required keys.

### Out of Memory During Dataset Build
Reduce limit:
```python
build(limit=16)  # Smaller dataset
```

## Best Practices for Colab

1. **Start small:** Use `limit=32` for first runs
2. **Check fallback:** Ensure trivial examples work before scaling
3. **Monitor size:** `!du -sh artifacts/data/` to track disk usage
4. **Cache results:** Download dataset after building to avoid rebuilding
5. **Version control:** Keep track of which dataset version you're using

## Dataset Download/Upload

### Download Built Dataset
```python
from google.colab import files

# Zip and download
!zip -r dataset.zip artifacts/data/
files.download('dataset.zip')
```

### Upload to Next Session
```python
from google.colab import files

# Upload the zip
uploaded = files.upload()
!unzip dataset.zip
```

## Verifying Dataset Quality

```python
import json
from collections import Counter

# Load and analyze
with open("artifacts/data/mini_mathlib.jsonl") as f:
    examples = [json.loads(line) for line in f]

print(f"Total examples: {len(examples)}")

# Count tactics
tactics = Counter(ex['tactic'] for ex in examples)
print(f"Unique tactics: {len(tactics)}")
print(f"Top 10 tactics: {tactics.most_common(10)}")

# Check for diversity
goals = [ex['goal'] for ex in examples]
unique_goals = len(set(goals))
print(f"Unique goals: {unique_goals} ({unique_goals/len(goals)*100:.1f}%)")
```

## Next Steps

After dataset is ready:
1. ✅ Run SFT training: `!python prover/train/sft_hf.py --config orchestration/configs/colab_sft_gpu.yaml`
2. ✅ Verify model learns: Check loss curves in `artifacts/checkpoints/`
3. ✅ Test inference: Generate tactics for new goals
4. ✅ Scale up: Increase dataset size for better performance

---

**Summary:** For Colab, just run `build(limit=32)` and you're ready. The system handles everything automatically with smart fallbacks.

