<!-- beddf4df-108d-4bdd-9bef-b6dd9ff352f4 a854a818-0e10-4977-97fc-ae1ab903bfa4 -->
# Consolidate ARCHON-RH Repository Structure

## Overview

Fix duplicate directory structure, missing imports, and incompatible dependencies to work seamlessly in both Google Colab and local development.

## Phase 1: Move libs Module to Root

Currently `libs/` only exists in `archon-rh/` subdirectory, but root-level modules need it.

**Actions:**

- Move `archon-rh/libs/` to root level `libs/`
- Verify `libs/rh_common/` contains:
  - `__init__.py` (exports: get_logger, ensure_dir, write_json, read_json)
  - `logging.py`
  - `artifacts.py`

## Phase 2: Add Missing **init**.py Files

Root-level modules are missing package markers.

**Actions:**

Create empty `__init__.py` in:

- `prover/__init__.py`
- `prover/datasets/__init__.py`
- `prover/inference/__init__.py`
- `prover/metrics/__init__.py`
- `prover/rl/__init__.py`
- `prover/train/__init__.py`
- `conjecture/__init__.py`
- `conjecture/funsearch/__init__.py`
- `conjecture/objectives/__init__.py`
- `numerics/__init__.py`
- `numerics/arb_bindings/__init__.py`
- `numerics/explicit_bounds/__init__.py`
- `numerics/zeros/__init__.py`
- `orchestration/__init__.py`
- `orchestration/pipelines/__init__.py`
- `formal/__init__.py`
- `formal/leandojo_bridge/__init__.py`
- `safety/__init__.py`
- `safety/audits/__init__.py`
- `safety/policies/__init__.py`

## Phase 3: Update pyproject.toml for Python 3.12

Current issues:

- `ray==2.10.0` doesn't exist for Python 3.12 (available: 2.31.0+)
- Some dependency versions may be outdated

**Actions:**

Update `pyproject.toml`:

```toml
requires-python = ">=3.10,<3.13"

dependencies = [
  # Core
  "click>=8.1.7,<9.0",
  "numpy>=1.26,<2.1",
  "pandas>=2.1,<2.3",
  "pyarrow>=14.0,<18.0",
  
  # ML/DL
  "torch>=2.2.2,<2.6",
  "torchaudio>=2.2.2,<2.6",
  "torchvision>=0.17.2,<0.20",
  "accelerate>=0.31,<0.35",
  "transformers>=4.40,<4.48",
  "sentencepiece>=0.1.99,<0.3.0",
  "bitsandbytes>=0.43,<0.45; platform_system == 'Linux'",
  
  # Math/Science
  "mpmath>=1.3,<2.0",
  "sympy>=1.12,<1.14",
  
  # Utilities
  "rich>=13.7,<14.0",
  "pydantic>=2.7,<2.10",
  "pyyaml>=6.0,<7.0",
  "tqdm>=4.66,<5.0",
  "networkx>=3.2,<3.5",
  
  # Data
  "datasets>=2.19,<3.0",
  "huggingface-hub>=0.22,<0.26",
  
  # Testing
  "pytest>=8.2,<8.4",
  "pytest-asyncio>=0.23,<0.25",
  
  # Misc
  "tenacity>=8.2,<9.0",
  "cryptography>=42,<44",
  "shortuuid>=1.0,<2.0"
]

[project.optional-dependencies]
dev = [
  "ruff",
  "pytest",
  "pytest-cov",
  "mypy",
  "types-PyYAML",
  "types-requests"
]

# Ray is optional (not needed for Colab, fails on Python 3.12)
ray = [
  "ray[default]>=2.31.0,<2.51"
]

local = [
  "uvicorn>=0.29,<0.31",
  "fastapi>=0.111,<0.113",
  "typer>=0.12,<0.14"
]
```

## Phase 4: Remove Duplicate archon-rh Directory

After moving libs/ to root, the nested directory is redundant.

**Actions:**

- Verify no unique files in `archon-rh/` that aren't in root
- Keep `archon-rh/colab/` notebooks (they're old versions to archive)
- Delete entire `archon-rh/` directory

Files to preserve before deletion:

- Check if any configs in `archon-rh/orchestration/configs/` are newer
- Archive old notebooks from `archon-rh/colab/` to `colab/archive/` for reference

## Phase 5: Consolidate Colab Notebooks

Currently have:

- `colab/` (new, updated) at root
- `archon-rh/colab/` (old versions)

**Actions:**

- Keep root-level `colab/` directory with updated notebooks
- Move old notebooks to `colab/archive/` for reference:
  - `archon_setup.ipynb`
  - `archon_pipeline.ipynb`
  - `archon_lean.ipynb`

## Phase 6: Update Import Paths

Files importing `libs.rh_common` should work once libs/ is at root level, but verify:

**Files to check:**

- `prover/datasets/build_mathlib_dataset.py:8`
- `prover/rl/ppo_loop.py:15`
- `prover/train/sft_hf.py:14`
- `prover/inference/serve_vllm.py:8`

All use: `from libs.rh_common import ...`

This will work once `libs/` exists at root and `pyproject.toml` packages include it.

**Update pyproject.toml packages line:**

```toml
[tool.setuptools]
packages = {find = {include = ["libs*", "formal*", "prover*", "conjecture*", "numerics*", "orchestration*", "safety*"]}}
```

## Phase 7: Update Configuration Files

Verify all config paths reference correct structure:

**Check configs in:**

- `orchestration/configs/*.yaml` - ensure paths are relative to root
- Colab configs especially: `colab_sft_gpu.yaml`, `colab_rl_gpu.yaml`

**Update if needed:**

- Dataset paths: `artifacts/data/mini_mathlib.jsonl`
- Output paths: `artifacts/checkpoints/...`
- Config references in notebooks

## Phase 8: Update Colab Notebooks

Update installation cell in both notebooks:

**In `colab/ARCHON_RH_Complete_Setup.ipynb` and `colab/Quick_Start.ipynb`:**

Replace Step 4 (Install Dependencies) with:

```python
import sys
import os

%cd /content/archon-rh

# Add to Python path
sys.path.insert(0, '/content/archon-rh')

# Create any missing __init__.py files (shouldn't be needed after fixes)
directories = [
    'prover', 'prover/datasets', 'prover/inference', 'prover/metrics',
    'prover/rl', 'prover/train', 'conjecture', 'conjecture/funsearch',
    'conjecture/objectives', 'numerics', 'numerics/arb_bindings',
    'numerics/zeros', 'orchestration', 'orchestration/pipelines',
    'libs', 'libs/rh_common', 'safety', 'safety/policies'
]

for d in directories:
    init_file = Path(d) / '__init__.py'
    init_file.touch(exist_ok=True)

# Install dependencies (ray optional, skip for Colab)
print("Installing dependencies...")
!pip install -q -e ".[dev]"

# Verify imports
from prover.datasets.build_mathlib_dataset import build
from conjecture.funsearch.funsearch_loop import run_loop
from libs.rh_common import get_logger

print("✓ Installation complete!")
```

## Phase 9: Update Documentation

Update references to structure in:

**Files to update:**

- `README.md` - architecture section, quick start
- `colab/README.md` - installation instructions
- `colab/dataset_setup_guide.md` - paths and structure
- `colab/SETUP_SUMMARY.md` - directory structure diagrams
- `RIEMANN_HYPOTHESIS_CHECKLIST.md` - file references
- `docs/COLAB.md` - installation steps

**Key changes:**

- Remove references to nested `archon-rh/` structure
- Update import examples
- Update directory tree diagrams
- Clarify that ray is optional (for local orchestration only)

## Phase 10: Verify and Test

After all changes:

**Local verification:**

1. Run `pip install -e .[dev]` from root
2. Run `pytest` to verify tests pass
3. Test imports:
   ```python
   from prover.datasets.build_mathlib_dataset import build
   from libs.rh_common import get_logger
   from conjecture.funsearch.funsearch_loop import run_loop
   ```


**Colab verification:**

1. Push changes to GitHub
2. Test updated notebooks in Colab
3. Verify dataset building works
4. Confirm training scripts can import properly

## Phase 11: Git Commit and Push

Commit all changes:

```bash
git add .
git commit -m "Restructure: consolidate to single root structure, fix imports, update Python 3.12 compatibility"
git push origin main
```

Update Colab links will automatically use new structure.

## Files Changed Summary

**Created:**

- 20+ `__init__.py` files in root modules
- `libs/` at root level (moved from archon-rh/)
- `colab/archive/` with old notebooks

**Modified:**

- `pyproject.toml` (dependencies, packages)
- `colab/ARCHON_RH_Complete_Setup.ipynb` (installation cell)
- `colab/Quick_Start.ipynb` (installation cell)
- `README.md`, `colab/README.md`, and other docs

**Deleted:**

- `archon-rh/` entire subdirectory (after moving libs/)

## Expected Outcomes

After completion:

- Single, clean directory structure at root level
- All imports work without path hacks
- Compatible with Python 3.10-3.12
- Works in Google Colab without modifications
- Works locally with `pip install -e .`
- Ray is optional (install separately if needed)
- Clear, updated documentation