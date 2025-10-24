# ARCHON-RH: Riemann Hypothesis Research Checklist

This is a comprehensive roadmap for using ARCHON-RH to work toward proving or advancing research on the Riemann Hypothesis (RH).

## ⚠️ Reality Check

The Riemann Hypothesis is the most famous unsolved problem in mathematics (since 1859). This system provides **tools and infrastructure** for:
- Automated theorem proving in Lean
- Conjecture discovery via evolutionary search
- Numerical verification of zero locations
- Reinforcement learning for tactic exploration

**This does NOT guarantee a proof**, but provides a systematic, verifiable research environment.

---

## Phase 1: Foundation & Infrastructure Setup ✅

### 1.1 Environment Setup
- [ ] **Local Development**
  - [ ] Python 3.10+ installed
  - [ ] CUDA-capable GPU (16GB+ VRAM recommended)
  - [ ] Run `make dev` to create virtual environment
  - [ ] Run `make test` to verify all tests pass
  - [ ] Verify Lean 4 toolchain installed

- [ ] **Google Colab Setup** (Alternative)
  - [ ] Open `colab/ARCHON_RH_Complete_Setup.ipynb`
  - [ ] Enable GPU runtime (T4/V100/A100)
  - [ ] Clone repository with your credentials
  - [ ] Run all setup cells

### 1.2 Core Component Verification
- [ ] **Prover Module**
  - [ ] Dataset builder works (`prover/datasets/build_mathlib_dataset.py`)
  - [ ] SFT training runs (`pytest -k test_sft_trainer`)
  - [ ] RL environment connects to Lean (`pytest -k test_datasets`)

- [ ] **Formal System**
  - [ ] LeanDojo bridge operational (`pytest -k toy_lemma`)
  - [ ] Mathlib imports work
  - [ ] Can prove trivial lemmas

- [ ] **Numerics Module**
  - [ ] Interval arithmetic bindings load (`numerics/arb_bindings/`)
  - [ ] Can verify zeros numerically (`pytest -k numeric_cert`)
  - [ ] Certificate generation works

- [ ] **Safety & Verification**
  - [ ] Reward signing functional
  - [ ] Policy enforcement active
  - [ ] Audit logs generated

---

## Phase 2: Dataset & Knowledge Base 📚

### 2.1 Mathematical Dataset
- [ ] **Analytic Number Theory**
  - [ ] Extract Mathlib lemmas on Dirichlet L-functions
  - [ ] Collect theorems on zeta function properties
  - [ ] Gather results on prime number theorem
  - [ ] Index lemmas on complex analysis

- [ ] **RH-Specific Theorems**
  - [ ] Riemann-von Mangoldt formula
  - [ ] Zero-free regions
  - [ ] Functional equation of ζ(s)
  - [ ] Critical line and critical strip
  - [ ] Explicit formulas
  - [ ] Equivalences to RH (e.g., Li criterion)

- [ ] **Supporting Lemmas**
  - [ ] Complex integration theory
  - [ ] Fourier analysis on R and T
  - [ ] Mellin transforms
  - [ ] Contour integration techniques

### 2.2 Build Custom Datasets
```bash
# Extract RH-relevant tactics from Mathlib
python prover/datasets/build_mathlib_dataset.py --filter="Riemann|Zeta|Prime|Dirichlet" --limit=10000

# Create RH-specific training set
python prover/datasets/builders.py --domain="analytic_number_theory"
```

- [ ] Dataset: ≥1000 tactic examples on analytic number theory
- [ ] Dataset: ≥500 examples on complex analysis
- [ ] Dataset: ≥100 examples directly mentioning zeta or L-functions

---

## Phase 3: Supervised Fine-Tuning 🧠

### 3.1 Train Base Prover
```bash
# Start with tiny config for testing
python prover/train/sft_hf.py --config orchestration/configs/sft_tiny.yaml

# Scale to production
python prover/train/sft_hf.py --config orchestration/configs/sft_hf_prod.yaml
```

- [ ] **Tiny Model** (debugging)
  - [ ] Trains without errors
  - [ ] Can predict simple tactics (rfl, simp, trivial)
  - [ ] Loss decreases monotonically

- [ ] **Medium Model** (development)
  - [ ] 12 layers, 768 embedding
  - [ ] Trains on 5K examples
  - [ ] Can handle complex_integration tactics

- [ ] **Large Model** (production)
  - [ ] 24+ layers, 1024+ embedding
  - [ ] Trains on full dataset (10K+ examples)
  - [ ] Achieves <1.5 perplexity on validation set
  - [ ] Can suggest non-trivial analytic_NT tactics

### 3.2 Evaluation Metrics
- [ ] Tactic prediction accuracy >60%
- [ ] Pass@1 on test theorems >30%
- [ ] Pass@10 on test theorems >60%
- [ ] Can close at least 10 real Mathlib gaps

---

## Phase 4: Reinforcement Learning & Self-Play 🎮

### 4.1 PPO Training on Lean Environment
```bash
# Tiny test
python prover/rl/ppo_loop.py --config orchestration/configs/rl_tiny.yaml

# Production self-play
python prover/rl/run_selfplay.py --config orchestration/configs/rl_ppo_prod.yaml
```

- [ ] **Mock Environment** (testing)
  - [ ] PPO converges on toy proofs
  - [ ] Policy improves over random baseline

- [ ] **Lean Environment** (production)
  - [ ] Agent interacts with real Lean via LeanDojo
  - [ ] Explores proof search trees
  - [ ] Discovers novel tactic sequences
  - [ ] Reward shaping: length penalty + proof bonus

### 4.2 RL Success Criteria
- [ ] Solves 10+ Mathlib "sorry" gaps automatically
- [ ] Average proof length <20 tactics
- [ ] Discovers at least 3 non-standard proof patterns
- [ ] Policy trained for ≥100K episodes

---

## Phase 5: Numerical Verification 🔢

### 5.1 Zero Computation
```bash
# Run zero checks with interval arithmetic
python numerics/zeros/run_zero_checks.py --config orchestration/configs/numeric_verify.yaml
```

- [ ] **Critical Line Verification**
  - [ ] Verify first 10^5 zeros on σ=1/2
  - [ ] Compute zeros to height T=10^6 (or higher)
  - [ ] Generate rigorous certificates for each zero

- [ ] **Zero-Free Regions**
  - [ ] Verify classical zero-free region (σ > 1 - 1/log t)
  - [ ] Check improved bounds from recent literature

### 5.2 Numerical Support for Conjectures
- [ ] Li criterion evaluation for N=10^6 terms
- [ ] Sign checks on ζ derivatives
- [ ] Argmax computations for S(t)

---

## Phase 6: Conjecture Discovery (FunSearch) 🔬

### 6.1 Automated Conjecture Generation
```bash
# Run FunSearch loop
python conjecture/funsearch/funsearch_loop.py --config orchestration/configs/funsearch_loop.yaml
```

- [ ] **Objective: Li Criterion**
  - [ ] Generate candidate kernels for Li's criterion
  - [ ] Evaluate numerically for large N
  - [ ] Evolve population over 1000+ generations
  - [ ] Extract top-10 scoring functions

- [ ] **Custom Objectives**
  - [ ] Zero density estimates
  - [ ] Lindelöf hypothesis proxies
  - [ ] Alternative equivalent formulations

### 6.2 Conjecture Evaluation
- [ ] At least 5 conjectures generated
- [ ] Each numerically verified for N≥10^4
- [ ] At least 1 survives adversarial testing
- [ ] Export to Lean for formal statement

---

## Phase 7: Formal Verification in Lean 📜

### 7.1 Formalize Generated Conjectures
```lean
-- In formal/leanlib/ArchonRiemann/

-- Example: New Li-type criterion
theorem archon_li_criterion (n : ℕ) : 
  RiemannHypothesis ↔ ∀ k ≤ n, archon_generated_sum k ≥ 0 := by
  sorry  -- Goal: remove this sorry
```

- [ ] Formalize top conjecture from FunSearch
- [ ] State dependencies clearly
- [ ] Link to existing Mathlib results
- [ ] Document assumptions

### 7.2 Automated Proof Attempts
```bash
# Use trained RL agent to attempt proof
python prover/inference/serve_vllm.py --backend hf --prompts "theorem archon_li_criterion"
```

- [ ] Agent generates proof attempts
- [ ] At least 10 partial proofs
- [ ] At least 1 complete auxiliary lemma
- [ ] Manual review and completion

### 7.3 Human-in-the-Loop
- [ ] Export proof states to mathematicians
- [ ] Manually complete gaps where agent fails
- [ ] Use agent suggestions as guidance
- [ ] Iterate on proof strategy

---

## Phase 8: Integration & Iteration 🔄

### 8.1 Feedback Loop
```
Numerical Evidence → FunSearch → Conjecture → Lean Formalization
                                                     ↓
                                              Proof Attempt (RL)
                                                     ↓
                                              Manual Review → REPEAT
```

- [ ] Run complete pipeline end-to-end
- [ ] Identify bottlenecks
- [ ] Retrain models with new data
- [ ] Refine conjectures based on proof attempts

### 8.2 Scaling
- [ ] Multi-GPU training for larger models
- [ ] Distributed RL with Ray
- [ ] Kubernetes deployment for long runs
- [ ] Artifact versioning and reproducibility

---

## Phase 9: Safety & Verification ✅

### 9.1 Reward Verification
```bash
# Sign all critical events
python safety/audits/sign_rewards.py

# Verify signatures before optimization
python safety/policies/reward_verifier.py
```

- [ ] All training rewards signed
- [ ] Audit log complete and immutable
- [ ] No unsigned rewards consumed by optimizer
- [ ] Merkle chain verification passes

### 9.2 Sandboxing
- [ ] Model execution in seccomp sandbox
- [ ] Network egress blocked during inference
- [ ] Filesystem access restricted
- [ ] Resource limits enforced

---

## Phase 10: Publication & Validation 📄

### 10.1 Documentation
- [ ] **Technical Report**
  - [ ] System architecture
  - [ ] Training methodology
  - [ ] Generated conjectures
  - [ ] Numerical evidence
  - [ ] Partial proofs

- [ ] **Lean Formalization**
  - [ ] Complete ArchonRiemann library
  - [ ] All conjectures stated formally
  - [ ] Auxiliary lemmas proved
  - [ ] Integrated with Mathlib

### 10.2 Independent Verification
- [ ] Share artifacts with mathematicians
- [ ] Submit to arXiv
- [ ] Lean code reviewed by experts
- [ ] Numerical certificates checked independently

### 10.3 Open Science
- [ ] Publish all code (already done if public repo)
- [ ] Share trained models
- [ ] Release datasets
- [ ] Document reproducibility steps

---

## Success Criteria 🎯

### Minimum Viable Contribution
- [ ] 5+ new formalized lemmas in ArchonRiemann library
- [ ] 3+ automated proofs of existing "sorry" gaps
- [ ] 1+ novel conjecture with numerical support
- [ ] Numerical verification of 10^5 zeros with certificates

### Significant Advancement
- [ ] 20+ new formalized theorems
- [ ] 10+ automated proofs
- [ ] 5+ novel conjectures, each with strong evidence
- [ ] At least 1 conjecture partially proved
- [ ] Numerical verification to height T=10^7

### Major Breakthrough (Ambitious)
- [ ] Complete proof of a non-trivial equivalent to RH
- [ ] Novel approach published in peer-reviewed journal
- [ ] Adopted by broader Lean mathematics community
- [ ] Advances state-of-the-art in automated theorem proving

---

## Timeline Estimate

| Phase | Optimistic | Realistic | Conservative |
|-------|------------|-----------|--------------|
| 1-2: Setup & Data | 1 week | 2 weeks | 1 month |
| 3: SFT Training | 3 days | 1 week | 2 weeks |
| 4: RL Training | 1 week | 2 weeks | 1 month |
| 5: Numerics | 3 days | 1 week | 2 weeks |
| 6: FunSearch | 1 week | 2 weeks | 1 month |
| 7: Formalization | 2 weeks | 1 month | 3 months |
| 8: Integration | 1 week | 2 weeks | 1 month |
| 9-10: Validation | 1 month | 3 months | 6 months |
| **TOTAL** | **2-3 months** | **4-6 months** | **12-18 months** |

**Note:** These timelines assume full-time work by an experienced team with strong ML and mathematics background.

---

## Resources & References

### Papers
- Riemann (1859): Original paper on ζ(s)
- Li (1997): Criterion equivalent to RH
- Odlyzko (2001): Computational verification of zeros
- Booker & Trudgian (2021): Recent computational records

### Lean Libraries
- Mathlib: `NumberTheory.ZetaFunction`
- Mathlib: `Analysis.Complex.Basic`
- LeanDojo: Automated theorem proving interface

### ARCHON-RH Modules
- `formal/leanlib/ArchonRiemann/`: RH-specific Lean code
- `numerics/`: Interval arithmetic and zero checks
- `conjecture/`: FunSearch loop
- `prover/`: ML-based theorem prover

---

## Contact & Collaboration

For questions, contributions, or collaboration:
1. Open GitHub issue
2. Review `CONTRIBUTING.md` (if exists)
3. Join research discussion forums
4. Share results via arXiv preprints

---

**Good luck! May your zeros all lie on the critical line. 🎲**

