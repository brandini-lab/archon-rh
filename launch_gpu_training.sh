#!/bin/bash
# Quick launch script for GPU training on any platform
# Works on: Colab, Lambda Labs, RunPod, Vast.ai, local GPU

set -e  # Exit on error

echo "============================================"
echo "ARCHON-RH GPU Training Launcher"
echo "============================================"

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No GPU detected - will train on CPU (slow!)"
fi

# Check Python and PyTorch
echo ""
echo "Checking environment..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')"

# Install dependencies if needed
if ! python3 -c "import transformers" 2>/dev/null; then
    echo ""
    echo "Installing dependencies..."
    pip install -q -e .
fi

# Choose training mode
echo ""
echo "Select training mode:"
echo "1) SFT (Supervised Fine-tuning) - Small GPU (T4, RTX 3090)"
echo "2) SFT Production - Large GPU (A100, V100)"
echo "3) RL PPO - Reinforcement Learning"
echo "4) FunSearch - Conjecture Discovery (CPU-bound)"
echo "5) Numerics - Zero Verification (CPU-bound)"
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo "Running SFT training (small GPU config)..."
        python prover/train/sft_hf.py orchestration/configs/colab_sft_gpu.yaml
        ;;
    2)
        echo "Running SFT production (large GPU config)..."
        python prover/train/sft_hf.py orchestration/configs/gpu_production.yaml
        ;;
    3)
        echo "Running RL PPO training..."
        python prover/rl/ppo_loop.py orchestration/configs/colab_rl_gpu.yaml
        ;;
    4)
        echo "Running FunSearch..."
        python conjecture/funsearch/funsearch_loop.py --config orchestration/configs/funsearch_loop.yaml
        ;;
    5)
        echo "Running numerical zero verification..."
        python numerics/zeros/run_zero_checks.py --config orchestration/configs/numeric_verify.yaml
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Training complete! Check artifacts/ directory"
echo "============================================"
ls -lh artifacts/checkpoints/ 2>/dev/null || echo "No checkpoints found"

