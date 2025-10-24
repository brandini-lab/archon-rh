@echo off
REM Quick launch script for GPU training on Windows
REM Works with WSL2 CUDA or native Windows CUDA

echo ============================================
echo ARCHON-RH GPU Training Launcher (Windows)
echo ============================================

REM Check if GPU is available
where nvidia-smi >nul 2>nul
if %errorlevel% == 0 (
    echo ✓ GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
) else (
    echo ⚠ No GPU detected - will train on CPU (slow!)
)

echo.
echo Checking environment...
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')" 2>nul
if %errorlevel% neq 0 (
    echo Installing PyTorch with CUDA...
    pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
)

REM Install dependencies
python -c "import transformers" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo Installing dependencies...
    pip install -q -e .
)

echo.
echo Select training mode:
echo 1^) SFT (Supervised Fine-tuning) - Small GPU
echo 2^) SFT Production - Large GPU
echo 3^) RL PPO - Reinforcement Learning
echo 4^) FunSearch - Conjecture Discovery
echo 5^) Numerics - Zero Verification
set /p choice="Enter choice [1-5]: "

if "%choice%"=="1" (
    echo Running SFT training (small GPU config)...
    python prover/train/sft_hf.py orchestration/configs/colab_sft_gpu.yaml
) else if "%choice%"=="2" (
    echo Running SFT production (large GPU config)...
    python prover/train/sft_hf.py orchestration/configs/gpu_production.yaml
) else if "%choice%"=="3" (
    echo Running RL PPO training...
    python prover/rl/ppo_loop.py orchestration/configs/colab_rl_gpu.yaml
) else if "%choice%"=="4" (
    echo Running FunSearch...
    python conjecture/funsearch/funsearch_loop.py --config orchestration/configs/funsearch_loop.yaml
) else if "%choice%"=="5" (
    echo Running numerical zero verification...
    python numerics/zeros/run_zero_checks.py --config orchestration/configs/numeric_verify.yaml
) else (
    echo Invalid choice
    exit /b 1
)

echo.
echo ============================================
echo Training complete! Check artifacts/ directory
echo ============================================
dir /s artifacts\checkpoints\ 2>nul || echo No checkpoints found
pause

