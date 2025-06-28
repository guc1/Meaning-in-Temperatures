#!/bin/bash
#SBATCH -J setup_pytorch_env
#SBATCH -p gpu_a100
#SBATCH --account=vusr98230
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                # High memory for compilations
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup_pytorch_env-%j.out

#################################################################################
##  Creates a Python virtual environment with a user-managed PyTorch stack,    ##
##  a compatible Hugging Face ecosystem, and source-builds FlashAttention.     ##
##  This environment is stable for both single and multi-GPU training/inference.##
#################################################################################

set -euo pipefail

# ─────────────────────────── Configuration ────────────────────────────────────
TARGET_TORCH_VERSION="2.3.1"
TORCH_CUDA_SUFFIX="cu121"
FLASH_ATTN_VERSION="2.5.8"
SYSTEM_CUDA_MODULE="CUDA/12.1.1"
TOOLCHAIN="2023"
VENV_NAME="pt231_fa_qwen_env" # Descriptive name
VENV_PATH="$HOME/envs/${VENV_NAME}"

TRANSFORMERS_VERSION="4.41.2"
TRL_VERSION="0.8.6"
PEFT_VERSION="0.11.1"
ACCELERATE_VERSION="0.29.3"

echo "INFO: Target venv path   : ${VENV_PATH}"
echo "INFO: Target PyTorch     : ${TARGET_TORCH_VERSION}+${TORCH_CUDA_SUFFIX}"
echo "INFO: Target flash-attn  : ${FLASH_ATTN_VERSION}"
echo "INFO: System CUDA module : ${SYSTEM_CUDA_MODULE} (Toolchain: ${TOOLCHAIN})"
echo

# ───────────────────────────── Module loading ────────────────────────────────
echo "INFO: Loading system modules..."
module purge
module load ${TOOLCHAIN}
module load Python/3.11.3-GCCcore-12.3.0
module load ${SYSTEM_CUDA_MODULE} || { echo "ERROR: CUDA module failed to load"; exit 1; }
echo "INFO: Loaded modules:"; module list; echo

# ───────────────────────────── CUDA environment ──────────────────────────────
echo "INFO: Setting up CUDA environment..."
if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -z "${EBROOTCUDA:-}" ]]; then
        echo "ERROR: CUDA_HOME and EBROOTCUDA are not set. Cannot determine CUDA path."
        exit 1
    fi
    export CUDA_HOME="${EBROOTCUDA}"
fi
echo "INFO: Using CUDA_HOME=${CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "INFO: nvcc path: $(which nvcc)"; nvcc --version | head -n1; echo

# ───────────────────────────── Virtual environment ───────────────────────────
echo "INFO: (Re-)creating virtual environment at $VENV_PATH..."
[[ -d "$VENV_PATH" ]] && rm -rf "$VENV_PATH"
python -m venv "$VENV_PATH" || { echo "ERROR: venv creation failed"; exit 1; }
source "$VENV_PATH/bin/activate" || { echo "ERROR: venv activation failed"; exit 1; }
echo "INFO: Active Python: $(which python)"
python -m pip install --upgrade pip wheel setuptools
echo

# ───────────────────────────── Core PyTorch stack ────────────────────────────
echo "INFO: Installing PyTorch stack..."
python -m pip install torch==${TARGET_TORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA_SUFFIX}
python -m pip install torchvision torchaudio # Install separately to avoid version conflicts if needed
python -c "import torch; print(f'PyTorch {torch.__version__}; CUDA compile {torch.version.cuda}; is_available={torch.cuda.is_available()}'); assert torch.cuda.is_available(), 'Torch CUDA not available'" || exit 1
echo

# ───────────────────────────── Helper build tools & Python basics ────────────────────────────
echo "INFO: Installing helper build tools and foundational packages..."
python -m pip install typing_extensions einops ninja cmake
echo

# ───────────────────────────── HF + ecosystem libs ───────────────────────────
echo "INFO: Installing Hugging Face ecosystem libraries..."
python -m pip install "transformers==${TRANSFORMERS_VERSION}" "datasets" \
                    "trl==${TRL_VERSION}" "peft==${PEFT_VERSION}" \
                    "accelerate==${ACCELERATE_VERSION}" "sentencepiece"
python -m pip install "bitsandbytes>=0.43.0"
echo

# ───────────────────────────── Flash‑Attention build (FORCE SOURCE BUILD) ─────
echo "INFO: Forcing flash-attn build from source..."
# For A100 GPUs (Ampere architecture)
export TORCH_CUDA_ARCH_LIST="80"
export CMAKE_CUDA_ARCHITECTURES="80"
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-8}
export FLASH_ATTENTION_SKIP_CUDA_BUILD=0

echo "Attempting to install flash-attn==${FLASH_ATTN_VERSION} by building from source..."
python -m pip install --no-cache-dir --no-binary flash-attn "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
if [ $? -ne 0 ]; then
    echo "ERROR: flash-attn==${FLASH_ATTN_VERSION} build from source FAILED."
    exit 1
else
    echo "INFO: flash-attn==${FLASH_ATTN_VERSION} installed successfully (built from source)."
fi
unset TORCH_CUDA_ARCH_LIST CMAKE_CUDA_ARCHITECTURES MAX_JOBS FLASH_ATTENTION_SKIP_CUDA_BUILD
echo

# ───────────────────────────── Final Sanity Check ────────────────────────────
echo "INFO: Environment setup complete. To use, run: 'source ${VENV_PATH}/bin/activate'"