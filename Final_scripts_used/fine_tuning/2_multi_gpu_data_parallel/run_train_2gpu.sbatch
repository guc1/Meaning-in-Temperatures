#!/bin/bash
#SBATCH -J qwen_ft_2gpu
#SBATCH -p gpu_a100
#SBATCH --account=vusr98230
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=36
#SBATCH --mem=240G
#SBATCH --time=01:30:00 # 2 GPUs should be faster for same number of effective samples
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/qwen_ft_2gpu-%j.out

echo "################################################################################"
echo "## 2-GPU TRAINING JOB START: $(date)"
echo "## JOB ID: $SLURM_JOB_ID, HOSTNAME: $(hostname)"
echo "################################################################################"
echo

VENV_PATH="$HOME/envs/pt231_fa_qwen_env" # Use the created environment
# Use the same python script, as it's compatible
PYTHON_SCRIPT_TO_RUN="$SLURM_SUBMIT_DIR/fine_tuning/1_single_gpu_baseline/train_qwen25_lora_single_gpu.py"
MERGE_COPY_SCRIPT="$SLURM_SUBMIT_DIR/utils/merge_and_copy.py"

# --- Environment Setup ---
module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1
source "${VENV_PATH}/bin/activate" || { echo "ERROR: venv activation failed"; exit 1; }
export TOKENIZERS_PARALLELISM=false

# --- Data and Model Paths ---
DATASET_SOURCE_PATH="$SLURM_SUBMIT_DIR/data/dataset.chatml.jsonl"
DATASET_LOCAL_PATH="$TMPDIR/dataset_${SLURM_JOB_ID}.jsonl"

MODEL_NAME="Qwen/Qwen2.5-3B"
JOB_BASE_NAME="qwen_output_2gpu_${SLURM_JOB_ID}"
OUTPUT_DIR_LOCAL="$TMPDIR/${JOB_BASE_NAME}"
FINAL_MODEL_DIR_HOME="$HOME/final_models/${MODEL_NAME//\//-}-ft-2gpu-$(date +%Y%m%d-%H%M%S)_${SLURM_JOB_ID}"

# --- Pre-run Checks and Setup ---
cp "${DATASET_SOURCE_PATH}" "${DATASET_LOCAL_PATH}" || { echo "ERROR: Dataset copy failed"; exit 1; }
mkdir -p "$OUTPUT_DIR_LOCAL" || { echo "ERROR: Output dir creation failed"; exit 1; }

# --- Training Parameters ---
TRAIN_STEPS=100
SAVE_STEPS=50
LOGGING_STEPS=10
# For data parallelism, this batch size is PER GPU
PER_DEVICE_BATCH_SIZE=1 
GRAD_ACCUM=16 # Effective global batch size will be 1 * 2 GPUs * 16 = 32

# --- Run Training ---
echo "INFO: Starting training with accelerate launch..."
echo "INFO: Effective global batch size = ${PER_DEVICE_BATCH_SIZE} * 2 * ${GRAD_ACCUM} = $((2 * PER_DEVICE_BATCH_SIZE * GRAD_ACCUM))"

# accelerate will automatically detect the 2 GPUs and launch 2 processes
accelerate launch "$PYTHON_SCRIPT_TO_RUN" \
       --model_name_or_path "$MODEL_NAME" \
       --train_file "$DATASET_LOCAL_PATH" \
       --out "$OUTPUT_DIR_LOCAL" \
       --steps $TRAIN_STEPS \
       --save_steps $SAVE_STEPS \
       --logging_steps $LOGGING_STEPS \
       --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
       --grad_accum_steps $GRAD_ACCUM \
       --block 8192 \
       --lr 2e-5 \
       --lora_r 32 \
       --lora_alpha 64 &
TRAIN_PID=$!
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?
echo "INFO: Training script finished with exit code: $TRAIN_EXIT_CODE"

# --- Copy Final Merged Model ---
if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
    python "$MERGE_COPY_SCRIPT" "$OUTPUT_DIR_LOCAL" "$FINAL_MODEL_DIR_HOME"
fi
echo "################################################################################"
echo "## 2-GPU TRAINING JOB END: $(date)"
echo "################################################################################"