#!/bin/bash
#SBATCH -J qwen_eval_inference
#SBATCH -p gpu_a100
#SBATCH --account=vusr98230
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/qwen_eval-%j.out

echo "################################################################################"
echo "## INFERENCE JOB START: $(date)"
echo "################################################################################"
echo

VENV_PATH="$HOME/envs/pt231_fa_qwen_env"
PYTHON_SCRIPT_TO_RUN="$SLURM_SUBMIT_DIR/inference/run_model_inference.py"

# --- Environment Setup ---
module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1
source "${VENV_PATH}/bin/activate" || exit 1
export TOKENIZERS_PARALLELISM=false

# --- Configuration & Paths ---
# !!! IMPORTANT: UPDATE THIS PATH TO THE MERGED MODEL YOU WANT TO TEST !!!
FT_MODEL_PATH="/home/ybloetjes/final_models/qwen2.5-3B-ft-2gpu-..." # Example

TRUTH_DATA_FILE_SOURCE="$SLURM_SUBMIT_DIR/data/prompts_and_truths.jsonl"
TRUTH_DATA_FILE_LOCAL="$TMPDIR/prompts_and_truths_${SLURM_JOB_ID}.jsonl"

# --- Inference Parameters to Experiment With ---
INTERNAL_INFERENCE_BATCH_SIZE=12        # Optimal batch size found for 3B/7B on A100-40GB
COMPILE_INFERENCE_MODEL="--compile_model" # Set to "--compile_model" or ""
MONTH_SAMPLING_INTERVAL=1               # 1 for all, N for every Nth month

# --- Dynamic Naming and Setup ---
if [ -n "$COMPILE_INFERENCE_MODEL" ]; then COMPILE_SUFFIX="_compiled"; else COMPILE_SUFFIX=""; fi
JOB_BASE_NAME="infer_bs${INTERNAL_INFERENCE_BATCH_SIZE}${COMPILE_SUFFIX}_interval${MONTH_SAMPLING_INTERVAL}_${SLURM_JOB_ID}"
RAW_PREDICTIONS_FILE_PERSISTENT="$SLURM_SUBMIT_DIR/results/${JOB_BASE_NAME}.jsonl"
NVIDIA_SMI_CSV_PERSISTENT="$SLURM_SUBMIT_DIR/results/nvidia_smi_logs/nvidia_smi_${JOB_BASE_NAME}.csv"

echo "INFO: Model Path: ${FT_MODEL_PATH}"
echo "INFO: Batch Size: ${INTERNAL_INFERENCE_BATCH_SIZE}"
echo "INFO: Compile Mode: ${COMPILE_INFERENCE_MODEL:-"Disabled"}"

# --- Pre-run Setup (Copy data, make dirs, start monitoring) ---
cp "${TRUTH_DATA_FILE_SOURCE}" "${TRUTH_DATA_FILE_LOCAL}" || exit 1
mkdir -p "$(dirname ${RAW_PREDICTIONS_FILE_PERSISTENT})" "$(dirname ${NVIDIA_SMI_CSV_PERSISTENT})" || exit 1
NVIDIA_SMI_LOG_FILE_TMP="$TMPDIR/nvidia_smi_tmp_${SLURM_JOB_ID}.csv"
echo "timestamp,gpu_idx,utilization.gpu [%],utilization.memory [%],memory.used [MiB],power.draw [W]" > "${NVIDIA_SMI_LOG_FILE_TMP}"
( while true; do nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits >> "${NVIDIA_SMI_LOG_FILE_TMP}"; sleep 1; done ) &
NVIDIA_SMI_PID=$!
trap 'kill $NVIDIA_SMI_PID; cp ${NVIDIA_SMI_LOG_FILE_TMP} ${NVIDIA_SMI_CSV_PERSISTENT};' EXIT

# --- Run Inference ---
echo -e "\n--- RUNNING INFERENCE ---"
python "$PYTHON_SCRIPT_TO_RUN" \
    --model_path "$FT_MODEL_PATH" \
    --prompts_file "$TRUTH_DATA_FILE_LOCAL" \
    --predictions_output_file "$RAW_PREDICTIONS_FILE_PERSISTENT" \
    --month_interval ${MONTH_SAMPLING_INTERVAL} \
    --max_new_tokens 128 \
    --block 8192 \
    --batch_size ${INTERNAL_INFERENCE_BATCH_SIZE} \
    --do_sample \
    --temperature 0.6 \
    --top_p 0.9 \
    --seed 42 \
    ${COMPILE_INFERENCE_MODEL}

echo "################################################################################"
echo "## INFERENCE JOB END: $(date)"
echo "################################################################################"