This folder collects the code that produced the thesis results.

Subdirectories:
- `environment_setup` – SLURM script to build the Python environment with PyTorch, HuggingFace libraries and FlashAttention.
- `fine_tuning` – scripts for training Qwen models using QLoRA on one or multiple GPUs.
- `inference` – job script for running inference on the trained models.
- `ResultsBenchmark` – evaluation outputs and helper scripts for the OpenTas benchmark.
- `results` – hardware logs produced during inference.
- `utils` – helper utilities, e.g. merging LoRA adapters.

