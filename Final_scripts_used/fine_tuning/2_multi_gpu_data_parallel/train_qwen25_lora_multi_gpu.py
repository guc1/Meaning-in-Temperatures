# The `train_qwen25_lora_single_gpu.py` script provided above
# is already robust enough to be used for multi-GPU data parallel
# training when launched with `accelerate launch`.
# The key features that make it compatible are:
# 1. It uses `device_map={"": torch.cuda.current_device()}`.
#    When `accelerate launch` runs, it sets `CUDA_VISIBLE_DEVICES` for each process,
#    so `torch.cuda.current_device()` correctly resolves to 0 for the first process
#    (which sees physical GPU 0), 0 for the second process (which sees physical GPU 1), etc.
# 2. `transformers.Trainer` is aware of the distributed environment set up by Accelerate
#    and handles DistributedDataParallel (DDP) wrapping and gradient synchronization automatically.
# 3. For saving models, the Trainer ensures only the main process (rank 0) writes the final files.
#
# Therefore, you can simply reuse `train_qwen25_lora_single_gpu.py`
# or copy it to this new directory for organizational purposes.