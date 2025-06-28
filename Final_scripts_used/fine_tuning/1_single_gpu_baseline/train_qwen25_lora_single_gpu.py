#!/usr/bin/env python
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, set_seed
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import argparse
import os
import logging
import signal
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Argument Parsing ---
p = argparse.ArgumentParser(description="Fine-tune Qwen model with LoRA on a single GPU.")
p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-3B", help="Base model name or path.")
p.add_argument("--train_file", type=str, required=True, help="Path to the training data JSONL file.")
p.add_argument("--out", type=str, required=True, help="Output directory for checkpoints and logs.")
p.add_argument("--resume", default=None, type=str, help="Path to checkpoint to resume from.")
p.add_argument("--steps", type=int, default=100, help="Number of training steps.")
p.add_argument("--block", type=int, default=8192, help="Max sequence length (block size).")
p.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
p.add_argument("--grad_accum_steps", type=int, default=16, help="Gradient accumulation steps.")
p.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
p.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X steps.")
p.add_argument("--logging_steps", type=int, default=10, help="Log every X steps.")
p.add_argument("--lora_r", type=int, default=32, help="LoRA r parameter.")
p.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha parameter.")
p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
args = p.parse_args()

# --- Basic Setup ---
os.environ["PYTHONUNBUFFERED"] = "1"
set_seed(args.seed)

logger.info(f"Script arguments: {args}")
logger.info(f"Torch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# --- Tokenizer ---
logger.info(f"Loading tokenizer: {args.model_name_or_path}")
tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, padding_side="left")
if tok.pad_token is None:
    logger.warning("Tokenizer has no pad_token. Setting to eos_token.")
    tok.pad_token = tok.eos_token
logger.info(f"Tokenizer pad_token: {tok.pad_token} (ID: {tok.pad_token_id})")

# --- Model with QLoRA ---
logger.info(f"Loading base model: {args.model_name_or_path} with QLoRA")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model_dtype = torch.bfloat16
logger.info(f"Loading model with torch_dtype: {model_dtype} and Flash Attention 2")

base = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    quantization_config=quantization_config,
    torch_dtype=model_dtype,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    device_map={"": torch.cuda.current_device()}
)
logger.info("Base model loaded.")
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
logger.info("Model prepared for k-bit training.")

# --- LoRA Configuration ---
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
logger.info(f"LoRA target modules: {lora_target_modules}")
lora_cfg = LoraConfig(
    r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_target_modules,
    lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(base, lora_cfg)
logger.info("PEFT model created.")
model.print_trainable_parameters()

# --- Dataset ---
logger.info(f"Loading dataset from: {args.train_file}")
try:
    ds = load_dataset("json", data_files=args.train_file, split="train")
    logger.info(f"Dataset loaded. Number of examples: {len(ds)}")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}", exc_info=True)
    sys.exit(1)

# --- Trainer Setup ---
logger.info("Initializing SFTTrainer...")
training_args_obj = TrainingArguments(
    output_dir=args.out,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.grad_accum_steps,
    max_steps=args.steps,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=args.logging_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=3,
    optim="paged_adamw_8bit",
    disable_tqdm=False,
    resume_from_checkpoint=args.resume,
    report_to=[],
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    seed=args.seed,
    bf16=(model_dtype == torch.bfloat16),
    fp16=(model_dtype == torch.float16),
)

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds,
    dataset_kwargs={"skip_prepare_dataset": False},
    max_seq_length=args.block, packing=True, args=training_args_obj,
)
logger.info("SFTTrainer initialized.")

# --- Signal Handler ---
def handle_sigusr1(sig, frame):
    logger.warning("Signal USR1 received, saving checkpoint and exiting...")
    trainer.save_model(os.path.join(args.out, "checkpoint-sigusr1"))
    trainer.save_state()
    sys.exit(0)
signal.signal(signal.SIGUSR1, handle_sigusr1)

# --- Training ---
logger.info(f"Starting training for {args.steps} steps...")
try:
    trainer.train(resume_from_checkpoint=args.resume)
except Exception as e:
    logger.error(f"Exception during training: {e}", exc_info=True)
    try:
        trainer.save_model(os.path.join(args.out, "checkpoint-exception"))
        trainer.save_state()
    except Exception as save_e:
        logger.error(f"Could not save state after exception: {save_e}", exc_info=True)
    sys.exit(1)

# --- Merging LoRA and Saving ---
logger.info("Merging LoRA adapter into base model...")
try:
    final_checkpoint = trainer.state.best_model_checkpoint or os.path.join(args.out, f"checkpoint-{args.steps}")
    if not os.path.exists(final_checkpoint): # Fallback to finding latest if best/last isn't clear
        dirs = [os.path.join(args.out, d) for d in os.listdir(args.out) if d.startswith("checkpoint-")]
        if dirs:
            final_checkpoint = max(dirs, key=os.path.getmtime)
        else:
            raise ValueError(f"No checkpoint found in {args.out} to merge.")

    logger.info(f"Using adapter from checkpoint: {final_checkpoint}")

    base_model_for_merge = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=model_dtype, trust_remote_code=True, device_map="cpu"
    )
    peft_model = PeftModel.from_pretrained(base_model_for_merge, final_checkpoint)
    merged_model = peft_model.merge_and_unload()

    merged_model_save_path = os.path.join(args.out, "merged")
    logger.info(f"Saving merged model to: {merged_model_save_path}")
    merged_model.save_pretrained(merged_model_save_path)
    tok.save_pretrained(merged_model_save_path)
    logger.info("Merged model and tokenizer saved.")
except Exception as e:
    logger.error(f"Error during model merging and saving: {e}", exc_info=True)
    sys.exit(1)

logger.info("Fine-tuning and merging process finished successfully.")