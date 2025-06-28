import shutil, sys, pathlib

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <source_base_dir> <destination_model_dir>")
    print("  <source_base_dir>: The main output directory from SFTTrainer (e.g., $TMPDIR/qwen_output_...).")
    print("  <destination_model_dir>: The final path where the merged model will be copied.")
    sys.exit(1)

src_base = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])

src_merged = src_base / "merged"

if not src_merged.is_dir():
    print(f"ERROR: Merged model directory not found at '{src_merged}'")
    sys.exit(1)

print(f"Copying merged model from '{src_merged}' to '{dst}'...")
try:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_merged, dst, dirs_exist_ok=True)
    print(f"Model copied successfully to {dst}")
except Exception as e:
    print(f"ERROR: Failed to copy model: {e}")
    sys.exit(1)