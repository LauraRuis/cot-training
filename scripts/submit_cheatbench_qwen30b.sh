#!/bin/bash
# Run by the user on the cluster login node. Downloads happen in the allocation.
set -euo pipefail
cd "${PILOT_CODE_ROOT:-/data/scratch/lruis/cot-training}"
export PILOT_MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507
export PILOT_MODEL_REVISION=144afc2f379b542fdd4e85a1fcd5e1f79112d95d
export PILOT_VENV="${PILOT_VENV:-/data/scratch/lruis/LLMScience/llm_science_eval_env}"
export PILOT_WATCHFILES_OVERLAY="${PILOT_WATCHFILES_OVERLAY:-/data/scratch/lruis/cot-training/cheatbench_work/server_overlays/2820742}"
# One BF16 model, one concurrent sequence, no CUDA graph allocation.
export PILOT_GPU_MEMORY=0.90
export PILOT_MAX_SEQUENCES=1
export PILOT_ENFORCE_EAGER=1
export PILOT_MAX_BATCHED_TOKENS=2048
export PILOT_STARTUP_CHECKS=360
# Separate local tunnel from the ongoing 4B run on port 8000.
export PILOT_LOCAL_PORT=8001
sbatch --job-name=cheatbench_30b --time=06:00:00 --mem=128G \
    scripts/serve_cheatbench_qwen.sbatch "$@"
