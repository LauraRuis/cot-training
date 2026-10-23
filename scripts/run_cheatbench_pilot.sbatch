#!/bin/bash
# Inference only, on the prepared task; the user submits this job.
#SBATCH --job-name=cheatbench_qwen4b
#SBATCH --account=lingo
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/data/scratch/lruis/cheatbench_qwen4b-%j.log
#SBATCH --error=/data/scratch/lruis/cheatbench_qwen4b-%j.log
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch so downloads and inference run on a compute node.'
    exit 2
fi

cd "${PILOT_CODE_ROOT:-/data/scratch/lruis/cot-training}"
source /data/scratch/lruis/LLMScience/llm_science_eval_env/bin/activate
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export TRITON_CACHE_DIR=/data/scratch/lruis/.triton
export TRITON_HOME=/data/scratch/lruis/.triton
export XDG_CACHE_HOME=/data/scratch/lruis/.cache
PILOT_WORK_ROOT=${PILOT_WORK_ROOT:-/data/scratch/lruis/cot-training/cheatbench_work}
test -f "$PILOT_WORK_ROOT/prepared/READY" || { echo 'Run the preparation job successfully first.'; exit 1; }
pilot_out="$PILOT_WORK_ROOT/runs/$SLURM_JOB_ID"
mkdir -p "$pilot_out"
export APPTAINER_CACHEDIR="$PILOT_WORK_ROOT/apptainer_cache"
pilot_tmp=$(mktemp -d /tmp/cheatbench-inference.XXXXXX)
export APPTAINER_TMPDIR="$pilot_tmp"
pilot_server_pid=""
cleanup() {
    if [[ -n "$pilot_server_pid" ]]; then
        kill "$pilot_server_pid" 2>/dev/null || true
        wait "$pilot_server_pid" 2>/dev/null || true
    fi
    rm -rf -- "$pilot_tmp"
}
trap cleanup EXIT

# Recheck isolation before loading/downloading weights.
python - "$PILOT_WORK_ROOT/prepared/task.sif" "$pilot_out" <<'PY'
import sys
from pathlib import Path
from contract.cheatbench.pilot import check_runtime
check_runtime(Path(sys.argv[1]), Path(sys.argv[2]))
PY
python - <<'PY' > "$pilot_out/packages.txt"
from importlib.metadata import distributions
print("\n".join(sorted(
    f"{dist.metadata['Name']}=={dist.version}"
    for dist in distributions() if dist.metadata['Name']
)))
PY
nvidia-smi > "$pilot_out/gpu.txt"
git rev-parse HEAD > "$pilot_out/code-commit.txt"
git diff HEAD > "$pilot_out/code-diff.patch"
# Include untracked pilot code too; a git diff alone would omit it.
cp -r contract/cheatbench "$pilot_out/code"
cp contract/llm.py "$pilot_out/llm.py"
cp scripts/run_cheatbench_pilot.sbatch "$pilot_out/launch.sbatch"

pilot_port=$((20000 + SLURM_JOB_ID % 20000))
pilot_model=Qwen/Qwen3-4B
# Pin the resolved HF revision once, for both tokenizer and server; downloads
# happen inside this allocated job using the user's existing HF cache.
pilot_revision=$(python -c 'from huggingface_hub import model_info; print(model_info("Qwen/Qwen3-4B").sha)')
echo "$pilot_revision" > "$pilot_out/model-revision.txt"
vllm serve "$pilot_model" --revision "$pilot_revision" --tokenizer-revision "$pilot_revision" \
    --host 127.0.0.1 --port "$pilot_port" --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --max-num-seqs 2 > "$pilot_out/vllm.log" 2>&1 &
pilot_server_pid=$!
pilot_ready=0
for ((i=0; i<180; i++)); do
    kill -0 "$pilot_server_pid" 2>/dev/null || { echo "vLLM failed: $pilot_out/vllm.log"; exit 1; }
    if curl --max-time 2 -fsS "http://127.0.0.1:$pilot_port/health" >/dev/null 2>&1; then
        pilot_ready=1
        break
    fi
    sleep 10
done
[[ "$pilot_ready" == 1 ]] || { echo "vLLM startup timed out: $pilot_out/vllm.log"; exit 1; }
python -m contract.cheatbench.pilot run --work-root "$PILOT_WORK_ROOT" \
    --out "$pilot_out/episodes" --model "$pilot_model" --model-revision "$pilot_revision" \
    --url "http://127.0.0.1:$pilot_port/v1" "$@"
