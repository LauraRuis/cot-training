#!/bin/bash
# Model server only. The user submits this and establishes the SSH tunnel.
#SBATCH --job-name=cheatbench_server
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/data/scratch/lruis/cheatbench_server-%j.log
#SBATCH --error=/data/scratch/lruis/cheatbench_server-%j.log
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch so downloads and inference run on a compute node.'
    exit 2
fi

cd "${PILOT_CODE_ROOT:-/data/scratch/lruis/cot-training}"
source "${PILOT_VENV:-/data/scratch/lruis/LLMScience/llm_science_eval_env}/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${PILOT_WATCHFILES_OVERLAY:-}" ]]; then
    python -m contract.cheatbench.watchfiles_overlay "$PILOT_WATCHFILES_OVERLAY"
    export PYTHONPATH="$PILOT_WATCHFILES_OVERLAY/site-packages${PYTHONPATH:+:$PYTHONPATH}"
fi
python - <<'PY'
import sys
from importlib.util import find_spec
print(f"Server Python: {sys.executable}", flush=True)
if find_spec("vllm") is None:
    raise SystemExit("vLLM is not installed in this environment. Set PILOT_VENV to an environment containing vLLM.")
PY
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export TRITON_CACHE_DIR=/data/scratch/lruis/.triton
export TRITON_HOME=/data/scratch/lruis/.triton
export XDG_CACHE_HOME=/data/scratch/lruis/.cache
pilot_out="${PILOT_WORK_ROOT:-/data/scratch/lruis/cot-training/cheatbench_work}/servers/$SLURM_JOB_ID"
mkdir -p "$pilot_out"
if [[ -n "${PILOT_WATCHFILES_OVERLAY:-}" ]]; then
    cp "$PILOT_WATCHFILES_OVERLAY/manifest.json" "$pilot_out/watchfiles-overlay.json"
fi
python - <<'PY' > "$pilot_out/packages.txt"
from importlib.metadata import distributions
print("\n".join(sorted(
    f"{dist.metadata['Name']}=={dist.version}"
    for dist in distributions() if dist.metadata['Name']
)))
PY
nvidia-smi > "$pilot_out/gpu.txt"
cp scripts/serve_cheatbench_qwen.sbatch "$pilot_out/launch.sbatch"

pilot_port=$((20000 + SLURM_JOB_ID % 20000))
pilot_model="${PILOT_MODEL:-Qwen/Qwen3-4B}"
pilot_gpu_memory="${PILOT_GPU_MEMORY:-0.85}"
pilot_max_sequences="${PILOT_MAX_SEQUENCES:-2}"
pilot_local_port="${PILOT_LOCAL_PORT:-8000}"
pilot_extra_args=()
if [[ "${PILOT_ENFORCE_EAGER:-0}" == 1 ]]; then
    pilot_extra_args+=(--enforce-eager)
fi
if [[ -n "${PILOT_MAX_BATCHED_TOKENS:-}" ]]; then
    pilot_extra_args+=(--enable-chunked-prefill --max-num-batched-tokens "$PILOT_MAX_BATCHED_TOKENS")
fi
pilot_revision="${PILOT_MODEL_REVISION:-}"
if [[ -z "$pilot_revision" ]]; then
    pilot_revision=$(python -c 'import sys; from huggingface_hub import model_info; print(model_info(sys.argv[1]).sha)' "$pilot_model")
fi
echo "$pilot_revision" > "$pilot_out/model-revision.txt"
pilot_node=$(hostname -f)
python - "$pilot_out/server.json" "$pilot_node" "$pilot_port" "$pilot_revision" "$pilot_model" "$pilot_gpu_memory" "$pilot_max_sequences" "${PILOT_ENFORCE_EAGER:-0}" <<'PY'
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "host": sys.argv[2], "port": int(sys.argv[3]),
    "model": sys.argv[5], "model_revision": sys.argv[4],
    "context_length": 32768,
    "gpu_memory_utilization": float(sys.argv[6]),
    "max_num_seqs": int(sys.argv[7]), "enforce_eager": sys.argv[8] == "1",
    "max_num_batched_tokens": int(os.environ.get("PILOT_MAX_BATCHED_TOKENS", "0")) or None,
}, indent=2) + "\n")
PY

# Raw completions retain reasoning. Do not enable a reasoning parser here.
python -m vllm.entrypoints.cli.main serve "$pilot_model" --revision "$pilot_revision" --tokenizer-revision "$pilot_revision" \
    --host 127.0.0.1 --port "$pilot_port" --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization "$pilot_gpu_memory" --max-num-seqs "$pilot_max_sequences" \
    "${pilot_extra_args[@]}" > "$pilot_out/vllm.log" 2>&1 &
pilot_server_pid=$!
cleanup() {
    kill "$pilot_server_pid" 2>/dev/null || true
    wait "$pilot_server_pid" 2>/dev/null || true
}
trap cleanup EXIT
pilot_ready=0
for ((i=0; i<${PILOT_STARTUP_CHECKS:-180}; i++)); do
    if ! kill -0 "$pilot_server_pid" 2>/dev/null; then
        echo "vLLM failed: $pilot_out/vllm.log"
        tail -n 100 "$pilot_out/vllm.log"
        exit 1
    fi
    if curl --max-time 2 -fsS "http://127.0.0.1:$pilot_port/health" >/dev/null 2>&1; then
        pilot_ready=1
        break
    fi
    sleep 10
done
if [[ "$pilot_ready" != 1 ]]; then
    echo "vLLM startup timed out: $pilot_out/vllm.log"
    tail -n 100 "$pilot_out/vllm.log"
    exit 1
fi
echo "READY: $pilot_node:$pilot_port model=$pilot_model model_revision=$pilot_revision"
echo 'On your Mac, using your established CSAIL SSH configuration:'
echo "ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:$pilot_local_port:127.0.0.1:$pilot_port lruis@$pilot_node"
echo "Use --model $pilot_model --model-revision $pilot_revision for the local pilot. Artifacts: $pilot_out"
wait "$pilot_server_pid"
