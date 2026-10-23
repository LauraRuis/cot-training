#!/bin/bash
# Install one replacement wheel outside the existing environment, on compute.
#SBATCH --job-name=cheatbench_watchfiles
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=00:10:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/data/scratch/lruis/cheatbench_watchfiles-%j.log
#SBATCH --error=/data/scratch/lruis/cheatbench_watchfiles-%j.log
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch so package downloads run on a compute node.'
    exit 2
fi

cd "${PILOT_CODE_ROOT:-/data/scratch/lruis/cot-training}"
pilot_env=${PILOT_VENV:-/data/scratch/lruis/LLMScience/llm_science_eval_env}
source "$pilot_env/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
pilot_uv=$(command -v uv || true)
if [[ -z "$pilot_uv" && -x "$HOME/.local/bin/uv" ]]; then
    pilot_uv="$HOME/.local/bin/uv"
fi
if [[ -z "$pilot_uv" ]]; then
    echo 'uv was not found on PATH or in ~/.local/bin. No packages were changed.'
    exit 1
fi

pilot_overlay="${PILOT_WORK_ROOT:-/data/scratch/lruis/cot-training/cheatbench_work}/server_overlays/$SLURM_JOB_ID"
mkdir -p "$(dirname "$pilot_overlay")"
mkdir "$pilot_overlay"  # never overwrite an earlier preparation
pilot_version=$(python -c 'from importlib.metadata import version; print(version("watchfiles"))')
python - "$pilot_overlay/manifest.json" <<'PY'
from importlib.metadata import version
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "base_env": sys.prefix,
    "python_version": list(sys.version_info[:2]),
    "watchfiles_version": version("watchfiles"),
}, indent=2) + "\n")
PY
echo "Installing watchfiles==$pilot_version into $pilot_overlay/site-packages"
# --target is the only install destination. --no-deps prevents changes to
# anyio, vLLM, Torch or other dependencies; require a compatible binary wheel.
"$pilot_uv" --no-config --no-cache pip install \
    --python "$pilot_env/bin/python" --target "$pilot_overlay/site-packages" \
    --no-deps --only-binary :all: "watchfiles==$pilot_version"

if ! python -m contract.cheatbench.watchfiles_overlay "$pilot_overlay" --preflight \
    > "$pilot_overlay/preflight.log" 2>&1; then
    cat "$pilot_overlay/preflight.log"
    echo 'Preflight failed. No further packages were installed; base environment unchanged.'
    exit 1
fi
cat "$pilot_overlay/preflight.log"
printf 'Same-version replacement and server imports verified.\n' > "$pilot_overlay/READY"
echo "READY: $pilot_overlay"
echo 'Submit the server with this job-scoped replacement:'
printf 'PILOT_VENV=%q PILOT_WATCHFILES_OVERLAY=%q sbatch scripts/serve_cheatbench_qwen.sbatch\n' \
    "$pilot_env" "$pilot_overlay"
