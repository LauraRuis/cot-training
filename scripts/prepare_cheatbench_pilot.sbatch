#!/bin/bash
# CPU-only preparation; the user submits this job.
#SBATCH --job-name=cheatbench_prepare
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/scratch/lruis/cheatbench_prepare-%j.log
#SBATCH --error=/data/scratch/lruis/cheatbench_prepare-%j.log
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch so downloads run on a compute node.'
    exit 2
fi

cd "${PILOT_CODE_ROOT:-/data/scratch/lruis/cot-training}"
source /data/scratch/lruis/LLMScience/llm_science_eval_env/bin/activate
PILOT_WORK_ROOT=${PILOT_WORK_ROOT:-/data/scratch/lruis/cot-training/cheatbench_work}
mkdir -p "$PILOT_WORK_ROOT/apptainer_cache"
export APPTAINER_CACHEDIR="$PILOT_WORK_ROOT/apptainer_cache"
pilot_tmp=$(mktemp -d /tmp/cheatbench-prepare.XXXXXX)
export APPTAINER_TMPDIR="$pilot_tmp"
trap 'rm -rf -- "$pilot_tmp"' EXIT
python -m contract.cheatbench.pilot prepare --work-root "$PILOT_WORK_ROOT" "$@"
