#!/bin/bash
# Submit from the cluster: sbatch scripts/check_cheatbench_apptainer.sbatch
# Infrastructure probe only: downloads Ubuntu, but no models or benchmark images.
# No GPU requested. Artifacts stay under CHEATBENCH_WORK_ROOT for inspection.
#SBATCH --job-name=cheatbench_container_check
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/data/scratch/lruis/cheatbench_container_check-%j.log
#SBATCH --error=/data/scratch/lruis/cheatbench_container_check-%j.log

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit this script with sbatch so downloads run on a compute node.'
    exit 2
fi

probe_root=${CHEATBENCH_WORK_ROOT:-/data/scratch/lruis/cot-training}
probe_run="$probe_root/container_checks/$SLURM_JOB_ID"
mkdir -p "$probe_run/workspace" "$probe_root/apptainer_cache"
export APPTAINER_CACHEDIR="$probe_root/apptainer_cache"
# Image conversion needs local temporary storage, not a network filesystem.
probe_tmp=$(mktemp -d /tmp/cheatbench-container-check.XXXXXX)
export APPTAINER_TMPDIR="$probe_tmp"
trap 'rm -rf -- "$probe_tmp"' EXIT

hostname
apptainer --version
echo "Artifacts: $probe_run"
echo 'CHECK: pull a small OCI image (Ubuntu 22.04)'
apptainer pull "$probe_run/ubuntu.sif" docker://ubuntu:22.04

# Test the fallback where repository files live in a per-episode host directory.
echo 'CHECK: writable bind mount'
apptainer exec --containall --cleanenv --pwd /workspace \
    --bind "$probe_run/workspace:/workspace" "$probe_run/ubuntu.sif" \
    sh -ec 'printf "bind-ok\n" > /workspace/probe.txt'
test "$(cat "$probe_run/workspace/probe.txt")" = bind-ok
echo 'PASS: writable bind mount'

# More faithful Docker-like semantics: edits anywhere in the container persist
# between separate tool invocations. Fakeroot is namespace-local, not host sudo.
echo 'CHECK: persistent writable overlay and fakeroot'
overlay_status=FAIL
if apptainer overlay create --size 64 --fakeroot "$probe_run/overlay.img"; then
    if apptainer exec --containall --cleanenv --fakeroot --pwd / \
        --overlay "$probe_run/overlay.img" "$probe_run/ubuntu.sif" \
        sh -ec 'mkdir -p /cheatbench-probe; printf "overlay-ok\n" > /cheatbench-probe/state'; then
        if apptainer exec --containall --cleanenv --fakeroot --pwd / \
            --overlay "$probe_run/overlay.img" "$probe_run/ubuntu.sif" \
            sh -ec 'test "$(cat /cheatbench-probe/state)" = overlay-ok'; then
            overlay_status=PASS
        fi
    fi
fi
echo "$overlay_status: persistent writable overlay and fakeroot"
echo 'This checks container primitives only; SWE image/setup/verifier compatibility remains untested.'
if [[ "$overlay_status" != PASS ]]; then
    echo 'Keep this log: the error determines whether to use bind mounts or change the runtime configuration.'
    exit 1
fi
