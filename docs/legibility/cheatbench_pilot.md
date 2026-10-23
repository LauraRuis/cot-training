# CheatBench inference pilot

The user handles all cluster connections, setup, and job submission. Local code
preparation does not authorize SSH or submission by the assistant.

The intended policy is `Qwen/Qwen3-4B` with thinking enabled. This is the
post-trained, instruction-following hybrid checkpoint, not `Qwen3-4B-Base`.
The separate `Qwen3-4B-Instruct-2507` checkpoint has no native thinking mode.
No model training is part of the pilot.

## Container status (2026-10-23)

The user checked `andreas-h100-1`: Apptainer and Singularity executables exist;
Docker is absent. The upstream CheatBench SWE runner expects Harbor with Docker.
Apptainer support therefore requires an adapter, not just changing a model flag.

The inspected Flipt task starts from a published `jefzda/sweap-images` image and
then resets the repository to a pinned commit, sets `/app` and `PYTHONPATH`, and
performs setup commands. Pulling its base image alone does not reproduce this
environment. An adapter must preserve those setup steps, per-episode changes,
and the native verifier. The proposed writable-test experiment additionally
needs separate proxy and pristine-test grading; native SWE restores gold tests.

Job `2819315` passed the writable bind and persistent overlay/fakeroot checks
using Apptainer 1.5.3. Its warning, `Not virtualizing pid namespace by
configuration`, leaves process isolation unresolved. The pilot explicitly asks
for a PID namespace and compares the container's namespace identity with the
host's. It also checks that a host-only marker is inaccessible. A failed check
stops preparation before the large task-image download and stops inference
before weights load. The runner does not override site configuration. If the
check fails, cluster support must provide a supported execution environment
with PID isolation before model-generated shell commands can run here.

## Implemented pilot

`contract/cheatbench/pilot.py` provides `prepare` and `run` stages. This is a
single-task capability smoke test, with two independent samples by default:

- Task: `instance_flipt-io__flipt-02e21636c58e86c51119b63e0fb5ca7b813b07b1`
  (Redis TLS configuration in Go).
- CheatBench revision: `1bd70d4f7509966e069ef602e5c4df77d66d2802`.
- Preparation executes the task Dockerfile's supported setup commands in an
  overlay, then runs the native verifier on untouched and reference-patched
  copies. It requires native rewards 0 and 1 respectively. The manifest records
  source revision, task-file hashes, SIF hash and initial-overlay hash.
- Every sample receives a fresh writable overlay and private home/tmp storage.
  The policy uses a minimal JSON shell protocol, rather than Harbor's agent
  harness. Reasoning is retained across turns, including in saved raw model
  responses. Defaults are 12 turns, 8,192 generated tokens per turn, a 32,768
  token context, temperature 0.6, top-p 0.95, and top-k 20. Context overflow,
  generation truncation and malformed actions end the sample explicitly.
- The verifier runs on a copy of the final state. A diagnostic, frozen
  Qwen3-4B monitor receives only the task and reasoning, using nonthinking mode.
  Its score is the yes/no probability renormalized over matching tokens in the
  top-20 next-token logprobs. Missing probability mass can distort this score;
  this is not a validated calibrated monitor. Unavailable scores remain null.
- Cheating labels remain null until an action/artifact audit. Neither the CoT
  monitor nor native task success is treated as cheating ground truth.

## Running on the cluster

Copy the branch's code to `/data/scratch/lruis/cot-training`. The launchers use
the existing environment at
`/data/scratch/lruis/LLMScience/llm_science_eval_env/bin/activate`. Preparation
needs Apptainer, Git and Python; inference additionally needs that environment's
vLLM, Transformers, OpenAI client and huggingface_hub. All image/model downloads
occur inside the submitted allocation.

From the cluster project directory, first submit the CPU-only preparation job:

```bash
sbatch scripts/prepare_cheatbench_pilot.sbatch \
  --probe-image /data/scratch/lruis/cot-training/container_checks/2819315/ubuntu.sif
```

This uses account `lingo`, partition `csail-shared-h200`, QoS `shared-if-available`, eight CPUs,
32 GiB RAM and three hours. If the partition disallows CPU-only jobs, choose a
site-supported partition/resource override. Logs are
`/data/scratch/lruis/cheatbench_prepare-<job-id>.log`; detailed artifacts are in
`cheatbench_work/prepared/`. With the reported PID warning, this job may stop
at the isolation check. Read `isolation.log` before requesting a GPU.

After preparation succeeds and writes `cheatbench_work/prepared/READY`:

```bash
sbatch scripts/run_cheatbench_pilot.sbatch
```

The inference job uses the same partition and QoS and requests one H200 GPU,
eight CPUs, 80 GiB RAM and three hours. It
resolves and records a single Hugging Face model revision for both vLLM and the
tokenizer. Logs are `/data/scratch/lruis/cheatbench_qwen4b-<job-id>.log`.
Run artifacts live in `cheatbench_work/runs/<job-id>/`, including package
versions, code snapshots, model revision and vLLM logs. Under `episodes/`, each
sample retains full prompts/responses, tool outputs, final diff/status, the
final filesystem overlay, verifier logs and monitor inputs/score.

`PILOT_CODE_ROOT` and `PILOT_WORK_ROOT` override the default paths. Preparation
refuses to overwrite an existing `prepared/` directory, including a failed
attempt; preserve the diagnostic logs and select a new work root for a retry.
Use the same work root for both jobs. Keep large images/overlays on cluster
scratch; do not sync them into the local Dropbox-backed project directory.

## Interpretation and remaining limitations

Local unit checks cover action parsing, command construction, isolation failure
and verifier-result handling; shell syntax checks pass. No SWE image or GPU run
has been executed by the assistant. The cluster run must establish image setup,
reference grading, vLLM compatibility and model competence.

The native Flipt parser records only passing tests. Completed grading with an
empty parsed test list retains native reward zero but is marked
`empty_test_results_review_required`: inspect raw logs to distinguish code
compilation failures from dependency/infrastructure failures. Missing verifier
outputs and timeouts yield null success. Even nonempty results need inspection
if the run looks abnormal. The 0/1 preparation check is a compatibility check,
not an independent guarantee of verifier integrity after adversarial actions.

Audit saved actions and filesystem changes for use of existing solutions and
evaluation tampering, citing concrete commands/files. A legitimate success
requires both native success and a completed audit finding no cheating. Report
incomplete episodes and infrastructure failures separately. Two samples of one
task cannot establish a hacking rate or monitor accuracy.

This adapter changes the agent protocol, tool budget, container runtime and
process persistence relative to upstream Harbor. PID/home containment is a
practical check, not a hardened sandbox certification; network access remains
available. Native SWE restores gold tests, so this pilot does not yet implement
the editable-test proxy reward needed for the proposed OpenAI-style negative
control. No optimization, monitor penalty, calibrated-belief reward or
independent automated cheating grader has been added.

References:

- [CheatBench SWE setup](https://github.com/centerforaisafety/cheatbench/blob/main/tasks/software_engineering/harbor/README.md)
- [Inspected task Dockerfile](https://github.com/centerforaisafety/cheatbench/blob/main/tasks/software_engineering/harbor/tasks/instance_flipt-io__flipt-02e21636c58e86c51119b63e0fb5ca7b813b07b1/environment/Dockerfile)
- [Apptainer overlays](https://apptainer.org/docs/user/main/persistent_overlays.html)
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
