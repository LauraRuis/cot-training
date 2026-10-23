# CheatBench inference pilot

The user handles all cluster connections, setup, and job submission. Local code
preparation does not authorize SSH or submission by the assistant.

The intended policy is `Qwen/Qwen3-4B` with thinking enabled. This is the
post-trained, instruction-following hybrid checkpoint, not `Qwen3-4B-Base`.
The separate `Qwen3-4B-Instruct-2507` checkpoint has no native thinking mode.
No model training is part of the pilot.

## Current route: Docker on the Mac, Qwen on Slurm

**Local compatibility result, 2026-10-23:** preparation completed successfully
at `/private/tmp/cot-cheatbench-docker/prepared/`. The untouched solution scored
0; its gold tests failed to compile because the requested TLS fields and
`NewClient` were missing. The reference solution scored 1, with all four
required tests passing (16 parsed passing tests total). The untouched check
took approximately 7.5 minutes; neither it nor the reference check timed out.
The initial Docker image is approximately 4.87 GB, ID
`sha256:08fd2153ce3132d418b7655fd3c274d4476cf1bbe133d319fb11c4ed8ab0a577`.
`READY`, the manifest and raw verifier logs are present. The API-client
environment is installed at `/private/tmp/cot-cheatbench-client-env`, with
versions recorded in `/private/tmp/cot-cheatbench-docker/client-packages.txt`.
Ten local unit checks pass. No model inference or cluster job was run.

The PID-isolation probe failed on `andreas-h100-1`, `aia-h200-2` and
`groenig-1`. While TIG advises on cluster execution, use Docker Desktop locally
for the environment and a separate H100 allocation solely for model serving.
The Mac had an Intel-only Docker Desktop 4.4.2 installation; replace it with
the Apple-silicon build before attempting this route.

The Docker backend builds the upstream Dockerfile directly for `linux/amd64`.
On Apple silicon this uses emulation, so compilation/runtime compatibility must
be checked before inference. Each sample has a fresh container with no host
mounts, a private PID namespace, a 6 GiB RAM limit, four CPUs and a 512-process
limit. Commands run in `/app`; stopping the container after every tool call
ends background processes, including on client timeout. A Docker image snapshot
preserves the final filesystem; native grading runs in a separate container
with verifier assets mounted only then. Containers/snapshots are retained for
audit; their IDs are written in the run directory. Do not prune Docker storage
until the audit is complete. Networking remains enabled for benchmark tasks.

First prepare locally, from this repository directory (no GPU):

```bash
python3 -m contract.cheatbench.pilot prepare --runtime docker \
  --docker-context desktop-linux --work-root /private/tmp/cot-cheatbench-docker
```

Only Python's standard library, Git and a working Docker engine are needed for
preparation. Logs and the native base/reference checks are under
`/private/tmp/cot-cheatbench-docker/prepared/`. `READY` requires base=0 and
reference=1. Preparation refuses to overwrite an existing `prepared/` folder;
preserve failed-run logs and choose a fresh work root for a retry. Docker build
layers can still be reused. Local outputs must stay outside the Dropbox-backed
repository; `/private/tmp` is temporary and should be archived elsewhere after
the pilot. Keep prepared images in the same Docker context used for the run.

Once preparation succeeds, copy the new server launcher to the cluster and
submit it there:

```bash
sbatch scripts/serve_cheatbench_qwen.sbatch
```

It requests one H100 on `lingo-h100`, QoS `lingo-main`, account
`lingo`, for three hours. It needs no Apptainer or prepared task directory. It
pins a Qwen3-4B revision for model and tokenizer and listens on compute-node
loopback only. Read `/data/scratch/lruis/cheatbench_server-<job-id>.log`; when
ready, it prints the exact SSH tunnel command and the model revision. Establish
your CSAIL SSH/DUO connection manually, then run that printed command on the Mac
using your existing CSAIL SSH configuration. No assistant-initiated SSH is
needed. The tunnel forwards Mac port 8000 to the compute node, not the login
node. Test it locally with `curl http://127.0.0.1:8000/health`.

The local client needs `openai` and `transformers` (tokenizer only; no local
model weights or PyTorch are needed). Use an isolated Python 3.12 environment:

```bash
python3.12 -m venv /private/tmp/cot-cheatbench-client-env
/private/tmp/cot-cheatbench-client-env/bin/pip install 'openai>=1,<3' 'transformers>=4.51,<5'
```

With the tunnel running, replace `REVISION_FROM_SERVER_LOG` below with the
printed commit hash:

```bash
/private/tmp/cot-cheatbench-client-env/bin/python -m contract.cheatbench.pilot run \
  --runtime docker --docker-context desktop-linux \
  --work-root /private/tmp/cot-cheatbench-docker \
  --out /private/tmp/cot-cheatbench-docker/run-001 \
  --url http://127.0.0.1:8000/v1 \
  --tool-timeout 900 \
  --model-revision REVISION_FROM_SERVER_LOG
```

The local command explicitly uses a 15-minute shell timeout because the first
cold compilation under emulation exceeded five minutes; the CLI's legacy
default remains two minutes. Record this budget difference when comparing runs.
Keep the Mac awake and the tunnel open. Use a new output directory for each
run. If emulation hits the tool timeout, record it as an incomplete episode;
adjust budgets deliberately before a new run, rather than silently counting
timeouts as model failures. Preemption or loss of the server/tunnel stops API
calls; this pilot does not automatically resume partial trajectories. Stop
the Slurm server job with `scancel <job-id>` when finished.

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

## Original all-cluster route (blocked on PID isolation)

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

Local unit checks cover action parsing, command construction, isolation failure,
verifier-result handling, Docker cleanup on timeout/error and a mocked split
rollout with retained reasoning and independent audit labels. Shell syntax
checks pass. Actual Docker preparation results are recorded in the local work
root. No GPU/model pilot has been run; vLLM compatibility and model competence
remain untested.

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
