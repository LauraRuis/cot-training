"""Docker environment backend; model inference may live behind an SSH tunnel.

Each sample has an unmounted, private-PID container. Stop it after each command
so background processes do not leak into later tool calls. Keep stopped sample
containers and final image snapshots for audit; never mount the Docker socket.
"""
import json
import uuid

from contract.cheatbench.pilot import REVISION, TASK, checked, digest, execute, read_grade, save


class DockerRuntime:
    def __init__(self, args, image):
        self.args, self.image = args, image
        self.cli = ["docker"]
        if args.docker_context:
            self.cli += ["--context", args.docker_context]

    def validate(self, out):
        checked(self.cli + ["info"], out / "docker-info.log", 30)
        checked(self.cli + ["image", "inspect", self.image], out / "image-inspect.json", 30)
        info = json.loads((out / "image-inspect.json").read_text())[0]
        if info["Id"] != self.image:
            raise RuntimeError("Use the prepared immutable Docker image ID")
        if info["Config"].get("Volumes"):
            raise RuntimeError("Image declares volumes: filesystem snapshots would omit their changes")

    def create(self, image, out, mounts=()):
        name = "cheatbench-" + uuid.uuid4().hex[:12]
        argv = self.cli + ["create", "--name", name, "--platform", "linux/amd64",
            "--label", "cot-training.cheatbench=true", "--init", "--cpus", str(self.args.docker_cpus),
            "--memory", self.args.docker_memory, "--pids-limit", "512",
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
            "--network", "bridge", "--workdir", "/app", "--entrypoint", "/bin/sleep"]
        for mount in mounts:
            argv += ["--mount", mount]
        # Model commands never enter this host-side argv as executable syntax.
        checked(argv + [image, "infinity"], out / "container-create.log", 60)
        checked(self.cli + ["inspect", name], out / "container-inspect.json", 30)
        info = json.loads((out / "container-inspect.json").read_text())[0]
        if info["HostConfig"].get("PidMode") or info["HostConfig"].get("Privileged"):
            raise RuntimeError("Expected a private, unprivileged Docker container")
        if not mounts and info.get("Mounts"):
            raise RuntimeError("Agent container unexpectedly has host mounts or volumes")
        save(out / "container.json", {"name": name, "image": image})
        return name

    def new_episode(self, out):
        return self.create(self.image, out)

    def command(self, state, command, log, timeout):
        checked(self.cli + ["start", state], log.with_suffix(".start.log"), 60)
        try:
            result = execute(self.cli + ["exec", "--workdir", "/app", state,
                             "/bin/bash", "-c", command], log, timeout)
        finally:
            # Killing docker exec's client alone does not stop container code.
            checked(self.cli + ["stop", "--time", "0", state], log.with_suffix(".stop.log"), 60)
        checked(self.cli + ["inspect", state], log.with_suffix(".state.json"), 30)
        result["container_state"] = json.loads(log.with_suffix(".state.json").read_text())[0]["State"]
        return result

    def snapshot(self, state, out):
        checked(self.cli + ["commit", state], out / "snapshot-id.txt", 600)
        image = (out / "snapshot-id.txt").read_text().strip()
        if not image.startswith("sha256:"):
            raise RuntimeError("Docker commit did not return an immutable image ID")
        return image

    def grade(self, state, task, out):
        out.mkdir()
        snapshot = self.snapshot(state, out)
        grader = self.create(snapshot, out, [
            f"type=bind,src={task / 'tests'},dst=/tests,readonly",
            f"type=bind,src={out},dst=/logs/verifier"])
        result = self.command(grader, "bash /tests/test.sh", out / "verifier.log",
                              self.args.verifier_timeout)
        return read_grade(out, result)


def prepare_docker(args):
    # Check the engine before creating the preparation directory, so starting
    # Docker Desktop is enough to retry a missing-daemon failure.
    root = args.work_root
    root.mkdir(parents=True, exist_ok=True)
    runtime = DockerRuntime(args, None)
    checked(runtime.cli + ["info"], root / "docker-info.log", 30)
    out = root / "prepared"
    out.mkdir()
    source = out / "cheatbench"
    task_path = "tasks/software_engineering/harbor/tasks/" + TASK
    checked(["git", "clone", "--filter=blob:none", "--no-checkout",
             "https://github.com/centerforaisafety/cheatbench.git", source], out / "clone.log", 600)
    checked(["git", "-C", source, "sparse-checkout", "set", task_path], out / "sparse.log", 600)
    checked(["git", "-C", source, "checkout", "--detach", REVISION], out / "checkout.log", 600)
    task = source / task_path
    # Build the upstream Dockerfile directly, including all task setup steps.
    print(f"Building linux/amd64 task image; log: {out / 'build.log'}", flush=True)
    checked(runtime.cli + ["build", "--platform", "linux/amd64", "--iidfile", out / "image-id.txt",
            "-f", task / "environment/Dockerfile", task / "environment"], out / "build.log", 7200)
    runtime.image = (out / "image-id.txt").read_text().strip()
    runtime.validate(out)
    config = json.loads((task / "tests/config.json").read_text())
    grades = {}
    for name in ("base", "reference"):
        print(f"Checking {name} solution (cold builds can take several minutes)", flush=True)
        sample = out / name
        sample.mkdir()
        state = runtime.new_episode(sample)
        if name == "reference":
            patch = sample / "reference.patch"
            patch.write_text(config["patch"])
            checked(runtime.cli + ["cp", patch, f"{state}:/reference.patch"], sample / "patch-copy.log", 60)
            applied = runtime.command(state, "git apply /reference.patch && rm /reference.patch",
                                      sample / "patch-apply.log", 60)
            if applied["returncode"]:
                raise RuntimeError(f"Reference patch failed: {sample}")
        grades[name] = runtime.grade(state, task, sample / "verifier")
    manifest = {"runtime": "docker", "platform": "linux/amd64", "image_id": runtime.image,
                "cheatbench_revision": REVISION, "task_id": TASK, "grades": grades,
                "task_files_sha256": {str(p.relative_to(task)): digest(p)
                                      for p in task.rglob("*") if p.is_file()},
                "mode": "native_swe_capability_pilot"}
    save(out / "manifest.json", manifest)
    if grades["base"]["success"] != 0 or grades["reference"]["success"] != 1:
        raise RuntimeError("Base/reference checks did not produce 0/1; inspect prepared/manifest.json")
    (out / "READY").write_text("Base fails; reference passes.\n")
    print(f"Prepared Docker task {TASK}; base=0, reference=1", flush=True)
