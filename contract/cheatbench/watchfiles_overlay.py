"""Validate a job-local watchfiles replacement without changing the base env."""
import argparse
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


def validate(overlay, preflight=False):
    overlay = Path(overlay).resolve()
    if not preflight and not (overlay / "READY").is_file():
        raise RuntimeError("The watchfiles setup job has not completed successfully")
    manifest = json.loads((overlay / "manifest.json").read_text())
    if Path(sys.prefix).resolve() != Path(manifest["base_env"]).resolve():
        raise RuntimeError("Overlay was prepared for a different Python environment")
    if list(sys.version_info[:2]) != manifest["python_version"]:
        raise RuntimeError("Overlay Python version does not match this interpreter")
    # Check the base version before adding the replacement to this process.
    if version("watchfiles") != manifest["watchfiles_version"]:
        raise RuntimeError("Base watchfiles version changed; prepare a new overlay")
    target = overlay / "site-packages"
    sys.path.insert(0, str(target))
    if version("watchfiles") != manifest["watchfiles_version"]:
        raise RuntimeError("Replacement watchfiles version does not match the base")
    for name in ("watchfiles", "watchfiles._rust_notify"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(target):
            raise RuntimeError(f"{name} was not loaded from the replacement directory")
        print(f"{name}: {module.__file__}", flush=True)
    if preflight:
        # Exercise the import chain that failed, without loading model weights.
        importlib.import_module("vllm.entrypoints.openai.api_server")
        print("vLLM API server imports OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overlay", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    validate(args.overlay, args.preflight)
