"""Test replacement selection in fresh processes, without installing packages."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class WatchfilesOverlayTests(unittest.TestCase):
    def fixture(self, root, replacement_version="1.0", native=True):
        base = root / "base-packages"
        overlay = root / "overlay"
        target = overlay / "site-packages"
        for directory, package_version in ((base, "1.0"), (target, replacement_version)):
            package = directory / "watchfiles"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("from . import _rust_notify\n")
            metadata = directory / f"watchfiles-{package_version}.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text(f"Name: watchfiles\nVersion: {package_version}\n")
        # The base installation is deliberately broken, as on the cluster.
        if native:
            (target / "watchfiles/_rust_notify.py").write_text("MARKER = 'replacement'\n")
        (overlay / "manifest.json").write_text(json.dumps({
            "base_env": sys.prefix, "python_version": list(sys.version_info[:2]),
            "watchfiles_version": "1.0"}))
        (overlay / "READY").write_text("ready\n")
        return base, overlay

    def check(self, base, overlay):
        code = """
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from contract.cheatbench.watchfiles_overlay import validate
validate(sys.argv[2])
"""
        return subprocess.run([sys.executable, "-c", code, str(base), str(overlay)],
                              capture_output=True, text=True)

    def test_replacement_imports_without_repairing_base(self):
        with tempfile.TemporaryDirectory() as directory:
            base, overlay = self.fixture(Path(directory))
            before = {str(p): p.read_bytes() for p in base.rglob("*") if p.is_file()}
            result = self.check(base, overlay)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(overlay / "site-packages/watchfiles/_rust_notify.py"), result.stdout)
            after = {str(p): p.read_bytes() for p in base.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((base / "watchfiles/_rust_notify.py").exists())

    def test_rejects_different_replacement_version(self):
        with tempfile.TemporaryDirectory() as directory:
            base, overlay = self.fixture(Path(directory), replacement_version="2.0")
            result = self.check(base, overlay)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Replacement watchfiles version does not match", result.stderr)

    def test_missing_backend_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            base, overlay = self.fixture(Path(directory), native=False)
            result = self.check(base, overlay)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("_rust_notify", result.stderr)

    def test_requires_successful_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            base, overlay = self.fixture(Path(directory))
            (overlay / "READY").unlink()
            result = self.check(base, overlay)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("setup job has not completed", result.stderr)


if __name__ == "__main__":
    unittest.main()
