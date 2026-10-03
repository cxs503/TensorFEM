#!/usr/bin/env python3
"""Build and install TensorFEM artifacts without contacting package indexes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("LICENSE", "README.md", "CHANGELOG.md", "pyproject.toml", "api/public-api-v1.json")


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def static_audit() -> dict[str, object]:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    manifest = json.loads((ROOT / "api/public-api-v1.json").read_text()) if not missing else {}
    return {"ok": not missing, "missing": missing, "api_symbols": len(manifest.get("entries", []))}


def build_and_smoke() -> dict[str, object]:
    audit = static_audit()
    if not audit["ok"]:
        raise RuntimeError(f"missing release files: {audit['missing']}")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required for reproducible offline sdist/wheel builds")
    with tempfile.TemporaryDirectory(prefix="tensorfem-release-") as raw:
        temp = Path(raw)
        dist = temp / "dist"
        run([uv, "build", "--offline", "--out-dir", str(dist)], cwd=ROOT)
        wheel, sdist = next(dist.glob("*.whl")), next(dist.glob("*.tar.gz"))
        with zipfile.ZipFile(wheel) as archive:
            wheel_names = set(archive.namelist())
        required_wheel = {"tensorfem/__init__.py", "tensorfem/cli.py", "tensorfem/public_api_manifest.json"}
        if not required_wheel <= wheel_names:
            raise RuntimeError(f"wheel package-data missing: {sorted(required_wheel-wheel_names)}")
        with tarfile.open(sdist) as archive:
            sdist_names = {name.split("/", 1)[-1] for name in archive.getnames() if "/" in name}
        required_sdist = {"LICENSE", "README.md", "CHANGELOG.md", "api/public-api-v1.json"}
        if not required_sdist <= sdist_names:
            raise RuntimeError(f"sdist files missing: {sorted(required_sdist-sdist_names)}")
        venv = temp / "venv"
        run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
        python = venv / "bin/python"
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None or torch_spec.origin is None:
            raise RuntimeError("current interpreter has no local PyTorch dependency")
        local_site = run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True,
        ).stdout.strip()
        # A venv made from a venv does not inherit the parent's site-packages.
        # Point only at the already installed dependency directory; no downloads.
        Path(local_site, "tensorfem-local-dependencies.pth").write_text(
            str(Path(torch_spec.origin).parent.parent) + "\n", encoding="utf-8"
        )
        # The parent interpreter may expose source-tree egg-info through
        # --system-site-packages; force installation proves the artifact itself.
        install_env = dict(os.environ)
        # Do not let a caller's warning policy turn pip's vendored
        # pkg_resources warning into an unrelated TensorFEM release failure.
        install_env["PYTHONWARNINGS"] = "default"
        run([str(python), "-m", "pip", "install", "--no-index", "--no-deps",
             "--force-reinstall", str(wheel)], env=install_env)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        code = ("import importlib.metadata as m, importlib.resources as r, json, tensorfem; "
                "assert m.version('tensorfem') == tensorfem.__version__; "
                "assert json.loads(r.files('tensorfem').joinpath('public_api_manifest.json').read_text())['entries']")
        run([str(python), "-Werror", "-c", code], cwd=temp, env=env)
        run([str(venv / "bin/tensorfem"), "capabilities"], cwd=temp, env=env,
            stdout=subprocess.DEVNULL)
        return {**audit, "wheel": wheel.name, "sdist": sdist.name, "smoke": "pass"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="also build/install artifacts")
    args = parser.parse_args()
    result = build_and_smoke() if args.build else static_audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
