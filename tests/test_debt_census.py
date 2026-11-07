"""Regression tests for ``tools/debt_census.py``.

The tool is the TensorFEM port of TensorFVM's ``tools/debt_census.py``; the
four debt metrics (``numpy_only`` / ``sync_points`` / ``no_device_constructs``
/ ``host_loops``) and the JSON schema are frozen by that port.  These tests
pin three properties the deliverable is judged on:

* the emitted schema is byte-compatible with the FVM original (same top-level
  keys, ``fem-debt-census/1`` schema id, ``repo`` = TensorFEM),
* the four metrics are exact on a synthetic fixture whose every construct is
  written down (no "trust me" assertions),
* two consecutive runs produce identical output, from any working directory,
  and no absolute path leaks into the JSON.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "debt_census.py"

# Top-level keys of the FVM census (TensorFVM artifacts/p8/fvm_debt.json),
# frozen here so a schema drift in either direction fails loudly.
FROZEN_KEYS = [
    "schema_version",
    "repo",
    "roots",
    "excluded_dir_names",
    "files_scanned",
    "files_parsed",
    "files_excluded",
    "excluded_files",
    "ast_coverage_pct",
    "uncovered_files",
    "source_digest",
    "numpy_only_count",
    "sync_points",
    "no_device_constructs",
    "host_loops",
    "breakdown",
    "cross_check",
    "numpy_only_modules",
    "sync_sites",
    "no_device_sites",
    "per_file",
]

HOST_MODULE = '''\
import numpy as np


def f(x):
    a = np.zeros(3)
    for i in range(3):
        a[i] = x.cpu()
    b = np.asarray(x.numpy())
    while a[0] > 0:
        a[0] -= 1
    return a, b
'''

DEV_MODULE = '''\
import torch


def g(x):
    y = torch.zeros(3, device=x.device)
    z = torch.ones(3)
    w = torch.zeros_like(z)
    u = torch.tensor([1.0], dtype=torch.float32)
    for i in range(2):
        u[0] = 0.0
    return y, z, w, u.cpu().numpy()
'''

BROKEN_MODULE = "def f(:\n    pass\n"

HOST_ONLY_TEST = "for i in range(3):\n    pass\n"


def _load_tool():
    spec = importlib.util.spec_from_file_location("debt_census_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fixture_repo(tmp_path):
    """Minimal repo: 2 parsable src modules, 1 unparsable, 1 excluded test."""
    pkg = tmp_path / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "host.py").write_text(HOST_MODULE, encoding="utf-8")
    (pkg / "dev.py").write_text(DEV_MODULE, encoding="utf-8")
    (pkg / "broken.py").write_text(BROKEN_MODULE, encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text(HOST_ONLY_TEST, encoding="utf-8")
    return tmp_path


def test_schema_is_fvm_compatible(fixture_repo):
    census = _load_tool().build_census(fixture_repo, ("src",))
    assert list(census) == FROZEN_KEYS
    assert census["schema_version"] == "fem-debt-census/1"
    assert census["repo"] == "TensorFEM"
    assert census["roots"] == ["src"]
    assert census["source_digest"].startswith("sha256:")


def test_four_metrics_exact_on_fixture(fixture_repo):
    census = _load_tool().build_census(fixture_repo, ("src",))

    assert census["files_scanned"] == 3  # host.py, dev.py, broken.py
    assert census["files_parsed"] == 2
    # EXCLUDED_DIR_NAMES only prunes subtrees *inside* a scanned root, so the
    # repository-level tests/ directory is simply out of scope (never scanned).
    assert census["files_excluded"] == 0
    assert census["excluded_files"] == []

    # D1: host.py imports numpy but no device framework.  dev.py imports torch.
    assert census["numpy_only_count"] == 1
    assert census["numpy_only_modules"] == ["src/pkg/host.py"]

    # D2: host.py has .cpu() + .numpy(); dev.py has .cpu() + .numpy().
    assert census["sync_points"] == 4
    assert census["breakdown"]["sync_points"] == {"cpu": 2, "numpy": 2}
    # .cpu()/.numpy() are also synchronising but reported separately.
    assert census["breakdown"]["sync_extended"]["tolist"] == 0

    # D3: torch.ones(3) and torch.tensor([...], dtype=...) state no device;
    # torch.zeros(..., device=...) is explicit; torch.zeros_like inherits.
    assert census["no_device_constructs"] == 2
    constructs = census["breakdown"]["tensor_constructs"]
    assert constructs["device_explicit"] == 1
    assert constructs["device_inherited_like"] == 1
    assert constructs["device_unverified_spread"] == 0
    assert sorted(site["factory"] for site in census["no_device_sites"]) == ["ones", "tensor"]

    # D4: host.py for + while, dev.py for.
    assert census["host_loops"] == 3
    assert census["breakdown"]["host_loops_by_kind"]["while"] == 1
    assert census["breakdown"]["vectorizable_loops"] == 3


def test_excluded_dir_names_prune_inside_root(tmp_path):
    """tests/ and examples/ subtrees below a scanned root are host-only."""
    pkg = tmp_path / "src" / "pkg"
    (pkg / "tests").mkdir(parents=True)
    (pkg / "examples").mkdir(parents=True)
    (pkg / "core.py").write_text("import torch\n", encoding="utf-8")
    (pkg / "tests" / "t.py").write_text(HOST_ONLY_TEST, encoding="utf-8")
    (pkg / "examples" / "e.py").write_text(HOST_ONLY_TEST, encoding="utf-8")

    census = _load_tool().build_census(tmp_path, ("src",))

    assert census["files_scanned"] == 1
    assert census["excluded_files"] == ["src/pkg/examples/e.py", "src/pkg/tests/t.py"]
    assert census["ast_coverage_pct"] == 100.0


def test_unparsable_file_is_listed_in_full(fixture_repo):
    census = _load_tool().build_census(fixture_repo, ("src",))

    assert census["ast_coverage_pct"] == pytest.approx(66.67, abs=0.01)
    assert [entry["file"] for entry in census["uncovered_files"]] == ["src/pkg/broken.py"]
    reason = census["uncovered_files"][0]["reason"]
    assert reason.startswith("SyntaxError:")
    assert len(reason) <= 200


def test_two_runs_are_byte_identical(fixture_repo):
    tool = _load_tool()
    first = json.dumps(tool.build_census(fixture_repo, ("src",)), indent=2, ensure_ascii=True)
    second = json.dumps(tool.build_census(fixture_repo, ("src",)), indent=2, ensure_ascii=True)
    assert first == second
    # No absolute path / environment value may leak into the payload.
    assert str(fixture_repo) not in first


def test_cli_is_deterministic_and_cwd_independent(fixture_repo, tmp_path_factory):
    out_a = fixture_repo / "a.json"
    out_b = fixture_repo / "b.json"
    for out, cwd in ((out_a, REPO_ROOT), (out_b, tmp_path_factory.mktemp("elsewhere"))):
        subprocess.run(
            [sys.executable, str(TOOL), "--repo-root", str(fixture_repo),
             "--roots", "src", "--out", str(out), "--quiet"],
            cwd=cwd, check=True, capture_output=True,
        )
    assert out_a.read_bytes() == out_b.read_bytes()


def test_real_repository_census(fixture_repo):
    """Smoke the real tree: metrics are integers, coverage is reported honestly."""
    census = _load_tool().build_census(REPO_ROOT, ("src",))

    for key in ("numpy_only_count", "sync_points", "no_device_constructs", "host_loops"):
        assert isinstance(census[key], int) and census[key] >= 0

    scanned = census["files_scanned"]
    parsed = census["files_parsed"]
    assert scanned > 200  # the src/ tree of this repository
    assert parsed + len(census["uncovered_files"]) == scanned
    assert census["ast_coverage_pct"] == pytest.approx(100.0 * parsed / scanned, abs=0.01)
    # honest-FAIL rule: below 50 % the uncovered table must name every file.
    if census["ast_coverage_pct"] < 50.0:
        assert len(census["uncovered_files"]) == scanned - parsed
    # TensorFEM is a torch library, but a real share of modules is numpy-only.
    assert census["breakdown"]["device_module_count"] > 0
    assert census["numpy_only_count"] > 0
    assert census["sync_points"] > 0
    assert census["host_loops"] > 0