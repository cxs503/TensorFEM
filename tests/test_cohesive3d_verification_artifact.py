"""Gates on ``benchmarks/cohesive3d_verification.json`` (the 3-D cohesive evidence).

The artifact used to be a free-floating claim: nothing in the test suite tied its
numbers to the code that produced it, so ``python benchmarks/cohesive3d_verification.py``
could be re-run forever and the only thing that ever changed was the timestamp.
These tests make the artifact *checkable*:

* **schema** -- all eleven criteria are present, each carries its closed-form
  statement, and ``all_passed`` really is the conjunction of the eleven ``passed``
  flags (no silent partial pass);
* **tamper gate** -- ``checks_digest`` must hash the stored criteria, so editing a
  number by hand (or merging a partial regeneration) breaks the test without
  re-running the 30--45 s benchmark;
* **reproducibility** -- the fast closed-form checks are recomputed from the
  current checkout and compared leaf by leaf with the committed numbers within
  ``atol + rtol * max(|a|, |b|)``; this is the binding that a timestamp-only
  regeneration can never satisfy dishonestly;
* **freshness** -- the recorded SHA-256 of every source module must still match
  the checkout, i.e. the artifact cannot outlive the code it describes;
* **no host leakage** -- the artifact is repo-relative (no ``/root/...`` paths).

The expensive coupled-specimen half of the artifact is covered by
``python benchmarks/cohesive3d_verification.py --check`` (full recompute, exit
code 3 on a stale source hash); this module deliberately stays under ~10 s so it
can run in the default suite.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "cohesive3d_verification.py"
ARTIFACT = REPO / "benchmarks" / "cohesive3d_verification.json"

EXPECTED_CHECKS = (
    "law_reduces_to_2d_bk",
    "law_energy",
    "law_initiation",
    "law_tangent_vs_ad",
    "element_patch",
    "element_rigid_rotation",
    "element_mesh_objectivity",
    "coupled_block_mode_i",
    "coupled_block_mode_ii",
    "coupled_block_mesh",
    "coupled_block_energy",
)

#: The closed-form checks that cost < 10 s in total; the three coupled specimen
#: solves (~8 s each) are left to the ``--check`` CLI mode.
FAST_CHECKS = (
    "law_reduces_to_2d_bk",
    "law_energy",
    "law_initiation",
    "law_tangent_vs_ad",
    "element_patch",
    "element_rigid_rotation",
    "element_mesh_objectivity",
    "coupled_block_mesh",
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def bench():
    """The benchmark script itself, imported (it sets the global torch default to float64).

    The default dtype is restored at module teardown so that this gate cannot
    change the numerics of any other test module in the same session.
    """
    prior_dtype = torch.get_default_dtype()
    spec = importlib.util.spec_from_file_location("cohesive3d_verification", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        torch.set_default_dtype(prior_dtype)


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.exists(), f"missing artifact: run {SCRIPT.name} first"
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------- #
# Schema and honest aggregation
# --------------------------------------------------------------------------- #


def test_every_criterion_is_present_and_stated(artifact):
    criteria = artifact["criteria"]
    assert tuple(criteria) == EXPECTED_CHECKS
    assert artifact["schema"] == "tensorfem.cohesive3d-verification/1.0"
    assert artifact["generator"] == "benchmarks/cohesive3d_verification.py"
    for name, check in criteria.items():
        assert isinstance(check.get("criterion"), str) and check["criterion"], name
        assert isinstance(check.get("passed"), bool), name


def test_all_passed_is_the_conjunction_of_the_checks(artifact):
    flags = {name: check["passed"] for name, check in artifact["criteria"].items()}
    assert artifact["all_passed"] is all(flags.values())
    assert all(flags.values()), f"failing criteria: {[n for n, ok in flags.items() if not ok]}"


def test_provenance_binds_the_source_modules(artifact):
    provenance = artifact["provenance"]
    files = provenance["source_files"]
    assert set(files) == {
        "src/tensorfem/cohesive3d.py",
        "src/tensorfem/mixed_mode_cohesive.py",
        "src/tensorfem/solid3d.py",
    }
    for path, digest in files.items():
        assert HEX64.match(digest), path
    assert provenance["module"] == "tensorfem.cohesive3d"
    # a git HEAD is recorded when the artifact was generated inside a checkout
    assert provenance["git_head"] is None or re.match(r"^[0-9a-f]{40}$", provenance["git_head"])


def test_artifact_does_not_leak_host_paths(artifact):
    text = json.dumps(artifact)
    for marker in ("/root/", "/home/", "\\\\"):
        assert marker not in text, f"artifact leaks a host path via {marker!r}"


def test_reproduce_commands_are_recorded(artifact):
    reproduce = artifact["reproduce"]
    assert reproduce["write"] == "python benchmarks/cohesive3d_verification.py"
    assert reproduce["check"].endswith("--check")
    assert reproduce["runtime_s"] > 0.0


# --------------------------------------------------------------------------- #
# Tamper gate
# --------------------------------------------------------------------------- #


def test_checks_digest_hashes_the_stored_numbers(bench, artifact):
    assert artifact["checks_digest"] == bench.criteria_digest(artifact["criteria"])


def test_checks_digest_changes_when_a_single_number_is_edited(bench, artifact):
    tampered = copy.deepcopy(artifact["criteria"])
    tampered["coupled_block_mode_i"]["max_load_rel_error"] = 1e-9  # a "nicer" number
    assert bench.criteria_digest(tampered) != artifact["checks_digest"]


# --------------------------------------------------------------------------- #
# Reproducibility of the committed numbers (fast subset)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def artifact_fast(artifact) -> dict:
    """The committed numbers for exactly the checks the pytest gate re-runs."""
    return {name: artifact["criteria"][name] for name in FAST_CHECKS}


@pytest.fixture(scope="module")
def fresh_fast(bench) -> dict:
    return {name: getattr(bench, _fn_name(name))() if name != "coupled_block_mesh"
            else bench.check_coupled_mesh() for name in FAST_CHECKS}


def _fn_name(check: str) -> str:
    return {
        "law_reduces_to_2d_bk": "check_law_reduces_to_2d",
        "law_energy": "check_law_energies",
        "law_initiation": "check_law_initiation",
        "law_tangent_vs_ad": "check_law_tangent",
        "element_patch": "check_element_patch",
        "element_rigid_rotation": "check_element_rigid_rotation",
        "element_mesh_objectivity": "check_mesh_objectivity",
        "coupled_block_mesh": "check_coupled_mesh",
    }[check]


def test_committed_numbers_reproduce_from_the_current_code(bench, artifact_fast, fresh_fast):
    comparison = bench.compare_payload({"criteria": artifact_fast}, {"criteria": fresh_fast})
    assert comparison["ok"], comparison["diffs"]


def test_tolerance_absorbs_float_noise_but_not_physical_drift(bench, artifact_fast, fresh_fast):
    noisy = copy.deepcopy(fresh_fast)
    key = "coupled_block_mesh"
    noise = 1e-12 * abs(artifact_fast[key]["rows"][0]["reaction"])
    noisy[key]["rows"][0]["reaction"] += noise
    assert bench.compare_payload({"criteria": artifact_fast}, {"criteria": noisy})["ok"]

    drifted = copy.deepcopy(fresh_fast)
    drifted[key]["rows"][0]["reaction"] *= 1.01
    report = bench.compare_payload({"criteria": artifact_fast}, {"criteria": drifted})
    assert not report["ok"]
    assert report["diffs"][0]["kind"] == "numeric"


def test_a_stale_artifact_is_detected(bench, artifact, artifact_fast, tmp_path):
    """An artifact whose numbers came from other code must not pass ``--check``."""
    stale = copy.deepcopy(artifact)
    stale["criteria"]["coupled_block_mesh"]["rows"][1]["reaction"] *= 1.02  # 2 % off
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(stale))
    report = bench.check_artifact(path, fresh={"criteria": artifact_fast,
                                              "provenance": artifact["provenance"]})
    assert not report["ok"]
    assert report["exit_code"] == 1
    assert any(d["path"].endswith("coupled_block_mesh/rows/1/reaction")
               for d in report["comparison"]["diffs"])


def test_numeric_gate_has_a_documented_absolute_floor(bench, artifact_fast, fresh_fast):
    """Honest limit: the numeric gate cannot see a change below ``atol``.

    Machine-precision error metrics (~1e-16) legitimately move by more than
    ``rtol`` between architectures, so the comparison is absolute-floor-limited
    at ``atol = 1e-10``.  A residual-space edit is therefore caught by the
    ``checks_digest`` gate (byte-level), not by the numeric one -- both gates
    together cover the artifact, neither alone does.
    """
    tweaked = copy.deepcopy(fresh_fast)
    tweaked["law_reduces_to_2d_bk"]["max_rel_error"] = 1e-24  # below the atol floor
    assert bench.compare_payload({"criteria": artifact_fast},
                                 {"criteria": tweaked})["ok"]
    assert bench.criteria_digest(tweaked["law_reduces_to_2d_bk"]) != bench.criteria_digest(
        artifact_fast["law_reduces_to_2d_bk"])


def test_source_hashes_match_the_checkout(bench, artifact):
    """The artifact must describe the code that is checked out (regen in ~40 s)."""
    stale = bench.stale_sources(artifact)
    assert stale == [], (
        "cohesive3d_verification.json was generated from other module contents: "
        f"{stale}. Regenerate with `python benchmarks/cohesive3d_verification.py`."
    )


def test_missing_artifact_exits_with_two(bench, tmp_path):
    assert bench.main(["--check", "--out", str(tmp_path / "nope.json")]) == 2


def test_artifact_file_hash_matches_its_own_content():
    """Cheap integrity anchor: the file on disk is exactly the JSON it contains."""
    raw = ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(json.dumps(
        json.loads(raw), indent=2).encode()).hexdigest()