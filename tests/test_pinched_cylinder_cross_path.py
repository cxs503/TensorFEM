"""Cross-path agreement of the pinched-cylinder case (known-gap pin).

The same benchmark case is solved in this tree by two independent code paths:

``A`` ``tensorfem.pinched_cylinder`` (anchor certificate + case driver)
``B`` ``tensorfem.shell_benchmark_suite`` (main-solver case registry)

They do not agree: at 4x4 path A reads -2.656e-6 in (85.4 % error) while path B
reads -1.773e-5 in (2.86 % error), a factor 6.7.  The engine of the comparison
and the full table is ``benchmarks/verified/fem_pinched_cylinder/cross_path_audit.py``
/ ``cross_path_audit.json``; these tests keep the finding executable:

* the *gap* is pinned with a strict ``xfail``: the day the two paths agree the
  test starts failing as XPASS, i.e. the pin has to be removed deliberately
  rather than being forgotten;
* the *alignment* that does hold is asserted normally: the committed case
  evidence reproduces path B, not path A, so no reader can be told that the
  0.167 % reading at 16x16 comes from the code path the anchor certifies.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfem.benchmark_cases import run_case
from tensorfem.pinched_cylinder import solve_pinched_cylinder_sparse
from tensorfem.shell_benchmark_suite import pinched_cylinder_case

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "verified" / "fem_pinched_cylinder"


def _path_b() -> dict[int, float]:
    return {r["mesh"]: r["computed"] for r in run_case(pinched_cylinder_case())["results"]}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known gap: the anchor path and the main-solver path do not solve the same "
        "discrete model (4x4 disagree by a factor 6.7); see cross_path_audit.json"
    ),
)
def test_two_paths_agree_on_shared_rungs():
    shared = {3: None, 4: None}
    path_b = _path_b()
    for n in shared:
        a = solve_pinched_cylinder_sparse(n).displacement
        assert abs(a / path_b[n] - 1.0) < 1e-9, f"n={n}: anchor {a:.6e} vs main solver {path_b[n]:.6e}"


def test_committed_case_evidence_reproduces_the_main_solver_path():
    """The committed ladder is a path-B ladder; that must stay explicit."""
    path_b = _path_b()
    committed = json.loads((CASE / "case_result.json").read_text())
    compared = 0
    for level in committed["levels"]:
        n = level["n"]
        if n not in path_b:
            continue
        compared += 1
        assert abs(level["displacement_in"] / path_b[n] - 1.0) < 1e-9, (
            f"rung {n}x{n}: committed {level['displacement_in']:.9e} vs main solver {path_b[n]:.9e}"
        )
    assert compared >= 3, "no rung was comparable -- the case evidence changed shape"


def test_cross_path_audit_artifact_matches_its_script_contract():
    """The committed audit is self-describing and still records the gap."""
    audit = json.loads((CASE / "cross_path_audit.json").read_text())
    assert audit["schema"] == "tensorfem.cross-path-audit/1"
    assert audit["verdict"]["agreement"] is False
    assert audit["verdict"]["max_disagreement"] > 0.5
    assert audit["verdict"]["committed_evidence_matches"] == "path_b"
    assert audit["rows"] and all(r["ratio_b_over_a"] > 1.0 for r in audit["rows"])
    assert audit["falsified_hypothesis"]["verdict"].startswith("falsified")