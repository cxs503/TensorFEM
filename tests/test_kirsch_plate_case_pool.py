"""CASE-FEM-KIRSCH-PLATE case-pool integration-surface gates.

The solver-side physics (Kirsch K_t = 3 against the closed form) is verified by
``tests/test_kirsch_plate_anchor.py`` and ``tests/test_kirsch_hole_reference.py``;
this module asserts the *delivery contract* the platform case-pool settlement
gate checks: the Kirsch case is reachable through the integration surface
(:func:`tensorfem.integration_callers.run_kirsch_plate_case`) and returns a
machine-checkable verdict dict with no CLI parsing, exactly like the
NAFEMS-LE11 caller.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

KT_KIRSCH = 3.0
POOL_TOLERANCE = 0.05  # pool mandate: < 5 % against the Kirsch reference


def test_integration_surface_exposes_kirsch_caller():
    """The case-pool integration surface must expose a Kirsch adapter."""
    from tensorfem.integration_callers import (
        KIRSCH_PLATE_QUANTITIES, run_kirsch_plate_case,
    )
    assert len(KIRSCH_PLATE_QUANTITIES) == 2
    names = {q[0] for q in KIRSCH_PLATE_QUANTITIES}
    assert {"K_tg", "K_tn"} <= names


def test_kirsch_caller_returns_machine_checkable_verdict():
    """Coarse mesh (same code path as the committed ladder) must PASS the pool."""
    from tensorfem.integration_callers import run_kirsch_plate_case
    out = run_kirsch_plate_case(n_boundary=12, n_radial=12, grading=6.0)
    assert out["case"] == "CASE-FEM-KIRSCH-PLATE"
    assert out["verdict"] in ("PASS", "FAIL")
    assert len(out["rows"]) == 3  # K_tg, K_tn, load_balance
    # the gross factor must already be inside the pool tolerance on the coarse mesh
    ktg = next(r for r in out["rows"] if r["quantity"] == "K_tg")
    assert ktg["ref"] == pytest.approx(KT_KIRSCH)
    assert ktg["rel_error"] < POOL_TOLERANCE
    # load balance is a statically-exact invariant
    lb = next(r for r in out["rows"] if r["quantity"] == "load_balance")
    assert lb["passed"] is True
    assert lb["rel_error"] < 1e-10


def test_kirsch_caller_converges_to_three():
    """Refining once must move K_tg monotonically toward the Kirsch limit 3."""
    from tensorfem.integration_callers import run_kirsch_plate_case
    coarse = run_kirsch_plate_case(n_boundary=12, n_radial=12, grading=6.0)
    fine = run_kirsch_plate_case(n_boundary=24, n_radial=24, grading=6.0)
    c = next(r for r in coarse["rows"] if r["quantity"] == "K_tg")["got"]
    f = next(r for r in fine["rows"] if r["quantity"] == "K_tg")["got"]
    assert abs(f - KT_KIRSCH) < abs(c - KT_KIRSCH)


def test_kirsch_caller_writes_artifact():
    """With ``output_dir`` the caller must persist a machine-checkable JSON."""
    from tensorfem.integration_callers import run_kirsch_plate_case
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = run_kirsch_plate_case(n_boundary=12, n_radial=12, grading=6.0,
                                    output_dir=d)
        assert out["artifact"] is not None
        p = Path(out["artifact"])
        assert p.is_file()
        payload = json.loads(p.read_text())
        assert payload["case"] == "CASE-FEM-KIRSCH-PLATE"
        assert payload["verdict"] == out["verdict"]


def test_registry_wires_kirsch_case():
    """The benchmark registry must carry the Kirsch case-pool evidence id."""
    reg = (ROOT / "src" / "tensorfem" / "benchmark_registry.py").read_text()
    assert re.search(r"kirsch\.(plate|hole)", reg), \
        "benchmark_registry.py must register the Kirsch case-pool evidence"
