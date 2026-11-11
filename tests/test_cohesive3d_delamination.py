"""Regression gates for the specimen-level 3-D cohesive benchmark.

``benchmarks/cohesive3d_delamination.py`` carries the *precision* evidence (it
runs the declared ``MESH`` and stores ``benchmarks/cohesive3d_delamination.json``
plus the figure).  This module is the CI-sized gate on top of it: it re-runs the
same code paths on a coarse mesh with the pre-declared ``QUICK_TOL`` bands, plus
the three structural facts whose violation produced the defect this suite exists
for (see ``docs/cohesive3d-delamination-verification.md``):

* the DCB must be clamped at the root and loaded at the mouth -- clamping the
  mouth and loading the bonded root makes the specimen 4 orders of magnitude
  too stiff (C = 9e-10 m/N instead of 1.6e-5 m/N);
* the ENF pre-crack faces must be unilateral -- without contact the arms
  interpenetrate (5.4 um measured) and the specimen is ~24 % too compliant;
* the prescribed mouth DOFs must leave the unknown set, otherwise the Newton
  solve relaxes them and the cohesive zone never receives any load.

Every assertion is a statement about a closed form or about the *structure* of
the solution, never a snapshot of a previously computed number.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "cohesive3d_delamination.py"
EVIDENCE = ROOT / "benchmarks" / "cohesive3d_delamination.json"


def load_benchmark():
    import sys

    spec = importlib.util.spec_from_file_location("cohesive3d_delamination", BENCH)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve ``cls.__module__`` through sys.modules while the module
    # body executes, so register it before exec_module
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_MODULE = None


def benchmark():
    """Cached import of the benchmark script (it is not an importable package)."""
    global _MODULE
    if _MODULE is None:
        _MODULE = load_benchmark()
    return _MODULE


# ---------------------------------------------------------------------------
# linear specimen routes
# ---------------------------------------------------------------------------
def test_linear_routes_pass_quick_bands():
    B = benchmark()
    """DCB/ENF compliance and SERR on the coarse mesh, within the declared bands."""
    criteria, detail = B.linear_criteria(B.QUICK_MESH, B.QUICK_TOL)
    failed = {k: v for k, v in criteria.items() if not v["passed"]}
    assert not failed, failed
    # the shipped (inverted) rig missed these two by two orders of magnitude
    assert criteria["dcb_linear_compliance"]["rel_error"] < 0.10
    assert criteria["enf_linear_compliance"]["rel_error"] < 0.20
    # the SERR routes must be energy-consistent with the compliance derivative
    assert criteria["dcb_linear_serr"]["got"] == pytest.approx(
        detail["dcb"]["G_I_fem"], rel=1e-12
    )


def test_dcb_rig_is_not_the_rigid_one():
    B = benchmark()
    """Clamp at the root / load at the mouth: the beams must actually bend."""
    mesh = B.QUICK_MESH
    c_fem, _ = B.dcb_compliance_fe(mesh, B.CRACK_DCB)
    c_ref = B.dcb_compliance(B.CRACK_DCB)
    assert 0.75 * c_ref < c_fem < 1.25 * c_ref
    # the inverted rig (clamped mouth, loaded bonded root) gave C = 9.0e-10 m/N
    assert c_fem > 1.0e-6


def test_enf_precrack_faces_are_unilateral():
    B = benchmark()
    """Contact is what makes the ENF a mode II test; without it the arms penetrate."""
    mesh = B.QUICK_MESH
    c_contact, _, pen = B.enf_compliance_fe(mesh, B.CRACK_ENF)
    assert pen < 1.0e-8  # metres

    spec = B.build_specimen(mesh["nx"], mesh["nz"], B.CRACK_ENF)
    model = B.InterfaceModel(spec, B.make_law(), with_contact=False)
    fixed, load_z = B.enf_rig(spec)
    fext = np.zeros(spec.n_dofs)
    fext[load_z] = -1.0 / load_z.size
    u, _ = B.linear_solution(model, fixed, fext)
    c_free = float(-u[torch.from_numpy(load_z)].mean())
    # the free-face solution interpenetrates and is ~24 % more compliant
    pairs = spec.mouth_pairs()
    gap = (u[torch.from_numpy(pairs[:, 0])]
           - u[torch.from_numpy(pairs[:, 1])]).numpy()
    assert gap.min() < -1.0e-7
    assert c_free > 1.15 * c_contact
    assert c_contact < c_free


# ---------------------------------------------------------------------------
# cohesive-zone propagation machinery
# ---------------------------------------------------------------------------
def test_cohesive_zone_machinery():
    B = benchmark()
    """Newton must converge through a growing damage front on a coarse mesh."""
    r = B.simulate_dcb_czm(B.QUICK_CZM_MESH, d_max=B.QUICK_CZM_D_MAX,
                           n_steps=B.QUICK_CZM_STEPS)
    assert r["converged_all"], r["curve"][-3:]
    peak_damage = max(c["max_damage"] for c in r["curve"])
    assert 0.0 < peak_damage < 1.0
    assert r["peak_load"] > 0.0
    # the prescribed mouth DOFs must not be unknowns: with them free the load
    # collapses to ~0 (the old defect) while the closed-form opening load here is
    # of order 10 N
    assert r["peak_load"] > 1.0
    # the load must grow monotonically while the ligament is still elastic enough
    loads = [c["load"] for c in r["curve"]]
    assert loads[0] < loads[-1]
    onset = B.dcb_onset_load()
    assert r["peak_load"] < 2.0 * onset  # coarse mesh: no wild overshoot


# ---------------------------------------------------------------------------
# stored evidence
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not EVIDENCE.exists(), reason="artifact not generated")
def test_stored_artifact_is_self_consistent():
    B = benchmark()
    out = json.loads(EVIDENCE.read_text())
    assert out["all_passed"] is True, out["criteria"]
    assert set(out["criteria"]) == {
        "dcb_linear_compliance", "dcb_linear_serr",
        "enf_linear_compliance", "enf_linear_serr",
    }
    # references are recomputed from the closed forms, not trusted from the JSON
    refs = {
        "dcb_linear_compliance": B.dcb_compliance(B.CRACK_DCB),
        "dcb_linear_serr": B.dcb_g(1.0, B.CRACK_DCB),
        "enf_linear_compliance": B.enf_compliance(B.CRACK_ENF),
        "enf_linear_serr": B.enf_g(1.0, B.CRACK_ENF),
    }
    for key, ref in refs.items():
        crit = out["criteria"][key]
        assert crit["ref"] == pytest.approx(ref, rel=1e-12)
        assert crit["rel_error"] == pytest.approx(
            abs(crit["got"] - crit["ref"]) / crit["ref"], rel=1e-9
        )
        assert crit["passed"]
    # the declared model must be the one the artifact was produced with
    assert out["model"]["G_Ic"] == B.GIC
    assert out["model"]["a_dcb_m"] == B.CRACK_DCB
    assert out["model"]["a_enf_m"] == B.CRACK_ENF
    assert out["mesh"]["nx"] * out["mesh"]["nz"] > 0
    assert math.isclose(out["model"]["process_zone_m"],
                        B.E_ADH * B.GIC / B.TN0**2, rel_tol=1e-12)
    assert "dcb_cohesive_zone_propagation" in out["diagnostics"]