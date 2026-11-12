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
    assert criteria["dcb_linear_compliance"]["rel_error"] < 0.08
    assert criteria["enf_linear_compliance"]["rel_error"] < 0.08
    # the SERR routes must be energy-consistent with the compliance derivative
    assert criteria["dcb_linear_serr"]["got"] == pytest.approx(
        detail["dcb"]["G_I_fem"], rel=1e-12
    )


def test_reference_uses_the_strip_modulus():
    B = benchmark()
    """Material control (nu = 0 vs 0.33) on the same mesh and geometry.

    A wide strip restrained against transverse strain is a plane-strain
    problem: the 3-D solution must follow ``E/(1-nu^2)``.  At ``nu = 0`` the two
    moduli coincide, so the FE compliance must then match the closed form; at
    ``nu = 0.33`` the *plane-stress* closed form is missed by O(10 %) while the
    strip form is not.  This is what justifies ``E_STRIP`` in the reference --
    without it the criterion would be measuring the specimen's 3-D restraint
    instead of the cohesive element.
    """
    mesh = B.QUICK_MESH
    a = B.CRACK_ENF

    def c_of(nu):
        spec = B.build_specimen(mesh["nx"], mesh["nz"], a)
        model = B.InterfaceModel(spec, B.make_law(), nu=nu, with_contact=True)
        fixed, load_z = B.enf_rig(spec)
        fext = np.zeros(spec.n_dofs)
        fext[load_z] = -1.0 / load_z.size
        u, _ = B.linear_solution(model, fixed, fext)
        return float(-u[torch.from_numpy(load_z)].mean())

    c0, c33 = c_of(0.0), c_of(0.33)
    ref_plane_stress = B.enf_compliance(a, e=B.E_ADH)
    ref_strip = B.enf_compliance(a)  # E_STRIP

    # nu = 0: both references coincide, the FE tracks the beam closed form
    assert abs(c0 - ref_plane_stress) / ref_plane_stress < 0.06
    # nu = 0.33: the plane-stress form is off by the 3-D restraint factor
    assert abs(c33 - ref_plane_stress) / ref_plane_stress > 0.06
    assert abs(c33 - ref_strip) / ref_strip < 0.06
    # and the shift is the Poisson factor, 1 - nu^2, to within the mesh budget
    assert c33 / c0 == pytest.approx(1.0 - B.NU_ADH**2, rel=0.05)


def test_dcb_root_offset_is_a_constant_crack_extension():
    B = benchmark()
    """ASTM D5528 data reduction: da from two crack lengths predicts a third.

    The DCB closed form assumes the arm root at the crack tip is built in.  It is
    not: the bonded ligament rotates, which is equivalent to a constant apparent
    crack extension ``da`` (measured here as a fraction of the arm thickness).
    Reducing ``da`` at a = 30/60 mm and *predicting* the compliance at the working
    length a = 45 mm is a falsifiable statement about that model, so it is used as
    a criterion rather than a tuned reference.
    """
    ro = B.root_offset(B.QUICK_MESH, 30.0e-3, 60.0e-3)
    # the reduction is a difference of two compliances, so on QUICK_MESH
    # (dx = 1 mm, ~4 % discretisation error) the offset itself is only resolved
    # to a fraction of an element; what must hold is that it stays a small
    # correction and that it still predicts the working crack length.  The
    # sharp ``da/h`` range is asserted on the declared mesh, in the stored
    # artifact test.
    assert 0.0 < ro["da_m"] < 0.2 * 30.0e-3, ro
    assert ro["prediction_rel_error"] < B.TOL_QUICK_ROOT_OFFSET, ro


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
        "dcb_root_offset_prediction",
    }
    # references are recomputed from the closed forms, not trusted from the JSON
    # the DCB reference carries the root offset the artifact reduced at run time
    da = out["criteria"]["dcb_root_offset_prediction"]["da_m"]
    refs = {
        "dcb_linear_compliance": B.dcb_compliance(B.CRACK_DCB, da=da),
        "dcb_linear_serr": B.dcb_g(1.0, B.CRACK_DCB, da=da),
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
    # reference conventions are part of the evidence: plane-strain strip modulus
    # and the DCB root-flexibility offset
    assert math.isclose(out["model"]["E_strip_Pa"],
                        B.E_ADH / (1.0 - B.NU_ADH**2), rel_tol=1e-12)
    assert math.isclose(out["model"]["dcb_root_offset_m"],
                        B.ROOT_OFFSET_H * B.H_ARM, rel_tol=1e-12)
    ro = out["criteria"]["dcb_root_offset_prediction"]
    assert ro["passed"] and ro["rel_error"] < B.TOL_DCB_ROOT_OFFSET
    assert 0.3 < ro["da_over_h"] < 0.9

    assert math.isclose(out["model"]["WIDTH_m"], B.WIDTH, rel_tol=1e-12)
    assert "dcb_cohesive_zone_propagation" in out["diagnostics"]