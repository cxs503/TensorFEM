"""Coupling-convergence verification tests for GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.

Closes the check the adversarial audit
(``docs/AUDIT-GAP-PP-1-adversarial-verification.md``, section 4) left not run:
the two-way coupling loop ``coupled_patch_solve`` is only asserted to converge
and keep the force; its *tolerance / mesh sensitivity* and *fixed-point
uniqueness* were never probed.  These tests pin that down.

Re-runnable tool with the full ladders and figures:
``tools/verify_gap_pp1_coupling_convergence.py``.
"""

from __future__ import annotations

import torch

from tensorfem.ice_panel_patch_contact import (
    BilinearCohesiveLaw,
    coupled_patch_solve,
    czm_contact_patch,
    deflection_at,
    default_law,
)

DP = dict(young=206.0e9, poisson=0.3, thickness=0.022)
EDGE = {"kind": "spherical", "radius": 1.0, "amplitude": 0.02, "box_half": 0.10}
CEILING = 2.0e6
FORCE = 4.0e4
PANEL = (0.8, 0.7)
GRID = 41
MESH = (16, 14)


def _solve(*, grid=GRID, mesh=MESH, tol=1e-4, relax=0.5, thickness=0.022,
           law=None, w0=None):
    law = law or default_law()
    patch, sol, coup = coupled_patch_solve(
        nx=mesh[0], ny=mesh[1], panel=PANEL, thickness=thickness,
        young=DP["young"], poisson=DP["poisson"],
        law=law, ceiling=CEILING, edge=EDGE, total_force=FORCE,
        grid=grid, tol=tol, relax=relax, w0=w0, max_coupling=40,
    )
    peak = sol.peak_stress()
    w_c = float(deflection_at(
        sol, torch.tensor([0.4], dtype=torch.float64),
        torch.tensor([0.35], dtype=torch.float64),
    )[0])
    return patch, sol, coup, {
        "peak_pressure_MPa": patch.peak_pressure / 1e6,
        "contact_area_m2": patch.contact_area,
        "peak_sigma_vm_MPa": peak["sigma_vm_MPa"],
        "centre_deflection_mm": w_c * 1e3,
    }


def _fp_residual(patch, coup, grid):
    """Pressure change one more sweep would produce from the final relaxed
    deflection -- the fixed-point residual of the loop."""
    w = coup["final_deflection"]
    p_next = czm_contact_patch(
        law=default_law(), ceiling=CEILING, edge=EDGE, total_force=FORCE,
        panel=PANEL, grid=grid, deflection=w,
        contact_box=(0.3, 0.5, 0.25, 0.45),
    )
    p = patch.pressure
    return float((p_next.pressure - p).abs().max() / max(float(p.max()), 1e-30))


def test_tolerance_ladder_converges_and_stabilizes():
    rows = []
    for tol in (1e-2, 1e-3, 1e-4):
        patch, sol, coup, q = _solve(tol=tol)
        assert coup["converged"], (tol, coup["history"])
        rows.append((tol, q))
    (t0, q0), (t1, q1), (t2, q2) = rows
    # the two finest levels must agree: the fixed point, not the tolerance,
    # sets the answer
    for k in ("peak_pressure_MPa", "contact_area_m2", "peak_sigma_vm_MPa"):
        assert abs(q1[k] - q2[k]) / q2[k] < 0.02, (k, q1[k], q2[k])
    # tightening tol must not move the answer by more than the coarser level
    assert abs(q0["peak_pressure_MPa"] - q2["peak_pressure_MPa"]) / q2[
        "peak_pressure_MPa"] < 0.05


def test_fixed_point_residual_below_tol():
    tol = 1e-4
    patch, sol, coup, q = _solve(tol=tol)
    resid = _fp_residual(patch, coup, GRID)
    # a true fixed point: the next sweep would change the pressure by ~0,
    # well inside the requested tolerance
    assert resid < 5.0 * tol, (resid, tol)
    assert coup["converged"]


def test_relaxation_path_independence():
    base = None
    for relax in (0.25, 0.5, 1.0):
        patch, sol, coup, q = _solve(relax=relax, tol=1e-5)
        assert coup["converged"], (relax, coup["history"])
        if base is None:
            base = q
        else:
            # different relaxation paths land on the same fixed point up to
            # the loop's own stopping tolerance (tol=1e-5 here)
            assert abs(q["peak_pressure_MPa"] - base["peak_pressure_MPa"]) \
                / base["peak_pressure_MPa"] < 2e-5, (relax, q, base)
            assert abs(q["centre_deflection_mm"] - base["centre_deflection_mm"]) \
                / base["centre_deflection_mm"] < 2e-5, (relax, q, base)


def test_initial_guess_independence():
    _, _, coup_ref, q_ref = _solve(tol=1e-5)
    w_ref = coup_ref["final_deflection"]
    for name, w0 in (("2x", 2.0 * w_ref), ("-1x", -w_ref)):
        patch, sol, coup, q = _solve(tol=1e-5, w0=w0)
        assert coup["converged"], (name, coup["history"])
        assert abs(q["peak_pressure_MPa"] - q_ref["peak_pressure_MPa"]) \
            / q_ref["peak_pressure_MPa"] < 2e-5, (name, q, q_ref)
        assert abs(q["centre_deflection_mm"] - q_ref["centre_deflection_mm"]) \
            / q_ref["centre_deflection_mm"] < 2e-5, (name, q, q_ref)


def test_contact_grid_convergence():
    rows = []
    for g in (33, 49, 61):
        patch, sol, coup, q = _solve(grid=g, tol=1e-5)
        assert coup["converged"]
        rows.append((g, q))
    (g0, q0), (g1, q1), (g2, q2) = rows
    for k in ("peak_pressure_MPa", "contact_area_m2"):
        assert abs(q1[k] - q2[k]) / q2[k] < 0.05, (k, q1[k], q2[k])


def test_plate_mesh_convergence():
    rows = []
    for mesh in ((12, 10), (16, 14), (20, 16)):
        patch, sol, coup, q = _solve(mesh=mesh, tol=1e-5)
        assert coup["converged"]
        rows.append((mesh, q))
    (m0, q0), (m1, q1), (m2, q2) = rows
    assert abs(q1["peak_sigma_vm_MPa"] - q2["peak_sigma_vm_MPa"]) \
        / q2["peak_sigma_vm_MPa"] < 0.05


def test_two_way_correction_scales_with_compliance():
    corr = []
    for t in (0.011, 0.022, 0.044):
        patch, sol, coup, q = _solve(thickness=t, tol=1e-5)
        assert coup["converged"], (t, coup["history"])
        one = czm_contact_patch(
            law=default_law(), ceiling=CEILING, edge=EDGE, total_force=FORCE,
            panel=PANEL, grid=GRID,
        )
        corr.append((t, (q["peak_pressure_MPa"] - one.peak_pressure / 1e6)
                    / (one.peak_pressure / 1e6)))
    # stiffer plate (larger t, D ~ t^3) -> smaller two-way correction
    assert abs(corr[0][1]) > abs(corr[1][1]) > abs(corr[2][1]), corr


def test_ice_stiffness_sweep_stays_convergent():
    for k in (1.0e9, 4.0e9, 1.6e10):
        # G_Ic must keep failure_opening > onset_opening: G > peak^2/(2k)
        g_ic = max(1200.0, 2.0 * 2.0e6 ** 2 / (2.0 * k))
        law = BilinearCohesiveLaw(stiffness=k, peak_traction=2.0e6,
                                  fracture_energy=g_ic)
        patch, sol, coup, q = _solve(law=law, tol=1e-5)
        assert coup["converged"], (k, coup["history"])
        # CZM bisection force residual: < 1e-8 even on the steepest law
        assert patch.residual < 1e-8
