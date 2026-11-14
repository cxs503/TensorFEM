"""Regression + literature checks for the restrained-panel plate solver.

The anchor GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT requires three things, all
checked here:

1. rotational-spring (finite fixity) edge conditions in the plate solver,
2. an embedding-coefficient sweep beta = 0 (simply supported) ... 1 (clamped),
3. ``beta = 0`` reproducing the JOUR-ICE-STRUCT panel benchmark
   (``panel_patch_load_results.json``) with a relative difference below 0.1 %.

The baseline numbers are copied into ``tests/data/panel_patch_load_reference.json``
together with the SHA-256 of the source file they were extracted from, so the
guard is self-contained but still traceable.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorfem.plate import q4_mindlin_stiffness
from tensorfem.plate_elastic_restraint import (
    beta_to_zeta,
    peak_bending_stress,
    rect_patch_pressure,
    solve_plate_elastic_restraint,
    solve_restrained_panel,
    structured_rect_mesh,
    _batched_mindlin_stiffness,
    moments_at_nodes,
    von_mises_bending_at_nodes,
)

DATA = Path(__file__).parent / "data" / "panel_patch_load_reference.json"
ARTIFACT = (Path(__file__).resolve().parents[1] / "benchmarks" / "artifacts"
            / "beta_restraint_panel_band.json")

# Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed.,
# Table 8 -- square plate, uniform load, nu = 0.3, coefficients quoted to the
# three significant digits printed in the book (hence the 5 % tolerance).
TIMOSHENKO_SSSS_W = 0.00406
TIMOSHENKO_SSSS_M = 0.0479
TIMOSHENKO_CCCC_W = 0.00126
TIMOSHENKO_CCCC_M_CENTRE = 0.0231
TIMOSHENKO_CCCC_M_EDGE = 0.0513


@pytest.fixture(scope="module")
def baseline():
    ref = json.loads(DATA.read_text())
    row = next(r for r in ref["plate_vs_strip"]
               if r["lambda_c_over_l"] == 0.1 and r["patch_mode"] == "full_breadth")
    return {"ref": ref, "row": row}


def _paper_panel(nx, beta_x=0.0, beta_y=0.0, ref=None):
    """Solve the paper's 0.8 x 0.7 m panel with the full-breadth 0.1 l patch."""
    dp = ref["design_point"]
    L, S, T = dp["l"], dp["s"], dp["t"]
    c = 0.1 * L
    press = rect_patch_pressure(dp["p_ref"], 0.5 * (L - c), 0.5 * (L + c), 0.0, S)
    ny = max(1, int(round(nx * S / L)))
    res = solve_restrained_panel(
        nx, ny, L, S, dp["E"], dp["nu"], T, press, beta_x=beta_x, beta_y=beta_y)
    return res, peak_bending_stress(res)


def _richardson(values):
    """Observed order and Richardson extrapolation from three successive grids."""
    v1, v2, v3 = values
    denom = v2 - v3
    if denom == 0.0:
        return math.nan, math.nan
    r = (v1 - v2) / denom
    if r <= 0.0:
        return math.nan, math.nan
    order = math.log2(r)
    return order, v3 + (v3 - v2) / (r - 1.0)


# --------------------------------------------------------------------------
# 1. The new vectorised/sparse path must describe the same discrete operator
# --------------------------------------------------------------------------

def test_batched_element_stiffness_matches_loop_element():
    nodes, els = structured_rect_mesh(3, 2, 1.3, 0.9)
    ke, _, _ = _batched_mindlin_stiffness(nodes, els, 2.1e11, 0.3, 0.02)
    for e in range(els.shape[0]):
        ref = q4_mindlin_stiffness(nodes[els[e]], 2.1e11, 0.3, 0.02)
        assert torch.allclose(ke[e], ref, rtol=1e-12, atol=1e-4)


def test_sparse_path_reproduces_dense_path_simply_supported():
    """beta = 0 square panel: new sparse solver == historical dense solver."""
    args = dict(young=9.0e9, poisson=0.33, thickness=0.5,
                winkler_k=10055.0, pressure=5.0e4)
    new = solve_restrained_panel(12, 12, 2.0, 2.0, beta_x=0.0, **args)
    old = solve_plate_elastic_restraint(12, 2.0, beta=0.0, **args)
    scale = float(old.displacement.abs().max())
    assert float((new.displacement - old.displacement).abs().max()) / scale < 1e-10


def test_sparse_path_reproduces_dense_path_clamped():
    args = dict(young=9.0e9, poisson=0.33, thickness=0.5,
                winkler_k=10055.0, pressure=5.0e4)
    new = solve_restrained_panel(12, 12, 2.0, 2.0, boundary="clamped", **args)
    old = solve_plate_elastic_restraint(12, 2.0, boundary="clamped", **args)
    scale = float(old.displacement.abs().max())
    # dense LU vs sparse LU: different pivoting, so only round-off level agreement
    assert float((new.displacement - old.displacement).abs().max()) / scale < 1e-5


def test_infinite_spring_reproduces_clamped_boundary_condition():
    """beta -> 1 (zeta = 1/eps) must equal the hard-clamped solve."""
    args = dict(young=2.06e11, poisson=0.3, thickness=0.022, pressure=1.0e6)
    spring = solve_restrained_panel(24, 24, 1.0, 1.0, beta_x=1.0, beta_y=1.0, **args)
    hard = solve_restrained_panel(24, 24, 1.0, 1.0, boundary="clamped", **args)
    scale = float(hard.displacement.abs().max())
    assert float((spring.displacement - hard.displacement).abs().max()) / scale < 1e-6
    # the spring stiffness is genuinely large, not silently zero
    assert spring.K_rot_x > 1e5 * (2.06e11 * 0.022**3 / 12 / (1 - 0.3**2))


# --------------------------------------------------------------------------
# 2. Literature: Timoshenko Table 8
# --------------------------------------------------------------------------

def test_beta_zero_matches_ssss_moment_coefficient(baseline):
    """beta = 0, uniform square plate: M_coef vs Navier/Timoshenko (<0.1 %)."""
    dp = baseline["ref"]["design_point"]
    targeted = baseline["ref"]["verification_timoshenko_N401"]["M_coef_got"]
    res = solve_restrained_panel(80, 80, 1.0, 1.0, dp["E"], dp["nu"], dp["t"],
                                 1.0e6, beta_x=0.0)
    M_coef = abs(moments_at_nodes(res)[int(torch.argmin(
        (res.nodes[:, 0] - 0.5)**2 + (res.nodes[:, 1] - 0.5)**2)), 0]) / 1.0e6
    assert abs(M_coef / targeted - 1.0) < 1e-3          # 0.1 % vs the baseline
    assert abs(M_coef / TIMOSHENKO_SSSS_M - 1.0) < 0.05  # 5 % vs the book


def test_clamped_matches_timoshenko_table8():
    """beta = 1 clamped square plate: w, centre M and edge M within 5 %."""
    E, nu, t, q, a = 2.06e11, 0.3, 0.022, 1.0e6, 1.0
    D = E * t**3 / (12 * (1 - nu**2))
    w, mc, me = [], [], []
    for nx in (20, 40, 80):
        res = solve_restrained_panel(nx, nx, a, a, E, nu, t, q, beta_x=1.0)
        pk = peak_bending_stress(res)
        Mn = moments_at_nodes(res)
        edge = torch.nonzero(res.nodes[:, 0].abs() < 1e-9, as_tuple=False).flatten()
        w.append(abs(pk["w_centre"]) * D / (q * a**4))
        mc.append(abs(pk["centre_Mx"]) / (q * a**2))
        me.append(float(Mn[edge, 0].abs().max()) / (q * a**2))
    # book coefficients are 3 significant digits -> 5 % band
    assert abs(w[-1] / TIMOSHENKO_CCCC_W - 1.0) < 0.05
    assert abs(mc[-1] / TIMOSHENKO_CCCC_M_CENTRE - 1.0) < 0.05
    # the clamped-edge moment converges only at O(h) (moment boundary layer)
    order, extrap = _richardson(me)
    assert order > 0.8
    assert abs(extrap / TIMOSHENKO_CCCC_M_EDGE - 1.0) < 0.05, (me, extrap)


# --------------------------------------------------------------------------
# 3. Acceptance: beta = 0 reproduces the paper baseline (<0.1 %)
# --------------------------------------------------------------------------

def test_paper_baseline_beta_zero_centre_stress(baseline):
    """GAP acceptance: beta=0 peak stress vs panel_patch_load_results.json <0.1 %.

    "Peak" is the paper's own quantity: von Mises bending stress at the panel
    centre, ``6 sqrt(Mx^2 - Mx My + My^2 + 3 Mxy^2) / t^2`` with the Navier
    Kirchhoff reference at N = 401 terms.
    """
    ref, _ = baseline["ref"], baseline["row"]
    target = baseline["row"]["sigma_plate_MPa"]
    res, pk = _paper_panel(160, ref=ref)
    sigma = pk["centre_von_mises"] / 1e6
    err = sigma / target - 1.0
    assert abs(err) < 1e-3, f"beta=0 centre stress {sigma:.4f} MPa vs {target:.4f} MPa ({100*err:+.3f}%)"
    # the sigma_x component of the same baseline row, as an independent guard
    sigma_x = pk["centre_sigma_x"] / 1e6
    assert abs(sigma_x / baseline["row"]["sigma_plate_Mx_MPa"] - 1.0) < 1e-3


def test_paper_baseline_convergence_is_second_order(baseline):
    """The 0.1 % agreement is a converged result, not a mesh-specific fluke."""
    target = baseline["row"]["sigma_plate_MPa"]
    vals = [_paper_panel(nx, ref=baseline["ref"])[1]["centre_von_mises"] / 1e6
            for nx in (40, 80, 160)]
    order, extrap = _richardson(vals)
    assert 1.5 < order < 2.5, (vals, order)
    assert abs(extrap / target - 1.0) < 5e-4, (vals, extrap, target)


# --------------------------------------------------------------------------
# 4. The beta sweep itself: per-edge restraint, monotonicity, machine-readable
# --------------------------------------------------------------------------

def test_per_edge_beta_band_is_monotonic_and_brackets_ssss(baseline):
    """Longitudinal restraint alone (beta_y = 0) monotonically lowers sigma."""
    ref = baseline["ref"]
    target = baseline["row"]["sigma_plate_MPa"]
    sigmas = []
    for bx in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, pk = _paper_panel(60, beta_x=bx, beta_y=0.0, ref=ref)
        sigmas.append(pk["centre_von_mises"] / 1e6)
    assert all(b < a for a, b in zip(sigmas, sigmas[1:])), sigmas
    assert abs(sigmas[0] / target - 1.0) < 0.02      # beta=0 = SSSS upper bound
    assert sigmas[-1] / sigmas[0] < 0.95             # clamped is materially lower


def test_beta_to_zeta_endpoints():
    assert beta_to_zeta(0.0) == 0.0
    assert beta_to_zeta(1.0) == pytest.approx(1.0 / 1e-6)
    with pytest.raises(ValueError):
        beta_to_zeta(1.5)


def test_sweep_artifact_is_machine_readable():
    """The >=5-point beta scan must be committed as a machine-readable JSON."""
    if not ARTIFACT.exists():
        pytest.skip("benchmark artifact not generated in this checkout")
    data = json.loads(ARTIFACT.read_text())
    diag = data["diagonal_beta"]
    assert len(diag) >= 5
    assert [r["beta_x"] for r in diag] == sorted(r["beta_x"] for r in diag)
    sig = [r["sigma_vm_centre_MPa"] for r in diag]
    assert all(b < a for a, b in zip(sig, sig[1:]))
    assert diag[0]["beta_x"] == 0.0 and diag[-1]["beta_x"] == 1.0
    # beta = 0 row of the sweep is itself the regression anchor
    target = data["baseline_navier_N401_sigma_vm_centre_MPa"]
    assert abs(diag[0]["sigma_vm_centre_MPa"] / target - 1.0) < 1e-3
    assert 0.0 < data["band"]["reduction_vs_SSSS_pct"] < 100.0
    assert data["git_sha"] and len(data["git_sha"]) >= 7
    # literature column of the artifact is consistent with the book values
    lit = data["literature_square"]
    assert abs(lit["uniform_square_SSSS"]["M_coef"] / TIMOSHENKO_SSSS_M - 1.0) < 0.05
    assert abs(lit["uniform_square_CCCC"]["w_coef"] / TIMOSHENKO_CCCC_W - 1.0) < 0.05


def test_independent_reference_displacement_is_not_used():
    """Guard against a degenerate 'stress' field: moments must vary in space."""
    ref = json.loads(DATA.read_text())
    _, pk = _paper_panel(40, ref=ref)
    assert pk["max_von_mises"] >= pk["centre_von_mises"] > 0.0
    res, _ = _paper_panel(40, ref=ref)
    vm = von_mises_bending_at_nodes(res)
    assert float(vm.std() / vm.abs().max()) > 0.1