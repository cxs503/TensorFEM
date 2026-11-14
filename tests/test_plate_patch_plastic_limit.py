"""Tests for the plate-panel patch-load elastoplastic capacity module.

Anchor ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET``: (1) elastoplastic plate with
isotropic hardening and a low-temperature flow stress, (2) permanent-deflection
post-processing, (3) consistency regression against the elastic solution on the
same mesh.

The closed-form references (Navier series, yield-line upper bound, strip lower
bound) are literature-anchored and tested first; the finite element tests then
pin the elastoplastic path against those references *and* against the same-mesh
elastic response, because the layered Shell4 element carries a known elastic
bias that is quantified in the module docstring and in the case document.
"""
import math
import pytest
import torch

from tensorfem.plate_patch_plastic_limit import (
    STEEL_GRADES,
    analytic_first_yield_pressure,
    elastic_patch_plate,
    flow_stress_table,
    low_temperature_flow_stress,
    patch_plate_mesh,
    plastic_moment,
    run_patch_plastic_path,
    strip_lower_bound,
    surface_extrapolated_mises,
    yield_line_patch_upper_bound,
    _patch_mechanism_volume_ratio,
    _patch_mechanism_volume_ratio_quadrature,
)
from tensorfem.layered_shell4_plasticity import (
    LayeredShell4State,
    solve_layered_shell4_increment,
)

A = B = 1.0
T = 0.02
E, NU, FY = 210e9, 0.3, 355e6
PATCH = (0.5 * A, 0.5 * B)
M_P = plastic_moment(T, FY)


# ---------------------------------------------------------------------------
# closed-form references
# ---------------------------------------------------------------------------

def test_plastic_moment_is_quarter_sigma_t_squared():
    assert plastic_moment(0.02, 355e6) == pytest.approx(355e6 * 0.02 ** 2 / 4)
    with pytest.raises(ValueError):
        plastic_moment(0.0, 355e6)


def test_yield_line_uniform_load_recovers_classical_24mp_over_a2():
    result = yield_line_patch_upper_bound(A, B, T, FY, (A, B))
    assert result["pressure"] == pytest.approx(24 * M_P / A ** 2, rel=1e-12)
    assert result["mechanism_volume"] == pytest.approx(A * B / 3, rel=1e-12)


def test_yield_line_closed_form_matches_quadrature():
    for patch in ((A, B), (0.5 * A, 0.5 * B), (0.25 * A, 0.5 * B), (0.1 * A, 0.1 * B)):
        exact = _patch_mechanism_volume_ratio(A, B, patch)
        numeric = _patch_mechanism_volume_ratio_quadrature(A, B, patch, n=200)
        assert exact == pytest.approx(numeric, rel=2e-2), patch


def test_patch_upper_bound_exceeds_lower_bound_and_grows_with_concentration():
    previous = None
    for fraction in (1.0, 0.5, 0.25):
        patch = (fraction * A, fraction * B)
        upper = yield_line_patch_upper_bound(A, B, T, FY, patch)["pressure"]
        lower = strip_lower_bound(A, B, T, FY, patch)["pressure"]
        assert upper > lower > 0.0
        if previous is not None:
            assert upper > previous
        previous = upper


def test_navier_uniform_load_matches_timoshenko_coefficients():
    q = 1e5
    result = elastic_patch_plate(A, B, T, E, NU, (A, B), q)
    D = E * T ** 3 / (12 * (1 - NU ** 2))
    assert result["deflection"][0] == pytest.approx(0.00406 * q * A ** 4 / D, rel=1e-2)
    assert result["Mx"][0] == pytest.approx(0.0479 * q * A ** 2, rel=1e-2)


def test_navier_patch_load_tends_to_the_point_load_limit():
    """Small patch -> Timoshenko's concentrated-load coefficient 0.0116."""
    D = E * T ** 3 / (12 * (1 - NU ** 2))
    fraction = 0.05
    pressure = 1.0 / (fraction * A * fraction * B)     # unit total load
    w = elastic_patch_plate(A, B, T, E, NU, (fraction * A, fraction * B), pressure)["deflection"][0]
    assert w * D / (1.0 * A ** 2) == pytest.approx(0.0116, rel=2e-2)


def test_first_yield_pressure_scales_with_strength_and_thickness_squared():
    base = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=9, terms=21)["pressure"]
    strong = analytic_first_yield_pressure(A, B, T, E, NU, 2 * FY, PATCH, grid=9, terms=21)["pressure"]
    thick = analytic_first_yield_pressure(A, B, 1.5 * T, E, NU, FY, PATCH, grid=9, terms=21)["pressure"]
    assert strong == pytest.approx(2 * base, rel=1e-9)
    assert thick == pytest.approx(base * 1.5 ** 2, rel=1e-9)   # sigma ~ p/t^2


# ---------------------------------------------------------------------------
# low-temperature flow stress
# ---------------------------------------------------------------------------

def test_flow_stress_reference_and_amplification():
    room = low_temperature_flow_stress("DH36", 20.0)
    assert room.yield_stress == pytest.approx(355e6)
    assert room.ductile and room.amplification == pytest.approx(1.0)
    arctic = low_temperature_flow_stress("DH36", -40.0)
    assert arctic.yield_stress > room.yield_stress
    assert arctic.amplification == pytest.approx(1.08, abs=0.02)
    monotone = [low_temperature_flow_stress("DH36", T_).yield_stress
                for T_ in (20.0, 0.0, -20.0, -40.0, -60.0)]
    assert all(x < y for x, y in zip(monotone, monotone[1:]))


def test_flow_stress_withholds_plasticity_below_dbtt():
    brittle = low_temperature_flow_stress("DH36", -80.0)
    assert brittle.ductile is False
    assert brittle.yield_stress == pytest.approx(355e6)   # no extrapolation
    assert "DBTT" in brittle.note
    with pytest.raises(KeyError):
        low_temperature_flow_stress("NOT-A-GRADE", 0.0)
    assert set(STEEL_GRADES) >= {"AH36", "DH36", "EH36", "FH36"}


def test_flow_stress_table_is_monotone():
    table = flow_stress_table("EH36", [20.0, -20.0, -40.0])
    assert [row["temperature_c"] for row in table] == [20.0, -20.0, -40.0]
    assert table[0]["yield_stress_MPa"] == pytest.approx(355.0)


# ---------------------------------------------------------------------------
# mesh contract
# ---------------------------------------------------------------------------

def test_mesh_rejects_patch_that_does_not_follow_element_edges():
    with pytest.raises(ValueError):
        patch_plate_mesh(A, B, T, 6, 6, young=E, poisson=NU, yield_stress=FY,
                         hardening=2e9, patch=(0.37 * A, 0.5 * B))
    with pytest.raises(ValueError):
        run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                               hardening=2e9, patch=PATCH, load_factors=[1e5],
                               nx=5, ny=5)


def test_patch_pressure_forces_carry_the_right_total_load():
    model, info = patch_plate_mesh(A, B, T, 6, 6, young=E, poisson=NU,
                                   yield_stress=FY, hardening=2e9, patch=PATCH)
    assert float(info["reference_load"].sum()) == pytest.approx(PATCH[0] * PATCH[1])
    assert int((info["reference_load"] != 0).sum()) == 3 * 3   # 3x3 nodes of the 0.5a patch


# ---------------------------------------------------------------------------
# finite element path (single small mesh, shared fixture)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def capacity():
    """One overloaded-and-unloaded path plus one elastic probe (small mesh)."""
    py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=11, terms=31)
    steps = [factor * py["pressure"] for factor in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8)]
    path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                  hardening=2e10, patch=PATCH, load_factors=steps,
                                  nx=6, ny=6, layers=3, unload_steps=3)
    probe = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                   hardening=2e10, patch=PATCH,
                                   load_factors=[0.5 * py["pressure"]],
                                   nx=6, ny=6, layers=3)
    return {"analytic": py, "path": path, "probe": probe}


def test_elastic_branch_scales_linearly_with_the_load(capacity):
    steps = capacity["path"].loading[:3]
    ratios = []
    for step in steps:
        analytic = elastic_patch_plate(A, B, T, E, NU, PATCH, step.load_factor)["deflection"][0]
        ratios.append(step.centre_deflection / analytic)
    assert max(ratios) - min(ratios) < 5e-3            # linear elastic, no drift
    assert all(step.max_alpha == 0.0 for step in steps)


def test_same_mesh_elastic_bias_is_quantified_and_bounded(capacity):
    probe = capacity["probe"].loading[0]
    analytic = capacity["analytic"]["sigma_eq_per_pa"]
    fe = probe.max_surface_mises / probe.load_factor
    bias = fe / analytic - 1.0
    # the layered Shell4 element is soft in bending; this bound is a regression
    # guard, the measured number is reported in the case document
    assert 0.0 < bias < 0.25


def test_yield_is_detected_on_the_surface_before_any_layer_yields(capacity):
    path = capacity["path"]
    first = path.first_yield_factor
    assert first is not None
    surface_yielded = [s for s in path.loading if s.max_surface_mises >= FY]
    assert surface_yielded, "surface von Mises never reached the flow stress"
    assert first <= surface_yielded[0].load_factor
    # through-thickness layering delays the *layer* plastic strain: with n layers
    # sampled at mid-layer points the outer layer sits at |z| = t/2 (1 - 1/n)
    layer_yielded = [s for s in path.loading if s.max_alpha > 0.0]
    assert layer_yielded, "no layer ever became plastic"
    ratio = layer_yielded[0].load_factor / first
    assert 1.0 < ratio < 1.0 / (1.0 - 1.0 / 3) + 0.35   # + one load step (0.3 py) of granularity


def test_transition_point_matches_same_mesh_elastic_field_within_5pct(capacity):
    path = capacity["path"]
    probe = capacity["probe"].loading[0]
    fe_sigma_per_pa = probe.max_surface_mises / probe.load_factor
    same_mesh_yield = FY / fe_sigma_per_pa
    error = (path.first_yield_factor - same_mesh_yield) / same_mesh_yield
    assert abs(error) < 0.05


def test_transition_point_raw_error_versus_analytic_yield_load(capacity):
    """Raw comparison against the closed-form yield load (no bias normalisation).

    Reported, not tuned: the layered plate element under-predicts the elastic peak
    moment, so the raw first-yield load lands several percent low.
    """
    path = capacity["path"]
    analytic = capacity["analytic"]["pressure"]
    error = (path.first_yield_factor - analytic) / analytic
    assert -0.20 < error < 0.05


def test_permanent_deflection_is_positive_and_unloads_monotonically(capacity):
    path = capacity["path"]
    assert path.permanent_deflection is not None
    assert path.permanent_deflection > 0.0
    assert path.peak_factor > capacity["analytic"]["pressure"]
    residuals = [step.centre_deflection for step in path.unloading]
    assert residuals == sorted(residuals, reverse=True)
    assert residuals[-1] == pytest.approx(path.permanent_deflection)


def test_permanent_set_is_a_fraction_of_the_peak_deflection(capacity):
    path = capacity["path"]
    peak = path.loading[-1].centre_deflection
    assert 0.0 < path.permanent_deflection < peak


def test_surface_extrapolation_is_exact_for_linear_stress():
    thickness, layers = 0.02, 5
    dz = thickness / layers
    z = [(-thickness / 2 + (k + 0.5) * dz) for k in range(layers)]
    stress = torch.zeros((1, 4, layers, 3), dtype=torch.float64)
    for k, zk in enumerate(z):
        stress[0, :, k, 0] = 100e6 * zk / (thickness / 2)     # linear bending stress
    assert surface_extrapolated_mises(stress, thickness) == pytest.approx(100e6, rel=1e-9)
    with pytest.raises(ValueError):
        surface_extrapolated_mises(torch.zeros((1, 4, 1, 3), dtype=torch.float64), thickness)


def test_collapse_load_matches_the_yield_line_bound_after_layering_correction():
    """Near-ideally plastic plate: the collapse load, corrected for the
    through-thickness layering factor ``(1 - 1/n)`` (a mid-layer stress sample
    reaches the flow stress only at ``|z| = t/2 (1-1/n)``), must approach the
    yield-line upper bound from below and stay above the strip lower bound.
    """
    py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=11, terms=31)
    upper = yield_line_patch_upper_bound(A, B, T, FY, PATCH)["pressure"]
    lower = strip_lower_bound(A, B, T, FY, PATCH)["pressure"]
    layers = 3
    path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                  hardening=1e6, patch=PATCH,
                                  load_factors=[0.6 * py["pressure"], 1.0 * py["pressure"],
                                                1.4 * py["pressure"], 1.8 * py["pressure"],
                                                2.4 * py["pressure"], 3.0 * py["pressure"]],
                                  nx=6, ny=6, layers=layers)
    collapse = path.collapse_factor or path.loading[-1].load_factor
    corrected = collapse * (1.0 - 1.0 / layers)
    assert lower < corrected < upper
    assert abs(corrected / upper - 1.0) < 0.15


# ---------------------------------------------------------------------------
# 3-level mesh convergence (anchor acceptance "3 档网格收敛")
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh_sweep():
    """Run the FE path at three even mesh levels and return the observables.

    The acceptance criterion requires the numerical collapse / first-yield /
    permanent-set path to converge monotonically with refinement, and the
    transition point to stay within the same-mesh elastic-field tolerance at
    every level.  Even element counts keep the plate centre a node.
    """
    py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=11, terms=31)
    upper = yield_line_patch_upper_bound(A, B, T, FY, PATCH)["pressure"]
    lower = strip_lower_bound(A, B, T, FY, PATCH)["pressure"]
    rows = []
    for nx in (4, 6, 8):
        steps = [f * py["pressure"] for f in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.4, 3.0)]
        path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                      hardening=2e10, patch=PATCH, load_factors=steps,
                                      nx=nx, ny=nx, layers=3, unload_steps=3)
        probe = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                       hardening=2e10, patch=PATCH,
                                       load_factors=[0.5 * py["pressure"]],
                                       nx=nx, ny=nx, layers=3)
        fe_sigma_per_pa = probe.loading[0].max_surface_mises / probe.loading[0].load_factor
        same_mesh_yield = FY / fe_sigma_per_pa
        rows.append({
            "nx": nx,
            "first": path.first_yield_factor,
            "first_err_same_mesh": (path.first_yield_factor - same_mesh_yield) / same_mesh_yield,
            "collapse": path.collapse_factor or path.loading[-1].load_factor,
            "permanent": path.permanent_deflection,
            "peak": path.loading[-1].centre_deflection,
        })
    return {"py": py, "upper": upper, "lower": lower, "rows": rows}


def test_collapse_load_converges_monotonically_to_the_upper_bound(mesh_sweep):
    """Collapse must approach the yield-line upper bound from below, monotonically.

    The raw collapse is over-strengthened by the through-thickness layering
    (outer stress sample at |z| = t/2 (1-1/n)), so it is reported *layering-
    corrected* by (1 - 1/n) — the same convention as the single-mesh collapse
    test — and must then sit between the strip lower and yield-line upper bounds
    and be monotone non-decreasing with refinement.
    """
    upper = mesh_sweep["upper"]
    lower = mesh_sweep["lower"]
    collapses = [r["collapse"] * (1.0 - 1.0 / 3) for r in mesh_sweep["rows"]]
    assert all(c < upper for c in collapses), "corrected collapse must stay below the upper bound"
    assert all(c > lower for c in collapses), "corrected collapse must stay above the strip lower bound"
    assert all(collapses[i] <= collapses[i + 1] for i in range(len(collapses) - 1)), \
        "corrected collapse load must be monotone non-decreasing with refinement"


def test_transition_point_stays_within_5pct_at_every_mesh_level(mesh_sweep):
    """The elastic->plastic transition must match the same-mesh elastic field < 5%."""
    for r in mesh_sweep["rows"]:
        assert abs(r["first_err_same_mesh"]) < 0.05, f"mesh {r['nx']} transition error {r['first_err_same_mesh']:.3f}"


def test_permanent_set_is_bounded_and_converges(mesh_sweep):
    """Permanent set must be a positive, bounded fraction of the peak deflection."""
    for r in mesh_sweep["rows"]:
        assert 0.0 < r["permanent"] < r["peak"], f"mesh {r['nx']} permanent set out of range"
    perms = [r["permanent"] for r in mesh_sweep["rows"]]
    # permanent set should not grow without bound as the mesh refines
    assert max(perms) < 1.5 * min(perms), "permanent set must be mesh-converged (spread < 50%)"
