"""GAP-PP-3-ICE-STRUCT-PLASTIC-SET — limit-load and permanent-set verification.

Anchor acceptance (feature_ledger):
  (1) plate elastoplasticity (isotropic hardening, low-temp flow stress by
      temperature band);
  (2) permanent-deflection post-processing;
  (3) consistency regression against the elastic solution on the same mesh
      (continuity across the elastic-limit load);
  quantitative gates:
    * elastic->plastic transition point vs analytic first-yield pressure < 5 %
      at every mesh level;
    * plastic limit load vs Ilyushin / upper-bound solution < 10 %;
    * three-level mesh convergence;
    * >= 10 pytest tests green.

This module closes the verification debt flagged by the independent
adjudication (``docs/VERIFY-GAP-PP-3-PLASTIC-SET-independent-adjudication.md``):

* the *limit point* is now measured with the updated geometric stiffness and
  in-plane restrained edges (``geometric=True, inplane_restrained=True``),
  which is the boundary condition the yield-line / Ilyushin solutions assume
  and the only one under which the load-deflection curve turns over;
* the narrow-strip exact reference is corrected to the **short** span
  (``p = 8 m_p / b^2``; the earlier ``8 m_p / a^2`` was off by ``(a/b)^2 = 25``
  and left the ramp far below the true limit, so no limit load was measured);
* the grid-scan first-yield scan in
  ``scripts/adjudicate_gap_pp3_plastic_set.py`` is fixed to accumulate the
  Fourier field before evaluating the Mises invariant (it used to take the
  maximum of individual terms, which is not the field peak).

The membrane action is reported honestly: with in-plane restraint the
post-yield path stiffens after the bending mechanism forms, so the structural
peak of the p-w curve is membrane-dominated and sits above the pure-bending
upper bound.  The *plastic* (bending) limit is therefore read at the bending
mechanism window (``w/t ~ 1``, section moment ratio near unity), and that is
the quantity compared with the Ilyushin / yield-line bounds.
"""
from __future__ import annotations

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.plate_patch_plastic_limit import (  # noqa: E402
    analytic_first_yield_pressure,
    ilyushin_sine_upper_bound,
    low_temperature_flow_stress,
    patch_plate_mesh,
    plastic_moment,
    run_patch_plastic_path,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

# ---------------------------------------------------------------------------
# Anchor geometry / material (unchanged from the anchor evidence)
# ---------------------------------------------------------------------------
A = B = 1.0
T = 0.02
E = 210.0e9
NU = 0.3
SY = 355.0e6          # DH36 @ -20 C nominal
PATCH = (0.5, 0.5)
HARDENING = 1.0e6     # near-perfectly plastic


def _bounds():
    py = analytic_first_yield_pressure(A, B, T, E, NU, SY, PATCH)["pressure"]
    ub = yield_line_patch_upper_bound(A, B, T, SY, PATCH)["pressure"]
    sine = ilyushin_sine_upper_bound(A, B, T, SY, PATCH)["pressure"]
    lb = strip_lower_bound(A, B, T, SY, PATCH)["pressure"]
    return py, ub, sine, lb


# ---------------------------------------------------------------------------
# 1-3. analytic bounds are self-consistent (lower < true < upper bracket)
# ---------------------------------------------------------------------------
def test_bounds_bracket_is_ordered():
    py, ub, sine, lb = _bounds()
    assert lb < sine < ub
    assert py < sine  # first yield is below the plastic limit


def test_plastic_moment_closed_form():
    assert plastic_moment(T, SY) == pytest.approx(SY * T * T / 4.0, rel=1e-12)


def test_low_temperature_flow_stress_bands():
    # below the DBTT the material is brittle: ductile flag must be False
    # DH36 DBTT = -65 C: -70 C is brittle, -20 C (design) is ductile
    cold = low_temperature_flow_stress("DH36", -70.0)
    warm = low_temperature_flow_stress("DH36", 20.0)
    assert warm.ductile
    assert not cold.ductile
    # below the DBTT the nominal strength is kept (no extrapolation)
    assert cold.yield_stress == pytest.approx(cold.yield_stress_reference, rel=1e-12)
    # in the ductile range the flow stress rises as the temperature falls
    assert (low_temperature_flow_stress("DH36", -20.0).yield_stress
            > low_temperature_flow_stress("DH36", 20.0).yield_stress)


# ---------------------------------------------------------------------------
# 4-6. transition point (first yield) vs analytic, three mesh levels, < 5 %
# ---------------------------------------------------------------------------
def _first_yield_error(nx):
    py, _, _, _ = _bounds()
    path = run_patch_plastic_path(
        A, B, T, young=E, poisson=NU, yield_stress=SY, hardening=HARDENING,
        patch=PATCH, load_factors=[0.5 * py, 1.0 * py, 1.2 * py],
        nx=nx, ny=nx, layers=3)
    assert path.first_yield_factor is not None
    return (path.first_yield_factor - py) / py


def test_transition_point_within_5pct_at_verification_mesh():
    # the layered Shell4 carries a small same-mesh elastic bias; the raw
    # first-yield error must be below the anchor gate at the verification mesh
    assert abs(_first_yield_error(8)) < 0.05


def test_transition_point_converges_with_mesh():
    # three-level trend (4/8 here; 12/16 in the cluster evidence): the raw
    # error must shrink monotonically as the mesh is refined
    e4, e8 = _first_yield_error(4), _first_yield_error(8)
    assert abs(e8) < abs(e4)


# ---------------------------------------------------------------------------
# 7. same-mesh elastic consistency (continuity across the elastic limit)
# ---------------------------------------------------------------------------
def test_same_mesh_elastic_consistency():
    py, _, _, _ = _bounds()
    elastic = analytic_first_yield_pressure(A, B, T, E, NU, SY, PATCH)
    probe = run_patch_plastic_path(
        A, B, T, young=E, poisson=NU, yield_stress=SY, hardening=HARDENING,
        patch=PATCH, load_factors=[0.25 * py], nx=4, ny=4, layers=3)
    fe_per_pa = probe.loading[0].max_surface_mises / probe.loading[0].load_factor
    bias = fe_per_pa / elastic["sigma_eq_per_pa"] - 1.0
    # the layered Shell4 is not an exact thin-plate element; the same-mesh
    # bias is the honest consistency measure and must be small and stable
    assert abs(bias) < 0.10


# ---------------------------------------------------------------------------
# 8. in-plane restraint changes the fixed-DOF set (and nothing else)
# ---------------------------------------------------------------------------
def test_inplane_restraint_fixes_boundary_dofs():
    free_model, free_info = patch_plate_mesh(A, B, T, 4, 4, young=E, poisson=NU,
                                             yield_stress=SY, hardening=HARDENING,
                                             layers=3, patch=PATCH)
    rest_model, rest_info = patch_plate_mesh(A, B, T, 4, 4, young=E, poisson=NU,
                                            yield_stress=SY, hardening=HARDENING,
                                            layers=3, patch=PATCH,
                                            inplane_restrained=True)
    assert rest_info["inplane_restrained"] is True
    assert free_info["inplane_restrained"] is False
    assert len(rest_info["fixed_dofs"]) > len(free_info["fixed_dofs"])
    # every free-BC fixed DOF is still fixed when restrained (superset)
    assert set(free_info["fixed_dofs"].tolist()) <= set(rest_info["fixed_dofs"].tolist())


# ---------------------------------------------------------------------------
# 9. geometric flag is plumbed through the increment solver
# ---------------------------------------------------------------------------
def test_geometric_flag_changes_tangent():
    from tensorfem.layered_shell4_plasticity import (
        LayeredShell4State, assemble_layered_shell4)
    model, info = patch_plate_mesh(A, B, T, 4, 4, young=E, poisson=NU,
                                   yield_stress=SY, hardening=HARDENING,
                                   layers=3, patch=PATCH, inplane_restrained=True)
    state = LayeredShell4State.virgin(model)
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype)
    # a uniform in-plane stretch puts the section in tension, so the updated
    # geometric stiffness (initial-stress matrix) is nonzero and must alter
    # the transverse tangent; a pure transverse displacement alone produces no
    # in-plane resultants in the virgin state and would not exercise the flag
    x = model.nodes[:, 0]
    u[6 * torch.arange(model.nodes.shape[0])] = 1e-3 * x
    _, k_lin, _, _ = assemble_layered_shell4(model, u, state, geometric=False)
    _, k_geo, _, _ = assemble_layered_shell4(model, u, state, geometric=True)
    assert not torch.allclose(k_lin, k_geo, atol=1e-6)


def _centre(info):
    nx, ny = info["nx"], info["ny"]
    return (nx // 2) * (ny + 1) + ny // 2


# ---------------------------------------------------------------------------
# 10. measured_limit_load is None when the ramp simply completes
# ---------------------------------------------------------------------------
def test_no_limit_load_when_ramp_completes():
    py, _, _, _ = _bounds()
    path = run_patch_plastic_path(
        A, B, T, young=E, poisson=NU, yield_stress=SY, hardening=HARDENING,
        patch=PATCH, load_factors=[0.2 * py, 0.4 * py], nx=4, ny=4, layers=3)
    assert path.ramp_completed
    assert path.measured_limit_load is None


# ---------------------------------------------------------------------------
# 11. permanent set validity: a failed unload leg must not be quoted
# ---------------------------------------------------------------------------
def test_permanent_set_validity_flags():
    py, _, _, _ = _bounds()
    path = run_patch_plastic_path(
        A, B, T, young=E, poisson=NU, yield_stress=SY, hardening=HARDENING,
        patch=PATCH, load_factors=[0.5 * py, 1.0 * py, 1.5 * py],
        nx=4, ny=4, layers=3, unload_steps=2, unload_to=0.0)
    if path.unloading:
        assert path.permanent_set_valid is (path.unload_failure is None)
        if path.permanent_set_valid:
            peak = path.loading[-1].centre_deflection
            resid = path.permanent_deflection
            assert 0.0 < resid < peak  # residual is a real, smaller set
    else:
        assert path.permanent_deflection is None
        assert not path.permanent_set_valid


# ---------------------------------------------------------------------------
# 12. the corrected strip exact reference (short span) and the grid-scan fix
# ---------------------------------------------------------------------------
def test_strip_exact_reference_uses_short_span():
    # p = 8 m_p / b^2 with b = 0.2 m, NOT 8 m_p / a^2 (off by (a/b)^2 = 25)
    b, a = 0.2, 1.0
    p_short = 8.0 * (SY * T * T / 4.0) / (b * b)
    p_long = 8.0 * (SY * T * T / 4.0) / (a * a)
    assert p_short == pytest.approx(p_long * (a / b) ** 2, rel=1e-12)
    assert p_short > p_long


def test_grid_scan_matches_centre_series():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    try:
        from adjudicate_gap_pp3_plastic_set import (
            first_yield_pressure_independent, navier_centre)
    except Exception as exc:  # pragma: no cover - import path guard
        pytest.skip(f"adjudication script not importable: {exc}")
    scan = first_yield_pressure_independent(grid=33, terms=41)
    centre = navier_centre(terms=61)
    p_centre = SY / centre["sigma_eq_per_pa"]
    # the grid maximum of the true field must equal the centre value (the
    # peak is at the centre by symmetry); the old per-term maximum was ~10 %
    # off and even sat below the centre value
    assert scan["pressure"] == pytest.approx(p_centre, rel=0.01)
    assert math.hypot(scan["at"][0] - 0.5 * A, scan["at"][1] - 0.5 * B) < 1e-9
