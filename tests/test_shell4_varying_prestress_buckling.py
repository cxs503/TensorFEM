"""Fail-closed gates for non-uniform (linearly varying) pre-stress buckling.

The reference is ``tensorfem.plate_varying_buckling_reference`` (Ritz double-sine
energy minimisation), which is independent of the finite-element path.  All
tolerances below are hard AssertionErrors: a regression breaks the suite rather
than degrading a stored number.
"""
import json
import math
from pathlib import Path

import pytest
import torch

from tensorfem.plate_varying_buckling_reference import (
    ritz_uniaxial_buckling, ritz_uniaxial_convergence,
)
from tensorfem.shell4_buckling import (
    linearly_varying_uniaxial_field, navier_plate_load,
    shell4_geometric_stiffness, simply_supported_plate_buckling,
    simply_supported_plate_buckling_field,
)

PLATE = dict(young=210e9, poisson=0.3, thickness=0.01, a=1.0, b=1.0)


def _coefficient(value: float, case: dict) -> float:
    D = case["young"] * case["thickness"] ** 3 / (12 * (1 - case["poisson"] ** 2))
    return value * case["b"] ** 2 / (math.pi ** 2 * D)


def test_ritz_reproduces_closed_form_uniform_navier():
    out = ritz_uniaxial_buckling(psi=1.0, harmonics=12, **PLATE)
    exact = navier_plate_load(PLATE["a"], PLATE["b"], PLATE["young"],
                              PLATE["poisson"], PLATE["thickness"])
    assert abs(out["critical_resultant"] - exact) / exact < 1e-9
    assert out["dominant_halfwaves"] == (1, 1)


def test_ritz_edge_to_zero_matches_classical_seven_eight_one():
    out = ritz_uniaxial_buckling(psi=0.0, harmonics=12, **PLATE)
    assert abs(out["buckling_coefficient"] - 7.81) / 7.81 < 1e-3


def test_ritz_harmonic_convergence_is_monotone_in_change():
    rows = ritz_uniaxial_convergence(harmonics=(6, 8, 12, 16), psi=-1.0, **PLATE)
    changes = [row["successive_change"] for row in rows[1:]]
    assert min(changes) >= 0
    assert changes[-1] < 1e-6


def test_constant_field_api_is_identical_to_constant_path():
    common = dict(a=1.0, b=1.0, nx=8, ny=8, young=210e9, poisson=0.3, thickness=0.01)
    constant = float(simply_supported_plate_buckling(**common, ny_ratio=0.).load_factors[0])
    field = float(simply_supported_plate_buckling_field(
        **common, stress_field=linearly_varying_uniaxial_field(1.0)).load_factors[0])
    assert abs(constant - field) / constant < 1e-12


def test_bilinear_nodal_resultants_match_constant_triple():
    xyz = torch.tensor([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]],
                       dtype=torch.float64)
    triple = torch.tensor([3., 1., 0.2], dtype=torch.float64)
    nodal = triple.reshape(1, 3).expand(4, 3)
    a = shell4_geometric_stiffness(xyz, triple)
    b = shell4_geometric_stiffness(xyz, nodal)
    assert a.shape == (24, 24)
    assert torch.allclose(a, b, atol=1e-12)
    assert torch.allclose(b, b.T, atol=1e-12)


def test_linear_nodal_field_is_not_piecewise_constant():
    """A linear nodal field must differ from the element mean of the same field."""
    xyz = torch.tensor([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]],
                       dtype=torch.float64)
    nodal = torch.tensor([[1., 0., 0.], [1., 0., 0.], [3., 0., 0.], [3., 0., 0.]],
                         dtype=torch.float64)
    field = shell4_geometric_stiffness(xyz, nodal)
    mean = shell4_geometric_stiffness(xyz, torch.tensor([2., 0., 0.], dtype=torch.float64))
    assert torch.linalg.vector_norm(field - mean) > 1e-6


def test_edge_to_zero_shell4_is_within_three_percent():
    ref = ritz_uniaxial_buckling(psi=0.0, harmonics=16, **PLATE)["critical_resultant"]
    for n in (8, 16, 24):
        value = float(simply_supported_plate_buckling_field(
            nx=n, ny=n, stress_field=linearly_varying_uniaxial_field(0.0), **PLATE).load_factors[0])
        assert abs(value - ref) / ref < 0.03, f"mesh {n}x{n} outside the 3% gate"


def test_pure_bending_shell4_converges_but_coarse_mesh_stays_failed():
    ref = ritz_uniaxial_buckling(psi=-1.0, harmonics=16, **PLATE)["critical_resultant"]
    errors = {}
    for n in (8, 16, 24):
        value = float(simply_supported_plate_buckling_field(
            nx=n, ny=n, stress_field=linearly_varying_uniaxial_field(-1.0),
            **PLATE).load_factors[0])
        errors[n] = (value - ref) / ref
    assert abs(errors[8]) > 0.03          # recorded failure, not hidden
    assert abs(errors[16]) < 0.03
    assert abs(errors[24]) < 0.01


def test_pure_bending_uses_antisymmetric_mode():
    result = simply_supported_plate_buckling_field(
        nx=16, ny=16, stress_field=linearly_varying_uniaxial_field(-1.0), **PLATE)
    w = result.modes[0::3, 0].reshape(17, 17)
    assert float(w[:8, :].sum() + w[9:, :].sum()) < 0.5 * float(w.abs().sum())
    assert float(result.modes[:, 0][result.modes[:, 0].abs() > 0].abs().sum()) > 0


def test_indefinite_prestress_does_not_raise_and_is_physical():
    """Pure bending makes the initial-stress matrix indefinite; the solver must still work."""
    uniform = float(simply_supported_plate_buckling_field(
        nx=12, ny=12, stress_field=linearly_varying_uniaxial_field(1.0), **PLATE).load_factors[0])
    bending = float(simply_supported_plate_buckling_field(
        nx=12, ny=12, stress_field=linearly_varying_uniaxial_field(-1.0), **PLATE).load_factors[0])
    assert bending > uniform
    assert math.isfinite(bending)


@pytest.mark.parametrize("field", [
    lambda x, y: (1.0, 0.0),
    lambda x, y: (float("nan"), 0.0, 0.0),
    "not-callable",
])
def test_malformed_stress_field_is_rejected(field):
    with pytest.raises((TypeError, ValueError)):
        simply_supported_plate_buckling_field(nx=4, ny=4, stress_field=field, **PLATE)


def test_applied_field_magnitude_scales_load_factor_reciprocally():
    """The reported eigenvalue is a load factor: doubling the applied field halves it."""
    kw = dict(nx=8, ny=8, stress_field=linearly_varying_uniaxial_field(0.0), **PLATE)
    base = float(simply_supported_plate_buckling_field(**kw).load_factors[0])
    doubled = float(simply_supported_plate_buckling_field(
        nx=8, ny=8, stress_field=lambda x, y: (2.0 * (1.0 - y), 0.0, 0.0),
        **PLATE).load_factors[0])
    # reported value is the load factor on the applied field, so 2*field -> /2
    assert abs(doubled * 2.0 / base - 1.0) < 1e-9


def test_stored_evidence_artifact_matches_recomputation():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "varying-prestress-buckling.json"
    if not path.exists():
        pytest.skip("evidence artifact not built in this checkout")
    stored = json.loads(path.read_text())
    psi0 = float(simply_supported_plate_buckling_field(
        nx=16, ny=16, stress_field=linearly_varying_uniaxial_field(0.0), **PLATE).load_factors[0])
    recorded = next(r for r in stored["shell4"]["edge_to_zero"] if r["mesh"] == "16x16")
    assert abs(psi0 - recorded["critical_resultant"]) / psi0 < 1e-10
    assert stored["verdict"]["uniform"]["within_3pct_all_meshes"] is True
    assert stored["verdict"]["pure_bending"]["within_3pct_all_meshes"] is False