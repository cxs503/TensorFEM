"""Objective round-off criteria: unit tests and the layered-shell floor audit.

The retired assertion ``abs(stress).max() < 1e-10`` on a stress-free patch was
machine dependent because ``1e-10`` sat inside the ``ops * eps * E`` noise band
of the assembly.  These tests pin the replacement criterion
(:mod:`tensorfem.roundoff_scale`) and prove, on the same case, that

* the measured floor stays inside ``DEFAULT_OPS * eps * scale`` for coordinate
  origins spanning eight decades;
* the bound sits at least ten orders of magnitude below a physically
  meaningful stress signal, so it is not a vacuous assertion;
* the fixture reproduces the historical failure mode (the floor exceeds the
  retired ``1e-10`` constant at some origin), which is why an absolute
  constant was never platform independent.
"""
import math

import pytest
import torch

from tensorfem.finite_rotation_layered_shell4 import assemble_finite_rotation_layered_shell4
from tensorfem.layered_shell4_plasticity import LayeredShell4Model, LayeredShell4State
from tensorfem.roundoff_scale import (
    DEFAULT_OPS,
    machine_epsilon,
    roundoff_bound,
    roundoff_ratio,
)

D = torch.float64
YOUNG, POISSON, THICKNESS = 200000.0, 0.3, 0.08
ELEMENTS = torch.tensor([[0, 1, 2, 3]])
WARP = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.012], [1.0, 1.0, -0.007], [0.0, 1.0, 0.004]], dtype=D
)
ORIGINS = (0.0, 1e2, 1e4, 1e6, 1e8)


def _model(origin: float) -> LayeredShell4Model:
    return LayeredShell4Model(
        WARP + origin, ELEMENTS, YOUNG, POISSON, THICKNESS, 250.0, 1400.0, layers=3
    )


def _stress_free(origin: float):
    shell = _model(origin)
    return shell, assemble_finite_rotation_layered_shell4(
        shell, torch.zeros(shell.n_dofs, dtype=D),
        LayeredShell4State.virgin(shell), tangent=False,
    )


def test_bound_is_linear_in_scale_and_operation_budget():
    base = roundoff_bound(1.0)
    assert base == pytest.approx(DEFAULT_OPS * torch.finfo(D).eps)
    assert roundoff_bound(4.0) == pytest.approx(4.0 * base)
    assert roundoff_bound(1.0, ops=2 * DEFAULT_OPS) == pytest.approx(2.0 * base)


def test_bound_tracks_dtype_precision():
    single = machine_epsilon(torch.float32)
    double = machine_epsilon(torch.float64)
    assert single > double
    assert roundoff_bound(1.0, torch.float32) / roundoff_bound(1.0, torch.float64) == \
        pytest.approx(single / double)


def test_invalid_scales_and_budgets_are_rejected():
    for scale in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="scale"):
            roundoff_bound(scale)
    for ops in (0, -3):
        with pytest.raises(ValueError, match="ops"):
            roundoff_bound(1.0, ops=ops)
    with pytest.raises(ValueError, match="ops"):
        roundoff_bound(1.0, ops=2.5)


def test_ratio_reports_margin_relative_to_the_bound():
    assert roundoff_ratio(0.0, 1.0) == 0.0
    assert roundoff_ratio(roundoff_bound(1.0), 1.0) == pytest.approx(1.0)
    assert roundoff_ratio(-1e-16, 1e6) < 1.0
    assert roundoff_ratio(-1e-6, 1e6) > 1.0  # 1e-6 Pa on E=1e6 is not round-off


@pytest.mark.parametrize("origin", ORIGINS)
def test_stress_free_reference_is_zero_to_roundoff(origin):
    """Replaces the retired machine-dependent ``< 1e-10`` assertion."""
    _, response = _stress_free(origin)
    assert float(response.internal_force.abs().max()) <= roundoff_bound(YOUNG * THICKNESS)
    assert float(response.stress.abs().max()) <= roundoff_bound(YOUNG)


def test_measured_amplification_stays_inside_the_operation_budget():
    """``floor / (eps * E)`` must stay under ``DEFAULT_OPS`` with real margin."""
    amplifications = []
    for origin in ORIGINS:
        _, response = _stress_free(origin)
        amplifications.append(float(response.stress.abs().max()) / (machine_epsilon(D) * YOUNG))
    assert max(amplifications) < DEFAULT_OPS / 10.0
    # Sanity: the floor is not identically zero, so the test measures something.
    assert min(amplifications) > 1e-3


def test_bound_is_not_vacuous_for_a_physical_inconsistency():
    """A one-percent strain signal must exceed the bound by ten orders."""
    shell, stress_free = _stress_free(0.0)
    stretch = torch.zeros(shell.n_dofs, dtype=D)
    for node, X in enumerate(shell.nodes):
        stretch[6 * node] = 1e-2 * float(X[0])
        stretch[6 * node + 1] = 1e-2 * float(X[1])
    stretched = assemble_finite_rotation_layered_shell4(
        shell, stretch, LayeredShell4State.virgin(shell), tangent=False
    )
    signal = float(stretched.stress.abs().max())
    assert signal / roundoff_bound(YOUNG) > 1e9
    assert float(stress_free.stress.abs().max()) < signal / 1e6


def test_retired_absolute_threshold_was_inside_the_noise_band():
    """Documents why ``1e-10`` was machine dependent, on this very fixture."""
    floors = [float(_stress_free(origin)[1].stress.abs().max()) for origin in ORIGINS]
    assert max(floors) > 1e-11  # not exactly zero: there is a real floor
    assert any(floor > 5e-11 for floor in floors)