"""Adjudication tests for GAP-PP-3-ICE-STRUCT-PLASTIC-SET (verification debt).

The anchor module's own tests compare the finite element against *its* analytic
helpers, so a shared algebra error would pass.  These tests re-derive the
references independently (own series, own quadrature) and additionally pin the
reporting semantics that the verification found broken:

* a failed unloading leg must not be reported as a permanent set,
* ``collapse_factor`` must say whether it is a genuine limit point or merely the
  last prescribed load factor of the ramp.

All checks here are analytic or bookkeeping only (no FE solve), so they stay in
the fast lane; the FE mesh sweep is exercised by the existing slow tests.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.plate_patch_plastic_limit import (
    PatchPath,
    PatchPathPoint,
    analytic_first_yield_pressure,
    ilyushin_sine_upper_bound,
    plastic_moment,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

A, B, T = 1.0, 1.0, 0.02
E, NU, SY = 210.0e9, 0.3, 355.0e6
PATCH = (0.5 * A, 0.5 * B)


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------

def _mp(t: float = T, sy: float = SY) -> float:
    return sy * t * t / 4.0


def _navier_sigma_per_pa(a=A, b=B, t=T, e=E, nu=NU, patch=PATCH, terms=61) -> float:
    """Navier double series, written from the load expansion directly."""
    u, v = patch
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    mx = my = 0.0
    for m in range(1, terms + 1, 2):
        am = m * math.pi / a
        dx = math.cos(am * x1) - math.cos(am * x2)
        for n in range(1, terms + 1, 2):
            bn = n * math.pi / b
            dy = math.cos(bn * y1) - math.cos(bn * y2)
            qmn = 4.0 * dx * dy / (math.pi ** 2 * m * n)
            lam = am * am + bn * bn
            s = math.sin(am * 0.5 * a) * math.sin(bn * 0.5 * b)
            mx += qmn * (am * am + nu * bn * bn) / (lam * lam) * s
            my += qmn * (bn * bn + nu * am * am) / (lam * lam) * s
    sx, sy = 6.0 * mx / (t * t), 6.0 * my / (t * t)
    return math.sqrt(max(sx * sx - sx * sy + sy * sy, 0.0))


def _mechanism_volume(a=A, b=B, patch=PATCH, samples=400) -> float:
    """``int_patch (1 - 2 max(|x|/a,|y|/b)) dA`` by midpoint quadrature."""
    u, v = patch
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    dx, dy = (x2 - x1) / samples, (y2 - y1) / samples
    total = 0.0
    for i in range(samples):
        x = x1 + (i + 0.5) * dx - 0.5 * a
        ax = abs(x) / a
        row = 0.0
        for j in range(samples):
            y = y1 + (j + 0.5) * dy - 0.5 * b
            row += 1.0 - 2.0 * max(ax, abs(y) / b)
        total += row * dx * dy
    return total


def _yield_line_ub(a=A, b=B, t=T, sy=SY, patch=PATCH) -> float:
    """Four-triangle pyramid mechanism: 4 kinks dissipate m_p w0 (a/b + b/a)."""
    return 4.0 * _mp(t, sy) * (a / b + b / a) / _mechanism_volume(a, b, patch)


def _sine_ub(a=A, b=B, t=T, sy=SY, patch=PATCH) -> float:
    u, v = patch
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    internal = 4.0 * _mp(t, sy) * a * b * (1.0 / a ** 2 + 1.0 / b ** 2)
    ext = ((a / math.pi) * (math.cos(math.pi * x1 / a) - math.cos(math.pi * x2 / a))
           * (b / math.pi) * (math.cos(math.pi * y1 / b) - math.cos(math.pi * y2 / b)))
    return internal / ext


def _strip_lb(a=A, b=B, t=T, sy=SY, patch=PATCH) -> float:
    u, v = patch
    mp = _mp(t, sy)
    return 8.0 * mp / (u * (2.0 * a - u)) + 8.0 * mp / (v * (2.0 * b - v))


# ---------------------------------------------------------------------------
# 1. The module's analytic references against the independent ones
# ---------------------------------------------------------------------------

def test_plastic_moment_matches_independent_formula():
    assert plastic_moment(T, SY) == pytest.approx(_mp(), rel=1e-12)
    assert _mp() == pytest.approx(35500.0, rel=1e-12)          # N m / m for DH36 t=20 mm


def test_module_first_yield_pressure_is_0p804_mpa_not_the_reported_0p48():
    """The merged verification doc quotes ``p_y ~ 480 kPa``.  An independent
    Navier series evaluated at the plate centre -- where the symmetric patch
    field peaks -- gives 804 kPa, and the module's own lattice scan reproduces it
    to machine precision, so the doc number is an unsupported placeholder
    (recorded here so it cannot come back)."""
    independent = SY / _navier_sigma_per_pa()
    module = analytic_first_yield_pressure(A, B, T, E, NU, SY, PATCH,
                                           grid=41, terms=61)["pressure"]
    assert module == pytest.approx(independent, rel=1e-9)
    assert module == pytest.approx(0.804e6, rel=5e-3)
    assert abs(module / 0.48e6 - 1.0) > 0.5


def test_yield_line_bound_matches_independent_quadrature():
    got = yield_line_patch_upper_bound(A, B, T, SY, PATCH)["pressure"]
    assert got == pytest.approx(_yield_line_ub(), rel=1e-3)
    assert got == pytest.approx(1.704e6, rel=5e-3)


def test_sine_bound_matches_independent_integrals():
    got = ilyushin_sine_upper_bound(A, B, T, SY, PATCH)["pressure"]
    assert got == pytest.approx(_sine_ub(), rel=1e-9)
    assert got == pytest.approx(1.401e6, rel=5e-3)


def test_strip_lower_bound_matches_independent_expression():
    got = strip_lower_bound(A, B, T, SY, PATCH)["pressure"]
    assert got == pytest.approx(_strip_lb(), rel=1e-12)
    assert got == pytest.approx(0.757e6, rel=5e-3)


def test_admissible_bracket_ordering_for_the_anchor_case():
    """lower < tighter upper < looser upper, and the bracket is *wide*: the
    in-house evidence must not present a <10 % agreement with the upper bound as
    if the limit load were known to <10 %."""
    lower = _strip_lb()
    sine = _sine_ub()
    yl = _yield_line_ub()
    assert lower < sine < yl
    assert sine / lower == pytest.approx(1.85, abs=0.05)
    assert yl / sine == pytest.approx(1.216, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Reporting semantics repaired by this verification
# ---------------------------------------------------------------------------

def _path(**kwargs) -> PatchPath:
    return PatchPath(model=None, info={}, **kwargs)     # model unused by the properties


def test_failed_unload_is_not_reported_as_a_permanent_set():
    path = _path(loading=[PatchPathPoint(1.0e6, 0.02, 0.01)],
                 unloading=[], unload_steps_requested=3,
                 unload_failure="unloading step to 0.9e6 did not converge")
    assert path.permanent_deflection is None
    assert path.permanent_set_valid is False


def test_converged_unload_reports_a_valid_permanent_set():
    path = _path(loading=[PatchPathPoint(1.0e6, 0.02, 0.01)],
                 unloading=[PatchPathPoint(5.0e5, 0.015, 0.01),
                            PatchPathPoint(0.0, 0.011, 0.01)],
                 unload_steps_requested=2, unload_converged_steps=2)
    assert path.permanent_deflection == pytest.approx(0.011)
    assert path.permanent_set_valid is True


def test_collapse_provenance_distinguishes_limit_point_from_ramp_end():
    limit = _path(loading=[PatchPathPoint(1.0e6, 0.02, 0.01)],
                  collapse_factor=1.0e6, limit_point_searched=True)
    ramp_end = _path(loading=[PatchPathPoint(1.0e6, 0.02, 0.01)],
                     collapse_factor=None, ramp_completed=True)
    assert limit.collapse_is_limit_point is True
    assert ramp_end.collapse_is_limit_point is False
    assert ramp_end.permanent_deflection is None