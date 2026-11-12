"""Thick-plate (Mindlin-Reissner) bending moments vs an exact Navier series.

Closes the documented limitation of ``GAP-FEM-WINKLER``'s moment evidence
(``docs/GAP-FEM-WINKLER-moment-verification.md``, point 5.1): that page compared
the FE (Mindlin) moment field with the *Kirchhoff* series, which is only an
``O((t/a)^2)`` approximation of the FE kinematics.  Here the reference is the
Mindlin-Reissner Navier solution itself
(:func:`~tensorfem.foundation_plate_bending.navier_mindlin_foundation_moments`),
exact for hard simple support at any ``t/a``.

Layers:

1. **Classical identity, independent of the code** -- for a simply supported
   Mindlin plate *without* foundation the rotations (hence the bending moments)
   are identical to the Kirchhoff solution while the deflection gains the shear
   term ``1 + D(a^2+b^2)/S`` per mode.  Both halves of that statement are
   asserted exactly, which pins the new series against physics rather than
   against itself.
2. **Foundation coupling** -- with ``k > 0`` the identity breaks: shear
   deformation shifts the moments by ``O(k/S) = O(k/(G t))``; the measured shift
   is checked for the predicted ``1/t`` scaling.
3. **FE vs the exact thick-plate series** -- moment field within 3 % at
   ``t/a = 0.1`` and ``0.2`` (where the FE model is genuinely thick).
4. **Reference-validity boundary** -- at ``t/a = 0.2`` the Kirchhoff *deflection*
   reference is 23 % off the FE solution while the Mindlin reference is 0.04 %
   off: the same FE field "fails" a 15 % gate purely because the truth model is
   wrong.  Recorded as a regression guard on the benchmark's reference selection.
"""
import math

import pytest
import torch

from tensorfem.foundation_plate_bending import (
    compare_finite_plate_moments,
    navier_foundation_deflection,
    navier_foundation_moments,
    navier_mindlin_foundation_deflection,
    navier_mindlin_foundation_moments,
    plate_rigidity,
)

YOUNG, POISSON = 3.0e10, 0.3
SIDE, PRESSURE = 10.0, 1.0e4
SHEAR_CORRECTION = 5.0 / 6.0


def _center():
    return torch.tensor([[0.5 * SIDE, 0.5 * SIDE]], dtype=torch.float64)


def _quarter():
    return torch.tensor([[0.25 * SIDE, 0.25 * SIDE]], dtype=torch.float64)


def _shear_stiffness(thickness):
    return SHEAR_CORRECTION * YOUNG / (2.0 * (1.0 + POISSON)) * thickness


# ---------------------------------------------------------------------------
# 1. classical Mindlin/Kirchhoff identity for the hard-simply-supported plate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("thickness", [0.1, 0.5, 1.0, 2.0, 3.0])
def test_moments_identical_to_kirchhoff_without_foundation(thickness):
    """No foundation: shear deformation leaves Mx, My, Mxy *unchanged*.

    The per-mode Mindlin equilibrium forces ``X = -a W``, ``Y = -b W`` with
    ``W`` the Kirchhoff amplitude, so the moment field is mode-by-mode the
    Kirchhoff field although ``W`` itself carries the shear correction.  A
    violation would mean the rotations of the new series are wrong.
    """
    kwargs = dict(side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
                  thickness=thickness, pressure=PRESSURE, n_modes=99)
    points = torch.cat((_center(), _quarter()), dim=0)
    mindlin = navier_mindlin_foundation_moments(points, **kwargs)
    kirchhoff = navier_foundation_moments(points, **kwargs)
    scale = float(kirchhoff.abs().max())
    assert float((mindlin - kirchhoff).abs().max()) <= 1e-10 * scale


@pytest.mark.parametrize("thickness", [0.1, 1.0, 2.0])
def test_deflection_shear_correction_single_mode_is_exact(thickness):
    """Single mode: ``w_mindlin / w_kirchhoff = 1 + D(a^2 + b^2)/S`` exactly."""
    args = dict(side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
                thickness=thickness, pressure=PRESSURE, n_modes=1)
    ratio = float(
        navier_mindlin_foundation_deflection(_center(), **args)[0]
        / navier_foundation_deflection(_center(), **args)[0]
    )
    a = math.pi / SIDE
    inertia = plate_rigidity(YOUNG, POISSON, thickness)
    expected = 1.0 + inertia * (a**2 + a**2) / _shear_stiffness(thickness)
    assert ratio == pytest.approx(expected, rel=1e-12)


def test_thick_plate_gets_thicker_deflection_but_same_moments():
    """The shear correction is ``O((t/a)^2)``: 4x thickness -> 16x w excess."""
    excessive = []
    for thickness in (0.5, 1.0, 2.0):
        args = dict(side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
                    thickness=thickness, pressure=PRESSURE, n_modes=99)
        w_m = float(navier_mindlin_foundation_deflection(_center(), **args)[0])
        w_k = float(navier_foundation_deflection(_center(), **args)[0])
        excessive.append((w_m - w_k) / w_k)
    assert excessive[0] < excessive[1] < excessive[2]
    assert excessive[2] / excessive[0] == pytest.approx(16.0, rel=0.10)


# ---------------------------------------------------------------------------
# 2. foundation coupling breaks the identity: O(k/(G t)) moment shift
# ---------------------------------------------------------------------------

def test_foundation_shear_shift_scales_like_inverse_thickness():
    """With a Winkler bed the moment identity fails at ``O(k/(G t))``.

    ``k = 1e6`` N/m^3 gives 0.107 / 0.057 / 0.029 % centre-``Mx`` shifts at
    ``t = 0.5 / 1 / 2`` m -- the ``1/t`` law of ``k/S``.
    """
    shifts = {}
    for thickness in (0.5, 1.0, 2.0):
        args = dict(side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
                    thickness=thickness, winkler=1.0e6, pressure=PRESSURE,
                    n_modes=99)
        m_m = float(navier_mindlin_foundation_moments(_center(), **args)[0, 0])
        m_k = float(navier_foundation_moments(_center(), **args)[0, 0])
        shifts[thickness] = abs(m_k - m_m) / m_m
    assert 0.0 < shifts[0.5] < 2.0e-3            # small but non-zero
    assert shifts[0.5] > shifts[1.0] > shifts[2.0]
    assert shifts[0.5] / shifts[1.0] == pytest.approx(2.0, rel=0.10)
    assert shifts[1.0] / shifts[2.0] == pytest.approx(2.0, rel=0.10)


def test_weak_foundation_collapses_to_kirchhoff():
    """``k/S -> 0`` (shear rigidity dominant) recovers the thin-plate moments.

    The controlling small parameter is ``k/(G t)``, not ``t/a``: with
    ``k = 1e2`` N/m^3 at ``t = 1`` m the shift is ``~6e-8`` relative.
    """
    args = dict(side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
                thickness=1.0, winkler=1.0e2, pressure=PRESSURE, n_modes=99)
    mindlin = navier_mindlin_foundation_moments(_center(), **args)
    kirchhoff = navier_foundation_moments(_center(), **args)
    assert float(mindlin[0, 0]) == pytest.approx(float(kirchhoff[0, 0]), rel=1e-6)


# ---------------------------------------------------------------------------
# 3. FE (Mindlin Q4/MITC) vs the exact thick-plate series
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("thickness,slenderness", [(1.0, 0.1), (2.0, 0.2)])
def test_fe_thick_plate_moments_within_3pct(thickness, slenderness):
    """``t/a = 0.1, 0.2``: Mx, My, Mxy and w all within 3 % of the exact series."""
    report = compare_finite_plate_moments(
        side_x=SIDE, side_y=SIDE, nx=32, ny=32,
        young=YOUNG, poisson=POISSON, thickness=thickness, pressure=PRESSURE,
        reference="mindlin",
    )
    assert report["reference"] == "mindlin"
    assert report["center_mx_rel_error"] <= 3.0e-2
    assert report["max_mx_rel_error"] <= 3.0e-2
    assert report["max_my_rel_error"] <= 3.0e-2
    assert report["mxy_rel_error"] <= 3.0e-2
    assert report["max_w_rel_error"] <= 3.0e-2


def test_fe_thick_plate_on_winkler_bed_within_3pct():
    """Same gate with a Winkler bed (``k = 1e5`` N/m^3), ``t/a = 0.2``."""
    report = compare_finite_plate_moments(
        side_x=SIDE, side_y=SIDE, nx=32, ny=32,
        young=YOUNG, poisson=POISSON, thickness=2.0, winkler=1.0e5,
        pressure=PRESSURE, reference="mindlin",
    )
    assert report["max_mx_rel_error"] <= 3.0e-2
    assert report["mxy_rel_error"] <= 3.0e-2
    assert report["max_w_rel_error"] <= 3.0e-2


def test_fe_thick_plate_on_pasternak_bed_within_3pct():
    """Two-parameter bed (``k = 1e5``, ``G = 1e6``) at ``t/a = 0.2``.

    The foundation shear modulus enters the series as an extra ``S + G`` in the
    ``w`` equation, so this also exercises the Pasternak branch of the new
    reference.  The Kirchhoff reference would report a bogus 23 % deflection
    error here while the moments stay sub-percent.
    """
    common = dict(side_x=SIDE, side_y=SIDE, nx=32, ny=32, thickness=2.0,
                  winkler=1.0e5, pasternak=1.0e6, young=YOUNG,
                  poisson=POISSON, pressure=PRESSURE)
    mindlin = compare_finite_plate_moments(**common, reference="mindlin")
    kirchhoff = compare_finite_plate_moments(**common, reference="kirchhoff")
    assert mindlin["max_mx_rel_error"] <= 3.0e-2
    assert mindlin["mxy_rel_error"] <= 3.0e-2
    assert mindlin["max_w_rel_error"] <= 3.0e-2
    assert kirchhoff["max_w_rel_error"] >= 0.15


def test_fe_moment_error_converges_under_refinement_thick_plate():
    """Second-order moment convergence is preserved at ``t/a = 0.2``."""
    errors = [
        compare_finite_plate_moments(
            side_x=SIDE, side_y=SIDE, nx=n, ny=n, thickness=2.0,
            young=YOUNG, poisson=POISSON, pressure=PRESSURE, reference="mindlin",
        )["max_mx_rel_error"]
        for n in (16, 32)
    ]
    assert errors[1] < errors[0]
    assert errors[0] / errors[1] >= 3.0          # ~4 for O(h^2)


# ---------------------------------------------------------------------------
# 4. reference-validity boundary (the defect this page fixes)
# ---------------------------------------------------------------------------

def test_kirchhoff_reference_is_invalid_for_thick_plate_deflection():
    """``t/a = 0.2``: FE deflection is 23 % "off" a Kirchhoff truth, 0.04 % off
    the exact Mindlin truth -- an argument for ``reference="mindlin"``."""
    common = dict(side_x=SIDE, side_y=SIDE, nx=32, ny=32, thickness=2.0,
                  young=YOUNG, poisson=POISSON, pressure=PRESSURE)
    thin_ref = compare_finite_plate_moments(**common, reference="kirchhoff")
    thick_ref = compare_finite_plate_moments(**common, reference="mindlin")
    assert thin_ref["max_w_rel_error"] >= 0.15          # bogus 23 % "error"
    assert thick_ref["max_w_rel_error"] <= 1.0e-3       # true 0.04 %
    # The *moment* field is far less sensitive to the reference choice.
    assert thin_ref["max_mx_rel_error"] == pytest.approx(
        thick_ref["max_mx_rel_error"], abs=0.01)


def test_reference_argument_is_validated():
    with pytest.raises(ValueError):
        compare_finite_plate_moments(reference="mindlin-reissner")
    default = compare_finite_plate_moments(nx=8, ny=8)
    assert default["reference"] == "kirchhoff"   # backward compatible default