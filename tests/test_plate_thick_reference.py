"""Tests for the exact FSDT (Mindlin) simply supported plate spectrum.

The reference is only worth something if it is independent of the solver *and*
of the textbook transcription: the closed form is re-derived symbolically in
``scripts/derive_mindlin_sss_determinant.py`` and pinned here, then used to
judge the four-node SRI Mindlin element outside its thin-plate qualification.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from tensorfem.plate_thick_reference import (
    SHEAR_CORRECTION,
    mindlin_sss_modal_frequencies,
    mindlin_sss_spectrum,
)
from tensorfem.plate_vibration import (
    kirchhoff_angular_frequency,
    shear_deformation_bias,
    solve_plate_vibration,
)

STEEL = dict(young=200.0e9, poisson=0.3, density=7850.0)
SIDE = 1.0


def spectrum(thickness: float, **kwargs):
    return mindlin_sss_spectrum(
        length=SIDE, width=SIDE, thickness=thickness, **STEEL, **kwargs
    )


def relative(value: float, reference: float) -> float:
    return value / reference - 1.0


def test_thin_limit_recovers_kirchhoff_navier():
    """a/h = 1e4 must collapse onto the Navier thin-plate frequency."""
    thickness = 1.0e-4
    exact = spectrum(thickness)
    navier = kirchhoff_angular_frequency(
        length=SIDE, width=SIDE, thickness=thickness, **STEEL
    )
    assert relative(exact.kirchhoff_angular, navier) < 1e-12
    assert abs(relative(exact.flexural_angular, navier)) < 1e-5


def test_dropping_rotary_inertia_reproduces_first_order_shear_bias():
    """Without rho h^3/12 the spectrum is exactly the documented first-order bias.

    ``tensorfem.plate_vibration.shear_deformation_bias`` quotes
    ``omega^2 = omega_K^2 / (1 + D R / (kappa G h))``; the FSDT quadratic with
    ``rotary = 0`` must reproduce it to machine precision.  This is the check
    that ties the new reference to the previously documented approximation.
    """
    for aspect in (5.0, 10.0, 20.0, 100.0):
        thickness = SIDE / aspect
        exact = spectrum(thickness, rotary_inertia=False)
        kirchhoff = kirchhoff_angular_frequency(
            length=SIDE, width=SIDE, thickness=thickness, **STEEL
        )
        first_order = shear_deformation_bias(
            length=SIDE,
            width=SIDE,
            thickness=thickness,
            young=STEEL["young"],
            poisson=STEEL["poisson"],
        )
        assert relative(exact.flexural_angular, kirchhoff * (1.0 + first_order)) < 1e-12


def test_infinite_shear_stiffness_leaves_kirchhoff_with_rotary_inertia():
    """kappa -> infinity must leave ``omega^2 = D R^2 / (rho h + rho h^3 R / 12)``."""
    thickness = 0.1
    exact = spectrum(thickness, shear_correction=1.0e9)
    rigidity = STEEL["young"] * thickness**3 / (12.0 * (1.0 - 0.3**2))
    rotary = STEEL["density"] * thickness**3 / 12.0
    wave = exact.wave_number
    closed = math.sqrt(rigidity * wave**2 / (STEEL["density"] * thickness + rotary * wave))
    assert abs(relative(exact.flexural_angular, closed)) < 1e-6


def test_branch_ordering_and_positivity():
    """Bending branch < thickness-shear branch < pure-rotation branch."""
    for aspect in (5.0, 10.0, 50.0, 100.0):
        exact = spectrum(SIDE / aspect)
        assert exact.flexural_angular > 0.0
        assert exact.flexural_angular < exact.thickness_shear_angular
        assert exact.rotation_angular > 0.0
        assert exact.flexural_angular < exact.kirchhoff_angular
        assert exact.shear_bias < 0.0


def test_shear_correction_increases_the_bending_branch():
    """A larger kappa means a stiffer transverse shear term, hence higher f."""
    low = spectrum(0.1, shear_correction=0.5)
    high = spectrum(0.1, shear_correction=1.0)
    assert high.flexural_angular > low.flexural_angular
    assert high.shear_bias > low.shear_bias


def test_square_plate_degeneracy_and_ordering():
    """(1,2) and (2,1) are degenerate; the family sorts by frequency."""
    family = mindlin_sss_modal_frequencies(
        ((1, 1), (1, 2), (2, 1), (2, 2)),
        length=SIDE,
        width=SIDE,
        thickness=0.1,
        **STEEL,
    )
    assert len(family) == 4
    assert math.isclose(family[1].wave_number, family[2].wave_number, rel_tol=1e-12)
    ordered = [item.flexural_angular for item in family]
    assert ordered == sorted(ordered)


def test_rectangular_plate_matches_wave_number_ordering():
    """An a/b = 2 plate has R(2,1) < R(1,2); the spectrum must follow."""
    wide = mindlin_sss_spectrum(
        length=2.0, width=1.0, thickness=0.1, **STEEL,
        half_waves_x=2, half_waves_y=1,
    )
    tall = mindlin_sss_spectrum(
        length=2.0, width=1.0, thickness=0.1, **STEEL,
        half_waves_x=1, half_waves_y=2,
    )
    assert wide.flexural_angular < tall.flexural_angular


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"thickness": 0.0}, ValueError),
        ({"thickness": -0.01}, ValueError),
        ({"thickness": True}, TypeError),
        ({"length": 0.0}, ValueError),
        ({"density": -1.0}, ValueError),
        ({"poisson": 0.5}, ValueError),
        ({"poisson": 1.0}, ValueError),
        ({"half_waves_x": 0}, ValueError),
        ({"half_waves_y": 1.5}, TypeError),
        ({"shear_correction": 0.0}, ValueError),
    ],
)
def test_input_guards(kwargs, error):
    base = dict(length=SIDE, width=SIDE, thickness=0.1, **STEEL)
    base.update(kwargs)
    with pytest.raises(error):
        mindlin_sss_spectrum(**base)


def test_empty_mode_family_is_rejected():
    with pytest.raises(ValueError):
        mindlin_sss_modal_frequencies((), length=SIDE, width=SIDE, thickness=0.1, **STEEL)


@pytest.mark.parametrize("aspect, gate", [(100.0, 0.02), (20.0, 0.02), (10.0, 0.02)])
def test_sri_mindlin_element_tracks_the_fsdt_branch(aspect, gate):
    """The element must converge to its own theory, not only to Kirchhoff."""
    thickness = SIDE / aspect
    reference = spectrum(thickness)
    result = solve_plate_vibration(
        12, 12, length=SIDE, width=SIDE, thickness=thickness, modes=2,
        boundary="simple", **STEEL,
    )
    fe = float(result.frequencies_hz[0])
    assert abs(relative(fe, reference.flexural_hz)) <= gate


def test_thick_plate_is_outside_the_kirchhoff_reference():
    """Honest scope statement: at a/h = 5 Kirchhoff is >10 % too high."""
    exact = spectrum(1.0 / 5.0)
    assert relative(exact.kirchhoff_hz, exact.flexural_hz) > 0.10
    assert abs(exact.shear_bias) > 0.10


def test_shear_correction_constant_is_the_rectangular_section_value():
    assert math.isclose(SHEAR_CORRECTION, 5.0 / 6.0, rel_tol=1e-15)


def test_symbolic_derivation_reproduces_the_library_closed_form():
    """Run the sympy derivation script and require residual == 0."""
    pytest.importorskip("sympy")
    script = Path(__file__).resolve().parents[1] / "scripts" / "derive_mindlin_sss_determinant.py"
    spec = importlib.util.spec_from_file_location("derive_mindlin", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["derive_mindlin"] = module
    spec.loader.exec_module(module)
    assert module.main() == 0