"""Fail-closed locks for the edge-midpoint transverse-shear probe.

Closes ``docs/plate-3d-exact-anchor.md`` §5.4 / §7⑤: the through-thickness
``tau_xz`` section must be sampled at the edge midpoint ``(0, b/2)``, not at the
plate centre where the field is identically zero.

The numbers asserted here are the *physics* of the exact 3-D solution, not a
regression snapshot: a parabolic shear distribution must reproduce the classical
3/2 peak-to-average factor to well inside the 3 % acceptance band, its integral
must equal the plate-theory shear force, and the centre section must stay at
round-off (so that "quietly moving the probe back to the centre" fails).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorfem.plate_3d_exact import SSPlateExact3D

_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "plate_3d_exact_shear_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("plate_3d_exact_shear_probe", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probe


@pytest.fixture(scope="module")
def data():
    ref = SSPlateExact3D(a=1.0, b=1.0, h=0.1, young=2.0e11, poisson=0.3, q0=1.0e6)
    return _load_probe()(ref, n_z=81)


def test_peak_shear_is_at_the_midsurface(data):
    """The 3-D shear section peaks at z = 0 (no bending/torsion coupling here)."""
    assert abs(data["edge_section"]["z_at_peak_over_h"]) < 1.0 / 80 + 1e-12
    assert 2.30 < data["edge_section"]["tau_xz_peak_over_q0"] < 2.45


def test_parabolic_three_halves_factor(data):
    """peak / (V_x/h) must be the classical 1.5 parabolic factor within 0.5 %."""
    ratio = data["plate_theory"]["peak_over_fsdt_constant"]
    assert abs(ratio - 1.5) / 1.5 < 5e-3


def test_shear_integral_equals_plate_theory_shear_force(data):
    """int tau_xz dz = -V_x (sign convention), magnitude to round-off after Richardson."""
    eq = data["equilibrium"]
    assert eq["residual_rel_magnitude"] < 1e-5      # plain trapezoid on 401 points
    assert eq["richardson_residual_rel_magnitude"] < 1e-9
    assert eq["signed_integral"] * eq["v_x"] < 0.0  # tau_xz = -dpsi_x/dz


def test_dual_section_is_symmetric(data):
    """Square plate: the tau_yz section repeats the tau_xz one."""
    assert data["dual_section"]["tau_yz_peak_over_q0"] == pytest.approx(
        data["edge_section"]["tau_xz_peak_over_q0"], rel=1e-12
    )


def test_centre_section_stays_at_round_off(data):
    """Negative control: the old sampling point cannot carry a physical shear peak."""
    assert data["negative_control_centre"]["tau_xz_max_abs_over_q0"] < 1e-12