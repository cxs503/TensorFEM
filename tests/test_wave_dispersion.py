"""Gates for the Rayleigh--Lamb dispersion anchor (docs/lamb-wave-dispersion-verification.md).

The analytic oracle gates are positive verification.  The Bloch--Floquet gates are
fail-closed documentation of an *open* defect: they assert the defect is still there,
so fixing it forces this file and the document to be updated together.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfem.wave_dispersion import (
    bloch_cell_frequencies,
    bloch_cell_matrices,
    flexural_phase_velocity,
    lamb_residual,
    longitudinal_speed,
    plate_speed,
    rayleigh_lamb_frequencies,
    rayleigh_speed,
    shear_speed,
)

E, NU, RHO, H = 210.0e9, 0.3, 7850.0, 0.01
RESULT = Path(__file__).resolve().parents[1] / "benchmarks/verified/fem_wave_dispersion/result.json"


def test_wave_speeds_are_closed_form():
    cT, cL = shear_speed(E, NU, RHO), longitudinal_speed(E, NU, RHO)
    assert cL / cT == pytest.approx(math.sqrt((1 - NU) / (0.5 - NU)), rel=1e-12)
    assert plate_speed(E, NU, RHO) / cT == pytest.approx(1.690309, rel=2e-2)
    assert rayleigh_speed(E, NU, RHO) / cT == pytest.approx(0.927413, rel=1e-5)


def test_s0_converges_to_plate_velocity():
    """S0 phase velocity extrapolated to kh -> 0 equals sqrt(E/(rho(1-nu^2)))."""
    speeds = []
    for kh in (0.05, 0.1, 0.2):
        k = kh / H
        speeds.append(rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=1)["symmetric"][0] / k)
    # Richardson-consistent quadratic fit in kh^2 (three points, cp = leading term).
    import numpy as np

    xs = np.array([kh ** 2 for kh in (0.05, 0.1, 0.2)])
    coef, *_ = np.linalg.lstsq(np.vstack([np.ones(3), xs]).T, np.array(speeds), rcond=None)
    assert abs(coef[0] / plate_speed(E, NU, RHO) - 1.0) < 2e-3


def test_a0_matches_thin_plate_and_rayleigh_limits():
    k = 0.05 / H
    thin = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=1)["antisymmetric"][0] / k
    assert abs(thin / flexural_phase_velocity(k, E, NU, RHO, H) - 1.0) < 1e-2
    k = 20.0 / H
    thick = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=1)["antisymmetric"][0] / k
    assert abs(thick / rayleigh_speed(E, NU, RHO) - 1.0) < 1e-2


def test_roots_satisfy_the_equation():
    cL, cT = longitudinal_speed(E, NU, RHO), shear_speed(E, NU, RHO)
    for kh in (0.1, 0.5, 2.0):
        k = kh / H
        roots = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=2)
        for branch, key in (("symmetric", "symmetric"), ("antisymmetric", "antisymmetric")):
            for omega in roots[key]:
                assert abs(lamb_residual(omega, k, cL, cT, H, branch)) < 2e-2


def test_consistent_mass_conserves_mass():
    cell = bloch_cell_matrices(0.0, half_thickness=H, young=E, poisson=NU, density=RHO,
                              nx=8, ny=4)
    # One rho*int(Ni Nj) block per translation direction.
    assert cell.total_mass == pytest.approx(2.0 * RHO * cell.cell_length * 2.0 * H, rel=1e-12)


def test_bloch_floquet_reduction_defect_is_open():
    """FAIL-CLOSED negative gate: the Floquet reduction does not yet work.

    At the Gamma point the periodic cell must own five zero eigenvalues (three
    rigid modes plus the A0/S0 acoustic branches).  It does not, and the resulting
    phase-velocity error is mesh-insensitive, so the fault is in the reduction and
    not in the discretisation.  See section 8 of the anchor document.  When this
    test starts failing, the defect has been fixed -- update the document and
    gates_verified.json in the same commit.
    """
    omega = bloch_cell_frequencies(0.0, half_thickness=H, young=E, poisson=NU,
                                   density=RHO, nx=8, ny=4, n_modes=8)
    assert omega[4] / omega[5] > 1e-3, "Floquet reduction now reproduces the Gamma-point zeros?"
    k = 0.5 / H
    exact = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=1)["antisymmetric"][0]
    fem = bloch_cell_frequencies(k, half_thickness=H, young=E, poisson=NU, density=RHO,
                                 nx=16, ny=8, n_modes=2)[0]
    assert abs(fem / exact - 1.0) > 3e-2, "Bloch cell now reproduces the A0 branch?"


def test_recorded_anchor_numbers_match_recomputation():
    if not RESULT.exists():
        pytest.skip("anchor result.json not present")
    data = json.loads(RESULT.read_text())
    k = 0.5 / H
    exact = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=1)
    row = next(r for r in data["analytic"] if r["kh"] == 0.5)
    assert row["a0_phase"] == pytest.approx(exact["antisymmetric"][0] / k / shear_speed(E, NU, RHO), rel=2e-2)
    assert row["s0_phase"] == pytest.approx(exact["symmetric"][0] / k / shear_speed(E, NU, RHO), rel=2e-2)
    assert data["gates"]["s0_plate_velocity_extrapolated"]["verdict"] == "PASS"
    assert data["gates"]["fem_phase_error_fine"]["verdict"] == "FAIL"
