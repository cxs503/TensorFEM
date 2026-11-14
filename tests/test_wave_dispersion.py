"""Gates for the Rayleigh--Lamb dispersion anchor (docs/lamb-wave-dispersion-verification.md).

Every gate here is positive verification: the analytic roots must satisfy the
Rayleigh--Lamb equation (including the higher ones the anchor does not score) and
the Bloch--Floquet cell must reproduce the analytic branches.  No test asserts that
a known defect is still present.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import numpy as np
from tensorfem.continuum import rectangular_q4_mesh

from tensorfem.wave_dispersion import (
    bloch_cell_frequencies,
    bloch_cell_matrices,
    bloch_cell_modes,
    midplane_family_fraction,
    reduced_dof_count,
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


def test_lowest_roots_satisfy_the_equation():
    """The two roots the anchor scores (lowest of each family) are true roots."""
    cL, cT = longitudinal_speed(E, NU, RHO), shear_speed(E, NU, RHO)
    for kh in (0.1, 0.5, 2.0):
        k = kh / H
        roots = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=2)
        for branch in ("symmetric", "antisymmetric"):
            omega = roots[branch][0]
            assert abs(lamb_residual(omega, k, cL, cT, H, branch)) < 2e-2


def test_higher_roots_are_genuine():
    """Positive gate: the multi-root scan returns only genuine roots.

    The scanner used to accept sign changes across the ``tan``/``tanh`` poles as
    roots: at ``q = 0`` (``omega = k c_T``) the ``4 k^2 p q`` term vanishes
    identically, and near-tangential brackets were accepted too, so
    ``rayleigh_lamb_frequencies(n_roots>=2)`` reported candidates whose normalised
    residual reached O(0.1)..1 -- e.g. a spurious antisymmetric root at
    ``omega/k/c_T = 1.0``.  Each bisection result is now re-evaluated and kept only
    if its residual is below ``1e-6``, so every reported root -- not just the
    lowest one the anchor scores -- must satisfy the Rayleigh--Lamb equation.
    """
    cL, cT = longitudinal_speed(E, NU, RHO), shear_speed(E, NU, RHO)
    for kh in (0.1, 0.5, 2.0):
        k = kh / H
        roots = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=2)
        for branch in ("symmetric", "antisymmetric"):
            assert len(roots[branch]) == 2
            for omega in roots[branch]:
                assert abs(lamb_residual(omega, k, cL, cT, H, branch)) < 1e-6


def test_scan_reports_no_root_off_the_equation_on_the_anchor_grid():
    """Sweep the anchor's own ``kh`` grid: no reported root is a pole crossing.

    Same three-branch request the anchor's analytic table makes (``n_roots=3``).
    Before the fix the antisymmetric list carried a pseudo-root at
    ``omega/k/c_T = 1.0`` (residual 1.0) and one pole crossing per cut-off
    (residual ~0.7) at every ``kh``.
    """
    cL, cT = longitudinal_speed(E, NU, RHO), shear_speed(E, NU, RHO)
    worst = 0.0
    for kh in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        k = kh / H
        roots = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=3)
        for branch in ("symmetric", "antisymmetric"):
            assert roots[branch], f"no root found at kh={kh}, {branch}"
            assert roots[branch] == sorted(roots[branch])
            for omega in roots[branch]:
                worst = max(worst, abs(lamb_residual(omega, k, cL, cT, H, branch)))
    assert worst < 1e-6


def test_consistent_mass_conserves_mass():
    cell = bloch_cell_matrices(0.0, half_thickness=H, young=E, poisson=NU, density=RHO,
                              nx=8, ny=4)
    # One rho*int(Ni Nj) block per translation direction.
    assert cell.total_mass == pytest.approx(2.0 * RHO * cell.cell_length * 2.0 * H, rel=1e-12)


def test_floquet_reduction_lives_in_dof_space():
    """Regression for the node/dof aliasing defect (docs §8).

    The reduced cell carries two unknowns per retained node.  Numbering the reduced
    unknowns by node id collapses ``u_x`` and ``u_z`` of distinct nodes into one
    column, which is how the pre-fix anchor produced a mesh-converging but wrong
    dispersion curve.
    """
    for kh in (0.0, 0.2, 1.0):
        k = kh / H
        cell = bloch_cell_matrices(k, half_thickness=H, young=E, poisson=NU,
                                   density=RHO, nx=8, ny=4)
        coords = cell.nodes.numpy()
        n_right = int(np.sum(np.isclose(coords[:, 0], cell.cell_length)))
        assert reduced_dof_count(cell) == 2 * (len(coords) - n_right)
        assert cell.transform.shape == (2 * len(coords), 2 * (len(coords) - n_right))


def test_gamma_point_owns_two_rigid_translations():
    """Periodic cell at k=0: two zero modes (translations), not one, not five."""
    omega = bloch_cell_frequencies(0.0, half_thickness=H, young=E, poisson=NU,
                                   density=RHO, nx=8, ny=4, n_modes=8)
    assert omega[0] <= 1e-12 * omega[-1]
    assert omega[1] <= 1e-10 * omega[-1]
    assert omega[2] > 1e-3 * omega[-1]


def test_parity_classifier_selects_a0_and_s0_without_the_oracle():
    """Branch picking by mid-plane parity must land on A0/S0 at every kh."""
    for kh in (0.2, 0.5, 1.0):
        k = kh / H
        exact = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=1)
        nodes, _ = rectangular_q4_mesh(2.0 * math.pi / (4.0 * k), 2.0 * H, 16, 8)
        omega, modes = bloch_cell_modes(k, half_thickness=H, young=E, poisson=NU,
                                        density=RHO, nx=16, ny=8, n_modes=8)
        fractions = [midplane_family_fraction(nodes, modes[:, i]) for i in range(len(omega))]
        a_index = next(i for i, fr in enumerate(fractions) if fr < 0.5)
        s_index = next(i for i, fr in enumerate(fractions) if fr >= 0.5)
        assert abs(omega[a_index] / exact["antisymmetric"][0] - 1.0) < 5e-2
        assert abs(omega[s_index] / exact["symmetric"][0] - 1.0) < 5e-2


def test_fine_cell_matches_the_analytic_branches():
    """Positive verification gate: A0 and S0 within 1% on the 32x16 cell, kh<=1."""
    for kh in (0.2, 0.5, 1.0):
        k = kh / H
        exact = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=1)
        nodes, _ = rectangular_q4_mesh(2.0 * math.pi / (4.0 * k), 2.0 * H, 32, 16)
        omega, modes = bloch_cell_modes(k, half_thickness=H, young=E, poisson=NU,
                                        density=RHO, nx=32, ny=16, n_modes=8)
        fractions = [midplane_family_fraction(nodes, modes[:, i]) for i in range(len(omega))]
        a_index = next(i for i, fr in enumerate(fractions) if fr < 0.5)
        s_index = next(i for i, fr in enumerate(fractions) if fr >= 0.5)
        assert abs(omega[a_index] / exact["antisymmetric"][0] - 1.0) < 1e-2
        assert abs(omega[s_index] / exact["symmetric"][0] - 1.0) < 1e-2


def test_recorded_anchor_numbers_match_recomputation():
    if not RESULT.exists():
        pytest.skip("anchor result.json not present")
    data = json.loads(RESULT.read_text())
    k = 0.5 / H
    exact = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=1)
    row = next(r for r in data["analytic"] if r["kh"] == 0.5)
    assert row["a0_phase"] == pytest.approx(exact["antisymmetric"][0] / k / shear_speed(E, NU, RHO), rel=2e-2)
    assert row["s0_phase"] == pytest.approx(exact["symmetric"][0] / k / shear_speed(E, NU, RHO), rel=2e-2)
    gates = data["gates"]
    assert gates["s0_plate_velocity_extrapolated"]["passed"] is True
    assert gates["fem_phase_error_fine"]["passed"] is True
    assert gates["fem_phase_error_fine"]["value"] < 3e-2
    assert gates["gamma_zero_mode_count"]["value"] == 2.0
    # a branch picked positionally would score S0 against A1 and blow up here
    assert all(row["s0_error"] < 3e-2 for row in data["fem"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
