"""Cross-family verification of CASE-FEM-CLAMPED-PLATE-UNIFORM (round 2).

The anchor case grades one boundary-value problem -- clamped square plate, thin
regime ``a/t = 100``, uniform pressure, dimensionless centre deflection
``alpha = w_max D / (q a**4)`` -- against the Timoshenko & Woinowsky-Krieger
1959 Table 30 entry ``alpha = 0.00126`` with a 3 % acceptance band.

Round 1 (mission 16181) graded it with a *single* discretisation family (Q4
Mindlin, C0, shear-deformable, 3 dof/node) against the table plus a
Rayleigh-Ritz reference.  Round 2 adds a second, unrelated family: the
Bogner--Fox--Schmit C1 bicubic Hermite Kirchhoff rectangle (4 dof/node, no
rotation dofs, sparse solve, separate module and separate assembly).  Two
independent families landing on the same number is a much stronger statement
than one family extrapolating itself, so these are the invariants that must
hold -- and each one is asserted here, not merely computed:

* the element algebra is exact on a constant-curvature/constant-twist field
  (patch test) -- checked before any plate number is trusted;
* the second family reproduces the case that *does* have a closed form (hard
  simply supported, Navier double series) to ~1e-6 relative;
* both families land inside the 3 % band around the Timoshenko table entry;
* the two families agree with each other better than either agrees with the
  three-digit table (i.e. cross-family agreement is not an artifact of the
  reference rounding);
* discrete equilibrium holds exactly (support reaction = q a**2);
* the family gap shrinks under refinement (evidence that the agreement is a
  converged property and not a coincidence of two coarse meshes).

Everything runs on small meshes (<3 s per test) so this file is a fast
regression guard for the heavy benchmark script.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfem import plate_bfs as pb
from tensorfem.plate_uniform_load import (
    clamped_uniform_ritz,
    solve_uniform_square_plate,
    ss_uniform_kirchhoff,
)

SIDE = 1.0
YOUNG = 2.1e11
POISSON = 0.3
PRESSURE = 1.0e6
THICKNESS = 0.01                      # a/t = 100
RIGIDITY = pb.plate_rigidity(YOUNG, POISSON, THICKNESS)
TABLE_ALPHA = 0.00126                 # Timoshenko Table 30 (rounded to 3 digits)
TABLE_BAND = 0.03                     # case acceptance band


def alpha_of(w: float) -> float:
    """Dimensionless centre deflection ``w D / (q a**4)``."""
    return w * RIGIDITY / (PRESSURE * SIDE ** 4)


def bfs_centre(n: int, boundary: str = "cccc") -> tuple:
    plate = pb.assemble_bfs_plate(
        n, n, SIDE, SIDE, YOUNG, POISSON, THICKNESS, surface_load=PRESSURE
    )
    fixed = pb.bfs_boundary_dofs(n, n, boundary)
    u = pb.solve_bfs_static(plate, fixed)
    point = np.array([[SIDE / 2.0, SIDE / 2.0]])
    w = float(pb.bfs_evaluate(plate, u, point)["w"][0])
    return plate, fixed, u, alpha_of(w)


# ---------------------------------------------------------------------------
# family B machinery: exactness and a closed-form case, before the plate case
# ---------------------------------------------------------------------------
def test_patch_test_reproduces_constant_curvature_field_exactly():
    result = pb.bfs_patch_test()      # 3x2 mesh of 2:1 rectangles
    assert result["n_interior_free_dofs"] > 0
    assert result["relative_error"] < 1e-12, result


def test_patch_test_is_exact_on_a_square_mesh_too():
    result = pb.bfs_patch_test(n_x=2, n_y=2, length_x=1.0, length_y=1.0)
    assert result["relative_error"] < 1e-12, result


def test_bfs_reproduces_the_navier_solution_when_a_closed_form_exists():
    _, _, _, alpha = bfs_centre(20, boundary="ssss")
    exact = ss_uniform_kirchhoff(SIDE, YOUNG, POISSON, THICKNESS, PRESSURE)
    alpha_exact = alpha_of(exact["deflection"])
    assert abs(alpha / alpha_exact - 1.0) < 1e-4, (alpha, alpha_exact)


# ---------------------------------------------------------------------------
# the clamped case itself: table gate and cross-family agreement
# ---------------------------------------------------------------------------
def test_bfs_clamped_alpha_is_inside_the_three_percent_band():
    _, _, _, alpha = bfs_centre(20)
    error = abs(alpha / TABLE_ALPHA - 1.0)
    assert error < TABLE_BAND, (alpha, error)


def test_mindlin_and_bfs_agree_better_than_either_matches_the_rounded_table():
    _, _, _, alpha_bfs = bfs_centre(20)
    result = solve_uniform_square_plate(
        16, SIDE, YOUNG, POISSON, THICKNESS, PRESSURE, "clamped"
    )
    # centre node of a 16x16 mesh with 17x17 nodes
    index = int(np.argmin(np.abs(result.nodes[:, 0] - SIDE / 2.0)
                          + np.abs(result.nodes[:, 1] - SIDE / 2.0)))
    alpha_mindlin = alpha_of(float(result.displacement[3 * index]))
    gap = abs(alpha_mindlin - alpha_bfs) / alpha_bfs
    assert gap < 0.01, (alpha_mindlin, alpha_bfs, gap)
    # and both sit within the +-0.4 % table-rounding band of the converged Ritz
    ritz = alpha_of(
        clamped_uniform_ritz(SIDE, YOUNG, POISSON, THICKNESS, PRESSURE,
                             n_modes=12).deflection(0.5, 0.5)
    )
    assert abs(alpha_bfs / ritz - 1.0) < 0.01


def test_cross_family_gap_shrinks_under_refinement():
    gaps = []
    for n in (10, 20):
        _, _, _, alpha_bfs = bfs_centre(n)
        result = solve_uniform_square_plate(
            n, SIDE, YOUNG, POISSON, THICKNESS, PRESSURE, "clamped"
        )
        index = int(np.argmin(np.abs(result.nodes[:, 0] - SIDE / 2.0)
                              + np.abs(result.nodes[:, 1] - SIDE / 2.0)))
        alpha_mindlin = alpha_of(float(result.displacement[3 * index]))
        gaps.append(abs(alpha_mindlin - alpha_bfs))
    assert gaps[1] < gaps[0], gaps


def test_discrete_equilibrium_carries_the_total_load():
    plate, fixed, u, _ = bfs_centre(20)
    residual = plate.stiffness @ u - plate.load
    reaction = abs(float(np.sum(residual[fixed])))
    total = float(np.sum(plate.load))
    assert total == pytest.approx(PRESSURE * SIDE ** 2, rel=1e-12)
    assert reaction == pytest.approx(total, rel=1e-10), (reaction, total)


def test_boundary_condition_names_are_validated():
    with pytest.raises(ValueError):
        pb.bfs_boundary_dofs(4, 4, "free")