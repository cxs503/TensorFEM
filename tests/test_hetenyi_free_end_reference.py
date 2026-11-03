"""Verification of the general-end-condition Winkler-beam reference solution.

The exact oracle ``finite_beam_bvp`` (Hetényi end-conditioning in matrix form)
is pinned by six independent oracles:

* the exact propagator against ``torch.matrix_exp``;
* the existing Fourier-sine series for simply supported ends;
* the infinite-beam Green's function for all end conditions in the far field;
* global equilibrium ``integral k_b w dx = total load`` for free ends;
* symmetry of the symmetric problem and the rigid-beam limit;
* an independent Euler--Bernoulli FE model with a consistent foundation.

The audit write-up is ``docs/hetanyi-free-end-reference.md``.
"""
import math

import pytest
import torch

from tensorfem.hetenyi_reference import (
    beam_state_propagator,
    finite_beam_bvp,
    finite_beam_series,
    foundation_reaction_integral,
    infinite_beam_point_load,
)
from tensorfem.modal import beam_bending_stiffness

D = torch.float64

EI = 604.8
KB = 4.0e5
LAM = (KB / (4.0 * EI)) ** 0.25
LC = 1.0 / LAM
P = 1000.0


def _fe_free_free(beam_length, elements, load, *, end="free"):
    """Euler--Bernoulli FE beam on a consistent Winkler foundation.

    ``end='free'`` leaves both ends unconstrained (the foundation alone is
    stable); ``end='clamped'`` fixes w and w' at both end nodes.
    """
    le = beam_length / elements
    ndof = 2 * (elements + 1)
    k_winkler = KB * le / 420.0 * torch.tensor(
        [[156.0, 22 * le, 54.0, -13 * le],
         [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
         [54.0, 13 * le, 156.0, -22 * le],
         [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]], dtype=D)
    stiffness = torch.zeros((ndof, ndof), dtype=D)
    for e in range(elements):
        stiffness[2 * e:2 * e + 4, 2 * e:2 * e + 4] += beam_bending_stiffness(EI, le) + k_winkler
    force = torch.zeros(ndof, dtype=D)
    force[2 * (elements // 2)] = load
    fixed = []
    if end == "clamped":
        fixed = [0, 1, ndof - 2, ndof - 1]
    free = [i for i in range(ndof) if i not in fixed]
    displacement = torch.zeros(ndof, dtype=D)
    reduced = stiffness[free][:, free]
    displacement[free] = torch.linalg.solve(reduced, force[free])
    return displacement


def test_propagator_matches_matrix_exponential():
    generator = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [-4.0, 0.0, 0.0, 0.0]],
        dtype=D)
    for xi in (0.0, 0.1, 0.75, 2.5, 7.0):
        reference = torch.matrix_exp(generator * xi)
        assert torch.allclose(beam_state_propagator(xi), reference, rtol=0.0, atol=1e-10), xi
    assert torch.allclose(beam_state_propagator(0.0), torch.eye(4, dtype=D), atol=1e-15)


def test_simply_supported_matches_fourier_series():
    length = 1.0 * LC
    x = torch.linspace(0.01 * length, 0.99 * length, 41, dtype=D)
    bvp = finite_beam_bvp(
        x, beam_length=length, lam=LAM, k_b=KB, EI=EI,
        point_loads=[(length / 2, P)], end_a="simply_supported", end_b="simply_supported")
    series = finite_beam_series(x, length, LAM, KB, EI, [(length / 2, P)],
                                boundary="simply_supported", n_terms=800)
    relative = float(((bvp.deflection - series) / series).abs().max())
    assert relative < 1e-8, relative


@pytest.mark.parametrize("end", ["free", "clamped", "simply_supported", "guided"])
def test_infinite_beam_limit_is_end_condition_independent(end):
    length = 60.0 * LC
    x = torch.linspace(10.0 * LC, 50.0 * LC, 41, dtype=D)
    bvp = finite_beam_bvp(
        x, beam_length=length, lam=LAM, k_b=KB, EI=EI,
        point_loads=[(length / 2, P)], end_a=end, end_b=end)
    reference = infinite_beam_point_load(x, length / 2, P, LAM, KB)
    relative = float(((bvp.deflection - reference) / reference).abs().max())
    assert relative < 1e-3, (end, relative)


@pytest.mark.parametrize("end", ["free", "clamped", "simply_supported", "guided"])
def test_global_equilibrium_identity(end):
    """``integral k_b w dx + V(0) - V(L) = total load``: the exact ODE invariant."""
    length = 2.0 * LC
    x = torch.linspace(0.0, length, 4001, dtype=D)
    solution = finite_beam_bvp(
        x, beam_length=length, lam=LAM, k_b=KB, EI=EI,
        point_loads=[(length / 2, P)], end_a=end, end_b=end)
    foundation_load = foundation_reaction_integral(solution) / P
    end_shear = (float(solution.shear[0]) - float(solution.shear[-1])) / P
    assert foundation_load + end_shear == pytest.approx(1.0, abs=1e-7)
    if end in ("free", "guided"):
        # no vertical end reactions are possible, so the foundation carries all of it
        assert foundation_load == pytest.approx(1.0, abs=1e-7)


def test_free_end_symmetry_and_boundary_conditions():
    length = 2.5 * LC
    x = torch.linspace(0.0, length, 501, dtype=D)
    solution = finite_beam_bvp(
        x, beam_length=length, lam=LAM, k_b=KB, EI=EI, point_loads=[(length / 2, P)])
    assert float(solution.deflection[0]) == pytest.approx(float(solution.deflection[-1]), rel=1e-12)
    mid = len(x) // 2
    assert abs(float(solution.slope[mid])) < 1e-12 * float(solution.slope.abs().max()) + 1e-18
    # free ends carry no moment and no shear
    assert abs(float(solution.moment[0])) < 1e-9
    assert abs(float(solution.moment[-1])) < 1e-9
    assert abs(float(solution.shear[0])) < 1e-9
    assert abs(float(solution.shear[-1])) < 1e-9
    # antisymmetric load keeps the symmetric half intact
    off = finite_beam_bvp(
        x, beam_length=length, lam=LAM, k_b=KB, EI=EI, point_loads=[(0.3 * length, P)])
    assert float(off.deflection.max()) > float(off.deflection.min())


def test_clamped_end_boundary_conditions():
    length = 1.0 * LC
    x = torch.linspace(0.0, length, 201, dtype=D)
    solution = finite_beam_bvp(
        x, beam_length=length, lam=LAM, k_b=KB, EI=EI,
        point_loads=[(length / 2, P)], end_a="fixed", end_b="fixed")
    assert abs(float(solution.deflection[0])) < 1e-15
    assert abs(float(solution.deflection[-1])) < 1e-15
    assert abs(float(solution.slope[0])) < 1e-12
    assert abs(float(solution.slope[-1])) < 1e-12


def test_rigid_beam_limit_gives_uniform_foundation_pressure():
    length = 0.02 * LC
    x = torch.linspace(0.0, length, 101, dtype=D)
    solution = finite_beam_bvp(
        x, beam_length=length, lam=LAM, k_b=KB, EI=EI, point_loads=[(length / 2, P)])
    rigid = P / (KB * length)
    relative = float((solution.deflection / rigid - 1.0).abs().max())
    assert relative < 5e-3, relative


@pytest.mark.parametrize("length_factor", [1.0, 2.0])
def test_fe_free_end_crosscheck(length_factor):
    length = length_factor * LC
    exact = finite_beam_bvp(
        torch.tensor([length / 2], dtype=D), beam_length=length, lam=LAM, k_b=KB, EI=EI,
        point_loads=[(length / 2, P)])
    displacement = _fe_free_free(length, 32, P)
    fe_mid = float(displacement[2 * 16])
    relative = abs(fe_mid / float(exact.deflection[0]) - 1.0)
    assert relative < 0.03, relative


def test_fe_clamped_crosscheck():
    length = 1.0 * LC
    exact = finite_beam_bvp(
        torch.tensor([length / 2], dtype=D), beam_length=length, lam=LAM, k_b=KB, EI=EI,
        point_loads=[(length / 2, P)], end_a="clamped", end_b="clamped")
    displacement = _fe_free_free(length, 32, P, end="clamped")
    relative = abs(float(displacement[2 * 16]) / float(exact.deflection[0]) - 1.0)
    assert relative < 0.03, relative


def test_guided_end_conditions_and_superposition():
    length = 2.0 * LC
    x = torch.linspace(0.0, length, 401, dtype=D)
    guided = finite_beam_bvp(
        x, beam_length=length, lam=LAM, k_b=KB, EI=EI,
        point_loads=[(length / 2, P)], end_a="free", end_b="guided")
    assert abs(float(guided.slope[-1])) < 1e-14
    assert abs(float(guided.shear[-1])) < 1e-9
    a = finite_beam_bvp(x, beam_length=length, lam=LAM, k_b=KB, EI=EI,
                        point_loads=[(0.3 * length, P)])
    b = finite_beam_bvp(x, beam_length=length, lam=LAM, k_b=KB, EI=EI,
                        point_loads=[(0.7 * length, 2.0 * P)])
    both = finite_beam_bvp(x, beam_length=length, lam=LAM, k_b=KB, EI=EI,
                           point_loads=[(0.3 * length, P), (0.7 * length, 2.0 * P)])
    scale = float(both.deflection.abs().max())
    assert float((both.deflection - a.deflection - b.deflection).abs().max()) < 1e-12 * scale


def test_input_validation():
    x = torch.linspace(0.0, 1.0, 5, dtype=D)
    with pytest.raises(ValueError, match="end condition"):
        finite_beam_bvp(x, beam_length=1.0, lam=LAM, k_b=KB, EI=EI, end_a="elastic")
    with pytest.raises(ValueError, match="strictly inside"):
        finite_beam_bvp(x, beam_length=1.0, lam=LAM, k_b=KB, EI=EI, point_loads=[(0.0, P)])
    with pytest.raises(ValueError, match="inside"):
        finite_beam_bvp(torch.tensor([-0.1], dtype=D), beam_length=1.0, lam=LAM, k_b=KB, EI=EI)
    with pytest.raises(ValueError, match="positive"):
        finite_beam_bvp(x, beam_length=1.0, lam=0.0, k_b=KB, EI=EI)
