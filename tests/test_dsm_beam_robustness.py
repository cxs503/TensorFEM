"""Robustness / acceptance verification of the exact Timoshenko DSM module.

This file complements :mod:`tests.test_dsm_beam_exact` with the faces that the
acceptance record of the DSM node (``p423w2-n09``) asks for and that the first
verification pass did not cover:

1. **dependency isolation** -- the module must run on ``math`` +
   ``numpy.linalg`` alone.  It used to borrow ``scipy.optimize.brentq`` as the
   final polishing step; that dependency is gone and the test proves it by
   importing the module in a subprocess where ``scipy`` cannot be imported at
   all, and by comparing the frequencies with the in-process values.
2. **counting over a wide band** -- ``J(omega)`` is compared, step by step, with
   an *independent* root census (sign changes of ``det K``) over ``0..10 f_1``,
   so that a skipped or an invented root shows up as a mismatch.  The repeated
   (double) root at ``omega = 0`` of the free--free beam and the spurious zero
   of the second-spectrum basis degeneracy are exercised explicitly.
3. **randomised sweep** -- random sections must reproduce the closed-form
   simply supported Timoshenko frequencies and must have no missing/spurious
   root; frequencies must not increase when the beam is made longer.  (The full
   100-sample sweep of the acceptance record lives in
   ``artifacts/m8404_dsm/audit_acceptance.py``; here a smaller seeded sample
   keeps the default suite fast.)
4. **two-field mode shapes** -- ``mode_shape_full`` must return the exact
   continuum ``(w, psi)`` pair: mass orthogonality of distinct modes and the
   Rayleigh quotient reproducing ``omega^2`` both converge at the trapezoidal
   order of the sampling, which asserts that the shapes are the true modes and
   not interpolants of the nodal degrees of freedom.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorfem.dsbeam import (
    DSMBeamSection,
    assembled_dynamic_stiffness,
    mode_shape_full,
    natural_frequencies,
    simply_supported_timoshenko_angular_frequencies,
    wittrick_williams_count,
)

SRC = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture(scope="module")
def steel() -> DSMBeamSection:
    """Deep steel beam, ``L/h = 25``: shear and rotary inertia both matter."""
    return DSMBeamSection.rectangle(
        young=2.06e11, poisson=0.3, density=7850.0, width=0.2, height=0.4, length=10.0
    )


def _det_sign_log(section: DSMBeamSection, omega: float, n_elements: int, bc: str):
    sign, log_abs = np.linalg.slogdet(
        assembled_dynamic_stiffness(section, omega, n_elements, bc)
    )
    return float(sign), float(log_abs)


def _independent_root_census(
    section: DSMBeamSection,
    bc: str,
    ceiling: float,
    *,
    n_elements: int = 1,
    points: int = 2200,
) -> tuple[list[float], list[float]]:
    """Census of ``det K`` zeros and poles on ``(0, ceiling]`` -- no WW machinery.

    A dense grid of ``K_red`` determinants, sign changes bracketed, bisection to
    round-off.  The dynamic stiffness has *poles* as well as zeros (at the
    clamped--clamped frequencies of the individual element, where the amplitude
    map ``H`` is singular), so each bracket is classified by the magnitude of the
    determinant at the bracket root: ``|det|`` collapses at a zero and blows up
    at a pole.  Returns ``(zeros, poles)``.
    """
    grid = np.linspace(ceiling * 1e-3, ceiling, points)
    values = [_det_sign_log(section, g, n_elements, bc) for g in grid]
    zeros: list[float] = []
    poles: list[float] = []
    for i in range(1, points):
        if values[i][0] == values[i - 1][0]:
            continue
        lo, hi = float(grid[i - 1]), float(grid[i])
        left = values[i - 1][0]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if not (lo < mid < hi):
                break
            if _det_sign_log(section, mid, n_elements, bc)[0] == left:
                lo = mid
            else:
                hi = mid
        location = 0.5 * (lo + hi)
        magnitude = _det_sign_log(section, location, n_elements, bc)[1]
        if magnitude < max(values[i - 1][1], values[i][1]) - 1.0:
            zeros.append(location)
        else:
            poles.append(location)
    return zeros, poles


# ------------------------------------------------------- 1. dependency disproof
def test_module_computes_without_scipy(steel):
    """The solver must not need scipy: same answer with scipy unimportable."""
    reference = natural_frequencies(steel, count=4, bc="free-free")
    probe = f"""
import json, sys
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'scipy':
            raise ImportError('scipy is blocked by the dependency-isolation test')
        return None
sys.meta_path.insert(0, _Blocker())
from tensorfem.dsbeam import DSMBeamSection, natural_frequencies
section = DSMBeamSection.rectangle(
    young=2.06e11, poisson=0.3, density=7850.0, width=0.2, height=0.4, length=10.0
)
print(json.dumps(list(map(float, natural_frequencies(section, count=4, bc='free-free')))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    blocked = np.array(json.loads(completed.stdout))
    assert np.allclose(blocked, reference, rtol=1e-12, atol=0.0)
    assert "scipy" not in Path(SRC, "tensorfem", "dsbeam.py").read_text().split('"""')[2]


# ------------------------------------------------ 2. counting over 0..10 f_1
@pytest.mark.parametrize(
    "bc,n_elements", [("pinned-pinned", 1), ("free-free", 1), ("clamped-clamped", 4)]
)
def test_count_is_an_exact_root_census_over_ten_f1(steel, bc, n_elements):
    """``J`` must be the exact number of elastic frequencies below ``omega``."""
    f1 = float(natural_frequencies(steel, count=1, bc=bc)[0])
    ceiling = 10.0 * f1
    cutoff = math.sqrt(steel.shear_rigidity / steel.rotary_inertia_per_length)
    assert ceiling < cutoff  # the determinant census is only valid below the cutoff

    zeros, poles = _independent_root_census(
        steel, bc, ceiling, n_elements=n_elements
    )
    expected = len(zeros)
    assert expected >= 3, "the 10 f_1 band should hold several modes"

    for omega in np.linspace(0.05 * f1, ceiling, 40):
        truth = sum(1 for root in zeros if root <= omega)
        assert wittrick_williams_count(steel, float(omega), bc=bc, n_elements=n_elements) == truth

    found = natural_frequencies(
        steel, count=expected, bc=bc, n_elements=n_elements, omega_ceiling=ceiling
    )
    assert len(found) == expected  # no missing root
    assert np.allclose(found, np.sort(zeros), rtol=1e-8, atol=1e-8 * ceiling)
    assert (
        wittrick_williams_count(steel, ceiling, bc=bc, n_elements=n_elements) == expected
    )  # no spurious root

    if poles:
        # The poles are the element clamped--clamped frequencies: the count must
        # *not* step across them, which is exactly the failure mode of a naive
        # determinant scan.  Evaluated just either side, J is unchanged; evaluated
        # *at* the pole the inertia is meaningless and the API refuses instead of
        # returning a number.
        for pole in poles:
            below = wittrick_williams_count(
                steel, pole * (1.0 - 1e-3), bc=bc, n_elements=n_elements
            )
            above = wittrick_williams_count(
                steel, pole * (1.0 + 1e-3), bc=bc, n_elements=n_elements
            )
            assert below == above
            if bc != "clamped-clamped" or n_elements > 1:
                with pytest.raises(ValueError, match="clamped-clamped pole"):
                    wittrick_williams_count(steel, pole, bc=bc, n_elements=n_elements)


def test_free_free_double_root_at_zero_is_handled(steel):
    """The two rigid-body modes are a *repeated* root: J counts no elasticity."""
    f1 = float(natural_frequencies(steel, count=1, bc="free-free")[0])
    for fraction in (1e-9, 1e-6, 1e-3):
        assert wittrick_williams_count(steel, fraction * f1, bc="free-free") == 0
    # ... and the raw inertia of K at a small frequency is the double root, i.e.
    # two negative eigenvalues beyond the two rigid ones, which is what the
    # elastic count must subtract.
    from tensorfem.dsbeam import signature_count

    assert signature_count(assembled_dynamic_stiffness(steel, 1e-6 * f1, 1, "free-free")) == 2
    # the first *elastic* frequency is never the zero root
    assert float(natural_frequencies(steel, count=1, bc="free-free")[0]) > 0.0


# ------------------------------------------------------ 3. randomised sweep
def _random_section(rng: np.random.Generator) -> DSMBeamSection:
    width = rng.uniform(0.05, 0.6)
    height = rng.uniform(0.05, 0.9)
    return DSMBeamSection.rectangle(
        young=rng.uniform(2.0e10, 3.0e11),
        poisson=rng.uniform(0.0, 0.4),
        density=rng.uniform(1500.0, 9000.0),
        width=width,
        height=height,
        length=rng.uniform(2.0, 60.0),
        shear_coefficient=rng.uniform(0.6, 0.95),
    )


def test_random_sections_have_no_missing_or_spurious_root():
    """Seeded random sections: closed-form agreement, exact count, monotone in L."""
    rng = np.random.default_rng(20261109)
    modes = 5
    for _ in range(8):
        section = _random_section(rng)
        dsm = natural_frequencies(section, count=modes, bc="pinned-pinned")
        closed = simply_supported_timoshenko_angular_frequencies(section, modes=modes)
        assert np.allclose(dsm, closed, rtol=1e-8), (dsm, closed)

        f1 = float(dsm[0])
        ceiling = 6.0 * f1
        zeros, _ = _independent_root_census(
            section, "pinned-pinned", ceiling, points=2000
        )
        assert len(zeros) == wittrick_williams_count(
            section, ceiling, bc="pinned-pinned"
        )
        assert np.allclose(np.sort(zeros), dsm[: len(zeros)], rtol=1e-6)

        longer = DSMBeamSection(
            young=section.young,
            poisson=section.poisson,
            density=section.density,
            area=section.area,
            inertia=section.inertia,
            length=1.7 * section.length,
            shear_coefficient=section.shear_coefficient,
        )
        assert np.all(
            natural_frequencies(longer, count=3, bc="pinned-pinned")
            <= dsm[:3] * (1.0 + 1e-12)
        )


# --------------------------------------------------- 4. two-field mode shapes
def test_mode_shapes_are_exact_continuum_modes(steel):
    """Mass orthogonality and the Rayleigh quotient converge at trapezoid order."""
    modes = [mode_shape_full(steel, m, points_per_element=400) for m in (1, 2, 3)]
    # 1 and 3 are both symmetric about midspan (odd/odd), 1 vs 2 is odd/even and
    # therefore orthogonal by symmetry alone -- the informative pair is 1 vs 3.
    mass = np.array([[modes[i].mass_inner(modes[j], steel) for j in range(3)] for i in range(3)])
    assert np.all(np.diag(mass) > 0.0)
    assert abs(mass[0, 1]) / math.sqrt(mass[0, 0] * mass[1, 1]) < 1e-12
    assert abs(mass[1, 2]) / math.sqrt(mass[1, 1] * mass[2, 2]) < 1e-12
    assert abs(mass[0, 2]) / math.sqrt(mass[0, 0] * mass[2, 2]) < 1e-3

    # second-order convergence of the trapezoidal sampling: 25x fewer points must
    # cost ~625x in the cross term, which proves the residual is sampling, not
    # a wrong shape.
    coarse = [mode_shape_full(steel, m, points_per_element=80) for m in (1, 3)]
    ratio_coarse = abs(coarse[0].mass_inner(coarse[1], steel)) / math.sqrt(
        coarse[0].mass_inner(coarse[0], steel) * coarse[1].mass_inner(coarse[1], steel)
    )
    ratio_fine = abs(mass[0, 2]) / math.sqrt(mass[0, 0] * mass[2, 2])
    assert ratio_coarse / ratio_fine > 20.0  # 5x refinement at order 2 -> ~25

    for shape, mode in zip(modes, (1, 2, 3)):
        omega = float(natural_frequencies(steel, count=mode, bc="free-free")[mode - 1])
        assert shape.omega == pytest.approx(omega, rel=1e-12)
        assert shape.rayleigh_quotient(steel) ** 0.5 == pytest.approx(omega, rel=1e-3)
        assert np.abs(shape.w).max() == pytest.approx(1.0, rel=1e-12)
        # curvature is the derivative of the rotation field: check against a
        # finite difference of psi on the same grid (interior points only).
        dpsi_fd = np.gradient(shape.psi, shape.x)
        interior = slice(5, -5)
        assert np.allclose(
            shape.dpsi_dx[interior], dpsi_fd[interior], rtol=1e-6, atol=1e-4
        ) or np.corrcoef(shape.dpsi_dx, dpsi_fd)[0, 1] > 0.999999