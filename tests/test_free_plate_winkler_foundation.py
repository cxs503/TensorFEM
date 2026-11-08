"""Free-edge plate on a Winkler foundation: verification against exact references.

The free-edge path of ``tensorfem.plate_vibration`` was verified for ``k_w = 0``
in ``tests/test_free_plate_vibration.py``; the combination *free edge + elastic
foundation* was explicitly listed as not verified.  This module locks it with
three references that do not come from the FE solver:

1. ``omega^2 = k_w / (rho h)`` for the vertical translation mode -- exact for any
   boundary condition, and exact for a consistent Q4 element set (the foundation
   matrix and the translatory mass share the same shape-function products);
2. the exact tributary-area row sums of the assembled foundation matrix;
3. the shifted free-plate Rayleigh--Ritz spectrum
   ``lambda_k^2(kappa) = lambda_k^2(0) + kappa`` (``kappa = k_w a^4 / D``), which
   holds for *every* Kirchhoff eigenpair because the Winkler potential is exactly
   proportional to the mass operator.

Every assertion is fail-closed: the coarse-mesh third mode that misses the 3 %
gate is asserted to miss it, so a later refactor cannot turn a documented failure
into a silent pass.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfem.plate_mindlin import q4_winkler_foundation_stiffness
from tensorfem.plate_ritz_reference import (
    foundation_energy_matrix,
    legendre_operator_matrices,
    ritz_free_plate_eigenvalues,
)
from tensorfem.plate_vibration import (
    BOUNDARY_FREE,
    plate_rigidity,
    q4_plate_mass_matrix,
    solve_plate_vibration,
    structured_rectangular_mesh,
    winkler_translation_angular_frequency,
)

YOUNG = 200.0e9
POISSON = 0.30
THICKNESS = 0.01
DENSITY = 7850.0
LENGTH = 1.0
RIGIDITY = plate_rigidity(YOUNG, POISSON, THICKNESS)
AREAL = DENSITY * THICKNESS
DTYPE = torch.float64
SOFT_MODES = 4


def _assemble(mesh, length, width, winkler_k):
    """Global foundation translatory block and translatory mass block."""
    nodes, elements = structured_rectangular_mesh(
        mesh[0], mesh[1], length, width, dtype=DTYPE
    )
    count = nodes.shape[0]
    foundation = torch.zeros((count, count), dtype=DTYPE)
    mass = torch.zeros((count, count), dtype=DTYPE)
    for connectivity in elements:
        xy = nodes[connectivity]
        foundation[connectivity[:, None], connectivity] += (
            q4_winkler_foundation_stiffness(xy, winkler_k)[0::3, 0::3]
        )
        mass[connectivity[:, None], connectivity] += (
            q4_plate_mass_matrix(xy, DENSITY, THICKNESS)[0::3, 0::3]
        )
    return nodes, elements, foundation, mass


def _ritz(kappa: float, count: int = 8, width: float | None = None):
    return ritz_free_plate_eigenvalues(
        length=LENGTH, width=width, poisson=POISSON, order=14, count=count,
        winkler_parameter=kappa,
    )


def _ritz_angular_squared(width: float | None = None, count: int = 8):
    """Ritz ``omega^2`` in physical units (Kirchhoff, ``kappa = 0``)."""
    values = _ritz(0.0, count=count, width=width).dimensionless**2
    return values * RIGIDITY / (AREAL * LENGTH**4)


def _free(mesh, winkler_k, modes=10, width=None, length=LENGTH, **kwargs):
    return solve_plate_vibration(
        mesh[0], mesh[1], length=length, width=width, young=YOUNG, poisson=POISSON,
        thickness=THICKNESS, density=DENSITY, boundary=BOUNDARY_FREE, modes=modes,
        winkler_k=winkler_k, **kwargs,
    )


# --------------------------------------------------------------- reference side
def test_foundation_operator_reproduces_analytic_moments():
    """``W`` must integrate constants and coordinate products exactly."""
    order, a, b = 8, 1.0, 0.5
    matrix = foundation_energy_matrix(order=order, length=a, width=b)
    count = order + 1

    def moment(vector):
        return float(vector @ matrix @ vector)

    constant = np.zeros(count * count)
    constant[0] = 1.0
    assert moment(constant) == pytest.approx(a * b, rel=1e-14)

    xi = np.zeros(count * count)
    xi[1 * count] = 1.0  # P_1 in x, P_0 in y
    assert moment(xi) == pytest.approx(a * b / 3.0, rel=1e-13)

    eta = np.zeros(count * count)
    eta[0 * count + 1] = 1.0  # P_0 in x, P_1 in y
    assert moment(eta) == pytest.approx(a * b / 3.0, rel=1e-13)

    mixed = np.zeros(count * count)
    mixed[1 * count + 1] = 1.0
    assert moment(mixed) == pytest.approx(a * b / 9.0, rel=1e-13)


def test_foundation_operator_agrees_with_the_kronecker_mass_operator():
    """Two different quadrature rules must assemble the same operator."""
    for width in (None, 0.5):
        result = _ritz(1.0, count=4, width=width)
        assert result.foundation_check < 1e-12
    # And the same check on the plain mass operator path, order by order.
    for order in (6, 10, 14):
        length = 1.0
        width = 0.5
        mass, _, _, _ = legendre_operator_matrices(order, length)
        mass_y, _, _, _ = legendre_operator_matrices(order, width)
        kronecker = np.kron(mass, mass_y)
        operator = foundation_energy_matrix(order=order, length=length, width=width)
        assert np.linalg.norm(operator - kronecker) / np.linalg.norm(kronecker) < 1e-12


def test_ritz_foundation_shift_identity_is_a_small_residual():
    """``lambda^2(kappa) - lambda^2(0) = kappa`` for every retained elastic mode."""
    for kappa in (0.5, 1.0, 10.0):
        base = _ritz(0.0, count=6)
        shifted = _ritz(kappa, count=9)
        residual = shifted.dimensionless[3:] ** 2 - base.dimensionless**2 - kappa
        assert np.max(np.abs(residual)) < 1e-5 * kappa
    # The three lifted rigid modes must sit at ``kappa`` themselves.
    shifted = _ritz(1.0, count=6)
    assert np.max(np.abs(shifted.dimensionless[:3] ** 2 - 1.0)) < 1e-5


def test_ritz_kappa_default_keeps_the_published_free_plate_reference():
    """Backward compatibility: no foundation -> Leissa-anchored free-plate values."""
    result = _ritz(0.0, count=4)
    assert result.winkler_parameter == 0.0
    assert result.foundation_check == 0.0
    assert result.dimensionless[0] == pytest.approx(13.4682, rel=1e-4)
    assert result.dimensionless[1] == pytest.approx(19.5961, rel=1e-4)
    assert result.rigid_body_defect < 1e-10


@pytest.mark.parametrize("bad", [-1.0, True, float("nan"), float("inf")])
def test_ritz_rejects_invalid_foundation_parameter(bad):
    with pytest.raises((TypeError, ValueError)):
        _ritz(bad, count=2)


# -------------------------------------------------------------------- FE anchors
def test_translation_rayleigh_quotient_hits_the_exact_value():
    """Machine-precision anchor on the assembled foundation/mass blocks."""
    cases = (
        ((6, 6), 1.0, 1.0, 5.0e6),
        ((6, 6), 1.0, 1.0, 1.0e9),
        ((8, 4), 1.0, 0.5, 5.0e6),
        ((8, 4), 1.0, 0.5, 1.0e9),
    )
    for mesh, length, width, winkler_k in cases:
        _, _, foundation, mass = _assemble(mesh, length, width, winkler_k)
        vector = torch.ones(foundation.shape[0], dtype=DTYPE)
        measured = math.sqrt(
            float(vector @ (foundation @ vector)) / float(vector @ (mass @ vector))
        )
        exact = winkler_translation_angular_frequency(
            density=DENSITY, thickness=THICKNESS, winkler_k=winkler_k
        )
        assert measured / exact - 1.0 == pytest.approx(0.0, abs=1e-12)


def test_foundation_row_sums_equal_the_exact_tributary_areas():
    for mesh, length, width in (((6, 6), 1.0, 1.0), ((8, 4), 1.0, 0.5)):
        winkler_k = 3.0e6
        _, _, foundation, _ = _assemble(mesh, length, width, winkler_k)
        areas = foundation.sum(dim=1).numpy() / winkler_k
        side_x, side_y = length / mesh[0], width / mesh[1]
        expected = []
        for index in range(areas.size):
            node_y, node_x = divmod(index, mesh[0] + 1)  # x is the fast index
            share_x = 1.0 if 0 < node_x < mesh[0] else 0.5
            share_y = 1.0 if 0 < node_y < mesh[1] else 0.5
            expected.append(share_x * side_x * share_y * side_y)
        expected = np.array(expected)
        assert np.max(np.abs(areas - expected)) < 1e-12 * expected.max()
        assert areas.sum() == pytest.approx(length * width, rel=1e-13)


def test_translation_helper_validates_its_inputs():
    with pytest.raises((TypeError, ValueError)):
        winkler_translation_angular_frequency(density=DENSITY, thickness=THICKNESS, winkler_k=-1.0)
    with pytest.raises(TypeError):
        winkler_translation_angular_frequency(density=DENSITY, thickness=THICKNESS, winkler_k=True)
    assert winkler_translation_angular_frequency(
        density=DENSITY, thickness=THICKNESS, winkler_k=1.0e6
    ) == pytest.approx(math.sqrt(1.0e6 / AREAL), rel=1e-15)


# ------------------------------------------------------------------ FE behaviour
def test_free_foundation_lifts_every_mechanism():
    """``k_w = 0`` keeps the SRI hourglass (4 zero modes); ``k_w > 0`` lifts all 4."""
    bare = _free((8, 8), 0.0, modes=4)
    assert bare.rigid_body_modes == 4
    lifted = _free((8, 8), 1.0e6, modes=4)
    assert lifted.rigid_body_modes == 0
    lowest = float((lifted.angular_frequencies[0] ** 2))
    assert lowest > 0.9 * 1.0e6 / AREAL


def test_soft_modes_saturate_at_the_exact_translation_frequency():
    """All four lifted mechanisms sit at ``k_w/(rho h)`` (rotary inertia for two)."""
    for winkler_k in (1.0e5, 1.0e6, 1.0e8):
        shift = winkler_k / AREAL
        result = _free((8, 8), winkler_k, modes=6)
        ratios = (result.angular_frequencies[:SOFT_MODES] ** 2).numpy() / shift
        # two modes are translation-like (machine precision), two carry rotary inertia
        assert np.max(np.abs(ratios[2:] - 1.0)) < 1e-6
        assert np.max(np.abs(ratios - 1.0)) < 1e-3
        # below the exact value, never above: the rotary block only adds mass
        assert np.max(ratios) <= 1.0 + 1e-9


def test_rotary_inertia_deficit_of_the_lifted_modes_is_mesh_independent():
    """The 1e-4/4e-4 deficits are analytic (rotary inertia), not discretisation."""
    square = [
        (_free((mesh, mesh), 1.0e6, modes=4).angular_frequencies[:2] ** 2).numpy()
        / (1.0e6 / AREAL)
        for mesh in (6, 12)
    ]
    assert np.max(np.abs(square[0] - square[1])) < 1e-7
    rectangle = [
        _free((nx, ny), 1.0e6, modes=4, width=0.5, length=1.0).angular_frequencies[:2] ** 2
        for nx, ny in ((8, 4), (12, 6))
    ]
    rectangle = [row.numpy() / (1.0e6 / AREAL) for row in rectangle]
    assert np.max(np.abs(rectangle[0] - rectangle[1])) < 1e-7
    # the 2:1 plate has a larger rotation-to-translation ratio -> larger deficit
    assert np.all(rectangle[0] < square[0])


def test_bending_modes_match_the_shifted_rayleigh_ritz_reference():
    """12x12 free square, ``k_w = 1e6`` and ``1e8``: all three bending modes < 3 %."""
    reference = _ritz_angular_squared()
    for winkler_k in (1.0e6, 1.0e8):
        shift = winkler_k / AREAL
        result = _free((12, 12), winkler_k)
        measured = (result.angular_frequencies[SOFT_MODES : SOFT_MODES + 3] ** 2).numpy()
        error = (measured - (reference[:3] + shift)) / (reference[:3] + shift)
        assert np.max(np.abs(error)) < 0.03


def test_two_to_one_plate_matches_the_shifted_reference():
    reference = _ritz_angular_squared(width=0.5)
    shift = 1.0e6 / AREAL
    result = _free((16, 8), 1.0e6, width=0.5)
    measured = (result.angular_frequencies[SOFT_MODES : SOFT_MODES + 3] ** 2).numpy()
    error = (measured - (reference[:3] + shift)) / (reference[:3] + shift)
    assert np.max(np.abs(error)) < 0.03


def test_de_shifted_bending_error_is_independent_of_the_foundation():
    """The shift is additive, so subtracting it recovers the ``k_w = 0`` error."""
    reference = _ritz_angular_squared()
    errors = {}
    for winkler_k in (1.0e5, 1.0e6):
        shift = winkler_k / AREAL
        result = _free((12, 12), winkler_k)
        measured = (result.angular_frequencies[SOFT_MODES : SOFT_MODES + 3] ** 2).numpy()
        errors[winkler_k] = ((measured - shift) - reference[:3]) / reference[:3]
    assert np.max(np.abs(errors[1.0e5] - errors[1.0e6])) < 1e-3
    # and it agrees with the unfounded free plate at the same mesh
    bare = _free((12, 12), 0.0)
    bare_error = ((bare.angular_frequencies[:3] ** 2).numpy() - reference[:3]) / reference[:3]
    assert np.max(np.abs(errors[1.0e6] - bare_error)) < 1e-3


def test_coarse_mesh_third_mode_still_misses_the_gate():
    """Fail-closed record: 6x6 mode 3 is *not* within 3 % and must not be claimed."""
    reference = _ritz_angular_squared()
    result = _free((6, 6), 1.0e6)
    measured = (result.angular_frequencies[SOFT_MODES : SOFT_MODES + 3] ** 2).numpy()
    error = np.abs((measured - (reference[:3] + 1.0e6 / AREAL)) / (reference[:3] + 1.0e6 / AREAL))
    assert error[2] > 0.03
    # modes 1-2 are still inside the gate at the same coarse mesh
    assert error[0] < 0.03


def test_soft_foundation_needs_a_tighter_rigid_mode_tolerance():
    """Documented usability boundary of the free-boundary mechanism counter."""
    lookup = {"shift": 1.0e4 / AREAL}
    default = _free((8, 8), 1.0e4, modes=4)
    tight = _free((8, 8), 1.0e4, modes=4, rigid_body_relative_tolerance=1e-12)
    # default tolerance classifies the lifted foundation modes as mechanisms ...
    assert default.rigid_body_modes == 4
    # ... and the caller can recover them
    assert tight.rigid_body_modes == 0
    # the two translation-like modes of the lifted quartet hit k_w/(rho h) exactly
    assert float(tight.angular_frequencies[2] ** 2) == pytest.approx(lookup["shift"], rel=1e-8)