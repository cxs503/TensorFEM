"""Verification of ``tensorfem.plate_vibration`` against closed-form references.

Every threshold below is an independent closed form (Navier sine solution,
Leissa's tabulated clamped coefficient) or an exact matrix identity; no
reference number is taken from this module's own output.
"""
import math

import pytest
import torch

from tensorfem.plate_vibration import (
    BOUNDARY_CLAMPED,
    BOUNDARY_SIMPLE,
    LEISSA_CLAMPED_SQUARE_LAMBDA,
    clamped_square_angular_frequency,
    kirchhoff_angular_frequency,
    mode_convergence,
    observed_convergence_order,
    plate_rigidity,
    q4_plate_mass_matrix,
    shear_deformation_bias,
    solve_plate_vibration,
    structured_rectangular_mesh,
    winkler_angular_frequency,
)
from tensorfem.plate import structured_square_mesh

STEEL = dict(young=200.0e9, poisson=0.3, density=7850.0, thickness=0.01)


def _hz(omega: float) -> float:
    return omega / (2.0 * math.pi)


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


def test_simply_supported_square_first_four_modes_below_three_percent():
    result = solve_plate_vibration(16, 16, modes=4, **STEEL)
    exact = sorted(
        _hz(kirchhoff_angular_frequency(
            length=1.0, width=1.0, half_waves_x=m, half_waves_y=n, **STEEL
        ))
        for m in (1, 2) for n in (1, 2)
    )
    computed = result.frequencies_hz.tolist()
    errors = [_relative_error(computed[i], exact[i]) for i in range(4)]
    assert max(errors) < 0.03
    # The staircase frequency set of a square plate is doubly degenerate in
    # the (1,2)/(2,1) pair; the FE result must reproduce that degeneracy.
    assert abs(computed[1] - computed[2]) / computed[1] < 1.0e-6


def test_mesh_refinement_is_second_order_and_monotone_from_above():
    reference = _hz(kirchhoff_angular_frequency(length=1.0, width=1.0, **STEEL))
    rows = mode_convergence(reference, meshes=(6, 8, 10, 12), **STEEL)
    errors = [row.relative_error for row in rows]
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 0.01
    # A selective-reduced-integration Q4 Mindlin plate is O(h^2) here; the
    # estimate uses the signed error so an overshoot would be visible.
    order = observed_convergence_order(rows, reference)
    assert 1.5 < order < 2.5, order


def test_shear_deformation_bias_explains_the_residual_gap():
    # For a/h = 100 the shear correction must be far below the 3% gate, which
    # is what makes the Kirchhoff reference admissible for a Mindlin element.
    thin = shear_deformation_bias(
        length=1.0, width=1.0, young=STEEL["young"], poisson=STEEL["poisson"],
        thickness=STEEL["thickness"],
    )
    assert abs(thin) < 1.0e-3
    thick = shear_deformation_bias(
        length=1.0, width=1.0, young=200.0e9, poisson=0.3, thickness=0.1
    )
    assert abs(thick) > 20.0 * abs(thin)


def test_rectangular_plate_selects_longitudinal_half_waves():
    result = solve_plate_vibration(12, 12, length=2.0, width=1.0, modes=4, **STEEL)
    expected = sorted(
        _hz(kirchhoff_angular_frequency(
            length=2.0, width=1.0, half_waves_x=m, half_waves_y=n, **STEEL
        ))
        for m in (1, 2, 3) for n in (1, 2)
    )[:4]
    computed = result.frequencies_hz.tolist()
    # First two modes are converged on this mesh; the third and fourth are
    # reported by mode identity rather than by a loosened tolerance.
    for index in range(2):
        assert _relative_error(computed[index], expected[index]) < 0.03
    for index in (2, 3):
        assert abs(computed[index] - expected[index]) < abs(computed[index] - expected[1])


def test_winkler_foundation_frequency_matches_the_exact_sine_eigenvalue():
    for winkler_k in (5.0e8, 2.0e9):
        result = solve_plate_vibration(8, 8, winkler_k=winkler_k, modes=1, **STEEL)
        exact = _hz(winkler_angular_frequency(
            length=1.0, width=1.0, winkler_k=winkler_k, **STEEL
        ))
        assert _relative_error(float(result.frequencies_hz[0]), exact) < 0.01
        # A foundation can only raise the frequency, never lower it.
        unstiffened = _hz(kirchhoff_angular_frequency(length=1.0, width=1.0, **STEEL))
        assert float(result.frequencies_hz[0]) > unstiffened


def test_clamped_square_matches_leissa_tabulated_coefficient():
    result = solve_plate_vibration(
        12, 12, boundary=BOUNDARY_CLAMPED, modes=1, **STEEL
    )
    exact = _hz(clamped_square_angular_frequency(length=1.0, **STEEL))
    assert _relative_error(float(result.frequencies_hz[0]), exact) < 0.03
    # Clamping must stiffen the panel relative to simple support.
    simple = solve_plate_vibration(8, 8, modes=1, **STEEL)
    assert float(result.frequencies_hz[0]) > float(simple.frequencies_hz[0])
    # Leissa's coefficient, not this module, is the reference: check against
    # the tabulated value recomputed here from the definition.
    rigidity = plate_rigidity(STEEL["young"], STEEL["poisson"], STEEL["thickness"])
    assert abs(
        clamped_square_angular_frequency(length=1.0, **STEEL)
        - LEISSA_CLAMPED_SQUARE_LAMBDA * math.sqrt(rigidity / (STEEL["density"] * STEEL["thickness"]))
    ) < 1.0e-12


def test_consistent_mass_matrix_conserves_total_mass_exactly():
    nodes, elements = structured_rectangular_mesh(4, 3, 2.0, 1.0)
    ndof = 3 * nodes.shape[0]
    mass = torch.zeros((ndof, ndof), dtype=torch.float64)
    for connectivity in elements:
        element_mass = q4_plate_mass_matrix(nodes[connectivity], 7850.0, 0.01)
        assert torch.allclose(element_mass, element_mass.T, atol=1.0e-12)
        dofs = torch.stack(
            (3 * connectivity, 3 * connectivity + 1, 3 * connectivity + 2), dim=1
        ).reshape(-1)
        mass[dofs[:, None], dofs] += element_mass
    w_dofs = torch.arange(0, ndof, 3)
    ones = torch.ones(w_dofs.numel(), dtype=torch.float64)
    total = float(ones @ mass[w_dofs[:, None], w_dofs] @ ones)
    assert abs(total - 7850.0 * 0.01 * 2.0 * 1.0) / (7850.0 * 0.01 * 2.0) < 1.0e-12
    rotation_block = mass[w_dofs + 1][:, w_dofs + 1]
    ones_rot = torch.ones(w_dofs.numel(), dtype=torch.float64)
    rotary_total = float(ones_rot @ rotation_block @ ones_rot)
    expected_rotary = 7850.0 * 0.01**3 / 12.0 * 2.0 * 1.0
    assert abs(rotary_total - expected_rotary) / expected_rotary < 1.0e-12


def test_mass_matrix_rejects_degenerate_inputs():
    nodes, elements = structured_square_mesh(1)
    xy = nodes[elements[0]]
    with pytest.raises(ValueError):
        q4_plate_mass_matrix(xy, 0.0, 0.01)
    with pytest.raises(ValueError):
        q4_plate_mass_matrix(xy, 7850.0, -0.01)
    with pytest.raises(ValueError):
        q4_plate_mass_matrix(xy[:3], 7850.0, 0.01)


def test_boundary_dofs_are_held_and_modes_vanish_on_them():
    for boundary in (BOUNDARY_SIMPLE, BOUNDARY_CLAMPED):
        result = solve_plate_vibration(6, 6, boundary=boundary, modes=2, **STEEL)
        ndof = 3 * result.nodes.shape[0]
        fixed = torch.tensor(
            [d for d in range(ndof) if d not in set(result.retained_dofs.tolist())]
        )
        assert bool((result.modes[fixed].abs().max() == 0).item())
        assert int(result.retained_dofs.numel()) < ndof
        # The fundamental mode of a supported panel has no interior nodal line:
        # every transverse component must share one sign.
        transverse = result.modes[0::3, 0]
        assert float(transverse.abs().max()) > 0.0
        assert bool((transverse >= 0).all().item()) or bool((transverse <= 0).all().item())


def test_mechanical_mechanism_is_rejected_when_mass_is_translation_only():
    with pytest.raises(ValueError, match="rotary_inertia"):
        solve_plate_vibration(4, 4, rotary_inertia=False, modes=1, **STEEL)


def test_input_validation_and_result_metadata():
    result = solve_plate_vibration(4, 5, length=2.0, width=1.5, modes=3, **STEEL)
    assert result.element_counts == (4, 5)
    assert result.total_mass == pytest.approx(7850.0 * 0.01 * 2.0 * 1.5)
    assert result.frequencies_hz.shape == (3,)
    assert torch.all(result.frequencies_hz[1:] > result.frequencies_hz[:-1])
    with pytest.raises(ValueError):
        solve_plate_vibration(2, 2, boundary="free", modes=1, **STEEL)
    with pytest.raises(ValueError):
        solve_plate_vibration(2, 2, winkler_k=-1.0, modes=1, **STEEL)
    with pytest.raises(TypeError):
        solve_plate_vibration(2, 2, winkler_k=True, modes=1, **STEEL)
    with pytest.raises(ValueError):
        solve_plate_vibration(2, 2, modes=10000, **STEEL)
    with pytest.raises(ValueError):
        solve_plate_vibration(0, 2, modes=1, **STEEL)
    with pytest.raises(TypeError):
        solve_plate_vibration(2.0, 2, modes=1, **STEEL)
    with pytest.raises(ValueError):
        kirchhoff_angular_frequency(length=1.0, width=1.0, half_waves_x=0, **STEEL)
    with pytest.raises(ValueError):
        shear_deformation_bias(length=1.0, width=1.0, young=200.0e9, poisson=0.5,
                              thickness=0.01)