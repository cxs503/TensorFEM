"""MITC4 shear field as an option of ``solve_plate_vibration``.

The historical SRI rule leaves one spurious ``w`` hourglass mode on a free
plate; the MITC4 assumed-shear-strain rule is rank sufficient.  These tests pin
both facts and, more importantly, pin that the new option does not disturb the
already-qualified supported-boundary cases.

Evidence bundle: ``benchmarks/free_plate_mitc_vibration_evidence.json``,
report: ``docs/free-plate-mitc-vibration-validation.md``.
"""
import math

import pytest
import torch

from tensorfem.plate import q4_mindlin_stiffness, structured_square_mesh
from tensorfem.plate_ritz_reference import ritz_free_plate_eigenvalues
from tensorfem.plate_vibration import (
    LEISSA_CLAMPED_SQUARE_LAMBDA,
    SHEAR_MITC,
    SHEAR_SRI,
    solve_plate_vibration,
    winkler_translation_angular_frequency,
)

YOUNG = 200.0e9
POISSON = 0.30
THICKNESS = 0.01
DENSITY = 7850.0
LENGTH = 1.0
RIGIDITY = YOUNG * THICKNESS**3 / (12.0 * (1.0 - POISSON**2))


def _dimensionless(frequency_hz: float) -> float:
    return (
        float(frequency_hz)
        * 2.0
        * math.pi
        * LENGTH**2
        * math.sqrt(DENSITY * THICKNESS / RIGIDITY)
    )


def _free_stiffness(n: int, shear_treatment: str) -> torch.Tensor:
    nodes, elements = structured_square_mesh(n, LENGTH)
    ndof = 3 * nodes.shape[0]
    stiffness = torch.zeros((ndof, ndof), dtype=torch.float64)
    for connectivity in elements:
        element = q4_mindlin_stiffness(
            nodes[connectivity], YOUNG, POISSON, THICKNESS,
            shear_treatment=shear_treatment,
        )
        dofs = torch.stack(
            (3 * connectivity, 3 * connectivity + 1, 3 * connectivity + 2), dim=1
        ).reshape(-1)
        stiffness[dofs[:, None], dofs] += element
    return 0.5 * (stiffness + stiffness.T)


def _rigid_motions(n: int) -> torch.Tensor:
    nodes, _ = structured_square_mesh(n, LENGTH)
    count = nodes.shape[0]
    zeros, ones = torch.zeros(count, dtype=torch.float64), torch.ones(
        count, dtype=torch.float64
    )
    return torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=1),
            torch.stack((nodes[:, 0], -ones, zeros), dim=1),
            torch.stack((-nodes[:, 1], zeros, ones), dim=1),
        ),
        dim=2,
    ).reshape(-1, 3)


def _zero_energy_count(stiffness: torch.Tensor, relative_tolerance: float = 1e-9) -> int:
    eigenvalues = torch.linalg.eigvalsh(stiffness)
    scale = float(eigenvalues.abs().max())
    return int((eigenvalues.abs() <= relative_tolerance * scale).sum().item())


# ------------------------------------------------------------ contract / default
def test_default_shear_treatment_is_the_historical_sri_path():
    default = solve_plate_vibration(6, boundary="free")
    explicit = solve_plate_vibration(6, boundary="free", shear_treatment=SHEAR_SRI)
    assert default.shear_treatment == SHEAR_SRI == "sri"
    assert torch.equal(default.frequencies_hz, explicit.frequencies_hz)
    assert torch.equal(default.modes, explicit.modes)


def test_invalid_shear_treatment_is_rejected():
    with pytest.raises(ValueError, match="shear_treatment"):
        solve_plate_vibration(4, shear_treatment="assumed")
    with pytest.raises(ValueError, match="shear_treatment"):
        solve_plate_vibration(4, shear_treatment="")


def test_result_reports_the_rule_that_was_used():
    result = solve_plate_vibration(4, boundary="free", shear_treatment=SHEAR_MITC)
    assert result.shear_treatment == "mitc"


# ------------------------------------------------- stiffness null space (rank)
@pytest.mark.parametrize("n", [4, 6, 8])
def test_free_plate_stiffness_null_space_depends_on_the_shear_rule(n):
    sri = _free_stiffness(n, SHEAR_SRI)
    mitc = _free_stiffness(n, SHEAR_MITC)
    assert _zero_energy_count(sri) == 4   # 3 rigid + 1 w hourglass
    assert _zero_energy_count(mitc) == 3  # rigid only


def test_rigid_motions_are_zero_energy_under_both_rules():
    for treatment in (SHEAR_SRI, SHEAR_MITC):
        stiffness = _free_stiffness(6, treatment)
        scale = float(torch.linalg.matrix_norm(stiffness, ord=2))
        for motion in _rigid_motions(6).T:
            residual = float(torch.linalg.vector_norm(stiffness @ motion))
            assert residual / (scale * float(torch.linalg.vector_norm(motion))) < 1e-12


def test_the_extra_sri_mode_is_a_spurious_mechanism_not_a_rigid_motion():
    """The 4th SRI zero-energy field carries energy under MITC4 => not rigid."""
    n = 6
    sri = _free_stiffness(n, SHEAR_SRI)
    mitc = _free_stiffness(n, SHEAR_MITC)
    eigenvalues, eigenvectors = torch.linalg.eigh(sri)
    scale = float(eigenvalues.abs().max())
    zero_vectors = eigenvectors[:, eigenvalues.abs() <= 1e-9 * scale]
    assert zero_vectors.shape[1] == 4
    rigid = _rigid_motions(n)
    candidate = zero_vectors[:, 3]
    coefficients = torch.linalg.lstsq(rigid, candidate).solution
    field = candidate - rigid @ coefficients
    field = field / torch.linalg.vector_norm(field)
    energy_sri = float(0.5 * field @ (sri @ field))
    energy_mitc = float(0.5 * field @ (mitc @ field))
    assert abs(energy_sri) / energy_mitc < 1e-12  # invisible to the one-point rule
    # and it is a transverse-displacement pattern: rotations vanish
    assert float(torch.linalg.vector_norm(field.reshape(-1, 3)[:, 1:])) < 1e-8


# ------------------------------------------------------- free-boundary spectrum
@pytest.mark.parametrize("n", [6, 8, 12])
def test_free_plate_mitc_keeps_exactly_three_rigid_body_modes(n):
    result = solve_plate_vibration(
        n, length=LENGTH, young=YOUNG, poisson=POISSON, thickness=THICKNESS,
        density=DENSITY, boundary="free", modes=3, shear_treatment=SHEAR_MITC,
    )
    assert result.rigid_body_modes == 3


def test_free_plate_sri_still_reports_the_documented_hourglass():
    result = solve_plate_vibration(
        8, length=LENGTH, young=YOUNG, poisson=POISSON, thickness=THICKNESS,
        density=DENSITY, boundary="free", modes=3, shear_treatment=SHEAR_SRI,
    )
    assert result.rigid_body_modes == 4


def test_free_plate_mitc_frequencies_match_the_rayleigh_ritz_reference():
    reference = ritz_free_plate_eigenvalues(
        length=LENGTH, poisson=POISSON, order=18, count=3
    )
    result = solve_plate_vibration(
        12, length=LENGTH, young=YOUNG, poisson=POISSON, thickness=THICKNESS,
        density=DENSITY, boundary="free", modes=3, shear_treatment=SHEAR_MITC,
    )
    for measured, expected in zip(result.frequencies_hz, reference.dimensionless):
        assert _dimensionless(measured) / float(expected) - 1.0 == pytest.approx(
            0.0, abs=0.03
        )


def test_free_plate_mitc_reproduces_the_exact_foundation_translation_mode():
    """The pure translation mode stays an exact eigenpair of the MITC4 system.

    A uniform Winkler foundation makes ``w = 1, theta = 0`` an exact eigenvector
    with ``omega^2 = k_w / (rho h)`` for *any* boundary.  The two lifted
    *rotation* rigid modes are **not** exact at that value: they carry rotary
    inertia, so they sit ``~(h^2/12) * (int1 / int x^2) / 2 = 5e-5`` below it.
    That is measured here rather than hidden, which is why the translation mode
    is identified by matching, not by index.
    """
    winkler_k = 1.0e6
    result = solve_plate_vibration(
        8, length=LENGTH, young=YOUNG, poisson=POISSON, thickness=THICKNESS,
        density=DENSITY, winkler_k=winkler_k, boundary="free", modes=3,
        shear_treatment=SHEAR_MITC,
    )
    exact = winkler_translation_angular_frequency(
        density=DENSITY, thickness=THICKNESS, winkler_k=winkler_k
    )
    deviations = [float(a) / exact - 1.0 for a in result.angular_frequencies]
    assert min(abs(d) for d in deviations) < 1e-7
    rotary_bias = -0.5 * (THICKNESS**2 / 12.0) * 12.0
    assert sorted(deviations)[:2] == pytest.approx([rotary_bias, rotary_bias], abs=1e-6)


# ------------------------------------------- supported boundaries are unchanged
@pytest.mark.parametrize("boundary", ["simple", "clamped"])
def test_supported_boundaries_are_indifferent_to_the_shear_rule(boundary):
    common = dict(
        length=LENGTH, young=YOUNG, poisson=POISSON, thickness=THICKNESS,
        density=DENSITY, boundary=boundary, modes=3,
    )
    sri = solve_plate_vibration(8, shear_treatment=SHEAR_SRI, **common)
    mitc = solve_plate_vibration(8, shear_treatment=SHEAR_MITC, **common)
    assert sri.rigid_body_modes == mitc.rigid_body_modes == 0
    for measured, reference in zip(mitc.frequencies_hz, sri.frequencies_hz):
        assert float(measured) / float(reference) - 1.0 == pytest.approx(0.0, abs=1e-3)


def test_clamped_mitc_still_matches_leissa():
    result = solve_plate_vibration(
        12, length=LENGTH, young=YOUNG, poisson=POISSON, thickness=THICKNESS,
        density=DENSITY, boundary="clamped", modes=1, shear_treatment=SHEAR_MITC,
    )
    assert _dimensionless(result.frequencies_hz[0]) / LEISSA_CLAMPED_SQUARE_LAMBDA - 1.0 == pytest.approx(
        0.0, abs=0.03
    )