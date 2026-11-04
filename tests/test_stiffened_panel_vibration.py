"""Verification of ``tensorfem.stiffened_panel_vibration`` against independent references.

Two kinds of reference appear here, neither of them produced by this package's
solver output:

* the exact first-order shear deformation (Mindlin) **Navier** eigenvalue of the
  orthotropic plate, solved from the 3x3 generalised problem in the module
  docstring.  Its isotropic reduction is additionally checked against the
  classical ``omega = D R^2 / sqrt(rho h)`` limit, which is an independent
  constraint on the derivation itself;
* hand-computed stiffness algebra for the flat-bar smearing (``I_s + A e^2``
  about the plating mid-surface, per unit width).

The finite element is the Q4 Mindlin plate of :mod:`tensorfem.plate` with an
orthotropic :class:`tensorfem.plate.PlateRigidity`; the isotropic identity test
proves that adding that path changed nothing for existing models.
"""
import math

import pytest
import torch

from tensorfem.plate import PlateRigidity, q4_mindlin_stiffness, structured_square_mesh
from tensorfem.plate_vibration import kirchhoff_angular_frequency, plate_rigidity
from tensorfem.stiffened_panel_vibration import (
    StiffenerGeometry,
    fsdt_navier_angular_frequency,
    fsdt_navier_spectrum,
    identify_modes,
    panel_mode_table,
    smeared_panel_rigidity,
    smearing_wavelength_warning,
    solve_panel_vibration,
)

STEEL = dict(young=200.0e9, poisson=0.3, density=7850.0)
THICKNESS = 0.01
PANEL = dict(length=1.2, width=0.8, thickness=THICKNESS)
SHEAR_CORRECTION = 5.0 / 6.0


def _shear_modulus(young=STEEL["young"], poisson=STEEL["poisson"]):
    return young / (2.0 * (1.0 + poisson))


def _orthotropic_rigidity(ratio=6.0, thickness=THICKNESS):
    """Synthetic orthotropic rigidity with ``dx/dy = ratio`` and isotropic-like coupling."""
    rigid = plate_rigidity(STEEL["young"], STEEL["poisson"], thickness)
    shear = SHEAR_CORRECTION * _shear_modulus() * thickness
    return PlateRigidity(
        dx=ratio * rigid, dy=rigid, dxy=STEEL["poisson"] * math.sqrt(ratio) * rigid,
        dt=rigid * (1.0 - STEEL["poisson"]) / 2.0, sx=shear, sy=shear,
    )


@pytest.fixture(scope="module")
def orthotropic_panel():
    rigidity = _orthotropic_rigidity()
    result = solve_panel_vibration(
        16, 12, rigidity=rigidity, density=STEEL["density"], modes=4, **PANEL
    )
    rows = panel_mode_table(result, rigidity=rigidity, density=STEEL["density"], **PANEL)
    return rigidity, result, rows


def test_isotropic_rigidity_reproduces_the_builtin_element_exactly():
    nodes, elements = structured_square_mesh(2, 1.0)
    xy = nodes[elements[0]]
    isotropic = PlateRigidity.isotropic(STEEL["young"], STEEL["poisson"], THICKNESS)
    for treatment in ("sri", "mitc"):
        built_in = q4_mindlin_stiffness(
            xy, STEEL["young"], STEEL["poisson"], THICKNESS, shear_treatment=treatment
        )
        via_rigidity = q4_mindlin_stiffness(xy, shear_treatment=treatment, rigidity=isotropic)
        deviation = float((built_in - via_rigidity).abs().max() / built_in.abs().max())
        assert deviation == 0.0, (treatment, deviation)


def test_navier_twist_of_an_isotropic_rigidity_equals_the_flexural_rigidity():
    # Timoshenko's H must collapse to D for an isotropic plate, otherwise the
    # engineering-curvature convention of the element would be inconsistent.
    for thickness in (0.005, 0.01, 0.05):
        rigidity = PlateRigidity.isotropic(STEEL["young"], STEEL["poisson"], thickness)
        reference = plate_rigidity(STEEL["young"], STEEL["poisson"], thickness)
        assert abs(rigidity.navier_twist - reference) / reference < 1.0e-12


def test_fsdt_reference_reduces_to_kirchhoff_when_thin_and_softens_with_thickness():
    biases = []
    for thickness in (0.01, 0.02, 0.05, 0.1):
        rigidity = PlateRigidity.isotropic(STEEL["young"], STEEL["poisson"], thickness)
        fsdt = fsdt_navier_angular_frequency(
            rigidity=rigidity, density=STEEL["density"], thickness=thickness,
            length=1.0, width=1.0,
        )
        kirchhoff = kirchhoff_angular_frequency(
            length=1.0, width=1.0, young=STEEL["young"], poisson=STEEL["poisson"],
            thickness=thickness, density=STEEL["density"],
        )
        biases.append(fsdt / kirchhoff - 1.0)
    # a/h = 100 must be inside the 0.1% envelope used by plate_vibration, and the
    # shear bias must grow monotonically as the plate thickens.
    assert abs(biases[0]) < 1.0e-3, biases
    assert all(b < 0.0 for b in biases), biases
    assert biases == sorted(biases, reverse=True), biases
    assert -0.05 < biases[-1] < -0.02, biases


def test_rigidity_validation_is_fail_closed():
    with pytest.raises(ValueError):
        PlateRigidity(dx=-1.0, dy=1.0, dxy=0.0, dt=1.0, sx=1.0, sy=1.0)
    with pytest.raises(ValueError):
        PlateRigidity(dx=1.0, dy=1.0, dxy=1.0, dt=1.0, sx=1.0, sy=1.0)  # dxy^2 >= dx*dy
    with pytest.raises(TypeError):
        PlateRigidity(dx="1", dy=1.0, dxy=0.0, dt=1.0, sx=1.0, sy=1.0)
    with pytest.raises(ValueError):
        PlateRigidity.isotropic(STEEL["young"], 0.5, THICKNESS)


def test_smearing_matches_the_hand_computed_flat_bar_algebra():
    stiffener = StiffenerGeometry(depth=0.2, web_thickness=0.012, spacing=0.8)
    rigidity = smeared_panel_rigidity(
        young=STEEL["young"], poisson=STEEL["poisson"], thickness=THICKNESS,
        longitudinal=stiffener,
    )
    plate = plate_rigidity(STEEL["young"], STEEL["poisson"], THICKNESS)
    area = 0.2 * 0.012
    own = 0.012 * 0.2 ** 3 / 12.0
    eccentricity = 0.5 * (THICKNESS + 0.2)
    expected_dx = plate + STEEL["young"] * (own + area * eccentricity ** 2) / 0.8
    assert rigidity.dx == pytest.approx(expected_dx, rel=1.0e-12)
    assert rigidity.dy == pytest.approx(plate, rel=1.0e-12)
    assert rigidity.dxy == pytest.approx(
        STEEL["poisson"] * math.sqrt(expected_dx * plate), rel=1.0e-12)
    shear = _shear_modulus()
    assert rigidity.dt == pytest.approx(shear * THICKNESS ** 3 / 12.0, rel=1.0e-12)
    assert rigidity.sx == pytest.approx(SHEAR_CORRECTION * shear * THICKNESS, rel=1.0e-12)
    assert rigidity.sy == pytest.approx(rigidity.sx, rel=1.0e-12)


def test_smearing_without_stiffeners_reproduces_the_isotropic_rigidity():
    smeared = smeared_panel_rigidity(
        young=STEEL["young"], poisson=STEEL["poisson"], thickness=THICKNESS
    )
    isotropic = PlateRigidity.isotropic(STEEL["young"], STEEL["poisson"], THICKNESS)
    for name in ("dx", "dy", "dxy", "dt", "sx", "sy"):
        assert getattr(smeared, name) == pytest.approx(getattr(isotropic, name), rel=1.0e-12)


def test_stiffener_geometry_rejects_non_physical_input_and_reports_section_properties():
    for kwargs in (dict(depth=0.0, web_thickness=0.01, spacing=0.8),
                   dict(depth=0.2, web_thickness=-0.01, spacing=0.8),
                   dict(depth=0.2, web_thickness=0.01, spacing=0.0)):
        with pytest.raises(ValueError):
            StiffenerGeometry(**kwargs)
    stiffener = StiffenerGeometry(depth=0.2, web_thickness=0.01, spacing=0.8)
    assert stiffener.area == pytest.approx(2.0e-3)
    assert stiffener.second_moment == pytest.approx(0.01 * 0.2 ** 3 / 12.0)
    assert stiffener.torsion_constant == pytest.approx(0.2 * 0.01 ** 3 / 3.0)
    assert stiffener.eccentric_bending(THICKNESS) == pytest.approx(
        stiffener.second_moment + stiffener.area * (0.5 * (THICKNESS + 0.2)) ** 2)


def test_closed_form_spectrum_is_ordered_and_carries_its_labels():
    rigidity = _orthotropic_rigidity()
    spectrum = fsdt_navier_spectrum(
        rigidity=rigidity, density=STEEL["density"], thickness=THICKNESS,
        length=PANEL["length"], width=PANEL["width"],
        max_half_waves_x=2, max_half_waves_y=2,
    )
    frequencies = [mode.angular_frequency for mode in spectrum]
    assert frequencies == sorted(frequencies)
    assert {(mode.half_waves_x, mode.half_waves_y) for mode in spectrum} == {
        (1, 1), (1, 2), (2, 1), (2, 2)}
    assert spectrum[0].frequency_hz == pytest.approx(spectrum[0].angular_frequency / (2 * math.pi))


def test_orthotropic_panel_modes_are_below_three_percent_and_correctly_labelled(orthotropic_panel):
    _, _, rows = orthotropic_panel
    labels = {(row.half_waves_x, row.half_waves_y) for row in rows}
    assert labels == {(1, 1), (1, 2), (2, 1), (2, 2)}, labels
    assert min(row.overlap for row in rows) > 0.99
    errors = [row.relative_error for row in rows]
    assert max(errors) < 0.03, errors
    # The fundamental of a 6:1 orthotropic panel is inside 1% at 16x12; the
    # higher modes carry the discretisation error, not the reference.
    assert rows[0].relative_error < 0.01, errors


def test_fundamental_frequency_converges_from_above_at_second_order():
    rigidity = _orthotropic_rigidity()
    errors = []
    frequencies = []
    for nx, ny in ((8, 6), (12, 9), (16, 12)):
        result = solve_panel_vibration(
            nx, ny, rigidity=rigidity, density=STEEL["density"], modes=1, **PANEL)
        reference = fsdt_navier_angular_frequency(
            rigidity=rigidity, density=STEEL["density"], thickness=THICKNESS,
            length=PANEL["length"], width=PANEL["width"],
        ) / (2.0 * math.pi)
        frequency = float(result.frequencies_hz[0])
        frequencies.append(frequency)
        errors.append(abs(frequency - reference) / reference)
        assert frequency > reference, "a displacement formulation must not fall below the exact frequency"
    assert errors == sorted(errors, reverse=True), errors
    order = math.log(errors[0] / errors[-1]) / math.log(16.0 / 8.0)
    assert 1.5 < order < 2.5, order


def test_mode_identification_returns_unit_overlap_for_the_sine_basis():
    rigidity = _orthotropic_rigidity()
    result = solve_panel_vibration(
        12, 9, rigidity=rigidity, density=STEEL["density"], modes=2, **PANEL)
    labels = identify_modes(result, length=PANEL["length"], width=PANEL["width"])
    assert len(labels) == 2
    for m, n, overlap in labels:
        assert (m, n) == (1, 1) or (m, n) == (2, 1) or (m, n) == (1, 2)
        assert overlap > 0.98


def test_wavelength_warning_flags_modes_that_are_shorter_than_the_stiffener_spacing():
    rigidity = _orthotropic_rigidity()
    flagged = smearing_wavelength_warning(
        rigidity=rigidity, density=STEEL["density"], thickness=THICKNESS,
        length=3.2, width=2.4, stiffener_spacing=0.8,
        max_half_waves_x=2, max_half_waves_y=2,
    )
    flagged_labels = {(mode.half_waves_x, mode.half_waves_y) for mode in flagged}
    # half wavelengths for a 3.2 x 2.4 panel: (1,1) -> 2.4 m, (2,1) -> 1.6 m,
    # (1,2) -> 1.2 m, (2,2) -> 1.2 m; the threshold is 2 x 0.8 m = 1.6 m.
    assert flagged_labels == {(1, 2), (2, 2)}, flagged_labels
    loose = smearing_wavelength_warning(
        rigidity=rigidity, density=STEEL["density"], thickness=THICKNESS,
        length=3.2, width=2.4, stiffener_spacing=0.5,
        max_half_waves_x=2, max_half_waves_y=2,
    )
    assert loose == tuple(), [mode.half_waves_x for mode in loose]


def test_solve_panel_vibration_rejects_a_non_rigidity_argument():
    with pytest.raises(TypeError):
        solve_panel_vibration(
            4, 4, rigidity=1.0, density=STEEL["density"], modes=1, **PANEL)


def test_plate_rigidity_rejects_swapped_extremes():
    # A rigidity with dy = dx and dxy = sqrt(dx dy) is singular in bending; the
    # constructor must refuse it rather than return a mechanism frequency.
    rigid = plate_rigidity(STEEL["young"], STEEL["poisson"], THICKNESS)
    with pytest.raises(ValueError):
        PlateRigidity(dx=rigid, dy=rigid, dxy=rigid, dt=rigid, sx=rigid, sy=rigid)
    with torch.no_grad():
        assert torch.isfinite(torch.tensor(rigid))