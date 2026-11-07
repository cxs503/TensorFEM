"""Tests for the hull-girder wet/dry modal benchmark (module + benchmark scripts).

The tests are deliberately cheap (small meshes, capped sample counts) but each
one asserts a physical property rather than a stored number:

* the section properties are verified by an independent Green's-theorem path
  and by the thin-wall asymptote;
* the Lewis added mass reproduces the three exact limits of the family;
* the first-order Timoshenko correction stays within 2 % of the exact
  Hu--Timoshenko root over the validity range and raises outside it;
* the finite element model converges at second order and reproduces the
  wet/dry reduction ``1/sqrt(1+kappa)``;
* the consistent added-mass matrix conserves the rigid-translation kinetic
  energy, which is the partition-of-unity property the wet modal assembly
  relies on.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorfem.hull_girder_modal import (
    BoxGirderSection,
    added_mass_matrix,
    ballast_mass_per_length,
    beam_simply_supported_modes,
    exact_hull_girder_angular_frequencies,
    floating_mass_per_length,
    gci,
    lewis_fit,
    lewis_heave_added_mass,
    lewis_heave_coefficient,
    timoshenko_corrected_angular_frequencies,
    variable_added_mass_modes,
    wet_dry_frequency_ratio,
    wet_simply_supported_modes,
)
from tensorfem.timoshenko_beam_modal import (
    euler_bernoulli_simply_supported_angular_frequencies,
    simply_supported_timoshenko_modes,
)

REPO = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO / "benchmarks" / "hull_girder_modal"

GATE_GIRDER = dict(length=200.0, breadth=16.0, depth=4.0, wall=0.020, draft=2.4)
SHIP_GIRDER = dict(length=180.0, breadth=24.0, depth=14.0, wall=0.022, draft=8.4)


def girder(**overrides) -> BoxGirderSection:
    spec = dict(GATE_GIRDER)
    spec.update(overrides)
    return BoxGirderSection(**spec)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_geometry_verified_by_independent_path():
    section = girder()
    audit = section.geometry_audit()
    # Green's theorem on the outer/inner polygons: same numbers, other formula.
    assert audit["area_rel_error"] < 1e-12
    assert audit["inertia_rel_error"] < 1e-12
    # Thin-wall asymptote: an O(t/D) approximation, so a per-mille agreement.
    assert audit["area_thin_wall_rel_dev"] < 5e-3
    # the asymptote residual carries the O(t/D) coefficient ~2.3
    assert audit["inertia_thin_wall_rel_dev"] < 2e-2
    assert 0.0 < section.block_coefficient < 1.0


def test_equivalent_section_reproduces_the_four_beam_products():
    section = girder()
    beam = section.to_timoshenko_section()
    assert beam.young * beam.inertia == pytest.approx(section.bending_stiffness, rel=1e-12)
    assert beam.rho_area == pytest.approx(section.mass_per_length, rel=1e-12)
    assert beam.density * beam.inertia == pytest.approx(
        section.density * section.inertia, rel=1e-12
    )
    assert beam.shear_coefficient * beam.shear_modulus * beam.area == pytest.approx(
        section.shear_stiffness, rel=1e-12
    )


def test_invalid_section_is_rejected():
    with pytest.raises(ValueError):
        girder(wall=0.0).validate()
    with pytest.raises(ValueError):
        girder(wall=2.5).validate()  # thicker than half the depth
    with pytest.raises(ValueError):
        girder(draft=100.0).validate()


# ---------------------------------------------------------------------------
# Lewis section added mass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "half_b, half_h, exact",
    [
        (1.0, 1.0, math.pi),  # circle of radius 1
        (2.5, 2.5, math.pi * 2.5**2),  # circle of radius 2.5
        (2.0, 1.0, math.pi * 2.0**2),  # ellipse: vertical motion -> rho pi b^2
        (5.0, 0.05, math.pi * 5.0**2),  # flat plate limit
    ],
)
def test_lewis_reproduces_exact_limits(half_b, half_h, exact):
    section = lewis_fit(half_b, half_h, area=math.pi * half_b * half_h)
    assert section.a3 == pytest.approx(0.0, abs=1e-8)  # area target selects the ellipse
    assert section.heave_added_mass(1.0) == pytest.approx(exact, rel=1e-9)


def test_lewis_fit_stays_inside_its_bounding_box():
    for half_b, half_h in ((1.0, 1.0), (8.0, 2.4), (1.0, 0.5), (5.0, 0.05)):
        section = lewis_fit(half_b, half_h)
        x_max = max(abs(section.contour(2.0 * math.pi * i / 720)[0]) for i in range(720))
        y_max = max(abs(section.contour(2.0 * math.pi * i / 720)[1]) for i in range(720))
        assert x_max <= half_b * (1.0 + 1e-6)
        assert y_max <= half_h * (1.0 + 1e-6)
        assert section.half_breadth == pytest.approx(half_b)
        assert section.half_draft == pytest.approx(half_h)


def test_boxier_section_has_more_added_mass_and_area():
    ellipse = lewis_fit(2.0, 1.0, area=math.pi * 2.0)
    boxy = lewis_fit(2.0, 1.0)
    assert boxy.a3 < 0.0
    assert boxy.area(256) > ellipse.area(256)
    assert boxy.heave_added_mass(1.0) > ellipse.heave_added_mass(1.0)


def test_area_target_beyond_the_family_is_reported_not_clipped():
    # The doubled box area 2*B*T cannot be reached by a smooth Lewis form.
    section = lewis_fit(8.0, 2.4, area=2.0 * 16.0 * 2.4)
    assert 0.85 < section.area_ratio < 1.0
    assert section.area_ratio == pytest.approx(
        section.area(128) / (2.0 * 16.0 * 2.4), rel=1e-12
    )


def test_added_mass_scales_with_breadth_squared_and_density():
    base = lewis_heave_added_mass(4.0, 1.6, 1000.0)[0]
    wider = lewis_heave_added_mass(8.0, 1.6, 1000.0)[0]
    heavier = lewis_heave_added_mass(4.0, 1.6, 2000.0)[0]
    assert heavier == pytest.approx(2.0 * base, rel=1e-12)
    assert wider > 3.5 * base  # same draft, double breadth -> roughly 4x


def test_coefficient_table_matches_direct_evaluation():
    # The tabulated coefficient is linearly interpolated in log(h/b).  In the
    # hull-like range (h/b < 0.7) the interpolation error is below 1e-4; the
    # boxiness saturation at h/b = 1 makes the curve kinked there, so the
    # tolerance is looser at that single point (documented in the docstring).
    for ratio in (0.05, 0.2, 0.3, 0.4, 0.6):
        assert lewis_heave_coefficient(ratio) == pytest.approx(
            lewis_heave_coefficient(ratio, direct=True), rel=1e-4
        )
    for ratio in (1.0, 2.0):
        assert lewis_heave_coefficient(ratio) == pytest.approx(
            lewis_heave_coefficient(ratio, direct=True), rel=5e-3
        )
    assert lewis_heave_coefficient(0.05, direct=True) > math.pi


# ---------------------------------------------------------------------------
# Timoshenko correction
# ---------------------------------------------------------------------------
def test_first_order_correction_within_two_percent_on_the_gate_girder():
    section = girder()
    first = timoshenko_corrected_angular_frequencies(section, 4)
    exact = exact_hull_girder_angular_frequencies(section, 4)
    relative = ((first - exact).abs() / exact).max()
    assert float(relative) * 100.0 <= 2.0


def test_shear_correction_lowers_the_euler_bernoulli_frequency():
    section = girder()
    beam = section.to_timoshenko_section()
    eb = euler_bernoulli_simply_supported_angular_frequencies(beam, 4)
    exact = exact_hull_girder_angular_frequencies(section, 4)
    assert torch.all(exact < eb)
    assert float(exact[0] / eb[0]) < 0.999  # the correction is not negligible


def test_first_order_correction_raises_outside_its_validity_range():
    section = BoxGirderSection(**SHIP_GIRDER)
    with pytest.raises(ValueError):
        timoshenko_corrected_angular_frequencies(section, 4)


def test_finite_element_frequencies_converge_at_second_order():
    section = girder()
    beam = section.to_timoshenko_section()
    exact = exact_hull_girder_angular_frequencies(section, 4) / (2.0 * math.pi)
    series = []
    for elements in (20, 40, 80):
        modes = simply_supported_timoshenko_modes(beam, elements, 4)
        series.append([float(v) for v in modes.frequencies_hz])
    for index in range(4):
        fine, medium, coarse = series[2][index], series[1][index], series[0][index]
        study = gci(fine, medium, coarse, refinement=2.0)
        assert study["observed_order"] >= 1.9
        assert abs(coarse - float(exact[index])) / float(exact[index]) * 100.0 < 2.0


def test_gci_rejects_a_non_asymptotic_history():
    with pytest.raises(ValueError):
        gci(1.0, 2.0, 1.5)  # oscillatory
    with pytest.raises(ValueError):
        gci(1.0, 1.0, 1.0)  # degenerate


# ---------------------------------------------------------------------------
# wet modes
# ---------------------------------------------------------------------------
def test_wet_frequency_is_below_dry_and_ratio_matches_closed_form():
    section = girder()
    added = 5.0 * floating_mass_per_length(section)
    ballast = ballast_mass_per_length(section)
    kappa = added / floating_mass_per_length(section)
    dry = beam_simply_supported_modes(section, 40, (ballast,), 4)
    wet = wet_simply_supported_modes(section, 40, added, ballast, 4)
    closed = wet_dry_frequency_ratio(kappa)
    for dry_mode, wet_mode in zip(dry.frequencies_hz, wet.frequencies_hz):
        assert float(wet_mode) < float(dry_mode)
        assert float(wet_mode / dry_mode) == pytest.approx(closed, rel=1e-3)


def test_added_mass_matrix_conserves_rigid_translation_energy():
    section = girder()
    added = 12345.0
    matrix = added_mass_matrix(section, 20, added)
    nodes = 21
    rigid = torch.zeros(2 * nodes, dtype=torch.float64)
    rigid[0::2] = 1.0
    assert float(rigid @ matrix @ rigid) == pytest.approx(added * section.length, rel=1e-12)
    # symmetric, positive semi-definite
    assert torch.allclose(matrix, matrix.T)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert float(eigenvalues.min()) > -1e-9 * float(eigenvalues.max())


def test_uniform_variable_added_mass_reduces_to_the_uniform_case():
    section = girder()
    added = 100000.0
    ballast = ballast_mass_per_length(section)
    uniform = wet_simply_supported_modes(section, 30, added, ballast, 2)
    variable = variable_added_mass_modes(section, 30, lambda x: added, ballast, 2)
    assert torch.allclose(
        uniform.frequencies_hz, variable.frequencies_hz, rtol=1e-10, atol=1e-12
    )


def test_tapered_added_mass_changes_the_ratio():
    section = girder()
    length = section.length
    ballast = ballast_mass_per_length(section)
    uniform_added = 5.0 * floating_mass_per_length(section)

    def taper(x: float) -> float:
        eta = abs(2.0 * x / length - 1.0)
        return uniform_added * (1.0 - 0.7 * eta**3)

    dry = beam_simply_supported_modes(section, 40, (ballast,), 1)
    wet_uniform = wet_simply_supported_modes(section, 40, uniform_added, ballast, 1)
    wet_taper = variable_added_mass_modes(section, 40, taper, ballast, 1)
    ratio_uniform = float(wet_uniform.frequencies_hz[0] / dry.frequencies_hz[0])
    ratio_taper = float(wet_taper.frequencies_hz[0] / dry.frequencies_hz[0])
    # removing added mass at the ends (where mode 1 has most displacement)
    # raises the wet frequency
    assert ratio_taper > ratio_uniform


def test_wet_dry_ratio_is_one_without_added_mass():
    assert wet_dry_frequency_ratio(0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wet_dry_frequency_ratio(-1.0)


# ---------------------------------------------------------------------------
# benchmark artefacts
# ---------------------------------------------------------------------------
def test_benchmark_scripts_import_and_expose_compute():
    sys.path.insert(0, str(BENCHMARK_DIR))
    try:
        import timoshenko_comparison as tc
        import wet_dry_ratio as wd
    finally:
        sys.path.pop(0)
    dry = tc.analyse_case(tc.GIRDERS["girder_g"])
    assert len(dry["modes"]) == 4
    assert dry["geometry_audit"]["area_rel_error"] < 1e-12
    assert dry["grid_convergence"][0]["observed_order"] >= 1.9
    section = wd.build(wd.GATE_GIRDER)
    added = wd.hull_added_mass(section)
    assert added["added_mass_per_length_kg_m"] > 0.0
    assert added["lewis_area_ratio_vs_doubled_box"] < 1.0
    for row in wd.lewis_exact_checks():
        assert row["rel_error"] < 1e-9


def test_committed_result_json_matches_the_recomputed_gates():
    path = BENCHMARK_DIR / "result.json"
    assert path.exists(), "result.json must be committed with the benchmark"
    payload = json.loads(path.read_text())
    gates = payload["gates"]
    assert gates["all_pass"] is True
    assert gates["gate1_timoshenko_modes"]["first_order_max_rel_error_pct"] <= 2.0
    assert gates["gate2_added_mass_scaling"]["uniform_ratio_max_rel_error_pct"] <= 5.0
    assert gates["gate3_wet_dry_ratio"]["max_rel_error_pct"] <= 3.0
    assert gates["gate4_grid_convergence"]["min_observed_order"] >= 2.0
    # the committed kappa must equal the freshly computed one
    section = girder()
    fresh = lewis_heave_added_mass(
        section.half_breadth,
        section.section_draft,
        section.water_density,
        area=2.0 * section.breadth * section.section_draft,
    )[0]
    assert gates["gate3_wet_dry_ratio"]["kappa_33"] == pytest.approx(
        fresh / floating_mass_per_length(section), rel=1e-9
    )


def test_benchmark_scripts_run_as_modules():
    for script in ("timoshenko_comparison.py", "wet_dry_ratio.py"):
        completed = subprocess.run(
            [sys.executable, str(BENCHMARK_DIR / script)],
            capture_output=True,
            text=True,
            timeout=900,
            env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        )
        assert completed.returncode == 0, completed.stderr
        assert "mode" in completed.stdout.lower() or "lewis" in completed.stdout.lower()