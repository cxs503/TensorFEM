import pytest

from tensorfem.hertz_3d_qualification import (
    Hertz3DRun, deformable_hertz_reference, qualify_deformable_hertz_3d,
)


def reference():
    return deformable_hertz_reference(1000., .01, 210e9, .3, 210e9, .3)


def run(cells, scale, domain=12., **kw):
    ref = reference()
    return Hertz3DRun(cells, domain, cells**3, 2*cells**3,
                      1000.*(1+scale), 1000.,
                      ref.contact_radius*(1+scale), ref.indentation*(1+scale),
                      ref.peak_pressure*(1+scale), abs(scale), **kw)


def test_two_deformable_body_hertz_oracle_values():
    ref = reference()
    assert ref.effective_modulus == pytest.approx(115384615384.61537)
    assert ref.contact_radius == pytest.approx(0.000402072575858906)
    assert ref.indentation == pytest.approx(1.6166235625781572e-5)
    assert ref.peak_pressure == pytest.approx(2953469442.9062705)


def test_complete_converged_evidence_can_pass_every_gate():
    runs = (run(4, .08), run(8, .045), run(16, .02))
    result = qualify_deformable_hertz_3d(1000., reference(), runs, run(16, .019, 18.))
    assert result.status == "qualified"
    assert not result.missing_evidence
    assert max(result.errors.values()) < .03


@pytest.mark.parametrize("override, phrase", [
    ({"sphere_is_deformable": False}, "two deformable"),
    ({"spatial_dimension": 2}, "three-dimensional"),
    ({"pressure_was_integrated": False}, "integrated surface pressure"),
])
def test_axisymmetric_rigid_or_nonintegrated_shortcuts_fail_closed(override, phrase):
    runs = tuple(run(n, e, **override) for n, e in ((4,.08),(8,.045),(16,.02)))
    result = qualify_deformable_hertz_3d(1000., reference(), runs, run(16,.019,18.))
    assert result.status == "blocked"
    assert any(phrase in item for item in result.missing_evidence)


def test_requires_three_meshes_and_a_valid_domain_study():
    runs = (run(4,.08), run(8,.02))
    result = qualify_deformable_hertz_3d(1000., reference(), runs, None)
    assert result.status == "blocked"
    assert "at least three contact-zone mesh densities" in result.missing_evidence
    assert "larger-domain truncation run" in result.missing_evidence


def test_current_repository_has_no_general_3d_run_and_is_explicitly_blocked():
    result = qualify_deformable_hertz_3d(1000., reference(), (), None)
    assert result.status == "blocked"
    assert result.errors == {}
    assert "global deformable-to-deformable 3-D contact results" in result.missing_evidence


def test_nonconverged_or_above_threshold_results_remain_blocked():
    runs = (run(4,.04), run(8,.05), run(16,.031))
    result = qualify_deformable_hertz_3d(1000., reference(), runs, run(16,.03,18.))
    assert result.status == "blocked"
    assert any("below 3%" in item for item in result.missing_evidence)
    assert any("monotone" in item for item in result.missing_evidence)
