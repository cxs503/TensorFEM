import pytest
from tensorfem.standard_benchmarks import (cantilever_beam, cantilever_beam_field,
    simply_supported_plate_center, hertz_contact_force, convergence,
    axial_bar, axial_bar_field, pure_shear_patch, pure_shear_patch_field,
    benchmark_case_catalog)

def test_closed_form_fields_and_references():
    beam=cantilever_beam(); assert beam["tip_displacement"]>0 and beam["root_moment"]==100
    field=cantilever_beam_field(0.5,0.1); assert field.displacement[1]>0 and field.stress[0]<0
    assert simply_supported_plate_center()["center_deflection"]>0
    assert hertz_contact_force(0.0)==0.0
    assert convergence([(4,1.0),(8,1.1)],1.0)[1].relative_error == pytest.approx(0.1)

def test_hertz_rejects_negative_displacement():
    with pytest.raises(ValueError): hertz_contact_force(-1e-6)


def test_axial_bar_and_shear_patch_closed_form_fields():
    bar = axial_bar()
    assert bar["tip_displacement"] == pytest.approx(1000/(210e9*1e-4))
    assert axial_bar_field(0.25).stress[0] == pytest.approx(1.0e7)
    patch = pure_shear_patch()
    assert patch["top_displacement"] == pytest.approx(1e6/80e9)
    assert pure_shear_patch_field(.5, .5).stress[2] == pytest.approx(1e6)


def test_catalog_contains_five_documented_cases():
    catalog = benchmark_case_catalog()
    assert len(catalog) >= 5
    for case in catalog.values():
        assert case["problem"] and case["conditions"] and case["procedure"]
        assert case["result_fields"] and case["visualizations"]
