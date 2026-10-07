import pytest
from tensorfem.standard_benchmarks import (cantilever_beam, cantilever_beam_field,
    simply_supported_plate_center, hertz_contact_force, convergence)

def test_closed_form_fields_and_references():
    beam=cantilever_beam(); assert beam["tip_displacement"]>0 and beam["root_moment"]==100
    field=cantilever_beam_field(0.5,0.1); assert field.displacement[1]>0 and field.stress[0]<0
    assert simply_supported_plate_center()["center_deflection"]>0
    assert hertz_contact_force(0.0)==0.0
    assert convergence([(4,1.0),(8,1.1)],1.0)[1].relative_error == pytest.approx(0.1)

def test_hertz_rejects_negative_displacement():
    with pytest.raises(ValueError): hertz_contact_force(-1e-6)
