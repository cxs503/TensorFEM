import pytest
from tensorfem.hertz_quarter_patch import build_hertz_quarter_patch,quarter_patch_quality

def test_quarter_patch_resolves_contact_and_far_field_with_quality_gate():
    patch=build_hertz_quarter_patch(contact_radius=.2)
    quality=quarter_patch_quality(patch)
    assert quality["passed"] and quality["contact_cells_per_radius"]==8
    assert quality["domain_radii"]==8
    assert quality["surface_spacing_relative_to_radius"]==pytest.approx(1/8)
    assert float(patch.x[-1])==pytest.approx(1.6) and float(patch.z[-1])==pytest.approx(1.6)
    assert quality["maximum_adjacent_growth_ratio"]<=3

def test_quarter_patch_rejects_underresolved_shortcut():
    with pytest.raises(ValueError,match=">=6 contact cells"):
        build_hertz_quarter_patch(contact_radius=.2,contact_cells=4)
