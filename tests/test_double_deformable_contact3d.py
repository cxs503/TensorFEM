import pytest
from tensorfem.double_deformable_contact3d import (
    _reaction_oracle, run_double_deformable_contact_qualification)


def test_two_compliant_body_global_contact_closure_passes():
    r = run_double_deformable_contact_qualification()
    assert r["passed"] and r["reaction_relative_error"] < .03
    assert r["force_imbalance"] < 1e-10 and r["moment_imbalance"] < 1e-10
    assert r["objectivity_relative_error"] < 1e-10
    assert r["master_slave_interchange_relative_error"] < .03
    assert r["rollback_exact"] is True
    assert "not general double-sided mortar" in r["scope"]


def test_independent_series_compliance_oracle():
    value = _reaction_oracle(force=100., clearance=.05, normal_penalty=5e4,
        slave_stiffness=1e3, master_node_stiffness=1e3)
    assert value == pytest.approx(61.57635467980296)
