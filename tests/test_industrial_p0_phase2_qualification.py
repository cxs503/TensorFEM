import copy
import pytest
from tensorfem.industrial_p0_phase2_qualification import (
    run_industrial_p0_phase2_qualification,validate_industrial_p0_phase2_qualification)


@pytest.fixture(scope="module")
def report():return run_industrial_p0_phase2_qualification()


def test_phase2_real_state_paths_pass(report):
    assert validate_industrial_p0_phase2_qualification(report) is report
    assert report["shell"]["material_relative_error"]<.03
    assert report["shell"]["permanent_extension"]>0
    assert report["contact"]["force_imbalance"]<1e-10


def test_external_and_solver_boundaries_stay_fail_closed(report):
    assert report["external_contract_status"]["bathe-bolourchi-1979-large-rotation-pure-bending"]=="qualified"
    assert report["external_contract_status"]["karatas-yuksel-2021-clamped-ring-load"]=="blocked"
    assert report["external_contract_status"]["hertz-1882-sphere-elastic-halfspace"]=="blocked"
    assert report["categories"]["finite_rotation_shell_plasticity_postbuckling"]["status"]=="not_qualified"
    changed=copy.deepcopy(report);changed["categories"]["general_mortar_self_contact"]["status"]="qualified"
    with pytest.raises(ValueError,match="hash mismatch"):validate_industrial_p0_phase2_qualification(changed)
