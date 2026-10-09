import copy
import pytest

from tensorfem.industrial_p0_qualification import (
    run_industrial_p0_qualification,validate_industrial_p0_qualification)


@pytest.fixture(scope="module")
def report(): return run_industrial_p0_qualification()


def test_p0_vertical_slices_pass_and_are_traceable(report):
    assert validate_industrial_p0_qualification(report) is report
    assert report["shell"]["residual_norm"]<3e-7
    assert report["contact"]["maximum_residual"]<1e-9
    assert report["workflow"]["lossless_roundtrip"]


def test_unfinished_industrial_boundaries_remain_fail_closed(report):
    categories=report["categories"]
    assert categories["shell_integration_point_plasticity"]["status"]=="not_qualified"
    assert categories["general_3d_mortar_self_contact"]["status"]=="not_qualified"
    assert categories["distributed_production_solver"]["status"]=="not_qualified"
    changed=copy.deepcopy(report);changed["categories"]["distributed_production_solver"]["status"]="qualified"
    with pytest.raises(ValueError,match="hash mismatch"):validate_industrial_p0_qualification(changed)
