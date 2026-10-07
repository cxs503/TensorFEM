import pytest

from tensorfem.marine_discrete_panel import (
    discrete_stiffener_buckling, nonuniform_corrosion_buckling,
    run_discrete_panel_qualification,
)


def test_discrete_stiffener_and_nonuniform_corrosion_pass_gates():
    report = run_discrete_panel_qualification()
    assert report["passed"] is True
    assert report["stiffener"]["relative_error"] < 0.03
    errors = [r["relative_error"] for r in report["nonuniform_corrosion"]["results"]]
    assert errors[-1] < 0.03
    assert errors[0] > 0.03
    assert max(errors[1:]) < 1e-12
    assert "not stiffener tripping" in report["scope"]


def test_discrete_stiffener_is_not_a_smeared_rigidity():
    row = discrete_stiffener_buckling(stiffener_count=4)
    assert len(row["attachment_locations"]) == 4
    assert row["discrete_beam_contribution"] > 0.0
    assert row["relative_error"] < 1e-12


def test_nonuniform_loss_reduces_ritz_buckling_load():
    pristine = nonuniform_corrosion_buckling(cells=16, loss_intensity=0.0)
    corroded = nonuniform_corrosion_buckling(cells=16, loss_intensity=0.4)
    assert corroded["critical_line_load"] < pristine["critical_line_load"]
    assert corroded["minimum_thickness"] < 0.012


@pytest.mark.parametrize("call", [
    lambda: discrete_stiffener_buckling(stiffener_count=0),
    lambda: discrete_stiffener_buckling(poisson=0.5),
    lambda: nonuniform_corrosion_buckling(cells=True),
    lambda: nonuniform_corrosion_buckling(cells=8, loss_intensity=1.0),
])
def test_invalid_inputs_fail_closed(call):
    with pytest.raises(ValueError):
        call()
