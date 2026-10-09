import pytest

from tensorfem.marine_degradation_qualification import (
    _residual_yield_onset, run_marine_degradation_qualification,
)


def test_stiffener_corrosion_and_residual_sensitivities_pass_three_percent_gate():
    report = run_marine_degradation_qualification()
    assert report["passed"] is True
    assert report["tolerance"] == 0.03
    assert report["stiffened_panel"]["relative_error"] < 0.03
    assert max(row["navier_relative_error"] for row in report["uniform_corrosion"]) < 0.03
    assert max(row["ratio_relative_error"] for row in report["uniform_corrosion"]) < 1e-12
    residual = report["residual_stress"]
    assert residual["relative_errors"][-1] < 0.03
    assert all(a > b for a, b in zip(residual["relative_errors"], residual["relative_errors"][1:]))
    assert residual["continuum_onset"] < residual["zero_residual_onset"]
    assert "not discrete tripping" in report["scope"]


@pytest.mark.parametrize("kwargs", [
    {"residual_stress": -0.1}, {"residual_stress": 1.0},
    {"residual_stress": 0.2, "fibres": True},
    {"residual_stress": 0.2, "fibres": 3},
])
def test_residual_yield_oracle_rejects_nonphysical_inputs(kwargs):
    with pytest.raises(ValueError):
        _residual_yield_onset(**kwargs)
