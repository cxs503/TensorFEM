import json
import math
from pathlib import Path

import pytest
import torch

from tensorfem.marine_hydrodynamics import (
    AiryWave, MorisonMember, airy_kinematics, force_history_resultants,
    load_force_history, load_marine_case, morison_base_actions,
    morison_quarter_phase_oracle, run_marine_case,
)


ROOT = Path(__file__).parents[1]


def test_dispersion_and_boundary_kinematics():
    wave = AiryWave(2.0, 8.0, 30.0)
    k = wave.wave_number
    assert abs(wave.gravity * k * math.tanh(k * wave.depth) / wave.omega**2 - 1.0) < 1e-12
    kin = airy_kinematics(wave, torch.tensor([-30.0, 0.0]), 0.0, 0.0)
    assert torch.isfinite(torch.stack(list(kin.values()))).all()
    assert kin["w"][0] == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(ValueError, match="within"):
        airy_kinematics(wave, 0.1, 0.0, 0.0)


@pytest.mark.parametrize("current", [0.0, 0.8, -0.4])
def test_morison_actions_match_closed_form_oracle_below_three_percent(current):
    wave = AiryWave(3.0, 9.0, 25.0)
    member = MorisonMember(1.2, 1.05, 2.0)
    time = -math.pi / (2.0 * wave.omega)  # phase = pi/2 at x=0
    computed = morison_base_actions(wave, member, 0.0, time, current_velocity=current, quadrature_order=24)
    oracle = morison_quarter_phase_oracle(wave, member, current_velocity=current)
    for key in oracle:
        assert abs(computed[key] / oracle[key] - 1.0) < 0.03


def test_versioned_case_and_units_fail_closed(tmp_path):
    case = ROOT / "examples" / "marine" / "fixed_pile_airy_morison.json"
    output = run_marine_case(case)
    assert output["schema"] == "tensorfem.marine-load-result.v1"
    raw = json.loads(case.read_text())
    raw["units"]["force"] = "kN"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="SI"):
        load_marine_case(bad)
    raw["units"]["force"] = "N"
    raw["unexpected"] = 1
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema"):
        load_marine_case(bad)


def test_tensorlbm_force_history_adapter_and_resultants():
    history = load_force_history(ROOT / "examples" / "marine" / "tensorlbm_force_history.json")
    result = force_history_resultants(history, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]))
    assert result["force"].shape == (2, 3)
    assert result["force"][0].tolist() == pytest.approx([30.0, 0.0, 0.0])
    assert result["moment_about_origin"][0].tolist() == pytest.approx([0.0, 40.0, 0.0])


def test_odd_order_quadrature_is_supported():
    wave = AiryWave(1.0, 7.0, 18.0)
    member = MorisonMember(0.8, 1.0, 2.0)
    result = morison_base_actions(wave, member, 0.0, 0.2, quadrature_order=15)
    assert all(math.isfinite(value) for value in result.values())


def test_tensorlbm_exchange_rejects_wrong_shape_and_nonmonotone_time(tmp_path):
    source = ROOT / "examples" / "marine" / "tensorlbm_force_history.json"
    raw = json.loads(source.read_text())
    raw["times"] = [0.1, 0.0]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="increasing"):
        load_force_history(bad)
    raw["times"] = [0.0, 0.1]
    raw["forces"] = [[[1.0, 2.0]], [[1.0, 2.0]]]
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="shape"):
        load_force_history(bad)


def test_invalid_physical_parameters_fail_closed():
    with pytest.raises(ValueError, match="height"):
        AiryWave(20.0, 8.0, 5.0)
    with pytest.raises(ValueError, match="non-negative"):
        MorisonMember(1.0, -1.0, 2.0)
