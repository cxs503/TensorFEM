"""Verification tests for :mod:`tensorfem.welding_thermo_mechanical`.

Fast tests for CI: small meshes, short times.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem import welding_thermo_mechanical as wtm

# ---------------------------------------------------------------------------
# 1. Goldak source tests (analytical, fast)
# ---------------------------------------------------------------------------


def test_goldak_peak_at_source():
    src = wtm.GoldakSource()
    q_center = src.q_surface(0.0, 0.0, 0.0)
    q_offset = src.q_surface(0.001, 0.0, 0.0)
    assert q_center > q_offset
    assert q_center > 0


def test_goldak_power_integral():
    src = wtm.GoldakSource(Q=5000.0)
    assert src.total_power_check() == pytest.approx(5000.0, rel=1e-10)


def test_goldak_front_rear_asymmetry():
    src = wtm.GoldakSource(a_f=0.004, a_r=0.008)
    q_front = src.q_surface(-0.002, 0.0, 0.0)
    q_rear = src.q_surface(0.006, 0.0, 0.0)
    assert q_front != q_rear


def test_goldak_decay_with_distance():
    src = wtm.GoldakSource()
    q0 = src.q_surface(0.0, 0.0, 0.0)
    q_far = src.q_surface(0.02, 0.02, 0.0)
    assert q_far < q0 * 0.01


# ---------------------------------------------------------------------------
# 2. Rosenthal analytical solution tests (analytical, fast)
# ---------------------------------------------------------------------------


def test_rosenthal_on_axis_decay():
    T0 = wtm.rosenthal_on_axis(5000.0, 0.005, 45.0, 1.14e-5, 0.001, 300.0)
    T_far = wtm.rosenthal_on_axis(5000.0, 0.005, 45.0, 1.14e-5, 0.01, 300.0)
    assert T0 > T_far > 300.0


def test_rosenthal_far_field_finite():
    T = wtm.rosenthal_temperature(5000.0, 0.005, 45.0, 1.14e-5, 0.03, 0.0, 0.0, 300.0)
    assert 300.0 < T < 2000.0


def test_rosenthal_haz_radius_positive():
    haz = wtm.rosenthal_haz_radius(5000.0, 0.005, 45.0, 1.14e-5,
                                    T_haz=723.0 + 273.15, T0=300.0)
    assert 0.001 < haz < 0.02


def test_rosenthal_zero_speed_limit():
    Q, k = 5000.0, 45.0
    T_slow = wtm.rosenthal_temperature(Q, 1e-6, k, 1.14e-5, 0.01, 0.0, 0.0, 300.0)
    T_stat = 300.0 + Q / (2.0 * math.pi * k * 0.01)
    assert abs(T_slow - T_stat) / T_stat < 0.05


# ---------------------------------------------------------------------------
# 3. Timoshenko curvature tests (analytical, fast)
# ---------------------------------------------------------------------------


def test_timoshenko_zero_mismatch():
    kappa = wtm.timoshenko_curvature(1.2e-5, 1.2e-5, 100.0, 0.002,
                                      200e9, 200e9, 0.001, 0.001)
    assert abs(kappa) < 1e-15


def test_timoshenko_symmetric_layers():
    kappa = wtm.timoshenko_curvature(1.7e-5, 1.2e-5, 100.0, 0.002,
                                      200e9, 200e9, 0.001, 0.001)
    expected = 0.375
    assert kappa == pytest.approx(expected, rel=1e-12)


def test_timoshenko_linear_in_dT():
    k1 = wtm.timoshenko_curvature(1.7e-5, 1.2e-5, 50.0, 0.002,
                                   200e9, 200e9, 0.001, 0.001)
    k2 = wtm.timoshenko_curvature(1.7e-5, 1.2e-5, 100.0, 0.002,
                                   200e9, 200e9, 0.001, 0.001)
    assert k2 / k1 == pytest.approx(2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Thermal solve tests (small mesh, short time)
# ---------------------------------------------------------------------------


def test_thermal_solve_energy_positive():
    src = wtm.GoldakSource(Q=3000.0, v=0.003)
    params = wtm.WeldThermalParams()
    result = wtm.solve_welding_thermal(src, params, Lx=0.06, Ly=0.03,
                                        nx=12, ny=6, t_end=1.0, dt=0.05)
    assert result["energy_stored"] > 0
    assert result["energy_in"] > 0


def test_thermal_solve_peak_above_ambient():
    src = wtm.GoldakSource(Q=5000.0, v=0.005)
    params = wtm.WeldThermalParams(T0=300.0)
    result = wtm.solve_welding_thermal(src, params, Lx=0.08, Ly=0.04,
                                        nx=16, ny=8, t_end=2.0, dt=0.05)
    assert result["peak_T"] > params.T0


def test_thermal_solve_stable():
    src = wtm.GoldakSource(Q=3000.0, v=0.003)
    params = wtm.WeldThermalParams(T0=300.0)
    result = wtm.solve_welding_thermal(src, params, Lx=0.06, Ly=0.03,
                                        nx=12, ny=6, t_end=1.0, dt=0.05)
    assert result["peak_T"] < 3000.0


# ---------------------------------------------------------------------------
# 5. Thermo-elastic stress tests
# ---------------------------------------------------------------------------


def test_thermoelastic_stress_nonzero():
    nodes, elements = wtm.build_rectangular_plate(0.1, 0.05, 15, 8)
    T_field = torch.full((len(nodes),), 500.0, dtype=torch.float64)
    mech = wtm.WeldMechanicalParams()
    result = wtm.solve_thermoelastic_stress(nodes, elements, T_field, mech)
    assert result["max_sigma_vm"] > 0


def test_thermoelastic_stress_scales_with_dT():
    nodes, elements = wtm.build_rectangular_plate(0.1, 0.05, 10, 5)
    mech = wtm.WeldMechanicalParams()
    T1 = torch.full((len(nodes),), 400.0, dtype=torch.float64)
    T2 = torch.full((len(nodes),), 500.0, dtype=torch.float64)
    r1 = wtm.solve_thermoelastic_stress(nodes, elements, T1, mech)
    r2 = wtm.solve_thermoelastic_stress(nodes, elements, T2, mech)
    dT1 = 400.0 - mech.T_ref
    dT2 = 500.0 - mech.T_ref
    ratio_expected = dT2 / dT1
    if r1["max_sigma_vm"] > 0:
        ratio_actual = r2["max_sigma_vm"] / r1["max_sigma_vm"]
        assert abs(ratio_actual - ratio_expected) < 0.2 * ratio_expected


# ---------------------------------------------------------------------------
# 6. Angular deformation tests
# ---------------------------------------------------------------------------


def test_angular_deformation_positive():
    theta = wtm.estimate_angular_deformation(
        T_max=1500.0, T_amb=300.0, alpha_T=1.2e-5,
        plate_thickness=0.01, weld_length=0.1, E=210e9, sigma_y=250e6
    )
    assert theta > 0


def test_angular_deformation_scales_with_delta_T():
    theta1 = wtm.estimate_angular_deformation(
        800.0, 300.0, 1.2e-5, 0.01, 0.1, 210e9, 250e6
    )
    theta2 = wtm.estimate_angular_deformation(
        1300.0, 300.0, 1.2e-5, 0.01, 0.1, 210e9, 250e6
    )
    assert theta2 / theta1 == pytest.approx(2.0, rel=1e-10)


# ---------------------------------------------------------------------------
# 7. Helper tests
# ---------------------------------------------------------------------------


def test_relative_error_helper():
    assert wtm.relative_error(105.0, 100.0) == pytest.approx(0.05)
    assert wtm.relative_error(95.0, 100.0) == pytest.approx(0.05)
    assert wtm.relative_error(0.0, 0.0) == 0.0


def test_energy_closure_helper():
    assert wtm.energy_closure(100.0, 80.0, 15.0) == pytest.approx(0.05)
    assert wtm.energy_closure(100.0, 50.0, 50.0) == pytest.approx(0.0)


def test_gci_helper():
    g = wtm.gci(0.25, 1.0, 4.0, r=2.0)
    assert g >= 0


# ---------------------------------------------------------------------------
# 8. Full workflow tests (small)
# ---------------------------------------------------------------------------


def test_full_workflow_runs():
    case = wtm.WeldingCase(
        source=wtm.GoldakSource(Q=3000.0, v=0.003),
        t_end=1.0,
    )
    result = wtm.run_welding_thermo_mechanical(case, nx=12, ny=6, dt=0.05)
    assert "thermal" in result
    assert "mechanical" in result
    assert "verification" in result
    assert result["verification"]["T_peak_fem"] > case.thermal.T0


def test_full_workflow_verification_metrics():
    case = wtm.WeldingCase(
        source=wtm.GoldakSource(Q=4000.0, v=0.004),
        t_end=2.0,
    )
    result = wtm.run_welding_thermo_mechanical(case, nx=15, ny=8, dt=0.05)
    v = result["verification"]
    assert v["T_rosenthal_far_field"] > case.thermal.T0
    assert v["haz_rosenthal_half_width"] > 0
    assert v["kappa_timoshenko"] == v["kappa_timoshenko"]
    assert v["theta_estimate"] > 0
    assert v["energy_in"] > 0


def test_gate_a_goldak_vs_rosenthal():
    case = wtm.WeldingCase(source=wtm.GoldakSource(Q=4000.0, v=0.004), t_end=1.0)
    result = wtm.run_welding_thermo_mechanical(case, nx=12, ny=6, dt=0.05)
    v = result["verification"]
    assert v["T_rosenthal_far_field"] > case.thermal.T0
    assert v["T_peak_fem"] > case.thermal.T0


def test_gate_b_haz_boundary():
    case = wtm.WeldingCase(source=wtm.GoldakSource(Q=4000.0, v=0.004), t_end=1.0)
    result = wtm.run_welding_thermo_mechanical(case, nx=12, ny=6, dt=0.05)
    v = result["verification"]
    assert v["haz_rosenthal_half_width"] > 0.001


def test_gate_c_timoshenko_curvature():
    case = wtm.WeldingCase()
    result = wtm.run_welding_thermo_mechanical(case, nx=12, ny=6, dt=0.05)
    v = result["verification"]
    assert v["kappa_timoshenko"] == v["kappa_timoshenko"]
    assert abs(v["kappa_timoshenko"]) > 0


def test_gate_f_energy_conservation():
    case = wtm.WeldingCase(source=wtm.GoldakSource(Q=3000.0, v=0.003), t_end=1.0)
    result = wtm.run_welding_thermo_mechanical(case, nx=12, ny=6, dt=0.05)
    v = result["verification"]
    assert v["energy_in"] > v["energy_stored"]
    assert v["energy_stored"] > 0
