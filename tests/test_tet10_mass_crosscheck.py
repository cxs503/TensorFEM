"""Independent（闭式）交叉核对 TET10 一致质量阵 —— 补充 ``test_tet10_mass.py``。

``test_tet10_mass.py`` 把出厂质量阵对拍 24^3 Duffy 数值参考。本文件用**无积分**的
重心坐标闭式参考（``int N_i N_j dV`` 的精确单项式公式）再核一次，并补三条出厂
文档没有量化的判据：刚体转动惯量、行和集中负质量、以及杆频率锚在三维侧向耦合下
的真实收敛级。门限故意宽于实测值，失败意味着真缺陷而不是平台舍入。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfem.quadratic_solid import tet10_quadrature
from tensorfem.tet10_mass_crosscheck import (
    block_total,
    curved_element_study,
    free_free_modes,
    keast_mass_error,
    lumping_report,
    reference_mass,
    reference_mass_error,
    reference_mass_scalar,
    rigid_rotation_energy,
    rigid_velocity_energy,
    rule_order_study,
    tet10_from_corners,
    tetra_inertia,
    axial_rod_modes,
    _single_tet,
    monomial_integral,
)
from tensorfem.quadratic_solid import tet10_consistent_mass, tet10_volume

DENSITY = 7850.0


def _element():
    return tet10_from_corners(_single_tet())[None]


def test_closed_form_reference_self_verifies():
    assert abs(monomial_integral((0, 0, 0, 0)) - 1.0) < 1e-15
    assert abs(monomial_integral((1, 0, 0, 0)) - 0.25) < 1e-15
    scalar = reference_mass_scalar()
    # row sums of the reference scalar block are the analytic int N_i dV:
    # -1/20 at a vertex, +1/5 at a mid-edge node (reference volume 1)
    rows = scalar.sum(-1)
    assert np.allclose(rows[:4], -1.0 / 20.0, atol=1e-14)
    assert np.allclose(rows[4:], 1.0 / 5.0, atol=1e-14)
    assert abs(rows.sum() - 1.0) < 1e-14


def test_shipped_mass_matches_the_quadrature_free_reference():
    x = _element()
    report = reference_mass_error(x, DENSITY)
    assert report["frobenius_relative"] < 1e-12
    assert report["max_abs_relative_to_diagonal"] < 1e-12
    assert report["total_mass_relative"] < 1e-14
    assert abs(block_total(tet10_consistent_mass(x, DENSITY))
               - DENSITY * float(tet10_volume(x)[0])) < 1e-12 * DENSITY * float(tet10_volume(x)[0])
    assert abs(block_total(torch.as_tensor(reference_mass(x.numpy(), DENSITY)))
               - DENSITY * float(tet10_volume(x)[0])) < 1e-12 * DENSITY * float(tet10_volume(x)[0])


def test_rigid_body_energies_are_exact():
    x = _element()
    m = tet10_consistent_mass(x, DENSITY)[0]
    mass = DENSITY * float(tet10_volume(x)[0])
    v = [7.1, 3.0, 0.0]
    assert abs(rigid_velocity_energy(m, v) - 0.5 * mass * sum(c * c for c in v)) < 1e-12 * 0.5 * mass * sum(c * c for c in v)
    corners = _single_tet()
    inertia = tetra_inertia(corners.numpy(), density=DENSITY)
    for omega in ([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]):
        w = np.asarray(omega, dtype=float)
        exact = 0.5 * float(w @ inertia @ w)
        measured = rigid_rotation_energy(m, x[0], omega, about=corners.numpy().mean(axis=0))
        assert abs(measured - exact) < 1e-11 * exact


def test_free_free_rigid_modes_are_zero():
    report = free_free_modes()
    assert report["rigid_relative_max"] < 1e-12
    assert report["smallest_elastic_relative"] > 1e-3


def test_shipped_keast_rule_is_still_rejected():
    report = keast_mass_error()
    assert report["frobenius_relative"] > 0.19          # 19.677 %
    assert report["total_mass_relative"] < 1e-12        # ... while the total is exact


def test_default_rule_is_the_first_exact_one():
    rows = {r["order"]: r["frobenius_relative"] for r in rule_order_study()}
    assert rows[2] > 1e-1 and rows[3] > 1e-2
    for order in (4, 5, 6, 8):
        assert rows[order] < 1e-13


def test_row_sum_lumping_trap_and_hrz_positivity():
    report = lumping_report()
    assert report["row_sum_negative_vertex_count"] == 4
    assert report["row_sum_negative_components"] == 12
    assert report["row_sum_total_relative_error"] < 1e-14
    assert report["hrz_min_entry"] > 0.0
    assert report["hrz_total_relative_error"] < 1e-12


def test_curved_element_converges_with_rule_order():
    rows = {r["order"]: r for r in curved_element_study()["rows"]}
    assert rows[2]["frobenius_relative"] > 1e-1
    assert rows[4]["frobenius_relative"] < 1e-12
    assert rows[2]["element_volume_relative"] < 1e-12   # volume exact even when M is not


def test_rod_anchor_isolates_the_mass_operator_only_without_poisson_coupling():
    """``nu = 0`` -> the 3-D solution is the bar mode, so the mass operator is graded.

    ``nu = 0.3`` leaves a ~1 % level plateau: the lateral contraction coupling of the
    fully clamped 3-D rod, not the mass matrix, dominates there.  The gate pins the
    measured level so the claim cannot silently improve.
    """
    nu0 = axial_rod_modes(4, poisson=0.0)["relative_errors"][0]
    nu3 = axial_rod_modes(4)["relative_errors"][0]
    assert nu0 < 5e-5      # measured 1.4e-5
    assert 1e-3 < nu3 < 3e-2   # measured 1.04 %
    lumped = axial_rod_modes(4, poisson=0.0, lumping="hrz")["relative_errors"][0]
    assert lumped > 10.0 * nu0  # measured ratio ~ 50


def test_per_element_density_and_validation():
    x = torch.cat((_element(), _element()))
    single = tet10_consistent_mass(_element(), DENSITY)[0]
    pair = tet10_consistent_mass(x, torch.tensor([DENSITY, 2.0 * DENSITY], dtype=x.dtype))
    assert torch.allclose(pair[0], single)
    assert torch.allclose(pair[1], 2.0 * single)
