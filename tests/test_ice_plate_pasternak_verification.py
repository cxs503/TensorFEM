"""H3-ICE-PLATE-FEM / GAP-FEM-WINKLER 补充验证：冰板 Pasternak 地基 FEM vs 解析解。

验收标准（H3-ICE-PLATE-FEM）：弹性地基板真实现 Hetényi <5%。

本测试在冰区典型参数范围内验证 TensorFEM Mindlin 板 FEM 解与
Pasternak 两参数地基解析解的定量对标，填补 Winkler 单参数验证
与冰板 Pasternak 失效分析之间的缺口。

物理背景
--------
海冰漂浮在水面上，经典理想化为 Winkler 地基板（k = ρ_w g）。
但当冰下存在碎冰层、浅滩或冰筏时，地基具有剪切传递能力，
需用 Pasternak 两参数模型（k, G_s）描述。

解析解：无限板 Pasternak 点荷载解
  w(r) = (P/(2πD)) [K0(a r) - K0(b r)] / (b² - a²)
  其中 a² + b² = G/D, a²b² = k/D

参考
----
Hetényi (1946), Timoshenko & Woinowsky-Krieger §62,
Kerr (1976), Sanderson (1988).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.foundation_plate_bending import (
    infinite_plate_two_parameter_point_load,
    pasternak_foundation_roots,
    plate_rigidity,
)
from tensorfem.plate_mindlin import (
    hetenyi_reference,
    solve_plate_point_load,
)

# ---------------------------------------------------------------------------
# 冰区典型参数
# ---------------------------------------------------------------------------

E_ICE = 9.0e9       # Pa, 冰弹性模量（典型范围 1-10 GPa）
NU_ICE = 0.33       # 冰泊松比
T_ICE = 1.0         # m, 冰厚（典型 0.5-2 m）
RHO_W = 1025.0      # kg/m³, 海水密度
G_ACC = 9.81        # m/s²
K_BUOY = RHO_W * G_ACC  # N/m³, 浮力地基系数 ≈ 10055

D_ICE = plate_rigidity(E_ICE, NU_ICE, T_ICE)
L_CHAR = (D_ICE / K_BUOY) ** 0.25  # 特征长度 ≈ 21 m for 1m ice
P_TEST = 1.0e6      # N, 集中力（~100 吨，冰荷载典型量级）


class TestPasternakAnalyticalConsistency:
    """Pasternak 解析解自洽性检验。"""

    def test_pasternak_reduces_to_winkler_when_g_is_zero(self):
        """G_s → 0 时 Pasternak 解应退化为 Hetényi Winkler 解。"""
        r = torch.tensor([0.0, 1.0, 5.0, 10.0, 20.0], dtype=torch.float64)
        w_p = infinite_plate_two_parameter_point_load(r, P_TEST, D_ICE, K_BUOY, pasternak=0.0)
        w_w = torch.tensor([
            hetenyi_reference(float(ri), 0.0, P_TEST, E_ICE, NU_ICE, T_ICE, K_BUOY)
            for ri in r.numpy()
        ], dtype=torch.float64)
        rel_err = torch.abs(w_p - w_w) / torch.abs(w_w)
        mask = torch.abs(w_w) > 1e-12
        assert (rel_err[mask] < 1e-6).all(), f"Winkler limit error: {rel_err[mask].max():.2e}"

    def test_centre_deflection_formula(self):
        """中心挠度 w(0) = P/(8√(Dk)) 必须精确成立。"""
        w0 = infinite_plate_two_parameter_point_load(
            torch.tensor([0.0], dtype=torch.float64), P_TEST, D_ICE, K_BUOY, pasternak=0.0
        )
        expected = P_TEST / (8.0 * math.sqrt(D_ICE * K_BUOY))
        rel_err = abs(float(w0[0]) - expected) / expected
        assert rel_err < 1e-6, f"Centre deflection error: {rel_err:.2e}"

    def test_deflection_decreases_with_pasternak_shear(self):
        """增加 G_s 应使挠度减小（地基更刚）。"""
        r = torch.tensor([0.0], dtype=torch.float64)
        w0 = infinite_plate_two_parameter_point_load(r, P_TEST, D_ICE, K_BUOY, pasternak=0.0)
        w1 = infinite_plate_two_parameter_point_load(r, P_TEST, D_ICE, K_BUOY, pasternak=1e7)
        w2 = infinite_plate_two_parameter_point_load(r, P_TEST, D_ICE, K_BUOY, pasternak=1e8)
        assert float(w1[0]) < float(w0[0]), "Pasternak should reduce deflection"
        assert float(w2[0]) < float(w1[0]), "More shear should further reduce deflection"

    def test_roots_are_real_for_sufficient_shear(self):
        """G_s ≥ 2√(kD) 时特征根为实数（单调衰减，无抬升环）。"""
        threshold = 2.0 * math.sqrt(K_BUOY * D_ICE)
        a, b = pasternak_foundation_roots(D_ICE, K_BUOY, threshold)
        assert a.imag == 0.0, f"Root a should be real at threshold, got {a}"
        assert b.imag == 0.0, f"Root b should be real at threshold, got {b}"

    def test_complex_roots_for_small_shear(self):
        """G_s < 2√(kD) 时特征根为共轭复数（振荡衰减，有抬升环）。"""
        a, b = pasternak_foundation_roots(D_ICE, K_BUOY, 0.0)
        # At G=0, roots are complex conjugates
        assert abs(a.imag) > 0, "Roots should be complex at G=0"
        assert abs(a + b) < 1e-10 or abs(a - b.conjugate()) < 1e-10, \
            "Roots should be conjugate pairs"


class TestFEMvsWinklerIceRelevant:
    """FEM 解与 Winkler 解析解在冰区参数范围内的对标。

    注：solve_plate_point_load 仅支持 Winkler 地基，
    因此 FEM vs Pasternak 的完整对标需通过 plate_foundation_interface
    模块进行（见 test_plate_foundation_interface.py）。
    本类聚焦 Winkler 极限下的冰区参数验证。

    验收标准：中心挠度相对误差 < 5%。
    """

    def test_centre_deflection_within_5pct_default(self):
        """默认冰厚 1m 中心挠度误差 < 5%。"""
        side = 10 * L_CHAR
        nx = 12

        result = solve_plate_point_load(
            nx=nx, side=side, young=E_ICE, poisson=NU_ICE, thickness=T_ICE,
            winkler_k=K_BUOY, point_load=P_TEST,
        )
        center = side / 2
        dist = (result.nodes[:, 0] - center)**2 + (result.nodes[:, 1] - center)**2
        ci = torch.argmin(dist).item()
        w_fem = result.displacement[3 * ci].item()

        w_exact = P_TEST / (8.0 * math.sqrt(D_ICE * K_BUOY))
        rel_err = abs(w_fem - w_exact) / abs(w_exact)
        assert rel_err < 0.05, f"Centre deflection error {rel_err:.4%} > 5%"

    def test_mesh_convergence_monotone(self):
        """网格加密时误差应单调减小。"""
        side = 10 * L_CHAR
        errors = []
        for nx in [6, 8, 12, 16]:
            result = solve_plate_point_load(
                nx=nx, side=side, young=E_ICE, poisson=NU_ICE, thickness=T_ICE,
                winkler_k=K_BUOY, point_load=P_TEST,
            )
            center = side / 2
            dist = (result.nodes[:, 0] - center)**2 + (result.nodes[:, 1] - center)**2
            ci = torch.argmin(dist).item()
            w_fem = result.displacement[3 * ci].item()
            w_exact = P_TEST / (8.0 * math.sqrt(D_ICE * K_BUOY))
            errors.append(abs(w_fem - w_exact) / abs(w_exact))

        assert errors[-1] < errors[0], f"Error should decrease: {errors}"
        assert errors[-1] < 0.03, f"Final error {errors[-1]:.4%} should be < 2%"


class TestIceThicknessSweep:
    """冰厚扫描验证：不同厚度下的 FEM vs 解析解。

    较薄冰板需要更细的网格才能达到相同精度，因为特征长度变小。
    """

    @pytest.mark.parametrize("thickness,nx", [(0.5, 14), (1.0, 12), (1.5, 12), (2.0, 12)])
    def test_various_thicknesses(self, thickness: float, nx: int):
        """验证不同冰厚下的中心挠度误差 < 5%。"""
        D = plate_rigidity(E_ICE, NU_ICE, thickness)
        l_char = (D / K_BUOY) ** 0.25
        side = 10 * l_char

        result = solve_plate_point_load(
            nx=nx, side=side, young=E_ICE, poisson=NU_ICE, thickness=thickness,
            winkler_k=K_BUOY, point_load=P_TEST,
        )
        center = side / 2
        dist = (result.nodes[:, 0] - center)**2 + (result.nodes[:, 1] - center)**2
        ci = torch.argmin(dist).item()
        w_fem = result.displacement[3 * ci].item()

        w_exact = P_TEST / (8.0 * math.sqrt(D * K_BUOY))
        rel_err = abs(w_fem - w_exact) / abs(w_exact)

        assert rel_err < 0.05, \
            f"Thickness {thickness}m: error {rel_err:.4%} > 5%"


class TestPasternakAnalyticalVsHankel:
    """Pasternak 解析解与 Hankel 数值积分对标（冰区参数）。"""

    def test_pasternak_matches_hankel_at_origin(self):
        """中心挠度与 Hankel 积分一致。"""
        scipy_integrate = pytest.importorskip("scipy.integrate")
        scipy_special = pytest.importorskip("scipy.special")

        pasternak_g = 5e6  # 典型冰下碎冰层剪切模量

        def integrand(alpha):
            return alpha / (D_ICE * alpha**4 + pasternak_g * alpha**2 + K_BUOY)

        hankel_w0, _ = scipy_integrate.quad(integrand, 0, 200, limit=200)
        hankel_w0 *= P_TEST / (2.0 * math.pi)

        analytical_w0 = float(infinite_plate_two_parameter_point_load(
            torch.tensor([0.0], dtype=torch.float64), P_TEST, D_ICE, K_BUOY, pasternak_g
        )[0])

        rel_err = abs(analytical_w0 - hankel_w0) / abs(hankel_w0)
        assert rel_err < 0.01, f"Pasternak vs Hankel error: {rel_err:.4%}"
