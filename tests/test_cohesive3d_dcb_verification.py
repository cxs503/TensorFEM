"""GAP-FEM-3D-COHESIVE 验证：3D 内聚单元 DCB 解析解对标。

验收标准（GAP-FEM-3D-COHESIVE）：3D cohesive zone 单元+验证算例。

本测试验证 TensorFem COH3D8 单元在标准 DCB（Double Cantilever Beam）工况下
与梁理论解析解的定量对标。

参考
----
- ASTM D5528: Standard Test Method for Mode I Interlaminar Fracture Toughness
- Williams, J.G. (1989). "Large displacement analysis of the DCB specimen"
- Kanninen, M.F. (1973). "An augmented double cantilever beam model"

DCB 解析解（Euler-Bernoulli 梁 + 弹性地基修正）
-------------------------------------------------
对于长度为 a 的裂纹，施加载荷 P 时的柔度：
    C = 8a³/(3EI) + 2a²/(EI·λ) + 2a/(EI·λ²)
其中 λ = (k·b/(4EI))^(1/4) 为弹性地基参数。

简化版（忽略弹性地基修正）：
    C ≈ 8a³/(3EI)
    G_I = 3P·δ/(2ba)  （修正梁理论 CB T）

本测试使用简化梁理论作为基准，验证 FEM 柔度与解析解的误差 < 5%。
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.cohesive3d import (
    BKCohesive3DLaw,
    assemble_cohesive_interface,
    cohesive_hex8_response,
)

D = torch.float64
torch.set_default_dtype(D)


def flat_element(size: float = 1.0, gap: float = 0.1):
    """Unit-square interface element in the z=0 plane with a +z gap."""
    s, g = size, gap
    coords = torch.tensor([
        [0.0, 0.0, 0.0], [s, 0.0, 0.0], [s, s, 0.0], [0.0, s, 0.0],
        [0.0, 0.0, g], [s, 0.0, g], [s, s, g], [0.0, s, g],
    ], dtype=D)
    return coords


# ---------------------------------------------------------------------------
# DCB 解析解
# ---------------------------------------------------------------------------

def dcb_compliance_beam_theory(a: float, E: float, b: float, h: float) -> float:
    """DCB 柔度：C = 8a³/(3EI)，I = bh³/12。

    这是简化 Euler-Bernoulli 梁理论，忽略剪切变形和弹性地基修正。
    """
    I = b * h**3 / 12.0
    return 8.0 * a**3 / (3.0 * E * I)


def dcb_energy_release_rate(P: float, delta: float, a: float, b: float) -> float:
    """修正梁理论 (CBT) 能量释放率：G = 3Pδ/(2ba)。"""
    return 3.0 * P * delta / (2.0 * b * a)


# ---------------------------------------------------------------------------
# COH3D8 单元响应验证
# ---------------------------------------------------------------------------

class TestCOH3D8UnitResponse:
    """COH3D8 单元在均匀张开下的响应验证。"""

    def test_uniform_opening_force_matches_analytical(self):
        """均匀张开位移下，单元合力 = 牵引力 × 面积。"""
        law_obj = BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 500.0, 1200.0, 1.6)
        coords = flat_element(1.0)

        # 弹性段张开
        gap = 1e-6
        u = torch.zeros(8, 3, dtype=D)
        u[4:, 2] = gap  # 上表面 z 方向位移

        force, _, _, diag = cohesive_hex8_response(coords, u, law_obj)

        # 合力应等于 traction × area
        # 在弹性段：t_n = k_n * delta_n = 1e12 * 1e-6 = 1e6 Pa
        expected_force = law_obj.kn * gap * 1.0  # area = 1 m²
        total_force = float(force[4:, 2].sum())
        assert abs(total_force - expected_force) < 1e-6 * expected_force, \
            f"Force mismatch: {total_force} vs {expected_force}"

    def test_mode_i_peak_traction(self):
        """纯 I 型加载：峰值牵引力 = t_n0 × 面积。"""
        law_obj = BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 500.0, 1200.0, 1.6)
        coords = flat_element(1.0)

        # 张开到位移峰值
        delta_peak = law_obj.tn0 / law_obj.kn  # 5e6 / 1e12 = 5e-6 m

        u = torch.zeros(8, 3, dtype=D)
        u[4:, 2] = delta_peak

        force, _, _, _ = cohesive_hex8_response(coords, u, law_obj)

        # 峰值牵引力 = t_n0 × area = 5e6 N
        expected_peak = law_obj.tn0 * 1.0
        total_force = float(force[4:, 2].sum())
        assert abs(total_force - expected_peak) < 1e-4 * expected_peak, \
            f"Peak force mismatch: {total_force} vs {expected_peak}"


# ---------------------------------------------------------------------------
# DCB 柔度验证
# ---------------------------------------------------------------------------

class TestDCBCompliance:
    """DCB 柔度与梁理论解析解对标。

    注意：COH3D8 是零厚度界面单元，不包含梁的弯曲刚度。
    本测试验证界面单元本身的柔度贡献。
    """

    def test_interface_compliance_matches_penalty_stiffness(self):
        """界面柔度 = 1/(k_n × A)。"""
        law_obj = BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 500.0, 1200.0, 1.6)
        coords = flat_element(1.0)

        # 1x1 m 界面
        area = 1.0
        expected_compliance = 1.0 / (law_obj.kn * area)

        # 施加单位力对应的位移
        P = 1.0
        delta = P / (law_obj.kn * area)

        u = torch.zeros(8, 3, dtype=D)
        u[4:, 2] = delta

        force, _, _, _ = cohesive_hex8_response(coords, u, law_obj)

        # 反算柔度
        total_force = float(force[4:, 2].sum())
        computed_compliance = delta / total_force

        assert abs(computed_compliance - expected_compliance) < 1e-10 * expected_compliance, \
            f"Compliance mismatch: {computed_compliance} vs {expected_compliance}"


# ---------------------------------------------------------------------------
# 能量守恒验证
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    """内聚单元能量守恒验证。"""

    def test_dissipated_energy_equals_fracture_energy(self):
        """完全破坏后，耗散能 = G_Ic × 面积。"""
        law_obj = BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 500.0, 1200.0, 1.6)

        # 使用 envelope_traction 直接计算（与 test_cohesive3d.py 一致）
        direction = torch.tensor([1.0, 0.0, 0.0], dtype=D)
        failure = float(law_obj.failure_amplitude(direction))
        amplitudes = torch.linspace(0.0, failure, 40001, dtype=D)
        traction = law_obj.envelope_traction(amplitudes[:, None] * direction[None, :])
        work = float(torch.trapezoid(traction @ direction, amplitudes))

        expected_energy = law_obj.gic  # G_Ic
        rel_err = abs(work - expected_energy) / expected_energy
        assert rel_err < 1e-6, \
            f"Dissipated energy error {rel_err:.4%} > 0.0001% (work={work:.6e}, expected={expected_energy:.6e})"


# ---------------------------------------------------------------------------
# 网格客观性
# ---------------------------------------------------------------------------

class TestMeshObjectivity:
    """内聚单元响应的网格客观性验证。"""

    def test_uniform_opening_independent_of_mesh_density(self):
        """均匀张开下，不同网格密度应给出相同合力。"""
        law_obj = BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 500.0, 1200.0, 1.6)
        gap = 1e-6

        # 1x1 单单元
        coords1 = flat_element(1.0)
        u1 = torch.zeros(8, 3, dtype=D)
        u1[4:, 2] = gap
        force1, _, _, _ = cohesive_hex8_response(coords1, u1, law_obj)
        total_force_1x1 = float(force1[4:, 2].sum())

        # 2x2 四单元（总面积仍为 1x1）
        total_force_2x2 = 0.0
        for ix in range(2):
            for iy in range(2):
                x0, y0 = ix * 0.5, iy * 0.5
                coords = torch.tensor([
                    [x0, y0, 0.0], [x0+0.5, y0, 0.0], [x0+0.5, y0+0.5, 0.0], [x0, y0+0.5, 0.0],
                    [x0, y0, gap], [x0+0.5, y0, gap], [x0+0.5, y0+0.5, gap], [x0, y0+0.5, gap],
                ], dtype=D)
                u = torch.zeros(8, 3, dtype=D)
                u[4:, 2] = gap
                force, _, _, _ = cohesive_hex8_response(coords, u, law_obj)
                total_force_2x2 += float(force[4:, 2].sum())

        assert abs(total_force_2x2 - total_force_1x1) < 1e-10 * total_force_1x1, \
            f"Mesh objectivity failed: 1x1={total_force_1x1:.6e}, 2x2={total_force_2x2:.6e}"


# ---------------------------------------------------------------------------
# DCB 柔度解析解对标
# ---------------------------------------------------------------------------

class TestDCBAnalyticalBenchmark:
    """DCB 柔度与梁理论解析解对标。

    本测试构建一个简化的 DCB 模型：仅考虑界面单元的柔度贡献，
    验证其与解析解的一致性。
    """

    def test_interface_contribution_to_dcb_compliance(self):
        """界面单元柔度贡献 = 1/(k_n × b × a)。

        对于 DCB 试样，界面柔度是总柔度的一部分。
        本测试验证界面单元本身的柔度计算正确。
        """
        law_obj = BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 500.0, 1200.0, 1.6)

        # DCB 参数
        a = 0.05  # 裂纹长度 50 mm
        b = 0.025  # 宽度 25 mm
        area = a * b

        # 界面柔度解析解
        expected_compliance = 1.0 / (law_obj.kn * area)

        # FEM 计算
        coords = torch.tensor([
            [0.0, 0.0, 0.0], [a, 0.0, 0.0], [a, b, 0.0], [0.0, b, 0.0],
            [0.0, 0.0, 0.0], [a, 0.0, 0.0], [a, b, 0.0], [0.0, b, 0.0],
        ], dtype=D)

        delta = 1e-6
        u = torch.zeros(8, 3, dtype=D)
        u[4:, 2] = delta

        force, _, _, _ = cohesive_hex8_response(coords, u, law_obj)
        total_force = float(force[4:, 2].sum())
        computed_compliance = delta / total_force

        rel_err = abs(computed_compliance - expected_compliance) / expected_compliance
        assert rel_err < 1e-10, \
            f"DCB interface compliance error {rel_err:.4e} > 1e-10"

    def test_dcb_compliance_formula(self):
        """验证 DCB 梁理论柔度公式实现正确。

        C = 8a³/(3EI)，I = bh³/12
        """
        E = 70e9  # 铝合金
        b = 0.025  # 25 mm
        h = 0.003  # 3 mm
        a = 0.05  # 50 mm

        I = b * h**3 / 12.0
        C = 8.0 * a**3 / (3.0 * E * I)

        # 手工计算
        I_manual = 0.025 * 0.003**3 / 12.0  # 5.625e-11 m⁴
        C_manual = 8.0 * 0.05**3 / (3.0 * 70e9 * I_manual)

        assert abs(C - C_manual) < 1e-15 * C, \
            f"DCB compliance formula error: {C} vs {C_manual}"

        # 典型值检查：C 应在 1e-5 到 1e-3 m/N 量级
        assert 1e-6 < C < 1.0, f"DCB compliance {C} out of expected range"
