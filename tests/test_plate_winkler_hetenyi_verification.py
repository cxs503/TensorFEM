"""GAP-FEM-WINKLER 验证：弹性地基板 FEM 解 vs Hetényi 解析解。

验收标准（GAP-FEM-WINKLER）：弹性地基板真实现 vs Hetényi <5%。

本测试验证 TensorFEM Q4 Mindlin 板单元 + Winkler 地基实现与 Hetényi (1946)
无限板精确解的定量对标。

参考
----
M. Hetényi, *Beams on Elastic Foundation*, Univ. of Michigan Press, 1946.
Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., §62.

关键公式
--------
无限板中心挠度：w(0) = P / (8·√(D·k_f))
特征长度：l = (D/k_f)^(1/4)
其中 D = E·t³/(12(1-ν²)) 为板弯曲刚度。

当有限板尺寸 ≥ 8l 时，边界效应可忽略，FEM 解应与 Hetényi 解一致。
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.plate_mindlin import (
    hetenyi_reference,
    solve_plate_point_load,
)


# ---------------------------------------------------------------------------
# 基准参数（冰板典型工况）
# ---------------------------------------------------------------------------

E_ICE = 9.0e9       # Pa, 冰弹性模量
NU_ICE = 0.33       # 冰泊松比
T_ICE = 0.5         # m, 冰厚
K_F_WATER = 1.0e4   # N/m³, 水地基系数（ρg ≈ 1025×9.81）
P_TEST = 1.0e5      # N, 集中力（~10 吨）

D_ICE = E_ICE * T_ICE**3 / (12.0 * (1.0 - NU_ICE**2))
L_CHAR = (D_ICE / K_F_WATER) ** 0.25  # 特征长度
W_HETENYI_0 = P_TEST / (8.0 * math.sqrt(D_ICE * K_F_WATER))


def _find_center_node(nodes: torch.Tensor, side: float) -> int:
    """Find the node closest to the plate center."""
    center = side / 2
    dist = (nodes[:, 0] - center)**2 + (nodes[:, 1] - center)**2
    return torch.argmin(dist).item()


# ---------------------------------------------------------------------------
# 1. Hetényi 解析解自洽性
# ---------------------------------------------------------------------------

class TestHetenyiAnalytical:
    """验证 Hetényi 解析解的基本性质。"""

    def test_center_deflection_formula(self):
        """w(0) = P/(8√(Dk)) 必须精确成立。"""
        w = hetenyi_reference(0.0, 0.0, P_TEST, E_ICE, NU_ICE, T_ICE, K_F_WATER)
        expected = P_TEST / (8.0 * math.sqrt(D_ICE * K_F_WATER))
        rel_err = abs(w - expected) / abs(expected)
        assert rel_err < 1e-6, f"Hetényi w(0) error: {rel_err:.2e}"

    def test_decay_with_distance(self):
        """挠度应随距离衰减。"""
        w0 = hetenyi_reference(0.0, 0.0, P_TEST, E_ICE, NU_ICE, T_ICE, K_F_WATER)
        w1 = hetenyi_reference(L_CHAR, 0.0, P_TEST, E_ICE, NU_ICE, T_ICE, K_F_WATER)
        w2 = hetenyi_reference(2 * L_CHAR, 0.0, P_TEST, E_ICE, NU_ICE, T_ICE, K_F_WATER)
        assert abs(w1) < abs(w0), "Deflection should decay at r=l"
        assert abs(w2) < abs(w1), "Deflection should decay at r=2l"

    def test_uplift_ring(self):
        """Kelvin 解在 r≈3.8l 处出现抬升环（w 变号）。"""
        w_near = hetenyi_reference(3.5 * L_CHAR, 0.0, P_TEST, E_ICE, NU_ICE, T_ICE, K_F_WATER)
        w_far = hetenyi_reference(4.5 * L_CHAR, 0.0, P_TEST, E_ICE, NU_ICE, T_ICE, K_F_WATER)
        assert w_near > 0, f"Expected downward deflection at 3.5l, got {w_near}"
        assert w_far < 0, f"Expected uplift at 4.5l, got {w_far}"


# ---------------------------------------------------------------------------
# 2. FEM vs Hetényi 定量对标（核心验收）
# ---------------------------------------------------------------------------

class TestFEMvsHetenyi:
    """FEM 解与 Hetényi 解析解的定量比较。

    验收标准：中心挠度相对误差 < 5%。
    """

    def _solve_and_compare(self, nx: int, side_mult: float, boundary: str = "simply_supported"):
        """求解并返回 (w_fem, w_hetenyi, rel_err)。"""
        side = side_mult * L_CHAR
        result = solve_plate_point_load(
            nx=nx, side=side,
            young=E_ICE, poisson=NU_ICE, thickness=T_ICE,
            winkler_k=K_F_WATER, point_load=P_TEST,
            boundary=boundary,
        )
        ci = _find_center_node(result.nodes, side)
        w_fem = result.displacement[3 * ci].item()
        w_het = W_HETENYI_0
        rel_err = abs(w_fem - w_het) / abs(w_het)
        return w_fem, w_het, rel_err

    def test_convergence_8l_ss(self):
        """8l 板简支：nx=12 时误差 < 5%（验收线）。"""
        w_fem, w_het, rel_err = self._solve_and_compare(nx=12, side_mult=8.0)
        assert rel_err < 0.05, \
            f"FEM vs Hetényi error {rel_err:.4%} > 5% (w_fem={w_fem:.4e}, w_het={w_het:.4e})"

    def test_convergence_8l_fine(self):
        """8l 板简支：nx=16 时误差 < 3%。"""
        w_fem, w_het, rel_err = self._solve_and_compare(nx=16, side_mult=8.0)
        assert rel_err < 0.03, \
            f"FEM vs Hetényi error {rel_err:.4%} > 3% (w_fem={w_fem:.4e}, w_het={w_het:.4e})"

    def test_convergence_10l(self):
        """10l 板简支：nx=12 时误差 < 5%。"""
        w_fem, w_het, rel_err = self._solve_and_compare(nx=12, side_mult=10.0)
        assert rel_err < 0.05, \
            f"FEM vs Hetényi error {rel_err:.4%} > 5%"

    def test_clamped_boundary(self):
        """固支边界：大板中心挠度也应与 Hetényi 一致（边界效应可忽略）。"""
        w_fem, w_het, rel_err = self._solve_and_compare(nx=12, side_mult=8.0, boundary="clamped")
        assert rel_err < 0.05, \
            f"Clamped FEM vs Hetényi error {rel_err:.4%} > 5%"

    def test_mesh_convergence_monotone(self):
        """网格细化应单调收敛。"""
        errors = []
        for nx in [6, 8, 12]:
            _, _, rel_err = self._solve_and_compare(nx=nx, side_mult=8.0)
            errors.append(rel_err)
        assert errors[2] < errors[1] < errors[0], \
            f"Error not monotonically decreasing: {errors}"


# ---------------------------------------------------------------------------
# 3. 特征长度参数扫描
# ---------------------------------------------------------------------------

class TestCharacteristicLengthScaling:
    """验证不同特征长度下的缩放律。"""

    def test_deflection_scales_with_P(self):
        """挠度应与载荷成正比。"""
        side = 8 * L_CHAR
        r1 = solve_plate_point_load(
            nx=12, side=side, young=E_ICE, poisson=NU_ICE, thickness=T_ICE,
            winkler_k=K_F_WATER, point_load=P_TEST,
        )
        r2 = solve_plate_point_load(
            nx=12, side=side, young=E_ICE, poisson=NU_ICE, thickness=T_ICE,
            winkler_k=K_F_WATER, point_load=2 * P_TEST,
        )
        ci = _find_center_node(r1.nodes, side)
        ratio = r2.displacement[3 * ci].item() / r1.displacement[3 * ci].item()
        assert abs(ratio - 2.0) < 0.01, f"Linearity ratio {ratio} != 2"

    def test_deflection_scales_with_k(self):
        """w(0) ∝ 1/√k_f。

        注意：改变 k_f 会改变特征长度，因此需要相应调整板尺寸以保持
        相同的"无限板"近似质量。
        """
        k2 = 4 * K_F_WATER
        l2 = (D_ICE / k2) ** 0.25

        # 各自使用合适的板尺寸
        side1 = 8 * L_CHAR
        side2 = 8 * l2

        r1 = solve_plate_point_load(
            nx=12, side=side1, young=E_ICE, poisson=NU_ICE, thickness=T_ICE,
            winkler_k=K_F_WATER, point_load=P_TEST,
        )
        r2 = solve_plate_point_load(
            nx=12, side=side2, young=E_ICE, poisson=NU_ICE, thickness=T_ICE,
            winkler_k=k2, point_load=P_TEST,
        )
        ci1 = _find_center_node(r1.nodes, side1)
        ci2 = _find_center_node(r2.nodes, side2)

        ratio = r1.displacement[3 * ci1].item() / r2.displacement[3 * ci2].item()
        # w ∝ 1/√k, so w1/w2 = √(k2/k1) = √4 = 2
        assert abs(ratio - 2.0) < 0.02, f"k-scaling ratio {ratio} != 2"


# ---------------------------------------------------------------------------
# 4. 挠度场轮廓 vs Hetényi 径向分布
# ---------------------------------------------------------------------------

class TestDeflectionProfile:
    """验证 FEM 挠度场沿径向的分布与 Hetényi 解一致。"""

    def test_radial_profile(self):
        """沿 x 轴的 FEM 挠度应与 Hetényi 解吻合（r ≤ 3l 内误差 < 10%）。"""
        side = 10 * L_CHAR
        nx = 12
        result = solve_plate_point_load(
            nx=nx, side=side, young=E_ICE, poisson=NU_ICE, thickness=T_ICE,
            winkler_k=K_F_WATER, point_load=P_TEST,
        )
        nodes = result.nodes
        center_x = side / 2

        # 沿中心线 (y = side/2) 取最近节点
        y_center = side / 2
        dist_y = (nodes[:, 1] - y_center).abs()
        line_mask = dist_y < (side / nx)
        line_indices = line_mask.nonzero(as_tuple=False).flatten()
        line_nodes = nodes[line_indices]

        # 排序
        order = torch.argsort(line_nodes[:, 0])
        sorted_indices = line_indices[order]
        x_vals = line_nodes[order, 0].numpy()
        w_fem_vals = result.displacement[3 * sorted_indices].numpy()

        # 与 Hetényi 比较（r ≤ 3l）
        max_rel_err = 0.0
        n_points = 0
        for x, w_fem in zip(x_vals, w_fem_vals):
            r = abs(x - center_x)
            if r <= 3 * L_CHAR:
                w_het = hetenyi_reference(r, 0.0, P_TEST, E_ICE, NU_ICE, T_ICE, K_F_WATER)
                if abs(w_het) > 1e-12:
                    rel_err = abs(w_fem - w_het) / abs(w_het)
                    max_rel_err = max(max_rel_err, rel_err)
                    n_points += 1

        assert n_points > 5, f"Too few comparison points: {n_points}"
        assert max_rel_err < 0.10, \
            f"Radial profile max error {max_rel_err:.4%} > 10% (n={n_points})"
