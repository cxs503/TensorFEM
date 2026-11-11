"""Mindlin-Reissner Q4 板单元 + Winkler 弹性地基耦合求解器。

本模块复用 ``plate.py`` 的 Q4 SRI 单元实现，提供：
- Winkler 弹性地基耦合刚度
- 一致载荷向量（高斯积分）
- 简支/固支边界条件
- Hetényi 解析参考解（无限地基板）
- 集中力点载荷求解器（用于与 Hetényi 解析解对标验证）

API 面向外部调用：挠度场、单元应力场、内 face 应力。

取舍说明：
- 复用 ``plate.q4_mindlin_stiffness`` 避免重复实现单元刚度
- 新增 Winkler 地基刚度和求解器封装
- Hetényi 参考解用于库级验证（SI-1 口径复现）
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .hetenyi_reference import infinite_plate_point_load
from .plate import q4_mindlin_stiffness, _shape, structured_square_mesh


@dataclass(frozen=True)
class MindlinPlateResult:
    """Mindlin 板求解结果。"""
    displacement: torch.Tensor  # (ndof,) 节点位移 [w, theta_x, theta_y]
    nodes: torch.Tensor         # (n_nodes, 2)
    elements: torch.Tensor      # (n_elements, 4)
    young: float
    poisson: float
    thickness: float
    winkler_k: float


def q4_winkler_foundation_stiffness(
    xy: torch.Tensor,
    winkler_k: float,
) -> torch.Tensor:
    """Q4 Winkler 弹性地基刚度矩阵 (仅 w 自由度耦合)。
    
    Parameters
    ----------
    xy : (4, 2) tensor
        单元节点坐标
    winkler_k : float
        Winkler 地基系数 [N/m^3]
    
    Returns
    -------
    ke_winkler : (12, 12) tensor
        地基刚度矩阵，仅 3*i 行/列 (w 自由度) 非零
    """
    dtype, device = xy.dtype, xy.device
    ke = torch.zeros((12, 12), dtype=dtype, device=device)
    
    # 2x2 Gauss 积分
    gp = 1.0 / 3.0**0.5
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            xi = xy.new_tensor(xif)
            eta = xy.new_tensor(etaf)
            N, dN_nat = _shape(xi, eta)
            J = dN_nat.T @ xy
            detJ = torch.linalg.det(J)
            
            # 仅 w 自由度耦合: ke_w[i,j] = k_f * N_i * N_j * detJ
            for i in range(4):
                for j in range(4):
                    ke[3*i, 3*j] += winkler_k * N[i] * N[j] * detJ
    
    return ke


def consistent_load_vector(
    xy: torch.Tensor,
    pressure: torch.Tensor | float | callable,
) -> torch.Tensor:
    """一致载荷向量（高斯积分）。
    
    Parameters
    ----------
    xy : (4, 2) tensor
        单元节点坐标
    pressure : float, tensor, or callable
        压力载荷。若为 callable，签名 pressure(x, y) -> scalar
    
    Returns
    -------
    fe : (12,) tensor
        单元载荷向量，仅 w 自由度非零
    """
    dtype, device = xy.dtype, xy.device
    fe = torch.zeros(12, dtype=dtype, device=device)
    
    gp = 1.0 / 3.0**0.5
    for xif in (-gp, gp):
        for etaf in (-gp, gp):
            xi = xy.new_tensor(xif)
            eta = xy.new_tensor(etaf)
            N, dN_nat = _shape(xi, eta)
            J = dN_nat.T @ xy
            detJ = torch.linalg.det(J)
            
            # 物理坐标
            p = N @ xy
            x, y = p[0], p[1]
            
            # 压力值
            if callable(pressure):
                q = pressure(x, y)
                if not torch.is_tensor(q):
                    q = xy.new_tensor(q)
            else:
                q = torch.as_tensor(pressure, dtype=dtype, device=device)
            
            # 一致载荷: fe_i = N_i * q * detJ
            for i in range(4):
                fe[3*i] += N[i] * q * detJ
    
    return fe


def hetenyi_reference(
    x: float,
    y: float,
    P: float,
    E: float,
    nu: float,
    t: float,
    k_f: float,
) -> float:
    """无限 Winkler 地基板集中力精确解（Kelvin kei 函数）。

    控制方程（轴对称）：``D∇⁴w + k_f·w = P·δ(r)/(2πr)``，
    ``D = E t³/(12(1-ν²))``，特征长度 ``l = (D/k_f)^(1/4)``：

        w(r) = -P/(2π·k_f·l²) · kei(r/l),   w(0) = P/(8·√(D·k_f))

    参考：Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, §62；
    Hetényi (1946)。数值实现见 :func:`tensorfem.hetenyi_reference.kelvin_kei`。

    注意点（审计结论，2026-11-02）
    ------------------------------
    本函数此前返回 ``w0·exp(-βr)``——一个自行声明的「简化指数衰减近似」，
    既不是 Kelvin 解，也不是 kei 的任何渐近形式：它在 r≈1.2·l 处与精确解
    偏差 >80%，且丢失了精确解在载荷外围的抬升环（w 变号）。现改为精确解，
    校验见 ``tests/test_foundation_reference_audit.py`` 与
    ``docs/hetanyi-reference-audit.md``。

    Parameters
    ----------
    x, y : float
        评估点坐标（相对于力作用点，单位 m）。
    P : float
        集中力大小 [N]（向下为正）。
    E : float
        弹性模量 [Pa]。
    nu : float
        泊松比。
    t : float
        板厚 [m]。
    k_f : float
        Winkler 地基系数 [N/m³]。

    Returns
    -------
    w : float
        挠度 [m]（向下为正）。
    """
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    r = math.sqrt(x * x + y * y)
    return float(infinite_plate_point_load(torch.tensor([r], dtype=torch.float64), P, D, k_f)[0])


def _apply_boundary_conditions(
    nodes: torch.Tensor,
    side: float,
    ndof: int,
    boundary: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (fixed_dofs, free_dofs)。"""
    tol = side * 1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - side).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - side).abs() < tol)
    edge = xedge | yedge
    
    if boundary == "simply_supported":
        fixed = torch.cat((
            3 * torch.nonzero(edge, as_tuple=False).flatten(),
            3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1,
            3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2,
        )).unique()
    elif boundary == "clamped":
        fixed = torch.cat((
            3 * torch.nonzero(edge, as_tuple=False).flatten(),
            3 * torch.nonzero(edge, as_tuple=False).flatten() + 1,
            3 * torch.nonzero(edge, as_tuple=False).flatten() + 2,
        )).unique()
    else:
        raise ValueError("boundary must be 'simply_supported' or 'clamped'")
    
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    return fixed, free


def _assemble_plate_stiffness(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young: float,
    poisson: float,
    thickness: float,
    winkler_k: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """组装板刚度矩阵（弯曲 + Winkler 地基）。"""
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes
    K = torch.zeros((ndof, ndof), dtype=dtype)
    
    for conn in elements:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, young, poisson, thickness)
        ke += q4_winkler_foundation_stiffness(xy, winkler_k)
        dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
    
    return K


def solve_plate_on_winkler_foundation(
    nx: int,
    side: float,
    young: float,
    poisson: float,
    thickness: float,
    winkler_k: float,
    pressure: float | callable,
    boundary: str = "simply_supported",
    dtype=torch.float64,
) -> MindlinPlateResult:
    """求解 Winkler 弹性地基上的 Mindlin 板。
    
    Parameters
    ----------
    nx : int
        每边单元数
    side : float
        板边长 [m]
    young : float
        弹性模量 [Pa]
    poisson : float
        泊松比
    thickness : float
        板厚 [m]
    winkler_k : float
        Winkler 地基系数 [N/m^3]
    pressure : float or callable
        压力载荷（均匀压力或 pressure(x,y) 函数）
    boundary : str
        边界条件："simply_supported" 或 "clamped"
    dtype : torch.dtype
        数据类型
    
    Returns
    -------
    result : MindlinPlateResult
        求解结果
    """
    if boundary not in ("simply_supported", "clamped"):
        raise ValueError("boundary must be 'simply_supported' or 'clamped'")
    
    nodes, elements = structured_square_mesh(nx, side, dtype=dtype)
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes
    
    # 组装全局刚度和载荷
    K = _assemble_plate_stiffness(nodes, elements, young, poisson, thickness, winkler_k, dtype)
    f = torch.zeros(ndof, dtype=dtype)
    
    for conn in elements:
        xy = nodes[conn]
        fe = consistent_load_vector(xy, pressure)
        dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
        f[dofs] += fe
    
    # 施加边界条件
    _, free = _apply_boundary_conditions(nodes, side, ndof, boundary)
    
    K_ff = K[free][:, free]
    f_f = f[free]
    
    u_f = torch.linalg.solve(K_ff, f_f)
    
    u = torch.zeros(ndof, dtype=dtype)
    u[free] = u_f
    
    return MindlinPlateResult(
        displacement=u,
        nodes=nodes,
        elements=elements,
        young=young,
        poisson=poisson,
        thickness=thickness,
        winkler_k=winkler_k,
    )


def solve_plate_point_load(
    nx: int,
    side: float,
    young: float,
    poisson: float,
    thickness: float,
    winkler_k: float,
    point_load: float,
    load_x: Optional[float] = None,
    load_y: Optional[float] = None,
    boundary: str = "simply_supported",
    dtype=torch.float64,
) -> MindlinPlateResult:
    """求解 Winkler 弹性地基上受集中力的 Mindlin 板。

    用于与 Hetényi 无限板解析解对标验证。当板尺寸 ≥ 8 倍特征长度时，
    中心挠度与 Hetényi 解误差 < 1%。

    Parameters
    ----------
    nx : int
        每边单元数
    side : float
        板边长 [m]
    young : float
        弹性模量 [Pa]
    poisson : float
        泊松比
    thickness : float
        板厚 [m]
    winkler_k : float
        Winkler 地基系数 [N/m^3]
    point_load : float
        集中力大小 [N]（向下为正）
    load_x, load_y : float, optional
        力作用点坐标，默认板中心 (side/2, side/2)
    boundary : str
        边界条件："simply_supported" 或 "clamped"
    dtype : torch.dtype
        数据类型

    Returns
    -------
    result : MindlinPlateResult
        求解结果
    """
    if boundary not in ("simply_supported", "clamped"):
        raise ValueError("boundary must be 'simply_supported' or 'clamped'")
    
    nodes, elements = structured_square_mesh(nx, side, dtype=dtype)
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes
    
    # 组装刚度
    K = _assemble_plate_stiffness(nodes, elements, young, poisson, thickness, winkler_k, dtype)
    f = torch.zeros(ndof, dtype=dtype)
    
    # 集中力施加到最近节点
    if load_x is None:
        load_x = side / 2
    if load_y is None:
        load_y = side / 2
    
    dist = (nodes[:, 0] - load_x)**2 + (nodes[:, 1] - load_y)**2
    load_node = torch.argmin(dist).item()
    f[3 * load_node] = point_load
    
    # 边界条件
    _, free = _apply_boundary_conditions(nodes, side, ndof, boundary)
    
    K_ff = K[free][:, free]
    f_f = f[free]
    
    u_f = torch.linalg.solve(K_ff, f_f)
    
    u = torch.zeros(ndof, dtype=dtype)
    u[free] = u_f
    
    return MindlinPlateResult(
        displacement=u,
        nodes=nodes,
        elements=elements,
        young=young,
        poisson=poisson,
        thickness=thickness,
        winkler_k=winkler_k,
    )


def compute_face_stresses(
    result: MindlinPlateResult,
) -> list[dict]:
    """计算单元内 face 应力。
    
    对于每个单元的 4 条边，计算中点处的应力状态。
    
    Returns
    -------
    faces : list of dict
        每个 face 包含：
        - id: int
        - element: int
        - edge: int (0-3)
        - centroid: (3,) array [x, y, z=0]
        - normal: (3,) array
        - area: float
        - stress: dict with 'sigma_x', 'sigma_y', 'tau_xy', 'tau_xz', 'tau_yz'
    """
    faces = []
    face_id = 0
    
    D = result.young * result.thickness**3 / (12 * (1 - result.poisson**2))
    
    for elem_idx, conn in enumerate(result.elements):
        xy = result.nodes[conn]
        
        # 单元中心
        xi_c = xy.new_tensor(0.0)
        eta_c = xy.new_tensor(0.0)
        _, dN_nat_c = _shape(xi_c, eta_c)
        J_c = dN_nat_c.T @ xy
        dN_c = dN_nat_c @ torch.linalg.inv(J_c).T
        
        # 提取单元位移
        u_elem = result.displacement[torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)]
        w = u_elem[0::3]
        theta_x = u_elem[1::3]
        theta_y = u_elem[2::3]
        
        # 计算曲率
        kappa_x = (dN_c[:, 0] @ theta_x).item()
        kappa_y = (dN_c[:, 1] @ theta_y).item()
        kappa_xy = (dN_c[:, 0] @ theta_y + dN_c[:, 1] @ theta_x).item()
        
        # 板顶/底面应力（z = ±t/2）
        z = result.thickness / 2
        sigma_x = D / result.thickness**3 * 12 * (kappa_x + result.poisson * kappa_y) * z
        sigma_y = D / result.thickness**3 * 12 * (kappa_y + result.poisson * kappa_x) * z
        tau_xy = D / result.thickness**3 * 12 * (1 - result.poisson) / 2 * kappa_xy * z
        
        # 4 条边
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for edge_idx, (i, j) in enumerate(edges):
            centroid_2d = 0.5 * (xy[i] + xy[j])
            centroid = torch.tensor([centroid_2d[0].item(), centroid_2d[1].item(), 0.0])
            
            edge_vec = xy[j] - xy[i]
            normal_2d = torch.tensor([edge_vec[1], -edge_vec[0]])
            normal_2d = normal_2d / torch.linalg.norm(normal_2d)
            normal = torch.tensor([normal_2d[0].item(), normal_2d[1].item(), 0.0])
            
            length = torch.linalg.norm(edge_vec).item()
            area = length * result.thickness
            
            faces.append({
                "id": face_id,
                "element": elem_idx,
                "edge": edge_idx,
                "centroid": centroid.tolist(),
                "normal": normal.tolist(),
                "area": area,
                "stress": {
                    "sigma_x": sigma_x,
                    "sigma_y": sigma_y,
                    "tau_xy": tau_xy,
                    "tau_xz": 0.0,
                    "tau_yz": 0.0,
                }
            })
            face_id += 1
    
    return faces


def export_failure_faces_json(faces: list[dict]) -> dict:
    """导出失效面清单为 JSON 格式。
    
    接口冻结（上下游契约）：
    failure_faces.json = {
        "faces": [
            {
                "id": int,
                "area": float,
                "normal": [x, y, z],
                "centroid": [x, y, z],
                "G": float,
                "released_energy": float
            }
        ],
        "units": "SI"
    }
    """
    return {
        "faces": [
            {
                "id": f["id"],
                "area": f["area"],
                "normal": f["normal"],
                "centroid": f["centroid"],
                "G": f.get("G", 0.0),
                "released_energy": f.get("released_energy", 0.0),
            }
            for f in faces
        ],
        "units": "SI",
    }
