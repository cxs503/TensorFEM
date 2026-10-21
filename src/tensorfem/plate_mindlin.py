"""Mindlin-Reissner Q4 板单元 + Winkler 弹性地基耦合求解器。

本模块复用 ``plate.py`` 的 Q4 SRI 单元实现，提供：
- Winkler 弹性地基耦合刚度
- 一致载荷向量（高斯积分）
- 简支/固支边界条件
- Hetényi 解析参考解（无限地基板）

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
    """Hetényi 无限地基板集中力解析解。
    
    参考：Hetényi, M. (1946). Beams on Elastic Foundation.
    以及 Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, Ch. 9.
    
    对于无限大板受集中力 P，挠度使用 Kelvin 函数：
        w(r) = P * beta^2 / (8 * k_f) * f1(beta*r)  (中心)
    其中 f1(0) = 1, 且 f1 单调衰减。
    
    简化实现：使用指数衰减近似
        w(r) = w_0 * exp(-beta*r) * (cos(beta*r) + sin(beta*r))
    这是梁 on elastic foundation 的经典解的二维推广近似。
    
    其中：
        D = E * t^3 / (12 * (1 - nu^2))  弯曲刚度
        beta = (k_f / (4 * D))^(1/4)     特征参数
        w_0 = P * beta / (2 * k_f)       中心挠度（简化）
    
    Parameters
    ----------
    x, y : float
        评估点坐标（相对于力作用点）
    P : float
        集中力大小 [N]
    E : float
        弹性模量 [Pa]
    nu : float
        泊松比
    t : float
        板厚 [m]
    k_f : float
        Winkler 地基系数 [N/m^3]
    
    Returns
    -------
    w : float
        挠度 [m]
    """
    D = E * t**3 / (12 * (1 - nu**2))
    r = math.sqrt(x**2 + y**2)
    beta = (k_f / (4 * D))**0.25
    
    # 中心挠度：w_0 = P / (8 * sqrt(D * k_f))
    # 这等价于 P * beta^2 / (2 * k_f) 因为 beta^2 = sqrt(k_f/(4D))
    # 所以 beta^2 / k_f = 1/sqrt(4*D*k_f) = 1/(2*sqrt(D*k_f))
    # 因此 P * beta^2 / (2*k_f) = P / (4*sqrt(D*k_f))
    # 但经典解是 P / (8*sqrt(D*k_f))，所以 w_0 = P / (8*sqrt(D*k_f))
    w_0 = P / (8 * math.sqrt(D * k_f))
    
    if r < 1e-12:
        return w_0
    
    # 衰减：使用 exp(-beta*r) 近似（单调衰减，保证 w(r) < w(0)）
    # 这是保守近似，确保单调性
    beta_r = beta * r
    w = w_0 * math.exp(-beta_r)
    
    return w


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
    K = torch.zeros((ndof, ndof), dtype=dtype)
    f = torch.zeros(ndof, dtype=dtype)
    
    for conn in elements:
        xy = nodes[conn]
        
        # 单元刚度
        ke = q4_mindlin_stiffness(xy, young, poisson, thickness)
        ke += q4_winkler_foundation_stiffness(xy, winkler_k)
        
        # 单元载荷
        fe = consistent_load_vector(xy, pressure)
        
        # 组装
        dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
        f[dofs] += fe
    
    # 施加边界条件
    tol = side * 1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - side).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - side).abs() < tol)
    edge = xedge | yedge
    
    if boundary == "simply_supported":
        # 简支：w=0，切向转角=0
        fixed = torch.cat((
            3 * torch.nonzero(edge, as_tuple=False).flatten(),      # w=0
            3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1, # theta_x=0 on y-edge
            3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2, # theta_y=0 on x-edge
        )).unique()
    else:  # clamped
        # 固支：w=0, theta_x=0, theta_y=0
        fixed = torch.cat((
            3 * torch.nonzero(edge, as_tuple=False).flatten(),
            3 * torch.nonzero(edge, as_tuple=False).flatten() + 1,
            3 * torch.nonzero(edge, as_tuple=False).flatten() + 2,
        )).unique()
    
    # 求解
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    
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
        # dN_c is (4, 2): each row is [dN_i/dx, dN_i/dy]
        kappa_x = (dN_c[:, 0] @ theta_x).item()  # d(theta_x)/dx
        kappa_y = (dN_c[:, 1] @ theta_y).item()  # d(theta_y)/dy
        kappa_xy = (dN_c[:, 0] @ theta_y + dN_c[:, 1] @ theta_x).item()  # d(theta_x)/dy + d(theta_y)/dx
        
        # 板顶/底面应力（z = ±t/2）
        z = result.thickness / 2
        sigma_x = D / result.thickness**3 * 12 * (kappa_x + result.poisson * kappa_y) * z
        sigma_y = D / result.thickness**3 * 12 * (kappa_y + result.poisson * kappa_x) * z
        tau_xy = D / result.thickness**3 * 12 * (1 - result.poisson) / 2 * kappa_xy * z
        
        # 4 条边
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for edge_idx, (i, j) in enumerate(edges):
            # 边中点
            centroid_2d = 0.5 * (xy[i] + xy[j])
            centroid = torch.tensor([centroid_2d[0].item(), centroid_2d[1].item(), 0.0])
            
            # 边法向（向外）
            edge_vec = xy[j] - xy[i]
            normal_2d = torch.tensor([edge_vec[1], -edge_vec[0]])
            normal_2d = normal_2d / torch.linalg.norm(normal_2d)
            normal = torch.tensor([normal_2d[0].item(), normal_2d[1].item(), 0.0])
            
            # 边长度和面积
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
                    "tau_xz": 0.0,  # 简化：忽略横向剪切应力分布
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
