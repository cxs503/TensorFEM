"""Broken-ice floe field: buoyant rigid-floe dynamics, contact and channel clearing.

锚点 H1-BROKEN-ICE-DEM（碎冰场 DEM 动力学）在 TensorFEM 侧的验证级实现面。

本模块给出碎冰场（broken / brash ice）的**参考级刚体浮冰块动力学**：

* 垂直向：每块浮冰按阿基米德浮力平衡求解吃水/干舷，非线性浮力
  ``F_b = rho_w * g * A * clip(h/2 - z, 0, h)``（z 为块心相对静水面的高度），
  求解线性化回复刚度 ``k = rho_w * g * A`` 与垂荡固有频率
  ``omega = sqrt(rho_w * g / (rho_i * h))``；
* 水平向：圆盘刚体之间的脉冲法接触（带回弹性系数 restitution，可选库仑摩擦）、
  与刚性船首楔形体（两条斜边 + 平行中体）的接触，均按无限质量刚体处理；
* 航道清空：船首楔形体以定速 V 扫过碎冰场，统计航道走廊内的冰面覆盖率下降、
  块心侧向清空率、以及被推向前方（"推土"）的比例。

定位（诚实声明）
----------------
1. 这是**刚体圆盘 + 脉冲接触**的参考实现，不是生产级 DEM：多面体/团簇、滚动阻力、
   粘结破坏、GPU 大规模并行等属于 TensorDEM 仓的 ``broken_ice_micro`` /
   ``brash_ice_rheology`` / ``buoyancy_verification`` 等模块。本模块的用途是
   （a）提供可机器裁剪的解析基准（阿基米德浮态、垂荡频率、回复刚度、楔形稳态
   侧向速度），供任何 DEM 实现（含 TensorDEM）对表；（b）量化"清空率"这一工程统计量
   在**无标定数据**时的模型行为与敏感性。
2. 浮冰只保留垂荡一个竖向自由度（无横摇/纵摇，圆盘水平面惯性矩对称，倾覆不予
   描述）；冰面覆盖率用走廊网格采样定义。接触摩擦默认 ``friction = 0.0``
   （无摩擦），需要时显式开启并在报告中声明。
3. 航道清空率**无仓内实测数据集**可对标：本模块如实输出清空率随航速的敏感性与
   单调性，不声称与实测统计的定量一致（见 docs 内误差判定节）。

参考
----
* Archimedes 浮态：任意流体静力学教材；海冰密度 900-920 kg/m³ vs 海水 1025 kg/m³。
* 无限大质量刚性斜面以定速 V 推进时，无摩擦单质点的稳态速度
  ``v = (V . n) n``，n 为斜面外法向；对半顶角 ``alpha`` 的楔边
  ``n = (cos alpha, sin alpha)``，故稳态侧向速度 ``v_x = V sin alpha cos alpha``。
  本模块以此作为接触子模型的定量真值。
* 垂荡固有频率：``omega^2 = rho_w g A / (rho_i h A) = rho_w g / (rho_i h)``。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

D = torch.float64


# ---------------------------------------------------------------------------
# 解析基准（浮态 / 刚度 / 频率）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchimedesState:
    """Single floe hydrostatic equilibrium (flat, level floe)."""

    thickness: float
    ice_density: float
    water_density: float
    draft: float                 # 吃水 (m)
    freeboard: float             # 干舷 (m)
    submerged_fraction: float    # 吃水/厚度 = rho_i / rho_w
    center_elevation: float      # 块心相对静水面高度 (m)，负值表示块心在水下


def archimedes_equilibrium(
    thickness: float,
    ice_density: float = 900.0,
    water_density: float = 1025.0,
) -> ArchimedesState:
    """Flat-floe floatation equilibrium: ``draft = h * rho_i / rho_w``.

    The freeboard follows from mass balance alone (no calibration): a level floe
    displaces its own weight, ``rho_w * draft = rho_i * h``.
    """
    for name, value in (
        ("thickness", thickness),
        ("ice_density", ice_density),
        ("water_density", water_density),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if ice_density >= water_density:
        raise ValueError(
            "floating equilibrium requires ice_density < water_density "
            f"(got {ice_density} >= {water_density})"
        )
    ratio = ice_density / water_density
    draft = thickness * ratio
    return ArchimedesState(
        thickness=thickness,
        ice_density=ice_density,
        water_density=water_density,
        draft=draft,
        freeboard=thickness - draft,
        submerged_fraction=ratio,
        center_elevation=0.5 * thickness - draft,
    )


def hydrostatic_restoring_stiffness(
    waterplane_area: float,
    water_density: float = 1025.0,
    gravity: float = 9.81,
) -> float:
    """Linear heave restoring stiffness ``k = rho_w * g * A`` (N/m)."""
    if not math.isfinite(waterplane_area) or waterplane_area <= 0.0:
        raise ValueError("waterplane_area must be finite and positive")
    if not math.isfinite(water_density) or water_density <= 0.0:
        raise ValueError("water_density must be finite and positive")
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive")
    return water_density * gravity * waterplane_area


def heave_natural_frequency(
    thickness: float,
    ice_density: float = 900.0,
    water_density: float = 1025.0,
    gravity: float = 9.81,
) -> float:
    """Undamped heave natural frequency ``omega = sqrt(rho_w g / (rho_i h))`` (rad/s)."""
    equilibrium = archimedes_equilibrium(thickness, ice_density, water_density)
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive")
    return math.sqrt(water_density * gravity / (equilibrium.ice_density * thickness))


def floe_mass(radius: float, thickness: float, ice_density: float = 900.0) -> float:
    """Mass of a cylindrical floe ``m = rho_i * pi * r^2 * h`` (kg)."""
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness must be finite and positive")
    if not math.isfinite(ice_density) or ice_density <= 0.0:
        raise ValueError("ice_density must be finite and positive")
    return ice_density * math.pi * radius * radius * thickness


# ---------------------------------------------------------------------------
# 场配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloeFieldConfig:
    """Broken-ice field + structure configuration (SI units throughout)."""

    n_floes: int = 80
    radius_range: tuple[float, float] = (1.5, 3.0)
    thickness_range: tuple[float, float] = (0.4, 1.0)
    ice_density: float = 900.0
    water_density: float = 1025.0
    gravity: float = 9.81
    restitution: float = 0.2
    friction: float = 0.0
    contact_iterations: int = 2
    heave_damping_ratio: float = 0.7
    drag_time: float | None = 8.0
    dt: float = 0.01
    structure_speed: float = 1.5
    bow_half_angle_deg: float = 30.0
    bow_length: float = 8.0
    bow_half_thickness: float = 0.1
    parallel_body_length: float = 45.0
    channel_half_width: float | None = None
    field_length: float = 60.0
    field_half_width: float = 25.0
    coverage_cell: float = 0.25
    seed: int = 20261113

    def validate(self) -> None:
        if isinstance(self.n_floes, bool) or not isinstance(self.n_floes, int) or self.n_floes <= 0:
            raise ValueError("n_floes must be a positive integer")
        for name in ("radius_range", "thickness_range"):
            band = getattr(self, name)
            if len(band) != 2 or not all(math.isfinite(v) and v > 0.0 for v in band):
                raise ValueError(f"{name} must be two finite positive bounds")
            if band[0] > band[1]:
                raise ValueError(f"{name} lower bound exceeds upper bound")
        if not math.isfinite(self.ice_density) or self.ice_density <= 0.0:
            raise ValueError("ice_density must be finite and positive")
        if not math.isfinite(self.water_density) or self.water_density <= 0.0:
            raise ValueError("water_density must be finite and positive")
        if self.ice_density >= self.water_density:
            raise ValueError("floating ice requires ice_density < water_density")
        if not math.isfinite(self.gravity) or self.gravity <= 0.0:
            raise ValueError("gravity must be finite and positive")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must lie in [0, 1]")
        if self.friction < 0.0 or not math.isfinite(self.friction):
            raise ValueError("friction must be finite and nonnegative")
        if (
            isinstance(self.contact_iterations, bool)
            or not isinstance(self.contact_iterations, int)
            or self.contact_iterations < 1
        ):
            raise ValueError("contact_iterations must be a positive integer")
        if not 0.0 <= self.heave_damping_ratio < 1.0:
            raise ValueError("heave_damping_ratio must lie in [0, 1)")
        if self.drag_time is not None and (
            not math.isfinite(self.drag_time) or self.drag_time <= 0.0
        ):
            raise ValueError("drag_time must be None or a finite positive number")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(self.structure_speed) or self.structure_speed <= 0.0:
            raise ValueError("structure_speed must be finite and positive")
        if not 0.0 < self.bow_half_angle_deg < 90.0:
            raise ValueError("bow_half_angle_deg must lie in (0, 90)")
        for name in (
            "bow_length",
            "bow_half_thickness",
            "parallel_body_length",
            "field_length",
            "field_half_width",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        # 冰厚上限必须小于水深无关量；垂荡稳定要求 omega*dt 远小于 1
        omega_max = heave_natural_frequency(
            self.thickness_range[0], self.ice_density, self.water_density, self.gravity
        )
        if omega_max * self.dt > 0.1:
            raise ValueError("dt too large for the stiffest floe heave frequency")

    # -- derived geometry ---------------------------------------------------
    @property
    def bow_half_width(self) -> float:
        return self.bow_length * math.tan(math.radians(self.bow_half_angle_deg))

    @property
    def effective_channel_half_width(self) -> float:
        if self.channel_half_width is not None:
            return float(self.channel_half_width)
        return self.bow_half_width

    @property
    def sweep_time(self) -> float:
        """Time for the apex to travel from the field entry to the field exit."""
        return (self.field_length + 2.0 * self.bow_length) / self.structure_speed


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass
class FloeFieldResult:
    """State history summary of a floe-field sweep."""

    config: FloeFieldConfig
    positions0: torch.Tensor          # (N, 2)
    positions: torch.Tensor           # (N, 2)
    radii: torch.Tensor               # (N,)
    thickness: torch.Tensor           # (N,)
    heave: torch.Tensor               # (N,) 块心相对静水面高度 (m)
    heave_velocity: torch.Tensor      # (N,)
    velocity: torch.Tensor            # (N, 2)
    t_end: float
    max_penetration: float
    max_relative_speed: float
    heave_history: torch.Tensor | None = None
    corridor: dict = field(default_factory=dict)
    diagnostic: dict = field(default_factory=dict)


class _single_thread_intraop:
    """Pin torch intra-op threads to 1 for the integration loop.

    The per-step tensors are tiny (N_floes^2 contact pairs at most a few
    thousand entries), so the OpenMP/threadpool dispatch overhead dominates the
    arithmetic: on a 128-thread node the 80-floe sweep runs ~10x faster with a
    single intra-op thread.  The previous setting is always restored.
    """

    def __enter__(self) -> "_single_thread_intraop":
        self._previous = torch.get_num_threads()
        if self._previous > 1:
            torch.set_num_threads(1)
        return self

    def __exit__(self, *exc_info) -> None:
        torch.set_num_threads(self._previous)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _place_floes(
    config: FloeFieldConfig, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-overlapping random floe placement (random sequential adsorption).

    Sampling is done in plain Python floats against a spatial hash keyed on
    ``2 * radius_range[1]`` cells: a scalar ``torch`` call per candidate would
    dominate the runtime of the whole study, and the placement has to be
    bit-reproducible for a given seed (it is — the generator draw order is
    fixed).
    """
    radii = (
        torch.rand(config.n_floes, generator=generator, dtype=D)
        * (config.radius_range[1] - config.radius_range[0])
        + config.radius_range[0]
    )
    thickness = (
        torch.rand(config.n_floes, generator=generator, dtype=D)
        * (config.thickness_range[1] - config.thickness_range[0])
        + config.thickness_range[0]
    )
    radii_list = [float(v) for v in radii]
    gap = 0.1                       # 初始最小间隙 (m)
    cell = 2.0 * config.radius_range[1] + gap
    buckets: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    coords: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = 500 * config.n_floes
    u = torch.rand(2 * config.n_floes * 8, generator=generator, dtype=D).tolist()
    cursor = 0

    def next_uniform() -> float:
        nonlocal cursor
        if cursor >= len(u):
            u.extend(torch.rand(4096, generator=generator, dtype=D).tolist())
        cursor += 1
        return u[cursor - 1]

    for k in range(config.n_floes):
        while True:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    "cannot place the requested floes without overlap; "
                    "reduce n_floes / radius_range or enlarge the field"
                )
            x = next_uniform() * 2.0 * config.field_half_width - config.field_half_width
            y = next_uniform() * config.field_length
            rk = radii_list[k]
            gx = int(math.floor(x / cell))
            gy = int(math.floor(y / cell))
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for ox, oy, other_radius in buckets.get((gx + dx, gy + dy), ()):
                        lim = rk + other_radius + gap
                        if (x - ox) ** 2 + (y - oy) ** 2 < lim * lim:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                coords.append((x, y))
                buckets.setdefault((gx, gy), []).append((x, y, rk))
                break
    positions = torch.tensor(coords, dtype=D)
    return positions, radii, thickness


def _bow_segments(apex_y: float, config: FloeFieldConfig) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Rigid bow: two divergent faces plus two straight parallel-body sides."""
    half_width = config.bow_half_width
    aft_y = apex_y - config.bow_length
    side_end_y = aft_y - config.parallel_body_length
    segments = []
    for side in (-1.0, 1.0):
        segments.append(
            (
                torch.tensor([0.0, apex_y], dtype=D),
                torch.tensor([side * half_width, aft_y], dtype=D),
            )
        )
        segments.append(
            (
                torch.tensor([side * half_width, aft_y], dtype=D),
                torch.tensor([side * half_width, side_end_y], dtype=D),
            )
        )
    return segments


def _closest_point_on_segment(
    points: torch.Tensor, start: torch.Tensor, end: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised closest point on a segment for a batch of points (N, 2)."""
    seg = end - start
    seg_len2 = float(torch.dot(seg, seg))
    rel = points - start
    t = (rel @ seg) / seg_len2
    t = torch.clamp(t, 0.0, 1.0)
    return start + t.unsqueeze(1) * seg, t


def _apply_pair_contact(
    positions: torch.Tensor,
    velocity: torch.Tensor,
    inv_mass: torch.Tensor,
    radii: torch.Tensor,
    restitution: float,
    friction: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Impulse-based floe-floe contact (one Jacobi pass).

    Returns updated (positions, velocity, max_penetration, max_closing_speed).
    """
    n = positions.shape[0]
    if n < 2:
        return positions, velocity, 0.0, 0.0
    delta = positions.unsqueeze(0) - positions.unsqueeze(1)          # (i, j, 2), i->j is -delta
    dist = torch.linalg.norm(delta, dim=2)
    contact_radius = radii.unsqueeze(0) + radii.unsqueeze(1)
    eye = torch.eye(n, dtype=torch.bool)
    overlapping = (dist < contact_radius) & (~eye)
    penetration = torch.where(overlapping, contact_radius - dist, torch.zeros_like(dist))
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    active = overlapping & upper
    max_penetration = float(penetration.max()) if bool(overlapping.any()) else 0.0
    if not bool(active.any()):
        return positions, velocity, max_penetration, 0.0

    idx = torch.nonzero(active, as_tuple=False)                       # (K, 2) = (i, j)
    i = idx[:, 0]
    j = idx[:, 1]
    gap = positions[j] - positions[i]
    gap_dist = torch.linalg.norm(gap, dim=1)
    safe_dist = torch.clamp(gap_dist, min=1e-12)
    normal = gap / safe_dist.unsqueeze(1)                            # i -> j
    pen = contact_radius[i, j] - gap_dist
    closing = (velocity[i] - velocity[j]) * normal
    closing_speed = closing.sum(dim=1)
    # closing_speed > 0 表示相互接近，取正向最大值作为该步最大接近速度
    max_closing = float(torch.clamp(closing_speed, min=0.0).max())

    inv_sum = inv_mass[i] + inv_mass[j]
    # 法向冲量：closing_speed > 0 为接近，jn = (1+e) * closing / (1/m_i + 1/m_j) >= 0
    jn = torch.clamp((1.0 + restitution) * closing_speed, min=0.0) / inv_sum

    dv = jn.unsqueeze(1) * normal
    velocity = velocity.index_add(0, i, -dv * inv_mass[i].unsqueeze(1))
    velocity = velocity.index_add(0, j, dv * inv_mass[j].unsqueeze(1))

    if friction > 0.0:
        vt = (velocity[i] - velocity[j]) - (
            ((velocity[i] - velocity[j]) * normal).sum(dim=1, keepdim=True) * normal
        )
        vt_mag = torch.linalg.norm(vt, dim=1)
        tangent = torch.where(
            vt_mag.unsqueeze(1) > 1e-12, vt / torch.clamp(vt_mag, min=1e-12).unsqueeze(1),
            torch.zeros_like(vt),
        )
        jt = torch.clamp(vt_mag, max=friction * jn) / inv_sum
        dvt = jt.unsqueeze(1) * tangent
        velocity = velocity.index_add(0, i, -dvt * inv_mass[i].unsqueeze(1))
        velocity = velocity.index_add(0, j, dvt * inv_mass[j].unsqueeze(1))

    # 位置修正：按逆质量比例分离，消除重叠（Baumgarte 式直接投影）
    share = inv_mass[i] / inv_sum
    correction = pen.unsqueeze(1) * normal
    positions = positions.index_add(0, i, -correction * share.unsqueeze(1))
    positions = positions.index_add(0, j, correction * (1.0 - share).unsqueeze(1))
    return positions, velocity, max_penetration, max_closing


def _apply_structure_contact(
    positions: torch.Tensor,
    velocity: torch.Tensor,
    inv_mass: torch.Tensor,
    radii: torch.Tensor,
    segments: Sequence[tuple[torch.Tensor, torch.Tensor]],
    structure_velocity: torch.Tensor,
    restitution: float,
    friction: float,
    half_thickness: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Floe vs infinite-mass moving segments (bow faces + parallel body)."""
    max_penetration = 0.0
    limit = radii + half_thickness
    for start, end in segments:
        closest, _ = _closest_point_on_segment(positions, start, end)
        gap = positions - closest
        dist = torch.linalg.norm(gap, dim=1)
        dist = torch.clamp(dist, min=1e-12)
        pen = limit - dist
        active = pen > 0.0
        if not bool(active.any()):
            continue
        max_penetration = max(max_penetration, float(pen[active].max()))
        normal = gap / dist.unsqueeze(1)                              # 段 -> 冰
        rel = velocity - structure_velocity.unsqueeze(0)
        vn = (rel * normal).sum(dim=1)
        # 无限质量刚体：单块冲量经自身质量除回，故 Δv = -(1+e) v_n n
        dvn = torch.where(active & (vn < 0.0), -(1.0 + restitution) * vn, torch.zeros_like(vn))
        velocity = velocity + dvn.unsqueeze(1) * normal
        if friction > 0.0:
            rel2 = velocity - structure_velocity.unsqueeze(0)
            vt = rel2 - (rel2 * normal).sum(dim=1, keepdim=True) * normal
            vt_mag = torch.linalg.norm(vt, dim=1)
            tangent = torch.where(
                vt_mag.unsqueeze(1) > 1e-12, vt / torch.clamp(vt_mag, min=1e-12).unsqueeze(1),
                torch.zeros_like(vt),
            )
            jt = torch.clamp(vt_mag, max=friction * dvn.abs())
            velocity = velocity - jt.unsqueeze(1) * tangent
        push = torch.where(active, pen, torch.zeros_like(pen))
        positions = positions + normal * push.unsqueeze(1)
    return positions, velocity, max_penetration


def corridor_coverage(
    positions: torch.Tensor,
    radii: torch.Tensor,
    half_width: float,
    y_range: tuple[float, float],
    cell: float = 0.25,
) -> float:
    """Footprint coverage of the strip ``|x| <= half_width`` (grid sampling)."""
    nx = max(int(math.ceil(2.0 * half_width / cell)), 1)
    ny = max(int(math.ceil((y_range[1] - y_range[0]) / cell)), 1)
    xs = (
        torch.arange(nx, dtype=D) * cell
        - half_width
        + 0.5 * cell
    )
    ys = torch.arange(ny, dtype=D) * cell + y_range[0] + 0.5 * cell
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    covered = torch.zeros(grid.shape[0], dtype=torch.bool)
    for center, radius in zip(positions, radii):
        d2 = ((grid - center) ** 2).sum(dim=1)
        covered |= d2 <= radius * radius
    return float(covered.to(D).mean())


# ---------------------------------------------------------------------------
# 主模拟
# ---------------------------------------------------------------------------


def simulate_floe_field(
    config: FloeFieldConfig,
    *,
    initial_heave_offset: float = 0.0,
    initial_heave: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    radii: torch.Tensor | None = None,
    thickness: torch.Tensor | None = None,
    t_end: float | None = None,
    structure: bool = True,
    record_history: bool = False,
) -> FloeFieldResult:
    """Integrate the buoyant floe field (single intra-op thread, see below).

    Vertical: symplectic Euler on the nonlinear buoyancy plus added damping.
    Horizontal: impulse-based contact against other floes and the sweeping bow.

    The integration runs with the torch intra-op thread pool pinned to one
    thread (``_single_thread_intraop``) and restores the previous setting on
    exit: per-step tensors are tiny, so the pool dispatch overhead dominates.
    """
    with _single_thread_intraop():
        return _simulate_floe_field(
            config,
            initial_heave_offset=initial_heave_offset,
            initial_heave=initial_heave,
            positions=positions,
            radii=radii,
            thickness=thickness,
            t_end=t_end,
            structure=structure,
            record_history=record_history,
        )


def _simulate_floe_field(
    config: FloeFieldConfig,
    *,
    initial_heave_offset: float = 0.0,
    initial_heave: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    radii: torch.Tensor | None = None,
    thickness: torch.Tensor | None = None,
    t_end: float | None = None,
    structure: bool = True,
    record_history: bool = False,
) -> FloeFieldResult:
    """Integrate the buoyant floe field with the sweeping bow.

    Vertical: symplectic Euler on the nonlinear buoyancy + added damping.
    Horizontal: impulse-based contact against other floes and the bow.
    """
    config.validate()
    generator = torch.Generator().manual_seed(config.seed)
    if positions is None:
        positions, radii, thickness = _place_floes(config, generator)
    else:
        if radii is None or thickness is None:
            raise ValueError("radii and thickness must accompany explicit positions")
    assert radii is not None and thickness is not None
    n = positions.shape[0]
    radii = radii.to(D).clone()
    thickness = thickness.to(D).clone()
    positions = positions.to(D).clone()

    areas = math.pi * radii * radii
    masses = config.ice_density * areas * thickness
    inv_mass = 1.0 / masses
    omega_sq = config.water_density * config.gravity / (config.ice_density * thickness)
    # 线性粘性阻尼 c = 2 zeta omega m（临界阻尼 2 sqrt(k m) = 2 omega m）
    damping = 2.0 * config.heave_damping_ratio * torch.sqrt(omega_sq) * masses
    buoy_k = config.water_density * config.gravity * areas

    # 初始垂荡默认取块底刚好触水（块心 z = h/2）：浮态必须由动力学自己沉降出来，
    # 而不是从解析平衡态出发（否则"干舷对阿基米德"的验证就退化成恒等式）。
    if initial_heave is None:
        heave = 0.5 * thickness + initial_heave_offset
    else:
        heave = initial_heave.to(D).clone()
    heave_vel = torch.zeros(n, dtype=D)
    velocity = torch.zeros(n, 2, dtype=D)

    positions0 = positions.clone()
    dt = config.dt
    if t_end is None:
        t_end = config.sweep_time
    n_steps = int(math.ceil(t_end / dt))
    apex_y = -config.bow_length - 2.0
    meters_per_step = config.structure_speed * dt
    max_penetration = 0.0
    max_relative_speed = 0.0
    structure_velocity = torch.tensor([0.0, config.structure_speed], dtype=D)
    history: list[torch.Tensor] = []

    for _ in range(n_steps):
        submerged = torch.clamp(0.5 * thickness - heave, min=0.0)
        submerged = torch.minimum(submerged, thickness)
        force = -masses * config.gravity + buoy_k * submerged - damping * heave_vel
        heave_vel = heave_vel + dt * force / masses
        heave = heave + dt * heave_vel

        # --- 水平 ---
        if config.drag_time is not None:
            velocity = velocity * (1.0 - dt / config.drag_time)

        for _ in range(config.contact_iterations):
            positions, velocity, pen, closing = _apply_pair_contact(
                positions, velocity, inv_mass, radii, config.restitution, config.friction
            )
            max_penetration = max(max_penetration, pen)
            max_relative_speed = max(max_relative_speed, closing)

        if structure:
            segments = _bow_segments(apex_y, config)
            positions, velocity, pen = _apply_structure_contact(
                positions,
                velocity,
                inv_mass,
                radii,
                segments,
                structure_velocity,
                config.restitution,
                config.friction,
                config.bow_half_thickness,
            )
            max_penetration = max(max_penetration, pen)
            apex_y += meters_per_step

        positions = positions + velocity * dt
        if record_history and n == 1:
            history.append(heave.clone())

    # --- 清空统计 ---
    half_width = config.effective_channel_half_width
    y_range = (0.0, config.field_length)
    in_corridor0 = positions0[:, 0].abs() <= half_width
    corridor0_idx = torch.nonzero(in_corridor0, as_tuple=False).reshape(-1)
    n_corridor0 = int(corridor0_idx.numel())
    corridor: dict = {
        "half_width": half_width,
        "n_corridor_initial": n_corridor0,
        "coverage_initial": corridor_coverage(positions0, radii, half_width, y_range, config.coverage_cell),
        "clearing_rate_centroid": None,
        "clearing_rate_coverage": None,
        "mean_lateral_displacement": None,
        "dragged_forward_fraction": None,
        "residual_fraction": None,
    }
    if n_corridor0 > 0:
        # 走廊内的块在扫掠结束后分三类（空间窗口统一为 y in [0, field_length]，
        # 与覆盖率统计一致）：
        #   cleared   : 块心横向被船首抛出走廊 |x| > half_width；
        #   dragged   : 未侧向清出但被顶着推出分析窗口（y > field_length），即"推土"；
        #   residual  : 仍在走廊内的残留冰。
        lateral = positions[corridor0_idx, 0].abs() - half_width
        cleared = lateral > 0.0
        y_final = positions[corridor0_idx, 1]
        dragged = (~cleared) & (y_final > config.field_length)
        residual = (~cleared) & (~dragged)
        corridor["clearing_rate_centroid"] = float(cleared.to(D).mean())
        corridor["dragged_forward_fraction"] = float(dragged.to(D).mean())
        corridor["residual_fraction"] = float(residual.to(D).mean())
        corridor["mean_lateral_displacement"] = float(lateral.mean())
        coverage_final = corridor_coverage(positions, radii, half_width, y_range, config.coverage_cell)
        corridor["coverage_final"] = coverage_final
        if corridor["coverage_initial"] > 0.0:
            corridor["clearing_rate_coverage"] = 1.0 - coverage_final / corridor["coverage_initial"]

    diagnostic = {
        "n_steps": n_steps,
        "t_end": t_end,
        "omega_min": float(torch.sqrt(omega_sq).min()),
        "omega_max": float(torch.sqrt(omega_sq).max()),
        "omega_dt_max": float(torch.sqrt(omega_sq).max()) * dt,
        "max_penetration": max_penetration,
        "max_closing_speed": max_relative_speed,
        "apex_travel": apex_y + config.bow_length + 2.0,
    }

    return FloeFieldResult(
        config=config,
        positions0=positions0,
        positions=positions,
        radii=radii,
        thickness=thickness,
        heave=heave,
        heave_velocity=heave_vel,
        velocity=velocity,
        t_end=t_end,
        max_penetration=max_penetration,
        max_relative_speed=max_relative_speed,
        heave_history=(
            torch.stack(history).reshape(-1) if record_history and n == 1 else None
        ),
        corridor=corridor,
        diagnostic=diagnostic,
    )


def simulate_free_float(
    thickness: float,
    radius: float = 3.0,
    config: FloeFieldConfig | None = None,
    *,
    initial_heave_offset: float = 0.0,
    damping_ratio: float | None = None,
    t_end: float = 20.0,
) -> dict:
    """Single-floe heave benchmark: measured freeboard vs Archimedes.

    Returns the measured freeboard, the analytic freeboard and the relative
    error, plus the measured heave period when the damping is switched off.
    """
    base = config or FloeFieldConfig(n_floes=1, drag_time=None)
    cfg_kwargs = dict(
        n_floes=1,
        radius_range=(radius, radius),
        thickness_range=(thickness, thickness),
        ice_density=base.ice_density,
        water_density=base.water_density,
        gravity=base.gravity,
        restitution=base.restitution,
        friction=base.friction,
        heave_damping_ratio=(
            base.heave_damping_ratio if damping_ratio is None else damping_ratio
        ),
        drag_time=None,
        dt=base.dt,
        seed=base.seed,
    )
    cfg = FloeFieldConfig(**cfg_kwargs)
    positions = torch.zeros(1, 2, dtype=D)
    radii = torch.tensor([radius], dtype=D)
    ice = torch.tensor([thickness], dtype=D)
    analytic = archimedes_equilibrium(thickness, cfg.ice_density, cfg.water_density)
    start = None
    if initial_heave_offset != 0.0:
        start = torch.full((1,), analytic.center_elevation + initial_heave_offset, dtype=D)
    result = simulate_floe_field(
        cfg,
        initial_heave=start,
        positions=positions,
        radii=radii,
        thickness=ice,
        t_end=t_end,
        structure=False,
    )
    measured_freeboard = float(result.heave[0]) + 0.5 * thickness
    return {
        "freeboard_measured": measured_freeboard,
        "freeboard_analytic": analytic.freeboard,
        "relative_error": abs(measured_freeboard - analytic.freeboard) / analytic.freeboard,
        "draft_measured": thickness - measured_freeboard,
        "draft_analytic": analytic.draft,
        "heave": float(result.heave[0]),
        "heave_velocity": float(result.heave_velocity[0]),
        "omega_analytic": heave_natural_frequency(
            thickness, cfg.ice_density, cfg.water_density, cfg.gravity
        ),
    }


def simulate_wedge_floe_sideways(
    half_angle_deg: float = 30.0,
    speed: float = 1.0,
    *,
    radius: float = 1.5,
    thickness: float = 0.8,
    restitution: float = 0.0,
    t_end: float = 30.0,
    config: FloeFieldConfig | None = None,
) -> dict:
    """Single floe pushed by an inclined face: measured vs analytic lateral speed.

    Analytic steady state for a frictionless contact against an infinite-mass
    face with normal ``n = (cos a, sin a)`` moving at ``V (0, 1)``:
    ``v = (V . n) n`` hence ``v_x = V sin a cos a``.
    """
    base = config or FloeFieldConfig(n_floes=1, drag_time=None)
    cfg = FloeFieldConfig(
        n_floes=1,
        radius_range=(radius, radius),
        thickness_range=(thickness, thickness),
        ice_density=base.ice_density,
        water_density=base.water_density,
        gravity=base.gravity,
        restitution=restitution,
        friction=0.0,
        heave_damping_ratio=base.heave_damping_ratio,
        drag_time=None,
        dt=base.dt,
        structure_speed=speed,
        bow_half_angle_deg=half_angle_deg,
        bow_length=base.bow_length,
        parallel_body_length=base.parallel_body_length,
        seed=base.seed,
    )
    # 冰盘放在右舷斜面中段的外法向一侧（距离 = r + 段半厚 + 微小间隙）：
    # 若放在楔尖附近会撞到斜面端点，端点接触的法向不是斜面法向，
    # 解析稳态 v = V sin(a) cos(a) 就不再适用（15 度工况曾因此偏 91%）。
    apex_y0 = -cfg.bow_length - 2.0
    face_point = torch.tensor(
        [0.6 * cfg.bow_half_width, apex_y0 - 0.6 * cfg.bow_length], dtype=D
    )
    angle_rad = math.radians(cfg.bow_half_angle_deg)
    normal = torch.tensor([math.cos(angle_rad), math.sin(angle_rad)], dtype=D)
    positions = (face_point + normal * (radius + cfg.bow_half_thickness + 0.05)).reshape(1, 2)
    result = simulate_floe_field(
        cfg,
        positions=positions,
        radii=torch.tensor([radius], dtype=D),
        thickness=torch.tensor([thickness], dtype=D),
        t_end=t_end,
        structure=True,
    )
    angle = math.radians(half_angle_deg)
    analytic_vx = speed * math.sin(angle) * math.cos(angle)
    return {
        "vx_measured": float(result.velocity[0, 0]),
        "vx_analytic": analytic_vx,
        "relative_error": abs(float(result.velocity[0, 0]) - analytic_vx) / analytic_vx,
        "vy_measured": float(result.velocity[0, 1]),
        "vy_analytic": speed * math.sin(angle) ** 2,
    }


def run_clearing_study(
    speeds: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    config: FloeFieldConfig | None = None,
) -> list[dict]:
    """Channel-clearing sweep: clearing rate and bulldozing vs advance speed."""
    base = config or FloeFieldConfig()
    rows: list[dict] = []
    for speed in speeds:
        cfg = FloeFieldConfig(
            **{
                **{
                    name: getattr(base, name)
                    for name in FloeFieldConfig.__dataclass_fields__
                    if name
                    in {
                        "n_floes",
                        "radius_range",
                        "thickness_range",
                        "ice_density",
                        "water_density",
                        "gravity",
                        "restitution",
                        "friction",
                        "heave_damping_ratio",
                        "drag_time",
                        "dt",
                        "bow_half_angle_deg",
                        "bow_length",
                        "bow_half_thickness",
                        "parallel_body_length",
                        "channel_half_width",
                        "field_length",
                        "field_half_width",
                        "coverage_cell",
                        "seed",
                    }
                },
                "structure_speed": speed,
            }
        )
        result = simulate_floe_field(cfg)
        rows.append(
            {
                "speed": speed,
                "clearing_rate_centroid": result.corridor["clearing_rate_centroid"],
                "clearing_rate_coverage": result.corridor["clearing_rate_coverage"],
                "mean_lateral_displacement": result.corridor["mean_lateral_displacement"],
                "dragged_forward_fraction": result.corridor["dragged_forward_fraction"],
                "residual_fraction": result.corridor["residual_fraction"],
                "n_corridor_initial": result.corridor["n_corridor_initial"],
                "coverage_initial": result.corridor["coverage_initial"],
                "max_penetration": result.max_penetration,
                "froude_like": speed
                / math.sqrt(base.gravity * float(result.thickness.mean())),
            }
        )
    return rows


def measure_heave_period(
    thickness: float,
    radius: float = 3.0,
    *,
    offset: float = 0.002,
    t_end: float = 12.0,
    config: FloeFieldConfig | None = None,
) -> dict:
    """Undamped free heave oscillation: measured period vs ``2 pi / omega``.

    The floe starts at the analytic floatation equilibrium plus a small
    ``offset``; upward zero crossings of ``z - z_eq`` give the period, so this
    is an independent check of the restoring stiffness the integrator sees
    (it does not reuse the closed-form frequency that builds the damping).
    """
    base = config or FloeFieldConfig(n_floes=1, drag_time=None)
    cfg = FloeFieldConfig(
        n_floes=1,
        radius_range=(radius, radius),
        thickness_range=(thickness, thickness),
        ice_density=base.ice_density,
        water_density=base.water_density,
        gravity=base.gravity,
        heave_damping_ratio=0.0,
        drag_time=None,
        dt=base.dt,
        seed=base.seed,
    )
    analytic = archimedes_equilibrium(thickness, cfg.ice_density, cfg.water_density)
    start = torch.tensor([analytic.center_elevation + offset], dtype=D)
    result = simulate_floe_field(
        cfg,
        initial_heave=start,
        positions=torch.zeros(1, 2, dtype=D),
        radii=torch.tensor([radius], dtype=D),
        thickness=torch.tensor([thickness], dtype=D),
        t_end=t_end,
        structure=False,
        record_history=True,
    )
    history = result.heave_history
    assert history is not None
    signal = history - analytic.center_elevation
    crossings: list[float] = []
    for k in range(1, signal.numel()):
        if float(signal[k - 1]) <= 0.0 < float(signal[k]):
            # 线性插值穿越时刻
            prev = float(signal[k - 1])
            cur = float(signal[k])
            frac = -prev / (cur - prev)
            crossings.append((k - 1 + frac) * cfg.dt)
    periods = [b - a for a, b in zip(crossings, crossings[1:])]
    if not periods:
        raise RuntimeError("no heave zero crossings recorded; increase t_end")
    period_measured = sum(periods) / len(periods)
    omega = heave_natural_frequency(thickness, cfg.ice_density, cfg.water_density, cfg.gravity)
    period_analytic = 2.0 * math.pi / omega
    amplitude = float(signal.abs().max())
    return {
        "period_measured": period_measured,
        "period_analytic": period_analytic,
        "relative_error": abs(period_measured - period_analytic) / period_analytic,
        "n_periods": len(periods),
        "amplitude": amplitude,
    }

