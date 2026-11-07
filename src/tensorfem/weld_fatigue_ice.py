"""Ice-region welded-joint fatigue assessment chain (SI units).

This module owns the TensorFEM-side post-processing chain that turns an ice
crushing load history into a welded-joint fatigue damage number:

1. :class:`WeldGeometry` -- parametric butt / fillet weld toe description
   (reinforcement or leg size, toe radius, flank angle).
2. :func:`toe_stress_concentration` -- a finite-element weld-toe notch
   submodel in elliptic-cylinder coordinates (orthogonal body-fitted Q4 mesh,
   near-tip edge length well below 2 mm) solved with
   :func:`tensorfem.continuum.solve_continuum`.  The submodel recovers the toe
   (notch) stress concentration and the surface-path stress used for the
   IIW / DNV hot-spot extrapolations.
3. :class:`DNVSNCurve` / :class:`IIWFatClass` -- two-slope S-N curves
   (DNV-RP-C203 curve D in air, IIW FAT classes) plus the low-temperature
   penalty ``k_T``.
4. :func:`ice_pressure_history` -- DEM-calibrated stochastic surrogate for the
   ice crushing pressure on a side-shell panel (a DEM solver itself lives in
   TensorDEM; this function is *not* a DEM run).
5. :func:`exact_sdof_response` / :func:`response_energy_ledger` -- exact
   piecewise-linear SDOF structural response with closed energy bookkeeping
   (load work = strain + kinetic + damping dissipation).
6. :func:`fatigue_assessment` / :func:`run_ice_weld_fatigue_matrix` -- rainflow
   counting, Palmgren-Miner accumulation, 18-case ice-thickness / speed / weld
   type scan.

Nothing here is a class-society approval: the numbers are engineering
verification evidence with documented assumptions and explicit limitations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from .continuum import ContinuumModel, solve_continuum
from .marine_fatigue_advanced import rainflow_cycles

__all__ = [
    "WeldGeometry",
    "NotchSubmodel",
    "toe_stress_concentration",
    "hotspot_extrapolation",
    "DNVSNCurve",
    "IIWFatClass",
    "low_temperature_factor",
    "SNTemperatureCorrection",
    "PanelStructure",
    "ice_pressure_history",
    "exact_sdof_response",
    "response_energy_ledger",
    "miner_damage",
    "fatigue_assessment",
    "run_ice_weld_fatigue_matrix",
    "elliptical_hole_kt_fe",
    "CASE_ICE_THICKNESS_M",
    "CASE_SPEED_KN",
    "CASE_WELD_TYPES",
]

#: Ice thickness levels of the campaign scan (m).
CASE_ICE_THICKNESS_M: tuple[float, ...] = (0.3, 0.5, 0.8)
#: Ship speed levels of the campaign scan (kn).
CASE_SPEED_KN: tuple[float, ...] = (3.0, 5.0, 7.0)
#: Weld type levels of the campaign scan.
CASE_WELD_TYPES: tuple[str, ...] = ("butt", "fillet")

#: Knots -> m/s.
KNOT_TO_M_S = 0.514444


def _real(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


# ---------------------------------------------------------------------------
# 1. Weld geometry parameterisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeldGeometry:
    """Parametric weld toe geometry.

    Parameters
    ----------
    weld_type:
        ``"butt"`` (对接) or ``"fillet"`` (角接).
    plate_thickness_m:
        Attached plate thickness ``t`` (m).
    reinforcement_m:
        Butt welds: cap reinforcement height ``h_r`` (m).  Fillet welds: weld
        leg size ``a`` (m).
    toe_radius_m:
        Weld toe transition radius ``r_t`` (m).  As-welded values of 0.5 mm to
        2 mm are the practical range.
    flank_angle_deg:
        Weld flank angle measured from the plate surface (deg).
    fillet_notch_fraction:
        Fraction of the fillet leg that is treated as the active notch depth of
        the toe region; only used for fillet welds.  This is an engineering
        idealisation of the convex fillet flank (see the module docs).
    """

    weld_type: str = "butt"
    plate_thickness_m: float = 0.012
    reinforcement_m: float = 1.5e-3
    toe_radius_m: float = 1.0e-3
    flank_angle_deg: float = 30.0
    fillet_notch_fraction: float = 0.15

    def __post_init__(self) -> None:
        kind = str(self.weld_type).lower()
        if kind not in ("butt", "fillet"):
            raise ValueError("weld_type must be 'butt' or 'fillet'")
        object.__setattr__(self, "weld_type", kind)
        thickness = _real(self.plate_thickness_m, "plate_thickness_m", positive=True)
        _real(self.reinforcement_m, "reinforcement_m", positive=True)
        _real(self.toe_radius_m, "toe_radius_m", positive=True)
        angle = _real(self.flank_angle_deg, "flank_angle_deg")
        if not 5.0 <= angle <= 85.0:
            raise ValueError("flank_angle_deg must be within [5, 85]")
        fraction = _real(self.fillet_notch_fraction, "fillet_notch_fraction")
        if not 0.0 < fraction <= 0.5:
            raise ValueError("fillet_notch_fraction must be within (0, 0.5]")
        if kind == "butt" and self.reinforcement_m > 0.25 * thickness:
            raise ValueError("butt reinforcement must not exceed 25% of the plate thickness")
        if kind == "fillet" and self.reinforcement_m > thickness:
            raise ValueError("fillet leg size must not exceed the plate thickness")

    @property
    def flank_angle_rad(self) -> float:
        return math.radians(self.flank_angle_deg)

    @property
    def equivalent_notch_width_ratio(self) -> float:
        """Equivalent notch half-width over tip radius, ``A / r_t``.

        The toe region is idealised as a semi-elliptical surface notch of tip
        radius ``r_t`` and half-width ``A``.  Its width follows the flank angle:
        a steeper (smaller-angle) flank exposes a narrower arc at the toe, so
        ::

            A / r_t = 1 + 1.6 cos(theta)

        The constant is a documented idealisation of the convex fillet flank and
        is treated as an input assumption, not as a class-rule constant; the
        benchmark reports the sensitivity of ``K_t`` to it.
        """
        return 1.0 + 1.6 * math.cos(self.flank_angle_rad)

    @property
    def notch_depth_m(self) -> float:
        """Equivalent notch depth ``d = r_t sqrt(A / r_t)`` (m).

        For a semi-ellipse with tip radius ``r_t`` and half-width ``A`` the depth
        is ``d = sqrt(A r_t)``; the weld-type inputs (butt reinforcement ``h_r``
        or fillet leg ``a``) enter through :attr:`equivalent_notch_width_ratio`
        only in so far as they set the flank angle of the real detail.
        """
        return self.toe_radius_m * math.sqrt(self.equivalent_notch_width_ratio)

    @property
    def notch_aspect_ratio(self) -> float:
        """``d / r_t`` of the equivalent toe notch."""
        return self.notch_depth_m / self.toe_radius_m

    @property
    def weld_toe_ratio(self) -> float:
        """Butt reinforcement (or fillet leg) to plate thickness."""
        return self.reinforcement_m / self.plate_thickness_m


# ---------------------------------------------------------------------------
# 2. Finite-element weld-toe notch submodel (elliptic-cylinder coordinates)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NotchSubmodel:
    """Result container of the FE notch submodel (all stresses divided by the
    remote tension ``sigma_inf``)."""

    depth_m: float
    radius_m: float
    kt_toe: float
    kt_hotspot_iiw: float
    kt_hotspot_dnv: float
    surface_path_m: tuple[float, ...]
    surface_stress_ratio: tuple[float, ...]
    min_edge_m: float
    n_nodes: int
    n_elements: int

    @property
    def mesh_fine_enough(self) -> bool:
        """Campaign requirement: near-toe mesh refinement to 2 mm or finer."""
        return self.min_edge_m <= 2.0e-3


def _elliptic_edge_notch_mesh(depth_m: float, radius_m: float, *, n_xi: int, n_eta: int, outer_ratio: float):
    """Body-fitted Q4 mesh of a semi-elliptical edge notch in a half-plane.

    Elliptic-cylinder coordinates ``x = c cosh(eta) cos(xi)``,
    ``y = c sinh(eta) sin(xi)`` with ``xi in [0, pi]``, ``eta in [eta0, eta_out]``.
    The notch surface is the arc ``eta = eta0`` (depth ``d`` at ``xi = pi/2``,
    tip radius ``rho = d^2 / A``), the flat free surface is ``xi = 0`` and
    ``xi = pi``, and the far field is ``eta = eta_out``.
    """
    depth = float(depth_m)
    radius = float(radius_m)
    if depth <= radius:
        raise ValueError("notch depth must exceed the tip radius")
    semi_axis = depth * depth / radius  # A, semi-axis along x
    c = math.sqrt(semi_axis * semi_axis - depth * depth)
    eta0 = math.asinh(depth / c)
    cosh_out = outer_ratio * semi_axis / c
    if cosh_out <= math.cosh(eta0):
        raise ValueError("outer_ratio too small for the requested patch size")
    eta_out = math.acosh(cosh_out)
    xi = math.pi * _cluster(np.linspace(0.0, 1.0, n_xi + 1), 0.6)
    eta = eta0 + (eta_out - eta0) * np.linspace(0.0, 1.0, n_eta + 1) ** 2
    xx, yy = np.meshgrid(np.cosh(eta), np.sin(xi), indexing="ij")
    ss, cc = np.meshgrid(np.sinh(eta), np.cos(xi), indexing="ij")
    nodes = np.stack((c * xx * cc, c * ss * yy), axis=-1).reshape(-1, 2)
    cells = []
    for j in range(n_eta):
        for i in range(n_xi):
            n0 = j * (n_xi + 1) + i
            cells.append((n0, n0 + n_xi + 1, n0 + n_xi + 2, n0 + 1))
    return nodes, np.asarray(cells, dtype=np.int64), semi_axis


def _cluster(s: np.ndarray, beta: float) -> np.ndarray:
    """Cluster parameter values towards ``s = 0.5`` (``0 <= beta < 1``)."""
    return s - beta * np.sin(2.0 * np.pi * s) / (2.0 * np.pi)


def _min_edge(nodes: np.ndarray, cells: np.ndarray) -> float:
    quad = nodes[cells]
    edges = [np.linalg.norm(quad[:, (k + 1) % 4] - quad[:, k], axis=1) for k in range(4)]
    return float(min(edge.min() for edge in edges))


def _outward_x_forces(
    nodes: np.ndarray,
    top: np.ndarray,
    sigma_inf: float,
    *,
    closed: bool = False,
    axis: str = "x",
) -> np.ndarray:
    """Consistent nodal forces for a remote tension ``sigma_inf``.

    On an outward-normal boundary ``n_x ds = dy`` and ``n_y ds = -dx``, so a
    boundary segment carries ``sigma_inf * (y2 - y1)`` (x tension) or
    ``-sigma_inf * (x2 - x1)`` (y tension), split equally on its two end nodes.
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    chain = np.append(top, top[0]) if closed else top
    nodal = np.zeros(nodes.shape[0])
    for a, b in zip(chain[:-1], chain[1:]):
        if axis == "x":
            share = 0.5 * sigma_inf * (nodes[b, 1] - nodes[a, 1])
        else:
            share = -0.5 * sigma_inf * (nodes[b, 0] - nodes[a, 0])
        nodal[a] += share
        nodal[b] += share
    forces = np.zeros(2 * nodes.shape[0])
    forces[0 if axis == "x" else 1 :: 2] = nodal
    return forces


def _nodal_stress_xx(nodes: np.ndarray, cells: np.ndarray, stress_xx: np.ndarray) -> np.ndarray:
    acc = np.zeros(nodes.shape[0])
    cnt = np.zeros(nodes.shape[0])
    for element, value in zip(cells, stress_xx):
        acc[element] += value
        cnt[element] += 1.0
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)


def _solve_notch(nodes: np.ndarray, cells: np.ndarray, *, n_xi: int) -> tuple[np.ndarray, np.ndarray]:
    top = np.arange(nodes.shape[0] - (n_xi + 1), nodes.shape[0])
    forces = _outward_x_forces(nodes, top, 1.0)
    node_tensor = torch.as_tensor(nodes, dtype=torch.float64)
    model = ContinuumModel(
        nodes=node_tensor,
        elements=torch.as_tensor(cells, dtype=torch.long),
        young_modulus=torch.tensor(210.0e9, dtype=torch.float64),
        poisson_ratio=torch.tensor(0.3, dtype=torch.float64),
        thickness=torch.tensor(1.0, dtype=torch.float64),
        forces=torch.as_tensor(forces, dtype=torch.float64),
        fixed_dofs=torch.tensor(
            [0, 1, 2 * (n_xi + 1) + 1], dtype=torch.long
        ),
        element_type="q4",
        plane="stress",
    )
    result = solve_continuum(model)
    return result.displacement.detach().numpy(), result.stress[:, 0].detach().numpy()


@lru_cache(maxsize=32)
def _notch_submodel_cached(
    depth_m: float, radius_m: float, plate_thickness_m: float, n_xi: int, n_eta: int, outer_ratio: float
) -> NotchSubmodel:
    semi_axis = depth_m * depth_m / radius_m
    patch = 1.7 * plate_thickness_m
    ratio = max(float(outer_ratio), 1.0 + patch / semi_axis)
    nodes, cells, _ = _elliptic_edge_notch_mesh(
        depth_m, radius_m, n_xi=n_xi, n_eta=n_eta, outer_ratio=ratio
    )
    _, stress_xx = _solve_notch(nodes, cells, n_xi=n_xi)
    tip = n_xi // 2  # xi = pi/2 -> notch root
    # The two first-ring elements adjacent to the root node carry the peak.
    kt_toe = float(np.mean((stress_xx[tip - 1], stress_xx[tip])))

    nodal = _nodal_stress_xx(nodes, cells, stress_xx)
    left = np.arange(0, (cells.shape[0] // n_xi) * (n_xi + 1), n_xi + 1)
    xs = nodes[left, 0]
    stresses = nodal[left]
    order = np.argsort(xs)
    xs, stresses = xs[order], stresses[order]
    toe_x = float(xs[0])

    distances = tuple(float(v) for v in np.linspace(0.0, 1.6 * plate_thickness_m, 33))
    path = tuple(float(np.interp(toe_x + d, xs, stresses)) for d in distances)
    kt_iiw = hotspot_extrapolation(path, distances, plate_thickness_m, method="iiw")
    kt_dnv = hotspot_extrapolation(path, distances, plate_thickness_m, method="dnv")
    return NotchSubmodel(
        depth_m=float(depth_m),
        radius_m=float(radius_m),
        kt_toe=kt_toe,
        kt_hotspot_iiw=kt_iiw,
        kt_hotspot_dnv=kt_dnv,
        surface_path_m=distances,
        surface_stress_ratio=path,
        min_edge_m=_min_edge(nodes, cells),
        n_nodes=int(nodes.shape[0]),
        n_elements=int(cells.shape[0]),
    )


def hotspot_extrapolation(
    surface_stress_ratio: Sequence[float],
    distances_m: Sequence[float],
    plate_thickness_m: float,
    *,
    method: str = "iiw",
) -> float:
    """Linear surface extrapolation of the structural hot-spot stress.

    ``method="iiw"`` uses the 0.4t / 1.0t sampling pair, whose linear
    extrapolation weights are ``(1.67, -0.67)``;
    ``method="dnv"`` uses the 0.5t / 1.5t pair with weights ``(1.5, -0.5)``.
    Both are applied to stresses normalised by the remote tension.
    """
    keys = {"iiw": (0.4, 1.0), "dnv": (0.5, 1.5)}
    if method not in keys:
        raise ValueError("method must be 'iiw' or 'dnv'")
    thickness = _real(plate_thickness_m, "plate_thickness_m", positive=True)
    x = np.asarray(distances_m, dtype=float)
    y = np.asarray(surface_stress_ratio, dtype=float)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("distances and stresses must be matching arrays")
    first, second = keys[method]
    s1 = float(np.interp(first * thickness, x, y))
    s2 = float(np.interp(second * thickness, x, y))
    span = (second - first) * thickness
    return s1 - first * thickness * (s2 - s1) / span


def toe_stress_concentration(
    geometry: WeldGeometry, *, n_xi: int = 48, n_eta: int = 28, outer_ratio: float = 16.0
) -> NotchSubmodel:
    """FE weld-toe notch submodel for the given geometry.

    Returns the toe (peak) stress concentration ``kt_toe = sigma_toe / sigma_nom``
    and the two hot-spot extrapolation variants.
    """
    if not isinstance(geometry, WeldGeometry):
        raise TypeError("geometry must be a WeldGeometry")
    return _notch_submodel_cached(
        round(geometry.notch_depth_m, 12),
        round(geometry.toe_radius_m, 12),
        round(geometry.plate_thickness_m, 12),
        int(n_xi),
        int(n_eta),
        round(float(outer_ratio), 6),
    )


@lru_cache(maxsize=16)
def _ring_hole_kt(depth_m: float, radius_m: float, n_xi: int, n_eta: int, outer_ratio: float) -> float:
    """FE stress concentration of an *internal* elliptical hole loaded along x.

    Exact reference: ``sigma_tip / sigma_inf = 1 + 2 a / b`` (Inglis), where
    ``A``, ``B`` are the semi-axes across and along the load (``A > B``); the
    load is applied along ``y`` so that the tip at ``xi = 0`` gives
    ``sigma_tip / sigma_inf = 1 + 2 A / B``.  The mesh uses the same
    elliptic-cylinder mapping with ``xi in [0, 2 pi]``.
    """
    b = float(depth_m)  # semi-axis along the load
    rho = float(radius_m)
    a = b * b / rho  # tip radius rho = b^2 / a
    c = math.sqrt(max(a * a - b * b, 1e-30))
    eta0 = math.asinh(b / c)
    eta_out = math.acosh(outer_ratio * a / c)
    xi = 2.0 * math.pi * _cluster(np.linspace(0.0, 1.0, n_xi + 1)[:-1], 0.6)
    eta = eta0 + (eta_out - eta0) * np.linspace(0.0, 1.0, n_eta + 1) ** 2
    rings = []
    for e in eta:
        rings.append(np.stack((c * math.cosh(e) * np.cos(xi), c * math.sinh(e) * np.sin(xi)), axis=1))
    nodes = np.concatenate(rings, axis=0)
    n_x = len(xi)
    cells = []
    for j in range(n_eta):
        for i in range(n_x):
            n0 = j * n_x + i
            cells.append(
                (n0, (j + 1) * n_x + i, (j + 1) * n_x + (i + 1) % n_x, j * n_x + (i + 1) % n_x)
            )
    cells = np.asarray(cells, dtype=np.int64)
    top = np.arange(n_eta * n_x, (n_eta + 1) * n_x)
    forces = _outward_x_forces(nodes, top, 1.0, closed=True, axis="y")
    model = ContinuumModel(
        nodes=torch.as_tensor(nodes, dtype=torch.float64),
        elements=torch.as_tensor(cells, dtype=torch.long),
        young_modulus=torch.tensor(210.0e9, dtype=torch.float64),
        poisson_ratio=torch.tensor(0.3, dtype=torch.float64),
        thickness=torch.tensor(1.0, dtype=torch.float64),
        forces=torch.as_tensor(forces, dtype=torch.float64),
        fixed_dofs=torch.tensor([0, 1, 2 * (n_x // 4)], dtype=torch.long),
        element_type="q4",
        plane="stress",
    )
    result = solve_continuum(model)
    stress_yy = result.stress[:, 1].detach().numpy()
    return float(np.mean((stress_yy[0], stress_yy[n_x - 1])))


def elliptical_hole_kt_fe(a_over_b: float, *, n_xi: int = 64, n_eta: int = 24, outer_ratio: float = 40.0) -> float:
    """FE K_t of an internal elliptical hole (validation helper for the notch mesh)."""
    ratio = _real(a_over_b, "a_over_b", positive=True)
    if ratio < 1.0:
        raise ValueError("a_over_b must be >= 1")
    semi_b = 1.0e-3
    radius = semi_b / ratio  # rho = b^2 / a  with a = ratio * b
    return _ring_hole_kt(semi_b, radius, int(n_xi), int(n_eta), float(outer_ratio))


# ---------------------------------------------------------------------------
# 3. S-N curves and the low-temperature penalty
# ---------------------------------------------------------------------------

KNEE_STRESS_TOLERANCE = 0.03


@dataclass(frozen=True)
class DNVSNCurve:
    """DNV-RP-C203 two-slope S-N curve (stress range in Pa, cycles in 1).

    Default values are curve D *in air*: ``log10 N = 12.164 - 3 log10 ds`` below
    the knee at ``1e7`` cycles and ``log10 N = 15.606 - 5 log10 ds`` above it,
    which gives ``ds(N=1e7) = 52.63 MPa``.
    """

    name: str = "DNV-RP-C203 D air"
    log_a1: float = 12.164
    slope_m1: float = 3.0
    log_a2: float = 15.606
    slope_m2: float = 5.0
    knee_cycles: float = 1.0e7

    def cycles_to_failure(self, stress_range_pa: float) -> float:
        # The DNV formulation is written with the stress range in MPa.
        stress = _real(stress_range_pa, "stress_range_pa", positive=True) / 1.0e6
        n1 = 10.0 ** self.log_a1 * stress ** (-self.slope_m1)
        if n1 <= self.knee_cycles:
            return n1
        return 10.0 ** self.log_a2 * stress ** (-self.slope_m2)

    def stress_at_cycles(self, cycles: float) -> float:
        n = _real(cycles, "cycles", positive=True)
        if n <= self.knee_cycles:
            return 1.0e6 * (10.0 ** self.log_a1 / n) ** (1.0 / self.slope_m1)
        return 1.0e6 * (10.0 ** self.log_a2 / n) ** (1.0 / self.slope_m2)

    @property
    def knee_stress_pa(self) -> float:
        return self.stress_at_cycles(self.knee_cycles)

    def thickness_corrected(
        self, plate_thickness_m: float, *, reference_m: float = 0.025, exponent: float = 0.2
    ) -> "DNVSNCurve":
        """DNV thickness correction ``(t/t_ref)^k`` for ``t > 25 mm``.

        A thickness correction is implemented by lowering the reference stress,
        which is equivalent to raising the stress range of every cycle.
        """
        t = _real(plate_thickness_m, "plate_thickness_m", positive=True)
        if t <= reference_m:
            return self
        factor = (t / reference_m) ** exponent
        shift = math.log10(factor) * self.slope_m1
        return DNVSNCurve(
            name=f"{self.name} (t={t*1e3:.0f} mm)",
            log_a1=self.log_a1 - shift,
            slope_m1=self.slope_m1,
            slope_m2=self.slope_m2,
            log_a2=self.log_a2 - shift * self.slope_m2 / self.slope_m1,
            knee_cycles=self.knee_cycles,
        )


@dataclass(frozen=True)
class IIWFatClass:
    """IIW FAT class: ``FAT`` MPa at 2e6 cycles, slope 3, knee at 1e7, slope 5."""

    fat_mpa: float = 90.0
    knee_cycles: float = 1.0e7
    slope_m1: float = 3.0
    slope_m2: float = 5.0
    cutoff_cycles: float = 1.0e9

    def cycles_to_failure(self, stress_range_pa: float) -> float:
        stress = _real(stress_range_pa, "stress_range_pa", positive=True) / 1.0e6
        n1 = 2.0e6 * (self.fat_mpa / stress) ** self.slope_m1
        if n1 <= self.knee_cycles:
            return n1
        knee_stress = self.fat_mpa * (2.0e6 / self.knee_cycles) ** (1.0 / self.slope_m1)
        n2 = self.knee_cycles * (knee_stress / stress) ** self.slope_m2
        return min(n2, self.cutoff_cycles)

    def stress_at_cycles(self, cycles: float) -> float:
        n = _real(cycles, "cycles", positive=True)
        if n <= self.knee_cycles:
            return 1.0e6 * self.fat_mpa * (2.0e6 / n) ** (1.0 / self.slope_m1)
        knee_stress = self.fat_mpa * (2.0e6 / self.knee_cycles) ** (1.0 / self.slope_m1)
        return knee_stress * (self.knee_cycles / n) ** (1.0 / self.slope_m2)


def low_temperature_factor(
    temperature_c: float, *, reference_c: float = -5.0, factor: float = 1.2
) -> float:
    """Low-temperature penalty ``k_T`` on the applied stress range.

    ``k_T = 1`` for ``T >= 0 degC``, ``k_T = 1.2`` for ``T <= -5 degC``, linear in
    between.  The penalty represents the reduced weld-toughness margin at low
    temperature (campaign specification; it is *not* a DNV rule curve).
    """
    t = _real(temperature_c, "temperature_c")
    _real(factor, "factor", positive=True)
    if factor < 1.0:
        raise ValueError("factor must be >= 1")
    if t >= 0.0:
        return 1.0
    if t <= reference_c:
        return factor
    return 1.0 + (factor - 1.0) * (t / reference_c)


@dataclass(frozen=True)
class SNTemperatureCorrection:
    """Container describing the low-temperature S-N shift."""

    curve: DNVSNCurve
    temperature_c: float

    @property
    def factor(self) -> float:
        return low_temperature_factor(self.temperature_c)

    def cycles_to_failure(self, stress_range_pa: float) -> float:
        return self.curve.cycles_to_failure(self.factor * stress_range_pa)

    def stress_at_cycles(self, cycles: float) -> float:
        return self.curve.stress_at_cycles(cycles) / self.factor

    def strength_reduction(self, cycles: float) -> float:
        """Relative reduction of the allowable stress range at ``cycles``."""
        return 1.0 - self.stress_at_cycles(cycles) / self.curve.stress_at_cycles(cycles)


# ---------------------------------------------------------------------------
# 4. Structural transfer and ice load spectrum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelStructure:
    """Side-shell panel / lowest global mode used as the structural transfer.

    The static bending stress of a continuous (clamped) plate strip of span
    ``frame_spacing_m`` under pressure ``p`` is ``sigma = 0.25 p s^2 / t^2``.
    The SDOF mode shape factor is normalised so that the quasi-static response
    reproduces that beam value exactly.
    """

    plate_thickness_m: float = 0.025
    frame_spacing_m: float = 0.35
    span_m: float = 2.4
    natural_frequency_hz: float = 12.0
    damping_ratio: float = 0.03
    density_kg_m3: float = 7850.0
    modal_area_m2: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "plate_thickness_m",
            "frame_spacing_m",
            "span_m",
            "natural_frequency_hz",
            "density_kg_m3",
            "modal_area_m2",
        ):
            _real(getattr(self, name), name, positive=True)
        ratio = _real(self.damping_ratio, "damping_ratio")
        if not 0.0 < ratio < 1.0:
            raise ValueError("damping_ratio must be within (0, 1)")

    @property
    def stress_per_pressure(self) -> float:
        """Quasi-static plate bending stress per unit pressure (Pa/Pa)."""
        return 0.25 * self.frame_spacing_m**2 / self.plate_thickness_m**2

    @property
    def modal_mass_kg(self) -> float:
        volume = self.plate_thickness_m * self.frame_spacing_m * self.span_m
        return 0.5 * self.density_kg_m3 * volume

    @property
    def modal_stiffness_n_m(self) -> float:
        omega = 2.0 * math.pi * self.natural_frequency_hz
        return self.modal_mass_kg * omega * omega

    @property
    def modal_damping_n_s_m(self) -> float:
        return 2.0 * self.damping_ratio * math.sqrt(self.modal_stiffness_n_m * self.modal_mass_kg)

    @property
    def stress_gain_pa_per_m(self) -> float:
        """Maps modal displacement to nominal plate stress (Pa/m)."""
        return self.stress_per_pressure * self.modal_stiffness_n_m


def ice_pressure_history(
    duration_s: float,
    sample_rate_hz: float,
    ice_thickness_m: float,
    speed_kn: float,
    *,
    seed: int = 20261106,
    crushing_pressure_pa: float = 1.2e6,
    mean_pressure_ratio: float = 0.20,
    rate_coefficient_hz: float = 0.5,
) -> dict[str, np.ndarray | float]:
    """DEM-calibrated stochastic ice crushing pressure history (Pa).

    The surrogate superposes intermittent crushing pulses onto a slowly varying
    background pressure:

    * nominal crushing pressure ``p_G = p_ref (h/h_ref)^0.5 (v/v_ref)^0.1``:
      the local peak pressure grows sub-linearly with ice thickness because the
      contact patch grows with ``h`` (documented surrogate assumption),
    * mean level ``mean_pressure_ratio * p_G``,
    * grinding event rate ``f = rate_coefficient_hz * v / h`` (Hz), pulses of
      half-sine shape with Weibull(shape=2) amplitudes,
    * 8% multiplicative slow scatter (2 Hz interpolation) on the background
      level, so the rainflow reversals are dominated by the crushing pulses.

    This is a *surrogate* for a DEM time history, not a DEM solution; the module
    documents that limitation explicitly.
    """
    duration = _real(duration_s, "duration_s", positive=True)
    rate = _real(sample_rate_hz, "sample_rate_hz", positive=True)
    thickness = _real(ice_thickness_m, "ice_thickness_m", positive=True)
    speed = _real(speed_kn, "speed_kn", positive=True) * KNOT_TO_M_S
    n = int(round(duration * rate)) + 1
    t = np.arange(n) / rate
    rng = np.random.default_rng(int(seed))
    p_g = _real(crushing_pressure_pa, "crushing_pressure_pa", positive=True)
    p_g *= (thickness / 0.5) ** 0.5 * (max(speed, 0.1) / 2.57222) ** 0.1
    level = mean_pressure_ratio * p_g
    coarse = rng.standard_normal(int(duration * 2.0) + 2)
    background = level * (1.0 + 0.08 * np.interp(t, np.linspace(0.0, duration, coarse.size), coarse))
    background = np.clip(background, 0.0, None)
    event_rate = rate_coefficient_hz * speed / thickness
    count = int(max(1.0, event_rate * duration))
    times = np.sort(rng.uniform(0.0, duration, count))
    amplitudes = level * rng.weibull(2.0, count) * 1.2533  # unit mean
    widths = rng.uniform(0.35, 0.9, count) / max(event_rate, 0.1)
    pressure = background.copy()
    for start, amplitude, width in zip(times, amplitudes, widths):
        i0 = int(start * rate)
        i1 = min(n, i0 + max(int(width * rate), 3))
        if i1 <= i0:
            continue
        local = np.pi * (t[i0:i1] - t[i0]) / max(t[i1 - 1] - t[i0], 1e-9)
        pressure[i0:i1] += amplitude * np.sin(local)
    return {
        "time_s": t,
        "pressure_pa": pressure,
        "event_rate_hz": float(event_rate),
        "reference_pressure_pa": float(p_g),
        "mean_pressure_pa": float(level),
        "duration_s": float(duration),
        "sample_rate_hz": float(rate),
    }


# ---------------------------------------------------------------------------
# 5. Exact piecewise-linear SDOF response and energy ledger
# ---------------------------------------------------------------------------


def exact_sdof_response(
    force: np.ndarray, dt: float, *, mass: float, damping: float, stiffness: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact response of ``m u'' + c u' + k u = F(t)`` for piecewise-linear ``F``.

    The load is linear on every step ``[t_i, t_i + dt]``; the closed-form
    ramp solution (particular ``a0 + a1 tau`` plus the underdamped homogeneous
    part) is advanced step by step, so no time-integration error is introduced.
    """
    f = np.asarray(force, dtype=float)
    if f.ndim != 1 or f.size < 2:
        raise ValueError("force must be a one-dimensional array with >= 2 samples")
    series = f.tolist()
    step = _real(dt, "dt", positive=True)
    m = _real(mass, "mass", positive=True)
    c = _real(damping, "damping")
    k = _real(stiffness, "stiffness", positive=True)
    if c < 0.0:
        raise ValueError("damping must be >= 0")
    omega = math.sqrt(k / m)
    zeta = c / (2.0 * math.sqrt(k * m))
    if zeta >= 1.0:
        raise ValueError("exact ramp response requires an underdamped mode")
    omega_d = omega * math.sqrt(1.0 - zeta * zeta)
    out_u = [0.0] * len(series)
    out_v = [0.0] * len(series)
    z = zeta * omega
    expo = math.exp(-z * step)
    cos_d = math.cos(omega_d * step)
    sin_d = math.sin(omega_d * step)
    u_i = 0.0
    v_i = 0.0
    for i in range(len(series) - 1):
        f0, f1 = series[i], series[i + 1]
        a1 = (f1 - f0) / step / k
        a0 = (f0 - c * a1) / k
        amp = u_i - a0
        b = (v_i + z * amp - a1) / omega_d
        u_i = a0 + a1 * step + expo * (amp * cos_d + b * sin_d)
        v_i = a1 + expo * ((omega_d * b - z * amp) * cos_d - (omega_d * amp + z * b) * sin_d)
        out_u[i + 1] = u_i
        out_v[i + 1] = v_i
    return np.asarray(out_u), np.asarray(out_v)


_GAUSS_5 = (
    (-0.9061798459386640, 0.2369268850561891),
    (-0.5384693101056831, 0.4786286704993665),
    (0.0, 0.5688888888888889),
    (0.5384693101056831, 0.4786286704993665),
    (0.9061798459386640, 0.2369268850561891),
)


def response_energy_ledger(
    force: np.ndarray,
    displacement: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    *,
    mass: float,
    damping: float,
    stiffness: float,
) -> dict[str, float]:
    """Closed energy ledger of the SDOF response.

    ``work_load = strain + kinetic + damping_dissipation`` is integrated with
    5-point Gauss quadrature on the exact piecewise-linear response inside each
    step, so the residual is a quadrature error only (no time-marching error).
    """
    f = np.asarray(force, dtype=float)
    u = np.asarray(displacement, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if not (f.shape == u.shape == v.shape):
        raise ValueError("force, displacement and velocity must share a shape")
    step = _real(dt, "dt", positive=True)
    c = _real(damping, "damping")
    k = _real(stiffness, "stiffness", positive=True)
    m = _real(mass, "mass", positive=True)
    omega = math.sqrt(k / m)
    zeta = c / (2.0 * math.sqrt(k * m))
    if zeta >= 1.0:
        raise ValueError("energy ledger requires an underdamped mode")
    omega_d = omega * math.sqrt(1.0 - zeta * zeta)
    z = zeta * omega
    work = 0.0
    damp = 0.0
    for i in range(f.size - 1):
        f0, f1 = float(f[i]), float(f[i + 1])
        a1 = (f1 - f0) / step / k
        a0 = (f0 - c * a1) / k
        amp = float(u[i]) - a0
        b = (float(v[i]) + z * amp - a1) / omega_d
        for s, w in _GAUSS_5:
            tau = 0.5 * step * (s + 1.0)
            env = math.exp(-z * tau)
            trig_u = amp * math.cos(omega_d * tau) + b * math.sin(omega_d * tau)
            trig_v = (omega_d * b - z * amp) * math.cos(omega_d * tau) - (
                omega_d * amp + z * b
            ) * math.sin(omega_d * tau)
            vel = a1 + env * trig_v
            force_tau = f0 + (f1 - f0) * tau / step
            weight = w * 0.5 * step
            work += weight * force_tau * vel
            damp += weight * c * vel * vel
    strain = 0.5 * k * float(u[-1]) ** 2
    kinetic = 0.5 * m * float(v[-1]) ** 2
    scale = max(abs(work), 1e-30)
    residual = (work - strain - kinetic - damp) / scale
    return {
        "work_load_j": float(work),
        "strain_energy_j": float(strain),
        "kinetic_energy_j": float(kinetic),
        "damping_dissipation_j": float(damp),
        "relative_residual": float(residual),
    }


# ---------------------------------------------------------------------------
# 6. Rainflow / Miner accumulation and the 18-case scan
# ---------------------------------------------------------------------------


def miner_damage(cycles, curve, *, temperature_c: float = -20.0) -> dict[str, object]:
    """Palmgren-Miner damage of a rainflow spectrum with a low-temperature shift."""
    correction = SNTemperatureCorrection(curve=curve, temperature_c=temperature_c)
    total = 0.0
    blocks = []
    count = 0.0
    for cycle in cycles:
        stress = correction.factor * float(cycle.range_pa)
        if stress <= 0.0:
            continue
        life = curve.cycles_to_failure(stress)
        part = float(cycle.count) / life
        total += part
        count += float(cycle.count)
        blocks.append(
            {
                "range_pa": float(cycle.range_pa),
                "count": float(cycle.count),
                "corrected_range_pa": stress,
                "cycles_to_failure": life,
                "damage": part,
            }
        )
    blocks.sort(key=lambda row: row["damage"], reverse=True)
    return {
        "damage": total,
        "cycles": count,
        "factor": correction.factor,
        "blocks": blocks,
    }


@dataclass(frozen=True)
class IceFatigueCase:
    ice_thickness_m: float
    speed_kn: float
    weld_type: str

    @property
    def key(self) -> str:
        return f"h{self.ice_thickness_m:.1f}_v{self.speed_kn:.0f}_{self.weld_type}"


def weld_geometry_for(weld_type: str, plate_thickness_m: float = 0.025) -> WeldGeometry:
    """Typical as-welded toe geometry per weld type (documented defaults)."""
    kind = str(weld_type).lower()
    if kind == "butt":
        return WeldGeometry(
            weld_type="butt",
            plate_thickness_m=plate_thickness_m,
            reinforcement_m=0.06 * plate_thickness_m,
            toe_radius_m=1.0e-3,
            flank_angle_deg=30.0,
        )
    if kind == "fillet":
        return WeldGeometry(
            weld_type="fillet",
            plate_thickness_m=plate_thickness_m,
            reinforcement_m=0.4 * plate_thickness_m,
            toe_radius_m=1.0e-3,
            flank_angle_deg=45.0,
            fillet_notch_fraction=0.12,
        )
    raise ValueError("weld_type must be 'butt' or 'fillet'")


def fatigue_assessment(
    case: IceFatigueCase,
    *,
    structure: PanelStructure | None = None,
    curve: DNVSNCurve | None = None,
    temperature_c: float = -20.0,
    window_s: float = 3.0 * 3600.0,
    sample_rate_hz: float = 100.0,
    exposure_h: float = 500.0,
    seed: int = 20261106,
    weld_override: float | None = None,
    n_xi: int = 48,
    n_eta: int = 28,
) -> dict[str, object]:
    """Full chain for one ice / speed / weld-type case."""
    struct = structure or PanelStructure()
    sn = curve or DNVSNCurve()
    geometry = weld_geometry_for(case.weld_type, struct.plate_thickness_m)
    submodel = toe_stress_concentration(geometry, n_xi=n_xi, n_eta=n_eta)
    kt = submodel.kt_toe if weld_override is None else _real(weld_override, "weld_override", positive=True)
    load = ice_pressure_history(
        window_s, sample_rate_hz, case.ice_thickness_m, case.speed_kn, seed=seed
    )
    pressure = np.asarray(load["pressure_pa"])
    dt = 1.0 / float(load["sample_rate_hz"])
    modal_force = pressure * struct.modal_area_m2
    u, v = exact_sdof_response(
        modal_force,
        dt,
        mass=struct.modal_mass_kg,
        damping=struct.modal_damping_n_s_m,
        stiffness=struct.modal_stiffness_n_m,
    )
    ledger = response_energy_ledger(
        modal_force,
        u,
        v,
        dt,
        mass=struct.modal_mass_kg,
        damping=struct.modal_damping_n_s_m,
        stiffness=struct.modal_stiffness_n_m,
    )
    nominal = struct.stress_gain_pa_per_m * u
    hotspot = kt * nominal
    cycles = rainflow_cycles(hotspot - hotspot.mean())
    miner = miner_damage(cycles, sn, temperature_c=temperature_c)
    per_window = float(miner["damage"])
    exposure_factor = exposure_h / (window_s / 3600.0)
    return {
        "case": case.key,
        "ice_thickness_m": case.ice_thickness_m,
        "speed_kn": case.speed_kn,
        "weld_type": case.weld_type,
        "kt_toe": kt,
        "kt_hotspot_iiw": submodel.kt_hotspot_iiw,
        "kt_hotspot_dnv": submodel.kt_hotspot_dnv,
        "notch_depth_m": geometry.notch_depth_m,
        "toe_radius_m": geometry.toe_radius_m,
        "event_rate_hz": load["event_rate_hz"],
        "mean_pressure_pa": load["mean_pressure_pa"],
        "rms_hotspot_pa": float(np.std(hotspot)),
        "max_hotspot_pa": float(np.abs(hotspot).max()),
        "cycles_counted": float(miner["cycles"]),
        "damage_per_window": per_window,
        "damage_per_1000h": per_window * (1000.0 / (window_s / 3600.0)),
        "exposure_h": exposure_h,
        "design_damage": per_window * exposure_factor,
        "temperature_c": temperature_c,
        "sn_factor": miner["factor"],
        "energy_ledger": ledger,
        "window_s": window_s,
        "seed": int(seed),
    }


def run_ice_weld_fatigue_matrix(
    *,
    structure: PanelStructure | None = None,
    curve: DNVSNCurve | None = None,
    temperature_c: float = -20.0,
    exposure_h: float = 500.0,
    window_s: float = 3.0 * 3600.0,
    sample_rate_hz: float = 100.0,
    seed: int = 20261106,
) -> dict[str, object]:
    """Run the 18-case ice / speed / weld-type fatigue scan."""
    cases = [
        IceFatigueCase(thickness, speed, kind)
        for thickness in CASE_ICE_THICKNESS_M
        for speed in CASE_SPEED_KN
        for kind in CASE_WELD_TYPES
    ]
    rows = [
        fatigue_assessment(
            case,
            structure=structure,
            curve=curve,
            temperature_c=temperature_c,
            exposure_h=exposure_h,
            window_s=window_s,
            sample_rate_hz=sample_rate_hz,
            seed=seed,
        )
        for case in cases
    ]
    return {"cases": rows, "exposure_h": exposure_h, "window_s": window_s}