"""Hull-girder longitudinal-bending chain: wave load -> beam FE -> CSR/DNV check.

This module closes the longitudinal-strength chain for a tanker-type hull girder
(KVLCC2 calibration case) as one auditable pipeline:

1. Hull form: symmetric sectional-area curve calibrated to ``Cb`` with
   numerical volume checks (Cb, displacement, L/B/T ratios).
2. Still-water load: block weight distribution (steel / cargo / machinery /
   outfitting) and buoyancy ``rho g A(x)``; equilibrium draft and trim are
   solved so that net force and net moment both vanish.
3. Wave load: regular design wave with Smith correction; zero net force and
   zero net trim moment are enforced station by station.
4. Equivalent beam: 2-node Euler-Bernoulli (Hermite) finite elements with a
   varying ``EI(x)``; bending moment and stress are recovered from the FE
   curvature.
5. Midship section: plate/stiffener scantling set, inertia and section moduli
   by parallel-axis summation with an independent strip-integration
   cross-check; deck scantling closed against the IACS UR S11 minimum modulus.
6. Rule checks: DNV GL 2019 / IACS UR S11 wave bending moments, CSR allowable
   stress and safety factors.
7. Ultimate capacity: Smith-type incremental-curvature collapse with plate and
   column (Johnson-Ostenfeld) compression limits on the same section.

All quantities are SI (m, N, N.m, Pa).  Sign convention: transverse load
``q = buoyancy - weight`` is upward positive, deflection ``v`` upward positive,
so ``M = EI v'' > 0`` is **sagging** and ``M < 0`` is **hogging**; stress is
``sigma = -E v'' y`` with ``y`` upward from the neutral axis, hence the deck is
compressed while the hull sags and stretched while it hogs.

Scope and limitations: the hull is idealised as wall-sided and symmetric with a
smooth sectional-area curve; the scantling set is a CSR-consistent
reconstruction (the proprietary KRISO structural drawings are not public); the
beam model is linear elastic (no shear lag, no warping, no dynamic
amplification); local plate/stiffener buckling enters only through
Johnson-Ostenfeld compression limits.  See
``docs/hull-girder-csr-bending-chain.md`` for the full evidence trail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .hull_girder_progressive import HullComponent, solve_component_section

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

WATER_DENSITY = 1025.0            # kg/m3, sea water
GRAVITY = 9.81                    # m/s2
STEEL_E = 2.06e11                 # Pa
STEEL_NU = 0.3
MILD_STEEL_YIELD = 235.0e6        # Pa
CSR_GAMMA_M = 1.43                # material factor used by the mission spec
CSR_PERMISSIBLE_STRESS = 220.0e6  # Pa, CSR hull-girder permissible stress

#: KVLCC2 public particulars (KRISO hull form; structure is reconstructed here).
KVLCC2_PARTICULARS = {
    "name": "KVLCC2",
    "Lpp": 320.0,
    "B": 58.0,
    "D": 30.0,
    "T": 20.8,
    "Cb": 0.8095,
    "displacement_t": 320_000.0,
    "speed_kn": 15.5,
}


def _positive(name: str, value: object) -> float:
    """Return ``value`` as a finite positive float, otherwise raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return out


def _trapezoid(values: np.ndarray, x: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(x)))


def _cumulative_trapezoid(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(values)
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(x))
    return out


# ---------------------------------------------------------------------------
# 1. Rule wave bending moments (DNV GL 2019 / IACS UR S11)
# ---------------------------------------------------------------------------

def wave_coefficient(L: float) -> float:
    """Wave coefficient ``C`` of IACS UR S11 / DNV-RU-SHIP Pt.3 Ch.4 Sec.5."""
    L = _positive("L", L)
    if L < 90.0 or L > 500.0:
        raise ValueError("L outside the validity range 90 m <= L <= 500 m")
    if L < 300.0:
        return 10.75 - ((300.0 - L) / 100.0) ** 1.5
    if L <= 350.0:
        return 10.75
    return 10.75 - ((L - 350.0) / 150.0) ** 1.5


def rule_wave_bending_moment(
    *,
    L: float,
    B: float,
    Cb: float,
    f_nl: float = 1.0,
    f_m: float = 1.0,
    f_h: float = 1.0,
) -> dict:
    """Design vertical wave bending moments at midship (rule values, N.m).

    ``M_wh = 0.19 f_nl f_m f_h C L^2 B Cb``              [kN.m] (hogging)
    ``M_ws = 0.11 f_nl f_m f_h C L^2 B (Cb + 0.7)``     [kN.m] (sagging)

    The distribution factors are unity for a 320 m double-hull tanker
    (``f_nl = 1`` for ``L >= 300 m``, ``f_m = 1`` for ``Cb <= 0.9``, ``f_h = 1``
    for hull-girder assessment); they stay explicit arguments so the assumption
    is visible instead of hidden in a constant.
    """
    L = _positive("L", L)
    B = _positive("B", B)
    Cb = _positive("Cb", Cb)
    for name, value in (("f_nl", f_nl), ("f_m", f_m), ("f_h", f_h)):
        _positive(name, value)
    C = wave_coefficient(L)
    scale = f_nl * f_m * f_h * C * L ** 2 * B * 1.0e3  # kN.m -> N.m
    return {"C": C, "M_wh": 0.19 * scale * Cb, "M_ws": 0.11 * scale * (Cb + 0.7)}


def holtrop_wave_bending_moment(
    *,
    L: float,
    B: float,
    Cb: float,
    h_w: float,
    k_w: float = 1.0,
    rho: float = WATER_DENSITY,
    g: float = GRAVITY,
) -> float:
    """``M_w = k_w * rho * g * B * L^2 * Cb * h_w / 100``  [N.m].

    ``k_w`` is a non-dimensional shape/calibration coefficient.  ``k_w = 1``
    exposes the raw design-wave physics; use
    :func:`equivalent_wave_coefficient` for the value implied by the DNV/UR S11
    closed form.
    """
    L = _positive("L", L)
    B = _positive("B", B)
    Cb = _positive("Cb", Cb)
    h_w = _positive("h_w", h_w)
    rho = _positive("rho", rho)
    g = _positive("g", g)
    if isinstance(k_w, bool) or not isinstance(k_w, (int, float)) or not math.isfinite(k_w):
        raise ValueError("k_w must be a finite real scalar")
    return float(k_w) * rho * g * B * L ** 2 * Cb * h_w / 100.0


def equivalent_wave_coefficient(
    *,
    L: float,
    B: float,
    Cb: float,
    h_w: float,
    rho: float = WATER_DENSITY,
    g: float = GRAVITY,
) -> dict:
    """``k_w`` (hogging / sagging) implied by the DNV/UR S11 closed form.

    ``k_w = 100 M_wh / (rho g B L^2 Cb h_w)``.  For a 320 m tanker this is
    O(0.01): the ``k_w = 11.6`` used in earlier drafts of this chain is a
    screening-factor artefact that over-predicts the wave moment by an order of
    magnitude (75.8 GNm instead of 9.8 GNm at ``h_w = 13.5 m``).
    """
    rule = rule_wave_bending_moment(L=L, B=B, Cb=Cb)
    unit = holtrop_wave_bending_moment(L=L, B=B, Cb=Cb, h_w=h_w, k_w=1.0, rho=rho, g=g)
    return {"k_w_hog": rule["M_wh"] / unit, "k_w_sag": rule["M_ws"] / unit}# ---------------------------------------------------------------------------
# 2. Hull form: sectional-area curve with numerical closure checks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HullForm:
    """Symmetric, wall-sided hull described by a sectional-area curve.

    ``A(xi) = A_mid * sin(pi xi)**q`` with ``xi = x / Lpp`` from the aft
    perpendicular.  ``A_mid`` is scaled numerically so the integrated block
    coefficient equals the input ``Cb`` to integration accuracy; the curve is
    symmetric, so the buoyancy centre sits at ``Lpp/2``.
    """

    Lpp: float
    B: float
    D: float
    T: float
    Cb: float
    shape_exponent: float = 0.5
    n_x: int = 801
    rho: float = WATER_DENSITY
    g: float = GRAVITY

    def __post_init__(self) -> None:
        for name in ("Lpp", "B", "D", "T", "Cb", "shape_exponent", "rho", "g"):
            _positive(name, getattr(self, name))
        if not 0.3 <= self.Cb <= 0.9:
            raise ValueError("Cb outside the physically meaningful range 0.3..0.9")
        if self.D <= self.T:
            raise ValueError("moulded depth D must exceed the design draft T")
        if not isinstance(self.n_x, int) or self.n_x < 101:
            raise ValueError("n_x must be an integer >= 101")

    def x(self) -> np.ndarray:
        """Station positions from the AP (m)."""
        return np.linspace(0.0, self.Lpp, self.n_x)

    def sectional_area(self) -> np.ndarray:
        """Sectional area (m2) on :meth:`x`."""
        x = self.x()
        shape = np.sin(np.pi * x / self.Lpp) ** self.shape_exponent
        target_volume = self.Cb * self.B * self.T * self.Lpp
        area_mid = target_volume / _trapezoid(shape, x)
        return area_mid * shape

    def breadth(self) -> np.ndarray:
        """Waterline breadth (m) for wall-sided sections, ``A / T``."""
        return self.sectional_area() / self.T

    def verify(self) -> dict:
        """Geometric closure checks that must pass before any load analysis."""
        x = self.x()
        area = self.sectional_area()
        volume = _trapezoid(area, x)
        cb_int = volume / (self.Lpp * self.B * self.T)
        return {
            "L_over_B": self.Lpp / self.B,
            "B_over_T": self.B / self.T,
            "B_over_D": self.B / self.D,
            "Cb_input": self.Cb,
            "Cb_integrated": cb_int,
            "Cb_error": abs(cb_int - self.Cb) / self.Cb,
            "displacement_t": self.rho * volume / 1000.0,
            "lcb_fraction": _trapezoid(area * x, x) / (volume * self.Lpp),
            "max_section_area": float(area.max()),
        }


def kvlcc2_hull_form(**overrides) -> HullForm:
    """KVLCC2 hull form (public particulars, symmetric idealisation)."""
    data = dict(KVLCC2_PARTICULARS)
    fields = {key: data[key] for key in ("Lpp", "B", "D", "T", "Cb")}
    fields.update(overrides)
    return HullForm(**fields)


# ---------------------------------------------------------------------------
# 3. Weight distribution, buoyancy, equilibrium draft and trim
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeightItem:
    """A block mass spread over ``[x0, x1]`` from the AP.

    ``shape = "uniform"`` spreads it uniformly; ``shape = "hull_area"`` spreads
    it proportionally to the local sectional area (steel weight following the
    hull envelope).
    """

    name: str
    mass_t: float
    x0: float
    x1: float
    shape: str = "uniform"

    def __post_init__(self) -> None:
        _positive("mass_t", self.mass_t)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("weight item name must be a non-empty string")
        if not 0.0 <= float(self.x0) < float(self.x1):
            raise ValueError("weight item needs 0 <= x0 < x1")
        if self.shape not in ("uniform", "hull_area"):
            raise ValueError("shape must be 'uniform' or 'hull_area'")


def homogeneous_loaded_tanker_weights(form: HullForm) -> tuple[list[WeightItem], float]:
    """Block weights for a homogeneous-loaded VLCC (documented assumptions).

    Fractions of the design displacement: hull steel 0.20, cargo 0.70,
    machinery 0.035, outfitting/deckhouse 0.025, remaining systems 0.04.
    """
    delta_t = form.rho * form.Cb * form.Lpp * form.B * form.T / 1000.0
    L = form.Lpp
    items = [
        WeightItem("hull_steel", 0.200 * delta_t, 0.0, L, "hull_area"),
        WeightItem("remaining_system", 0.040 * delta_t, 0.0, L, "hull_area"),
        WeightItem("cargo", 0.700 * delta_t, 0.18 * L, 0.92 * L, "uniform"),
        WeightItem("machinery", 0.035 * delta_t, 0.05 * L, 0.20 * L, "uniform"),
        WeightItem("deckhouse_outfitting", 0.025 * delta_t, 0.0, 0.06 * L, "uniform"),
    ]
    return items, sum(item.mass_t for item in items)


def weight_per_length(form: HullForm, items: Sequence[WeightItem]) -> np.ndarray:
    """Weight distribution ``w(x)`` in N/m on :meth:`HullForm.x`."""
    x = form.x()
    area = form.sectional_area()
    total = np.zeros_like(x)
    for item in items:
        mask = (x >= item.x0) & (x <= item.x1)
        if not np.any(mask):
            raise ValueError(f"weight item {item.name!r} covers no station")
        weight_n = item.mass_t * 1000.0 * form.g
        # Normalise the block shape on the discrete stations (including the
        # half-interval at each block end) so the trapezoidal integral of the
        # block equals its mass exactly.
        shape = np.ones_like(x) if item.shape == "uniform" else area
        pattern = np.where(mask, shape, 0.0)
        total += weight_n * pattern / _trapezoid(pattern, x)
    return total


def weight_centroid(form: HullForm, items: Sequence[WeightItem]) -> float:
    """Longitudinal centre of gravity measured from the AP (m)."""
    x = form.x()
    w = weight_per_length(form, items)
    return _trapezoid(w * x, x) / _trapezoid(w, x)


def _buoyancy(form: HullForm, draft_scale: float, trim_shift: float) -> np.ndarray:
    """Buoyancy (N/m) for a scaled draft and a longitudinal shift."""
    x = form.x()
    area = form.sectional_area()
    shifted = np.interp(np.clip(x - trim_shift, 0.0, form.Lpp), x, area)
    return form.rho * form.g * shifted * draft_scale


def solve_equilibrium_draft_trim(
    form: HullForm,
    total_mass_t: float,
    lcg_m: float,
    *,
    iterations: int = 60,
) -> dict:
    """Solve draft scale and longitudinal shift so buoyancy balances the mass.

    Unknowns ``[draft_scale, trim_shift]``; residuals are net vertical force and
    net moment about the AP.  Damped Newton iteration with numerical Jacobian.
    """
    g = form.g
    target_force = total_mass_t * 1000.0 * g
    target_moment = target_force * lcg_m
    x = form.x()
    scale = np.array([target_force, target_force * form.Lpp])

    def residual(p: np.ndarray) -> np.ndarray:
        b = _buoyancy(form, p[0], p[1])
        return np.array([_trapezoid(b, x) - target_force,
                         _trapezoid(b * x, x) - target_moment])

    p = np.array([1.0, 0.0])
    for _ in range(iterations):
        r = residual(p) / scale
        if float(np.linalg.norm(r)) < 1e-13:
            break
        jac = np.zeros((2, 2))
        step = np.array([1e-6, 1e-4])
        for j in range(2):
            dp = np.zeros(2)
            dp[j] = step[j]
            jac[:, j] = (residual(p + dp) - residual(p)) / (step[j] * scale)
        try:
            delta = np.linalg.solve(jac, -r)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate geometry
            raise RuntimeError("equilibrium Jacobian is singular")
        p = p + 0.8 * delta
        if not np.all(np.isfinite(p)):
            raise RuntimeError("draft/trim equilibrium diverged")
    b = _buoyancy(form, p[0], p[1])
    force_residual = _trapezoid(b, x) - target_force
    moment_residual = _trapezoid(b * x, x) - target_moment
    return {
        "draft_scale": float(p[0]),
        "draft_m": form.T * float(p[0]),
        "trim_shift_m": float(p[1]),
        "trim_m": float(p[1]),
        "force_residual_rel": abs(force_residual) / target_force,
        "moment_residual_rel": abs(moment_residual) / abs(target_moment),
        "total_mass_t": total_mass_t,
        "lcg_m": lcg_m,
        "buoyancy_N_per_m": b,
    }


# ---------------------------------------------------------------------------
# 4. Wave load distributed along the hull (design wave + Smith correction)
# ---------------------------------------------------------------------------

def wave_load_distribution(
    form: HullForm,
    *,
    h_w: float,
    wavelength: float | None = None,
    crest_position: float | None = None,
    smith_correction: bool = True,
) -> np.ndarray:
    """Wave-induced load ``q_W(x)`` (N/m, upward positive).

    A regular design wave ``eta = (h_w/2) cos(2 pi (x - x_c)/lambda)`` with crest
    at ``x_c`` (default midships, the hogging case) changes the immersed area by
    ``delta_A = B_wl eta exp(-kT)`` (wall-sided section, Smith correction
    ``k = 2 pi / lambda``).  A wave cannot change displacement or trim, so the
    net force and net moment of ``delta_A`` are removed by subtracting the linear
    draft/trim correction before moment integration.
    """
    h_w = _positive("h_w", h_w)
    wavelength = form.Lpp if wavelength is None else _positive("wavelength", wavelength)
    crest = form.Lpp / 2.0 if crest_position is None else float(crest_position)
    x = form.x()
    k = 2.0 * math.pi / wavelength
    eta = 0.5 * h_w * np.cos(k * (x - crest))
    attenuation = math.exp(-k * form.T) if smith_correction else 1.0
    delta_area = form.breadth() * eta * attenuation
    length = form.Lpp
    a0 = _trapezoid(delta_area, x) / length
    a1 = _trapezoid(delta_area * (x - length / 2.0), x) * 12.0 / length ** 3
    delta_area = delta_area - a0 - a1 * (x - length / 2.0)
    return form.rho * form.g * delta_area


# ---------------------------------------------------------------------------
# 5. Equivalent-beam Euler-Bernoulli finite elements with varying EI
# ---------------------------------------------------------------------------

def beam_moment_from_load(x: np.ndarray, q: np.ndarray) -> dict:
    """Closed-form double integration for a simply supported beam.

    ``V = int_0^x q ds``, ``M = int_0^x V ds - (x/L) int_0^L V ds``: the linear
    correction enforces ``M(0) = M(L) = 0`` and is exact for the given load.
    """
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    shear = _cumulative_trapezoid(q, x)
    moment = _cumulative_trapezoid(shear, x)
    length = x[-1] - x[0]
    moment = moment - moment[-1] * (x - x[0]) / length
    return {"shear": shear, "moment": moment}


def beam_fem(
    x_nodes: np.ndarray,
    q_nodes: np.ndarray,
    ei_nodes: np.ndarray,
    *,
    supports: tuple[int, int] = (0, -1),
) -> dict:
    """2-node Hermite beam FE with linearly varying load and EI.

    DOFs per node: deflection ``v`` (upward positive) and rotation
    ``theta = dv/dx``.  Element stiffness uses the midpoint ``EI``; the
    consistent load vector uses 2-point Gauss quadrature (exact for a linear
    load on a cubic Hermite field).  Returns nodal deflection, rotation,
    recovered moment ``M = EI v''`` and the support reactions.  Simple supports
    (``v = 0``) are applied at ``supports``.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    q_nodes = np.asarray(q_nodes, dtype=float)
    ei_nodes = np.asarray(ei_nodes, dtype=float)
    if not (x_nodes.ndim == q_nodes.ndim == ei_nodes.ndim == 1):
        raise ValueError("beam inputs must be one-dimensional")
    if not (len(x_nodes) == len(q_nodes) == len(ei_nodes)) or len(x_nodes) < 3:
        raise ValueError("beam inputs must share a length >= 3")
    if np.any(np.diff(x_nodes) <= 0.0):
        raise ValueError("beam nodes must be strictly increasing")
    if np.any(ei_nodes <= 0.0):
        raise ValueError("EI must be positive everywhere")

    n_el = len(x_nodes) - 1
    n_dof = 2 * len(x_nodes)
    stiffness = np.zeros((n_dof, n_dof))
    load = np.zeros(n_dof)
    gp = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))
    n1 = lambda s: 1.0 - 3.0 * s ** 2 + 2.0 * s ** 3
    n2 = lambda s, le: le * (s - 2.0 * s ** 2 + s ** 3)
    n3 = lambda s: 3.0 * s ** 2 - 2.0 * s ** 3
    n4 = lambda s, le: le * (-s ** 2 + s ** 3)
    for e in range(n_el):
        le = x_nodes[e + 1] - x_nodes[e]
        ei = 0.5 * (ei_nodes[e] + ei_nodes[e + 1])
        ke = ei / le ** 3 * np.array([
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le ** 2, -6.0 * le, 2.0 * le ** 2],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le ** 2, -6.0 * le, 4.0 * le ** 2],
        ])
        dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        stiffness[np.ix_(dofs, dofs)] += ke
        fe = np.zeros(4)
        for s in gp:
            q_s = (1.0 - s) * q_nodes[e] + s * q_nodes[e + 1]
            shape = np.array([n1(s), n2(s, le), n3(s), n4(s, le)])
            fe += shape * q_s * le * 0.5
        load[dofs] += fe

    n_nodes = len(x_nodes)
    fixed = sorted({2 * (index % n_nodes) for index in supports})
    free = np.setdiff1d(np.arange(n_dof), np.array(fixed))
    displacement = np.zeros(n_dof)
    displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], load[free])
    reaction = stiffness @ displacement - load

    moment_nodes = np.zeros(n_nodes)
    for e in range(n_el):
        le = x_nodes[e + 1] - x_nodes[e]
        ei = 0.5 * (ei_nodes[e] + ei_nodes[e + 1])
        d = displacement[2 * e:2 * e + 4]
        for s, node in ((0.0, e), (1.0, e + 1)):
            v_second = ((-6.0 + 12.0 * s) * d[0] + le * (-4.0 + 6.0 * s) * d[1]
                        + (6.0 - 12.0 * s) * d[2] + le * (-2.0 + 6.0 * s) * d[3]) / le ** 2
            moment_nodes[node] = ei * v_second
    return {
        "displacement": displacement[0::2],
        "rotation": displacement[1::2],
        "moment": moment_nodes,
        "reaction": np.array([reaction[2 * (index % n_nodes)] for index in supports]),
    }


def section_inertia_profile(form: HullForm, i_mid: float) -> np.ndarray:
    """Approximate ``I(x)`` from the sectional-area curve and local depth.

    ``I(x)/I_mid = (A(x)/A_mid) (D(x)/D)^2`` with
    ``D(x) = D (0.55 + 0.45 A(x)/A_mid)``: a documented idealisation of the loss
    of depth and material towards the ends (sheer, rise of floor).  Only the
    midship value ``I_mid`` is calibrated, and it comes from the scantling set.
    """
    i_mid = _positive("i_mid", i_mid)
    area = form.sectional_area()
    # 2% floor: the extreme stations carry only local structure, and a strictly
    # positive EI is required by the beam finite element (documented
    # idealisation; it has no influence on the midship moment).
    ratio = np.clip(area / area.max(), 0.02, 1.0)
    depth_ratio = 0.55 + 0.45 * ratio
    return i_mid * ratio * depth_ratio ** 2# ---------------------------------------------------------------------------
# 6. Midship section: scantlings, inertia and section modulus
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionMember:
    """Longitudinal member lumped at its centroid.

    ``vertical_span`` (m) is the vertical extent; the member's own second moment
    ``area * vertical_span^2 / 12`` is then included (negligible for horizontal
    plating, where the own inertia is ``breadth * t^3 / 12``).
    """

    name: str
    area: float
    y: float
    vertical_span: float = 0.0
    thickness: float = 0.0
    stiffener_spacing: float = 0.9

    def __post_init__(self) -> None:
        _positive("area", self.area)
        _positive("thickness", self.thickness)
        _positive("stiffener_spacing", self.stiffener_spacing)
        if isinstance(self.y, bool) or not isinstance(self.y, (int, float)):
            raise TypeError("y must be a real scalar")
        if not math.isfinite(float(self.y)):
            raise ValueError("y must be finite")
        if float(self.vertical_span) < 0.0:
            raise ValueError("vertical_span must be non-negative")

    @property
    def own_inertia(self) -> float:
        if self.vertical_span:
            return self.area * self.vertical_span ** 2 / 12.0
        return 0.0


@dataclass(frozen=True)
class MidshipSection:
    """Longitudinal members of a midship section plus derived properties."""

    members: tuple[SectionMember, ...]
    young: float = STEEL_E
    yield_stress: float = MILD_STEEL_YIELD

    def __post_init__(self) -> None:
        if len(self.members) < 4:
            raise ValueError("a midship section needs at least four members")
        if len({m.name for m in self.members}) != len(self.members):
            raise ValueError("member names must be unique")
        _positive("young", self.young)
        _positive("yield_stress", self.yield_stress)

    def total_area(self) -> float:
        return sum(m.area for m in self.members)

    def neutral_axis(self) -> float:
        return sum(m.area * m.y for m in self.members) / self.total_area()

    def moment_of_inertia(self) -> float:
        y_na = self.neutral_axis()
        return sum(m.area * (m.y - y_na) ** 2 + m.own_inertia for m in self.members)

    def extreme_fibres(self) -> tuple[float, float]:
        """``(y_top, y_bottom)`` distances from the neutral axis (m)."""
        y_na = self.neutral_axis()
        top = max(m.y + 0.5 * m.vertical_span for m in self.members)
        bottom = min(m.y - 0.5 * m.vertical_span for m in self.members)
        return top - y_na, y_na - bottom

    def section_moduli(self) -> tuple[float, float]:
        """``(W_deck, W_bottom)`` in m3."""
        y_top, y_bottom = self.extreme_fibres()
        inertia = self.moment_of_inertia()
        return inertia / y_top, inertia / y_bottom

    def strip_inertia(self, strips: int = 200) -> float:
        """Independent strip integration of the same members.

        Vertical members are cut into ``strips`` equal-area slices, the centroid
        is recomputed from the slices and the inertia is summed slice by slice.
        Used to cross-check the closed-form parallel-axis summation.
        """
        if not isinstance(strips, int) or strips < 8:
            raise ValueError("strips must be an integer >= 8")
        areas, ys = [], []
        for m in self.members:
            if m.vertical_span:
                dy = m.vertical_span / strips
                centres = (m.y - m.vertical_span / 2.0) + (np.arange(strips) + 0.5) * dy
                areas.extend([m.area / strips] * strips)
                ys.extend(centres.tolist())
            else:
                areas.append(m.area)
                ys.append(m.y)
        areas = np.asarray(areas)
        ys = np.asarray(ys)
        y_na = float(np.sum(areas * ys) / np.sum(areas))
        return float(np.sum(areas * (ys - y_na) ** 2))

    def axial_residual(self) -> float:
        """``|int y dA| / int |y| dA`` about the neutral axis (must be ~0)."""
        y_na = self.neutral_axis()
        scale = sum(m.area * (abs(m.y - y_na) + 0.5 * m.vertical_span) for m in self.members)
        first = sum(m.area * (m.y - y_na) for m in self.members)
        return abs(first) / scale if scale else 0.0


def kvlcc2_midship_section(deck_thickness: float = 0.030) -> MidshipSection:
    """CSR-consistent KVLCC2 midship scantling set (documented reconstruction).

    Longitudinal members follow CSR minimum-thickness practice for a 320 m
    double-hull tanker: deck 30 mm plus bulb-flat longitudinals at 0.9 m
    spacing, bottom shell 32 mm, inner bottom 28 mm, side shell 26 mm, two
    longitudinal bulkheads 24 mm, hopper plating 24 mm, double-bottom depth
    2.7 m, moulded depth 30 m, inner bottom breadth 46 m.  The set is
    *reconstructed*, not copied from the proprietary KRISO structural drawings;
    the deck plate thickness is the free parameter used to close the section
    modulus against the IACS UR S11 minimum (see :func:`close_deck_scantling`).

    Members that lump a stiffener set carry the thickness of the plating they
    are attached to, because their compression limit is governed by that plating
    panel and by the stiffener column, never by a fictitious thin panel of the
    stiffener area.
    """
    deck_thickness = _positive("deck_thickness", deck_thickness)
    B, D = 58.0, 30.0
    side_height = D - 2.7
    side_centre = 0.5 * (2.7 + D)
    members = (
        SectionMember("deck_plate", B * deck_thickness, D - 0.5 * deck_thickness,
                      thickness=deck_thickness),
        SectionMember("deck_longitudinals", (B / 0.9) * 0.012, D - 0.20,
                      thickness=deck_thickness),
        SectionMember("bottom_shell", B * 0.032, 0.016, thickness=0.032),
        SectionMember("bottom_longitudinals", (B / 0.9) * 0.012, 0.30,
                      thickness=0.032),
        SectionMember("inner_bottom", 46.0 * 0.028, 2.714, thickness=0.028),
        SectionMember("inner_bottom_longitudinals", (46.0 / 0.9) * 0.010, 2.55,
                      thickness=0.028),
        SectionMember("hopper_plating", 2 * 5.5 * 0.024, 1.35,
                      vertical_span=2.7, thickness=0.024),
        SectionMember("side_shell", 2 * side_height * 0.026, side_centre,
                      vertical_span=side_height, thickness=0.026),
        SectionMember("side_longitudinals", 2 * (side_height / 0.9) * 0.010,
                      side_centre, vertical_span=side_height, thickness=0.026),
        SectionMember("longitudinal_bulkheads", 2 * side_height * 0.024,
                      side_centre, vertical_span=side_height, thickness=0.024),
    )
    return MidshipSection(members)


def ur_s11_minimum_section_modulus(*, L: float, B: float, Cb: float) -> float:
    """IACS UR S11 minimum midship section modulus at deck (m3).

    ``W_min = C L^2 B (Cb + 0.7)`` [cm3], converted to m3.  For KVLCC2
    (``L = 320 m``, ``B = 58 m``, ``Cb = 0.8095``, ``C = 10.75``) this gives
    96.4 m3, a physically meaningful deck modulus for a 320 kt tanker and the
    reason the earlier ``I_ref = 1.15e7 m4`` reference used in this repository
    is discarded (it is three orders of magnitude too large).
    """
    L = _positive("L", L)
    B = _positive("B", B)
    Cb = _positive("Cb", Cb)
    return wave_coefficient(L) * L ** 2 * B * (Cb + 0.7) * 1.0e-6


def close_deck_scantling(
    section: MidshipSection | None = None,
    *,
    target_modulus: float,
    max_iterations: int = 60,
) -> tuple[MidshipSection, dict]:
    """Thicken the deck plating until ``W_deck >= target_modulus``.

    A scalar bisection on the deck plate thickness (longitudinals follow it) is
    used; the returned dict is the audit record of the closure loop.
    """
    target_modulus = _positive("target_modulus", target_modulus)
    base = kvlcc2_midship_section() if section is None else section
    w_start, _ = base.section_moduli()
    if w_start >= target_modulus:
        return base, {
            "iterations": 0,
            "W_deck_start": w_start,
            "thickness_start": 0.030,
            "thickness_final": 0.030,
            "W_deck_final": w_start,
            "target_modulus": target_modulus,
            "converged": True,
        }
    lo, hi = 0.030, 0.080
    history = []
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        trial = kvlcc2_midship_section(deck_thickness=mid)
        w_deck, _ = trial.section_moduli()
        history.append({"thickness": mid, "W_deck": w_deck})
        if w_deck >= target_modulus:
            hi = mid
        else:
            lo = mid
    final = kvlcc2_midship_section(deck_thickness=hi)
    w_deck, _ = final.section_moduli()
    return final, {
        "iterations": len(history),
        "W_deck_start": w_start,
        "thickness_start": 0.030,
        "thickness_final": hi,
        "W_deck_final": w_deck,
        "target_modulus": target_modulus,
        "converged": w_deck >= target_modulus,
        "history": history[-6:],
    }


# ---------------------------------------------------------------------------
# 7. CSR / DNV rule checks
# ---------------------------------------------------------------------------

def csr_allowable_stress(
    yield_stress: float = MILD_STEEL_YIELD,
    *,
    gamma_m: float = CSR_GAMMA_M,
    permissible_stress: float = CSR_PERMISSIBLE_STRESS,
) -> dict:
    """Allowable hull-girder normal stress (Pa).

    Two criteria are reported and the smaller one governs: the mission /
    limit-state form ``sigma_y / gamma_m`` (material factor 1.43) and the CSR
    hull-girder permissible stress for mild steel (220 MPa).  ``SF =
    sigma_allow / sigma``.
    """
    yield_stress = _positive("yield_stress", yield_stress)
    gamma_m = _positive("gamma_m", gamma_m)
    permissible_stress = _positive("permissible_stress", permissible_stress)
    sigma_gamma = yield_stress / gamma_m
    return {
        "sigma_y": yield_stress,
        "gamma_m": gamma_m,
        "sigma_allow_gamma": sigma_gamma,
        "sigma_allow_csr": permissible_stress,
        "sigma_allow": min(sigma_gamma, permissible_stress),
        "governing": "sigma_y/gamma_m" if sigma_gamma <= permissible_stress else "CSR 220 MPa",
    }


def csr_check(
    *,
    moment: float,
    section_modulus: float,
    yield_stress: float = MILD_STEEL_YIELD,
    gamma_m: float = CSR_GAMMA_M,
    permissible_stress: float = CSR_PERMISSIBLE_STRESS,
    ultimate_moment: float | None = None,
) -> dict:
    """CSR / DNV limit-state check of one governing bending case."""
    moment = abs(float(moment))
    section_modulus = _positive("section_modulus", section_modulus)
    allowable = csr_allowable_stress(
        yield_stress, gamma_m=gamma_m, permissible_stress=permissible_stress
    )
    sigma = moment / section_modulus
    out = {
        "moment": moment,
        "section_modulus": section_modulus,
        "sigma": sigma,
        "sigma_allow": allowable["sigma_allow"],
        "sigma_allow_gamma": allowable["sigma_allow_gamma"],
        "sigma_allow_csr": allowable["sigma_allow_csr"],
        "governing": allowable["governing"],
        "safety_factor": allowable["sigma_allow"] / sigma if sigma > 0.0 else float("inf"),
        "passes": sigma <= allowable["sigma_allow"],
    }
    if ultimate_moment is not None:
        ultimate_moment = _positive("ultimate_moment", ultimate_moment)
        out["ultimate_moment"] = ultimate_moment
        out["utilisation"] = moment / ultimate_moment
        out["ultimate_passes"] = moment / ultimate_moment <= 1.0
    return out


# ---------------------------------------------------------------------------
# 8. Ultimate bending capacity (Smith-type incremental curvature)
# ---------------------------------------------------------------------------

def johnson_ostenfeld(sigma_y: float, sigma_e: float) -> float:
    """Johnson-Ostenfeld inelastic buckling strength (Pa)."""
    sigma_y = _positive("sigma_y", sigma_y)
    sigma_e = _positive("sigma_e", sigma_e)
    if sigma_e >= 0.5 * sigma_y:
        return sigma_y * (1.0 - sigma_y / (4.0 * sigma_e))
    return sigma_e


def member_compression_strength(
    member: SectionMember,
    *,
    young: float = STEEL_E,
    poisson: float = STEEL_NU,
    yield_stress: float = MILD_STEEL_YIELD,
    strut_length: float = 5.0,
    strut_radius_of_gyration: float = 0.055,
) -> dict:
    """Compression strength of a stiffened plate member (plate and column).

    Plate Euler stress (panel, ``k = 4``)::

        sigma_E,plate = 4 pi^2 E / (12 (1 - nu^2)) (t/b)^2

    Column Euler stress of the stiffener plus effective plating as a strut
    between transverse frames (frame spacing 5.0 m, radius of gyration 0.055 m
    for a 400 mm bulb flat with effective plating)::

        sigma_E,col = pi^2 E r^2 / l^2

    Governing strength is ``min(JO(plate), JO(column), sigma_y)``; tensile
    strength stays ``sigma_y``.  Stiffener tripping and tripping/plate
    interaction are excluded (see module limitations).
    """
    young = _positive("young", young)
    yield_stress = _positive("yield_stress", yield_stress)
    strut_length = _positive("strut_length", strut_length)
    strut_radius_of_gyration = _positive("strut_radius_of_gyration", strut_radius_of_gyration)
    t = _positive("thickness", member.thickness)
    b = _positive("stiffener_spacing", member.stiffener_spacing)
    sigma_e_plate = 4.0 * math.pi ** 2 * young / (12.0 * (1.0 - poisson ** 2)) * (t / b) ** 2
    sigma_e_col = math.pi ** 2 * young * strut_radius_of_gyration ** 2 / strut_length ** 2
    sigma_plate = johnson_ostenfeld(yield_stress, sigma_e_plate)
    sigma_col = johnson_ostenfeld(yield_stress, sigma_e_col)
    return {
        "sigma_E_plate": sigma_e_plate,
        "sigma_E_column": sigma_e_col,
        "sigma_compression": min(sigma_plate, sigma_col, yield_stress),
        "governing": "plate" if sigma_plate <= sigma_col else "column",
    }


def _collapse_components(
    section: MidshipSection,
    *,
    mirror: bool = False,
    strips_vertical: int = 6,
    **strength_kwargs,
) -> tuple[HullComponent, ...]:
    """Build :class:`HullComponent` objects (optionally mirrored in y)."""
    components: list[HullComponent] = []
    sign = -1.0 if mirror else 1.0
    for member in section.members:
        strength = member_compression_strength(
            member, young=section.young, yield_stress=section.yield_stress,
            **strength_kwargs
        )
        if member.vertical_span:
            dy = member.vertical_span / strips_vertical
            for i in range(strips_vertical):
                y = member.y - member.vertical_span / 2.0 + (i + 0.5) * dy
                components.append(HullComponent(
                    f"{member.name}_{i}", member.area / strips_vertical, sign * y,
                    section.young, section.yield_stress,
                    compression_strength=strength["sigma_compression"],
                ))
        else:
            components.append(HullComponent(
                member.name, member.area, sign * member.y, section.young,
                section.yield_stress,
                compression_strength=strength["sigma_compression"],
            ))
    return tuple(components)


def hull_girder_ultimate_moment(
    section: MidshipSection,
    *,
    direction: str = "hog",
    n_curvatures: int = 60,
    curvature_factor: float = 12.0,
    **strength_kwargs,
) -> dict:
    """Ultimate vertical bending moment by incremental-curvature collapse.

    The section is loaded with plane sections until full plasticity: each
    component is elastic-perfectly plastic in tension (``sigma_y``) and limited
    in compression by :func:`member_compression_strength`.  ``direction="hog"``
    puts the deck in compression; ``"sag"`` mirrors the section so the bottom is
    compressed.  Returns ``M_u``, the moment-curvature curve, the first-yield
    moment and the elastic yield moment ``sigma_y I / y``.
    """
    if direction not in ("hog", "sag"):
        raise ValueError("direction must be 'hog' or 'sag'")
    if not isinstance(n_curvatures, int) or n_curvatures < 10:
        raise ValueError("n_curvatures must be an integer >= 10")
    curvature_factor = _positive("curvature_factor", curvature_factor)
    components = _collapse_components(section, mirror=direction == "sag", **strength_kwargs)
    y_max = max(abs(c.y) for c in components)
    kappa_y = section.yield_stress / (section.young * y_max)
    curvatures = [0.0] + [
        kappa_y * curvature_factor * (i / n_curvatures) ** 1.5
        for i in range(1, n_curvatures + 1)
    ]
    points = solve_component_section(components, curvatures)
    moments = np.array([p.moment for p in points])
    index = int(np.argmax(moments))
    first_yield = next(
        (p for p in points if any(s.mode != "elastic" for s in p.states)), points[-1]
    )
    inertia = section.moment_of_inertia()
    # First-yield moment of this bending direction: the component that reaches
    # its own limit (tension sigma_y, compression sigma_c) earliest governs.
    # First-yield moment of this bending direction, defined at the extreme
    # fibres (members within 75% of the maximum lever arm), which is the
    # hull-girder convention: sigma_y * W_deck for hogging and the compressive
    # limit * W_deck (deck) or sigma_y * W_bottom (bottom), whichever is smaller.
    y_na_local = sum(c.area * c.y for c in components) / sum(c.area for c in components)
    extent = max(abs(c.y - y_na_local) for c in components)
    yield_moment = min(
        (c.yield_stress if c.y > y_na_local else c.compression_strength)
        * inertia / abs(c.y - y_na_local)
        for c in components
        if abs(c.y - y_na_local) >= 0.75 * extent
    )
    return {
        "direction": direction,
        "M_ultimate": float(moments[index]),
        "curvature_ultimate": float(points[index].curvature),
        "M_first_yield": float(first_yield.moment),
        "M_elastic_yield": float(yield_moment),
        "shape_factor": float(moments[index] / yield_moment),
        "axial_residual_rel": float(max(abs(p.axial_residual) for p in points)
                                    / sum(c.area * c.yield_stress for c in components)),
        "curvature": [p.curvature for p in points],
        "moment_curve": moments.tolist(),
        "compression_limits": {
            c.name: c.compression_strength for c in components
        },
    }# ---------------------------------------------------------------------------
# 9. Full-chain driver for the KVLCC2 calibration case
# ---------------------------------------------------------------------------

def run_kvlcc2_csr_chain(
    *,
    design_wave_height: float = 13.5,
    n_elements: int = 320,
    gamma_m: float = CSR_GAMMA_M,
    permissible_stress: float = CSR_PERMISSIBLE_STRESS,
) -> dict:
    """Run the complete wave-load -> beam FE -> CSR chain for KVLCC2.

    Parameters
    ----------
    design_wave_height:
        Physically based 100-year North Atlantic design wave height (m).  Its
        *rule-equivalent* value is derived inside the chain; the raw value is
        kept so the calibration factor can be reported.
    n_elements:
        Number of beam finite elements along the hull.

    Returns a dictionary with every acceptance number (C1..C5) and the
    distributions needed for plotting.
    """
    design_wave_height = _positive("design_wave_height", design_wave_height)
    if not isinstance(n_elements, int) or n_elements < 40:
        raise ValueError("n_elements must be an integer >= 40")

    # -- geometry ----------------------------------------------------------
    form = kvlcc2_hull_form()
    geometry = form.verify()
    if geometry["Cb_error"] > 1e-3:
        raise RuntimeError("sectional-area curve does not reproduce Cb")

    # -- weights, buoyancy, equilibrium -------------------------------------
    items, total_mass_t = homogeneous_loaded_tanker_weights(form)
    lcg = weight_centroid(form, items)
    equilibrium = solve_equilibrium_draft_trim(form, total_mass_t, lcg)
    weight = weight_per_length(form, items)
    buoyancy = equilibrium.pop("buoyancy_N_per_m")
    x = form.x()
    q_still_water = buoyancy - weight

    # -- rule wave moment and design-wave calibration -----------------------
    rule = rule_wave_bending_moment(L=form.Lpp, B=form.B, Cb=form.Cb)
    k_w = equivalent_wave_coefficient(
        L=form.Lpp, B=form.B, Cb=form.Cb, h_w=design_wave_height
    )
    # The chain keeps the *physical* design wave; its rule-equivalent height is
    # reported only as a consistency diagnostic, never used to force agreement.
    q_wave = wave_load_distribution(form, h_w=design_wave_height)
    m_wave = beam_moment_from_load(x, q_wave)["moment"]
    mid = int(np.argmin(np.abs(x - form.Lpp / 2.0)))
    raw_mid = abs(float(m_wave[mid]))
    h_w_equivalent = design_wave_height * rule["M_wh"] / raw_mid

    # -- midship section and CSR modulus closure ----------------------------
    w_min = ur_s11_minimum_section_modulus(L=form.Lpp, B=form.B, Cb=form.Cb)
    base_section = kvlcc2_midship_section()
    section, closure = close_deck_scantling(base_section, target_modulus=w_min)
    inertia = section.moment_of_inertia()
    strip_inertia = section.strip_inertia()
    w_deck, w_bottom = section.section_moduli()
    y_top, y_bottom = section.extreme_fibres()
    y_na = section.neutral_axis()

    # -- beam FE with EI(x) -------------------------------------------------
    i_profile = section_inertia_profile(form, inertia)
    ei_profile = section.young * i_profile
    nodes = np.linspace(0.0, form.Lpp, n_elements + 1)
    q_sw_nodes = np.interp(nodes, x, q_still_water)
    q_w_nodes = np.interp(nodes, x, q_wave)
    ei_nodes = np.interp(nodes, x, ei_profile)

    fem_sw = beam_fem(nodes, q_sw_nodes, ei_nodes)
    fem_w = beam_fem(nodes, q_w_nodes, ei_nodes)
    fem_hog = beam_fem(nodes, q_sw_nodes + q_w_nodes, ei_nodes)
    fem_sag = beam_fem(nodes, q_sw_nodes - q_w_nodes, ei_nodes)
    analytic_sw = beam_moment_from_load(x, q_still_water)["moment"]
    analytic_w = beam_moment_from_load(x, q_wave)["moment"]
    analytic_hog = beam_moment_from_load(x, q_still_water + q_wave)["moment"]
    analytic_sag = beam_moment_from_load(x, q_still_water - q_wave)["moment"]

    e_profile = np.interp(nodes, x, np.full_like(x, section.young))
    i_nodes = np.interp(nodes, x, i_profile)

    def stress_profile(moment_nodes: np.ndarray, y_extreme: float) -> np.ndarray:
        """``sigma = -E v'' y = -M y / I`` recovered from the FE solution."""
        return -moment_nodes * y_extreme / i_nodes

    mid_node = int(np.argmin(np.abs(nodes - form.Lpp / 2.0)))
    m_sw_mid = float(fem_sw["moment"][mid_node])
    m_w_mid = float(fem_w["moment"][mid_node])
    m_hog_mid = float(fem_hog["moment"][mid_node])   # negative = hogging
    m_sag_mid = float(fem_sag["moment"][mid_node])   # positive = sagging
    m_total_hog = -m_hog_mid
    m_total_sag = m_sag_mid

    sigma_deck_hog = float(stress_profile(fem_hog["moment"], y_top)[mid_node])
    sigma_bottom_hog = float(stress_profile(fem_hog["moment"], -y_bottom)[mid_node])
    sigma_deck_sag = float(stress_profile(fem_sag["moment"], y_top)[mid_node])
    sigma_bottom_sag = float(stress_profile(fem_sag["moment"], -y_bottom)[mid_node])

    # -- solver verification against the closed-form integration ------------
    analytic_hog_nodes = np.interp(nodes, x, analytic_hog)
    analytic_sag_nodes = np.interp(nodes, x, analytic_sag)
    scale = max(abs(analytic_hog).max(), abs(analytic_sag).max())
    beam_error = max(
        float(np.max(np.abs(fem_hog["moment"] - analytic_hog_nodes))),
        float(np.max(np.abs(fem_sag["moment"] - analytic_sag_nodes))),
    ) / scale
    sigma_analytic_deck_hog = -m_hog_mid * y_top / inertia
    sigma_error = abs(sigma_deck_hog - sigma_analytic_deck_hog) / abs(sigma_analytic_deck_hog)
    sigma_analytic_bottom_sag = -m_sag_mid * (-y_bottom) / inertia
    sigma_error_bottom = abs(sigma_bottom_sag - sigma_analytic_bottom_sag) / abs(sigma_analytic_bottom_sag)

    # -- ultimate capacity --------------------------------------------------
    ultimate_hog = hull_girder_ultimate_moment(section, direction="hog")
    ultimate_sag = hull_girder_ultimate_moment(section, direction="sag")

    # -- CSR checks ---------------------------------------------------------
    check_hog = csr_check(
        moment=m_total_hog, section_modulus=w_deck, gamma_m=gamma_m,
        permissible_stress=permissible_stress,
        ultimate_moment=ultimate_hog["M_ultimate"],
    )
    check_sag = csr_check(
        moment=m_total_sag, section_modulus=w_bottom, gamma_m=gamma_m,
        permissible_stress=permissible_stress,
        ultimate_moment=ultimate_sag["M_ultimate"],
    )

    # -- C1..C5 acceptance numbers -----------------------------------------
    c1_error = float(abs(inertia - strip_inertia) / strip_inertia)
    # DNV-composed design moments: still-water moment (sagging positive) plus the
    # rule wave moment in the appropriate sense.
    m_total_hog_rule = abs(m_sw_mid - rule["M_wh"])
    m_total_sag_rule = abs(m_sw_mid + rule["M_ws"])
    c2_reference = m_total_hog_rule
    c2_error = float(abs(m_total_hog - m_total_hog_rule) / m_total_hog_rule)
    c2_error_sag = float(abs(m_total_sag - m_total_sag_rule) / m_total_sag_rule)
    c3_error = float(max(beam_error, sigma_error, sigma_error_bottom))
    criteria = {
        "C1_I_midship_error": c1_error,
        "C1_pass": c1_error <= 0.05,
        "C2_M_total_hog_vs_DNV_error": c2_error,
        "C2_M_total_sag_vs_DNV_error": c2_error_sag,
        "C2_pass": c2_error <= 0.10 and c2_error_sag <= 0.10,
        "C3_stress_vs_analytic_error": c3_error,
        "C3_pass": c3_error <= 0.01,
        "C4_safety_factor_hog": check_hog["safety_factor"],
        "C4_safety_factor_sag": check_sag["safety_factor"],
        "C4_pass": check_hog["safety_factor"] >= 1.0 and check_sag["safety_factor"] >= 1.0,
        "C5_utilisation_hog": check_hog["utilisation"],
        "C5_utilisation_sag": check_sag["utilisation"],
        "C5_pass": max(check_hog["utilisation"], check_sag["utilisation"]) <= 1.0,
    }
    criteria["all_pass"] = all(
        criteria[key] for key in ("C1_pass", "C2_pass", "C3_pass", "C4_pass", "C5_pass")
    )

    return {
        "schema": "tensorfem.hull-girder-csr-chain/1.0",
        "ship": "KVLCC2",
        "particulars": dict(KVLCC2_PARTICULARS),
        "geometry": geometry,
        "weights": {
            "items": [
                {"name": item.name, "mass_t": item.mass_t,
                 "x0": item.x0, "x1": item.x1, "shape": item.shape}
                for item in items
            ],
            "total_mass_t": total_mass_t,
            "lcg_m": lcg,
        },
        "equilibrium": equilibrium,
        "rule": {
            "C": rule["C"],
            "M_wh": rule["M_wh"],
            "M_ws": rule["M_ws"],
            "W_min_deck": w_min,
            "k_w_hog": k_w["k_w_hog"],
            "k_w_sag": k_w["k_w_sag"],
        },
        "design_wave": {
            "h_w_input": design_wave_height,
            "h_w_rule_equivalent": h_w_equivalent,
            "wavelength": form.Lpp,
            "smith_correction": True,
            "M_w_raw_mid": raw_mid,
            "M_w_mid": abs(m_w_mid),
            "calibration_ratio": raw_mid / rule["M_wh"],
        },
        "section": {
            "y_neutral_axis": y_na,
            "I_midship": inertia,
            "I_strip": strip_inertia,
            "W_deck": w_deck,
            "W_bottom": w_bottom,
            "y_deck": y_top,
            "y_bottom": y_bottom,
            "total_area": section.total_area(),
            "axial_residual": section.axial_residual(),
            "scantling_closure": closure,
            "members": [
                {"name": m.name, "area": m.area, "y": m.y,
                 "thickness": m.thickness, "vertical_span": m.vertical_span}
                for m in section.members
            ],
        },
        "moments": {
            "M_SW_mid": m_sw_mid,
            "M_W_mid": m_w_mid,
            "M_total_hog": m_total_hog,
            "M_total_sag": m_total_sag,
            "M_wh_rule": rule["M_wh"],
            "M_ws_rule": rule["M_ws"],
            "M_total_hog_rule": m_total_hog_rule,
            "M_total_sag_rule": m_total_sag_rule,
            "C2_reference": c2_reference,
        },
        "stresses": {
            "deck_hog": sigma_deck_hog,
            "bottom_hog": sigma_bottom_hog,
            "deck_sag": sigma_deck_sag,
            "bottom_sag": sigma_bottom_sag,
            "sigma_analytic_deck_hog": sigma_analytic_deck_hog,
            "sigma_analytic_bottom_sag": sigma_analytic_bottom_sag,
        },
        "beam": {
            "n_elements": n_elements,
            "max_moment_error_vs_analytic": beam_error,
            "reaction_hog_N": fem_hog["reaction"].tolist(),
            "reaction_sag_N": fem_sag["reaction"].tolist(),
        },
        "csr": {
            "allowable": csr_allowable_stress(
                gamma_m=gamma_m, permissible_stress=permissible_stress
            ),
            "hogging": check_hog,
            "sagging": check_sag,
        },
        "ultimate": {
            "hogging": {k: v for k, v in ultimate_hog.items()
                        if k not in ("compression_limits",)},
            "sagging": {k: v for k, v in ultimate_sag.items()
                        if k not in ("compression_limits",)},
        },
        "criteria": criteria,
        "distributions": {
            "x": x.tolist(),
            "nodes": nodes.tolist(),
            "q_still_water": np.interp(nodes, x, q_still_water).tolist(),
            "q_wave": np.interp(nodes, x, q_wave).tolist(),
            "M_SW": fem_sw["moment"].tolist(),
            "M_W": fem_w["moment"].tolist(),
            "M_hog": fem_hog["moment"].tolist(),
            "M_sag": fem_sag["moment"].tolist(),
            "M_analytic_hog": analytic_hog_nodes.tolist(),
            "M_analytic_sag": analytic_sag_nodes.tolist(),
            "sigma_deck_hog": stress_profile(fem_hog["moment"], y_top).tolist(),
            "sigma_bottom_hog": stress_profile(fem_hog["moment"], -y_bottom).tolist(),
            "sigma_deck_sag": stress_profile(fem_sag["moment"], y_top).tolist(),
            "sigma_bottom_sag": stress_profile(fem_sag["moment"], -y_bottom).tolist(),
            "I_profile": i_profile.tolist(),
        },
        "unit_note": "N, N.m, Pa; moments reported as magnitudes, hogging positive-definite",
    }


# ---------------------------------------------------------------------------
# 10. Figures
# ---------------------------------------------------------------------------

def plot_hull_girder_chain(result: dict, outdir: str = "artifacts/hull_girder_csr_chain") -> list[dict]:
    """Render the four chain figures and return ``[{name, caption, path}]``.

    matplotlib is imported lazily so the numerical pipeline has no plotting
    dependency.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist = result["distributions"]
    x = np.asarray(dist["nodes"]) / 1000.0
    os.makedirs(outdir, exist_ok=True)
    figures: list[dict] = []

    # 1. load and bending-moment distributions
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    axes[0].plot(x, np.asarray(dist["q_still_water"]) / 1e3, label="still-water load $q_{SW}$")
    axes[0].plot(x, np.asarray(dist["q_wave"]) / 1e3, label="wave load $q_W$")
    axes[0].set_ylabel("load (kN/m)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("KVLCC2 load and vertical bending moment along the hull")
    axes[1].plot(x, np.asarray(dist["M_SW"]) / 1e9, label="$M_{SW}$ (FE)")
    axes[1].plot(x, np.asarray(dist["M_W"]) / 1e9, label="$M_W$ (FE)")
    axes[1].plot(x, -np.asarray(dist["M_hog"]) / 1e9, label="$M_{total}$ hogging")
    axes[1].plot(x, np.asarray(dist["M_sag"]) / 1e9, "--", label="$M_{total}$ sagging")
    axes[1].plot(x, -np.asarray(dist["M_analytic_hog"]) / 1e9, ":", lw=1.0,
                 label="analytic hogging")
    axes[1].set_xlabel("x from AP (km)")
    axes[1].set_ylabel("bending moment (GN$\\cdot$m)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "moment_distribution.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figures.append({"name": "moment_distribution", "path": path,
                    "caption": "Still-water, wave and total vertical bending moment along the "
                               "KVLCC2 hull from the Euler-Bernoulli beam FE (320 elements, "
                               "EI(x) from the scantling set) against the closed-form "
                               "double integration."})

    # 2. deck/bottom stress distributions
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.plot(x, np.asarray(dist["sigma_deck_hog"]) / 1e6, label="deck, hogging")
    ax.plot(x, np.asarray(dist["sigma_bottom_hog"]) / 1e6, label="bottom, hogging")
    ax.plot(x, np.asarray(dist["sigma_deck_sag"]) / 1e6, "--", label="deck, sagging")
    ax.plot(x, np.asarray(dist["sigma_bottom_sag"]) / 1e6, "--", label="bottom, sagging")
    ax.axhline(result["csr"]["allowable"]["sigma_allow"] / 1e6, color="k", lw=1.0,
               label="allowable $\\sigma_y/\\gamma_m$")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("x from AP (km)")
    ax.set_ylabel("hull-girder stress (MPa)")
    ax.set_title("Deck and bottom longitudinal stress (FE curvature recovery)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "stress_distribution.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figures.append({"name": "stress_distribution", "path": path,
                    "caption": "Deck and bottom longitudinal stress along the hull for the "
                               "hogging and sagging design cases, compared with the CSR "
                               "allowable stress."})

    # 3. midship section stress diagram
    sec = result["section"]
    y_na = sec["y_neutral_axis"]
    m_hog = result["moments"]["M_total_hog"]
    m_sag = result["moments"]["M_total_sag"]
    inertia = sec["I_midship"]
    depth = np.linspace(0.0, 30.0, 200)
    sigma_hog = -(m_hog * -1.0) * (depth - y_na) / inertia
    sigma_sag = -(m_sag * 1.0) * (depth - y_na) / inertia
    fig, ax = plt.subplots(figsize=(5.2, 6.0))
    ax.plot(sigma_hog / 1e6, depth, label="hogging (deck tension)")
    ax.plot(sigma_sag / 1e6, depth, label="sagging (deck compression)")
    ax.axvline(0.0, color="k", lw=0.6)
    ax.axhline(y_na, color="k", ls=":", lw=1.0)
    ax.text(ax.get_xlim()[0], y_na + 0.4, f"NA y = {y_na:.2f} m", fontsize=8)
    ax.set_xlabel("bending stress (MPa)")
    ax.set_ylabel("height above keel (m)")
    ax.set_title("Midship section stress at the design moments")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "section_stress.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figures.append({"name": "section_stress", "path": path,
                    "caption": "Linear bending-stress distribution over the midship section "
                               "(neutral axis at y = %.2f m) for the governing hogging and "
                               "sagging moments." % y_na})

    # 4. ultimate moment-curvature curves
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for direction, style in (("hogging", "-"), ("sagging", "--")):
        data = result["ultimate"][direction]
        ax.plot(np.asarray(data["curvature"]), np.asarray(data["moment_curve"]) / 1e9,
                style, label=f"{direction}: $M_u$ = {data['M_ultimate']/1e9:.2f} GN$\\cdot$m")
    ax.axhline(result["moments"]["M_total_hog"] / 1e9, color="C0", ls=":", lw=1.0,
               label="design hogging moment")
    ax.axhline(result["moments"]["M_total_sag"] / 1e9, color="C1", ls=":", lw=1.0,
               label="design sagging moment")
    ax.set_xlabel("curvature (1/m)")
    ax.set_ylabel("bending moment (GN$\\cdot$m)")
    ax.set_title("Ultimate strength: incremental-curvature collapse of the midship section")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "ultimate_moment_curvature.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figures.append({"name": "ultimate_moment_curvature", "path": path,
                    "caption": "Moment-curvature response up to ultimate collapse for hogging "
                               "and sagging, with the design total moments marked."})

    return figures