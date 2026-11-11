"""Floating ice sheet bending fracture driven by a level-set tracked crack.

Physical model
--------------
An ice sheet of thickness ``h`` floats on water.  The water is modelled as a
Winkler foundation ``p = rho_w g w`` (hydrostatic restoring), so the sheet obeys
``D grad^4 w + rho_w g w = q`` with ``D = E h^3/(12(1-nu^2))`` and the
characteristic length ``l = (D/(rho_w g))^(1/4)`` (Hetényi 1946; Kerr 1976).  A
vertical load ``P`` distributed over a circular patch of radius ``a_l`` (an
ice-structure contact footprint) bends the sheet; a pre-existing straight
through-crack of half-length ``a`` under the load opens in mode I.

Fracture model
--------------
* driving force: energy release rate ``G`` of the *explicit* crack in the
  Mindlin/Winkler plate, evaluated by virtual crack extension
  (:func:`tensorfem.levelset.plate_crack.energy_release_rate_vce`);
* criterion: ``G >= G_c`` with ``G_c = K_IC^2 (1-nu^2)/E`` (ice toughness
  ``K_IC = 0.1 MPa m^(1/2)``, range 0.05-0.2 in the literature);
* growth: the crack front is the zero contour of a level-set field; each
  increment advances the tip by the current grid step along the front normal
  (constant velocity extension), re-extracts the crack span from the level set
  and re-solves, so the level set is the single source of truth for the crack
  geometry.

Analytic anchors
----------------
1. *Uncracked sheet*: the exact Kelvin-function centre moment of a circular
   patch load, ``M_max = p l^2 X kei'(X)``, ``X = a_l/l``
   (:func:`tensorfem.levelset.plate_crack.patch_max_moment`).
2. *Small-crack fracture load*: with the far field equibiaxial at the patch
   centre, the bending-Griffith relation
   ``G = pi c^2 M^2 a/(D h)``, ``c = 1/(1+nu)`` gives

       P_f = sqrt(G_c D h/(pi a)) / (c * m_hat),   M = m_hat P

   (:func:`analytical_fracture_load`).  This is the "reference value" used by
   the acceptance criterion; the Faltinsen ice-plate experiment dataset is not
   available offline, which is declared explicitly in the report.
3. *First-crack (strength) load* of classical ice engineering,
   ``M_max(P) = sigma_f h^2/6`` (:func:`strength_first_crack_load`), is reported
   as a cross-check together with the intrinsic flaw size that reconciles the
   two criteria (:func:`equivalent_flaw_size`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .crack_criteria import growth_velocity
from .level_set import LevelSetField, crack_level_set
from .plate_crack import (
    energy_release_rate_vce,
    graded_axis,
    kelvin_kei_derivative,
    patch_max_moment,
    patch_max_moment_nevel,
    patch_moment_sum,
    plate_rigidity,
    winkler_length,
)

__all__ = [
    "IceSheet",
    "crack_span_from_level_set",
    "energy_release_rate_ice",
    "analytical_fracture_load",
    "strength_first_crack_load",
    "equivalent_flaw_size",
    "level_set_crack_growth",
    "uncracked_moment_check",
    "nevel_form_reference",
    "run_ice_plate_fracture_benchmark",
]


@dataclass(frozen=True)
class IceSheet:
    """Material and thickness parameters of a floating ice sheet."""

    young: float = 7.0e9            # Pa
    poisson: float = 0.30
    thickness: float = 0.30         # m
    rho_ice: float = 917.0          # kg/m^3
    rho_water: float = 1025.0       # kg/m^3
    gravity: float = 9.81           # m/s^2
    toughness: float = 1.0e5        # Pa m^0.5  (0.1 MPa m^0.5)
    flexural_strength: float = 6.0e5  # Pa

    def __post_init__(self) -> None:
        if self.thickness <= 0.0:
            raise ValueError("thickness must be positive")
        if self.young <= 0.0 or not (-1.0 < self.poisson < 0.5):
            raise ValueError("invalid elastic constants")
        if self.toughness <= 0.0:
            raise ValueError("toughness must be positive")

    @property
    def foundation(self) -> float:
        """Winkler modulus of the water, ``rho_w g`` (N/m^3)."""
        return self.rho_water * self.gravity

    @property
    def rigidity(self) -> float:
        return plate_rigidity(self.young, self.poisson, self.thickness)

    @property
    def length_scale(self) -> float:
        return winkler_length(self.rigidity, self.foundation)

    @property
    def critical_energy_release_rate(self) -> float:
        """``G_c = K_IC^2 (1-nu^2)/E`` (plane strain)."""
        return self.toughness ** 2 * (1.0 - self.poisson ** 2) / self.young

    @property
    def surface_stress_per_moment(self) -> float:
        """``sigma = 6 M/h^2`` -- surface bending stress per unit moment."""
        return 6.0 / self.thickness ** 2

    def describe(self) -> dict[str, float]:
        return {
            "young": self.young,
            "poisson": self.poisson,
            "thickness": self.thickness,
            "foundation": self.foundation,
            "rigidity": self.rigidity,
            "length_scale": self.length_scale,
            "critical_energy_release_rate": self.critical_energy_release_rate,
            "toughness": self.toughness,
            "flexural_strength": self.flexural_strength,
        }


# ---------------------------------------------------------------------------
# analytic anchors
# ---------------------------------------------------------------------------
def _moment_per_load(patch_radius: float, sheet: IceSheet) -> float:
    """``m_hat``: centre moment of a circular patch load per unit total load."""
    pressure = 1.0
    m = patch_max_moment(pressure, patch_radius, sheet.rigidity, sheet.foundation,
                         sheet.poisson)
    total = pressure * math.pi * patch_radius ** 2
    return m / total


def analytical_fracture_load(half_length: float, patch_radius: float,
                             sheet: IceSheet) -> float:
    """Small-crack energy-criterion fracture load (see module docstring)."""
    c = 1.0 / (1.0 + sheet.poisson)
    m_hat = _moment_per_load(patch_radius, sheet)
    g_per_force2 = math.pi * c ** 2 * m_hat ** 2 * half_length / (
        sheet.rigidity * sheet.thickness
    )
    return math.sqrt(sheet.critical_energy_release_rate / g_per_force2)


def strength_first_crack_load(patch_radius: float, sheet: IceSheet) -> float:
    """Classical first-crack load from the flexural-strength criterion."""
    m_hat = _moment_per_load(patch_radius, sheet)
    m_crit = sheet.flexural_strength * sheet.thickness ** 2 / 6.0
    return m_crit / m_hat


def equivalent_flaw_size(first_crack_load: float, patch_radius: float,
                         sheet: IceSheet) -> float:
    """Crack half-length that makes the energy criterion agree with strength."""
    m = _moment_per_load(patch_radius, sheet) * first_crack_load
    c = 1.0 / (1.0 + sheet.poisson)
    return sheet.critical_energy_release_rate * sheet.rigidity * sheet.thickness / (
        math.pi * c ** 2 * m ** 2
    )


def nevel_form_reference(pressure: float, patch_radius: float,
                         sheet: IceSheet) -> dict[str, float]:
    """Compare the exact Kelvin patch moment with the Nevel/Westergaard form."""
    total = pressure * math.pi * patch_radius ** 2
    exact = patch_max_moment(pressure, patch_radius, sheet.rigidity, sheet.foundation,
                             sheet.poisson)
    nevel = patch_max_moment_nevel(total, patch_radius, sheet.rigidity, sheet.foundation,
                                   sheet.poisson)
    return {"exact": exact, "nevel": nevel, "relative_difference": (nevel - exact) / exact}


# ---------------------------------------------------------------------------
# cracked-sheet finite element model
# ---------------------------------------------------------------------------
def _plate_grid(domain: float, cells: int,
                h_min: float | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Tensor-product plate grid; graded towards the centre when ``h_min`` is given."""
    if h_min is None or h_min >= domain / cells:
        xs = np.linspace(-0.5 * domain, 0.5 * domain, cells + 1)
    else:
        xs = graded_axis(0.5 * domain, cells, h_min)
    return xs, xs.copy(), cells // 2


def crack_span_from_level_set(field: LevelSetField, xs: np.ndarray, y_c: float,
                             *, tip_tol: float = 1e-9) -> tuple[int, int]:
    """Map the level-set zero contour to the crack-span node indices of the grid.

    The contour is the crack; its two extremes along ``x`` are the tips, which
    are snapped to the nearest grid nodes.  This is the level-set -> FE hand-off
    used by :func:`level_set_crack_growth`.
    """
    contours = field.zero_contour()
    if not contours:
        raise ValueError("level set has no zero contour")
    pts = np.vstack(contours)
    on_line = pts[np.abs(pts[:, 1] - y_c) <= field.h + tip_tol]
    if on_line.shape[0] < 2:
        raise ValueError("zero contour does not lie on the crack line")
    # a signed distance to a *finite* segment keeps a zero contour along the
    # tangent beyond both tips; keep only the points where the field really
    # changes sign across the crack line (the physical crack)
    up = field.values_at(np.stack((on_line[:, 0], on_line[:, 1] + 1.5 * field.h), axis=1))
    down = field.values_at(np.stack((on_line[:, 0], on_line[:, 1] - 1.5 * field.h), axis=1))
    real = on_line[up * down < 0.0]
    if real.shape[0] < 2:
        raise ValueError("no genuine crack crossing found on the crack line")
    x_lo = float(np.min(real[:, 0]))
    x_hi = float(np.max(real[:, 0]))
    # snap outwards (the crack must not shrink by rounding): the first grid node
    # outside each tip, provided it is within one element of the tip
    i_left = int(np.searchsorted(xs, x_lo, side="right") - 1)
    i_right = int(np.searchsorted(xs, x_hi, side="left"))
    i_left = max(i_left, 0)
    i_right = min(i_right, xs.size - 1)
    spacing = float(np.min(np.diff(xs)))
    if xs[i_left] < x_lo - spacing:
        i_left = int(np.argmin(np.abs(xs - x_lo)))
    if xs[i_right] > x_hi + spacing:
        i_right = int(np.argmin(np.abs(xs - x_hi)))
    if i_right <= i_left:
        raise ValueError("degenerate crack span from level set")
    return i_left, i_right


def energy_release_rate_ice(
    half_length: float,
    pressure: float,
    patch_radius: float,
    sheet: IceSheet,
    *,
    domain: float = 4.0,
    cells: int = 120,
    h_min: float | None = 0.06,
    advances: tuple[int, ...] = (1, 2),
    cache: dict | None = None,
) -> dict[str, float]:
    """Energy release rate of a central crack in the floating sheet.

    ``domain`` is the edge length in units of the characteristic length ``l``.
    """
    xs, ys, jc = _plate_grid(domain * sheet.length_scale, cells, h_min)
    h_centre = float(xs[jc + 1] - xs[jc])
    if h_centre > 0.5 * half_length:
        raise ValueError("grid too coarse for the requested crack length")
    ih = int(round(half_length / h_centre))
    out = energy_release_rate_vce(
        xs, ys, jc=jc, i_left=jc - ih, i_right=jc + ih, advances=advances,
        thickness=sheet.thickness, young=sheet.young, poisson=sheet.poisson,
        foundation=sheet.foundation, pressure=pressure, patch_centre=(0.0, 0.0),
        patch_radius=patch_radius, stiffness_cache=cache,
    )
    out["applied_load"] = pressure * math.pi * patch_radius ** 2
    out["mesh_spacing"] = float(xs[jc + 1] - xs[jc])
    return out


def uncracked_moment_check(sheet: IceSheet, *, pressure: float, patch_radius: float,
                           domain: float = 4.0, cells: int = 120,
                           h_min: float | None = 0.06) -> dict[str, float]:
    """Compare the FE centre moment of the uncracked sheet with the Kelvin anchor."""
    from .plate_crack import solve_cracked_plate

    xs, ys, jc = _plate_grid(domain * sheet.length_scale, cells, h_min)
    res = solve_cracked_plate(
        xs, ys, jc=jc, i_left=None, i_right=None, thickness=sheet.thickness,
        young=sheet.young, poisson=sheet.poisson, foundation=sheet.foundation,
        pressure=pressure, patch_centre=(0.0, 0.0), patch_radius=patch_radius,
    )
    mom = res.moment_field(young=sheet.young, poisson=sheet.poisson,
                           thickness=sheet.thickness)
    centres = mom["centres"]
    r = np.hypot(centres[:, 0], centres[:, 1])
    patch_moment = 0.5 * (mom["Mx"] + mom["My"])
    near = r <= patch_radius
    fe_centre = float(np.max(patch_moment[near]))  # fmax, not a window average
    fe_window_mean = float(np.mean(patch_moment[near]))
    anchor = patch_max_moment(pressure, patch_radius, sheet.rigidity, sheet.foundation,
                              sheet.poisson)
    return {
        "finite_element": fe_centre,
        "kelvin_anchor": anchor,
        "relative_difference": (fe_centre - anchor) / anchor,
        "finite_element_window_mean": fe_window_mean,
        "moment_sum_anchor": patch_moment_sum(pressure, patch_radius, sheet.rigidity,
                                              sheet.foundation),
        "applied_load": res.total_load,
        "expected_load": pressure * math.pi * patch_radius ** 2,
    }


# ---------------------------------------------------------------------------
# level-set driven growth
# ---------------------------------------------------------------------------
def level_set_crack_growth(
    sheet: IceSheet,
    *,
    half_length_initial: float,
    pressure: float,
    patch_radius: float,
    domain: float = 4.0,
    cells: int = 120,
    h_min: float | None = 0.06,
    increments: int = 4,
    load_steps: int = 1,
    tip_speed: float = 1.0,
    kinetic_exponent: float = 1.0,
    cache: dict | None = None,
) -> dict[str, object]:
    """Grow a level-set tracked crack through the floating sheet.

    Each increment (i) solves the cracked plate at the crack geometry defined by
    the level-set zero contour, (ii) evaluates ``G`` by virtual crack extension,
    (iii) advances the front by one grid step when ``G > G_c`` using the kinetic
    law ``v = v0 (G/G_c - 1)^m`` as the level-set advection speed, and (iv)
    re-initialises the level set.  The history of ``G``, the crack length and the
    tip position is returned for the report figures.
    """
    xs, ys, jc = _plate_grid(domain * sheet.length_scale, cells, h_min)
    h = float(xs[jc + 1] - xs[jc])
    # level-set grid: uniform, symmetric about the crack centre, one node on the
    # crack line (odd count) and spanning the same physical domain as the FE grid
    n_ls = 2 * int(math.ceil(0.5 * domain * sheet.length_scale / h)) + 1
    field = crack_level_set(2.0 * half_length_initial, centre=(0.0, 0.0),
                            nx=n_ls, ny=n_ls, h=h)
    history: list[dict[str, float]] = []
    g_c = sheet.critical_energy_release_rate
    for step in range(increments + 1):
        i_left, i_right = crack_span_from_level_set(field, xs, 0.0)
        out = energy_release_rate_vce(
            xs, ys, jc=jc, i_left=i_left, i_right=i_right, advances=(1,),
            thickness=sheet.thickness, young=sheet.young, poisson=sheet.poisson,
            foundation=sheet.foundation, pressure=pressure, patch_centre=(0.0, 0.0),
            patch_radius=patch_radius, stiffness_cache=cache,
        )
        g = out["G"]
        tip = np.array([xs[i_right], 0.0])
        history.append({
            "step": float(step),
            "half_length": 0.5 * (xs[i_right] - xs[i_left]),
            "G": float(g),
            "G_over_Gc": float(g / g_c),
            "tip_x": float(tip[0]),
            "tip_y": float(tip[1]),
            "velocity": float(growth_velocity(g, g_c, tip_speed, kinetic_exponent)),
        })
        if step == increments:
            break
        if g <= g_c:
            break
        v = growth_velocity(g, g_c, tip_speed, kinetic_exponent)
        dt = h / v  # advance exactly one grid step
        # symmetric growth: the field is advected outwards on both sides of the
        # crack centre, so both tips advance by one grid step
        X, _ = field.coords()
        direction = np.sign(X) * np.abs(np.sign(X))
        field = field.advect(v * direction, np.zeros_like(direction), dt)
        field = field.reinitialize(band=4.0 * h, n_iter=int(6.0 * 4.0 / 0.3))
    return {
        "history": history,
        "final_half_length": history[-1]["half_length"],
        "initial_half_length": history[0]["half_length"],
        "g_c": g_c,
        "grid_spacing": h,
        "path_max_lateral_deviation": float(
            max(abs(row["tip_y"]) for row in history)
        ),
    }


def run_ice_plate_fracture_benchmark(
    *,
    sheet: IceSheet | None = None,
    patch_radius: float = 0.5,
    half_length: float = 0.30,
    pressure: float = 1.0e4,
    domain: float = 4.0,
    cells: int = 120,
    h_min: float = 0.06,
) -> dict[str, object]:
    """Full acceptance run: analytic anchors, G, fracture loads, level-set growth."""
    sheet = sheet or IceSheet()
    g_c = sheet.critical_energy_release_rate
    checks = uncracked_moment_check(sheet, pressure=pressure, patch_radius=patch_radius,
                                    domain=domain, cells=cells, h_min=h_min)
    cache: dict = {}
    g_ref = energy_release_rate_ice(half_length, pressure, patch_radius, sheet,
                                    domain=domain, cells=cells, h_min=h_min, cache=cache)
    load_ref = pressure * math.pi * patch_radius ** 2
    g = g_ref["G"]
    p_f_fe = math.sqrt(g_c / g) * pressure if g > 0.0 else float("nan")
    p_f_an = analytical_fracture_load(half_length, patch_radius, sheet)
    p_first = strength_first_crack_load(patch_radius, sheet)
    growth = level_set_crack_growth(
        sheet, half_length_initial=half_length, pressure=p_f_fe * 1.02,
        patch_radius=patch_radius, domain=domain, cells=cells, h_min=h_min,
        increments=4, cache=cache,
    )
    m_hat = _moment_per_load(patch_radius, sheet)
    return {
        "sheet": sheet.describe(),
        "scenario": {
            "patch_radius": patch_radius,
            "crack_half_length": half_length,
            "pressure": pressure,
            "domain_units_of_l": domain,
            "cells": cells,
            "h_min": h_min,
            "grid_spacing_at_tip": g_ref["mesh_spacing"],
            "applied_load": load_ref,
        },
        "A4a_uncracked_moment_vs_kelvin": checks,
        "energy_release_rate": g_ref,
        "A4_fracture_load": {
            "finite_element": p_f_fe,
            "analytic_small_crack": p_f_an,
            "relative_difference": (p_f_fe - p_f_an) / p_f_an,
            "strength_first_crack_load": p_first,
            "equivalent_flaw_size_of_strength_criterion": equivalent_flaw_size(
                p_first, patch_radius, sheet
            ),
            "moment_per_load": m_hat,
            "nevel_form_check": nevel_form_reference(pressure, patch_radius, sheet),
        },
        "level_set_growth": growth,
    }
