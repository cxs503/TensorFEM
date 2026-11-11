"""Mixed-mode crack path tracking with the level-set method.

Verification target
-------------------
The level-set field of :mod:`tensorfem.levelset.level_set` is *the* geometrical
representation of the crack in this package.  This module checks that it tracks
a curved propagation path faithfully, by comparing the tip trajectory of the
level-set front with an independently integrated reference path produced by the
same propagation law.

Propagation law
---------------
* the local mixity decides the kink angle through the maximum tangential
  (hoop) stress criterion, :func:`tensorfem.levelset.crack_criteria.kink_angle_max_hoop`;
* for an inclined crack of half-length ``a`` in an infinite plate under remote
  uniaxial tension ``sigma`` (load along ``y``, crack at angle ``beta`` from
  ``x``) the analytic stress intensity factors are
  ``K_I = sigma sin^2(beta) sqrt(pi a)``, ``K_II = sigma sin(beta)cos(beta) sqrt(pi a)``;
* a growing crack rotates until ``K_II -> 0`` (mode I attractor), so the mixity
  ratio is relaxed exponentially with the crack extension,
  ``rho(s) = K_II/K_I exp(-s/s0)``, which is the standard idealisation of the
  curved-path capture: a kink of the maximum-hoop-stress magnitude immediately
  followed by a rotation towards the direction perpendicular to the remote load.

Scale convention: ``beta`` and the kink angles are measured from ``+x``; the
sign of the kink is fixed by the requirement that the crack rotates *towards*
the axis perpendicular to the remote tension, which for the geometry above
corresponds to using ``-K_II`` in the hoop-stress criterion (the sign of the
shear traction depends on which bank is called upper).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .crack_criteria import inclined_crack_sifs, kink_angle_max_hoop
from .level_set import LevelSetField, crack_level_set, polyline_deviation

__all__ = [
    "PathGrowthResult",
    "mixity_schedule",
    "reference_path",
    "level_set_path",
    "path_comparison",
]


def mixity_schedule(beta: float, sigma: float, half_length: float,
                    decay_length: float) -> dict[str, float]:
    """Initial kink angle and mixity ratio of the inclined crack."""
    k_i, k_ii = inclined_crack_sifs(sigma, beta, half_length)
    if k_i <= 0.0:
        raise ValueError("crack must open (K_I > 0)")
    ratio = k_ii / k_i
    theta0 = kink_angle_max_hoop(k_i, -k_ii).theta
    return {"k_i": k_i, "k_ii": k_ii, "ratio": ratio, "kink_angle": theta0,
            "decay_length": decay_length}


def _direction(s: float, beta: float, ratio: float, decay: float) -> float:
    rho = ratio * math.exp(-s / decay) if decay > 0.0 else ratio
    return beta + kink_angle_max_hoop(1.0, -rho).theta


def reference_path(beta: float, ratio: float, decay: float, *, steps: int,
                   step: float, start: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Trapezoidal integration of the propagation law (independent reference)."""
    pts = [np.asarray(start, dtype=float)]
    alpha = _direction(0.0, beta, ratio, decay)
    for k in range(steps):
        s0 = k * step
        s1 = (k + 1) * step
        nxt_alpha = _direction(s1, beta, ratio, decay)
        mid = 0.5 * (alpha + nxt_alpha)
        pts.append(pts[-1] + step * np.array([math.cos(mid), math.sin(mid)]))
        alpha = nxt_alpha
    return np.asarray(pts)


@dataclass
class PathGrowthResult:
    path: np.ndarray
    kink_angles: list[float]
    tip_history: list[np.ndarray]
    initial_kink_angle: float
    final_direction: float
    field: LevelSetField


def level_set_path(beta: float, half_length: float, *, ratio: float, decay: float,
                   steps: int, step: float, nx: int = 241, ny: int = 241,
                   h: float | None = None, tip_window: float = 2.5,
                   reinit_band: float | None = None) -> PathGrowthResult:
    """Advance the crack front in a level-set field and record the tip path.

    The advection velocity is applied only in a compact window around the tip
    (``tip_window * step`` wide), which is the local form of the velocity
    extension: the crack grows at its tip instead of the whole front being
    translated.  After each increment the field is re-initialised, so the tip
    trajectory is the only memory of the growth.
    """
    h = h or (4.0 * half_length / (nx - 1))
    field = crack_level_set(2.0 * half_length, centre=(0.0, 0.0), angle=beta,
                            nx=nx, ny=ny, h=h)
    x0 = field.x0
    y0 = field.y0
    tips: list[np.ndarray] = [np.array([half_length * math.cos(beta),
                                        half_length * math.sin(beta)])]
    tip0 = tips[0].copy()
    kinks: list[float] = []
    path = [np.array([0.0, 0.0]), tips[0].copy()]
    alpha = _direction(0.0, beta, ratio, decay)
    for k in range(steps):
        s1 = (k + 1) * step
        nxt = _direction(s1, beta, ratio, decay)
        mid = 0.5 * (alpha + nxt)
        direction = np.array([math.cos(mid), math.sin(mid)])
        kinks.append(nxt - alpha)
        # local velocity: unit speed in `direction` inside a window at the tip
        X, Y = field.coords()
        tip = tips[-1]
        w = max(tip_window * step, 3.0 * h)
        weight = np.exp(-(((X - tip[0]) ** 2 + (Y - tip[1]) ** 2) / w ** 2))
        vx = step * direction[0] * weight
        vy = step * direction[1] * weight
        field = field.advect(vx, vy, 1.0, scheme="sl2")
        band = reinit_band or 4.0 * step
        field = field.reinitialize(band=band, n_iter=int(6.0 * band / (0.3 * h)) + 5)
        # the tip is measured *from the level set*, not from the integrator: the
        # point of the zero contour furthest along the propagation direction
        new_tip = _tip_from_contour(field, direction, previous=tip,
                                    window=1.5 * step + 3.0 * h)
        tips.append(new_tip)
        path.append(new_tip.copy())
        alpha = nxt
    initial_kink = _direction(0.0, beta, ratio, decay) - beta
    return PathGrowthResult(
        path=np.asarray(path[1:]), kink_angles=kinks, tip_history=tips,
        initial_kink_angle=initial_kink, final_direction=alpha, field=field,
    )


def _tip_from_contour(field: LevelSetField, direction: np.ndarray,
                      previous: np.ndarray, window: float) -> np.ndarray:
    """Furthest zero-contour point along ``direction`` near the previous tip.

    A signed distance to a *finite* segment has a zero contour that also runs
    along the tangent beyond both tips (the field flips sign there), so the
    extracted contour is trimmed to the ball of radius ``window`` around the
    previous tip before the leading point is taken.
    """
    contours = field.zero_contour()
    if not contours:
        return previous + 0.0
    pts = np.vstack(contours)
    keep = np.hypot(*(pts - previous).T) <= window
    if not np.any(keep):
        return previous + 0.0
    pts = pts[keep]
    adv = (pts - previous) @ direction
    return pts[int(np.argmax(adv))].copy()


def path_comparison(candidate: np.ndarray, reference: np.ndarray, *,
                    samples: int = 200) -> dict[str, float]:
    """Normalised deviation of a tracked path from the reference path."""
    return polyline_deviation(candidate, reference, samples=samples)
