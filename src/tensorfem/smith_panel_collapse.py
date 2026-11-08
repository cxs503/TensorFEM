"""Smith progressive collapse of a stiffened panel unit (plate + stiffener).

Why this module exists
----------------------
``tensorfem.stiffened_panel_ultimate`` carries a *limiting-stress* load-end
shortening law: each component is linear elastic up to its limiting stress and
then a constant plateau.  That law has no post-peak branch, so a Smith
progressive-collapse run built on it can never exhibit the negative stiffness
that drives the peak of a panel's mean stress--strain curve, and it cannot
represent the interaction between a plate that is already softening and a
stiffener that is still picking up load.

This module supplies the missing chain, for one stiffened plate unit
(plate field of span ``a``, stiffener spacing ``b``, thickness ``t``, one
longitudinal stiffener):

1. **Plate load--shortening curve** from a large-deflection (Marguerre /
   von Karman) single-mode solution with an explicit initial deflection
   ``w_0 = c_0 beta^2 t`` and a longitudinal-edge yield criterion
   ``sigma + sigma_1 = sigma_y`` (``sigma_1`` is the membrane stress
   concentration at the longitudinal edge).  The welded residual stress field
   is superposed through the classical two-zone idealisation (tension block
   ``sigma_rt = sigma_y`` at the plate edges over a fraction ``2 eta``,
   compression ``sigma_rc`` elsewhere; ``2 eta sigma_rt = (1 - 2 eta) sigma_rc``
   fixes ``eta`` for a given ``sigma_rc``).  The post-ultimate branch is a
   documented linear softening (see :data:`POST_ULTIMATE_SLOPE`).
2. **Stiffener load--shortening curve** from the smaller of the beam-column
   strength (Perry--Robertson with the initial bow, i.e. the Euler load reduced
   by the bow imperfection) and the local buckling of the flange outstand and of
   the web.
3. **Smith combination**: plate and stiffener in parallel, weighted by their
   areas, giving ``sigma(eps)`` for the panel unit and its peak ``sigma_u``.
4. **Interaction surface** for biaxial compression plus shear in the
   CSR-style quadratic form (see :func:`interaction_load_factor`), evaluated
   along proportional load paths.
5. **CSR-style closed-form cross-check**: an *independent* rule-type min-chain
   (Faulkner plate ultimate + Johnson--Ostenfeld column with the effective
   attached plate) evaluated at the same geometry, used as the reference value
   for the progressive-collapse peak (:func:`csr_panel_chain`).
6. The **IACS CSR No. 47 standard panel** of the validation brief
   (``a = 2400 mm``, ``s = 800 mm``, ``t = 12 mm``, stiffener ``HP 200 x 10``)
   as :func:`csr_no47_panel`.

Scope and honesty statement
---------------------------
* The plate branch is *derived*, not tabulated: the Marguerre single-mode
  equilibrium (Galerkin projection of the von Karman plate equations) is solved
  here, the critical mode number is selected by minimising the elastic buckling
  stress over ``m = 1..12``, and the ultimate is the largest mean stress before
  the longitudinal edge reaches yield.  No literature coefficient is used in
  that branch except ``c_0 = 0.1`` (the reference imperfection amplitude
  ``w_0 = 0.1 beta^2 t``, which the caller supplies) and Poisson's ratio.
* The result of that branch is *compared* with the Faulkner (1975) effective
  width ``2/beta - 1/beta^2`` as an independent closed form.  Both descend from
  the same effective-width physics, so this comparison is a cross-check of the
  numerical route against the closed-form fit, not an experimental validation.
  The report states this explicitly.
* The **residual stress** model is the standard two-zone idealisation; the zone
  stresses are superposed on the plate mean-stress curve at equal strain with
  yield clipping.  It captures the loss of plate capacity, not the local
  redistribution near the weld.
* The **post-ultimate softening slope** is an assumption
  (:data:`POST_ULTIMATE_SLOPE`), not a derived quantity; the sensitivity of the
  panel ultimate to it is reported by the evidence run.
* The **bulb flat** ``HP 200 x 10`` is replaced by an equivalent T-section by
  :func:`bulb_flat_equivalent_t`; no catalogue area is quoted and the
  idealisation is stated in every artefact.
* **No experimental comparison value is invented.**  The Paik & Thayamballi
  (2003) specimen table is not available offline; :func:`compare_with_experiments`
  is the harness for it and the shipped specimen file is marked ``WITHHELD``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "PlateField",
    "Stiffener",
    "StiffenedPanelSmith",
    "bulb_flat_equivalent_t",
    "csr_no47_panel",
    "marguerre_plate_path",
    "plate_load_shortening",
    "stiffener_ultimate",
    "stiffener_load_shortening",
    "panel_load_shortening",
    "csrf_faulkner_plate_ultimate",
    "csr_panel_chain",
    "interaction_load_factor",
    "interaction_components",
    "interaction_curve",
    "isotropic_reference_components",
    "plate_biaxial_path",
    "biaxial_equal_compression",
    "compare_with_experiments",
    "sig6",
    "sig6_tree",
    "POST_ULTIMATE_SLOPE",
    "PLATE_RESIDUAL_FLOOR",
]

POST_ULTIMATE_SLOPE = -0.10
"""Post-ultimate plate softening slope ``d sigma / d eps = slope * E``.

Documented assumption (see the module docstring): the large-deflection
single-mode solution delivers the rising branch and the ultimate but not the
post-ultimate decay, so a linear softening at ``-0.10 E`` is appended, which is
the range used by Smith-method load-shortening curves.  The evidence run
re-runs the acceptance chain at ``-0.05 E`` and ``-0.20 E`` to show the panel
ultimate is insensitive to it.
"""

PLATE_RESIDUAL_FLOOR = 0.40
"""Plate mean stress floor after softening, as a fraction of the ultimate."""

# --------------------------------------------------------------------------
# small helpers (input validation + 6-significant-digit landing)
# --------------------------------------------------------------------------
def _positive(name: str, value: float) -> float:
    x = float(value)
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return x


def _finite(name: str, value: float) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


def sig6(value: float) -> float:
    """Round to six significant digits (the landing format of the artefacts)."""
    return float(f"{float(value):.6g}")


def sig6_tree(obj):
    """Recursively round every float in a container to six significant digits."""
    if isinstance(obj, dict):
        return {k: sig6_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sig6_tree(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sig6_tree(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return sig6(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# --------------------------------------------------------------------------
# 1. plate field: Marguerre large-deflection path
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PlateField:
    """One plate field between longitudinal stiffeners.

    ``length`` is the stiffener span ``a`` (m), ``spacing`` the stiffener
    spacing ``b`` (m), ``thickness`` the plate thickness (m).  The plate mid
    surface is the reference surface; compression is positive throughout this
    module.
    """

    length: float
    spacing: float
    thickness: float
    young: float = 2.06e11
    poisson: float = 0.3
    yield_stress: float = 235e6
    imperfection_ratio: float = 0.1
    """``w_0 = imperfection_ratio * beta^2 * t``; the validation reference is 0.1."""
    residual_ratio: float = 0.2
    """``sigma_rc = residual_ratio * sigma_y``; 0 disables residual stresses."""
    mode_limit: int = 12

    def __post_init__(self) -> None:
        for name in ("length", "spacing", "thickness", "young", "yield_stress"):
            _positive(name, getattr(self, name))
        nu = _finite("poisson", self.poisson)
        if not -1.0 < nu < 0.5:
            raise ValueError("poisson must lie in (-1, 0.5)")
        if _finite("imperfection_ratio", self.imperfection_ratio) < 0.0:
            raise ValueError("imperfection_ratio must be non-negative")
        rr = _finite("residual_ratio", self.residual_ratio)
        if not 0.0 <= rr < 1.0:
            raise ValueError("residual_ratio must lie in [0, 1)")
        if int(self.mode_limit) < 1:
            raise ValueError("mode_limit must be at least 1")

    # -- geometry -------------------------------------------------------
    @property
    def aspect(self) -> float:
        return self.length / self.spacing

    @property
    def flexural_rigidity(self) -> float:
        return self.young * self.thickness**3 / (12.0 * (1.0 - self.poisson**2))

    @property
    def beta(self) -> float:
        """Plate slenderness ``beta = (b/t) sqrt(sigma_y/E)``."""
        return (self.spacing / self.thickness) * math.sqrt(self.yield_stress / self.young)

    @property
    def yield_strain(self) -> float:
        return self.yield_stress / self.young

    @property
    def imperfection_amplitude(self) -> float:
        """Initial deflection amplitude ``w_0`` (m)."""
        return self.imperfection_ratio * self.beta**2 * self.thickness

    @property
    def residual_stress(self) -> float:
        """Compressive residual stress in the plate middle (Pa, compression +)."""
        return self.residual_ratio * self.yield_stress

    @property
    def residual_zone_fraction(self) -> float:
        """Edge fraction ``2 eta`` carrying the tensile residual ``sigma_rt``.

        The tension block reaches yield (``sigma_rt = sigma_y``, the usual
        weld idealisation) and the field is self-equilibrated:
        ``2 eta sigma_rt = (1 - 2 eta) sigma_rc``.
        """
        rc = self.residual_stress
        if rc <= 0.0:
            return 0.0
        return rc / (self.yield_stress + rc)

    # -- elastic buckling -------------------------------------------------
    def buckling_stress(self, mode: int | None = None) -> dict:
        """Elastic uniaxial buckling stress, minimised over ``m`` unless given.

        ``sigma_cr = D (alpha^2 + beta_m^2)^2 / (t alpha^2)`` with
        ``alpha = m pi / a`` and ``beta_m = pi / b`` (simply supported edges).
        """
        k = int(mode) if mode is not None else None
        if k is not None and k < 1:
            raise ValueError("mode must be >= 1")
        best = None
        candidates = [k] if k is not None else range(1, int(self.mode_limit) + 1)
        for m in candidates:
            alpha = m * math.pi / self.length
            bet = math.pi / self.spacing
            sigma = (self.flexural_rigidity * (alpha**2 + bet**2) ** 2
                     / (self.thickness * alpha**2))
            if best is None or sigma < best[0]:
                best = (sigma, m)
        sigma, m = best
        return {
            "sigma_cr": sigma,
            "mode": m,
            "sigma_cr_over_yield": sigma / self.yield_stress,
            "buckling_coefficient": sigma * self.spacing**2 * self.thickness
            / (math.pi**2 * self.flexural_rigidity),
        }

    def shear_buckling_stress(self) -> float:
        """Elastic shear buckling stress, ``k_s = 5.34 + 4 (b/a)^2``."""
        k_s = 5.34 + 4.0 * (self.spacing / self.length) ** 2
        return k_s * math.pi**2 * self.flexural_rigidity / (self.spacing**2 * self.thickness)

    def validate_geometry(self) -> dict:
        """Geometric consistency checks before any strength claim."""
        checks = {
            "aspect_ratio": self.aspect,
            "slenderness_beta": self.beta,
            "aspect_in_range": 1.0 <= self.aspect <= 8.0,
            "slenderness_in_range": 0.5 <= self.beta <= 6.0,
            "thickness_over_spacing": self.thickness / self.spacing,
        }
        if not checks["aspect_in_range"]:
            raise ValueError(f"plate aspect ratio {self.aspect:.3f} outside 1..8")
        if not checks["slenderness_in_range"]:
            raise ValueError(f"plate slenderness beta {self.beta:.3f} outside 0.5..6")
        return checks


def _marguerre_amplitudes(plate: PlateField, count: int = 4000,
                          span_factor: float = 12.0) -> np.ndarray:
    """Additional-deflection amplitudes ``A`` covering the whole path."""
    w0 = plate.imperfection_amplitude
    start = max(1e-9, 1e-4 * plate.thickness)
    stop = span_factor * plate.thickness + 4.0 * w0
    return np.linspace(start, stop, int(count))


def _path_point(plate: PlateField, alpha: float, bet: float, w0: float,
                amplitude: np.ndarray, sigma_cr: float,
                stress_ratio: float = 0.0) -> dict:
    """Evaluate the single-mode Galerkin equilibrium at prescribed amplitudes.

    ``sigma = (sigma_cr - sigma_rc) A/(A + w_0) + sigma_1/2
              + (beta^2/alpha^2) sigma_2/2``, ``sigma_1 = E Psi alpha^2/8``,
    ``sigma_2 = E Psi beta^2/8``, ``Psi = A(A + 2 w_0)``;
    mean shortening ``eps = sigma/E + alpha^2 Psi/8``; the longitudinal edge
    carries ``sigma + sigma_1``.  The ``- sigma_rc`` on the leading term is the
    residual-stress destabilisation of the buckle discussed in the module
    docstring.
    """
    r = float(stress_ratio)
    psi = amplitude * (amplitude + 2.0 * w0)
    sigma_1 = plate.young * psi * alpha**2 / 8.0
    sigma_2 = plate.young * psi * bet**2 / 8.0
    leading = float(max(sigma_cr - plate.residual_stress, 1e-3 * sigma_cr))
    denom = alpha**2 + r * bet**2
    sigma = (leading * alpha**2 * amplitude / ((amplitude + w0) * denom)
             + 0.5 * (alpha**2 * sigma_1 + bet**2 * sigma_2) / denom)
    eps = sigma / plate.young + alpha**2 * psi / 8.0
    sx = sigma + sigma_1
    sy = r * sigma + sigma_2
    if r > 0.0:
        # Biaxial: von Mises on the two edge membrane stresses.
        edge = np.sqrt(np.maximum(sx**2 - sx * sy + sy**2, 0.0))
    else:
        # Uniaxial: the classical edge-yield criterion is on the longitudinal
        # membrane stress alone (the transverse oscillation is a self-equilibrated
        # residual-type field, and this is the criterion the Faulkner closed form
        # is built on).
        edge = sx
    return {"sigma": sigma, "eps": eps, "edge": edge, "sigma_x": sx, "sigma_y": sy,
            "sigma_1": sigma_1, "leading": leading}


def plate_biaxial_path(plate: PlateField, stress_ratio: float = 1.0, *,
                       mode: int | None = None, count: int = 4000,
                       span_factor: float = 12.0) -> dict:
    """Large-deflection path under proportional biaxial compression.

    ``stress_ratio`` is ``sigma_y / sigma_x`` (1 -> equal biaxial compression).
    Two things change with respect to :func:`marguerre_plate_path`: the buckling
    lead term becomes ``sigma_cr alpha^2/(alpha^2 + r beta^2)`` (for ``r = 1``
    and ``alpha = beta`` this is the classical halving of the biaxial buckling
    stress), and the edge criterion becomes von Mises on the edge membrane
    stresses.  Identical to the uniaxial path at ``stress_ratio = 0``.
    """
    plate.validate_geometry()
    r = _finite("stress_ratio", stress_ratio)
    if r < 0.0:
        raise ValueError("stress_ratio must be non-negative")
    chosen = plate.buckling_stress(mode)
    m = chosen["mode"]
    alpha = m * math.pi / plate.length
    bet = math.pi / plate.spacing
    w0 = plate.imperfection_amplitude
    A = _marguerre_amplitudes(plate, count=count, span_factor=span_factor)
    point = _path_point(plate, alpha, bet, w0, A, chosen["sigma_cr"], r)
    admissible = point["edge"] <= plate.yield_stress
    if not admissible.any():
        raise ValueError("the edge is yielding at the smallest amplitude")
    idx = int(np.argmax(np.where(admissible, point["sigma"], -np.inf)))
    return {
        "mode": m,
        "stress_ratio": r,
        "amplitude": A,
        "mean_stress": point["sigma"],
        "mean_strain": point["eps"],
        "von_mises_edge_stress": point["edge"],
        "elastic_buckling_stress": chosen["sigma_cr"],
        "ultimate_stress": float(point["sigma"][idx]),
        "ultimate_over_yield": float(point["sigma"][idx] / plate.yield_stress),
        "ultimate_strain": float(point["eps"][idx]),
    }


def marguerre_plate_path(plate: PlateField, *, mode: int | None = None,
                         count: int = 4000, span_factor: float = 12.0) -> dict:
    """Single-mode large-deflection path of the plate field, up to edge yield.

    The Galerkin projection of the von Karman plate equations onto
    ``w = A sin(m pi x/a) sin(pi y/b)`` delivers ``sigma(A)`` and ``eps(A)``
    (see :func:`_path_point`).  The plate ultimate is the largest mean stress
    reached before the longitudinal edge reaches yield; the crossing is located
    by bisection so the reported ultimate does not depend on the amplitude grid.
    """
    plate.validate_geometry()
    chosen = plate.buckling_stress(mode)
    m = chosen["mode"]
    alpha = m * math.pi / plate.length
    bet = math.pi / plate.spacing
    w0 = plate.imperfection_amplitude
    A = _marguerre_amplitudes(plate, count=count, span_factor=span_factor)
    point = _path_point(plate, alpha, bet, w0, A, chosen["sigma_cr"])
    sigma, eps, edge = point["sigma"], point["eps"], point["edge"]
    sigma_1 = point["sigma_1"]
    if edge[0] > plate.yield_stress:
        raise ValueError(
            "the longitudinal edge is already yielding at the smallest "
            "amplitude; the geometry is too stocky for this single-mode path"
        )
    admissible = edge <= plate.yield_stress
    idx = int(np.argmax(np.where(admissible, sigma, -np.inf)))
    if idx + 1 < A.size:
        lo, hi = float(A[idx]), float(A[idx + 1])
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            test = _path_point(plate, alpha, bet, w0, np.array([mid]), chosen["sigma_cr"])
            if float(test["edge"][0]) <= plate.yield_stress:
                lo = mid
            else:
                hi = mid
        refined = _path_point(plate, alpha, bet, w0, np.array([lo]), chosen["sigma_cr"])
        A = np.append(A[: idx + 1], lo)
        sigma = np.append(sigma[: idx + 1], refined["sigma"][0])
        eps = np.append(eps[: idx + 1], refined["eps"][0])
        edge = np.append(edge[: idx + 1], refined["edge"][0])
        sigma_1 = np.append(sigma_1[: idx + 1], refined["sigma_1"][0])
        idx = A.size - 1
    return {
        "mode": m,
        "amplitude": A,
        "mean_stress": sigma,
        "mean_strain": eps,
        "edge_stress": edge,
        "membrane_oscillation": sigma_1,
        "elastic_buckling_stress": chosen["sigma_cr"],
        "effective_buckling_stress": float(point["leading"]),
        "ultimate_stress": float(sigma[idx]),
        "ultimate_strain": float(eps[idx]),
        "ultimate_over_yield": float(sigma[idx] / plate.yield_stress),
        "ultimate_index": idx,
        "edge_stress_at_ultimate": float(edge[idx]),
    }


# --------------------------------------------------------------------------
# 2. residual stresses + plate load-shortening curve
# --------------------------------------------------------------------------
def _plate_residual_mean(plate: PlateField, sigma_plate: np.ndarray) -> np.ndarray:
    """Two-zone residual-stress superposition at equal strain.

    Each zone follows the same plate mean-stress law shifted by its residual
    stress (compression positive) and clipped at tensile/compressive yield; the
    area-weighted mean is the plate mean stress.  With ``residual_ratio = 0``
    this reduces exactly to the input curve, and while no zone has yielded it
    returns the input curve identically (the residual field is self-equilibrated).
    """
    eta2 = plate.residual_zone_fraction
    if eta2 <= 0.0:
        return np.array(sigma_plate, dtype=float)
    sy = plate.yield_stress
    rc = plate.residual_stress
    rt = sy  # the weld tension block reaches yield
    mid = np.clip(sigma_plate + rc, -sy, sy)
    edge = np.clip(sigma_plate - rt, -sy, sy)
    return (1.0 - eta2) * mid + eta2 * edge


def _rising_curve(plate: PlateField, path: dict) -> tuple:
    """Assemble the pre-ultimate curve: elastic branch + post-buckling branch.

    The uniform-strain solution ``sigma = E eps`` is used until the *additional*
    deflection amplitude equals the initial imperfection amplitude ``w_0`` --
    below that amplitude the plate is still in its pre-buckling state and the
    single-mode solution's geometric shortening would be spurious (it would
    report a 20-40 % stiffness loss that no real plate shows).  Beyond the
    junction the strain increment is taken from the large-deflection path, which
    supplies the correct post-buckling compliance, and the two branches are
    matched in strain and stress.
    """
    sigma_p = np.asarray(path["mean_stress"], dtype=float)
    eps_p = np.asarray(path["mean_strain"], dtype=float)
    amplitude = np.asarray(path["amplitude"], dtype=float)
    w0 = plate.imperfection_amplitude
    i0 = int(np.argmin(np.abs(amplitude - w0)))
    sig_j = sigma_p[i0]
    eps_j = sig_j / plate.young
    eps_post = eps_j + (eps_p[i0:] - eps_p[i0])
    sig_post = sigma_p[i0:]
    eps_elastic = np.linspace(0.0, eps_j, 100)
    sig_elastic = plate.young * eps_elastic
    eps_curve = np.concatenate([eps_elastic, eps_post[1:]])
    sig_curve = np.concatenate([sig_elastic, sig_post[1:]])
    keep = np.concatenate([[True], np.diff(eps_curve) > 0.0])
    return eps_curve[keep], sig_curve[keep]


def plate_load_shortening(plate: PlateField, *, strains: Sequence[float] | None = None,
                          path: dict | None = None, n_points: int = 400) -> dict:
    """Plate load--shortening curve ``sigma(eps)`` with residual stresses.

    Returns the tabulated curve, its peak (the plate ultimate) and the flags the
    acceptance criteria read.  ``strains`` overrides the default grid, which
    spans ``0 .. 8 eps_y``.
    """
    path = path if path is not None else marguerre_plate_path(plate)
    eps_curve, sig_curve = _rising_curve(plate, path)
    sig_curve = _plate_residual_mean(plate, sig_curve)
    peak_index = int(np.argmax(sig_curve))
    eps_u = float(eps_curve[peak_index])
    sig_u = float(sig_curve[peak_index])
    rising_eps = eps_curve[: peak_index + 1]
    rising_sig = sig_curve[: peak_index + 1]

    ey = plate.yield_strain
    if strains is None:
        strains = np.concatenate([
            np.linspace(0.0, eps_u, n_points // 2, endpoint=False),
            np.linspace(eps_u, 8.0 * ey, n_points // 2),
        ])
    eps = np.asarray(strains, dtype=float)
    sigma = np.zeros_like(eps)
    branch = np.empty(eps.size, dtype=object)
    for i, e in enumerate(eps):
        if e <= eps_u:
            sigma[i] = np.interp(e, rising_eps, rising_sig)
            branch[i] = "rising"
        else:
            soften = sig_u + POST_ULTIMATE_SLOPE * plate.young * (e - eps_u)
            floor = PLATE_RESIDUAL_FLOOR * sig_u
            if soften > floor:
                sigma[i] = soften
                branch[i] = "softening"
            else:
                sigma[i] = floor
                branch[i] = "residual"

    peak = int(np.argmax(sigma))
    return {
        "strains": eps,
        "stress": sigma,
        "branch": branch,
        "rising_strains": rising_eps,
        "rising_stress": rising_sig,
        "beta": plate.beta,
        "imperfection_amplitude": plate.imperfection_amplitude,
        "imperfection_over_thickness": plate.imperfection_amplitude / plate.thickness,
        "residual_stress": plate.residual_stress,
        "residual_zone_fraction": plate.residual_zone_fraction,
        "elastic_buckling_stress": path["elastic_buckling_stress"],
        "effective_buckling_stress": path["effective_buckling_stress"],
        "mode": path["mode"],
        "sigma_u": float(sigma[peak]),
        "eps_u": float(eps[peak]),
        "sigma_u_over_yield": float(sigma[peak] / plate.yield_stress),
        "eps_u_over_yield_strain": float(eps[peak] / plate.yield_strain),
        "has_negative_stiffness": bool(np.any(np.diff(sigma[peak:]) < 0.0)),
        "post_peak_slope": float(np.gradient(sigma, eps)[-1]),
    }


# --------------------------------------------------------------------------
# 3. stiffener
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Stiffener:
    """T-section longitudinal stiffener with an attached plate flange."""

    span: float
    web_height: float
    web_thickness: float
    flange_width: float
    flange_thickness: float
    young: float = 2.06e11
    poisson: float = 0.3
    yield_stress: float = 235e6
    initial_bow_ratio: float = 0.002
    """Initial lateral bow ``delta_0 = initial_bow_ratio * span``."""

    def __post_init__(self) -> None:
        for name in ("span", "web_height", "web_thickness", "flange_width",
                     "flange_thickness", "young", "yield_stress"):
            _positive(name, getattr(self, name))
        if self.flange_width < self.web_thickness:
            raise ValueError("flange_width must not be narrower than the web")
        if _finite("initial_bow_ratio", self.initial_bow_ratio) < 0.0:
            raise ValueError("initial_bow_ratio must be non-negative")

    @property
    def area(self) -> float:
        return self.web_height * self.web_thickness + self.flange_width * self.flange_thickness

    @property
    def centroid_from_plate(self) -> float:
        """Distance from the plate mid-surface to the stiffener-only centroid."""
        a_w = self.web_height * self.web_thickness
        a_f = self.flange_width * self.flange_thickness
        return (a_w * 0.5 * self.web_height
                + a_f * (self.web_height + 0.5 * self.flange_thickness)) / (a_w + a_f)

    @property
    def own_inertia(self) -> float:
        """Second moment of the stiffener alone about the plate mid-surface."""
        a_w = self.web_height * self.web_thickness
        a_f = self.flange_width * self.flange_thickness
        i_w = self.web_thickness * self.web_height**3 / 12.0 + a_w * (0.5 * self.web_height) ** 2
        i_f = (self.flange_width * self.flange_thickness**3 / 12.0
               + a_f * (self.web_height + 0.5 * self.flange_thickness) ** 2)
        return i_w + i_f

    @property
    def initial_bow(self) -> float:
        return self.initial_bow_ratio * self.span

    # -- local buckling ---------------------------------------------------
    def flange_outstand_buckling(self) -> float:
        """Elastic buckling of the flange outstand, ``k = 0.425``."""
        outstand = max(0.5 * (self.flange_width - self.web_thickness), 1e-12)
        return (0.425 * math.pi**2 * self.young
                / (12.0 * (1.0 - self.poisson**2)) * (self.flange_thickness / outstand) ** 2)

    def web_local_buckling(self) -> float:
        """Elastic buckling of the web as a plate, ``k = 4``."""
        return (4.0 * math.pi**2 * self.young
                / (12.0 * (1.0 - self.poisson**2)) * (self.web_thickness / self.web_height) ** 2)

    # -- column ------------------------------------------------------------
    def attached_section(self, plate: PlateField, effective_width: float) -> dict:
        """Section properties of stiffener + effective plate width."""
        a_p = effective_width * plate.thickness
        a_s = self.area
        a_e = a_p + a_s
        # Stiffener hangs below the plate mid-surface (negative z), the plate
        # sits on it; the centroid is measured downwards from the plate.
        z_s = self.centroid_from_plate
        z_c = a_s * z_s / a_e
        i_e = (self.own_inertia - a_s * z_s**2) + a_s * (z_s - z_c) ** 2 + a_p * z_c**2
        return {
            "area": a_e,
            "stiffener_area": a_s,
            "plate_area": a_p,
            "centroid": z_c,
            "inertia": i_e,
            "effective_width": effective_width,
        }


def stiffener_ultimate(stiffener: Stiffener, plate: PlateField,
                       *, effective_width: float | None = None) -> dict:
    """Stiffener axial ultimate: beam-column (Perry) vs local buckling.

    The beam-column strength uses the Perry--Robertson form

    ``sigma_u = (sigma_y + sigma_E (1 + eta)) / 2
                - sqrt( ((sigma_y + sigma_E (1 + eta))/2)^2 - sigma_y sigma_E )``

    with ``eta = delta_0 z_c A_e / I_e`` the initial-bow imperfection parameter
    and ``sigma_E`` the Euler stress of stiffener + effective plate.  The
    effective attached plate width defaults to the plate effective width at its
    own ultimate, ``b_e = b sigma_u,plate / sigma_y``.
    """
    if effective_width is None:
        curve = plate_load_shortening(plate)
        effective_width = min(plate.spacing, plate.spacing * curve["sigma_u_over_yield"])
    _positive("effective_width", effective_width)
    section = stiffener.attached_section(plate, effective_width)
    i_e = section["inertia"]
    a_e = section["area"]
    sigma_e = math.pi**2 * stiffener.young * i_e / (a_e * stiffener.span**2)
    sy = stiffener.yield_stress
    eta = stiffener.initial_bow * section["centroid"] * a_e / i_e
    mean = 0.5 * (sy + sigma_e * (1.0 + eta))
    disc = mean**2 - sy * sigma_e
    sigma_column = mean - math.sqrt(max(disc, 0.0))
    sigma_flange = stiffener.flange_outstand_buckling()
    sigma_web = stiffener.web_local_buckling()
    candidates = {
        "beam_column_perry": sigma_column,
        "flange_outstand": min(sigma_flange, sy),
        "web_local": min(sigma_web, sy),
        "yield": sy,
    }
    governing = min(candidates, key=lambda k: candidates[k])
    sigma_u = min(candidates[governing], sy)
    return {
        "euler_stress": sigma_e,
        "euler_over_yield": sigma_e / sy,
        "initial_bow": stiffener.initial_bow,
        "perry_eta": eta,
        "beam_column_strength": sigma_column,
        "flange_outstand_stress": sigma_flange,
        "web_local_stress": sigma_web,
        "governing_mode": governing,
        "sigma_u": sigma_u,
        "sigma_u_over_yield": sigma_u / sy,
        "section": section,
    }


def stiffener_load_shortening(stiffener: Stiffener, plate: PlateField, *,
                              strains: Sequence[float] | None = None,
                              ultimate: dict | None = None,
                              plateau_end: float = 3.0,
                              softening_slope: float = -0.05) -> dict:
    """Stiffener ``sigma(eps)``: elastic, plastic plateau, linear softening."""
    ultimate = ultimate if ultimate is not None else stiffener_ultimate(stiffener, plate)
    sy = stiffener.yield_stress
    ey = sy / stiffener.young
    sig_u = ultimate["sigma_u"]
    eps_u = sig_u / stiffener.young
    eps_plateau_end = max(plateau_end * ey, eps_u)
    if strains is None:
        strains = np.linspace(0.0, 8.0 * ey, 400)
    eps = np.asarray(strains, dtype=float)
    sigma = np.where(
        eps <= eps_u, stiffener.young * eps,
        np.where(eps <= eps_plateau_end, sig_u,
                 np.maximum(sig_u + softening_slope * stiffener.young * (eps - eps_plateau_end),
                            0.3 * sig_u)))
    return {
        "strains": eps,
        "stress": sigma,
        "sigma_u": sig_u,
        "eps_u": eps_u,
        "eps_plateau_end": eps_plateau_end,
        "ultimate": ultimate,
    }


# --------------------------------------------------------------------------
# 4. panel unit and the Smith combination
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StiffenedPanelSmith:
    """One stiffened plate unit: a plate field plus one longitudinal stiffener."""

    plate: PlateField
    stiffener: Stiffener
    label: str = "panel"

    def __post_init__(self) -> None:
        if not isinstance(self.plate, PlateField):
            raise TypeError("plate must be a PlateField")
        if not isinstance(self.stiffener, Stiffener):
            raise TypeError("stiffener must be a Stiffener")
        if abs(self.stiffener.young - self.plate.young) > 1e-6 * self.plate.young:
            raise ValueError("plate and stiffener must share the same Young modulus")
        self.plate.validate_geometry()

    @property
    def area(self) -> float:
        return self.plate.spacing * self.plate.thickness + self.stiffener.area

    @property
    def plate_area_fraction(self) -> float:
        return self.plate.spacing * self.plate.thickness / self.area

    def summary(self) -> dict:
        return {
            "label": self.label,
            "length": self.plate.length,
            "spacing": self.plate.spacing,
            "thickness": self.plate.thickness,
            "slenderness_beta": self.plate.beta,
            "aspect_ratio": self.plate.aspect,
            "stiffener_area": self.stiffener.area,
            "plate_area": self.plate.spacing * self.plate.thickness,
            "total_area": self.area,
            "plate_area_fraction": self.plate_area_fraction,
            "yield_stress": self.plate.yield_stress,
            "young": self.plate.young,
            "imperfection_ratio": self.plate.imperfection_ratio,
            "residual_ratio": self.plate.residual_ratio,
            "stiffener_web": (self.stiffener.web_height, self.stiffener.web_thickness),
            "stiffener_flange": (self.stiffener.flange_width, self.stiffener.flange_thickness),
        }


def panel_load_shortening(panel: StiffenedPanelSmith, *,
                          strains: Sequence[float] | None = None) -> dict:
    """Smith combination: area-weighted parallel plate + stiffener curves."""
    plate = panel.plate
    ey = plate.yield_strain
    if strains is None:
        strains = np.concatenate([
            np.linspace(0.0, 1.5 * ey, 300),
            np.linspace(1.5 * ey, 8.0 * ey, 200)[1:],
        ])
    eps = np.asarray(strains, dtype=float)
    p_curve = plate_load_shortening(plate, strains=eps)
    s_curve = stiffener_load_shortening(panel.stiffener, plate, strains=eps)
    a_p = plate.spacing * plate.thickness
    a_s = panel.stiffener.area
    sigma = (a_p * p_curve["stress"] + a_s * s_curve["stress"]) / (a_p + a_s)
    peak = int(np.argmax(sigma))
    return {
        "strains": eps,
        "panel_stress": sigma,
        "plate_stress": p_curve["stress"],
        "stiffener_stress": s_curve["stress"],
        "sigma_u": float(sigma[peak]),
        "eps_u": float(eps[peak]),
        "sigma_u_over_yield": float(sigma[peak] / plate.yield_stress),
        "eps_u_over_yield_strain": float(eps[peak] / ey),
        "plate_curve": p_curve,
        "stiffener_curve": s_curve,
        "has_negative_stiffness": bool(np.any(np.diff(sigma[peak:]) < 0.0)),
    }


# --------------------------------------------------------------------------
# 5. CSR-style closed-form cross-check (independent of the Smith integration)
# --------------------------------------------------------------------------
def csrf_faulkner_plate_ultimate(beta: float, yield_stress: float) -> float:
    """Faulkner (1975) uniaxial plate ultimate ``sigma_y (2/beta - 1/beta^2)``."""
    b = _positive("beta", beta)
    if b <= 1.0:
        return yield_stress
    return yield_stress * (2.0 / b - 1.0 / b**2)


def _johnson_ostenfeld(sigma_y: float, sigma_e: float) -> float:
    """Inelastic column curve used by several class rules."""
    if sigma_e >= 0.5 * sigma_y:
        return sigma_y * (1.0 - sigma_y / (4.0 * sigma_e))
    return sigma_e


def csr_panel_chain(panel: StiffenedPanelSmith) -> dict:
    """Rule-type closed-form axial capacity of the panel unit.

    Independent of :func:`panel_load_shortening`: the plate side uses the
    Faulkner effective width, the stiffener side the Johnson--Ostenfeld column
    curve with the attached effective plate (``b_e = b sigma_cp / sigma_y``), and
    the unit capacity is the area-weighted mean of the two component strengths.
    """
    plate = panel.plate
    sy = plate.yield_stress
    sigma_cp = csrf_faulkner_plate_ultimate(plate.beta, sy)
    b_e = min(plate.spacing, plate.spacing * sigma_cp / sy)
    section = panel.stiffener.attached_section(plate, b_e)
    i_e, a_e = section["inertia"], section["area"]
    sigma_e = math.pi**2 * plate.young * i_e / (a_e * panel.stiffener.span**2)
    sigma_cs = min(_johnson_ostenfeld(sy, sigma_e), sy)
    sigma_cs = min(sigma_cs, panel.stiffener.flange_outstand_buckling(),
                   panel.stiffener.web_local_buckling())
    a_p = plate.spacing * plate.thickness
    a_s = panel.stiffener.area
    sigma_panel = (a_p * sigma_cp + a_s * sigma_cs) / (a_p + a_s)
    return {
        "plate_ultimate": sigma_cp,
        "plate_ultimate_over_yield": sigma_cp / sy,
        "attached_effective_width": b_e,
        "column_euler_stress": sigma_e,
        "column_strength": sigma_cs,
        "column_strength_over_yield": sigma_cs / sy,
        "panel_capacity": sigma_panel,
        "panel_capacity_over_yield": sigma_panel / sy,
        "governing_component": "plate" if sigma_cp < sigma_cs else "stiffener",
    }


# --------------------------------------------------------------------------
# 6. interaction surface (biaxial compression + shear)
# --------------------------------------------------------------------------
def _transverse_plate_ultimate(panel: StiffenedPanelSmith) -> float:
    """Panel mean stress at transverse (across-stiffener) collapse.

    The plate field's uniaxial ultimate is a function of the slenderness
    ``beta = (b/t) sqrt(sigma_y/E)`` only, and transverse compression loads the
    same strips between longitudinal stiffeners with the same ``k = 4`` buckling
    family (the transverse frames act as the end supports), so the plate
    ultimate is direction independent in this model.  The stiffeners carry no
    transverse membrane load, so the *panel mean stress* is the plate ultimate
    weighted by the plate area fraction -- this is what makes the panel
    interaction surface anisotropic, and the anisotropy is reported rather than
    assumed away.
    """
    plate_u = plate_load_shortening(panel.plate)["sigma_u"]
    return plate_u * panel.plate_area_fraction


def interaction_load_factor(panel: StiffenedPanelSmith, sigma_x: float, sigma_y: float = 0.0,
                            tau: float = 0.0, *, coupling: float = 0.0,
                            components: dict | None = None) -> dict:
    """CSR-style quadratic interaction utilisation (``= 1`` at the surface).

    ``(sx/sxu)^2 + (sy/syu)^2 - coupling (sx sy)/(sxu syu) + (tau/tauu)^2 = 1``

    ``coupling = 0`` is the conservative uncoupled quadratic form; a positive
    coupling is the usual Smith-type refinement that allows extra capacity when
    both stresses have the same sign.  ``tau_u`` is the smaller of the von Mises
    shear yield ``sigma_y/sqrt(3)`` and the elastic shear buckling stress (the
    tension-field reserve is deliberately omitted, so the shear side is
    conservative).
    """
    comp = components if components is not None else interaction_components(panel)
    sx = _finite("sigma_x", sigma_x)
    sy = _finite("sigma_y", sigma_y)
    tq = _finite("tau", tau)
    c = _finite("coupling", coupling)
    if not -1.0 <= c <= 1.0:
        raise ValueError("coupling must lie in [-1, 1]")
    util = (sx / comp["sigma_xu"]) ** 2 + (sy / comp["sigma_yu"]) ** 2 \
        - c * sx * sy / (comp["sigma_xu"] * comp["sigma_yu"]) + (tq / comp["tau_u"]) ** 2
    return {
        "utilisation_squared": float(util),
        "utilisation": float(math.sqrt(max(util, 0.0))),
        "components": comp,
        "coupling": c,
    }


def interaction_components(panel: StiffenedPanelSmith) -> dict:
    """Component capacities entering the interaction surface."""
    curve = panel_load_shortening(panel)
    sigma_xu = curve["sigma_u"]
    sigma_yu = _transverse_plate_ultimate(panel)
    plate = panel.plate
    tau_yield = plate.yield_stress / math.sqrt(3.0)
    tau_cr = plate.shear_buckling_stress()
    return {
        "sigma_xu": sigma_xu,
        "sigma_xu_over_yield": sigma_xu / plate.yield_stress,
        "sigma_yu": sigma_yu,
        "sigma_yu_over_yield": sigma_yu / plate.yield_stress,
        "tau_u": min(tau_yield, tau_cr),
        "tau_yield": tau_yield,
        "tau_buckling": tau_cr,
        "plate_sigma_u": curve["plate_curve"]["sigma_u"],
        "yield_stress": plate.yield_stress,
    }


def biaxial_equal_compression(panel: StiffenedPanelSmith | None = None, *,
                              components: dict | None = None,
                              coupling: float = 0.0) -> dict:
    """Ultimate under equal biaxial compression (proportional path).

    Solving the interaction surface for ``sigma_x = sigma_y = sigma`` gives
    ``sigma = 1/sqrt(1/sxu^2 + 1/syu^2 - c/(sxu syu))``; for an *isotropic*
    panel (``syu = sxu``) and ``c = 0`` this is exactly ``sxu/sqrt(2)``, the
    classical ``1/sqrt(2)`` interaction reduction.  For the CSR panel the
    stiffeners do not carry the transverse load, so ``syu < sxu`` and the
    reduction is stronger than ``1/sqrt(2)``; both numbers are reported.
    """
    if components is None:
        if panel is None:
            raise ValueError("either panel or components is required")
        components = interaction_components(panel)
    comp = components
    c = _finite("coupling", coupling)
    inv = 1.0 / comp["sigma_xu"] ** 2 + 1.0 / comp["sigma_yu"] ** 2 \
        - c / (comp["sigma_xu"] * comp["sigma_yu"])
    sigma = 1.0 / math.sqrt(inv)
    yield_stress = comp.get("yield_stress")
    return {
        "sigma": sigma,
        "sigma_over_uniaxial_ultimate": sigma / comp["sigma_xu"],
        "sigma_over_yield": (sigma / yield_stress) if yield_stress else None,
        "target_over_uniaxial_ultimate": 1.0 / math.sqrt(2.0),
        "isotropic_limit_over_uniaxial_ultimate": 1.0 / math.sqrt(2.0),
        "anisotropy_ratio_yu_over_xu": comp["sigma_yu"] / comp["sigma_xu"],
        "components": comp,
        "coupling": c,
    }


def isotropic_reference_components(plate: PlateField) -> dict:
    """Interaction components of an isotropic plate (``sxu = syu``).

    Used to expose the ``1/sqrt(2)`` interaction reduction on a case where the
    surface is isotropic, and as the reference for the physics-based biaxial
    large-deflection model of :func:`plate_biaxial_path`.
    """
    curve = plate_load_shortening(plate)
    sigma_u = curve["sigma_u"]
    return {
        "sigma_xu": sigma_u,
        "sigma_xu_over_yield": sigma_u / plate.yield_stress,
        "sigma_yu": sigma_u,
        "sigma_yu_over_yield": sigma_u / plate.yield_stress,
        "tau_u": min(plate.yield_stress / math.sqrt(3.0), plate.shear_buckling_stress()),
        "plate_sigma_u": sigma_u,
        "yield_stress": plate.yield_stress,
    }


def interaction_curve(panel: StiffenedPanelSmith, *, angles: Iterable[float] | None = None,
                      coupling: float = 0.0) -> dict:
    """Radial load-factor scan of the biaxial surface (``tau = 0``)."""
    comp = interaction_components(panel)
    if angles is None:
        angles = np.linspace(0.0, 0.5 * math.pi, 37)
    rows = []
    for theta in angles:
        cx, cy = math.cos(theta), math.sin(theta)
        lf = 1.0 / math.sqrt((cx / comp["sigma_xu"]) ** 2 + (cy / comp["sigma_yu"]) ** 2
                             - coupling * cx * cy / (comp["sigma_xu"] * comp["sigma_yu"]))
        rows.append({
            "theta_deg": math.degrees(theta),
            "sigma_x": lf * cx,
            "sigma_y": lf * cy,
            "load_factor": lf,
            "sigma_x_over_yield": lf * cx / panel.plate.yield_stress,
            "sigma_y_over_yield": lf * cy / panel.plate.yield_stress,
        })
    return {"rows": rows, "components": comp, "coupling": coupling}


# --------------------------------------------------------------------------
# 7. the IACS CSR No. 47 reference panel
# --------------------------------------------------------------------------
def bulb_flat_equivalent_t(height: float, thickness: float, *,
                           flange_width_ratio: float = 0.30,
                           web_height_ratio: float = 0.90,
                           yield_stress: float = 235e6,
                           young: float = 2.06e11,
                           poisson: float = 0.3) -> Stiffener:
    """Idealise a bulb flat as an equivalent T-section.

    A bulb flat is not a T-bar; the equivalent T used here keeps the overall
    depth (``web_height_ratio * height`` for the web, the rest absorbed by the
    flange thickness) and replaces the bulb by a flange of width
    ``flange_width_ratio * height`` and the same plate thickness.  No catalogue
    area is quoted: the idealisation is explicit so the capacity can be
    reproduced from the geometry.
    """
    _positive("height", height)
    _positive("thickness", thickness)
    _positive("flange_width_ratio", flange_width_ratio)
    _positive("web_height_ratio", web_height_ratio)
    return Stiffener(
        span=1.0,  # overwritten by the caller
        web_height=web_height_ratio * height,
        web_thickness=thickness,
        flange_width=flange_width_ratio * height,
        flange_thickness=thickness,
        young=young,
        poisson=poisson,
        yield_stress=yield_stress,
    )


def csr_no47_panel(*, span: float = 2.4, spacing: float = 0.8, thickness: float = 0.012,
                   stiffener_height: float = 0.200, stiffener_thickness: float = 0.010,
                   yield_stress: float = 235e6, young: float = 2.06e11,
                   poisson: float = 0.3, imperfection_ratio: float = 0.1,
                   residual_ratio: float = 0.2) -> StiffenedPanelSmith:
    """The IACS CSR No. 47 reference panel of the validation brief.

    ``a = 2400 mm``, ``s = 800 mm``, ``t = 12 mm``, longitudinal ``HP 200 x 10``
    (bulb flat, idealised by :func:`bulb_flat_equivalent_t`), steel
    ``sigma_y = 235 MPa``, ``E = 206 GPa``.
    """
    plate = PlateField(length=span, spacing=spacing, thickness=thickness, young=young,
                       poisson=poisson, yield_stress=yield_stress,
                       imperfection_ratio=imperfection_ratio,
                       residual_ratio=residual_ratio)
    base = bulb_flat_equivalent_t(stiffener_height, stiffener_thickness,
                                  yield_stress=yield_stress, young=young, poisson=poisson)
    stiffener = Stiffener(span=span, web_height=base.web_height,
                          web_thickness=base.web_thickness,
                          flange_width=base.flange_width,
                          flange_thickness=base.flange_thickness,
                          young=young, poisson=poisson, yield_stress=yield_stress)
    return StiffenedPanelSmith(plate=plate, stiffener=stiffener, label="IACS-CSR-No47")


# --------------------------------------------------------------------------
# 8. experimental harness (no data invented)
# --------------------------------------------------------------------------
def compare_with_experiments(specimens: Sequence[dict], *,
                             young: float = 2.06e11,
                             poisson: float = 0.3) -> dict:
    """Run the Smith chain on a specimen table and tabulate the deviations.

    Each specimen is ``{"name", "a", "s", "t", "web_height", "web_thickness",
    "flange_width", "flange_thickness", "yield_stress", "sigma_u_measured"}``
    with SI units (m, Pa).  The returned ``status`` is ``"WITHHELD"`` when the
    table is empty, so a missing reference can never be reported as agreement.
    """
    specimens = list(specimens)
    if not specimens:
        return {
            "status": "WITHHELD",
            "reason": "no measured ultimate strengths were supplied",
            "rows": [],
            "max_relative_error": None,
        }
    rows = []
    for spec in specimens:
        plate = PlateField(length=spec["a"], spacing=spec["s"], thickness=spec["t"],
                           young=young, poisson=poisson,
                           yield_stress=spec["yield_stress"],
                           imperfection_ratio=spec.get("imperfection_ratio", 0.1),
                           residual_ratio=spec.get("residual_ratio", 0.2))
        stiff = Stiffener(span=spec["a"], web_height=spec["web_height"],
                          web_thickness=spec["web_thickness"],
                          flange_width=spec["flange_width"],
                          flange_thickness=spec["flange_thickness"],
                          young=young, poisson=poisson,
                          yield_stress=spec["yield_stress"])
        panel = StiffenedPanelSmith(plate=plate, stiffener=stiff, label=spec["name"])
        computed = panel_load_shortening(panel)["sigma_u"]
        measured = _positive("sigma_u_measured", spec["sigma_u_measured"])
        rows.append({
            "name": spec["name"],
            "sigma_u_computed": computed,
            "sigma_u_measured": measured,
            "relative_error": computed / measured - 1.0,
        })
    worst = max(abs(r["relative_error"]) for r in rows)
    return {
        "status": "COMPUTED",
        "rows": rows,
        "max_relative_error": worst,
        "specimen_count": len(rows),
    }