"""Local contact pressure--area measurement for the ice-coupled 6-DOF study (W24).

What this module is
-------------------
The W22/W23 ice-coupled-6DOF chain used the ISO 19906 global pressure--area law

    p_G = C_R (A / A_0) ** ex                                        (1)

as the *pressure source* of the hull--ice contact (see
``benchmarks/ice_coupled_6dof/cummins_ice_6dof.py``).  Every force therefore
contains (1) by construction, so a power law fitted to the simulated
(pressure, area) pairs of that chain can only return the exponent that was put
in: it is a closure check of the bookkeeping, *not* a test of the standard.
This module supplies the two ingredients the campaign node
``paper-exp-ice-coupled-6dof-4`` needs to separate those two statements:

1. **A recorder.**  :class:`RecordingContactLaw` is injected into
   :meth:`Cummins6DOF.run(..., contact_law=...)` and records, for every ice
   impact, the peak contact force ``N``, the nominal contact area ``A`` at that
   instant and the pressure ``p = N / A``, plus the impulse-weighted mean
   pressure ``p_bar = int N dt / int A dt``.  Those are exactly the two
   quantities a pressure--area law is written in.

2. **An exponent-free pressure source.**  :class:`RateDependentCrushing` sets
   the local contact pressure from the *indentation strain rate* only,

       p = p_ref * (epsdot / epsdot_ref) ** n,
       epsdot = V_close / (4 D),                                        (2)

   with ``D`` the instantaneous indentation width and ``V_close`` the closing
   speed obtained from the penetration history.  Equation (2) is the classical
   indentation-rate idealisation (Michel & Tomin 1978, ``epsdot = V/(4D)``;
   Michel 1978, ``p ~ epsdot**n`` in the brittle range) and contains **no**
   ``p(A)`` exponent: the area dependence of the *observed* pressure is then an
   output of the coupled dynamics (the contact widens as the penetration grows,
   which lowers ``epsdot``, and the ensemble mixes thicknesses and approach
   speeds), not an input.  That is what makes the fitted exponent a falsifiable
   number when compared with any externally given ``ex``.

Definitions and calibres (important for any comparison)
------------------------------------------------------
* ``A`` here is the **nominal** contact area of the hull--ice interface patch,
  ``A = w * h`` with ``w`` the penetrating contact width of the wedged bow and
  ``h`` the floe thickness -- the same geometry the W23 chain used.
* The pressure ``p = N / A`` is therefore the **global (area-average) calibre**:
  the model carries one pressure over the whole contact zone and has no
  sub-patch pressure field.  A *local* calibre (pressure on a high-pressure
  zone of area ``A_loc << A``) is **not** available from this model; the two
  calibres are bridged analytically in
  :func:`global_local_bridge` and the incomparable parts are flagged there.
* For a measurement point we use the *peak-force* sample
  ``(p_peak, A_peak)`` (the load condition an area law is calibrated on) and,
  as a secondary, the impulse-weighted pair ``(p_bar, A_bar)``.

Fitting
-------
:func:`power_law_fit` is a log--log ordinary least squares of ``p`` on ``A``
returning ``ex``, the prefactor ``C`` at ``A_0`` and ``R^2``; the intercept is
reported in MPa when ``A_0 = 1 m^2``.  Degenerate samples (``n < 3`` or zero
variance in ``ln A``) return ``fit_ok = False`` instead of a silently wrong
slope.  :func:`bootstrap_power_law` resamples the impact points for an interval
and :func:`quantile_summary` gives the median/5--95 % quantiles used across
seeds.

Honesty notes
-------------
* Equation (2) parameters (``p_ref``, ``epsdot_ref``, ``n``) are *declared*
  engineering inputs, not measurements of ice.  ``n`` is swept; the whole
  pressure level scales exactly linearly in ``p_ref``.
* The contact is still the plastic/rebound idealisation of the W23 chain: the
  floe is a point mass, the ice is level and homogeneous, and the crushed layer
  is not resolved.  No spalling length statistics enter.
* No normative number is stored in this module.  Anything that would need an
  ISO 19906 table entry stays absent by design.

All quantities SI (m, kg, s, N, Pa); pressures are given in MPa where the field
name says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

__all__ = [
    "ContactTrace",
    "RecordingContactLaw",
    "ISOAreaLawContact",
    "RateDependentCrushing",
    "RandomStrengthContact",
    "calibrate_reference_pressure",
    "power_law_fit",
    "bootstrap_power_law",
    "quantile_summary",
    "multifactor_log_fit",
    "decompose_total_exponent",
    "global_local_bridge",
    "IMPACT_COLUMNS",
]

IMPACT_COLUMNS = (
    "model",
    "concentration",
    "seed",
    "floe_index",
    "t_start_s",
    "thickness_m",
    "diameter_m",
    "approach_speed_mps",
    "peak_force_N",
    "area_at_peak_m2",
    "pressure_at_peak_MPa",
    "impulse_Ns",
    "area_time_integral_m3s",
    "mean_pressure_MPa",
    "strain_rate_at_peak_s-1",
    "closing_speed_at_peak_mps",
    "n_samples",
    "closed",
)


# --------------------------------------------------------------------------- #
# per-impact bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class ContactTrace:
    """Everything recorded for one floe impact."""

    floe_index: int
    t_start_s: float
    thickness_m: float
    diameter_m: float
    approach_speed_mps: float
    peak_force_N: float = 0.0
    area_at_peak_m2: float = 0.0
    pressure_at_peak_MPa: float = 0.0
    impulse_Ns: float = 0.0
    area_time_integral_m3s: float = 0.0
    strain_rate_at_peak: float = 0.0
    closing_speed_at_peak: float = 0.0
    last_force_N: float = 0.0
    last_area_m2: float = 0.0
    last_pen_m: float = 0.0
    n_samples: int = 0
    closed: bool = False

    @property
    def mean_pressure_MPa(self) -> float:
        if self.area_time_integral_m3s <= 0.0:
            return float("nan")
        return self.impulse_Ns / self.area_time_integral_m3s / 1.0e6

    def row(self, model: str, concentration: float, seed: int) -> dict:
        return {
            "model": model,
            "concentration": float(concentration),
            "seed": int(seed),
            "floe_index": int(self.floe_index),
            "t_start_s": self.t_start_s,
            "thickness_m": self.thickness_m,
            "diameter_m": self.diameter_m,
            "approach_speed_mps": self.approach_speed_mps,
            "peak_force_N": self.peak_force_N,
            "area_at_peak_m2": self.area_at_peak_m2,
            "pressure_at_peak_MPa": self.pressure_at_peak_MPa,
            "impulse_Ns": self.impulse_Ns,
            "area_time_integral_m3s": self.area_time_integral_m3s,
            "mean_pressure_MPa": self.mean_pressure_MPa,
            "strain_rate_at_peak_s-1": self.strain_rate_at_peak,
            "closing_speed_at_peak_mps": self.closing_speed_at_peak,
            "n_samples": int(self.n_samples),
            "closed": bool(self.closed),
        }


class RecordingContactLaw:
    """Base class: a contact law that records the (pressure, area) of each impact.

    Subclasses implement :meth:`pressure_MPa`; the base class converts the
    pressure into the contact force ``N = p * A``, advances the per-impact
    integrals and keeps the peak-force sample.  The object is injected through
    ``Cummins6DOF.run(..., contact_law=...)`` and must be given the same time
    step as the integration (it differentiates the penetration to obtain the
    closing speed used by rate-dependent laws).
    """

    name = "recording_law"

    def __init__(self, dt_s: float):
        if not (dt_s > 0.0):
            raise ValueError("dt_s must be positive")
        self.dt_s = float(dt_s)
        self._traces: dict = {}
        self._order: list = []
        self.current_event = None

    # -- to be implemented ---------------------------------------------------
    def pressure_MPa(self, area_m2: float, penetration_m: float,
                     thickness_m: float, closing_speed_mps: float) -> float:
        raise NotImplementedError

    # -- recorder -----------------------------------------------------------
    def contact_force(self, area_m2, penetration_m, thickness_m, event) -> float:
        area = float(area_m2)
        pen = float(penetration_m)
        thick = float(thickness_m)
        key = id(event)
        trace = self._traces.get(key)
        if trace is None:
            trace = ContactTrace(
                floe_index=len(self._order),
                t_start_s=float(event.get("t_start", 0.0)),
                thickness_m=thick,
                diameter_m=float(event.get("diameter", float("nan"))),
                approach_speed_mps=float(event.get("v_rel_in", 0.0)),
            )
            self._traces[key] = trace
            self._order.append(key)

        self.current_event = event
        dt = self.dt_s
        closing = (pen - trace.last_pen_m) / dt if trace.n_samples else float(
            max(event.get("v_rel_in", 0.0), 0.0))
        closing = max(0.0, closing)
        pressure = float(self.pressure_MPa(area, pen, thick, closing))
        if not math.isfinite(pressure) or pressure < 0.0:
            raise ValueError(f"{self.name}: non-physical pressure {pressure}")
        force = pressure * 1.0e6 * area

        if trace.n_samples:
            trace.impulse_Ns += 0.5 * (trace.last_force_N + force) * dt
            trace.area_time_integral_m3s += 0.5 * (trace.last_area_m2 + area) * dt
        if force > trace.peak_force_N:
            trace.peak_force_N = force
            trace.area_at_peak_m2 = area
            trace.pressure_at_peak_MPa = pressure
            width = max(area / max(thick, 1e-9), 1.0e-6)
            trace.strain_rate_at_peak = closing / (4.0 * width)
            trace.closing_speed_at_peak = closing
        trace.last_force_N = force
        trace.last_area_m2 = area
        trace.last_pen_m = pen
        trace.n_samples += 1
        trace.closed = bool(event.get("done", False))
        return force

    # -- access -------------------------------------------------------------
    def traces(self) -> list:
        return [self._traces[k] for k in self._order]

    def rows(self, model: str, concentration: float, seed: int,
             min_samples: int = 3) -> list:
        return [t.row(model, concentration, seed) for t in self.traces()
                if t.n_samples >= min_samples and t.area_at_peak_m2 > 0.0
                and t.peak_force_N > 0.0]


class ISOAreaLawContact(RecordingContactLaw):
    """Recorder for the imposed ISO 19906 area law (the W22/W23 pressure source).

    This is the *closure* arm: the pressure is ``p = C_R (A/A_0)**ex`` by
    construction, so a fit to its own samples must return the input exponent.
    Keeping it as an explicit arm makes the non-discriminating power of the
    chain visible instead of implicitly claiming a validation.
    """

    name = "iso_area_law"

    def __init__(self, pressure_law, dt_s: float,
                 helper=None, contact_area_fn=None):
        super().__init__(dt_s)
        self.pressure_law = pressure_law
        self.helper = helper
        self.contact_area_fn = contact_area_fn

    def pressure_MPa(self, area_m2, penetration_m, thickness_m,
                     closing_speed_mps) -> float:
        _ = penetration_m, thickness_m, closing_speed_mps
        return float(self.pressure_law.pressure_MPa(area_m2))


class RateDependentCrushing(RecordingContactLaw):
    """Exponent-free contact pressure: ``p = p_ref (epsdot/epsdot_ref)**n``.

    ``epsdot = V_close / (4 D)`` is the indentation strain rate of Michel &
    Tomin (1978) with ``D`` the instantaneous indentation width (taken as
    ``A / h``, i.e. the width of the nominal contact patch).  No ``p(A)``
    exponent enters; the area dependence of ``p`` is produced by the coupled
    dynamics through ``V_close`` and ``D``.
    """

    name = "rate_crushing"

    def __init__(self, dt_s: float, reference_pressure_MPa: float = 1.0,
                 reference_strain_rate_s: float = 0.01,
                 exponent: float = 0.33,
                 max_pressure_MPa: float | None = None):
        super().__init__(dt_s)
        if reference_pressure_MPa <= 0.0:
            raise ValueError("reference_pressure_MPa must be positive")
        if reference_strain_rate_s <= 0.0:
            raise ValueError("reference_strain_rate_s must be positive")
        self.reference_pressure_MPa = float(reference_pressure_MPa)
        self.reference_strain_rate_s = float(reference_strain_rate_s)
        self.exponent = float(exponent)
        self.max_pressure_MPa = (None if max_pressure_MPa is None
                                 else float(max_pressure_MPa))
        self.name = f"rate_crushing_n{self.exponent:.2f}".replace(".", "")

    def pressure_MPa(self, area_m2, penetration_m, thickness_m,
                     closing_speed_mps) -> float:
        width = max(area_m2 / max(thickness_m, 1.0e-9), 1.0e-6)
        epsdot = max(closing_speed_mps, 0.0) / (4.0 * width)
        ratio = max(epsdot, 1.0e-12) / self.reference_strain_rate_s
        pressure = self.reference_pressure_MPa * ratio ** self.exponent
        if self.max_pressure_MPa is not None:
            pressure = min(pressure, self.max_pressure_MPa)
        return pressure


class RandomStrengthContact(RecordingContactLaw):
    """Negative control: per-floe random strength with **no** area dependence.

    ``p = sigma_i`` with ``sigma_i`` drawn once per floe from a lognormal with a
    declared mean and coefficient of variation.  Because the contact pressure is
    independent of the contact area by construction (``ex = 0`` exactly), any
    slope the estimator returns from an ensemble is spurious -- this arm measures
    that bias and its uncertainty, i.e. it is the control that tells whether an
    emergent exponent from :class:`RateDependentCrushing` is resolvable.
    """

    name = "random_strength"

    def __init__(self, dt_s: float, mean_pressure_MPa: float = 1.9,
                 coefficient_of_variation: float = 0.35, seed: int = 0):
        super().__init__(dt_s)
        if mean_pressure_MPa <= 0.0:
            raise ValueError("mean_pressure_MPa must be positive")
        self.mean_pressure_MPa = float(mean_pressure_MPa)
        self.cv = float(coefficient_of_variation)
        self._rng = np.random.default_rng(seed)
        self._strength: dict = {}

    def pressure_MPa(self, area_m2, penetration_m, thickness_m,
                     closing_speed_mps) -> float:
        _ = area_m2, penetration_m, thickness_m, closing_speed_mps
        key = id(self.current_event)
        value = self._strength.get(key)
        if value is None:
            if self.cv > 0.0:
                sigma = math.sqrt(math.log(1.0 + self.cv ** 2))
                value = float(self._rng.lognormal(
                    math.log(self.mean_pressure_MPa) - 0.5 * sigma ** 2, sigma))
            else:
                value = self.mean_pressure_MPa
            self._strength[key] = value
        return value


def calibrate_reference_pressure(exponent: float, *,
                                 reference_strain_rate_s: float = 1.0,
                                 target_pressure_MPa: float = 1.9) -> float:
    """One-constant calibration of a rate-dependent crushing law.

    Returns the ``p_ref`` that makes ``p(p_ref, epsdot_ref, n)`` equal
    ``target_pressure_MPa`` at ``epsdot = reference_strain_rate_s``.  Only the
    *level* is calibrated; the pressure--area slope remains a prediction of the
    coupled dynamics, which is what keeps the exponent comparison non-circular.
    """
    if exponent < 0.0:
        raise ValueError("exponent must be non-negative")
    if reference_strain_rate_s <= 0.0 or target_pressure_MPa <= 0.0:
        raise ValueError("reference values must be positive")
    return float(target_pressure_MPa / reference_strain_rate_s ** exponent)


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def power_law_fit(area_m2, pressure_MPa, *, reference_area_m2: float = 1.0,
                  min_points: int = 3) -> dict:
    """Log--log OLS fit of ``p = C (A/A_0)**ex``.

    Returns ``exponent``, ``prefactor_MPa`` (= ``C`` in MPa), ``r2`` and the
    sample description.  ``fit_ok`` is False -- with the reason in ``note`` --
    when the sample cannot identify the slope (too few points, non-positive
    area/pressure, or zero variance in ``ln A``); no number is then reported
    for ``ex``/``C``.
    """
    a = np.asarray(area_m2, dtype=float).ravel()
    p = np.asarray(pressure_MPa, dtype=float).ravel()
    if a.size != p.size:
        raise ValueError("area and pressure must have the same length")
    good = np.isfinite(a) & np.isfinite(p) & (a > 0.0) & (p > 0.0) & (a >= 1.0e-9)
    a, p = a[good], p[good]
    n = int(a.size)
    out = {
        "n_points": n,
        "reference_area_m2": float(reference_area_m2),
        "area_min_m2": float(a.min()) if n else None,
        "area_max_m2": float(a.max()) if n else None,
        "pressure_min_MPa": float(p.min()) if n else None,
        "pressure_max_MPa": float(p.max()) if n else None,
        "fit_ok": False,
        "note": "",
    }
    if n < min_points:
        out["note"] = f"only {n} usable impact samples (< {min_points})"
        return out
    x = np.log(a / reference_area_m2)
    y = np.log(p)
    sx = float(np.std(x))
    if sx <= 1.0e-12:
        out["note"] = "zero variance in ln(A): the slope is not identifiable"
        return out
    xm, ym = float(np.mean(x)), float(np.mean(y))
    sxx = float(np.sum((x - xm) ** 2))
    sxy = float(np.sum((x - xm) * (y - ym)))
    slope = sxy / sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - ym) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    out.update({
        "fit_ok": True,
        "exponent": float(slope),
        "prefactor_MPa": float(math.exp(intercept)),
        "r2": float(r2),
        "stderr_exponent": float(math.sqrt(ss_res / (n - 2) / sxx)),
        "stderr_ln_prefactor": float(math.sqrt(ss_res / (n - 2)
                                              * (1.0 / n + xm ** 2 / sxx))),
        "residual_rms_ln": float(math.sqrt(ss_res / n)),
    })
    return out


def bootstrap_power_law(area_m2, pressure_MPa, *, n_boot: int = 400,
                        seed: int = 20261112, reference_area_m2: float = 1.0,
                        min_points: int = 3) -> dict:
    """Percentile bootstrap of the fitted exponent / prefactor."""
    a = np.asarray(area_m2, dtype=float).ravel()
    p = np.asarray(pressure_MPa, dtype=float).ravel()
    n = int(min(a.size, p.size))
    if n < min_points:
        return {"n_boot": 0, "note": "not enough points for a bootstrap"}
    rng = np.random.default_rng(seed)
    ex = []
    pref = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        fit = power_law_fit(a[idx], p[idx], reference_area_m2=reference_area_m2)
        if fit["fit_ok"]:
            ex.append(fit["exponent"])
            pref.append(fit["prefactor_MPa"])
    if len(ex) < max(10, int(0.2 * n_boot)):
        return {"n_boot": len(ex), "note": "too many degenerate resamples"}
    return {
        "n_boot": len(ex),
        "exponent_p05": float(np.percentile(ex, 5.0)),
        "exponent_median": float(np.percentile(ex, 50.0)),
        "exponent_p95": float(np.percentile(ex, 95.0)),
        "prefactor_p05_MPa": float(np.percentile(pref, 5.0)),
        "prefactor_median_MPa": float(np.percentile(pref, 50.0)),
        "prefactor_p95_MPa": float(np.percentile(pref, 95.0)),
    }


def quantile_summary(values, *, qs=(5.0, 50.0, 95.0)) -> dict:
    """Median and 5--95 % quantiles of a sample (linear interpolation)."""
    v = np.asarray([x for x in np.asarray(values, dtype=float).ravel()
                    if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0}
    out = {"n": int(v.size), "min": float(v.min()), "max": float(v.max())}
    for q in qs:
        key = {5.0: "p05", 50.0: "median", 95.0: "p95"}.get(
            float(q), f"p{float(q):g}")
        out[key] = float(np.percentile(v, q))
    out["mean"] = float(v.mean())
    out["std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return out


def multifactor_log_fit(area_m2, thickness_m, closing_speed_mps,
                        pressure_MPa) -> dict:
    """Decompose ``ln p`` on ``ln A``, ``ln h`` and ``ln V_close``.

    The measured pressure--area exponent of a mechanism-based contact law is a
    *confounded* number: the ensemble varies the floe thickness and the closing
    speed together with the contact area (``A = w h``).  This regression makes
    the confounders explicit, so an exponent can be attributed instead of merely
    reported.
    """
    a = np.asarray(area_m2, dtype=float)
    h = np.asarray(thickness_m, dtype=float)
    v = np.asarray(closing_speed_mps, dtype=float)
    p = np.asarray(pressure_MPa, dtype=float)
    good = (np.isfinite(a) & np.isfinite(h) & np.isfinite(p) & np.isfinite(v)
            & (a > 0) & (h > 0) & (p > 0) & (v > 0))
    a, h, v, p = a[good], h[good], v[good], p[good]
    n = int(a.size)
    if n < 6:
        return {"n_points": n, "fit_ok": False, "note": "too few points"}
    X = np.column_stack([np.ones(n), np.log(a), np.log(h), np.log(v)])
    y = np.log(p)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n_points": n,
        "fit_ok": True,
        "intercept": float(coef[0]),
        "dlnp_dlnA": float(coef[1]),
        "dlnp_dlnh": float(coef[2]),
        "dlnp_dlnV": float(coef[3]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "note": ("partial exponents: dlnp_dlnA is the area coefficient at fixed "
                 "thickness and closing speed"),
    }


def decompose_total_exponent(area_m2, thickness_m, closing_speed_mps,
                            pressure_MPa) -> dict:
    """Split the *apparent* (raw) ``dln p / dln A`` into its two sources.

    For any pressure law that is a deterministic function of
    ``(A, h, V_close)`` the raw log--log slope of ``p`` on ``A`` is exactly

        dlnp/dlnA |_total = a_A + a_h * beta_h + a_V * beta_V,            (3)

    where ``a`` are the partial exponents of :func:`multifactor_log_fit` and
    ``beta_h = dln h/dln A``, ``beta_V = dln V_close/dln A`` are the auxiliary
    regressions of the ensemble.  Equation (3) is the bridge between a
    *mechanism-level* exponent (a material/contact law) and the *design-calibre*
    exponent an observer measures over a population: the second and third terms
    are the population's covariance structure, not the contact law.

    Returns the partial exponents, the auxiliary betas, the identity check and
    the residual ``total - (a_A + a_h beta_h + a_V beta_V)`` (which is zero up to
    least-squares algebra when the law is deterministic in the three
    variables).
    """
    a = np.asarray(area_m2, dtype=float)
    h = np.asarray(thickness_m, dtype=float)
    v = np.asarray(closing_speed_mps, dtype=float)
    p = np.asarray(pressure_MPa, dtype=float)
    good = (np.isfinite(a) & np.isfinite(h) & np.isfinite(p) & np.isfinite(v)
            & (a > 0) & (h > 0) & (p > 0) & (v > 0))
    a, h, v, p = a[good], h[good], v[good], p[good]
    n = int(a.size)
    if n < 6:
        return {"n_points": n, "identity_ok": False, "note": "too few points"}
    partial = multifactor_log_fit(a, h, v, p)
    if not partial.get("fit_ok"):
        return {"n_points": n, "identity_ok": False, "note": "degenerate sample"}
    la = np.log(a)
    beta_h = float(np.polyfit(la, np.log(h), 1)[0])
    beta_v = float(np.polyfit(la, np.log(v), 1)[0])
    raw = float(np.polyfit(la, np.log(p), 1)[0])
    predicted = (partial["dlnp_dlnA"] + partial["dlnp_dlnh"] * beta_h
                 + partial["dlnp_dlnV"] * beta_v)
    return {
        "n_points": n,
        "partial_exponents": {"A": partial["dlnp_dlnA"],
                              "h": partial["dlnp_dlnh"],
                              "V_close": partial["dlnp_dlnV"]},
        "partial_r2": partial["r2"],
        "beta_h_vs_A": beta_h,
        "beta_V_vs_A": beta_v,
        "raw_total_exponent": raw,
        "predicted_total_exponent": predicted,
        "residual": raw - predicted,
        "identity_ok": True,
        "note": ("total = a_A + a_h beta_h + a_V beta_V; the last two terms are "
                 "the population covariance, not the contact law"),
    }


# --------------------------------------------------------------------------- #
# calibre bridge
# --------------------------------------------------------------------------- #
def global_local_bridge(prefactor_MPa: float, exponent: float, *,
                        reference_area_m2: float = 1.0,
                        local_area_m2: float = 0.1,
                        nominal_area_m2: float = 1.0) -> dict:
    """Bridge the global (nominal-area) and local (high-pressure-zone) calibres.

    The area law is a statement about *one* area definition.  With the same
    coefficients, the pressure on a high-pressure zone of area ``A_loc`` and the
    area-average pressure over a nominal area ``A_nom`` that contains
    ``N_hp`` such zones are

        p_loc = C (A_loc / A_0)**ex,
        p_global = p_loc * N_hp * A_loc / A_nom = kappa * p_loc,

    so the two calibres differ by the *area participation factor*
    ``kappa = N_hp A_loc / A_nom <= 1``.  A fitted exponent measured on one
    calibre cannot be compared with an exponent defined on the other without
    ``kappa``: if ``kappa`` is itself area dependent, it absorbs into the fitted
    slope.  This function returns the numbers for a declared ``kappa``; the
    model used in this study carries ``kappa = 1`` by construction (one uniform
    pressure over the whole contact patch), which is exactly why its
    local-calibre comparison is WITHHELD.
    """
    kappa = float(local_area_m2) / float(nominal_area_m2)
    p_loc = prefactor_MPa * (local_area_m2 / reference_area_m2) ** exponent
    return {
        "prefactor_MPa": float(prefactor_MPa),
        "exponent": float(exponent),
        "local_area_m2": float(local_area_m2),
        "nominal_area_m2": float(nominal_area_m2),
        "participation_factor_kappa": kappa,
        "pressure_local_calibre_MPa": float(p_loc),
        "pressure_global_calibre_MPa": float(kappa * p_loc),
        "note": ("kappa = 1 in the simulated model (single uniform pressure over "
                 "the contact patch), so a local-calibre (high-pressure-zone) "
                 "comparison is WITHHELD"),
    }