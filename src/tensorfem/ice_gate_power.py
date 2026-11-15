"""Information-content gate for the ice-chain heave added mass (GAP-ICE-GATE-POWER).

Background
----------
The ice-coupled 6-DOF chain :mod:`benchmarks.ice_coupled_6dof.cummins_ice_6dof`
declares a *synthetic* radiation model whose infinite-frequency heave added mass
is ``A_inf = 1.00 * m`` (``m = rho C_b L B T`` = 8.610e6 kg), i.e.

    A_inf / L = 86.1 t/m     (declared, per unit length)

Report #19234 (mission 21257) compared that with an independent 2-D potential-flow
solution, ``A33 = 228.6 t/m`` (two routes, mutual disagreement 3.8%), and found a
165.5% deviation.  Two follow-up rounds (m22156, m22213) failed to turn that
finding into an *acceptance criterion*: the reference they built was 3-D where the
judged quantity is 2-D, used a different geometry (T = 8 m instead of 6 m), and
their "rejected negative control" was an arithmetic identity
(``declared = reference * 2.65``), so the criterion carried no information.

What this module adds
---------------------
1. **The judged quantity, computed with the chain's own geometry.**  The declared
   value is read from the chain module itself (no transcription), divided by ``L``
   so that it is a 2-D coefficient ``[kg/m]`` like the reference.
2. **An in-repository, independent reference route** built on
   :mod:`tensorfem.added_mass_bem2d` (constant-strength single-layer BEM), used
   through the free-surface mirror trick:

   * ``free_surface="omega0"``   (free-surface condition ``dphi/dz = 0``, the
     zero-frequency heave limit): even mirror  -> ``A = m_full / 2``;
   * ``free_surface="infinity"`` (``phi = 0``, the infinite-frequency limit that
     the chain's ``A_inf`` denotes): odd mirror -> ``A = force on the real half``.

   Both mirror routes are validated inside the module against the exact
   deeply-submerged-cylinder value ``rho pi a**2`` (`validate_reference_routes`),
   and the panel-count convergence is measured, not assumed
   (`convergence_study`).
3. **A decision rule with stated information content** (the "POWER" of the gap
   name).  Adopted from the metrology consensus on conformity assessment --
   ILAC-G8:09/2019 *Guidelines on decision rules and statements of conformity*
   and JCGM 106:2012 *The role of measurement uncertainty in conformity
   assessment* -- instead of the naive ``|dev| <= tol`` comparison:

   ====================  ===========================================
   acceptance limit      ``tol - k * u_rel``   -> accept
   rejection limit       ``tol + k * u_rel``   -> reject
   between the two       *indeterminate* (report, do not pass)
   ====================  ===========================================

   where ``u_rel`` is the measured relative residual of the reference route and
   ``k`` the coverage factor.  The module then computes the **detection rate**
   (statistical power) of the criterion for any true deviation, its **false
   acceptance** complement, and the **smallest detectable deviation** at a
   requested power.
4. **Negative controls that are actually rejected.**  Every control value below
   is a *real* number produced somewhere else (chain channel, an independent
   route, a rejected earlier-round reference, zero), not ``reference * const``,
   so the pass/reject outcomes are not decided by construction.  A **meta-gate**
   additionally rejects *degenerate gates*: a tolerance so wide that the audited
   failure would be accepted, and a tolerance so tight that a conforming value
   could not be accepted.

All quantities SI (kg, m, s); added mass per unit length is in kg/m, and the
reporting layer converts to t/m (1 t/m = 1000 kg/m).
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np

from .added_mass_bem2d import influence_matrices, rectangle_panels

__all__ = [
    "GAP_SLUG",
    "DEFAULT_TOL",
    "DEFAULT_U_REL",
    "DEFAULT_K",
    "CHAIN_MODULE_RELPATH",
    "REFERENCE_CHAIN_GEOMETRY",
    "REFERENCE_19234",
    "chain_model",
    "chain_particulars",
    "declared_added_mass_per_length",
    "section_added_mass_2d",
    "deep_cylinder_check",
    "validate_reference_routes",
    "convergence_study",
    "richardson_extrapolate",
    "acceptance_limit",
    "rejection_limit",
    "decide",
    "decide_naive",
    "detection_power",
    "detection_power_analytic",
    "false_accept_probability",
    "critical_deviation",
    "meta_gate",
    "gate_verdict",
]

# --------------------------------------------------------------------------- #
# constants and provenance
# --------------------------------------------------------------------------- #
GAP_SLUG = "GAP-ICE-GATE-POWER"

#: Tolerance of the acceptance criterion (relative deviation of the declared
#: heave added mass from the independent reference).  15% is the level at which
#: the ecosystem treats a 2-D strip/potential-value comparison as "the same
#: physics"; it is a *declared* requirement, not a fitted number.
DEFAULT_TOL = 0.15

#: Relative residual of the reference route, ``u_rel``.  Taken from report
#: #19234 section Q5: the independent BEM and finite-difference potential routes
#: disagree by 3.8% (=> per-estimate 3.8/sqrt(2) = 2.7%), and the
#: deeply-submerged-cylinder self-verification residual of the same report spans
#: -0.32% .. +2.90%.  The larger, 2.90%, is used (conservative).  It also covers
#: the free-surface-condition band computed here (225.3 t/m .. 234.4 t/m, i.e.
#: +/-1.98% about the midpoint).
DEFAULT_U_REL = 0.029

#: Coverage factor of the guard band (``k = 2`` <-> ~95% one-sided coverage).
DEFAULT_K = 2.0

CHAIN_MODULE_RELPATH = pathlib.Path("benchmarks/ice_coupled_6dof/cummins_ice_6dof.py")

#: Geometry of the chain that the reference must use.  Read back from the chain
#: module by :func:`chain_particulars`; repeated here only as documentation.
REFERENCE_CHAIN_GEOMETRY = {
    "length_m": 100.0,
    "breadth_m": 20.0,
    "draught_m": 6.0,
    "block_coefficient": 0.70,
    "water_density": 1025.0,
}

#: Independent 2-D heave added mass of report #19234 (mission 21257), section Q5.
REFERENCE_19234 = {
    "value_kg_per_m": 228.6e3,
    "source": "tensor_reports#19234 (mission 21257) section Q5: BEM mirror Green "
              "single-layer + finite-difference five-point Laplace, mutual "
              "disagreement 3.8%, deep-submerged-cylinder self-check -0.32%..+2.90%",
    "geometry": "chain geometry L=100 m, B=20 m, T=6 m, C_b=0.70",
}

#: Reference of the reopened round m22156 (rejected by audit m22176, finding F2):
#: a strip-theory estimate of the wrong quantity, kept as a negative control.
REJECTED_M22156_REFERENCE = {
    "value_kg_per_m": 440.10e3,
    "source": "m22156 tests/test_hydro_ice_anchor.py heave_added_mass_rectangular_barge "
              "(strip_theory_newman_1977 label); audit m22176 finding F2: +92.5% "
              "against the verified BEM anchor",
}


# --------------------------------------------------------------------------- #
# the judged quantity: what the chain declares
# --------------------------------------------------------------------------- #
def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def chain_model(**overrides):
    """Import the ice-chain module by path and return ``(ShipModel(), module)``."""
    path = _repo_root() / CHAIN_MODULE_RELPATH
    spec = importlib.util.spec_from_file_location("_ice_chain_cummins", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load chain module at {path}")
    module = importlib.util.module_from_spec(spec)
    # ``dataclasses`` resolves field annotations through ``sys.modules``, so the
    # chain module has to be registered before it is executed.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ShipModel(**overrides), module


def chain_particulars(**overrides) -> dict:
    """Particulars of the chain ship (read from the chain, not transcribed)."""
    model, _ = chain_model(**overrides)
    return {
        "length_m": model.length_m,
        "breadth_m": model.breadth_m,
        "draught_m": model.draught_m,
        "block_coefficient": model.block_coefficient,
        "water_density": model.water_density,
        "mass_kg": model.mass_kg,
    }


def declared_added_mass_per_length(dof: str = "heave", **overrides) -> dict:
    """Declared infinite-frequency added mass of ``dof``, per unit length [kg/m].

    Straight read of ``ShipModel.radiation[dof].added_mass_inf / L``: no number in
    this function is transcribed, so the criterion cannot drift from the chain
    without the test suite failing.
    """
    model, _ = chain_model(**overrides)
    if dof not in model.radiation:
        raise KeyError(f"unknown dof {dof!r}; chain has {sorted(model.radiation)}")
    total = float(model.radiation[dof].added_mass_inf)
    return {
        "dof": dof,
        "added_mass_total_kg": total,
        "length_m": model.length_m,
        "value_kg_per_m": total / model.length_m,
        "value_t_per_m": total / model.length_m / 1.0e3,
        "declared_mass_ratio": total / model.mass_kg,
    }


# --------------------------------------------------------------------------- #
# the reference: 2-D potential-flow added mass of the section
# --------------------------------------------------------------------------- #
def section_added_mass_2d(breadth: float, draught: float, rho: float = 1025.0,
                          per_edge: int = 160, free_surface: str = "omega0") -> float:
    """2-D heave added mass per unit length [kg/m] of a surface-piercing rectangle.

    The body occupies ``|x| <= breadth/2``, ``-draught <= z <= 0``, the free
    surface is ``z = 0``.  The mirror trick folds the half-plane problem into the
    unbounded problem for the doubled rectangle ``2c x 2h`` with ``c = B/2`` and
    ``h = T``:

    * ``free_surface="omega0"``: the zero-frequency free-surface condition
      ``dphi/dz = 0`` is an even extension, so the doubled body translates as a
      whole and the half-domain energy is half the full energy,
      ``A = m_full / 2``;
    * ``free_surface="infinity"``: ``phi = 0`` on the free surface is an odd
      extension, so the doubled body moves antisymmetrically and the added mass
      of the real half is the single-layer force integral taken on the lower half
      only, ``A = -rho * sum_lower(phi * n_y * l)``.

    The *magnitude* of that odd-extension integral is returned: its sign follows
    the outward-normal bookkeeping at the waterline (positive for the deep
    cylinder, negative for the surface-piercing rectangle) and the physical added
    mass is ``|A|``.  The magnitude is what the deep-cylinder validation of
    :func:`validate_reference_routes` pins to ``rho pi a**2``.

    Both routes reproduce the exact deeply-submerged-cylinder value
    ``rho * pi * a**2`` (see :func:`validate_reference_routes`).
    """
    c = 0.5 * float(breadth)
    h = float(draught)
    if c <= 0.0 or h <= 0.0:
        raise ValueError("breadth and draught must be positive")
    panels = rectangle_panels(c, h, per_edge=per_edge)
    A, D = influence_matrices(panels)
    nrm = panels["normals"]
    ln = panels["lengths"]
    mid = panels["midpoints"]

    if free_surface == "omega0":
        g = nrm[:, 1].copy()
        phi = A @ np.linalg.solve(D, g)
        m_full = -rho * float(np.sum(phi * nrm[:, 1] * ln))
        return m_full / 2.0
    if free_surface == "infinity":
        g = np.where(mid[:, 1] > 0.0, -1.0, 1.0) * nrm[:, 1]
        phi = A @ np.linalg.solve(D, g)
        lower = mid[:, 1] < 0.0
        return abs(-rho * float(np.sum((phi * nrm[:, 1] * ln)[lower])))
    raise ValueError(f"unknown free_surface condition {free_surface!r}")


def _two_circles(a: float, depth: float, n: int):
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    lower = np.stack((a * np.cos(th), -depth + a * np.sin(th)), axis=1)
    upper = np.stack((a * np.cos(th), depth + a * np.sin(th)), axis=1)
    return lower, upper


def deep_cylinder_check(a: float = 1.0, depth: float = 8.0, rho: float = 1.0,
                        n: int = 720) -> dict:
    """Both mirror routes on a deeply submerged cylinder vs the exact ``rho pi a**2``."""
    from .added_mass_bem2d import _curve_panels

    lower, upper = _two_circles(a, depth, n)
    xy = np.concatenate([lower, upper])
    panels = _curve_panels(xy[:, 0], xy[:, 1])
    A, D = influence_matrices(panels)
    nrm, ln = panels["normals"], panels["lengths"]
    n_low = len(lower)
    exact = rho * math.pi * a ** 2

    phi_even = A @ np.linalg.solve(D, nrm[:, 1])
    m_even_full = -rho * float(np.sum(phi_even * nrm[:, 1] * ln))

    g = nrm[:, 1].copy()
    g[n_low:] *= -1.0
    phi_odd = A @ np.linalg.solve(D, g)
    m_odd_low = -rho * float(np.sum((phi_odd * nrm[:, 1] * ln)[:n_low]))

    return {
        "a": a, "depth_over_a": depth / a, "exact_kg_per_m": exact,
        "even_mirror": m_even_full / 2.0,
        "odd_mirror": m_odd_low,
        "err_even": m_even_full / 2.0 / exact - 1.0,
        "err_odd": m_odd_low / exact - 1.0,
    }


def validate_reference_routes(depths=(4.0, 8.0), n: int = 720) -> list:
    """Self-verification of the two mirror routes (deep cylinder -> ``rho pi a**2``)."""
    return [deep_cylinder_check(depth=d, n=n) for d in depths]


def richardson_extrapolate(values) -> dict:
    """Observed order and Richardson estimate from three successive refinements."""
    v = [float(x) for x in values]
    if len(v) < 3:
        raise ValueError("need at least three refinement levels")
    e1, e2 = v[-2] - v[-3], v[-1] - v[-2]
    if e2 == 0.0:
        return {"observed_order": float("inf"), "richardson": v[-1],
                "last_increment": e2}
    ratio = e1 / e2
    if ratio <= 0.0:
        return {"observed_order": float("nan"), "richardson": float("nan"),
                "last_increment": e2}
    p = math.log(ratio) / math.log(2.0)
    return {"observed_order": p, "richardson": v[-1] + e2 / (2.0 ** p - 1.0),
            "last_increment": e2}


def convergence_study(breadth: float = 20.0, draught: float = 6.0,
                      rho: float = 1025.0, per_edges=(40, 80, 160, 320),
                      free_surface: str = "omega0") -> dict:
    """Panel-count convergence of :func:`section_added_mass_2d` (observed order)."""
    values = [section_added_mass_2d(breadth, draught, rho=rho, per_edge=pe,
                                    free_surface=free_surface)
              for pe in per_edges]
    rich = richardson_extrapolate(values[-3:])
    return {
        "free_surface": free_surface,
        "per_edges": list(per_edges),
        "values_kg_per_m": values,
        "orders": [richardson_extrapolate(values[i - 2:i + 1])["observed_order"]
                   for i in range(2, len(values))],
        "observed_order": rich["observed_order"],
        "richardson_kg_per_m": rich["richardson"],
    }


# --------------------------------------------------------------------------- #
# decision rule and its information content
# --------------------------------------------------------------------------- #
def acceptance_limit(tol: float = DEFAULT_TOL, u_rel: float = DEFAULT_U_REL,
                     k: float = DEFAULT_K) -> float:
    """Largest deviation accepted with confidence (guard band = ``k * u_rel``)."""
    return tol - k * u_rel


def rejection_limit(tol: float = DEFAULT_TOL, u_rel: float = DEFAULT_U_REL,
                    k: float = DEFAULT_K) -> float:
    """Smallest deviation rejected with confidence."""
    return tol + k * u_rel


def decide(deviation: float, tol: float = DEFAULT_TOL,
           u_rel: float = DEFAULT_U_REL, k: float = DEFAULT_K) -> str:
    """Guard-banded conformity decision for a signed relative deviation."""
    acc, rej = acceptance_limit(tol, u_rel, k), rejection_limit(tol, u_rel, k)
    a = abs(float(deviation))
    if a <= acc:
        return "accept"
    if a >= rej:
        return "reject"
    return "indeterminate"


def decide_naive(deviation: float, tol: float = DEFAULT_TOL) -> str:
    """The naive rule the earlier rounds used (no guard band), for comparison."""
    return "accept" if abs(float(deviation)) <= tol else "reject"


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def detection_power_analytic(delta: float, tol: float = DEFAULT_TOL,
                             u_rel: float = DEFAULT_U_REL,
                             k: float = DEFAULT_K) -> float:
    """``P(reject | true deviation = delta)`` in the normal-approximation model.

    The reference estimate carries a multiplicative residual ``eps ~ N(0, u_rel)``,
    so the observed deviation is ``delta - eps`` and the guard-banded rule rejects
    when ``|delta - eps| >= tol + k*u_rel``.
    """
    rej = rejection_limit(tol, u_rel, k)
    return _normal_cdf((delta - rej) / u_rel) + _normal_cdf((-delta - rej) / u_rel)


def detection_power(delta: float, tol: float = DEFAULT_TOL,
                    u_rel: float = DEFAULT_U_REL, k: float = DEFAULT_K,
                    n_mc: int = 200_000, seed: int = 20261111) -> float:
    """Monte-Carlo detection rate of the criterion at a true deviation ``delta``."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, u_rel, size=int(n_mc))
    observed = float(delta) - eps
    rej = rejection_limit(tol, u_rel, k)
    return float(np.mean(np.abs(observed) >= rej))


def false_accept_probability(delta: float, tol: float = DEFAULT_TOL,
                             u_rel: float = DEFAULT_U_REL,
                             k: float = DEFAULT_K) -> float:
    """``P(accept | true deviation = delta)``: one minus the detection rate."""
    acc = acceptance_limit(tol, u_rel, k)
    if acc <= 0.0:          # guard band consumes the acceptance region
        return 0.0
    return _normal_cdf((acc - delta) / u_rel) + _normal_cdf((-acc - delta) / u_rel)


def critical_deviation(tol: float = DEFAULT_TOL, u_rel: float = DEFAULT_U_REL,
                       k: float = DEFAULT_K, target: float = 0.95) -> float:
    """Smallest true deviation detected with probability ``target`` (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if detection_power_analytic(mid, tol, u_rel, k) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def meta_gate(tol: float, declared_deviation: float,
              u_rel: float = DEFAULT_U_REL, k: float = DEFAULT_K) -> dict:
    """Reject degenerate gates: a criterion must be able to fail *and* to pass.

    A gate is *informative* only if

    * it rejects the audited deviation with at least 95% power (``power >= 0.95``), and
    * it accepts a conforming value (``delta = 0``) with at most 5% false rejection.
    """
    power_at_audited = detection_power_analytic(abs(declared_deviation), tol, u_rel, k)
    false_reject_at_zero = 1.0 - false_accept_probability(0.0, tol, u_rel, k)
    informative = (power_at_audited >= 0.95) and (false_reject_at_zero <= 0.05)
    return {
        "tol": tol,
        "power_at_audited_deviation": power_at_audited,
        "false_reject_at_zero": false_reject_at_zero,
        "informative": informative,
        "reason": ("has power and can also pass" if informative else
                   ("cannot reject the audited failure" if power_at_audited < 0.95
                    else "cannot accept a conforming value")),
    }


# --------------------------------------------------------------------------- #
# the full gate
# --------------------------------------------------------------------------- #
def gate_verdict(tol: float = DEFAULT_TOL, u_rel: float = DEFAULT_U_REL,
                 k: float = DEFAULT_K, per_edge: int = 160,
                 dofs=("heave", "surge"), geometry_overrides=None,
                 n_mc: int = 200_000) -> dict:
    """Evaluate the GAP-ICE-GATE-POWER criterion on the ice chain.

    Returns a certificate dict with the four elements the ecosystem's
    information-content gate requires:

    (a) ``baseline_source``  -- where the reference number comes from;
    (b) ``power``            -- the threshold expressed as a detection rate
                                (and the observed convergence order of the route);
    (c) ``controls``         -- real values that must be accepted / rejected, each
                                with the decision actually returned;
    (d) ``meta_gate``        -- proof that the criterion is not degenerate.
    """
    geometry_overrides = dict(geometry_overrides or {})
    particulars = chain_particulars(**geometry_overrides)
    declared = {d: declared_added_mass_per_length(d, **geometry_overrides) for d in dofs}

    breadth, draught = particulars["breadth_m"], particulars["draught_m"]
    rho = particulars["water_density"]

    # in-repository reference routes (both free-surface conditions)
    route_omega0 = section_added_mass_2d(breadth, draught, rho=rho,
                                         per_edge=per_edge, free_surface="omega0")
    route_inf = section_added_mass_2d(breadth, draught, rho=rho,
                                      per_edge=per_edge, free_surface="infinity")
    conv_omega0 = convergence_study(breadth, draught, rho=rho,
                                    per_edges=(max(10, per_edge // 4), max(20, per_edge // 2),
                                               per_edge, 2 * per_edge),
                                    free_surface="omega0")
    conv_inf = convergence_study(breadth, draught, rho=rho,
                                 per_edges=(max(10, per_edge // 4), max(20, per_edge // 2),
                                            per_edge, 2 * per_edge),
                                 free_surface="infinity")

    # The primary reference is *computed*: the mean of the two free-surface
    # variants of the in-repository route.  Report #19234's 228.6 t/m then enters
    # as an independent cross-check control (P1), not as a transcribed constant
    # that the verdict would depend on.
    reference = 0.5 * (route_omega0 + route_inf)
    band = {"min_kg_per_m": min(route_omega0, route_inf),
            "max_kg_per_m": max(route_omega0, route_inf)}
    band_half_width = 0.5 * (band["max_kg_per_m"] - band["min_kg_per_m"]) / reference
    anchor_deviation = (REFERENCE_19234["value_kg_per_m"] - reference) / reference
    # total reference uncertainty: conservative maximum of the documented
    # contributions (route residual, free-surface-condition band, cross-check
    # deviation of the independent anchor)
    u_used = max(u_rel, band_half_width, abs(anchor_deviation))

    deviations = {d: (v["value_kg_per_m"] - reference) / reference
                  for d, v in declared.items()}
    decisions = {d: decide(dv, tol, u_used, k) for d, dv in deviations.items()}
    decisions_naive = {d: decide_naive(dv, tol) for d, dv in deviations.items()}

    acc, rej = acceptance_limit(tol, u_used, k), rejection_limit(tol, u_used, k)
    heave_dev = abs(deviations["heave"])
    power = {
        "u_used": u_used,
        "at_audited_deviation": detection_power_analytic(heave_dev, tol, u_used, k),
        "mc_at_audited_deviation": detection_power(heave_dev, tol, u_used, k, n_mc=n_mc),
        "at_tolerance": detection_power_analytic(tol, tol, u_used, k),
        "at_rejection_limit": detection_power_analytic(rej, tol, u_used, k),
        "orthogonal_mc_check": {
            "delta": 0.30,
            "analytic": detection_power_analytic(0.30, tol, u_used, k),
            "monte_carlo": detection_power(0.30, tol, u_used, k, n_mc=n_mc, seed=7),
        },
        "critical_deviation_95": critical_deviation(tol, u_used, k),
        "false_accept_at_tolerance": false_accept_probability(tol, tol, u_used, k),
        # the naive rule (guard band k = 0) flips exactly at the tolerance, i.e.
        # it is a coin flip on the very deviation the tolerance is meant to catch
        "naive_power_at_tolerance": detection_power_analytic(tol, tol, u_used, 0.0),
        "naive_false_accept_at_tolerance": false_accept_probability(tol, tol, u_used, 0.0),
        "naive_false_reject_at_zero": (1.0 - false_accept_probability(0.0, tol, u_used, 0.0)),
    }

    # ---- controls: real numbers from elsewhere in the ecosystem --------------
    controls = []

    def add(name, value_kg_per_m, expected, provenance):
        dev = (value_kg_per_m - reference) / reference
        got = decide(dev, tol, u_used, k)
        controls.append({
            "name": name, "value_kg_per_m": value_kg_per_m,
            "value_t_per_m": value_kg_per_m / 1.0e3,
            "deviation": dev, "expected": expected, "got": got,
            "passed": got == expected, "provenance": provenance,
        })

    add("P1_independent_anchor_19234", REFERENCE_19234["value_kg_per_m"], "accept",
        REFERENCE_19234["source"])
    add("P2_repo_route_odd_mirror", route_inf, "accept",
        "tensorfem.ice_gate_power.section_added_mass_2d(free_surface='infinity')")
    add("P3_repo_route_even_mirror", route_omega0, "accept",
        "tensorfem.ice_gate_power.section_added_mass_2d(free_surface='omega0')")
    if "heave" in declared:
        add("N1_chain_heave_declared", declared["heave"]["value_kg_per_m"], "reject",
            "benchmarks/ice_coupled_6dof/cummins_ice_6dof.py ShipModel.radiation['heave']")
    if "surge" in declared:
        add("N2_chain_surge_channel", declared["surge"]["value_kg_per_m"], "reject",
            "benchmarks/ice_coupled_6dof/cummins_ice_6dof.py ShipModel.radiation['surge']")
        add("N3_sign_flipped_heave", -declared["heave"]["value_kg_per_m"], "reject",
            "sign-flipped chain declaration (a physical impossibility probe)")
    add("N4_lock_dead_zero", 0.0, "reject", "zero added mass (lock-dead probe)")
    add("N5_rejected_m22156_reference", REJECTED_M22156_REFERENCE["value_kg_per_m"],
        "reject", REJECTED_M22156_REFERENCE["source"])

    mg = meta_gate(tol, heave_dev, u_used, k)
    degenerate = [meta_gate(t_try, heave_dev, u_used, k) for t_try in (2.0, 0.0)]

    passed = (
        all(c["passed"] for c in controls)
        and decisions["heave"] == "reject"
        and mg["informative"]
        and not any(d["informative"] for d in degenerate)
        and conv_omega0["observed_order"] > 0.0
        and conv_inf["observed_order"] > 0.0
    )

    return {
        "gap_slug": GAP_SLUG,
        "judged_quantity": "2-D infinite/low-frequency heave added mass per unit "
                           "length of the chain mid-section [kg/m]",
        "geometry": particulars,
        "declared": declared,
        "deviations": deviations,
        "decision": decisions,
        "decision_naive_rule": decisions_naive,
        "rule": {"tol": tol, "u_rel": u_rel, "u_used": u_used, "k": k,
                 "acceptance_limit": acc, "rejection_limit": rej,
                 "u_contributions": {"route_residual_19234": u_rel,
                                     "free_surface_band_half_width": band_half_width,
                                     "independent_anchor_deviation": abs(anchor_deviation)}},
        "reference": {
            "primary_kg_per_m": reference,
            "primary_t_per_m": reference / 1.0e3,
            "primary_source": "mean of the two free-surface variants of "
                              "section_added_mass_2d (computed in this repository)",
            "independent_anchor_deviation": anchor_deviation,
            "repo_route_omega0_kg_per_m": route_omega0,
            "repo_route_infinity_kg_per_m": route_inf,
            "band": band,
            "band_half_width_rel": band_half_width,
            "provenance": REFERENCE_19234,
        },
        "baseline_source": {
            "primary": REFERENCE_19234["source"],
            "self_verification": validate_reference_routes(),
            "convergence": {"omega0": conv_omega0, "infinity": conv_inf},
        },
        "power": power,
        "controls": controls,
        "meta_gate": mg,
        "degenerate_gates_rejected": degenerate,
        "pass": bool(passed),
        "verdict": ("REJECT-DECLARATION (chain heave added mass is outside the "
                    "acceptance band with power ~1)" if decisions.get("heave") == "reject"
                    else "declaration accepted"),
    }