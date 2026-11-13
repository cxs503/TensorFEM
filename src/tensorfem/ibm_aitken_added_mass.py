"""Added-mass cross-validation of the implicit IBM + Aitken coupling.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (reference side:
joconnor22/LIFE -- surface-marker immersed boundary with implicit
sub-iterations under-relaxed by Aitken's Delta^2 rule).

Why this module exists
----------------------
:mod:`tensorfem.ibm_aitken_implicit` validates the IBM operator chain (kernel
partition of unity, epsilon conservation, force transmission) and the
*algorithmic* Aitken recursion on a scalar probe.  What it could not validate
is the **coupled** answer: a beam whose markers sit on the structural
centreline has no analytic reference, because its numerical thickness is set by
the delta support rather than by the structural section (see
``docs/xval_implicit_ibm_aitken_v2.md`` section 6, open item "submerged
frequency").  This module supplies a configuration that *does* have an exact
reference:

    a **thin rigid plate** of half-width ``c`` (marker segment along ``x``),
    restrained by a linear spring of stiffness ``k`` and constrained to
    translate transversely, immersed in a quiescent periodic D2Q9 box and
    coupled by implicit direct forcing with Aitken sub-iterations.

For a flat plate in 2-D potential flow the added mass per unit length for
broadside translation is exactly

    m_a = rho_f * pi * c^2          (Lamb 1932, art. 71; ellipse with the
                                     semi-axis along the motion -> 0)

so the transverse oscillator obeys

    (m_s + m_a) y'' + k y = 0   =>   f = (1/2 pi) sqrt(k / (m_s + m_a)),

i.e. ``f / f_vacuum = sqrt(m_s / (m_s + m_a))``.  Two quantities are
cross-validated against this:

1. ``m_a`` recovered from a *forced* oscillation (least-squares fit of the
   interface force onto ``{y'', y', y, 1}``) vs ``rho_f pi c^2``;
2. the free-vibration frequency of the *coupled* implicit IBM/Aitken solver vs
   the analytic ``sqrt(k/(m_s+m_a))`` above -- and against the *measured* ``m_a``
   of (1), which separates "is the IBM operator physically right" from "is the
   staggered implicit coupling right".

A thin plate is chosen over a closed body on purpose: a closed marker ring
traps the *interior* fluid, which is dragged only through the interface and
adds ~1.5 ``m_a`` of non-analytic inertia (measured in this work), and a
circular boundary smears to an effective radius ``a + O(dx)`` (a 10 % added-mass
error at ``c = 10 dx``).  For a plate both effects are absent to leading order:
there is no interior, and the thickness-induced error is ``O((dx/2c)^2)``.

Configuration regime
--------------------
The default ``m_a/m_s ~ 0.6`` sits in the stable band of the staggered
(fluid-frozen) interface iteration; the sweep in
``benchmarks/xval_aitken_added_mass.py`` runs ``m_a/m_s`` from 0.15 to 1.5 and
finds the band edge -- i.e. the classical added-mass instability of partitioned
schemes (Causin, Gerbeau & Nobile 2005; Foerster, Wall & Ramm 2007) -- together
with what Aitken relaxation buys inside it.

Coupling order (important)
--------------------------
The interface force of time level ``k`` is applied **inside** that level's LBM
collision (``LBMFluid2D.step(force=...)``, Guo half split), not one step later.
With the one-step lag the interface force alternates sign with a growth factor
of ~2 per step for this configuration (measured: 0.07 -> -0.2 -> 0.86 -> -1.2 ->
3.1 -> -5.6), so the lag is a defect, not a modelling choice.

Deviations from LIFE (same as the companion module, plus one):
* periodic box, no walls/inflow -- the coupling algorithm, not the boundary
  treatment, is under test;
* markers carry the force only (no interior forcing), as LIFE does for thin
  structures;
* the structure is a 1-dof rigid oscillator rather than a finite-element body:
  that is what makes the reference analytic.

Public API: :func:`analytic_added_mass`, :func:`oscillator_frequency`,
:func:`plate_markers`, :func:`marker_ring`, :class:`SpringMountedBody`,
:class:`IBMImplicitFSI`, :func:`estimate_period`, :func:`estimate_period_fft`,
:func:`structural_probe`, :func:`forced_oscillation`,
:func:`coupled_free_vibration`, :func:`solve`.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from .ibm_aitken_implicit import AitkenRelaxation, IBMInterface, LBMFluid2D

__all__ = [
    "analytic_added_mass",
    "oscillator_frequency",
    "plate_markers",
    "marker_ring",
    "SpringMountedBody",
    "IBMImplicitFSI",
    "estimate_period",
    "estimate_period_fft",
    "structural_probe",
    "forced_oscillation",
    "coupled_free_vibration",
    "solve",
]


# ---------------------------------------------------------------------------
# analytic references
# ---------------------------------------------------------------------------
def analytic_added_mass(half_extent, rho_f=1.0):
    """2-D potential-flow added mass per unit length: ``rho_f * pi * c^2``.

    ``c`` is the half-width for a thin plate in broadside motion (Lamb 1932,
    art. 71) and the radius for a circular ring (art. 73); both give the same
    expression, which is why one analytic formula covers the two geometries.
    """
    return float(rho_f) * math.pi * float(half_extent) ** 2


def oscillator_frequency(stiffness, mass):
    """Natural frequency in cycles per unit time of ``m y'' + k y = 0``."""
    return math.sqrt(float(stiffness) / float(mass)) / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# marker geometries
# ---------------------------------------------------------------------------
def plate_markers(center, half_width, dx=1.0):
    """Markers equally spaced (``<= dx``) along a plate of half-width ``c``."""
    n = max(2, int(round(2.0 * half_width / dx)) + 1)
    x = np.linspace(center[0] - half_width, center[0] + half_width, n)
    return np.stack((x, np.full(n, float(center[1]))), axis=1)


def marker_ring(center, radius, dx=1.0, n_markers=None):
    """Markers on a circle, arclength spacing ``<= dx`` (closed-body variant)."""
    n = int(math.ceil(2.0 * math.pi * radius / dx)) if n_markers is None \
        else int(n_markers)
    n = max(n, 8)
    theta = 2.0 * math.pi * np.arange(n) / n
    return np.stack((center[0] + radius * np.cos(theta),
                     center[1] + radius * np.sin(theta)), axis=1)


class SpringMountedBody:
    """Rigid marker set with one transverse (``y``) dof and a linear spring.

    State is committed only through :meth:`commit`, so sub-iterations can
    evaluate trial states without corrupting the time level.
    """

    def __init__(self, positions0, mass=500.0, stiffness=3.0, y0=0.0):
        self.positions0 = np.asarray(positions0, dtype=float).copy()
        self.mass = float(mass)
        self.stiffness = float(stiffness)
        self.set_state(y0)

    # -- geometry -----------------------------------------------------------
    def markers(self, y=None):
        y = self.y if y is None else float(y)
        m = self.positions0.copy()
        m[:, 1] += y
        return m

    @property
    def half_width(self):
        return 0.5 * float(self.positions0[:, 0].max()
                           - self.positions0[:, 0].min())

    def body_velocity(self, y=None, dt=None, beta=0.25, gamma=0.5):
        """Marker velocities consistent with trial displacement ``y`` (Newmark)."""
        if y is None or dt is None:
            vy = self.v
        else:
            vy = (gamma / (beta * dt)) * (float(y) - self.y_n) \
                + (1.0 - gamma / beta) * self.v_n \
                + dt * (1.0 - gamma / (2.0 * beta)) * self.a_n
        return np.array([[0.0, vy]] * len(self.positions0))

    # -- time integration ---------------------------------------------------
    def newmark_step(self, force_y, dt, beta=0.25, gamma=0.5):
        """Trial Newmark update from the committed state under ``force_y``."""
        acc = 1.0 / (beta * dt ** 2)
        y_pred = self.y_n + dt * self.v_n + dt ** 2 * (0.5 - beta) * self.a_n
        y_new = (float(force_y) + self.mass * acc * y_pred) / \
            (self.mass * acc + self.stiffness)
        a_new = (y_new - y_pred) * acc
        v_new = self.v_n + dt * ((1.0 - gamma) * self.a_n + gamma * a_new)
        return y_new, v_new, a_new

    def commit(self, y, v, a):
        self.y, self.v, self.a = float(y), float(v), float(a)
        self.y_n, self.v_n, self.a_n = float(y), float(v), float(a)

    def set_state(self, y=0.0, v=0.0, a=0.0):
        self.y = self.y_n = float(y)
        self.v = self.v_n = float(v)
        self.a = self.a_n = float(a)


# ---------------------------------------------------------------------------
# coupled driver
# ---------------------------------------------------------------------------
class IBMImplicitFSI:
    """Implicit IBM + Aitken sub-iteration loop for a sprung rigid marker set.

    Step order: read the pre-force (Guo-free) fluid velocity of this time level,
    drive the interface fixed point to ``sub_tol`` (structure implicit, Aitken
    under-relaxation), commit the body, then apply the Lagrange force inside the
    same LBM collision.
    """

    def __init__(self, body, fluid, dt=1.0, dx=1.0, ref_length=None,
                 relax_max=1.0, sub_tol=1e-8, max_sub_iter=30,
                 relax_mode="aitken", fixed_omega=1.0, correct_epsilon=True):
        self.body = body
        self.fluid = fluid
        self.dt = float(dt)
        self.dx = float(dx)
        self.ref_length = float(ref_length if ref_length is not None
                                else 2.0 * body.half_width)
        self.ibm = IBMInterface(body.markers(), fluid.shape, dx=dx,
                               correct_epsilon=correct_epsilon)
        self.aitken = AitkenRelaxation(relax_max=relax_max, relax0=relax_max)
        self.relax_mode = relax_mode
        self.fixed_omega = float(fixed_omega)
        self.sub_tol = float(sub_tol)
        self.max_sub_iter = int(max_sub_iter)
        self.sub_iterations = []
        self.residuals = []
        self.relaxes = []
        self.n_maxit_hits = 0
        self.diverged = False
        self.diverged_reason = ""

    def interface_force(self, rho, u, y_k):
        """Hydrodynamic force on the body at trial displacement ``y_k``."""
        self.ibm.move(self.body.markers(y_k))
        rho_i, mom_i = self.ibm.interpolate(rho, u)
        f_lag = self.ibm.force(rho_i, mom_i,
                               self.body.body_velocity(y_k, self.dt),
                               dt=self.dt)
        w = self.ibm.eps_used * self.ibm.ds
        return -float(np.dot(f_lag[:, 1], w)), f_lag

    def step(self):
        body = self.body
        # pre-force fluid state of this time level: the interface force decided
        # below acts inside this step's collision (no one-step lag)
        rho, u = self.fluid.macroscopic(include_force=False)
        y_k = body.y_n + self.dt * body.v_n \
            + self.dt ** 2 * 0.25 * body.a_n           # Newmark predictor
        r_km1 = None
        omega = self.aitken.relax_max if self.relax_mode == "aitken" \
            else self.fixed_omega
        converged, res, it = False, float("inf"), 0
        for it in range(1, self.max_sub_iter + 1):
            force_on_body, _ = self.interface_force(rho, u, y_k)
            y_new, _v, _a = body.newmark_step(force_on_body, self.dt)
            r_k = np.array([y_new - y_k])
            res = float(np.linalg.norm(r_k)) / self.ref_length
            if res <= self.sub_tol:
                y_k, converged = y_new, True
                break
            if self.relax_mode != "aitken":
                omega = self.fixed_omega
            elif it >= 2:
                omega = self.aitken.omega_for(r_km1, r_k)
            else:
                omega = self.aitken.relax_max
            r_km1 = r_k.copy()
            y_k = y_k + omega * (y_new - y_k)
        if not converged:
            self.n_maxit_hits += 1
        force_on_body, f_lag = self.interface_force(rho, u, y_k)
        _y, v_new, a_new = body.newmark_step(force_on_body, self.dt)
        body.commit(y_k, v_new, a_new)
        self.fluid.step(force=self.ibm.spread(f_lag))
        self.sub_iterations.append(it)
        self.residuals.append(res)
        self.relaxes.append(float(omega))
        return {"y": float(body.y), "v": float(body.v), "a": float(body.a),
                "sub_iterations": it, "residual": res,
                "converged": bool(converged), "omega": float(omega),
                "body_force": force_on_body,
                "finite": bool(np.isfinite(body.y) and abs(body.y) < 1.0e3)}

    def run(self, n_steps):
        hist = []
        self.diverged = False
        self.diverged_reason = ""
        for _ in range(int(n_steps)):
            try:
                rec = self.step()
            except (ValueError, FloatingPointError) as exc:
                self.diverged, self.diverged_reason = True, str(exc)
                break
            hist.append(rec)
            if not rec["finite"]:
                self.diverged = True
                self.diverged_reason = "interface state left the lattice"
                break
        return hist


# ---------------------------------------------------------------------------
# signal analysis
# ---------------------------------------------------------------------------
def estimate_period(series, dt=1.0):
    """Period from upward zero crossings of the mean-removed signal."""
    y = np.asarray(series, dtype=float)
    if y.size < 4 or not np.all(np.isfinite(y)):
        return float("nan")
    y = y - y.mean()
    idx = np.where((y[:-1] <= 0.0) & (y[1:] > 0.0))[0]
    if len(idx) < 2:
        return float("nan")
    return float((idx[-1] - idx[0]) * dt / (len(idx) - 1))


def estimate_period_fft(series, dt=1.0, oversample=8):
    """Period from the dominant spectral peak (Hann window, zero padded)."""
    y = np.asarray(series, dtype=float)
    if y.size < 8 or not np.all(np.isfinite(y)):
        return float("nan")
    y = (y - y.mean()) * np.hanning(y.size)
    n = int(2 ** math.ceil(math.log2(max(y.size * oversample, 16))))
    spec = np.abs(np.fft.rfft(y, n=n))
    freqs = np.fft.rfftfreq(n, d=dt)
    if spec.size < 3:
        return float("nan")
    k = int(np.argmax(spec[1:])) + 1
    return 1.0 / float(freqs[k]) if freqs[k] > 0.0 else float("nan")


def _pct(got, ref):
    return float("nan") if ref == 0 else 100.0 * (got - ref) / abs(ref)


# ---------------------------------------------------------------------------
# experiment 0: structure alone (integrator control)
# ---------------------------------------------------------------------------
def structural_probe(mass=500.0, stiffness=3.0, y0=0.2, dt=1.0, n_steps=600):
    """Free vibration of the spring-mass dof *without* fluid (control)."""
    body = SpringMountedBody(plate_markers((0.0, 0.0), 10.0), mass=mass,
                             stiffness=stiffness, y0=y0)
    ys = []
    for _ in range(int(n_steps)):
        y_new, v_new, a_new = body.newmark_step(0.0, dt)
        body.commit(y_new, v_new, a_new)
        ys.append(body.y)
    period = estimate_period(ys, dt)
    f_num = 1.0 / period if period == period else float("nan")
    f_ref = oscillator_frequency(stiffness, mass)
    return {"mode": "structural_probe", "dt": dt, "n_steps": int(n_steps),
            "period": period, "frequency": f_num, "frequency_ref": f_ref,
            "error_pct": _pct(f_num, f_ref), "y_history": ys}


# ---------------------------------------------------------------------------
# experiment 1: forced oscillation -> added-mass coefficient
# ---------------------------------------------------------------------------
def forced_oscillation(n_steps=1200, dt=1.0, half_width=20.0,
                       center=(80.0, 80.0), lattice=(160, 160),
                       amplitude=0.25, rho_f=1.0, drive_period=160.0,
                       omega_lbm=1.8, dx=1.0, sub_tol=1e-10,
                       max_sub_iter=40):
    """Prescribe ``y(t) = A sin(2 pi t / T_d)``; fit the interface force.

    The total force on the body is fitted onto ``{y'', y', y, 1}``::

        F_y = -m_a y'' - c y' + O(y, 1)

    so ``m_a`` is recovered independently of the structural dynamics and of the
    relaxation scheme; ``y''`` and ``y'`` are the prescribed analytic
    derivatives, not finite differences of the record.
    """
    body = SpringMountedBody(plate_markers(center, half_width), mass=1.0e9,
                             stiffness=0.0)
    fluid = LBMFluid2D(lattice, omega=omega_lbm, rho0=rho_f)
    solver = IBMImplicitFSI(body, fluid, dt=dt, dx=dx, sub_tol=sub_tol,
                            max_sub_iter=max_sub_iter)
    w = 2.0 * math.pi / float(drive_period)
    ys, fs, accs, vels = [], [], [], []
    for k in range(1, int(n_steps) + 1):
        t = k * dt
        y = amplitude * math.sin(w * t)
        v = amplitude * w * math.cos(w * t)
        acc = -amplitude * w ** 2 * math.sin(w * t)
        if acc == 0.0:
            acc = -amplitude * w ** 2 * 1e-12   # keep the basis well posed
        rho, u = solver.fluid.macroscopic(include_force=False)
        solver.ibm.move(body.markers(y))
        rho_i, mom_i = solver.ibm.interpolate(rho, u)
        u_body = np.array([[0.0, v]] * len(body.positions0))
        f_lag = solver.ibm.force(rho_i, mom_i, u_body, dt=dt)
        wgt = solver.ibm.eps_used * solver.ibm.ds
        solver.fluid.step(force=solver.ibm.spread(f_lag))
        ys.append(y)
        fs.append(-float(np.dot(f_lag[:, 1], wgt)))
        accs.append(acc)
        vels.append(v)
    ys = np.asarray(ys)
    fs = np.asarray(fs)
    accs = np.asarray(accs)
    vels = np.asarray(vels)
    skip = int(2 * drive_period / dt)          # drop the start-up transient
    basis = np.stack([accs[skip:], vels[skip:], ys[skip:],
                      np.ones_like(ys[skip:])], axis=1)
    coef, *_ = np.linalg.lstsq(basis, fs[skip:], rcond=None)
    resid = fs[skip:] - basis @ coef
    denom = float(np.sum((fs[skip:] - fs[skip:].mean()) ** 2)) or 1.0
    m_a = -float(coef[0])
    m_a_ref = analytic_added_mass(half_width, rho_f)
    return {
        "mode": "forced_oscillation", "dt": dt, "n_steps": int(n_steps),
        "half_width": half_width, "amplitude": amplitude,
        "drive_period": drive_period, "lattice": list(lattice),
        "markers": int(len(body.positions0)),
        "added_mass_fit": m_a, "added_mass_ref": m_a_ref,
        "added_mass_error_pct": _pct(m_a, m_a_ref),
        "damping_fit": -float(coef[1]), "fit_r2":
            float(1.0 - np.sum(resid ** 2) / denom),
        "y_history": ys.tolist(), "force_history": fs.tolist(),
    }


# ---------------------------------------------------------------------------
# experiment 2: coupled free vibration -> frequency shift
# ---------------------------------------------------------------------------
def coupled_free_vibration(n_steps=900, dt=1.0, half_width=20.0,
                           center=(80.0, 80.0), lattice=(160, 160),
                           mass=500.0, stiffness=3.0, y0=0.2, rho_f=1.0,
                           omega_lbm=1.8, dx=1.0, relax_mode="aitken",
                           fixed_omega=1.0, relax_max=1.0, sub_tol=1e-8,
                           max_sub_iter=30, correct_epsilon=True,
                           added_mass=None):
    """Plate released from ``y0`` in a quiescent fluid; frequency from the record.

    ``added_mass`` optionally overrides the analytic ``m_a`` used for the
    reference (used to build the reference from the *measured* ``m_a``).
    """
    body = SpringMountedBody(plate_markers(center, half_width), mass=mass,
                             stiffness=stiffness, y0=y0)
    fluid = LBMFluid2D(lattice, omega=omega_lbm, rho0=rho_f)
    solver = IBMImplicitFSI(body, fluid, dt=dt, dx=dx, relax_mode=relax_mode,
                            fixed_omega=fixed_omega, relax_max=relax_max,
                            sub_tol=sub_tol, max_sub_iter=max_sub_iter,
                            correct_epsilon=correct_epsilon)
    hist = solver.run(n_steps)
    ys = [h["y"] for h in hist]
    period = estimate_period(ys, dt)
    period_fft = estimate_period_fft(ys, dt)
    f_num = 1.0 / period if period == period else float("nan")
    f_fft = 1.0 / period_fft if period_fft == period_fft else float("nan")
    m_a_ref = analytic_added_mass(half_width, rho_f) if added_mass is None \
        else float(added_mass)
    f_vac = oscillator_frequency(stiffness, mass)
    f_ref = oscillator_frequency(stiffness, mass + m_a_ref)
    ratio_meas = float("nan")
    if f_num == f_num and f_num > 0.0:
        ratio_meas = stiffness / (2.0 * math.pi * f_num) ** 2 - mass
    return {
        "mode": "coupled_free_vibration", "relax_mode": relax_mode,
        "fixed_omega": fixed_omega, "dt": dt, "n_steps": int(n_steps),
        "steps_completed": len(hist), "half_width": half_width, "mass": mass,
        "stiffness": stiffness, "y0": y0,
        "markers": int(len(body.positions0)),
        "added_mass_ratio": m_a_ref / mass if mass else float("nan"),
        "period": period, "period_fft": period_fft,
        "frequency": f_num, "frequency_fft": f_fft,
        "frequency_vacuum": f_vac, "frequency_ref": f_ref,
        "frequency_ratio": f_num / f_vac if f_vac else float("nan"),
        "frequency_ratio_ref": f_ref / f_vac if f_vac else float("nan"),
        "frequency_error_pct": _pct(f_num, f_ref),
        "frequency_error_fft_pct": _pct(f_fft, f_ref),
        "added_mass_measured": ratio_meas,
        "added_mass_measured_error_pct": _pct(ratio_meas, m_a_ref),
        "mean_sub_iterations": float(np.mean(solver.sub_iterations))
            if solver.sub_iterations else float("nan"),
        "max_sub_iterations": int(max(solver.sub_iterations))
            if solver.sub_iterations else 0,
        "maxit_hits": int(solver.n_maxit_hits),
        "converged_all": bool(all(h["converged"] for h in hist)) if hist
            else False,
        "diverged": bool(solver.diverged),
        "diverged_reason": solver.diverged_reason,
        "y_history": ys,
        "iter_history": solver.sub_iterations,
        "relax_history": solver.relaxes,
        "residual_history": solver.residuals,
    }


def solve(kind="coupled", **kwargs):
    """Generic entry: ``kind`` in ``{"structural", "forced", "coupled"}``."""
    table = {"structural": structural_probe,
             "forced": forced_oscillation,
             "coupled": coupled_free_vibration}
    if kind not in table:
        raise ValueError(f"unknown kind {kind!r}; use {sorted(table)}")
    return table[kind](**kwargs)


def main(argv=None):        # pragma: no cover - CLI convenience
    argv = list(sys.argv[1:] if argv is None else argv)
    kind = argv[0] if argv else "coupled"
    kw = {}
    for token in argv[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            kw[k] = json.loads(v)
    rec = solve(kind, **kw)
    rec.pop("y_history", None)
    rec.pop("force_history", None)
    print(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":        # pragma: no cover
    main()