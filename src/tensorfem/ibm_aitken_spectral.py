"""Spectral (multi-DOF) law for Aitken Delta^2 interface relaxation.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (joconnor22/LIFE implicit
immersed boundary + Aitken Delta^2 under-relaxation).  Rounds v1-v11 of this
anchor verified a *scalar* interface map: a single heave degree of freedom, for
which the relaxation factor ``omega* = 1/(1+R)`` reaches the fixed point in two
relaxations (three sub-iterations) for every added-mass gain ``R``
(``docs/xval_implicit_ibm_aitken_v11.md``).

That closed form is the ``n = 1`` case of a much older statement.  With the
residual ``r_k = G(y_k) - y_k`` of an affine interface map
``G(y) = A y + b`` and the relaxed iterate ``y_{k+1} = y_k + omega_k r_k``, the
Aitken Delta^2 update that :class:`tensorfem.ibm_aitken_implicit.AitkenRelaxation`
implements,

    omega_k = -omega_{k-1} (r_{k-1} . (r_k - r_{k-1})) / |r_k - r_{k-1}|^2 ,

substituted with ``r_k - r_{k-1} = -omega_{k-1} M r_{k-1}`` (``M := I - A``, the
interface operator), is *identically*

    omega_k = (r_{k-1} . M r_{k-1}) / |M r_{k-1}|^2 ,

i.e. the Rayleigh quotient that minimises ``|(I - omega M) r_{k-1}|`` over
``omega``.  Aitken Delta^2 relaxation **is** minimal-residual / steepest-descent
relaxation with an exact line search; equivalently Anderson acceleration with
memory one (Walker & Ni 2011; Evans et al. 2020 give the Anderson(1) <->
nonlinear-CG equivalence).  Consequences, all exact:

* ``n = 1`` (or a residual inside one eigenvector): ``omega_k = 1/mu`` with
  ``mu`` the only eigenvalue, ``r_{k+1} = 0`` -- finite termination, the v11 law,
  with ``mu = 1 + R``;
* ``n > 1``: the rule is a *rank-one* line search, so it cannot zero a residual
  that spans two eigen-directions; the iteration is linear with the classical
  steepest-descent bound ``|r_{k+1}| / |r_k| <= (kappa - 1)/(kappa + 1)``,
  ``kappa = cond(M)``, and no finite termination exists;
* one global ``omega`` for a body whose degrees of freedom have *different*
  added-mass ratios (translation vs rotation of a 6-DOF LIFE-type body) is
  therefore bounded by the *worst* ratio, while a per-DOF-group (block) Aitken
  recovers three sub-iterations per group.

The functions below are the closed forms used by
``benchmarks/xval_aitken_spectral_law.py``; the analytic added-mass matrix is
the ellipse of Lamb (1932) art. 105 / Newman (1977) §4.6 (2-D, per unit span).

Public API: :func:`ellipse_added_mass`, :func:`ellipse_body_inertia`,
:func:`added_mass_operator`, :func:`operator_spectrum`,
:func:`rayleigh_step`, :func:`minres_relaxation`, :func:`rate_bound`,
:func:`geometric_rate`, :func:`stability_edge`, :func:`block_minres`.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "ellipse_added_mass",
    "ellipse_body_inertia",
    "added_mass_operator",
    "operator_spectrum",
    "rayleigh_step",
    "minres_relaxation",
    "rate_bound",
    "geometric_rate",
    "stability_edge",
    "block_minres",
]


def ellipse_added_mass(a, b, rho=1.0):
    """Added-mass matrix entries of a 2-D ellipse in potential flow.

    ``a``/``b`` are the semi-axes along ``x``/``y``.  Lamb (1932) art. 105 and
    Newman (1977) §4.6, per unit span:

    * surge (translation along ``x``): ``rho pi b^2``
    * heave (translation along ``y``): ``rho pi a^2``
    * pitch (rotation about the centre): ``rho pi (a^2 - b^2)^2 / 8``

    A circular cylinder (``a = b``) has zero rotational added mass and keeps the
    translational value ``rho pi a^2`` -- the limit used as a test case below.
    """
    a, b, rho = float(a), float(b), float(rho)
    return {"surge": rho * math.pi * b * b,
            "heave": rho * math.pi * a * a,
            "pitch": rho * math.pi * (a * a - b * b) ** 2 / 8.0}


def ellipse_body_inertia(a, b, rho_s=1.0):
    """Solid elliptical section: ``(mass, pitch inertia)`` per unit span.

    ``mass = rho_s pi a b`` and ``I = mass (a^2 + b^2)/4`` (second moment of a
    solid ellipse about its centre).
    """
    a, b, rho_s = float(a), float(b), float(rho_s)
    mass = rho_s * math.pi * a * b
    return mass, mass * (a * a + b * b) / 4.0


def added_mass_operator(mass_matrix, added_matrix):
    """Interface operator ``M = I + Ms^{-1} Ma`` of the added-mass exchange.

    The implicit direct-forcing coupling hands the fluid force computed at the
    trial displacement back to the structure (``g = -S/m_eff`` in the scalar
    notation of ``tests/test_aitken_strong_coupling.py``); written for a body
    with mass matrix ``Ms`` and added-mass matrix ``Ma`` the fixed point map is
    ``G(y) = -Ms^{-1} Ma y + b``, so ``M = I - A = I + Ms^{-1} Ma``.  Its
    eigenvalues ``mu_i = 1 + R_i`` are the per-DOF gains; the v11 scalar law uses
    ``mu = 1 + R``.
    """
    ms = np.atleast_2d(np.asarray(mass_matrix, dtype=float))
    ma = np.atleast_2d(np.asarray(added_matrix, dtype=float))
    if ms.shape != ma.shape:
        raise ValueError(f"mass/added matrices differ in shape: {ms.shape} vs {ma.shape}")
    return np.eye(ms.shape[0]) + np.linalg.solve(ms, ma)


def operator_spectrum(operator):
    """Real eigenvalues of the interface operator, ascending, as floats.

    A complex pair (a genuinely non-symmetric coupled added-mass matrix, e.g.
    sway-pitch of a body with an offset centre of mass) is rejected rather than
    silently cast: the closed forms below are stated for real spectra.
    """
    op = np.atleast_2d(np.asarray(operator, dtype=float))
    vals = np.linalg.eigvals(op)
    if np.max(np.abs(vals.imag)) > 1e-9 * max(1.0, float(np.max(np.abs(vals)))):
        raise ValueError("interface operator has a complex spectrum; closed forms "
                         "used here are stated for real eigenvalues")
    return np.sort(vals.real)


def rayleigh_step(r_prev, r_now, omega_prev):
    """Exact-line-search factor of the Aitken recursion (the Rayleigh quotient).

    Independent implementation of ``(r . M r) / |M r|^2`` with
    ``M r = (r_prev - r_now)/omega_prev`` -- the residual pair alone determines
    it, so this can be compared against the production recursion without
    touching the solver.  ``omega_prev`` must be the factor that produced
    ``r_now`` from ``r_prev`` (the ``omega_applied`` convention of the solver).
    """
    r_prev = np.atleast_1d(np.asarray(r_prev, dtype=float))
    r_now = np.atleast_1d(np.asarray(r_now, dtype=float))
    w = float(omega_prev)
    if w == 0.0:
        raise ValueError("omega_prev must be non-zero")
    m_r = (r_prev - r_now) / w
    den = float(np.dot(m_r, m_r))
    if den == 0.0:
        raise ValueError("degenerate residual pair: M r = 0")
    return float(np.dot(r_prev, m_r)) / den


def minres_relaxation(operator, r0, tol=1e-6, first_omega=None, max_iter=200,
                      variant="lagged"):
    """Residual sequence of minimal-residual (== Aitken Delta^2) relaxation.

    Iterates ``r_{k+1} = (I - omega_k M) r_k`` with the exact-line-search factor
    ``omega_k = (r_k . M r_k)/|M r_k|^2``.  ``r0`` is the first residual of a
    time step.  ``first_omega`` reproduces the production loop: the *first*
    correction is relaxed by that fixed factor (the shipped code starts from
    ``relax_max``) and the exact line search afterwards; ``None`` searches from
    the first relaxation on.

    ``variant`` selects which residual the Rayleigh quotient is evaluated at:

    * ``"lagged"`` (default) -- ``omega_k = (r_{k-1} . M r_{k-1})/|M r_{k-1}|^2``,
      the factor the shipped ``AitkenRelaxation`` actually produces from the
      residual pair ``(r_{k-1}, r_k)``: its numerator uses ``r_{k-1}``, while the
      factor is applied to the correction leaving ``r_k``.  This is the model
      that matches the production code to round-off;
    * ``"aligned"`` -- ``omega_k = (r_k . M r_k)/|M r_k|^2``, the exact line
      search at the residual being corrected (minimal residual proper).  For
      ``n = 1`` the two coincide, which is why the v11 scalar law needs no such
      distinction.

    Returns the predicted per-iteration residual norms, factors and the
    iteration count at which ``|r| <= tol |r0|``.
    """
    op = np.atleast_2d(np.asarray(operator, dtype=float))
    r = np.atleast_1d(np.asarray(r0, dtype=float)).copy()
    scale = float(np.linalg.norm(r))
    if scale == 0.0:
        raise ValueError("r0 is zero")
    residuals = [scale]
    omegas = []
    r_prev = r.copy()
    converged = scale <= tol * scale
    k = 0
    while not converged and k < max_iter:
        m_r = op @ r
        den = float(np.dot(m_r, m_r))
        if den == 0.0:
            break
        if k == 0 and first_omega is not None:
            omega = float(first_omega)
        elif variant == "lagged" and k > 0:
            m_prev = op @ r_prev
            den_prev = float(np.dot(m_prev, m_prev))
            omega = float(np.dot(r_prev, m_prev)) / den_prev
        elif variant == "aligned":
            omega = float(np.dot(r, m_r)) / den
        else:
            omega = float(np.dot(r, m_r)) / den
        if variant not in ("lagged", "aligned"):
            raise ValueError(f"unknown variant {variant!r}")
        omegas.append(omega)
        r_prev = r
        r = r - omega * m_r
        k += 1
        norm = float(np.linalg.norm(r))
        residuals.append(norm)
        converged = norm <= tol * residuals[0]
    return {"residuals": residuals, "omegas": omegas, "iterations": k,
            "converged": bool(converged),
            "first_omega": None if first_omega is None else float(first_omega)}


def rate_bound(eigenvalues):
    """``(kappa, (kappa-1)/(kappa+1))`` for a positive spectrum."""
    vals = np.sort(np.asarray(eigenvalues, dtype=float))
    if vals[0] <= 0.0:
        raise ValueError("spectrum must be positive (an indefinite interface "
                         "operator cannot be stabilised by a line search)")
    kappa = float(vals[-1] / vals[0])
    return kappa, (kappa - 1.0) / (kappa + 1.0)


def geometric_rate(residuals):
    """Observed per-iteration ratio of a residual-norm sequence.

    Returns ``(per_step_ratios, asymptotic_ratio, log_linear_r2)``; the
    asymptotic ratio is the geometric mean of the last third of the ratios and
    ``r2`` the coefficient of determination of ``log|r_k|`` against ``k`` (a
    rank-one interface gives ratios that are constant, so ``r2 -> 1``).
    """
    res = np.asarray([r for r in residuals if r > 0.0], dtype=float)
    if res.size < 3:
        raise ValueError("need at least three positive residuals")
    ratios = res[1:] / res[:-1]
    tail = ratios[max(0, (2 * ratios.size) // 3):]
    k = np.arange(res.size, dtype=float)
    logr = np.log(res)
    slope, intercept = np.polyfit(k, logr, 1)
    fit = slope * k + intercept
    ss_res = float(np.sum((logr - fit) ** 2))
    ss_tot = float(np.sum((logr - logr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return ratios, float(np.exp(np.mean(np.log(tail)))), r2


def stability_edge(operator):
    """Largest fixed relaxation factor that still converges.

    For a symmetric positive operator the iteration matrix is ``I - omega M`` and
    ``|1 - omega mu_max| < 1`` gives ``0 < omega < 2/mu_max``; the ``omega = 1``
    case is ``mu_max < 2``, i.e. ``R < 1`` -- the Causin-Gerbeau-Nobile (2005)
    added-mass instability threshold that v11 measured.
    """
    vals = np.sort(np.asarray(operator_spectrum(operator)))
    return 2.0 / float(vals[-1])


def block_minres(operator, r0, tol=1e-6, groups=None, max_iter=200):
    """Per-block (per-DOF-group) Aitken: relax each invariant block separately.

    ``groups`` is a list of index arrays; the default is one group per DOF, which
    is the best case an implementation can reach when the interface operator is
    diagonal (each group then obeys the ``n = 1`` law).  The returned count is the
    worst group's, which is the wall-clock cost of the time step.
    """
    op = np.atleast_2d(np.asarray(operator, dtype=float))
    r0 = np.atleast_1d(np.asarray(r0, dtype=float))
    if groups is None:
        groups = [np.array([i]) for i in range(op.shape[0])]
    counts, residuals = [], []
    for g in groups:
        idx = np.atleast_1d(np.asarray(g, dtype=int))
        sub = op[np.ix_(idx, idx)]
        out = minres_relaxation(sub, r0[idx], tol=tol, first_omega=1.0,
                                max_iter=max_iter)
        counts.append(out["iterations"] + 1)   # sub-iterations = relaxations + 1
        residuals.append(out["residuals"])
    return {"iterations": int(max(counts)), "counts": counts,
            "residuals": residuals}
