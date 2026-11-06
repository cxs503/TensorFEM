"""Large-deflection cantilever under a transverse tip load: elastica oracle vs FE beam.

The merged large-rotation evidence in this repository
(``large_rotation_shell_benchmark.py``) is *displacement controlled*: the end
rotation is prescribed and the reference is the circular-elastica formula.  It
therefore never exercises **load control**, and a circular arc is a very special
elastica.  A transverse tip force on a cantilever is the classical complementary
case: no limit point exists, the free-end slope becomes large, and the exact
shape is a genuinely non-circular Kirchhoff/Clebsch elastica.

Oracle (derived here, not digitised)
------------------------------------
With arc length ``s`` from the free end, tangent angle ``theta`` measured from
the undeformed axis and ``k^2 = P/(E I)``::

    theta''(s) = -k^2 cos(theta),  theta(0)=theta0, theta'(0)=0, theta(L)=0

The first integral is ``theta'^2 = 2 k^2 (sin(theta0) - sin(theta))``, hence

    I_len(theta0) := int_0^theta0 dtheta / sqrt(2 (sin(theta0)-sin(theta))) = sqrt(P L^2/(E I))

which fixes the free-end slope ``theta0``; the tip kinematics follow from the
same kernel with ``cos``/``sin`` numerators::

    x_tip/L = I_cos/sqrt(alpha),   w_tip/L = I_sin/sqrt(alpha),   alpha = P L^2/(E I)

The integrand is evaluated after ``theta = theta0 - t^2``, which removes the
endpoint singularity.  Two independent numerical routes are provided so the
oracle can be audited: Gauss-Legendre quadrature of that kernel
(:func:`solve_elastica`) and a direct boundary-value solve of the ODE system
(:func:`elastica_from_ode`).  Neither uses the finite-element solution.

Finite-element side
-------------------
:func:`corotational_beam_cantilever` is a load-controlled geometrically
nonlinear 2-D beam (corotational Euler-Bernoulli, 3 DOF per node) whose tangent
comes from the exact Hessian of its potential energy, solved with adaptive
load increments.  It is deliberately cheap (milliseconds) so the whole
mesh/load sweep can live in the test suite.

Because the shell strip in this repository is *not* load-control capable at
acceptable cost, the FE side here is the dedicated beam path; this module does
therefore validate the elastica oracle and the load-controlled nonlinear
solution path, not the Shell4 facet element.  That boundary is stated
explicitly in ``docs/elastica-cantilever-validation.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch

# ---------------------------------------------------------------------------
# exact (elastica) oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElasticaSolution:
    """Exact inextensible-elastica solution for a cantilever tip load."""

    alpha: float          # P L^2 / (E I), dimensionless
    tip_rotation: float   # theta0 [rad]
    deflection_ratio: float   # w / L, transverse tip deflection / L
    shortening_ratio: float   # (L - x_tip) / L, axial pull-in / L

    def tip_deflection(self, length: float) -> float:
        return self.deflection_ratio * length

    def tip_shortening(self, length: float) -> float:
        return self.shortening_ratio * length


@lru_cache(maxsize=None)
def _gauss(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Legendre abscissae (``leggauss`` is expensive to rebuild)."""
    t, w = np.polynomial.legendre.leggauss(nodes)
    t.setflags(write=False)
    w.setflags(write=False)
    return t, w


def _kernel(theta0: float, nodes: int = 512):
    """Return (I_len, I_cos, I_sin) after the theta = theta0 - t^2 substitution."""
    limit = math.pi / 2 - 1e-12
    if not 0.0 < theta0 < limit:
        raise ValueError("theta0 must lie strictly inside (0, pi/2)")
    t, w = _gauss(nodes)
    tmax = math.sqrt(theta0)
    tt = 0.5 * tmax * (t + 1.0)
    ww = 0.5 * tmax * w
    theta = theta0 - tt * tt
    # sin(theta0) - sin(theta), written to avoid catastrophic cancellation
    diff = 2.0 * np.cos(0.5 * (theta + theta0)) * np.sin(0.5 * (theta0 - theta))
    integrand = 2.0 * tt / np.sqrt(2.0 * diff)     # |dtheta/dt| / sqrt(2(sin0-sin))
    i_len = float(np.sum(ww * integrand))
    i_cos = float(np.sum(ww * integrand * np.cos(theta)))
    i_sin = float(np.sum(ww * integrand * np.sin(theta)))
    return i_len, i_cos, i_sin


def solve_elastica(alpha: float, *, nodes: int = 512, tol: float = 1e-14) -> ElasticaSolution:
    """Solve the tip-loaded cantilever elastica for ``alpha = P L^2 / (E I)``."""
    from scipy.optimize import brentq

    if not alpha > 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be positive and finite")
    target = math.sqrt(alpha)
    lo, hi = 1e-12, math.pi / 2 - 1e-7
    if _kernel(lo, nodes)[0] > target:
        raise ValueError("alpha too small for this bracket")
    theta0 = brentq(lambda th: _kernel(th, nodes)[0] - target, lo, hi,
                    xtol=tol, rtol=8.9e-16, maxiter=200)
    i_len, i_cos, i_sin = _kernel(theta0, nodes)
    scale = 1.0 / target
    return ElasticaSolution(alpha=alpha, tip_rotation=theta0,
                            deflection_ratio=scale * i_sin,
                            shortening_ratio=1.0 - scale * i_cos)


def elastica_from_ode(alpha: float, *, steps: int = 40) -> ElasticaSolution:
    """Independent oracle: solve the elastica boundary-value problem directly.

    Unit length ``L = 1``: states ``[theta, theta', x, z]`` with
    ``theta' = -k^2 cos(theta)`` (``k^2 = alpha``), ``x' = cos(theta)``,
    ``z' = sin(theta)``; boundary conditions ``theta(0) = x(0) = z(0) = 0``
    (clamped root) and ``theta'(1) = 0`` (moment-free tip).  Continuation in
    ``k`` supplies the initial guess.  No elliptic integral, no shooting.
    """
    from scipy.integrate import solve_bvp

    if not alpha > 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be positive and finite")
    s = np.linspace(0.0, 1.0, 201)

    def system(_s, y, k2):
        theta, dtheta, _x, _z = y
        return np.vstack((dtheta, -k2 * np.cos(theta), np.cos(theta), np.sin(theta)))

    def bc(ya, yb):
        return np.array([ya[0], ya[2], ya[3], yb[1]])

    y = np.zeros((4, s.size))
    for k2 in np.linspace(0.0, alpha, steps)[1:]:
        sol = solve_bvp(lambda _s, yy, k=k2: system(_s, yy, k), bc, s, y,
                        tol=1e-11, max_nodes=8000)
        if sol.status != 0:
            raise RuntimeError(f"elastica BVP failed at k^2={k2}: {sol.message}")
        y = sol.sol(s)
    theta, _dtheta, x, z = y
    return ElasticaSolution(alpha=alpha, tip_rotation=float(theta[-1]),
                            deflection_ratio=float(z[-1]),
                            shortening_ratio=1.0 - float(x[-1]))


def oracle_cross_check(alpha: float) -> dict[str, float]:
    """Compare the two independent oracle routes; returns absolute differences."""
    a = solve_elastica(alpha)
    b = elastica_from_ode(alpha)
    return {
        "alpha": alpha,
        "d_tip_rotation": abs(a.tip_rotation - b.tip_rotation),
        "d_deflection_ratio": abs(a.deflection_ratio - b.deflection_ratio),
        "d_shortening_ratio": abs(a.shortening_ratio - b.shortening_ratio),
        "quadrature_deflection_ratio": a.deflection_ratio,
        "bvp_deflection_ratio": b.deflection_ratio,
    }


# ---------------------------------------------------------------------------
# load-controlled corotational beam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CantileverCase:
    alpha: float
    elements: int
    converged: bool
    load_factor: float
    tip_deflection: float     # transverse tip deflection (positive), length units
    tip_shortening: float     # axial pull-in of the tip (positive inward), length units
    exact: ElasticaSolution
    length: float
    load: float
    iterations: int = 0

    @property
    def deflection_error(self) -> float:
        return abs(self.tip_deflection / self.exact.tip_deflection(self.length) - 1.0)

    @property
    def shortening_error(self) -> float:
        return abs(self.tip_shortening / self.exact.tip_shortening(self.length) - 1.0)


def _beam_energy(nodes0: torch.Tensor, dofs: torch.Tensor, ea: float, ei: float,
                 lengths: torch.Tensor) -> torch.Tensor:
    """Corotational 2-D Euler-Bernoulli potential energy (quadratic local form)."""
    x = nodes0[:, 0] + dofs[:, 0]
    z = nodes0[:, 1] + dofs[:, 1]
    theta = dofs[:, 2]
    dx = x[1:] - x[:-1]
    dz = z[1:] - z[:-1]
    current = torch.sqrt(dx * dx + dz * dz)
    phi = torch.atan2(dz, dx)
    extension = current - lengths
    a_i = theta[:-1] - phi
    a_j = theta[1:] - phi
    axial = 0.5 * ea / lengths * extension * extension
    bending = 2.0 * ei / lengths * (a_i * a_i + a_i * a_j + a_j * a_j)
    return torch.sum(axial + bending)


def corotational_beam_cantilever(alpha: float, elements: int, *, length: float = 10.0,
                                 young: float = 1.2e6, thickness: float = 0.1,
                                 width: float = 1.0, poisson: float = 0.0,
                                 increment: float = 0.1, minimum_increment: float = 1e-3,
                                 max_iterations: int = 15,
                                 tolerance: float = 1e-10) -> CantileverCase:
    """Clamp a 2-D beam and pull its tip with a transverse dead load.

    ``E I = 10`` force*length^2 for the default properties, so ``P = 0.1 alpha``.
    """
    if elements < 1:
        raise ValueError("elements must be at least 1")
    if not alpha > 0.0:
        raise ValueError("alpha must be positive")
    inertia = width * thickness ** 3 / 12.0
    flexural = young * inertia
    load = alpha * flexural / length ** 2
    axial = young * width * thickness
    dtype = torch.float64

    xs = np.linspace(0.0, length, elements + 1)
    nodes0 = torch.tensor(np.stack([xs, np.zeros_like(xs)], axis=1), dtype=dtype)
    lengths = torch.full((elements,), length / elements, dtype=dtype)

    dofs = torch.zeros(3 * (elements + 1), dtype=dtype)
    free = torch.arange(3, dofs.numel())        # node 0 clamped (u, w, theta)
    force = torch.zeros_like(dofs)
    force[3 * elements + 1] = -load             # transverse tip dead load

    def gradient(q_flat):
        q = q_flat.detach().clone().requires_grad_(True)
        return torch.autograd.grad(
            _beam_energy(nodes0, q.reshape(-1, 3), axial, flexural, lengths), q)[0]

    def tangent(q, h: float = 1e-6):
        """Symmetric central-difference Hessian (kept explicit: analytic AD Hessian
        of the chord-frame energy is rank deficient at the straight state)."""
        base = q.detach().clone()
        cols = []
        for j in range(base.numel()):
            step = torch.zeros_like(base)
            step[j] = h
            cols.append((gradient(base + step) - gradient(base - step)) / (2 * h))
        K = torch.stack(cols, dim=1)
        return 0.5 * (K + K.T)

    factor, step, iterations = 0.0, min(increment, 1.0), 0
    q = dofs.clone()
    while factor < 1.0 - 1e-14:
        target = min(1.0, factor + step)
        trial = q.clone()
        converged = False
        for _ in range(max_iterations):
            iterations += 1
            res = gradient(trial) - target * force
            sol = res[free].detach()
            if float(torch.linalg.vector_norm(sol)) <= tolerance:
                converged = True
                break
            try:
                delta = torch.linalg.solve(tangent(trial)[free[:, None], free], -sol)
            except torch.linalg.LinAlgError:
                break
            trial = (trial.detach().clone())
            trial[free] += delta
        if converged:
            q = trial.detach().clone()
            factor = target
            step = min(step * 1.5, 0.5, 1.0 - factor + 1e-15) if factor < 1.0 else step
        else:
            step *= 0.5
            if step < minimum_increment:
                break
    q = q.detach().reshape(-1, 3)
    tip = q[-1]
    return CantileverCase(alpha=alpha, elements=elements,
                          converged=bool(factor >= 1.0 - 1e-9),
                          load_factor=factor,
                          tip_deflection=-float(tip[1]), tip_shortening=-float(tip[0]),
                          exact=solve_elastica(alpha), length=length, load=load,
                          iterations=iterations)


# ---------------------------------------------------------------------------
# benchmark aggregation / gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElasticaBenchmark:
    cases: tuple[CantileverCase, ...]
    tolerance: float = 0.03

    @property
    def finest(self) -> tuple[CantileverCase, ...]:
        if not self.cases:
            return ()
        best = max(c.elements for c in self.cases)
        return tuple(c for c in self.cases if c.elements == best)

    @property
    def all_converged(self) -> bool:
        return bool(self.cases) and all(
            c.converged and c.load_factor >= 1.0 - 1e-9 for c in self.cases)

    @property
    def qualified(self) -> bool:
        """Fail-closed: every finest-mesh response must be *below* the gate."""
        if not self.all_converged or not self.finest:
            return False
        return all(c.deflection_error < self.tolerance and
                   c.shortening_error < self.tolerance for c in self.finest)

    def monotone(self) -> bool:
        """Refinement must not increase the deflection error of any load level."""
        for alpha in sorted({c.alpha for c in self.cases}):
            group = sorted((c for c in self.cases if c.alpha == alpha),
                           key=lambda c: c.elements)
            errs = [c.deflection_error for c in group]
            if any(b > a + 1e-12 for a, b in zip(errs, errs[1:])):
                return False
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "tolerance": self.tolerance,
            "qualified": self.qualified,
            "monotone_refinement": self.monotone(),
            "all_converged": self.all_converged,
            "cases": [
                {"alpha": c.alpha, "elements": c.elements, "converged": c.converged,
                 "load_factor": c.load_factor, "load": c.load,
                 "iterations": c.iterations,
                 "tip_deflection": c.tip_deflection,
                 "tip_deflection_exact": c.exact.tip_deflection(c.length),
                 "deflection_error": c.deflection_error,
                 "tip_shortening": c.tip_shortening,
                 "tip_shortening_exact": c.exact.tip_shortening(c.length),
                 "shortening_error": c.shortening_error,
                 "tip_rotation": c.exact.tip_rotation}
                for c in self.cases
            ],
        }


def cantilever_benchmark(alphas=(1.0, 2.0, 5.0, 10.0), meshes=(2, 4, 8, 16),
                         **kwargs) -> ElasticaBenchmark:
    """Run the load-level x mesh sweep used by the validation report."""
    cases = []
    for alpha in alphas:
        for n in meshes:
            cases.append(corotational_beam_cantilever(float(alpha), int(n), **kwargs))
    return ElasticaBenchmark(tuple(cases))


# Published tabulated deflection ratios, used only as a coarse sanity check of
# the independently derived oracle (Mattiasson 1981; Bisshopp & Drucker 1945 use
# the same load parameter).  They are never used to build the reference, and the
# table itself is not reproduced here.
PUBLISHED_CROSS_CHECK = {
    1.0: (0.30172, 0.05643),
    2.0: (0.49346, 0.16064),
    5.0: (0.71379, 0.39004),
    10.0: (0.81061, 0.55538),
}