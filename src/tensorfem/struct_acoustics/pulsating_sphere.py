"""Exterior acoustic radiation of a pulsating sphere: exact monopole anchor + radial Helmholtz FEM.

Physical background
-------------------
A rigid sphere of radius ``a`` vibrates in its breathing mode (Legendre order
``n = 0``) with uniform radial surface velocity ``v0`` (peak amplitude,
``e^{-i omega t}`` convention) and radiates into an unbounded acoustic medium
``(rho, c)``.  The spherically symmetric Helmholtz problem

.. math:: \\frac{1}{r^2}\\frac{d}{dr}\\left(r^2\\frac{dp}{dr}\\right)
          + k^2 p = 0, \\qquad a < r < \\infty, \\qquad k = \\omega/c

has the outgoing solution

.. math:: p(r) = \\rho c v_0 \\frac{a}{r}\\,e^{ik(r-a)} \\frac{1}{1 + i/(ka)} .

Two limits are worth naming because the campaign node quotes both:

* ``ka >> 1`` (ray/plane-wave limit): the surface pressure tends to the plane
  wave value ``rho c v0`` and

  .. math:: p(r) \\to \\rho c v_0 \\frac{a}{r} e^{ik(r-a)},
            \\qquad W \\to 2\\pi a^2 \\rho c v_0^2 .

  This is exactly the expression quoted in the node specification.  Its
  relative error to the exact monopole is ``1/sqrt(1 + (ka)^2)``, i.e. 98 % at
  ``ka = 0.21`` -- see :func:`plane_wave_form_relative_error`.

* ``ka << 1`` (compact source): the surface pressure *collapses* as ``ka``
  because the medium is inertia limited,

  .. math:: p(a) \\approx -i\\,\\rho c v_0\\, k a,
            \\qquad W \\approx 2\\pi a^2 \\rho c v_0^2 (ka)^2 .

  So the exact radiated power of the node formula is only attained in the
  high-``ka`` limit; :func:`radiated_power` returns the exact value and
  :func:`plane_wave_power` the node's asymptotic one.

Independent anchors used here
-----------------------------
1. Exact closed-form monopole (above) -- no numerical model, no fitted constant.
2. Radiation impedance ``Z = p(a)/v0 = rho c (ka)(ka - i)/(1 + (ka)^2)``, whose
   real part must reproduce the far-field power integral
   ``W = 1/2 Re(Z) 4 pi a^2 v0^2``.
3. Energy flux conservation of the discretised field: the power crossing the
   inner surface must equal the power crossing the outer truncation surface
   (:func:`solve_pulsating_sphere` reports both routes).

Numerical method
----------------
The spherical acoustic domain ``a <= r <= R`` is discretised with **3-node
quadratic radial finite elements** and the Galerkin weak form of the radial
Helmholtz equation,

.. math:: \\int_a^R \\frac{r^2}{\\rho} p' q'\\,dr
          - \\omega^2 \\int_a^R \\frac{r^2}{\\rho c^2} p q\\,dr
          = \\left[\\frac{r^2}{\\rho}\\frac{\\partial p}{\\partial r} q\\right]_a^R,

so the factor ``4 pi r^2`` of the 3-D volume element is carried exactly (this is
the exact 1-D reduction of the 3-D Helmholtz operator for a spherically
symmetric field, not a 1-D surrogate).

* Inner surface: the structural velocity enters as a *Neumann* condition
  ``dp/dr|_a = i omega rho v0`` (normal acceleration continuity
  ``rho_f d^2 u_n/dt^2 = -dp/dn``), i.e. the fluid loading of the structure is
  the domain solution and not a bookkeeping term.
* Outer truncation: the **exact** first-order radiation condition for a
  spherical outgoing wave, ``dp/dr|_R = (ik - 1/R) p(R)``.  This is an exact
  absorbing condition (Bayliss-Turkel order 1 is exact for the ``l = 0``
  harmonic), so the only error in the scheme is radial discretisation -- which
  is what makes the closed-form comparison a *code* verification rather than a
  model-agreement exercise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

RHO_WATER = 1000.0
C_WATER = 1500.0

# 8-point Gauss-Legendre rule on [-1, 1] (exact to degree 15).
_GL_X = (
    -0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
    -0.1834346424956498, 0.1834346424956498, 0.5255324099163290,
    0.7966664774136267, 0.9602898564975363,
)
_GL_W = (
    0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
    0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
    0.2223810344533745, 0.1012285362903763,
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


# ---------------------------------------------------------------------------
# Exact closed-form monopole
# ---------------------------------------------------------------------------

def wavenumber(frequency: float, c: float = C_WATER) -> float:
    """Acoustic wavenumber ``k = 2 pi f / c`` in 1/m."""
    return 2.0 * math.pi * _positive("frequency", frequency) / _positive("c", c)


def surface_impedance(
    radius: float, frequency: float, rho: float = RHO_WATER, c: float = C_WATER
) -> complex:
    """Exact specific acoustic impedance ``p(a)/v0`` of a pulsating sphere.

    ``Z = rho c (ka)(ka - i)/(1 + (ka)^2)``; ``Re Z -> rho c (ka)^2`` for
    ``ka -> 0`` (inertia limited) and ``Re Z -> rho c`` for ``ka -> inf``
    (plane wave limit).
    """
    a = _positive("radius", radius)
    ka = wavenumber(frequency, c) * a
    z0 = _positive("rho", rho) * _positive("c", c)
    return z0 * (ka * (ka - 1.0j)) / (1.0 + ka * ka)


def pressure(
    r,
    radius: float,
    frequency: float,
    v0: float,
    rho: float = RHO_WATER,
    c: float = C_WATER,
):
    """Exact monopole pressure field ``p(r)`` for a pulsating sphere.

    ``p(r) = rho c v0 (a/r) exp(ik(r-a)) / (1 + i/(ka))``.  Accepts a float or
    a ``torch.Tensor`` of radii and returns complex128.
    """
    a = _positive("radius", radius)
    k = wavenumber(frequency, c)
    ka = k * a
    amp = _positive("rho", rho) * _positive("c", c) * complex(v0)
    rr = torch.as_tensor(r, dtype=torch.float64)
    if torch.any(rr <= 0.0):
        raise ValueError("radii must be positive")
    scale = amp / (1.0 + 1.0j / ka)
    field = scale * (a / rr) * torch.exp(1.0j * k * (rr - a))
    return field if field.ndim else complex(field)


def surface_pressure(
    radius: float, frequency: float, v0: float,
    rho: float = RHO_WATER, c: float = C_WATER,
) -> complex:
    """Exact surface pressure ``p(a) = Z v0`` (compare :func:`surface_impedance`)."""
    return surface_impedance(radius, frequency, rho, c) * complex(v0)


def radiated_power(
    radius: float, frequency: float, v0: float,
    rho: float = RHO_WATER, c: float = C_WATER,
) -> float:
    """Exact time-averaged radiated power ``W = 1/2 Re(Z) 4 pi a^2 |v0|^2``."""
    a = _positive("radius", radius)
    z = surface_impedance(a, frequency, rho, c)
    return 0.5 * 4.0 * math.pi * a * a * z.real * abs(complex(v0)) ** 2


def plane_wave_pressure(
    r,
    radius: float,
    frequency: float,
    v0: float,
    rho: float = RHO_WATER,
    c: float = C_WATER,
):
    """Node-specification form ``rho c v0 (a/r) exp(ik(r-a))`` (``ka >> 1`` limit)."""
    a = _positive("radius", radius)
    k = wavenumber(frequency, c)
    amp = _positive("rho", rho) * _positive("c", c) * complex(v0)
    rr = torch.as_tensor(r, dtype=torch.float64)
    if torch.any(rr <= 0.0):
        raise ValueError("radii must be positive")
    field = amp * (a / rr) * torch.exp(1.0j * k * (rr - a))
    return field if field.ndim else complex(field)


def plane_wave_power(
    radius: float, v0: float, rho: float = RHO_WATER, c: float = C_WATER
) -> float:
    """Node-specification power ``2 pi a^2 rho c v0^2`` (``ka >> 1`` limit)."""
    a = _positive("radius", radius)
    return 2.0 * math.pi * a * a * _positive("rho", rho) * _positive("c", c) * abs(
        complex(v0)
    ) ** 2


def plane_wave_form_relative_error(ka: float) -> float:
    """Relative error of the node's pressure form to the exact monopole.

    The node form equals the exact field times ``(1 + i/(ka))``, hence
    ``|p_node - p_exact| / |p_exact| = 1/(ka)``.  It stays above 3 % until
    ``ka > 33.3``, which is why the node form is verified in its asymptotic
    domain (50 m sphere, ``ka = 10.5 ... 104.7``) and the exact form is used at
    the engineering frequencies of the campaign.
    """
    ka = _positive("ka", ka)
    return 1.0 / ka


def plane_wave_power_relative_error(ka: float) -> float:
    """Relative error of the node's power form ``2 pi a^2 rho c v0^2``: ``1/(ka)^2``."""
    ka = _positive("ka", ka)
    return 1.0 / (ka * ka)


# ---------------------------------------------------------------------------
# Radial finite element mesh
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RadialMesh:
    """Uniform 3-node quadratic mesh on ``[inner, outer]`` (``2*n + 1`` nodes)."""

    nodes: torch.Tensor
    inner: float
    outer: float
    elements: int

    @classmethod
    def uniform(cls, inner: float, outer: float, elements: int) -> "RadialMesh":
        inner = _positive("inner", inner)
        outer = _positive("outer", outer)
        if outer <= inner:
            raise ValueError("outer radius must exceed inner radius")
        n = _count("elements", elements)
        nodes = torch.linspace(inner, outer, 2 * n + 1, dtype=torch.float64)
        return cls(nodes=nodes, inner=inner, outer=outer, elements=n)

    def element_nodes(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.elements:
            raise IndexError("element index out of range")
        return self.nodes[2 * index : 2 * index + 3]


def _shape(xi: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return (
        0.5 * xi * (xi - 1.0),
        1.0 - xi * xi,
        0.5 * xi * (xi + 1.0),
    )


def _dshape(xi: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return (xi - 0.5, -2.0 * xi, xi + 0.5)


def _quadrature() -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(_GL_X, dtype=torch.float64),
        torch.tensor(_GL_W, dtype=torch.float64),
    )


def assemble_radial_helmholtz(
    mesh: RadialMesh,
    frequency: float,
    rho: float = RHO_WATER,
    c: float = C_WATER,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assemble the complex radial Helmholtz operator and inner Neumann load.

    Returns ``(A, b)`` with ``A`` of shape ``(n_nodes, n_nodes)`` complex128 and
    ``b`` the unit-velocity load vector (``v0 = 1``); the actual load is
    ``b * v0``.  ``A`` already contains the exact outer radiation condition
    ``dp/dr = (ik - 1/R) p``.
    """
    rho = _positive("rho", rho)
    c = _positive("c", c)
    k = wavenumber(frequency, c)
    omega = 2.0 * math.pi * _positive("frequency", frequency)
    xi, w = _quadrature()
    n_nodes = mesh.nodes.numel()
    stiff = torch.zeros((n_nodes, n_nodes), dtype=torch.float64)
    mass = torch.zeros((n_nodes, n_nodes), dtype=torch.float64)
    for e in range(mesh.elements):
        re = mesh.element_nodes(e)
        n = _shape(xi)
        dn = _dshape(xi)
        radius = sum(n_i * r_i for n_i, r_i in zip(n, re))
        dr_dxi = sum(dn_i * r_i for dn_i, r_i in zip(dn, re))
        ww = w * radius * radius / rho * dr_dxi
        stiff_e = torch.zeros((3, 3), dtype=torch.float64)
        mass_e = torch.zeros((3, 3), dtype=torch.float64)
        for i in range(3):
            for j in range(3):
                integrand = ww * dn[i] * dn[j] / (dr_dxi * dr_dxi)
                stiff_e[i, j] = torch.sum(integrand)
                mass_e[i, j] = torch.sum(ww * n[i] * n[j])
        idx = torch.arange(2 * e, 2 * e + 3)
        stiff[idx[:, None], idx[None, :]] += stiff_e
        mass[idx[:, None], idx[None, :]] += mass_e
    scale = 4.0 * math.pi
    operator = (stiff - omega * omega / (c * c) * mass).to(torch.complex128) * scale
    big_r = mesh.outer
    radiation = (4.0 * math.pi * big_r * big_r / rho) * complex(1.0j * k - 1.0 / big_r)
    operator[-1, -1] -= radiation
    load = torch.zeros(n_nodes, dtype=torch.complex128)
    load[0] = -scale * mesh.inner * mesh.inner * 1.0j * omega
    return operator, load


@dataclass(frozen=True)
class PulsatingSphereSolution:
    """Result of the radial FEM solve plus the two independent power routes."""

    nodes: torch.Tensor
    pressure: torch.Tensor
    frequency: float
    radius: float
    v0: float
    elements: int
    k: float

    @property
    def surface_pressure(self) -> complex:
        return complex(self.pressure[0])

    @property
    def impedance(self) -> complex:
        """FEM specific impedance ``p(a)/v0``."""
        return self.surface_pressure / complex(self.v0)

    @property
    def power_inner_surface(self) -> float:
        """Radiated power from the inner-surface flux ``1/2 Re(p(a) v0*) A``."""
        return 0.5 * 4.0 * math.pi * self.radius**2 * (
            self.surface_pressure * complex(self.v0).conjugate()
        ).real

    def flux_profile(self, rho: float = RHO_WATER) -> Tuple[torch.Tensor, torch.Tensor]:
        """Radiated power evaluated at every element Gauss point.

        In a source-free spherically symmetric acoustic field the radial energy
        flux ``4 pi r^2 (1/2) Re(p v_r*)`` is *constant* in ``r``, so the spread
        of this profile is a direct conservation diagnostic of the discretised
        field.  Evaluating at Gauss points (not nodes) avoids the derivative
        loss at element ends.
        """
        omega = 2.0 * math.pi * self.frequency
        xi, _ = _quadrature()
        radii = []
        powers = []
        for e in range(self.elements):
            re = self.nodes[2 * e : 2 * e + 3]
            pe = self.pressure[2 * e : 2 * e + 3]
            n = _shape(xi)
            dn = _dshape(xi)
            radius = sum(n_i * r_i for n_i, r_i in zip(n, re))
            dr_dxi = sum(dn_i * r_i for dn_i, r_i in zip(dn, re))
            p_g = sum(n_i * p for n_i, p in zip(n, pe))
            dxi = sum(dn_i * p for dn_i, p in zip(dn, pe))
            dp_dr = dxi / dr_dxi
            v_r = dp_dr / complex(1.0j * omega * rho)
            power = 0.5 * 4.0 * math.pi * radius * radius * (p_g * torch.conj(v_r)).real
            radii.append(radius.detach())
            powers.append(power.detach())
        return torch.cat(radii), torch.cat(powers)

    @property
    def power_outer_surface(self) -> float:
        """Radiated power from the outermost Gauss point of the last element."""
        radii, powers = self.flux_profile()
        return float(powers[-1])

    @property
    def flux_spread(self) -> float:
        """``(max - min)/mean`` of the radial flux profile (conservation error)."""
        _, powers = self.flux_profile()
        return float((powers.max() - powers.min()) / powers.mean())

    def relative_error(self, rho: float = RHO_WATER, c: float = C_WATER) -> torch.Tensor:
        """Element-wise ``|p_fem - p_exact| / |p_exact|`` over all nodes."""
        exact = pressure(self.nodes, self.radius, self.frequency, self.v0, rho, c)
        return torch.abs(self.pressure - exact) / torch.abs(exact)


def solve_pulsating_sphere(
    frequency: float,
    radius: float = 1.0,
    v0: float = 1.0,
    truncation_ratio: float = 10.0,
    elements: int = 16,
    rho: float = RHO_WATER,
    c: float = C_WATER,
) -> PulsatingSphereSolution:
    """Solve the exterior monopole problem on ``[a, R]`` with ``R = ratio * a``."""
    a = _positive("radius", radius)
    ratio = _positive("truncation_ratio", truncation_ratio)
    if ratio <= 1.0:
        raise ValueError("truncation_ratio must exceed 1")
    mesh = RadialMesh.uniform(a, ratio * a, elements)
    operator, load = assemble_radial_helmholtz(mesh, frequency, rho, c)
    field = torch.linalg.solve(operator, load * complex(v0))
    return PulsatingSphereSolution(
        nodes=mesh.nodes,
        pressure=field,
        frequency=float(frequency),
        radius=a,
        v0=float(v0),
        elements=int(elements),
        k=wavenumber(frequency, c),
    )


def convergence_study(
    frequency: float,
    radius: float = 1.0,
    element_counts: Tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    truncation_ratio: float = 10.0,
    v0: float = 1.0,
    rho: float = RHO_WATER,
    c: float = C_WATER,
) -> list[dict]:
    """Radial h-refinement study; returns max/mean nodal error per mesh."""
    rows = []
    for n in element_counts:
        sol = solve_pulsating_sphere(
            frequency, radius, v0, truncation_ratio, n, rho, c
        )
        err = sol.relative_error(rho, c)
        rows.append(
            {
                "elements": int(n),
                "nodes": int(sol.nodes.numel()),
                "max_relative_error": float(err.max()),
                "mean_relative_error": float(err.mean()),
                "power_inner": sol.power_inner_surface,
                "power_outer": sol.power_outer_surface,
            }
        )
    return rows


def radiation_efficiency(
    radius: float, frequency: float, c: float = C_WATER
) -> float:
    """Monopole radiation efficiency ``sigma = (ka)^2/(1 + (ka)^2)``.

    Defined by ``W = rho c S <v^2> sigma`` with ``S = 4 pi a^2`` and
    ``<v^2>`` the time-averaged mean-square surface velocity.
    """
    ka = wavenumber(frequency, c) * _positive("radius", radius)
    return ka * ka / (1.0 + ka * ka)