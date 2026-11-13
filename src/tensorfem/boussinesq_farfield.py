"""Closed-form Boussinesq far field for truncated elastic half-space models.

A contact/indentation model that replaces the elastic half-space by a finite
patch must cut the far field somewhere, and the usual choice (clamped bottom,
clamped outer radius) makes the patch *stiffer* than the half-space: the
material that the exact solution would let move at the cut is held in place.
For the axisymmetric Hertz problem this is a ``O(a/D)`` compliance deficit in
the indentation (measured: ``-1.01%`` at ``D = 96 a``, still ``-0.48%`` at
``D = 768 a``), whose very slow ``1/D`` tail is what makes Richardson
extrapolation on that ladder unreliable.

This module supplies the other way out: the *exact* half-space displacement
produced by the computed surface traction, so the truncated boundary can
carry absorbing Dirichlet data instead of zero data.  The kernel is the
classical Boussinesq solution for a vertical point load on a homogeneous,
isotropic half-space (Poulos & Davis 1974, *Elastic Solutions for Soil and Rock
Mechanics*, §3.2; Love 1944; Johnson 1985 §2.2)::

    u_r = P/(4 pi G) [ r z / R^3 - (1 - 2 nu) r / (R (R + z)) ]
    u_z = P/(4 pi G) [ z^2 / R^3 + 2 (1 - nu) / R ],      R = sqrt(r^2 + z^2)

with ``z >= 0`` measured *into* the body and ``u_z`` positive *downwards*.
Note the two conventions of this repository: FEM nodes live at ``z <= 0``
(surface at ``z = 0``) and a downward displacement is a *negative* ``u_z``,
so callers map ``u_z^FEM = -u_z^Boussinesq``.

All routines are axisymmetric: the traction is a set of ring forces (one per
surface node of a radial mesh), and the field point is ``(r, depth)``.  The
$\\theta$ integral is evaluated with Gauss-Legendre, the sign convention of the
horizontal projection is ``(r - a cos theta)/s`` (the unit vector from the
load point to the field point, projected on the radial basis vector at the
field point).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "theta_quadrature",
    "point_load_displacement",
    "ring_load_displacement",
    "nodal_force_displacement",
    "uniform_disc_displacement",
    "disc_axis_settlement",
    "disc_center_settlement",
    "disc_edge_settlement",
    "effective_compliance",
]


def theta_quadrature(n_theta: int = 32):
    """Nodes/weights on ``[0, 2 pi]`` (the integrand is ``theta -> 2 pi - theta``
    symmetric, so half the interval is integrated and doubled)."""
    x, w = np.polynomial.legendre.leggauss(int(n_theta))
    theta = 0.5 * math.pi * (x + 1.0)
    weights = 0.5 * math.pi * w * 2.0
    return theta, weights


def effective_compliance(young: float, poisson: float) -> float:
    """``1/(4 pi G)`` with ``G = E/(2(1+nu))`` -- the single scalar that scales
    every Boussinesq displacement per unit load."""
    if young <= 0:
        raise ValueError("young must be positive")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("poisson must lie in (-1, 1/2)")
    shear = young / (2.0 * (1.0 + poisson))
    return 1.0 / (4.0 * math.pi * shear)


def point_load_displacement(radii, depths, load=1.0, young=1.0, poisson=0.3):
    """Closed-form displacement at ``(r, depth)`` due to a vertical surface point
    load at the axis.  Returns ``(u_r, u_z_down)``.

    Validated against the two textbook surface values
    ``u_z(r,0) = (1-nu^2) P/(pi E r)`` and ``u_r(r,0) = -(1-2nu)(1+nu) P/(2 pi E r)``.
    """
    r = np.atleast_1d(np.asarray(radii, dtype=float))
    d = np.atleast_1d(np.asarray(depths, dtype=float))
    if np.any(d < 0.0):
        raise ValueError("depth is measured into the body and must be >= 0")
    c = effective_compliance(young, poisson) * float(load)
    radius = np.sqrt(r * r + d * d)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_r = c * (r * d / radius**3 - (1.0 - 2.0 * poisson) * r / (radius * (radius + d)))
        u_z = c * (d * d / radius**3 + 2.0 * (1.0 - poisson) / radius)
    singular = (radius == 0.0)
    if np.any(singular):
        u_r = np.where(singular, 0.0, u_r)
        u_z = np.where(singular, np.inf, u_z)
    return u_r, u_z


def ring_load_displacement(
    radii,
    depths,
    ring_radius: float,
    ring_force: float,
    young: float = 1.0,
    poisson: float = 0.3,
    n_theta: int = 32,
):
    """Displacement from a total force ``ring_force`` spread uniformly over the
    ring ``r' = ring_radius`` on the surface (axisymmetric quadrature)."""
    r = np.atleast_1d(np.asarray(radii, dtype=float))
    d = np.atleast_1d(np.asarray(depths, dtype=float))
    if np.any(d < 0.0):
        raise ValueError("depth is measured into the body and must be >= 0")
    if ring_radius < 0.0:
        raise ValueError("ring_radius must be >= 0")
    theta, weights = theta_quadrature(n_theta)
    c = effective_compliance(young, poisson) * float(ring_force) / (2.0 * math.pi)
    rr = r[:, None]
    dd = d[:, None]
    cos_t = np.cos(theta)[None, :]
    s2 = rr * rr + ring_radius * ring_radius - 2.0 * rr * ring_radius * cos_t
    s2 = np.maximum(s2, 0.0)
    big_r = np.sqrt(s2 + dd * dd)
    if np.any(big_r == 0.0):
        raise ValueError("field point coincides with the ring (singular kernel)")
    with np.errstate(divide="ignore", invalid="ignore"):
        u_r = c * np.sum(
            (dd / big_r**3 - (1.0 - 2.0 * poisson) / (big_r * (big_r + dd)))
            * (rr - ring_radius * cos_t)
            * weights,
            axis=1,
        )
        u_z = c * np.sum(
            (dd * dd / big_r**3 + 2.0 * (1.0 - poisson) / big_r) * weights, axis=1
        )
    return u_r, u_z


def nodal_force_displacement(
    field_radii,
    field_depths,
    ring_radii,
    ring_forces,
    young: float = 1.0,
    poisson: float = 0.3,
    n_theta: int = 32,
):
    """Superpose the ring forces of an axisymmetric surface traction.

    ``ring_radii`` / ``ring_forces`` come from the FEM contact state (the nodal
    contact forces ``lam_i`` of a radial surface mesh; each one is the resultant
    of its ring).  Returns ``(u_r, u_z_down)`` at the field points.
    """
    ring_radii = np.asarray(ring_radii, dtype=float)
    ring_forces = np.asarray(ring_forces, dtype=float)
    if ring_radii.shape != ring_forces.shape:
        raise ValueError("ring_radii and ring_forces must have the same shape")
    field_radii = np.atleast_1d(np.asarray(field_radii, dtype=float))
    field_depths = np.atleast_1d(np.asarray(field_depths, dtype=float))
    u_r = np.zeros(field_radii.shape)
    u_z = np.zeros(field_radii.shape)
    for radius, force in zip(ring_radii, ring_forces):
        if force == 0.0:
            continue
        du_r, du_z = ring_load_displacement(
            field_radii,
            field_depths,
            float(radius),
            float(force),
            young=young,
            poisson=poisson,
            n_theta=n_theta,
        )
        u_r += du_r
        u_z += du_z
    return u_r, u_z


def uniform_disc_displacement(
    radii,
    depths,
    radius: float,
    pressure: float,
    young: float = 1.0,
    poisson: float = 0.3,
    n_r: int = 64,
    n_theta: int = 32,
):
    """Displacement under a *uniform* pressure on a disc of radius ``radius``
    (Gauss-Legendre in the radial direction, azimuthal quadrature per ring)."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    x, w = np.polynomial.legendre.leggauss(int(n_r))
    ring_radius = 0.5 * radius * (x + 1.0)
    ring_force = float(pressure) * 2.0 * math.pi * ring_radius * (0.5 * radius * w)
    return nodal_force_displacement(
        radii,
        depths,
        ring_radius,
        ring_force,
        young=young,
        poisson=poisson,
        n_theta=n_theta,
    )


# ---------------------------------------------------------------------------
# Closed forms used as references (nodal-force quadrature must reproduce them)
# ---------------------------------------------------------------------------


def disc_axis_settlement(radius: float, pressure: float, depth: float, young: float, poisson: float):
    """Settlement on the axis beneath a uniformly loaded disc (closed form).

    Integrating the Boussinesq point-load kernel over the disc gives

    ``u_z(0,z) = p (1+nu)/E [ z - z^2/S + 2 (1-nu) (S - z) ]``,  ``S = sqrt(a^2+z^2)``

    which reduces to ``2 (1-nu^2) p a/E`` at ``z = 0`` (Boussinesq 1885).  Note
    the widely quoted ``(p/E*)(S - z)`` is the *incremental* form valid only up
    to a factor: at ``z = 0`` it gives half the correct centre settlement.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    s = math.sqrt(radius * radius + depth * depth)
    return (
        float(pressure)
        * (1.0 + poisson)
        / young
        * (depth - depth * depth / s + 2.0 * (1.0 - poisson) * (s - depth))
    )


def disc_center_settlement(radius: float, pressure: float, young: float, poisson: float):
    """``u_z(0,0) = 2 (1 - nu^2) p a / E`` (Boussinesq 1885)."""
    return 2.0 * (1.0 - poisson**2) * float(pressure) * radius / young


def disc_edge_settlement(radius: float, pressure: float, young: float, poisson: float):
    """``u_z(a,0) = 4 (1 - nu^2) p a / (pi E)`` (uniform disc, edge)."""
    return 4.0 * (1.0 - poisson**2) * float(pressure) * radius / (math.pi * young)