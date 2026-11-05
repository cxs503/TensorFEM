"""Analytic baselines and the benchmark model for the HEX20 hemisphere oracle.

Every reference below is closed form, so the oracle is certified before it is
used to judge the shell element:

* ``lame_ring_validation`` -- plane-strain Lamé ring with a prescribed inner
  radial displacement and a traction-free outer surface.  Exercises curved
  geometry, assembly and stress recovery against exact elasticity.
* ``thin_strip_bending_validation`` -- plane-strain Timoshenko cantilever strip
  at the aspect ratio of the hemisphere benchmark, which quantifies the
  through-thickness bending accuracy the oracle can deliver on a thin shell.
* ``hemisphere_hex20_model`` -- the MacNeal--Harder quarter model itself.
"""
from __future__ import annotations

import math

import torch

from .hemisphere_hex20_oracle import (MH_HOLE_POLAR_DEGREES, MH_POISSON, MH_RADIUS,
                                      MH_THICKNESS, MH_YOUNG, hex20_parametric_centre,
                                      rotate_ring_stress, solve_shell_solid_hex20,
                                      shell_solid_hex20_mesh)

MU = lambda young, poisson: young / (2 * (1 + poisson))
LAMBDA = lambda young, poisson: young * poisson / ((1 + poisson) * (1 - 2 * poisson))


# --------------------------------------------------------------------------
# 1. exact Lamé ring: oracle baseline on curved (but axisymmetric) geometry
# --------------------------------------------------------------------------
def lame_ring_model(nphi: int, ntheta: int, nz: int, *, radius=MH_RADIUS,
                    thickness=MH_THICKNESS, young=MH_YOUNG, poisson=MH_POISSON,
                    inner_displacement=1e-3):
    """Plane-strain Lamé ring with a prescribed radial displacement on the bore.

    Every node is restrained axially, so the exact solution is the plane-strain
    Lamé field ``u_r = A r + B/r`` with a traction-free outer surface.  Because
    ``A`` and ``B`` are set by ``u_r(a) = delta`` and ``sigma_r(b) = 0``, all
    three stress components are known in closed form at every element centroid.
    """
    nodes, elements = shell_solid_hex20_mesh(
        nphi, ntheta, nz, radius=radius, thickness=thickness, cylinder=True,
        phi_start=0.0, phi_end=1.0)
    ndof = 3 * len(nodes)
    forces = torch.zeros(ndof, dtype=nodes.dtype)
    a = radius - 0.5 * thickness
    polar = torch.linalg.vector_norm(nodes[:, :2], dim=1)
    bore = torch.where((polar - a).abs() < 1e-9)[0]
    if not len(bore):
        raise ValueError("bore surface nodes were not found")
    theta = torch.atan2(nodes[bore, 1], nodes[bore, 0])
    fixed, values = [], []
    for n, node in enumerate(bore.tolist()):
        fixed.extend((3 * node + 0, 3 * node + 1))
        values.extend((inner_displacement * math.cos(float(theta[n])),
                       inner_displacement * math.sin(float(theta[n]))))
    # circumferential symmetry: the two theta-cut planes must not separate, so the
    # normal displacement vanishes there (u_y on theta=0, u_x on theta=90).
    for node in torch.where(nodes[:, 1].abs() < 1e-9)[0].tolist():
        fixed.append(3 * node + 1)
        values.append(0.0)
    for node in torch.where(nodes[:, 0].abs() < 1e-9)[0].tolist():
        fixed.append(3 * node + 0)
        values.append(0.0)
    for node in range(len(nodes)):
        fixed.append(3 * node + 2)
        values.append(0.0)
    order = sorted(range(len(fixed)), key=lambda t: fixed[t])
    fixed = torch.tensor([fixed[t] for t in order], dtype=torch.long)
    values = torch.tensor([values[t] for t in order], dtype=nodes.dtype)
    result = solve_shell_solid_hex20(nodes, elements, young=young, poisson=poisson,
                                     forces=forces, fixed_dofs=fixed, prescribed_values=values)
    centroid = hex20_parametric_centre(nodes, elements)
    r = torch.linalg.vector_norm(centroid[:, :2], dim=1)
    coefficients = lame_ring_coefficients(a, radius + 0.5 * thickness, young, poisson,
                                          inner_displacement)
    exact = torch.stack([torch.tensor(lame_ring_stress(float(rho), *coefficients, young, poisson))
                         for rho in r.unbind()])
    theta = torch.atan2(centroid[:, 1], centroid[:, 0])
    local = rotate_ring_stress(result.stress, theta)
    fe = torch.stack((local[:, 0], local[:, 1], local[:, 2]), 1)
    # the axial ends of the ring are traction free, so a three-dimensional
    # boundary layer lives in the outermost element layers; exclude it exactly
    axial = centroid[:, 2]
    span = float(axial.max() - axial.min())
    middle = (axial - axial.min()).abs().gt(0.25 * span) & (axial - axial.min()).abs().lt(0.75 * span)
    if not bool(middle.any()):
        raise ValueError("no element centroid lies in the axial middle half")
    fe, exact, r = fe[middle], exact[middle], r[middle]
    scale = float(exact[:, 1].abs().max())
    errors = {name: float((fe[:, i] - exact[:, i]).abs().max()) / scale
              for i, name in enumerate(("radial", "hoop", "axial"))}
    return {
        "case": "lame_ring_plane_strain",
        "mesh": {"nphi": nphi, "ntheta": ntheta, "nz": nz, "nodes": len(nodes),
                 "elements": len(elements)},
        "max_relative_error": errors,
        "hoop_reference_max": scale,
        "equilibrium": result.equilibrium(),
    }


def lame_ring_coefficients(a: float, b: float, young: float, poisson: float,
                           inner_displacement: float):
    """Return ``(A, B)`` of ``u_r = A r + B/r`` for the Lamé ring."""
    mu, lam = MU(young, poisson), LAMBDA(young, poisson)
    c = (lam + mu) / mu
    A = inner_displacement / (a + c * b * b / a)
    return A, c * A * b * b


def lame_ring_stress(r: float, A: float, B: float, young: float, poisson: float):
    """Exact plane-strain stresses ``(sigma_r, sigma_theta, sigma_z)``."""
    mu, lam = MU(young, poisson), LAMBDA(young, poisson)
    hoop = 2 * (lam + mu) * A + 2 * mu * B / (r * r)
    radial = 2 * (lam + mu) * A - 2 * mu * B / (r * r)
    return (radial, hoop, 2 * lam * A)


# --------------------------------------------------------------------------
# 2. thin strip bending: bending accuracy at the benchmark aspect ratio
# --------------------------------------------------------------------------
def thin_strip_bending_validation(nx: int = 24, ny: int = 2, nz: int = 2, *,
                                  length=4.0, width=1.0, thickness=MH_THICKNESS,
                                  tip_load=-1e-2, young=MH_YOUNG, poisson=MH_POISSON):
    """Cantilever strip whose element size / thickness matches the hemisphere mesh.

    The returned error is the through-thickness bending penalty that the oracle
    carries into the shell benchmark, so it is an explicit accuracy budget rather
    than an assumption.  Reference: Euler--Bernoulli with the plane-strain
    modulus ``E/(1-nu^2)`` (``width/thickness = 25`` makes the strip a plate).
    """
    from .hex20 import structured_hex20_mesh

    nodes, elements = structured_hex20_mesh(length, width, thickness, nx, ny, nz)
    ndof = 3 * len(nodes)
    forces = torch.zeros(ndof, dtype=nodes.dtype)
    x = nodes[:, 0]
    tip = torch.where(x > length * (1 - 1e-9))[0]
    forces[3 * tip + 2] = tip_load / len(tip)
    root = torch.where(x.abs() < 1e-12)[0]
    fixed = torch.stack((3 * root, 3 * root + 1, 3 * root + 2), 1).reshape(-1)
    # restrain the out-of-plane displacement everywhere: the strip is a plate, so
    # the Euler--Bernoulli reference must use the plane-strain modulus E/(1-nu^2).
    plane = 3 * torch.arange(len(nodes)) + 1
    result = solve_shell_solid_hex20(nodes, elements, young=young, poisson=poisson,
                                     forces=forces,
                                     fixed_dofs=torch.unique(torch.cat((fixed, plane))))
    computed = float(result.displacement[3 * tip + 2].mean())
    inertia = width * thickness ** 3 / 12
    reference = tip_load * length ** 3 / (3 * (young / (1 - poisson ** 2)) * inertia)
    return {
        "case": "thin_strip_bending",
        "mesh": {"nx": nx, "ny": ny, "nz": nz, "nodes": len(nodes), "elements": len(elements),
                 "element_size_over_thickness": (length / nx) / thickness},
        "tip_deflection": computed,
        "euler_bernoulli_plane_strain": reference,
        "relative_error": abs(computed / reference - 1.0),
        "equilibrium": result.equilibrium(),
    }


# --------------------------------------------------------------------------
# 3. the MacNeal--Harder quarter model
# --------------------------------------------------------------------------
def hemisphere_hex20_model(nphi: int, ntheta: int, nz: int, *,
                           radius=MH_RADIUS, thickness=MH_THICKNESS,
                           young=MH_YOUNG, poisson=MH_POISSON,
                           hole_degrees=MH_HOLE_POLAR_DEGREES, dtype=torch.float64):
    """Quarter model with the benchmark symmetry conditions and unit corner loads.

    Symmetry planes: normal translation restrained on ``theta=0`` (``u_y=0``) and
    ``theta=90`` (``u_x=0``); one vertical DOF is the rigid-translation gauge.
    The two equator corner loads are unit radial forces with opposite sign, split
    equally over the ``2*nz+1`` through-thickness nodes so that no spurious
    through-thickness moment is introduced.
    """
    nodes, elements = shell_solid_hex20_mesh(
        nphi, ntheta, nz, radius=radius, thickness=thickness,
        phi_start=math.radians(hole_degrees), phi_end=math.pi / 2, dtype=dtype)
    ndof = 3 * len(nodes)
    forces = torch.zeros(ndof, dtype=dtype)
    scale = max(float(torch.linalg.vector_norm(nodes, dim=1).max()), 1.0)
    tol = 1e-9 * scale
    x, y, z = nodes.unbind(1)
    polar = torch.linalg.vector_norm(nodes[:, :2], dim=1)
    sym_zero = torch.where(y.abs() < tol)[0]        # theta = 0 symmetry plane
    sym_ninety = torch.where(x.abs() < tol)[0]     # theta = 90 symmetry plane
    fixed, values = [], []
    for node in sym_zero.tolist():
        fixed.append(3 * node + 1)
        values.append(0.0)
    for node in sym_ninety.tolist():
        fixed.append(3 * node + 0)
        values.append(0.0)
    equator = torch.where(z.abs() < tol)[0]
    corner_a = equator[y[equator].abs() < tol]
    corner_b = equator[x[equator].abs() < tol]
    if not len(corner_a) or not len(corner_b):
        raise ValueError("equator corner nodes were not found")
    for node in corner_a.tolist():
        forces[3 * node + 0] += 1.0 / len(corner_a)          # +x radial at theta=0
    for node in corner_b.tolist():
        forces[3 * node + 1] += -1.0 / len(corner_b)         # -y radial at theta=90
    mid = torch.where((z.abs() < tol) & (y.abs() < tol) &
                      ((polar - radius).abs() < tol))[0]
    if len(mid) != 1:
        raise ValueError("mid-surface equator probe node is not unique")
    probe = int(mid[0])
    fixed.append(3 * probe + 2)                              # vertical gauge
    values.append(0.0)
    order = sorted(range(len(fixed)), key=lambda t: fixed[t])
    fixed = torch.tensor([fixed[t] for t in order], dtype=torch.long)
    values = torch.tensor([values[t] for t in order], dtype=dtype)
    unique_fixed = torch.unique(fixed)
    result = solve_shell_solid_hex20(nodes, elements, young=young, poisson=poisson,
                                     forces=forces, fixed_dofs=unique_fixed,
                                     prescribed_values=_align(unique_fixed, fixed, values))
    return result, probe


def _align(unique_fixed, fixed, values):
    """Prescribed values ordered like the de-duplicated fixed DOF list."""
    lookup = {}
    for dof, value in zip(fixed.tolist(), values.tolist()):
        lookup.setdefault(dof, value)
    return torch.tensor([lookup[int(d)] for d in unique_fixed], dtype=values.dtype)
