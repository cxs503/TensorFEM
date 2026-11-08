"""Hertz elastic contact, Cattaneo--Mindlin partial slip and unilateral
complementarity verification kernels.

What this module owns
---------------------
1. **Closed-form oracles** (no FE): Hertz point contact (sphere--sphere,
   sphere--plane) and Hertz line contact (cylinder--plane) for the contact
   radius, peak pressure and approach; Cattaneo--Mindlin partial slip for the
   stick radius ``c/a``, the tangential force--displacement relation and the
   traction profile; and the elastic strain energy of the Hertzian state.
2. **Condensed FE half-space compliance**: an axisymmetric Q4 mesh (point
   contact) or a plane-strain Q4 mesh (line contact) is assembled once and
   statically condensed onto the contact-surface degrees of freedom.  The
   condensation is exact linear algebra on the assembled FE stiffness
   (``u_C = H f_C``), so no stiffness information is discarded; only the
   contact nonlinearity lives in the small condensed problem.
3. **Unilateral solvers** on the condensed problem: nodal penalty normal
   contact, augmented Lagrangian (ALM) normal contact with multiplier
   iteration, and incremental Cattaneo--Mindlin tangential contact (rigid
   stick set plus a monotonically growing Coulomb slip annulus).

Conventions (stated once, used everywhere)
------------------------------------------
* Body: ``r`` (or ``x``) ``>= 0``, top surface at ``z = 0``, depth ``-D``.
* ``u_z`` is positive **upward**.  Contact pressure ``p >= 0`` is compressive
  and therefore acts on the body as the nodal force ``f_z = -p``.
* Rigid indenter displaced downward by ``delta_R >= 0``.  For a spherical tip
  of radius ``R`` the free gap between the two surfaces is

      g_i = r_i^2/(2 R) - delta_R - u_z(r_i) = offset_i + (H_zz f)_i ,
      offset_i = r_i^2/(2 R) - delta_R ,

  where ``f_i >= 0`` is the compressive *nodal force* (``f_z = -f`` on the body)
  and ``H_zz`` the condensed normal compliance.  The unilateral condition is
  the linear complementarity problem

      0 <= f_i  _|_  g_i >= 0          (f >= 0, g >= 0, f . g = 0).

  Penalty contact permits ``g < 0`` (penetration); an admissible ALM solution
  returns ``g >= 0`` with ``f . g = 0``.  Physical pressure and traction are
  ``p_i = f_i / A_i`` with ``A_i`` the tributary measure, so the Coulomb
  condition ``|tau| <= mu p`` is ``|q_i| <= mu f_i`` on nodal forces.* Tangential
  ``q >= 0`` is the radial nodal force transferred from the indenter to the
  body and ``s`` the rigid tangential shift of the indenter; stick means
  ``u_r = s``.

Dimensionless scaling
---------------------
The compliance is assembled with unit modulus on a mesh whose lengths are
measured in units of the Hertz contact radius ``a_H`` (point contact) or
half-width ``b_H`` (line contact).  Linear elasticity fixes the scaling
exactly -- ``H_phys = H_dim/(E_fe a_H)`` for point contact and
``H_dim/E_fe`` for line contact -- so a single factorization per mesh level
serves every load level and every configuration.  The contact radius is only
used to *place* the mesh (standard a-priori grading); no Hertz formula enters
the equilibrium, the contact detection, or the recovery of ``a``/``p_0``,
which are read off the computed pressure field.

Limitations
-----------
* Small deformation, small sliding, normal-frictionless contact against a
  *rigid* indenter; the body is linearly elastic and isotropic.
* The Cattaneo solver assumes a monotone tangential loading path at constant
  normal load (the classical Cattaneo/Mindlin history).  Unloading, reverse
  slip and rolling paths are out of scope.
* ``c/a`` is a nodal-resolution estimate of the stick/slip boundary; the mesh
  convergence table is reported rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .hertz_axisymmetric import assemble_axisymmetric_stiffness

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# 1. closed-form oracles
# ---------------------------------------------------------------------------
def hertz_point(load: float, radius: float, e_star: float) -> dict[str, float]:
    """Hertz point contact: contact radius ``a``, peak pressure ``p_0``, approach.

    ``radius`` is the *effective* radius (``1/R = 1/R_1 + 1/R_2``) and
    ``e_star`` the effective modulus
    (``1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2``).
    """
    if load <= 0 or radius <= 0 or e_star <= 0:
        raise ValueError("Hertz parameters must be positive")
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    p0 = 3.0 * load / (2.0 * math.pi * a * a)
    delta = a * a / radius
    return {"a": a, "p0": p0, "delta": delta, "load": load, "radius": radius,
            "e_star": e_star}


def hertz_point_from_indentation(indentation: float, radius: float,
                                 e_star: float) -> dict[str, float]:
    """Inverse Hertz relation: prescribed approach -> load, ``a``, ``p_0``."""
    if indentation <= 0:
        raise ValueError("indentation must be positive")
    a = math.sqrt(radius * indentation)
    load = 4.0 * e_star * a ** 3 / (3.0 * radius)
    return hertz_point(load, radius, e_star)


def hertz_line(load_per_length: float, radius: float, e_star: float) -> dict[str, float]:
    """Hertz line contact (cylinder--plane): half-width ``b``, ``p_0``, approach.

    Johnson (1985) Sec. 4.2: ``b = sqrt(4 P R/(pi E*))``, ``p_0 = 2P/(pi b)``
    and the approach of distant points
    ``delta = (P/(pi E*)) (2 ln(4R/b) - 1)``.  The logarithm makes the
    line-contact approach sensitive to the assumed reference distance, so it
    is cross-checked by :func:`hertz_line_approach_quadrature`, which derives
    the same quantity from the Flamant kernel at a declared reference radius.
    """
    if load_per_length <= 0 or radius <= 0 or e_star <= 0:
        raise ValueError("Hertz line parameters must be positive")
    b = math.sqrt(4.0 * load_per_length * radius / (math.pi * e_star))
    p0 = 2.0 * load_per_length / (math.pi * b)
    delta = (load_per_length / (math.pi * e_star)) * (2.0 * math.log(4.0 * radius / b) - 1.0)
    return {"b": b, "p0": p0, "delta": delta, "load_per_length": load_per_length,
            "radius": radius, "e_star": e_star}


def hertz_line_approach_quadrature(load_per_length: float, radius: float,
                                   e_star: float, reference: float,
                                   samples: int = 40001) -> float:
    """Line-contact approach from the Flamant kernel (no log constant fitted).

    The plane-strain surface displacement of a half-space under a pressure
    ``p`` is ``w(x) = (2(1-nu^2)/(pi E)) int p(s) ln(1/|x-s|) ds`` plus a
    constant that is *independent of x* for a fixed pressure.  The approach
    between the contact centre and a surface point at distance ``reference``
    therefore needs no constant::

        delta(reference) = (2/(pi E*)) int_{-b}^{b} p(s) ln(|reference-s|/|s|) ds

    with ``2/(pi E*) = 2(1-nu^2)/(pi E)``.  Cosine clustering resolves the
    integrable ``ln|s|`` singularity at the contact centre.
    """
    hz = hertz_line(load_per_length, radius, e_star)
    b, p0 = hz["b"], hz["p0"]
    if reference <= b:
        raise ValueError("reference distance must exceed the contact half-width")
    theta = torch.linspace(0.0, math.pi, samples, dtype=DTYPE)
    s = b * torch.cos(theta)
    p = p0 * torch.sin(theta)                 # sqrt(1 - (s/b)^2) = sin(theta)
    jac = b * torch.sin(theta)                # ds = -b sin(theta) dtheta
    kernel = torch.log((reference - s).abs() / s.abs())
    value = torch.trapezoid(p * jac * kernel, theta)
    return float(2.0 * value / (math.pi * e_star))


def hertz_traction(r, a: float, p0: float) -> torch.Tensor:
    """Hertz pressure profile ``p(r) = p_0 sqrt(1 - (r/a)^2)`` on ``r <= a``."""
    ratio = torch.clamp(torch.as_tensor(r, dtype=DTYPE) / a, max=1.0)
    return p0 * torch.sqrt(torch.clamp(1.0 - ratio * ratio, min=0.0))


def hertz_work(load: float, delta: float) -> float:
    """Elastic strain energy of the Hertz point-contact state: ``(2/5) F delta``."""
    return 0.4 * load * delta


def hertz_line_work(load_per_length: float, delta: float) -> float:
    """Line-contact elastic energy ``(2/3) P delta``."""
    return (2.0 / 3.0) * load_per_length * delta


def _ellipk(m: torch.Tensor) -> torch.Tensor:
    """Complete elliptic integral of the first kind ``K(m)`` (parameter ``m = k^2``)."""
    a = torch.ones_like(m)
    b = torch.sqrt(torch.clamp(1.0 - m, min=0.0))
    for _ in range(80):
        a, b = 0.5 * (a + b), torch.sqrt(a * b)
    return math.pi / (2.0 * a)


def hertz_point_settlement(r, radius: float, load: float, e_star: float,
                           samples: int = 4001) -> torch.Tensor:
    """Surface settlement ``w(r)`` (positive downward) under a Hertz pressure.

    Boussinesq kernel quadrature over the loaded circle::

        w(r) = (1/(pi E*)) int_0^a p(s) s ds  int_0^{2 pi} dtheta / rho
             = (1/(pi E*)) int_0^a p(s) s ds  4 K(k) / (r + s),
        k^2 = 4 r s/(r + s)^2

    with ``K`` the complete elliptic integral (AGM).  The integrable
    ``ln|r - s|`` behaviour of ``K`` at ``s = r`` is resolved by dense
    sampling.  The closed form is used both as an independent oracle and as a
    reference-relative approach that does not depend on a logarithm convention.
    """
    hz = hertz_point(load, radius, e_star)
    a, p0 = hz["a"], hz["p0"]
    r = torch.atleast_1d(torch.as_tensor(r, dtype=DTYPE))
    s = torch.linspace(0.0, a, samples, dtype=DTYPE)
    pressure = p0 * torch.sqrt(torch.clamp(1.0 - (s / a) ** 2, min=0.0))
    out = torch.empty_like(r)
    for index, value in enumerate(r.tolist()):
        if value < 1e-14 * a:
            total = (2.0 / e_star) * torch.trapezoid(pressure, s)
        else:
            k2 = 4.0 * value * s / (value + s) ** 2
            integrand = pressure * s * 4.0 * _ellipk(k2) / (value + s)
            total = torch.trapezoid(integrand, s) / (math.pi * e_star)
        out[index] = total
    return out if out.numel() > 1 else out[0]


def hertz_point_approach(radius: float, load: float, e_star: float,
                         reference: float) -> float:
    """Reference-relative Hertz approach ``w(0) - w(reference)`` (positive).

    Because the settlement is evaluated from the same Boussinesq kernel at both
    points, no logarithm/constant convention enters: this is the quantity a
    displacement-based FE model with a clamped far field actually measures.
    """
    if reference <= 0.0:
        raise ValueError("reference distance must be positive")
    w0 = hertz_point_settlement(0.0, radius, load, e_star)
    wr = hertz_point_settlement(reference, radius, load, e_star)
    return float(w0 - wr)


def cattaneo_stick_radius(load_tangential: float, load_normal: float, mu: float,
                          a: float) -> float:
    """Stick radius ``c = a (1 - Q/(mu N))^{1/3}`` (Cattaneo 1938)."""
    if load_normal <= 0 or mu <= 0 or a <= 0:
        raise ValueError("Cattaneo parameters must be positive")
    ratio = load_tangential / (mu * load_normal)
    if ratio > 1.0 + 1e-12:
        raise ValueError("tangential load exceeds the friction limit")
    return a * max(0.0, 1.0 - ratio) ** (1.0 / 3.0)


def cattaneo_traction(r, a: float, c: float, p0: float, mu: float) -> torch.Tensor:
    """Cattaneo tangential traction: Coulomb in the annulus, relieved inside.

    ``q(r) = mu p_0 sqrt(1-(r/a)^2)`` for ``c <= r <= a``, and
    ``q(r) = mu p_0 [sqrt(1-(r/a)^2) - (c/a) sqrt(1-(r/c)^2)]`` for ``r < c``.
    """
    r = torch.as_tensor(r, dtype=DTYPE)
    outer = torch.sqrt(torch.clamp(1.0 - (r / a) ** 2, min=0.0))
    inner = torch.sqrt(torch.clamp(1.0 - (r / c) ** 2, min=0.0))
    q = mu * p0 * torch.where(r < c, outer - (c / a) * inner, outer)
    return torch.clamp(q, min=0.0)


def cattaneo_displacement(load_tangential: float, load_normal: float, mu: float,
                          a: float, shear_modulus: float, poisson: float) -> float:
    """Tangential approach of the contact centre (Cattaneo/Mindlin closed form).

    ``u_t = (3 mu N (2-nu)/(16 G a)) [1 - (1 - Q/(mu N))^{2/3}]``.

    The prefactor is pinned by the small-``Q`` limit, which must reproduce the
    classical tangential contact stiffness ``dQ/du_t = 8 G a/(2 - nu)``
    (Mindlin 1949).  :func:`cattaneo_stiffness_small_q` exposes that limit so
    the verification driver can audit the two independently.
    """
    ratio = load_tangential / (mu * load_normal)
    if ratio > 1.0 + 1e-12:
        raise ValueError("tangential load exceeds the friction limit")
    coef = 3.0 * mu * load_normal * (2.0 - poisson) / (16.0 * shear_modulus * a)
    return coef * (1.0 - max(0.0, 1.0 - ratio) ** (2.0 / 3.0))


def cattaneo_stiffness_small_q(load_normal: float, a: float,
                               shear_modulus: float, poisson: float) -> float:
    """``dQ/du_t`` at ``Q -> 0``: ``8 G a/(2 - nu)`` (Mindlin 1949)."""
    return 8.0 * shear_modulus * a / (2.0 - poisson)# ---------------------------------------------------------------------------
# 2. FE kernels and condensed compliance
# ---------------------------------------------------------------------------
def _constitutive_plane_strain(young: float, poisson: float) -> torch.Tensor:
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return torch.tensor(
        [[lam + 2.0 * mu, lam, 0.0],
         [lam, lam + 2.0 * mu, 0.0],
         [0.0, 0.0, mu]], dtype=DTYPE)


def plane_strain_mesh(nx: int, nz: int, width: float, depth: float, *,
                      grading: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured ``(x, z)`` mesh with ``x in [0, width]``, ``z in [-depth, 0]``.

    ``grading > 1`` clusters nodes towards ``x = 0`` and ``z = 0`` (the contact
    surface), matching :func:`tensorfem.hertz_axisymmetric.axisymmetric_mesh`.
    """
    if nx < 2 or nz < 2 or width <= 0.0 or depth <= 0.0:
        raise ValueError("invalid mesh")
    if grading < 1.0:
        raise ValueError("grading must be at least one")
    ux = torch.linspace(0.0, 1.0, nx + 1, dtype=DTYPE)
    x = width * ux ** grading
    uz = torch.linspace(0.0, 1.0, nz + 1, dtype=DTYPE)
    z = -depth * uz ** grading
    nodes = torch.stack(torch.meshgrid(x, z, indexing="ij"), dim=-1).reshape(-1, 2)
    elems = []

    def node(i: int, j: int) -> int:
        return i * (nz + 1) + j

    for i in range(nx):
        for j in range(nz):
            elems.append([node(i, j + 1), node(i + 1, j + 1), node(i + 1, j), node(i, j)])
    return nodes, torch.tensor(elems, dtype=torch.long)


def assemble_plane_strain_stiffness(nodes: torch.Tensor, elements: torch.Tensor,
                                    young: float, poisson: float) -> torch.Tensor:
    """Q4 plane-strain stiffness (unit thickness), DOFs ordered ``(u_x, u_z)``."""
    C = _constitutive_plane_strain(young, poisson)
    nnode = nodes.shape[0]
    K = torch.zeros(2 * nnode, 2 * nnode, dtype=DTYPE)
    gp = torch.tensor([-1.0, 1.0], dtype=DTYPE) / math.sqrt(3.0)
    for elem in elements.tolist():
        coords = nodes[elem]
        x, z = coords[:, 0], coords[:, 1]
        ke = torch.zeros(8, 8, dtype=DTYPE)
        for xi in gp:
            for eta in gp:
                dN_dxi = torch.tensor(
                    [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=DTYPE) / 4.0
                dN_deta = torch.tensor(
                    [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)], dtype=DTYPE) / 4.0
                jac = torch.stack([
                    torch.stack([(dN_dxi * x).sum(), (dN_dxi * z).sum()]),
                    torch.stack([(dN_deta * x).sum(), (dN_deta * z).sum()]),
                ])
                det = torch.linalg.det(jac)
                if float(det) <= 0.0:
                    raise ValueError("degenerate plane-strain element")
                inv = torch.linalg.inv(jac)
                dN = torch.stack([inv[0, 0] * dN_dxi + inv[1, 0] * dN_deta,
                                  inv[0, 1] * dN_dxi + inv[1, 1] * dN_deta])
                B = torch.zeros(3, 8, dtype=DTYPE)
                B[0, 0::2] = dN[0]
                B[1, 1::2] = dN[1]
                B[2, 0::2] = dN[1]
                B[2, 1::2] = dN[0]
                ke += B.T @ C @ B * det
        dofs = torch.tensor([d for n in elem for d in (2 * n, 2 * n + 1)],
                            dtype=torch.long)
        K[dofs[:, None], dofs[None, :]] += ke
    return 0.5 * (K + K.T)


def _ring_measures(radii: torch.Tensor) -> torch.Tensor:
    """Tributary annulus areas ``2 pi r dr`` for nodes on an axisymmetric surface."""
    radii = torch.as_tensor(radii, dtype=DTYPE)
    edges = torch.empty(radii.numel() + 1, dtype=DTYPE)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (radii[:-1] + radii[1:])
    edges[-1] = radii[-1]
    return math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


def _line_measures(coords: torch.Tensor) -> torch.Tensor:
    """Tributary surface lengths for 1-D (plane-strain) contact nodes."""
    coords = torch.as_tensor(coords, dtype=DTYPE)
    edges = torch.empty(coords.numel() + 1, dtype=DTYPE)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (coords[:-1] + coords[1:])
    edges[-1] = coords[-1]
    return edges[1:] - edges[:-1]


@dataclass(frozen=True)
class ContactCompliance:
    """Condensed FE compliance ``u_C = H f_C`` on the contact surface.

    ``H`` has interleaved columns ``[t_1, n_1, t_2, n_2, ...]`` where ``t`` is
    the tangential (radial for axisymmetric, ``x`` for plane strain) and ``n``
    the normal (``z``) degree of freedom.  ``coords`` holds the tangential
    coordinates of the contact nodes and ``measures`` their tributary ring
    areas / line lengths.  Everything is dimensionless: unit modulus, lengths
    in units of the Hertz contact radius / half-width.
    """

    H: torch.Tensor
    coords: torch.Tensor
    measures: torch.Tensor
    kind: str
    ndof: int = 0
    memory: dict = field(default_factory=dict)

    @property
    def normal(self) -> torch.Tensor:
        return self.H[1::2, 1::2]

    @property
    def tangential(self) -> torch.Tensor:
        return self.H[0::2, 0::2]


def _condense(K: torch.Tensor, contact_dofs: torch.Tensor,
              fixed_dofs: torch.Tensor) -> torch.Tensor:
    """Static condensation of ``K`` onto ``contact_dofs`` with ``fixed_dofs`` held.

    Returns ``H`` with ``u_C = H f_C``.  Exact linear algebra on the assembled
    FE stiffness: one LU factorization with one right-hand side per contact
    DOF.
    """
    ndof = K.shape[0]
    fixed = torch.zeros(ndof, dtype=torch.bool)
    fixed[fixed_dofs] = True
    cont = torch.zeros(ndof, dtype=torch.bool)
    cont[contact_dofs] = True
    if bool(torch.any(fixed & cont)):
        raise ValueError("a DOF cannot be both fixed and a contact DOF")
    free = torch.nonzero(~fixed & ~cont, as_tuple=False).flatten()
    Kff = K[free][:, free]
    Kfc = K[free][:, contact_dofs]
    Kcc = K[contact_dofs][:, contact_dofs]
    X = torch.linalg.solve(Kff, Kfc)          # Kff^-1 Kfc
    M = Kcc - Kfc.T @ X
    H = torch.linalg.inv(0.5 * (M + M.T))
    return 0.5 * (H + H.T)


def _graded_coords(segments: list[tuple[float, float]], total: float,
                   growth: float) -> torch.Tensor:
    """Node coordinates of ``[0, total]`` from ``(end, step)`` segments + growth.

    Each segment ``(end_i, step_i)`` is filled with uniform steps, so the mesh
    can be refined exactly where the contact physics needs it (the patch and,
    for tangential problems, the stick/slip edge), and the outermost zone grows
    geometrically to the far field.
    """
    if not segments or segments[-1][0] >= total - 1e-12:
        raise ValueError("the explicit segments must stop before the far field")
    coords = [0.0]
    for end, step in segments:
        if end <= coords[-1] or step <= 0.0:
            raise ValueError("segment ends must increase and steps be positive")
        count = max(1, int(round((end - coords[-1]) / step)))
        start = coords[-1]
        coords += [start + (i + 1) * (end - start) / count for i in range(count)]
    # geometric far field beyond the last segment end
    if coords[-1] < total - 1e-12:
        step = coords[-1] - coords[-2]
        while coords[-1] < total - 1e-12:
            step *= growth
            coords.append(min(coords[-1] + step, total))
    return torch.tensor(coords, dtype=DTYPE)


def graded_halfspace_mesh(radial: list[tuple[float, float]],
                          axial: list[tuple[float, float]], total: float, *,
                          growth: float = 1.3
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Graded Q4 mesh of the half-space ``r in [0, total]``, ``z in [-total, 0]``.

    ``radial``/``axial`` are ``(end, step)`` segment lists measured from the
    contact centre / surface; the axial list must reach ``total`` in depth.
    """
    r = _graded_coords(radial, total, growth)
    z = -_graded_coords(axial, total, growth)
    nodes = torch.stack(torch.meshgrid(r, z, indexing="ij"), dim=-1).reshape(-1, 2)
    nz = z.numel()
    elems = []

    def node(i: int, j: int) -> int:
        return i * nz + j

    for i in range(r.numel() - 1):
        for j in range(nz - 1):
            elems.append([node(i, j + 1), node(i + 1, j + 1), node(i + 1, j), node(i, j)])
    return nodes, torch.tensor(elems, dtype=torch.long)


def default_segments(n_patch: int, extent: float, total: float,
                     patch_depth: float = 2.0) -> tuple[list, list]:
    """Uniform patch resolution ``h = extent/n_patch`` from centre to far field."""
    step = extent / n_patch
    return ([(extent, step)], [(patch_depth, step)])


def axisymmetric_compliance(*, n_in: int, nz_in: int, poisson: float, extent: float = 2.0,
                            depth_in: float = 2.0, total: float = 30.0,
                            growth: float = 1.3, radial: list | None = None,
                            axial: list | None = None) -> ContactCompliance:
    """Condensed compliance of the axisymmetric half-space (unit modulus)."""
    if radial is None or axial is None:
        radial, axial = default_segments(n_in, extent, total, depth_in)
    nodes, elements = graded_halfspace_mesh(radial, axial, total, growth=growth)
    K = assemble_axisymmetric_stiffness(nodes, elements, 1.0, poisson)
    contact_nodes = torch.arange(0, nodes.shape[0], dtype=torch.long)
    contact_nodes = contact_nodes[nodes[:, 1].abs() < 1e-14]
    contact_nodes = contact_nodes[1:]                          # r > 0 only
    contact_nodes = contact_nodes[nodes[contact_nodes, 0] < total - 1e-12]
    contact_dofs = torch.stack(
        [2 * contact_nodes, 2 * contact_nodes + 1], dim=1).flatten()
    fixed = []
    for i, (r, z) in enumerate(nodes.tolist()):
        if abs(r) < 1e-14:                                     # axis: u_r = 0
            fixed.append(2 * i)
        if abs(z + total) < 1e-12:                             # base: u_r = u_z = 0
            fixed.extend((2 * i, 2 * i + 1))
        if abs(r - total) < 1e-12:                             # far field: u_r = 0
            fixed.append(2 * i)
    H = _condense(K, contact_dofs, torch.tensor(sorted(set(fixed)), dtype=torch.long))
    coords = nodes[contact_nodes, 0]
    return ContactCompliance(H=H, coords=coords, measures=_ring_measures(coords),
                             kind="axisymmetric", ndof=2 * nodes.shape[0],
                             memory={"n_in": n_in, "nz_in": nz_in, "extent": extent,
                                     "depth_in": depth_in, "total": total,
                                     "growth": growth, "radial": radial,
                                     "axial": axial, "nodes": int(nodes.shape[0]),
                                     "elements": int(elements.shape[0])})


def plane_strain_compliance(*, n_in: int, nz_in: int, poisson: float, extent: float = 2.0,
                            depth_in: float = 2.0, total: float = 30.0,
                            growth: float = 1.3, radial: list | None = None,
                            axial: list | None = None) -> ContactCompliance:
    """Condensed compliance of the plane-strain half-space (unit modulus)."""
    if radial is None or axial is None:
        radial, axial = default_segments(n_in, extent, total, depth_in)
    nodes, elements = graded_halfspace_mesh(radial, axial, total, growth=growth)
    K = assemble_plane_strain_stiffness(nodes, elements, 1.0, poisson)
    contact_nodes = torch.arange(0, nodes.shape[0], dtype=torch.long)
    contact_nodes = contact_nodes[nodes[:, 1].abs() < 1e-14]
    contact_nodes = contact_nodes[1:]                          # x > 0 only
    contact_nodes = contact_nodes[nodes[contact_nodes, 0] < total - 1e-12]
    contact_dofs = torch.stack(
        [2 * contact_nodes, 2 * contact_nodes + 1], dim=1).flatten()
    fixed = []
    for i, (x, z) in enumerate(nodes.tolist()):
        if abs(x) < 1e-14:                                     # symmetry plane
            fixed.append(2 * i)
        if abs(z + total) < 1e-12:
            fixed.extend((2 * i, 2 * i + 1))
        if abs(x - total) < 1e-12:
            fixed.append(2 * i)
    H = _condense(K, contact_dofs, torch.tensor(sorted(set(fixed)), dtype=torch.long))
    coords = nodes[contact_nodes, 0]
    return ContactCompliance(H=H, coords=coords, measures=_line_measures(coords),
                             kind="plane_strain", ndof=2 * nodes.shape[0],
                             memory={"n_in": n_in, "nz_in": nz_in, "extent": extent,
                                     "depth_in": depth_in, "total": total,
                                     "growth": growth, "radial": radial,
                                     "axial": axial, "nodes": int(nodes.shape[0]),
                                     "elements": int(elements.shape[0])})


# ---------------------------------------------------------------------------
# 3. unilateral solvers
# ---------------------------------------------------------------------------
def complementarity_residual(pressure: torch.Tensor, gap: torch.Tensor,
                             load: torch.Tensor | float,
                             reference_length: float) -> float:
    """Dimensionless complementarity residual ``||p . g||_1 / (N L_ref)``.

    ``p . g`` is an energy per unit contact area; dividing by the contact load
    and the indentation reference length makes the measure dimensionless and
    directly comparable between penalty and ALM solutions.  It vanishes for an
    admissible triple ``p >= 0``, ``g >= 0``, ``p . g = 0``; for penalty
    contact the leftover is the penetration work, which is exactly the
    quantity that shrinks when ``k_n`` grows.
    """
    p = torch.as_tensor(pressure, dtype=DTYPE)
    e = torch.as_tensor(gap, dtype=DTYPE)
    scale = abs(float(load)) * float(reference_length)
    if scale <= 0.0:
        return 0.0
    return float(torch.sum(torch.abs(p * e))) / scale


@dataclass(frozen=True)
class NormalContactSolution:
    nodal_force: torch.Tensor
    pressure: torch.Tensor
    gap: torch.Tensor
    load: torch.Tensor
    iterations: int
    converged: bool
    condition_number: float
    residual: float


def solve_normal_penalty(Hn: torch.Tensor, offset: torch.Tensor, penalty: float,
                         measures: torch.Tensor, *, max_iterations: int = 200
                         ) -> NormalContactSolution:
    """Nodal penalty normal contact ``f = k_n max(-g, 0)`` by active sets.

    The active set solves ``(I + k_n H_AA) f_A = -k_n offset_A``.  The matrix is
    SPD, so the iteration only has to decide membership; the condition number
    of the active-set system is reported because it grows with ``k_n`` -- that
    is the residual-versus-conditioning trade-off the mission asks to expose
    rather than hide.
    """
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    nc = Hn.shape[0]
    eye = torch.eye(nc, dtype=DTYPE)
    force = torch.zeros(nc, dtype=DTYPE)
    active = offset < 0
    condition = float("nan")
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        ids = torch.nonzero(active, as_tuple=False).flatten()
        force = torch.zeros(nc, dtype=DTYPE)
        if ids.numel():
            A = eye[ids][:, ids] + penalty * Hn[ids][:, ids]
            condition = float(torch.linalg.cond(A))
            force[ids] = torch.linalg.solve(A, -penalty * offset[ids])
        gap = offset + Hn @ force
        updated = gap < 0.0
        if torch.equal(updated, active):
            converged = True
            break
        active = updated
    gap = offset + Hn @ force
    load = torch.sum(force)
    residual = complementarity_residual(force / measures, gap, load,
                                        float(torch.max(torch.abs(offset))))
    return NormalContactSolution(nodal_force=force, pressure=force / measures,
                                 gap=gap, load=load, iterations=iteration,
                                 converged=converged,
                                 condition_number=condition, residual=residual)


def solve_normal_alm(Hn: torch.Tensor, offset: torch.Tensor, measures: torch.Tensor,
                     *, rho_scale: float = 1.0, tolerance: float = 1e-12,
                     max_iterations: int = 100,
                     max_subproblem: int = 200) -> NormalContactSolution:
    """Augmented-Lagrangian normal contact (exact non-penetration).

    The ALM subproblem ``f = rho max(0, lam/rho - gap(f))`` with
    ``gap = offset + H f`` is algebraically the *penalty* problem for the shifted
    offset ``offset - lam/rho``, so each subproblem reuses the exact active-set
    penalty solver and the multiplier update ``lam <- max(0, lam - rho gap)``
    removes the remaining penetration.  The converged state satisfies
    ``f >= 0``, ``gap >= 0`` and ``f . gap = 0`` to the requested tolerance --
    the complementarity gate the mission asks for -- without ever inverting the
    (far-field ill-conditioned) compliance matrix and without an unbounded
    penalty: ``rho`` only escalates when the residual stalls.
    """
    nc = Hn.shape[0]
    h_scale = float(torch.median(torch.diagonal(Hn).abs()))
    if not math.isfinite(h_scale) or h_scale <= 0.0:
        raise ValueError("degenerate compliance matrix")
    rho = rho_scale / h_scale
    multiplier = torch.zeros(nc, dtype=DTYPE)
    force = torch.zeros(nc, dtype=DTYPE)
    gap = offset.clone()
    condition = float("nan")
    converged = False
    iteration = 0
    residual = float("inf")
    scale = float(torch.max(torch.abs(offset)))
    for iteration in range(1, max_iterations + 1):
        sub = solve_normal_penalty(Hn, offset - multiplier / rho, rho, measures,
                                   max_iterations=max_subproblem)
        force = sub.nodal_force
        condition = sub.condition_number
        gap = offset + Hn @ force
        load = torch.sum(force)
        residual = complementarity_residual(force / measures, gap, load, scale)
        if residual <= tolerance:
            converged = True
            break
        previous = multiplier
        multiplier = torch.clamp(multiplier - rho * gap, min=0.0)
        if iteration % 6 == 0 and residual > 1e3 * tolerance:
            rho *= 4.0
            multiplier = previous
    gap = offset + Hn @ force
    load = torch.sum(force)
    residual = complementarity_residual(force / measures, gap, load, scale)
    return NormalContactSolution(nodal_force=force, pressure=force / measures,
                                 gap=gap, load=load, iterations=iteration,
                                 converged=converged, condition_number=condition,
                                 residual=residual)


def recover_hertz_pressure(pressure: torch.Tensor, coords: torch.Tensor) -> tuple[float, float]:
    """Least-squares recovery of ``(a, p_0)`` from the computed pressure field.

    Hertz implies ``p(r)^2 = p_0^2 - (p_0^2/a^2) r^2`` inside the contact, which
    is linear in ``r^2``; the fit uses only the computed nodal pressures, and
    only nodes carrying at least 0.5 % of the peak pressure (the clustered
    edge nodes have the largest discretisation error and no measurable load).
    """
    p = torch.as_tensor(pressure, dtype=DTYPE)
    r = torch.as_tensor(coords, dtype=DTYPE)
    peak = float(p.max())
    if peak <= 0.0:
        return 0.0, 0.0
    mask = p > 0.005 * peak
    if int(mask.sum()) < 3:
        radius = float(r[p > 0].max()) if bool(torch.any(p > 0)) else 0.0
        return radius, peak
    design = torch.stack([torch.ones_like(r[mask]), r[mask] ** 2], dim=1)
    solution = torch.linalg.lstsq(design, p[mask] ** 2).solution
    alpha, beta = float(solution[0]), float(solution[1])
    if alpha <= 0.0 or beta >= 0.0:
        radius = float(r[p > 0].max())
        return radius, peak
    return math.sqrt(-alpha / beta), math.sqrt(alpha)


@dataclass(frozen=True)
class CattaneoPath:
    shift: torch.Tensor          # rigid tangential shift samples (dimensionless)
    tangential_load: torch.Tensor
    stick: torch.Tensor          # (nshift, ncontact) stick mask
    traction: torch.Tensor       # (nshift, ncontact) tangential traction
    stick_radius: torch.Tensor   # nodal estimate of c per sample
    slip_radius: torch.Tensor    # first slip node radius (upper bracket of c)


def solve_cattaneo_tangential(Hr: torch.Tensor, pressure: torch.Tensor,
                              measures: torch.Tensor, coords: torch.Tensor,
                              mu: float, shifts: torch.Tensor) -> CattaneoPath:
    """Incremental Cattaneo--Mindlin tangential contact on the condensed problem.

    The normal pressure is held fixed (``N`` applied first, then ``Q``), the
    stick set is *rigid* (``u_r = s`` inside the stick zone, exactly Coulomb
    ``q = mu p`` in the slip annulus), which is the discrete counterpart of the
    Cattaneo solution.  Loading is monotone in ``s`` so the slip annulus only
    grows; each sample is solved by an active-set loop that removes nodes whose
    stick traction would exceed ``mu p``.
    """
    if mu <= 0:
        raise ValueError("friction coefficient must be positive")
    p = torch.as_tensor(pressure, dtype=DTYPE)
    contact = p > 0.0
    nc = p.numel()
    stick = contact.clone()
    tractions = []
    sticks = []
    c_values = []
    slip_values = []
    loads = []
    tol = 1e-14
    for s in torch.as_tensor(shifts, dtype=DTYPE):
        if float(s) == 0.0:
            q = torch.zeros(nc, dtype=DTYPE)
            tractions.append(q)
            sticks.append(stick.clone())
            c_values.append(float(coords[stick].max()) if bool(torch.any(stick)) else 0.0)
            slip_values.append(0.0)
            loads.append(torch.tensor(0.0, dtype=DTYPE))
            continue
        stick = contact.clone()
        while True:
            S = torch.nonzero(stick & contact, as_tuple=False).flatten()
            F = torch.nonzero(~stick & contact, as_tuple=False).flatten()
            q = torch.zeros(nc, dtype=DTYPE)
            if F.numel():
                q[F] = mu * p[F]
            if S.numel():
                rhs = s.expand(S.numel()) - Hr[S][:, F] @ q[F]
                q[S] = torch.linalg.solve(Hr[S][:, S], rhs)
            excess = q - mu * p
            violations = torch.nonzero((excess > tol * float(p[contact].max())) & stick,
                                       as_tuple=False).flatten()
            if not violations.numel():
                break
            stick[violations] = False
        tractions.append(q)
        sticks.append(stick.clone())
        if bool(torch.any(stick)):
            c_values.append(float(coords[stick].max()))
            slip = torch.nonzero((~stick) & contact, as_tuple=False).flatten()
            slip_values.append(float(coords[slip].min()) if slip.numel() else float("nan"))
        else:
            c_values.append(0.0)
            slip_values.append(float(coords[contact].min()))
        loads.append(torch.sum(q))
    return CattaneoPath(shift=torch.as_tensor(shifts, dtype=DTYPE),
                        tangential_load=torch.stack(loads),
                        stick=torch.stack(sticks),
                        traction=torch.stack(tractions),
                        stick_radius=torch.tensor(c_values, dtype=DTYPE),
                        slip_radius=torch.tensor(slip_values, dtype=DTYPE))