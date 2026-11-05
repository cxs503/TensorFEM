"""Rotating annular disk under centrifugal body force: closed-form oracle + HEX8 driver.

Why this case
-------------
The 3-D solid path (:mod:`tensorfem.solid3d` + :mod:`tensorfem.sparse_core`) is
already benchmarked against closed-form solutions that are all driven by
* nodal * loads: Saint-Venant torsion, pure bending, Kirsch, Lame, Hertz.  No
merged case exercises (a) a *body force* field that varies linearly over the
mesh, ``b = rho*omega**2*(x, y, 0)``, and (b) an * axisymmetric * stress state
that is not uniform along any mesh line.  The rotating disk supplies both and is
the textbook design case for flywheels, brake discs, turbine rotors and
propulsion-shaft flanges, so it is the cheapest missing verification anchor.

Exact solution
--------------
Plane stress, thin annular disk of density ``rho``, angular velocity ``omega``,
inner radius ``a``, outer radius ``b``, free at both rims (Timoshenko & Goodier,
*Theory of Elasticity*, 3rd ed., section 32 -- rotating disk with a central hole,
plane stress):

    s      = (3 + nu) / 8 * rho * omega**2
    sigma_r     = s * (b**2 + a**2 - a**2*b**2/r**2 - r**2)
    sigma_theta = s * (b**2 + a**2 + a**2*b**2/r**2 - (1 + 3*nu)/(3 + nu) * r**2)
    u_r         = (1 - nu)/(8*E) * rho * omega**2
                  * ((3 + nu)*(b**2 + a**2)*r
                     + (3 + nu)*(1 + nu)/(1 - nu)*a**2*b**2/r
                     - (1 + nu)*r**3)

``u_theta = u_z = 0``.  Taking ``a -> 0`` recovers the solid-disk solution
``sigma_r = s*(b**2 - r**2)``, ``u_r = (1-nu)/(8*E)*rho*omega**2*((3+nu)*b**2*r
- (1+nu)*r**3)``, which the package also exposes.

This module deliberately *does not* derive the field from the code under test:
the closed form is the reference, and :func:`tensorfem.rotating_disk.equilibrium_residual`
/ :func:`~tensorfem.rotating_disk.compatibility_residual` /
:meth:`RotatingDisk.rim_tractions` are the self-checks that must vanish before
the field is used to grade the finite-element path.

Finite-element driver
---------------------
A quarter annulus with a mid-plane symmetry plane is meshed with HEX8 elements
in polar blocks; the centrifugal field is converted to consistent nodal loads by
an *independent* trilinear quadrature (no ``B`` matrix, Jacobian or stiffness of
the code under test is reused).  Symmetry on ``x = 0``, ``y = 0`` and ``z = 0``
removes the six rigid-body modes and is exactly compatible with the plane-stress
solution, so no Dirichlet data of the displacement field is imposed anywhere --
the answer is produced by the stiffness of :func:`tensorfem.solid3d.hex8_stiffness`
and the solver of :func:`tensorfem.sparse_core.solve_sparse_static` alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .solid3d import elasticity_matrix_3d, hex8_stiffness
from .sparse_core import (assemble_coo, conjugate_gradient, constraint_transform,
                          sparse_mv)

__all__ = [
    "RotatingDisk",
    "DiskResult",
    "quarter_disk_mesh",
    "centrifugal_load",
    "rotating_disk_run",
    "cylindrical_stress",
    "solid_disk_reference",
]


# --------------------------------------------------------------------------- #
# closed-form oracle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RotatingDisk:
    """Plane-stress rotating annular disk, free at both rims.

    Parameters are SI (``E`` in Pa, ``rho`` in kg/m^3, ``omega`` in rad/s).  A
    zero ``inner`` radius is allowed and selects the solid-disk formulas.
    """

    outer: float = 1.0
    inner: float = 0.5
    young_modulus: float = 200.0e9
    poisson_ratio: float = 0.3
    density: float = 7850.0
    omega: float = 314.1592653589793
    thickness: float = 0.02

    def __post_init__(self) -> None:
        for name, value in (
            ("outer", self.outer),
            ("inner", self.inner),
            ("young_modulus", self.young_modulus),
            ("density", self.density),
            ("omega", self.omega),
            ("thickness", self.thickness),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.outer <= 0.0:
            raise ValueError("outer radius must be positive")
        if not self.inner < self.outer:
            raise ValueError("need inner radius < outer radius")
        if self.young_modulus <= 0.0 or self.density <= 0.0:
            raise ValueError("young_modulus and density must be positive")
        if self.thickness <= 0.0:
            raise ValueError("thickness must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")

    # -- helpers ---------------------------------------------------------- #
    @property
    def scale(self) -> float:
        """``(3 + nu)/8 * rho * omega**2`` -- the common factor of the stresses."""
        return (3.0 + self.poisson_ratio) / 8.0 * self.density * self.omega ** 2

    @property
    def _poisson_factor(self) -> float:
        return (1.0 + 3.0 * self.poisson_ratio) / (3.0 + self.poisson_ratio)

    def _check_radius(self, r: torch.Tensor) -> torch.Tensor:
        if torch.any(r <= 0.0):
            raise ValueError("radius must be positive (the axis r = 0 is singular)")
        return r

    # -- exact fields ------------------------------------------------------ #
    def sigma_r(self, r: torch.Tensor) -> torch.Tensor:
        r = self._check_radius(torch.as_tensor(r))
        a2, b2 = self.inner ** 2, self.outer ** 2
        return self.scale * (b2 + a2 - a2 * b2 / r ** 2 - r ** 2)

    def sigma_theta(self, r: torch.Tensor) -> torch.Tensor:
        r = self._check_radius(torch.as_tensor(r))
        a2, b2 = self.inner ** 2, self.outer ** 2
        return self.scale * (b2 + a2 + a2 * b2 / r ** 2 - self._poisson_factor * r ** 2)

    def u_r(self, r: torch.Tensor) -> torch.Tensor:
        r = self._check_radius(torch.as_tensor(r))
        a2, b2 = self.inner ** 2, self.outer ** 2
        nu = self.poisson_ratio
        return ((1.0 - nu) / (8.0 * self.young_modulus) * self.density * self.omega ** 2
                * ((3.0 + nu) * (b2 + a2) * r
                   + (3.0 + nu) * (1.0 + nu) / (1.0 - nu) * a2 * b2 / r
                   - (1.0 + nu) * r ** 3))

    def displacement(self, xyz: torch.Tensor, *, contraction: bool = True) -> torch.Tensor:
        """Exact displacement ``(u_r*cos(theta), u_r*sin(theta), u_z)``.

        ``contraction=True`` (default) adds the plane-stress thickness
        contraction ``u_z = -nu*(sigma_r + sigma_theta)/E * z``, measured from the
        mid-plane ``z = 0``.  A three-dimensional model of a thin disk always
        carries it (it is the ``eps_z = -nu*(sigma_r+sigma_theta)/E`` of the
        plane-stress state), so comparing a solid-element solution with
        ``contraction=False`` charges the element for a real, correct
        out-of-plane displacement; on the ``32 x 16`` mesh that single term is
        45% of the whole displacement error.
        """
        xyz = torch.as_tensor(xyz)
        r = torch.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
        ur = self.u_r(r)
        if contraction:
            uz = (-self.poisson_ratio * (self.sigma_r(r) + self.sigma_theta(r))
                  / self.young_modulus * xyz[..., 2])
        else:
            uz = torch.zeros_like(ur)
        return torch.stack((ur * xyz[..., 0] / r, ur * xyz[..., 1] / r, uz), dim=-1)

    def sigma_peak(self) -> float:
        """Exact peak hoop stress, attained on the inner rim ``r = a``."""
        a2, b2 = self.inner ** 2, self.outer ** 2
        return float(self.scale * (2.0 * b2 + a2 * (1.0 - self._poisson_factor)))

    def u_peak(self) -> float:
        """Exact maximum radial displacement (occurs in the interior or at ``b``)."""
        r = torch.linspace(self.inner, self.outer, 20001, dtype=torch.float64)[1:]
        return float(self.u_r(r).max())

    # -- self-checks that must vanish for the field to be usable ---------- #
    def equilibrium_residual(self, r: torch.Tensor, h: float = 1e-6) -> torch.Tensor:
        """Normalised ``(d(sigma_r)/dr + (sigma_r - sigma_theta)/r
        + rho*omega**2*r) / (rho*omega**2*r)``.

        This is the axisymmetric radial equilibrium of the rotating disk; it is
        zero for the exact field and non-zero for any perturbation, so it is a
        genuine test of a candidate reference, not a tautology.
        """
        r = self._check_radius(torch.as_tensor(r, dtype=torch.float64))
        derivative = (self.sigma_r(r + h) - self.sigma_r(r - h)) / (2.0 * h)
        sr, st = self.sigma_r(r), self.sigma_theta(r)
        load = self.density * self.omega ** 2 * r
        return (derivative + (sr - st) / r + load) / load

    def compatibility_residual(self, r: torch.Tensor, h: float = 1e-6) -> torch.Tensor:
        """``d(r*eps_theta)/dr - eps_r`` with the plane-stress constitutive law."""
        r = self._check_radius(torch.as_tensor(r, dtype=torch.float64))

        def g(q: torch.Tensor) -> torch.Tensor:
            return q * (self.sigma_theta(q) - self.poisson_ratio * self.sigma_r(q)) \
                / self.young_modulus

        eps_r = (self.sigma_r(r) - self.poisson_ratio * self.sigma_theta(r)) \
            / self.young_modulus
        return (g(r + h) - g(r - h)) / (2.0 * h) - eps_r

    def rim_tractions(self) -> tuple[float, float]:
        """Exact radial stress on the inner and outer rims (both must vanish)."""
        return (float(self.sigma_r(torch.tensor(self.inner, dtype=torch.float64))),
                float(self.sigma_r(torch.tensor(self.outer, dtype=torch.float64))))

    def hoop_force_balance(self, samples: int = 200001) -> tuple[float, float]:
        """Global radial equilibrium: ``int_a^b sigma_theta dr`` vs ``rho*omega**2*(b^3-a^3)/3``.

        Cutting the annulus along a diameter, the centrifugal resultant of the
        half ring ``2*rho*omega**2*(b**3-a**3)/3`` (per unit thickness) must be
        carried by the two hoop-stress resultants ``2*int sigma_theta dr``.  The
        identity is independent of the pointwise solution and therefore
        cross-checks the closed form as a whole.
        """
        r = torch.linspace(self.inner, self.outer, samples, dtype=torch.float64)
        lhs = float(torch.trapz(self.sigma_theta(r), r))
        rhs = self.density * self.omega ** 2 * (self.outer ** 3 - self.inner ** 3) / 3.0
        return lhs, rhs

    def as_dict(self) -> dict[str, float]:
        return {
            "outer_radius": self.outer,
            "inner_radius": self.inner,
            "young_modulus": self.young_modulus,
            "poisson_ratio": self.poisson_ratio,
            "density": self.density,
            "omega": self.omega,
            "thickness": self.thickness,
            "rpm": self.omega * 60.0 / (2.0 * math.pi),
            "peak_hoop_stress": self.sigma_peak(),
            "peak_radial_displacement": self.u_peak(),
        }


def solid_disk_reference(disk: RotatingDisk, r: torch.Tensor) -> dict[str, torch.Tensor]:
    """Solid-disk (no bore) limit of :class:`RotatingDisk` in closed form."""
    r = torch.as_tensor(r, dtype=torch.float64)
    nu, b2 = disk.poisson_ratio, disk.outer ** 2
    s = (3.0 + nu) / 8.0 * disk.density * disk.omega ** 2
    return {
        "sigma_r": s * (b2 - r ** 2),
        "sigma_theta": s * (b2 - (1.0 + 3.0 * nu) / (3.0 + nu) * r ** 2),
        "u_r": ((1.0 - nu) / (8.0 * disk.young_modulus) * disk.density * disk.omega ** 2
                * ((3.0 + nu) * b2 * r - (1.0 + nu) * r ** 3)),
    }


# --------------------------------------------------------------------------- #
# HEX8 quarter-annulus mesh (radial/angular/thickness blocks)
# --------------------------------------------------------------------------- #
def quarter_disk_mesh(disk: RotatingDisk, radial: int, angular: int, layers: int = 2,
                      *, dtype: torch.dtype = torch.float64):
    """Quarter annulus ``theta in [0, pi/2]`` of the *half* thickness ``h/2``.

    The mesh is laid out as ``n(i, j, k) = i*(angular+1)*(layers+1)
    + j*(layers+1) + k`` (C order of the ``(radial, angular, thickness)`` grid)
    with ``i`` radial, ``j`` angular and ``k`` through the half thickness.  Element corners follow the HEX8 ordering of
    :func:`tensorfem.solid3d.structured_hex_mesh`, so the map
    ``(radial, angular, thickness) -> (r_hat, theta_hat, z_hat)`` is right handed
    and every Jacobian is positive.
    """
    for name, value in (("radial", radial), ("angular", angular), ("layers", layers)):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    rs = torch.linspace(disk.inner, disk.outer, radial + 1, dtype=dtype)
    ts = torch.linspace(0.0, 0.5 * math.pi, angular + 1, dtype=dtype)
    zs = torch.linspace(0.0, 0.5 * disk.thickness, layers + 1, dtype=dtype)
    rr, tt, zz = torch.meshgrid(rs, ts, zs, indexing="ij")   # (nr+1, nt+1, nz+1)
    nodes = torch.stack((rr * torch.cos(tt), rr * torch.sin(tt), zz), dim=-1).reshape(-1, 3)

    def idx(i: int, j: int, k: int) -> int:
        # matches the C-order flattening of the (radial, angular, thickness) grid
        return i * (angular + 1) * (layers + 1) + j * (layers + 1) + k

    cells = []
    for k in range(layers):
        for j in range(angular):
            for i in range(radial):
                cells.append((idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k),
                              idx(i, j + 1, k), idx(i, j, k + 1), idx(i + 1, j, k + 1),
                              idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)))
    return nodes, torch.tensor(cells, dtype=torch.long)


# --------------------------------------------------------------------------- #
# consistent centrifugal nodal loads, integrated independently of the solver
# --------------------------------------------------------------------------- #
#: one-dimensional Gauss-Legendre rules on ``[-1, 1]``, written out so the load
#: integrator shares no quadrature code with the element under test
_GAUSS = {
    1: ((0.0,), (2.0,)),
    2: ((-0.5773502691896257, 0.5773502691896257), (1.0, 1.0)),
    3: ((-0.7745966692414834, 0.0, 0.7745966692414834),
        (0.5555555555555556, 0.8888888888888888, 0.5555555555555556)),
}

_HEX_SIGNS = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
              (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))


def centrifugal_load(disk: RotatingDisk, nodes: torch.Tensor, hexes: torch.Tensor,
                     gauss: int = 2) -> torch.Tensor:
    """``f_a = int_e N_a * rho*omega**2*(x, y, 0) dV`` by trilinear Gauss quadrature.

    Shape functions, ``det J`` and the quadrature rule are implemented here from
    scratch: nothing from :mod:`tensorfem.solid3d` enters, so the load vector is
    an independent quantity and a wrong ``B``/Jacobian in the element cannot hide
    itself by corrupting both the load and the stiffness in the same way.
    """
    if gauss not in _GAUSS:
        raise ValueError("gauss must be 1, 2 or 3")
    pts, wts = (torch.tensor(_GAUSS[gauss][0], dtype=nodes.dtype),
                torch.tensor(_GAUSS[gauss][1], dtype=nodes.dtype))
    signs = torch.tensor(_HEX_SIGNS, dtype=nodes.dtype)
    corners = nodes[hexes]                                   # (e,8,3)
    force = torch.zeros(len(nodes), 3, dtype=nodes.dtype)
    rho_omega2 = disk.density * disk.omega ** 2
    for p, xi in enumerate(pts):
        for q, eta in enumerate(pts):
            for s, zeta in enumerate(pts):
                n = ((1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta)
                     * (1.0 + signs[:, 2] * zeta)) / 8.0                 # (8,)
                dn = torch.stack((
                    signs[:, 0] * (1.0 + signs[:, 1] * eta) * (1.0 + signs[:, 2] * zeta),
                    signs[:, 1] * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 2] * zeta),
                    signs[:, 2] * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta),
                ), dim=1) / 8.0                                          # (8,3)
                jac = torch.einsum("eai,aj->eij", corners, dn)            # (e,3,3)
                det = torch.linalg.det(jac)
                if torch.any(det <= 0.0):
                    raise ValueError("non-positive Jacobian in the centrifugal load")
                coords = torch.einsum("a,eai->ei", n, corners)            # (e,3)
                body = coords * rho_omega2
                body[:, 2] = 0.0
                weight = det * float(wts[p] * wts[q] * wts[s])            # (e,)
                # (8 nodes, e elements, 3) -> element-major (e*8, 3) to match
                # hexes.reshape(-1)
                contribution = (n[:, None, None] * body[None, :, :]
                                * weight[None, :, None]).permute(1, 0, 2)
                force.index_add_(0, hexes.reshape(-1), contribution.reshape(-1, 3))
    return force.reshape(-1)


def cylindrical_stress(sigma: torch.Tensor, coords: torch.Tensor,
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate Cartesian Voigt stress ``[xx, yy, zz, xy, yz, xz]`` to ``(r, theta, z)``."""
    x, y = coords[:, 0], coords[:, 1]
    r = torch.sqrt(x * x + y * y)
    c, s = x / r, y / r
    xx, yy, zz, xy, yz, xz = (sigma[:, k] for k in range(6))
    srr = xx * c * c + yy * s * s + 2.0 * xy * s * c
    stt = xx * s * s + yy * c * c - 2.0 * xy * s * c
    srt = (yy - xx) * s * c + xy * (c * c - s * s)
    return srr, stt, srt, zz


def _jacobi_static(K: torch.Tensor, force: torch.Tensor, dirichlet: dict[int, float],
                   *, rtol: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor, object]:
    """Constrained static solve with a Jacobi-preconditioned CG.

    :func:`tensorfem.sparse_core.solve_sparse_static` runs an unpreconditioned CG.
    A thin disk has an element aspect ratio of about 1:100, so the unpreconditioned
    iteration stagnates around ``1e-6`` on this case; the Jacobi preconditioner
    (diagonal of the reduced operator, which for unit-vector constraint columns is
    just the kept diagonal of ``K``) reaches the requested tolerance.  The
    equations, constraints and operator are exactly the merged ones -- only the
    iteration is preconditioned, and the residual is reported so the run can be
    audited.
    """
    ndof = force.numel()
    T, u0, independent = constraint_transform(ndof, dirichlet, (), dtype=force.dtype)
    tt = T.transpose(0, 1)

    def reduced_mv(q: torch.Tensor) -> torch.Tensor:
        return sparse_mv(tt, sparse_mv(K, sparse_mv(T, q)))

    rhs = sparse_mv(tt, force - sparse_mv(K, u0))
    diagonal = K.diagonal() if K.layout == torch.strided else K.to_dense().diagonal()
    diagonal = diagonal[torch.tensor(independent, dtype=torch.long)].clamp_min(
        torch.finfo(force.dtype).tiny)
    solved = conjugate_gradient(reduced_mv, rhs, rtol=rtol, diagonal=diagonal)
    if not solved.converged:
        raise RuntimeError(
            f"CG did not converge: residual={solved.relative_residual:g} "
            f"after {solved.iterations} iterations")
    u = sparse_mv(T, solved.x) + u0
    return u, sparse_mv(K, u) - force, solved


@dataclass
class DiskResult:
    """Outcome of one finite-element run, all errors against the closed form."""

    radial: int
    angular: int
    layers: int
    nodes: int
    elements: int
    dofs: int
    u_error: float
    hoop_error: float
    radial_error: float
    hoop_ring_error: float
    hoop_peak_got: float
    hoop_peak_exact: float
    rim_radial_max: float
    plane_stress_ratio: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "radial": self.radial, "angular": self.angular, "layers": self.layers,
            "nodes": self.nodes, "elements": self.elements, "dofs": self.dofs,
            "relative_l2_displacement_error": self.u_error,
            "relative_l2_hoop_stress_error": self.hoop_error,
            "relative_l2_radial_stress_error": self.radial_error,
            "bore_ring_hoop_error": self.hoop_ring_error,
            "hoop_peak_fe": self.hoop_peak_got,
            "hoop_peak_exact": self.hoop_peak_exact,
            "rim_radial_stress_over_peak_hoop": self.rim_radial_max,
            "max_sigma_zz_over_peak_hoop": self.plane_stress_ratio,
        }
        out.update(self.extra)
        return out


def rotating_disk_run(disk: RotatingDisk, radial: int = 16, angular: int = 8,
                      layers: int = 2, *, rtol: float = 1e-12,
                      dtype: torch.dtype = torch.float64) -> DiskResult:
    """Solve the quarter model on the merged HEX8 path and grade it.

    Only symmetry conditions are imposed (``u_x = 0`` on ``x = 0``, ``u_y = 0`` on
    ``y = 0``, ``u_z = 0`` on the mid-plane ``z = 0``); every other degree of
    freedom, including the whole free inner bore, is determined by the assembled
    stiffness and the consistent centrifugal load.

    Error definitions, all against the closed form of :class:`RotatingDisk`:

    * ``u_error``      -- relative L2 norm of the nodal displacement error;
    * ``hoop_error``   -- ``||sigma_theta_FE - sigma_theta_exact|| / ||sigma_theta_exact||``
      at element centroids (same for ``radial_error``);
    * ``hoop_ring_error`` -- same norm restricted to the innermost radial ring,
      i.e. the bore where the peak lives;
    * ``rim_radial_max`` -- ``max |sigma_r,FE - sigma_r,exact|`` at the innermost
      and outermost ring centroids, divided by the exact peak hoop stress: the
      accuracy where the stress state has its steepest gradient and where the
      exact traction vanishes;
    * ``plane_stress_ratio`` -- ``max |sigma_zz|`` divided by the exact peak hoop
      stress; a thin disk must return nearly zero.
    """
    nodes, hexes = quarter_disk_mesh(disk, radial, angular, layers, dtype=dtype)
    ne, ndof = len(hexes), 3 * len(nodes)
    young = nodes.new_full((ne,), disk.young_modulus)
    poisson = nodes.new_full((ne,), disk.poisson_ratio)
    D = elasticity_matrix_3d(young, poisson)
    ke, B = hex8_stiffness(nodes[hexes], D)
    edofs = (3 * hexes[:, :, None] + torch.arange(3)).reshape(ne, 24)
    K = assemble_coo(ke, edofs, ndof)

    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    tol = 1e-12 * max(disk.outer, 1.0)
    dirichlet: dict[int, float] = {}
    for node in torch.where(x.abs() <= tol)[0].tolist():
        dirichlet[3 * node] = 0.0
    for node in torch.where(y.abs() <= tol)[0].tolist():
        dirichlet[3 * node + 1] = 0.0
    for node in torch.where(z.abs() <= tol)[0].tolist():
        dirichlet[3 * node + 2] = 0.0

    force = centrifugal_load(disk, nodes, hexes)
    u, reaction, iterative = _jacobi_static(K, force, dirichlet, rtol=rtol)

    exact = disk.displacement(nodes).reshape(-1)
    u_error = float(torch.linalg.vector_norm(u - exact) / torch.linalg.vector_norm(exact))

    centres = nodes[hexes].mean(dim=1)
    strain = torch.einsum("eij,ej->ei", B, u.flatten()[edofs])
    stress = torch.einsum("eij,ej->ei", D, strain)
    srr, stt, srt, szz = cylindrical_stress(stress, centres)
    rc = torch.sqrt(centres[:, 0] ** 2 + centres[:, 1] ** 2)
    sr_exact = disk.sigma_r(rc)
    st_exact = disk.sigma_theta(rc)
    hoop_error = float(torch.linalg.vector_norm(stt - st_exact)
                       / torch.linalg.vector_norm(st_exact))
    radial_error = float(torch.linalg.vector_norm(srr - sr_exact)
                         / torch.linalg.vector_norm(sr_exact))
    # innermost ring: elements whose centroid radius is the smallest one
    ring = rc <= rc.min() + 1e-9 * max(disk.outer, 1.0)
    hoop_ring = float(torch.linalg.vector_norm(stt[ring] - st_exact[ring])
                      / torch.linalg.vector_norm(st_exact[ring]))
    peak_hoop = disk.sigma_peak()
    rim_tol = 1e-9 * max(disk.outer, 1.0)
    rim = (rc <= rc.min() + rim_tol) | (rc >= rc.max() - rim_tol)
    rim_radial = float((srr[rim] - sr_exact[rim]).abs().max() / peak_hoop)
    return DiskResult(
        radial=radial, angular=angular, layers=layers, nodes=len(nodes),
        elements=ne, dofs=ndof,
        u_error=u_error, hoop_error=hoop_error, radial_error=radial_error,
        hoop_ring_error=hoop_ring,
        hoop_peak_got=float(stt.max()), hoop_peak_exact=peak_hoop,
        rim_radial_max=rim_radial,
        plane_stress_ratio=float(szz.abs().max() / peak_hoop),
        extra={
            "centre_radii": rc.tolist(),
            "hoop_fe": stt.tolist(),
            "hoop_exact": st_exact.tolist(),
            "radial_fe": srr.tolist(),
            "radial_exact": sr_exact.tolist(),
            "displacement_fe": u.reshape(-1, 3).tolist(),
            "nodes": nodes.tolist(),
            "reaction_max": float(reaction.abs().max()),
            "rtol": rtol,
            "applied_load_resultant": float(force.reshape(-1, 3).sum(dim=0).norm()),
            "reaction_closure": float((reaction.reshape(-1, 3).sum(dim=0)
                                       + force.reshape(-1, 3).sum(dim=0)).norm()
                                      / force.reshape(-1, 3).sum(dim=0).norm()),
            "cg_iterations": int(iterative.iterations),
            "cg_relative_residual": float(iterative.relative_residual),
        },
    )