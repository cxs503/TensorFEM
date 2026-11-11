"""Axisymmetric shell-of-revolution element (2-node curved-meridian ring element).

Motivation (documented gap)
---------------------------
``docs/benchmark-qualification-status.md`` records a fail-closed negative result
for the curved-shell family: the flat-facet ``Shell4`` element
(``tensorfem.spherical_shell.projected_shell4_stiffness``) under internal
pressure on a closed sphere misses the exact membrane force ``pR/2`` by ~1006%
at 216 facets, and extrapolation says ~2e4 facets are needed to reach the 3%
gate.  The report attributes that failure to faceted bending dominating a
thin + curved + pressure state and records that no qualified route to the exact
membrane field exists in the tree.  This module supplies such a route: a ring
element whose meridian may be a genuine circular arc (``meridian_radius=R`` for
a sphere / torus, ``None`` for a cone or cylinder), i.e. the geometry the
element sees is the shell it claims to be.

Formulation
-----------
Reissner-Mindlin (Ahmad-Irons-Zienkiewicz degeneration) with three DOFs per
node -- meridional displacement ``u``, normal displacement ``w`` and meridional
rotation of the normal ``beta`` -- and five generalised strains::

    eps_s       = du/ds + w / R_s
    eps_theta   = (u cos(phi) + w sin(phi)) / r
    kappa_s     = d beta / ds
    kappa_theta = beta cos(phi) / r
    gamma       = dw/ds + beta - u / R_s      (transverse shear, Kirchhoff defect)

with ``phi`` the angle between the outward normal and the axis of revolution,
``r(s)`` the hoop radius at meridian arc length ``s`` and ``R_s`` the meridional
radius of curvature (``inf`` for a straight generatrix).  Constitutive law
(plane stress, shear correction ``k = 5/6``)::

    N = Cm (eps + nu eps'),   Cm = E t / (1 - nu^2)
    M = Db (kappa + nu kappa'), Db = E t^3 / (12 (1 - nu^2))
    Q = k G t gamma

Membrane and bending terms are integrated with 2-point Gauss; the transverse
shear term is integrated at the element centre only (selective reduced
integration, SRI), which is what keeps the element free of shear locking in the
thin limit.  A full-integration variant is kept behind ``shear_integration`` so
the test-suite can show, as a negative control, that it locks.

Orientation convention
----------------------
Nodes must be ordered so the outward normal ``n = (-dz/ds, dr/ds)`` has
non-negative ``r`` and ``z`` components, i.e. ``cos(phi) = dr/ds >= 0`` and
``sin(phi) = -dz/ds >= 0``.  For a cylinder that means ordering *top -> bottom*,
for a sphere *pole -> equator*, for a cone *small radius -> large radius with z
decreasing*.  The constructor rejects an inverted ordering instead of silently
flipping the pressure side.

Loads
-----
Uniform normal pressure ``p`` on the wall (outward positive) plus an axial nodal
force helper for closed-vessel end caps
(:meth:`AxisymShell.apply_axial_force`).  The ``2 pi r`` hoop measure is exact,
so pressure resultants are exactly ``p * A``; the verification script checks
that against the analytic axial reaction.
"""
from __future__ import annotations

import math

import torch

TWOPI = 2.0 * math.pi
SHEAR_FACTOR = 5.0 / 6.0
_GAUSS = 1.0 / math.sqrt(3.0)


# --------------------------------------------------------------------------- #
# meridian geometry
# --------------------------------------------------------------------------- #
class Meridian:
    """Circular-arc (or straight) generator of a shell-of-revolution segment.

    Parameters
    ----------
    r1, z1, r2, z2 : float
        End nodes in the meridional ``(r, z)`` plane, ordered along the element.
    radius : float, optional
        Meridional radius of curvature ``R_s``.  ``None`` (default) means a
        straight generator (cone / cylinder).  A positive value means ``phi``
        increases along the element (the mid-surface curves towards ``-n``),
        which is the case for a sphere traversed from pole towards the equator.
    """

    def __init__(self, r1: float, z1: float, r2: float, z2: float,
                 radius: float | None = None):
        self.r1, self.z1 = float(r1), float(z1)
        self.r2, self.z2 = float(r2), float(z2)
        self.chord = math.hypot(self.r2 - self.r1, self.z2 - self.z1)
        if self.chord <= 0.0:
            raise ValueError("degenerate element: coincident nodes")
        cr = (self.r2 - self.r1) / self.chord      # dr/ds of the chord
        cz = (self.z2 - self.z1) / self.chord      # dz/ds of the chord
        if cr < 0.0 or cz > 0.0:
            raise ValueError(
                "nodes must be ordered so the outward normal n = (-dz/ds, dr/ds) "
                "has non-negative r and z components; got dr/ds=%g, dz/ds=%g" % (cr, cz))
        self.sn = -cz           # sin(phi) at the chord mid-point
        self.cn = cr            # cos(phi) at the chord mid-point
        self.phi_mid = math.atan2(self.sn, self.cn)
        if radius is None:
            self.radius = math.inf
            self.length = self.chord
            self.theta = 0.0
            self.sagitta = 0.0
            self.r_mid = 0.5 * (self.r1 + self.r2)
            self.z_mid = 0.5 * (self.z1 + self.z2)
        else:
            radius = float(radius)
            if radius <= 0.0:
                raise ValueError("meridian radius must be positive")
            if self.chord > 2.0 * radius * (1.0 + 1e-12):
                raise ValueError("chord longer than the diameter of the meridian arc")
            self.radius = radius
            self.theta = math.asin(min(1.0, self.chord / (2.0 * radius)))
            self.length = 2.0 * radius * self.theta
            self.sagitta = radius * (1.0 - math.cos(self.theta))
            self.r_mid = 0.5 * (self.r1 + self.r2) + self.sagitta * self.sn
            self.z_mid = 0.5 * (self.z1 + self.z2) + self.sagitta * self.cn

    # -- kinematics of the generator ---------------------------------------- #
    def phi(self, s: float) -> float:
        if math.isinf(self.radius):
            return self.phi_mid
        return self.phi_mid + (s - 0.5 * self.length) / self.radius

    def radius_at(self, s: float) -> float:
        """Hoop radius ``r(s)`` of the mid-surface (``dr/ds = cos phi``)."""
        if math.isinf(self.radius):
            return self.r_mid + (s - 0.5 * self.length) * self.cn
        return self.r_mid + self.radius * (math.sin(self.phi(s)) - math.sin(self.phi_mid))

    def point(self, s: float) -> tuple[float, float]:
        """``(r, z)`` of the mid-surface at arc length ``s``."""
        if math.isinf(self.radius):
            return (self.r_mid + (s - 0.5 * self.length) * self.cn,
                    self.z_mid - (s - 0.5 * self.length) * self.sn)
        return (self.r_mid + self.radius * (math.sin(self.phi(s)) - math.sin(self.phi_mid)),
                self.z_mid + self.radius * (math.cos(self.phi(s)) - math.cos(self.phi_mid)))

    def curvature(self) -> float:
        return 0.0 if math.isinf(self.radius) else 1.0 / self.radius

    def gauss(self) -> list[tuple[float, float]]:
        """Two-point Gauss abscissae/weights on ``[0, length]``."""
        half = 0.5 * self.length
        return [(half * (1.0 - _GAUSS), half), (half * (1.0 + _GAUSS), half)]


def _constitutive(young: float, poisson: float, thickness: float) -> torch.Tensor:
    cm = young * thickness / (1.0 - poisson * poisson)
    db = young * thickness ** 3 / (12.0 * (1.0 - poisson * poisson))
    ks = SHEAR_FACTOR * young / (2.0 * (1.0 + poisson)) * thickness
    d = torch.zeros((5, 5), dtype=torch.float64)
    d[0, 0] = d[1, 1] = cm
    d[0, 1] = d[1, 0] = poisson * cm
    d[2, 2] = d[3, 3] = db
    d[2, 3] = d[3, 2] = poisson * db
    d[4, 4] = ks
    return d


def _b_matrix(mer: Meridian, s: float) -> torch.Tensor:
    r = mer.radius_at(s)
    if r <= 1e-12:
        raise ValueError("Gauss point on the axis of revolution (r=0); a pole-adjacent "
                         "element must keep both Gauss points at r > 0")
    phi = mer.phi(s)
    sn, cn = math.sin(phi), math.cos(phi)
    n1, n2 = 1.0 - s / mer.length, s / mer.length
    d1, d2 = -1.0 / mer.length, 1.0 / mer.length
    inv_rs = mer.curvature()
    b = torch.zeros((5, 6), dtype=torch.float64)
    # eps_s = du/ds + w / R_s
    b[0, 0], b[0, 3] = d1, d2
    b[0, 1], b[0, 4] = n1 * inv_rs, n2 * inv_rs
    # eps_theta = (u cos(phi) + w sin(phi)) / r
    b[1, 0], b[1, 1] = cn * n1 / r, sn * n1 / r
    b[1, 3], b[1, 4] = cn * n2 / r, sn * n2 / r
    # kappa_s = d beta / ds
    b[2, 2], b[2, 5] = d1, d2
    # kappa_theta = beta cos(phi) / r
    b[3, 2], b[3, 5] = cn * n1 / r, cn * n2 / r
    # gamma = dw/ds + beta - u / R_s
    b[4, 0], b[4, 3] = -n1 * inv_rs, -n2 * inv_rs
    b[4, 1], b[4, 4] = d1, d2
    b[4, 2], b[4, 5] = n1, n2
    return b


def element_stiffness(mer: Meridian, young: float, poisson: float, thickness: float,
                      *, shear_integration: str = "reduced") -> torch.Tensor:
    """``6 x 6`` element stiffness in the local ``(u, w, beta)`` ordering."""
    if shear_integration not in ("reduced", "full"):
        raise ValueError("shear_integration must be 'reduced' or 'full'")
    d = _constitutive(young, poisson, thickness)
    ke = torch.zeros((6, 6), dtype=torch.float64)
    for s, w in mer.gauss():
        b = _b_matrix(mer, s)
        ke += b[:4].T @ d[:4, :4] @ b[:4] * (TWOPI * mer.radius_at(s) * w)
    if shear_integration == "reduced":
        s = 0.5 * mer.length
        b = _b_matrix(mer, s)
        ke += b[4:5].T @ d[4:, 4:] @ b[4:5] * (TWOPI * mer.radius_at(s) * mer.length)
    else:
        for s, w in mer.gauss():
            b = _b_matrix(mer, s)
            ke += b[4:5].T @ d[4:, 4:] @ b[4:5] * (TWOPI * mer.radius_at(s) * w)
    return ke


def element_pressure_load(mer: Meridian, pressure: float) -> torch.Tensor:
    """Equivalent nodal load of a uniform outward normal pressure."""
    f = torch.zeros(6, dtype=torch.float64)
    for s, w in mer.gauss():
        r = mer.radius_at(s)
        n1, n2 = 1.0 - s / mer.length, s / mer.length
        f[1] += pressure * n1 * (TWOPI * r * w)
        f[4] += pressure * n2 * (TWOPI * r * w)
    return f


# --------------------------------------------------------------------------- #
# assembled model
# --------------------------------------------------------------------------- #
class AxisymShell:
    """Axisymmetric shell-of-revolution model on a meridional node chain.

    Parameters
    ----------
    nodes : sequence of (r, z)
        Meridional nodes in the orientation convention of :class:`Meridian`.
    young, poisson, thickness : float
        Isotropic material and constant wall thickness.
    pressure : float
        Uniform outward normal pressure on the wall (0 = none).
    meridian_radius : float or sequence of float, optional
        ``R_s`` per element; a scalar is broadcast, ``None`` means straight
        generators everywhere.
    shear_integration : str
        ``"reduced"`` (production, SRI) or ``"full"`` (negative control).
    """

    def __init__(self, nodes, *, young: float, poisson: float, thickness: float,
                 pressure: float = 0.0, meridian_radius=None,
                 shear_integration: str = "reduced"):
        self.nodes = [(float(r), float(z)) for r, z in nodes]
        if len(self.nodes) < 2:
            raise ValueError("at least two meridional nodes are required")
        self.young, self.poisson = float(young), float(poisson)
        self.thickness = float(thickness)
        self.pressure = float(pressure)
        self.shear_integration = shear_integration
        n_el = len(self.nodes) - 1
        if meridian_radius is None:
            radii: list[float | None] = [None] * n_el
        elif isinstance(meridian_radius, (int, float)):
            radii = [float(meridian_radius)] * n_el
        else:
            radii = list(meridian_radius)
            if len(radii) != n_el:
                raise ValueError("meridian_radius must provide one value per element")
        self.meridians = [Meridian(r1, z1, r2, z2, None if r is None else float(r))
                          for (r1, z1), (r2, z2), r in zip(self.nodes[:-1], self.nodes[1:],
                                                           radii)]
        self.ndof = 3 * len(self.nodes)
        self.K = torch.zeros((self.ndof, self.ndof), dtype=torch.float64)
        self.f = torch.zeros(self.ndof, dtype=torch.float64)
        self._assemble()

    # -- assembly ----------------------------------------------------------- #
    def _assemble(self) -> None:
        for e, mer in enumerate(self.meridians):
            ke = element_stiffness(mer, self.young, self.poisson, self.thickness,
                                   shear_integration=self.shear_integration)
            fe = element_pressure_load(mer, self.pressure)
            dofs = [3 * e, 3 * e + 1, 3 * e + 2, 3 * e + 3, 3 * e + 4, 3 * e + 5]
            idx = torch.tensor(dofs)
            self.K[idx[:, None], idx[None, :]] += ke
            self.f[idx] += fe

    # -- helpers ------------------------------------------------------------ #
    def dof(self, node: int, component: str) -> int:
        comp = {"u": 0, "w": 1, "beta": 2}[component]
        return 3 * int(node) + comp

    def apply_axial_force(self, node: int, force_z: float) -> None:
        """Add an axial (``+z``) ring force at a node, e.g. a closed end cap.

        The local basis at the node is ``t = (cos phi, -sin phi)`` and
        ``n = (sin phi, cos phi)``; a global ``+z`` force therefore resolves to
        ``-F sin(phi)`` on ``u`` and ``+F cos(phi)`` on ``w``.
        """
        node = int(node)
        if node < 0 or node >= len(self.nodes):
            raise IndexError("node index out of range")
        if node == 0:
            phi = self.meridians[0].phi(0.0)
        else:
            ascent = self.meridians[node - 1]
            phi = ascent.phi(ascent.length)
        self.f[self.dof(node, "u")] += -force_z * math.sin(phi)
        self.f[self.dof(node, "w")] += force_z * math.cos(phi)

    def solve(self, dirichlet: dict[int, float] | None = None) -> torch.Tensor:
        """Solve ``K u = f`` with Dirichlet DOFs (dense reduced system).

        The models qualified here are small (O(100) DOFs); a dense reduction
        keeps the verification free of iterative-solver tolerance discussions.
        """
        dirichlet = dict(dirichlet or {})
        u = torch.zeros(self.ndof, dtype=torch.float64)
        fixed = sorted(dirichlet)
        for d in fixed:
            u[d] = float(dirichlet[d])
        free = torch.ones(self.ndof, dtype=torch.bool)
        if fixed:
            free[torch.tensor(fixed)] = False
        if not bool(free.any()):
            return u
        k_ff = self.K[free][:, free]
        rhs = self.f[free] - self.K[free][:, ~free] @ u[~free]
        u[free] = torch.linalg.solve(k_ff, rhs)
        return u

    def reaction(self, u: torch.Tensor) -> torch.Tensor:
        return self.K @ u - self.f

    # -- post-processing ---------------------------------------------------- #
    def element_response(self, u: torch.Tensor, element: int) -> dict:
        """Generalised forces at the two Gauss points of one element."""
        mer = self.meridians[int(element)]
        d = _constitutive(self.young, self.poisson, self.thickness)
        dofs = torch.tensor([3 * element, 3 * element + 1, 3 * element + 2,
                             3 * element + 3, 3 * element + 4, 3 * element + 5])
        ue = u[dofs]
        rows = {"s": [], "r": [], "phi": [], "N_s": [], "N_theta": [],
                "M_s": [], "M_theta": [], "Q": []}
        for s, _w in mer.gauss():
            b = _b_matrix(mer, s)
            strain = b @ ue
            mem = d[:4, :4] @ strain[:4]
            rows["s"].append(s)
            rows["r"].append(mer.radius_at(s))
            rows["phi"].append(mer.phi(s))
            rows["N_s"].append(float(mem[0]))
            rows["N_theta"].append(float(mem[1]))
            rows["M_s"].append(float(mem[2]))
            rows["M_theta"].append(float(mem[3]))
            rows["Q"].append(float(d[4, 4] * strain[4]))
        return rows

    def field(self, u: torch.Tensor) -> dict:
        """Concatenate the Gauss-point response of every element."""
        out = {k: [] for k in ("s", "r", "phi", "N_s", "N_theta", "M_s", "M_theta", "Q")}
        for e in range(len(self.meridians)):
            resp = self.element_response(u, e)
            for k in out:
                out[k].extend(resp[k])
        return out

    def strain_energy(self, u: torch.Tensor) -> float:
        return float(0.5 * u @ (self.K @ u))