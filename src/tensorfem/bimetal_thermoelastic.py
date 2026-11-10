"""Thermo-elastic bimetal strip: Timoshenko (1925) closed form vs a 2-D FEM.

Why this module exists
----------------------
The package verified 3-D thermo-elasticity against *restraint* anchors
(``polar_thermal_stress``: ``E alpha dT/(1-nu)`` on a biaxially clamped plate,
``thermoelastic_annulus``) and 1-D *steady conduction* against closed forms.
It had no anchor for the one thermo-elastic configuration in which a uniform
temperature change produces a **self-equilibrated bending** field with a
classical exact solution: a two-layer (bimetal) strip heated while
mechanically free.  That case exercises the thermal strain load, the
traction-free surfaces, the inter-layer equilibrium and the bending
kinematics at the same time, and its solution has a long pedigree
(Timoshenko, *Analysis of bi-metal thermostats*, J. Opt. Soc. Am. Rev. Sci.
Instrum. 11 (1925) 233; re-derived in Clyne & Hull, Prog. Mater. Sci. 40
(1996) 81).

The closed form
---------------
With ``m = t1/t2`` (thickness ratio), ``n = E1/E2`` (modulus ratio),
``h = t1 + t2`` and ``dT`` the uniform temperature rise::

    kappa = 6 (alpha1 - alpha2) dT (1 + m)**2
            / ( h [ 3 (1 + m)**2 + (1 + m n) (m**2 + 1/(m n)) ] )

The same expression is produced here by a first-principles 2x2 closed-section
equilibrium (:func:`section_equilibrium`: resultant force and moment about the
interface both vanish).  Keeping both derivations side by side means a
transcription slip in either one is caught at machine precision instead of
being absorbed into a "FEM vs analysis" error budget
(:func:`curvature_agreement`).

Kinematic convention (signs matter)
-----------------------------------
* layer 1 is the **top** layer (``+y``), layer 2 the bottom one, the interface
  is at ``y = 0`` and ``y`` is measured upwards from it;
* ``eps_xx(y) = eps0 + kappa y`` and ``kappa > 0`` when ``alpha1 > alpha2``:
  the top layer wants to be longer, so it ends up on the outside of the arc;
* Euler-Bernoulli gives ``eps_xx = -y w''`` so ``w'' = -kappa`` and the strip
  built in at ``x = 0`` and free at ``x = L`` deflects as
  ``w(x) = -kappa x**2 / 2``; the **signed** tip deflection is
  ``-kappa L**2 / 2`` (for ``alpha1 > alpha2`` the free end dips).

FEM side
--------
:func:`solve_bimetal_strip` builds a structured **Q9** (biquadratic Lagrange,
3x3 Gauss, exactly integrated) plane-stress mesh: two bonded layers, a shared
node line on the interface, a uniform ``dT`` applied as a thermal strain load,
the whole ``x = 0`` edge clamped, all other edges traction free.  Q9 is used
deliberately: bilinear Q4 elements lock in bending, so their deflection error
would be dominated by spurious stiffness instead of by discretisation of the
exact solution, which makes an observed convergence order meaningless.

What is *not* modelled (known limitations)
------------------------------------------
* small strain, linear elasticity, temperature-independent ``E, alpha``; no
  plasticity, creep or buckling of the compressed layer;
* ``nu`` is equal in both layers -- the classical formula is a beam result and
  is independent of ``nu`` (plane stress, ``E`` enters unscaled), so a Poisson
  mismatch would add a 2-D effect outside the anchor's assumptions;
* the closed form is the pure-bending state: it is *not* valid in the
  Saint-Venant boundary layer at the clamped edge (length ``O(h)``) nor at the
  free ends of the interface (stress singularity there).  The studies
  therefore use the tip deflection (a global quantity) and the through-
  thickness stress at mid-length, and the energy gate is restricted to the
  region outside the clamped boundary layer;
* plane stress (a strip, not a plate in plane strain), matching the beam
  assumptions of the closed form.

Every number in the evidence JSON, report and tests is computed by this
module; nothing is hand copied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

FLOAT = torch.float64

# ---------------------------------------------------------------------------
# analytical solution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BimetalCase:
    """Geometry, material and thermal load of one bimetal strip."""

    m: float = 1.0           # thickness ratio t1/t2
    n: float = 1.0           # modulus ratio E1/E2
    dT: float = 100.0        # uniform temperature rise [K]
    h: float = 2.0e-3        # total thickness [m]
    length: float = 40.0e-3  # strip length [m]  (L/h = 20 by default)
    E1: float = 200.0e9      # layer-1 Young's modulus [Pa]
    alpha1: float = 1.7e-5   # layer-1 CTE [1/K] (higher-expansion, top)
    alpha2: float = 1.2e-5   # layer-2 CTE [1/K] (lower-expansion, bottom)
    nu: float = 0.3          # Poisson ratio, equal in both layers

    @property
    def t1(self) -> float:
        return self.m * self.h / (1.0 + self.m)

    @property
    def t2(self) -> float:
        return self.h / (1.0 + self.m)

    @property
    def E2(self) -> float:
        return self.E1 / self.n

    @property
    def slenderness(self) -> float:
        return self.length / self.h

    def validate(self) -> None:
        if self.m <= 0 or self.n <= 0:
            raise ValueError("m and n must be positive")
        if self.h <= 0 or self.length <= 0:
            raise ValueError("h and length must be positive")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError("nu must be in (-1, 0.5)")


@dataclass(frozen=True)
class BimetalSolution:
    """Closed-form (Timoshenko) field of a free bimetal strip."""

    case: BimetalCase
    curvature: float                 # kappa [1/m]
    axial_strain: float              # eps0 at the interface
    tip_deflection: float            # signed, w(L) = -kappa L^2/2 [m]
    sigma_interface_top: float       # sigma_xx in layer 1 at y = 0+
    sigma_interface_bottom: float    # sigma_xx in layer 2 at y = 0-
    sigma_surface_top: float         # sigma_xx at y = +t1
    sigma_surface_bottom: float      # sigma_xx at y = -t2
    strain_energy_per_length: float  # per unit width and length [J/m^2]
    strain_energy: float             # over the strip length, unit width [J/m]

    @property
    def max_abs_sigma_interface(self) -> float:
        return max(abs(self.sigma_interface_top), abs(self.sigma_interface_bottom))

    def strain_energy_in(self, x_lo: float, x_hi: float) -> float:
        """Strain energy per unit width carried by the section ``[x_lo, x_hi]``."""
        return self.strain_energy_per_length * (x_hi - x_lo)

    def sigma_xx(self, y: float) -> float:
        """Closed-form ``sigma_xx(y)`` (any section outside the boundary layers)."""
        E, alpha = (
            (self.case.E1, self.case.alpha1) if y >= 0.0 else (self.case.E2, self.case.alpha2)
        )
        return E * (self.axial_strain + self.curvature * y - alpha * self.case.dT)


def timoshenko_curvature(m: float, n: float, dT: float, h: float,
                         alpha1: float = 1.0, alpha2: float = 0.0) -> float:
    """Timoshenko (1925) bimetal curvature.

    The defaults reproduce the classical unit thermal-mismatch form
    ``kappa = 6 dT (1+m)^2 / (h [3(1+m)^2 + (1+mn)(m^2 + 1/(mn))])``.
    """
    num = 6.0 * (alpha1 - alpha2) * dT * (1.0 + m) ** 2
    den = h * (3.0 * (1.0 + m) ** 2 + (1.0 + m * n) * (m * m + 1.0 / (m * n)))
    return num / den


def section_equilibrium(case: BimetalCase) -> tuple[float, float]:
    """Independent derivation of ``(eps0, kappa)`` from a 2x2 section balance.

    Resultant axial force and moment about the interface of the layered
    section are required to vanish.  This *is* the classical derivation of the
    Timoshenko formula, written without any of its algebra, so that a
    transcription slip in :func:`timoshenko_curvature` cannot hide behind an
    FEM error budget.
    """
    case.validate()
    t1, t2 = case.t1, case.t2
    E1, E2 = case.E1, case.E2
    e1 = case.alpha1 * case.dT       # free thermal strain, layer 1
    e2 = case.alpha2 * case.dT       # free thermal strain, layer 2
    a11 = E1 * t1 + E2 * t2
    a12 = 0.5 * (E1 * t1 * t1 - E2 * t2 * t2)
    b1 = E1 * t1 * e1 + E2 * t2 * e2
    a21 = a12
    a22 = (E1 * t1**3 + E2 * t2**3) / 3.0
    b2 = 0.5 * (E1 * t1 * t1 * e1 - E2 * t2 * t2 * e2)
    det = a11 * a22 - a12 * a21
    eps0 = (b1 * a22 - a12 * b2) / det
    kappa = (a11 * b2 - b1 * a21) / det
    return eps0, kappa


def analytical_solution(case: BimetalCase) -> BimetalSolution:
    """Full closed-form field: curvature, strains, stresses and energy."""
    case.validate()
    eps0, kappa_eq = section_equilibrium(case)
    kappa_closed = timoshenko_curvature(
        case.m, case.n, case.dT, case.h, case.alpha1, case.alpha2
    )
    tol = 1e-10 * max(abs(kappa_eq), 1e-30)
    if abs(kappa_closed - kappa_eq) > tol:
        raise AssertionError(
            "Timoshenko closed form disagrees with section equilibrium: "
            f"{kappa_closed!r} vs {kappa_eq!r}"
        )
    kappa = kappa_eq
    t1, t2 = case.t1, case.t2
    E1 = case.E1

    def sigma(y: float, layer: int) -> float:
        E, alpha = (E1, case.alpha1) if layer == 1 else (case.E2, case.alpha2)
        return E * (eps0 + kappa * y - alpha * case.dT)

    energy_density = 0.0
    for lo, hi, E, alpha in ((0.0, t1, E1, case.alpha1), (-t2, 0.0, case.E2, case.alpha2)):
        e0 = eps0 - alpha * case.dT
        # int (e0 + kappa y)^2 dy over one layer
        integ = (
            e0 * e0 * (hi - lo)
            + e0 * kappa * (hi * hi - lo * lo)
            + kappa * kappa * (hi**3 - lo**3) / 3.0
        )
        energy_density += 0.5 * E * integ
    return BimetalSolution(
        case=case,
        curvature=kappa,
        axial_strain=eps0,
        tip_deflection=-kappa * case.length**2 / 2.0,
        sigma_interface_top=sigma(0.0, 1),
        sigma_interface_bottom=sigma(0.0, 2),
        sigma_surface_top=sigma(t1, 1),
        sigma_surface_bottom=sigma(-t2, 2),
        strain_energy_per_length=energy_density,
        strain_energy=energy_density * case.length,
    )


def parameter_cases(dT: float = 100.0) -> list[BimetalCase]:
    """The 9 ``(m, n)`` combinations of the campaign node at ``dT``."""
    return [BimetalCase(m=m, n=n, dT=dT) for m in (0.5, 1.0, 2.0) for n in (0.5, 1.0, 3.0)]


def temperature_cases() -> list[BimetalCase]:
    """``dT`` ladder at the reference ``(m, n) = (1, 1)``."""
    return [BimetalCase(dT=dT) for dT in (50.0, 100.0, 200.0)]


def all_cases() -> list[BimetalCase]:
    """The 9 (m, n) combinations plus the 3-``dT`` ladder (12 cases total)."""
    return parameter_cases() + temperature_cases()


def curvature_agreement(cases: list[BimetalCase] | None = None) -> float:
    """Max relative gap between the closed form and the section equilibrium."""
    cases = cases if cases is not None else all_cases()
    worst = 0.0
    for case in cases:
        k_closed = timoshenko_curvature(
            case.m, case.n, case.dT, case.h, case.alpha1, case.alpha2
        )
        _, k_eq = section_equilibrium(case)
        if k_eq == 0.0:
            worst = max(worst, abs(k_closed))
        else:
            worst = max(worst, abs(k_closed - k_eq) / abs(k_eq))
    return worst


# ---------------------------------------------------------------------------
# Q9 (biquadratic Lagrange) plane-stress FE with a thermal strain load
# ---------------------------------------------------------------------------

_G = math.sqrt(3.0 / 5.0)
_GAUSS = [(xi, eta) for xi in (-_G, 0.0, _G) for eta in (-_G, 0.0, _G)]
_WEIGHTS = [wx * wy for wx in (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
            for wy in (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)]


def _lagrange(t: float) -> tuple[list[float], list[float]]:
    """1-D quadratic Lagrange basis on the nodes ``-1, 0, 1`` and its derivative."""
    values = [0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)]
    derivs = [t - 0.5, -2.0 * t, t + 0.5]
    return values, derivs


def q9_shape(xi: float, eta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q9 shape functions and natural derivatives.

    Local numbering ``p = 3*a + b`` maps to the mesh-grid offset
    ``(2i + a, 2j + b)`` (``a`` along ``x``/``xi``, ``b`` along ``y``/``eta``):
    corners at ``p = 0, 2, 6, 8``, edge mid-sides at ``p = 1, 3, 5, 7``
    (left, bottom, right, top) and the centre node at ``p = 4``.
    """
    lx, dlx = _lagrange(xi)
    ly, dly = _lagrange(eta)
    n = torch.tensor([lx[a] * ly[b] for a in range(3) for b in range(3)], dtype=FLOAT)
    dn_dxi = torch.tensor([dlx[a] * ly[b] for a in range(3) for b in range(3)], dtype=FLOAT)
    dn_deta = torch.tensor([lx[a] * dly[b] for a in range(3) for b in range(3)], dtype=FLOAT)
    return n, dn_dxi, dn_deta


@dataclass(frozen=True)
class BimetalMesh:
    nodes: torch.Tensor        # [n_nodes, 2], global index = j * (2*nx+1) + i
    elements: torch.Tensor     # [n_elems, 9] Q9 connectivity, order p = 3*a + b
    material: torch.Tensor     # [n_elems] 0 = layer 2 (bottom), 1 = layer 1 (top)
    nx: int
    ny_per_layer: int
    x_lines: torch.Tensor
    y_lines: torch.Tensor
    y_interface: float

    @property
    def n_dofs(self) -> int:
        return 2 * self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    def element_index(self, i: int, j: int) -> int:
        return j * self.nx + i


def graded_x_lines(length: float, nx: int, grading: float = 1.0) -> torch.Tensor:
    """Element boundaries along ``x``, geometrically graded toward ``x = 0``.

    ``grading = 1`` gives a uniform mesh.  ``grading > 1`` clusters elements at
    the clamped edge, which is where the pure-bending closed form fails (the
    Saint-Venant boundary layer of a clamped bimetal strip is ``O(h)`` thick);
    a graded family is the standard remedy for such a layer and is what makes
    the observed order of the code meaningful.
    """
    if nx < 1:
        raise ValueError("nx must be >= 1")
    if grading <= 0.0:
        raise ValueError("grading must be positive")
    if abs(grading - 1.0) < 1e-12:
        return torch.linspace(0.0, length, nx + 1, dtype=FLOAT)
    k = torch.arange(nx + 1, dtype=FLOAT)
    x = (grading ** k - 1.0) / (grading**nx - 1.0) * length
    return x


def strip_mesh(case: BimetalCase, nx: int, ny_per_layer: int,
               grading: float = 1.0) -> BimetalMesh:
    """Structured Q9 mesh: ``ny_per_layer`` elements per layer, shared interface."""
    case.validate()
    if nx < 1 or ny_per_layer < 1:
        raise ValueError("nx and ny_per_layer must be >= 1")
    nytot = 2 * ny_per_layer
    x_lines = graded_x_lines(case.length, nx, grading)
    y_bot = torch.linspace(-case.t2, 0.0, ny_per_layer + 1, dtype=FLOAT)
    y_top = torch.linspace(0.0, case.t1, ny_per_layer + 1, dtype=FLOAT)
    y_lines = torch.cat((y_bot, y_top[1:]))
    x_half = torch.empty(2 * nx + 1, dtype=FLOAT)
    x_half[0::2] = x_lines
    x_half[1::2] = 0.5 * (x_lines[:-1] + x_lines[1:])
    y_half = torch.empty(2 * nytot + 1, dtype=FLOAT)
    y_half[0::2] = y_lines
    y_half[1::2] = 0.5 * (y_lines[:-1] + y_lines[1:])
    yy, xx = torch.meshgrid(y_half, x_half, indexing="ij")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), 1)
    npx = 2 * nx + 1
    cells, material = [], []
    for j in range(nytot):
        for i in range(nx):
            cells.append([(2 * j + b) * npx + (2 * i + a) for a in range(3) for b in range(3)])
            material.append(0 if j < ny_per_layer else 1)
    return BimetalMesh(
        nodes=nodes,
        elements=torch.tensor(cells, dtype=torch.long),
        material=torch.tensor(material, dtype=torch.long),
        nx=nx,
        ny_per_layer=ny_per_layer,
        x_lines=x_lines,
        y_lines=y_lines,
        y_interface=0.0,
    )


def constitutive(case: BimetalCase, mesh: BimetalMesh) -> tuple[torch.Tensor, torch.Tensor]:
    """Plane-stress ``D`` matrix and thermal strain ``eps_th`` per element."""
    ne = mesh.n_elements
    top = mesh.material == 1
    E = torch.where(top, torch.tensor(case.E1, dtype=FLOAT), torch.tensor(case.E2, dtype=FLOAT))
    alpha = torch.where(top, torch.tensor(case.alpha1, dtype=FLOAT),
                        torch.tensor(case.alpha2, dtype=FLOAT))
    fac = E / (1.0 - case.nu * case.nu)
    D = torch.zeros((ne, 3, 3), dtype=FLOAT)
    D[:, 0, 0] = fac
    D[:, 1, 1] = fac
    D[:, 0, 1] = fac * case.nu
    D[:, 1, 0] = fac * case.nu
    D[:, 2, 2] = fac * (1.0 - case.nu) / 2.0
    eth = torch.zeros((ne, 3), dtype=FLOAT)
    eth[:, 0] = alpha * case.dT
    eth[:, 1] = alpha * case.dT
    return D, eth


def _gauss_data(case: BimetalCase, mesh: BimetalMesh) -> tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor]:
    """Per-Gauss-point ``B``, jacobian determinant and physical coordinates."""
    coords = mesh.nodes[mesh.elements]
    n_gp = len(_GAUSS)
    B = torch.zeros((mesh.n_elements, n_gp, 3, 18), dtype=FLOAT)
    detJ = torch.zeros((mesh.n_elements, n_gp), dtype=FLOAT)
    xs = torch.zeros((mesh.n_elements, n_gp), dtype=FLOAT)
    ys = torch.zeros((mesh.n_elements, n_gp), dtype=FLOAT)
    for g, (xi, eta) in enumerate(_GAUSS):
        n, dn_dxi, dn_deta = q9_shape(xi, eta)
        dx_dxi = coords[:, :, 0] @ dn_dxi
        dy_dxi = coords[:, :, 1] @ dn_dxi
        dx_deta = coords[:, :, 0] @ dn_deta
        dy_deta = coords[:, :, 1] @ dn_deta
        det = dx_dxi * dy_deta - dy_dxi * dx_deta
        if bool((det <= 0).any()):
            raise ValueError("non-positive jacobian: inverted Q9 element")
        detJ[:, g] = det
        dN_dx = (dn_dxi[None, :] * dy_deta[:, None]
                 - dn_deta[None, :] * dy_dxi[:, None]) / det[:, None]
        dN_dy = (-dn_dxi[None, :] * dx_deta[:, None]
                 + dn_deta[None, :] * dx_dxi[:, None]) / det[:, None]
        for a in range(9):
            B[:, g, 0, 2 * a] = dN_dx[:, a]
            B[:, g, 1, 2 * a + 1] = dN_dy[:, a]
            B[:, g, 2, 2 * a] = dN_dy[:, a]
            B[:, g, 2, 2 * a + 1] = dN_dx[:, a]
        xs[:, g] = coords[:, :, 0] @ n
        ys[:, g] = coords[:, :, 1] @ n
    return B, detJ, xs, ys


@dataclass
class StripResult:
    """Everything measured on one FEM mesh."""

    case: BimetalCase
    mesh: BimetalMesh
    displacement: torch.Tensor
    tip_deflection: float            # signed v at (L, 0)
    mid_deflection: float            # signed v at the node nearest x = L/2, y = 0
    mid_x: float                     # actual x of that node (graded meshes)
    sigma_interface_top: float       # sigma_xx in layer 1 at (L/2, 0+)
    sigma_interface_bottom: float    # sigma_xx in layer 2 at (L/2, 0-)
    strain_energy: float             # whole strip, unit width
    thermal_pool: float              # 0.5 * int eps_th^T D eps_th dV  ("C")
    load_work_half: float            # 0.5 * u^T f_th
    gauss_x: torch.Tensor
    gauss_y: torch.Tensor
    gauss_sigma_xx: torch.Tensor
    gauss_energy_density: torch.Tensor
    element_D: torch.Tensor
    element_eth: torch.Tensor

    @property
    def energy_identity_gap(self) -> float:
        """Relative residual of the discrete thermo-elastic energy identity.

        For a thermal load the elastic energy is *not* ``u^T f_th / 2`` (that
        Clapeyron form holds for mechanical loads): expanding
        ``0.5 (eps - eps_th)^T D (eps - eps_th)`` and using ``K u = f_th`` gives

            U_el = C - 0.5 u^T f_th,      C = 0.5 int eps_th^T D eps_th dV.

        The residual of that identity measures assembly consistency; it must sit
        at solver round-off, otherwise the element matrices and the recovery
        are inconsistent with each other.
        """
        return abs(self.strain_energy - (self.thermal_pool - self.load_work_half)) / \
            abs(self.strain_energy)

    @property
    def curvature_estimate(self) -> float:
        """Curvature implied by the tip deflection, ``-2 w(L)/L^2``."""
        return -2.0 * self.tip_deflection / self.case.length**2

    @property
    def max_abs_sigma_interface(self) -> float:
        return max(abs(self.sigma_interface_top), abs(self.sigma_interface_bottom))

    def strain_energy_above(self, x_lo: float) -> float:
        """Strain energy of ``x >= x_lo`` by Gauss-point restriction.

        The closed form is the pure-bending state, which is not valid inside
        the Saint-Venant boundary layer at the clamped edge; energy comparisons
        therefore exclude that layer instead of hiding it in the tolerance.
        """
        mask = self.gauss_x >= x_lo - 1e-15
        return float(self.gauss_energy_density[mask].sum())


def solve_bimetal_strip(
    case: BimetalCase, nx: int, ny_per_layer: int, *, thickness: float = 1.0,
    grading: float = 1.0,
) -> StripResult:
    """Solve the clamped-free bimetal strip under a uniform temperature rise."""
    case.validate()
    mesh = strip_mesh(case, nx, ny_per_layer, grading)
    B, detJ, xs, ys = _gauss_data(case, mesh)
    D, eth = constitutive(case, mesh)
    w = torch.tensor(_WEIGHTS, dtype=FLOAT)[None, :]
    dV = detJ * w * thickness

    DB = torch.einsum("eab,egbj->egaj", D, B)                # [ne, gp, 3, 18]
    ke = torch.einsum("eg,egai,egaj->eij", dV, DB, B)        # [ne, 18, 18]
    fth = torch.einsum("eab,eb->ea", D, eth)                 # [ne, 3]
    fe = torch.einsum("eg,egai,ea->ei", dV, B, fth)          # [ne, 18]

    edofs = torch.empty((mesh.n_elements, 18), dtype=torch.long)
    edofs[:, 0::2] = 2 * mesh.elements
    edofs[:, 1::2] = 2 * mesh.elements + 1
    n_dofs = mesh.n_dofs
    f = torch.zeros(n_dofs, dtype=FLOAT).index_add_(0, edofs.reshape(-1), fe.reshape(-1))

    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    rows = np.repeat(edofs.numpy(), 18, axis=1).reshape(-1)
    cols = np.tile(edofs.numpy(), (1, 18)).reshape(-1)
    K = sp.coo_matrix((ke.numpy().reshape(-1), (rows, cols)),
                      shape=(n_dofs, n_dofs)).tocsr()

    clamped_nodes = torch.nonzero(mesh.nodes[:, 0] <= 1e-15).reshape(-1)
    fixed = torch.cat((2 * clamped_nodes, 2 * clamped_nodes + 1)).sort().values
    free_mask = torch.ones(n_dofs, dtype=torch.bool)
    free_mask[fixed] = False
    free = torch.nonzero(free_mask).reshape(-1)

    u = torch.zeros(n_dofs, dtype=FLOAT)
    u[free] = torch.from_numpy(
        np.asarray(spla.spsolve(K[free.numpy()][:, free.numpy()].tocsc(), f[free].numpy()),
                   dtype=np.float64)
    )

    npx = 2 * mesh.nx + 1
    j_int = 2 * ny_per_layer                                  # node row at y = 0
    tip_node = j_int * npx + 2 * mesh.nx                      # (x = L, y = 0)
    xs_half = mesh.nodes[:npx, 0]
    i_mid = int(torch.argmin(torch.abs(xs_half - 0.5 * case.length)))
    mid_node = j_int * npx + i_mid                            # node nearest (L/2, 0)
    tip_deflection = float(u[2 * tip_node + 1])
    mid_deflection = float(u[2 * mid_node + 1])
    mid_x = float(xs_half[i_mid])

    ue = u[edofs].reshape(mesh.n_elements, 18)
    strain = torch.einsum("egij,ej->egi", B, ue)
    mech = strain - eth[:, None, :]
    stress = torch.einsum("eij,egj->egi", D, mech)
    energy_density = 0.5 * (mech * stress).sum(dim=2) * dV

    result = StripResult(
        case=case,
        mesh=mesh,
        displacement=u,
        tip_deflection=tip_deflection,
        mid_deflection=mid_deflection,
        mid_x=mid_x,
        sigma_interface_top=float("nan"),
        sigma_interface_bottom=float("nan"),
        strain_energy=float(energy_density.sum()),
        thermal_pool=0.5 * float((torch.einsum("eij,ei,ej->e", D, eth, eth)[:, None] * dV).sum()),
        load_work_half=0.5 * float(u @ f),
        gauss_x=xs.reshape(-1),
        gauss_y=ys.reshape(-1),
        gauss_sigma_xx=stress[:, :, 0].reshape(-1),
        gauss_energy_density=energy_density.reshape(-1),
        element_D=D,
        element_eth=eth,
    )
    bottom, top = stress_xx_at(result, 0.5 * case.length, 0.0)
    result.sigma_interface_bottom = bottom
    result.sigma_interface_top = top
    return result


def stress_xx_at(result: StripResult, x: float, y: float) -> tuple[float, float]:
    """``sigma_xx`` at ``(x, y)``: ``(from the bottom layer, from the top layer)``.

    At ``y = 0`` both layers are sampled separately, because the interface
    stress is discontinuous there.  For any other ``y`` the first entry holds
    the value and the second is ``nan``.
    """
    mesh = result.mesh
    i = int(torch.clamp(torch.searchsorted(mesh.x_lines, torch.tensor(x, dtype=FLOAT),
                                           right=True) - 1, 0, mesh.nx - 1))
    j = int(torch.clamp(torch.searchsorted(mesh.y_lines, torch.tensor(y, dtype=FLOAT),
                                           right=True) - 1, 0, 2 * mesh.ny_per_layer - 1))
    xi = 2.0 * (x - float(mesh.x_lines[i])) / float(mesh.x_lines[i + 1] - mesh.x_lines[i]) - 1.0

    def sample(j_elem: int) -> float:
        eta = 2.0 * (y - float(mesh.y_lines[j_elem])) / float(
            mesh.y_lines[j_elem + 1] - mesh.y_lines[j_elem]) - 1.0
        e = mesh.element_index(i, int(j_elem))
        conn = mesh.elements[e]
        _, dn_dxi, dn_deta = q9_shape(xi, eta)
        xy = mesh.nodes[conn]
        dx_dxi = float(xy[:, 0] @ dn_dxi)
        dy_dxi = float(xy[:, 1] @ dn_dxi)
        dx_deta = float(xy[:, 0] @ dn_deta)
        dy_deta = float(xy[:, 1] @ dn_deta)
        det = dx_dxi * dy_deta - dy_dxi * dx_deta
        dN_dx = (dn_dxi * dy_deta - dn_deta * dy_dxi) / det
        dN_dy = (-dn_dxi * dx_deta + dn_deta * dx_dxi) / det
        ue = torch.empty(18, dtype=FLOAT)
        ue[0::2] = result.displacement[2 * conn]
        ue[1::2] = result.displacement[2 * conn + 1]
        strain = torch.tensor(
            [float(dN_dx @ ue[0::2]),
             float(dN_dy @ ue[1::2]),
             float(dN_dy @ ue[0::2]) + float(dN_dx @ ue[1::2])],
            dtype=FLOAT,
        )
        mech = strain - result.element_eth[e]
        return float(result.element_D[e][0] @ mech)

    if abs(y - mesh.y_interface) < 1e-15:
        return sample(mesh.ny_per_layer - 1), sample(mesh.ny_per_layer)
    return sample(j), float("nan")


def section_resultants(result: StripResult, x: float | None = None) -> dict:
    """Resultant axial force and moment of the FE stress field at section ``x``.

    The free end is traction free, so the internal force and moment resultants of
    the exact solution vanish at *every* section outside the clamped layer.  This
    is an equilibrium check on the FE field itself (it never uses the closed
    form).

    Implementation note: ``sigma_xx`` is linear in ``y`` inside each layer, so two
    samples per layer pin it exactly and the resulting polynomial integrals are
    evaluated **analytically**.  A trapezoidal rule on ``y * sigma_xx`` is *not*
    exact (the integrand is quadratic) and leaves a mesh-independent spurious
    moment of ``1.6e-4`` of the scale -- a trap that makes a correct field look
    out of balance.
    """
    case = result.case
    x = 0.5 * case.length if x is None else x
    force = moment = 0.0
    for lo, hi in ((-case.t2, 0.0), (0.0, case.t1)):
        span = hi - lo
        # two strictly interior samples pin the (linear) layer stress without
        # having to disambiguate the shared interface node
        y1, y2 = lo + 0.25 * span, lo + 0.75 * span
        s1 = stress_xx_at(result, x, y1)[0]
        s2 = stress_xx_at(result, x, y2)[0]
        slope = (s2 - s1) / (y2 - y1)
        icpt = s1 - slope * y1                      # sigma of this line at y = 0
        force += icpt * span + 0.5 * slope * (hi**2 - lo**2)
        moment += 0.5 * icpt * (hi**2 - lo**2) + slope * (hi**3 - lo**3) / 3.0
    e_scale = max(case.E1, case.E2) * abs(case.alpha1 - case.alpha2) * abs(case.dT)
    force_scale = e_scale * case.h
    moment_scale = e_scale * case.h * case.h
    return {
        "x_over_L": x / case.length,
        "axial_force_per_width": force,
        "moment_per_width": moment,
        "force_scale": force_scale,
        "moment_scale": moment_scale,
        "force_rel": abs(force) / force_scale,
        "moment_rel": abs(moment) / moment_scale,
    }


def section_profile(result: StripResult, x: float, n_points: int = 41) -> dict:
    """Through-thickness ``sigma_xx(y)`` at section ``x`` (FEM)."""
    mesh = result.mesh
    ys = torch.linspace(float(mesh.y_lines[0]), float(mesh.y_lines[-1]), n_points,
                        dtype=FLOAT)
    values = []
    for y in ys:
        yy = float(y)
        if abs(yy - mesh.y_interface) < 1e-14:      # exactly on the interface
            yy = mesh.y_interface + 1e-12
        values.append(stress_xx_at(result, x, yy)[0])
    return {"x": x, "y": ys, "sigma_xx": torch.tensor(values, dtype=FLOAT)}


# ---------------------------------------------------------------------------
# convergence / GCI helpers
# ---------------------------------------------------------------------------


def relative_error(value: float, reference: float) -> float:
    """Signed relative error ``(value - reference)/|reference|``."""
    if reference == 0.0:
        raise ValueError("relative error against an exact zero reference")
    return (value - reference) / abs(reference)


def observed_order(f_coarse: float, f_mid: float, f_fine: float, ratio: float = 2.0) -> float:
    """Grid-triplet observed order of convergence (Celik et al. 2008)."""
    d1 = f_mid - f_coarse
    d2 = f_fine - f_mid
    if d1 == 0.0 or d2 == 0.0 or d1 * d2 <= 0.0:
        return float("nan")
    # e ~ h^p  =>  d2/d1 = 2**-p  =>  p = log(|d1/d2|)/log(r)
    return math.log(abs(d1 / d2)) / math.log(ratio)


def gci_fine(f_coarse: float, f_mid: float, f_fine: float, ratio: float = 2.0,
             safety: float = 1.25) -> tuple[float, float]:
    """``(p, GCI_fine)`` for a grid triplet; the GCI is returned as a fraction."""
    p = observed_order(f_coarse, f_mid, f_fine, ratio)
    if not math.isfinite(p):
        return p, float("nan")
    if f_fine == 0.0:
        return p, float("nan")
    e = abs((f_fine - f_mid) / f_fine)
    return p, safety * e / (ratio**p - 1.0)


@dataclass(frozen=True)
class MeshLevel:
    nx: int
    ny_per_layer: int


DEFAULT_LEVELS: tuple[MeshLevel, ...] = (
    MeshLevel(8, 2),
    MeshLevel(16, 4),
    MeshLevel(32, 8),
    MeshLevel(64, 16),
)


def convergence_study(
    case: BimetalCase | None = None,
    nxs: tuple[int, ...] = (8, 16, 32, 64, 128),
    ny_per_layer: int = 4,
    grading: float = 1.0,
) -> dict:
    """h-refinement (in ``x``) of the tip deflection and of the mid-span deflection.

    ``ny_per_layer`` has no effect on the answer -- the Q9 element reproduces the
    through-thickness linear bending strain exactly (verified separately by
    ``thickness_independence``) -- so the family refines ``x`` only, which is
    where the clamped boundary layer lives.  The observed order ``p`` is computed
    from the grid triplet differences (Celik et al. 2008), i.e. it measures the
    discretisation order of the code; the constant ``O(h/L)`` offset between the
    2-D clamped model and the beam closed form cancels in the differences.
    """
    case = case or BimetalCase()
    ref = analytical_solution(case)
    rows = []
    for nx in nxs:
        res = solve_bimetal_strip(case, nx, ny_per_layer, grading=grading)
        rows.append({
            "nx": nx,
            "ny_per_layer": ny_per_layer,
            "grading": grading,
            "elements": res.mesh.n_elements,
            "dofs": res.mesh.n_dofs,
            "dx_min_over_h": float(res.mesh.x_lines[1] - res.mesh.x_lines[0]) / case.h,
            "tip_deflection": res.tip_deflection,
            "tip_rel_error": relative_error(res.tip_deflection, ref.tip_deflection),
            "mid_deflection": res.mid_deflection,
            "mid_x_over_L": res.mid_x / case.length,
            "mid_rel_error": relative_error(res.mid_deflection,
                                            -ref.curvature * res.mid_x**2 / 2.0),
            "sigma_interface_top": res.sigma_interface_top,
            "sigma_top_rel_error": relative_error(res.sigma_interface_top,
                                                  ref.sigma_interface_top),
            "energy_identity_gap": res.energy_identity_gap,
        })
    out = {"case": case, "reference": ref, "grading": grading, "levels": rows}
    for qoi in ("tip_deflection", "mid_deflection"):
        seq = [r[qoi] for r in rows]
        triplets = []
        for a in range(len(seq) - 2):
            p, gci = gci_fine(seq[a], seq[a + 1], seq[a + 2])
            triplets.append({"levels": f"{nxs[a]}-{nxs[a+1]}-{nxs[a+2]}", "order": p, "gci": gci})
        out[qoi] = {"triplets": triplets,
                    "order": triplets[-1]["order"],
                    "gci": triplets[-1]["gci"]}
    return out


def thickness_independence(
    case: BimetalCase | None = None, nx: int = 48, nys: tuple[int, ...] = (2, 4, 8, 16)
) -> list[dict]:
    """Show that the through-thickness mesh carries no discretisation error.

    Q9 reproduces a strain that is linear in ``y`` exactly, and the closed-form
    field of a free bimetal strip is exactly that (``sigma_yy = tau_xy = 0``),
    so refining ``ny`` must not move the answer beyond round-off.  This is what
    licenses the ``x``-only refinement family of :func:`convergence_study`.
    """
    case = case or BimetalCase()
    ref = analytical_solution(case)
    out = []
    for ny in nys:
        res = solve_bimetal_strip(case, nx, ny)
        out.append({
            "ny_per_layer": ny,
            "dofs": res.mesh.n_dofs,
            "tip_deflection": res.tip_deflection,
            "tip_rel_error": relative_error(res.tip_deflection, ref.tip_deflection),
            "sigma_interface_top": res.sigma_interface_top,
            "sigma_top_rel_error": relative_error(res.sigma_interface_top,
                                                  ref.sigma_interface_top),
        })
    return out


def slenderness_study(
    slenderness: tuple[float, ...] = (10.0, 20.0, 40.0),
    elements_per_thickness: float = 2.5,
    ny_per_layer: int = 8,
) -> list[dict]:
    """Quantify the clamped-edge boundary layer: deflection ratio vs ``L/h``.

    The closed form is the pure-bending state while the FEM clamps one end, so
    the measured ratio ``w_FEM/w_analytic`` approaches 1 as the strip gets
    slender.  This is the *evidence* behind the default ``L/h`` (a modelling
    decision), not a knob applied to the acceptance tolerance.
    """
    h = BimetalCase().h
    out = []
    for s in slenderness:
        case = BimetalCase(length=s * h)
        ref = analytical_solution(case)
        nx = max(2, int(round(elements_per_thickness * s)))   # dx/h held constant
        res = solve_bimetal_strip(case, nx, ny_per_layer)
        out.append({
            "slenderness": s,
            "nx": nx,
            "dx_over_h": case.length / nx / h,
            "tip_deflection": res.tip_deflection,
            "analytic_tip_deflection": ref.tip_deflection,
            "ratio": res.tip_deflection / ref.tip_deflection,
            "rel_error": relative_error(res.tip_deflection, ref.tip_deflection),
            "sigma_top_rel_error": relative_error(res.sigma_interface_top,
                                                  ref.sigma_interface_top),
        })
    return out


def energy_balance(
    case: BimetalCase | None = None, nx: int = 64, ny_per_layer: int = 16,
    boundary_layers: float = 2.0,
) -> dict:
    """Strain energy vs the work of the thermal load, and vs the closed form."""
    case = case or BimetalCase()
    ref = analytical_solution(case)
    res = solve_bimetal_strip(case, nx, ny_per_layer)
    x_lo = boundary_layers * case.h
    fem_reg = res.strain_energy_above(x_lo)
    ana_reg = ref.strain_energy_in(x_lo, case.length)
    return {
        "case": case,
        "strain_energy_fem": res.strain_energy,
        "strain_energy_analytic": ref.strain_energy,
        "whole_strip_rel_error": relative_error(res.strain_energy, ref.strain_energy),
        "thermal_pool": res.thermal_pool,
        "half_load_work": res.load_work_half,
        "energy_identity_gap": res.energy_identity_gap,
        "boundary_layer_excluded": x_lo,
        "strain_energy_fem_region": fem_reg,
        "strain_energy_analytic_region": ana_reg,
        "region_rel_error": relative_error(fem_reg, ana_reg),
        "region_fraction": fem_reg / res.strain_energy,
    }


def single_material_limit(case: BimetalCase | None = None) -> dict:
    """``alpha1 -> alpha2`` must kill the curvature (bimetal-ness vanishes)."""
    ref_case = case or BimetalCase()
    ref = analytical_solution(ref_case)
    same = BimetalCase(m=ref_case.m, n=ref_case.n, dT=ref_case.dT, h=ref_case.h,
                       length=ref_case.length, E1=ref_case.E1,
                       alpha1=ref_case.alpha2, alpha2=ref_case.alpha2, nu=ref_case.nu)
    ana = analytical_solution(same)
    res = solve_bimetal_strip(same, 32, 8)
    return {
        "reference_deflection": ref.tip_deflection,
        "analytic_curvature": ana.curvature,
        "analytic_deflection": ana.tip_deflection,
        "fem_tip_deflection": res.tip_deflection,
        "analytic_relative_deflection": abs(ana.tip_deflection) / abs(ref.tip_deflection),
        "fem_relative_deflection": abs(res.tip_deflection) / abs(ref.tip_deflection),
        "fem_curvature": res.curvature_estimate,
        "fem_max_abs_sigma_interface": res.max_abs_sigma_interface,
        "analytic_sigma_interface_top": ana.sigma_interface_top,
    }


def parameter_sweep(
    cases: list[BimetalCase] | None = None,
    nx: int = 64,
    ny_per_layer: int = 16,
) -> list[dict]:
    """FEM vs closed form over a list of ``(m, n, dT)`` cases."""
    cases = cases if cases is not None else all_cases()
    out = []
    for case in cases:
        ref = analytical_solution(case)
        res = solve_bimetal_strip(case, nx, ny_per_layer)
        out.append({
            "m": case.m,
            "n": case.n,
            "dT": case.dT,
            "t1": case.t1,
            "t2": case.t2,
            "E2": case.E2,
            "analytic_curvature": ref.curvature,
            "fem_curvature": res.curvature_estimate,
            "curvature_rel_error": relative_error(res.curvature_estimate, ref.curvature),
            "analytic_tip_deflection": ref.tip_deflection,
            "fem_tip_deflection": res.tip_deflection,
            "deflection_rel_error": relative_error(res.tip_deflection, ref.tip_deflection),
            "analytic_sigma_top": ref.sigma_interface_top,
            "fem_sigma_top": res.sigma_interface_top,
            "sigma_top_rel_error": relative_error(res.sigma_interface_top,
                                                  ref.sigma_interface_top),
            "analytic_sigma_bottom": ref.sigma_interface_bottom,
            "fem_sigma_bottom": res.sigma_interface_bottom,
            "sigma_bottom_rel_error": relative_error(res.sigma_interface_bottom,
                                                     ref.sigma_interface_bottom),
            "analytic_max_abs_sigma": ref.max_abs_sigma_interface,
            "fem_max_abs_sigma": res.max_abs_sigma_interface,
            "sigma_max_rel_error": relative_error(res.max_abs_sigma_interface,
                                                  ref.max_abs_sigma_interface),
            "energy_identity_gap": res.energy_identity_gap,
        })
    return out