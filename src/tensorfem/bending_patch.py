"""Constant-curvature ("bending") patch test for the Q4 Mindlin-Reissner plate.

Why this module exists
----------------------
``tensorfem.patch_test`` covers the *continuum* patch test (Robinson A/B/C for
Q4/Q8/TET10).  A plane-stress patch test says nothing about plate **bending**
elements, because those carry an independent rotation field and a transverse
shear energy that can lock.  This module adds the missing piece: the
constant-curvature patch test as stated for plate elements in the FE
literature.

Criterion (the reference)
-------------------------
A plate patch with *free interior nodes*, loaded only through boundary
displacements that belong to a state of **constant curvature**, must reproduce
the interior displacement, the rotations, the curvatures/moments and the strain
energy exactly (to round-off).  Statement of the test: Irons & Ahmad,
*Techniques of Finite Elements* (1980), ch. 8; restated for plates in
Zienkiewicz & Taylor, *The Finite Element Method*, Vol. 2 (5th ed.), §10.4 and
applied to plate elements in Batoz, Bathe & Ho, *Int. J. Numer. Methods Eng.*
**15** (1980) 1771-1812, §4.  See ``docs/bending-patch.md`` for the full
provenance note and ``docs/oss-survey-CASE-BENDING-PATCH.md`` for the
open-source / method-genealogy survey.

The exact field
---------------
With the Mindlin sign convention used by :mod:`tensorfem.plate` (transverse
shear ``gamma_xz = w,x + theta_x``), a shear-free state of prescribed
curvatures ``(kappa_x, kappa_y, kappa_xy)`` is

    w       = -(kappa_x x^2 + kappa_xy x y + kappa_y y^2) / 2
    theta_x =   kappa_x x + kappa_xy y / 2
    theta_y =   kappa_xy x / 2 + kappa_y y

so that ``gamma_xz = gamma_yz = 0`` and

    kappa = [theta_x,x ; theta_y,y ; theta_x,y + theta_y,x]
          = [kappa_x ; kappa_y ; kappa_xy]           (constant)

The moments follow the isotropic Kirchhoff/Mindlin law with
``D = E t^3 / (12 (1 - nu^2))``:

    M_x  = D (kappa_x + nu kappa_y)
    M_y  = D (kappa_y + nu kappa_x)
    M_xy = D (1 - nu)/2 * kappa_xy

(i.e. ``M = Db kappa`` with the very ``Db`` that :func:`tensorfem.plate.
q4_mindlin_stiffness` uses).  No table is hand-copied: every target value is
evaluated from this closed form, and :func:`oracle_selfcheck` re-verifies the
field (zero shear, constant curvature, constant moments) before any finite
element is assembled.

Mesh families and what each one can prove
-----------------------------------------
* ``affine`` / ``skewed`` (parallelogram elements, i.e. an affine map of the
  reference patch): the interpolant of the quadratic ``w`` has *zero* parasitic
  shear at the element centre, so a shear-robust Q4 reproduces the state to
  machine precision.  This is the *pass/fail* form of the patch test.
* ``distorted`` (non-affine quads, interior nodes displaced): the bilinear
  interpolation of a quadratic ``w`` is not exact, so a residual appears.  It is
  not a patch-test failure of the element but a property of the interpolation
  space; it is reported as a measured, h-convergent number (see
  ``docs/bending-patch.md`` §5).  Claiming "exact" here would be false, so the
  module computes it instead of asserting it.
* ``full`` shear integration is deliberately included as a **negative control**:
  the same patch with 2x2 shear integration locks and must fail, which proves
  that the passing gates are not vacuous.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .plate import _shape, q4_mindlin_stiffness

__all__ = [
    "CurvatureState",
    "Patch",
    "plate_moment_matrix",
    "q4_reference_stiffness",
    "affine_patch_mesh",
    "distorted_patch_mesh",
    "element_dofs",
    "assemble_plate_stiffness",
    "exact_nodal_dofs",
    "solve_dirichlet_patch",
    "q4_gauss_curvatures",
    "moment_error",
    "strain_energy",
    "run_patch_case",
    "strip_pure_bending",
    "oracle_selfcheck",
]

GAUSS = 1.0 / math.sqrt(3.0)
_GAUSS_2X2 = ((-GAUSS, -GAUSS), (GAUSS, -GAUSS), (GAUSS, GAUSS), (-GAUSS, GAUSS))


# --------------------------------------------------------------------------- #
# exact oracle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CurvatureState:
    """A constant-curvature (shear-free) plate bending state."""

    kappa_x: float = 1.0e-3
    kappa_y: float = 0.0
    kappa_twist: float = 0.0

    @property
    def kappa(self) -> torch.Tensor:
        return torch.tensor([self.kappa_x, self.kappa_y, self.kappa_twist],
                            dtype=torch.float64)

    def field(self, xy: torch.Tensor) -> torch.Tensor:
        """Exact nodal DOFs ``(w, theta_x, theta_y)`` at ``xy`` (n, 2)."""
        x, y = xy[:, 0], xy[:, 1]
        kx, ky, kxy = self.kappa_x, self.kappa_y, self.kappa_twist
        # The element measures the *engineering* twist curvature
        # kappa_xy = theta_x,y + theta_y,x (see PlateRigidity.dt), so the twist is
        # split half-half between the two rotations: theta_x,y = theta_y,x = kxy/2
        # and w,xy = -kxy/2.
        w = -0.5 * (kx * x * x + kxy * x * y + ky * y * y)
        thx = kx * x + 0.5 * kxy * y
        thy = ky * y + 0.5 * kxy * x
        return torch.stack((w, thx, thy), dim=1)

    def moments(self, young: float = 210.0e3, poisson: float = 0.3,
                thickness: float = 1.0) -> torch.Tensor:
        """Exact constant moments ``(M_x, M_y, M_xy)``."""
        return plate_moment_matrix(young, poisson, thickness) @ self.kappa

    def label(self) -> str:
        return f"kx={self.kappa_x:g},ky={self.kappa_y:g},kxy={self.kappa_twist:g}"


def plate_moment_matrix(young: float, poisson: float, thickness: float) -> torch.Tensor:
    """Isotropic bending rigidity ``Db`` (3x3), identical to the element's."""
    d = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    return d * torch.tensor(
        [[1.0, poisson, 0.0],
         [poisson, 1.0, 0.0],
         [0.0, 0.0, (1.0 - poisson) / 2.0]],
        dtype=torch.float64,
    )


def oracle_selfcheck(state: CurvatureState | None = None,
                     young: float = 210.0e3, poisson: float = 0.3,
                     thickness: float = 1.0) -> dict:
    """Verify the analytic oracle before any FE work.

    Checks (all analytic, no solver involved):

    * ``gamma_xz = w,x + theta_x`` and ``gamma_yz = w,y + theta_y`` vanish -- a
      pure bending state carries no transverse shear, which is what makes it a
      *stress* patch test for the bending part;
    * ``kappa = [theta_x,x, theta_y,y, theta_x,y + theta_y,x]`` equals the
      prescribed constant (``kappa_xy`` is the *engineering* twist curvature, the
      measure ``tensorfem.plate`` integrates: ``dt`` multiplies
      ``theta_x,y + theta_y,x``);
    * ``M = Db kappa`` is constant and matches the closed form above;
    * the equality ``w,x = -theta_x`` holds, i.e. the two fields are one state.
    """
    s = state or CurvatureState()
    h = 1.0e-6
    pts = [(0.13, 0.71), (0.62, 0.24), (0.91, 0.88)]

    def f(x, y):
        return s.field(torch.tensor([[x, y]], dtype=torch.float64))[0]

    shear, curv = [], []
    for (x, y) in pts:
        w_p, tx_p, _ = f(x + h, y)
        w_m, tx_m, _ = f(x - h, y)
        _, _, ty_p = f(x, y + h)
        _, _, ty_m = f(x, y - h)
        dwdx = (w_p - w_m).item() / (2 * h)
        dwdy = (f(x, y + h)[0] - f(x, y - h)[0]).item() / (2 * h)
        tx, ty = f(x, y)[1].item(), f(x, y)[2].item()
        shear.append(abs(dwdx + tx) + abs(dwdy + ty))
        curv.append([(tx_p - tx_m).item() / (2 * h),
                     (ty_p - ty_m).item() / (2 * h),
                     (f(x, y + h)[1] - f(x, y - h)[1]).item() / (2 * h)
                     + (f(x + h, y)[2] - f(x - h, y)[2]).item() / (2 * h)])

    curv_t = torch.tensor(curv, dtype=torch.float64)
    db = plate_moment_matrix(young, poisson, thickness)
    mom = curv_t @ db.T
    return {
        "shear_residual": max(shear),
        "curvature_deviation": float((curv_t - s.kappa.unsqueeze(0)).abs().max()),
        "moment_deviation": float((mom - s.moments(young, poisson, thickness)
                                   .unsqueeze(0)).abs().max()),
        "moments": s.moments(young, poisson, thickness).tolist(),
    }


# --------------------------------------------------------------------------- #
# elements
# --------------------------------------------------------------------------- #
def q4_reference_stiffness(xy: torch.Tensor, *, young: float, poisson: float,
                           thickness: float, shear_treatment: str = "sri",
                           shear_correction: float = 5.0 / 6.0) -> torch.Tensor:
    """Q4 Mindlin stiffness; ``shear_treatment='full'`` is the locking control.

    ``sri`` and ``mitc`` delegate to the production element
    :func:`tensorfem.plate.q4_mindlin_stiffness`.  ``full`` re-integrates the
    transverse shear with the 2x2 rule and exists *only* as the negative control
    of the patch test -- it is intentionally shear-locking and must not be used
    as a solver.
    """
    if shear_treatment in ("sri", "mitc"):
        return q4_mindlin_stiffness(xy, young=young, poisson=poisson,
                                    thickness=thickness,
                                    shear_correction=shear_correction,
                                    shear_treatment=shear_treatment)
    if shear_treatment != "full":
        raise ValueError("shear_treatment must be 'sri', 'mitc' or 'full'")

    dtype, device = xy.dtype, xy.device
    db = plate_moment_matrix(young, poisson, thickness).to(dtype=dtype, device=device)
    ds = shear_correction * young / (2.0 * (1.0 + poisson)) * thickness
    ds = ds * torch.eye(2, dtype=dtype, device=device)
    ke = torch.zeros((12, 12), dtype=dtype, device=device)
    for xif, etaf in _GAUSS_2X2:
        xi, eta = xy.new_tensor(xif), xy.new_tensor(etaf)
        n, dN_nat = _shape(xi, eta)
        jac = dN_nat.T @ xy
        detj = torch.linalg.det(jac)
        if bool((detj <= 0).item()):
            raise ValueError("plate element has non-positive Jacobian")
        dN = dN_nat @ torch.linalg.inv(jac).T
        bb = torch.zeros((3, 12), dtype=dtype, device=device)
        bs = torch.zeros((2, 12), dtype=dtype, device=device)
        for i in range(4):
            bb[0, 3 * i + 1] = dN[i, 0]
            bb[1, 3 * i + 2] = dN[i, 1]
            bb[2, 3 * i + 1] = dN[i, 1]
            bb[2, 3 * i + 2] = dN[i, 0]
            bs[0, 3 * i] = dN[i, 0]
            bs[0, 3 * i + 1] = n[i]
            bs[1, 3 * i] = dN[i, 1]
            bs[1, 3 * i + 2] = n[i]
        ke = ke + (bb.T @ db @ bb + bs.T @ ds @ bs) * detj
    return ke


# --------------------------------------------------------------------------- #
# patches
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Patch:
    """A structured Q4 patch: nodes, connectivity and the interior-node mask."""

    nodes: torch.Tensor        # (n_nodes, 2)
    elements: torch.Tensor     # (n_elems, 4)
    interior: torch.Tensor     # (n_nodes,) bool
    n: int
    family: str

    def __post_init__(self) -> None:
        if self.elements.shape[1] != 4:
            raise ValueError("Patch expects Q4 connectivity")

    @property
    def interior_elements(self) -> torch.Tensor:
        return self.interior[self.elements].all(dim=1)

    def min_jacobian(self) -> float:
        worst = math.inf
        for e in self.elements.tolist():
            xy = self.nodes[torch.tensor(e)]
            for xif in (-GAUSS, 0.0, GAUSS):
                for etaf in (-GAUSS, 0.0, GAUSS):
                    _, dN_nat = _shape(xy.new_tensor(xif), xy.new_tensor(etaf))
                    worst = min(worst, float(torch.linalg.det(dN_nat.T @ xy)))
        return worst

    def quad_area(self) -> float:
        total = 0.0
        for e in self.elements.tolist():
            xy = self.nodes[torch.tensor(e)]
            a = 0.0
            for i in range(4):
                p, q = xy[i], xy[(i + 1) % 4]
                a += float(p[0] * q[1] - q[0] * p[1])
            total += abs(a) * 0.5
        return total


def _structured_patch(n: int, side: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n < 2:
        raise ValueError("a patch needs at least one free interior node (n >= 2)")
    g = torch.linspace(0.0, side, n + 1, dtype=torch.float64)
    xx, yy = torch.meshgrid(g, g, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    elems = []
    for j in range(n):
        for i in range(n):
            n0 = j * (n + 1) + i
            elems.append((n0, n0 + 1, n0 + n + 2, n0 + n + 1))
    ij = torch.arange(nodes.shape[0]).reshape(n + 1, n + 1)
    interior = torch.zeros(nodes.shape[0], dtype=torch.bool)
    interior[ij[1:-1, 1:-1].reshape(-1)] = True
    return nodes, torch.tensor(elems, dtype=torch.long), interior


def affine_patch_mesh(n: int, side: float = 1.0, skew: float = 0.0) -> Patch:
    """Patch of *parallelogram* elements: the affine map ``x' = x + skew*y``."""
    nodes, elems, interior = _structured_patch(n, side)
    nodes = torch.stack((nodes[:, 0] + skew * nodes[:, 1], nodes[:, 1]), dim=1)
    return Patch(nodes, elems, interior, n, family="skew" if skew else "affine")


def distorted_patch_mesh(n: int, side: float = 1.0, amplitude: float = 0.30,
                         seed: int = 20261111) -> Patch:
    """Non-affine quad patch: every interior node moves by up to ``amplitude*h``
    in each direction (``h = side/n``); boundary nodes stay on the square."""
    nodes, elems, interior = _structured_patch(n, side)
    h = side / n
    gen = torch.Generator().manual_seed(seed)
    idx = torch.nonzero(interior).reshape(-1)
    shift = torch.rand((idx.numel(), 2), generator=gen, dtype=torch.float64) * 2.0 - 1.0
    nodes = nodes.clone()
    nodes[idx] += shift * amplitude * h
    patch = Patch(nodes, elems, interior, n, family="distorted")
    if patch.min_jacobian() <= 0.0:
        raise ValueError("distortion produced a non-positive Jacobian")
    return patch


# --------------------------------------------------------------------------- #
# assembly / solve
# --------------------------------------------------------------------------- #
def element_dofs(elements: torch.Tensor) -> torch.Tensor:
    """(n_elems, 12) global DOF indices, node-major ``(w, theta_x, theta_y)``."""
    return (elements.unsqueeze(2) * 3 + torch.arange(3)).reshape(-1, 12)


def assemble_plate_stiffness(patch: Patch, *, young: float, poisson: float,
                             thickness: float, shear_treatment: str = "sri",
                             shear_correction: float = 5.0 / 6.0) -> torch.Tensor:
    ndof = 3 * patch.nodes.shape[0]
    k = torch.zeros((ndof, ndof), dtype=patch.nodes.dtype, device=patch.nodes.device)
    dofs = element_dofs(patch.elements)
    for ei, e in enumerate(patch.elements.tolist()):
        ke = q4_reference_stiffness(patch.nodes[torch.tensor(e)], young=young,
                                    poisson=poisson, thickness=thickness,
                                    shear_treatment=shear_treatment,
                                    shear_correction=shear_correction)
        idx = dofs[ei]
        k[idx.unsqueeze(1), idx.unsqueeze(0)] += ke
    return k


def exact_nodal_dofs(patch: Patch, state: CurvatureState) -> torch.Tensor:
    return state.field(patch.nodes).reshape(-1)


def solve_dirichlet_patch(patch: Patch, state: CurvatureState, *, young: float,
                          poisson: float, thickness: float,
                          shear_treatment: str = "sri",
                          shear_correction: float = 5.0 / 6.0) -> dict:
    """Prescribe the exact field on every boundary node, solve the free interior."""
    k = assemble_plate_stiffness(patch, young=young, poisson=poisson,
                                 thickness=thickness, shear_treatment=shear_treatment,
                                 shear_correction=shear_correction)
    u_exact = exact_nodal_dofs(patch, state)
    free_mask = patch.interior.unsqueeze(1).expand(-1, 3).reshape(-1)
    free = torch.nonzero(free_mask).reshape(-1)
    fixed = torch.nonzero(~free_mask).reshape(-1)
    u = u_exact.clone()
    kff = k[free][:, free]
    kfc = k[free][:, fixed]
    rhs = -(kfc @ u_exact[fixed])
    u[free] = torch.linalg.solve(kff, rhs)
    residual = float((kff @ u[free] - rhs).abs().max())
    return {"u": u, "stiffness": k, "free": free, "residual": residual}


def q4_gauss_curvatures(xy: torch.Tensor, ue: torch.Tensor) -> torch.Tensor:
    """Curvatures ``(4, 3)`` at the 2x2 Gauss points of one Q4 element."""
    dtype, device = xy.dtype, xy.device
    out = []
    for xif, etaf in _GAUSS_2X2:
        xi, eta = xy.new_tensor(xif), xy.new_tensor(etaf)
        _, dN_nat = _shape(xi, eta)
        jac = dN_nat.T @ xy
        dN = dN_nat @ torch.linalg.inv(jac).T
        bb = torch.zeros((3, 12), dtype=dtype, device=device)
        for i in range(4):
            bb[0, 3 * i + 1] = dN[i, 0]
            bb[1, 3 * i + 2] = dN[i, 1]
            bb[2, 3 * i + 1] = dN[i, 1]
            bb[2, 3 * i + 2] = dN[i, 0]
        out.append(bb @ ue)
    return torch.stack(out)


def moment_error(patch: Patch, u: torch.Tensor, state: CurvatureState, *, young: float,
                 poisson: float, thickness: float,
                 interior_only: bool = True) -> float:
    """Max moment error over the sampled Gauss points, normalised by |M|_max."""
    db = plate_moment_matrix(young, poisson, thickness)
    exact = state.moments(young, poisson, thickness)
    scale = float(exact.abs().max())
    if scale == 0.0:
        raise ValueError("reference moment vanishes; relative error undefined")
    mask = (patch.interior_elements if interior_only
            else torch.ones(patch.elements.shape[0], dtype=torch.bool))
    dofs = element_dofs(patch.elements)
    worst = 0.0
    for ei in torch.nonzero(mask).reshape(-1).tolist():
        e = patch.elements[ei]
        m = q4_gauss_curvatures(patch.nodes[e], u[dofs[ei]]) @ db.T
        worst = max(worst, float((m - exact.unsqueeze(0)).abs().max()))
    return worst / scale


def strain_energy(u: torch.Tensor, k: torch.Tensor) -> float:
    return 0.5 * float(u @ (k @ u))


def run_patch_case(patch: Patch, state: CurvatureState, *, young: float = 210.0e3,
                   poisson: float = 0.3, thickness: float = 1.0,
                   shear_treatment: str = "sri",
                   shear_correction: float = 5.0 / 6.0) -> dict:
    """Run one patch case and return every graded quantity.

    Error norms are normalised by the maximum modulus of the exact field over
    the *free interior* DOFs, so a zero solution can never fake a pass.
    """
    sol = solve_dirichlet_patch(patch, state, young=young, poisson=poisson,
                                thickness=thickness, shear_treatment=shear_treatment,
                                shear_correction=shear_correction)
    u, u_exact, free = sol["u"], exact_nodal_dofs(patch, state), sol["free"]
    w_err = float((u[free][0::3] - u_exact[free][0::3]).abs().max())
    rot_err = float(torch.stack((u[free][1::3] - u_exact[free][1::3],
                                 u[free][2::3] - u_exact[free][2::3])).abs().max())
    w_scale = float(u_exact[free][0::3].abs().max())
    rot_scale = float(torch.stack((u_exact[free][1::3],
                                   u_exact[free][2::3])).abs().max())
    if w_scale == 0.0 or rot_scale == 0.0:
        raise ValueError("degenerate reference state: exact field vanishes on free DOFs")

    db = plate_moment_matrix(young, poisson, thickness)
    energy_density = 0.5 * float(state.kappa @ (db @ state.kappa))
    energy_exact = energy_density * patch.quad_area()
    energy = strain_energy(u, sol["stiffness"])

    return {
        "family": patch.family,
        "n": patch.n,
        "elements": int(patch.elements.shape[0]),
        "free_dofs": int(free.numel()),
        "shear_treatment": shear_treatment,
        "state": state.label(),
        "kappa": state.kappa.tolist(),
        "moments_exact": state.moments(young, poisson, thickness).tolist(),
        "displacement_rel_error": w_err / w_scale,
        "rotation_rel_error": rot_err / rot_scale,
        "moment_rel_error": moment_error(patch, u, state, young=young,
                                         poisson=poisson, thickness=thickness),
        "energy_rel_error": abs(energy - energy_exact) / energy_exact,
        "strain_energy": energy,
        "strain_energy_exact": energy_exact,
        "equilibrium_residual": sol["residual"],
        "min_jacobian": patch.min_jacobian(),
    }


# --------------------------------------------------------------------------- #
# load-driven strip: a genuine BVP with a closed-form answer
# --------------------------------------------------------------------------- #
def strip_pure_bending(n_x: int, n_y: int = 2, *, moment: float = 1.0e3,
                       length: float = 1.0, width: float = 0.2,
                       young: float = 210.0e3, poisson: float = 0.3,
                       thickness: float = 0.01,
                       lateral: str = "symmetry",
                       shear_treatment: str = "sri",
                       shear_correction: float = 5.0 / 6.0) -> dict:
    """Cantilever plate strip under a tip bending moment per unit width.

    This is a *load-driven* boundary value problem (nothing of the answer is
    prescribed), so it exercises the load path, the sign conventions and the
    ``Db``/``Ds`` magnitudes on top of the patch test.

    Two lateral conditions are offered because they are two different physical
    problems with two different closed forms:

    ``lateral="symmetry"`` (``theta_y = 0`` on ``y = 0`` and ``y = width``,
    w free) -- the strip is one symmetry cell of an infinite wide plate, so the
    response is **cylindrical bending** with the plate rigidity
    ``D = E t^3 / (12 (1 - nu^2))``::

        w(L) = M L^2 / (2 D),     theta_x(L) = -M L / D

    ``lateral="free"`` -- a narrow strip whose free lateral edges force
    ``M_y = 0`` (plane stress in ``y``), i.e. the **beam** rigidity
    ``D_b = E t^3 / 12 = D (1 - nu^2)`` and the same formulae with ``D_b``.

    Both are standard results (Timoshenko & Woinowsky-Krieger, *Theory of Plates
    and Shells*, 2nd ed., Art. 10; Timoshenko, *Strength of Materials* II, ch. 3).
    Their ratio is exactly ``1/(1 - nu^2)``, so running both is a physical
    cross-check: an implementation error in the Poisson coupling cannot satisfy
    both at once.

    The moment enters as the natural boundary condition: a consistent edge load
    conjugate to ``theta_x`` on ``x = length`` (corner nodes carry half the
    tributary length).  ``w = theta_x = theta_y = 0`` on the clamped edge.
    """
    if n_x < 1 or n_y < 1:
        raise ValueError("n_x and n_y must be positive")
    if lateral not in ("symmetry", "free"):
        raise ValueError("lateral must be 'symmetry' or 'free'")
    gx = torch.linspace(0.0, length, n_x + 1, dtype=torch.float64)
    gy = torch.linspace(0.0, width, n_y + 1, dtype=torch.float64)
    xx, yy = torch.meshgrid(gx, gy, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    elems = []
    for j in range(n_y):
        for i in range(n_x):
            n0 = j * (n_x + 1) + i
            elems.append((n0, n0 + 1, n0 + n_x + 2, n0 + n_x + 1))
    elements = torch.tensor(elems, dtype=torch.long)
    dofs = element_dofs(elements)
    ndof = 3 * nodes.shape[0]
    k = torch.zeros((ndof, ndof), dtype=torch.float64)
    f = torch.zeros(ndof, dtype=torch.float64)
    for ei, e in enumerate(elements.tolist()):
        xy = nodes[torch.tensor(e)]
        ke = q4_reference_stiffness(xy, young=young, poisson=poisson,
                                    thickness=thickness,
                                    shear_treatment=shear_treatment,
                                    shear_correction=shear_correction)
        idx = dofs[ei]
        k[idx.unsqueeze(1), idx.unsqueeze(0)] += ke
    # consistent edge moment: int N_i m ds on the theta_x DOF of the loaded edge.
    # The negative sign makes a positive |M| give w(L) > 0 in this model's
    # convention (gamma_xz = w,x + theta_x  =>  theta_x = -w,x).
    trib = width / n_y
    for j in range(n_y + 1):
        weight = 0.5 if j in (0, n_y) else 1.0
        f[3 * (j * (n_x + 1) + n_x) + 1] += -abs(moment) * trib * weight
    # clamped edge x = 0
    clamp = []
    for j in range(n_y + 1):
        base = 3 * (j * (n_x + 1))
        clamp += [base, base + 1, base + 2]
    # symmetry on the lateral edges: theta_y = 0 (dy symmetry plane)
    if lateral == "symmetry":
        for i in range(n_x + 1):
            clamp.append(3 * i + 2)                                    # y = 0
            clamp.append(3 * (n_y * (n_x + 1) + i) + 2)                # y = width
    fixed = torch.tensor(sorted(set(clamp)), dtype=torch.long)
    mask = torch.ones(ndof, dtype=torch.bool)
    mask[fixed] = False
    free = torch.nonzero(mask).reshape(-1)
    u = torch.zeros(ndof, dtype=torch.float64)
    u[free] = torch.linalg.solve(k[free][:, free], f[free])

    d_cyl = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    d_beam = young * thickness ** 3 / 12.0
    d_ref = d_cyl if lateral == "symmetry" else d_beam
    m = abs(moment)
    theta_exact = -m * length / d_ref
    w_kirchhoff = m * length ** 2 / (2.0 * d_ref)
    g_shear = young / (2.0 * (1.0 + poisson))
    w_mindlin = w_kirchhoff + m * length / (shear_correction * g_shear * thickness)
    tip = n_y // 2 * (n_x + 1) + n_x
    w_tip = float(u[3 * tip])
    theta_tip = float(u[3 * tip + 1])
    return {
        "n_x": n_x, "n_y": n_y, "lateral": lateral,
        "shear_treatment": shear_treatment,
        "moment": moment, "length": length, "width": width, "thickness": thickness,
        "slenderness": length / thickness,
        "rigidity_used": "D_cyl" if lateral == "symmetry" else "D_beam",
        "rigidity": d_ref,
        "w_tip": w_tip,
        "theta_tip": theta_tip,
        "w_exact_kirchhoff": w_kirchhoff,
        "w_exact_mindlin": w_mindlin,
        "theta_exact": theta_exact,
        "w_rel_error_kirchhoff": abs(w_tip - w_kirchhoff) / w_kirchhoff,
        "w_rel_error_mindlin": abs(w_tip - w_mindlin) / w_mindlin,
        "theta_rel_error": abs(theta_tip - theta_exact) / abs(theta_exact),
        "free_dofs": int(free.numel()),
    }
