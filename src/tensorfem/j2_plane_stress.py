"""Small-strain J2 plane-stress (membrane) Q4 element with radial return mapping.

The module supplies the missing half of the repository's J2 family: the 3-D radial
return in :mod:`tensorfem.plasticity` is *plane-strain* by construction
(``sigma_zz`` is whatever the strain gives), whereas a membrane / shell material
point must satisfy ``sigma_zz = 0``.  The plane-stress operator is built here as a
thin wrapper around the 3-D kernel, so the return mapping keeps exactly one
implementation:

* the out-of-plane strain ``eps_zz`` is the scalar unknown that enforces
  ``sigma_zz = 0``; it is solved by a scalar Newton iteration whose derivative is
  the ``zz`` entry of the 3-D algorithmic tangent (:func:`_tangent_3d`, the
  batched mirror of :func:`tensorfem.plasticity.j2_consistent_tangent`);
* the plane-stress consistent tangent is the Schur complement of that 3-D tangent
  w.r.t. the ``zz`` row/column -- the implicit-function-theorem result for the
  constraint ``sigma_zz(eps_zz(eps_inplane)) = 0``;
* the plastic work increment is the exact hardening integral
  ``int (sigma_y + H alpha) dalpha``, so the global energy balance
  ``W_ext = U_elastic + W_plastic`` can be audited instead of assumed.

The finite-element surface is a 2x2 Gauss, plane-stress Q4 membrane with a
Newton-Raphson driver (force and displacement control) and per-Gauss-point state
commit, so a failed increment is retried from the committed state.

Scope: infinitesimal strain, isotropic linear elasticity, von Mises yield, *linear
isotropic hardening* ``sigma_y(alpha) = sigma_y0 + H alpha``, monotone or
proportional paths.  Softening is rejected at construction (it is not a local
well-posedness claim).  Kinematic / combined hardening, rate dependence, finite
strain, damage and pressure-sensitive yield are **not** part of this surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

Tensor = torch.Tensor

__all__ = [
    "J2PlaneStressMaterial",
    "MembraneState",
    "Q4Kinematics",
    "MembraneProblem",
    "MembraneResponse",
    "CookMembraneModel",
    "DentSpecimen",
    "plane_stress_return",
    "plane_stress_return_3d",
    "q4_kinematics",
    "make_problem",
    "gauss_response",
    "solve_step",
    "run_path",
    "elastic_solve",
    "mises_plane_stress",
    "uniaxial_plane_stress",
    "cook_membrane_model",
    "dent_specimen_model",
]

# Voigt order [xx, yy, zz, xy, yz, zx]; in-plane subset for membranes.
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))
_IN_PLANE = (0, 1, 3)
_NODE_NAT = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
_GAUSS_1D = 1.0 / math.sqrt(3.0)
_GAUSS_NAT = ((-_GAUSS_1D, -_GAUSS_1D), (_GAUSS_1D, -_GAUSS_1D),
              (_GAUSS_1D, _GAUSS_1D), (-_GAUSS_1D, _GAUSS_1D))


# --------------------------------------------------------------------------- #
# material / state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class J2PlaneStressMaterial:
    """Isotropic elastic + von Mises plastic card with linear isotropic hardening."""

    young: float
    poisson: float
    yield_stress: float
    hardening: float = 0.0
    name: str = "j2-plane-stress"

    def __post_init__(self) -> None:
        if not (self.young > 0.0 and math.isfinite(self.young)):
            raise ValueError("young must be positive and finite")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must lie in (-1, 0.5)")
        if not (self.yield_stress > 0.0 and math.isfinite(self.yield_stress)):
            raise ValueError("yield_stress must be positive and finite")
        if not (self.hardening >= 0.0 and math.isfinite(self.hardening)):
            raise ValueError("hardening must be non-negative (softening is out of contract)")

    @property
    def shear(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def bulk(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    @property
    def elastic_modulus_plane_stress(self) -> float:
        return self.young / (1.0 - self.poisson ** 2)

    def plane_stress_matrix(self, dtype=torch.float64, device=None) -> Tensor:
        """Elastic ``[xx, yy, xy]`` matrix acting on engineering shear ``gamma_xy``."""
        c = self.elastic_modulus_plane_stress
        return torch.tensor([[c, c * self.poisson, 0.0],
                             [c * self.poisson, c, 0.0],
                             [0.0, 0.0, 0.5 * c * (1.0 - self.poisson)]],
                            dtype=dtype, device=device)

    def uniaxial_hardening_parameter(self, strain) -> Tensor:
        """Closed-form ``alpha(eps_xx)``: ``alpha = (E eps - sigma_y) / (E + H)``.

        The elastic strain keeps growing as the yield surface hardens, so the
        plastic strain is *not* ``eps - sigma_y/E`` (that is the perfectly plastic
        limit); solving ``alpha = eps - (sigma_y + H alpha)/E`` gives the closed
        form below, whose small-strain slope is ``E H / (E + H)``.
        """
        eps = torch.as_tensor(strain, dtype=torch.float64)
        return torch.clamp((self.young * eps - self.yield_stress)
                           / (self.young + self.hardening), min=0.0)

    def uniaxial_stress(self, strain) -> Tensor:
        """Closed-form plane-stress uniaxial curve ``sigma_xx(eps_xx)``."""
        eps = torch.as_tensor(strain, dtype=torch.float64)
        alpha = self.uniaxial_hardening_parameter(eps)
        return torch.where(alpha > 0.0, self.yield_stress + self.hardening * alpha,
                           self.young * eps)

    def uniaxial_lateral_strain(self, strain) -> Tensor:
        """Closed-form ``eps_yy(eps_xx)`` for plane-stress uniaxial tension."""
        alpha = self.uniaxial_hardening_parameter(strain)
        sigma = self.uniaxial_stress(strain)
        return -self.poisson * sigma / self.young - 0.5 * alpha

    def plastic_work_closed_form(self, alpha) -> Tensor:
        """``int_0^alpha (sigma_y0 + H a) da`` -- exact for linear hardening."""
        a = torch.as_tensor(alpha, dtype=torch.float64)
        return self.yield_stress * a + 0.5 * self.hardening * a * a


@dataclass
class MembraneState:
    """Committed per-Gauss-point state (elastic strain is recoverable from it)."""

    plastic_strain: Tensor  # (M, 3, 3)
    alpha: Tensor           # (M,) equivalent plastic strain
    eps_zz: Tensor          # (M,) converged plane-stress out-of-plane strain

    def __len__(self) -> int:
        return int(self.alpha.numel())

    @classmethod
    def virgin(cls, n: int, *, dtype=torch.float64, device=None) -> "MembraneState":
        if n < 0:
            raise ValueError("point count must be non-negative")
        return cls(torch.zeros((n, 3, 3), dtype=dtype, device=device),
                   torch.zeros((n,), dtype=dtype, device=device),
                   torch.zeros((n,), dtype=dtype, device=device))

    def clone(self) -> "MembraneState":
        return MembraneState(self.plastic_strain.clone(), self.alpha.clone(),
                             self.eps_zz.clone())


# --------------------------------------------------------------------------- #
# vectorised 3-D kernel (1:1 mirror of tensorfem.plasticity.update_j2 and
# j2_consistent_tangent; the mirror is pinned by tests)
# --------------------------------------------------------------------------- #
def _strain_tensor(exx: Tensor, eyy: Tensor, ezz: Tensor, gxy: Tensor) -> Tensor:
    m = exx.shape[0]
    out = exx.new_zeros((m, 3, 3))
    out[:, 0, 0] = exx
    out[:, 1, 1] = eyy
    out[:, 2, 2] = ezz
    out[:, 0, 1] = out[:, 1, 0] = 0.5 * gxy
    return out


def _return_3d(mat: J2PlaneStressMaterial, strain: Tensor,
               plastic_strain: Tensor, alpha: Tensor):
    """Batched radial return (backward Euler, associative flow, linear hardening)."""
    dt, dev = strain.dtype, strain.device
    eye = torch.eye(3, dtype=dt, device=dev)
    shear, bulk = mat.shear, mat.bulk
    elastic_strain = strain - plastic_strain
    tr = torch.diagonal(elastic_strain, dim1=-2, dim2=-1).sum(-1)
    deviator = elastic_strain - (tr / 3.0)[:, None, None] * eye
    s_trial = 2.0 * shear * deviator
    seq = torch.sqrt(1.5 * (s_trial * s_trial).sum((-1, -2)))
    surface = mat.yield_stress + mat.hardening * alpha
    f = seq - surface
    dgamma = torch.where(f > 0.0, f / (3.0 * shear + mat.hardening), torch.zeros_like(seq))
    safe = torch.where(seq > 0.0, seq, torch.ones_like(seq))
    direction = 1.5 * s_trial / safe[:, None, None]
    stress = (bulk * tr[:, None, None] * eye
              + s_trial - 2.0 * shear * dgamma[:, None, None] * direction)
    new_plastic = plastic_strain + dgamma[:, None, None] * direction
    return stress, new_plastic, alpha + dgamma, dgamma, s_trial, seq, tr


def _tangent_3d(mat: J2PlaneStressMaterial, strain: Tensor,
                plastic_strain: Tensor, dgamma: Tensor) -> Tensor:
    """Batched algorithmic tangent, Voigt ``[xx,yy,zz,xy,yz,zx]``, engineering shear."""
    dt, dev = strain.dtype, strain.device
    m = strain.shape[0]
    eye = torch.eye(3, dtype=dt, device=dev)
    ones = torch.ones((3, 3), dtype=dt, device=dev)
    shear, bulk, hard = mat.shear, mat.bulk, mat.hardening
    elastic_strain = strain - plastic_strain
    tr = torch.diagonal(elastic_strain, dim1=-2, dim2=-1).sum(-1)
    deviator = elastic_strain - (tr / 3.0)[:, None, None] * eye
    s_trial = 2.0 * shear * deviator
    seq = torch.sqrt(1.5 * (s_trial * s_trial).sum((-1, -2)))
    tangent = torch.zeros((m, 6, 6), dtype=dt, device=dev)
    tangent[:, :3, :3] = bulk * ones + 2.0 * shear * (eye - ones / 3.0)
    tangent[:, 3, 3] = shear
    tangent[:, 4, 4] = shear
    tangent[:, 5, 5] = shear
    active = torch.nonzero(dgamma > 0.0, as_tuple=False).flatten()
    if active.numel():
        s = seq[active]
        dg = dgamma[active]
        a = 1.0 - 3.0 * shear * dg / s
        da = -3.0 * shear * (1.0 / ((3.0 * shear + hard) * s) - dg / (s * s))
        c = 2.0 * shear * (1.5 / s) * da
        tangent[active, :3, :3] = (bulk * ones
                                   + 2.0 * shear * a[:, None, None] * (eye - ones / 3.0))
        tangent[active, 3, 3] = shear * a
        tangent[active, 4, 4] = shear * a
        tangent[active, 5, 5] = shear * a
        vec = torch.stack([s_trial[active][:, i, j] for i, j in _VOIGT_PAIRS], dim=1)
        tangent[active] += c[:, None, None] * (vec[:, :, None] * vec[:, None, :])
    return tangent


def _condense(tangent: Tensor) -> Tensor:
    """Schur-condense the batched ``6x6`` tangent to the in-plane ``3x3`` block."""
    ip = list(_IN_PLANE)
    block = tangent[:, ip][:, :, ip]
    col = tangent[:, ip, 2]
    row = tangent[:, 2, ip]
    normal = tangent[:, 2, 2]
    if bool((normal.abs() <= 0.0).any()):
        raise RuntimeError("plane-stress condensation hit a singular sigma_zz tangent")
    return block - col[:, :, None] * row[:, None, :] / normal[:, None, None]


# --------------------------------------------------------------------------- #
# plane-stress return mapping
# --------------------------------------------------------------------------- #
def plane_stress_return(mat: J2PlaneStressMaterial, strain: Tensor,
                        state: MembraneState, *, max_iter: int = 40,
                        tol: float | None = None):
    """Plane-stress return map for ``strain`` ``(M, 3)`` = ``[exx, eyy, gxy]``.

    Returns ``(stress, new_state, dWp, tangent, diagnostics)``:

    * ``stress`` ``(M, 3)`` = ``[sxx, syy, sxy]`` with ``sigma_zz = 0`` enforced,
    * ``tangent`` ``(M, 3, 3)`` consistent algorithmic tangent,
    * ``dWp`` ``(M,)`` exact plastic work increment ``int (sigma_y + H a) da``,
    * ``diagnostics`` with the ``sigma_zz`` residual and Newton iteration count.
    """
    strain = torch.atleast_2d(strain)
    if strain.shape[-1] != 3:
        raise ValueError("plane-stress strain must have 3 columns [exx, eyy, gxy]")
    if state.plastic_strain.shape[0] != strain.shape[0]:
        raise ValueError("state and strain point counts differ")
    exx, eyy, gxy = strain[:, 0], strain[:, 1], strain[:, 2]
    tol_sigma = (1e-11 * mat.yield_stress) if tol is None else float(tol)
    # elastic (sigma_zz = 0) start value: eps_zz = -nu/(1-nu) (exx + eyy)
    ezz = -mat.poisson / (1.0 - mat.poisson) * (exx + eyy)
    sigma_zz = torch.zeros_like(ezz)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        strain3 = _strain_tensor(exx, eyy, ezz, gxy)
        stress3, _, _, dgamma, _, _, _ = _return_3d(mat, strain3, state.plastic_strain,
                                                   state.alpha)
        sigma_zz = stress3[:, 2, 2]
        if float(sigma_zz.abs().max()) <= tol_sigma:
            break
        tangent3 = _tangent_3d(mat, strain3, state.plastic_strain, dgamma)
        slope = tangent3[:, 2, 2]
        floor = 1e-6 * (mat.bulk + mat.shear)
        slope = torch.where(slope > floor, slope, torch.full_like(slope, floor))
        ezz = ezz - sigma_zz / slope
    strain3 = _strain_tensor(exx, eyy, ezz, gxy)
    stress3, new_plastic, new_alpha, dgamma, _, _, _ = _return_3d(
        mat, strain3, state.plastic_strain, state.alpha)
    tangent = _condense(_tangent_3d(mat, strain3, state.plastic_strain, dgamma))
    stress = torch.stack([stress3[:, 0, 0], stress3[:, 1, 1], stress3[:, 0, 1]], dim=1)
    dWp = ((mat.yield_stress + mat.hardening * state.alpha) * dgamma
           + 0.5 * mat.hardening * dgamma * dgamma)
    new_state = MembraneState(new_plastic, new_alpha, ezz)
    diagnostics = {
        "sigma_zz_max_abs": float(sigma_zz.abs().max()) if sigma_zz.numel() else 0.0,
        "newton_iterations": iterations,
        "stress_3d": stress3,
    }
    return stress, new_state, dWp, tangent, diagnostics


def plane_stress_return_3d(mat: J2PlaneStressMaterial, strain: Tensor,
                           state: MembraneState, **kwargs):
    """Convenience wrapper returning the full ``3x3`` plane-stress stress tensor."""
    stress, new_state, dWp, tangent, diagnostics = plane_stress_return(
        mat, strain, state, **kwargs)
    return diagnostics["stress_3d"], new_state, dWp, tangent, diagnostics


def mises_plane_stress(stress: Tensor) -> Tensor:
    """von Mises equivalent of a plane-stress tensor ``[sxx, syy, sxy]``.

    ``q = sqrt(sxx^2 - sxx syy + syy^2 + 3 sxy^2)`` -- the plane-stress form of
    ``q = sqrt(3/2 s:s)`` with ``s_zz = -(sxx + syy)``.  Using the deviatoric
    expression on *total* stresses (``3(sxx^2 + syy^2 + sxx syy)``) would inflate
    ``q`` by ``sqrt(3)`` in uniaxial tension, so this form is the one to keep.
    """
    sxx, syy, sxy = stress[..., 0], stress[..., 1], stress[..., 2]
    return torch.sqrt(torch.clamp(sxx * sxx - sxx * syy + syy * syy
                                  + 3.0 * sxy * sxy, min=0.0))


def uniaxial_plane_stress(mat: J2PlaneStressMaterial, strain_xx, *,
                          state: MembraneState | None = None):
    """Evaluate the plane-stress operator on the *closed-form* uniaxial strain path.

    The analytic small-strain solution of plane-stress uniaxial tension is

    ``alpha = (E eps - sigma_y)/(E + H)``, ``sigma = sigma_y + H alpha`` (or
    ``sigma = E eps`` when ``alpha = 0``), ``eps_yy = eps_zz = -nu sigma/E - alpha/2``.

    Feeding that strain triple to the return mapping is a two-sided check: it must
    return ``sigma_xx`` equal to the closed form *and* ``sigma_yy = sigma_zz = 0``
    with the same hardening parameter.  A stress-driven solve (root find on
    ``eps_yy``) is deliberately *not* used: ``sigma_yy(eps_yy)`` is monotone only on
    each branch and kinks at yield, so a bracketed bisection is invalid and an
    undamped Newton cycles across the kink (observed to diverge on unload paths).

    Returns ``(stress_xx, strain_yy, new_state, diagnostics)``.
    """
    exx = torch.as_tensor(strain_xx, dtype=torch.float64).flatten()
    m = exx.numel()
    st = MembraneState.virgin(m) if state is None else state
    alpha = mat.uniaxial_hardening_parameter(exx)
    sigma = mat.uniaxial_stress(exx)
    lateral = -mat.poisson * sigma / mat.young - 0.5 * alpha
    strain = torch.stack([exx, lateral, torch.zeros_like(exx)], dim=1)
    stress, new_state, _, _, diagnostics = plane_stress_return(mat, strain, st)
    return stress[:, 0], lateral, new_state, diagnostics


# --------------------------------------------------------------------------- #
# element kinematics / problem / assembly
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Q4Kinematics:
    b: Tensor       # (E, 4, 3, 8)
    weight: Tensor  # (E, 4) Gauss volume weight (detJ * thickness)
    det_j: Tensor   # (E, 4)


def q4_kinematics(nodes: Tensor, elements: Tensor, thickness: float) -> Q4Kinematics:
    """2x2 Gauss Q4 gradient operators and weights (full integration)."""
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("Q4 connectivity must be (E, 4)")
    x = nodes[elements]
    dndnat = torch.empty((4, 4, 2), dtype=nodes.dtype, device=nodes.device)
    for g, (xi, eta) in enumerate(_GAUSS_NAT):
        for i, (xn, en) in enumerate(_NODE_NAT):
            dndnat[g, i, 0] = 0.25 * xn * (1.0 + eta * en)
            dndnat[g, i, 1] = 0.25 * en * (1.0 + xi * xn)
    jac = torch.einsum("gnj,eni->egij", dndnat, x)
    det = torch.linalg.det(jac)
    if bool((det <= 0.0).any()):
        raise ValueError("non-positive Jacobian: check element node ordering")
    inv = torch.linalg.inv(jac)
    dndx = torch.einsum("egji,gnj->egni", inv, dndnat)
    b = torch.zeros((elements.shape[0], 4, 3, 8), dtype=nodes.dtype, device=nodes.device)
    for i in range(4):
        b[:, :, 0, 2 * i] = dndx[:, :, i, 0]
        b[:, :, 1, 2 * i + 1] = dndx[:, :, i, 1]
        b[:, :, 2, 2 * i] = dndx[:, :, i, 1]
        b[:, :, 2, 2 * i + 1] = dndx[:, :, i, 0]
    weight = det * float(thickness)
    return Q4Kinematics(b, weight, det)


@dataclass
class MembraneProblem:
    nodes: Tensor
    elements: Tensor
    thickness: float
    material: J2PlaneStressMaterial
    fixed_dofs: Tensor

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must be (n, 2)")
        if self.thickness <= 0.0:
            raise ValueError("thickness must be positive")
        two = torch.tensor([0, 1], dtype=self.elements.dtype, device=self.elements.device)
        self._element_dofs = (2 * self.elements[:, :, None] + two).reshape(
            self.elements.shape[0], 8)
        self._n_dofs = 2 * int(self.nodes.shape[0])
        self._kin = q4_kinematics(self.nodes, self.elements, self.thickness)
        self.fixed_dofs = torch.as_tensor(self.fixed_dofs, dtype=torch.long,
                                          device=self.nodes.device).flatten()

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    @property
    def element_dofs(self) -> Tensor:
        return self._element_dofs

    @property
    def kinematics(self) -> Q4Kinematics:
        return self._kin

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])

    @property
    def n_gauss(self) -> int:
        return int(self.elements.shape[0] * 4)


@dataclass
class MembraneResponse:
    displacement: Tensor
    fint: Tensor
    stiffness: Tensor | None
    stress: Tensor          # (E, 4, 3) [sxx, syy, sxy]
    strain: Tensor          # (E, 4, 3) [exx, eyy, gxy]
    state: MembraneState
    dWp: Tensor             # (E, 4) plastic work increments
    residual_norm: float = 0.0
    iterations: int = 0
    converged: bool = False

    def energy_density(self) -> Tensor:
        """Elastic strain energy density per Gauss point ``(E, 4)``."""
        s, e = self.stress, self.strain
        plastic = self.state.plastic_strain.reshape(self.stress.shape[0], 4, 3, 3)
        exx = e[..., 0] - plastic[..., 0, 0]
        eyy = e[..., 1] - plastic[..., 1, 1]
        gxy = e[..., 2] - 2.0 * plastic[..., 0, 1]
        return 0.5 * (s[..., 0] * exx + s[..., 1] * eyy + s[..., 2] * gxy)


def make_problem(nodes, elements, thickness, material, fixed_dofs) -> MembraneProblem:
    return MembraneProblem(torch.as_tensor(nodes, dtype=torch.float64),
                           torch.as_tensor(elements, dtype=torch.long),
                           float(thickness), material,
                           torch.as_tensor(fixed_dofs, dtype=torch.long))


def gauss_response(problem: MembraneProblem, u: Tensor, committed: MembraneState,
                   *, tangent: bool = True) -> MembraneResponse:
    """Strain -> return map at every Gauss point -> assemble internal force (and K)."""
    kin = problem.kinematics
    ue = u[problem.element_dofs]
    strain = torch.einsum("egij,ej->egi", kin.b, ue)
    e = problem.n_elements
    flat = strain.reshape(e * 4, 3)
    sigma, new_state, dWp, dmat, _ = plane_stress_return(problem.material, flat, committed)
    stress = sigma.reshape(e, 4, 3)
    dmat = dmat.reshape(e, 4, 3, 3)
    weight = kin.weight
    fint_e = torch.einsum("egki,egk,eg->ei", kin.b, stress, weight)
    fint = torch.zeros(problem.n_dofs, dtype=u.dtype, device=u.device)
    fint.index_add_(0, problem.element_dofs.reshape(-1), fint_e.reshape(-1))
    stiffness = None
    if tangent:
        ke = torch.einsum("egki,egkl,eglj,eg->eij", kin.b, dmat, kin.b, weight)
        ndof = problem.n_dofs
        rows = problem.element_dofs[:, :, None].expand(-1, 8, 8).reshape(-1)
        cols = problem.element_dofs[:, None, :].expand(-1, 8, 8).reshape(-1)
        flatk = torch.zeros(ndof * ndof, dtype=u.dtype, device=u.device)
        flatk.index_add_(0, rows * ndof + cols, ke.reshape(-1))
        stiffness = flatk.view(ndof, ndof)
    return MembraneResponse(u.clone(), fint, stiffness, stress, strain,
                            new_state, dWp.reshape(e, 4))


def _free_mask(problem: MembraneProblem, prescribed: Tensor | None) -> Tensor:
    mask = torch.ones(problem.n_dofs, dtype=torch.bool, device=problem.nodes.device)
    if prescribed is not None and len(prescribed):
        mask[prescribed] = False
    return mask


def elastic_solve(problem: MembraneProblem, fext: Tensor | None = None,
                  prescribed: Tensor | None = None, values: Tensor | None = None):
    """One linear-elastic plane-stress solve (elastic anchor / yield-load scaling)."""
    ndof = problem.n_dofs
    dtype = problem.nodes.dtype
    u = torch.zeros(ndof, dtype=dtype)
    idx = problem.fixed_dofs if prescribed is None else torch.as_tensor(prescribed)
    if idx is not None and len(idx):
        u[idx] = 0.0 if values is None else values.to(dtype)
    c = problem.material.plane_stress_matrix(dtype=dtype)
    kin = problem.kinematics
    ke = torch.einsum("egki,egkl,eglj,eg->eij", kin.b,
                      c[None, None].expand(problem.n_elements, 4, 3, 3),
                      kin.b, kin.weight)
    rows = problem.element_dofs[:, :, None].expand(-1, 8, 8).reshape(-1)
    cols = problem.element_dofs[:, None, :].expand(-1, 8, 8).reshape(-1)
    flatk = torch.zeros(ndof * ndof, dtype=dtype)
    flatk.index_add_(0, rows * ndof + cols, ke.reshape(-1))
    stiffness = flatk.view(ndof, ndof)
    free = _free_mask(problem, idx)
    f = torch.zeros(ndof, dtype=dtype) if fext is None else fext.to(dtype)
    if bool(free.any()):
        u[free] = torch.linalg.solve(stiffness[free][:, free], f[free])
    fint = stiffness @ u
    strain = torch.einsum("egij,ej->egi", kin.b, u[problem.element_dofs])
    stress = torch.einsum("egkl,egl->egk", c[None, None].expand(problem.n_elements, 4, 3, 3),
                          strain)
    return {"u": u, "fint": fint, "stiffness": stiffness, "strain": strain,
            "stress": stress, "weights": kin.weight}


def solve_step(problem: MembraneProblem, committed: MembraneState, u0: Tensor, *,
               prescribed: Tensor | None = None, values: Tensor | None = None,
               fext: Tensor | None = None, tol: float = 1e-11,
               max_iter: int = 40, line_search: int = 6):
    """One Newton increment from a committed state (the state is not mutated)."""
    u = u0.clone()
    if prescribed is not None and len(prescribed):
        u[prescribed] = values
    free = _free_mask(problem, prescribed)
    if fext is None:
        fext = torch.zeros(problem.n_dofs, dtype=u.dtype)
    resp = gauss_response(problem, u, committed)
    scale = max(float(fext.abs().max()) if fext.numel() else 0.0,
                float(resp.fint.abs().max()) if resp.fint.numel() else 0.0, 1e-30)
    for _ in range(max_iter):
        r = fext - resp.fint
        rn = float(r[free].abs().max()) / scale if bool(free.any()) else 0.0
        resp.residual_norm, resp.iterations = rn, resp.iterations + 1
        if rn <= tol:
            resp.converged = True
            return u, resp, True
        du = torch.linalg.solve(resp.stiffness[free][:, free], r[free])
        step = du.clone()
        accepted = False
        for _ in range(line_search):
            trial = u.clone()
            trial[free] += step
            trial_resp = gauss_response(problem, trial, committed, tangent=False)
            trial_rn = (float((fext - trial_resp.fint)[free].abs().max()) / scale
                        if bool(free.any()) else 0.0)
            if trial_rn < rn or trial_rn <= tol:
                u = trial
                accepted = True
                break
            step = 0.5 * step
        if not accepted:
            return u, resp, False
        resp = gauss_response(problem, u, committed)
    resp.converged = False
    return u, resp, False


def _advance(problem: MembraneProblem, state: MembraneState, u: Tensor, idx: Tensor,
             values: Tensor, fext_from: Tensor, fext_to: Tensor, tol: float,
             max_iter: int, depth: int = 0, max_depth: int = 5):
    """Newton increment with automatic bisection of a failing load step."""
    u_new, resp, ok = solve_step(problem, state, u, prescribed=idx, values=values,
                                 fext=fext_to, tol=tol, max_iter=max_iter)
    if ok or depth >= max_depth:
        return u_new, resp, ok, 2 ** depth
    middle = 0.5 * (u[idx] + values)
    half_fext = 0.5 * (fext_from + fext_to)
    u1, r1, ok1, c1 = _advance(problem, state, u, idx, middle, fext_from, half_fext,
                               tol, max_iter, depth + 1, max_depth)
    if not ok1:
        return u1, r1, False, c1
    u2, r2, ok2, c2 = _advance(problem, r1.state, u1, idx, values, half_fext, fext_to,
                               tol, max_iter, depth + 1, max_depth)
    return u2, r2, ok2, c1 + c2


def run_path(problem: MembraneProblem, targets, *, tol: float = 1e-11,
             max_iter: int = 40, state: MembraneState | None = None,
             u0: Tensor | None = None, response_dofs=None) -> dict:
    """Drive a sequence of increments and audit energies at every step.

    ``targets`` = list of ``{"fext": Tensor|None, "prescribed": (idx, values)|None,
    "label": str}``.  The external work is integrated along the equilibrium path
    (``W_ext = int fint . du`` over converged increments, trapezoidal), the stored
    energy from ``0.5 sigma:eps_e`` and the plastic work from the closed-form
    hardening integral.
    """
    state = MembraneState.virgin(problem.n_gauss) if state is None else state.clone()
    u = torch.zeros(problem.n_dofs, dtype=torch.float64) if u0 is None else u0.clone()
    if problem.fixed_dofs.numel():
        u[problem.fixed_dofs] = 0.0
    history = []
    w_ext = 0.0
    w_plastic_incremental = 0.0
    fint_prev = torch.zeros_like(u)
    fext_prev = torch.zeros_like(u)
    for target in targets:
        fext = target.get("fext")
        fext = torch.zeros_like(u) if fext is None else fext.to(u.dtype)
        presc = target.get("prescribed")
        idx = problem.fixed_dofs
        vals = torch.zeros(idx.numel(), dtype=u.dtype)
        if presc is not None:
            idx = torch.cat([problem.fixed_dofs,
                             torch.as_tensor(presc[0], dtype=torch.long)])
            vals = torch.cat([torch.zeros(problem.fixed_dofs.numel(), dtype=u.dtype),
                              torch.as_tensor(presc[1], dtype=u.dtype)])
        u_new, resp, ok, substeps = _advance(problem, state, u, idx, vals,
                                             fint_prev * 0.0 + fext_prev, fext, tol,
                                             max_iter)
        if not ok:
            history.append({"label": target.get("label", ""), "converged": False,
                            "substeps": substeps,
                            "residual": resp.residual_norm,
                            "iterations": resp.iterations})
            break
        w_ext += float(0.5 * (fint_prev + resp.fint) @ (u_new - u))
        kin = problem.kinematics
        elastic_energy = float((resp.energy_density() * kin.weight).sum())
        alpha = resp.state.alpha.reshape(problem.n_elements, 4)
        plastic_work = float((problem.material.plastic_work_closed_form(alpha)
                              * kin.weight).sum())
        w_plastic_incremental += float((resp.dWp * kin.weight).sum())
        entry = {
            "label": target.get("label", ""),
            "converged": True,
            "substeps": substeps,
            "iterations": resp.iterations,
            "residual": resp.residual_norm,
            "max_mises": float(mises_plane_stress(resp.stress).max()),
            "max_alpha": float(alpha.max()),
            "elastic_energy": elastic_energy,
            "plastic_work": plastic_work,
            "plastic_work_incremental": w_plastic_incremental,
            "external_work": w_ext,
        }
        if response_dofs is not None:
            entry["response"] = [float(u_new[d]) for d in response_dofs]
        if presc is not None:
            entry["reaction_sum"] = float(resp.fint[torch.as_tensor(presc[0])].sum())
            entry["prescribed_values"] = [float(v) for v in torch.as_tensor(presc[1])]
        history.append(entry)
        u, fint_prev = u_new, resp.fint.clone()
        fext_prev = fext.clone()
        state = resp.state
        if target.get("stop", False):
            break
    return {"history": history, "state": state, "displacement": u,
            "external_work": w_ext}


# --------------------------------------------------------------------------- #
# benchmark builders
# --------------------------------------------------------------------------- #
@dataclass
class CookMembraneModel:
    problem: MembraneProblem
    response_dof: int
    load_pattern: Tensor        # unit-resultant nodal load vector
    right_edge: Tensor
    left_edge: Tensor

    def load(self, resultant: float) -> Tensor:
        return self.load_pattern * float(resultant)


def cook_membrane_model(n: int, material: J2PlaneStressMaterial, *,
                        thickness: float = 1.0, dtype=torch.float64) -> CookMembraneModel:
    """Classical Cook tapered membrane, published geometry (Simo & Rifai 1990)."""
    from .continuum_benchmarks import mapped_q4_mesh
    if n % 2:
        raise ValueError("Cook model requires an even subdivision count")
    corners = torch.tensor(((0., 0.), (48., 44.), (48., 60.), (0., 44.)), dtype=dtype)
    nodes, elements = mapped_q4_mesh(corners, n, n)
    right = torch.arange(n, (n + 1) * (n + 1), n + 1)
    left = torch.arange(0, (n + 1) * (n + 1), n + 1)
    fixed = torch.stack((2 * left, 2 * left + 1), 1).reshape(-1)
    pattern = torch.zeros(2 * nodes.shape[0], dtype=dtype)
    weights = torch.full((n + 1,), 1.0 / n, dtype=dtype)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    pattern[2 * right + 1] = weights
    response_dof = 2 * int(right[n // 2]) + 1
    problem = MembraneProblem(nodes, elements, float(thickness), material, fixed)
    return CookMembraneModel(problem, response_dof, pattern, right, left)


@dataclass
class DentSpecimen:
    problem: MembraneProblem
    top_edge: Tensor
    bottom_edge: Tensor
    ligament_width: float
    limit_load: float
    material: J2PlaneStressMaterial
    thickness: float


def dent_specimen_model(nx: int, ny: int, *, width: float, height: float,
                        notch_length: float, notch_height: float,
                        thickness: float, material: J2PlaneStressMaterial,
                        dtype=torch.float64) -> DentSpecimen:
    """Double-edge-notched plane-stress tension specimen (stair-step notches).

    The meshed ``width x height`` rectangle carries two symmetrical side notches of
    depth ``notch_length`` and height ``notch_height`` centred on mid-height.  The
    elements inside the notches are removed and orphan nodes dropped, so the
    ligament width is ``width - 2 * notch_length`` exactly (mesh aligned).
    """
    from .continuum_benchmarks import mapped_q4_mesh
    corners = torch.tensor(((0., 0.), (width, 0.), (width, height), (0., height)),
                           dtype=dtype)
    nodes, elements = mapped_q4_mesh(corners, nx, ny)
    centroid = nodes[elements].mean(dim=1)
    in_notch = (((centroid[:, 0] < notch_length) | (centroid[:, 0] > width - notch_length))
                & (centroid[:, 1] > 0.5 * height - 0.5 * notch_height)
                & (centroid[:, 1] < 0.5 * height + 0.5 * notch_height))
    keep = elements[~in_notch]
    if keep.shape[0] == 0:
        raise ValueError("notch removal deleted the whole mesh")
    used = torch.unique(keep)
    remap = torch.full((nodes.shape[0],), -1, dtype=torch.long)
    remap[used] = torch.arange(used.numel())
    nodes = nodes[used]
    keep = remap[keep]
    ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
    tol_y = 1e-9 * max(height, 1.0)
    bottom = torch.nonzero((nodes[:, 1] - ymin).abs() < tol_y, as_tuple=False).flatten()
    top = torch.nonzero((nodes[:, 1] - ymax).abs() < tol_y, as_tuple=False).flatten()
    fixed = torch.cat([2 * bottom, 2 * bottom + 1,
                       torch.tensor([2 * int(bottom[0])], dtype=torch.long)])
    problem = MembraneProblem(nodes, keep, float(thickness), material,
                              torch.unique(fixed))
    ligament = width - 2.0 * notch_length
    return DentSpecimen(problem, 2 * top + 1, 2 * bottom + 1, ligament,
                        material.yield_stress * thickness * ligament, material,
                        float(thickness))