"""Unified foundation interface for the Q4 Mindlin plate + tangential contact stiffness.

Why this module exists
----------------------
The repository already carries the pieces of a plate-on-foundation model
(``plate.q4_mindlin_stiffness``, ``plate_mindlin.q4_winkler_foundation_stiffness``,
``foundation_plate_bending.q4_pasternak_shear_stiffness``,
``elastic_halfspace.settlement``) but they are *not* interchangeable: each one
lives in its own solver, each one is validated against its own reference, and no
common reference exists that carries all of the physics at once.  That makes two
engineering questions unanswerable:

1. *How much does the interface tangential contact stiffness matter?*  For a
   Mindlin plate the cross-section rotation ``psi`` is an independent field, so
   the plate's bottom face moves tangentially by ``u_b = z psi`` at ``z = -h/2``.
   A bonded (no-slip) interface therefore exerts a *moment* on the section,

       m_t = R psi,        R = k_t h^2 / 4  [N/m],

   where ``k_t`` [N/m^3] is the tangential contact stiffness per unit area of the
   interface.  This term is *absent* from both the Winkler and the Pasternak
   models as they are normally written, and it cannot be represented by them:
   it acts on the rotation DOFs and scales with ``h^2``.  It is derived and
   implemented here as :func:`q4_mindlin_tangential_contact_stiffness`.

2. *Is a foundation-model switch safe?*  Winkler (``p = k w``), Pasternak
   (``p = k w - G_p lap(w)``, ``G_p`` = tangential/shear-layer stiffness [N/m])
   and an elastic half-space (Boussinesq patch compliance) differ in *kind* --
   local vs non-local -- so a switch can change the deflection by tens of
   per cent.  Here all three are reachable through one object
   (:func:`make_foundation`), one solver (:func:`solve_mindlin_plate_foundation`)
   and one exact reference (:func:`fsdt_navier_ss_deflection`) that carries
   ``k``, ``G_p`` and ``k_t`` simultaneously.

Exact reference: first-order shear deformation (Mindlin-Timoshenko) Navier solution
----------------------------------------------------------------------------------
For a hard simply supported rectangular plate (``w = 0`` and zero tangential
rotation on the edges), double-sine expansion with
``lam = (m pi/a)^2 + (n pi/b)^2`` gives the modal amplitude (derivation in
``docs/verification_P9263_N01_mindlin_tangential_foundation.md``)

    W_mn = q_mn / [ k + G_p lam + S lam (R + D lam) / (R + D lam + S) ],

with ``D = E h^3 / (12 (1 - nu^2))``, ``S = kappa G h``, ``G = E / (2 (1 + nu))``,
``R = k_t h^2 / 4`` and the uniform-load coefficients
``q_mn = 16 q / (pi^2 m n)`` for odd ``m, n``.  Three limits are built into that
single formula and are used as analytic checks:

* ``k_t = 0`` (``R = 0``)  ->  denominator ``k + G_p lam + D lam^2 / (1 + D lam / S)``,
  i.e. the classical Mindlin plate on a two-parameter foundation
  (Kirchhoff plus the first-order shear deflection ``q_mn / (S lam)``).
* ``S -> infinity`` (Kirchhoff limit)  ->  denominator ``k + G_p lam + D lam^2``.
* ``k_t -> infinity`` (perfectly bonded interface, ``psi = 0`` everywhere)
  ->  denominator ``k + G_p lam + S lam``, which is the Fourier symbol of
  ``S lap(w) + k w - G_p lap(w) + q = 0`` with ``w = 0`` on the edges -- the pure
  transverse-shear (Poisson) problem.  That is an *independent* analytic target
  for the tangential contact term: no bending stiffness at all.

Elastic half-space model
------------------------
The half-space is non-local: a loaded patch settles under the Boussinesq kernel
``w(x) = (1 - nu^2)/(pi E) int_A p / |x - xi| dA``.  The model here is the
patch-influence ("boundary-element") discretisation: every plate element is one
uniformly loaded patch, the patch-average settlement matrix
``C_p[i][j]`` is integrated with the closed-form rectangle integral of
``elastic_halfspace.settlement``, and the plate is coupled through the
energy-equivalent transverse stiffness ``K = D M^-1 D`` with
``M[i][j] = A_i C_p[i][j]`` (``M`` is symmetric by Betti reciprocity, ``K`` is
symmetric positive definite because the kernel matrix is).  The model is graded
against the classical elastic-half-space settlement factors for a uniformly
loaded flexible square (``w_avg = 0.95 q B (1 - nu^2)/E``,
``w_centre = 1.12 ...``, Poulos & Davis 1974).

Scope limits (do not over-read)
-------------------------------
* Isotropic linear elasticity, one shear correction factor (``5/6``), uniform
  thickness, hard simple support (SS1) or clamped edges; no contact
  non-linearity (tensionless/lift-off is in ``foundation_plate_tensionless``),
  no plasticity, no in-plane (membrane) DOFs -- the tangential contact term here
  couples the *section rotation* to the interface, it is not a membrane
  Coulomb-friction model.
* The half-space patch model resolves the loaded area only; it does not model a
  layered (multi-stratum) ground (``kerr_foundation`` is the three-parameter
  alternative for that).
* ``k_t -> infinity`` is imposed through the exact Navier limit, not through a
  penalty in the FE solve; the FE tangential matrix is used at finite ``k_t``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .elastic_halfspace import settlement as half_space_settlement
from .foundation_plate_bending import q4_pasternak_shear_stiffness
from .plate import _shape, q4_mindlin_stiffness, structured_square_mesh
from .plate_mindlin import q4_winkler_foundation_stiffness

KAPPA_DEFAULT = 5.0 / 6.0

_G = 1.0 / math.sqrt(3.0)
_GAUSS_2 = ((-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G))


# ---------------------------------------------------------------------------
# Element matrices
# ---------------------------------------------------------------------------

def q4_normal_foundation_stiffness(xy: torch.Tensor, k_normal: float) -> torch.Tensor:
    """Winkler normal stiffness ``k int N^T N dA`` on the transverse DOFs.

    ``p = k w`` is the foundation pressure, so the weak form contributes
    ``k int N_i N_j dA`` to the (w, w) entries (DOF ``3i``).  Delegates to
    :func:`tensorfem.plate_mindlin.q4_winkler_foundation_stiffness`.
    """
    if k_normal < 0.0:
        raise ValueError("normal foundation stiffness must be non-negative")
    if k_normal == 0.0:
        return torch.zeros((12, 12), dtype=xy.dtype, device=xy.device)
    return q4_winkler_foundation_stiffness(xy, float(k_normal))


def q4_shear_layer_stiffness(xy: torch.Tensor, g_shear: float) -> torch.Tensor:
    """Pasternak tangential (shear-layer) stiffness ``G int grad(N) . grad(N)^T dA``.

    ``p = k w - G_p lap(w)`` integrated by parts gives the shear-layer energy
    ``(G_p / 2) int |grad w|^2 dA``; ``G_p`` [N/m] is the *tangential contact
    stiffness* of the interface (a soft layer of shear modulus ``G_f`` and
    thickness ``h_f`` has ``G_p = G_f h_f``).  Constant ``w`` carries no energy,
    so the block is positive semi-definite (rank 3 on a Q4 element).
    """
    if g_shear < 0.0:
        raise ValueError("shear-layer (tangential) stiffness must be non-negative")
    return q4_pasternak_shear_stiffness(xy, float(g_shear))


def q4_mindlin_tangential_contact_stiffness(
    xy: torch.Tensor,
    k_tangential: float,
    thickness: float,
) -> torch.Tensor:
    """Mindlin plate tangential contact stiffness matrix (rotation DOFs).

    Derivation
    ----------
    With the plate occupying ``z in [-h/2, h/2]`` and Mindlin kinematics
    ``u(z) = z psi_x``, ``v(z) = z psi_y``, a bonded interface transmits a
    tangential traction ``tau = k_t u_b`` at the plate bottom face
    ``z = -h/2``, where ``u_b = -(h/2) psi_x``.  The interface energy is

        U_t = (1/2) int k_t (u_b^2 + v_b^2) dA
            = (1/2) int (k_t h^2 / 4) (psi_x^2 + psi_y^2) dA
            = (1/2) int m_t . psi dA,        m_t = R psi,  R = k_t h^2 / 4.

    Hence the 12x12 element matrix carries ``R int N^T N dA`` on the rotation
    diagonal blocks ``(3i+1, 3j+1)`` and ``(3i+2, 3j+2)`` and nothing else.  It
    vanishes for ``h -> 0`` (consistently: a vanishing section has no lever arm),
    which is why it is a *Mindlin* (thickness-dependent) contact term.  In the
    exact Navier solution it reduces the modal amplitude to the pure shear
    (Poisson) problem as ``k_t -> infinity``.

    Parameters
    ----------
    xy : (4, 2) element node coordinates.
    k_tangential : interface tangential contact stiffness [N/m^3] >= 0.
    thickness : plate thickness ``h`` [m].

    Returns
    -------
    (12, 12) tensor, rotation blocks only.
    """
    if k_tangential < 0.0:
        raise ValueError("tangential contact stiffness must be non-negative")
    dtype, device = xy.dtype, xy.device
    ke = torch.zeros((12, 12), dtype=dtype, device=device)
    if k_tangential == 0.0 or thickness == 0.0:
        return ke
    rigidity = float(k_tangential) * float(thickness) ** 2 / 4.0
    block = torch.zeros((4, 4), dtype=dtype, device=device)
    for xif, etaf in _GAUSS_2:
        xi, eta = xy.new_tensor(xif), xy.new_tensor(etaf)
        N, dN_nat = _shape(xi, eta)
        detJ = torch.linalg.det(dN_nat.T @ xy)
        block = block + rigidity * torch.outer(N, N) * detJ
    for offset in (1, 2):
        idx = torch.arange(4, device=device) * 3 + offset
        ke[idx[:, None], idx] += block
    return ke


# ---------------------------------------------------------------------------
# Elastic half-space patch model
# ---------------------------------------------------------------------------

def half_space_patch_compliance(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young: float,
    poisson: float,
    *,
    gauss: int = 2,
) -> torch.Tensor:
    """Patch-average settlement matrix ``C_p`` [m/Pa] of a Boussinesq half-space.

    ``C_p[i][j]`` is the settlement of patch ``i`` *averaged over its area* when
    a unit uniform pressure acts on patch ``j``.  Each plate element is treated
    as one uniformly loaded rectangular patch; the source integral over ``j`` is
    closed-form (:func:`tensorfem.elastic_halfspace.settlement`), the receiver
    average is a ``gauss x gauss`` rule over ``i``.  ``A_i C_p[i][j]`` is
    symmetric by Betti reciprocity -- asserted in the test suite.
    """
    if young <= 0.0:
        raise ValueError("young must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson out of range")
    nodes_np = nodes.detach().cpu().numpy()
    elem_np = elements.detach().cpu().numpy()
    n_elem = elem_np.shape[0]
    x1 = nodes_np[elem_np, 0].min(axis=1)
    x2 = nodes_np[elem_np, 0].max(axis=1)
    y1 = nodes_np[elem_np, 1].min(axis=1)
    y2 = nodes_np[elem_np, 1].max(axis=1)
    lx = x2 - x1
    ly = y2 - y1
    if (lx <= 0.0).any() or (ly <= 0.0).any():
        raise ValueError("degenerate plate element in mesh")

    if gauss == 2:
        fracs = (0.5 - _G / 2.0, 0.5 + _G / 2.0)
    elif gauss == 1:
        fracs = (0.5,)
    else:  # pragma: no cover - defensive
        raise ValueError("only gauss=1 or gauss=2 is supported")
    c_np = torch.zeros((n_elem, n_elem), dtype=torch.float64).numpy()
    for fx in fracs:
        for fy in fracs:
            xg = x1 + fx * lx
            yg = y1 + fy * ly
            for j in range(n_elem):
                # settlement at the Gauss point due to unit pressure on patch j
                dx = xg - x1[j]
                dy = yg - y1[j]
                c_np[:, j] += [
                    half_space_settlement(1.0, young, poisson, lx[j], ly[j], a, b)
                    for a, b in zip(dx.tolist(), dy.tolist())
                ]
    c_np /= float(len(fracs) ** 2)
    return torch.from_numpy(c_np).to(dtype=nodes.dtype)


def half_space_transverse_stiffness(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young: float,
    poisson: float,
    *,
    gauss: int = 2,
) -> torch.Tensor:
    """Transverse (w-only) stiffness ``B^T D M^-1 D B`` of a Boussinesq half-space.

    ``B`` averages the four corner ``w`` values onto the patch and
    ``D = diag(A_i)``; see the module docstring for the energy argument.
    Returned matrix is ``(n_node, n_node)`` acting on the transverse DOF only.
    """
    c_p = half_space_patch_compliance(nodes, elements, young, poisson, gauss=gauss)
    area = _element_areas(nodes, elements)
    m = c_p * area[:, None]            # M[i][j] = A_i C_p[i][j], symmetric by Betti
    m = 0.5 * (m + m.T)                # remove round-off asymmetry
    k_patch = diag_scale(torch.linalg.inv(m), area)   # D M^-1 D  [N/m^2 per patch]
    k_patch = 0.5 * (k_patch + k_patch.T)

    n_node = nodes.shape[0]
    n_epn = elements.shape[1]
    scale = 1.0 / float(n_epn)
    idx = elements.detach().to(torch.long)
    k_w = torch.zeros((n_node, n_node), dtype=nodes.dtype)
    for a in range(n_epn):
        for b in range(n_epn):
            _accumulate_dense(k_w, idx[:, a], idx[:, b], k_patch * (scale * scale))
    return 0.5 * (k_w + k_w.T)


def diag_scale(matrix: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``diag(scale) @ matrix @ diag(scale)`` without forming the diagonal."""
    return matrix * scale[:, None] * scale[None, :]


def _accumulate_dense(target: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
                      values: torch.Tensor) -> None:
    """Scatter-add a dense block ``values`` onto ``target[rows, cols]``."""
    target.index_put_((rows[:, None], cols[None, :]), values, accumulate=True)


def _element_areas(nodes: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
    """Areas of the (rectangular) Q4 elements."""
    xy = nodes[elements]
    x1, x2 = xy[:, :, 0].min(dim=1).values, xy[:, :, 0].max(dim=1).values
    y1, y2 = xy[:, :, 1].min(dim=1).values, xy[:, :, 1].max(dim=1).values
    return (x2 - x1) * (y2 - y1)


# ---------------------------------------------------------------------------
# Unified foundation models
# ---------------------------------------------------------------------------

class FoundationModel:
    """Common interface for Winkler / Pasternak / elastic half-space foundations."""

    kind = "abstract"
    non_local = False

    def element_stiffness(self, xy: torch.Tensor, *, thickness: float = 0.0) -> torch.Tensor:
        raise NotImplementedError

    def navier_extra(self, lam: float) -> float:
        """Term added to the Navier denominator by this foundation alone."""
        raise NotImplementedError

    def global_transverse_stiffness(self, nodes: torch.Tensor,
                                    elements: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.kind} foundation has no non-local transverse block"
        )


@dataclass(frozen=True)
class WinklerFoundation(FoundationModel):
    """One-parameter foundation: pressure ``p = k w`` (no tangential coupling)."""

    k_normal: float
    kind = "winkler"
    non_local = False

    def __post_init__(self) -> None:
        if self.k_normal < 0.0:
            raise ValueError("k_normal must be non-negative")

    def element_stiffness(self, xy: torch.Tensor, *, thickness: float = 0.0) -> torch.Tensor:
        return q4_normal_foundation_stiffness(xy, self.k_normal)

    def navier_extra(self, lam: float) -> float:
        return float(self.k_normal)


@dataclass(frozen=True)
class PasternakFoundation(FoundationModel):
    """Two-parameter foundation: ``p = k w - G_p lap(w)``.

    ``g_shear`` (``G_p`` [N/m]) is the tangential contact stiffness of the
    interface/shear layer; ``G_p > 0`` always raises the plate stiffness.
    """

    k_normal: float
    g_shear: float
    kind = "pasternak"
    non_local = False

    def __post_init__(self) -> None:
        if self.k_normal < 0.0 or self.g_shear < 0.0:
            raise ValueError("Pasternak parameters must be non-negative")

    def element_stiffness(self, xy: torch.Tensor, *, thickness: float = 0.0) -> torch.Tensor:
        return (q4_normal_foundation_stiffness(xy, self.k_normal)
                + q4_shear_layer_stiffness(xy, self.g_shear))

    def navier_extra(self, lam: float) -> float:
        return float(self.k_normal) + float(self.g_shear) * lam


@dataclass(frozen=True)
class ElasticHalfSpaceFoundation(FoundationModel):
    """Elastic half-space (Boussinesq) foundation, non-local patch model."""

    young: float
    poisson: float
    kind = "half_space"
    non_local = True

    def __post_init__(self) -> None:
        if self.young <= 0.0:
            raise ValueError("young must be positive")
        if not -1.0 < self.poisson < 0.5:
            raise ValueError("poisson out of range")

    def element_stiffness(self, xy: torch.Tensor, *, thickness: float = 0.0) -> torch.Tensor:
        raise NotImplementedError(
            "the elastic half-space is non-local; assemble it with "
            "global_transverse_stiffness() over the whole mesh"
        )

    def navier_extra(self, lam: float) -> float:
        raise NotImplementedError(
            "the elastic half-space has no local (k, G_p) Navier symbol; "
            "grade it against the Boussinesq settlement factors instead"
        )

    def global_transverse_stiffness(self, nodes: torch.Tensor,
                                    elements: torch.Tensor) -> torch.Tensor:
        return half_space_transverse_stiffness(nodes, elements, self.young, self.poisson)


_FOUNDATION_ALIASES = {
    "winkler": "winkler",
    "one_parameter": "winkler",
    "pasternak": "pasternak",
    "two_parameter": "pasternak",
    "half_space": "half_space",
    "half-space": "half_space",
    "elastic_half_space": "half_space",
    "boussinesq": "half_space",
}


def make_foundation(kind: str, **params: float) -> FoundationModel:
    """One-click factory: ``make_foundation("pasternak", k_normal=..., g_shear=...)``.

    Accepted kinds: ``winkler``, ``pasternak``, ``half_space`` (aliases
    ``one_parameter``, ``two_parameter``, ``elastic_half_space``, ``boussinesq``).
    """
    key = _FOUNDATION_ALIASES.get(str(kind).strip().lower())
    if key is None:
        raise ValueError(f"unknown foundation kind {kind!r}")
    if key == "winkler":
        return WinklerFoundation(**{k: params[k] for k in ("k_normal",) if k in params})
    if key == "pasternak":
        allowed = {k: params[k] for k in ("k_normal", "g_shear") if k in params}
        return PasternakFoundation(**allowed)
    allowed = {k: params[k] for k in ("young", "poisson") if k in params}
    return ElasticHalfSpaceFoundation(**allowed)


# ---------------------------------------------------------------------------
# Solver: Mindlin plate on any foundation model
# ---------------------------------------------------------------------------

@dataclass
class PlateFoundationSolution:
    """Outcome of :func:`solve_mindlin_plate_foundation`."""

    displacement: torch.Tensor     # (3*n_node,) [w, psi_x, psi_y] per node
    nodes: torch.Tensor            # (n_node, 2)
    elements: torch.Tensor         # (n_elem, 4)
    young: float
    poisson: float
    thickness: float
    side: float
    boundary: str
    shear_treatment: str
    foundation: FoundationModel
    tangential_contact: float
    kappa: float
    _patch_pressure: Optional[torch.Tensor] = None
    _reactions: Optional[torch.Tensor] = None
    _load_total: float = 0.0

    @property
    def transverse(self) -> torch.Tensor:
        """Nodal transverse displacements ``w``."""
        return self.displacement[0::3]

    @property
    def rotations(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.displacement[1::3], self.displacement[2::3]

    @property
    def centre_node(self) -> int:
        centroid = self.nodes.mean(dim=0)
        return int(torch.argmin(torch.linalg.norm(self.nodes - centroid, dim=1)))

    def centre_deflection(self) -> float:
        """Transverse displacement at the node closest to the plate centre."""
        return float(self.transverse[self.centre_node])

    def foundation_pressure(self) -> torch.Tensor:
        """Foundation reaction per patch (half-space) or per node (local models)."""
        if self._patch_pressure is not None:
            return self._patch_pressure
        w = self.transverse
        if isinstance(self.foundation, (WinklerFoundation, PasternakFoundation)):
            return float(self.foundation.k_normal) * w
        raise NotImplementedError("pressure field unavailable for this model")

    def support_reaction(self) -> float:
        """Total transverse reaction carried by the edge supports [N], positive up.

        ``K u - f`` at a constrained DOF is minus the external force needed to
        hold that DOF (downward loads are positive in ``f``), so the upward
        support reaction is its negative.
        """
        if self._reactions is None:
            raise RuntimeError("reactions are not available for this solution")
        return -float(self._reactions[0::3].sum())

    def foundation_total_reaction(self) -> float:
        """Total transverse reaction carried by the foundation [N].

        Local models: ``int k w dA`` evaluated element-wise (one-point rule on
        the transverse field); non-local models: the equilibrium complement
        ``Q_applied - support_reaction`` -- the patch-pressure field is an
        ill-conditioned inverse (see the module docstring) and its *sum* is the
        well-posed quantity.
        """
        if isinstance(self.foundation, WinklerFoundation):
            w_patch = self.transverse[self.elements].mean(dim=1)
            area = _element_areas(self.nodes, self.elements)
            return float(self.foundation.k_normal * (w_patch * area).sum())
        return self._load_total - self.support_reaction()

    def equilibrium_residual(self) -> float:
        """``|(foundation + supports)/applied - 1|`` -- expected to be ~1e-15."""
        total = self.foundation_total_reaction() + self.support_reaction()
        if self._load_total == 0.0:
            return abs(total)
        return abs(total / self._load_total - 1.0)


def solve_mindlin_plate_foundation(
    n: int,
    *,
    side: float = 1.0,
    young: float = 2.0e11,
    poisson: float = 0.3,
    thickness: float = 0.05,
    pressure: float | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = 0.0,
    boundary: str = "simply_supported",
    foundation: Optional[FoundationModel] = None,
    tangential_contact: float = 0.0,
    kappa: float = KAPPA_DEFAULT,
    shear_treatment: str = "mitc",
    dtype=torch.float64,
) -> PlateFoundationSolution:
    """Solve a square Q4 Mindlin plate on a switchable foundation model.

    Parameters
    ----------
    n : number of elements per side.
    pressure : uniform pressure [Pa] or ``callable(x, y) -> scalar``.
    boundary : ``"simply_supported"`` (SS1: ``w = 0`` and zero tangential
        rotation on the edges) or ``"clamped"``.
    foundation : any :class:`FoundationModel`; ``None`` means no foundation.
    tangential_contact : ``k_t`` [N/m^3], the interface tangential contact
        stiffness applied through :func:`q4_mindlin_tangential_contact_stiffness`.
    shear_treatment : ``"mitc"`` (default, Dvorkin-Bathe assumed shear strain) or
        ``"sri"`` (historical one-point shear).
    """
    if boundary not in ("simply_supported", "clamped"):
        raise ValueError("boundary must be 'simply_supported' or 'clamped'")
    nodes, elements = structured_square_mesh(n, side, dtype=dtype)
    n_node = nodes.shape[0]
    ndof = 3 * n_node

    K = torch.zeros((ndof, ndof), dtype=dtype)
    f = torch.zeros(ndof, dtype=dtype)

    pressure_fn = _as_pressure_fn(pressure, dtype)
    for conn in elements:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, young, poisson, thickness, shear_correction=kappa,
                                  shear_treatment=shear_treatment)
        if foundation is not None and not foundation.non_local:
            ke = ke + foundation.element_stiffness(xy, thickness=thickness)
        if tangential_contact:
            ke = ke + q4_mindlin_tangential_contact_stiffness(
                xy, tangential_contact, thickness)
        fe = _consistent_load(xy, pressure_fn)
        dofs = _element_dofs(conn)
        K[dofs[:, None], dofs] += ke
        f[dofs] += fe

    patch_pressure = None
    if foundation is not None and foundation.non_local:
        k_w = foundation.global_transverse_stiffness(nodes, elements)
        idx = 3 * torch.arange(n_node)
        K[idx[:, None], idx] += k_w

    fixed = _boundary_dofs(nodes, side, boundary)
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    u = torch.zeros(ndof, dtype=dtype)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])

    solution = PlateFoundationSolution(
        displacement=u, nodes=nodes, elements=elements, young=young, poisson=poisson,
        thickness=thickness, side=side, boundary=boundary,
        shear_treatment=shear_treatment, foundation=foundation if foundation else WinklerFoundation(0.0),
        tangential_contact=float(tangential_contact), kappa=kappa,
        _reactions=K @ u - f,
        _load_total=float(f[0::3].sum()),
    )
    if isinstance(foundation, ElasticHalfSpaceFoundation):
        # patch pressure from the inverse of the compliance relation w_avg = C_p p
        c_p = half_space_patch_compliance(nodes, elements, foundation.young, foundation.poisson)
        w_patch = solution.transverse[elements].mean(dim=1)
        patch_pressure = torch.linalg.solve(0.5 * (c_p + c_p.T), w_patch)
    solution._patch_pressure = patch_pressure
    return solution


def _as_pressure_fn(pressure, dtype) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if callable(pressure):
        return pressure
    value = float(pressure)

    def uniform(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(value)

    return uniform


def _consistent_load(xy: torch.Tensor, pressure_fn) -> torch.Tensor:
    """Consistent nodal load from a transverse pressure field (only ``w`` DOFs)."""
    fe = torch.zeros(12, dtype=xy.dtype, device=xy.device)
    for xif, etaf in _GAUSS_2:
        xi, eta = xy.new_tensor(xif), xy.new_tensor(etaf)
        N, dN_nat = _shape(xi, eta)
        detJ = torch.linalg.det(dN_nat.T @ xy)
        p = N @ xy
        q = pressure_fn(p[0], p[1])
        fe[0::3] += N * q * detJ
    return fe


def _element_dofs(conn: torch.Tensor) -> torch.Tensor:
    return torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)


def _boundary_dofs(nodes: torch.Tensor, side: float, boundary: str) -> torch.Tensor:
    tol = side * 1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - side).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - side).abs() < tol)
    edge = xedge | yedge
    if boundary == "simply_supported":
        w_dofs = 3 * torch.nonzero(edge, as_tuple=False).flatten()
        psi_x = 3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1
        psi_y = 3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2
        return torch.cat((w_dofs, psi_x, psi_y)).unique()
    edge_idx = torch.nonzero(edge, as_tuple=False).flatten()
    return torch.cat((3 * edge_idx, 3 * edge_idx + 1, 3 * edge_idx + 2)).unique()


def element_bending_moments(solution: PlateFoundationSolution) -> dict:
    """Bending moments per element at its centre: ``Mx, My, Mxy`` [N].

    ``M = D_b kappa`` with ``kappa = (psi_x,x, psi_y,y, psi_x,y + psi_y,x)`` and
    ``D_b = D [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu)/2]]``; the gradients of
    the nodal rotations are taken at the element centre (natural origin), which
    is the standard patch-recovery point for a Q4 element.
    """
    D = solution.young * solution.thickness ** 3 / (12.0 * (1.0 - solution.poisson ** 2))
    nu = solution.poisson
    n_elem = solution.elements.shape[0]
    mx = torch.zeros(n_elem, dtype=solution.displacement.dtype)
    my = torch.zeros_like(mx)
    mxy = torch.zeros_like(mx)
    for e, conn in enumerate(solution.elements):
        xy = solution.nodes[conn]
        xi = eta = xy.new_tensor(0.0)
        _, dN_nat = _shape(xi, eta)
        dN = dN_nat @ torch.linalg.inv(dN_nat.T @ xy).T
        psi_x = solution.displacement[3 * conn + 1]
        psi_y = solution.displacement[3 * conn + 2]
        kx = dN[:, 0] @ psi_x
        ky = dN[:, 1] @ psi_y
        kxy = dN[:, 0] @ psi_y + dN[:, 1] @ psi_x
        mx[e] = D * (kx + nu * ky)
        my[e] = D * (ky + nu * kx)
        mxy[e] = D * (1.0 - nu) / 2.0 * kxy
    return {"Mx": mx, "My": my, "Mxy": mxy, "D": D}


# ---------------------------------------------------------------------------
# Exact FSDT (Mindlin-Timoshenko) Navier reference
# ---------------------------------------------------------------------------

def fsdt_navier_ss_deflection(
    *,
    q: float,
    a: float,
    b: float,
    young: float,
    poisson: float,
    thickness: float,
    k_normal: float = 0.0,
    g_shear: float = 0.0,
    k_tangential: float = 0.0,
    kappa: float = KAPPA_DEFAULT,
    terms: int = 81,
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> dict:
    """Exact FSDT double-sine solution for a simply supported plate on a
    two-parameter foundation *with* tangential contact stiffness.

    Returns a dict with the deflection at ``(x, y)`` (default: plate centre),
    the modal breakdown ``bending`` / ``shear`` and the Kirchhoff-only value.
    ``k_tangential = math.inf`` is the perfectly bonded limit.

    The modal amplitude (``lam = (m pi/a)^2 + (n pi/b)^2``) is

        W_mn = q_mn / [ k + G_p lam + S lam (R + D lam)/(R + D lam + S) ]

    with ``R = k_t h^2 / 4``; see the module docstring for the three limits.
    """
    if a <= 0.0 or b <= 0.0 or thickness <= 0.0:
        raise ValueError("plate geometry must be positive")
    D = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    shear_modulus = young / (2.0 * (1.0 + poisson))
    S = kappa * shear_modulus * thickness
    R = math.inf if math.isinf(k_tangential) else k_tangential * thickness ** 2 / 4.0
    xc = 0.5 * a if x is None else x
    yc = 0.5 * b if y is None else y

    total = 0.0
    bending = 0.0
    shear = 0.0
    kirchhoff = 0.0
    for m in range(1, 2 * terms, 2):
        sx = math.sin(m * math.pi * xc / a)
        am = m * math.pi / a
        for k in range(1, 2 * terms, 2):
            lam = am ** 2 + (k * math.pi / b) ** 2
            qmn = 16.0 * q / (math.pi ** 2 * m * k)
            shape = sx * math.sin(k * math.pi * yc / b)
            foundation = k_normal + g_shear * lam
            shear_coupling = S * lam
            if math.isinf(R):
                coupling = shear_coupling           # bonded limit: psi = 0
            else:
                coupling = shear_coupling * (R + D * lam) / (R + D * lam + S)
            denom = foundation + coupling
            total += qmn / denom * shape
            bending += qmn / (foundation + D * lam ** 2) * shape
            shear += qmn / (foundation + shear_coupling) * shape
            kirchhoff += qmn / (D * lam ** 2) * shape
    return {
        "deflection": total,
        "kirchhoff_deflection": kirchhoff,
        "bending_series": bending,
        "shear_limit_deflection": shear,
        "flexural_rigidity": D,
        "shear_rigidity": S,
        "tangential_rigidity": R,
        "terms": terms,
    }


def fsdt_navier_from_foundation(model: FoundationModel, **kwargs) -> dict:
    """Convenience: exact FSDT Navier deflection for a foundation model object."""
    if model.non_local:
        raise NotImplementedError("no local Navier symbol for a non-local foundation")
    params = dict(kwargs)
    params.setdefault("k_normal", float(getattr(model, "k_normal", 0.0)))
    params.setdefault("g_shear", float(getattr(model, "g_shear", 0.0)))
    return fsdt_navier_ss_deflection(**params)