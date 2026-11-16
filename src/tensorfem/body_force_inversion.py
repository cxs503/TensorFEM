"""Sparse-observation body-force field inversion for 2-D plane-stress FEM.

Why this module exists
----------------------
``tensorfem.inverse`` identifies a *handful* of scalar parameters (``E``, ``tau``,
``A``) from data.  The next question in any calibrated structural workflow is a
different one: the load is not a number but a *field*, and the measurements are
sparse -- strain gauges and displacement rosettes sit on a few percent of the
nodes, never on all of them.  Recovering a distributed volume load from such
data is a linear ill-posed problem, and the honest deliverables are (i) a
recovered field, (ii) how far it can be trusted, and (iii) which parts of it the
observation set cannot see at all.

Forward model (the part that must be right before any inversion is meaningful)
-----------------------------------------------------------------------------
2-D plane stress, bilinear Q4, on ``Omega = [0, Lx] x [0, Ly]``::

    -div(sigma(u)) = b(x, y)      in Omega
    u = 0                         on x = 0   (clamped edge)

with the volume load expanded in a smooth modal basis

    phi_{mn}(x, y) = sin(m pi x / Lx) sin(n pi y / Ly),   m, n = 1 .. P,

separately for the ``x`` and ``y`` components.  The unknown coefficient vector
``c`` (dimension ``2 P^2``) enters the nodal load vector *linearly* through
consistent (3x3 Gauss) loads ``L``, so the map from ``c`` to the sampled
displacements is the matrix

    A = S K^{-1} L,        A in R^{n_obs x 2P^2},

whose columns follow from stiffness back-substitutions.  The inverse problem
``min ||A c - d||^2 + alpha_eff ||c||^2`` is solved with CGLS, so **each
iteration costs exactly one forward and one adjoint solve**: the anchor's
iteration budget (<= 50 adjoint iterations) is a counted quantity, not a figure
of speech.

Why the modal basis, and what dropping it costs
-----------------------------------------------
The observation operator is a *sampling* operator: it destroys information.  For
a full nodal unknown (``2 n_node`` components, ``mode="nodal"``) the matrix ``A``
is rank deficient by construction -- there are load distributions that produce
exactly zero displacement at the sampled nodes.  Inverting in the full space with
Tikhonov damping returns the *minimum-norm* field, which is not the true field.
:func:`run_body_force_inversion` runs that variant as a negative control and
reports ``null_space_dim`` beside the recovery error instead of presenting the
nodal answer as a measurement.

Damping is dimensionless
------------------------
``alpha_eff = alpha_rel * sigma_max(A)^2``.  A raw weight on ``||c||^2`` is
dimensionful and unit dependent: with ``E = 210 GPa`` the entries of ``A`` are
``~1e-13``, so any absolute weight above ``~1e-30`` swamps the data term and
collapses the solution to ``c = 0``.  That failure was measured (see
``docs/CASE-FEM-BODY-FORCE-INVERSION.md``), and is why the option is
``alpha_rel``.

Diagnostics that are part of the result
---------------------------------------
``observability``
    Singular values of the observation matrix: which modes the sparse sample set
    can resolve at all, and with what conditioning.
``field_rel_l2``
    Relative L2 error of the recovered field against a supplied truth, measured
    on a 61x61 grid that is independent of the finite-element mesh.  This is
    verification instrumentation: with no truth supplied the key is absent, never
    guessed.
``out_of_span``
    Set when the truth carries content outside the inversion basis; a large error
    is then expected model error, not a solver defect.
``stop_reason``
    ``"tolerance"``, ``"budget"`` or ``"gradient_exhausted"``.  CGLS stagnates
    once the recursion residual is dominated by round-off; that is reported as
    stagnation, never as convergence.
``convergence``
    Per-iteration record (one entry = one forward + one adjoint solve) with misfit
    and, in verification runs, the field error -- enough to plot and re-read a
    convergence curve from the JSON the driver writes.

Schemas
-------
``tensorfem.body-force-inversion/1``       the returned result dict
``tensorfem.body-force-inversion-iter/1``  one ``convergence`` entry

References for the elements/oracles used by the tests
-----------------------------------------------------
* Q4 plane-stress stiffness, 2x2/3x3 Gauss: standard isoparametric formulation,
  e.g. Zienkiewicz & Taylor, *The Finite Element Method*, vol. 1, ch. 7.
* Consistent nodal loads ``int N_a b dV`` (not lumped): same reference, ch. 8.
* CGLS for ``min ||Ax-b||^2 + alpha||x||^2``: Bjorck, *Numerical Methods for
  Least Squares Problems*, SIAM 1996, sec. 7.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

__all__ = [
    "PlaneStressMesh",
    "PlaneStressModel",
    "ModalBasis",
    "NodalBasis",
    "ObservationSet",
    "InversionOptions",
    "run_body_force_inversion",
    "SCHEMA_RESULT",
    "SCHEMA_ITER",
]

SCHEMA_RESULT = "tensorfem.body-force-inversion/1"
SCHEMA_ITER = "tensorfem.body-force-inversion-iter/1"

_FLOAT = torch.float64

# 3x3 Gauss-Legendre on [-1, 1]: exact through degree 5, so stiffness products
# (bilinear gradients) and consistent loads (bilinear shape function times a
# smooth mode) carry no quadrature error at the accuracy the 5 % band needs.
_GAUSS_3 = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
_GAUSS_3_W = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)


def _shape(xi: float, eta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear Q4 shape functions and their reference-space gradients."""
    N = torch.tensor(
        [
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        ],
        dtype=_FLOAT,
    )
    dN = torch.tensor(
        [
            [-0.25 * (1 - eta), -0.25 * (1 - xi)],
            [0.25 * (1 - eta), -0.25 * (1 + xi)],
            [0.25 * (1 + eta), 0.25 * (1 + xi)],
            [-0.25 * (1 + eta), 0.25 * (1 - xi)],
        ],
        dtype=_FLOAT,
    )
    return N, dN


# Quadrature table computed once: 9 points, each with (xi, eta, w, N, dN).
_QUAD = tuple(
    (xi, eta, wi * wj, *_shape(xi, eta))
    for xi, wi in zip(_GAUSS_3, _GAUSS_3_W)
    for eta, wj in zip(_GAUSS_3, _GAUSS_3_W)
)


# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlaneStressMesh:
    """Structured Q4 mesh on ``[0, Lx] x [0, Ly]``.

    Node ``(ix, iy)`` has index ``iy * (nx + 1) + ix``; its DOFs are ``2*index``
    (x) and ``2*index + 1`` (y).  Elements are numbered with ``ix`` fastest and
    list their nodes counter-clockwise from the lower-left corner, matching the
    shape-function table above.
    """

    Lx: float
    Ly: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        if not (self.Lx > 0.0 and self.Ly > 0.0):
            raise ValueError("Lx and Ly must be positive")
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be >= 1")

    @property
    def n_node(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def n_elem(self) -> int:
        return self.nx * self.ny

    @property
    def n_dof(self) -> int:
        return 2 * self.n_node

    def node_index(self, ix: int, iy: int) -> int:
        return iy * (self.nx + 1) + ix

    def coords(self) -> torch.Tensor:
        """``(n_node, 2)`` node coordinates, ``x`` axial."""
        xs = torch.linspace(0.0, self.Lx, self.nx + 1, dtype=_FLOAT)
        ys = torch.linspace(0.0, self.Ly, self.ny + 1, dtype=_FLOAT)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    def elements(self) -> torch.Tensor:
        """``(n_elem, 4)`` connectivity, CCW from (xi, eta) = (-1, -1)."""
        nx, ny = self.nx, self.ny
        ix = torch.arange(nx, dtype=torch.long).repeat(ny)
        iy = torch.arange(ny, dtype=torch.long).repeat_interleave(nx)
        n0 = iy * (nx + 1) + ix
        n1 = n0 + 1
        n3 = n0 + (nx + 1)
        n2 = n3 + 1
        return torch.stack([n0, n1, n2, n3], dim=1)

    def element_dofs(self) -> torch.Tensor:
        """``(n_elem, 8)`` DOF map, ``[u0, v0, u1, v1, ...]``."""
        el = self.elements()
        dofs = torch.stack([2 * el, 2 * el + 1], dim=2)
        return dofs.reshape(el.shape[0], 8)

    def fixed_dofs(self, side: str = "left") -> torch.Tensor:
        """Clamped DOFs on the ``x = 0`` edge."""
        if side != "left":
            raise ValueError("only side='left' (x = 0) clamping is implemented")
        iy = torch.arange(self.ny + 1, dtype=torch.long)
        nodes = self.node_index(torch.zeros_like(iy), iy)
        return torch.stack([2 * nodes, 2 * nodes + 1], dim=1).reshape(-1)

    def node_field(self, values: torch.Tensor) -> torch.Tensor:
        """Reshape a nodal scalar field to ``(ny + 1, nx + 1)``."""
        return values.reshape(self.ny + 1, self.nx + 1)


# --------------------------------------------------------------------------- #
# Forward model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlaneStressModel:
    """Plane-stress Q4 forward model ``K u = f`` with a clamped ``x = 0`` edge."""

    mesh: PlaneStressMesh
    E: float = 210.0e9
    nu: float = 0.3
    thickness: float = 1.0

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise ValueError("E must be positive")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError("nu must lie in (-1, 0.5)")
        if self.thickness <= 0.0:
            raise ValueError("thickness must be positive")

    def D(self) -> torch.Tensor:
        """Plane-stress elasticity matrix."""
        e, nu = self.E, self.nu
        c = e / (1.0 - nu * nu)
        return torch.tensor(
            [[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1.0 - nu) / 2.0]],
            dtype=_FLOAT,
        )

    def _element_geometry(
        self, xy: torch.Tensor
    ) -> list[tuple[float, torch.Tensor, torch.Tensor]]:
        """Per quadrature point ``(weight * detJ, N, dN/dx)`` for one element."""
        out = []
        Jinv = None
        detJ = 0.0
        for _, _, w, N, dN in _QUAD:
            J = xy.T @ dN
            if Jinv is None:
                detJ = float(torch.linalg.det(J))
                if detJ <= 0.0:
                    raise ValueError("element has non-positive Jacobian")
                Jinv = torch.linalg.inv(J)
            dNdx = dN @ Jinv  # dN/dx_i = sum_j dN/dxi_j * (J^{-1})_{j i}
            out.append((w * detJ * self.thickness, N, dNdx))
        return out

    def stiffness(self) -> torch.Tensor:
        """Dense symmetric positive-definite ``(n_dof, n_dof)`` stiffness matrix."""
        mesh = self.mesh
        coords = mesh.coords()
        el = mesh.elements()
        edof = mesh.element_dofs()
        D = self.D()
        K = torch.zeros(mesh.n_dof, mesh.n_dof, dtype=_FLOAT)
        for e in range(mesh.n_elem):
            xy = coords[el[e]]
            Ke = torch.zeros(8, 8, dtype=_FLOAT)
            for wdet, _, dNdx in self._element_geometry(xy):
                B = torch.zeros(3, 8, dtype=_FLOAT)
                B[0, 0::2] = dNdx[:, 0]
                B[1, 1::2] = dNdx[:, 1]
                B[2, 0::2] = dNdx[:, 1]
                B[2, 1::2] = dNdx[:, 0]
                Ke += wdet * (B.T @ D @ B)
            idx = edof[e]
            K[idx.unsqueeze(1), idx.unsqueeze(0)] += Ke
        return K

    def consistent_loads(
        self,
        phi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        component: int,
    ) -> torch.Tensor:
        """Nodal loads of the volume load ``phi(x, y) e_component`` (single mode)."""
        return self.load_matrix([(phi, component)])

    def load_matrix(
        self,
        modes: Sequence[tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], int]],
    ) -> torch.Tensor:
        """``(n_dof, n_mode)`` matrix of consistent nodal loads, one pass over elements.

        ``f_a^{(k)} = int N_a phi_k dV`` on the same 3x3 rule as the stiffness, so
        a mode and its assembled load are consistent to quadrature accuracy.
        """
        mesh = self.mesh
        coords = mesh.coords()
        el = mesh.elements()
        edof = mesh.element_dofs()
        n_mode = len(modes)
        F = torch.zeros(mesh.n_dof, n_mode, dtype=_FLOAT)
        for e in range(mesh.n_elem):
            xy = coords[el[e]]
            Fe = torch.zeros(8, n_mode, dtype=_FLOAT)
            for wdet, N, _ in self._element_geometry(xy):
                x = torch.tensor(float(N @ xy[:, 0]), dtype=_FLOAT)
                y = torch.tensor(float(N @ xy[:, 1]), dtype=_FLOAT)
                for k, (phi, comp) in enumerate(modes):
                    if comp not in (0, 1):
                        raise ValueError("component must be 0 (x) or 1 (y)")
                    Fe[comp::2, k] += wdet * N * float(phi(x, y))
            F[edof[e]] += Fe
        return F


# --------------------------------------------------------------------------- #
# Unknown-field parameterisations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModalBasis:
    """Smooth body-force basis ``sin(m pi x/Lx) sin(n pi y/Ly)`` in both components.

    ``order`` is ``P``: the basis holds ``P*P`` modes per displacement component,
    so the unknown vector has ``2 P^2`` entries (x-modes first, then y-modes).
    Every mode vanishes on the boundary, the physically sensible class for a
    volume load, and keeps the load orthogonal to the clamped edge.
    """

    mesh: PlaneStressMesh
    order: int = 2

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be >= 1")

    @property
    def n_mode(self) -> int:
        return self.order * self.order

    @property
    def n_unknown(self) -> int:
        return 2 * self.n_mode

    def mode(self, k: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if not 0 <= k < self.n_mode:
            raise ValueError(f"mode index out of range: {k}")
        m = k // self.order + 1
        n = k % self.order + 1
        Lx, Ly = self.mesh.Lx, self.mesh.Ly

        def phi(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.sin(m * math.pi * x / Lx) * torch.sin(n * math.pi * y / Ly)

        return phi

    def component(self, k: int) -> int:
        return 0 if k < self.n_mode else 1

    def mode_of(self, k: int) -> int:
        return k % self.n_mode

    def evaluate(
        self, c: torch.Tensor, component: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Field value of one component at arbitrary points."""
        out = torch.zeros_like(x)
        for k in range(self.n_mode):
            out = out + c[k + component * self.n_mode] * self.mode(k)(x, y)
        return out

    def nodal_field(self, c: torch.Tensor, component: int) -> torch.Tensor:
        coords = self.mesh.coords()
        return self.evaluate(c, component, coords[:, 0], coords[:, 1])

    def modes(self) -> list[tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], int]]:
        return [(self.mode(self.mode_of(k)), self.component(k)) for k in range(self.n_unknown)]

    def load_matrix(self, model: PlaneStressModel) -> torch.Tensor:
        """``L`` with ``L c`` the consistent nodal loads of the expanded field."""
        return model.load_matrix(self.modes())


class NodalBasis:
    """Full nodal unknown (``2 n_node``), no smoothing: the underdetermined control.

    Present so that the ill-posedness of "invert every nodal load" is *measured*
    in the same harness rather than asserted in prose.
    """

    def __init__(self, mesh: PlaneStressMesh) -> None:
        self.mesh = mesh

    @property
    def n_unknown(self) -> int:
        return self.mesh.n_dof

    def nodal_field(self, c: torch.Tensor, component: int) -> torch.Tensor:
        return c[component::2]


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObservationSet:
    """The node indices where both displacement components are measured."""

    node_ids: torch.Tensor
    mesh: PlaneStressMesh

    def __post_init__(self) -> None:
        if self.node_ids.numel() == 0:
            raise ValueError("observation set is empty")
        if int(self.node_ids.max()) >= self.mesh.n_node or int(self.node_ids.min()) < 0:
            raise ValueError("observation node id out of range")

    @property
    def fraction(self) -> float:
        return self.node_ids.numel() / self.mesh.n_node

    def dofs(self) -> torch.Tensor:
        return torch.stack([2 * self.node_ids, 2 * self.node_ids + 1], dim=1).reshape(-1)

    @staticmethod
    def sample(
        mesh: PlaneStressMesh, fraction: float, *, seed: int = 0
    ) -> "ObservationSet":
        """Uniformly random node subset of ``floor(fraction * n_node)`` nodes.

        ``floor``, not ``ceil``: the acceptance criterion is a hard cap on the
        observed fraction, and ``ceil(0.10 * 91) = 10`` nodes is 11.0 %, which
        would silently violate it.
        """
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must lie in (0, 1]")
        n_take = max(1, int(math.floor(fraction * mesh.n_node)))
        gen = torch.Generator().manual_seed(seed)
        ids = torch.randperm(mesh.n_node, generator=gen)[:n_take].sort().values
        return ObservationSet(node_ids=ids, mesh=mesh)


# --------------------------------------------------------------------------- #
# Inversion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InversionOptions:
    """CGLS budget and damping; one iteration = one forward + one adjoint solve."""

    max_adjoint: int = 50
    tol: float = 1.0e-10
    # Damping relative to sigma_max(A)^2, see the module docstring.
    alpha_rel: float = 1.0e-10
    noise: float = 0.0  # relative observation noise (fraction of RMS(d))
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_adjoint < 1:
            raise ValueError("max_adjoint must be >= 1")
        if self.tol <= 0.0:
            raise ValueError("tol must be positive")
        if self.alpha_rel < 0.0:
            raise ValueError("alpha_rel must be non-negative")
        if self.noise < 0.0:
            raise ValueError("noise must be non-negative")


def _cgls(
    forward: Callable[[torch.Tensor], torch.Tensor],
    adjoint: Callable[[torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    n_unknown: int,
    opts: InversionOptions,
    alpha_eff: float,
    on_iter: Callable[[int, torch.Tensor, float], None],
) -> tuple[torch.Tensor, list[dict[str, Any]], str]:
    """Damped CGLS; each iteration performs exactly one forward and one adjoint solve."""
    sqrt_alpha = math.sqrt(alpha_eff)
    x = torch.zeros(n_unknown, dtype=_FLOAT)
    r_data = data.clone()
    r_reg = torch.zeros(n_unknown, dtype=_FLOAT)
    s = adjoint(r_data)
    p = s.clone()
    gamma = float(s @ s)
    gamma0 = gamma
    norm_d = float(data.norm())
    if norm_d == 0.0:
        raise ValueError("observation vector is identically zero; nothing to invert")
    history: list[dict[str, Any]] = []
    stop_reason = "budget"
    for it in range(1, opts.max_adjoint + 1):
        # Round-off floor of the recursion: once the gradient is numerically gone,
        # further iterations would only report noise as progress.
        if gamma <= 1.0e-30 * gamma0 or gamma <= 0.0:
            stop_reason = "gradient_exhausted"
            break
        Ap = forward(p)
        denom = float(Ap @ Ap) + alpha_eff * float(p @ p)
        if denom <= 0.0:
            stop_reason = "gradient_exhausted"
            break
        step = gamma / denom
        x = x + step * p
        r_data = r_data - step * Ap
        r_reg = r_reg - step * sqrt_alpha * p
        resid = math.sqrt(float(r_data @ r_data)) / norm_d
        history.append(
            {
                "schema": SCHEMA_ITER,
                "iteration": it,
                "relative_data_residual": resid,
                "step": step,
                "forward_solve": it,
                "adjoint_solve": it,
            }
        )
        on_iter(it, x, resid)
        if resid <= opts.tol:
            stop_reason = "tolerance"
            break
        s = adjoint(r_data) + sqrt_alpha * r_reg
        gamma_new = float(s @ s)
        beta = gamma_new / gamma
        p = s + beta * p
        gamma = gamma_new
    return x, history, stop_reason


def _field_error(
    basis: ModalBasis | NodalBasis,
    c_rec: torch.Tensor,
    c_true: torch.Tensor,
    truth_extra: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    mesh: PlaneStressMesh,
) -> float:
    """Relative L2 field error on a 61x61 grid independent of the FE mesh."""
    if isinstance(basis, NodalBasis):
        num = 0.0
        den = 0.0
        for comp in (0, 1):
            diff = basis.nodal_field(c_rec, comp) - basis.nodal_field(c_true, comp)
            num += float((diff**2).sum())
            den += float((basis.nodal_field(c_true, comp) ** 2).sum())
        return math.sqrt(num / den) if den > 0.0 else float("inf")

    n_q = 61
    xs = torch.linspace(0.0, mesh.Lx, n_q, dtype=_FLOAT)
    ys = torch.linspace(0.0, mesh.Ly, n_q, dtype=_FLOAT)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xf, yf = xx.reshape(-1), yy.reshape(-1)
    num = 0.0
    den = 0.0
    for comp in (0, 1):
        rec = basis.evaluate(c_rec, comp, xf, yf)
        tru = basis.evaluate(c_true, comp, xf, yf)
        if truth_extra is not None:
            tru = tru + truth_extra(xf, yf)
        num += float(((rec - tru) ** 2).mean())
        den += float((tru**2).mean())
    if den <= 0.0:
        return float("inf")
    return math.sqrt(num / den)


def run_body_force_inversion(
    *,
    truth_coeffs: Sequence[float],
    mesh: PlaneStressMesh | None = None,
    truth_extra: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    order: int = 2,
    fraction: float = 0.10,
    options: InversionOptions | None = None,
    mode: str = "modal",
    E: float = 210.0e9,
    nu: float = 0.3,
) -> dict[str, Any]:
    """Invert a volume-load field from ``fraction`` of the nodal displacement samples.

    Parameters
    ----------
    truth_coeffs
        Coefficients of the synthetic truth in the inversion basis (``2 P^2``
        entries for ``mode="modal"``, ``2 n_node`` for ``mode="nodal"``).
    truth_extra
        Optional callable added to the truth field *outside* the inversion basis.
        Its presence sets ``out_of_span`` and quantifies pure model error.
    mode
        ``"modal"`` (smooth basis, the acceptance case) or ``"nodal"`` (full-space
        negative control exposing the rank deficiency).

    Returns
    -------
    dict
        JSON-serialisable result; see the module docstring for the keys.
    """
    mesh = mesh if mesh is not None else PlaneStressMesh(Lx=1.0, Ly=0.25, nx=12, ny=6)
    opts = options if options is not None else InversionOptions()
    model = PlaneStressModel(mesh=mesh, E=E, nu=nu)

    if mode == "modal":
        basis: ModalBasis | NodalBasis = ModalBasis(mesh=mesh, order=order)
        n_unknown = basis.n_unknown
        if len(truth_coeffs) != n_unknown:
            raise ValueError(
                f"truth_coeffs must have {n_unknown} entries for order={order}, "
                f"got {len(truth_coeffs)}"
            )
        L = basis.load_matrix(model)
        if opts.alpha_rel <= 0.0:
            raise ValueError(
                "modal inversion requires alpha_rel > 0: the observation operator is a "
                "sampling operator, so the undamped normal equations are rank deficient"
            )
    elif mode == "nodal":
        basis = NodalBasis(mesh)
        n_unknown = basis.n_unknown
        if len(truth_coeffs) != n_unknown:
            raise ValueError(f"truth_coeffs must have {n_unknown} entries in nodal mode")
        if opts.alpha_rel <= 0.0:
            raise ValueError("nodal inversion requires alpha_rel > 0 (Tikhonov damping)")
        L = torch.eye(mesh.n_dof, dtype=_FLOAT)
    else:
        raise ValueError("mode must be 'modal' or 'nodal'")

    # -- forward operator: one LU factorisation, reused for every solve ----- #
    K = model.stiffness()
    free = torch.ones(mesh.n_dof, dtype=torch.bool)
    free[mesh.fixed_dofs()] = False
    lu, piv = torch.linalg.lu_factor(K[free][:, free])

    def solve_free(rhs_free: torch.Tensor) -> torch.Tensor:
        sol = torch.linalg.lu_solve(lu, piv, rhs_free.unsqueeze(-1)).squeeze(-1)
        out = torch.zeros(mesh.n_dof, dtype=_FLOAT)
        out[free] = sol
        return out

    obs = ObservationSet.sample(mesh, fraction, seed=opts.seed)
    obs_dofs = obs.dofs()

    def forward(c: torch.Tensor) -> torch.Tensor:
        return solve_free((L @ c)[free])[obs_dofs]

    def adjoint(r: torch.Tensor) -> torch.Tensor:
        g = torch.zeros(mesh.n_dof, dtype=_FLOAT)
        g[obs_dofs] = r
        return L.T @ solve_free(g[free])  # K symmetric: same factorisation

    # -- synthetic data ----------------------------------------------------- #
    c_true = torch.tensor(list(truth_coeffs), dtype=_FLOAT)
    d_clean = forward(c_true)
    if opts.noise > 0.0:
        gen = torch.Generator().manual_seed(opts.seed + 977)
        scale = opts.noise * float(d_clean.pow(2).mean().sqrt())
        d = d_clean + scale * torch.randn(d_clean.shape, generator=gen, dtype=_FLOAT)
    else:
        d = d_clean.clone()

    # -- observability of the observation matrix ---------------------------- #
    eye = torch.eye(n_unknown, dtype=_FLOAT)
    A_obs = torch.stack([forward(eye[:, k]) for k in range(n_unknown)], dim=1)
    sv = torch.linalg.svdvals(A_obs)
    sv_max = float(sv[0])
    sv_min = float(sv[-1])
    alpha_eff = opts.alpha_rel * sv_max * sv_max

    # -- CGLS with per-iteration bookkeeping -------------------------------- #
    trace: list[dict[str, Any]] = []

    def on_iter(it: int, x: torch.Tensor, resid: float) -> None:
        trace.append(
            {
                "schema": SCHEMA_ITER,
                "iteration": it,
                "relative_data_residual": resid,
                "field_rel_l2": _field_error(basis, x, c_true, truth_extra, mesh),
            }
        )

    c_rec, history, stop_reason = _cgls(
        forward, adjoint, d, n_unknown, opts, alpha_eff, on_iter
    )

    field_err = _field_error(basis, c_rec, c_true, truth_extra, mesh)
    null_dim = None
    if mode == "nodal":
        null_dim = int(n_unknown - torch.linalg.matrix_rank(A_obs))

    return {
        "schema": SCHEMA_RESULT,
        "mode": mode,
        "mesh": {
            "Lx": mesh.Lx,
            "Ly": mesh.Ly,
            "nx": mesh.nx,
            "ny": mesh.ny,
            "n_node": mesh.n_node,
            "n_dof": mesh.n_dof,
        },
        "basis": {
            "kind": mode,
            "order": order if mode == "modal" else None,
            "n_mode_per_component": basis.n_mode if mode == "modal" else None,
            "n_unknown": n_unknown,
        },
        "observations": {
            "fraction": obs.fraction,
            "fraction_limit": 0.10,
            "n_node_sampled": int(obs.node_ids.numel()),
            "n_observation": int(obs_dofs.numel()),
            "noise_relative": opts.noise,
            "seed": opts.seed,
        },
        "options": {
            "max_adjoint": opts.max_adjoint,
            "tol": opts.tol,
            "alpha_rel": opts.alpha_rel,
            "alpha_effective": alpha_eff,
        },
        "iterations": len(history),
        "adjoint_iteration_budget": opts.max_adjoint,
        "adjoint_solves": len(history),
        "forward_solves": len(history) + n_unknown + 1,
        "stop_reason": stop_reason,
        "relative_data_residual": float((forward(c_rec) - d).norm() / d.norm()),
        "field_rel_l2": field_err,
        "acceptance": {
            "fraction_ok": obs.fraction <= 0.10 + 1.0e-12,
            "field_within_5pct": field_err <= 0.05,
            "iterations_within_budget": len(history) <= 50,
        },
        "observability": {
            "sigma_max": sv_max,
            "sigma_min": sv_min,
            "condition_number": (sv_max / sv_min) if sv_min > 0.0 else None,
            "n_singular_values": int(sv.numel()),
            "singular_values": [float(s_) for s_ in sv],
            "scaled_singular_values": [float(s_ / sv_max) for s_ in sv],
        },
        "null_space_dim": null_dim,
        "out_of_span": truth_extra is not None,
        "coefficients": {
            "recovered": [float(v) for v in c_rec],
            "truth": [float(v) for v in c_true],
        },
        "convergence": trace,
        "iteration_log": history,
    }