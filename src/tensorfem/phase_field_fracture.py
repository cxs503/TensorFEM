"""Phase-field fracture (AT2) for two-dimensional plane-strain continua.

The module adds the damage-evolution physics that the production linear-elastic and
cohesive-zone paths do not carry: a scalar phase field ``d in [0, 1]`` (``d = 0``
intact, ``d = 1`` broken) governed by the regularised variational functional

    E(u, d) = int g(d) psi+ (eps(u)) dV + int psi- (eps(u)) dV
              + int Gc [ d^2 / (2 l) + (l/2) |grad d|^2 ] dV ,

with ``g(d) = (1 - d)^2 + k`` and the Amor volumetric-deviatoric split

    psi+ = 1/2 K <tr eps>^2 + mu eps_dev : eps_dev ,   psi- = 1/2 K <-tr eps>^2 .

Irreversibility is enforced by the pointwise history variable
``H(x, t) = max_{s <= t} psi+ (eps(x, s))`` which replaces ``psi+`` in the damage
residual, so ``d`` can only grow.

Solver: a monolithic Newton iteration on the coupled residual ``(R_u, R_d)`` with
the exact (non-symmetric) consistent tangent, where the damage driving force uses
the *current* tensile energy subject to the history floor.  A staggered
(alternate-minimisation) driver is available as well.

Conventions, identical to :mod:`tensorfem.continuum` and
:mod:`tensorfem.quadratic_plane`:

* Q4 nodes ``(-1,-1) (1,-1) (1,1) (-1,1)``, DOF order ``(u, v)`` per node;
* global unknown vector ``x = [u_0, v_0, u_1, v_1, ..., d_0, d_1, ...]``;
* engineering strain vector ``eps = (e_xx, e_yy, g_xy)``.

Nothing in the existing plane / shell / contact paths is touched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

_SQ3 = 1.0 / math.sqrt(3.0)
#: 2x2 Gauss rule: exact for the bilinear Q4 gradients and for ``N_a N_b``.
GAUSS_POINTS = ((-_SQ3, -_SQ3), (_SQ3, -_SQ3), (_SQ3, _SQ3), (-_SQ3, _SQ3))
GAUSS_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
#: Q4 parent-node coordinates (same order as ``continuum``).
_NODE_XI = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def structured_quad_mesh(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int):
    """Structured Q4 mesh, node index ``i + (nx + 1) * j``."""
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    quads = []
    for j in range(ny):
        for i in range(nx):
            quads.append(
                (idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i])
            )
    return grid, np.array(quads, dtype=np.int64)


def nodes_on_segment(nodes: np.ndarray, p0, p1, tol: float) -> np.ndarray:
    """Indices of nodes lying within ``tol`` of the segment ``p0 -> p1``."""
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    ab = b - a
    length = float(np.linalg.norm(ab))
    t = ((nodes - a) @ ab) / max(length * length, 1e-30)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    dist = np.linalg.norm(nodes - proj, axis=1)
    inside = np.abs(t - np.clip(t, 0.0, 1.0)) <= 1.0
    return np.where((dist <= tol) & inside)[0]


def _shape_and_derivatives():
    n = np.zeros((4, 4))
    dxi = np.zeros((4, 2, 4))
    for g, (xi, eta) in enumerate(GAUSS_POINTS):
        for k, (xk, yk) in enumerate(_NODE_XI):
            n[g, k] = 0.25 * (1.0 + xi * xk) * (1.0 + eta * yk)
            # dN/dxi_a
            dxi[k, 0, g] = 0.25 * xk * (1.0 + eta * yk)
            dxi[k, 1, g] = 0.25 * yk * (1.0 + xi * xk)
    return torch.tensor(n, dtype=torch.float64), torch.tensor(dxi, dtype=torch.float64)


_N_T, _DN_T = _shape_and_derivatives()          # N [g, 4], dN/dxi [4, 2, g]


# --------------------------------------------------------------------------------------
# problem definition
# --------------------------------------------------------------------------------------
@dataclass
class PhaseFieldProblem:
    """Geometry + AT2 material parameters for a 2-D plane-strain body."""

    nodes: np.ndarray
    elements: np.ndarray
    E: float
    nu: float
    Gc: float
    l: float
    thickness: float = 1.0
    k_res: float = 1.0e-6
    void_elements: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.elements = np.asarray(self.elements, dtype=np.int64)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must be [n_nodes, 2]")
        if self.elements.ndim != 2 or self.elements.shape[1] != 4:
            raise ValueError("elements must be [n_elem, 4] (Q4)")
        if self.l <= 0.0 or self.Gc <= 0.0:
            raise ValueError("Gc and l must be positive")
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.K_bulk = lam + 2.0 * self.mu / 3.0
        # plane-strain effective modulus used to convert G <-> K_I
        self.E_prime = self.E / (1.0 - self.nu ** 2)
        self.n_nodes = self.nodes.shape[0]
        self.n_elem = self.elements.shape[0]
        self.n_dof = 3 * self.n_nodes
        self._precompute()

    # -- geometry precomputation ---------------------------------------------------
    def _precompute(self) -> None:
        nodes_t = torch.tensor(self.nodes, dtype=torch.float64)
        elem_t = torch.tensor(self.elements, dtype=torch.long)
        coords = nodes_t[elem_t]                                    # [m, 4, 2]
        jac = torch.einsum("kag,mkb->mgab", _DN_T, coords)           # [m, g, 2, 2]
        det = jac[:, :, 0, 0] * jac[:, :, 1, 1] - jac[:, :, 0, 1] * jac[:, :, 1, 0]
        if bool((det <= 0.0).any()):
            raise ValueError("inverted or degenerate element in mesh")
        inv = torch.linalg.inv(jac)                                  # [m, g, 2, 2]
        # dN/dX[m, g, k, b] = sum_a dN_k/dxi_a * invJ[a, b]
        dndx = torch.einsum("kag,mgab->mgkb", _DN_T, inv)            # [m, 4, 4, 2]
        b = torch.zeros((self.n_elem, 4, 3, 8), dtype=torch.float64)
        b[:, :, 0, 0::2] = dndx[:, :, :, 0]
        b[:, :, 1, 1::2] = dndx[:, :, :, 1]
        b[:, :, 2, 0::2] = dndx[:, :, :, 1]
        b[:, :, 2, 1::2] = dndx[:, :, :, 0]
        self._b = b
        self._dndx = dndx
        self._n = _N_T.unsqueeze(0).expand(self.n_elem, 4, 4)        # [m, g, k]
        self._wdet = torch.tensor(GAUSS_WEIGHTS, dtype=torch.float64).view(1, 4) * det
        self._wdet = self._wdet * self.thickness
        self._coords = coords
        if not hasattr(self, "orphan_nodes"):
            self.orphan_nodes = np.zeros(0, dtype=np.int64)
        if self.void_elements is None:
            keep = np.ones((self.n_elem, 4))
        else:
            vid = np.zeros(self.n_elem, dtype=bool)
            vid[np.asarray(self.void_elements, dtype=np.int64)] = True
            keep = np.repeat((~vid)[:, None], 4, axis=1).astype(np.float64)
            incident = np.bincount(self.elements[~vid].ravel(), minlength=self.n_nodes)
            self.orphan_nodes = np.where(incident == 0)[0]
        self._keep = torch.tensor(keep, dtype=torch.float64)
        self.element_u_dofs = (2 * elem_t[:, :, None] + torch.tensor([0, 1])).reshape(self.n_elem, 8)
        self.element_d_dofs = self.n_nodes * 2 + elem_t

    # -- kinematics ----------------------------------------------------------------
    def strain(self, u: np.ndarray) -> torch.Tensor:
        """Strain at every Gauss point, ``[n_elem, 4, 3]``."""
        u_nodes = np.asarray(u, dtype=np.float64).reshape(self.n_nodes, 2)
        ue = torch.tensor(u_nodes[self.elements].reshape(self.n_elem, 8), dtype=torch.float64)
        return torch.einsum("mgij,mj->mgi", self._b, ue)

    def damage_at_gauss(self, d: np.ndarray) -> torch.Tensor:
        de = torch.tensor(d[self.elements], dtype=torch.float64)
        return torch.einsum("mgk,mk->mg", self._n, de)

    def damage_gradient(self, d: np.ndarray) -> torch.Tensor:
        de = torch.tensor(d[self.elements], dtype=torch.float64)
        return torch.einsum("mgkb,mk->mgb", self._dndx, de)

    # -- constitutive split --------------------------------------------------------
    def split(self, e: torch.Tensor):
        """Return ``(psi_p, psi_m, sigma_p, sigma_m)`` at all Gauss points."""
        exx, eyy, gxy = e[..., 0], e[..., 1], e[..., 2]
        tr = exx + eyy
        tr_p = torch.clamp(tr, min=0.0)
        tr_m = torch.clamp(-tr, min=0.0)
        mu, K = self.mu, self.K_bulk
        dev = mu * ((2.0 / 3.0) * (exx ** 2 + eyy ** 2) - (2.0 / 3.0) * exx * eyy + 0.5 * gxy ** 2)
        psi_p = 0.5 * K * tr_p ** 2 + dev
        psi_m = 0.5 * K * tr_m ** 2
        sig_p = torch.stack(
            (K * tr_p + mu * ((4.0 / 3.0) * exx - (2.0 / 3.0) * eyy),
             K * tr_p + mu * ((4.0 / 3.0) * eyy - (2.0 / 3.0) * exx),
             mu * gxy),
            dim=-1,
        )
        tr_neg = torch.clamp(tr, max=0.0)
        sig_m = torch.stack((K * tr_neg, K * tr_neg, torch.zeros_like(tr_neg)), dim=-1)
        return psi_p, psi_m, sig_p, sig_m

    def degradation(self, d: torch.Tensor) -> torch.Tensor:
        return (1.0 - d) ** 2 + self.k_res


# --------------------------------------------------------------------------------------
# assembled operators
# --------------------------------------------------------------------------------------
def _tocoo(rows, cols, data, n):
    rows = np.asarray(rows).ravel()
    cols = np.asarray(cols).ravel()
    data = np.asarray(data).ravel()
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def assemble(problem: PhaseFieldProblem, u: np.ndarray, d: np.ndarray, history: np.ndarray,
             load: np.ndarray | None = None):
    """Coupled residual vector and consistent tangent at ``(u, d)``.

    ``history`` is the irreversibility field ``H`` at the Gauss points
    (``[n_elem, 4]``); the damage driving force uses ``max(H, psi+(eps(u)))``.
    """
    ne = problem.n_elem
    n = problem.n_dof
    e = problem.strain(u)
    psi_p, psi_m, sig_p, sig_m = problem.split(e)
    dgp = problem.damage_at_gauss(d)
    grad_d = problem.damage_gradient(d)
    keep = problem._keep
    g = problem.degradation(dgp) * keep
    gp = -2.0 * (1.0 - dgp) * keep                            # dg/dd
    sig = g.unsqueeze(-1) * sig_p + keep.unsqueeze(-1) * sig_m

    hist = torch.tensor(history, dtype=torch.float64)
    drive = torch.where(psi_p > hist, psi_p, hist)
    active = (psi_p > hist)

    wdet = problem._wdet
    b = problem._b
    nmat = problem._n

    # --- element residuals -------------------------------------------------------
    bt_sig = torch.einsum("mgij,mgi->mgj", b, sig)
    ru = torch.einsum("mg,mgj->mj", wdet, bt_sig)                              # [m, 8]
    bt_sig_p = torch.einsum("mgij,mgi->mgj", b, sig_p)
    rd = torch.einsum("mg,mgk->mk", wdet * keep,
                      (problem.Gc / problem.l) * dgp.unsqueeze(-1) * nmat
                      + problem.Gc * problem.l * torch.einsum(
                          "mgkb,mgb->mgk", problem._dndx, grad_d)
                      + gp.unsqueeze(-1) * drive.unsqueeze(-1) * nmat)         # [m, 4]

    # --- consistent tangent ------------------------------------------------------
    # Kuu = int B^T (g C+ + C-) B : C+ = H(tr) Cvol + Cdev, C- = H(-tr) Cvol
    exx, eyy, gxy = e[..., 0], e[..., 1], e[..., 2]
    tr = exx + eyy
    hp = (tr >= 0).to(torch.float64)
    hm = (tr < 0).to(torch.float64)
    mu, K = problem.mu, problem.K_bulk
    cplus = torch.zeros((ne, 4, 3, 3), dtype=torch.float64)
    cminus = torch.zeros((ne, 4, 3, 3), dtype=torch.float64)
    vol = K
    cplus[:, :, 0, 0] = vol * hp
    cplus[:, :, 0, 1] = vol * hp
    cplus[:, :, 1, 0] = vol * hp
    cplus[:, :, 1, 1] = vol * hp
    cminus[:, :, 0, 0] = vol * hm
    cminus[:, :, 0, 1] = vol * hm
    cminus[:, :, 1, 0] = vol * hm
    cminus[:, :, 1, 1] = vol * hm
    # deviatoric block (engineering shear)
    cplus[:, :, 0, 0] += 4.0 * mu / 3.0
    cplus[:, :, 0, 1] += -2.0 * mu / 3.0
    cplus[:, :, 1, 0] += -2.0 * mu / 3.0
    cplus[:, :, 1, 1] += 4.0 * mu / 3.0
    cplus[:, :, 2, 2] += mu
    ctot = keep.unsqueeze(-1).unsqueeze(-1) * (
        g.unsqueeze(-1).unsqueeze(-1) * cplus + cminus)
    # (explicit form below is clearer and equally fast)
    kuu = torch.einsum("mg,mgia,mgik,mgkb->mab", wdet, b, ctot, b)             # [m, 8, 8]
    # Kud = int B^T gp sig_p N
    kud = torch.einsum("mg,mgj,mgk->mjk", wdet * gp, bt_sig_p, nmat)           # [m, 8, 4]
    # Kdu = int N gp sig_p^T B  (active set only)
    kdu = torch.einsum("mg,mgk,mgj->mkj", wdet * gp, nmat, bt_sig_p)           # [m, 4, 8]
    kdu = kdu * active.to(torch.float64).unsqueeze(-1)
    # Kdd = int (Gc/l + 2 H) N N + Gc l dN dN
    coef = ((problem.Gc / problem.l) + 2.0 * drive) * keep            # [m, g]
    kdd = torch.einsum("mg,mgk,mgl->mkl", wdet * coef, nmat, nmat)
    kdd = kdd + problem.Gc * problem.l * torch.einsum(
        "mg,mgkb,mglb->mkl", wdet, problem._dndx, problem._dndx)

    # --- scatter -----------------------------------------------------------------
    ud = problem.element_u_dofs.numpy()
    ddof = problem.element_d_dofs.numpy()
    rows_u = np.repeat(ud, 8, axis=1)
    cols_u = np.tile(ud, (1, 8))
    kuu_sp = _tocoo(rows_u, cols_u, kuu.numpy(), n)
    rows_d = np.repeat(ddof, 4, axis=1)
    cols_d = np.tile(ddof, (1, 4))
    kdd_sp = _tocoo(rows_d, cols_d, kdd.numpy(), n)
    rows_ud = np.repeat(ud, 4, axis=1)
    cols_ud = np.tile(ddof, (1, 8))
    kud_sp = _tocoo(rows_ud, cols_ud, kud.numpy(), n)
    kdu_sp = _tocoo(np.repeat(ddof, 8, axis=1), np.tile(ud, (1, 4)), kdu.numpy(), n)
    tangent = (kuu_sp + kdd_sp + kud_sp + kdu_sp).tocsr()

    R = np.zeros(n)
    np.add.at(R, ud.ravel(), ru.numpy().ravel())
    if load is not None:
        R[: 2 * problem.n_nodes] -= np.asarray(load, dtype=np.float64).ravel()
    np.add.at(R, ddof.ravel(), rd.numpy().ravel())

    diag = {
        "psi_p": psi_p.numpy(),
        "psi_m": psi_m.numpy(),
        "d_gp": dgp.numpy(),
        "grad_d": grad_d.numpy(),
        "history": drive.numpy(),
    }
    return R, tangent, diag


def energies(problem: PhaseFieldProblem, u, d, history):
    """Elastic (split) + fracture energy of a state, plus the current H field."""
    e = problem.strain(u)
    psi_p, psi_m, _, _ = problem.split(e)
    dgp = problem.damage_at_gauss(d)
    grad_d = problem.damage_gradient(d)
    keep = problem._keep
    g = problem.degradation(dgp) * keep
    w = problem._wdet
    elastic = float((w * (g * psi_p + keep * psi_m)).sum())
    fracture = float((w * keep * problem.Gc * (dgp ** 2 / (2.0 * problem.l)
                                        + 0.5 * problem.l * (grad_d ** 2).sum(-1))).sum())
    return {
        "elastic": elastic,
        "fracture": fracture,
        "total": elastic + fracture,
        "psi_p": psi_p.numpy(),
        "psi_m": psi_m.numpy(),
        "d_gp": dgp.numpy(),
    }


# --------------------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------------------
@dataclass
class NewtonRecord:
    residual_norms: list
    iterations: int
    converged: bool


class PhaseFieldSolver:
    """Monolithic Newton / staggered driver for the AT2 model."""

    def __init__(self, problem: PhaseFieldProblem, fixed: dict | None = None,
                 initial_damage: np.ndarray | None = None, load: np.ndarray | None = None):
        self.problem = problem
        self.load = (np.zeros(2 * problem.n_nodes) if load is None
                     else np.asarray(load, dtype=np.float64).ravel().copy())
        self.u = np.zeros(2 * problem.n_nodes)
        self.d = np.zeros(problem.n_nodes) if initial_damage is None else np.asarray(
            initial_damage, dtype=np.float64).copy()
        self.history = np.zeros((problem.n_elem, 4))
        self.fixed = np.zeros(problem.n_dof, dtype=bool)
        self.fixed_values = np.zeros(problem.n_dof)
        for dof, val in (fixed or {}).items():
            self.fixed[int(dof)] = True
            self.fixed_values[int(dof)] = float(val)
        self._apply_fixed()

    # -- helpers ---------------------------------------------------------------
    def _x(self):
        return np.concatenate([self.u, self.d])

    def _apply_fixed(self):
        x = self._x()
        x[self.fixed] = self.fixed_values[self.fixed]
        self.u = x[:2 * self.problem.n_nodes].copy()
        self.d = x[2 * self.problem.n_nodes:].copy()

    def fix_damage_nodes(self, idx, value: float = 1.0):
        for i in np.atleast_1d(idx):
            dof = 2 * self.problem.n_nodes + int(i)
            self.fixed[dof] = True
            self.fixed_values[dof] = value
        self._apply_fixed()

    def reactions(self):
        R, _, _ = assemble(self.problem, self.u, self.d, self.history, self.load)
        return R

    # -- Newton ---------------------------------------------------------------
    def newton_step(self, max_iter: int = 30, tol: float = 1e-9, verbose: bool = False,
                    line_search: bool = True) -> NewtonRecord:
        norms = []
        converged = False
        x = self._x()
        R, K, _ = assemble(self.problem, self.u, self.d, self.history, self.load)
        R = np.where(self.fixed, 0.0, R)
        norms.append(float(np.linalg.norm(R)))
        ref = max(norms[0], 1e-30)
        for it in range(max_iter):
            dnorm = float(np.linalg.norm(R))
            if dnorm <= tol * ref or (norms[0] > 0 and dnorm < 1e-12 * max(1.0, ref)):
                converged = True
                break
            dx = self._solve(K, -R)
            alpha = 1.0
            accepted = False
            for _ in range(6):
                trial = x + alpha * dx
                u_t = trial[:2 * self.problem.n_nodes]
                d_t = np.clip(trial[2 * self.problem.n_nodes:], 0.0, 1.0)
                R_t, K_t, _ = assemble(self.problem, u_t, d_t, self.history, self.load)
                R_t = np.where(self.fixed, 0.0, R_t)
                nt = float(np.linalg.norm(R_t))
                if nt < dnorm or not line_search:
                    x, R, K, dnorm = trial.copy(), R_t, K_t, nt
                    x[2 * self.problem.n_nodes:] = d_t
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                converged = dnorm <= tol * ref * 10.0
                break
            norms.append(dnorm)
            self.u = x[:2 * self.problem.n_nodes].copy()
            self.d = x[2 * self.problem.n_nodes:].copy()
            self._apply_fixed()
            x = self._x()
        else:
            it = max_iter
        self.u = x[:2 * self.problem.n_nodes].copy()
        self.d = np.clip(x[2 * self.problem.n_nodes:], 0.0, 1.0)
        self._apply_fixed()
        return NewtonRecord(norms, len(norms) - 1, converged)

    def _solve(self, K, rhs):
        mask = self.fixed.astype(np.float64)
        free = sp.diags(1.0 - mask)
        Kd = (free @ K @ free + sp.diags(mask)).tocsc()
        rhs = np.where(self.fixed, 0.0, rhs)
        lu = spla.splu(Kd)
        return lu.solve(rhs)

    def staggered_step(self, max_iter: int = 60, tol: float = 1e-8):
        """Alternate minimisation: solve u with d fixed, then d with H frozen."""
        prev = self._x().copy()
        for it in range(max_iter):
            # u-problem (linear)
            R, K, _ = assemble(self.problem, self.u, self.d, self.history, self.load)
            R = np.where(self.fixed, 0.0, R)
            dx = self._solve(K, -R)
            x = self._x() + dx
            self.u = x[:2 * self.problem.n_nodes]
            self.d = np.clip(x[2 * self.problem.n_nodes:], 0.0, 1.0)
            self._apply_fixed()
            # update H floor from the current strain
            e = self.problem.strain(self.u)
            psi_p = self.problem.split(e)[0].numpy()
            self.history = np.maximum(self.history, psi_p)
            R, K, _ = assemble(self.problem, self.u, self.d, self.history, self.load)
            R = np.where(self.fixed, 0.0, R)
            dx = self._solve(K, -R)
            x = self._x() + dx
            self.u = x[:2 * self.problem.n_nodes]
            self.d = np.clip(x[2 * self.problem.n_nodes:], 0.0, 1.0)
            self._apply_fixed()
            delta = np.linalg.norm(self._x() - prev)
            prev = self._x().copy()
            if delta < tol * max(1.0, np.linalg.norm(prev)):
                return it + 1
        return max_iter

    def update_history(self):
        e = self.problem.strain(self.u)
        psi_p = self.problem.split(e)[0].numpy()
        self.history = np.maximum(self.history, psi_p)


# --------------------------------------------------------------------------------------
# load-path driver
# --------------------------------------------------------------------------------------
def run_load_path(problem: PhaseFieldProblem, solver: PhaseFieldSolver,
                  control_dofs, values, *, scheme: str = "monolithic",
                  max_iter: int = 30, tol: float = 1e-9, stop=None):
    """Step a prescribed control DOF set through ``values``.

    Returns a dict of records: applied value, reaction sum, energies, damage
    maximum, Newton residual histories and irreversibility violation counters.
    """
    control_dofs = list(control_dofs)
    records = []
    work = 0.0
    # a pre-existing notch already carries fracture energy; only its growth is
    # paid for by external work, so the ledger uses the excess over this reference
    reference = energies(problem, solver.u, solver.d, solver.history)
    fracture0 = reference["fracture"]
    elastic0 = reference["elastic"]
    previous_d = solver.d.copy()
    previous_values = None
    for value in values:
        for dof in control_dofs:
            solver.fixed[int(dof)] = True
            solver.fixed_values[int(dof)] = float(value)
        solver._apply_fixed()
        if scheme == "staggered":
            iters = solver.staggered_step(max_iter=max_iter, tol=1e-8)
            rec = NewtonRecord([0.0], iters, True)
        else:
            rec = solver.newton_step(max_iter=max_iter, tol=tol)
        solver.update_history()
        solver._apply_fixed()
        react = solver.reactions()
        reaction = float(sum(react[int(dof)] for dof in control_dofs))
        if previous_values is not None:
            du = float(value) - previous_values
            work += 0.5 * (reaction + records[-1]["reaction"]) * du
        previous_values = float(value)
        en = energies(problem, solver.u, solver.d, solver.history)
        drop = previous_d - solver.d
        violations = int((drop > 1e-10).sum())
        records.append({
            "value": float(value),
            "reaction": reaction,
            "elastic": en["elastic"] - elastic0,
            "fracture": en["fracture"] - fracture0,
            "total": (en["elastic"] - elastic0) + (en["fracture"] - fracture0),
            "work": work,
            "d_max": float(solver.d.max()),
            "newton": rec.residual_norms,
            "newton_iters": rec.iterations,
            "converged": rec.converged,
            "irreversibility_violations": violations,
            "damage_drop_max": float(max(drop.max(), 0.0)),
        })
        previous_d = solver.d.copy()
        if stop is not None and stop(solver, records[-1]):
            break
    return records


def crack_line_elements(nodes: np.ndarray, elements: np.ndarray, p0, p1, tol: float) -> np.ndarray:
    """Element indices whose centroid lies within ``tol`` of the segment ``p0 -> p1``."""
    centroids = nodes[elements].mean(axis=1)
    return nodes_on_segment(centroids, p0, p1, tol)


def sharp_crack_energy_release(problems, load, fixed: dict, delta_a: float):
    """Energy-release rate ``G`` of a sharp (slit) crack by central differences.

    ``problems`` must be the three bodies with crack lengths ``a - delta_a``,
    ``a``, ``a + delta_a`` where the crack is a true slit (void element row).
    For dead loading and linear elasticity ``Pi = U - f.u`` and ``G = -dPi/da / t``.
    """
    potentials = []
    details = []
    for problem in problems:
        d = np.zeros(problem.n_nodes)
        bc = dict(fixed)
        # the crack is a true slit: the phase field is frozen at d = 0 everywhere
        for nd in range(problem.n_nodes):
            bc[2 * problem.n_nodes + nd] = 0.0
        for nd in getattr(problem, "orphan_nodes", []):
            bc[2 * int(nd)] = 0.0
            bc[2 * int(nd) + 1] = 0.0
            bc[2 * problem.n_nodes + int(nd)] = 0.0
        solver = PhaseFieldSolver(problem, fixed=bc, initial_damage=d, load=load)
        solver.newton_step(max_iter=12, tol=1e-12, line_search=True)
        u = solver.u
        strain_energy = problem_strain_energy(problem, u, d)
        external = float(np.asarray(load)[: 2 * problem.n_nodes] @ u)
        potentials.append(strain_energy - external)
        details.append({"strain_energy": strain_energy, "external": external,
                        "potential": strain_energy - external})
    dPi = (potentials[2] - potentials[0]) / (2.0 * delta_a)
    return {"G": float(-dPi / problems[1].thickness), "potential": potentials,
            "details": details}


def problem_strain_energy(problem: PhaseFieldProblem, u, d):
    """Degraded elastic energy of a state (Amor split, void elements excluded)."""
    e = problem.strain(u)
    psi_p, psi_m, _, _ = problem.split(e)
    keep = problem._keep
    g = problem.degradation(problem.damage_at_gauss(d)) * keep
    w = problem._wdet
    return float((w * (g * psi_p + keep * psi_m)).sum())
