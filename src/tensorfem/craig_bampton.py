"""Craig--Bampton component mode synthesis (CMS) for partitioned FE models.

Why this module exists
----------------------
Every modal routine in the library so far works on a *monolithic* stiffness and
mass matrix.  Ship hull-segment models, however, are built from components
(hold blocks, deck houses, machinery foundations) that are designed, meshed and
re-analysed independently and are only joined through a small interface.  Without
a component mode synthesis the only options are to re-solve the assembled model
from scratch every time a component changes, or to use static (Guyan) reduction,
which systematically over-stiffens the model because it drops all inertia of the
internal degrees of freedom.

This module implements the fixed-interface Craig--Bampton transformation

    u = T q,     T = [[Phi   Psi_c],
                      [ 0     I  ]]                     (1)

with ``Phi`` the *fixed-interface normal modes* (eigenmodes of the component with
all interface degrees of freedom held) and ``Psi_c = -K_ss^{-1} K_si`` the
*constraint modes* (exact static response of the internal degrees of freedom to
unit interface displacements).  ``T`` is applied component by component and the
reduced components are assembled on the shared interface degrees of freedom.

Properties that are checked in ``benchmarks/craig_bampton_verification.py`` and
in ``tests/test_craig_bampton.py`` (they are identities, not snapshots):

* ``T`` is ``M``-orthogonal across the constraint/fixed-interface split,
  ``Psi_c^T M_ss Phi = 0`` and ``Phi^T M_ss Phi = I`` (eq. 1 is exact).
* Retaining *all* internal modes reproduces the monolithic eigenvalues exactly.
* Retaining *zero* modes degenerates exactly to Guyan (static) reduction.
* Truncation never lowers a frequency: the Rayleigh--Ritz bound
  ``omega_reduced >= omega_exact`` holds for every retained set.
* A quasi-static interface load is carried exactly at any truncation level
  (the constraint modes alone transmit it).

Scope / limitations (honest)
----------------------------
* Small-strain linear elasto-dynamics only; no prestress, no damping coupling,
  no rotational-interface kinematics beyond what the host elements provide.
* Dense algebra; the intended use is component-level reduction of meshes with
  up to a few thousand degrees of freedom, not sparse industrial scale.
* Only the *fixed-interface* variant is offered.  Free-interface CMS and the
  residual-flexibility (Craig--Chang / Rubin) second-order correction are not
  implemented -- truncation error is therefore the classical fixed-interface
  one, which the convergence table quantifies.

References
----------
R.R. Craig Jr. and M.C.C. Bampton, "Coupling of substructures for dynamic
analyses", *AIAA Journal* 6(7):1313--1319, 1968.
R.R. Craig Jr. and A.J. Kurdila, *Fundamentals of Structural Dynamics*, 2nd ed.,
Wiley, 2006, ch. 15--16 (coupling of substructures).
O.C. Zienkiewicz and R.L. Taylor, *The Finite Element Method*, vol. 2, ch. 9
(substructuring and dynamic reduction).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "q4_consistent_mass",
    "assemble_dense",
    "plane_stress_component",
    "partition_dofs",
    "guyan_transformation",
    "fixed_interface_modes",
    "craig_bampton_transformation",
    "generalized_eigen",
    "build_superelement",
    "Superelement",
    "ReducedSystem",
    "assemble_superelements",
    "ComponentMesh",
    "extract_component",
    "restrict_system",
]


# ---------------------------------------------------------------------------
# Element mass matrix and assembly
# ---------------------------------------------------------------------------

def q4_consistent_mass(coordinates: torch.Tensor, density: torch.Tensor,
                       thickness: torch.Tensor) -> torch.Tensor:
    """Consistent mass matrix of a bilinear plane element (2 dofs per node).

    ``coordinates`` has shape ``[element, 4, 2]`` in counter-clockwise order,
    ``density`` / ``thickness`` broadcast to ``[element]``.  The result is
    ``[element, 8, 8]`` with the same ``rho t int N_i N_j dA`` block repeated on
    both displacement directions -- so the row sum of ``M`` recovers
    ``rho * t * area`` per direction, which is the mass-conservation check.
    """
    g = 1.0 / 3.0**0.5
    points = ((-g, -g), (g, -g), (g, g), (-g, g))
    ne = coordinates.shape[0]
    ke = torch.zeros((ne, 8, 8), dtype=coordinates.dtype, device=coordinates.device)
    scale = (density * thickness).reshape(-1, 1, 1)
    for xi, eta in points:
        shape = coordinates.new_tensor([
            (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta), (1 - xi) * (1 + eta),
        ]) / 4.0
        natural = coordinates.new_tensor([
            [-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
            [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)],
        ]) / 4.0
        jac = torch.einsum("eia,ib->eab", coordinates, natural)
        det = torch.linalg.det(jac)
        if torch.any(det <= 0):
            raise ValueError("Q4 element has non-positive Jacobian")
        block = torch.outer(shape, shape) * (det.reshape(-1, 1, 1) * scale)
        for d in range(2):
            ke[:, d::2, d::2] += block
    return ke


def assemble_dense(element_matrices: torch.Tensor, connectivity: torch.Tensor,
                   n_dofs: int, dofs_per_node: int = 2) -> torch.Tensor:
    """Scatter element matrices (dense) into a global square matrix."""
    nn = connectivity.shape[1]
    rows = (connectivity.unsqueeze(-1) * dofs_per_node
            + torch.arange(dofs_per_node, device=connectivity.device)).reshape(connectivity.shape[0], -1)
    assert rows.shape[1] == nn * dofs_per_node
    total = torch.zeros((n_dofs, n_dofs), dtype=element_matrices.dtype,
                        device=element_matrices.device)
    total.index_put_((rows[:, :, None], rows[:, None, :]), element_matrices, accumulate=True)
    return total


def plane_stress_component(nodes: torch.Tensor, elements: torch.Tensor, *,
                           young: float, poisson: float, density: float,
                           thickness: float = 1.0,
                           plane: str = "stress") -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble ``(K, M)`` of a 2-d Q4 component (2 dofs per node)."""
    from .continuum import elasticity_matrix, q4_stiffness

    coords = nodes[elements]
    ne = elements.shape[0]
    dtype = nodes.dtype
    dmat = elasticity_matrix(nodes.new_tensor(young).reshape(1),
                             nodes.new_tensor(poisson).reshape(1), plane).expand(ne, 3, 3)
    thick = nodes.new_tensor(thickness).reshape(1).expand(ne)
    stiffness, _ = q4_stiffness(coords, dmat, thick)
    mass = q4_consistent_mass(coords, nodes.new_tensor(density).reshape(1).expand(ne), thick)
    n_dofs = 2 * nodes.shape[0]
    K = assemble_dense(stiffness, elements, n_dofs)
    M = assemble_dense(mass, elements, n_dofs)
    return 0.5 * (K + K.T), 0.5 * (M + M.T)


# ---------------------------------------------------------------------------
# Partitioning and the two transformations
# ---------------------------------------------------------------------------

def partition_dofs(n_dofs: int, interface_dofs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(internal_dofs, interface_dofs_sorted)`` for a global dof count."""
    iface = torch.as_tensor(interface_dofs, dtype=torch.long).reshape(-1)
    if iface.numel() and (iface.min() < 0 or iface.max() >= n_dofs):
        raise ValueError("interface dof out of range")
    if torch.unique(iface).numel() != iface.numel():
        raise ValueError("interface dofs must be unique")
    iface = torch.sort(iface).values
    mask = torch.ones(n_dofs, dtype=torch.bool, device=iface.device)
    mask[iface] = False
    return torch.nonzero(mask).reshape(-1), iface


def guyan_transformation(K: torch.Tensor, interface_dofs: torch.Tensor) -> torch.Tensor:
    """Static (Guyan) transformation ``T = [[-K_ss^-1 K_si], [I]]``."""
    n = K.shape[0]
    if K.shape != (n, n):
        raise ValueError("K must be square")
    internal, iface = partition_dofs(n, interface_dofs)
    T = torch.zeros((n, iface.numel()), dtype=K.dtype, device=K.device)
    if iface.numel() == 0:
        return T
    if internal.numel():
        K_ss = K[internal[:, None], internal[None, :]]
        K_si = K[internal[:, None], iface[None, :]]
        T[internal] = torch.linalg.solve(K_ss, -K_si)
    T[iface] = torch.eye(iface.numel(), dtype=K.dtype, device=K.device)
    return T


def fixed_interface_modes(K: torch.Tensor, M: torch.Tensor,
                          interface_dofs: torch.Tensor,
                          n_modes: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-interface eigenpairs ``(omega2, Phi)`` of the internal partition.

    ``Phi`` is ``M_ss``-orthonormal with shape ``[n_internal, n_kept]``; the
    returned ``Phi`` is the full-length vector set (zeros on interface rows), so
    it can be used directly as the first block column of ``T``.
    """
    internal, iface = partition_dofs(K.shape[0], interface_dofs)
    n_int = internal.numel()
    if n_int == 0:
        return torch.zeros(0, dtype=K.dtype, device=K.device), torch.zeros(
            (K.shape[0], 0), dtype=K.dtype, device=K.device)
    K_ss = K[internal[:, None], internal[None, :]]
    M_ss = M[internal[:, None], internal[None, :]]
    keep = n_int if n_modes is None else int(n_modes)
    if keep < 0:
        raise ValueError("n_modes must be non-negative")
    keep = min(keep, n_int)
    omega2, phi_s = generalized_eigen(K_ss, M_ss, keep)
    phi = torch.zeros((K.shape[0], phi_s.shape[1]), dtype=K.dtype, device=K.device)
    phi[internal] = phi_s
    return omega2, phi


def craig_bampton_transformation(K: torch.Tensor, M: torch.Tensor,
                                 interface_dofs: torch.Tensor,
                                 n_modes: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(T, omega2_fixed)`` for the fixed-interface Craig--Bampton basis.

    ``T`` has ``n_interface + n_kept`` columns: the constraint modes first, then
    the retained fixed-interface modes (eq. 1 of the module docstring).
    ``n_modes=0`` reproduces :func:`guyan_transformation` exactly.
    """
    iface = torch.sort(torch.as_tensor(interface_dofs, dtype=torch.long).reshape(-1)).values
    Tg = guyan_transformation(K, iface)
    omega2, phi = fixed_interface_modes(K, M, iface, n_modes)
    return torch.cat((Tg, phi), dim=1), omega2


def generalized_eigen(K: torch.Tensor, M: torch.Tensor,
                      n_modes: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Lowest ``n_modes`` eigenpairs of ``K phi = omega^2 M phi`` (dense, SPD M)."""
    n = K.shape[0]
    if K.shape != (n, n) or M.shape != (n, n):
        raise ValueError("K and M must be square and equally sized")
    if n == 0:
        return torch.zeros(0, dtype=K.dtype, device=K.device), torch.zeros(
            (0, 0), dtype=K.dtype, device=K.device)
    L = torch.linalg.cholesky(0.5 * (M + M.T))
    Linv = torch.linalg.solve_triangular(
        L, torch.eye(n, dtype=K.dtype, device=K.device), upper=False)
    A = Linv @ K @ Linv.transpose(-1, -2)
    A = 0.5 * (A + A.T)
    w, v = torch.linalg.eigh(A)
    keep = n if n_modes is None else min(int(n_modes), n)
    phi = torch.linalg.solve_triangular(L.transpose(-1, -2), v[:, :keep], upper=True)
    # re-orthonormalise against M so the identity Phi^T M Phi = I holds to
    # machine precision even when the eigen-solver returns a near-degenerate pair
    MtPhi = M @ phi
    gram = phi.transpose(-1, -2) @ MtPhi
    gram = 0.5 * (gram + gram.T)
    Lg = torch.linalg.cholesky(gram)
    phi = torch.linalg.solve_triangular(Lg, phi.transpose(-1, -2), upper=False).transpose(-1, -2)
    return w[:keep], phi


def restrict_system(K: torch.Tensor, M: torch.Tensor,
                    keep_dofs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict ``(K, M)`` to ``keep_dofs`` (used to apply clamps at any level)."""
    keep = torch.as_tensor(keep_dofs, dtype=torch.long).reshape(-1)
    if keep.numel() == 0:
        return (torch.zeros((0, 0), dtype=K.dtype, device=K.device),
                torch.zeros((0, 0), dtype=K.dtype, device=K.device))
    return K[keep[:, None], keep[None, :]], M[keep[:, None], keep[None, :]]


# ---------------------------------------------------------------------------
# Component extraction from a monolithic mesh
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentMesh:
    """A sub-mesh of a monolithic model plus its local -> host dof map."""

    nodes: torch.Tensor
    elements: torch.Tensor
    dof_map: torch.Tensor          # local dof index -> host dof index

    @property
    def n_dofs(self) -> int:
        return 2 * int(self.nodes.shape[0])

    def dofs_on(self, axis: int, value: float, tol: float = 1e-12) -> torch.Tensor:
        """Local dofs located on the coordinate line ``x_axis = value``."""
        hit = torch.nonzero((self.nodes[:, axis] - value).abs() <= tol).reshape(-1)
        return torch.sort(torch.cat((2 * hit, 2 * hit + 1))).values

    def host_dofs(self, local_dofs: torch.Tensor) -> torch.Tensor:
        """Translate local dofs into the host numbering."""
        return self.dof_map[torch.as_tensor(local_dofs, dtype=torch.long).reshape(-1)]


def extract_component(nodes: torch.Tensor, elements: torch.Tensor,
                      element_mask: torch.Tensor) -> ComponentMesh:
    """Pull one component (a subset of elements) out of a monolithic mesh."""
    mask = torch.as_tensor(element_mask, dtype=torch.bool).reshape(-1)
    if mask.numel() != elements.shape[0]:
        raise ValueError("element_mask must have one entry per element")
    el = elements[mask]
    if el.numel() == 0:
        raise ValueError("empty component")
    used = torch.unique(el)
    remap = torch.full((nodes.shape[0],), -1, dtype=torch.long)
    remap[used] = torch.arange(used.numel())
    dof_map = torch.sort(torch.cat((2 * used, 2 * used + 1))).values
    return ComponentMesh(nodes=nodes[used], elements=remap[el], dof_map=dof_map)


# ---------------------------------------------------------------------------
# Superelements and their assembly
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Superelement:
    """A fixed-interface Craig--Bampton superelement.

    ``global_interface_dofs`` names the rows of the *assembled* model that the
    constraint modes attach to; ``local_interface_dofs`` are the same degrees of
    freedom in the component-local numbering of ``T``.
    """

    global_interface_dofs: torch.Tensor
    local_interface_dofs: torch.Tensor
    T: torch.Tensor
    K: torch.Tensor
    M: torch.Tensor
    fixed_interface_omega2: torch.Tensor
    n_component_dofs: int

    @property
    def n_interface(self) -> int:
        return int(self.global_interface_dofs.numel())

    @property
    def n_modes(self) -> int:
        return int(self.fixed_interface_omega2.numel())

    @property
    def n_reduced_dofs(self) -> int:
        return int(self.T.shape[1])


def build_superelement(K: torch.Tensor, M: torch.Tensor,
                       local_interface_dofs: torch.Tensor,
                       n_modes: int | None,
                       global_interface_dofs: torch.Tensor | None = None) -> Superelement:
    """Reduce one component to a superelement on its interface degrees of freedom.

    The interface columns of the returned ``T`` follow ascending
    ``global_interface_dofs`` so that :func:`assemble_superelements` can scatter
    them without any further reordering.
    """
    local = torch.as_tensor(local_interface_dofs, dtype=torch.long).reshape(-1)
    gif = (torch.as_tensor(global_interface_dofs, dtype=torch.long).reshape(-1)
           if global_interface_dofs is not None else local.clone())
    if gif.numel() != local.numel():
        raise ValueError("global and local interface dof counts differ")
    if torch.unique(gif).numel() != gif.numel():
        raise ValueError("duplicate global interface dofs")
    if torch.unique(local).numel() != local.numel():
        raise ValueError("duplicate local interface dofs")
    order = torch.argsort(gif)
    gif, local = gif[order], local[order]
    T, omega2 = craig_bampton_transformation(K, M, local, n_modes)
    n_if = int(local.numel())
    n_cols = int(T.shape[1])
    if n_if:
        # craig_bampton_transformation sorts the interface internally; restore the
        # caller-facing order (ascending global dof) before assembly
        _, sorted_idx = torch.sort(local)
        perm_iface = torch.argsort(sorted_idx)
        perm = torch.cat((perm_iface, torch.arange(n_if, n_cols, dtype=torch.long)))
        T = T[:, perm]
    Kr = T.transpose(-1, -2) @ K @ T
    Mr = T.transpose(-1, -2) @ M @ T
    return Superelement(global_interface_dofs=gif, local_interface_dofs=local,
                        T=T, K=0.5 * (Kr + Kr.T), M=0.5 * (Mr + Mr.T),
                        fixed_interface_omega2=omega2, n_component_dofs=int(K.shape[0]))


@dataclass(frozen=True)
class ReducedSystem:
    """Assembled reduced model: ``[interface dofs | modes(comp 1) | modes(comp 2) | ...]``."""

    components: tuple[Superelement, ...]
    interface_dofs: torch.Tensor
    K: torch.Tensor
    M: torch.Tensor
    offsets: tuple[int, ...]

    @property
    def n_dofs(self) -> int:
        return int(self.K.shape[0])

    def modes(self, n_modes: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return generalized_eigen(self.K, self.M, n_modes)

    def expand(self, phi_reduced: torch.Tensor) -> list[torch.Tensor]:
        """Map reduced eigenvectors back to each component's local dof vector.

        The vectors are expressed in the component numbering *after* any dofs the
        caller removed before reduction (e.g. clamped rows), not in the numbering
        of the un-reduced component mesh.
        """
        out: list[torch.Tensor] = []
        n_if = int(self.interface_dofs.numel())
        iface_pos = {int(d): i for i, d in enumerate(self.interface_dofs.reshape(-1).tolist())}
        for comp, off in zip(self.components, self.offsets):
            sel = torch.tensor([iface_pos[int(d)] for d in comp.global_interface_dofs.reshape(-1).tolist()],
                               dtype=torch.long)
            block = torch.arange(off, off + comp.n_modes, dtype=torch.long)
            q = torch.cat((phi_reduced[sel], phi_reduced[block]), dim=0)
            out.append(comp.T @ q)
        return out


def assemble_superelements(components: list[Superelement]) -> ReducedSystem:
    """Assemble reduced components through their shared global interface dofs."""
    if not components:
        raise ValueError("at least one component is required")
    iface = torch.unique(torch.cat([c.global_interface_dofs for c in components])).sort().values
    n_if = int(iface.numel())
    pos = {int(d): i for i, d in enumerate(iface.reshape(-1).tolist())}
    offsets: list[int] = []
    size = n_if
    for c in components:
        offsets.append(size)
        size += c.n_modes
    dtype = components[0].K.dtype
    device = components[0].K.device
    K = torch.zeros((size, size), dtype=dtype, device=device)
    M = torch.zeros((size, size), dtype=dtype, device=device)
    for comp, off in zip(components, offsets):
        ii = torch.tensor([pos[int(d)] for d in comp.global_interface_dofs.reshape(-1).tolist()],
                          dtype=torch.long)
        n_if_c = comp.n_interface
        blk = torch.arange(off, off + comp.n_modes, dtype=torch.long)
        for target, src in ((K, comp.K), (M, comp.M)):
            target.index_put_((ii[:, None], ii[None, :]), src[:n_if_c, :n_if_c], accumulate=True)
            if comp.n_modes:
                target.index_put_((ii[:, None], blk[None, :]),
                                  src[:n_if_c, n_if_c:], accumulate=True)
                target.index_put_((blk[:, None], ii[None, :]),
                                  src[n_if_c:, :n_if_c], accumulate=True)
                target.index_put_((blk[:, None], blk[None, :]),
                                  src[n_if_c:, n_if_c:], accumulate=True)
    return ReducedSystem(components=tuple(components), interface_dofs=iface,
                         K=0.5 * (K + K.T), M=0.5 * (M + M.T), offsets=tuple(offsets))