"""Independent three-dimensional continuum oracle for the 18-degree-hole hemisphere.

The production path for the MacNeal--Harder hemisphere is the flat-facet
``projected_shell4`` element (:mod:`tensorfem.spherical_shell`).  Its recovered
surface stresses have no independent reference, so they were only ever
*documented*, never *certified* (``stress_accuracy_qualified=False`` in
:func:`tensorfem.hemisphere_benchmark_fields.hemisphere_field_report`).

This module supplies that missing oracle with a deliberately different
mechanical model:

* no shell theory at all -- the shell is meshed as a **three-dimensional
  solid of revolution** (quadratic HEX20 bricks, :mod:`tensorfem.hex20`);
* the through-thickness direction is resolved with ``nz`` brick layers
  (``2*nz+1`` nodes), so the linear-through-thickness bending stress can be
  represented and extrapolated to the two free surfaces;
* nothing from the shell/shell4 code path is imported, so a shared bug is
  excluded by construction.

Because both the geometry and the mesh are written in the surface parameters
``(phi, theta, rho)`` -- polar angle, azimuth and through-thickness offset --
the same generator produces the hemisphere of the benchmark *and* the
cylindrical shell used to give the oracle itself an analytic baseline.

Only ``tensorfem.hex20`` (element matrices) and a sparse linear solve are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .hex20 import _EDGES, hex20_shape_gradients, hex20_stiffness
from .solid3d import _b, elasticity_matrix_3d

# MacNeal & Harder (1985) hemisphere benchmark, "proposed standard set of
# problems to test finite element accuracy", Finite Elements in Analysis and
# Design 1, 3-20, DOI 10.1016/0168-874X(85)90003-4.
MH_RADIUS = 10.0
MH_THICKNESS = 0.04
MH_HOLE_POLAR_DEGREES = 18.0
MH_YOUNG = 6.825e7
MH_POISSON = 0.3
MH_REFERENCE_DISPLACEMENT = 0.0924


@dataclass(frozen=True)
class Hex20SolidResult:
    """Nodal solution plus element-centroid stress of a HEX20 block."""

    nodes: torch.Tensor
    elements: torch.Tensor
    displacement: torch.Tensor
    reaction: torch.Tensor
    strain: torch.Tensor
    stress: torch.Tensor
    forces: torch.Tensor
    fixed_dofs: torch.Tensor
    stiffness: torch.Tensor

    def equilibrium(self) -> dict:
        """Solve-quality and global equilibrium diagnostics.

        The support reaction is the residual ``K u - f`` at the restrained DOFs,
        so ``sum(R) + sum(f_free)`` and the matching moment must vanish.  The
        free-DOF residual is reported relative to the load norm.
        """
        residual = (self.stiffness @ self.displacement - self.forces)
        free = torch.ones(self.displacement.numel(), dtype=torch.bool)
        free[self.fixed_dofs.to(torch.long)] = False
        applied = self.forces.reshape(-1, 3)
        applied_free = torch.where(free.reshape(-1, 3), applied, torch.zeros_like(applied))
        reaction = torch.where(free.reshape(-1, 3), torch.zeros_like(applied),
                               residual.reshape(-1, 3))
        imbalance = reaction.sum(0) + applied_free.sum(0)
        moment = (torch.linalg.cross(self.nodes, reaction, dim=1).sum(0)
                  + torch.linalg.cross(self.nodes, applied_free, dim=1).sum(0))
        scale = max(float(torch.linalg.vector_norm(applied_free.sum(0))),
                    float(torch.linalg.vector_norm(reaction, dim=1).max()),
                    float(torch.linalg.vector_norm(self.forces)), 1e-30)
        return {
            "free_residual_relative": float(torch.linalg.vector_norm(residual[free]))
            / max(float(torch.linalg.vector_norm(self.forces[free])),
                  float(torch.linalg.vector_norm(reaction)), 1e-30),
            "normalized_force_balance_error": float(torch.linalg.vector_norm(imbalance)) / scale,
            "normalized_moment_balance_error": float(torch.linalg.vector_norm(moment))
            / (MH_RADIUS * scale),
            "total_applied_force": applied.sum(0).tolist(),
            "total_reaction_force": reaction.sum(0).tolist(),
        }


def rotate_ring_stress(stress: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate ``(xx, yy, zz, xy, yz, zx)`` into the local ``(r, theta, z)`` frame."""
    sxx, syy, szz, sxy, syz, szx = stress.unbind(-1)
    c, s = torch.cos(theta), torch.sin(theta)
    rr = c * c * sxx + s * s * syy + 2 * c * s * sxy
    tt = s * s * sxx + c * c * syy - 2 * c * s * sxy
    rt = -c * s * sxx + c * s * syy + (c * c - s * s) * sxy
    return torch.stack((rr, tt, szz, rt, c * syz - s * szx, s * syz + c * szx), -1)


# --------------------------------------------------------------------------
# parametrised mesh of a thin shell of revolution, discretised as a solid ring
# --------------------------------------------------------------------------
def shell_solid_hex20_mesh(
    nphi: int,
    ntheta: int,
    nz: int,
    *,
    radius: float = MH_RADIUS,
    thickness: float = MH_THICKNESS,
    phi_start: float = math.radians(MH_HOLE_POLAR_DEGREES),
    phi_end: float = math.pi / 2,
    theta_start: float = 0.0,
    theta_end: float = math.pi / 2,
    cylinder: bool = False,
    dtype=torch.float64,
):
    """Structured HEX20 mesh of a spherical (or cylindrical) shell of revolution.

    Nodes are placed exactly on the ``(phi, theta, rho)`` parametrisation, so the
    curved surface is represented without chordal mid-edge error.  ``phi`` runs
    from ``phi_start`` to ``phi_end`` (polar angle from the apex for the sphere;
    axial coordinate for the cylinder), ``theta`` from ``theta_start`` to
    ``theta_end`` (a quarter model), ``rho`` over ``[-t/2, +t/2]``.
    """
    for name, value in (("nphi", nphi), ("ntheta", ntheta), ("nz", nz)):
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1")
    nphi, ntheta, nz = int(nphi), int(ntheta), int(nz)

    def coords(i: int, j: int, k: int):
        """Map the half-step integer grid (2*n+1 points) to global Cartesian."""
        sphi = i / (2.0 * nphi)
        sth = j / (2.0 * ntheta)
        srho = k / (2.0 * nz)
        phi = phi_start + (phi_end - phi_start) * sphi
        theta = theta_start + (theta_end - theta_start) * sth
        rho = -0.5 * thickness + thickness * srho
        radial = radius + rho
        if cylinder:
            # phi plays the role of axial length; the sign keeps the (phi, theta, rho)
            # natural ordering right-handed so every element Jacobian is positive.
            axial = -(phi - phi_start) * radius
            x = radial * math.cos(theta)
            y = radial * math.sin(theta)
            return (x, y, axial)
        x = radial * math.sin(phi) * math.cos(theta)
        y = radial * math.sin(phi) * math.sin(theta)
        z = radial * math.cos(phi)
        return (x, y, z)

    index: dict = {}
    nodes: list = []

    def node(key):
        found = index.get(key)
        if found is None:
            found = len(nodes)
            index[key] = found
            nodes.append(coords(*key))
        return found

    elements = []
    for k in range(nz):
        for j in range(ntheta):
            for i in range(nphi):
                base = ((2 * i, 2 * j, 2 * k), (2 * i + 2, 2 * j, 2 * k),
                        (2 * i + 2, 2 * j + 2, 2 * k), (2 * i, 2 * j + 2, 2 * k),
                        (2 * i, 2 * j, 2 * k + 2), (2 * i + 2, 2 * j, 2 * k + 2),
                        (2 * i + 2, 2 * j + 2, 2 * k + 2), (2 * i, 2 * j + 2, 2 * k + 2))
                row = [node(c) for c in base]
                row += [node(tuple((base[a][d] + base[b][d]) // 2 for d in range(3)))
                        for a, b in _EDGES]
                elements.append(row)
    return (torch.tensor(nodes, dtype=dtype),
            torch.tensor(elements, dtype=torch.long))


def shell_solid_hex20_mesh_index(nphi: int, ntheta: int, nz: int):
    """Node lookup ``(i, j, k) -> index`` for the mesh above (i,j,k half-steps)."""
    # The generator numbers nodes lazily; replay the same key order to recover it.
    index: dict = {}

    def node(key):
        if key not in index:
            index[key] = len(index)
        return index[key]

    for k in range(nz):
        for j in range(ntheta):
            for i in range(nphi):
                base = ((2 * i, 2 * j, 2 * k), (2 * i + 2, 2 * j, 2 * k),
                        (2 * i + 2, 2 * j + 2, 2 * k), (2 * i, 2 * j + 2, 2 * k),
                        (2 * i, 2 * j, 2 * k + 2), (2 * i + 2, 2 * j, 2 * k + 2),
                        (2 * i + 2, 2 * j + 2, 2 * k + 2), (2 * i, 2 * j + 2, 2 * k + 2))
                for c in base:
                    node(c)
                for a, b in _EDGES:
                    node(tuple((base[a][d] + base[b][d]) // 2 for d in range(3)))
    return index


# --------------------------------------------------------------------------
# linear solve
# --------------------------------------------------------------------------
def _solve_sparse(k: torch.Tensor, force: torch.Tensor, fixed: torch.Tensor, values: torch.Tensor):
    """Solve ``K u = f`` with the fixed DOFs eliminated.

    The assembled block is a thin-walled solid, whose conditioning is far beyond
    what Jacobi-preconditioned CG handles in a bounded iteration count, so an
    exact sparse factorisation is used when SciPy is available.
    """
    n = force.numel()
    free = torch.ones(n, dtype=torch.bool)
    free[fixed] = False
    free_idx = torch.arange(n)[free]
    u = torch.zeros(n, dtype=force.dtype)
    u[fixed] = values
    rhs = force[free] - _sparse_rows(k, free_idx, fixed) @ values if len(fixed) else force[free]
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        coo = k.to_sparse_coo().coalesce()
        mat = sp.coo_matrix((coo.values().numpy(),
                             (coo.indices()[0].numpy(), coo.indices()[1].numpy())),
                            shape=(n, n)).tocsr()
        sub = mat[free_idx.numpy()][:, free_idx.numpy()].tocsc()
        solution = spla.spsolve(sub, rhs.numpy())
        u[free_idx] = torch.as_tensor(solution, dtype=force.dtype)
    except ImportError:  # pragma: no cover - SciPy is present in the report env
        dense = _sparse_rows(k, free_idx, free_idx).to_dense()
        u[free_idx] = torch.linalg.solve(dense, rhs)
    return u


def _sparse_rows(k: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """Sparse sub-matrix ``k[rows][:, cols]`` of a COO matrix."""
    coo = k.to_sparse_coo().coalesce()
    index = coo.indices()
    row_map = torch.full((coo.shape[0],), -1, dtype=torch.long)
    row_map[rows] = torch.arange(len(rows))
    col_map = torch.full((coo.shape[1],), -1, dtype=torch.long)
    col_map[cols] = torch.arange(len(cols))
    r = row_map[index[0]]
    c = col_map[index[1]]
    keep = (r >= 0) & (c >= 0)
    return torch.sparse_coo_tensor(torch.stack((r[keep], c[keep])), coo.values()[keep],
                                   (len(rows), len(cols))).coalesce()


def solve_shell_solid_hex20(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    *,
    young: float = MH_YOUNG,
    poisson: float = MH_POISSON,
    forces: torch.Tensor,
    fixed_dofs: torch.Tensor,
    prescribed_values: torch.Tensor | None = None,
) -> Hex20SolidResult:
    """Assemble and solve one HEX20 solid-ring model."""
    dtype, device = nodes.dtype, nodes.device
    ne = len(elements)
    d = elasticity_matrix_3d(nodes.new_tensor(young).expand(ne), nodes.new_tensor(poisson).expand(ne))
    ke, b = hex20_stiffness(nodes[elements.long()], d)
    edof = torch.stack(tuple(3 * elements + i for i in range(3)), 2).reshape(ne, 60).long()
    ndof = 3 * len(nodes)
    rows = edof[:, :, None].expand(-1, -1, 60).reshape(-1)
    cols = edof[:, None, :].expand(-1, 60, -1).reshape(-1)
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        k = torch.sparse_coo_tensor(torch.stack((rows, cols)), ke.reshape(-1),
                                    (ndof, ndof), dtype=dtype, device=device).coalesce()
    fixed = fixed_dofs.to(device=device, dtype=torch.long)
    values = (torch.zeros(len(fixed), dtype=dtype, device=device)
              if prescribed_values is None else prescribed_values.to(device=device, dtype=dtype))
    u = _solve_sparse(k, forces.to(dtype), fixed, values)
    strain = torch.einsum("eij,ej->ei", b, u[edof])
    stress = torch.einsum("eij,ej->ei", d, strain)
    residual = torch.sparse.mm(k, u[:, None])[:, 0] - forces.to(dtype)
    return Hex20SolidResult(nodes, elements, u, residual, strain, stress,
                            forces.to(dtype), fixed, k)


def hex20_stress_at(nodes, elements, displacement, *, young=MH_YOUNG, poisson=MH_POISSON,
                    natural=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """Element stress at an explicit natural coordinate (e.g. a free surface)."""
    ne = len(elements)
    d = elasticity_matrix_3d(nodes.new_tensor(young).expand(ne), nodes.new_tensor(poisson).expand(ne))
    _, dn = hex20_shape_gradients(natural[0], natural[1], natural[2], like=nodes)
    x = nodes[elements.long()]
    jac = torch.einsum("eia,ib->eab", x, dn)
    grad = torch.einsum("ib,ebc->eic", dn, torch.linalg.inv(jac))
    b = _b(grad)
    edof = torch.stack(tuple(3 * elements + i for i in range(3)), 2).reshape(ne, 60).long()
    strain = torch.einsum("eij,ej->ei", b, displacement[edof])
    return torch.einsum("eij,ej->ei", d, strain)


def hex20_parametric_centre(nodes: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
    """Physical point where ``hex20_stiffness`` evaluates its centroid stress.

    For a curved element the mean of the twenty node coordinates is biased
    towards the chord, so the exact reference must be sampled at the parametric
    centre ``(r,s,t) = (0,0,0)`` instead.
    """
    values, _ = hex20_shape_gradients(0.0, 0.0, 0.0, like=nodes)
    return torch.einsum("i,eia->ea", values, nodes[elements.long()])


def von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Von Mises of ``[n,6]`` ordered ``xx, yy, zz, xy, yz, zx``."""
    sxx, syy, szz, sxy, syz, szx = stress.unbind(-1)
    return torch.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                      + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))

def shell_surface_stress_table(result: Hex20SolidResult, nphi: int, ntheta: int, nz: int, *,
                               outer: bool = True, young: float = MH_YOUNG,
                               poisson: float = MH_POISSON) -> dict:
    """Surface stress at the parametric centre of every ``(i, j)`` element column.

    The outermost (or innermost) brick layer is evaluated at its face centre
    ``t = +1`` (``-1``), where the traction-free condition ``sigma_nn = 0`` holds,
    so the resulting tensor is directly comparable to a shell element's recovered
    top/bottom surface stress.
    """
    layer = nz - 1 if outer else 0
    t = 1.0 if outer else -1.0
    ids = torch.tensor([[layer * ntheta * nphi + j * nphi + i for i in range(nphi)]
                        for j in range(ntheta)], dtype=torch.long)
    columns = ids.reshape(-1)
    stress = hex20_stress_at(result.nodes, result.elements[columns], result.displacement,
                             young=young, poisson=poisson, natural=(0.0, 0.0, t))
    values, _ = hex20_shape_gradients(0.0, 0.0, t, like=result.nodes)
    point = torch.einsum("i,eia->ea", values, result.nodes[result.elements[columns].long()])
    radial = torch.linalg.vector_norm(point, dim=1)
    phi = torch.arccos(torch.clamp(point[:, 2] / radial, -1.0, 1.0))
    theta = torch.atan2(point[:, 1], point[:, 0])
    if bool((radial <= 0).any()):
        raise ValueError("degenerate surface point at the axis")
    return {"element_index": columns, "point": point, "polar_angle": phi,
            "azimuth": theta, "stress": stress, "von_mises": von_mises(stress)}
