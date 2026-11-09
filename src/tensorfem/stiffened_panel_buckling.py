"""Linearised (initial-stress) bifurcation of stiffened panels and plate fields.

Why this module exists
----------------------
``tensorfem.layered_shell4_plasticity`` is documented as a *small-strain,
small-rotation* layered shell: its element tangent is the material tangent and
it carries **no geometric stiffness**.  ``tensorfem.stiffened_panel_unit_fe``
therefore traces a geometrically linear elasto-plastic path, and a slender panel
shows no loss of stiffness at all on that path: the campaign panel of this node
keeps a perfectly linear load--end-shortening relation (two Newton iterations,
zero plastic strain) up to ``0.63 sigma_y`` although its classical plate buckling
stress is ``0.47 sigma_y``.  Neither a buckling load nor a post-buckling limit
point can be extracted from that path.

This module supplies the missing *buckling* operator instead of pretending the
small-strain path can supply it:

1. **Panel unit-cell mesh** -- the plate field, web strip and flange strip of a
   longitudinally stiffened panel cell are taken from
   :func:`tensorfem.stiffened_panel_unit_fe.build_panel_unit`, so geometry,
   thicknesses, symmetry lines and edge supports are *the same* cell the
   nonlinear path uses (no second geometry implementation).
2. **Elastic bending operator** -- assembled from the verified MITC4 flat shell
   :func:`tensorfem.shell4.shell4_stiffness`.  The layered element's 1-point
   shear field carries hourglass modes, which make the pure plate bending
   operator singular, so the buckling operator uses the assumed-natural-strain
   shell (also the element used by the repository's shell buckling benchmarks).
3. **Uniform prebuckling state** -- under uniform axial end compression a
   prismatic panel has no linear redistribution: the plate, web and flange all
   carry the same axial stress, so the initial-stress resultants are
   ``Nx = sigma * t_element`` with ``Ny = Nxy = 0``.  The buckling load is
   reported as an *average stress* over the cell cross-section, which for this
   state equals the member stress.
4. **Initial-stress (geometric) matrix** --
   :func:`tensorfem.shell4_buckling.shell4_geometric_stiffness`, the verified
   transverse-displacement operator of the same facet.
5. **Linearised bifurcation** -- ``(K + lambda K_G) phi = 0`` in the bending
   subspace ``(w, rx, ry)`` (in a linear facet the membrane and drilling blocks
   are uncoupled from it and ``K_G`` touches only ``w``, so the restriction is
   exact).  The pencil is inverted onto the positive-definite elastic operator
   and solved with LAPACK.

Verification surface
--------------------
* :func:`classical_plate_buckling_stress` -- closed form
  ``sigma_cr = k pi^2 D / (b^2 t)`` with ``k = min_m (m b/a + a/(m b))^2``
  (Timoshenko & Gere; the same all-edges-simply-supported uniaxial form as
  Paik & Thayamballi Eq. 2.1).  It consumes nothing from the solve.
* :func:`bare_plate_buckling` -- the same assembly/eigen path on an unstiffened
  simply supported plate, i.e. the stiffened panel with the stiffener removed,
  graded against the closed form above.  This is what qualifies the operator.
* :func:`panel_buckling` -- the stiffened cell.  There the stiffener restrains
  the plate edge (it is not the idealised nodal line of the closed form), so the
  value is reported *against* the idealised plate value rather than as a
  reproduction of it.

Honesty statement
-----------------
* Linearised (eigenvalue) buckling only: resultants are those of the
  prebuckling state and are not updated.  No post-buckling reserve, no ultimate
  strength.
* Rotational geometric terms are omitted (thin-plate transverse term only); the
  bare-plate agreement with the closed form quantifies that omission.
* No experimental value is quoted anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .shell4 import shell4_stiffness
from .shell4_buckling import shell4_geometric_stiffness
from .stiffened_panel_ultimate import StiffenedPanel
from .stiffened_panel_unit_fe import build_panel_unit

__all__ = [
    "BucklingResult",
    "PanelCellMesh",
    "assembly_buckling",
    "bare_plate_buckling",
    "classical_plate_buckling_stress",
    "panel_buckling",
    "panel_cell_mesh",
    "plate_cell_mesh",
]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelCellMesh:
    """Flat-facet shell mesh with one thickness per element group."""

    nodes: torch.Tensor
    groups: tuple[tuple[torch.Tensor, float], ...]
    fixed: torch.Tensor

    @property
    def n_dofs(self) -> int:
        return 6 * len(self.nodes)

    def free_dofs(self) -> torch.Tensor:
        mask = torch.ones(self.n_dofs, dtype=torch.bool)
        mask[self.fixed.long()] = False
        return torch.arange(self.n_dofs)[mask]

    @property
    def element_count(self) -> int:
        return sum(len(elements) for elements, _ in self.groups)


@dataclass(frozen=True)
class BucklingResult:
    """Lowest linearised buckling modes of an assembled shell model."""

    critical_stress: float
    load_factors: np.ndarray
    modes: np.ndarray                     # (n_modes, n_nodes) transverse nodal field
    mesh: str
    n_dofs: int
    n_elements: int
    nodes: np.ndarray
    plate_elements: np.ndarray
    residuals: np.ndarray = None                 # pencil residual per mode

    def summary(self, reference: float | None = None) -> dict:
        out = {
            "critical_stress_pa": self.critical_stress,
            "load_factors_pa": [float(v) for v in self.load_factors],
            "mesh": self.mesh,
            "n_dofs": self.n_dofs,
            "n_elements": self.n_elements,
        }
        if reference:
            out["reference_stress_pa"] = float(reference)
            out["relative_error"] = float(self.critical_stress / reference - 1.0)
        return out


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def panel_cell_mesh(panel: StiffenedPanel, nx: int, ny: int) -> PanelCellMesh:
    """Unit-cell shell mesh, reusing the nonlinear panel cell's own builder."""
    case = build_panel_unit(panel, nx=nx, ny=ny, layers=1,
                            plate_imperfection_ratio=0.0, column_imperfection_ratio=0.0)
    groups = ((case.plate_model.elements, panel.plate_thickness),
              (case.web_model.elements, panel.web_thickness),
              (case.flange_model.elements, panel.flange_thickness))
    return PanelCellMesh(case.nodes, groups, case.fixed_dofs)


def plate_cell_mesh(a: float, b: float, thickness: float, nx: int, ny: int,
                    *, edge_rotation: str = "free") -> PanelCellMesh:
    """Plate field, ``a`` loaded and ``b`` the supported width.

    ``edge_rotation="free"`` gives the classical simply supported (four edges SS)
    plate that the closed form describes; ``"clamped"`` additionally restrains
    the rotation about the loaded axis on both long edges, which is the other
    classical limit and is used as an upper bracket for the stiffened cell (the
    stiffener restrains, but does not fully clamp, the plate edge).
    """
    if edge_rotation not in ("free", "clamped"):
        raise ValueError("edge_rotation must be 'free' or 'clamped'")
    xs = np.linspace(0.0, a, nx + 1)
    ys = np.linspace(0.0, b, ny + 1)
    stride = nx + 1
    nodes = torch.tensor([[x, y, 0.0] for y in ys for x in xs], dtype=torch.float64)
    elements = torch.tensor([(j * stride + i, j * stride + i + 1,
                              (j + 1) * stride + i + 1, (j + 1) * stride + i)
                             for j in range(ny) for i in range(nx)], dtype=torch.long)
    fixed = set()
    for j in range(ny + 1):
        for i in (0, nx):
            fixed.add(6 * (j * stride + i))        # u = 0 at x = 0, free elsewhere
            fixed.add(6 * (j * stride + i) + 2)    # w = 0 on the loaded edges
    for i in range(nx + 1):
        for j in (0, ny):
            fixed.add(6 * (j * stride + i) + 2)    # w = 0 on the long edges
    fixed.add(6 * (0 * stride + 0) + 1)            # v = 0 removes the in-plane drift
    if edge_rotation == "clamped":
        for i in range(nx + 1):
            for j in (0, ny):
                fixed.add(6 * (j * stride + i) + 3)    # rx = 0: rotation about the load axis
    return PanelCellMesh(nodes, ((elements, thickness),),
                         torch.tensor(sorted(fixed), dtype=torch.long))


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------
def _assemble(mesh: PanelCellMesh, young: float, poisson: float) -> torch.Tensor:
    stiffness = torch.zeros((mesh.n_dofs, mesh.n_dofs), dtype=mesh.nodes.dtype)
    for elements, thickness in mesh.groups:
        for conn in elements:
            conn = conn.long()
            ids = torch.stack(tuple(6 * conn + k for k in range(6)), 1).reshape(-1)
            block = shell4_stiffness(mesh.nodes[conn], young, poisson, thickness,
                                     plate_shear="mitc")
            stiffness.index_put_((ids[:, None], ids[None, :]), block, accumulate=True)
    return stiffness


def _assemble_geometric(mesh: PanelCellMesh, stress: float) -> torch.Tensor:
    """Initial-stress matrix for uniform uniaxial compression of averaged stress."""
    total = torch.zeros((mesh.n_dofs, mesh.n_dofs), dtype=mesh.nodes.dtype)
    for elements, thickness in mesh.groups:
        resultants = torch.tensor([stress * thickness, 0.0, 0.0], dtype=mesh.nodes.dtype)
        for conn in elements:
            conn = conn.long()
            ids = torch.stack(tuple(6 * conn + k for k in range(6)), 1).reshape(-1)
            block = shell4_geometric_stiffness(mesh.nodes[conn], resultants)
            total.index_put_((ids[:, None], ids[None, :]), block, accumulate=True)
    return total


def _bending_pencil(stiffness: torch.Tensor, geometric: torch.Tensor,
                    free: torch.Tensor, n_nodes: int):
    """Restrict both operators to the bending DOF ``(w, rx, ry)``.

    For a linear flat facet the membrane ``(u, v)`` and drilling ``(rz)`` blocks
    are uncoupled from the transverse/rotation block, and ``K_G`` is populated on
    ``w`` only, so the bifurcation eigenproblem lives entirely in
    ``(w, rx, ry)``.  The drilling block is a 1e-6 relative stabilisation and is
    deliberately excluded rather than allowed to set the conditioning.
    """
    index = torch.cat([6 * torch.arange(n_nodes) + k for k in (2, 3, 4)])
    is_bending = torch.zeros(stiffness.shape[0], dtype=torch.bool)
    is_bending[index] = True
    keep = free[is_bending[free]]
    k_bend = stiffness[keep[:, None], keep[None, :]].double()
    g_bend = torch.zeros_like(k_bend)
    w_all = (6 * torch.arange(n_nodes) + 2)
    lookup = {int(dof): i for i, dof in enumerate(keep.tolist())}
    w_pairs = [(int(d), lookup[int(d)]) for d in w_all.tolist() if int(d) in lookup]
    rows = torch.tensor([p[1] for p in w_pairs], dtype=torch.long)
    cols = torch.tensor([p[0] for p in w_pairs], dtype=torch.long)
    g_bend[rows[:, None], rows[None, :]] = geometric[cols, :][:, cols].double()
    return 0.5 * (k_bend + k_bend.T), g_bend, keep


def _solve_pencil(k_bend: torch.Tensor, kg_bend: torch.Tensor, modes: int):
    """Lowest positive eigenvalues of ``K phi = lambda Kg phi``.

    ``Kg`` is singular (a linear transverse field inside one facet carries no
    curvature, so the element geometric matrix has rank 3 of 4) while the elastic
    bending operator ``K`` is positive definite.  The pencil is therefore mapped
    to a standard symmetric problem with the Cholesky factor of ``K``:
    ``B = L^-1 Kg L^-T`` has eigenvalues ``1/lambda``, so the largest ``B``
    eigenvalues are the lowest buckling loads.  Only a *standard* symmetric
    solver is used (the generalised LAPACK driver aborts on this platform for a
    singular pencil, so it is avoided rather than trusted).
    """
    from scipy.linalg import solve_triangular
    k = k_bend.numpy()
    g = kg_bend.numpy()
    n = k.shape[0]
    if n < 2:
        raise ValueError("the bending space is too small")
    cholesky = np.linalg.cholesky(k)
    intermediate = solve_triangular(cholesky, g, lower=True)
    b = solve_triangular(cholesky, intermediate.T, lower=True).T
    b = 0.5 * (b + b.T)
    mu, vectors = np.linalg.eigh(b)
    order = np.argsort(mu)[::-1][:modes]
    load = np.full(modes, np.inf)
    for slot, index in enumerate(order):
        if mu[index] > 0.0:
            load[slot] = 1.0 / mu[index]
    eigen = solve_triangular(cholesky.T, vectors[:, order], lower=False)
    residual = np.zeros(modes)
    k_norm = np.linalg.norm(k, ord=np.inf)
    for slot, column in enumerate(order):
        phi = eigen[:, slot]
        if np.isfinite(load[slot]) and k_norm > 0.0:
            residual[slot] = (np.linalg.norm(k @ phi - load[slot] * (g @ phi))
                              / (k_norm * np.linalg.norm(phi)))
    return load, eigen, residual


def assembly_buckling(mesh: PanelCellMesh, young: float, poisson: float,
                      *, modes: int = 3, label: str = "") -> BucklingResult:
    """Lowest buckling loads of an assembled shell mesh under uniaxial stress."""
    stiffness = _assemble(mesh, young, poisson)
    geometric = _assemble_geometric(mesh, 1.0)          # one pascal of member stress
    free = mesh.free_dofs()
    k_bend, kg_bend, keep = _bending_pencil(stiffness, geometric, free, len(mesh.nodes))
    values, vectors, residuals = _solve_pencil(k_bend, kg_bend, modes)
    w_rows = np.array([i for i, dof in enumerate(keep.tolist()) if int(dof) % 6 == 2])
    nodes = (keep[w_rows] // 6).numpy()
    fields = np.zeros((len(values), len(mesh.nodes)))
    for column in range(vectors.shape[1]):
        amplitude = np.real(vectors[w_rows, column])
        peak = np.abs(amplitude).max()
        if peak > 0:
            amplitude = amplitude / peak
        fields[column, nodes] = amplitude
    plate = mesh.groups[0][0].numpy() if mesh.groups else np.zeros((0, 4), dtype=int)
    return BucklingResult(float(values[0]), np.asarray(values), fields,
                          label or mesh_str(mesh), int(len(free)), mesh.element_count,
                          mesh.nodes.numpy(), plate, np.asarray(residuals))


def mesh_str(mesh: PanelCellMesh) -> str:
    return f"{mesh.element_count} elements / {len(mesh.nodes)} nodes"


# --------------------------------------------------------------------------
# closed-form oracle
# --------------------------------------------------------------------------
def classical_plate_buckling_stress(a: float, b: float, thickness: float,
                                     young: float, poisson: float, *,
                                     half_waves: int | None = None) -> dict:
    """Uniaxial buckling of a simply supported plate (all four edges SS).

    ``a`` is the loaded length and ``b`` the supported width, so
    ``k = min_m (m b/a + a/(m b))^2``.  This is the classical result, quoted in
    the same form as Paik & Thayamballi Eq. 2.1 for a plate with all edges
    simply supported.
    """
    for name, value in (("a", a), ("b", b), ("thickness", thickness), ("young", young)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("poisson must lie in (-1, 0.5)")
    ratio = b / a
    candidates = [half_waves] if half_waves else list(range(1, 41))
    m, k = min(((m, (m * ratio + 1.0 / (m * ratio)) ** 2) for m in candidates),
               key=lambda item: item[1])
    rigidity = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    sigma = k * math.pi ** 2 * rigidity / (b ** 2 * thickness)
    return {"half_waves": int(m), "buckling_coefficient": float(k),
            "sigma_cr": float(sigma), "half_wavelength": float(a / m),
            "aspect_ratio": float(a / b), "width_over_thickness": float(b / thickness)}


# --------------------------------------------------------------------------
# public analyses
# --------------------------------------------------------------------------
def bare_plate_buckling(a: float, b: float, thickness: float, young: float, poisson: float,
                        nx: int, ny: int, *, modes: int = 3) -> BucklingResult:
    """Simply supported plate through the same assembly/condensation/eigen path."""
    mesh = plate_cell_mesh(a, b, thickness, nx, ny)
    return assembly_buckling(mesh, young, poisson, modes=modes,
                             label=f"plate {nx}x{ny}")


def panel_buckling(panel: StiffenedPanel, nx: int, ny: int, *, modes: int = 3) -> BucklingResult:
    """Linearised bifurcation of the stiffened-panel unit cell under compression."""
    mesh = panel_cell_mesh(panel, nx, ny)
    return assembly_buckling(mesh, panel.young, panel.poisson, modes=modes,
                             label=f"panel {nx}x{ny}")