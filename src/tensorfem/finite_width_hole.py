"""Finite-width plate with a central circular hole: Howland / Heywood verification.

Why this module exists
----------------------
``docs/kirsch-hole-stress-concentration.md`` verifies the production plane-stress
path against the *infinite-plate* Kirsch solution and records the remaining debt
explicitly: "正交异性板、含裂纹孔、**有限宽板（Howland 修正）**都未覆盖，
``K_t = 3`` 只对无限大板成立".  Ship structures are never infinite: hatch
corners, manholes and pipe penetrations sit in plate fields a few hole diameters
wide, where the gross-section stress-concentration factor is well below 3
(2.16 at ``d/W = 0.5``).  Applying ``K_t = 3`` there over-predicts the peak
stress by 39 %, so the finite-width branch needs its own reference.

Reference hierarchy
-------------------
1. **Exact limit (primary anchor).**  With the hole held at a fixed distance from
   the free edges and ``d/W -> 0``, the solution must collapse onto Kirsch
   (1898), ``K_tg -> 3``.  That limit is verified to 0.10 % by the Kirsch unit on
   the *same* production path, so it anchors the pipeline against an exact
   solution rather than against another discretisation.
2. **Statically exact check.**  Equilibrium fixes the axial force that any
   cross-section must carry, ``int sigma_xx dA = T W t``; the reaction on the
   symmetry plane is compared with the applied load on every mesh.
3. **Literature anchor.**  Howland's series solution (1930) for a strip of
   finite width, as tabulated in Peterson's *Stress Concentration Factors*
   (Pilkey & Pilkey, 3rd ed., Chart 4.1), plus Heywood's engineering fit
   ``K_tn = 2 + (1 - d/W)**3`` (Heywood 1952).

**Net-section vs gross-section convention (a real trap, measured here).**  The
tabulated finite-width factors are *net-section* (ligament) factors
``K_tn = sigma_peak / sigma_net`` with ``sigma_net = T W / (W - d)``; they decay
from Kirsch's 3 toward ~2 as the hole grows.  Dividing the FE peak by the
applied traction ``T`` instead returns the *gross-section* factor
``K_tg = K_tn (1 - d/W)``, which differs by up to a factor two at ``d/W = 0.5``.
Both are reported; only the net-section pair is compared with the literature.

The quarter model (``0 <= x <= L``, ``0 <= y <= H``) carries symmetry on
``y = 0`` (``u_y = 0``) and ``x = 0`` (``u_x = 0``), a traction-free hole and
lateral edge, and uniform tension ``T`` on ``x = L``.  ``L`` is chosen long
enough for Saint-Venant insensitivity, which is measured, not assumed.

The observed quantity is the axial stress at the hole edge on the symmetry line
``(0, a)``, where the peak sits by symmetry; it is recovered from the elements
touching that node.  ``K_tg = sigma_xx / T``.  Everything is double precision on
the *production* path (:class:`tensorfem.continuum.ContinuumModel` +
:func:`solve_continuum`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .continuum import ContinuumModel, solve_continuum
from .kirsch_hole import _q4_b

#: Net-section (ligament) stress-concentration factors ``K_tn = sigma_peak/sigma_net``
#: for a plate of finite width ``W`` with a central hole of diameter ``d`` under
#: uniaxial tension: Howland (1930) series solution as tabulated in R.E. Peterson,
#: *Stress Concentration Factors* (Pilkey & Pilkey, 3rd ed., Chart 4.1).
#: Chart read-off, quoted to +-1 %.
PETERSON_HOWLAND_KTN: dict = {
    0.10: 2.73,
    0.20: 2.51,
    0.30: 2.35,
    0.40: 2.24,
    0.50: 2.16,
}


def heywood_stress_concentration(ratio: float) -> float:
    """Heywood (1952) net-section fit ``K_tn = 2 + (1 - d/W)**3``.

    Returns 3 for ``d/W -> 0`` (Kirsch) and reproduces Peterson's Howland table
    to better than 2 % over ``0.1 <= d/W <= 0.5``.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("d/W must lie in [0, 1)")
    return 2.0 + (1.0 - ratio) ** 3


def peterson_stress_concentration(ratio: float) -> float | None:
    """Tabulated Howland ``K_tn`` when ``d/W`` is a tabulated ratio, else ``None``."""
    for tabulated, value in PETERSON_HOWLAND_KTN.items():
        if abs(ratio - tabulated) < 1e-12:
            return value
    return None


@dataclass(frozen=True)
class FiniteWidthPlate:
    """Quarter-plate idealisation of a finite-width plate with a central hole."""

    width: float = 1.0
    diameter: float = 0.2
    half_length: float = 1.5
    young: float = 200.0e9
    poisson: float = 0.3
    thickness: float = 0.01
    tension: float = 1.0e6

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.diameter <= 0.0:
            raise ValueError("width and diameter must be positive")
        if self.diameter >= self.width:
            raise ValueError("hole diameter must be smaller than the plate width")
        if self.half_length <= self.diameter:
            raise ValueError("half length must exceed the hole diameter")

    @property
    def hole_radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def half_width(self) -> float:
        return 0.5 * self.width

    @property
    def ratio(self) -> float:
        """``d/W``."""
        return self.diameter / self.width

    @property
    def net_section_stress(self) -> float:
        """Exact average axial stress on a cross-section through the hole."""
        return self.tension * self.width / (self.width - self.diameter)

    @property
    def heywood_ktn(self) -> float:
        """Heywood net-section fit at this ``d/W``."""
        return heywood_stress_concentration(self.ratio)

    @property
    def peterson_ktn(self) -> float | None:
        """Tabulated Howland net-section factor at this ``d/W`` (``None`` off-table)."""
        return peterson_stress_concentration(self.ratio)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def starburst_mesh(
    plate: FiniteWidthPlate,
    n_boundary: int = 24,
    n_radial: int = 24,
    grading: float = 6.0,
    *,
    dtype=torch.float64,
):
    """Graded quarter-plate Q4 mesh: quarter hole at the origin, rays to the edges.

    Outer nodes are distributed by **arc length** along ``x = L`` and then
    ``y = H``, so the corner ``(L, H)`` -- where the loaded end meets the free
    edge and the traction boundary condition jumps -- is always a mesh node.
    Rays run from the hole to the outer boundary, so element density is highest
    at the hole where the gradient is largest.
    """
    if n_boundary < 4 or n_radial < 1:
        raise ValueError("n_boundary >= 4 and n_radial >= 1 are required")
    a = plate.hole_radius
    half_width = plate.half_width
    length = plate.half_length
    perimeter = length + half_width
    n_loaded = max(1, int(round(n_boundary * half_width / perimeter)))
    n_free = max(1, n_boundary - n_loaded)
    boundary: list = []
    for k in range(n_loaded):
        boundary.append((length, half_width * (k / n_loaded), "loaded"))
    for k in range(n_free + 1):
        kind = "corner" if k == 0 else "free"
        boundary.append((length - length * (k / n_free), half_width, kind))
    thetas = np.array([math.atan2(y, x) for x, y, _ in boundary])
    radii = np.array([math.hypot(x, y) for x, y, _ in boundary])

    s = np.linspace(0.0, 1.0, n_radial + 1)
    if grading > 0.0:
        s = (np.exp(grading * s) - 1.0) / (math.exp(grading) - 1.0)
    ring = a + (radii[:, None] - a) * s[None, :]  # [n_rays, n_radial+1]
    nodes = np.stack(
        [ring * np.cos(thetas)[:, None], ring * np.sin(thetas)[:, None]], axis=-1
    ).reshape(-1, 2)

    n_rays = len(boundary)
    n_columns = n_radial + 1

    def index(i: int, j: int) -> int:
        return i * n_columns + j

    elements = np.empty(((n_rays - 1) * n_radial, 4), dtype=np.int64)
    for i in range(n_rays - 1):
        for j in range(n_radial):
            elements[i * n_radial + j] = (
                index(i, j), index(i, j + 1), index(i + 1, j + 1), index(i + 1, j)
            )

    meta = {
        "nodes": np.asarray(nodes),
        "elements": elements,
        "thetas": thetas,
        "radii": radii,
        "boundary_kind": [kind for _, _, kind in boundary],
        "n_boundary": n_boundary,
        "n_radial": n_radial,
        "grading": grading,
        "n_rays": n_rays,
        "n_columns": n_columns,
        "loaded_segments": [(i, i + 1) for i in range(n_rays - 1)
                            if boundary[i][2] == "loaded"
                            and boundary[i + 1][2] in ("loaded", "corner")],
        "hole_nodes": [index(i, 0) for i in range(n_rays)],
        "symmetry_y0": [index(0, j) for j in range(n_columns)],
        "symmetry_x0": [index(n_rays - 1, j) for j in range(n_columns)],
        "loaded_nodes": [index(i, n_radial) for i in range(n_rays)
                         if boundary[i][2] in ("loaded", "corner")],
    }
    return (
        torch.as_tensor(nodes, dtype=dtype),
        torch.as_tensor(elements, dtype=torch.long),
        meta,
    )


def edge_loads(plate: FiniteWidthPlate, meta: dict, *, dtype=torch.float64) -> torch.Tensor:
    """Consistent nodal loads for the uniform tension on the loaded end ``x = L``.

    Two-point Gauss on every boundary segment that lies on ``x = L``; the lateral
    edge ``y = H`` is traction free and receives nothing.
    """
    coordinates = np.asarray(meta["nodes"], dtype=np.float64)
    n_columns = meta["n_columns"]
    n_radial = meta["n_radial"]
    forces = np.zeros(2 * coordinates.shape[0], dtype=np.float64)
    gauss = ((-1.0 / math.sqrt(3.0), 1.0), (1.0 / math.sqrt(3.0), 1.0))
    for i, _ in meta["loaded_segments"]:
        first = i * n_columns + n_radial
        second = (i + 1) * n_columns + n_radial
        p1, p2 = coordinates[first], coordinates[second]
        if abs(p1[0] - plate.half_length) > 1e-9 or abs(p2[0] - plate.half_length) > 1e-9:
            raise ValueError("loaded segment is not on x = L")
        length = float(p2[1] - p1[1])
        for xi, weight in gauss:
            n1, n2 = 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)
            jacobian = 0.5 * length * weight
            forces[2 * first] += n1 * plate.tension * plate.thickness * jacobian
            forces[2 * second] += n2 * plate.tension * plate.thickness * jacobian
    return torch.as_tensor(forces, dtype=dtype)


def symmetry_dofs(meta: dict) -> list:
    """Zero-displacement dofs: ``u_y = 0`` on ``y = 0`` and ``u_x = 0`` on ``x = 0``."""
    fixed = [2 * node + 1 for node in meta["symmetry_y0"]]
    fixed += [2 * node for node in meta["symmetry_x0"]]
    return sorted(set(fixed))


def plate_model(plate: FiniteWidthPlate, meta: dict, *, dtype=torch.float64) -> ContinuumModel:
    """Production :class:`ContinuumModel` for the quarter finite-width plate."""
    nodes = meta["nodes"]
    elements = meta["elements"]
    return ContinuumModel(
        nodes=torch.as_tensor(nodes, dtype=dtype) if not torch.is_tensor(nodes) else nodes,
        elements=torch.as_tensor(elements, dtype=torch.long) if not torch.is_tensor(elements) else elements,
        young_modulus=torch.as_tensor(plate.young, dtype=dtype),
        poisson_ratio=torch.as_tensor(plate.poisson, dtype=dtype),
        thickness=torch.as_tensor(plate.thickness, dtype=dtype),
        forces=edge_loads(plate, meta, dtype=dtype),
        fixed_dofs=torch.as_tensor(symmetry_dofs(meta), dtype=torch.long),
        plane="stress",
        element_type="q4",
    )


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _constitutive(plate: FiniteWidthPlate) -> np.ndarray:
    e, nu = plate.young, plate.poisson
    return (e / (1.0 - nu * nu)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
    )


def hole_edge_stress(plate: FiniteWidthPlate, meta: dict, displacement) -> dict:
    """Axial stress along the hole edge by element-extrapolated nodal averaging.

    The peak of ``sigma_xx`` sits on the symmetry line ``(0, a)``; the whole edge
    profile is returned so the peak location is checked rather than assumed.
    """
    nodes = np.asarray(meta["nodes"], dtype=np.float64)
    elements = np.asarray(meta["elements"])
    u = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    dmat = _constitutive(plate)
    hole_nodes = set(meta["hole_nodes"])
    accumulate = {node: np.zeros(3) for node in hole_nodes}
    count = {node: 0 for node in hole_nodes}
    corner_parent = {0: (-1.0, -1.0), 1: (1.0, -1.0), 2: (1.0, 1.0), 3: (-1.0, 1.0)}
    for element in range(elements.shape[0]):
        connectivity = elements[element]
        hits = [local for local, node in enumerate(connectivity) if node in accumulate]
        if not hits:
            continue
        local_u = np.empty(8)
        local_u[0::2] = u[connectivity, 0]
        local_u[1::2] = u[connectivity, 1]
        coordinates = nodes[connectivity]
        for local in hits:
            xi, eta = corner_parent[local]
            b, _ = _q4_b(xi, eta, coordinates)
            accumulate[connectivity[local]] += dmat @ b @ local_u
            count[connectivity[local]] += 1
    axial: dict = {}
    hoop: dict = {}
    for index, node in enumerate(meta["hole_nodes"]):
        stress = accumulate[node] / max(count[node], 1)
        theta = float(meta["thetas"][index])
        key = round(theta, 12)
        c, s = math.cos(theta), math.sin(theta)
        axial[key] = float(stress[0])
        hoop[key] = float(stress[0] * s * s + stress[1] * c * c - 2.0 * stress[2] * s * c)
    peak_theta = max(hoop, key=lambda key: hoop[key])
    apex_key = round(math.pi / 2.0, 12)
    return {
        "peak_hoop": hoop[peak_theta],
        "peak_theta_degrees": math.degrees(peak_theta),
        "hoop_at_apex": hoop[apex_key],
        "sigma_xx_at_apex": axial[apex_key],
        "axial_at_waist": axial.get(round(0.0, 12), float("nan")),
        "hoop_at_waist": hoop.get(round(0.0, 12), float("nan")),
        "axial_profile": [[math.degrees(t), v] for t, v in sorted(axial.items())],
        "hoop_profile": [[math.degrees(t), v] for t, v in sorted(hoop.items())],
    }


def load_balance(plate: FiniteWidthPlate, meta: dict, reaction) -> dict:
    """Applied force vs the reaction on the symmetry plane ``x = 0`` (exact by statics)."""
    r = np.asarray(reaction, dtype=np.float64).reshape(-1, 2)
    applied = plate.tension * plate.thickness * plate.half_width
    reacted = -float(sum(r[node][0] for node in meta["symmetry_x0"]))
    return {
        "applied": applied,
        "reacted": reacted,
        "relative_error": abs(reacted - applied) / applied,
    }


def solve_case(
    plate: FiniteWidthPlate | None = None,
    n_boundary: int = 24,
    n_radial: int | None = None,
    grading: float = 6.0,
) -> dict:
    """Solve one mesh with the production path and measure the stress concentration."""
    plate = plate or FiniteWidthPlate()
    n_radial = n_boundary if n_radial is None else n_radial
    _, _, meta = starburst_mesh(plate, n_boundary, n_radial, grading)
    model = plate_model(plate, meta)
    result = solve_continuum(model)
    displacement = result.displacement.detach().cpu().numpy()
    edge = hole_edge_stress(plate, meta, displacement)
    balance = load_balance(plate, meta, result.reaction.detach().cpu().numpy())
    gross_apex = edge["hoop_at_apex"] / plate.tension
    gross_peak = edge["peak_hoop"] / plate.tension
    ktn_apex = gross_apex * (1.0 - plate.ratio)   # sigma_peak / sigma_net
    ktn_peak = gross_peak * (1.0 - plate.ratio)
    reference = plate.heywood_ktn
    table = plate.peterson_ktn
    return {
        "ratio": plate.ratio,
        "half_length_over_width": plate.half_length / plate.width,
        "n_boundary": n_boundary,
        "n_radial": n_radial,
        "grading": grading,
        "elements": int(meta["elements"].shape[0]),
        "dofs": int(2 * meta["nodes"].shape[0]),
        "ktn_fe": ktn_apex,
        "ktn_fe_profile_peak": ktn_peak,
        "ktg_fe": gross_apex,
        "peak_theta_degrees": edge["peak_theta_degrees"],
        "ktn_heywood": reference,
        "ktn_peterson": table,
        "error_vs_heywood": abs(ktn_apex - reference) / reference,
        "error_vs_peterson": (None if table is None else abs(ktn_apex - table) / table),
        "load_balance": balance,
        "hole_edge_hoop_profile_over_tension": [[theta, value / plate.tension]
                                                for theta, value in edge["hoop_profile"]],
        "hole_edge_axial_profile_over_tension": [[theta, value / plate.tension]
                                                 for theta, value in edge["axial_profile"]],
        "hoop_at_waist_over_tension": edge["hoop_at_waist"] / plate.tension,
        "axial_at_waist_over_tension": edge["axial_at_waist"] / plate.tension,
        "stress_concentration": {"gross": gross_apex, "net": ktn_apex},
        "net_section_average_over_tension": 1.0 / (1.0 - plate.ratio),
        "_displacement": displacement,
        "_meta": meta,
        "_stress": result.stress.detach().cpu().numpy(),
    }