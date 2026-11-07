#!/usr/bin/env python3
"""IN36-N01 anchor 1: cantilever bending vs Euler--Bernoulli, with a GCI ladder.

Physics
-------
A straight prismatic cantilever, length ``L = 10 m``, width ``b = 1 m``, depth
``h = 0.1 m`` (``L/h = 100``), steel ``E = 210 GPa``, ``nu = 0.3``, carries a
transverse tip load ``F = 10 kN``.  Euler--Bernoulli beam theory gives

    w_tip = F L^3 / (3 E I),      I = b h^3 / 12,
    theta_tip = F L^2 / (2 E I),
    sigma_xx(x, z=h/2) = 6 F (L - x) / (b h^2).

At ``L/h = 100`` the shear contribution ``F L / (kappa G A)`` is ``7.8e-5`` of
the bending term (0.008 %), so the classical solution is the correct yardstick
to two orders better than the 1 % gate; that residual is reported instead of
hidden.

Models (two independent discretisations of the same solid)
----------------------------------------------------------
1. **Q8 plane-stress** membrane of the ``(x, z)`` section, thickness ``b``.
   Plane stress is the 2-D idealisation whose exact limit *is* Euler--Bernoulli
   (uniaxial bending stress, ``sigma_yy = 0``); the 8-node serendipity element
   with 3x3 Gauss integration is ``O(h^2)`` in stress.  The axial mesh is
   Chebyshev-clustered because the fully clamped root and the tip point load
   are both boundary layers; clustering keeps the observed order in the
   asymptotic range instead of measuring the corner singularity.
2. **HEX20 3-D solid** of the same bar (uniform mesh) as a model-form
   cross-check: it exercises the three-dimensional kernel and shows how far the
   solid model is from beam theory at this slenderness.  It is graded at 2 %
   (not 1 %) because a full-integration quadratic brick on a ``L/h = 100``
   member is shear-locking limited - reported, not hidden.

Both systems are solved with a sparse direct factorisation (SuperLU through
``scipy.sparse.linalg``); the element kernels are TensorFEM's
(``tensorfem.quadratic_plane.q8_stiffness`` and ``tensorfem.hex20``), which is
what this anchor certifies.

Outputs (this directory): ``result.json``, ``gates_verified.json``,
``cantilever_field.png``, ``cantilever_convergence.png``.

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_cantilever/run.py
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from _harness import (  # noqa: E402  (path set-up above)
    Gate,
    gate_manifest,
    pyplot,
    richardson,
    utc_now,
    write_json,
)

from tensorfem.continuum import elasticity_matrix  # noqa: E402
from tensorfem.hex20 import hex20_stiffness, structured_hex20_mesh  # noqa: E402
from tensorfem.quadratic_plane import (  # noqa: E402
    GAUSS_POINTS_3X3,
    q8_element_dofs,
    q8_recovered_nodal_stress,
    q8_shape,
    q8_stiffness,
)
from tensorfem.solid3d import elasticity_matrix_3d  # noqa: E402

DTYPE = torch.float64

# --- declared benchmark parameters (campaign IN36-N01) ----------------------
YOUNG = 210.0e9
POISSON = 0.3
LENGTH = 10.0
WIDTH = 1.0
DEPTH = 0.1
LOAD = 10.0e3  # transverse tip load, applied downwards in z

# Q8 ladder: (axial elements, depth elements); ratio 2 per level.
Q8_LEVELS = ((20, 2), (40, 4), (80, 8), (160, 16))
# HEX20 ladder: (axial, width, depth) elements.
HEX20_LEVELS = ((16, 2, 2), (32, 4, 4), (64, 4, 4))

CRITERIA = {
    "tip_deflection_rel_error_max": 0.01,
    "gci_fine_max": 0.03,
    "observed_order_min": 1.8,
    "midspan_bending_stress_rel_error_max": 0.02,
    "solid3d_tip_deflection_rel_error_max": 0.02,
    "refinement_levels_min": 3,
}


def analytic() -> dict:
    """Closed-form Euler--Bernoulli cantilever response (the oracle)."""
    inertia = WIDTH * DEPTH**3 / 12.0
    w_tip = LOAD * LENGTH**3 / (3.0 * YOUNG * inertia)
    theta_tip = LOAD * LENGTH**2 / (2.0 * YOUNG * inertia)
    shear = LOAD * LENGTH / ((5.0 / 6.0) * (YOUNG / (2.0 * (1.0 + POISSON))) * WIDTH * DEPTH)
    return {
        "formula_deflection": "w = F L^3 / (3 E I),  I = b h^3 / 12",
        "formula_rotation": "theta = F L^2 / (2 E I)",
        "formula_stress": "sigma_xx(x,z) = M(x)(z - h/2)/I,  M(x) = F (L - x); at z = h: 6 F (L - x)/(b h^2)",
        "inertia": inertia,
        "slenderness_L_over_h": LENGTH / DEPTH,
        "slenderness_L_over_b": LENGTH / WIDTH,
        "tip_deflection": w_tip,
        "tip_rotation": theta_tip,
        "root_bending_stress": 6.0 * LOAD * LENGTH / (WIDTH * DEPTH**2),
        "shear_deflection": shear,
        "shear_deflection_over_bending": shear / w_tip,
    }


def q8_mesh(nx: int, nz: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Q8 serendipity mesh with a Chebyshev-clustered axial distribution.

    Clustering is applied along ``x`` only: the clamped root and the tip load
    are boundary layers in ``x``, while the through-thickness bending field is a
    smooth cubic that an even mesh resolves optimally.
    """
    ax = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, 2 * nx + 1)))
    az = np.linspace(0.0, 1.0, 2 * nz + 1)
    xs, zs = LENGTH * ax, DEPTH * az
    index: dict[tuple[int, int], int] = {}
    coords: list[tuple[float, float]] = []

    def node(i: int, j: int) -> int:
        key = (i, j)
        if key not in index:
            index[key] = len(coords)
            coords.append((float(xs[i]), float(zs[j])))
        return index[key]

    cells: list[list[int]] = []
    for j in range(nz):
        for i in range(nx):
            a, c = 2 * i, 2 * j
            cells.append(
                [
                    node(a, c),
                    node(a + 2, c),
                    node(a + 2, c + 2),
                    node(a, c + 2),
                    node(a + 1, c),
                    node(a + 2, c + 1),
                    node(a + 1, c + 2),
                    node(a, c + 1),
                ]
            )
    return torch.tensor(coords, dtype=DTYPE), torch.tensor(cells, dtype=torch.long)


def _assemble(ke: torch.Tensor, dofs: torch.Tensor, ndof: int) -> sp.csr_matrix:
    nn = ke.shape[1]
    rows = dofs[:, :, None].expand(-1, -1, nn).reshape(-1).numpy()
    cols = dofs[:, None, :].expand(-1, nn, -1).reshape(-1).numpy()
    return sp.coo_matrix((ke.reshape(-1).numpy(), (rows, cols)), shape=(ndof, ndof)).tocsr()


def solve_q8_cantilever(nx: int, nz: int, *, recover_field: bool = False) -> dict:
    """Solve the Q8 plane-stress cantilever and return fields plus diagnostics.

    ``recover_field`` also returns the Zienkiewicz--Zhu nodal stress field, which
    is only needed for the figure and therefore only computed on the finest
    level.
    """
    nodes, elements = q8_mesh(nx, nz)
    dmat = elasticity_matrix(
        torch.tensor([YOUNG], dtype=DTYPE), torch.tensor([POISSON], dtype=DTYPE), "stress"
    )[0]
    coordinates = nodes[elements]
    stiffness, gauss_matrices = q8_stiffness(coordinates, dmat, torch.tensor(WIDTH, dtype=DTYPE))
    dofs = q8_element_dofs(elements)
    ndof = 2 * len(nodes)
    K = _assemble(stiffness, dofs, ndof)

    force = np.zeros(ndof)
    tip = torch.where(nodes[:, 0] == LENGTH)[0]
    force[2 * tip.numpy() + 1] = -LOAD / len(tip)
    root = torch.where(nodes[:, 0] == 0.0)[0]
    fixed = np.concatenate([2 * root.numpy(), 2 * root.numpy() + 1])
    free = np.setdiff1d(np.arange(ndof), fixed)

    start = time.perf_counter()
    displacement = np.zeros(ndof)
    displacement[free] = spla.spsolve(K[free][:, free].tocsc(), force[free])
    seconds = time.perf_counter() - start

    u = torch.tensor(displacement, dtype=DTYPE)
    tip_deflection = float(-u[2 * tip + 1].mean())
    reaction = K @ displacement - force
    support_reaction = float(reaction[2 * root.numpy() + 1].sum())

    # Bending-stress probe: top-fibre Gauss point of the element nearest mid-span.
    local = torch.zeros((len(elements), 16), dtype=DTYPE)
    local[:, 0::2] = u[2 * elements]
    local[:, 1::2] = u[2 * elements + 1]
    strain = torch.einsum("ekij,ej->eki", gauss_matrices, local)  # (ne, 9, 3)
    gauss = torch.einsum("ij,ekj->eki", dmat, strain)
    centres = coordinates.mean(dim=1)
    top_row = centres[:, 1].max() - 1e-12  # topmost element row
    upper = torch.where(centres[:, 1] >= top_row)[0]
    probe = int(upper[torch.argmin(torch.abs(centres[upper, 0] - LENGTH / 2.0))])
    eta = math.sqrt(0.6)  # outermost ordinate of the 3x3 Gauss rule
    gauss_index = GAUSS_POINTS_3X3.index((0.0, eta))
    shape = q8_shape(0.0, eta)
    x_probe = float(shape @ coordinates[probe][:, 0])
    z_probe = float(shape @ coordinates[probe][:, 1])
    sigma_fe = float(gauss[probe, gauss_index, 0])
    # Plane bending: sigma_xx(x, z) = M(x) (z - h/2)/I with M(x) = F (L - x).
    sigma_ref = LOAD * (LENGTH - x_probe) * (z_probe - DEPTH / 2.0) * 12.0 / (WIDTH * DEPTH**3)
    nodal = q8_recovered_nodal_stress(nodes, elements, gauss) if recover_field else None

    return {
        "axial_elements": nx,
        "depth_elements": nz,
        "nodes": len(nodes),
        "dofs": ndof,
        "tip_deflection": tip_deflection,
        "relative_error": abs(tip_deflection / analytic()["tip_deflection"] - 1.0),
        "support_reaction_z": support_reaction,
        "reaction_equilibrium_error": abs(abs(support_reaction) / LOAD - 1.0),
        "stress_probe_x": x_probe,
        "stress_probe_z": z_probe,
        "stress_probe_sigma_xx": sigma_fe,
        "stress_probe_reference": sigma_ref,
        "stress_relative_error": abs(sigma_fe / sigma_ref - 1.0),
        "solve_seconds": seconds,
        "_field": (
            {
                "nodes": nodes,
                "elements": elements,
                "displacement": u,
                "sigma_xx": nodal[:, 0],
            }
            if recover_field
            else None
        ),
    }


def solve_hex20_cantilever(nx: int, ny: int, nz: int) -> dict:
    """HEX20 3-D solid cantilever of the same bar (uniform mesh)."""
    nodes, cells = structured_hex20_mesh(LENGTH, WIDTH, DEPTH, nx, ny, nz, dtype=DTYPE)
    ne = len(cells)
    dmat = elasticity_matrix_3d(
        torch.tensor(YOUNG, dtype=DTYPE).expand(ne),
        torch.tensor(POISSON, dtype=DTYPE).expand(ne),
    )
    start = time.perf_counter()
    ke, _ = hex20_stiffness(nodes[cells], dmat)
    ed = torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(ne, 60)
    ndof = 3 * len(nodes)
    K = _assemble(ke, ed, ndof)
    force = np.zeros(ndof)
    tip = torch.where(nodes[:, 0] == LENGTH)[0]
    force[3 * tip.numpy() + 2] = -LOAD / len(tip)
    root = torch.where(nodes[:, 0] == 0.0)[0]
    fixed = np.concatenate([3 * root.numpy() + i for i in range(3)])
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), force[free])
    seconds = time.perf_counter() - start
    tip_deflection = float(-u[3 * tip.numpy() + 2].mean())
    return {
        "axial_elements": nx,
        "width_elements": ny,
        "depth_elements": nz,
        "nodes": len(nodes),
        "dofs": ndof,
        "tip_deflection": tip_deflection,
        "relative_error": abs(tip_deflection / analytic()["tip_deflection"] - 1.0),
        "solve_seconds": seconds,
    }


def geometry_audit(level: dict) -> dict:
    """Check the mesh really is the declared L/b/h prism (no silent scaling)."""
    nodes = level["_field"]["nodes"]
    length = float(nodes[:, 0].max() - nodes[:, 0].min())
    depth = float(nodes[:, 1].max() - nodes[:, 1].min())
    return {
        "declared": {"length": LENGTH, "width": WIDTH, "depth": DEPTH},
        "axial_extent": length,
        "depth_extent": depth,
        "extent_rel_error": max(abs(length / LENGTH - 1.0), abs(depth / DEPTH - 1.0)),
        "section_second_moment": WIDTH * DEPTH**3 / 12.0,
        "slenderness_L_over_h": length / depth,
    }


def _q8_triangles(elements: np.ndarray) -> np.ndarray:
    """Triangulate each Q8 cell through its centre for plotting."""
    cells = [
        [
            [c[0], c[4], c[7]],
            [c[4], c[1], c[5]],
            [c[5], c[2], c[6]],
            [c[7], c[6], c[3]],
            [c[4], c[5], c[6]],
            [c[4], c[6], c[7]],
        ]
        for c in elements
    ]
    return np.asarray(cells, dtype=np.int64).reshape(-1, 3)


def figures(levels: list[dict], hex_levels: list[dict], order: dict) -> list[Path]:
    plt = pyplot()
    out: list[Path] = []

    finest = levels[-1]
    field = finest["_field"]
    nodes = field["nodes"].numpy()
    elements = field["elements"].numpy()
    u = field["displacement"].numpy()
    sigma = field["sigma_xx"].numpy()
    scale = 0.03 * LENGTH / max(float(np.abs(u).max()), 1e-30)
    deformed = nodes.copy()
    deformed[:, 0] += scale * u[0::2]
    deformed[:, 1] += scale * u[1::2]
    triangles = _q8_triangles(elements)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6))
    ax = axes[0]
    contour = ax.tricontourf(
        deformed[:, 0], deformed[:, 1], triangles, sigma / 1e6, levels=24, cmap="coolwarm"
    )
    ax.set_aspect("equal")
    ax.set_ylabel("z [m]  (deflection x30)")
    ax.set_title(
        rf"Q8 plane-stress cantilever, {finest['axial_elements']}x{finest['depth_elements']} "
        r"elements, $L/h=100$: recovered $\sigma_{xx}$ [MPa], $F$ = 10 kN"
    )
    fig.colorbar(contour, ax=ax, label=r"$\sigma_{xx}$ [MPa]")

    ax = axes[1]
    x = np.linspace(0.0, LENGTH, 400)
    sigma_analytic = 6.0 * LOAD * (LENGTH - x) / (WIDTH * DEPTH**2) / 1e6
    ax.plot(x, sigma_analytic, "k-", lw=1.6, label=r"Euler--Bernoulli $6F(L-x)/(b h^2)$")
    top = np.isclose(nodes[:, 1], DEPTH) | np.isclose(nodes[:, 1], 0.0)
    ax.plot(
        nodes[top, 0],
        sigma[top] / 1e6,
        "o",
        ms=2.4,
        alpha=0.55,
        color="tab:red",
        label="recovered nodal stress, top/bottom fibres",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\sigma_{xx}$ [MPa]")
    ax.set_title("bending stress along the span against the closed form")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = HERE / "cantilever_field.png"
    fig.savefig(path)
    plt.close(fig)
    out.append(path)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    h_q8 = np.array([LENGTH / lv["axial_elements"] for lv in levels])
    e_q8 = np.array([lv["relative_error"] for lv in levels])
    h_hex = np.array([LENGTH / lv["axial_elements"] for lv in hex_levels])
    e_hex = np.array([lv["relative_error"] for lv in hex_levels])
    ax.loglog(
        h_q8, e_q8, "o-", label=f"Q8 plane stress, $p_{{obs}}$ = {order['observed_order']:.2f}"
    )
    ax.loglog(h_hex, e_hex, "s--", label="HEX20 solid (locking-limited)")
    ax.loglog(h_q8, e_q8[-1] * (h_q8 / h_q8[-1]) ** 2.0, "k:", lw=1.0, label=r"slope $h^2$")
    ax.axhline(
        CRITERIA["tip_deflection_rel_error_max"], color="tab:red", lw=1.0, ls="--", label="1 % gate"
    )
    ax.set_xlabel("axial element size h [m]")
    ax.set_ylabel(r"$|w_{FE}/w_{EB} - 1|$")
    ax.set_title("cantilever tip deflection: observed convergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = HERE / "cantilever_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    out.append(path)
    return out


def run() -> dict:
    oracle = analytic()
    levels = [
        solve_q8_cantilever(*pair, recover_field=(pair == Q8_LEVELS[-1])) for pair in Q8_LEVELS
    ]
    hex_levels = [solve_hex20_cantilever(*triple) for triple in HEX20_LEVELS]
    finest = levels[-1]
    order = richardson(
        levels[-3]["tip_deflection"],
        levels[-2]["tip_deflection"],
        levels[-1]["tip_deflection"],
        refinement_ratio=2.0,
    )
    geometry = geometry_audit(finest)
    figs = figures(levels, hex_levels, order)
    for lv in levels:
        lv.pop("_field", None)

    gates = [
        Gate(
            id="geometry_extent",
            quantity="max relative error of the meshed extents L, h",
            reference="declared L = 10 m, b = 1 m, h = 0.1 m",
            value=geometry["extent_rel_error"],
            threshold=1e-12,
            note="geometry is validated before any field is trusted",
        ),
        Gate(
            id="tip_deflection",
            quantity="tip deflection of the finest Q8 level",
            reference="w = F L^3/(3 E I)",
            value=finest["relative_error"],
            threshold=CRITERIA["tip_deflection_rel_error_max"],
        ),
        Gate(
            id="gci_fine",
            quantity="Roache grid-convergence index of the finest triple",
            reference="GCI = 1.25 |(w3-w2)/w3| / (r^p - 1)",
            value=order["gci_fine"],
            threshold=CRITERIA["gci_fine_max"],
        ),
        Gate(
            id="observed_order",
            quantity="observed order of accuracy p from the finest triple",
            reference="p = ln((w3-w2)/(w2-w1))/ln r",
            value=order["observed_order"],
            threshold=CRITERIA["observed_order_min"],
            operator=">=",
            note="Chebyshev-clustered Q8 ladder; p is the empirical value, not a wish",
        ),
        Gate(
            id="midspan_bending_stress",
            quantity="sigma_xx at the top-half Gauss point nearest mid-span",
            reference="sigma_xx = F (L - x)(z - h/2)/I",
            value=finest["stress_relative_error"],
            threshold=CRITERIA["midspan_bending_stress_rel_error_max"],
        ),
        Gate(
            id="reaction_equilibrium",
            quantity="|sum(R_z) - F|/F at the clamped root",
            reference="global equilibrium of the discrete system",
            value=finest["reaction_equilibrium_error"],
            threshold=1e-5,
            note="audit of the linear solve, not of the physics; the 1e-5 band reflects "
            "the conditioning of a 100:1 slender system, not a modelling tolerance",
        ),
        Gate(
            id="solid3d_tip_deflection",
            quantity="tip deflection of the finest HEX20 3-D level",
            reference="w = F L^3/(3 E I)",
            value=hex_levels[-1]["relative_error"],
            threshold=CRITERIA["solid3d_tip_deflection_rel_error_max"],
            note="model-form cross-check; locking-limited, see limitations",
        ),
    ]
    manifest = gate_manifest(
        anchor="fem_cantilever",
        title="Cantilever bending: Euler--Bernoulli deflection, stress and GCI ladder",
        criteria=CRITERIA,
        gates=gates,
        artifacts={
            "result": "result.json",
            "gates": "gates_verified.json",
            "figures": [p.name for p in figs],
        },
        limitations=[
            "plane stress is a 2-D idealisation; plane strain or the anticlastic 3-D "
            "state shifts the deflection by O(nu (h/L)^2), i.e. far below the gate",
            "the fully clamped root is a boundary layer; Chebyshev clustering is what "
            "keeps the ladder inside the asymptotic range",
            "HEX20 full integration is shear-locking limited at L/h = 100, so the 3-D "
            "cross-check is graded at 2 % and its observed order is reported but not gated",
            "small strain, isotropic, linear elasticity only: no prestress, no contact, "
            "no follower load, no geometric nonlinearity",
        ],
    )
    result = {
        "benchmark": "fem_cantilever",
        "mission": "IN36-N01",
        "generated_at": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "dtype": "torch.float64",
            "linear_solver": "scipy.sparse.linalg.spsolve (SuperLU, sparse direct)",
        },
        "geometry": {
            "young_modulus": YOUNG,
            "poisson_ratio": POISSON,
            "length": LENGTH,
            "width": WIDTH,
            "depth": DEPTH,
            "tip_load": LOAD,
            "load_direction": "-z",
        },
        "analytic": oracle,
        "geometry_audit": geometry,
        "q8_levels": levels,
        "q8_richardson": order,
        "hex20_levels": hex_levels,
        "verdict": {
            "finest_relative_error": finest["relative_error"],
            "gate_passed": manifest["passed"],
            "failed_gates": manifest["failed_gates"],
        },
        "artifacts": manifest["artifacts"],
    }
    write_json(HERE / "result.json", result)
    write_json(HERE / "gates_verified.json", manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="cantilever bending anchor")
    parser.parse_args()
    result = run()
    verdict = result["verdict"]
    print(
        json.dumps(
            {
                "passed": verdict["gate_passed"],
                "failed_gates": verdict["failed_gates"],
                "tip_deflection_rel_error": verdict["finest_relative_error"],
                "observed_order": result["q8_richardson"]["observed_order"],
                "gci_fine": result["q8_richardson"]["gci_fine"],
            },
            indent=1,
        )
    )
    return 0 if verdict["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
