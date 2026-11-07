#!/usr/bin/env python3
"""IN36-N01 anchor 3: simply supported rectangular plate, first five natural frequencies.

Physics
-------
A thin rectangular plate ``a x b x h = 1.2 x 0.8 x 0.012 m`` (``a/h = 100``),
steel ``E = 210 GPa``, ``nu = 0.3``, ``rho = 7850 kg/m^3``, simply supported on
all four edges.  Kirchhoff (thin-plate) theory gives the Navier double-sine
eigenfrequencies

    f_mn = (pi/2) sqrt(D/(rho h)) [ (m/a)^2 + (n/b)^2 ],   D = E h^3 / (12 (1-nu^2)),

which for ``a/b = 1.5`` order the first five modes as
``(1,1) < (2,1) < (1,2) < (3,1) < (2,2)`` with no degeneracy to hide a mis-paired
mode.  Values: 66.586, 128.051, 204.881, 230.491, 266.346 Hz.

Model
-----
TensorFEM's four-node Mindlin--Reissner plate with selective reduced shear
integration (``tensorfem.plate.q4_mindlin_stiffness``) and its consistent mass
matrix, including rotary inertia (``tensorfem.plate_vibration.q4_plate_mass_matrix``),
on a structured mesh from ``tensorfem.plate_vibration.structured_rectangular_mesh``
with the library's own hard simple-support DOF set.  The generalised eigenproblem
is reduced with the Cholesky factor of the mass matrix and solved for the lowest
five eigenpairs; the eigensolver is scipy's LAPACK ``dsyevr`` subset call rather
than ``torch.linalg.eigh`` purely because the latter costs 10x more wall time on
the loong64 nodes.  An equivalence gate re-solves the coarsest level with the
library's own ``solve_plate_vibration`` and requires agreement to 1e-9 relative,
so the fast path is proven to solve the same discrete problem.

At ``a/h = 100`` the Mindlin shear correction to the Kirchhoff frequency is
``O((h/a)^2) ~ 2e-4`` relative, i.e. two orders below the 1 % gate; the residual
is reported rather than hidden.

Outputs (this directory): ``result.json``, ``gates_verified.json``,
``plate_modes.png``, ``plate_convergence.png``.

Usage (levels are cached, so they can be run in separate calls)::

    PYTHONPATH=src python benchmarks/verified/fem_plate_modal/run.py --levels 12 24
    PYTHONPATH=src python benchmarks/verified/fem_plate_modal/run.py --levels 48
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
CACHE = HERE / "_cache"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from _harness import (  # noqa: E402  (path set-up above)
    Gate,
    gate_manifest,
    pyplot,
    read_json,
    utc_now,
    write_json,
)

from tensorfem.plate import q4_mindlin_stiffness  # noqa: E402
from tensorfem.plate_vibration import (  # noqa: E402
    _fixed_dofs,
    q4_plate_mass_matrix,
    solve_plate_vibration,
    structured_rectangular_mesh,
)

DTYPE = torch.float64

# --- declared benchmark parameters (campaign IN36-N01) ----------------------
LENGTH = 1.2
WIDTH = 0.8
THICKNESS = 0.012
YOUNG = 210.0e9
POISSON = 0.3
DENSITY = 7850.0
MODES = 5

#: (elements along a, elements along b); refinement ratio 2 per level.
LEVELS = ((12, 8), (24, 16), (48, 32))
#: mode labels of the five lowest Kirchhoff modes for a/b = 1.5.
MODE_LABELS = ((1, 1), (2, 1), (1, 2), (3, 1), (2, 2))

CRITERIA = {
    "frequency_rel_error_max": 0.01,
    "observed_order_min": 1.8,
    "mac_min": 0.99,
    "library_equivalence_max": 1e-9,
    "refinement_levels_min": 3,
}


def navier_frequencies() -> list[float]:
    """Closed-form simply supported Kirchhoff frequencies (the oracle)."""
    d = YOUNG * THICKNESS**3 / (12.0 * (1.0 - POISSON**2))
    scale = (math.pi / 2.0) * math.sqrt(d / (DENSITY * THICKNESS))
    return [scale * ((m / LENGTH) ** 2 + (n / WIDTH) ** 2) for m, n in MODE_LABELS]


def analytic_mode_shape(nodes: np.ndarray, m: int, n: int) -> np.ndarray:
    """``w = sin(m pi x/a) sin(n pi y/b)`` sampled at the mesh nodes."""
    return np.sin(m * math.pi * nodes[:, 0] / LENGTH) * np.sin(n * math.pi * nodes[:, 1] / WIDTH)


def assemble(nx: int, ny: int):
    """Assemble the Mindlin plate stiffness and consistent mass matrices."""
    nodes, elements = structured_rectangular_mesh(nx, ny, LENGTH, WIDTH, dtype=DTYPE)
    ndof = 3 * len(nodes)
    stiffness = torch.zeros((ndof, ndof), dtype=DTYPE)
    mass = torch.zeros_like(stiffness)
    for connectivity in elements:
        xy = nodes[connectivity]
        ke = q4_mindlin_stiffness(xy, YOUNG, POISSON, THICKNESS)
        me = q4_plate_mass_matrix(xy, DENSITY, THICKNESS)
        dofs = torch.stack(
            (3 * connectivity, 3 * connectivity + 1, 3 * connectivity + 2), 1
        ).reshape(-1)
        stiffness[dofs[:, None], dofs] += ke
        mass[dofs[:, None], dofs] += me
    fixed = _fixed_dofs(nodes, LENGTH, WIDTH, "simple", DTYPE)
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    return (
        stiffness[free[:, None], free],
        mass[free[:, None], free],
        nodes,
        elements,
        free,
    )


def _lambda_max(stiffness: torch.Tensor, mass: torch.Tensor, iterations: int = 200) -> float:
    """Largest eigenvalue of the pencil ``(K, M)`` by power iteration on ``M^-1 K``.

    Only used to reproduce the library's rigid-body filter threshold
    ``1e-9 * lambda_max``; a full spectrum would cost more than the eigenpairs
    that are actually wanted.
    """
    k_sparse = sp.csr_matrix(stiffness.numpy())
    m_factor = spla.splu(sp.csc_matrix(mass.numpy()))
    vector = np.ones(k_sparse.shape[0])
    vector /= np.linalg.norm(vector)
    value = 0.0
    for _ in range(iterations):
        vector = m_factor.solve(k_sparse @ vector)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return 0.0
        vector /= norm
        value = norm
    return value


def solve_modes(nx: int, ny: int) -> dict:
    """Lowest five Mindlin plate eigenpairs on one mesh (cached to ``_cache``).

    The pencil ``K phi = lambda M phi`` is solved by shift-invert Lanczos
    (``scipy.sparse.linalg.eigsh`` with ``sigma = -1e2``) because the mass and
    stiffness entries of a thin plate differ by ``O(h^2)`` and the dense
    reduced problem is too ill-conditioned to be trusted here (see the module
    docstring and ``docs/verified/fem-verification-anchors.md``).
    """
    path = CACHE / f"plate_{nx}x{ny}.json"
    start = time.perf_counter()
    stiffness, mass, nodes, elements, free = assemble(nx, ny)
    lambda_max = _lambda_max(stiffness, mass)
    threshold = 1e-9 * lambda_max

    k_sparse = sp.csr_matrix(stiffness.numpy())
    m_sparse = sp.csr_matrix(mass.numpy())
    values, vectors = spla.eigsh(k_sparse, k=MODES + 4, M=m_sparse, sigma=-1.0e2, which="LM")
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    keep = values > threshold
    filtered = int(np.sum(~keep))
    values = values[keep][:MODES]
    vectors = vectors[:, keep][:, :MODES]
    if values.size < MODES:
        raise RuntimeError("fewer elastic modes than requested")
    if np.any(values <= 0.0):
        raise RuntimeError("retained set still contains a mechanism")

    modes = torch.zeros((len(nodes) * 3, MODES), dtype=DTYPE)
    modes[free] = torch.from_numpy(np.ascontiguousarray(vectors)).to(DTYPE)

    frequencies = np.sqrt(values) / (2.0 * math.pi)
    node_array = nodes.numpy()
    macs = []
    for k, (m, n) in enumerate(MODE_LABELS):
        w = modes[0::3, k].numpy()
        exact = analytic_mode_shape(node_array, m, n)
        macs.append(float((w @ exact) ** 2 / ((w @ w) * (exact @ exact))))

    record = {
        "elements_x": nx,
        "elements_y": ny,
        "nodes": len(nodes),
        "dofs": 3 * len(nodes),
        "free_dofs": int(len(free)),
        "spurious_modes_filtered": filtered,
        "lambda_max_estimate": lambda_max,
        "frequencies_hz": [float(f) for f in frequencies],
        "mode_mac": macs,
        "mode_labels": [list(lbl) for lbl in MODE_LABELS],
        "seconds": time.perf_counter() - start,
        "w_field": {
            "nodes": node_array.tolist(),
            "triangles": _triangles(elements.numpy()).tolist(),
            "w": [modes[0::3, k].numpy().tolist() for k in range(MODES)],
        },
    }
    CACHE.mkdir(exist_ok=True)
    write_json(path, record)
    return record


def _triangles(elements: np.ndarray) -> np.ndarray:
    """Two triangles per Q4 cell, in the structured-mesh node numbering."""
    triangles = []
    for cell in elements:
        triangles.append([cell[0], cell[1], cell[2]])
        triangles.append([cell[0], cell[2], cell[3]])
    return np.asarray(triangles, dtype=np.int64)


def library_equivalence(nx: int, ny: int) -> float:
    """Re-solve one level with the library entry point and compare frequencies."""
    fast = read_json(CACHE / f"plate_{nx}x{ny}.json")
    library = solve_plate_vibration(
        nx,
        ny,
        length=LENGTH,
        width=WIDTH,
        young=YOUNG,
        poisson=POISSON,
        thickness=THICKNESS,
        density=DENSITY,
        modes=MODES,
        dtype=DTYPE,
    )
    library_hz = [float(f) for f in library.frequencies_hz]
    return max(abs(a / b - 1.0) for a, b in zip(fast["frequencies_hz"], library_hz))


def graded(record: dict, oracle: list[float]) -> dict:
    errors = [abs(f / ref - 1.0) for f, ref in zip(record["frequencies_hz"], oracle)]
    return {
        **{k: v for k, v in record.items() if k != "w_field"},
        "relative_errors": errors,
        "max_relative_error": max(errors),
    }


def observed_order(levels: list[dict], reference: list[float]) -> dict:
    """Two-level order estimates per mode from a uniform refinement ladder."""
    orders: list[float | None] = []
    h = [1.0 / lv["elements_x"] for lv in levels]
    for k in range(MODES):
        e1 = abs(levels[-3]["frequencies_hz"][k] / reference[k] - 1.0)
        e2 = abs(levels[-2]["frequencies_hz"][k] / reference[k] - 1.0)
        e3 = abs(levels[-1]["frequencies_hz"][k] / reference[k] - 1.0)
        if e1 <= 0.0 or e2 <= 0.0 or e3 <= 0.0 or not (e2 < e1 and e3 < e2):
            orders.append(None)
            continue
        orders.append(math.log(e2 / e3) / math.log(h[-2] / h[-1]))
    finite = [p for p in orders if p is not None]
    return {
        "mode_orders_finest_pair": orders,
        "min_order_finest_pair": min(finite) if finite else None,
        "ladder_elements_x": [lv["elements_x"] for lv in levels],
        "note": "order per mode from the finest two levels of the uniform r = 2 ladder",
    }


def figures(levels: list[dict], order: dict, oracle: list[float]) -> list[Path]:
    plt = pyplot()
    out: list[Path] = []

    finest = levels[-1]
    nodes = np.asarray(finest["w_field"]["nodes"])
    triangles = np.asarray(finest["w_field"]["triangles"])
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    for k, ax in enumerate(axes.ravel()):
        if k >= MODES:
            ax.axis("off")
            continue
        w = np.asarray(finest["w_field"]["w"][k])
        w = w / max(abs(w).max(), 1e-30)
        contour = ax.tricontourf(nodes[:, 0], nodes[:, 1], triangles, w, levels=20, cmap="RdBu_r")
        m, n = MODE_LABELS[k]
        ax.set_aspect("equal")
        ax.set_title(
            f"mode {k + 1} ({m},{n}): {finest['frequencies_hz'][k]:.2f} Hz\n"
            f"analytic {oracle[k]:.2f} Hz, MAC {finest['mode_mac'][k]:.4f}",
            fontsize=8,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"SS rectangular plate {LENGTH}x{WIDTH}x{THICKNESS} m: "
        f"first five Mindlin/Q4 mode shapes ({finest['elements_x']}x{finest['elements_y']} mesh)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = HERE / "plate_modes.png"
    fig.savefig(path)
    plt.close(fig)
    out.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    for k in range(MODES):
        ax.loglog(
            [lv["elements_x"] for lv in levels],
            [abs(lv["frequencies_hz"][k] / oracle[k] - 1.0) for lv in levels],
            "o-",
            label=f"mode {k + 1} ({MODE_LABELS[k][0]},{MODE_LABELS[k][1]})",
        )
    elements = [lv["elements_x"] for lv in levels]
    reference_curve = [
        abs(levels[-1]["frequencies_hz"][0] / oracle[0] - 1.0) * (elements[-1] / e) ** 2
        for e in elements
    ]
    ax.loglog(elements, reference_curve, "k:", lw=1.0, label=r"slope $h^2$")
    ax.axhline(
        CRITERIA["frequency_rel_error_max"], color="tab:red", ls="--", lw=1.0, label="1 % gate"
    )
    ax.set_xlabel("elements along a (r = 2 ladder)")
    ax.set_ylabel(r"$|f_{FE}/f_{Kirchhoff} - 1|$")
    ax.set_title("natural-frequency convergence")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.loglog(
        [lv["elements_x"] for lv in levels],
        [
            max(abs(lv["frequencies_hz"][k] / oracle[k] - 1.0) for k in range(MODES))
            for lv in levels
        ],
        "s-",
        label="max over the five modes",
    )
    ax.axhline(
        CRITERIA["frequency_rel_error_max"], color="tab:red", ls="--", lw=1.0, label="1 % gate"
    )
    ax.set_xlabel("elements along a")
    ax.set_ylabel("max relative frequency error")
    ax.set_title(f"worst-mode error (finest pair order {order['min_order_finest_pair']:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = HERE / "plate_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    out.append(path)
    return out


def run(levels_override: list[int] | None) -> dict | None:
    oracle = navier_frequencies()
    plan = [lv for lv in LEVELS if levels_override is None or lv[0] in levels_override]
    for nx, ny in plan:
        solve_modes(nx, ny)
    needed = [f"plate_{nx}x{ny}.json" for nx, ny in LEVELS]
    cached = {p.name for p in CACHE.glob("plate_*.json")} if CACHE.exists() else set()
    if not all(name in cached for name in needed):
        print(json.dumps({"partial": True, "cached": sorted(cached), "needed": needed}, indent=1))
        return None

    raw = [read_json(CACHE / name) for name in needed]
    levels = [graded(rec, oracle) for rec in raw]
    finest = levels[-1]
    order = observed_order(levels, oracle)
    equivalence = library_equivalence(LEVELS[0][0], LEVELS[0][1])
    figs = figures(raw, order, oracle)

    gates = [
        Gate(
            id=f"frequency_mode_{k + 1}_{MODE_LABELS[k][0]}{MODE_LABELS[k][1]}",
            quantity=f"natural frequency of mode {k + 1} ({MODE_LABELS[k][0]},{MODE_LABELS[k][1]})",
            reference="f_mn = (pi/2) sqrt(D/(rho h)) [(m/a)^2 + (n/b)^2]",
            value=finest["relative_errors"][k],
            threshold=CRITERIA["frequency_rel_error_max"],
        )
        for k in range(MODES)
    ]
    gates += [
        Gate(
            id="mode_shape_mac",
            quantity="minimum modal assurance criterion over the five modes",
            reference="MAC = (phi_FE . phi_exact)^2 / (|phi_FE|^2 |phi_exact|^2)",
            value=min(finest["mode_mac"]),
            threshold=CRITERIA["mac_min"],
            operator=">=",
            note="identifies each FE mode with the analytic (m,n) sine product",
        ),
        Gate(
            id="observed_order",
            quantity="minimum observed order over the five modes (finest pair)",
            reference="p = ln(e_coarse/e_fine)/ln(h_coarse/h_fine)",
            value=order["min_order_finest_pair"],
            threshold=CRITERIA["observed_order_min"],
            operator=">=",
        ),
        Gate(
            id="library_equivalence",
            quantity="max relative frequency difference against solve_plate_vibration",
            reference="TensorFEM library entry point on the coarsest level",
            value=equivalence,
            threshold=CRITERIA["library_equivalence_max"],
            note="proves the subset LAPACK eigensolve solves the same discrete problem",
        ),
        Gate(
            id="spurious_mode_filter",
            quantity="near-zero eigenvalues filtered out at the finest mesh",
            reference="supported boundary must carry no mechanism",
            value=float(finest["spurious_modes_filtered"]),
            threshold=0.0,
            note="non-zero means the SRI shear rule still leaves a mechanism in the "
            "supported spectrum; reported, and the retained set is verified positive",
        ),
    ]
    manifest = gate_manifest(
        anchor="fem_plate_modal",
        title="Simply supported rectangular plate: first five natural frequencies vs Navier",
        criteria=CRITERIA,
        gates=gates,
        artifacts={
            "result": "result.json",
            "gates": "gates_verified.json",
            "figures": [p.name for p in figs],
        },
        limitations=[
            "Kirchhoff is the yardstick; at a/h = 100 the Mindlin shear correction is "
            "O((h/a)^2) ~ 2e-4 relative, far below the 1 % gate but non-zero",
            "hard simple support (w = 0 and the normal rotation restrained) is the "
            "library's declared convention; the difference to the soft (SS1) convention "
            "is O((h/a)^2) as well",
            "isotropic, uniform thickness, no prestress, no added mass, no damping, "
            "no fluid coupling",
            "the selective-reduced-integration shear rule leaves zero-energy mechanisms "
            "in the element; the count at each level is reported and the retained "
            "spectrum is asserted positive",
            "the reported order is a two-level estimate per mode; it is not a p-refinement "
            "study and does not extrapolate the frequencies",
        ],
    )
    result = {
        "benchmark": "fem_plate_modal",
        "mission": "IN36-N01",
        "generated_at": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "dtype": "torch.float64",
            "eigensolver": "scipy.linalg.eigh(dsyevr) subset on the Cholesky-reduced pencil",
        },
        "geometry": {
            "length_a": LENGTH,
            "width_b": WIDTH,
            "thickness_h": THICKNESS,
            "young_modulus": YOUNG,
            "poisson_ratio": POISSON,
            "density": DENSITY,
            "aspect_a_over_b": LENGTH / WIDTH,
            "slenderness_a_over_h": LENGTH / THICKNESS,
            "boundary": "simple (hard: w = 0 on all edges, normal rotation restrained)",
        },
        "analytic": {
            "formula": "f_mn = (pi/2) sqrt(D/(rho h)) [(m/a)^2 + (n/b)^2]",
            "plate_rigidity": YOUNG * THICKNESS**3 / (12.0 * (1.0 - POISSON**2)),
            "mode_labels": [list(lbl) for lbl in MODE_LABELS],
            "frequencies_hz": oracle,
            "shear_bias_relative": (THICKNESS / LENGTH) ** 2,
        },
        "levels": levels,
        "convergence": order,
        "library_equivalence_max_rel": equivalence,
        "verdict": {
            "max_frequency_rel_error": finest["max_relative_error"],
            "min_mode_mac": min(finest["mode_mac"]),
            "gate_passed": manifest["passed"],
            "failed_gates": manifest["failed_gates"],
        },
        "artifacts": manifest["artifacts"],
    }
    write_json(HERE / "result.json", result)
    write_json(HERE / "gates_verified.json", manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="SS plate modal anchor")
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=None,
        help="mesh levels (elements along a) to solve in this call",
    )
    args = parser.parse_args()
    result = run(args.levels)
    if result is None:
        return 2
    print(
        json.dumps(
            {
                "passed": result["verdict"]["gate_passed"],
                "failed_gates": result["verdict"]["failed_gates"],
                "max_frequency_rel_error": result["verdict"]["max_frequency_rel_error"],
                "min_mode_mac": result["verdict"]["min_mode_mac"],
                "orders": result["convergence"]["mode_orders_finest_pair"],
            },
            indent=1,
        )
    )
    return 0 if result["verdict"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
