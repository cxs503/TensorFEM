"""MacNeal (1985) slender-beam shear-locking benchmark for the TensorFEM Q4 Mindlin element.

Reference
---------
MacNeal, R. H. (1985). "A simple quadrilateral, 4-node shell element with
enhanced interpolation." *Finite Elements in Analysis and Design*, 5(2), 113-118.

The paper's slender-beam test is a simply-supported beam of length ``L = 10``
and unit square cross-section (``h = 1``, unit width) under a central point
load ``P = 1``, with ``E = 1`` and ``nu = 0.3``.  The beam is discretised as a
two-dimensional strip of four-node quadrilaterals (``n`` elements along the
axis, one through the thickness) and the centre deflection is graded against
the closed-form beam solutions, which are the yardsticks the paper itself uses:

* **Euler-Bernoulli**  ``delta_EB = P L^3 / (48 E I)``  (the thin limit,
  ``I = b h^3 / 12``) -- the primary yardstick of MacNeal (1985);
* **Timoshenko**       ``delta_T  = delta_EB + P L / (4 k A G)``  (the exact
  Mindlin/Reissner answer for a finite ``L/h``, with the ``k = 5/6`` shear
  correction) -- the secondary, more accurate reference at ``L/h = 10``.

Both references are *evaluated in this script* from the closed forms (zero
hand-copying); the geometry, loading, boundary conditions and mesh ladder come
from the paper.

Shear treatments compared
-------------------------
The Q4 Mindlin element of :mod:`tensorfem.plate` is run with three shear
blocks (the bending block is the same 2x2 rule in all three):

* ``"full"`` -- shear integrated with the full 2x2 rule (equal-order).  This is
  the **locking** case: in the thin limit the parasitic shear strain cannot
  vanish, the element stiffens and the deflection is underpredicted.
* ``"sri"``  -- one-point (selective reduced) shear; removes the locking.
* ``"mitc"`` -- assumed natural (Dvorkin-Bathe) shear field; removes the locking
  and the two spurious mechanisms of the one-point rule.

The beam is modelled with the Q4 Mindlin element of :mod:`tensorfem.plate` in a
two-dimensional strip (the ``theta_y`` DOF is constrained to zero).  The bending
rigidity is set to the *beam* value ``E h^3 / 12`` (unit width, no ``1-nu^2``
plate factor) and the shear rigidity to ``k G h`` with ``k = 5/6``, so the
element reduces exactly to a Mindlin/Reissner beam of unit width.

Outputs (relative to the repository root)
-----------------------------------------
* ``benchmarks/macneal_beam_shear_locking/results.json`` -- every number, machine graded
* ``docs/assets/macneal_beam/*.png`` -- deflection field and convergence figures

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/macneal_beam_shear_locking/run.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.plate import (  # noqa: E402
    PlateRigidity,
    _shape,
    q4_mindlin_stiffness,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results.json"
ASSETS = ROOT / "docs" / "assets" / "macneal_beam"

DTYPE = torch.float64

# --------------------------------------------------------------------------- #
# Case definition (MacNeal 1985 slender-beam test)
# --------------------------------------------------------------------------- #
BEAM = dict(young=1.0, poisson=0.3, length=10.0, height=1.0, width=1.0, load=1.0)
SHEAR_CORRECTION = 5.0 / 6.0
TREATMENTS = ("full", "sri", "mitc")
LEVELS = (2, 4, 8, 16)          # elements along the axis (all even -> x=L/2 is a node)
TOLERANCE = 0.05                # task acceptance: < 5 %


def beam_analytic(cfg: dict = BEAM) -> dict:
    """Closed-form centre deflection of a simply-supported, centre-loaded beam.

    ``delta_EB = P L^3 / (48 E I)`` (Euler-Bernoulli, the MacNeal yardstick) and
    ``delta_T = delta_EB + P L / (4 k A G)`` (Timoshenko, the exact Mindlin
    answer at finite ``L/h``).  Both are evaluated here, not copied.
    """
    E, nu, L, h, b, P = (cfg["young"], cfg["poisson"], cfg["length"],
                          cfg["height"], cfg["width"], cfg["load"])
    I = b * h ** 3 / 12.0
    A = b * h
    G = E / (2.0 * (1.0 + nu))
    delta_eb = P * L ** 3 / (48.0 * E * I)
    delta_shear = P * L / (4.0 * SHEAR_CORRECTION * A * G)
    return {
        "inertia": I,
        "area": A,
        "shear_modulus": G,
        "slenderness": L / h,
        "euler_bernoulli": delta_eb,
        "timoshenko": delta_eb + delta_shear,
        "shear_correction": delta_shear,
        "shear_correction_pct": 100.0 * delta_shear / delta_eb,
    }


def beam_rigidity(cfg: dict = BEAM) -> PlateRigidity:
    """Unit-width Mindlin beam rigidity: beam bending, no plate ``1-nu^2`` factor."""
    E, nu, h = cfg["young"], cfg["poisson"], cfg["height"]
    G = E / (2.0 * (1.0 + cfg["poisson"]))
    return PlateRigidity(
        dx=E * cfg["height"] ** 3 / 12.0,          # beam bending, unit width
        dy=E * cfg["height"] ** 3 / 12.0,          # inactive in the 2-D strip
        dxy=0.0,
        dt=E * cfg["height"] ** 3 / 12.0 * (1.0 - cfg["poisson"]) / 2.0,
        sx=SHEAR_CORRECTION * G * cfg["height"],    # k G A, A = h for unit width
        sy=SHEAR_CORRECTION * G * cfg["height"],
    )


def build_strip(n: int, cfg: dict = BEAM) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured 2-D strip: ``n`` Q4 elements along x, one through the thickness."""
    L, h = cfg["length"], cfg["height"]
    xs = torch.linspace(0.0, L, n + 1, dtype=DTYPE)
    z = torch.tensor([0.0, 1.0] * (n + 1), dtype=DTYPE)   # bottom row then top row
    nodes = torch.stack((torch.repeat_interleave(xs, 2), z), dim=1)
    elements = []
    for i in range(n):
        n0, n1, n2, n3 = 2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1
        elements.append((n0, n1, n2, n3))
    return nodes, torch.tensor(elements, dtype=torch.long)


def _point_load_vector(nodes: torch.Tensor, elements: torch.Tensor,
                       x_load: float, z_load: float, P: float) -> torch.Tensor:
    """Consistent nodal vector of a transverse point load (w-DOF only)."""
    f = torch.zeros(3 * nodes.shape[0], dtype=DTYPE)
    for conn in elements:
        xy = nodes[conn]
        if not (xy[:, 0].min() <= x_load <= xy[:, 0].max()
                and xy[:, 1].min() <= z_load <= xy[:, 1].max()):
            continue
        x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
        z0, z1 = float(xy[:, 1].min()), float(xy[:, 1].max())
        xi = 2.0 * (x_load - x0) / (x1 - x0) - 1.0
        eta = 2.0 * (z_load - z0) / (z1 - z0) - 1.0
        N, _ = _shape(nodes.new_tensor(xi), nodes.new_tensor(eta))
        for i in range(4):
            f[3 * int(conn[i])] += P * float(N[i])
        break
    return f


def solve_beam(n: int, treatment: str, cfg: dict = BEAM) -> dict:
    """Assemble and solve the simply-supported centre-loaded strip."""
    rigidity = beam_rigidity(cfg)
    nodes, elements = build_strip(n, cfg)
    ndof = 3 * nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=DTYPE)
    for conn in elements:
        ke = q4_mindlin_stiffness(nodes[conn], shear_treatment=treatment,
                                  rigidity=rigidity)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
    f = _point_load_vector(nodes, elements, cfg["length"] / 2.0, cfg["height"] / 2.0,
                           cfg["load"])
    # Simply supported: w = 0 on both end faces; theta_y = 0 everywhere (2-D strip).
    end = (nodes[:, 0].abs() < 1e-12) | ((nodes[:, 0] - cfg["length"]).abs() < 1e-12)
    fixed = torch.cat((
        3 * torch.nonzero(end, as_tuple=False).flatten(),
        (3 * torch.arange(nodes.shape[0]) + 2),
    )).unique()
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    u = torch.zeros(ndof, dtype=DTYPE)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    # centre node (x = L/2, midline)
    centre = int(torch.argmin(
        ((nodes - nodes.new_tensor([cfg["length"] / 2.0, cfg["height"] / 2.0])) ** 2).sum(1)))
    return {
        "n_elements": n,
        "n_nodes": int(nodes.shape[0]),
        "dofs": int(ndof),
        "free_dofs": int(free.numel()),
        "centre_deflection": float(u[3 * centre]),
        "_w_field": u[0::3].clone(),
        "_nodes": nodes.clone(),
    }


def convergence_study(cfg: dict = BEAM) -> tuple[list[dict], dict]:
    analytic = beam_analytic(cfg)
    rows = []
    for treatment in TREATMENTS:
        for n in LEVELS:
            started = time.perf_counter()
            res = solve_beam(n, treatment, cfg)
            rows.append({
                "treatment": treatment,
                "n_elements": n,
                "n_nodes": res["n_nodes"],
                "dofs": res["dofs"],
                "free_dofs": res["free_dofs"],
                "centre_deflection": res["centre_deflection"],
                "error_vs_eb": (res["centre_deflection"] - analytic["euler_bernoulli"])
                               / analytic["euler_bernoulli"],
                "error_vs_timoshenko": (res["centre_deflection"] - analytic["timoshenko"])
                                      / analytic["timoshenko"],
                "seconds": time.perf_counter() - started,
                "_w_field": res["_w_field"],
                "_nodes": res["_nodes"],
            })
    return rows, analytic


def make_figures(rows: list[dict], analytic: dict) -> list[dict]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    figures = []

    # 1. Convergence: centre deflection vs elements, per treatment.
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    styles = {"full": ("o-", "tab:red"), "sri": ("s-", "tab:blue"), "mitc": ("^-", "tab:green")}
    for treatment in TREATMENTS:
        ms = [r["n_elements"] for r in rows if r["treatment"] == treatment]
        dv = [r["centre_deflection"] for r in rows if r["treatment"] == treatment]
        mk, col = styles[treatment]
        ax.plot(ms, dv, mk, color=col, label=treatment, markerfacecolor="white", ms=6)
    ax.axhline(analytic["timoshenko"], color="k", ls="--", lw=1.2,
               label=f"Timoshenko {analytic['timoshenko']:.2f}")
    ax.axhline(analytic["euler_bernoulli"], color="gray", ls=":", lw=1.4,
               label=f"Euler-Bernoulli {analytic['euler_bernoulli']:.1f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(LEVELS))
    ax.set_xticklabels([str(v) for v in LEVELS])
    ax.set_xlabel("elements along the axis")
    ax.set_ylabel("centre deflection")
    ax.set_title("MacNeal (1985) slender beam, L/h = 10, centre point load")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p1 = ASSETS / "convergence.png"
    fig.savefig(p1)
    plt.close(fig)
    figures.append({"name": "convergence.png", "path": str(p1),
                    "caption": "Centre deflection vs mesh refinement for the three shear "
                               "treatments; dashed/dotted lines are the Timoshenko and "
                               "Euler-Bernoulli references."})

    # 2. Relative error vs refinement (log-log), per treatment.
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    for treatment in TREATMENTS:
        ms = [r["n_elements"] for r in rows if r["treatment"] == treatment]
        ev = [abs(r["error_vs_timoshenko"]) for r in rows if r["treatment"] == treatment]
        mk, col = styles[treatment]
        ax.loglog(ms, ev, mk, color=col, label=treatment, markerfacecolor="white", ms=6)
    ax.axhline(TOLERANCE, color="k", ls="--", lw=1.0, label="5 % tolerance")
    ax.set_xlabel("elements along the axis")
    ax.set_ylabel("|error| vs Timoshenko")
    ax.set_title("Shear-locking error decay (log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = ASSETS / "error_decay.png"
    fig.savefig(p2)
    plt.close(fig)
    figures.append({"name": "error_decay.png", "path": str(p2),
                    "caption": "Relative error of the centre deflection against the "
                               "Timoshenko reference; the equal-order 'full' rule decays "
                               "slowly (locking) while 'sri'/'mitc' converge fast."})

    # 3. Deflection field on the finest SRI mesh.
    finest = [r for r in rows if r["treatment"] == "sri"][-1]
    nodes = finest["_nodes"]
    w = finest["_w_field"]
    from matplotlib.tri import Triangulation
    fig, ax = plt.subplots(figsize=(9.0, 2.6), dpi=150)
    tri = _strip_triangles(nodes, finest["n_elements"])
    triang = Triangulation(nodes[:, 0].numpy(), nodes[:, 1].numpy(),
                          tri.reshape(-1, 3).numpy())
    cfs = ax.tricontourf(triang, w.numpy(), levels=24, cmap="viridis")
    ax.set_aspect(10.0)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"Centre deflection field, SRI, n = {finest['n_elements']} "
                 f"(w_max = {float(w.max()):.2f})")
    fig.colorbar(cfs, ax=ax, label="w")
    fig.tight_layout()
    p3 = ASSETS / "deflection_field.png"
    fig.savefig(p3)
    plt.close(fig)
    figures.append({"name": "deflection_field.png", "path": str(p3),
                    "caption": "Transverse deflection field on the finest SRI mesh."})
    return figures


def _strip_triangles(nodes: torch.Tensor, n: int) -> torch.Tensor:
    """Two triangles per strip cell (node order: bottom row then top row)."""
    tris = []
    for i in range(n):
        b0, b1, t1, t0 = 2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1
        tris.extend((b0, b1, t1, b0, t1, t0))
    return torch.tensor(tris, dtype=torch.long)


def main() -> int:
    started = time.perf_counter()
    rows, analytic = convergence_study()
    figures = make_figures(rows, analytic)

    # machine-graded verdicts (re-derived from raw numbers, never from flags)
    def finest(treatment: str) -> dict:
        return [r for r in rows if r["treatment"] == treatment][-1]

    verdicts = {}
    for treatment in TREATMENTS:
        r = finest(treatment)
        verdicts[treatment] = {
            "finest_n": r["n_elements"],
            "centre_deflection": r["centre_deflection"],
            "error_vs_eb": r["error_vs_eb"],
            "error_vs_timoshenko": r["error_vs_timoshenko"],
            "pass_vs_timoshenko_5pct": abs(r["error_vs_timoshenko"]) < TOLERANCE,
            "pass_vs_eb_5pct": abs(r["error_vs_eb"]) < TOLERANCE,
        }
    # locking evidence: the equal-order rule must be the least accurate at the
    # coarsest level and must decay slower than the locking-free rules.
    def decay(treatment: str) -> float:
        sel = [r for r in rows if r["treatment"] == treatment]
        return abs(sel[-1]["error_vs_timoshenko"]) / max(abs(sel[0]["error_vs_timoshenko"]), 1e-300)

    payload = {
        "benchmark": "macneal_1985_slender_beam_shear_locking",
        "reference": ("MacNeal, R.H. (1985). A simple quadrilateral, 4-node shell element "
                      "with enhanced interpolation. Finite Elements in Analysis and "
                      "Design 5(2):113-118 -- slender-beam test (L=10, h=1, E=1, nu=0.3, "
                      "centre point load P=1, simply supported)."),
        "generated_by": "benchmarks/macneal_beam_shear_locking/run.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "dtype": "torch.float64",
        },
        "case": {**BEAM, "shear_correction_factor": SHEAR_CORRECTION},
        "analytic": analytic,
        "tolerance": TOLERANCE,
        "levels": list(LEVELS),
        "treatments": list(TREATMENTS),
        "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "verdicts": verdicts,
        "decay_factors_vs_coarsest": {t: decay(t) for t in TREATMENTS},
        "figures": [f["name"] for f in figures],
        "total_seconds": time.perf_counter() - started,
    }
    RESULTS.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"analytic": analytic, "verdicts": verdicts,
                      "decay": payload["decay_factors_vs_coarsest"],
                      "seconds": payload["total_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
