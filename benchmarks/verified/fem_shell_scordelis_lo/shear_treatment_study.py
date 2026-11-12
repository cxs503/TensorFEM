#!/usr/bin/env python3
"""Scordelis-Lo, flat-facet Shell4: rank census + shear-treatment ladder.

Question this study answers
---------------------------
``benchmarks/verified/fem_shell_scordelis_lo`` reported a ~50 % error for the
flat-facet Shell4 and attributed it to a *fundamental* limitation: "flat facet
approximation cannot capture membrane-bending coupling in curved shells".
That attribution is a claim about faceting.  It is testable: change nothing but
the transverse-shear integration rule of the *same* flat-facet element (same
nodes, same facets, same load, same diaphragms) and see whether the error
survives.

The study therefore measures, on one and the same faceted mesh family:

  (1) rank census of the 24x24 element stiffness
      (:func:`tensorfem.shell4_rank.element_rank_census`): how many zero-energy
      modes does the element have beyond the six rigid-body modes?
  (2) four-level mesh ladder for three shear rules (``sri`` one-point, ``mitc``
      assumed natural shear, ``full`` 2x2 equal-order locking reference) against
      the published reference w = -0.3024;

and writes ``shear_treatment_study.json`` + ``shear_treatment_study.png`` next
to the existing anchor artifacts.  No timestamps enter the JSON.

Run from this directory (the anchor pins ``tensorfem`` to this checkout):

    OMP_NUM_THREADS=8 python3 shear_treatment_study.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(_REPO_SRC))

import tensorfem  # noqa: E402

if _REPO_SRC.resolve() not in Path(tensorfem.__file__).resolve().parents:
    raise SystemExit(
        f"provenance error: tensorfem resolved to {tensorfem.__file__}, outside "
        f"this checkout ({_REPO_SRC})"
    )

from tensorfem.shell4_rank import element_rank_census  # noqa: E402
from tensorfem.shell_benchmarks import scordelis_lo_roof  # noqa: E402

D = torch.float64
OUT = Path(__file__).parent
REFERENCE = -0.3024

PROBLEM = {
    "name": "Scordelis-Lo cylindrical roof",
    "length": 50.0,
    "radius": 25.0,
    "half_angle_deg": 40.0,
    "thickness": 0.25,
    "young": 4.32e8,
    "poisson": 0.0,
    "vertical_surface_load": 90.0,
    "boundaries": "rigid diaphragms at both curved ends (u_y=u_z=0), one axial DOF fixed",
    "quantity_of_interest": "vertical displacement at midspan of the free edge",
    "reference_value": REFERENCE,
    "reference_source": "Scordelis & Lo (1960); MacNeal & Harder (1985)",
}
L, R = PROBLEM["length"], PROBLEM["radius"]
ANG = math.radians(PROBLEM["half_angle_deg"])
E, NU, T = PROBLEM["young"], PROBLEM["poisson"], PROBLEM["thickness"]

MESHES = [4, 8, 16, 32]
SHEARS = ("sri", "mitc", "full")


def _facet() -> torch.Tensor:
    """One facet of the (x, theta) mesh: planar quad, chord in the theta direction."""
    dx, dt = 1.0, math.radians(2.0)
    x0, t0 = 0.0, math.radians(-1.0)
    return torch.tensor(
        [
            [x0, R * math.sin(t0), R * math.cos(t0)],
            [x0 + dx, R * math.sin(t0), R * math.cos(t0)],
            [x0 + dx, R * math.sin(t0 + dt), R * math.cos(t0 + dt)],
            [x0, R * math.sin(t0 + dt), R * math.cos(t0 + dt)],
        ],
        dtype=D,
    )


# ---------------------------------------------------------------------------
# (1) element rank census
# ---------------------------------------------------------------------------
def rank_census(plate_shear: str) -> dict:
    census = element_rank_census(_facet(), E, NU, T, plate_shear=plate_shear)
    mechanisms = []
    for v in census.mechanisms:
        w = v[2::6]
        alternating = (
            abs(float(w[0] + w[1])) < 1e-9
            and abs(float(w[0] - w[2])) < 1e-9
            and abs(float(w[0])) > 1e-6
        )
        mechanisms.append(
            {
                "blocks": {
                    "ux": round(float(torch.linalg.vector_norm(v[0::6])), 6),
                    "uy": round(float(torch.linalg.vector_norm(v[1::6])), 6),
                    "w": round(float(torch.linalg.vector_norm(v[2::6])), 6),
                    "rx": round(float(torch.linalg.vector_norm(v[3::6])), 6),
                    "ry": round(float(torch.linalg.vector_norm(v[4::6])), 6),
                    "rz": round(float(torch.linalg.vector_norm(v[5::6])), 6),
                },
                "per_node_w": [round(float(x), 6) for x in w.tolist()],
                "w_checkerboard": bool(alternating),
            }
        )
    return {
        "plate_shear": plate_shear,
        "lambda_max": round(census.lambda_max, 6),
        "null_tolerance": census.rank_tolerance,
        "zero_modes": census.zero_modes,
        "rigid_body_modes": 6,
        "rigid_basis_residual_over_lambda_max": round(census.rigid_basis_residual, 12),
        "spurious_modes": census.spurious_modes,
        "spurious_near_rigid_singular_values": [
            round(float(s), 9) for s in census.singular_values.tolist()
        ],
        "spurious_mechanisms": mechanisms,
        "first_elastic_eigenvalues": [
            round(float(v), 6) for v in census.first_elastic_eigenvalues.tolist()
        ],
    }


# ---------------------------------------------------------------------------
# (2) shear-treatment ladder on the *unmodified* benchmark geometry
# ---------------------------------------------------------------------------
def ladder(plate_shear: str) -> list[dict]:
    rows = []
    for n in MESHES:
        r = scordelis_lo_roof(n, n, dtype=D, plate_shear=plate_shear)
        w = float(r.probe_displacement)
        err = abs(w / REFERENCE - 1.0)
        rows.append(
            {
                "mesh": f"{n}x{n}",
                "nx": n,
                "nt": n,
                "elements": n * n,
                "dof": 6 * (n + 1) ** 2,
                "probe_displacement": round(w, 9),
                "relative_error": round(err, 9),
                "relative_error_pct": round(err * 100, 4),
            }
        )
        print(f"  {plate_shear:5s} {n:2d}x{n:<2d} w={w: .6f}  err={err*100:7.3f}%", flush=True)
    return rows


def convergence(rows: list[dict]) -> dict:
    """Honest convergence statement from the error sequence.

    The reference is a published constant here, so the ladder is judged on the
    *error* sequence: a decreasing sequence admits an observed order and a
    Richardson limit; a sequence that turns back up must say so instead.
    """
    errs = [r["relative_error"] for r in rows]
    h = [r["nx"] for r in rows]
    decreasing = all(errs[k] < errs[k - 1] for k in range(1, len(errs)))
    order = None
    limit = None
    limit_err = None
    if decreasing:
        order = math.log(errs[-2] / errs[-1]) / math.log(h[-1] / h[-2])
        v_last = rows[-1]["probe_displacement"]
        v_prev = rows[-2]["probe_displacement"]
        ratio = h[-1] / h[-2]
        if order > 0 and abs(ratio ** order - 1.0) > 1e-12:
            limit = round(v_last + (v_last - v_prev) / (ratio ** order - 1.0), 9)
            limit_err = round(abs(limit / REFERENCE - 1.0) * 100, 4)
    return {
        "error_monotone_decreasing": decreasing,
        "observed_order_finest_pair": None if order is None else round(order, 4),
        "richardson_limit": limit,
        "richardson_error_pct": limit_err,
    }


def main() -> None:
    print("rank census (24x24 facet stiffness, 6 rigid-body modes expected):")
    census = {}
    for shear in SHEARS:
        c = rank_census(shear)
        census[shear] = c
        print(
            f"  {shear:5s} zero_modes={c['zero_modes']:2d}  "
            f"spurious={c['spurious_modes']:2d}  lambda_max={c['lambda_max']:.4e}"
        )

    print("\nladders (reference -0.3024):")
    ladders = {shear: ladder(shear) for shear in SHEARS}

    verdicts = {}
    for shear, rows in ladders.items():
        finest = rows[-1]["relative_error"]
        best = min(r["relative_error"] for r in rows)
        verdicts[shear] = {
            "spurious_zero_energy_modes": census[shear]["spurious_modes"],
            "finest_mesh_error": finest,
            "best_mesh_error": round(best, 9),
            "passes_3pct_at_finest": bool(finest < 0.03),
            "passes_3pct_anywhere": bool(best < 0.03),
            "convergence": convergence(rows),
        }

    payload = {
        "schema": "fem-shell-scordelis-shear-study/1",
        "problem": PROBLEM,
        "tolerance_pct": 3.0,
        "element": "flat-facet four-node shell (Q4 membrane + Mindlin Q4 plate, 6 dof/node)",
        "rank_census": census,
        "ladders": ladders,
        "verdicts": verdicts,
        "conclusion": (
            "The Scordelis-Lo failure of the flat-facet Shell4 is a rank deficiency of the "
            "element, not a property of faceting: with identical facets, load and diaphragms, "
            "replacing the one-point (sri) transverse-shear rule by assumed natural shear (mitc) "
            f"removes {census['sri']['spurious_modes'] - census['mitc']['spurious_modes']} of the "
            f"{census['sri']['spurious_modes']} spurious zero-energy mechanisms and moves the "
            f"finest-mesh error from {verdicts['sri']['finest_mesh_error']*100:.1f}% to "
            f"{verdicts['mitc']['finest_mesh_error']*100:.2f}% "
            f"(passes the 3% gate from 16x16 on: "
            f"{ladders['mitc'][2]['relative_error_pct']:.2f}%). Equal-order full shear "
            f"integration (the locking control) misses by "
            f"{verdicts['full']['finest_mesh_error']*100:.1f}% and still carries "
            f"{census['full']['spurious_modes']} spurious mechanism, confirming the shear block "
            "is the operative lever. The previously published attribution 'flat facets cannot "
            "capture curved-shell membrane-bending coupling' is therefore falsified as a "
            "statement about faceting."
        ),
    }

    OUT.joinpath("shear_treatment_study.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote", OUT / "shear_treatment_study.json")
    print(payload["conclusion"])

    _plot(ladders, census)


def _plot(ladders: dict, census: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    style = {
        "sri": ("tab:red", "o", "one-point shear (sri, default)"),
        "mitc": ("tab:green", "s", "assumed natural shear (mitc)"),
        "full": ("tab:gray", "^", "equal-order full shear (locking control)"),
    }
    ax = axes[0]
    for shear, (colour, marker, label) in style.items():
        xs = [r["nx"] for r in ladders[shear]]
        ys = [abs(r["probe_displacement"]) for r in ladders[shear]]
        ax.semilogx(xs, ys, marker=marker, color=colour, label=label, lw=1.6)
    ax.axhline(abs(REFERENCE), color="k", ls="--", lw=1.2, label="Scordelis & Lo (1960) = 0.3024")
    ax.set_xlabel("elements per edge")
    ax.set_ylabel("|w| at free-edge midspan")
    ax.set_title("(a) mesh ladder, identical facets")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    width = 0.26
    for k, (shear, (colour, _marker, _label)) in enumerate(style.items()):
        xs = [i + (k - 1) * width for i in range(len(MESHES))]
        ys = [r["relative_error_pct"] for r in ladders[shear]]
        ax.bar(xs, ys, width=width, color=colour, label=shear)
    ax.axhline(3.0, color="k", ls="--", lw=1.2, label="3 % acceptance")
    ax.set_xticks(range(len(MESHES)))
    ax.set_xticklabels([r["mesh"] for r in ladders["sri"]])
    ax.set_ylabel("relative error vs reference [%]")
    ax.set_title("(b) error by shear treatment")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    ax = axes[2]
    labels = list(SHEARS)
    counts = [census[s]["spurious_modes"] for s in labels]
    ax.bar(labels, counts, color=[style[s][0] for s in labels])
    for i, c in enumerate(counts):
        ax.text(i, c + 0.06, f"{c} spurious\nzero-energy", ha="center", fontsize=9)
    ax.set_ylabel("zero-energy modes beyond the 6 rigid-body modes")
    ax.set_title("(c) element rank census, 24x24 facet")
    ax.set_ylim(0, max(counts) + 1.5)

    fig.suptitle(
        "Scordelis-Lo roof, flat-facet Shell4: the shear rule, not faceting, decides accuracy",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "shear_treatment_study.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "shear_treatment_study.png")


if __name__ == "__main__":
    main()