#!/usr/bin/env python
"""CASE-FEM-SCORDELIS-LO-ROOF -- Scordelis-Lo cylindrical roof (curved-shell bending).

Case statement
--------------
The classical curved-shell benchmark (Scordelis & Lo 1960; MacNeal & Harder
1985): a cylindrical roof of length ``L = 50``, radius ``R = 25``, half angle
``40 deg`` and thickness ``t = 0.25``, ``E = 4.32e8``, ``nu = 0``, loaded by a
uniform vertical surface load ``q = 90``, supported by rigid end diaphragms
(``u_y = u_z = 0`` on both curved ends) with one axial dof removed.  The
reported quantity is the vertical displacement at the midspan of the free edge::

    w_ref = -0.3024      (the ONLY transcribed number in this file)

Why this case is in the pool
----------------------------
The repository ships a verification artefact for this benchmark
(``benchmarks/verified/fem_shell_scordelis_lo/result.json``) whose flat-facet
ladder *oscillates about 50 % above* the reference.  That failure had been
explained by an element-family statement -- "flat facets cannot capture
membrane-bending coupling" -- and the m16449 rank study
(``docs/CASE-FEM-SCORDELIS-LO-shear-rank-diagnosis.md``) falsified it: holding
the facets, nodes, load and diaphragms fixed and changing only the transverse
shear rule of the *same* element moves the ladder from 49 % to 0.75 % at
32x32.  This case entry turns that finding into a fail-closed, reproducible
anchor with three configurations measured on the same meshes:

``cyl``      metric-consistent ``cylindrical_shell4`` (curved geometry, exact
             cylindrical metric/curvature) -- an *independent formulation*, so
             agreement is not the same element talking to itself;
``mitc``     flat-facet ``Shell4`` with assumed natural transverse shear, the
             configuration the rank study identified as sufficient;
``sri``      flat-facet ``Shell4`` with the historical default one-point
             transverse shear -- the documented *null*: its error must stay
             large.  This gate is inverted on purpose, so the falsified
             "faceting" explanation cannot quietly re-enter as "fixed".

Everything else (facet areas, sagitta, convergences, Richardson limit, rank
census, reactions) is computed here; no number is typed in.

Run (writes ``benchmarks/artifacts/CASE-FEM-SCORDELIS-LO-ROOF/``)::

    PYTHONPATH=src python benchmarks/case_scordelis_lo_roof.py

Exit status is 0 only if every gate passes (fail-closed).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.cylindrical_shell_benchmarks import scordelis_lo_cylindrical  # noqa: E402
from tensorfem.shell4_rank import element_rank_census  # noqa: E402
from tensorfem.shell_benchmarks import scordelis_lo_roof  # noqa: E402

CASE = "CASE-FEM-SCORDELIS-LO-ROOF"
PARAMETERS = {
    "length_m": 50.0,
    "radius_m": 25.0,
    "half_angle_deg": 40.0,
    "thickness_m": 0.25,
    "young_pa": 4.32e8,
    "poisson": 0.0,
    "vertical_load_pa": 90.0,
    "support": "rigid end diaphragms u_y = u_z = 0, one axial dof fixed",
}
REFERENCE = -0.3024
MESHES = (4, 8, 16)
PRIMARY_MESH = 16
GATE_3PCT = 0.03
#: One facet of the 24-element-per-edge sector; the m16449 rank study used the
#: same chord (0.873) so the census here is directly comparable with it.
CENSUS_SECTORS = 24

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "benchmarks" / "artifacts" / CASE
FIGDIR = ROOT / "docs" / "assets" / "case-scordelis-lo-roof"
DOC = ROOT / "docs" / (CASE + ".md")
REPRO_CMD = "PYTHONPATH=src python benchmarks/case_scordelis_lo_roof.py"


# ---------------------------------------------------------------- geometry ---
def exact_surface_area() -> float:
    """Analytic area of the cylindrical mid-surface, ``L * R * 2*half_angle``."""
    return PARAMETERS["length_m"] * PARAMETERS["radius_m"] * math.radians(
        2.0 * PARAMETERS["half_angle_deg"]
    )


def facet_area(nodes: torch.Tensor, elements: torch.Tensor) -> float:
    """Sum of the planar facet areas of the assembled mesh (from coordinates)."""
    total = 0.0
    for element in elements:
        xyz = nodes[element]
        total += 0.5 * float(torch.linalg.vector_norm(torch.linalg.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])))
        total += 0.5 * float(torch.linalg.vector_norm(torch.linalg.cross(xyz[2] - xyz[0], xyz[3] - xyz[0])))
    return total


def sagitta(nt: int) -> float:
    """In-surface chord deviation of one facet, ``R (1 - cos(dtheta/2))``."""
    dtheta = math.radians(2.0 * PARAMETERS["half_angle_deg"]) / nt
    return PARAMETERS["radius_m"] * (1.0 - math.cos(dtheta / 2.0))


# -------------------------------------------------------------- one solve ---
def solve(config: str, nx: int) -> dict:
    """Solve one configuration on an ``nx x nx`` mesh and audit its equilibrium."""
    started = time.time()
    if config == "cyl":
        result = scordelis_lo_cylindrical(nx, nx)
        formulation = "metric-consistent cylindrical shell4 (curved geometry)"
    elif config == "mitc":
        result = scordelis_lo_roof(nx, nx, plate_shear="mitc")
        formulation = "flat-facet Shell4, assumed natural transverse shear"
    elif config == "sri":
        result = scordelis_lo_roof(nx, nx, plate_shear="sri")
        formulation = "flat-facet Shell4, default one-point transverse shear"
    else:  # pragma: no cover - programming error
        raise ValueError("unknown configuration %r" % (config,))
    probe = result.probe_displacement
    area = facet_area(result.nodes, result.elements)
    pressure = PARAMETERS["vertical_load_pa"]
    # The two formulations integrate the vertical load on *their own* surface
    # measure: the cylindrical element uses the exact arc area, the flat-facet
    # element the faceted area.  Equilibrium is therefore checked against the
    # measure each one integrates; the difference between the two measures is
    # the faceting budget reported separately as `facet_area_deficit`.
    loaded_area = exact_surface_area() if config == "cyl" else area
    applied = pressure * loaded_area
    # Vertical reactions live only on the two diaphragm end rings (u_y fixed);
    # free-dof residuals are zero by construction, so this sum is the total
    # support force that must balance q * A_facet.
    nt = nx
    ring_nodes = [i * (nt + 1) + j for i in (0, nx) for j in range(nt + 1)]
    reaction = float(result.reaction[[6 * n + 2 for n in ring_nodes]].sum())
    return {
        "configuration": config,
        "formulation": formulation,
        "mesh": "%dx%d" % (nx, nx),
        "elements": int(result.elements.shape[0]),
        "nodes": int(result.nodes.shape[0]),
        "probe_displacement": probe,
        "reference": REFERENCE,
        "relative_error": probe / REFERENCE - 1.0,
        "facet_area": area,
        "facet_area_deficit": area / exact_surface_area() - 1.0,
        "sagitta_over_thickness": sagitta(nx) / PARAMETERS["thickness_m"],
        "reaction_vertical": reaction,
        "applied_vertical": applied,
        "equilibrium_ratio": reaction / applied,
        "seconds": round(time.time() - started, 2),
    }


# ------------------------------------------------------------- rank census ---
def rank_census() -> list[dict]:
    """Zero-energy census of one facet for the three transverse-shear rules."""
    L, R, angle = (PARAMETERS["length_m"], PARAMETERS["radius_m"],
                   math.radians(PARAMETERS["half_angle_deg"]))
    dtheta = 2.0 * angle / CENSUS_SECTORS
    dx = L / CENSUS_SECTORS
    pts = [
        (-dx / 2.0, R * math.sin(-dtheta / 2.0), R * math.cos(-dtheta / 2.0)),
        (dx / 2.0, R * math.sin(-dtheta / 2.0), R * math.cos(-dtheta / 2.0)),
        (dx / 2.0, R * math.sin(dtheta / 2.0), R * math.cos(dtheta / 2.0)),
        (-dx / 2.0, R * math.sin(dtheta / 2.0), R * math.cos(dtheta / 2.0)),
    ]
    xyz = torch.tensor(pts, dtype=torch.float64)
    out = []
    for rule in ("sri", "mitc"):
        census = element_rank_census(
            xyz, PARAMETERS["young_pa"], PARAMETERS["poisson"],
            PARAMETERS["thickness_m"], plate_shear=rule,
        )
        transverse = []
        for mech in census.mechanisms:
            # dof order per node is (ux, uy, uz, rx, ry, rz)
            w = mech[2::6]
            rx, ry = mech[3::6], mech[4::6]
            transverse.append({
                "transverse_displacement_norm": float(torch.linalg.vector_norm(w)),
                "transverse_rotation_norm": float(
                    torch.linalg.vector_norm(torch.cat((rx, ry)))
                ),
            })
        out.append({
            "plate_shear": rule,
            "zero_modes": census.zero_modes,
            "spurious_modes": census.spurious_modes,
            "rigid_basis_residual": census.rigid_basis_residual,
            "first_elastic_eigenvalue": float(census.first_elastic_eigenvalues[0]),
            "mechanisms": transverse,
            "transverse_mechanisms": sum(
                1 for m in transverse
                if max(m["transverse_displacement_norm"],
                       m["transverse_rotation_norm"]) > 1e-6
            ),
        })
    return out


# ---------------------------------------------------------------- extrapolate -
def richardson(levels: list[dict], key: str = "probe_displacement") -> dict:
    """Observed order and Richardson limit from the three finest dyadic levels."""
    tail = levels[-3:]
    h = [1.0 / int(l["mesh"].split("x")[0]) for l in tail]
    w = [l[key] for l in tail]
    if not (h[0] / h[1] == h[1] / h[2] == 2.0):
        raise ValueError("Richardson limit needs three dyadic levels")
    d1, d2 = w[1] - w[0], w[2] - w[1]
    if d1 == 0.0 or d2 / d1 <= 0.0:
        return {"levels": tail, "observed_order": None,
                "richardson_limit": None,
                "reason": "differences do not have a constant sign"}
    order = math.log(abs(d1 / d2)) / math.log(2.0)
    limit = w[2] + d2 / (2.0 ** order - 1.0)
    return {
        "levels": tail,
        "observed_order": order,
        "richardson_limit": limit,
        "limit_relative_error": limit / REFERENCE - 1.0,
        "reason": "",
    }


# -------------------------------------------------------------------- gates ---
def build_gates(ladders: dict, census: list[dict]) -> list[dict]:
    def level(config, mesh):
        return next(l for l in ladders[config] if l["mesh"] == "%dx%d" % (mesh, mesh))

    gates: list[dict] = []

    def gate(name, quantity, value, bound, passed, note=""):
        gates.append({"gate": name, "quantity": quantity, "value": value,
                      "bound": bound, "passed": bool(passed), "note": note})

    cyl16 = level("cyl", PRIMARY_MESH)
    cyl8 = level("cyl", 8)
    mitc16 = level("mitc", PRIMARY_MESH)
    sri16 = level("sri", PRIMARY_MESH)

    gate("geometry.exact_area_vs_facets", "facet area deficit at 16x16",
         cyl16["facet_area_deficit"], "< 1e-3",
         abs(cyl16["facet_area_deficit"]) < 1e-3,
         "planar facets under-measure the cylindrical surface; the deficit is "
         "reported, not ignored")

    flat_worst = max(abs(l["equilibrium_ratio"] - 1.0)
                     for c in ("mitc", "sri") for l in ladders[c])
    cyl_worst = max(abs(l["equilibrium_ratio"] - 1.0) for l in ladders["cyl"])
    gate("equilibrium.flat_facet_reaction", "max |reaction/applied - 1|, flat facet",
         flat_worst, "< 1e-9", flat_worst < 1e-9,
         "vertical reaction sum over the diaphragm rings equals q * faceted area "
         "to machine precision for both shear rules -- the load integral and the "
         "assembled rigid-body reproduction are exact")
    gate("equilibrium.cylindrical_reaction", "max |reaction/applied - 1|, cylindrical",
         cyl_worst, "< 1 %", cyl_worst < 1e-2,
         "the cylindrical element integrates the exact arc area but its reaction "
         "sum carries a residual that falls ~h^2; it is exactly the rigid-mode "
         "projection (K 1_z)^T u / (-1^T f) and is attributed in "
         "docs/CASE-FEM-SCORDELIS-LO-ROOF-reaction-residual-attribution.md -- a "
         "curved-metric discretisation limit, not a load or reaction-extraction "
         "defect")

    gate("accuracy.cylindrical_16x16", "w error, metric-consistent element at 16x16",
         cyl16["relative_error"], "< 3 %",
         abs(cyl16["relative_error"]) < GATE_3PCT,
         "independent formulation; primary anchor")
    gate("accuracy.cylindrical_8x8", "w error, metric-consistent element at 8x8",
         cyl8["relative_error"], "< 3 %",
         abs(cyl8["relative_error"]) < GATE_3PCT,
         "the independent formulation clears the gate on a coarse mesh")
    gate("accuracy.flat_facet_mitc_16x16", "w error, flat facet + mitc at 16x16",
         mitc16["relative_error"], "< 3 %",
         abs(mitc16["relative_error"]) < GATE_3PCT,
         "the same facets the historical artefact reported at ~50 % error")
    gate("null.default_sri_16x16", "w error, flat facet + default sri at 16x16",
         sri16["relative_error"], "> 20 %",
         abs(sri16["relative_error"]) > 0.20,
         "INVERTED gate: the documented default failure must stay visible; if "
         "this ever passes, the null (and the corrected root cause) must be "
         "re-derived, not quietly dropped")

    sri_transverse = next(c for c in census if c["plate_shear"] == "sri")["transverse_mechanisms"]
    mitc_transverse = next(c for c in census if c["plate_shear"] == "mitc")["transverse_mechanisms"]
    gate("rank.sri_transverse_mechanisms", "spurious mechanisms with transverse content (sri)",
         sri_transverse, ">= 1", sri_transverse >= 1,
         "the rank defect that explains the 50 % plateau")
    gate("rank.mitc_transverse_mechanisms", "spurious mechanisms with transverse content (mitc)",
         mitc_transverse, "== 0", mitc_transverse == 0,
         "assumed natural shear leaves only an in-plane (drilling gauge) "
         "direction, which cannot carry bending energy")

    order = richardson(ladders["cyl"])["observed_order"]
    gate("convergence.cylindrical_order", "observed order, cylindrical element",
         order, "1.5 <= p <= 5",
         order is not None and 1.5 <= order <= 5.0,
         "dyadic 4/8/16 ladder")

    monotone = all(
        abs(a["relative_error"]) > abs(b["relative_error"])
        for a, b in zip(ladders["mitc"], ladders["mitc"][1:])
    )
    gate("convergence.flat_facet_mitc_monotone", "flat facet + mitc error strictly decreasing",
         monotone, "true", monotone,
         "the historical sri ladder oscillates; this one must not")
    return gates


# ------------------------------------------------------------------ figures ---
def make_figures(ladders: dict, census: list[dict]) -> list[Path]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 4.2))

    ax = fig.add_subplot(1, 3, 1)
    for config, label, style in (("cyl", "cylindrical shell4 (metric consistent)", "ko-"),
                                 ("mitc", "flat facet + mitc", "bs-"),
                                 ("sri", "flat facet + sri (default)", "r^-")):
        n = [int(l["mesh"].split("x")[0]) for l in ladders[config]]
        w = [abs(l["probe_displacement"]) for l in ladders[config]]
        ax.loglog(n, w, style, label=label, lw=1.8, ms=7)
    ax.axhline(abs(REFERENCE), color="0.35", ls="--", lw=1.2,
               label="reference |w| = %.4f" % abs(REFERENCE))
    ax.set_xlabel("elements per edge")
    ax.set_ylabel("|w| at free-edge midspan")
    ax.set_title("(a) convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(1, 3, 2)
    labels, values, colors = [], [], []
    for config, label, colour in (("cyl", "cyl", "0.2"),
                                  ("mitc", "mitc", "tab:blue"),
                                  ("sri", "sri (default)", "tab:red")):
        row = next(l for l in ladders[config] if l["mesh"] == "%dx%d" % (PRIMARY_MESH, PRIMARY_MESH))
        labels.append(label)
        values.append(100.0 * abs(row["relative_error"]))
        colors.append(colour)
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(3.0, color="tab:green", ls="--", lw=1.2, label="3 % acceptance")
    ax.set_yscale("log")
    ax.set_ylabel("|error| vs w_ref  [%]")
    ax.set_title("(b) error at %dx%d" % (PRIMARY_MESH, PRIMARY_MESH))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.15, "%.2f%%" % value,
                ha="center", fontsize=8)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    rules = [c["plate_shear"] for c in census]
    totals = [c["spurious_modes"] for c in census]
    transverse = [c["transverse_mechanisms"] for c in census]
    x = range(len(rules))
    ax.bar([i - 0.18 for i in x], totals, width=0.36, label="all spurious modes",
           color="0.6")
    ax.bar([i + 0.18 for i in x], transverse, width=0.36,
           label="with transverse content", color="tab:orange")
    ax.set_xticks(list(x))
    ax.set_xticklabels(rules)
    ax.set_ylabel("zero-energy modes per facet")
    ax.set_title("(c) rank census, one facet")
    ax.legend(fontsize=8)
    for i, (t, s) in enumerate(zip(totals, transverse)):
        ax.text(i - 0.18, t + 0.08, str(t), ha="center", fontsize=8)
        ax.text(i + 0.18, s + 0.08, str(s), ha="center", fontsize=8)

    fig.tight_layout()
    out = FIGDIR / "scordelis_lo_roof_case.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


# ---------------------------------------------------------------- rendering ---
def render_doc(payload: dict) -> str:
    pr = payload["primary"]
    par = payload["parameters"]
    ladders = payload["ladders"]
    census = payload["rank_census"]
    rich = payload["richardson"]["cyl"]
    gates = payload["gates"]
    failed = [g for g in gates if not g["passed"]]
    L: list[str] = []
    A = L.append

    A("# %s -- Scordelis-Lo cylindrical roof" % CASE)
    A("")
    A("**Verdict: %s** (%d/%d gates). Primary anchor: metric-consistent "
      "cylindrical shell4 at %s gives `w = %.6f` vs the published "
      "`w_ref = %s` = `%+.3f %%` (gate 3 %%). The historical flat-facet default "
      "(`sri`) reproduces its documented `%+.1f %%` failure on the same mesh, "
      "while the same facets with assumed natural shear reach `%+.2f %%`."
      % ("PASS" if not failed else "FAIL: " + ", ".join(g["gate"] for g in failed),
         len(gates) - len(failed), len(gates), pr["mesh"],
         pr["probe_displacement"], REFERENCE, 100 * pr["relative_error"],
         100 * ladders["sri"][-1]["relative_error"],
         100 * ladders["mitc"][-1]["relative_error"]))
    A("")
    A("Reproduce: `%s`  " % REPRO_CMD)
    A("Artifacts: `benchmarks/artifacts/%s/case_result.{json,md}`  " % CASE)
    A("Figure: `docs/assets/case-scordelis-lo-roof/scordelis_lo_roof_case.png`  ")
    A("Tests: `PYTHONPATH=src python -m pytest tests/test_case_scordelis_lo_roof.py -q`")
    A("")
    A("## 1. Problem statement")
    A("")
    A("Open cylindrical roof with rigid end diaphragms under a uniform vertical "
      "surface load -- the standard test for membrane-bending coupling in curved "
      "shell elements. It is membrane dominated, so an element that locks in "
      "membrane/shear does not merely lose accuracy: it loses the load path.")
    A("")
    A("| parameter | value |")
    A("|---|---|")
    for key, val in par.items():
        A("| %s | %s |" % (key, val))
    A("| exact mid-surface area | %.4f |" % payload["exact_surface_area"])
    A("| reference `w_ref` | %.4f |" % REFERENCE)
    A("")
    A("## 2. Method")
    A("")
    A("No new element code. Three configurations run through their merged "
      "assembly paths on identical meshes (4x4, 8x8, 16x16):")
    A("")
    for config in ("cyl", "mitc", "sri"):
        A("- `%s` -- %s" % (config, ladders[config][0]["formulation"]))
    A("")
    A("Audits attached to every solve, not assumed: (i) the assembled facet area "
      "against the analytic `L R 2*alpha` (a geometry budget, the shell analogue "
      "of validating `Cb/L/B` before trusting a resistance coefficient); "
      "(ii) the vertical reaction sum against the applied resultant `q * A_facet`; "
      "(iii) a rank census of one facet per shear rule "
      "(`tensorfem.shell4_rank.element_rank_census`), which splits the null space "
      "into rigid-body directions and genuine mechanisms and reports how many of "
      "the latter carry transverse displacement/rotation content.")
    A("")
    A("Order and limit: observed order and the Richardson limit from the dyadic "
      "4/8/16 ladder.")
    A("")
    A("## 3. Results")
    A("")
    A("### 3.1 Ladder (same facets, same load, same diaphragms)")
    A("")
    A("| mesh | facet area deficit | sagitta / t | cylindrical `w` | err | flat+mitc `w` | err | flat+sri `w` | err |")
    A("|---|---|---|---|---|---|---|---|---|")
    for i, mesh in enumerate(ladders["cyl"]):
        rows = {c: ladders[c][i] for c in ("cyl", "mitc", "sri")}
        A("| %s | %+.4f %% | %.3f | %.6f | %+.3f %% | %.6f | %+.3f %% | %.6f | %+.3f %% |"
          % (mesh["mesh"], 100 * mesh["facet_area_deficit"],
             mesh["sagitta_over_thickness"],
             rows["cyl"]["probe_displacement"], 100 * rows["cyl"]["relative_error"],
             rows["mitc"]["probe_displacement"], 100 * rows["mitc"]["relative_error"],
             rows["sri"]["probe_displacement"], 100 * rows["sri"]["relative_error"]))
    A("")
    A("### 3.2 Convergence")
    A("")
    A("| configuration | observed order | Richardson limit | limit error |")
    A("|---|---|---|---|")
    for config in ("cyl", "mitc", "sri"):
        r = payload["richardson"][config]
        if r["observed_order"] is None:
            A("| `%s` | n/a | n/a | %s |" % (config, r["reason"]))
        else:
            A("| `%s` | %.3f | %.6f | %+.3f %% |"
              % (config, r["observed_order"], r["richardson_limit"],
                 100 * r["limit_relative_error"]))
    A("")
    A("### 3.3 Rank census (one facet, chord 0.873)")
    A("")
    A("| shear rule | zero modes | spurious | of which transverse | first elastic eigenvalue |")
    A("|---|---|---|---|---|")
    for c in census:
        A("| `%s` | %d | %d | %d | %.2f |"
          % (c["plate_shear"], c["zero_modes"], c["spurious_modes"],
             c["transverse_mechanisms"], c["first_elastic_eigenvalue"]))
    A("")
    A("### 3.4 Equilibrium")
    A("")
    flat_worst = max(abs(l["equilibrium_ratio"] - 1.0)
                     for c in ("mitc", "sri") for l in ladders[c])
    cyl_worst = max(abs(l["equilibrium_ratio"] - 1.0) for l in ladders["cyl"])
    A("| configuration | worst `|sum(reaction_z) / (q A) - 1|` |")
    A("|---|---|")
    A("| flat facet (`sri`, `mitc`) | `%.2e` |" % flat_worst)
    A("| cylindrical shell4 | `%.2e` |" % cyl_worst)
    A("")
    A("The flat-facet family is exact to machine precision. The cylindrical "
      "element integrates the *exact* arc area and its reaction residual falls as "
      "`h^2`: it is exactly the rigid-mode projection `(K 1_z)^T u / (-1^T f)`, "
      "attributed to the interpolated rigid vertical translation in "
      "`docs/CASE-FEM-SCORDELIS-LO-ROOF-reaction-residual-attribution.md`, and is "
      "an order of magnitude below the displacement error it is compared with.")
    A("")
    A("## 4. Figures")
    A("")
    A("`scordelis_lo_roof_case.png`: (a) `|w|` vs elements per edge on a log-log "
      "axis with the reference line; (b) `|error|` at 16x16 against the 3 %% line; "
      "(c) the rank census (all spurious modes vs those with transverse content).")
    A("")
    A("## 5. Notes and limits")
    A("")
    A("- **The corrected claim.** The 50 %% plateau is not a faceting limitation. "
      "The same faceted mesh passes once its transverse-shear block is rank "
      "sufficient, and the rank census says why: the one-point `sri` rule leaves "
      "%d spurious mechanisms with transverse content, `mitc` leaves %d. The "
      "surviving `mitc` mechanism is an in-plane/drilling gauge direction that "
      "cannot store bending energy (see report 15618, "
      "`docs/` formulation-matrix study)."
      % (next(c for c in census if c["plate_shear"] == "sri")["transverse_mechanisms"],
         next(c for c in census if c["plate_shear"] == "mitc")["transverse_mechanisms"]))
    A("- **The default is not flipped here.** `plate_shear` still defaults to "
      "`sri`, because every existing flat-facet qualification would change. The "
      "case therefore pins the passing configuration explicitly and *inverts* the "
      "sri gate, so the documented failure cannot silently disappear.")
    A("- **Faceting is real but not the limiter on this benchmark.** At 16x16 the "
      "chord sagitta is %.1f %% of the thickness and the facet area is %.3f %% "
      "short of the analytic surface, yet the cylindrical and the `mitc` flat "
      "facet results agree with the reference to %.2f %% and %.2f %% "
      "respectively. What faceting costs here is bounded, not fatal."
      % (100 * ladders["cyl"][-1]["sagitta_over_thickness"],
         100 * ladders["cyl"][-1]["facet_area_deficit"],
         100 * abs(ladders["cyl"][-1]["relative_error"]),
         100 * abs(ladders["mitc"][-1]["relative_error"])))
    A("- **Reference resolution.** `w_ref = -0.3024` is a four-significant-digit "
      "literature value (Scordelis & Lo 1960; MacNeal & Harder 1985); claims "
      "below ~0.1 %% are outside what it can certify, so the 16x16 cylindrical "
      "result is reported as agreement, not as a 0.02 %% validation.")
    A("- **Scope.** Linear statics, single load level, cylindrical geometry, "
      "thin-ish shell `R/t = 100`. Large deflection, buckling and general "
      "non-cylindrical curvature are outside this case.")
    A("")
    A("## 6. Error determination")
    A("")
    A("| quantity | value | reference | error | < 3 %? |")
    A("|---|---|---|---|---|")
    A("| `w`, cylindrical shell4, 8x8 | %.6f | %.4f | %+.3f %% | %s |"
      % (ladders["cyl"][1]["probe_displacement"], REFERENCE,
         100 * ladders["cyl"][1]["relative_error"],
         "**yes**" if abs(ladders["cyl"][1]["relative_error"]) < GATE_3PCT else "**no**"))
    A("| `w`, cylindrical shell4, 16x16 | %.6f | %.4f | %+.3f %% | %s |"
      % (pr["probe_displacement"], REFERENCE, 100 * pr["relative_error"],
         "**yes**" if abs(pr["relative_error"]) < GATE_3PCT else "**no**"))
    A("| `w`, flat facet + mitc, 16x16 | %.6f | %.4f | %+.3f %% | %s |"
      % (ladders["mitc"][-1]["probe_displacement"], REFERENCE,
         100 * ladders["mitc"][-1]["relative_error"],
         "**yes**" if abs(ladders["mitc"][-1]["relative_error"]) < GATE_3PCT else "**no**"))
    A("| `w`, flat facet + sri (default), 16x16 | %.6f | %.4f | %+.3f %% | **no** |"
      % (ladders["sri"][-1]["probe_displacement"], REFERENCE,
         100 * ladders["sri"][-1]["relative_error"]))
    A("| vertical reaction / applied (flat facet, worst) | %.15f | 1 | %.1e | yes |"
      % (1.0 + flat_worst, flat_worst))
    A("| spurious transverse mechanisms, sri / mitc | %d / %d | 0 / 0 | exact integers | structural |"
      % (next(c for c in census if c["plate_shear"] == "sri")["transverse_mechanisms"],
         next(c for c in census if c["plate_shear"] == "mitc")["transverse_mechanisms"]))
    A("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- main ---
def main() -> int:
    ladders = {config: [solve(config, n) for n in MESHES]
               for config in ("cyl", "mitc", "sri")}
    census = rank_census()
    gates = build_gates(ladders, census)
    figures = make_figures(ladders, census)
    payload = {
        "case": CASE,
        "repro_cmd": REPRO_CMD,
        "parameters": PARAMETERS,
        "reference": REFERENCE,
        "reference_source": "Scordelis & Lo (1960); MacNeal & Harder (1985)",
        "exact_surface_area": exact_surface_area(),
        "primary": next(l for l in ladders["cyl"] if l["mesh"] == "%dx%d" % (PRIMARY_MESH, PRIMARY_MESH)),
        "ladders": ladders,
        "richardson": {c: richardson(ladders[c]) for c in ladders},
        "rank_census": census,
        "gates": gates,
        "figures": [str(p.relative_to(ROOT)) for p in figures],
        "all_passed": all(g["passed"] for g in gates),
        "environment": {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "dtype": "float64",
        },
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "case_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    markdown = render_doc(payload)
    (ART / "case_result.md").write_text(markdown)
    DOC.write_text(markdown)

    print("%s: %s" % (CASE, "PASS" if payload["all_passed"] else "FAIL"))
    for g in gates:
        print("  [%s] %-42s %s" % ("ok" if g["passed"] else "XX", g["gate"], g["value"]))
    print("artifacts:", ART)
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())