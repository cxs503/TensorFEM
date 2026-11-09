"""Evidence generator for ``tensorfem.open_section_torsion``.

Produces ``docs/assets/open-section-torsion/open_section_torsion.json`` plus four
figures.  Every number in the JSON is recomputed here (no hand typing); the
report in ``docs/open-section-torsion.md`` quotes the JSON.

Run: ``OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/audit_open_section_torsion.py``
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem import box_torsion as bt  # noqa: E402
from tensorfem.marine_stiffener_failure import lateral_torsional_buckling_screen  # noqa: E402
from tensorfem.open_section_torsion import (  # noqa: E402
    angle_section, box_section, channel_section, fem_torsion_constant, flat_bar,
    i_section, prism_extrude, properties, rect_block_mesh, stiffener_properties,
    tee_section, weld_meshes, z_section)
from tensorfem.saint_venant_torsion import RectSection  # noqa: E402

OUT = pathlib.Path("docs/assets/open-section-torsion")
OUT.mkdir(parents=True, exist_ok=True)
H, B, TW, TF = 0.30, 0.12, 0.008, 0.012
result: dict = {"repo": "TensorFEM", "module": "tensorfem.open_section_torsion"}


def rel(a, b):
    return abs(a / b - 1.0)


# --------------------------------------------------------------------------- #
# 1. closed-form anchors
# --------------------------------------------------------------------------- #
i_p = properties(i_section(H, B, TW, TF))
c_p = properties(channel_section(H, B, 0.01, 0.01))
t_p = properties(tee_section(H, B, TW, TF))
l_p = properties(angle_section(0.10, 0.12, 0.008))
z_p = properties(z_section(H, B, TW, TF))
t = 0.01
anchors = {
    "i_section": {
        "cw": i_p.warping_constant, "cw_closed_form": TF * B ** 3 * H ** 2 / 24.0,
        "cw_rel_error": rel(i_p.warping_constant, TF * B ** 3 * H ** 2 / 24.0),
        "i_xx": i_p.i_xx,
        "i_xx_closed_form": 2.0 * TF * B * (0.5 * H) ** 2 + TW * H ** 3 / 12.0,
        "shear_centre": list(i_p.shear_centre), "centroid": list(i_p.centroid),
    },
    "channel": {
        "shear_centre": list(c_p.shear_centre),
        "shear_centre_closed_form": [-3.0 * B ** 2 / (6.0 * B + H), 0.5 * H],
        "shear_centre_rel_error": rel(abs(c_p.shear_centre[0]),
                                      3.0 * B ** 2 / (6.0 * B + H)),
        "cw": c_p.warping_constant,
        "cw_closed_form": (t * B ** 3 * H ** 2 / 12.0) * (3 * B + 2 * H) / (6 * B + H),
        "cw_rel_error": rel(c_p.warping_constant,
                            (t * B ** 3 * H ** 2 / 12.0) * (3 * B + 2 * H) / (6 * B + H)),
    },
    "tee": {"shear_centre": list(t_p.shear_centre), "cw": t_p.warping_constant,
            "cw_expected": 0.0},
    "angle": {"shear_centre": list(l_p.shear_centre), "cw": l_p.warping_constant,
              "cw_expected": 0.0},
    "z_section": {"shear_centre": list(z_p.shear_centre), "centroid": list(z_p.centroid),
                  "cw": z_p.warping_constant},
}
result["closed_form_anchors"] = anchors

# --------------------------------------------------------------------------- #
# 2. torsion constant: thin wall vs exact rectangle series vs 3-D HEX8
# --------------------------------------------------------------------------- #
ratios = [0.02, 0.05, 0.1, 0.2, 0.4]
bar_table = []
for r in ratios:
    width, thick = 0.10, 0.10 * r
    p = properties(flat_bar(width, thick))
    exact = RectSection(0.5 * width, 0.5 * thick).j
    bar_table.append({"t_over_b": r, "j_thin": p.j_thin, "j_rect": p.j_rect,
                      "j_exact": exact,
                      "thin_rel_error": rel(p.j_thin, exact),
                      "rect_rel_error": rel(p.j_rect, exact)})
result["flat_bar_thin_wall_limit"] = bar_table


def bar_mesh(width, thick, nw, nt, length, layers):
    nodes2d, quads = rect_block_mesh(np.linspace(0.0, width, nw + 1),
                                     np.linspace(0.0, thick, nt + 1))
    return prism_extrude(nodes2d, quads, length, layers)


def i_block_mesh(h, flange_b, web_t, flange_t, n_flange, n_web_thick, n_web,
                 n_web_depth, length, layers):
    """Conforming, overlap-free HEX8 I profile (flanges full width, web between)."""
    ytop, ybot = 0.5 * h, -0.5 * h
    xs_f = np.unique(np.concatenate([
        np.linspace(-0.5 * flange_b, -0.5 * web_t, n_flange + 1),
        np.linspace(-0.5 * web_t, 0.5 * web_t, n_web_thick + 1),
        np.linspace(0.5 * web_t, 0.5 * flange_b, n_flange + 1)]))
    top = rect_block_mesh(xs_f, np.linspace(ytop - flange_t, ytop, n_web + 1))
    bot = rect_block_mesh(xs_f, np.linspace(ybot, ybot + flange_t, n_web + 1))
    web = rect_block_mesh(np.linspace(-0.5 * web_t, 0.5 * web_t, n_web_thick + 1),
                          np.linspace(ybot + flange_t, ytop - flange_t, n_web_depth + 1))
    nodes2d, quads = weld_meshes([top, bot, web])
    return prism_extrude(nodes2d, quads, length, layers)


fem_bar = []
for nt in (1, 2):
    nodes, hexes = bar_mesh(0.10, 0.01, 8, nt, 0.5, 8)
    res = fem_torsion_constant(nodes, hexes, rtol=1e-9)
    exact = RectSection(0.05, 0.005).j
    fem_bar.append({"through_thickness": nt, **res, "j_exact": exact,
                    "rel_error": rel(res["j_fem"], exact)})
result["hex8_flat_bar"] = fem_bar

i_fem = []
p_i = properties(i_section(H, B, TW, TF))
for tag, args in (("coarse", dict(n_flange=2, n_web_thick=1, n_web=1, n_web_depth=3,
                                  layers=8)),
                  ):
    nodes, hexes = i_block_mesh(H, B, TW, TF, length=1.0, **args)
    try:
        res = fem_torsion_constant(nodes, hexes, rtol=1e-9)
        i_fem.append({"mesh": tag, **res, "j_thin": p_i.j_thin, "j_rect": p_i.j_rect,
                      "rel_error_vs_j_rect": rel(res["j_fem"], p_i.j_rect)})
    except RuntimeError as exc:                     # honest record, not a pass
        i_fem.append({"mesh": tag, "error": str(exc), "j_thin": p_i.j_thin,
                      "j_rect": p_i.j_rect})
result["hex8_i_section"] = i_fem

# --------------------------------------------------------------------------- #
# 3. closed-contour control (Bredt vs multiply connected Prandtl FD)
# --------------------------------------------------------------------------- #
boxes = []
for half, thick, step in ((0.11, 0.01, 0.0025), (0.11, 0.02, 0.005),
                          (0.11, 0.04, 0.01)):
    p = properties(box_section(2.0 * (half - 0.5 * thick), 2.0 * (half - 0.5 * thick),
                               thick))
    fd = bt.solve_box_prandtl(half, half, thick, step)
    boxes.append({"outer_half_dim": half, "wall": thick, "j_bredt": p.j_bredt,
                  "j_fd_energy": fd.j_energy, "j_fd_torque": fd.j_torque,
                  "bredt_vs_fd_energy": rel(p.j_bredt, fd.j_energy),
                  "bredt_vs_fd_torque": rel(p.j_bredt, fd.j_torque)})
result["closed_box_control"] = boxes

# --------------------------------------------------------------------------- #
# 4. ship payoff: stiffener LTB with computed J / Cw
# --------------------------------------------------------------------------- #
spans = np.linspace(1.5, 6.0, 10)
stiff = stiffener_properties(0.25, 0.010, 0.090, 0.012)
cyc = []
for L in spans:
    new = lateral_torsional_buckling_screen(
        segments=60, length=float(L), torsion_constant=stiff.j,
        warping_constant=stiff.warping_constant, weak_axis_inertia=stiff.i_yy)
    old = lateral_torsional_buckling_screen(
        segments=60, length=float(L), torsion_constant=1.2e-7,
        warping_constant=2.5e-8, weak_axis_inertia=8.0e-7)
    cyc.append({"span": float(L), "mcr_computed": new["critical_moment"],
                "mcr_placeholder": old["critical_moment"],
                "ratio": new["critical_moment"] / old["critical_moment"]})
result["stiffener_ltb"] = {
    "scantlings": {"web_depth": 0.25, "web_t": 0.010,
                   "flange_width": 0.090, "flange_t": 0.012},
    "j": stiff.j, "j_thin": stiff.j_thin, "cw": stiff.warping_constant,
    "i_yy": stiff.i_yy, "shear_centre": list(stiff.shear_centre),
    "span_sweep": cyc,
    "placeholder_constants": {"torsion_constant": 1.2e-7, "warping_constant": 2.5e-8},
}

# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def draw_section(ax, profile, props, title):
    nodes = np.asarray(profile["nodes"], dtype=float)
    w = np.asarray(props.node_warping)
    norm = np.abs(w).max() or 1.0
    cmap = plt.get_cmap("coolwarm")
    for a, b, thick in profile["edges"]:
        ax.plot([nodes[a, 0], nodes[b, 0]], [nodes[a, 1], nodes[b, 1]],
                color=cmap(0.5 + 0.5 * w[a] / norm), lw=4.0)
        ax.plot([nodes[a, 0], nodes[b, 0]], [nodes[a, 1], nodes[b, 1]],
                color="k", lw=0.6, alpha=0.5)
    sc = props.shear_centre
    ax.plot(*sc, marker="o", ms=6, mfc="none", mec="k")
    ax.plot(*props.centroid, marker="+", ms=7, color="k")
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n$C_w$={props.warping_constant:.3e}", fontsize=8)
    ax.tick_params(labelsize=6)


fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
for ax, prof, prop, tag in zip(
        axes,
        [i_section(H, B, TW, TF), channel_section(H, B, 0.01, 0.01),
         tee_section(H, B, TW, TF), angle_section(0.10, 0.12, 0.008),
         z_section(H, B, TW, TF)],
        [i_p, c_p, t_p, l_p, z_p],
        ["I", "channel", "T", "angle", "Z"]):
    draw_section(ax, prof, prop, tag)
fig.suptitle("Open thin-walled profiles: node warping (colour), centroid (+), "
             "shear centre (o)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "sections.png", dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.6))
tt = np.array([row["t_over_b"] for row in bar_table])
ax.loglog(tt, [row["thin_rel_error"] for row in bar_table], "o-",
          label="thin-wall $bt^3/3$")
ax.loglog(tt, [row["rect_rel_error"] for row in bar_table], "s--",
          label="rectangle series $J_{rect}$")
ax.loglog(tt, 0.63 * tt / 3.0, ":", color="grey", label="$0.63\\,t/3b$")
ax.set_xlabel("$t/b$")
ax.set_ylabel("relative error vs exact series")
ax.set_title("open-section torsion constant: thin-wall error", fontsize=9)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "thin_wall_limit.png", dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.6))
nts = [row["through_thickness"] for row in fem_bar]
ax.semilogx(nts, [row["j_fem"] / row["j_exact"] for row in fem_bar], "o-",
            label="HEX8 $J/T$ vs exact series")
ax.axhline(1.0, color="k", lw=0.8, label="exact")
ax.axhline(0.10 * 0.01 ** 3 / 3.0 / RectSection(0.05, 0.005).j, color="r",
           ls="--", lw=0.8, label="thin-wall $bt^3/3$")
ax.set_xlabel("elements through the thickness")
ax.set_ylabel("$J_{FE}/J_{exact}$")
ax.set_title("flat bar 100x10 mm, 8x8x8 HEX8", fontsize=9)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fem_flat_bar.png", dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.plot([c["span"] for c in cyc], [c["mcr_computed"] / 1e6 for c in cyc], "o-",
        label="computed $J$, $C_w$ (250x10/90x12 T)")
ax.plot([c["span"] for c in cyc], [c["mcr_placeholder"] / 1e6 for c in cyc], "s--",
        label="module placeholder constants")
ax.set_xlabel("unbraced span (m)")
ax.set_ylabel("$M_{cr}$ (MN m)")
ax.set_title("stiffener lateral-torsional buckling", fontsize=9)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "stiffener_ltb.png", dpi=140)
plt.close(fig)

(OUT / "open_section_torsion.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: result[k] for k in ("closed_form_anchors", "hex8_i_section",
                                         "closed_box_control")}, indent=2)[:2600])
print("wrote", OUT)
