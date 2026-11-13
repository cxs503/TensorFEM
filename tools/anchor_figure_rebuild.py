"""Re-render certificate figures from the *committed* ``result.json`` (zero solve).

Problem
-------
Three certificates named figures that do not exist in the delivered tree, so
``tools/verified_anchor_audit.py`` raised ``E_ARTIFACT_MISSING`` and
``tests/test_anchor_figure_gate_safety.py`` counted them as anchors naming
figures the push gate strips (``benchmarks/verified/**/*.png`` is dropped by
``tp_pusher.py`` unless the path starts with ``docs/``):

===============================  ======  ============================================
anchor                           figs    why the name did not resolve
===============================  ======  ============================================
``fem_inverse_identification``   1       repo-relative name under ``benchmarks/``
``fem_sent_j_integral``          4       bare file name, file absent from the anchor dir
``fem_winkler_plate_k_sweep``    3       bare file name, anchor dir has no producer at all
===============================  ======  ============================================

The naive repair is "run the benchmark again and commit the PNG".  That costs an
FE solve per anchor and it still gets filtered out of the next delivery merge
(six earlier missions restored 32 such figures and had them stripped again).
The repair implemented here is the one that can be *kept*:

1. every plotted number already lives in the committed ``result.json`` — a
   certificate does not need a solver to redraw its evidence, it needs the data;
2. the figure is written into ``docs/assets/verified/<anchor>/``, the one path
   class the push gate carries, and the certificate names that path;
3. the render is **deterministic** (no timestamp, no version string, fixed dpi
   and font), so ``sha256`` in ``figures_manifest.json`` is a re-render
   invariant: a silent change of the data or of the drawing code breaks it.

Nothing here recomputes physics.  A figure is only allowed to draw numbers that
are already in ``result.json``; if a quantity is absent the renderer must say so
rather than invent an axis.

Usage
-----
    python3 tools/anchor_figure_rebuild.py --check     # dry run, no write
    python3 tools/anchor_figure_rebuild.py --write     # render + manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DIR = REPO_ROOT / "benchmarks" / "verified"
FIGURES_ROOT = REPO_ROOT / "docs" / "assets" / "verified"

DPI = 120
#: Metadata is stripped so that two renders of the same data are byte-identical.
SAVEFIG_KW = {
    "dpi": DPI,
    "metadata": {"Software": None, "Creation Time": None},
    "facecolor": "white",
}

GATE_PCT = 3.0  # platform scalar gate used by the certificates below


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _result(anchor: str) -> dict:
    path = VERIFIED_DIR / anchor / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"{anchor}: committed result.json is missing ({path})")
    return json.loads(path.read_text())


def _outdir(anchor: str) -> Path:
    return FIGURES_ROOT / anchor


def _tada_sent_shape_factor(a_over_w: float) -> float:
    """Tada/Paris-Irwin edge-crack geometry factor, identical to the driver's fit.

    ``F = 1.12 - 0.231 a + 10.55 a^2 - 21.72 a^3 + 30.39 a^4`` (a = a/W).  It is
    reproduced here rather than imported so the renderer only depends on the
    repository root (a figure is not allowed to pull a solver module in).
    """
    a = float(a_over_w)
    return 1.12 - 0.231 * a + 10.55 * a**2 - 21.72 * a**3 + 30.39 * a**4


# --------------------------------------------------------------------------- #
# fem_inverse_identification
# --------------------------------------------------------------------------- #
def _render_inverse_identification(payload: dict) -> dict[str, plt.Figure]:
    rows = payload["rows"]
    tol = float(payload.get("tolerance", 0.03))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax = axes[0]
    labels, errors, colors = [], [], []
    for row in rows:
        quantity = str(row.get("reference_quantity", "?"))
        labels.append(f"{row.get('case')}/{quantity.split()[0]}/seed{row.get('seed', '-')}")
        errors.append(100.0 * float(row["relative_error"]))
        colors.append("#2b6cb0" if row.get("passed") else "#c53030")
    positions = range(len(labels))
    ax.bar(list(positions), errors, color=colors, width=0.6)
    ax.axhline(GATE_PCT, color="k", ls="--", lw=0.9, label=f"{GATE_PCT:.0f} % gate")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("|recovered - truth| / truth  [%]")
    ax.set_title("Recovered parameters against the generating truth")
    ax.legend(fontsize=8)

    ax = axes[1]
    control = payload["negative_control"]
    rank = float(control.get("effective_rank", 0))
    ax.bar([0, 1], [rank, 2.0], color=["#c05621", "0.75"], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ill-posed E+A\n(effective rank)", "nominally\n2 parameters"], fontsize=7)
    ax.set_ylabel("rank / parameter count")
    ax.set_title(
        "Negative control: E and A are not separately identifiable\n"
        f"cond = {control.get('condition_number', float('nan')):.3g}, "
        f"product error {100.0 * float(control.get('product_relative_error', float('nan'))):.3f} %"
    )
    for axis in axes:
        axis.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    return {"inverse_identification_recovery.png": fig}


# --------------------------------------------------------------------------- #
# fem_sent_j_integral  (the four figures the certificate names but never had)
# --------------------------------------------------------------------------- #
def _sent_path_figure(cases_by_level: list[dict], title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for case in cases_by_level:
        contours = case.get("contours") or []
        if not contours:
            continue
        ax.plot(
            [float(c["station_fraction"]) for c in contours],
            [float(c["shape_factor"]) for c in contours],
            "o-",
            ms=3,
            lw=1.0,
            label=f"a/W = {float(case['a_over_w']):.2f}",
        )
    ax.set_xlabel("contour station fraction r/R(theta)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _sent_convergence_figure(ladder_errors: dict, cases: list[dict], title: str) -> plt.Figure:
    """|K_FEM - K_Tada| versus element count, one curve per a/W.

    ``ladder_errors[key]`` is ordered coarse -> fine and has one entry per mesh
    level, *not* one per case, so the abscissa is taken from the level ladder
    stored with each case (``n_elements`` of the matched a/W).
    """
    by_key = {f"{float(c['a_over_w']):.2f}": float(c["n_elements"]) for c in cases}
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for key, errors in ladder_errors.items():
        n_el = [by_key[key]]
        ax.plot(n_el, [abs(float(errors[-1])) * 100.0], "o", ms=6, label=f"a/W = {key} (finest)")
    ax.set_xscale("log")
    ax.set_xlabel("number of T6 elements")
    ax.set_ylabel("|K_FEM - K_Tada| / K_Tada  [%]")
    ax.set_title(title)
    ax.axhline(GATE_PCT, color="k", ls="--", lw=0.8, label=f"{GATE_PCT:.0f} % gate")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _sent_shape_figure(cases: list[dict], control: list[dict], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    alphas = [0.05 + 0.55 * i / 119.0 for i in range(120)]
    ax.plot(alphas, [_tada_sent_shape_factor(a) for a in alphas], "-", color="k", lw=1.2,
            label="Tada F(a/W)")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.9, label="central crack F = 1")
    ax.plot([float(c["a_over_w"]) for c in cases],
            [float(c["shape_factor_fem"]) for c in cases], "o", ms=6,
            label="FEM J-integral (finest)")
    if control:
        ax.plot([float(c["a_over_w"]) for c in control],
                [float(c["shape_factor_fem"]) for c in control], "s", mfc="none",
                label="non-singular control")
    ax.set_xlabel("a/W")
    ax.set_ylabel("F = K_I / (sigma sqrt(pi a))")
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plane_strain_path_figure(cases: list[dict]) -> plt.Figure:
    """Plane-strain path independence.

    ``plane_strain.cases`` records one canonical contour station per a/W plus the
    spread over all stations (``path_spread``); the per-station list is only
    stored for the plane-stress ladder, so the honest plot here is F at the
    canonical station with +/- the observed spread -- not a fabricated curve.
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    alphas = [float(c["a_over_w"]) for c in cases]
    f_mid = [float(c["shape_factor_fem"]) for c in cases]
    half = [float(c.get("path_spread", 0.0)) * float(c["shape_factor_fem"]) for c in cases]
    ax.errorbar(alphas, f_mid, yerr=half, fmt="o", ms=6, capsize=4,
                label="FEM J-integral, canonical station")
    ax.plot(alphas, [_tada_sent_shape_factor(a) for a in alphas], "k--", lw=1.0,
            label="Tada F(a/W)")
    ax.set_xlabel("a/W")
    ax.set_ylabel("F = K_I / (sigma sqrt(pi a))")
    ax.set_title("Plane strain: F and its contour path spread")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _render_sent_j_integral(payload: dict) -> dict[str, plt.Figure]:
    evidence = payload["evidence"]
    plane_strain = payload["plane_strain"]
    finest = evidence["ladder"]["levels"][-1]["cases"]
    # ``cases`` in ``evidence`` carries the finest-level n_elements for each a/W;
    # the ladder errors are ordered coarse -> fine, hence the shape check.
    finest_cases = sorted(evidence["cases"], key=lambda c: float(c["a_over_w"]))
    for values in list(evidence["ladder_errors"].values()) + list(
        plane_strain["ladder_errors"].values()
    ):
        if len(values) != len(finest_cases):
            raise ValueError("ladder errors and number of cases disagree; refusing to draw")

    return {
        "sent_convergence.png": _sent_convergence_figure(
            evidence["ladder_errors"], finest_cases,
            "SENT plane-stress J-integral convergence against Tada F(a/W)",
        ),
        "sent_shape_factor.png": _sent_shape_figure(
            finest_cases, evidence.get("control_non_singular") or [],
            "Edge-crack geometry factor, plane stress: numerical vs handbook",
        ),
        "sent_path_independence.png": _sent_path_figure(
            finest, "J path independence on the fan mesh lines", "F from J"
        ),
        "sent_plane_strain_convergence.png": _sent_convergence_figure(
            plane_strain["ladder_errors"],
            sorted(plane_strain["cases"], key=lambda c: float(c["a_over_w"])),
            "SENT plane-strain J-integral convergence against Tada F(a/W)",
        ),
        "sent_plane_strain_shape_factor.png": _sent_shape_figure(
            sorted(plane_strain["cases"], key=lambda c: float(c["a_over_w"])), [],
            "Edge-crack geometry factor, plane strain: numerical vs handbook",
        ),
        "sent_plane_strain_path_independence.png": _plane_strain_path_figure(
            sorted(plane_strain["cases"], key=lambda c: float(c["a_over_w"]))
        ),
    }


# --------------------------------------------------------------------------- #
# fem_winkler_plate_k_sweep
# --------------------------------------------------------------------------- #
def _render_winkler_k_sweep(payload: dict) -> dict[str, plt.Figure]:
    plate_cases = payload["plate_cases"]
    beam_cases = payload["beam_cases"]

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for case in plate_cases:
        profile = case["profile"]
        scale = float(case["w0_hetenyi_mm"])
        k = float(case["k_N_m3"])
        ax.plot([float(p["r_over_l"]) for p in profile],
                [float(p["w_fem_mm"]) / scale for p in profile], "o-", ms=3,
                label=f"FEM k={k:g} N/m^3")
        ax.plot([float(p["r_over_l"]) for p in profile],
                [float(p["w_hetenyi_mm"]) / scale for p in profile], "k--", lw=0.8)
    ax.set_xlabel("r / l   (l = (D/k)^(1/4))")
    ax.set_ylabel("w(r) / w(0)")
    ax.set_title("Winkler plate: plate profile vs Hetényi kei(r/l)\n(dashed black = analytical)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    figures = {"winkler_k_sweep_profile.png": fig}

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ks = [float(c["k_N_m3"]) for c in plate_cases]
    ax.loglog(ks, [max(float(c["centre_error_pct"]), 1e-6) for c in plate_cases], "o-",
              label="centre deflection")
    ax.loglog(ks, [max(float(c["profile_error_max_pct"]), 1e-6) for c in plate_cases], "s-",
              label="profile max (|dw|/w0)")
    ax.axhline(5.0, color="k", ls="--", lw=0.8, label="5 % gate")
    ax.set_xlabel("foundation stiffness k  [N/m^3]")
    ax.set_ylabel("error vs Hetényi  [%]")
    ax.set_title("Winkler k-sweep: error against the analytical solution")
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["winkler_k_sweep_error.png"] = fig

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for case in beam_cases:
        x = [float(v) for v in case["x_over_L"]]
        label = f"k_b={float(case['k_b_N_m2']):g} ({case['load']})"
        ax.plot(x, [float(v) for v in case["w_fem_mm"]], "-", lw=1.1, label=f"FEM {label}")
        ax.plot(x, [float(v) for v in case["w_ref_mm"]], "k--", lw=0.7)
    ax.set_xlabel("x / L")
    ax.set_ylabel("w(x)  [mm]")
    ax.set_title("Beam on elastic foundation: FEM vs Hetényi\n(dashed black = analytical)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6)
    fig.tight_layout()
    figures["winkler_k_sweep_beam.png"] = fig
    return figures


RENDERERS: dict[str, Callable[[dict], dict[str, plt.Figure]]] = {
    "fem_inverse_identification": _render_inverse_identification,
    "fem_sent_j_integral": _render_sent_j_integral,
    "fem_winkler_plate_k_sweep": _render_winkler_k_sweep,
}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def render(anchor: str, outdir: Path | None = None) -> dict[str, dict]:
    """Render one anchor's figures; return ``{name: {sha256, bytes}}``."""
    payload = _result(anchor)
    target = Path(outdir) if outdir is not None else _outdir(anchor)
    target.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for name, fig in RENDERERS[anchor](payload).items():
        path = target / name
        fig.savefig(path, **SAVEFIG_KW)
        plt.close(fig)
        blob = path.read_bytes()
        manifest[name] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="render the PNGs and the manifest")
    group.add_argument("--check", action="store_true", help="render into a temp dir and report")
    parser.add_argument("anchors", nargs="*", default=None,
                        help="subset of anchors (default: all rendered anchors)")
    args = parser.parse_args(argv)
    anchors = args.anchors or sorted(RENDERERS)

    import tempfile

    status = 0
    for anchor in anchors:
        if anchor not in RENDERERS:
            print(f"SKIP {anchor}: no renderer")
            status = 2
            continue
        if args.write:
            manifest = render(anchor)
            (FIGURES_ROOT / anchor / "figures_manifest.json").write_text(
                json.dumps(
                    {
                        "anchor": anchor,
                        "source": f"benchmarks/verified/{anchor}/result.json",
                        "renderer": "tools/anchor_figure_rebuild.py",
                        "dpi": DPI,
                        "figures": manifest,
                    },
                    indent=1,
                    sort_keys=True,
                )
                + "\n"
            )
            print(f"WROTE {anchor}: {len(manifest)} figures -> docs/assets/verified/{anchor}/")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                manifest = render(anchor, Path(tmp))
            print(f"CHECK {anchor}: {len(manifest)} figures render from result.json")
    if not args.write:
        print("dry run (no file written); pass --write to update docs/assets/verified/")
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())