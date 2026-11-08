#!/usr/bin/env python3
"""Generate the effective-modal-mass verification evidence (no hand-copied numbers).

Writes
------
``docs/modal-participation-evidence.json``
    Every number quoted in ``docs/modal-participation-effective-mass.md``.
``docs/assets/modal-participation/cumulative-mass.png``
    Cumulative effective-mass fraction against the closed-form reference.
``docs/assets/modal-participation/mass-defect.png``
    Discrete rigid-body inertia ``r^T M r / (mu L)`` versus element count.

Usage::

    PYTHONPATH=src python scripts/gen_modal_participation_evidence.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorfem.modal import _CLAMPED_ROOTS, _CANTILEVER_ROOTS, uniform_beam_matrices
from tensorfem.modal_participation import (
    _characteristic_root,
    beam_effective_mass_reference,
    modal_participation,
    translation_influence,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets" / "modal-participation"
LENGTH = 1.0
EI = 1.0
MU = 1.0
LEVELS = (8, 16, 32, 64, 128)
MODES = 6


def _cantilever(ne: int, modes: int | None = MODES):
    K, M = uniform_beam_matrices(LENGTH, ne, EI, MU)
    ndof = 2 * (ne + 1)
    free = [i for i in range(ndof) if i not in (0, 1)]
    influence = translation_influence(ndof, free[0::2], dtype=torch.float64)
    return modal_participation(K, M, influence, (0, 1), modes), K, M, influence, free


def main() -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)
    reference = beam_effective_mass_reference("cantilever", 10)
    reference_cc = beam_effective_mass_reference("clamped_clamped", 10)
    payload: dict = {
        "schema": "tensorfem.modal-participation/1",
        "beam": {"length": LENGTH, "EI": EI, "mu": MU, "element": "Hermite cubic"},
        "reference": {
            "cantilever_effective_mass_fraction": [float(v) for v in reference],
            "cantilever_sum_first_10": float(reference.sum()),
            "clamped_clamped_effective_mass_fraction": [float(v) for v in reference_cc],
            "clamped_clamped_sum_first_10": float(reference_cc.sum()),
            "cantilever_roots_vs_tabulated": [
                {
                    "mode": n,
                    "computed": float(_characteristic_root("cantilever", n)),
                    "tabulated": tabulated,
                    "rel_error": abs(
                        _characteristic_root("cantilever", n) - tabulated
                    ) / tabulated,
                }
                for n, tabulated in enumerate(_CANTILEVER_ROOTS, start=1)
            ],
            "clamped_roots_vs_tabulated": [
                {
                    "mode": n,
                    "computed": float(_characteristic_root("clamped_clamped", n)),
                    "tabulated": tabulated,
                    "rel_error": abs(
                        _characteristic_root("clamped_clamped", n) - tabulated
                    ) / tabulated,
                }
                for n, tabulated in enumerate(_CLAMPED_ROOTS, start=1)
            ],
        },
        "levels": [],
        "completeness": [],
        "free_free": {},
        "clamped_clamped_fe": {},
        "static_truncation": [],
    }

    for ne in LEVELS:
        summary, K, M, influence, free = _cantilever(ne)
        fractions = summary.effective_mass_fraction.tolist()
        physical = summary.fraction_of(MU * LENGTH).tolist()
        rel = [
            abs(physical[i] - float(reference[i])) / float(reference[i]) for i in range(MODES)
        ]
        F = torch.zeros(2 * (ne + 1), dtype=torch.float64)
        F[2 * ne] = 1.0
        coefficients = summary.modes.T @ F
        exact = LENGTH ** 3 / (3.0 * EI)
        static = [
            float(
                summary.modes[2 * ne, :n] @ (coefficients[:n] / summary.angular_frequencies[:n] ** 2)
            )
            / exact
            for n in range(1, MODES + 1)
        ]
        payload["levels"].append(
            {
                "elements": ne,
                "frequencies_hz": summary.frequencies_hz.tolist()[:MODES],
                "effective_mass_fraction": fractions,
                "physical_mass_fraction": physical,
                "relative_error": rel,
                "total_modal_mass": summary.total_modal_mass,
                "rigid_inertia_ratio": summary.total_modal_mass / (MU * LENGTH),
                "mass_normalisation_error": summary.mass_normalisation_error,
                "modes_for_90pct": summary.mode_count_for(0.9),
                "static_tip_ratio": static,
            }
        )

        # completeness identity: retain every free DOF
        complete, *_ = _cantilever(ne, modes=None)
        payload["completeness"].append(
            {
                "elements": ne,
                "modes": complete.mode_count,
                "sum_effective_mass": float(complete.effective_mass.sum()),
                "total_modal_mass": complete.total_modal_mass,
                "relative_identity_error": abs(
                    float(complete.effective_mass.sum()) - complete.total_modal_mass
                )
                / complete.total_modal_mass,
            }
        )

    # clamped--clamped: antisymmetric modes must carry zero participation
    ne = 64
    Kc, Mc = uniform_beam_matrices(LENGTH, ne, EI, MU)
    ndof = 2 * (ne + 1)
    consolidated = (0, 1, ndof - 2, ndof - 1)
    free_cc = [i for i in range(ndof) if i not in consolidated]
    influence_cc = translation_influence(ndof, free_cc[0::2], dtype=torch.float64)
    cc = modal_participation(Kc, Mc, influence_cc, consolidated, MODES)
    cc_analytic = beam_effective_mass_reference("clamped_clamped", MODES)
    payload["clamped_clamped_fe"] = {
        "elements": ne,
        "frequencies_hz": cc.frequencies_hz.tolist(),
        "effective_mass_fraction": cc.effective_mass_fraction.tolist(),
        "analytic_fraction": [float(v) for v in cc_analytic],
        "relative_error": [
            abs(float(cc.effective_mass_fraction[i]) - float(cc_analytic[i])) / float(cc_analytic[i])
            for i in range(0, MODES, 2)
        ],
        "antisymmetric_max_fraction": float(cc.effective_mass_fraction[1::2].abs().max()),
    }

    # free--free: the rigid-body mode must absorb the whole participation
    K, M = uniform_beam_matrices(LENGTH, ne, EI, MU)
    free_all = list(range(ndof))
    influence_all = translation_influence(ndof, free_all[0::2], dtype=torch.float64)
    payload["free_free"] = []
    for ne_ff in (16, 32, 64):
        K_ff, M_ff = uniform_beam_matrices(LENGTH, ne_ff, EI, MU)
        inf_ff = translation_influence(2 * (ne_ff + 1), list(range(0, 2 * (ne_ff + 1), 2)),
                                       dtype=torch.float64)
        ff = modal_participation(K_ff, M_ff, inf_ff, (), 4)
        payload["free_free"].append({
            "elements": ne_ff,
            "frequencies_hz": ff.frequencies_hz.tolist(),
            "physical_mass_fraction": ff.fraction_of(MU * LENGTH).tolist(),
            "rigid_body_candidates": list(ff.rigid_body_candidates(1e-3)),
            "elastic_participation_max": float(ff.participation_factor.abs().max()),
            "mass_normalisation_error": ff.mass_normalisation_error,
        })

    out = DOCS / "modal-participation-evidence.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")

    _plot(payload, reference)
    print(json.dumps({
        "reference_first3": payload["reference"]["cantilever_effective_mass_fraction"][:3],
        "levels": [
            {
                "ne": level["elements"],
                "rel_err_pct": [round(100.0 * v, 5) for v in level["relative_error"]],
                "rigid_inertia_ratio": level["rigid_inertia_ratio"],
                "modes_for_90pct": level["modes_for_90pct"],
                "static_tip_ratio": [round(v, 6) for v in level["static_tip_ratio"]],
                "mass_norm": level["mass_normalisation_error"],
            }
            for level in payload["levels"]
        ],
        "completeness": payload["completeness"],
        "free_free": payload["free_free"],
        "cc": payload["clamped_clamped_fe"],
        "ne64": payload["levels"][3],
        "roots_rel_err": [r["rel_error"] for r in payload["reference"]["cantilever_roots_vs_tabulated"]]
        + [r["rel_error"] for r in payload["reference"]["clamped_roots_vs_tabulated"]],
    }, indent=1))


def _plot(payload: dict, reference: torch.Tensor) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    modes = list(range(1, len(reference) + 1))
    ax.plot(modes, reference.cumsum(0).tolist(), "k--", marker="s", label="closed form")
    for level in payload["levels"]:
        values = level["physical_mass_fraction"]
        ax.plot(
            list(range(1, len(values) + 1)),
            list(torch.tensor(values).cumsum(0).tolist()),
            marker="o",
            label=f"FE {level['elements']} el.",
        )
    ax.axhline(0.9, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("mode number")
    ax.set_ylabel(r"cumulative $m_{eff}/(\mu L)$")
    ax.set_title("Cantilever: cumulative effective modal mass")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "cumulative-mass.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    elements = [level["elements"] for level in payload["levels"]]
    ratios = [level["rigid_inertia_ratio"] for level in payload["levels"]]
    ax.plot(elements, ratios, marker="o", label=r"$r^\top M r/(\mu L)$")
    ax.plot(elements, [1.0 - 0.6286 / n for n in elements], "k--", lw=0.9,
            label=r"$1-0.6286/n_{el}$ fit")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("elements")
    ax.set_ylabel("discrete rigid-body inertia / beam mass")
    ax.set_title("Hermite consistent mass: clamped-end mass defect")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(ASSETS / "mass-defect.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()