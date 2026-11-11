#!/usr/bin/env python3
"""Assemble the GAP-FEM-ADAPTIVE evidence pack from the generated study payloads.

The point of this script is *provenance*: every number in the benchmark table is read from
the JSON written by the study scripts, never retyped, and the pack records the sha256 of
each payload plus the git commit and the interpreter/array-library versions, so a claim can
be traced back to the run that produced it.  Rows that fail their criterion are printed as
``FAIL`` and kept in the table (honest-FAIL is not deleted).

Inputs (produced by, in this order)

    PYTHONPATH=src python scripts/adaptive_zz_study.py
    PYTHONPATH=src python scripts/adaptive_continuation_study.py

Outputs

    artifacts/adaptive_zz/benchmark_table.md
    artifacts/adaptive_zz/evidence.json

Usage
-----
    PYTHONPATH=src python scripts/adaptive_evidence_pack.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ZZ = ART / "adaptive_zz"
CONT = ART / "adaptive_continuation"

#: literature anchors for every reference value used below
ANCHORS = {
    "williams_lambda": (
        "M. L. Williams, 'Stress singularities resulting from various boundary conditions in "
        "angular corners of plates in extension', J. Appl. Mech. 19 (1952) 526-528: "
        "lambda = 0.5444837367825 for a 2*alpha = 3*pi/2 re-entrant corner"
    ),
    "uniform_rate": (
        "classical h-convergence on quasi-uniform meshes for a corner singularity: "
        "||u-u_h||_E <= C N^(-lambda/2) (Szabo & Babuska, Finite Element Analysis, 1991, ch. 6); "
        "lambda from Williams (1952)"
    ),
    "optimal_rate": (
        "best N-term approximation bound for the energy norm on optimally graded 2D meshes: "
        "||u-u_h||_E <= C N^(-1/2) (Binev, Dahmen & DeVore, SIAM J. Numer. Anal. 42 (2004) 1163; "
        "Stevenson, Math. Comp. 76 (2007) 27)"
    ),
    "effectivity": (
        "Zienkiewicz & Zhu, 'A simple error estimator and adaptive procedure for practical "
        "engineering analysis', Int. J. Numer. Methods Eng. 24 (1987) 337-357, and the "
        "superconvergent patch recovery of Zienkiewicz & Zhu, Int. J. Numer. Methods Eng. "
        "33 (1992) 1331-1364: theta = estimated/true error -> 1 as h -> 0 on smooth fields"
    ),
    "marking": (
        "W. Doerfler, 'A convergent adaptive algorithm for Poisson's equation', "
        "SIAM J. Numer. Anal. 33 (1996) 1106-1124 (bulk marking carries a fixed error fraction)"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unavailable"


def rel_error(got: float, reference: float) -> float:
    return abs(got - reference) / abs(reference) if reference else float("nan")


def row(quantity, reference, got, tolerance, anchor, note=""):
    error = rel_error(got, reference)
    verdict = "PASS" if error <= tolerance else "FAIL"
    return {
        "quantity": quantity,
        "reference": reference,
        "measured": got,
        "rel_error": error,
        "tolerance": tolerance,
        "verdict": verdict,
        "anchor": anchor,
        "note": note,
    }


def build_rows() -> tuple[list[dict], dict]:
    zz = json.loads((ZZ / "study.json").read_text())
    crit = zz["criteria"]
    ext = zz["extended_adaptive"]["asymptotics"]

    rows: list[dict] = []
    rows.append(
        row(
            "Williams corner exponent lambda",
            0.5444837367825,
            zz["williams_lambda"],
            1e-9,
            ANCHORS["williams_lambda"],
            "measured from the eigenvalue equation, not assumed",
        )
    )
    rows.append(
        row(
            "uniform ladder tail rate (L-shape, last 3 intervals)",
            crit["theory_uniform_rate"],
            crit["uniform_tail_rate"],
            0.15,
            ANCHORS["uniform_rate"],
            f"global slope over the whole ladder = {crit['uniform_rate']:.4f}",
        )
    )
    rows.append(
        row(
            "adaptive ladder tail rate (L-shape, last 6 intervals)",
            crit["theory_optimal_rate"],
            crit["extended_tail_rate"],
            0.15,
            ANCHORS["optimal_rate"],
            "global slope of the same ladder "
            f"{crit['extended_adaptive_rate']:.4f} is pre-asymptotic and must not be quoted",
        )
    )
    # ZZ effectivity: the criterion is the interval [0.8, 1.2], scored through the worst level
    all_eta = list(ext["effectivities"])
    all_eta += [lv["effectivity"] for lv in zz["adaptive"]["levels"]]
    all_eta += [lv["effectivity"] for lv in zz["uniform"]["levels"]]
    worst = max(all_eta, key=lambda value: abs(value - 1.0))
    # |theta - 1| <= 0.2 is exactly theta in [0.8, 1.2], so the interval criterion is scored
    # as the worst level against the exact value 1 with a 20 % relative tolerance
    rows.append(
        row(
            "ZZ effectivity theta, worst of "
            f"{len(all_eta)} L-shape levels (criterion theta in [0.8, 1.2])",
            1.0,
            worst,
            0.2,
            ANCHORS["effectivity"],
            f"theta range {min(all_eta):.4f} - {max(all_eta):.4f}; worst level is the one "
            f"farthest from 1, so the row is the interval test",
        )
    )
    rows.append(
        row(
            "FE energy norm vs closed-form exact energy norm (finest uniform level)",
            crit["exact_energy_norm"],
            crit["fe_energy_norm"],
            0.03,
            "exact energy norm of the Williams field integrated analytically in radius and by "
            "Gauss-Legendre panels in angle (this module: exact_energy_norm_sq)",
            "the discrete solution cannot exceed the exact energy (Galerkin best approximation)",
        )
    )
    rows.append(
        row(
            "DOF saving at equal accuracy, adaptive vs uniform (2 rounds)",
            crit["dof_saving_at_equal_error"],
            crit["dof_saving_at_equal_error"],
            0.0,
            "self-consistency row: reference is the measured value, reported for the cost claim",
            f"uniform error at matched DOF {crit['final_uniform_error_at_matched_dofs']:.5f} vs "
            f"adaptive {crit['final_adaptive_error']:.5f}",
        )
    )

    poisson = json.loads((CONT / "study_2d.json").read_text())
    rates = poisson["rates_last_two"]
    theory = poisson["theory_rates"]
    for key, label in (("energy", "energy norm"), ("l2", "L2 norm")):
        rows.append(
            row(
                f"2D Q4 Poisson {label} order (smooth manufactured field, finest pair)",
                theory[key],
                rates[key],
                0.05,
                ANCHORS["uniform_rate"].split(";")[0],
                f"asymptotic (last three rungs) {poisson['rates_asymptotic'][key]:.4f}",
            )
        )
    rows.append(
        row(
            "2D Q4 Poisson recovered-gradient effectivity (1521 unknowns, criterion theta in [0.8, 1.2])",
            1.0,
            poisson["criteria"]["final_effectivity"],
            0.2,
            ANCHORS["effectivity"],
            f"range {poisson['criteria']['effectivity_range'][0]:.4f} - "
            f"{poisson['criteria']['effectivity_range'][1]:.4f} over the ladder",
        )
    )

    one_d = json.loads((CONT / "study_1d.json").read_text())
    for name, label in (("sinusoidal", "u = sin(pi x)"), ("power", "u = x^0.6")):
        problem = one_d["problems"][name]
        rows.append(
            row(
                f"1D P1 uniform {label} energy order (closed-form discrete error)",
                abs(problem["uniform"]["theory_rate"]),
                abs(problem["uniform"]["rate"]),
                0.05,
                "closed-form discrete energy error of the 1D model problem; theory rate is "
                "1 (smooth) and alpha - 1/2 (singular), see tensorfem.adaptive_1d",
                f"theory {problem['uniform']['theory_rate']:.4f}, "
                f"measured {problem['uniform']['rate']:.4f}",
            )
        )
    singular_eta = one_d["problems"]["power"]["criteria"]["final_effectivity"]
    rows.append(
        {
            "quantity": "1D P1 singular effectivity theta (honest FAIL, documented limitation)",
            "reference": 1.0,
            "measured": singular_eta,
            "rel_error": rel_error(singular_eta, 1.0),
            "tolerance": 0.2,
            "verdict": "FAIL",
            "anchor": ANCHORS["effectivity"],
            "note": "the 1D gradient-recovery estimator is a stable underestimate on the "
            "singular field; no asymptotic-exactness claim is made there (theta in [0.8, 1.2] "
            "is asserted only for the 2D indicators, which meet it)",
        }
    )
    return rows, {"williams": zz, "poisson": poisson, "one_d": one_d}


def manifest(inputs: list[Path]) -> dict:
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": len([line for line in git("status", "--porcelain").splitlines()]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "reproduce": [
            "PYTHONPATH=src python scripts/adaptive_zz_study.py",
            "PYTHONPATH=src python scripts/adaptive_continuation_study.py",
            "PYTHONPATH=src python scripts/adaptive_evidence_pack.py",
            "PYTHONPATH=src python -m pytest tests/test_adaptive_zz.py -q",
        ],
    }


def main() -> dict:
    needed = [ZZ / "study.json", CONT / "study_2d.json", CONT / "study_1d.json"]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise SystemExit(f"missing study payloads: {missing}; run the study scripts first")
    rows, payloads = build_rows()
    data = {
        "schema": "tensorfem.adaptive_zz.evidence/v1",
        "feature_slug": "GAP-FEM-ADAPTIVE",
        "criteria": {
            "theta_interval": [0.8, 1.2],
            "rate_tolerance": 0.15,
            "all_rows_pass": all(item["verdict"] == "PASS" for item in rows),
            "failing_rows": [item["quantity"] for item in rows if item["verdict"] == "FAIL"],
        },
        "rows": rows,
        "anchors": ANCHORS,
        "study_criteria": payloads["williams"]["criteria"],
        "manifest": manifest(needed),
    }
    (ZZ / "evidence.json").write_text(json.dumps(data, indent=2, default=float))
    (ZZ / "benchmark_table.md").write_text(markdown_table(data))
    print(json.dumps({k: v for k, v in data["criteria"].items()}, indent=2))
    return data


def markdown_table(data: dict) -> str:
    lines = [
        "# GAP-FEM-ADAPTIVE benchmark table (ref vs got)",
        "",
        "Generated by `scripts/adaptive_evidence_pack.py` from the JSON payloads listed in the",
        "manifest below; no value in this table is typed by hand.",
        "",
        "| # | quantity | reference | measured | rel. error | tolerance | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, item in enumerate(data["rows"], start=1):
        lines.append(
            f"| {index} | {item['quantity']} | {item['reference']:.6g} | {item['measured']:.6g} | "
            f"{100.0 * item['rel_error']:.2f} % | {100.0 * item['tolerance']:.0f} % | "
            f"**{item['verdict']}** |"
        )
    lines += ["", "## Notes and anchors", ""]
    for item in data["rows"]:
        lines.append(f"* **{item['quantity']}** — {item['note']}  \n  anchor: {item['anchor']}")
    lines += [
        "",
        "## Manifest",
        "",
        f"* commit `{data['manifest']['git_commit']}` on `{data['manifest']['git_branch']}`",
        f"* python {data['manifest']['python']} on `{data['manifest']['platform']}`",
    ]
    for name, digest in data["manifest"]["inputs"].items():
        lines.append(f"* `{name}` sha256 `{digest}`")
    lines += ["", "```bash", *data["manifest"]["reproduce"], "```", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()