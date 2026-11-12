"""Audit the scalar benchmark registry by evidence class (fail-closed).

Usage (inside the repo checkout)::

    OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/registry_evidence_audit.py

Writes ``runs/registry_evidence_class/audit.json``, the Markdown table
``docs/assets/registry-evidence-classes/audit.md``, the figure
``docs/assets/registry-evidence-classes/evidence_classes.png`` and prints a
summary.  Exit code 1 when any check is unclassified / stale (no silent
default) or when the independent spheroid oracle disagrees with the registered
reference by more than 1e-9 relative.
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.benchmark_registry import verification_report  # noqa: E402
from tensorfem.registry_evidence_audit import (  # noqa: E402
    audit_evidence_classes,
    spheroid_equator_displacement_closed_form,
)

CASE = dict(a=1.0, b=0.5, thickness=0.01, young=1.0, poisson=0.3, pressure=1.0)
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "registry_evidence_class")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "registry-evidence-classes")


def main() -> int:
    report = verification_report()
    audit = audit_evidence_classes(report)
    rows = {row["id"]: row for row in report["results"]}

    oracle = spheroid_equator_displacement_closed_form(**CASE)
    registered = rows["shell.spheroid.displacement_equator"]["reference"]
    fe = rows["shell.spheroid.displacement_equator"]["computed"]
    oracle_delta = abs(registered - oracle) / abs(oracle)
    fe_vs_oracle = abs(fe - oracle) / abs(oracle)
    audit["spheroid_oracle"] = {
        "case": CASE,
        "registered_reference": registered,
        "independent_closed_form": oracle,
        "reference_vs_oracle_rel": oracle_delta,
        "fe_vs_oracle_rel": fe_vs_oracle,
        "registered_error_vs_reference": rows["shell.spheroid.displacement_equator"]["error"],
    }

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)
    with open(os.path.join(OUT, "audit.json"), "w") as fh:
        json.dump(audit, fh, indent=1, sort_keys=True)
    with open(os.path.join(OUT, "registry_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)

    lines = [
        "| class | checks | min error | max error | meaning |",
        "|---|---:|---:|---:|---|",
    ]
    meaning = {
        "discretized": "mesh/FD/eigen/time-step solve vs closed form or literature",
        "trial_space": "numerical quadrature/Ritz/contour integral vs closed form",
        "closed_form": "closed form vs a second closed-form coding (no numerics)",
        "identity": "consistency identity enforced by the algorithm itself",
    }
    for name in ("discretized", "trial_space", "closed_form", "identity"):
        lo = audit["min_error"][name]
        hi = audit["max_error"][name]
        lines.append(
            f"| `{name}` | {audit['counts'][name]} | "
            f"{'—' if lo is None else f'{lo:.3e}'} | {'—' if hi is None else f'{hi:.3e}'} | "
            f"{meaning[name]} |"
        )
    lines += [
        "",
        f"Total registered checks: {audit['total_checks']}; classified: {audit['classified']}; "
        f"discretised share: {audit['discretized_share']:.4f}.",
        "",
        "| thin margin check | class | error/tolerance |",
        "|---|---|---:|",
    ]
    for item in audit["thin_margin"]:
        lines.append(f"| `{item['id']}` | {item['class']} | {item['margin']:.3f} |")
    with open(os.path.join(ASSETS, "audit.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    try:  # figure is optional for the numeric gate
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
        names = ["discretized", "trial_space", "closed_form", "identity"]
        values = [audit["counts"][n] for n in names]
        ax.bar(names, values, color=["#2b6cb0", "#2f855a", "#b7791f", "#822727"])
        for i, v in enumerate(values):
            ax.text(i, v + 0.4, str(v), ha="center", fontsize=9)
        ax.set_ylabel("registered checks")
        ax.set_title(
            f"TensorFEM scalar registry: {audit['total_checks']} checks by evidence class"
        )
        ax.set_ylim(0, max(values) * 1.18)
        fig.tight_layout()
        fig.savefig(os.path.join(ASSETS, "evidence_classes.png"))
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[warn] figure not written: {exc}")

    print(json.dumps({k: audit[k] for k in
                      ("total_checks", "classified", "counts", "unclassified_ids",
                       "stale_ids", "thin_margin", "spheroid_oracle")}, indent=1))
    ok = not audit["unclassified_ids"] and not audit["stale_ids"] and oracle_delta < 1e-9
    print(f"[gate] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())