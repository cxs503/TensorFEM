"""Audit the bounded Abaqus ``.inp`` keyword gate and the round-trip precision.

Two experiments, both programmatic (no number in the report is hand-copied):

1. **Gate decision matrix** -- every keyword of a checklist of real ship-deck
   keywords is appended to one valid base model and read.  The outcome is
   classified from the reader itself: ``no-op`` (accepted, model unchanged),
   ``model data`` (accepted and changes the ModelDB) or ``rejected`` (the gate
   named the keyword and the line).
2. **Round-trip precision** -- awkward coordinates are written with
   ``d`` significant digits, re-read and compared bitwise, to decide how many
   digits ``write_inp`` must emit.

Outputs ``artifacts/abaqus_keyword_gate/audit.json`` and ``keyword_gate.png``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tensorfem.abaqus_io import read_inp, write_inp, SUPPORTED_KEYWORDS, NON_MODEL_KEYWORDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "abaqus_keyword_gate"

BASE = """*Node
1,0,0
2,1,0
3,1,1
4,0,1
*Element, type=CPS4, elset=EO
1,1,2,3,4
*Material, name=M
*Elastic
2.1E11,0.3
*Solid Section, elset=EO, material=M
0.012
"""

#: (keyword line, data rows) appended to BASE; first ten are in BASE itself.
CANDIDATES = [
    ("*Node, nset=NALL", ["5,2,0"]),
    ("*Element, type=CPS4, elset=EO2", ["2,1,2,3,4"]),
    ("*Nset, nset=NS1", ["1,2"]),
    ("*Elset, elset=ES1", ["1"]),
    ("*Material, name=M2", ["*Elastic", "1.E9,0.25"]),
    ("*Solid Section, elset=EO, material=M", ["0.02"]),
    ("*Boundary", ["NALL,1,2"]),
    ("*Cload", ["2,2,-1.0"]),
    # deliberately outside the subset
    ("*Include, input=part2.inp", []),
    ("*Shell Section, elset=EO, material=M", ["0.02,"]),
    ("*Beam Section, elset=EO, material=M", ["RECT,"]),
    ("*Membrane Section, elset=EO, material=M", ["0.01"]),
    ("*Surface, type=ELEMENT, name=S1", ["EO, S1"]),
    ("*Tie, name=T1", ["S1, S2"]),
    ("*Distribution, name=D1, location=Element", ["1,1."]),
    # non-model / analysis control
    ("*Step, name=S1", ["*Static", "1.,1.", "*End Step"]),
    ("*Static", ["1.,1."]),
    ("*Output, field, variable=preselect", []),
    ("*Element Output", ["S,"]),
    ("*Node Print, nset=NS1", ["U,"]),
    ("*Controls, parameters=time incrementation", []),
    ("*Restart, write, frequency=0", []),
]


def _fields(db) -> dict:
    return {
        f: getattr(db, f)
        for f in (
            "nodes",
            "elements",
            "node_sets",
            "element_sets",
            "materials",
            "sections",
            "boundaries",
            "loads",
        )
    }


def gate_matrix() -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "base.inp"
        base_path.write_text(BASE, encoding="utf-8")
        base = read_inp(base_path)
        for k, (line, data) in enumerate(CANDIDATES):
            path = Path(tmp) / f"case_{k}.inp"
            text = BASE + line + "\n" + "".join(row + "\n" for row in data)
            # nested *Elastic inside the *Material candidate must stay attached
            path.write_text(text, encoding="utf-8")
            record = {
                "keyword": line.split(",")[0].lstrip("*").strip().upper(),
                "declared": (
                    "supported"
                    if line.split(",")[0].lstrip("*").strip().upper() in SUPPORTED_KEYWORDS
                    else "non-model"
                    if line.split(",")[0].lstrip("*").strip().upper() in NON_MODEL_KEYWORDS
                    else "outside"
                ),
            }
            try:
                db = read_inp(path)
            except ValueError as exc:
                record.update(outcome="rejected", message=str(exc))
            else:
                changed = _fields(db) != _fields(base)
                record.update(outcome="model data" if changed else "no-op", message="")
            rows.append(record)
    return rows


DIGITS = [6, 9, 12, 15, 17]
AWKWARD = [
    0.1 + 1.0 / 3.0,
    1.0 / 7.0,
    1234.5678901234567,
    -0.3 - 1e-17,
    math.pi,
    2.1e11,
    1e-9 / 3.0,
    0.012345678901234567,
]


def precision_study() -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for digits in DIGITS:
            lines = ["*Node"] + [f"{i + 1}, {v:.{digits}g}, 0.0" for i, v in enumerate(AWKWARD)]
            path = Path(tmp) / f"p{digits}.inp"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            db = read_inp(path)
            got = [db.nodes[i + 1][0] for i in range(len(AWKWARD))]
            worst = max(abs(g - v) / max(abs(v), 1e-300) for g, v in zip(got, AWKWARD))
            exact = sum(1 for g, v in zip(got, AWKWARD) if g == v)
            rows.append(
                {
                    "digits": digits,
                    "max_relative_error": worst,
                    "bit_exact": exact,
                    "samples": len(AWKWARD),
                    "bound": 10.0 ** (-(digits - 1)),
                }
            )
    return rows


def round_trip_study() -> dict:
    """Read -> write -> read must return bit-identical coordinates."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "in.inp"
        source.write_text(BASE, encoding="utf-8")
        first = read_inp(source)
        target = Path(tmp) / "out.inp"
        write_inp(target, first)
        second = read_inp(target)
        equal = _fields(first) == _fields(second)
        return {"bit_identical_round_trip": bool(equal), "files": [source.name, target.name]}


def figure(rows: list[dict], precision: list[dict], path: Path, dpi: int = 150) -> None:
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11.4, 5.9), dpi=dpi, gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    colors = {"no-op": "#4c9f70", "model data": "#2b6cb0", "rejected": "#dd6b20"}
    labels = {
        "no-op": "accepted (non-model no-op)",
        "model data": "accepted (model data)",
        "rejected": "rejected by the gate (ValueError)",
    }
    ys = range(len(rows))
    for y, row in zip(ys, rows):
        ax0.barh(y, 1.0, color=colors[row["outcome"]], alpha=0.88, edgecolor="white", height=0.78)
        ax0.text(1.02, y, row["outcome"], va="center", fontsize=7.5, color="#333")
    ax0.set_yticks(list(ys))
    ax0.set_yticklabels([f"{r['keyword']}  [{r['declared']}]" for r in rows], fontsize=8)
    ax0.invert_yaxis()
    ax0.set_xlim(0, 1.45)
    ax0.set_xticks([])
    ax0.set_title(
        "Keyword gate: one valid base model + one candidate keyword\n"
        "(decision read back from read_inp, not asserted by hand)",
        fontsize=10,
    )
    ax0.legend(
        handles=[
            mpatches.Patch(color=colors[k], label=labels[k])
            for k in ("model data", "no-op", "rejected")
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.95,
    )

    d = [p["digits"] for p in precision]
    err = [max(p["max_relative_error"], 1e-20) for p in precision]
    ax1.semilogy(d, err, "o-", color="#dd6b20", label="measured round-trip error")
    ax1.semilogy(
        d, [p["bound"] for p in precision], "k--", lw=1.0, label=r"sampling bound $10^{-(d-1)}$"
    )
    ax1.axhspan(1e-18, 1e-16, color="#4c9f70", alpha=0.35)
    ax1.text(6.1, 2.2e-17, "float64 unit roundoff", fontsize=7.5, color="#25603f")
    for p in precision:
        if p["bit_exact"] == p["samples"]:
            ax1.plot(
                p["digits"],
                max(p["max_relative_error"], 1e-20),
                "o",
                ms=9,
                mfc="none",
                mec="#2b6cb0",
                mew=1.6,
            )
    ax1.set_xlabel("significant digits emitted by write_inp")
    ax1.set_ylabel("max relative error of stored coordinate")
    ax1.set_title("Why the writer emits %.17g\n(open circle = bit-exact round trip)", fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = gate_matrix()
    precision = precision_study()
    report = {
        "schema": "tensorfem.abaqus-keyword-gate.v1",
        "supported_keywords": sorted(SUPPORTED_KEYWORDS),
        "non_model_keywords": sorted(NON_MODEL_KEYWORDS),
        "gate": rows,
        "gate_counts": {
            k: sum(1 for r in rows if r["outcome"] == k)
            for k in ("model data", "no-op", "rejected")
        },
        "precision": precision,
        "round_trip": round_trip_study(),
    }
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figure(rows, precision, OUT / "keyword_gate.png", dpi=150)
    figure(rows, precision, OUT / "keyword_gate_report.png", dpi=78)
    print(json.dumps({k: report[k] for k in ("gate_counts", "round_trip")}, indent=2))
    for r in rows:
        print(f"{r['keyword']:22s} [{r['declared']:9s}] -> {r['outcome']:10s} {r['message'][:90]}")


if __name__ == "__main__":
    main()
