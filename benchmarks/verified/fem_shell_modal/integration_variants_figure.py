"""Turn the integration-variant modal ladder log into evidence JSON + figure.

Provenance: the numbers are **parsed** from the raw run log
``integration_variants.log`` (a programmatic injection, no hand transcription),
produced by monkey-patching the element stiffness used by
``tensorfem.cylindrical_shell_modal`` and re-running the anchor's ladder.

Usage
-----
    python benchmarks/verified/fem_shell_modal/integration_variants_figure.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
LOG = HERE / "integration_variants.log"
OUT_JSON = HERE / "integration_variants.json"
OUT_PNG = HERE / "integration_variants.png"

ORACLE_HZ = 202.14          # DMV minimum at (m, n) = (1, 5) for R=0.5, L=1, t=0.005
ORACLE_TOL_PCT = 15.0       # anchor gate G3

LINE = re.compile(
    r"^\s+(?P<mesh>\d+x\d+)\s+n_modes=(?P<nmodes>\d+)\s+mech_in_first10=(?P<mech>\d+)"
    r"\s+lowest_physical=\s*(?P<low>[0-9.]+)\s+Hz\s+err=\s*(?P<err>[0-9.]+)\s+%")


def parse() -> dict:
    rules: dict[str, list] = {}
    current = None
    for raw in LOG.read_text().splitlines():
        if raw.startswith("--- membrane="):
            head = raw.replace("--- ", "").strip()
            current = head
            rules[current] = []
            continue
        m = LINE.match(raw)
        if m and current is not None:
            rules[current].append({
                "mesh": m.group("mesh"),
                "n_modes": int(m.group("nmodes")),
                "mechanisms_in_first_ten": int(m.group("mech")),
                "lowest_physical_hz": float(m.group("low")),
                "error_vs_oracle_pct": float(m.group("err")),
            })
    return {"oracle_min_hz": ORACLE_HZ, "oracle_tolerance_pct": ORACLE_TOL_PCT,
            "source_log": LOG.name, "rules": rules}


def figure(data: dict) -> None:
    labels, series = [], []
    for rule, rows in data["rules"].items():
        if len(rows) != 3:
            continue
        labels.append(rule.replace("membrane=", "m=").replace(" shear=", " s="))
        series.append(rows)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    meshes = [r["mesh"] for r in series[0]]
    x = range(len(meshes))
    width = 0.8 / len(series)
    for i, (label, rows) in enumerate(zip(labels, series)):
        vals = [r["lowest_physical_hz"] for r in rows]
        ax.bar([xi - 0.4 + width * (i + 0.5) for xi in x], vals, width=width, label=label)
        for xi, v, r in zip(x, vals, rows):
            ax.text(xi - 0.4 + width * (i + 0.5), v, f"{r['error_vs_oracle_pct']:+.0f}%",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.axhline(ORACLE_HZ, color="k", ls="--", lw=1.2, label=f"DMV oracle {ORACLE_HZ:.2f} Hz")
    ax.axhspan(ORACLE_HZ * (1 - ORACLE_TOL_PCT / 100), ORACLE_HZ * (1 + ORACLE_TOL_PCT / 100),
               color="tab:green", alpha=0.12, label="anchor gate G3 band (15 %)")
    ax.set_xticks(list(x), meshes)
    ax.set_xlabel("mesh (axial x circumferential), cylinder R=0.5 m, L=1 m, t=5 mm")
    ax.set_ylabel("lowest mode with radial participation > 1e-6  [Hz]")
    ax.set_title("cylindrical_shell4: integration rule vs DMV oracle\n"
                 "no variant enters the gate band; reduced integration is too soft, "
                 "full membrane integration locks")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    data = parse()
    OUT_JSON.write_text(json.dumps(data, indent=2) + "\n")
    figure(data)
    print(f"wrote {OUT_JSON.name} and {OUT_PNG.name}")
    for rule, rows in data["rules"].items():
        errs = ", ".join(f"{r['mesh']}:{r['lowest_physical_hz']:.2f}Hz({r['error_vs_oracle_pct']:+.2f}%)"
                         for r in rows)
        print(f"  {rule}: {errs}")


if __name__ == "__main__":
    main()