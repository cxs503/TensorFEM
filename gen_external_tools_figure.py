"""Generate the KPI-TOOLS-5 external-tool verification comparison figure.

Bar chart: for each probed tool category, the observed absolute error vs the
acceptance limit (log scale).  A category is REAL when its bar is at/below the
limit line.  This is the "comparison vs reference" figure for the report.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.external_tools_verification import run_suite


def main() -> None:
    ledger = run_suite()
    out = Path("artifacts") / "external_tools"
    out.mkdir(parents=True, exist_ok=True)

    cats = [c["tool_id"] for c in ledger["classes"]]
    errs = [c["abs_err"] if c["abs_err"] is not None else 1.0 for c in ledger["classes"]]
    limits = [c["limit"] for c in ledger["classes"]]
    statuses = [c["status"] for c in ledger["classes"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cats))
    # log-scale: errors are tiny (1e-16..1e-51); use log10 of abs_err
    log_err = [np.log10(max(e, 1e-60)) for e in errs]
    log_lim = [np.log10(max(l, 1e-60)) for l in limits]
    colors = ["#2ca02c" if s == "REAL" else "#d62728" for s in statuses]
    ax.bar(x, log_err, color=colors, alpha=0.85, label="observed |abs_err| (log10)")
    ax.plot(x, log_lim, "k--", marker="o", label="acceptance limit (log10)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel("log10(|abs_err|)")
    ax.set_title("KPI-TOOLS-5: external-tool integration verification\n"
                 "(bar at/below dashed line = REAL, within analytic reference)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    png = out / "external_tools_verification.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png} ({png.stat().st_size}B)")


if __name__ == "__main__":
    main()
