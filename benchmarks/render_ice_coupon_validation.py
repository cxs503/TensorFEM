"""Regenerate the ice validation plots from measured inputs and saved solves."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.ice_indentation import spherical_bearing_load

DATA = Path(__file__).parent / "data" / "ice_coupons"
OUT = Path("docs/assets/ice-coupons")


def evidence(case):
    return json.loads(Path(f"artifacts/ice_coupons/{case}.json").read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (bend, indent) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    rows = evidence("stroke")["rows"]
    for row, color in zip([rows[0], rows[-1]], ["#20568b", "#b54e34"]):
        bend.plot(
            np.array(row["deflection_m"]) * 1000, row["force_N"], label=f"FE {row['id']}", c=color
        )
        bend.scatter(
            row["measured_peak_deflection_m"] * 1000,
            row["measured_peak_N"],
            marker="x",
            s=72,
            c=color,
            label=f"Test {row['id']}",
        )
    bend.set(
        xlabel="Load-point deflection (mm)",
        ylabel="Load (N)",
        title="Three-point bending, held-out specimens",
    )
    transient = evidence("transient")["rows"]
    for row, color in zip([transient[0], transient[-1]], ["#20568b", "#b54e34"]):
        history = np.asarray(row["history"])
        bend.plot(
            1000 * np.r_[0, history[:, 2]],
            np.r_[0, history[:, 1]],
            "--",
            c=color,
            linewidth=1.0,
            label=f"Transient {row['id']}",
        )
    d = evidence("indentation")
    obs = np.genfromtxt(DATA / "indentation.csv", delimiter=",", names=True)
    indent.plot(1000 * obs["depth_m"], obs["force_N"], c="black", label="Fig. 6, measured cycle")
    row = d["rows"][0]
    indent.plot(np.asarray(row["depth_m"]) * 1000, row["force_N"], label="J2 FE, loading")
    h = np.linspace(0, obs["depth_m"][obs["force_N"].argmax()], 101)
    indent.plot(
        1000 * h,
        [spherical_bearing_load(x, d["radius_m"], d["bearing_limit"]["hardness_Pa"]) for x in h],
        label="Bearing limit, loading",
    )
    indent.set(
        xlabel="Indentation (mm)", ylabel="Load (N)", title="Spherical indentation, 3.8 µm/s"
    )
    for ax in (bend, indent):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "coupon_comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
