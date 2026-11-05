"""Spectral verification of the rainflow + Palmgren-Miner fatigue path.

Checks the *distribution* behaviour of the merged advanced-fatigue primitives
(``tensorfem.marine_fatigue_advanced``) against standalone mathematics:

1. narrow-band Rayleigh damage oracle (DNV-RP-C203 App. B / Bendat) versus the
   production rainflow + Miner pipeline on simulated band-limited Gaussian
   histories, swept over spectral width;
2. sampling-rate sensitivity of the counted damage (known trap: an
   under-sampled narrow-band peak biases every range inward);
3. record-length / start-point sensitivity of the residual half-cycle
   bookkeeping;
4. the exact counting identity ``sum(count) = (n_rev - 1) / 2`` on random
   closed histories;
5. the constant-geometry Paris power-law branch ``m != 2`` against the closed
   form (the existing qualification only covers the ``m = 2`` exponential
   branch).

Writes ``benchmarks/marine_fatigue_spectrum_evidence.json`` and four figures
under ``docs/assets/marine-fatigue-spectrum/``.

Run: ``OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_marine_fatigue_spectrum.py``
"""
from __future__ import annotations

import json
import math
import platform as host
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem import __version__
from tensorfem.marine_fatigue_advanced import (
    CrackGrowthBlock,
    ParisLaw,
    paris_crack_growth,
    rainflow_cycles,
    spectrum_miner_damage,
)
from tensorfem.marine_fatigue_qualification import SNLogLogCurve
from tensorfem.rainflow_spectrum_reference import (
    full_cycle_multiset,
    narrow_band_damage_case,
    narrow_band_process,
    paris_power_law_closed_form,
    rainflow_count_invariant,
    rayleigh_range_pdf,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "marine_fatigue_spectrum_evidence.json"
ASSETS = ROOT / "docs" / "assets" / "marine-fatigue-spectrum"
TOLERANCE = 0.03
CURVE = SNLogLogCurve(100.0e6, 2.0e6, 3.0)
RMS_PA = 40.0e6
CYCLES = 20000.0
BANDWIDTHS = (0.4, 0.2, 0.1, 0.05)
SAMPLING = (4.0, 6.0, 8.0, 12.0, 20.0, 40.0)
PARIS_CASES = (
    ("paris.m2_exponential", 2.0, 1.0e-24, ((100.0e6, 1000.0), (150.0e6, 500.0))),
    ("paris.m2p5_power", 2.5, 1.0e-24, ((100.0e6, 1000.0), (150.0e6, 500.0))),
    ("paris.m3_power", 3.0, 1.0e-30, ((100.0e6, 1000.0), (150.0e6, 500.0), (80.0e6, 2000.0))),
    ("paris.m4_power", 4.0, 1.0e-38, ((100.0e6, 1000.0), (150.0e6, 500.0))),
)
FIGURES: list[Path] = []


def _save(fig, name: str) -> str:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append(path)
    return str(path.relative_to(ROOT))


def bandwidth_study() -> list[dict[str, float]]:
    rows = []
    for bandwidth in BANDWIDTHS:
        case = narrow_band_damage_case(
            curve=CURVE, relative_bandwidth=bandwidth, cycles=CYCLES, rms_stress_pa=RMS_PA
        )
        case["passed"] = case["relative_error"] < TOLERANCE
        rows.append(case)
    return rows


def sampling_study() -> list[dict[str, float]]:
    rows = []
    for samples in SAMPLING:
        case = narrow_band_damage_case(
            curve=CURVE, relative_bandwidth=0.05, cycles=CYCLES,
            samples_per_cycle=samples, rms_stress_pa=RMS_PA,
        )
        rows.append({
            "samples_per_cycle": case["samples_per_cycle"],
            "samples": case["samples"],
            "damage": case["damage"],
            "damage_oracle": case["damage_oracle"],
            "damage_ratio": case["damage_ratio"],
            "sampling_factor_estimate": case["sampling_factor_estimate"],
        })
    return rows


def start_point_study() -> dict[str, list[dict[str, float]]]:
    """Residual half-cycle bookkeeping versus record length and start offset."""
    result: dict[str, list[dict[str, float]]] = {}
    for label, cycles in (("long_20000_cycles", CYCLES), ("short_200_cycles", 200.0)):
        history, meta = narrow_band_process(
            relative_bandwidth=0.05, cycles=cycles, samples_per_cycle=40.0, rms_stress_pa=RMS_PA
        )
        rows = []
        n = history.size
        for fraction in (0.0, 0.05, 0.2, 0.5):
            shifted = np.roll(history, -int(fraction * n))
            counted = rainflow_cycles(shifted.tolist())
            damage = spectrum_miner_damage(counted, CURVE).damage
            rows.append({
                "start_fraction": fraction,
                "cycles_counted": math.fsum(row.count for row in counted),
                "full_cycles": float(sum(1 for row in counted if row.count == 1.0)),
                "damage": damage,
                "damage_relative_to_first": 1.0,
            })
        base = rows[0]["damage"]
        for row in rows:
            row["damage_relative_to_first"] = row["damage"] / base
        result[label] = rows
    # the decomposition (not the total) is start-point dependent: quantify the
    # full-cycle multiset agreement on the long record
    history, _ = narrow_band_process(relative_bandwidth=0.05, cycles=200.0, samples_per_cycle=40.0)
    reference = full_cycle_multiset(history.tolist())
    matched = 0
    total = 0
    for fraction in (0.05, 0.2, 0.5):
        shifted = np.roll(history, -int(fraction * history.size))
        other = full_cycle_multiset(shifted.tolist())
        total += 1
        matched += int(other == reference)
    result["full_cycle_multiset_agreement"] = matched / total
    return result


def invariant_study(trials: int = 300) -> dict[str, float]:
    rng = np.random.default_rng(20261104)
    worst = 0.0
    violations = 0
    for k in range(trials):
        length = int(rng.integers(4, 14))
        walk = np.cumsum(np.random.default_rng(7000 + k).normal(size=length - 1))
        walk = walk - walk.mean()
        history = [float(walk[0]), *map(float, walk[1:]), float(walk[0])]
        stats = rainflow_count_invariant(history)
        worst = max(worst, stats["defect"])
        violations += int(stats["defect"] > 1e-12)
    return {
        "trials": float(trials),
        "violations": float(violations),
        "worst_absolute_defect": worst,
        "identity": "sum(count) = (n_reversals - 1) / 2 for a closed record",
    }


def paris_study() -> list[dict[str, float]]:
    rows = []
    for case_id, slope, coefficient, blocks in PARIS_CASES:
        law = ParisLaw(coefficient, slope, 1.0)
        typed = tuple(CrackGrowthBlock(stress, cycles) for stress, cycles in blocks)
        produced = paris_crack_growth(0.01, typed, law).final_crack_m
        oracle = paris_power_law_closed_form(0.01, typed, law)
        rows.append({
            "id": case_id,
            "exponent_m": slope,
            "coefficient_c": coefficient,
            "produced_m": produced,
            "oracle_m": oracle,
            "relative_error": abs(produced - oracle) / oracle,
            "tolerance": TOLERANCE,
            "passed": abs(produced - oracle) / oracle < TOLERANCE,
        })
    return rows


def figures(band: list[dict], sampling: list[dict], paris: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    # 1) counted range histogram against the Rayleigh oracle
    history, meta = narrow_band_process(
        relative_bandwidth=0.05, cycles=CYCLES, samples_per_cycle=40.0, rms_stress_pa=RMS_PA
    )
    counted = rainflow_cycles(history.tolist())
    ranges = np.array([row.range_pa for row in counted]) / 1e6
    weights = np.array([row.count for row in counted])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(ranges, bins=60, weights=weights, density=True, alpha=0.55,
            label="production ASTM E1049 rainflow", color="#1f77b4")
    grid = np.linspace(1e-6, ranges.max(), 400)
    ax.plot(grid, rayleigh_range_pdf(grid * 1e6, meta["rms_stress_pa"]) / 1e6,
            "k--", lw=1.8, label="Rayleigh narrow-band oracle")
    ax.set_xlabel("stress range S (MPa)")
    ax.set_ylabel("probability density (1/MPa)")
    ax.set_title(
        f"Counted ranges vs Rayleigh oracle, $\\varepsilon$={meta['spectral_width']:.4f}, "
        f"RMS={meta['rms_stress_pa'] / 1e6:.0f} MPa"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    out["range_histogram"] = _save(fig, "range-histogram-rayleigh.png")

    # 2) damage ratio vs spectral width
    eps = [row["spectral_width"] for row in band]
    ratio = [row["damage_ratio"] for row in band]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhspan(0.97, 1.03, color="#2ca02c", alpha=0.12, label="3% gate")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.plot(eps, ratio, "o-", color="#1f77b4", label="rainflow + Miner / Rayleigh oracle")
    for row in band:
        ax.annotate(f"{row['relative_error'] * 100:.2f}%",
                    (row["spectral_width"], row["damage_ratio"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("spectral width $\\varepsilon=\\sqrt{1-m_1^2/(m_0m_2)}$")
    ax.set_ylabel("damage ratio (numerical / narrow-band)")
    ax.set_title("Narrow-band limit: damage ratio versus spectral width")
    ax.legend()
    ax.grid(alpha=0.3)
    out["bandwidth_convergence"] = _save(fig, "bandwidth-convergence.png")

    # 3) sampling-rate bias against the leading-order estimate
    sample_x = [row["samples_per_cycle"] for row in sampling]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(sample_x, [row["damage_ratio"] for row in sampling], "o-",
            color="#d62728", label="measured damage ratio")
    ax.plot(sample_x, [row["sampling_factor_estimate"] for row in sampling], "s--",
            color="#7f7f7f", label="leading-order $(1-\\pi^2/6N^2)^m$")
    ax.axhspan(0.97, 1.03, color="#2ca02c", alpha=0.12, label="3% gate")
    ax.set_xscale("log")
    ax.set_xlabel("samples per cycle (fs / f0)")
    ax.set_ylabel("damage ratio (numerical / oracle)")
    ax.set_title("Under-sampled narrow-band peaks bias damage low")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    out["sampling_bias"] = _save(fig, "sampling-rate-bias.png")

    # 4) Paris power-law branch error
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = [row["id"].split(".", 1)[1] for row in paris]
    errors = [max(row["relative_error"], 1e-17) for row in paris]
    ax.bar(labels, errors, color="#9467bd")
    ax.set_yscale("log")
    ax.axhline(0.03, color="k", ls="--", lw=1.0, label="3% gate")
    ax.set_ylabel("|relative error| vs closed form")
    ax.set_title("Paris constant-geometry closed form, power-law branch")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out["paris_branch"] = _save(fig, "paris-power-law-branch.png")
    return out


def main() -> dict[str, object]:
    band = bandwidth_study()
    sampling = sampling_study()
    start = start_point_study()
    invariant = invariant_study()
    paris = paris_study()
    fig = figures(band, sampling, paris)

    narrow = min(band, key=lambda row: row["spectral_width"])
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tensorfem_version": __version__,
        "python": host.python_version(),
        "benchmark": "rainflow_spectrum_reference",
        "description": (
            "Spectral verification of the rainflow + Palmgren-Miner fatigue primitives "
            "against the narrow-band Rayleigh closed form and the Paris power-law branch"
        ),
        "parameters": {
            "sn_curve": {"reference_stress_pa": CURVE.reference_stress_pa,
                         "reference_cycles": CURVE.reference_cycles,
                         "slope_m": CURVE.slope_m},
            "rms_stress_pa": RMS_PA,
            "cycles_per_case": CYCLES,
            "bandwidth_sweep": list(BANDWIDTHS),
            "sampling_sweep": list(SAMPLING),
            "tolerance": TOLERANCE,
        },
        "bandwidth_study": band,
        "sampling_study": sampling,
        "start_point_study": start,
        "counting_invariant": invariant,
        "paris_study": paris,
        "figures": fig,
        "verdict": {
            "narrow_band_within_3pct": narrow["relative_error"] < TOLERANCE,
            "narrow_band_spectral_width": narrow["spectral_width"],
            "wide_band_damage_rate_positive": all(
                row["damage_rate_ratio"] >= 1.0 for row in band
            ),
            "counting_identity_holds": invariant["violations"] == 0.0,
            "paris_power_branch_all_within_3pct": all(row["passed"] for row in paris),
            "sampling_trap_documented": (
                min(row["damage_ratio"] for row in sampling) < 0.9
                and sampling[-1]["damage_ratio"] > 0.97
            ),
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    for row in band:
        print(f"band={row['relative_bandwidth']:.2f} eps={row['spectral_width']:.4f} "
              f"ratio={row['damage_ratio']:.5f} rate={row['damage_rate_ratio']:.5f} "
              f"countrate_err={row['count_rate_error'] * 100:5.2f}% "
              f"err={row['relative_error'] * 100:6.3f}%")
    for row in sampling:
        print(f"fs/f0={row['samples_per_cycle']:5.1f} ratio={row['damage_ratio']:.5f} "
              f"est={row['sampling_factor_estimate']:.5f}")
    print("counting invariant:", invariant)
    print("verdict:", payload["verdict"])
    return payload


if __name__ == "__main__":
    main()