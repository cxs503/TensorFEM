"""Verification driver for ``tensorfem.wave_spectrum``.

Produces ``artifacts/wave_spectrum_verification/validation_report.json`` plus the
figures used by ``docs/marine-wave-spectrum-verification.md`` and the platform
report.  Every number in the documentation is emitted from here, so no value is
typed by hand.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=src python3 scripts/audit_wave_spectrum.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.wave_spectrum import (
    Jonswap,
    PiersonMoskowitz,
    equivalent_design_wave,
    irregular_sea_realisation,
    rayleigh_exceedance,
    rayleigh_most_probable_extreme,
    short_term_statistics,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "wave_spectrum_verification"
FIGURES = ROOT / "docs" / "assets" / "wave_spectrum_verification"
THRESHOLD = 0.03  # 3 % acceptance band used across the repository


def relative_error(value: float, reference: float) -> float:
    return value / reference - 1.0


def closed_form_check() -> dict[str, object]:
    """A1: PM moments against their exact closed form, per order."""
    pm = PiersonMoskowitz(6.0, 10.0)
    rows = []
    for order in (0, 1, 2, 3):
        quadrature = pm.moment(order)
        exact = pm.closed_form_moment(order)
        rows.append(
            {
                "order": order,
                "quadrature": quadrature["moment"],
                "closed_form": exact,
                "relative_error": relative_error(quadrature["moment"], exact),
                "panels": quadrature["intervals"],
                "converged": quadrature["converged"],
            }
        )
    worst = max(abs(row["relative_error"]) for row in rows)
    return {
        "case": "PiersonMoskowitz(Hs=6 m, Tp=10 s)",
        "rows": rows,
        "worst_abs_relative_error": worst,
        "passed": worst < THRESHOLD,
    }


def reduction_check() -> dict[str, object]:
    """A2: gamma = 1 must reproduce Pierson-Moskowitz exactly."""
    pm = PiersonMoskowitz(5.0, 8.0)
    reduced = Jonswap(5.0, 8.0, 1.0)
    grid = torch.linspace(0.05, 6.0 * pm.peak_angular_frequency, 20001, dtype=torch.float64)
    values_pm = pm.density(grid)
    density_gap = float(torch.max(torch.abs(reduced.density(grid) - values_pm))) / float(
        torch.max(torch.abs(values_pm))
    )
    stats_pm, stats_reduced = short_term_statistics(pm), short_term_statistics(reduced)
    moment_gap = max(
        abs(getattr(stats_reduced, name) / getattr(stats_pm, name) - 1.0)
        for name in ("m0", "m1", "m2")
    )
    return {
        "case": "Jonswap(gamma=1) vs PiersonMoskowitz(Hs=5 m, Tp=8 s)",
        "max_density_relative_gap": density_gap,
        "max_moment_relative_gap": moment_gap,
        "alpha_relative_gap": relative_error(reduced.alpha, pm.alpha),
        "passed": max(density_gap, moment_gap) < 1e-9,
    }


def literature_alpha_check() -> dict[str, object]:
    """A3: alpha against the standard 5.061 (Hs/Tp^2)^2 (1 - 0.287 ln gamma)."""
    rows = []
    for hs, tp, gamma in ((6.0, 10.0, 3.3), (4.0, 8.0, 2.0), (9.0, 13.0, 5.0), (2.5, 6.5, 1.0)):
        alpha = Jonswap(hs, tp, gamma).alpha
        literature = 5.061 * (hs / tp**2) ** 2 * (1.0 - 0.287 * math.log(gamma))
        rows.append(
            {
                "hs_m": hs,
                "tp_s": tp,
                "gamma": gamma,
                "alpha_solved": alpha,
                "alpha_literature": literature,
                "relative_error": relative_error(alpha, literature),
            }
        )
    pm = PiersonMoskowitz(6.0, 10.0)
    pm_literature = 5.061 * (6.0 / 10.0**2) ** 2
    rows.append(
        {
            "hs_m": 6.0,
            "tp_s": 10.0,
            "gamma": 1.0,
            "alpha_solved": pm.alpha,
            "alpha_literature": pm_literature,
            "relative_error": relative_error(pm.alpha, pm_literature),
            "note": "Pierson-Moskowitz closed form",
        }
    )
    worst = max(abs(row["relative_error"]) for row in rows)
    return {"rows": rows, "worst_abs_relative_error": worst, "passed": worst < 0.01}


def rayleigh_check() -> dict[str, object]:
    """A4: classical Rayleigh values, exact-vs-asymptotic extreme."""
    rows = []
    for n_cycles in (100.0, 1000.0, 10000.0, 100000.0):
        exact = rayleigh_most_probable_extreme(n_cycles, 2.25)
        asymptotic = rayleigh_most_probable_extreme(n_cycles, 2.25, exact=False)
        rows.append(
            {
                "n_cycles": n_cycles,
                "exact_median_max_m": exact,
                "asymptotic_m": asymptotic,
                "relative_gap": relative_error(exact, asymptotic),
                "over_hs": exact / 6.0,
                "sqrt_ln_n_over_2": math.sqrt(math.log(n_cycles) / 2.0),
            }
        )
    exceed_hs = rayleigh_exceedance(6.0, 2.25)
    return {
        "exceedance_at_hs": exceed_hs,
        "exp_minus_2": math.exp(-2.0),
        "exceedance_relative_error": relative_error(exceed_hs, math.exp(-2.0)),
        "extreme_rows": rows,
        "passed": abs(relative_error(exceed_hs, math.exp(-2.0))) < 1e-12,
    }


def time_domain_check(seeds: int = 8, n_samples: int = 2**14) -> dict[str, object]:
    """A5: realised records against the spectral prediction, 8 seeds."""
    cases = {
        "jonswap_gamma3.3": Jonswap(6.0, 10.0, 3.3),
        "pierson_moskowitz": PiersonMoskowitz(6.0, 10.0),
    }
    out: dict[str, object] = {}
    for name, spectrum in cases.items():
        stats = short_term_statistics(spectrum)
        hs_err, tz_err, hm_err, max_heights, band_tz_err = [], [], [], [], []
        truncated = None
        for seed in range(seeds):
            record = irregular_sea_realisation(spectrum, n_samples=n_samples, seed=seed)
            band_tz_err.append(
                relative_error(
                    record["measured_zero_crossing_period"], record["band_zero_crossing_period"]
                )
            )
            hs_err.append(relative_error(record["measured_hs"], stats.hs))
            tz_err.append(
                relative_error(record["measured_zero_crossing_period"], stats.zero_crossing_period)
            )
            hm_err.append(
                relative_error(record["mean_wave_height"], math.sqrt(2.0 * math.pi * stats.m0))
            )
            max_heights.append(record["max_wave_height"])
            truncated = record["truncated_m0"]
        n_waves = record["zero_upcrossings"] - 1
        mpm = stats.most_probable_extreme(n_waves)
        out[name] = {
            "hs_target_m": stats.hs,
            "hs_measured_mean_m": stats.hs * (1.0 + sum(hs_err) / seeds),
            "hs_relative_error_worst": max(abs(v) for v in hs_err),
            "tz_spectral_s": stats.zero_crossing_period,
            "tz_measured_mean_s": stats.zero_crossing_period * (1.0 + sum(tz_err) / seeds),
            "tz_relative_error_worst": max(abs(v) for v in tz_err),
            "tz_relative_error_mean": sum(tz_err) / seeds,
            "tz_vs_band_limit_mean": sum(band_tz_err) / seeds,
            "tz_vs_band_limit_worst": max(abs(v) for v in band_tz_err),
            "band_zero_crossing_period_s": record["band_zero_crossing_period"],
            "band_m0_fraction": record["band_m0"] / stats.m0,
            "band_m2_fraction": record["band_m2"] / stats.m2,
            "mean_height_relative_error_worst": max(abs(v) for v in hm_err),
            "truncated_m0": truncated,
            "spectral_m0": stats.m0,
            "waves_per_record": n_waves,
            "max_height_mean_m": sum(max_heights) / seeds,
            "max_height_min_m": min(max_heights),
            "max_height_max_m": max(max_heights),
            "most_probable_extreme_m": mpm,
            "max_height_mean_vs_mpm": relative_error(sum(max_heights) / seeds, mpm),
            "record_length_s": record["record_length"],
            "sample_interval_s": record["sample_interval"],
        }
    summary_passed = all(
        row["hs_relative_error_worst"] < 0.01 and row["tz_relative_error_worst"] < 0.06
        for row in out.values()
    )
    return {"seeds": seeds, "n_samples": n_samples, "cases": out, "passed": summary_passed}


def design_wave_link() -> dict[str, object]:
    """A6: the spectral -> regular-wave hand-over used by the Morison path."""
    spectrum = Jonswap(6.0, 10.0, 3.3)
    wave = equivalent_design_wave(spectrum)
    k = wave.wave_number
    residual = abs(wave.gravity * k * math.tanh(k * wave.depth) / wave.omega**2 - 1.0)
    return {
        "height_m": wave.height,
        "period_s": wave.period,
        "depth_m": wave.depth,
        "dispersion_residual": residual,
        "passed": residual < 1e-12 and abs(wave.height / 6.0 - 1.0) < 1e-6,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_spectra() -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    omega = torch.linspace(0.05, 4.0, 4000, dtype=torch.float64)
    for spectrum, label in (
        (PiersonMoskowitz(6.0, 10.0), r"PM  $H_s$=6 m, $T_p$=10 s"),
        (Jonswap(6.0, 10.0, 1.0), r"JONSWAP $\gamma$=1 (PM limit)"),
        (Jonswap(6.0, 10.0, 3.3), r"JONSWAP $\gamma$=3.3"),
        (Jonswap(6.0, 10.0, 5.0), r"JONSWAP $\gamma$=5"),
    ):
        ax.loglog(omega, torch.clamp(spectrum.density(omega), min=1e-12), lw=1.6, label=label)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$S(\omega)$ [m$^2\cdot$s]")
    ax.set_title("Spectra with identical $H_s$ = 6 m, $T_p$ = 10 s")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "spectra.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_quadrature(closed_form: dict[str, object]) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pm = PiersonMoskowitz(6.0, 10.0)
    panels = [32 * 2**i for i in range(7)]
    for order, marker in zip((0, 1, 2, 3), ("o", "s", "^", "d")):
        exact = pm.closed_form_moment(order)
        from tensorfem.wave_spectrum import _moment_in_log_frequency

        errors = []
        for count in panels:
            value = _moment_in_log_frequency(
                pm._density_scalar, pm.peak_angular_frequency, order, max_intervals=count
            )
            errors.append(abs(relative_error(value["moment"], exact)) + 1e-17)
        ax.loglog(panels, errors, marker=marker, label=f"m{order}")
    ax.axhline(THRESHOLD, color="r", ls="--", lw=1, label="3 % acceptance")
    ax.set_xlabel("Gauss-Legendre panels on $t=\\ln(\\omega/\\omega_p)\\in[-45,45]$")
    ax.set_ylabel("relative error vs closed form")
    ax.set_title("Quadrature error of the PM moments")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "quadrature_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_time_domain() -> Path:
    spectrum = Jonswap(6.0, 10.0, 3.3)
    stats = short_term_statistics(spectrum)
    record = irregular_sea_realisation(spectrum, n_samples=2**14, seed=3)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6.6, 6.0))
    window = 900
    ax_top.plot(
        record["time"][:window].numpy(), record["elevation"][:window].numpy(), lw=0.9, color="C0"
    )
    ax_top.axhline(0.0, color="k", lw=0.6)
    ax_top.set_xlabel("t [s]")
    ax_top.set_ylabel(r"$\eta$ [m]")
    ax_top.set_title(
        f"JONSWAP $\\gamma$=3.3 realised record (seed 3), $H_s$ measured "
        f"{record['measured_hs']:.3f} m vs spectral {stats.hs:.3f} m"
    )
    ax_top.grid(alpha=0.3)

    heights = record["wave_heights"].numpy()
    heights.sort()
    exceed = 1.0 - (torch.arange(1, len(heights) + 1, dtype=torch.float64) / len(heights))
    ax_bottom.semilogy(heights, exceed, marker=".", ms=4, lw=0.8, label="realised record")
    grid = torch.linspace(0.05, float(heights.max()) * 1.05, 400, dtype=torch.float64)
    ax_bottom.semilogy(
        grid.numpy(),
        rayleigh_exceedance(grid, stats.m0).numpy(),
        "r--",
        lw=1.2,
        label=r"Rayleigh $\exp(-H^2/8m_0)$",
    )
    ax_bottom.set_xlabel("wave height H [m]")
    ax_bottom.set_ylabel("exceedance probability")
    ax_bottom.set_title("Wave-height distribution vs the narrow-band Rayleigh model")
    ax_bottom.grid(alpha=0.3, which="both")
    ax_bottom.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "time_domain_and_rayleigh.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_statistics(time_domain: dict[str, object]) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    names = list(time_domain["cases"])
    labels = ["spectral prediction", "realised records (8 seeds)"]
    width = 0.35
    positions = range(len(names))
    for offset, key in enumerate(("hs_target_m", "hs_measured_mean_m")):
        values = [time_domain["cases"][n][key] for n in names]
        ax.bar([p + (offset - 0.5) * width for p in positions], values, width, label=labels[offset])
    ax.set_xticks(list(positions))
    ax.set_xticklabels([n.replace("_", " ") for n in names])
    ax.set_ylabel("$H_s$ [m]")
    ax.set_ylim(0.0, 7.0)
    ax.set_title("Significant wave height: spectrum vs realised records")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIGURES / "hs_spectral_vs_realised.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> dict[str, object]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "tensorfem.wave-spectrum-verification.v1",
        "threshold_relative": THRESHOLD,
        "closed_form_moments": closed_form_check(),
        "jonswap_pm_limit": reduction_check(),
        "literature_alpha": literature_alpha_check(),
        "rayleigh": rayleigh_check(),
        "time_domain": time_domain_check(),
        "design_wave_link": design_wave_link(),
    }
    figures = {
        "spectra": figure_spectra(),
        "quadrature": figure_quadrature(report["closed_form_moments"]),
        "time_domain": figure_time_domain(),
        "statistics": figure_statistics(report["time_domain"]),
    }
    report["figures"] = {name: str(path.relative_to(ROOT)) for name, path in figures.items()}
    report["all_passed"] = all(
        block["passed"]
        for key, block in report.items()
        if isinstance(block, dict) and "passed" in block
    )
    path = ARTIFACTS / "validation_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"report": str(path), "all_passed": report["all_passed"]}, indent=2))
    return report


if __name__ == "__main__":
    main()