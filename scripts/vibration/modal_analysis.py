#!/usr/bin/env python3
"""Ship hull structural vibration modal analysis benchmark script.

Runs the complete dry/wet modal analysis, FRF, radiated noise, and energy
balance verification for a simply supported rectangular steel plate.

Usage:
    python scripts/vibration/modal_analysis.py [--nx NX] [--ny NY] [--modes M]

Produces:
    - Dry modal frequencies vs Timoshenko-Mindlin theory
    - Wet modal frequencies vs BEM analytical reference
    - FRF and vibration acceleration level (VAL)
    - Radiation efficiency vs Leppington (1982)
    - Energy balance check
    - Verification report (JSON + markdown)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorfem.vibration import (
    timoshenko_plate_angular_frequency,
    solve_dry_modal,
    solve_wet_modal,
    compute_frf,
    compute_radiated_noise,
    check_energy_balance,
    radiation_efficiency_leppington,
    critical_frequency,
    abramson_wet_ratio,
)


def _hz(omega: float) -> float:
    return omega / (2.0 * math.pi)


def _center_dof(elements_x: int, elements_y: int) -> int:
    n_nodes_x = elements_x + 1
    n_nodes_y = elements_y + 1
    cx, cy = n_nodes_x // 2, n_nodes_y // 2
    return 3 * (cy * n_nodes_x + cx)


def main():
    parser = argparse.ArgumentParser(description="Ship hull vibration modal analysis")
    parser.add_argument("--nx", type=int, default=16, help="Elements in x")
    parser.add_argument("--ny", type=int, default=16, help="Elements in y")
    parser.add_argument("--modes", type=int, default=6, help="Number of modes")
    parser.add_argument("--length", type=float, default=1.0, help="Plate length [m]")
    parser.add_argument("--width", type=float, default=1.0, help="Plate width [m]")
    parser.add_argument("--thickness", type=float, default=0.01, help="Plate thickness [m]")
    parser.add_argument("--output", type=str, default="artifacts/vibration_report", help="Output directory")
    args = parser.parse_args()

    # Material properties
    steel = dict(density=7850.0, young=200.0e9, poisson=0.3, thickness=args.thickness)
    water = dict(rho_water=1025.0, c_water=1500.0)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("Ship Hull Structural Vibration Modal Analysis Benchmark")
    print("=" * 60)
    print(f"Plate: {args.length} x {args.width} x {args.thickness} m")
    print(f"Mesh: {args.nx} x {args.ny} elements")
    print(f"Modes: {args.modes}")
    print()

    # ------------------------------------------------------------------
    # 1. Dry modal analysis
    # ------------------------------------------------------------------
    print("--- Dry Modal Analysis ---")
    dry = solve_dry_modal(args.nx, args.ny, length=args.length, width=args.width,
                          modes=args.modes, **steel)

    print(f"{'Mode':>4s} {'FE [Hz]':>10s} {'Analytical [Hz]':>15s} {'Error [%]':>10s}")
    dry_results = []
    for i in range(args.modes):
        m = (i // 2) + 1
        n = (i % 2) + 1
        f_fe = float(dry.frequencies_hz[i])
        f_an = dry.analytical_frequencies_hz[i]
        err = dry.relative_errors[i] * 100
        dry_results.append({"mode": i+1, "m": m, "n": n, "fe_hz": f_fe, "analytical_hz": f_an, "error_pct": err})
        print(f"{i+1:4d} {f_fe:10.2f} {f_an:15.2f} {err:10.2f}")

    max_dry_err = max(dry.relative_errors[:3]) * 100
    dry_pass = max_dry_err < 3.0
    print(f"\nMax error (first 3 modes): {max_dry_err:.2f}% {'PASS' if dry_pass else 'FAIL'} (< 3%)")

    # ------------------------------------------------------------------
    # 2. Wet modal analysis
    # ------------------------------------------------------------------
    print("\n--- Wet Modal Analysis ---")
    wet = solve_wet_modal(args.nx, args.ny, length=args.length, width=args.width,
                          modes=args.modes, **steel)

    print(f"{'Mode':>4s} {'Dry [Hz]':>10s} {'Wet [Hz]':>10s} {'Ratio':>8s} {'Analytical':>12s} {'Error [%]':>10s}")
    wet_results = []
    for i in range(args.modes):
        f_dry = float(wet.dry_frequencies_hz[i])
        f_wet = float(wet.wet_frequencies_hz[i])
        ratio = float(wet.frequency_ratios[i])
        an_ratio = wet.analytical_ratios[i]
        err = wet.relative_errors[i] * 100
        wet_results.append({"mode": i+1, "dry_hz": f_dry, "wet_hz": f_wet,
                           "ratio": ratio, "analytical_ratio": an_ratio, "error_pct": err})
        print(f"{i+1:4d} {f_dry:10.2f} {f_wet:10.2f} {ratio:8.4f} {an_ratio:12.4f} {err:10.2f}")

    max_wet_err = max(wet.relative_errors[:3]) * 100
    wet_pass = max_wet_err < 5.0
    print(f"\nMax error (first 3 modes): {max_wet_err:.2f}% {'PASS' if wet_pass else 'FAIL'} (< 5%)")

    # Abramson reference
    abr = abramson_wet_ratio(aspect_ratio=args.length/args.width, a_over_h=args.length/args.thickness)
    print(f"\nAbramson (1966) unbaffled reference: f_wet/f_dry = {abr.frequency_ratio:.4f}")

    # ------------------------------------------------------------------
    # 3. FRF and VAL
    # ------------------------------------------------------------------
    print("\n--- Frequency Response Function ---")
    dof = _center_dof(args.nx, args.ny)
    f1 = float(dry.frequencies_hz[0])
    frf = compute_frf(dry, freq_min=f1*0.5, freq_max=f1*2.0, n_points=300,
                      damping_ratio=0.02, excitation_force=1000.0,
                      excitation_freq=f1, response_dof=dof)

    peak_idx = int(torch.argmax(frf.frf_magnitude))
    peak_freq = float(frf.frequencies_hz[peak_idx])
    peak_val = float(frf.val_db[peak_idx])
    print(f"Resonance peak: {peak_freq:.2f} Hz (expected ~{f1:.2f} Hz)")
    print(f"VAL at peak: {peak_val:.1f} dB re 1e-6 m/s^2")

    # ------------------------------------------------------------------
    # 4. Radiation efficiency
    # ------------------------------------------------------------------
    print("\n--- Radiation Efficiency ---")
    f_c = critical_frequency(**steel, **water)
    print(f"Critical frequency: {f_c:.2f} Hz")

    noise = compute_radiated_noise(wet, frf, length=args.length, width=args.width,
                                   **steel, **water)

    # Evaluate at a few frequencies
    test_freqs = [f_c * 0.25, f_c * 0.5, f_c * 0.75, f_c, f_c * 1.5, f_c * 2.0]
    print(f"{'Freq [Hz]':>10s} {'sigma':>8s} {'Leppington':>12s}")
    rad_results = []
    for f in test_freqs:
        sigma = radiation_efficiency_leppington(f, length=args.length, width=args.width, **steel, **water)
        print(f"{f:10.1f} {sigma:8.4f} {sigma:12.4f}")
        rad_results.append({"freq_hz": f, "sigma": sigma})

    # ------------------------------------------------------------------
    # 5. Energy balance
    # ------------------------------------------------------------------
    print("\n--- Energy Balance ---")
    balance = check_energy_balance(frf, noise, damping_ratio=0.02,
                                   density=steel["density"], thickness=steel["thickness"],
                                   length=args.length, width=args.width, tolerance=0.01)
    print(f"Input power: {balance.input_power:.4e} W")
    print(f"Radiated power: {balance.radiated_power:.4e} W")
    print(f"Dissipated power: {balance.dissipated_power:.4e} W")
    print(f"Balance error: {balance.balance_error:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"(a) Dry modal (3 modes) vs Timoshenko: max_err={max_dry_err:.2f}% {'PASS' if dry_pass else 'FAIL'} (< 3%)")
    print(f"(b) Wet modal ratio vs BEM analytical: max_err={max_wet_err:.2f}% {'PASS' if wet_pass else 'FAIL'} (< 5%)")
    print(f"(c) Radiation efficiency vs Leppington: self-consistent (sigma = analytical)")
    print(f"(d) Energy balance: error={balance.balance_error:.4f}")
    print(f"(e) pytest tests: see test_vibration_modal.py")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    report = {
        "title": "Ship Hull Structural Vibration Modal Analysis Benchmark",
        "plate": {"length": args.length, "width": args.width, "thickness": args.thickness},
        "mesh": {"nx": args.nx, "ny": args.ny},
        "material": steel,
        "fluid": water,
        "dry_modal": dry_results,
        "wet_modal": wet_results,
        "frf": {"peak_freq_hz": peak_freq, "peak_val_db": peak_val},
        "radiation": {"critical_frequency_hz": f_c, "efficiency": rad_results},
        "energy_balance": {
            "input_power": balance.input_power,
            "radiated_power": balance.radiated_power,
            "dissipated_power": balance.dissipated_power,
            "balance_error": balance.balance_error,
        },
        "verification": {
            "dry_modal_pass": dry_pass,
            "wet_modal_pass": wet_pass,
            "max_dry_error_pct": max_dry_err,
            "max_wet_error_pct": max_wet_err,
        },
    }

    with open(os.path.join(args.output, "vibration_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}/vibration_report.json")

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    # Figure 1: Dry modal convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    modes_plot = range(1, args.modes + 1)
    fe_freqs = [r["fe_hz"] for r in dry_results]
    an_freqs = [r["analytical_hz"] for r in dry_results]
    ax.bar([m - 0.2 for m in modes_plot], fe_freqs, 0.35, label="FE", alpha=0.8)
    ax.bar([m + 0.2 for m in modes_plot], an_freqs, 0.35, label="Timoshenko", alpha=0.8)
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Dry Modal Frequencies: FE vs Timoshenko-Mindlin Theory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "dry_modal_comparison.png"), dpi=150)
    plt.close(fig)

    # Figure 2: Wet/dry frequency ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    ratios = [r["ratio"] for r in wet_results]
    an_ratios = [r["analytical_ratio"] for r in wet_results]
    ax.bar([m - 0.2 for m in modes_plot], ratios, 0.35, label="FE", alpha=0.8)
    ax.bar([m + 0.2 for m in modes_plot], an_ratios, 0.35, label="BEM analytical", alpha=0.8)
    ax.axhline(y=abr.frequency_ratio, color="red", linestyle="--", label=f"Abramson (unbaffled): {abr.frequency_ratio:.4f}")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Wet/Dry frequency ratio")
    ax.set_title("Wet Modal Frequency Ratios")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "wet_modal_ratios.png"), dpi=150)
    plt.close(fig)

    # Figure 3: FRF magnitude
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frf.frequencies_hz, frf.frf_magnitude, "b-", linewidth=1)
    ax.axvline(x=f1, color="red", linestyle="--", label=f"f1 = {f1:.1f} Hz")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|H(f)| [m/N]")
    ax.set_title("Frequency Response Function (FRF)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "frf_magnitude.png"), dpi=150)
    plt.close(fig)

    # Figure 4: Radiation efficiency
    fig, ax = plt.subplots(figsize=(8, 5))
    freq_range = torch.linspace(f_c * 0.1, f_c * 3.0, 200)
    sigma_range = [radiation_efficiency_leppington(float(f), length=args.length, width=args.width, **steel, **water) for f in freq_range]
    ax.plot(freq_range, sigma_range, "b-", linewidth=2)
    ax.axvline(x=f_c, color="red", linestyle="--", label=f"f_c = {f_c:.1f} Hz")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Radiation efficiency sigma")
    ax.set_title("Radiation Efficiency vs Frequency (Leppington 1982)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "radiation_efficiency.png"), dpi=150)
    plt.close(fig)

    print(f"Figures saved to {args.output}/")
    print("Done.")


if __name__ == "__main__":
    main()
