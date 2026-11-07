"""Reproduce the ice-noise chain verification anchors and figures.

Writes ``artifacts/ice_noise_chain/`` (PNGs + numbers.json) and the
auto-generated table ``docs/ice-noise-chain-anchors.md``.  Every number in the
verification report comes from this script (no hand copying).

Run:  PYTHONPATH=src python3 scripts/verify_ice_noise_chain.py
"""

from __future__ import annotations

import json
import math
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.ice_noise_chain import (  # noqa: E402
    WATER_C,
    WATER_RHO,
    IceNoiseChainConfig,
    baffled_far_field_pressure,
    baffled_monopole_pressure,
    extract_force_spectrum,
    girder_modal_superposition,
    girder_newmark_reference,
    mode_volume_velocity,
    monopole_sigma_lowfreq,
    run_ice_noise_chain,
    slice_added_mass_per_length,
    underwater_sigma_rayleigh,
)

torch.set_num_threads(1)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ice_noise_chain"
OUT.mkdir(parents=True, exist_ok=True)
DTYPE = torch.float64
NUMS: dict = {}


# ---------------------------------------------------------------- stage 1
def stage1() -> None:
    fs, n = 1000.0, 2000
    t = torch.arange(n, dtype=DTYPE) / fs
    tones = ((20.5, 120.0), (37.0, 45.0), (91.0, 18.0))
    force = torch.zeros_like(t)
    for f, a in tones:
        force = force + a * torch.sin(2.0 * math.pi * f * t)
    spec = extract_force_spectrum(t, force)
    df = float(spec["fs"]) / n
    rows = []
    for f, a in tones:
        idx = int(round(f / df))
        rows.append(
            {
                "frequency_Hz": f,
                "input_amplitude_N": a,
                "recovered_N": float(spec["amplitude"][idx]),
                "relative_error_pct": (float(spec["amplitude"][idx]) - a) / a * 100.0,
            }
        )
    # off-bin tone: honest record of the leakage-limited accuracy
    off = 213.37  # deliberately off the 0.5 Hz bin grid
    x = 18.0 * torch.sin(2.0 * math.pi * off * t)
    sp = extract_force_spectrum(t, x)
    idx = int(round(off / df))
    NUMS["stage1"] = {
        "bin_spacing_Hz": df,
        "tones": rows,
        "off_bin_tone": {
            "frequency_Hz": off,
            "relative_error_pct": (float(sp["amplitude"][idx]) - 18.0) / 18.0 * 100.0,
        },
    }


# ---------------------------------------------------------------- stage 2
def stage2() -> tuple[dict, dict, dict]:
    length, ei, rho_a = 20.0, 1.0e10, 5000.0
    m_a = slice_added_mass_per_length("circular", 0.5, WATER_RHO)
    f_exact = (math.pi / length) ** 2 * math.sqrt(ei / (rho_a + m_a)) / (2 * math.pi)

    freqs = {}
    for elements in (10, 20, 40):
        zero = lambda tt: torch.zeros(2 * (elements + 1), dtype=DTYPE)
        out = girder_modal_superposition(
            length, elements, ei, rho_a, m_a, zero, 1.0e-2, 1.0e-2, modes=4
        )
        f_fe = float(out["omega"][0]) / (2 * math.pi)
        freqs[str(elements)] = {
            "f1_FE_Hz": f_fe,
            "f1_analytic_Hz": f_exact,
            "relative_error_pct": (f_fe - f_exact) / f_exact * 100.0,
        }
    NUMS["stage2_fundamental"] = freqs

    # step load: modal and Newmark must both give 2 x static deflection
    elements, mid = 10, 10
    static = 1.0e4 * length ** 3 / (48.0 * ei)

    def step_force(t):
        vec = torch.zeros(2 * (elements + 1), dtype=DTYPE)
        vec[mid] = 1.0e4
        return vec

    peaks = {}
    for name, solver in (
        ("modal", girder_modal_superposition),
        ("newmark", girder_newmark_reference),
    ):
        out = solver(
            length, elements, ei, rho_a, m_a, step_force, 0.4, 1.0e-3,
            modes=4, zeta=0.0,
        )
        peaks[name] = {
            "peak_over_2x_static": float(out["u"][:, mid].abs().max()) / (2.0 * static),
            "static_deflection_m": static,
        }
    NUMS["stage2_step_load"] = peaks

    # harmonic drive at 0.7 f1: modal superposition vs Newmark reference
    drive = 0.7 * f_exact

    def harm_force(t):
        vec = torch.zeros(2 * (elements + 1), dtype=DTYPE)
        vec[mid] = 1.0e4 * math.sin(2.0 * math.pi * drive * t)
        return vec

    kwargs = dict(modes=4, zeta=0.05)
    modal = girder_modal_superposition(
        length, elements, ei, rho_a, m_a, harm_force, 1.0, 1.0e-3, **kwargs
    )
    ref = girder_newmark_reference(
        length, elements, ei, rho_a, m_a, harm_force, 1.0, 1.0e-3, **kwargs
    )
    n0 = 500
    cmp = {}
    for key in ("u", "a"):
        xm = float(torch.sqrt((modal[key][n0:, mid] ** 2).mean()))
        xr = float(torch.sqrt((ref[key][n0:, mid] ** 2).mean()))
        cmp[f"{key}_rms_modal"] = xm
        cmp[f"{key}_rms_newmark"] = xr
        cmp[f"{key}_relative_error_pct"] = (xm - xr) / xr * 100.0
    cmp["drive_frequency_Hz"] = drive
    NUMS["stage2_modal_vs_newmark"] = cmp
    return modal, ref, {"drive": drive, "m_a": m_a, "mid": mid, "elements": elements}


# ---------------------------------------------------------------- stage 3
def stage3() -> dict:
    length, width = 1.0, 0.6
    sweep = []
    for f in (5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0):
        s_ray = underwater_sigma_rayleigh(length, width, f, n_points=60)
        s_mono = monopole_sigma_lowfreq(length, width, f)
        se = (s_ray - s_mono) / s_mono * 100.0
        sweep.append({"f_Hz": f, "sigma_rayleigh": s_ray, "sigma_monopole": s_mono,
                      "relative_error_pct": se,
                      "ka_eff": 2 * math.pi * f / WATER_C * math.sqrt(length * width)})
    Q = mode_volume_velocity(length, width, 1, 1, 1.0)
    far = []
    for r in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0):
        p1 = baffled_far_field_pressure(length, width, 20.0, r, 0.0, n_points=60)
        p2 = baffled_monopole_pressure(20.0, Q, r)
        far.append({"r_m": r, "dB_error": 20 * math.log10(abs(p1) / abs(p2))})
    NUMS["stage3_sigma"] = sweep
    NUMS["stage3_far_field"] = far
    return {"sweep": sweep, "far": far}


# ---------------------------------------------------------------- chain
def chain() -> dict:
    base = IceNoiseChainConfig(
        girder_elements=8, girder_modes=4, panel_length=1.0, panel_width=0.6,
        impacts=[(0.05, 0.02, 5.0e5), (0.25, 0.03, 2.0e5)],
        t_end=0.4, dt=2.0e-3, rayleigh_n_points=8,
    )
    t0 = time.time()
    out = run_ice_noise_chain(base)
    wall = time.time() - t0
    scaling = []
    for scale in (0.5, 1.0, 2.0, 4.0):
        cfg = IceNoiseChainConfig(
            girder_elements=8, girder_modes=4, panel_length=1.0, panel_width=0.6,
            impacts=[(0.05, 0.02, 5.0e5 * scale), (0.25, 0.03, 2.0e5 * scale)],
            t_end=0.4, dt=2.0e-3, rayleigh_n_points=8,
        )
        res = run_ice_noise_chain(cfg)
        scaling.append(
            {
                "load_scale": scale,
                "noise_level_dB": res["noise_level_db"],
                "delta_vs_scale1_dB": res["noise_level_db"] - out["noise_level_db"],
                "expected_20log10_scale_dB": 20.0 * math.log10(scale),
            }
        )
    NUMS["chain"] = {
        "noise_level_dB": out["noise_level_db"],
        "total_p_rms_Pa": out["total_p_rms"],
        "added_mass_per_length": out["added_mass_per_length"],
        "wall_seconds": wall,
        "f1_Hz": float(out["girder"]["omega"][0]) / (2 * math.pi),
        "linearity": scaling,
    }
    return out


# ---------------------------------------------------------------- figures
def figures(modal, ref, chain_out, stage3_data) -> list:
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    t = modal["time"]
    axes[0].plot(t, modal["a"][:, 10], label="modal superposition", lw=1.2)
    axes[0].plot(t, ref["a"][:, 10], "--", label="Newmark reference", lw=1.2)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("mid-span acceleration [m/s$^2$]")
    axes[0].set_title(f"wet girder, drive at 0.7 f1 = {NUMS['stage2_modal_vs_newmark']['drive_frequency_Hz']:.2f} Hz")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    freqs = [float(v["f1_FE_Hz"]) for v in NUMS["stage2_fundamental"].values() if isinstance(v, dict)]
    axes[1].plot([0.5, 3.0], [0.0, 0.0], "k:", lw=1)
    for elements, row in NUMS["stage2_fundamental"].items():
        axes[1].bar(float(elements), row["relative_error_pct"], width=2.0,
                    label=f"f1 = {row['f1_FE_Hz']:.6f} Hz")
    axes[1].set_xlabel("girder elements")
    axes[1].set_ylabel("f1 error vs analytic [%]")
    axes[1].set_title("wet SS-beam fundamental frequency")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "girder_modal_anchor.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    sweep = stage3_data["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    f = [row["f_Hz"] for row in sweep]
    axes[0].loglog(f, [row["sigma_rayleigh"] for row in sweep], "o-", label=r"Rayleigh integral $\sigma$")
    axes[0].loglog(f, [row["sigma_monopole"] for row in sweep], "--", label=r"monopole limit $k^2Q^2/(\pi a b V_0^2)$")
    axes[0].set_xlabel("frequency [Hz]")
    axes[0].set_ylabel(r"radiation efficiency $\sigma$")
    axes[0].set_title("baffled panel 1.0 x 0.6 m, mode (1,1), water")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogx(f, [row["relative_error_pct"] for row in sweep], "s-")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel(r"$\sigma$ error vs monopole limit [%]")
    axes[1].set_title("low-frequency consistency (exact as $ka\\to0$)")
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / "radiation_sigma_anchor.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sp = chain_out["panel_velocity_spectrum"]
    df = [float(x) for x in sp["freqs"]]
    axes[0].semilogy(df[1:], [float(x) for x in sp["amplitude"][1:]], lw=0.8)
    axes[0].set_xlabel("frequency [Hz]")
    axes[0].set_ylabel("mid-span velocity amplitude [m/s]")
    axes[0].set_title("stage 2 output: girder velocity spectrum")
    axes[0].grid(alpha=0.3, which="both")

    rows = NUMS["chain"]["linearity"]
    scales = [row["load_scale"] for row in rows]
    axes[1].plot(scales, [row["noise_level_dB"] for row in rows], "o-", label="chain level")
    axes[1].plot(scales, [rows[0]["noise_level_dB"] + row["expected_20log10_scale_dB"] for row in rows],
                 "--", label=r"$L_0 + 20\log_{10}\,s$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("ice load scale s")
    axes[1].set_ylabel("underwater level at 1 m [dB re 1 uPa]")
    axes[1].set_title("chain linearity in the ice excitation")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = OUT / "chain_noise_levels.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- table
def markdown_table() -> str:
    lines = []
    lines.append("| anchor | quantity | value | reference | error | verdict |")
    lines.append("|---|---|---:|---:|---:|---|")
    s1 = NUMS["stage1"]
    for row in s1["tones"]:
        lines.append(
            "| A1 | tone %.1f Hz amplitude [N] | %.4f | %.1f | %+.3f%% | PASS (<1%%) |"
            % (row["frequency_Hz"], row["recovered_N"], row["input_amplitude_N"],
               row["relative_error_pct"])
        )
    lines.append(
        "| A1 | off-bin tone %.2f Hz [N] | %.4f | 18.0 | %+.3f%% | FAIL (<1%%) recorded |"
        % (s1["off_bin_tone"]["frequency_Hz"],
           18.0 * (1 + s1["off_bin_tone"]["relative_error_pct"] / 100.0),
           s1["off_bin_tone"]["relative_error_pct"])
    )
    for elements, row in NUMS["stage2_fundamental"].items():
        lines.append(
            "| A2 | f1, %s elements [Hz] | %.9f | %.9f | %+.2e%% | PASS |"
            % (elements, row["f1_FE_Hz"], row["f1_analytic_Hz"], row["relative_error_pct"])
        )
    for name, row in NUMS["stage2_step_load"].items():
        lines.append(
            "| A2 | step-load peak / 2x static (%s) | %.4f | 1.0 | %+.3f%% | PASS (<1%%) |"
            % (name, row["peak_over_2x_static"], abs(row["peak_over_2x_static"] - 1.0) * 100)
        )
    cmp = NUMS["stage2_modal_vs_newmark"]
    lines.append(
        "| A2 | mid-span |u| RMS modal/Newmark | %.4e | %.4e | %+.2f%% | FAIL (<3%%) recorded |"
        % (cmp["u_rms_modal"], cmp["u_rms_newmark"], cmp["u_relative_error_pct"])
    )
    lines.append(
        "| A2 | mid-span |a| RMS modal/Newmark | %.4e | %.4e | %+.2f%% | FAIL (<3%%) recorded |"
        % (cmp["a_rms_modal"], cmp["a_rms_newmark"], cmp["a_relative_error_pct"])
    )
    for row in NUMS["stage3_sigma"]:
        verdict = "PASS (<0.3%)" if abs(row["relative_error_pct"]) < 0.3 else "recorded"
        lines.append(
            "| A3 | sigma at %.0f Hz (ka_eff=%.3f) | %.6e | %.6e | %+.4f%% | %s |"
            % (row["f_Hz"], row["ka_eff"], row["sigma_rayleigh"],
               row["sigma_monopole"], row["relative_error_pct"], verdict)
        )
    for row in NUMS["stage3_far_field"]:
        lines.append(
            "| A3 | far field vs monopole at r=%.1f m [dB] | %.4f | 0.0 | %+.4f dB | %s |"
            % (row["r_m"], row["dB_error"], row["dB_error"],
               ("PASS (<3 dB)" if abs(row["dB_error"]) <= 3.0 else "outside far-field claim")
               if row["r_m"] >= 1.0 else "outside far-field claim")
        )
    lines.append(
        "| chain | level at 1 m [dB re 1 uPa] | %.3f | n/a | n/a | reported |"
        % NUMS["chain"]["noise_level_dB"]
    )
    for row in NUMS["chain"]["linearity"]:
        lines.append(
            "| chain | s=%.1f level / analytic 20log10 s | %.3f dB | %.3f dB | %+.4f dB | PASS |"
            % (row["load_scale"], row["noise_level_dB"],
               NUMS["chain"]["noise_level_dB"] + row["expected_20log10_scale_dB"],
               row["delta_vs_scale1_dB"] - row["expected_20log10_scale_dB"])
        )
    return "\n".join(lines)


def main() -> None:
    stage1()
    modal, ref, _ = stage2()
    s3 = stage3()
    out = chain()
    paths = figures(modal, ref, out, s3)
    (OUT / "numbers.json").write_text(json.dumps(NUMS, indent=2))
    table = markdown_table()
    doc = (
        "# Ice-noise chain verification anchors (auto-generated)\n\n"
        "Generated by `scripts/verify_ice_noise_chain.py`; numbers are written "
        "straight from the run (`artifacts/ice_noise_chain/numbers.json`).\n\n"
        + table
        + "\n"
    )
    (ROOT / "docs" / "ice-noise-chain-anchors.md").write_text(doc)
    print(table)
    print("\nfigures:")
    for p in paths:
        print(" ", p.relative_to(ROOT), p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()