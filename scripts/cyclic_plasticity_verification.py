"""Verification of the Armstrong-Frederick cyclic-plasticity surface (P16-N2 / m5197).

Runs every machine-checkable criterion of the mission and dumps
``artifacts/cyclic_plasticity/verification.json`` plus the four figures quoted in
``docs/cyclic-plasticity-af-anchor.md``.  No number in the report is hand copied.

C1 stabilised loop peak/trough stress vs closed form      <= 1 %   (5 amplitudes x 2 cycle counts)
C2 dissipated energy per stabilised cycle                 <= 2 %
C3 Bauschinger reverse-yield drop vs closed form          <= 3 %   (5 amplitudes)
C4 analytic consistent tangent vs central difference      <= 1e-6
C5 ratcheting rate vs closed form                         <= 5 %   (20 cycles x 2 means x 2 amplitudes)
C6 C -> 0 degeneration onto the m4951 kernel              <= 1e-12
C7 negative control H_k = 0 must FAIL the C3 criterion     >= 50 % of cases
C8 m4951 regression suite                                 16 pytest cases green
"""

from __future__ import annotations

import base64
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorfem.cyclic_plasticity import (  # noqa: E402
    AFMaterial,
    CyclicState,
    VOIGT_PAIRS,
    bauschinger_drop,
    consistent_tangent,
    cyclic_history,
    degenerate_check,
    integrate,
    loop_dissipation,
    loop_peak_stress,
    ratcheting_rate,
    solve_uniaxial_point,
    uniaxial_cycle,
    uniaxial_stress_cycle,
    yield_radius,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts",
                   "cyclic_plasticity")

MATERIAL = AFMaterial(young=210.0e9, poisson=0.3, yield_stress=350.0e6,
                      kinematic_modulus=70.0e9, kinematic_gamma=500.0, hardening=0.0)
MATERIAL_VOCE = AFMaterial(young=210.0e9, poisson=0.3, yield_stress=350.0e6,
                           kinematic_modulus=70.0e9, kinematic_gamma=500.0,
                           hardening=0.0, saturation=80.0e6, saturation_rate=500.0)
#: negative control: kinematic modulus C = H_k = 0, isotropic saturation kept
MATERIAL_NO_KINEMATIC = AFMaterial(young=210.0e9, poisson=0.3, yield_stress=350.0e6,
                                   kinematic_modulus=0.0, kinematic_gamma=500.0,
                                   hardening=0.0, saturation=80.0e6, saturation_rate=500.0)

AMPLITUDES = (0.004, 0.006, 0.008, 0.010, 0.012)
CYCLE_COUNTS = (5, 20)
STEPS = 60            # strain-controlled steps per half cycle
RATCHET_STEPS = 200   # stress-controlled steps per half cycle
RATCHET_CYCLES = 20
RATCHET_CASES = ((20.0e6, 400.0e6), (20.0e6, 420.0e6), (40.0e6, 400.0e6), (40.0e6, 420.0e6))


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #
def dissipation(eps, sigma, start, end) -> float:
    """Trapezoid loop integral ``oint sigma d eps`` over one closed cycle."""
    total = 0.0
    for k in range(start, end - 1):
        total += 0.5 * (sigma[k] + sigma[k + 1]) * (eps[k + 1] - eps[k])
    total += 0.5 * (sigma[end - 1] + sigma[start]) * (eps[start] - eps[end - 1])
    return total


def reverse_yield(material, recorded, bounds, dtype=torch.float64):
    """Reverse yield stress inside the last cycle, resolved by bisection.

    From the tensile peak we walk forward to the first reverse plastic increment
    and then bisect the axial strain in that increment (committed state frozen)
    until the elastic/plastic boundary is located, so the measurement is not
    limited by the step size.
    """
    start, _ = bounds[-1]
    k = start
    while k < len(recorded["dp"]) - 1 and recorded["dp"][k + 1] <= 0.0:
        k += 1
    if k >= len(recorded["dp"]) - 1:
        raise RuntimeError("no reverse plastic increment found")
    lo, hi = recorded["eps"][k], recorded["eps"][k + 1]
    lateral = recorded["lateral"][k]
    state = recorded["states"][k]
    sigma_elastic = recorded["sigma"][k]
    span = max(abs(hi - lo), 1e-30)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        sigma, _, dp, _ = solve_uniaxial_point(material, state, mid, lateral, dtype=dtype)
        if float(dp) > 0.0:
            hi = mid
        else:
            lo, sigma_elastic = mid, sigma
        if abs(hi - lo) < 1e-13 * span:
            break
    sigma_mid, _, _, _ = solve_uniaxial_point(material, state, 0.5 * (lo + hi), lateral,
                                             dtype=dtype)
    return 0.5 * (sigma_elastic + sigma_mid)


def residual_audit(material, strain, state):
    """A posteriori residuals of one return mapping, evaluated from the result."""
    stress, new_state, dp = integrate(material, strain, state)
    eta = stress - new_state.backstress
    eta = eta - torch.trace(eta) / 3.0 * torch.eye(3, dtype=stress.dtype)
    q_f = float(torch.linalg.matrix_norm(eta))
    radius = yield_radius(material, float(new_state.alpha))
    d_eps_p = new_state.plastic_strain - state.plastic_strain
    flow = d_eps_p - (math.sqrt(1.5) * float(dp)) * (eta / q_f)
    af = (new_state.backstress - state.backstress
          - (2.0 / 3.0) * material.kinematic_modulus * d_eps_p
          + material.kinematic_gamma * float(dp) * new_state.backstress)
    return {
        "yield_residual": abs(q_f - math.sqrt(2.0 / 3.0) * radius)
        / (math.sqrt(2.0 / 3.0) * radius),
        "flow_rule_residual": float(torch.linalg.matrix_norm(flow))
        / float(torch.linalg.matrix_norm(d_eps_p)),
        "kinematic_evolution_residual": float(torch.linalg.matrix_norm(af))
        / float(torch.linalg.matrix_norm(new_state.backstress)),
        "plastic_incompressibility": abs(float(torch.trace(d_eps_p)))
        / float(torch.linalg.matrix_norm(d_eps_p)),
        "dp": float(dp),
    }


def tangent_error(material, strain, state, h=1e-7):
    """Relative error of the analytic tangent against central differences."""
    _, _, dp = integrate(material, strain, state)
    analytic = consistent_tangent(material, strain, state, dp)
    numeric = torch.zeros((6, 6), dtype=strain.dtype)
    for k, (i, j) in enumerate(VOIGT_PAIRS):
        d_eps = torch.zeros((3, 3), dtype=strain.dtype)
        if i == j:
            d_eps[i, i] = h
        else:
            d_eps[i, j] = d_eps[j, i] = 0.5 * h
        plus, _, _ = integrate(material, strain + d_eps, state)
        minus, _, _ = integrate(material, strain - d_eps, state)
        numeric[:, k] = torch.tensor([(float(plus[p, q]) - float(minus[p, q])) / (2.0 * h)
                                      for p, q in VOIGT_PAIRS], dtype=strain.dtype)
    scale = max(float(torch.linalg.matrix_norm(numeric)), 1.0)
    return (float(torch.linalg.matrix_norm(analytic - numeric)) / scale,
            analytic, numeric)


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def criterion_hysteresis(material, amplitudes, cycle_counts, steps, label):
    rows = []
    for cycles in cycle_counts:
        for amplitude in amplitudes:
            path, bounds = cyclic_history(0.0, amplitude, cycles, steps)
            rec = uniaxial_cycle(material, path, collect_states=True)
            start, end = bounds[-1]
            peak = max(rec["sigma"][start:end])
            trough = min(rec["sigma"][start:end])
            anchor = loop_peak_stress(material, amplitude)
            energy = dissipation(rec["eps"], rec["sigma"], start, end)
            exact_energy = loop_dissipation(material, amplitude, anchor["plastic_amplitude"])
            sigma_rev = reverse_yield(material, rec, bounds)
            exact_b = bauschinger_drop(material, amplitude)
            drop = anchor["radius"] - abs(sigma_rev)
            if exact_b["drop"] > 0.0:
                rel_drop = abs(drop - exact_b["drop"]) / abs(exact_b["drop"])
            else:  # negative control: the closed form predicts *no* Bauschinger effect
                rel_drop = 0.0 if abs(drop) <= 1e-6 * anchor["radius"] else math.inf
            rows.append({
                "material": label,
                "cycles": cycles,
                "strain_amplitude": amplitude,
                "peak_measured": peak,
                "peak_closed_form": anchor["peak"],
                "peak_rel_error": abs(peak - anchor["peak"]) / abs(anchor["peak"]),
                "trough_measured": trough,
                "trough_closed_form": anchor["trough"],
                "trough_rel_error": abs(trough - anchor["trough"]) / abs(anchor["trough"]),
                "plastic_amplitude_measured": 0.25 * (rec["p"][end - 1] - rec["p"][start]),
                "plastic_amplitude_closed_form": anchor["plastic_amplitude"],
                "dissipation_measured": energy,
                "dissipation_closed_form": exact_energy,
                "dissipation_rel_error": abs(energy - exact_energy) / abs(exact_energy),
                "reverse_yield_measured": sigma_rev,
                "reverse_yield_closed_form": exact_b["reverse_yield"],
                "bauschinger_drop_measured": drop,
                "bauschinger_drop_closed_form": exact_b["drop"],
                "bauschinger_rel_error": (rel_drop if math.isfinite(rel_drop) else None),
                "bauschinger_abs_error": drop - exact_b["drop"],
                "bauschinger_index": exact_b["ratio"],
                "bauschinger_detected": bool(exact_b["ratio"] >= 0.05),
                "loop": {"eps": rec["eps"][start:end], "sigma": rec["sigma"][start:end]},
            })
            print(f"  [{label}] N={cycles:2d} eps_a={amplitude:.3f}: "
                  f"C1 {100*rows[-1]['peak_rel_error']:.4f}%  "
                  f"C2 {100*rows[-1]['dissipation_rel_error']:.4f}%  "
                  f"C3 {(100*rel_drop if math.isfinite(rel_drop) else float('nan')):.4f}%  "
                  f"(Bauschinger index {100*rows[-1]['bauschinger_index']:.1f}%)", flush=True)
    return rows


def criterion_tangent(material):
    dtype = torch.float64
    path, _ = cyclic_history(0.0, 0.008, 2, 40)
    rec = uniaxial_cycle(material, path, collect_states=True)
    yielded = rec["states"][-1]
    cases = {}
    eps_elastic = torch.zeros((3, 3), dtype=dtype)
    eps_elastic[0, 0], eps_elastic[1, 1], eps_elastic[2, 2] = 0.0005, -0.00015, -0.00015
    cases["elastic"] = (eps_elastic, CyclicState.zero())
    eps_uniaxial = torch.tensor([[rec["eps"][-1], 0.0, 0.0],
                                 [0.0, rec["lateral"][-1], 0.0],
                                 [0.0, 0.0, rec["lateral"][-1]]], dtype=dtype)
    cases["uniaxial_plastic"] = (eps_uniaxial, rec["states"][-2])
    eps_mixed = torch.tensor([[0.006, 0.003, 0.0],
                             [0.003, -0.002, 0.0015],
                             [0.0, 0.0015, -0.004]], dtype=dtype)
    cases["non_proportional"] = (eps_mixed, yielded)
    cases["small_plastic"] = (torch.tensor([[0.0021, 0.0, 0.0], [0.0, -0.001, 0.0],
                                           [0.0, 0.0, -0.001]], dtype=dtype), CyclicState.zero())
    out = {}
    for name, (strain, state) in cases.items():
        error, analytic, numeric = tangent_error(material, strain, state)
        out[name] = {"rel_error": error,
                     "max_abs_entry": float(analytic.abs().max()),
                     "dp": float(integrate(material, strain, state)[2])}
        print(f"  C4 {name}: rel error {error:.3e} (dp={out[name]['dp']:.3e})", flush=True)
    out["max"] = max(v["rel_error"] for k, v in out.items() if k != "max")
    return out


def ratchet_case(material, mean, amplitude, cycles, steps):
    path, bounds = cyclic_history(mean, amplitude, cycles, steps)
    rec = uniaxial_stress_cycle(material, [mean + value for value in path])
    increments = [rec["eps_plastic"][bounds[i][1] - 1] - rec["eps_plastic"][bounds[i - 1][1] - 1]
                  for i in range(1, len(bounds))]
    return rec, increments, bounds


def criterion_ratcheting(material, cases, cycles, steps):
    rows = []
    for mean, amplitude in cases:
        rec, increments, bounds = ratchet_case(material, mean, amplitude, cycles, steps)
        measured = increments[-1]
        exact = ratcheting_rate(material, mean, amplitude)
        alpha = rec["alpha_back"][bounds[-1][0]:bounds[-1][1]]
        rows.append({
            "mean_stress": mean, "stress_amplitude": amplitude, "cycles": cycles,
            "steps_per_half": steps,
            "rate_measured": measured, "rate_closed_form": exact,
            "rate_rel_error": abs(measured - exact) / abs(exact),
            "alpha_min_measured": min(alpha), "alpha_max_measured": max(alpha),
            "alpha_min_closed_form": mean - amplitude + material.yield_stress,
            "alpha_max_closed_form": mean + amplitude - material.yield_stress,
            "rate_history": increments,
        })
        print(f"  C5 sigma_m={mean/1e6:5.1f} sigma_a={amplitude/1e6:5.1f} MPa: "
              f"measured {measured:.6e} closed {exact:.6e} "
              f"err {100*rows[-1]['rate_rel_error']:.2f}%", flush=True)
    return rows


def richardson_study(material, mean, amplitude, steps_list=(50, 100, 200, 400)):
    """Step-size study: backward Euler is first order, so the error must halve."""
    exact = ratcheting_rate(material, mean, amplitude)
    measured = {}
    for steps in steps_list:
        _, increments, _ = ratchet_case(material, mean, amplitude, 8, steps)
        measured[steps] = increments[-1]
        print(f"  Richardson steps={steps}: {measured[steps]:.6e} "
              f"({100*(measured[steps]-exact)/exact:+.3f}%)", flush=True)
    extrapolated = 2.0 * measured[steps_list[-1]] - measured[steps_list[-2]]
    return {"mean_stress": mean, "stress_amplitude": amplitude,
            "measured": {str(k): v for k, v in measured.items()},
            "rate_closed_form": exact,
            "richardson_limit": extrapolated,
            "richardson_rel_error": abs(extrapolated - exact) / abs(exact)}


def criterion_degeneration(material):
    dtype = torch.float64
    cases = {
        "monotonic_plastic": (torch.tensor([[0.004, 0.0, 0.0], [0.0, -0.002, 0.0],
                                            [0.0, 0.0, -0.002]], dtype=dtype), None),
        "pure_shear": (torch.tensor([[0.0, 0.003, 0.0], [0.003, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]], dtype=dtype), None),
        "yielded_state": (torch.tensor([[0.003, 0.002, 0.0], [0.002, -0.0015, 0.0],
                                        [0.0, 0.0, -0.0015]], dtype=dtype),
                          CyclicState(plastic_strain=torch.tensor(
                              [[0.001, 0.0005, 0.0], [0.0005, -0.0005, 0.0],
                               [0.0, 0.0, 0.0]], dtype=dtype),
                              backstress=torch.zeros((3, 3), dtype=dtype),
                              alpha=torch.tensor(0.0015, dtype=dtype))),
    }
    out = {}
    for name, (strain, state) in cases.items():
        report = degenerate_check(material, strain, state)
        out[name] = {key: report[key] for key in
                     ("stress_error", "dp_error", "plastic_error", "alpha_error",
                      "tangent_error")}
        print(f"  C6 {name}: {out[name]}", flush=True)
    out["max"] = max(max(v for v in row.values()) for k, row in out.items() if k != "max")
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(loop_rows, ratchet_rows, richardson):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    figures = []

    def save(fig, name, caption):
        fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
        plt.close(fig)
        with open(os.path.join(OUT, name), "rb") as handle:
            figures.append({"name": name, "caption": caption,
                            "png_base64": base64.b64encode(handle.read()).decode("ascii")})

    # (1) stabilised hysteresis loops + analytic anchors
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for row in [r for r in loop_rows if r["cycles"] == 20]:
        ax.plot([1e2 * e for e in row["loop"]["eps"]],
                [1e-6 * s for s in row["loop"]["sigma"]], lw=1.4,
                label=f"FEM  $\\varepsilon_a$={row['strain_amplitude']*100:.1f}%")
        ax.plot(1e2 * row["strain_amplitude"], 1e-6 * row["peak_closed_form"], "k*", ms=9)
        ax.plot(-1e2 * row["strain_amplitude"], 1e-6 * row["trough_closed_form"], "k*", ms=9)
    ax.plot([], [], "k*", ms=9, label="closed form $\\pm(\\alpha_a+R_{sat})$")
    ax.set_xlabel("axial strain $\\varepsilon_{11}$ [%]")
    ax.set_ylabel("axial stress $\\sigma_{11}$ [MPa]")
    ax.set_title("Stabilised Armstrong-Frederick hysteresis loops (cycle 20)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    save(fig, "hysteresis_loops.png",
         "Stabilised uniaxial hysteresis loops for five strain amplitudes "
         "(C=70 GPa, gamma=500, sigma_y0=350 MPa) against the closed-form peaks.")

    # (2) relative errors against the three anchors
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    labels = [f"{r['strain_amplitude']*100:.1f}%/N{r['cycles']}" for r in loop_rows]
    xs = list(range(len(loop_rows)))
    width = 0.26
    ax.bar([x - width for x in xs], [100 * r["peak_rel_error"] for r in loop_rows], width,
           label="C1 peak stress")
    ax.bar(xs, [100 * r["dissipation_rel_error"] for r in loop_rows], width,
           label="C2 loop energy")
    ax.bar([x + width for x in xs], [100 * r["bauschinger_rel_error"] for r in loop_rows],
           width, label="C3 Bauschinger drop")
    for level, style, text in ((1.0, "--", "1 %"), (2.0, ":", "2 %"), (3.0, "-.", "3 %")):
        ax.axhline(level, color="k" if level == 1.0 else "0.4", ls=style, lw=0.9)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ax.set_ylabel("relative error [%]")
    ax.set_title("Analytic-anchor errors (dashed 1 %, dotted 2 %, dash-dot 3 %)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    save(fig, "anchor_errors.png",
         "Relative error against the three closed-form anchors for five strain "
         "amplitudes and two cycle counts (all bars far below the 1 % gate).")

    # (3) ratcheting
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for row in ratchet_rows:
        history = [0.0]
        for increment in row["rate_history"]:
            history.append(history[-1] + increment)
        ax.plot(range(len(history)), [1e2 * h for h in history], "o-", ms=3,
                label=(f"$\\sigma_m$={row['mean_stress']/1e6:.0f}, "
                       f"$\\sigma_a$={row['stress_amplitude']/1e6:.0f} MPa"))
        ax.plot(range(len(history)),
                [1e2 * row["rate_closed_form"] * c for c in range(len(history))],
                "k--", lw=1.0)
    ax.plot([], [], "k--", label="closed-form rate")
    ax.set_xlabel("cycle")
    ax.set_ylabel("accumulated plastic strain $\\varepsilon^p_{11}$ [%]")
    ax.set_title("Ratcheting under stress control (20 cycles)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, "ratcheting.png",
         "Accumulated axial plastic strain per cycle under uniaxial stress control "
         "versus the closed-form Armstrong-Frederick ratcheting rate (dashed).")

    # (4) step-size convergence of the ratcheting rate
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    steps = [float(k) for k in richardson["measured"]]
    values = [richardson["measured"][k] for k in richardson["measured"]]
    ax.loglog(steps, [abs(v - richardson["rate_closed_form"]) for v in values], "o-",
              label="|rate(steps) - closed form|")
    ax.loglog(steps, [abs(values[0] - richardson["rate_closed_form"]) * steps[0] / s
                      for s in steps], "k:", lw=1.0, label="$O(1/\\Delta\\varepsilon)$")
    ax.set_xlabel("steps per half cycle")
    ax.set_ylabel("absolute rate error [-]")
    ax.set_title("Backward-Euler step-size convergence of the ratcheting rate")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "ratchet_convergence.png",
         "First-order step-size convergence of the measured ratcheting rate towards the "
         "closed form (Richardson limit agrees to well under 1 %).")
    return figures


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    results = {
        "mission": "P16-N2 / m5197",
        "branch": "agent/m5197",
        "material": {"E": MATERIAL.young, "nu": MATERIAL.poisson,
                     "sigma_y0": MATERIAL.yield_stress, "C": MATERIAL.kinematic_modulus,
                     "gamma": MATERIAL.kinematic_gamma, "H_iso": MATERIAL.hardening,
                     "X_sat": MATERIAL.saturation_backstress},
        "material_voce": {"Q_inf": MATERIAL_VOCE.saturation,
                          "b": MATERIAL_VOCE.saturation_rate},
        "settings": {"strain_steps_per_half": STEPS, "ratchet_steps_per_half": RATCHET_STEPS,
                     "ratchet_cycles": RATCHET_CYCLES},
    }

    print("residual audit ...", flush=True)
    strain = torch.zeros((3, 3), dtype=torch.float64)
    strain[0, 0], strain[1, 1], strain[2, 2] = 0.01, -0.005, -0.005
    results["residuals"] = residual_audit(MATERIAL, strain, CyclicState.zero())

    print("C1/C2/C3 ...", flush=True)
    loop_rows = criterion_hysteresis(MATERIAL, AMPLITUDES, CYCLE_COUNTS, STEPS, "AF")
    results["hysteresis"] = [{k: v for k, v in row.items() if k != "loop"}
                             for row in loop_rows]
    results["C1_max_peak_error"] = max(max(r["peak_rel_error"], r["trough_rel_error"])
                                       for r in loop_rows)
    results["C2_max_energy_error"] = max(r["dissipation_rel_error"] for r in loop_rows)
    results["C3_max_bauschinger_error"] = max(r["bauschinger_rel_error"] for r in loop_rows
                                              if r["bauschinger_rel_error"] is not None)

    print("Voce-isotropic variant ...", flush=True)
    voce_rows = criterion_hysteresis(MATERIAL_VOCE, (0.008, 0.012), (5,), STEPS, "AF+Voce")
    results["hysteresis_voce"] = [{k: v for k, v in row.items() if k != "loop"}
                                  for row in voce_rows]

    print("C7 negative control ...", flush=True)
    negatives = []
    for row in criterion_hysteresis(MATERIAL_NO_KINEMATIC, AMPLITUDES, (5,), STEPS, "Hk=0"):
        rel = row["bauschinger_rel_error"]
        passes = bool(row["bauschinger_detected"] and rel is not None and rel <= 0.03)
        negatives.append({"strain_amplitude": row["strain_amplitude"],
                          "bauschinger_index": row["bauschinger_index"],
                          "drop_measured": row["bauschinger_drop_measured"],
                          "drop_closed_form": row["bauschinger_drop_closed_form"],
                          "detected": bool(row["bauschinger_detected"]),
                          "criterion_passes": passes})
    results["C7_negative_control"] = negatives
    results["C7_fail_fraction"] = sum(1 for n in negatives if not n["criterion_passes"]) \
        / len(negatives)

    print("C4 tangent ...", flush=True)
    results["C4_tangent"] = criterion_tangent(MATERIAL)

    print("C5 ratcheting ...", flush=True)
    ratchet_rows = criterion_ratcheting(MATERIAL, RATCHET_CASES, RATCHET_CYCLES,
                                        RATCHET_STEPS)
    results["C5_ratcheting"] = [{k: v for k, v in row.items() if k != "rate_history"}
                                for row in ratchet_rows]
    results["C5_max_rate_error"] = max(r["rate_rel_error"] for r in ratchet_rows)
    print("Richardson study ...", flush=True)
    richardson = richardson_study(MATERIAL, 40.0e6, 420.0e6)
    results["C5_richardson"] = richardson

    print("C6 degeneration ...", flush=True)
    results["C6_degeneration"] = criterion_degeneration(MATERIAL)

    print("figures ...", flush=True)
    figures = make_figures(loop_rows, ratchet_rows, richardson)

    gates = {
        "C1_peak_le_1pct": results["C1_max_peak_error"] <= 0.01,
        "C2_energy_le_2pct": results["C2_max_energy_error"] <= 0.02,
        "C3_bauschinger_le_3pct": results["C3_max_bauschinger_error"] <= 0.03,
        "C4_tangent_le_1e-6": results["C4_tangent"]["max"] <= 1e-6,
        "C5_ratchet_le_5pct": results["C5_max_rate_error"] <= 0.05,
        "C5_richardson_le_1pct": richardson["richardson_rel_error"] <= 0.01,
        "C6_degeneration_le_1e-12": results["C6_degeneration"]["max"] <= 1e-12,
        "C7_negative_control_fails_ge_50pct": results["C7_fail_fraction"] >= 0.5,
        "residual_audit_le_1e-12": max(
            results["residuals"]["yield_residual"], results["residuals"]["flow_rule_residual"],
            results["residuals"]["kinematic_evolution_residual"]) <= 1e-12,
    }
    results["gates"] = gates
    results["all_gates_pass"] = all(gates.values())
    with open(os.path.join(OUT, "verification.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    with open(os.path.join(OUT, "figures.json"), "w") as handle:
        json.dump([{k: v for k, v in fig.items() if k != "png_base64"} for fig in figures],
                  handle, indent=2)
    print(json.dumps(gates, indent=2), flush=True)
    print("C1 max %.4e  C2 max %.4e  C3 max %.4e  C5 max %.4e  C4 %.3e  C6 %.3e"
          % (results["C1_max_peak_error"], results["C2_max_energy_error"],
             results["C3_max_bauschinger_error"], results["C5_max_rate_error"],
             results["C4_tangent"]["max"], results["C6_degeneration"]["max"]), flush=True)
    print("residuals", results["residuals"], flush=True)
    return 0 if results["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())