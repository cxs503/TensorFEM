"""Qualification artifact: ice-induced slamming and hull-girder whipping.

Produces ``result.json`` (every number computed here, nothing transcribed) and
the figures of ``docs/ice-slamming-whipping.md``.  Run with

    PYTHONPATH=src python benchmarks/ice_slamming_whipping/make_result.py

Criteria are the machine-checkable ones of campaign node
``p9050-n04-ice-slamming-whipping``:

(a) peak slam force against model-test literature, >= 3 (V, h) combinations;
(b) whipping frequencies f2-f4 against hull-girder beam theory, error < 5 %;
(c) whipping decay time against the 5-15 s literature window, error < 30 %;
(d) three hard negative controls (V -> 0, h -> 0, C -> 0);
(e) falsifiability: F_slam_max < 1 MN at V = 10 kn, h = 0.5 m is a FAIL.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import tensorfem  # noqa: E402
import tensorfem.ice_slamming_whipping as m  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
DOCS = REPO / "docs"
FIGS = DOCS / "assets" / "ice-slamming-whipping"

# ----------------------------------------------------------------------------- 
# declared case (CASE_INPUT)
# -----------------------------------------------------------------------------
SHIP = m.IceClassShipCase(
    length=230.0,
    breadth=32.0,
    draft=13.5,
    block_coefficient=0.80,
    girder_depth=20.0,
    girder_wall=0.040,
    deadrise_deg=30.0,
    impact_station_x=103.5,
    impact_local_immersion=3.5,
    impact_half_breadth=6.5,
    slam_length=14.0,
)
ICE = m.IceField(thickness=0.5, concentration=0.6)
WAVE_TRANSIT = m.WaveField(significant_height=2.0, peak_period=7.0)
WAVE_SLAM = m.WaveField(significant_height=6.0, peak_period=9.5)
SPEED_KN = 10.0
SPEED = SPEED_KN * 0.514444
MODEL = m.WhippingModel(elements=80, modes=6, response_dt=0.01, duration=120.0)
SHORT = m.WhippingModel(elements=80, modes=6, response_dt=0.01, duration=60.0)
CURVE = m.SNCurve()

# literature band quoted by the campaign node for (V = 10 kn, h = 0.5 m)
LITERATURE_FORCE_BAND = (2.0e6, 5.0e6)
LITERATURE_FORCE_FLOOR = 1.0e6
LITERATURE_DECAY_BAND = (5.0, 15.0)


def _mn(value):
    return round(float(value) / 1.0e6, 4)


def main() -> dict:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = {
        "task_key": "p9050-n04-ice-slamming-whipping",
        "repository": "TensorFEM",
        "module": "tensorfem.ice_slamming_whipping",
        "tensorfem_version": tensorfem.__version__,
        "generated_by": {
            "script": "benchmarks/ice_slamming_whipping/make_result.py",
            "command": "PYTHONPATH=src python benchmarks/ice_slamming_whipping/make_result.py",
        },
    }

    geometry = SHIP.geometry_checks()
    frequency = m.frequency_check(SHIP, MODEL)
    static_check = m.static_moment_check(SHIP, MODEL)

    main_case = m.run_ice_slamming_case(SHIP, ICE, WAVE_TRANSIT, SPEED, MODEL, CURVE)

    # --- (b) frequencies f2..f4 against the exact beam-theory solution --------
    fe = frequency["frequencies_wet_fe_hz"][:3]
    theory = frequency["frequencies_exact_dsm_hz"][:3]
    eb = frequency["euler_bernoulli_hz"][:3]
    f_residual = [
        abs(a - b) / b for a, b in zip(fe, theory)
    ]
    criterion_b = {
        "requirement": "f_2..f_4 (FE wet girder) vs hull-girder beam theory < 5 %",
        "reference": "exact free-free Timoshenko dynamic-stiffness solution "
        "(tensorfem.dsbeam.natural_frequencies_hz) of the same girder",
        "measured_fe_hz": [round(v, 5) for v in fe],
        "measured_theory_hz": [round(v, 5) for v in theory],
        "euler_bernoulli_hz": [round(v, 5) for v in eb],
        "relative_error": [round(v, 6) for v in f_residual],
        "max_relative_error": round(max(f_residual), 6),
        "limit": 0.05,
        "passed": bool(max(f_residual) < 0.05),
        "shear_softening_fraction_vs_euler_bernoulli": [
            round(1.0 - t / e, 5) for t, e in zip(theory, eb)
        ],
        "static_moment_recovery": {
            "measured_nm": static_check["measured_moment"],
            "analytic_nm": static_check["expected_moment"],
            "relative_error": static_check["relative_error"],
        },
    }

    # --- (c) whipping decay time --------------------------------------------
    decay_low, decay_high = LITERATURE_DECAY_BAND
    tau = main_case.tau_decay
    tau_analytic = main_case.tau_decay_analytic
    criterion_c = {
        "requirement": "whipping decay time within the 5-15 s literature window",
        "measured_tau_decay_s": round(tau, 4),
        "analytic_tau_s": round(tau_analytic, 4),
        "measured_vs_analytic_relative_error": round(
            abs(tau - tau_analytic) / tau_analytic, 6
        ),
        "literature_window_s": [decay_low, decay_high],
        "inside_window": bool(decay_low <= tau <= decay_high),
        "declared_damping_ratio_mode_2": MODEL.damping_ratio_2,
        "damping_ratio_band_implying_the_window": [
            round(1.0 / (decay_high * 2.0 * math.pi * fe[0]), 5),
            round(1.0 / (decay_low * 2.0 * math.pi * fe[0]), 5),
        ],
        "inter_arrival_s": round(main_case.inter_arrival_s, 4),
        "note": "the decay time is set by the declared structural damping; the "
        "5-15 s window and the declared zeta = 2.5 % are consistent but the check "
        "is a consistency check, not an independent validation",
        "passed": bool(decay_low <= tau <= decay_high),
    }

    # --- (e) falsifiability -------------------------------------------------
    criterion_e = {
        "requirement": "F_slam_max at V=10 kn, h=0.5 m must not fall below 1 MN",
        "measured_peak_force_mn": _mn(main_case.peak_total_force),
        "measured_peak_ice_force_mn": _mn(main_case.peak_ice_force),
        "floor_mn": _mn(LITERATURE_FORCE_FLOOR),
        "literature_band_mn": [_mn(LITERATURE_FORCE_BAND[0]), _mn(LITERATURE_FORCE_BAND[1])],
        "inside_literature_band": bool(
            LITERATURE_FORCE_BAND[0] <= main_case.peak_total_force <= LITERATURE_FORCE_BAND[1]
        ),
        "passed": bool(main_case.peak_total_force >= LITERATURE_FORCE_FLOOR),
    }

    # --- (a) literature anchor + cross-formulation band ---------------------
    combos = [
        (SPEED, 0.5),
        (SPEED, 0.3),
        (SPEED * 0.7, 0.5),
        (SPEED, 0.8),
    ]
    grid = []
    for speed, thickness in combos:
        ice = m.IceField(thickness=thickness, concentration=ICE.concentration, seed=ICE.seed)
        case = m.run_ice_slamming_case(SHIP, ice, WAVE_TRANSIT, speed, SHORT, CURVE)
        grid.append(
            {
                "speed_kn": round(speed / 0.514444, 3),
                "thickness_m": thickness,
                "peak_force_mn": _mn(case.peak_ice_force),
                "peak_whipping_moment_mnm": _mn(case.peak_total_moment),
                "damage_slam_per_60s": case.damage_slam,
                "damage_wave_per_60s": case.damage_wave,
                "damage_ratio": case.damage_ratio,
                "pulse_count": case.pulse_counts["ice"],
                "contact_area_m2": thickness * min(2.0 * SHIP.impact_half_breadth, ICE.floe_length),
            }
        )
    # pressure sensitivity: the only free constitutive knob of the ice branch
    sensitivity = []
    for p_ref in (0.40e6, 0.66e6, 1.00e6):
        ice = m.IceField(
            thickness=ICE.thickness,
            concentration=ICE.concentration,
            crush_pressure_reference=p_ref,
            seed=ICE.seed,
        )
        case = m.run_ice_slamming_case(SHIP, ice, WAVE_TRANSIT, SPEED, SHORT, CURVE)
        sensitivity.append(
            {"crush_pressure_reference_mpa": p_ref / 1e6, "peak_force_mn": _mn(case.peak_ice_force)}
        )
    # structural scaling check of the constitutive law: dlnF/dlnh and dlnF/dlnV
    forces = {(row["speed_kn"], row["thickness_m"]): row["peak_force_mn"] for row in grid}
    exponent_h = math.log(forces[(SPEED_KN, 0.8)] / forces[(SPEED_KN, 0.5)]) / math.log(0.8 / 0.5)
    exponent_v = math.log(forces[(SPEED_KN, 0.5)] / forces[(7.0, 0.5)]) / math.log(SPEED_KN / 7.0)

    criterion_a = {
        "requirement": "F_slam_max vs ice model-test literature, >= 3 (V,h) "
        "combinations, error < 20 %",
        "verdict": "WITHHELD",
        "reason": "no authorised copy of an ice model-test force table is "
        "reachable from this environment; no number was written from memory",
        "acquisition_attempts": [
            "repository corpus grep for ice slamming / model test force tables: no table",
            "local PDF corpus: no ice model-test report",
            "network: not attempted (unauthenticated fetch of a commercial "
            "publication would not be an authorised copy)",
        ],
        "substitute_evidence": {
            "pressure_sensitivity_band_mn": [row["peak_force_mn"] for row in sensitivity],
            "literature_band_mn": [_mn(LITERATURE_FORCE_BAND[0]), _mn(LITERATURE_FORCE_BAND[1])],
            "band_overlap": bool(
                min(row["peak_force_mn"] for row in sensitivity) <= 5.0
                and max(row["peak_force_mn"] for row in sensitivity) >= 2.0
            ),
            "note": "range overlap with the quoted literature band is a "
            "consistency argument, not a validation",
        },
        "constitutive_exponent_check": {
            "dlnF_dlnh_measured": round(exponent_h, 4),
            "dlnF_dlnh_expected": round(1.0 + ICE.area_exponent, 4),
            "dlnF_dlnV_measured": round(exponent_v, 4),
            "dlnF_dlnV_expected": round(ICE.rate_exponent, 4),
            "relative_error_h": round(
                abs(exponent_h - (1.0 + ICE.area_exponent)) / (1.0 + ICE.area_exponent), 4
            ),
            "relative_error_v": round(abs(exponent_v - ICE.rate_exponent) / ICE.rate_exponent, 4),
            "passed": bool(
                abs(exponent_h - (1.0 + ICE.area_exponent)) / (1.0 + ICE.area_exponent) < 0.02
                and abs(exponent_v - ICE.rate_exponent) / ICE.rate_exponent < 0.02
            ),
        },
        "passed": False,
    }

    # --- (d) hard negative controls -----------------------------------------
    # the wave-slam branch peak for the h -> 0 control is also taken straight
    # from the pulse list, so the control does not depend on the case pipeline
    wave_branch_direct = max(
        (pulse.amplitude for pulse in m.wave_slam_pulses(SHIP, WAVE_SLAM, SPEED, 90.0)),
        default=0.0,
    )
    controls = m.negative_control_checks(
        SHIP, ICE, WAVE_SLAM, SPEED, m.WhippingModel(elements=80, modes=6, duration=90.0)
    )
    criterion_d = {
        "requirement": "V->0: load and decay vanish; h->0: load -> pure wave slam; "
        "C->0: damage ratio -> 0",
        "measured": controls,
        "v_to_zero_relative_residual": round(controls["v_to_zero"]["relative_residual"], 9),
        "v_to_zero_tau_ice_only": controls["v_to_zero"]["tau_decay_ice_only_limit"],
        "v_to_zero_tau_total": round(controls["v_to_zero"]["tau_decay_limit"], 4),
        "h_to_zero_relative_residual": round(controls["h_to_zero"]["relative_residual"], 9),
        "h_to_zero_wave_branch_pulse_list_mn": _mn(wave_branch_direct),
        "c_to_zero_damage_ratio": controls["c_to_zero"]["damage_ratio_limit"],
        "limit_relative_residual": 1e-6,
        "passed": bool(
            controls["v_to_zero"]["relative_residual"] < 1e-6
            and controls["v_to_zero"]["tau_decay_ice_only_limit"] == 0.0
            and controls["h_to_zero"]["relative_residual"] < 1e-9
            and wave_branch_direct > 0.0
            and controls["c_to_zero"]["damage_ratio_limit"] == 0.0
        ),
    }

    # --- wave-slam branch, von Karman closure and coupling -------------------
    wave_motion = m.seakeeping_relative_motion(SHIP, WAVE_SLAM, SPEED)
    re_entry = wave_motion.re_entry_velocity()
    wave_case = m.run_ice_slamming_case(SHIP, ICE, WAVE_SLAM, SPEED, SHORT, CURVE)
    slam_pulse = m.wave_slam_pulses(SHIP, WAVE_SLAM, SPEED, 60.0)
    sample_pulse = slam_pulse[0] if slam_pulse else None
    closure = {
        "re_entry_velocity_m_s": round(re_entry, 5),
        "von_karman_peak_mn": _mn(m.von_karman_peak_force(SHIP, re_entry)),
        "wagner_peak_mn": _mn(m.von_karman_peak_force(SHIP, re_entry, wagner=True)),
        "rise_impulse_ns": round(sample_pulse.rise_impulse, 3) if sample_pulse else 0.0,
        "added_mass_impulse_ns": round(m.added_mass_impulse(SHIP, re_entry), 3),
        "relative_residual": round(
            abs(sample_pulse.rise_impulse - m.added_mass_impulse(SHIP, re_entry))
            / m.added_mass_impulse(SHIP, re_entry),
            9,
        )
        if sample_pulse
        else None,
        "equivalent_ochi_cs": round(wave_case.equivalent_ochi_cs, 4),
        "wagner_to_von_karman_ratio": round((0.5 * math.pi) ** 2, 4),
    }
    coupling = {
        "sea_state": {"hs_m": WAVE_SLAM.significant_height, "tp_s": WAVE_SLAM.peak_period},
        "peak_ice_force_mn": _mn(wave_case.peak_ice_force),
        "peak_wave_slam_force_mn": _mn(wave_case.peak_wave_force),
        "peak_total_force_mn": _mn(wave_case.peak_total_force),
        "peak_total_moment_mnm": _mn(wave_case.peak_total_moment),
        "peak_wave_vbm_mnm": _mn(wave_case.peak_wave_vbm),
        "damage_wave": wave_case.damage_wave,
        "damage_slam": wave_case.damage_slam,
        "damage_ice_only": wave_case.damage_ice_only,
        "damage_wave_slam_only": wave_case.damage_wave_slam_only,
        "damage_combined": wave_case.damage_combined,
        "damage_ratio_slam_over_wave": wave_case.damage_ratio,
        "damage_ratio_combined_over_wave": wave_case.combined_ratio,
        "interference": {k: round(v, 5) for k, v in wave_case.interference.items()},
        "modal_peak_share": [round(v, 5) for v in wave_case.modal_peak_share],
    }

    out["case_caliber"] = {
        "ship": {
            "length_m": SHIP.length,
            "breadth_m": SHIP.breadth,
            "draft_m": SHIP.draft,
            "block_coefficient": SHIP.block_coefficient,
            "girder_depth_m": SHIP.girder_depth,
            "girder_wall_m": SHIP.girder_wall,
            "deadrise_deg": SHIP.deadrise_deg,
            "impact_station_x_m": SHIP.impact_station_x,
            "impact_local_immersion_m": SHIP.impact_local_immersion,
            "impact_half_breadth_m": SHIP.impact_half_breadth,
            "slam_length_m": SHIP.slam_length,
            "displacement_t": round(SHIP.displacement_mass / 1000.0, 1),
            "section_inertia_m4": round(SHIP.girder().inertia, 4),
            "section_modulus_m3": round(SHIP.section_modulus, 4),
            "geometry_checks": geometry,
        },
        "ice": {
            "thickness_m": ICE.thickness,
            "concentration": ICE.concentration,
            "floe_length_m": ICE.floe_length,
            "crush_pressure_reference_mpa": ICE.crush_pressure_reference / 1e6,
            "area_exponent": ICE.area_exponent,
            "rate_exponent": ICE.rate_exponent,
            "ice_density_kg_m3": ICE.ice_density,
        },
        "wave_transit": {
            "significant_height_m": WAVE_TRANSIT.significant_height,
            "peak_period_s": WAVE_TRANSIT.peak_period,
        },
        "wave_slam": {
            "significant_height_m": WAVE_SLAM.significant_height,
            "peak_period_s": WAVE_SLAM.peak_period,
        },
        "speed_kn": SPEED_KN,
        "speed_m_s": round(SPEED, 5),
        "whipping_model": {
            "elements": MODEL.elements,
            "modes": MODEL.modes,
            "response_dt_s": MODEL.response_dt,
            "duration_s": MODEL.duration,
            "damping_ratio_mode_2": MODEL.damping_ratio_2,
            "damping_ratio_mode_3": MODEL.damping_ratio_3,
            "hot_spot_scf": MODEL.hot_spot_scf,
            "added_mass_per_length_kg_m": round(m.added_mass_per_length(SHIP), 2),
            "vibrating_mass_per_length_kg_m": round(m.vibrating_mass_per_length(SHIP), 2),
        },
        "sn_curve": {
            "c1": CURVE.c1,
            "m1": CURVE.m1,
            "c2": CURVE.c2,
            "m2": CURVE.m2,
            "knee_cycles": CURVE.knee_cycles,
            "knee_stress_mpa": round(CURVE.knee_stress / 1e6, 3),
        },
        "grades": {
            "geometry_and_hydrostatics": "CASE_INPUT (declared)",
            "ice_crushing_pressure": "DECLARED_CONSTITUTIVE (Sanderson-type, "
            "sensitivity swept 0.40-1.00 MPa)",
            "structural_damping": "DECLARED (1.5-2.5 %, hull-girder range)",
            "added_mass_factor": "DECLARED (0.70 of the Lewis 2D value)",
            "sn_curve": "DECLARED (DNV-RP-C203 curve D in air, published pair)",
            "beam_theory_reference": "COMPUTED (exact DSM solution)",
        },
    }
    out["main_case"] = {
        "peak_ice_force_mn": _mn(main_case.peak_ice_force),
        "peak_wave_slam_force_mn": _mn(main_case.peak_wave_force),
        "peak_total_force_mn": _mn(main_case.peak_total_force),
        "peak_whipping_moment_mnm": _mn(main_case.peak_total_moment),
        "dynamic_amplification_factor": round(main_case.dynamic_amplification, 4),
        "peak_wave_vbm_mnm": _mn(main_case.peak_wave_vbm),
        "modal_frequencies_hz": [round(v, 5) for v in main_case.modal_frequencies_hz],
        "modal_peak_share": [round(v, 5) for v in main_case.modal_peak_share],
        "tau_decay_s": round(main_case.tau_decay, 4),
        "tau_decay_analytic_s": round(main_case.tau_decay_analytic, 4),
        "damage_wave": main_case.damage_wave,
        "damage_slam": main_case.damage_slam,
        "damage_ratio": round(main_case.damage_ratio, 5),
        "damage_combined": main_case.damage_combined,
        "combined_ratio": round(main_case.combined_ratio, 5),
        "pulse_counts": main_case.pulse_counts,
        "bow_emergence": main_case.motion.encounters_emergence,
        "re_entry_velocity_m_s": round(main_case.re_entry_velocity, 6),
        "inter_arrival_s": round(main_case.inter_arrival_s, 4),
        "interference": {k: round(v, 5) for k, v in main_case.interference.items()},
    }
    out["criteria"] = [criterion_a, criterion_b, criterion_c, criterion_d, criterion_e]
    out["vh_grid"] = grid
    out["pressure_sensitivity"] = sensitivity
    out["wave_slam_and_coupling"] = {**closure, **coupling}
    out["verdict"] = {
        "a_literature_force": "WITHHELD (honest FAIL: source not obtained)",
        "b_frequencies_lt_5pct": criterion_b["passed"],
        "c_decay_in_5_15s": criterion_c["passed"],
        "d_negative_controls": criterion_d["passed"],
        "e_force_above_1mn": criterion_e["passed"],
        "overall": "4 of 5 machine criteria pass; criterion (a) is WITHHELD "
        "because no authorised ice model-test table is reachable",
    }
    out["limits"] = [
        "the hull girder is a uniform equivalent box girder: a real hull is "
        "non-uniform, so the n^2 spacing of the free-free modes (f3/f2 = 2.42 "
        "here) differs from the 1.8-2.2 of a real ship",
        "the ice crushing pressure is a declared constitutive law with a "
        "1.5x band; it is the dominant uncertainty of the force level",
        "the added mass factor 0.70 of the Lewis 2D heave value is declared, "
        "not measured; it scales the frequencies by 1/sqrt(1+kappa)",
        "the seakeeping model is a 2-DOF linear heave/pitch with declared "
        "damping; the wave-slam branch only fires when the relative bow motion "
        "exceeds the local immersion (not in the 2 m transit sea state)",
        "the wave bending moment baseline is a self-equilibrated linear strip "
        "load without Smith correction or diffraction",
        "the impact train is deterministic (seeded lognormal floe lengths); no "
        "statistical convergence study of the damage was performed",
    ]
    (HERE / "result.json").write_text(json.dumps(out, indent=2) + "\n")

    # ------------------------------------------------------------------ figures
    response = main_case.response
    mid = response.midship_index()
    t = response.t
    win = t <= 40.0

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)
    history = m.build_load_history(SHIP, ICE, WAVE_TRANSIT, SPEED, MODEL)
    axes[0].plot(t[win], history["force_ice"][win] / 1e6, lw=0.8, color="C3")
    axes[0].set_ylabel("ice impact force [MN]")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t[win], response.moment_total[mid][win] / 1e6, lw=0.8, color="C0")
    axes[1].set_ylabel("$M_v$ midships [MN m]")
    axes[1].grid(alpha=0.3)
    wave_vbm = m.wave_bending_moment_history(SHIP, WAVE_TRANSIT, SPEED, t)
    axes[2].plot(t[win], wave_vbm[win] / 1e6, lw=1.2, color="C2", label="wave VBM (baseline)")
    axes[2].plot(
        t[win],
        (wave_vbm + response.moment_total[mid])[win] / 1e6,
        lw=0.6,
        color="C4",
        label="wave VBM + whipping",
    )
    axes[2].set_ylabel("$M_v$ [MN m]")
    axes[2].set_xlabel("t [s]")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.suptitle(
        "Ice-impact excitation and the whipping response it produces "
        f"(V = {SPEED_KN:.1f} kn, h = {ICE.thickness} m, C = {ICE.concentration})"
    )
    fig.tight_layout()
    fig.savefig(FIGS / "load-and-whipping.png", dpi=140)
    plt.close(fig)

    # frequency comparison
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    idx = np.arange(3)
    width = 0.26
    ax.bar(idx - width, fe, width, label="FE wet girder (free-free)", color="C0")
    ax.bar(idx, theory, width, label="exact DSM Timoshenko", color="C1")
    ax.bar(idx + width, eb, width, label="Euler-Bernoulli (no shear)", color="C2")
    ax.set_xticks(idx)
    ax.set_xticklabels(["2-node (f$_2$)", "3-node (f$_3$)", "4-node (f$_4$)"])
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(
        "Whipping frequencies: FE vs exact beam theory, "
        f"max error {criterion_b['max_relative_error'] * 100:.2f} %"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "frequencies.png", dpi=140)
    plt.close(fig)

    # decay of an isolated whipping event
    ice_pulses = [p for p in history["pulses"] if p.kind == "ice"]
    strongest = max(ice_pulses, key=lambda p: p.amplitude)
    event = m.isolated_whipping_event(
        SHIP, MODEL, strongest.amplitude, strongest.rise, duration=60.0
    )
    env_t = event.t
    env_q = np.abs(event.modal_q[0])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(env_t, event.modal_q[0] * 1e3, lw=0.8, color="C0", label="modal coordinate $q_2$")
    ax.plot(env_t, env_q * 1e3, lw=0.6, color="C3", alpha=0.7, label="envelope $|q_2|$")
    ref = env_q.max() * np.exp(-(env_t - env_t[np.argmax(env_q)]) / tau)
    ax.plot(env_t, ref * 1e3, "k--", lw=1.0, label=f"exp decay, $\\tau$ = {tau:.1f} s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$q_2$ [mm]")
    ax.set_title("Whipping transient of an isolated impact and its decay time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "decay.png", dpi=140)
    plt.close(fig)

    # force and damage versus speed / thickness
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    h_values = [0.3, 0.5, 0.8]
    for speed_kn in (7.0, 10.0, 13.0):
        forces = []
        for h in h_values:
            ice = m.IceField(thickness=h, concentration=ICE.concentration, seed=ICE.seed)
            pulses = m.ice_impact_pulses(SHIP, ice, speed_kn * 0.514444, 60.0)
            forces.append(max(p.amplitude for p in pulses) / 1e6 if pulses else 0.0)
        axes[0].plot(h_values, forces, "o-", label=f"V = {speed_kn:.0f} kn")
    axes[0].axhspan(*[v / 1e6 for v in LITERATURE_FORCE_BAND], color="0.85", label="quoted band")
    axes[0].set_xlabel("ice thickness h [m]")
    axes[0].set_ylabel("peak ice impact force [MN]")
    axes[0].set_title("Peak force: model vs the quoted literature band")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    speeds = [4.0, 6.0, 8.0, 10.0, 12.0]
    ratios = []
    for speed_kn in speeds:
        case = m.run_ice_slamming_case(SHIP, ICE, WAVE_TRANSIT, speed_kn * 0.514444, SHORT, CURVE)
        ratios.append(case.damage_ratio)
    axes[1].plot(speeds, ratios, "s-", color="C4")
    axes[1].set_xlabel("speed [kn]")
    axes[1].set_ylabel("$\\Delta D_{slam} / D_{wave}$")
    axes[1].set_title("Slam fatigue increment relative to wave fatigue")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "force-and-damage-trends.png", dpi=140)
    plt.close(fig)
    out["speed_sweep_damage_ratio"] = [
        {"speed_kn": s, "damage_ratio": round(r, 5)} for s, r in zip(speeds, ratios)
    ]
    (HERE / "result.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["verdict"], indent=2))
    print(json.dumps(result["main_case"], indent=2)[:1200])