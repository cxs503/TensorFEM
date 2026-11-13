"""Assessment harness: the four Ai-2011 rolling-resistance families vs analytic oracles.

Anchor ``XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS``.

Scenarios (every reference number below is analytic, derived in
``docs/XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS.md``):

S1  flat stop, pure families      ``v0 = 1 m/s``, no dashpot: stopping distance
                                  vs ``(1+k) v0^2/(2 mu_r g)``.
S1b flat stop, damped             same but with the EPSD2 dashpot at ``zeta = 1``:
                                  demonstrates that the dashpot, not ``mu_r``,
                                  controls the transient when ``omega >> omega_ref``.
S2  static hold                   horizontal plane, constant ``0.5 M_cap`` torque:
                                  does the model hold a static contact at all?
S2b jitter resolution             same, "cdt" at two step sizes: the artefact is a
                                  limit cycle whose amplitude scales with ``dt``.
S3  supercritical, pure families  ``theta = 20 deg`` (``tan > mu_r``), no dashpot:
                                  saturated acceleration and mean torque vs
                                  ``g(sin - mu_r cos)/(1+k)`` and ``-M_cap``.
S3b supercritical, damped         with the dashpot: terminal velocity vs
                                  ``(m g sin R - M_cap) R / c_r`` and ``tau_s``.
S4  subcritical                   ``theta = 5 deg`` (``tan < mu_r``): elastic wind-up
                                  ``phi_eq = phi_y tan/mu_r`` then rest.
S5  critical angle                bisection on the simulated drift for the damped
                                  families -> ``theta_c`` vs ``atan(mu_r)``.
S6  family reduction              ``h = 0`` must reproduce the epsd family exactly.
S7  hysteresis loop               sub-yield driven oscillation: dissipated energy
                                  vs ``h`` (0 -> conservative, growing with ``h``).

Outputs (``--out``, default ``artifacts``): ``rolling_model_assessment.json`` (all
numbers, zero hand-copying) and ``rolling_model_assessment.png`` (three panels).

Run::

    PYTHONPATH=src python examples/rolling_model_assessment.py --out artifacts
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorfem.rolling_resistance_models import (
    INERTIA_FACTOR,
    MODELS,
    RollingModel,
    cdt_stopping_distance,
    critical_incline_angle_deg,
    damped_terminal_velocity,
    damped_time_constant,
    saturated_acceleration,
    simulate,
    static_windup_angle,
    viscous_terminal_velocity,
    viscous_time_constant,
)

MU_R = 0.3
RADIUS = 0.01          # m
MASS = 1.0             # kg
THETA_SUPER = 20.0     # deg
THETA_SUB = 5.0        # deg


def _model(family: str, **kwargs) -> RollingModel:
    kwargs.setdefault("inertia_factor", INERTIA_FACTOR["sphere"])
    return RollingModel(model=family, mu_r=MU_R, radius=RADIUS, mass=MASS, **kwargs)


def rel_error(value: float, reference: float) -> float:
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / abs(reference)


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
def s1_flat_stop(dt: float = 1.0e-5, duration: float = 0.5, damping_ratio: float = 0.0) -> dict:
    out = {}
    cdt_ref = None
    for family in MODELS:
        model = _model(family, damping_ratio=damping_ratio)
        res = simulate(model, incline_deg=0.0, v0=1.0, duration=duration, dt=dt)
        x_end = float(res["x"][-1])
        v_end = float(res["v"][-1])
        out[family] = {
            "stopping_distance_m": x_end,
            "final_speed_m_s": v_end,
            "contact_work_J": float(res["work"][-1]),
        }
        if family == "cdt":
            ref = cdt_stopping_distance(model, 1.0)
            cdt_ref = ref
            out[family]["stopping_distance_reference_m"] = ref
            out[family]["stopping_distance_rel_error"] = rel_error(x_end, ref)
            ke = 0.5 * (1.0 + model.inertia_factor) * MASS * 1.0**2
            out[family]["kinetic_energy_J"] = ke
            out[family]["work_rel_error"] = rel_error(-float(res["work"][-1]), ke)
        if family == "viscous":
            tau_v = viscous_time_constant(model, 0.0)
            v_inf = viscous_terminal_velocity(model, 0.0)
            ref = v_inf + (1.0 - v_inf) * math.exp(-duration / tau_v)
            out[family]["final_speed_reference_m_s"] = ref
            out[family]["final_speed_rel_error"] = rel_error(v_end, ref)
    for family in ("hysteretic", "epsd"):
        out[family]["stopping_distance_vs_cdt_rel_error"] = rel_error(
            out[family]["stopping_distance_m"], cdt_ref
        )
    return out


def s2_static_hold(dt: float = 1.0e-5, duration: float = 1.0, damping_ratio: float = 1.0) -> dict:
    out = {}
    for family in MODELS:
        model = _model(family, damping_ratio=damping_ratio)
        cap = model.capacity(model.normal_force(0.0))
        applied = 0.5 * cap
        res = simulate(
            model, incline_deg=0.0, v0=0.0, duration=duration, dt=dt,
            external_torque=applied,
        )
        out[family] = {
            "applied_torque_Nm": applied,
            "displacement_m": float(res["x"][-1]),
            "final_speed_m_s": float(res["v"][-1]),
            "max_speed_m_s": float(res["v"].abs().max()),
            "elastic_strain_rad": float(res["phi"][-1]),
        }
        if family in ("hysteretic", "epsd"):
            # phi = applied/k_r = 0.5 phi_y -> x = R phi
            ref = RADIUS * 0.5 * model.yield_angle
            out[family]["displacement_reference_m"] = ref
            out[family]["displacement_rel_error"] = rel_error(float(res["x"][-1]), ref)
        if family == "viscous":
            # tau = C_v omega = applied  ->  omega = applied/C_v  ->  v = R omega
            v_ref = RADIUS * model.omega_ref * applied / cap
            out[family]["creep_speed_reference_m_s"] = v_ref
    return out


def s2b_jitter_resolution(duration: float = 0.2) -> dict:
    out = {}
    for dt in (1.0e-5, 1.0e-6):
        model = _model("cdt", damping_ratio=0.0)
        cap = model.capacity(model.normal_force(0.0))
        res = simulate(
            model, incline_deg=0.0, v0=0.0, duration=duration, dt=dt,
            external_torque=0.5 * cap,
        )
        tail = res["v"][len(res["v"]) // 2:]
        out[f"dt={dt:g}"] = {
            "step_s": dt,
            "rms_speed_m_s": float(tail.pow(2).mean().sqrt()),
            "displacement_m": float(res["x"][-1]),
            "analytic_spurious_acceleration_m_s2": model.gravity * MU_R
            / (1.0 + model.inertia_factor),
        }
    keys = sorted(out)
    out["rms_ratio_dt10"] = (
        out[keys[0]]["rms_speed_m_s"] / out[keys[1]]["rms_speed_m_s"]
        if out[keys[1]]["rms_speed_m_s"] > 0.0 else None
    )
    return out


def s3_supercritical(dt: float = 1.0e-5, duration: float = 0.1, damping_ratio: float = 0.0) -> dict:
    out = {}
    window = 2000  # steps averaged for the tail measurement
    for family in MODELS:
        model = _model(family, damping_ratio=damping_ratio)
        res = simulate(model, incline_deg=THETA_SUPER, v0=0.0, duration=duration, dt=dt)
        v = res["v"]
        measured = (float(v[-1]) - float(v[-1 - window])) / (window * dt)
        out[family] = {
            "tail_acceleration_m_s2": measured,
            "tail_window_s": window * dt,
            "mean_tail_torque_Nm": float(res["tau"][-window:].mean()),
        }
        if family in ("cdt", "hysteretic", "epsd"):
            ref = saturated_acceleration(model, THETA_SUPER)
            out[family]["acceleration_reference_m_s2"] = ref
            out[family]["acceleration_rel_error"] = rel_error(measured, ref)
            cap_ref = -model.capacity(model.normal_force(THETA_SUPER))
            out[family]["mean_tail_torque_reference_Nm"] = cap_ref
            out[family]["mean_tail_torque_rel_error"] = rel_error(
                out[family]["mean_tail_torque_Nm"], cap_ref
            )
        if family == "viscous":
            ref = viscous_terminal_velocity(model, THETA_SUPER)
            out[family]["terminal_velocity_reference_m_s"] = ref
            out[family]["tail_acceleration_analytic_m_s2"] = -model.gravity * math.sin(
                math.radians(THETA_SUPER)
            ) * 0.0  # documented below: viscous acceleration -> 0 at v_inf
    return out


def s3b_supercritical_damped(dt: float = 1.0e-5, duration: float = 0.1) -> dict:
    out = {}
    for family in ("hysteretic", "epsd"):
        model = _model(family, damping_ratio=1.0)
        res = simulate(model, incline_deg=THETA_SUPER, v0=0.0, duration=duration, dt=dt)
        v_end = float(res["v"][-1])
        v_ref = damped_terminal_velocity(model, THETA_SUPER)
        tau_s = damped_time_constant(model, THETA_SUPER)
        out[family] = {
            "final_speed_m_s": v_end,
            "terminal_velocity_reference_m_s": v_ref,
            "terminal_velocity_rel_error": rel_error(v_end, v_ref),
            "time_constant_s": tau_s,
            "damping_torque_Nm": model.damping(model.normal_force(THETA_SUPER)),
        }
    return out


def s4_subcritical(dt: float = 2.0e-5, duration: float = 2.0, damping_ratio: float = 1.0) -> dict:
    out = {}
    for family in MODELS:
        model = _model(family, damping_ratio=damping_ratio)
        res = simulate(model, incline_deg=THETA_SUB, v0=0.0, duration=duration, dt=dt)
        out[family] = {
            "displacement_m": float(res["x"][-1]),
            "final_speed_m_s": float(res["v"][-1]),
            "max_speed_m_s": float(res["v"].abs().max()),
            "elastic_strain_rad": float(res["phi"][-1]),
        }
        if family in ("hysteretic", "epsd"):
            ref = RADIUS * static_windup_angle(model, THETA_SUB)
            out[family]["displacement_reference_m"] = ref
            out[family]["displacement_rel_error"] = rel_error(float(res["x"][-1]), ref)
        if family == "viscous":
            tau_v = viscous_time_constant(model, THETA_SUB)
            v_inf = viscous_terminal_velocity(model, THETA_SUB)
            ref = v_inf * (1.0 - math.exp(-duration / tau_v))
            out[family]["final_speed_reference_m_s"] = ref
            out[family]["final_speed_rel_error"] = rel_error(float(res["v"][-1]), ref)
    return out


def s5_critical_angle(duration: float = 0.5, dt: float = 2.0e-5, speed_tol: float = 1.0e-6) -> dict:
    """Bisect the incline angle at which the damped families start to drift."""

    def drifts(model: RollingModel, angle: float) -> bool:
        res = simulate(model, incline_deg=angle, v0=0.0, duration=duration, dt=dt)
        return abs(float(res["v"][-1])) > speed_tol

    out = {}
    for family in ("hysteretic", "epsd"):
        model = _model(family, damping_ratio=1.0)
        lo, hi = 0.0, 60.0
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if drifts(model, mid):
                hi = mid
            else:
                lo = mid
        theta_c = 0.5 * (lo + hi)
        ref = critical_incline_angle_deg(MU_R)
        out[family] = {
            "simulated_critical_angle_deg": theta_c,
            "drift_threshold_m_s": speed_tol,
            "reference_deg": ref,
            "absolute_error_deg": abs(theta_c - ref),
            "rel_error": rel_error(theta_c, ref),
        }
    out["viscous"] = {
        "simulated_critical_angle_deg": 0.0,
        "reference_deg": 0.0,
        "terminal_velocity_at_1deg_m_s": viscous_terminal_velocity(
            _model("viscous"), 1.0
        ),
        "note": "no static threshold: v_inf > 0 for every theta > 0",
    }
    return out


def s6_family_reduction(dt: float = 1.0e-5, duration: float = 0.05) -> dict:
    """``h = 0`` must make the hysteretic family identical to the epsd family."""
    a = simulate(
        _model("hysteretic", hysteresis_ratio=0.0, damping_ratio=1.0),
        incline_deg=THETA_SUPER, v0=0.0, duration=duration, dt=dt,
    )
    b = simulate(
        _model("epsd", damping_ratio=1.0),
        incline_deg=THETA_SUPER, v0=0.0, duration=duration, dt=dt,
    )
    return {
        "max_velocity_difference_m_s": float((a["v"] - b["v"]).abs().max()),
        "max_velocity_m_s": float(a["v"].abs().max()),
    }


def s7_hysteresis_loop(dt: float = 2.0e-6, duration: float = 0.5, cycles_hz: float = 20.0) -> dict:
    """Sub-yield driven oscillation: dissipated energy must grow with ``h``."""
    base = _model("hysteretic", damping_ratio=0.0, hysteresis_ratio=0.0)
    cap = base.capacity(base.normal_force(0.0))
    amplitude = 0.4 * cap
    torque = lambda t: amplitude * math.sin(2.0 * math.pi * cycles_hz * t)  # noqa: E731
    out = {"applied_torque_amplitude_Nm": amplitude, "frequency_hz": cycles_hz}
    previous = None
    for ratio in (0.0, 0.1, 0.25, 0.5):
        model = _model("hysteretic", damping_ratio=0.0, hysteresis_ratio=ratio)
        res = simulate(
            model, incline_deg=0.0, v0=0.0, duration=duration, dt=dt, external_torque=torque,
        )
        dissipated = -float(res["work"][-1])
        out[f"h={ratio}"] = {
            "dissipated_energy_J": dissipated,
            "peak_elastic_strain_rad": float(res["phi"].abs().max()),
            "yield_strain_rad": model.yield_angle,
            "monotone_vs_previous": None if previous is None else dissipated > previous,
        }
        previous = dissipated
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"cdt": "C3", "viscous": "C0", "hysteretic": "C2", "epsd": "C1"}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    ax = axes[0]
    for family in MODELS:
        res = simulate(
            _model(family, damping_ratio=1.0), incline_deg=THETA_SUB, v0=0.0,
            duration=0.4, dt=2.0e-5,
        )
        ax.plot(res["t"].numpy(), res["x"].numpy() * 1.0e3, label=family,
                color=colours[family], lw=1.6)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rolled distance [mm]")
    ax.set_title(f"S4 subcritical incline ({THETA_SUB:.0f} deg, tan={math.tan(math.radians(THETA_SUB)):.3f} < mu_r={MU_R})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels, values = [], []
    for family in MODELS:
        model = _model(family, damping_ratio=1.0)
        res = simulate(
            model, incline_deg=0.0, v0=0.0, duration=1.0, dt=1.0e-5,
            external_torque=0.5 * model.capacity(model.normal_force(0.0)),
        )
        labels.append(family)
        values.append(abs(float(res["x"][-1])))
    bars = ax.bar(labels, values, color=[colours[f] for f in labels])
    ax.set_yscale("log")
    ax.set_ylabel("|displacement| after 1 s [m] (log)")
    ax.set_title("S2 static hold, applied torque 0.5 M_cap")
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2e}", (bar.get_x() + bar.get_width() / 2.0, value),
                    ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    cap = _model("hysteretic").capacity(_model("hysteretic").normal_force(0.0))
    torque = lambda t: 0.4 * cap * math.sin(2.0 * math.pi * 20.0 * t)  # noqa: E731
    for ratio, colour in ((0.0, "0.4"), (0.25, "C4"), (0.5, "C5")):
        model = _model("hysteretic", damping_ratio=0.0, hysteresis_ratio=ratio)
        res = simulate(
            model, incline_deg=0.0, v0=0.0, duration=0.5, dt=2.0e-6, external_torque=torque,
        )
        theta = res["theta_rolled"].numpy()
        tau = res["tau"].numpy()
        keep = slice(len(theta) // 2, len(theta) // 2 + 5000)
        ax.plot(theta[keep] * 1.0e3, tau[keep], lw=1.2, color=colour, label=f"h={ratio}")
    ax.set_xlabel("rolling strain [mrad]")
    ax.set_ylabel("contact torque [N m]")
    ax.set_title("S7 hysteresis loops, sub-yield drive")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    parser = argparse.ArgumentParser(description="rolling-resistance model assessment")
    parser.add_argument("--out", default="artifacts", help="output directory")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    results = {
        "anchor": "XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS",
        "parameters": {
            "mu_r": MU_R,
            "radius_m": RADIUS,
            "mass_kg": MASS,
            "inertia_factor": INERTIA_FACTOR["sphere"],
            "theta_super_deg": THETA_SUPER,
            "theta_sub_deg": THETA_SUB,
            "critical_angle_deg": critical_incline_angle_deg(MU_R),
        },
        "s1_flat_stop_pure": s1_flat_stop(),
        "s1b_flat_stop_damped": s1_flat_stop(damping_ratio=1.0),
        "s2_static_hold": s2_static_hold(),
        "s2b_jitter_resolution": s2b_jitter_resolution(),
        "s3_supercritical_pure": s3_supercritical(),
        "s3b_supercritical_damped": s3b_supercritical_damped(),
        "s4_subcritical": s4_subcritical(),
        "s5_critical_angle": s5_critical_angle(),
        "s6_family_reduction": s6_family_reduction(),
        "s7_hysteresis_loop": s7_hysteresis_loop(),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rolling_model_assessment.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    if not args.no_figure:
        make_figure(out_dir / "rolling_model_assessment.png")

    for key, value in results.items():
        if key.startswith("s"):
            printable = {k: v for k, v in value.items()
                         if isinstance(v, dict) and ("rel_error" in v or "absolute_error_deg" in v)}
            print(f"{key}: " + ", ".join(
                f"{fam}=%s" % (
                    next((f"{k}={v[k]:.3g}" for k in ("rel_error", "absolute_error_deg")
                          if k in v), "n/a")
                ) for fam, v in printable.items()
            ))
    print(f"wrote {json_path}")
    return results


if __name__ == "__main__":
    main()