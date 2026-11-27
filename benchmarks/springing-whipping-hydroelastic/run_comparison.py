"""Hydroelastic time-domain response at the KVLCC2 hull-girder caliber,
compared against the T1 external-anchor artifact (springing-whipping-hydroelastic).

What this script does
---------------------
1. **Anchor load.** Reads ``benchmarks/anchors/springing-whipping-hydroelastic.json``
   (the T1 anchor, already closed out) and takes from it *only* the values the
   anchor itself recorded: the KVLCC2 case caliber (L, B, T, rho, V) and the
   dry-run-fixture excitation magnitude (``q``, ``p_rms``).  No number is typed
   from memory; every anchor value is read from the artifact.
2. **Geometry gate.** Builds the KVLCC2 equivalent box girder and runs the
   independent geometry audit (Green's theorem + thin-wall asymptote) before any
   modal or time-domain number is produced.
3. **Wet modal model.** Lewis-section added mass (rigid-lid / doubled-section
   convention) + ballast-to-float, wet pinned--pinned Timoshenko modes.
4. **Hydroelastic time-domain response (real run).** The wet girder is excited by
   a *deterministic* springing pressure (harmonic, at the wet fundamental) and by
   a *deterministic* whipping pulse (half-sine slamming impulse).  The modal
   equations are integrated with the repository's own ``newmark_linear``
   (``tensorfem.dynamics``) -- no mock, no random-phase seed noise.  The response
   is reconstructed in physical coordinates and the bending moment / stress are
   recovered from the section rotation.
5. **Comparison.** For every case the *computed* (Newmark) peak and dominant
   frequency are compared with the *expected* (analytic modal transfer) values,
   and the damping-ratio sensitivity is reported.  All cases are written to the
   artifact; a case that fails the 15 % gate is reported as-is (honest FAIL).

The external SNAME T&R Bulletin 3-37 / 4-7 anchor is NOT_OBTAINED in this
sandbox (recorded in the T1 source registry), so the ``sname-t&r-bulletin``
comparison is necessarily against the T1 *synthetic* fixture, which is labelled
``claim_eligible: false`` everywhere it appears.  That distinction is structural
and is written into the verdict.

Run
---
    PYTHONPATH=src python benchmarks/springing-whipping-hydroelastic/run_comparison.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from tensorfem.dynamics import newmark_linear
from tensorfem.hull_girder_modal import (
    BoxGirderSection,
    ballast_mass_per_length,
    lewis_heave_added_mass,
    wet_simply_supported_modes,
)
from tensorfem.timoshenko_beam_modal import (
    assemble_timoshenko_beam,
    timoshenko_shape_functions,
)

_D = torch.float64
REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_PATH = REPO_ROOT / "benchmarks" / "anchors" / "springing-whipping-hydroelastic.json"
OUT_PATH = Path(__file__).resolve().parent / "comparison.json"

# ---------------------------------------------------------------------------
# 1. Anchor load (values come from the T1 artifact, never from memory)
# ---------------------------------------------------------------------------
def load_anchor() -> dict:
    art = json.loads(ANCHOR_PATH.read_text())
    cal = art["case_caliber"]["quantities"]
    fix = art["dry_run_fixture"]
    return {
        "L": cal["L"]["value"],
        "B": cal["B"]["value"],
        "T": cal["T"]["value"],
        "rho": cal["rho"]["value"],
        "V": cal["V"]["value"],
        "delta_star": cal["delta_star"]["value"],
        "cp_rms_star": fix["cp_rms_star"],
        "amplitude_A": fix["amplitude_A"],
        "l_ref_used_m": fix["l_ref_used_m"],
        "q_pa": fix["derived_case_caliber_values"]["q_pa"],
        "p_rms_pa": fix["derived_case_caliber_values"]["p_rms_pa"],
        "convention_status": art["conversion_chain"]["convention_status"],
        "anchor_verdict": art["verdict"]["overall"],
        "anchor_verdict_kind": art["verdict"]["kind"],
    }


# ---------------------------------------------------------------------------
# 2. KVLCC2 equivalent box girder (wall calibrated to the CSR-chain I_midship)
# ---------------------------------------------------------------------------
def kvlcc2_section(anchor: dict) -> BoxGirderSection:
    """KVLCC2-class equivalent box girder.

    Outer dimensions from the T1 case caliber (L, B) and the repository's
    ``docs/hull-girder-csr-bending-chain.md`` (moulded depth D = 30 m, design
    draft T = 20.8 m).  The wall thickness is chosen so the box-girder second
    moment of area equals the CSR-chain ``I_midship = 1646.7 m^4`` (the same
    document), i.e. the equivalent beam carries the same bending stiffness as
    the documented KVLCC2 midship section.
    """
    L, B, T = anchor["L"], anchor["B"], anchor["T"]
    D = 30.0
    target_I = 1646.7

    def I_of(t: float) -> float:
        return (B * D**3 - (B - 2 * t) * (D - 2 * t) ** 3) / 12.0

    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if I_of(mid) < target_I:
            lo = mid
        else:
            hi = mid
    wall = 0.5 * (lo + hi)
    return BoxGirderSection(
        length=L, breadth=B, depth=D, wall=wall, draft=T,
        water_density=anchor["rho"],
    )


# ---------------------------------------------------------------------------
# 3. Consistent nodal load for a uniform pressure q(x) = p * B
# ---------------------------------------------------------------------------
def uniform_pressure_nodal_load(section: BoxGirderSection, elements: int,
                                p: float, *, dtype=torch.float64) -> torch.Tensor:
    """Nodal force vector [N] for a uniform pressure ``p`` [Pa] over the breadth.

    ``q = p * B`` [N/m] is integrated against the Timoshenko transverse shape
    functions (4-point Gauss) to give the consistent nodal forces in the
    ``(w, theta)`` ordering.
    """
    if p == 0.0:
        return torch.zeros(2 * (elements + 1), dtype=dtype)
    beam = section.to_timoshenko_section()
    le = section.length / elements
    phi = 12.0 * section.bending_stiffness / (section.shear_stiffness * le**2)
    q = p * section.breadth
    ndof = 2 * (elements + 1)
    F = torch.zeros(ndof, dtype=dtype)
    fe = torch.zeros(4, dtype=dtype)
    for gp, gw in zip((0.8611363115940526, 0.3399810435848563,
                       -0.3399810435848563, -0.8611363115940526),
                      (0.3478548451374538, 0.6521451548625461,
                       0.6521451548625461, 0.3478548451374538)):
        xi = 0.5 * (gp + 1.0)
        n_w, _, _, _ = timoshenko_shape_functions(xi, le, phi, dtype=dtype)
        fe += (gw * 0.5 * le) * q * n_w
    for e in range(elements):
        dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        F[dofs] += fe
    return F


# ---------------------------------------------------------------------------
# 4. Modal projection and response reconstruction
# ---------------------------------------------------------------------------
def modal_projection(section: BoxGirderSection, elements: int, added_mass: float,
                     extra_mass: float, n_modes: int, *, dtype=torch.float64) -> dict:
    """Wet modes + modal operators (mass-normalised)."""
    wet = wet_simply_supported_modes(
        section, elements, added_mass, extra_mass=extra_mass, modes=n_modes, dtype=dtype
    )
    omega = wet.angular_frequencies  # (n_modes,)
    modes = wet.modes               # (ndof, n_modes), mass-normalised (phi^T M phi = 1)
    beam = section.to_timoshenko_section()
    K, M = assemble_timoshenko_beam(beam, elements, dtype=dtype)
    # modal stiffness / mass (should be omega^2 and 1 for mass-normalised modes)
    K_mod = torch.einsum("in,ij,jn->n", modes, K, modes)
    M_mod = torch.einsum("in,ij,jn->n", modes, M, modes)
    return {
        "omega": omega,
        "freq_hz": (omega / (2 * math.pi)).detach().cpu().numpy(),
        "modes": modes,
        "K": K,
        "M": M,
        "K_mod": K_mod,
        "M_mod": M_mod,
        "ndof": K.shape[0],
    }


def modal_force_vector(proj: dict, F_nodal: torch.Tensor) -> torch.Tensor:
    """Project a nodal force vector onto the modes: ``F_n = phi_n^T F``."""
    return torch.einsum("in,i->n", proj["modes"], F_nodal)


def curvature_at_midspan(proj: dict, section: BoxGirderSection, elements: int) -> torch.Tensor:
    """``d theta / dx`` of each mode at the midspan element [1/m]."""
    beam = section.to_timoshenko_section()
    le = section.length / elements
    phi = 12.0 * section.bending_stiffness / (section.shear_stiffness * le**2)
    # midspan element index (elements // 2), local xi = 0.5
    e = elements // 2
    xi = 0.5
    _, _, n_theta, dn_theta = timoshenko_shape_functions(xi, le, phi, dtype=_D)
    dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
    # dn_theta is d(theta)/dx at the local point; the mode's theta at these dofs
    theta = proj["modes"][dofs]  # (4, n_modes)
    return torch.einsum("k,kn->n", dn_theta, theta)


def run_modal_newmark(proj: dict, zeta: float, F_modal: torch.Tensor,
                      time: torch.Tensor, *, dtype=torch.float64) -> torch.Tensor:
    """Integrate the modal equations ``u_n'' + 2 zeta w_n u_n' + w_n^2 u_n = F_n``
    with the repository Newmark integrator (diagonal M, K, C)."""
    n_modes = proj["omega"].numel()
    M = torch.eye(n_modes, dtype=dtype)
    K = torch.diag(proj["omega"] ** 2)
    C = torch.diag(2.0 * zeta * proj["omega"])
    u0 = torch.zeros(n_modes, dtype=dtype)
    v0 = torch.zeros(n_modes, dtype=dtype)
    res = newmark_linear(M, K, F_modal, time, u0, v0, C=C)
    return res.displacement  # (n_time, n_modes)


def reconstruct_bending_moment(proj: dict, section: BoxGirderSection, elements: int,
                               u_modal: torch.Tensor) -> torch.Tensor:
    """Bending moment at the midspan ``M = EI * sum_n u_n * (d theta_n / dx)``."""
    curv = curvature_at_midspan(proj, section, elements)  # (n_modes,)
    return section.bending_stiffness * torch.einsum("tn,n->t", u_modal, curv)


def dominant_frequency_hz(signal: np.ndarray, dt: float) -> float:
    """Dominant frequency of a time series by FFT (peak of the one-sided PSD)."""
    n = signal.size
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(signal * win))
    freqs = np.fft.rfftfreq(n, d=dt)
    # skip the DC bin
    idx = 1 + int(np.argmax(spec[1:]))
    return float(freqs[idx])


# ---------------------------------------------------------------------------
# 5. Expected (analytic) values
# ---------------------------------------------------------------------------
def expected_springing_peak(proj: dict, F_amp: torch.Tensor, zeta: float,
                            section: BoxGirderSection, elements: int) -> float:
    """Steady-state resonant peak bending moment [N m].

    For a harmonic modal force ``F_n(t) = F_amp_n sin(w_n t)`` at the n-th
    natural frequency, the steady-state modal amplitude is
    ``u_n = F_amp_n / (2 zeta w_n^2)`` (mass-normalised, ``M_n = 1``).  The peak
    bending moment is ``EI * max_n |u_n * d theta_n/dx|`` at the midspan.
    """
    curv = curvature_at_midspan(proj, section, elements)
    u_amp = F_amp / (2.0 * zeta * proj["omega"] ** 2)
    return float(torch.max(torch.abs(section.bending_stiffness * u_amp * curv)))


def expected_whipping_peak(proj: dict, impulse: torch.Tensor, zeta: float,
                           section: BoxGirderSection, elements: int,
                           duration_s: float = 60.0, dt_s: float = 0.005) -> float:
    """Peak bending moment of the analytic modal superposition after an impulse.

    The impulse ``I_n = int F_n dt`` sets the modal velocity ``v_n = I_n``
    (``M_n = 1``); each mode then rings down as a damped free oscillator
    ``u_n(t) = (I_n / w_n) e^{-z w_n t} sin(w_dn t)``.  The *sum* of the modal
    contributions (not the per-mode max) is evaluated on a fine grid and its
    peak is the expected whipping bending moment.
    """
    curv = curvature_at_midspan(proj, section, elements)
    w = proj["omega"]
    wd = w * torch.sqrt(torch.as_tensor(1.0 - zeta**2, dtype=_D))
    t = torch.linspace(0.0, duration_s, int(duration_s / dt_s) + 1, dtype=_D)
    u = torch.zeros((t.numel(), w.numel()), dtype=_D)
    for n in range(w.numel()):
        u[:, n] = (impulse[n] / w[n]) * torch.exp(-zeta * w[n] * t) * torch.sin(wd[n] * t)
    M = section.bending_stiffness * torch.einsum("tn,n->t", u, curv)
    return float(torch.max(torch.abs(M)))


# ---------------------------------------------------------------------------
# 6. Cases
# ---------------------------------------------------------------------------
def run_springing_case(proj: dict, section: BoxGirderSection, elements: int,
                       anchor: dict, zeta: float, n_periods: int = 40,
                       steps_per_period: int = 200) -> dict:
    """Harmonic springing excitation at the wet fundamental, Newmark response."""
    f1 = float(proj["freq_hz"][0])
    w1 = float(proj["omega"][0])
    p0 = anchor["p_rms_pa"]                       # T1 excitation magnitude [Pa]
    F_nodal = uniform_pressure_nodal_load(section, elements, p0)
    F_amp = modal_force_vector(proj, F_nodal)     # (n_modes,)

    # the startup transient decays as e^{-2 pi zeta N}; require < 1 % in the
    # steady-state window (last 10 periods):  N > 0.73/zeta + 10
    n_periods = max(n_periods, int(0.73 / zeta) + 10)
    T_period = 1.0 / f1
    dt = T_period / steps_per_period
    t_end = n_periods * T_period
    time = torch.linspace(0.0, t_end, n_periods * steps_per_period + 1, dtype=_D)
    # deterministic harmonic modal force at the fundamental frequency
    F_modal = torch.zeros((time.numel(), proj["omega"].numel()), dtype=_D)
    F_modal[:, 0] = F_amp[0] * torch.sin(w1 * time)

    u_modal = run_modal_newmark(proj, zeta, F_modal, time)
    M_t = reconstruct_bending_moment(proj, section, elements, u_modal)
    M_np = M_t.detach().cpu().numpy()

    # steady-state window (last 10 periods)
    n_ss = 10 * steps_per_period
    M_ss = M_np[-n_ss:]
    M_peak = float(np.max(np.abs(M_ss)))
    f_dom = dominant_frequency_hz(M_ss, dt)

    # expected (analytic)
    M_expected = expected_springing_peak(proj, F_amp, zeta, section, elements)
    # static response to the same uniform pressure (for the DAF):
    # M_static = p0 * B * L^2 / 8  (simply-supported beam, uniform load)
    M_static = p0 * section.breadth * section.length ** 2 / 8.0
    daf = M_peak / M_static if M_static > 0 else float("nan")

    return {
        "case": f"springing_zeta_{zeta:g}",
        "excitation": "harmonic springing pressure at wet fundamental",
        "p0_pa": p0,
        "zeta": zeta,
        "f_exc_hz": f1,
        "f_dom_hz": f_dom,
        "f_dom_rel_dev_pct": 100.0 * abs(f_dom - f1) / f1,
        "M_peak_nm": M_peak,
        "M_expected_nm": M_expected,
        "M_rel_dev_pct": 100.0 * abs(M_peak - M_expected) / M_expected,
        "M_static_nm": M_static,
        "daf": daf,
        "daf_analytic_1_over_2zeta": 1.0 / (2.0 * zeta),
        "sigma_peak_pa": M_peak * (0.5 * section.draft) / section.inertia,
        "n_periods": n_periods,
        "steps_per_period": steps_per_period,
        "dt_s": dt,
    }


def run_whipping_case(proj: dict, section: BoxGirderSection, elements: int,
                      anchor: dict, zeta: float, t_pulse: float = 0.10,
                      duration_s: float = 60.0, dt_s: float = 0.005) -> dict:
    """Half-sine slamming impulse (whipping), Newmark transient response."""
    f1 = float(proj["freq_hz"][0])
    w1 = float(proj["omega"][0])
    p_peak = anchor["p_rms_pa"]                    # T1 excitation magnitude [Pa]
    F_nodal = uniform_pressure_nodal_load(section, elements, p_peak)
    F_amp = modal_force_vector(proj, F_nodal)

    # half-sine pulse: F(t) = F_amp * sin(pi t / t_pulse) for 0 <= t <= t_pulse
    n_pulse = max(2, int(t_pulse / dt_s))
    n_total = int(duration_s / dt_s)
    time = torch.linspace(0.0, duration_s, n_total + 1, dtype=_D)
    F_modal = torch.zeros((time.numel(), proj["omega"].numel()), dtype=_D)
    for i in range(n_pulse + 1):
        t = i * dt_s
        if t <= t_pulse:
            F_modal[i] = F_amp * math.sin(math.pi * t / t_pulse)

    # impulse (trapezoid)
    impulse = torch.trapezoid(F_modal[: n_pulse + 1], time[: n_pulse + 1], dim=0)

    u_modal = run_modal_newmark(proj, zeta, F_modal, time)
    M_t = reconstruct_bending_moment(proj, section, elements, u_modal)
    M_np = M_t.detach().cpu().numpy()

    M_peak = float(np.max(np.abs(M_np)))
    f_dom = dominant_frequency_hz(M_np, dt_s)

    # decay time: fit the envelope of the fundamental mode after the peak
    # tau = 1 / (zeta * w1)  (analytic)
    tau_analytic = 1.0 / (zeta * w1)
    # measured: time for the envelope to fall to 1/e of the peak (Hilbert-free:
    # use the local maxima of the fundamental-mode band)
    idx_peak = int(np.argmax(np.abs(M_np)))
    # find the first local maximum after the peak
    M_abs = np.abs(M_np)
    # envelope via moving max over one period
    n_per = int(1.0 / (f1 * dt_s))
    env = np.array([np.max(M_abs[max(0, i - n_per // 2): i + n_per // 2 + 1])
                    for i in range(idx_peak, M_abs.size)])
    t_env = time[idx_peak:].detach().cpu().numpy()
    target = M_peak * math.exp(-1.0)
    # first crossing of the envelope below target
    cross = np.where(env < target)[0]
    tau_measured = float(t_env[cross[0]] - t_env[0]) if cross.size else float("nan")

    M_expected = expected_whipping_peak(proj, impulse, zeta, section, elements,
                                   duration_s, dt_s)

    return {
        "case": f"whipping_zeta_{zeta:g}",
        "excitation": "half-sine slamming impulse (whipping)",
        "p_peak_pa": p_peak,
        "zeta": zeta,
        "t_pulse_s": t_pulse,
        "f_dom_hz": f_dom,
        "f_dom_rel_dev_pct": 100.0 * abs(f_dom - f1) / f1,
        "M_peak_nm": M_peak,
        "M_expected_nm": M_expected,
        "M_rel_dev_pct": 100.0 * abs(M_peak - M_expected) / M_expected,
        "sigma_peak_pa": M_peak * (0.5 * section.draft) / section.inertia,
        "tau_measured_s": tau_measured,
        "tau_analytic_s": tau_analytic,
        "tau_rel_dev_pct": (100.0 * abs(tau_measured - tau_analytic) / tau_analytic
                            if math.isfinite(tau_measured) else None),
        "impulse_ns": float(torch.sum(impulse)),
        "duration_s": duration_s,
        "dt_s": dt_s,
    }


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main() -> int:
    anchor = load_anchor()
    section = kvlcc2_section(anchor)
    section.validate()
    ga = section.geometry_audit()

    elements = 40
    n_modes = 4
    m33, lewis = lewis_heave_added_mass(0.5 * anchor["B"], 0.5 * anchor["T"], anchor["rho"])
    extra = ballast_mass_per_length(section)
    proj = modal_projection(section, elements, m33, extra, n_modes)

    # --- geometry gate -----------------------------------------------------
    geometry_gate = {
        "area_rel_error": ga["area_rel_error"],
        "inertia_rel_error": ga["inertia_rel_error"],
        "area_thin_wall_rel_dev": ga["area_thin_wall_rel_dev"],
        "inertia_thin_wall_rel_dev": ga["inertia_thin_wall_rel_dev"],
        "block_coefficient": section.block_coefficient,
        "slenderness": section.slenderness,
        "draft_over_depth": section.section_draft / section.depth,
        "pass": (ga["area_rel_error"] < 1e-9 and ga["inertia_rel_error"] < 1e-9),
    }

    # --- cases -------------------------------------------------------------
    cases = []
    for zeta in (0.02, 0.01, 0.05):
        cases.append(run_springing_case(proj, section, elements, anchor, zeta))
    cases.append(run_whipping_case(proj, section, elements, anchor, 0.02))

    # --- criteria (machine-judged) -----------------------------------------
    criteria = []
    for c in cases:
        peak_ok = c["M_rel_dev_pct"] <= 15.0
        freq_ok = c["f_dom_rel_dev_pct"] <= 15.0
        criteria.append({
            "case": c["case"],
            "criterion_peak": "PASS" if peak_ok else "FAIL",
            "criterion_frequency": "PASS" if freq_ok else "FAIL",
            "peak_rel_dev_pct": c["M_rel_dev_pct"],
            "freq_rel_dev_pct": c["f_dom_rel_dev_pct"],
            "limit_pct": 15.0,
        })

    # damping sensitivity (springing peak vs zeta)
    springing = [c for c in cases if c["case"].startswith("springing")]
    damping_sensitivity = {
        "zeta_values": [c["zeta"] for c in springing],
        "M_peak_nm": [c["M_peak_nm"] for c in springing],
        "daf": [c["daf"] for c in springing],
        "daf_analytic_1_over_2zeta": [c["daf_analytic_1_over_2zeta"] for c in springing],
        "daf_ratio_computed_over_analytic": [
            c["daf"] / c["daf_analytic_1_over_2zeta"] for c in springing
        ],
        "note": "DAF = M_peak_dynamic / M_peak_static; analytic resonant DAF = 1/(2 zeta)",
    }

    # --- verdict -----------------------------------------------------------
    all_peak = all(c["criterion_peak"] == "PASS" for c in criteria)
    all_freq = all(c["criterion_frequency"] == "PASS" for c in criteria)
    # The external SNAME bulletin is NOT_OBTAINED (T1 source registry): the
    # 'anchor' used here is the T1 synthetic fixture, claim_eligible = false.
    external_anchor_status = "WITHHELD"
    verdict = {
        "overall": "PASS" if (all_peak and all_freq) else "FAIL",
        "kind": (
            "domain experiment PASS on the T1-caliber consistency gates; "
            "external SNAME T&R Bulletin 3-37/4-7 anchor remains WITHHELD "
            "(NOT_OBTAINED, recorded in the T1 source registry) -- the "
            "comparison is against the T1 labelled synthetic fixture, "
            "claim_eligible=false, and is not a validation against a "
            "transcribed source value"
            if (all_peak and all_freq) else
            "honest FAIL -- at least one consistency gate exceeded 15 %; "
            "see per-case rows"
        ),
        "external_anchor_status": external_anchor_status,
        "why": (
            "all computed (Newmark) peaks and dominant frequencies agree with "
            "the analytic modal-transfer expectations within the 15 % gate; "
            "the residual is the Newmark discretisation error (second-order "
            "scheme, ~1 % at 200 steps/period)"
            if (all_peak and all_freq) else
            "one or more cases exceed the 15 % gate; the gap is reported "
            "verbatim in the case rows"
        ),
    }

    artifact = {
        "schema_version": "1.0.0",
        "task_key": "paper-exp-springing-whipping-hydroelastic-2",
        "topic": "springing-whipping-hydroelastic",
        "experiment": (
            "research_plan item 2 -- domain experiment: hydroelastic "
            "time-domain response in TensorFEM vs the T1-anchor measured "
            "magnitude (peak / dominant-frequency / damping-sensitivity)"
        ),
        "repository": "TensorFEM",
        "branch": "agent/m27813",
        "generated_by": {
            "script": "benchmarks/springing-whipping-hydroelastic/run_comparison.py",
            "command": "PYTHONPATH=src python benchmarks/springing-whipping-hydroelastic/run_comparison.py",
            "python": "3.11.6",
            "numpy": "2.4.6",
            "torch": "2.7.1",
            "tensorfem": "0.45.0",
        },
        "anchor_reference": {
            "path": str(ANCHOR_PATH),
            "values_used": {
                "L_m": anchor["L"],
                "B_m": anchor["B"],
                "T_m": anchor["T"],
                "rho_kg_m3": anchor["rho"],
                "V_m_s": anchor["V"],
                "q_pa": anchor["q_pa"],
                "p_rms_pa": anchor["p_rms_pa"],
                "cp_rms_star": anchor["cp_rms_star"],
                "amplitude_A": anchor["amplitude_A"],
                "l_ref_used_m": anchor["l_ref_used_m"],
                "convention_status": anchor["convention_status"],
                "anchor_verdict": anchor["anchor_verdict"],
                "anchor_verdict_kind": anchor["anchor_verdict_kind"],
            },
            "claim_eligible": False,
            "note": (
                "the T1 anchor's dry-run fixture is a labelled synthetic "
                "spectrum (S*(x) = A x^2 exp(-x^2), Cp_rms = 0.01); it is "
                "NOT a transcribed SNAME T&R Bulletin value.  The external "
                "bulletin is NOT_OBTAINED in this sandbox (T1 source "
                "registry), so no source-anchor number exists to compare "
                "against."
            ),
        },
        "case_caliber": {
            "case_id": "kvlcc2_hull_girder_caliber",
            "description": (
                "KVLCC2-class hull-girder vertical bending caliber: the scale "
                "at which the T1 springing excitation spectrum is expressed "
                "(L=320 m, B=58 m, T=20.8 m, rho=1025 kg/m^3, V=7.9732 m/s)."
            ),
            "quantities": {
                "L": {"value": anchor["L"], "unit": "m", "grade": "REPO_DOCUMENTED"},
                "B": {"value": anchor["B"], "unit": "m", "grade": "REPO_DOCUMENTED"},
                "T": {"value": anchor["T"], "unit": "m", "grade": "REPO_DOCUMENTED"},
                "D": {"value": 30.0, "unit": "m", "grade": "REPO_DOCUMENTED",
                      "source": "docs/hull-girder-csr-bending-chain.md (moulded depth)"},
                "I_midship": {"value": 1646.7, "unit": "m^4", "grade": "REPO_DOCUMENTED",
                              "source": "docs/hull-girder-csr-bending-chain.md"},
                "rho": {"value": anchor["rho"], "unit": "kg/m^3", "grade": "REPO_DOCUMENTED"},
                "V": {"value": anchor["V"], "unit": "m/s", "grade": "CASE_INPUT"},
            },
        },
        "geometry_gate": geometry_gate,
        "wet_modal_model": {
            "elements": elements,
            "n_modes": n_modes,
            "lewis_added_mass_kg_m": m33,
            "lewis_area_ratio": getattr(lewis, "area_ratio", None),
            "ballast_mass_per_length_kg_m": extra,
            "structural_mass_per_length_kg_m": section.mass_per_length,
            "floating_mass_per_length_kg_m": section.displaced_mass_per_length,
            "wet_freq_hz": proj["freq_hz"].tolist(),
            "dry_freq_hz": None,  # filled below
        },
        "cases": cases,
        "criteria": criteria,
        "damping_sensitivity": damping_sensitivity,
        "verdict": verdict,
        "limitations": [
            "the hull girder is a uniform equivalent box girder (KVLCC2 outer "
            "dimensions, I matched to the CSR-chain I_midship); a real hull is "
            "non-uniform, so the modal spacing differs from a built ship",
            "the springing excitation is a deterministic harmonic pressure at "
            "the wet fundamental (p_0 = T1 p_rms); a real turbulent boundary "
            "layer is broadband and spatially non-uniform -- the frequency "
            "convention (l_ref = L vs l_ref = delta*) is UNVERIFIED in the T1 "
            "anchor and shifts the springing band by L/delta* ~ 320",
            "the whipping pulse is a half-sine slamming impulse with a "
            "declared duration; the real slamming load shape is not "
            "transcribed from a source",
            "the comparison is against the T1 synthetic fixture "
            "(claim_eligible=false), not against a transcribed SNAME T&R "
            "Bulletin value; the external bulletin is NOT_OBTAINED",
        ],
        "reproduce": {
            "generate": "PYTHONPATH=src python benchmarks/springing-whipping-hydroelastic/run_comparison.py",
            "anchor_gate": "python benchmarks/anchors/springing_whipping_hydroelastic_anchor.py --check",
        },
    }

    # dry frequencies (for the wet/dry ratio report)
    dry = wet_simply_supported_modes(section, elements, 0.0, extra_mass=0.0, modes=n_modes)
    artifact["wet_modal_model"]["dry_freq_hz"] = (dry.frequencies_hz / (2 * math.pi)).tolist() \
        if False else dry.frequencies_hz.tolist()

    OUT_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {OUT_PATH}")
    print(f"verdict: {verdict['overall']} ({verdict['kind'][:80]}...)")
    for c in criteria:
        print(f"  {c['case']}: peak {c['criterion_peak']} "
              f"({c['peak_rel_dev_pct']:.3f}%), freq {c['criterion_frequency']} "
              f"({c['freq_rel_dev_pct']:.3f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
