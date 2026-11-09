"""External-anchor reconciliation and caliber transfer for the research topic
``springing-whipping-hydroelastic`` (W18 paper production line, experiment 1).

What this script does
---------------------
1. **Transcription gate.** Loads ``benchmarks/anchors/sources/
   sname_tr_bulletin_3-37_4-7.transcription.json`` and classifies every anchor
   entry as *claim eligible* or *claim ineligible*. An entry is eligible only
   when it carries a character-exact quote, a page/table locator and
   ``verbatim_status == "VERBATIM"``. Nothing in this file is typed from
   memory: the script has no built-in numeric anchor constants at all.
2. **Caliber transfer chain.** Implements the nondimensional -> case-caliber
   mapping for a hull-girder scale case (KVLCC2-class particulars, taken from
   the repository's own ``docs/hull-girder-csr-bending-chain.md``) as an
   explicitly unit-tagged sequence of steps, so that "same caliber" means the
   same, written-down, dimensionalised convention and not an impression.
3. **Machine checks.** Dimensional homogeneity of every step, round-trip
   residual, quadrature vs closed-form moment, unit-system invariance, spectral
   scale-exponent invariance, time-domain synthesis closure, and the
   *convention sensitivity* of the frequency axis (which is the reason the
   transfer cannot be completed without the source text).

Honesty contract
----------------
If the transcription registry holds no eligible entry, the anchor transfer is
reported ``WITHHELD`` and the overall experiment verdict is ``FAIL``
(honest FAIL). The machinery is nevertheless exercised end-to-end on a
*synthetic dry-run fixture* that is labelled as such everywhere it appears and
is never presented as a source value. That distinction is structural: the
fixture lives under ``dry_run_fixture`` in the artifact, and every derived
number carries ``claim_eligible: false``.

Run
---
    python benchmarks/anchors/springing_whipping_hydroelastic_anchor.py
    python benchmarks/anchors/springing_whipping_hydroelastic_anchor.py --check

``--check`` exits non-zero if any machine check fails or if the artifact would
claim more than the transcription gate allows.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ANCHOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANCHOR_DIR.parents[1]
TRANSCRIPTION_PATH = ANCHOR_DIR / "sources" / "sname_tr_bulletin_3-37_4-7.transcription.json"
OUT_PATH = ANCHOR_DIR / "springing-whipping-hydroelastic.json"

TASK_KEY = "paper-exp-springing-whipping-hydroelastic-1"
TOPIC = "springing-whipping-hydroelastic"

# --------------------------------------------------------------------------
# 1. Case caliber ("本算例口径")
# --------------------------------------------------------------------------
# Inputs are *declared case inputs*, not external anchors: each one is either
# taken from a file that lives in this repository (path recorded) or is an
# explicit modelling assumption of the case (grade CASE_INPUT).
CASE = {
    "case_id": "kvlcc2_hull_girder_caliber",
    "description": (
        "Hull-girder vertical bending caliber of a KVLCC2-class ship: the scale at "
        "which a springing excitation spectrum has to be expressed in order to be "
        "usable by the TensorFEM hull-girder modal / hydroelastic models."
    ),
    "quantities": {
        "L": {"value": 320.0, "unit": "m", "grade": "REPO_DOCUMENTED",
              "source": "docs/hull-girder-csr-bending-chain.md (length between perpendiculars, KVLCC2)"},
        "B": {"value": 58.0, "unit": "m", "grade": "REPO_DOCUMENTED",
              "source": "docs/hull-girder-csr-bending-chain.md (moulded breadth, KVLCC2)"},
        "T": {"value": 20.8, "unit": "m", "grade": "REPO_DOCUMENTED",
              "source": "docs/hull-girder-csr-bending-chain.md (design draft, KVLCC2)"},
        "rho": {"value": 1025.0, "unit": "kg/m^3", "grade": "REPO_DOCUMENTED",
                "source": "docs/hull-girder-csr-bending-chain.md (sea water density)"},
        "V": {"value": 7.9732, "unit": "m/s", "grade": "CASE_INPUT",
              "source": "declared service speed 15.5 kn (0.514444 m/s per kn); an assumption of this case, not an external anchor"},
        "delta_star": {"value": 1.0, "unit": "m", "grade": "CASE_INPUT",
                       "source": "nominal hull boundary-layer displacement thickness at the springing band; declared assumption, swept below"},
    },
}

RHO = CASE["quantities"]["rho"]["value"]
V = CASE["quantities"]["V"]["value"]
L = CASE["quantities"]["L"]["value"]
DELTA_STAR = CASE["quantities"]["delta_star"]["value"]


# --------------------------------------------------------------------------
# 2. Dimensional algebra (minimal, dependency free)
# --------------------------------------------------------------------------
# A dimension is a tuple of exponents over (mass [kg], length [m], time [s]).
KG, M, S = 0, 1, 2


def dim(mass: int = 0, length: int = 0, time: int = 0) -> tuple[int, int, int]:
    return (mass, length, time)


def dmul(*dims: tuple[int, int, int]) -> tuple[int, int, int]:
    out = [0, 0, 0]
    for d in dims:
        for i in range(3):
            out[i] += d[i]
    return tuple(out)  # type: ignore[return-value]


def dpow(d: tuple[int, int, int], n: int) -> tuple[int, int, int]:
    return tuple(e * n for e in d)  # type: ignore[return-value]


def dsqrt(d):
    """Square root of a dimension whose exponents are all even."""
    if any(e % 2 for e in d):
        raise ValueError(f"non-integer exponents in sqrt of {d}")
    return tuple(e // 2 for e in d)  # type: ignore[return-value]


def ddiv(a, b):
    return dmul(a, dpow(b, -1))


def dstr(d) -> str:
    names = ("kg", "m", "s")
    num = [f"{names[i]}^{e}" for i, e in enumerate(d) if e]
    return "*".join(num) if num else "1"


DIMENSIONLESS = dim()
PASCAL = dim(1, -1, -2)          # kg m^-1 s^-2
SECOND = dim(0, 0, 1)
HERTZ = dim(0, 0, -1)
PA2_PER_HZ = dmul(dpow(PASCAL, 2), SECOND)   # Pa^2 s == Pa^2 / Hz


# --------------------------------------------------------------------------
# 3. Caliber transfer chain
# --------------------------------------------------------------------------
def transfer_chain(cp_rms_star: float, spectrum_shape, l_ref: float,
                   rho: float = RHO, v: float = V, length: float = L):
    """Return the dimensionalised chain for one assumed convention.

    Convention (to be confirmed verbatim from the source before use):

        Cp_rms^2 = INT_0^inf S*(w*) dw*,      w* = w * l_ref / v
        q        = 0.5 * rho * v^2                      [Pa]
        S_p(w)   = q^2 * (l_ref / v) * S*(w l_ref / v)  [Pa^2 s]
        S_p(f)   = 2*pi * S_p(w)                        [Pa^2 s]
        p_rms    = sqrt(INT S_p(w) dw) = q * Cp_rms     [Pa]

    ``l_ref`` is the reference length that closes the nondimensionalisation.
    It is exactly the quantity the source text must supply verbatim: with
    ``l_ref = L`` (ship length) or ``l_ref = delta*`` (boundary-layer scale)
    the same nondimensional abscissa maps onto frequencies that differ by
    ``L / delta*``.
    """
    q = 0.5 * rho * v ** 2                      # Pa
    k_omega = v / l_ref                         # w* = w / k_omega

    def s_star_over_omega(omega):
        return spectrum_shape(np.asarray(omega, dtype=float) / k_omega)

    def s_p_omega(omega):
        return q ** 2 * (l_ref / v) * spectrum_shape(np.asarray(omega, dtype=float) / k_omega)

    def s_p_f(f):
        return 2.0 * math.pi * s_p_omega(2.0 * math.pi * np.asarray(f, dtype=float))

    p_rms = q * cp_rms_star
    return {
        "l_ref_m": l_ref,
        "q_pa": q,
        "s_star_over_omega": s_star_over_omega,
        "s_p_omega": s_p_omega,
        "s_p_f": s_p_f,
        "p_rms_pa": p_rms,
        "omega_star_of_f": lambda f: 2.0 * math.pi * f * l_ref / v,
        "f_of_omega_star": lambda ws: ws * v / (2.0 * math.pi * l_ref),
    }


def f_from_omega_star(ws: float, l_ref: float = L, v: float = V) -> float:
    """Dimensional frequency [Hz] carrying the reduced frequency ``ws``."""
    return ws * v / (2.0 * math.pi * l_ref)

CHAIN_STEPS = [
    {
        "id": "S1",
        "formula": "q = 0.5 * rho * V^2",
        "lhs_unit": PASCAL,
        "rhs_unit": dmul(dim(1, -3, 0), dpow(dim(0, 1, -1), 2)),
        "note": "dynamic pressure; kg/m^3 * (m/s)^2 = kg m^-1 s^-2 = Pa",
    },
    {
        "id": "S2",
        "formula": "omega = 2*pi*f",
        "lhs_unit": HERTZ,
        "rhs_unit": dpow(SECOND, -1),
        "note": "Hz = 1/s",
    },
    {
        "id": "S3",
        "formula": "S_p(omega) = q^2 * (l_ref/V) * S*(omega*l_ref/V)",
        "lhs_unit": PA2_PER_HZ,
        "rhs_unit": dmul(dpow(PASCAL, 2), ddiv(dim(0, 1, 0), dim(0, 1, -1)), DIMENSIONLESS),
        "note": "Pa^2 * (m / (m/s)) = Pa^2 s; the bracket (l_ref/V) carries s",
    },
    {
        "id": "S4",
        "formula": "S_p(f) = 2*pi*S_p(omega)",
        "lhs_unit": PA2_PER_HZ,
        "rhs_unit": PA2_PER_HZ,
        "note": "2*pi is dimensionless; spectral density per Hz equals per rad/s up to 2*pi",
    },
    {
        "id": "S5",
        "formula": "p_rms = sqrt(INT_0^inf S_p(omega) domega)",
        "lhs_unit": PASCAL,
        "rhs_unit": dsqrt(dmul(PA2_PER_HZ, HERTZ)),
        "note": "rhs: sqrt(Pa^2 s * 1/s) = Pa",
    },
    {
        "id": "S6",
        "formula": "omega* = omega * l_ref / V",
        "lhs_unit": DIMENSIONLESS,
        "rhs_unit": dmul(HERTZ, ddiv(dim(0, 1, 0), dim(0, 1, -1))),
        "note": "reduced frequency; 1/s * s = 1",
    },
]


# --------------------------------------------------------------------------
# 4. Synthetic dry-run fixture (NOT a source value)
# --------------------------------------------------------------------------
# S*(x) = A * x^2 * exp(-x^2)  with  INT_0^inf S* dx = A * sqrt(pi)/4.
# A is chosen so that Cp_rms = FIXTURE_CP_RMS.  Nothing here is taken from the
# bulletin; the shape and amplitude are arbitrary and exist only to exercise
# the chain.  Every number derived from them is emitted with
# claim_eligible = false.
FIXTURE_CP_RMS = 0.01
FIXTURE_P = 2
FIXTURE_Q = 2
FIXTURE_B = 1.0


def fixture_shape(x):
    x = np.asarray(x, dtype=float)
    return FIXTURE_A * x ** FIXTURE_P * np.exp(-FIXTURE_B * x ** FIXTURE_Q)


FIXTURE_M0_CLOSED_FORM = math.sqrt(math.pi) / 4.0   # INT x^2 e^{-x^2} dx
FIXTURE_A = FIXTURE_CP_RMS ** 2 / FIXTURE_M0_CLOSED_FORM
FIXTURE_INF = 40.0


def fixture_shape_scalar(x: float) -> float:
    return float(fixture_shape(x))


def fixture_shape_array(x):
    return fixture_shape(x)


# --------------------------------------------------------------------------
# 5. Machine checks
# --------------------------------------------------------------------------
def check_dimensional_homogeneity():
    rows = []
    for step in CHAIN_STEPS:
        ok = tuple(step["lhs_unit"]) == tuple(step["rhs_unit"])
        rows.append({
            "step": step["id"],
            "formula": step["formula"],
            "lhs": dstr(step["lhs_unit"]),
            "rhs": dstr(step["rhs_unit"]),
            "homogeneous": ok,
            "note": step["note"],
        })
    return {
        "id": "CHK1_dimensional_homogeneity",
        "method": "exponent algebra over (kg, m, s) for both sides of every conversion step",
        "steps": rows,
        "n_steps": len(rows),
        "n_homogeneous": sum(r["homogeneous"] for r in rows),
        "tolerance": "exact",
        "passed": all(r["homogeneous"] for r in rows),
    }


def check_round_trip():
    """Case caliber -> nondimensional -> case caliber, on a reduced-frequency band."""
    chain = transfer_chain(FIXTURE_CP_RMS, fixture_shape_array, l_ref=L)
    ws = np.linspace(0.2, 9.0, 4001)              # band where the fixture is non-negligible
    f = ws * V / (2.0 * math.pi * L)
    sp_f = chain["s_p_f"](f)
    s_back = sp_f / (2.0 * math.pi) / (chain["q_pa"] ** 2 * (L / V))
    f_back = ws * V / (2.0 * math.pi * L)
    ref = fixture_shape(ws)
    floor = 1e-200
    rel_spec = float(np.max(np.abs(s_back - ref) / np.maximum(np.abs(ref), floor)))
    rel_f = float(np.max(np.abs(f_back - f) / f))
    residual = max(rel_spec, rel_f)
    return {
        "id": "CHK2_round_trip_residual",
        "method": "apply S1..S3 forward on a reduced-frequency band (0.2..9, 4001 points), then invert both the ordinate and the abscissa and compare",
        "band_omega_star": [0.2, 9.0],
        "band_hz": [float(f[0]), float(f[-1])],
        "max_relative_residual": residual,
        "ordinate_residual": rel_spec,
        "abscissa_residual": rel_f,
        "tolerance": 1e-9,
        "passed": residual < 1e-9,
        "claim_eligible": False,
        "fixture_note": "evaluated on the synthetic dry-run fixture, not on a source anchor",
    }


def check_moment_closure():
    """INT S* dw* vs closed form, and INT S_p dw vs q^2 Cp_rms^2."""
    x = np.linspace(0.0, FIXTURE_INF, 400001)
    m0_num = float(np.trapezoid(fixture_shape(x), x))
    m0_cf = FIXTURE_A * FIXTURE_M0_CLOSED_FORM
    rel = abs(m0_num - m0_cf) / m0_cf

    chain = transfer_chain(FIXTURE_CP_RMS, fixture_shape_array, l_ref=L)
    w = np.linspace(0.0, 60.0, 1200001)
    sp = chain["s_p_omega"](w)
    var_num = float(np.trapezoid(sp, w))
    var_cf = chain["p_rms_pa"] ** 2
    rel2 = abs(var_num - var_cf) / var_cf
    return {
        "id": "CHK3_spectral_moment_closure",
        "method": "trapezoidal quadrature of the nondimensional spectrum vs its closed-form moment; then quadrature of the dimensionalised spectrum vs (q*Cp_rms)^2",
        "m0_star_numeric": m0_num,
        "m0_star_closed_form": m0_cf,
        "m0_relative_residual": rel,
        "variance_pa2_numeric": var_num,
        "variance_pa2_closed_form": var_cf,
        "variance_relative_residual": rel2,
        "tolerance": 1e-9,
        "passed": rel < 1e-9 and rel2 < 1e-9,
        "claim_eligible": False,
    }


def check_unit_system_invariance():
    """Recompute the chain in a coherent cm-based system and map back."""
    scale = 100.0                      # m -> cm
    rho_cm = RHO / scale ** 3          # kg/cm^3
    v_cm = V * scale                   # cm/s
    l_cm = L * scale                   # cm
    chain_si = transfer_chain(FIXTURE_CP_RMS, fixture_shape_scalar, l_ref=L)
    chain_cm = transfer_chain(FIXTURE_CP_RMS, fixture_shape_scalar, l_ref=l_cm,
                              rho=rho_cm, v=v_cm, length=l_cm)
    f_ref = f_from_omega_star(2.0)
    sp_si = chain_si["s_p_f"](f_ref)
    sp_cm = chain_cm["s_p_f"](f_ref)
    # 1 kg/(cm s^2) = 100 Pa, so Pa^2 s -> 1e4 x the cm-system value
    sp_cm_in_si = sp_cm * scale ** 2       # 1 kg/(cm s^2) = 100 Pa  ->  squared = (100)^2 = scale^2
    rel = abs(sp_cm_in_si - sp_si) / sp_si
    # the nondimensional abscissa must be identical in both systems
    ws_si = chain_si["omega_star_of_f"](f_ref)
    ws_cm = 2.0 * math.pi * f_ref * l_cm / v_cm
    rel_ws = abs(ws_cm - ws_si) / ws_si
    return {
        "id": "CHK4_unit_system_invariance",
        "method": "evaluate S_p(f) in (kg, m, s) and in (kg, cm, s); map the cm result back with 1 kg/(cm s^2) = 100 Pa and compare",
        "evaluated_at_hz": f_ref,
        "s_p_si": sp_si,
        "s_p_cm_mapped_to_si": sp_cm_in_si,
        "relative_residual": rel,
        "reduced_frequency_residual": rel_ws,
        "tolerance": 1e-12,
        "passed": rel < 1e-12 and rel_ws < 1e-12,
        "claim_eligible": False,
    }


def check_scale_exponents():
    """Rescale (rho, V, L) at fixed reduced frequency; compare with exponents."""
    lam = 2.0
    f_ref = f_from_omega_star(2.0)   # w* = 2 at the base state
    base = transfer_chain(FIXTURE_CP_RMS, fixture_shape_scalar, l_ref=L)
    scaled = transfer_chain(FIXTURE_CP_RMS, fixture_shape_scalar, l_ref=L * lam,
                            rho=RHO * lam, v=V * lam)
    lhs = scaled["s_p_f"](f_ref) / base["s_p_f"](f_ref)
    # analytic: S_p = q^2 (L/V) S*  ~  rho^2 V^4 * (L/V) = rho^2 V^3 L  at fixed w*
    expected = lam ** 2 * lam ** 3 * lam ** 1
    rel = abs(lhs - expected) / expected
    return {
        "id": "CHK5_spectral_scale_exponents",
        "method": "S_p(rho, V, L, f) rescaled by lambda at fixed reduced frequency (L/V is invariant under a common scaling of L and V, so f is held); compares against the analytic exponents rho^2 V^3 L^1",
        "lambda": lam,
        "measured_ratio": lhs,
        "analytic_ratio": expected,
        "relative_residual": rel,
        "tolerance": 1e-12,
        "passed": rel < 1e-12,
        "claim_eligible": False,
    }


def check_time_domain_closure():
    """Two closures: (a) exact orthogonal multi-sine; (b) seed-averaged random phase."""
    chain = transfer_chain(FIXTURE_CP_RMS, fixture_shape_array, l_ref=L)
    f_max = 5.0
    n_bins = 512
    df = f_max / n_bins
    f_c = np.arange(1, n_bins + 1) * df        # exact multiples of 1/T with T = 1/df
    sp = chain["s_p_f"](f_c)
    amp = np.sqrt(2.0 * sp * df)
    m0_quad = float(np.sum(sp * df))

    # (a) exact closure: orthogonal frequencies over one record
    dt = 1.0 / (2.0 * f_max)
    n = int(round((1.0 / df) / dt))            # exactly one period T = 1/df
    t = np.arange(n) * dt
    exact = np.zeros(n)
    for a, fc in zip(amp, f_c):
        exact += a * np.cos(2.0 * math.pi * fc * t)
    var_exact = float(np.mean(exact ** 2))
    rel_exact = abs(var_exact - m0_quad) / m0_quad

    # (b) statistical closure with random phases, averaged over seeds
    rng = np.random.default_rng(20261108)
    n_t = 1 << 15
    t2 = np.arange(n_t) * dt
    variances = []
    for _ in range(16):
        phase = rng.uniform(0.0, 2.0 * math.pi, size=n_bins)
        x = np.zeros(n_t)
        for a, fc, ph in zip(amp, f_c, phase):
            x += a * np.cos(2.0 * math.pi * fc * t2 + ph)
        variances.append(float(np.mean(x ** 2)))
    mean_var = float(np.mean(variances))
    rel_stat = abs(mean_var - m0_quad) / m0_quad
    return {
        "id": "CHK6_time_domain_closure",
        "method": "(a) multi-sine synthesis on orthogonal bins -> record variance vs spectral variance; (b) 16 random-phase realisations, record variance averaged over seeds (declared tolerance 3%, expected standard error ~1.1%)",
        "m0_from_quadrature_pa2": m0_quad,
        "variance_orthogonal_pa2": var_exact,
        "relative_residual_orthogonal": rel_exact,
        "variance_random_phase_mean_pa2": mean_var,
        "relative_residual_random_phase_mean": rel_stat,
        "tolerance_orthogonal": 1e-9,
        "tolerance_random_phase": 0.03,
        "passed": rel_exact < 1e-9 and rel_stat < 0.03,
        "claim_eligible": False,
        "seeding": "numpy default_rng(20261108), 16 seeds, 512 bins, 32768 samples",
    }


def check_convention_sensitivity():
    """How far the abscissa moves between the two candidate nondimensionalisations.

    This is the quantitative reason the transfer is blocked without the source:
    ``w* = w L / V`` and ``w* = w delta* / U`` differ by the factor L/delta*,
    so the same nondimensional abscissa maps onto frequencies that differ by
    that factor. Until the bulletin states its convention verbatim, no anchor
    can be placed on the case frequency axis.
    """
    band = (1.0, 10.0)      # a reduced-frequency band, dimensionless
    rows = []
    for ds in (0.1, 0.3, 1.0, 3.0):
        l_conv = L
        d_conv = ds
        f_l = band[0] * V / (2.0 * math.pi * l_conv)
        f_d = band[0] * V / (2.0 * math.pi * d_conv)
        rows.append({
            "delta_star_m": ds,
            "L_over_delta_star": l_conv / d_conv,
            "f_at_omega_star_min_if_l_ref_is_L_hz": f_l,
            "f_at_omega_star_min_if_l_ref_is_delta_star_hz": f_d,
            "frequency_shift_factor": f_d / f_l,
        })
    ratio = L / DELTA_STAR
    return {
        "id": "CHK7_convention_sensitivity",
        "method": "map an identical reduced-frequency band onto the case frequency axis under the two candidate reference lengths",
        "nominal_delta_star_m": DELTA_STAR,
        "nominal_L_over_delta_star": ratio,
        "rows": rows,
        "quantifies": "the frequency-axis ambiguity produced by an unresolved nondimensionalisation convention",
        "passed": ratio > 10.0,
        "decision_use": "supports WITHHELD on the anchor transfer: the convention must be resolved verbatim first",
    }


# --------------------------------------------------------------------------
# 6. Transcription gate
# --------------------------------------------------------------------------
def classify_entries(registry: dict):
    """No promotion without a character-exact quote and page/table locator."""
    rows = []
    for entry in registry.get("entries", []):
        quote = (entry.get("quote") or "").strip()
        locator = entry.get("locator") or {}
        has_locator = bool(locator.get("document_part") or locator.get("page"))
        status = entry.get("verbatim_status", "WITHHELD")
        eligible = bool(quote) and has_locator and status == "VERBATIM"
        rows.append({
            "anchor_id": entry.get("anchor_id"),
            "source_id": entry.get("source_id"),
            "verbatim_status": status,
            "has_quote": bool(quote),
            "has_locator": has_locator,
            "n_values": len(entry.get("values") or []),
            "claim_eligible": eligible,
            "rejection_reason": None if eligible else (
                "empty quote" if not quote else
                "missing page/table locator" if not has_locator else
                f"verbatim_status={status}"
            ),
        })
    return rows


# --------------------------------------------------------------------------
# 7. Artifact assembly
# --------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    except Exception:                                    # pragma: no cover
        return ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_artifact() -> dict:
    registry = json.loads(TRANSCRIPTION_PATH.read_text())
    entries = classify_entries(registry)
    eligible = [e for e in entries if e["claim_eligible"]]

    checks = [
        check_dimensional_homogeneity(),
        check_round_trip(),
        check_moment_closure(),
        check_unit_system_invariance(),
        check_scale_exponents(),
        check_time_domain_closure(),
        check_convention_sensitivity(),
    ]
    all_checks_pass = all(c["passed"] for c in checks)

    try:
        import tensorfem
        tf_version = getattr(tensorfem, "__version__", "unknown")
    except Exception:                                    # pragma: no cover
        tf_version = "not importable"

    # --- criteria -------------------------------------------------------
    criteria = [
        {
            "id": "C1",
            "requirement": "引句 + 表号/页码 + 数值逐字录入工件（值=来源原值，禁改写/禁四舍五入到失真）",
            "status": "WITHHELD",
            "measured": {
                "verbatim_entries_in_registry": len(eligible),
                "total_entries_in_registry": len(entries),
                "sources_sought": [s["id"] for s in registry.get("sources_sought", [])],
                "sources_obtained": [s["id"] for s in registry.get("sources_sought", [])
                                     if s.get("acquisition_status") != "NOT_OBTAINED"],
            },
            "verdict": "FAIL (honest)",
            "evidence": "benchmarks/anchors/sources/sname_tr_bulletin_3-37_4-7.transcription.json "
                        "(acquisition_status = NOT_OBTAINED for both sources; entries = [])",
            "gap": "No authorised copy of SNAME T&R Bulletin 3-37 (1997) or 4-7 is present in the "
                   "sandbox (filesystem, repository, platform corpus) and the series is a paid "
                   "SNAME publication, so no character-exact quote, table number or page can be "
                   "copied. Writing one from memory would violate the transcription discipline, "
                   "therefore no value was entered.",
        },
        {
            "id": "C2",
            "requirement": "同口径换算脚本落盘且可复算（复算残差 <1e-9）",
            "status": "PASS_PIPELINE_DRY_RUN_ONLY",
            "measured": {
                "script": "benchmarks/anchors/springing_whipping_hydroelastic_anchor.py",
                "round_trip_max_relative_residual": checks[1]["max_relative_residual"],
                "moment_closure_relative_residual": checks[2]["m0_relative_residual"],
                "unit_system_invariance_residual": checks[3]["relative_residual"],
                "tolerance": 1e-9,
            },
            "verdict": "PARTIAL — machinery reproducible to <1e-9 on the synthetic fixture; "
                       "the anchor-specific conversion cannot be executed (blocked by C1)",
            "evidence": "machine check CHK2/CHK3/CHK4 in this artifact",
            "gap": "the chain is parameterised on l_ref and on the spectrum normalisation, both of "
                   "which the source text must supply; only the fixture path is exercised",
        },
        {
            "id": "C3",
            "requirement": "量纲齐次与内部一致性机检通过",
            "status": "PASS_PIPELINE_DRY_RUN_ONLY",
            "measured": {
                "dimensional_homogeneity_steps": f"{checks[0]['n_homogeneous']}/{checks[0]['n_steps']}",
                "time_domain_closure_relative_residual": checks[5]["relative_residual_orthogonal"],
                "random_phase_mean_residual": checks[5]["relative_residual_random_phase_mean"],
                "scale_exponent_residual": checks[4]["relative_residual"],
                "all_checks_pass": all_checks_pass,
            },
            "verdict": "PASS on the chain itself (dims + closure + invariance), "
                       "no anchor number is validated by it",
            "evidence": "machine checks CHK1, CHK3, CHK4, CHK5, CHK6",
            "gap": "internal consistency constrains the transfer, it does not verify agreement with "
                   "the external anchor",
        },
        {
            "id": "C4",
            "requirement": "未核候选登记值不得直接引用（只能作为核对对照）",
            "status": "PASS",
            "measured": {
                "claim_eligible_entries": len(eligible),
                "claim_ineligible_entries": len(entries) - len(eligible),
                "candidate_values_recorded_from_memory": 0,
            },
            "verdict": "PASS — structurally enforced by classify_entries(); registry holds zero "
                       "numeric candidates, so nothing could be cited even accidentally",
            "evidence": "promotion_rule in the transcription registry + CHK guard test in "
                        "tests/test_springing_whipping_anchor.py",
            "gap": None,
        },
    ]

    domain_comparison = {
        "status": "NOT_PERFORMED",
        "reason": "the research question's second half (本课题领域实验结果 vs 锚) presupposes a "
                  "verified anchor; with C1 WITHHELD any 'difference to the anchor' number would be "
                  "a difference to an unverified candidate. Left explicitly uncomputed.",
    }

    return {
        "schema_version": "1.0.0",
        "task_key": TASK_KEY,
        "topic": TOPIC,
        "experiment": "research_plan item 1 — external anchor reconciliation (verbatim transcription "
                      "+ same-caliber transfer + consistency machine check)",
        "repository": "TensorFEM",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git("rev-parse", "HEAD"),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": {
            "script": "benchmarks/anchors/springing_whipping_hydroelastic_anchor.py",
            "script_sha256": _sha256(Path(__file__)),
            "command": "python benchmarks/anchors/springing_whipping_hydroelastic_anchor.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tensorfem": tf_version,
        },
        "case_caliber": CASE,
        "source_registry": {
            "path": "benchmarks/anchors/sources/sname_tr_bulletin_3-37_4-7.transcription.json",
            "sha256": _sha256(TRANSCRIPTION_PATH),
            "sources_sought": registry.get("sources_sought", []),
            "entry_classification": entries,
            "claim_eligible_entries": len(eligible),
        },
        "conversion_chain": {
            "convention_status": "UNVERIFIED — assumed for the dry run, must be resolved verbatim from the source",
            "assumed_convention": {
                "pressure_scale": "q = 0.5 * rho * V^2",
                "frequency_scale": "omega* = omega * l_ref / V",
                "spectrum_normalisation": "INT_0^inf S* domega* = Cp_rms^2",
                "reference_length_l_ref": "L (ship length) OR delta* (boundary-layer scale) — the "
                                          "source must state which; CHK7 quantifies the difference",
            },
            "steps": [
                {"id": s["id"], "formula": s["formula"], "unit": dstr(s["lhs_unit"]),
                 "note": s["note"]} for s in CHAIN_STEPS
            ],
            "closure_identity": "INT S_p domega = q^2 * Cp_rms^2 = p_rms^2",
        },
        "machine_checks": checks,
        "criteria": criteria,
        "dry_run_fixture": {
            "label": "SYNTHETIC — not a source value, not an anchor, comparison-only",
            "spectrum_shape": "S*(x) = A x^2 exp(-x^2)",
            "closed_form_moment": "A*sqrt(pi)/4",
            "cp_rms_star": FIXTURE_CP_RMS,
            "amplitude_A": FIXTURE_A,
            "l_ref_used_m": L,
            "derived_case_caliber_values": {
                "q_pa": 0.5 * RHO * V ** 2,
                "p_rms_pa": 0.5 * RHO * V ** 2 * FIXTURE_CP_RMS,
                "s_p_at_1hz_pa2_per_hz": transfer_chain(
                    FIXTURE_CP_RMS, fixture_shape_scalar, l_ref=L)["s_p_f"](1.0),
            },
            "claim_eligible": False,
        },
        "domain_comparison": domain_comparison,
        "verdict": {
            "overall": "FAIL",
            "kind": "honest FAIL — external anchor not reconciled",
            "why": "C1 WITHHELD: neither SNAME T&R Bulletin 3-37 (1997) nor 4-7 could be obtained in "
                   "this sandbox, so no verbatim quote / table / page could be transcribed. Criteria "
                   "C2 and C3 pass only as pipeline self-consistency on a labelled synthetic fixture; "
                   "C4 passes. The research question (how far the domain result sits from the anchor) "
                   "is therefore unanswered and explicitly not computed.",
            "what_would_close_it": [
                "operator supplies an authorised copy (or licensed access) of SNAME T&R Bulletin 3-37 (1997) and 4-7",
                "two independent transcribers fill entries[] with quote + table/page + printed values as strings",
                "resolve the nondimensionalisation block (pressure scale, frequency scale, reference length) verbatim",
                "re-run this script: C1 flips to PASS, the transfer executes on the real anchor, and only then may the anchor be compared with the domain result",
            ],
        },
        "limitations": [
            "The dry-run fixture coefficients are arbitrary; the numbers under dry_run_fixture carry no "
            "physical meaning and must not be quoted.",
            "The convention block (pressure scale / frequency scale / reference length) is assumed, not "
            "sourced; CHK7 shows the frequency axis moves by L/delta* ~ 320 when the reference length "
            "changes between the hull length and the boundary-layer scale, so an assumed convention is "
            "not good enough for a transfer.",
            "Dimensional homogeneity and moment closure validate the arithmetic of the chain only; they "
            "say nothing about agreement with the external anchor.",
            "delta* = 1.0 m and V = 7.9732 m/s are declared case inputs, not measured values of a "
            "specific ship.",
        ],
        "reproduce": {
            "generate": "python benchmarks/anchors/springing_whipping_hydroelastic_anchor.py",
            "gate": "python benchmarks/anchors/springing_whipping_hydroelastic_anchor.py --check",
            "tests": "pytest tests/test_springing_whipping_anchor.py -q",
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero unless every machine check passes and the verdict "
                             "does not over-claim")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)

    artifact = build_artifact()
    Path(args.out).write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")

    bad = [c["id"] for c in artifact["machine_checks"] if not c["passed"]]
    print(f"artifact: {args.out}")
    print(f"machine checks: {len(artifact['machine_checks']) - len(bad)}/"
          f"{len(artifact['machine_checks'])} passed" + (f" (failed: {bad})" if bad else ""))
    print(f"claim-eligible anchors: {artifact['source_registry']['claim_eligible_entries']}")
    print(f"verdict: {artifact['verdict']['overall']} — {artifact['verdict']['kind']}")
    if args.check:
        if bad:
            return 1
        # never allow a growing artifact to declare a pass it did not earn
        if artifact["criteria"][0]["status"] != "WITHHELD" and \
                artifact["source_registry"]["claim_eligible_entries"] == 0:
            print("error: C1 must stay WITHHELD while no verbatim entry exists", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())