"""External-anchor audit kernel for the research topic ``hull-vibration-analytic-kernel``.

Task key: ``paper-exp-hull-vibration-analytic-kernel-1``
(mission 7501; research_plan item 1: external anchor verbatim transcription +
same-口径 conversion + machine consistency checks).

What this script does (no number in the JSON artifact is hand-typed):

1. **Candidate registry** -- every value that this ecosystem currently attributes
   to ISO 6954:2000 is collected with ``file:line`` provenance.  A record only
   becomes a verified anchor when it carries the four citation pillars required
   by the traceability discipline: verbatim quote, table/clause identifier, page,
   and the value string *as printed*.  Promotion must pass :func:`promotion_gate`,
   which raises :class:`AnchorNotVerifiable` -- the machine embodiment of the rule
   "未核候选登记值不得直接引用（只能作为核对对照）".
2. **Same-口径 conversion chain** -- the standard's quantity is a frequency
   weighted r.m.s. *velocity* (mm/s, 1-80 Hz); the case kernel
   (:mod:`tensorfem.hull_girder_vibration_chain`) reports a weighted r.m.s.
   *acceleration* at the blade frequency.  The chain ``a = 2*pi*f*v`` is built as
   a dimensional operator, every chain is checked for SI base homogeneity, and the
   round-trip recomputation residual is gated at ``1e-9``.
3. **Machine checks** -- independent trapezoid integration, weighting-segment
   continuity, applicability domain of the single-frequency conversion, and the
   cross-registration consistency of the two candidate ladders once both are put
   into one 口径.
4. **Artifact** -- ``benchmarks/anchors/hull-vibration-analytic-kernel.json`` plus
   three figures.

Run from the repository root::

    python benchmarks/anchors/hull_vibration_anchor.py

Honesty contract: when the licensed standard text is unavailable, criterion C1 is
reported ``WITHHELD`` (verdict ``FAIL``) and the candidate values stay unusable for
any compliance claim; they serve only as cross-checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:  # pragma: no cover - path bootstrap
    sys.path.insert(0, str(REPO / "src"))

from tensorfem.hull_girder_vibration_chain import (  # noqa: E402
    integrate_spectrum,
    iso6954_weighted_rms,
    run_hull_girder_vibration_chain,
)

ARTIFACT = HERE / "hull-vibration-analytic-kernel.json"
FIGDIR = HERE / "figures"

TASK_KEY = "paper-exp-hull-vibration-analytic-kernel-1"
MISSION_ID = 7501
TOPIC = "hull-vibration-analytic-kernel"
STANDARD = "ISO 6954:2000"
STANDARD_TITLE = (
    "Mechanical vibration - Guidelines for the measurement, reporting and "
    "evaluation of vibration with regard to habitability on passenger and "
    "merchant ships"
)

# gates declared before any number is inspected
GATE_RECONCILE_RESIDUAL = 1e-9
GATE_MIN_VERIFIED_RECORDS = 1
GATE_REGISTRATION_CONSISTENCY = 1e-2
GATE_DIM_VIOLATIONS = 0

# ---------------------------------------------------------------------------
# SI base dimension algebra (length, time)
# ---------------------------------------------------------------------------
VELOCITY = (1, -1)
ACCELERATION = (1, -2)
FREQUENCY = (0, -1)
ACCEL_PSD = (2, -4)
DIMENSIONLESS = (0, 0)


def dim_mul(a, b):
    return (a[0] + b[0], a[1] + b[1])


def dim_div(a, b):
    return (a[0] - b[0], a[1] - b[1])


def dim_pow(a, n):
    return (a[0] * n, a[1] * n)


def dim_repr(d):
    return f"L^{d[0]} T^{d[1]}"


# ---------------------------------------------------------------------------
# 1. Candidate registry
# ---------------------------------------------------------------------------
@dataclass
class CandidateRecord:
    key: str
    repo: str
    path: str
    line: int
    quantity: str
    unit: str
    ladder: dict
    note: str = ""
    citation_quote: str | None = None
    table_or_clause: str | None = None
    page: str | None = None
    printed_values: dict | None = None

    def missing_pillars(self):
        missing = []
        if not (self.citation_quote or "").strip():
            missing.append("citation_quote")
        if not (self.table_or_clause or "").strip():
            missing.append("table_or_clause")
        if not (self.page or "").strip():
            missing.append("page")
        if not (self.printed_values or {}):
            missing.append("printed_values")
        return missing

    @property
    def verified(self):
        return not self.missing_pillars()

    def as_dict(self):
        return {
            "key": self.key,
            "provenance": f"{self.repo}:{self.path}:{self.line}",
            "claimed_quantity": self.quantity,
            "unit": self.unit,
            "ladder": {k: float(v) for k, v in self.ladder.items()},
            "citation_quote": self.citation_quote,
            "table_or_clause": self.table_or_clause,
            "page": self.page,
            "printed_values": self.printed_values,
            "missing_citation_pillars": self.missing_pillars(),
            "status": "VERIFIED_ANCHOR" if self.verified else "UNCITED_CANDIDATE",
            "usable_for_compliance_claim": bool(self.verified),
            "note": self.note,
        }


class AnchorNotVerifiable(RuntimeError):
    """Raised when an uncited candidate is pushed towards a compliance claim."""


def promotion_gate(record):
    missing = record.missing_pillars()
    if missing:
        raise AnchorNotVerifiable(f"{record.key} cannot be promoted: missing {missing}")
    return record


def candidate_registry():
    return [
        CandidateRecord(
            key="tensorfem_accel_ladder",
            repo="TensorFEM",
            path="src/tensorfem/hull_girder_vibration_chain.py",
            line=484,
            quantity="acceleration",
            unit="m/s^2 (frequency-weighted r.m.s.)",
            ladder={"A": 0.2, "B": 0.4, "C": 0.8},
            note=(
                "iso6954_grade() attributes this acceleration ladder to "
                "ISO 6954:2000; the module header quotes the 1984-edition title and "
                "carries no table number, no page and no printed value string."
            ),
        ),
        CandidateRecord(
            key="tensorfem_weighting",
            repo="TensorFEM",
            path="src/tensorfem/hull_girder_vibration_chain.py",
            line=429,
            quantity="acceleration",
            unit="dimensionless weighting ordinate",
            ladder={"plateau_value": 0.5},
            note=(
                "iso6954_weighted_rms(weighting='Wk') uses a piecewise band "
                "approximation with a 0.5 plateau over 4-8 Hz; the ordinates are "
                "not traceable to a cited table."
            ),
        ),
        CandidateRecord(
            key="ship_platform_velocity_ladder",
            repo="ship-performance-platform",
            path="backend/modules/ice_operability/iso2631.py",
            line=67,
            quantity="velocity",
            unit="mm/s (frequency-weighted r.m.s.)",
            ladder={"A": 4.0, "B": 6.0, "C": 8.0},
            note=(
                "ISO6954_ZONES attributes 4/6/8 mm/s to zones A/B/C of the same "
                "standard; no quote, no table number, no page."
            ),
        ),
        CandidateRecord(
            key="kb_chapter_quantity_claim",
            repo="ship-performance-platform",
            path=(
                "knowledge-center/books/chapters/"
                "船舶总体性能与AI赋能框架_第九章_振动噪声与智能控制.md"
            ),
            line=47,
            quantity="acceleration",
            unit="unspecified r.m.s.",
            ladder={},
            note=(
                "Internal knowledge-base chapter states the standard limits the "
                "r.m.s. of the vibration acceleration -- a third, mutually "
                "inconsistent statement of the quantity in scope; no citation."
            ),
        ),
    ]


def acquisition_attempts():
    return [
        {
            "target": "licensed standard text inside the working environment",
            "method": "filesystem scan of the worktree, benchmarks/, docs/ and the "
                      "ship-performance-platform knowledge-center",
            "outcome": "NOT_FOUND",
            "evidence": "no ISO 6954:2000 document exists here; the only hits are "
                        "the uncited code constants and prose in candidate_registry()",
        },
        {
            "target": "iso.org catalogue page",
            "method": "HTTPS GET https://www.iso.org/standard/23714.html",
            "outcome": "HTTP 403 (automated access refused)",
            "evidence": "no text retrieved; the catalogue page would only expose the "
                        "scope abstract, never the limit table",
        },
        {
            "target": "ISO Online Browsing Platform preview",
            "method": "review decision",
            "outcome": "NOT_ATTEMPTED",
            "evidence": "client-side rendered and, by design, withholds the tables; a "
                        "licensed copy is required",
        },
    ]


# ---------------------------------------------------------------------------
# 2. Conversion chain and operators
# ---------------------------------------------------------------------------
def candidate_weighting(freq):
    """Piecewise weighting of the TensorFEM kernel (mirrors source lines 429-482)."""
    f = np.asarray(freq, dtype=float)
    w = np.zeros_like(f)
    low = f < 4.0
    plateau = (f >= 4.0) & (f <= 8.0)
    mid = (f > 8.0) & (f <= 16.0)
    high = f > 16.0
    w[low] = 0.5 * f[low] / 4.0
    w[plateau] = 0.5
    w[mid] = 0.5 * 8.0 / f[mid]
    w[high] = 0.5 * 8.0 / 16.0 * 16.0 / f[high]
    return w


def trapezoid(y, x):
    """Independent trapezoid (deliberately not the repository helper)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def velocity_to_acceleration(v_mm_s, f_hz):
    """``a = 2*pi*f*v`` with mm->m; dimensional closure asserted."""
    v = np.asarray(v_mm_s, dtype=float) / 1000.0
    f = np.asarray(f_hz, dtype=float)
    chain = dim_mul(FREQUENCY, VELOCITY)
    assert chain == ACCELERATION, dim_repr(chain)
    return 2.0 * math.pi * f * v


def weighted_acceleration_rms(freq, accel_psd):
    f = np.asarray(freq, dtype=float)
    s = np.asarray(accel_psd, dtype=float)
    return math.sqrt(trapezoid(candidate_weighting(f) ** 2 * s, f))


def weighted_velocity_rms(freq, accel_psd):
    """``v_w^2 = integral (W/(2 pi f))^2 S_aa df`` -> mm/s."""
    f = np.asarray(freq, dtype=float)
    s = np.asarray(accel_psd, dtype=float)
    w = candidate_weighting(f)
    integrand = (w / (2.0 * math.pi * f)) ** 2 * s
    assert dim_mul(dim_pow(dim_div(DIMENSIONLESS, FREQUENCY), 2), ACCEL_PSD) == dim_pow(VELOCITY, 2)
    return 1000.0 * math.sqrt(trapezoid(integrand, f))


def tone_round_trip_residual(freqs):
    rows, worst = [], 0.0
    for f in np.asarray(freqs, dtype=float):
        a = 1.0e-2
        v_mm_s = a / (2.0 * math.pi * f) * 1000.0
        a_back = float(velocity_to_acceleration(v_mm_s, f))
        residual = abs(a_back - a) / a
        worst = max(worst, residual)
        rows.append({"f_hz": float(f), "a_rms_m_s2": a, "v_rms_mm_s": float(v_mm_s),
                     "a_back_m_s2": a_back, "residual": residual})
    return worst, rows


def bandwidth_validity(f_centre, rel_half_widths):
    """Error of the single-frequency conversion on a finite brick-wall band."""
    rows = []
    for r in rel_half_widths:
        f = np.linspace(f_centre * (1.0 - r), f_centre * (1.0 + r), 20001)
        s = np.ones_like(f)
        a = weighted_acceleration_rms(f, s)
        v = weighted_velocity_rms(f, s) / 1000.0
        ratio = 2.0 * math.pi * f_centre * v / a
        rows.append({"relative_half_width": float(r), "ratio_R": float(ratio),
                     "relative_error": float(abs(ratio - 1.0))})
    return rows


def registration_consistency(freqs, accel_ladder, vel_ladder):
    f = np.asarray(freqs, dtype=float)
    rows, worst = [], 0.0
    for grade in ("A", "B", "C"):
        a_from_v = velocity_to_acceleration(vel_ladder[grade], f)
        a_ref = float(accel_ladder[grade])
        dev = np.abs(a_from_v - a_ref) / a_ref
        worst = max(worst, float(np.max(dev)))
        rows.append({
            "grade": grade,
            "acceleration_ladder_limit_m_s2": a_ref,
            "velocity_ladder_limit_mm_s": float(vel_ladder[grade]),
            "acceleration_equivalent_by_frequency": {
                f"{float(fi):.3f}": float(ai) for fi, ai in zip(f, a_from_v)},
            "deviation_at_each_frequency": {
                f"{float(fi):.3f}": float(di) for fi, di in zip(f, dev)},
            "max_relative_deviation": float(np.max(dev)),
        })
    return worst, rows


# ---------------------------------------------------------------------------
# 3. Machine checks
# ---------------------------------------------------------------------------
def check_dimensional_homogeneity():
    chains = {
        "velocity_to_acceleration": dim_mul(FREQUENCY, VELOCITY),
        "acceleration_to_velocity": dim_div(ACCELERATION, FREQUENCY),
        "velocity_integrand (W/(2 pi f))^2 S_aa":
            dim_mul(dim_pow(dim_div(DIMENSIONLESS, FREQUENCY), 2), ACCEL_PSD),
        "acceleration_integrand W^2 S_aa": ACCEL_PSD,
        "mm_to_m_scale": DIMENSIONLESS,
        "sqrt_velocity_integrand": VELOCITY,
    }
    expected = {
        "velocity_to_acceleration": ACCELERATION,
        "acceleration_to_velocity": VELOCITY,
        "velocity_integrand (W/(2 pi f))^2 S_aa": dim_pow(VELOCITY, 2),
        "acceleration_integrand W^2 S_aa": dim_pow(ACCELERATION, 2),
        "mm_to_m_scale": DIMENSIONLESS,
        "sqrt_velocity_integrand": VELOCITY,
    }
    block = {}
    for name, d in chains.items():
        block[name] = {"derived_dimension": dim_repr(d),
                       "required_dimension": dim_repr(expected[name]),
                       "closed": d == expected[name]}
    violations = [n for n, d in chains.items() if d != expected[n]]
    return {"chains": block, "violations": violations, "n_violations": len(violations),
            "status": "PASS" if len(violations) <= GATE_DIM_VIOLATIONS else "FAIL"}


def check_weighting_continuity():
    eps = 1e-9
    jumps = []
    for b in (4.0, 8.0, 16.0):
        left = float(candidate_weighting(np.array([b - eps]))[0])
        right = float(candidate_weighting(np.array([b + eps]))[0])
        jumps.append({"breakpoint_hz": b, "left": left, "right": right,
                      "abs_jump": abs(right - left)})
    worst = max(j["abs_jump"] for j in jumps)
    return {"breakpoints": jumps, "max_abs_jump": worst, "threshold": 1e-6,
            "status": "PASS" if worst <= 1e-6 else "FAIL",
            "observation": "segments join continuously, but the ordinates (0.5 "
                           "plateau, 12 dB/oct skirts) are not traceable to a cited "
                           "table"}


def check_weighting_replication(freq, psd):
    a_repo = float(iso6954_weighted_rms(freq, psd, weighting="Wk"))
    a_local = weighted_acceleration_rms(freq, psd)
    res = abs(a_local - a_repo) / max(abs(a_repo), 1e-300)
    return {"a_w_repository_m_s2": a_repo, "a_w_local_operator_m_s2": a_local,
            "relative_residual": res, "threshold": GATE_RECONCILE_RESIDUAL,
            "status": "PASS" if res <= GATE_RECONCILE_RESIDUAL else "FAIL"}


def check_integration_independence(freq, psd):
    y = candidate_weighting(freq) ** 2 * psd
    a_repo = math.sqrt(float(integrate_spectrum(y, freq)))
    a_local = math.sqrt(trapezoid(y, freq))
    res = abs(a_local - a_repo) / max(a_repo, 1e-300)
    return {"value_repository_integrator": a_repo, "value_independent_trapezoid": a_local,
            "relative_residual": res, "threshold": GATE_RECONCILE_RESIDUAL,
            "status": "PASS" if res <= GATE_RECONCILE_RESIDUAL else "FAIL"}


def value_string_fidelity_check(value_strings):
    rows, ok = [], True
    for s in value_strings:
        as_float = float(s)
        back = repr(as_float)
        same = float(back) == as_float
        ok = ok and same
        rows.append({"printed": s, "as_float": as_float, "round_trip": back,
                     "fidelity_ok": same})
    return {"rows": rows, "status": "PASS" if ok else "FAIL",
            "n_strings_checked": len(rows),
            "note": "applies to values transcribed from a licensed source; the current "
                    "registry holds no printed string, so the check is vacuous and is "
                    "reported as such"}


# ---------------------------------------------------------------------------
# 4. Case 口径 (repository kernel output, nothing invented)
# ---------------------------------------------------------------------------
def case_state():
    r = run_hull_girder_vibration_chain(n_elements=20)
    return {
        "source": "src/tensorfem/hull_girder_vibration_chain.py::"
                  "run_hull_girder_vibration_chain (defaults, n_elements=20)",
        "ship": {"L_m": 200.0, "B_m": 32.0, "D_m": 16.0, "Cb": 0.65, "T_m": 8.0},
        "propeller": {"n_blades": 5, "rpm": 90.0, "f_blade_hz": float(r.blade_freq)},
        "natural_frequencies_hz": [float(x) for x in r.natural_frequencies[:5]],
        "case_quantity": {"a_w_m_s2": float(r.weighted_rms_accel),
                          "unit": "m/s^2 (frequency-weighted r.m.s. acceleration, "
                                  "case kernel output)",
                          "kernel_grade_under_candidate_accel_ladder": str(r.iso_grade)},
        "propeller_bandwidth_hz": 0.5,
        "relative_half_width": 0.5 / float(r.blade_freq),
    }


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
def make_figures(case, candidates, validity, consistency_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs = []
    accel = next(c for c in candidates if c.key == "tensorfem_accel_ladder")
    vel = next(c for c in candidates if c.key == "ship_platform_velocity_ladder")
    f_b = case["propeller"]["f_blade_hz"]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    f = np.logspace(0.0, np.log10(80.0), 300)
    for grade, colour in (("A", "tab:green"), ("B", "tab:orange"), ("C", "tab:red")):
        ax.plot(f, velocity_to_acceleration(vel.ladder[grade], f), color=colour, lw=2.0,
                label=f"velocity ladder {grade} = {vel.ladder[grade]:g} mm/s (ship-platform)")
        ax.axhline(accel.ladder[grade], color=colour, ls="--", lw=1.2,
                   label=f"acceleration ladder {grade} = {accel.ladder[grade]:g} m/s^2 (TensorFEM)")
    ax.axvline(f_b, color="k", ls=":", lw=1.0)
    ax.text(f_b * 1.06, 3e-3, f"case blade freq\n{f_b:g} Hz", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("limit as weighted r.m.s. acceleration [m/s$^2$]")
    ax.set_title("ISO 6954 candidate registrations converted to one unit (uncited, cross-check only)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.0, loc="upper left")
    p1 = FIGDIR / "fig1_candidate_ladders_one_kou_jing.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    figs.append({"name": p1.name, "caption":
                 "Both uncited candidate ladders expressed as acceleration. They "
                 "coincide near 8 Hz and diverge by an order of magnitude at the band "
                 "edges, so the anchor is unusable until verbatim transcription."})

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    r = [row["relative_half_width"] for row in validity]
    e = [max(row["relative_error"], 1e-16) for row in validity]
    ax.loglog(r, e, "o-", label="measured ratio error")
    ax.loglog(r, np.array(r) ** 2, "k--", lw=1.0, label=r"reference slope $r^2$")
    ax.axvline(case["relative_half_width"], color="tab:red", ls=":",
               label=f"case band r = {case['relative_half_width']:.3f}")
    ax.set_xlabel(r"relative half width of the band, $r=\sigma_f/f_c$")
    ax.set_ylabel(r"relative error of $2\pi f_c v_w/a_w-1$")
    ax.set_title("Applicability domain of the single-frequency v<->a conversion")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p2 = FIGDIR / "fig2_conversion_validity_domain.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    figs.append({"name": p2.name, "caption":
                 "Conversion error grows as the square of the relative bandwidth; at the "
                 "case bandwidth (r=0.067) the single-frequency conversion is good to "
                 "well below 1 %."})

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for row, colour in zip(consistency_rows, ("tab:green", "tab:orange", "tab:red")):
        keys = np.array([float(k) for k in row["deviation_at_each_frequency"]])
        vals = np.array(list(row["deviation_at_each_frequency"].values()))
        ax.loglog(keys, np.maximum(vals, 1e-16), "o-", color=colour,
                  label=f"grade {row['grade']}")
    ax.axhline(GATE_REGISTRATION_CONSISTENCY, color="k", ls="--", lw=1.0, label="gate 1 %")
    ax.set_xlabel("frequency at which the velocity ladder is converted [Hz]")
    ax.set_ylabel("relative deviation from the acceleration ladder")
    ax.set_title("Cross-registration consistency of the two candidate ladders")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p3 = FIGDIR / "fig3_cross_registration_deviation.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    figs.append({"name": p3.name, "caption":
                 "Deviation between the two registrations after the 口径 conversion. The "
                 "1 % gate holds only in a narrow window around the blade frequency, so "
                 "the registrations are not interchangeable."})
    return figs


# ---------------------------------------------------------------------------
# 6. Artifact assembly
# ---------------------------------------------------------------------------
def build_artifact(write=True, figures=True):
    candidates = candidate_registry()
    case = case_state()

    ref_freqs = np.array([1.0, case["natural_frequencies_hz"][0],
                          case["propeller"]["f_blade_hz"],
                          case["natural_frequencies_hz"][1], 80.0])

    dim_check = check_dimensional_homogeneity()
    weighting_continuity = check_weighting_continuity()

    f_syn = np.linspace(1.0, 80.0, 20001)
    s_syn = np.exp(-0.5 * ((f_syn - case["propeller"]["f_blade_hz"]) / 0.25) ** 2)
    weighting_rep = check_weighting_replication(f_syn, s_syn)
    integration_indep = check_integration_independence(f_syn, s_syn)

    worst_round_trip, round_trip_rows = tone_round_trip_residual(ref_freqs)
    validity = bandwidth_validity(case["propeller"]["f_blade_hz"],
                                  [0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    case_r = case["relative_half_width"]
    case_validity_err = float(np.interp(case_r,
                                        [v["relative_half_width"] for v in validity],
                                        [v["relative_error"] for v in validity]))

    accel_rec = next(c for c in candidates if c.key == "tensorfem_accel_ladder")
    vel_rec = next(c for c in candidates if c.key == "ship_platform_velocity_ladder")
    worst_consistency, consistency_rows = registration_consistency(
        ref_freqs, accel_rec.ladder, vel_rec.ladder)

    gate_results = {}
    for rec in candidates:
        try:
            promotion_gate(rec)
            gate_results[rec.key] = "PROMOTED"
        except AnchorNotVerifiable as exc:
            gate_results[rec.key] = f"REFUSED: {exc}"

    verified = [c for c in candidates if c.verified]
    fidelity = value_string_fidelity_check(
        [s for c in verified for s in (c.printed_values or {}).values()])

    f_b = case["propeller"]["f_blade_hz"]
    a_w = case["case_quantity"]["a_w_m_s2"]
    v_w = a_w / (2.0 * math.pi * f_b) * 1000.0
    margin_accel = accel_rec.ladder["A"] / a_w
    margin_velocity = vel_rec.ladder["A"] / v_w

    criteria = [
        {
            "id": "C1",
            "claim": "verbatim transcription of the frequency-weighted r.m.s. velocity "
                     "limit curve with quote + table/page + printed value",
            "threshold": f">= {GATE_MIN_VERIFIED_RECORDS} verified record",
            "measured": len(verified),
            "unit": "verified records",
            "status": "WITHHELD" if not verified else "PASS",
            "verdict": "FAIL" if not verified else "PASS",
            "evidence": {
                "verified_records": [c.as_dict() for c in verified],
                "candidate_records": [c.as_dict() for c in candidates],
                "acquisition_attempts": acquisition_attempts(),
            },
        },
        {
            "id": "C2",
            "claim": "same-口径 conversion chain is recomputable: tonal round trip "
                     "a -> v -> a closes below the residual gate",
            "threshold": f"<= {GATE_RECONCILE_RESIDUAL:g}",
            "measured": float(worst_round_trip),
            "unit": "relative residual",
            "status": "PASS" if worst_round_trip <= GATE_RECONCILE_RESIDUAL else "FAIL",
            "evidence": {"frequency_sweep_hz": [float(x) for x in ref_freqs],
                         "rows": round_trip_rows,
                         "formula": "v[mm/s] = a/(2*pi*f)*1000 ; a = 2*pi*f*v",
                         "residual_note": "the round trip is evaluated with the same "
                                          "multiplicative pair, so the residual is exactly "
                                          "zero in IEEE-754 arithmetic; the gate is still "
                                          "meaningful because it also fixes the operator "
                                          "and the unit scale"},
        },
        {
            "id": "C3",
            "claim": "dimensional homogeneity of every link in the conversion chain",
            "threshold": f"n_violations <= {GATE_DIM_VIOLATIONS}",
            "measured": dim_check["n_violations"],
            "unit": "violations",
            "status": dim_check["status"],
            "evidence": dim_check,
        },
        {
            "id": "C4",
            "claim": "cross-registration consistency: both candidate ladders agree to "
                     "1 % once converted into one 口径 over 1-80 Hz",
            "threshold": f"<= {GATE_REGISTRATION_CONSISTENCY:g} relative",
            "measured": float(worst_consistency),
            "unit": "relative deviation",
            "status": "PASS" if worst_consistency <= GATE_REGISTRATION_CONSISTENCY else "FAIL",
            "evidence": {"rows": consistency_rows,
                         "interpretation": "the two registrations describe different "
                                           "quantities; at least one is wrong, and "
                                           "without the licensed text neither can be "
                                           "used for a compliance claim"},
        },
        {
            "id": "C5",
            "claim": "uncited candidate values can never be promoted to a benchmark "
                     "reference (promotion gate refuses)",
            "threshold": "all uncited records refused",
            "measured": sum(1 for v in gate_results.values() if v.startswith("REFUSED")),
            "unit": "refused records",
            "status": "PASS" if all(v.startswith("REFUSED") for v in gate_results.values())
                      else "FAIL",
            "evidence": gate_results,
        },
        {
            "id": "C6",
            "claim": "numeric operators are implementation-independent (repository "
                     "integrator vs independent trapezoid, local vs repository weighting)",
            "threshold": f"<= {GATE_RECONCILE_RESIDUAL:g}",
            "measured": max(weighting_rep["relative_residual"],
                            integration_indep["relative_residual"]),
            "unit": "relative residual",
            "status": "PASS" if max(weighting_rep["relative_residual"],
                                    integration_indep["relative_residual"])
                               <= GATE_RECONCILE_RESIDUAL else "FAIL",
            "evidence": {"weighting_replication": weighting_rep,
                         "integration_independence": integration_indep,
                         "spectrum": "synthetic Gaussian band centred on the blade "
                                     "frequency, sigma = 0.25 Hz (operator validation, "
                                     "not a case result)",
                         "residual_note": "the repository integrator and the independent "
                                          "trapezoid reduce to the same floating-point "
                                          "sequence on a uniform grid, hence exact zeros; "
                                          "the check still catches grid/weighting "
                                          "mismatches"},
        },
        {
            "id": "C7",
            "claim": "case quantity expressed in the standard's 口径 with an explicit "
                     "applicability bound",
            "threshold": "validity error <= 1 % at the case bandwidth",
            "measured": case_validity_err,
            "unit": "relative error",
            "status": "PASS" if case_validity_err <= GATE_REGISTRATION_CONSISTENCY else "FAIL",
            "evidence": {"case_a_w_m_s2": a_w, "case_v_w_mm_s": v_w,
                         "case_blade_freq_hz": f_b,
                         "relative_half_width": case_r,
                         "bandwidth_validity_rows": validity,
                         "margin_under_accel_ladder_A": margin_accel,
                         "margin_under_velocity_ladder_A": margin_velocity},
        },
        {
            "id": "C8",
            "claim": "printed value strings survive the transcription pipeline without "
                     "re-rounding",
            "threshold": "round trip exact",
            "measured": fidelity["n_strings_checked"],
            "unit": "strings checked",
            "status": fidelity["status"],
            "evidence": fidelity,
        },
    ]

    overall = "FAIL" if any(c["verdict" if "verdict" in c else "status"] == "FAIL"
                            for c in criteria) else "PASS"
    anchor_status = "VERIFIED_ANCHOR" if verified else "WITHHELD"

    payload = {
        "benchmark": "anchors/hull-vibration-analytic-kernel",
        "task_key": TASK_KEY,
        "mission_id": MISSION_ID,
        "topic": TOPIC,
        "repository": "TensorFEM",
        "generated_by": "benchmarks/anchors/hull_vibration_anchor.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "external_anchor": {
            "standard": STANDARD,
            "title": STANDARD_TITLE,
            "quantity_in_scope_per_mission_brief": "frequency-weighted r.m.s. velocity, "
                                                   "mm/s, over the 1-80 Hz band",
            "quantity_corroboration": "the mission brief states the quantity is a "
                                      "frequency-weighted r.m.s. velocity; this is a "
                                      "platform-provided claim, NOT yet corroborated by "
                                      "the licensed text",
            "verification_status": anchor_status,
            "n_verified_records": len(verified),
            "candidate_registrations": [c.as_dict() for c in candidates],
            "acquisition_attempts": acquisition_attempts(),
            "candidate_values_used_as_reference": [],
            "candidate_values_role": "cross-check only; per the task rule no uncited "
                                     "candidate value is cited as a benchmark or used in "
                                     "any pass/fail decision",
            "conclusion": "the external anchor is NOT yet closed: no licensed copy of the "
                          "standard exists in this environment, so no verbatim quote / "
                          "table number / page can be recorded, and the four candidate "
                          "registrations in the ecosystem stay uncited and mutually "
                          "inconsistent (acceleration vs velocity; 0.2 m/s^2 vs 4 mm/s).",
        },
        "case_conventions": case,
        "case_value_in_standard_kou_jing": {
            "a_w_m_s2": a_w,
            "f_blade_hz": f_b,
            "v_w_mm_s_single_frequency_conversion":
                v_w,
            "conversion": "v_w[mm/s] = a_w/(2*pi*f_b)*1000 (valid for a narrowband "
                          "component; error = table bandwidth_validity_rows)",
            "applicability_error_at_case_bandwidth": case_validity_err,
            "margin_to_candidate_A_accel": margin_accel,
            "margin_to_candidate_A_velocity": margin_velocity,
            "note": "the value is derived from the case kernel output; it is NOT compared "
                    "against a verified limit because none exists yet",
        },
        "conversion_chain": {
            "steps": [
                {"step": 1, "from": "weighted r.m.s. velocity limit [mm/s]",
                 "to": "r.m.s. velocity [m/s]",
                 "operation": "divide by 1000 (dimensionless scale)",
                 "dimension_in": dim_repr(VELOCITY), "dimension_out": dim_repr(VELOCITY)},
                {"step": 2, "from": "r.m.s. velocity at f", "to": "r.m.s. acceleration at f",
                 "operation": "multiply by 2*pi*f",
                 "dimension_in": dim_repr(dim_mul(FREQUENCY, VELOCITY)),
                 "dimension_out": dim_repr(ACCELERATION)},
                {"step": 3, "from": "case weighted r.m.s. acceleration",
                 "to": "velocity 口径", "operation": "divide by 2*pi*f_blade",
                 "dimension_in": dim_repr(ACCELERATION),
                 "dimension_out": dim_repr(VELOCITY)},
            ],
            "dimensional_check": dim_check,
            "recompute_residual": float(worst_round_trip),
        },
        "machine_checks": {
            "dimensional_homogeneity": dim_check,
            "weighting_continuity": weighting_continuity,
            "weighting_replication": weighting_rep,
            "integration_independence": integration_indep,
            "conversion_bandwidth_validity": validity,
            "cross_registration_deviation_max": float(worst_consistency),
        },
        "criteria": criteria,
        "overall_status": overall,
        "gap_statement": (
            "Achieved: a recomputable same-口径 conversion chain (residual "
            f"{worst_round_trip:.2e} <= 1e-9), dimensional homogeneity on all links, an "
            "implementation-independence check of the numerics, a promotion gate that "
            "refuses uncited values, and the case value expressed in the standard's 口径 "
            f"(v_w = {v_w:.4f} mm/s at f = {f_b:g} Hz, conversion error <= "
            f"{case_validity_err * 100:.2f} % at the case bandwidth). "
            "Not achieved: the verbatim transcription of the limit curve. No licensed "
            "copy of ISO 6954:2000 is reachable from this environment, so the anchor "
            "stays WITHHELD and no error-vs-benchmark number can be claimed for the "
            "topic yet. The four candidate registrations already in the ecosystem are "
            "uncited and contradict each other (e.g. "
            f"max relative deviation {worst_consistency * 100:.1f} % after 口径 "
            "conversion), which is the quantified size of the risk the anchor would "
            "have removed."
        ),
        "stance": {
            "mission_brief_basis": "mission 7501 research_plan item 1",
            "independent_anchor": None,
            "anchor_unverified_reason": "licensed standard text not available in the "
                                        "environment; only memory-derived internal "
                                        "constants exist",
            "comparison_basis": "internal cross-registration only (declared, not an "
                                "external benchmark)",
        },
        "figures": [],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "command": "python benchmarks/anchors/hull_vibration_anchor.py",
        },
    }

    if figures:
        payload["figures"] = make_figures(case, candidates, validity, consistency_rows)

    if write:
        ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return payload


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    payload = build_artifact(write=True, figures="--no-figures" not in argv)
    print(f"wrote {ARTIFACT}")
    print(f"anchor status: {payload['external_anchor']['verification_status']}")
    for c in payload["criteria"]:
        print(f"  {c['id']}: {c.get('verdict', c['status']):8s} measured={c['measured']}")
    print(f"overall: {payload['overall_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())