"""W23 anchor register: Cole (1948) underwater-shock environment.

Why this script exists
----------------------
The W23 protocol requires the Cole peak-pressure law ``p_m = K (W^(1/3)/R)^A`` and
the bubble-pulsation period to be closed out against an **external anchor**.  An
anchor is only usable when its printed values are transcribed from an authorised
copy of the source document (quote, table/figure/page locator, printed value
strings).  This environment does not hold such a copy, and the sandbox blocks
outbound network access (both attempts are recorded verbatim below as they were
actually run).  The register therefore

* records the ``acquisition_status`` as ``NOT_OBTAINED`` and every acquisition
  attempt with its actual outcome, instead of filling in remembered numbers;
* records the parameters that *are* used by the chain as
  ``repo_history_implementation_parameter`` (they exist as declared
  implementation parameters in this repository at a named commit) or as
  ``uncited_candidate`` (no transcription, not usable for a validation claim);
* records the reference values of the case matrix from a **standalone**
  implementation (no import of ``tensorfem``), evaluated in two independent
  algebraic forms so that the chain's recomputation can be checked against it at
  the 1 % level -- that check is *transcription/implementation fidelity*, never a
  validation of the coefficient values.

Run ``python scripts/w23_cole_anchor_record.py`` to (re)write
``benchmarks/undex_shock/cole_anchor.json``.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "benchmarks" / "undex_shock" / "cole_anchor.json"

#: Case matrix of the ensemble.  W [kg TNT], R [m standoff], H [m depth].
#: Reduced ranges W^(1/3)/R are kept in the 0.10-0.16 band where the far-field
#: Cole fit is the intended description.
CASES = [
    {"case_id": "C1", "weight_kg": 5.0, "range_m": 12.0, "depth_m": 40.0},
    {"case_id": "C2", "weight_kg": 10.0, "range_m": 18.0, "depth_m": 40.0},
    {"case_id": "C3", "weight_kg": 25.0, "range_m": 25.0, "depth_m": 40.0},
    {"case_id": "C4", "weight_kg": 50.0, "range_m": 30.0, "depth_m": 40.0},
]

#: Parameters used by the chain.  ``value`` is what the chain evaluates with;
#: ``provenance`` says where the number comes from and whether it may be used for
#: a validation claim.
PARAMETERS = {
    "K": {
        "symbol": "K",
        "value": 52.4e6,
        "unit": "Pa",
        "role": "Cole peak-pressure coefficient, metric TNT fit",
        "provenance": {
            "kind": "repo_history_implementation_parameter",
            "file": "src/tensorfem/undex_shock.py",
            "commit": "a706ec07315bf1c4c810f569171b19370dcad3d9",
            "quote": ("The Cole coefficients are implementation parameters for a TNT "
                      "charge (metric fit widely quoted for UNDEX work): K = 52.4 MPa, "
                      "A = 1.13, K_theta = 0.084 ms, decay exponent -0.24, W in kg, "
                      "R in m."),
            "usable_for_validation_claim": False,
        },
    },
    "A": {
        "symbol": "A",
        "value": 1.13,
        "unit": "-",
        "role": "Cole peak-pressure exponent, metric TNT fit",
        "provenance": {
            "kind": "repo_history_implementation_parameter",
            "file": "src/tensorfem/undex_shock.py",
            "commit": "a706ec07315bf1c4c810f569171b19370dcad3d9",
            "quote": "same declared implementation-parameter block (K = 52.4 MPa, A = 1.13)",
            "usable_for_validation_claim": False,
        },
    },
    "K_theta": {
        "symbol": "K_theta",
        "value": 0.084e-3,
        "unit": "s",
        "role": "Cole shock-decay time coefficient, theta = K_theta W^(1/3) (W^(1/3)/R)^n",
        "provenance": {
            "kind": "repo_history_implementation_parameter",
            "file": "src/tensorfem/undex_shock.py",
            "commit": "a706ec07315bf1c4c810f569171b19370dcad3d9",
            "quote": "K_theta = 0.084 ms, decay exponent -0.24",
            "usable_for_validation_claim": False,
        },
    },
    "decay_exponent": {
        "symbol": "n",
        "value": -0.24,
        "unit": "-",
        "role": "exponent of the reduced range in the Cole decay time",
        "provenance": {
            "kind": "repo_history_implementation_parameter",
            "file": "src/tensorfem/undex_shock.py",
            "commit": "a706ec07315bf1c4c810f569171b19370dcad3d9",
            "quote": "K_theta = 0.084 ms, decay exponent -0.24",
            "usable_for_validation_claim": False,
        },
    },
    "T_b": {
        "symbol": "K_b",
        "value": 2.11,
        "unit": "s * m^(5/6) / kg^(1/3)",
        "role": "prefactor of the bubble-pulsation period T = K_b W^(1/3)/(H + 10.33)^(5/6)",
        "provenance": {
            "kind": "uncited_candidate",
            "file": "src/tensorfem/undex_chain.py",
            "quote": ("UNCITED CANDIDATE: carried as a declared implementation "
                      "parameter, NOT transcribed from Cole (1948)."),
            "usable_for_validation_claim": False,
        },
    },
    "depth_offset": {
        "symbol": "H_0",
        "value": 10.33,
        "unit": "m",
        "role": "atmospheric head expressed as metres of water (p_atm / (rho g))",
        "provenance": {
            "kind": "physical_constant",
            "quote": ("p_atm = 101325 Pa, rho = 1025 kg/m^3, g = 9.80665 m/s^2 "
                      "-> 10.08 m; the 10.33 m figure is the fresh-water/standard "
                      "value used in the law's own convention"),
            "usable_for_validation_claim": False,
        },
    },
    "bubble_exponent": {
        "symbol": "m",
        "value": 5.0 / 6.0,
        "unit": "-",
        "role": "exponent of the hydrostatic head in the bubble-period law",
        "provenance": {
            "kind": "derived_coefficient_free",
            "derivation": ("Rayleigh collapse of a gas bubble: R0 ~ (E/p_h)^(1/3) and "
                           "t_c ~ R0 sqrt(rho/p_h) => t_c ~ W^(1/3) p_h^(-1/3-1/2) "
                           "= W^(1/3) p_h^(-5/6); machine-checked in "
                           "src/tensorfem/undex_chain.py::bubble_scaling_exponents"),
            "usable_for_validation_claim": True,
        },
    },
}

SEAWATER_DENSITY = 1025.0
SEAWATER_SOUND_SPEED = 1500.0


def _record_value_form_power(W: float, R: float, H: float) -> dict:
    """Standalone evaluation, power form (no tensorfem import)."""
    red = W ** (1.0 / 3.0) / R
    p_m = PARAMETERS["K"]["value"] * red ** PARAMETERS["A"]["value"]
    theta = (PARAMETERS["K_theta"]["value"] * W ** (1.0 / 3.0)
             * red ** PARAMETERS["decay_exponent"]["value"])
    T_b = (PARAMETERS["T_b"]["value"] * W ** (1.0 / 3.0)
           / (H + PARAMETERS["depth_offset"]["value"]) ** PARAMETERS["bubble_exponent"]["value"])
    return {"peak_pressure_pa": p_m, "decay_time_s": theta,
            "bubble_period_s": T_b, "specific_impulse_pa_s": p_m * theta}


def _record_value_form_exp(W: float, R: float, H: float) -> dict:
    """Standalone evaluation, exp/log form (algebraically independent of above)."""
    ln_red = math.log(W) / 3.0 - math.log(R)
    p_m = math.exp(math.log(PARAMETERS["K"]["value"])
                   + PARAMETERS["A"]["value"] * ln_red)
    theta = math.exp(math.log(PARAMETERS["K_theta"]["value"]) + math.log(W) / 3.0
                     + PARAMETERS["decay_exponent"]["value"] * ln_red)
    T_b = math.exp(math.log(PARAMETERS["T_b"]["value"]) + math.log(W) / 3.0
                   - PARAMETERS["bubble_exponent"]["value"]
                   * math.log(H + PARAMETERS["depth_offset"]["value"]))
    return {"peak_pressure_pa": p_m, "decay_time_s": theta,
            "bubble_period_s": T_b, "specific_impulse_pa_s": p_m * theta}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git",) + args, cwd=str(REPO), capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as exc:                       # pragma: no cover - diagnostics
        return f"<git failed: {exc}>"


def acquisition_attempts() -> list[dict]:
    """The acquisition channels tried, with the outcome each one actually had."""
    return [
        {
            "channel": "local filesystem corpus",
            "action": ("grep -rIl -i 'cole|undex|underwater explosion|bubble period|"
                       "Cole (1948)' /root /opt /srv /data --include=*.py --include=*.json "
                       "--include=*.md --include=*.csv --include=*.txt --include=*.tex "
                       "--include=*.bib (excluding this worktree)"),
            "result": ("only this repository's own UNDEX code (src/tensorfem/undex_shock.py, "
                       "present in this repository's history at commit a706ec07) and "
                       "unrelated wall-law scripts (compare_wall_law_coles.py in /tmp/stage). "
                       "No Cole (1948) text, table or page image."),
        },
        {
            "channel": "local PDF corpus",
            "action": ("find / -maxdepth 5 -iname '*.pdf' -not -path '/proc/*' -not -path "
                       "'/sys/*'  (bounded depth-5 sweep; the full sweep exceeded the "
                       "tool timeout at 120 s after listing 35 files)"),
            "result": ("35 PDFs, all tooling/platform manuals (SDAA programming guide, "
                       "hermes-kanban spec, one unrelated paper under /root/.worktrees). "
                       "No Cole monograph or UNDEX report."),
        },
        {
            "channel": "network",
            "action": ("timeout 12 curl -sS -o /dev/null -w '%{http_code}' "
                       "https://en.wikipedia.org/wiki/Underwater_explosion ; "
                       "timeout 10 curl -sS -o /dev/null -w '%{http_code}' "
                       "https://www.google.com"),
            "result": ("both rejected by the execution sandbox: "
                       "'REJECTED: 命令被安全策略拦截' (security policy). No outbound HTTP, "
                       "so no full text and no authorised copy either way."),
        },
        {
            "channel": "operator-supplied deposition",
            "action": "ls benchmarks/anchors/sources/",
            "result": ("only sname_tr_bulletin_3-37_4-7.transcription.json (different "
                       "topic, SNAME Transactions); no Cole deposition."),
        },
        {
            "channel": "repository history",
            "action": "git log --all --oneline -- '*undex*'",
            "result": ("a706ec07 'undex shock chain: Cole(1948) environment -> Taylor "
                       "air-backed plate FSI ...' whose src/tensorfem/undex_shock.py "
                       "declares K, A, K_theta, -0.24 as implementation parameters. "
                       "Self-declared, therefore not an anchor."),
        },
    ]


def main() -> dict:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    recorded = []
    for case in CASES:
        W, R, H = case["weight_kg"], case["range_m"], case["depth_m"]
        a = _record_value_form_power(W, R, H)
        b = _record_value_form_exp(W, R, H)
        worst = max(abs(a[k] - b[k]) / abs(a[k]) for k in a)
        red = W ** (1.0 / 3.0) / R
        flux = a["peak_pressure_pa"] ** 2 * a["decay_time_s"] / (
            2.0 * SEAWATER_DENSITY * SEAWATER_SOUND_SPEED)
        recorded.append({
            **case,
            "reduced_range": red,
            **a,
            "energy_flux_j_m2": flux,
            "shock_energy_j": 4.0 * math.pi * R * R * flux,
            "form_cross_check_max_rel": worst,
        })

    payload = {
        "schema_version": "1.0.0",
        "artifact": "benchmarks/undex_shock/cole_anchor",
        "topic": "w23_undex_shock_chain",
        "mission_id": 15095,
        "repository": "TensorFEM",
        "generated_by": "scripts/w23_cole_anchor_record.py",
        "script_sha256": _sha256(Path(__file__).resolve()),
        "git_commit_at_record": _git("rev-parse", "HEAD"),
        "document": {
            "citation_as_assigned": ("Cole, R.H., Underwater Explosions, Princeton "
                                     "University Press, 1948 (anchor AN-COLE)"),
            "subject_quantity": ("peak shock overpressure p_m = K (W^(1/3)/R)^A and the "
                                 "bubble-pulsation period T (K, A charge-dependent "
                                 "implementation parameters)"),
            "why_it_matters": ("without a transcribed copy of Cole's tables the coefficient "
                               "values cannot be validated; the chain therefore reports "
                               "transcription fidelity of the recorded parameters, the "
                               "coefficient-free scaling structure, and the FSI/solver "
                               "verification separately"),
        },
        "acquisition_status": "NOT_OBTAINED",
        "acquisition_attempts": acquisition_attempts(),
        "usable_for_physics_validation": False,
        "usable_for_transcription_fidelity_check": True,
        "parameters": PARAMETERS,
        "cases": CASES,
        "recorded_values": recorded,
        "scaling_structure": {
            "peak_pressure": "p_m = K (W^(1/3)/R)^A (Hopkinson scaling, exponent A)",
            "decay_time": "theta = K_theta W^(1/3) (W^(1/3)/R)^n (linear in Hopkinson scale)",
            "bubble_period": ("T = K_b W^(1/3) (H + 10.33)^(-5/6); the 1/3 and -5/6 "
                              "exponents are derived from the Rayleigh collapse "
                              "integral, only K_b is empirical"),
        },
        "evidence_note": ("every value in recorded_values is produced by this script from "
                          "the parameters block above; the chain recomputes them through "
                          "tensorfem.undex_chain.cole_recompute_vs_anchor and must agree "
                          "within 1 %"),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)} ({len(recorded)} cases)")
    for row in recorded:
        print(f"  {row['case_id']}: W={row['weight_kg']:g} kg R={row['range_m']:g} m "
              f"Z={row['reduced_range']:.4f} p_m={row['peak_pressure_pa']/1e6:.4f} MPa "
              f"theta={row['decay_time_s']*1e6:.2f} us "
              f"T_b={row['bubble_period_s']:.5f} s "
              f"form-cross-check={row['form_cross_check_max_rel']:.2e}")
    return payload


if __name__ == "__main__":
    main()