"""m22346 board-audit verdict for m22319 / GAP-ICE-GATE-POWER (TensorFEM, level 2).

evidence keys follow the platform contract (impl_files / entry / run_artifacts /
tests / integration_callers / benchmark / repro_cmd / external_anchor).  Paths are
relative to this audit worktree (/srv/tensor/agent/work/m22346), where the audited
commit ee385cf3 is frozen byte-for-byte under docs/board_audit/m22346/verified/
(sha256 in verify_output_m22346.json).  integration_callers is deliberately empty:
the audited gate has no caller, so L3 is not claimed (finding F2).

Run: python docs/board_audit/m22346/insert_verdict_m22346.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = "TensorFEM"
SCOPE = "m22319 GAP-ICE-GATE-POWER"
MISSION_ID = 22346
LEVEL = 2

EVIDENCE = {
    "audited_commit": "ee385cf336b96c7cb9f64f9d7186455fdd327750 (branch agent/m22319)",
    "impl_files": [
        "docs/board_audit/m22346/verified/ice_gate_power.py"
        " (frozen copy of ee385cf3:src/tensorfem/ice_gate_power.py, 615 lines; "
        "section_added_mass_2d / decide / detection_power_analytic / critical_deviation / "
        "meta_gate / gate_verdict)",
    ],
    "entry": "docs/board_audit/m22346/verified/run_gate_evidence.py "
             "(= ee385cf3:benchmarks/ice_gate_power/run_gate_evidence.py, rerun by auditor: 41.2 s)",
    "run_artifacts": [
        "docs/board_audit/m22346/verified/gate_evidence.json",
        "docs/board_audit/m22346/verified/fig1_power_curve.png",
        "docs/board_audit/m22346/verified/fig2_convergence.png",
        "docs/board_audit/m22346/verified/fig3_band_vs_controls.png",
        "docs/board_audit/m22346/verify_output_m22346.json",
    ],
    "tests": [
        "tests/test_ice_gate_power.py::test_deep_cylinder_recovers_rho_pi_a_squared[omega0]",
        "tests/test_ice_gate_power.py::test_deep_cylinder_recovers_rho_pi_a_squared[infinity]",
        "tests/test_ice_gate_power.py::test_two_routes_bracket_the_independent_anchor",
        "tests/test_ice_gate_power.py::test_convergence_order_is_measured_and_positive[omega0]",
        "tests/test_ice_gate_power.py::test_convergence_order_is_measured_and_positive[infinity]",
        "tests/test_ice_gate_power.py::test_guard_band_arithmetic",
        "tests/test_ice_gate_power.py::test_power_at_audited_deviation_is_near_one",
        "tests/test_ice_gate_power.py::test_monte_carlo_matches_normal_approximation",
        "tests/test_ice_gate_power.py::test_guard_band_is_more_informative_than_the_naive_rule",
        "tests/test_ice_gate_power.py::test_critical_deviation_exceeds_the_tolerance",
        "tests/test_ice_gate_power.py::test_all_controls_behave_as_predeclared",
        "tests/test_ice_gate_power.py::test_negative_controls_are_real_numbers_not_scaled_synthetics",
        "tests/test_ice_gate_power.py::test_heave_declaration_is_rejected",
        "tests/test_ice_gate_power.py::test_degenerate_gates_are_rejected",
        "tests/test_ice_gate_power.py::test_coherent_mutation_makes_the_gate_accept",
        "tests/test_ice_gate_power.py::test_mutated_reference_cannot_be_decided",
        "(19 items, pytest 19 passed / 27.31 s, nothing deselected)",
    ],
    "integration_callers": "",
    "benchmark": {
        "registered_case_rows": 0,
        "registered_case_query": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
                                 "WHERE anchor_slug='GAP-ICE-GATE-POWER' -> 0 rows",
        "judged_quantity": "2-D heave added mass per unit length [t/m], chain geometry L=100 B=20 T=6 Cb=0.70",
        "ref_t_per_m": 229.619569379078,
        "ref_source": "mean of the two free-surface mirror routes (mean of 225.197 and 234.042); "
                      "independent anchor tensor_reports#19234 Q5 = 228.600 t/m",
        "got_t_per_m": 86.1,
        "deviation_rel": -0.6250319594587433,
        "decision": "reject (acceptance limit 0.092, rejection limit 0.208, detection power 1.00)",
        "auditor_independent_cross_checks": {
            "closed_form_bracket_t_per_m": [161.013, 322.013],
            "declared_vs_bracket_lower_bound_rel": -0.465239391211756,
            "finite_volume_independent_route_t_per_m": 225.83061996048474,
            "fv_vs_audited_omega0_route_rel": 0.002811615089636,
            "fv_force_vs_energy_functional_rel": -0.009536607082703,
            "audited_anchor_deviation_rel": -0.004440254730183,
            "judgement_resolution_discretisation_shift": {"omega0": 0.000429546916099,
                                                          "infinity": 0.000992582115342},
        },
        "verdict_robustness": "reject holds for every admissible reference in [161.0, 322.0] t/m "
                              "(worst case -46.5 %) and for any u <= 0.2375 (audited u = 0.029)",
    },
    "repro_cmd": "python benchmarks/ice_gate_power/run_gate_evidence.py  "
                 "(auditor rerun: gate_evidence.json bit-identical, 41.2 s) ; "
                 "python -m pytest tests/test_ice_gate_power.py -q  (19 passed, 27.31 s) ; "
                 "python docs/board_audit/m22346/verify_gate_power.py  (57 s, independent FV + "
                 "closed-form bracket)",
    "external_anchor": "tensor_reports#19234 (mission 21257) section Q5: A33 = 228.6 t/m from two "
                       "independent potential-flow routes (mutual 3.8 %, deep-cylinder self-check "
                       "-0.32 %..+2.90 %) + exact rho*pi*a^2 (Lamb art.71/73) for the mirror routes; "
                       "no literature table value (Vugts 1968) obtained -> external_anchor is "
                       "ecosystem-internal, literature anchor is an open gap",
    "level_rationale": {
        "L2_confirmed": "real module (615 lines, input=chain module -> compute -> structured verdict), "
                        "executable entry, artifacts regenerable to a key-identical JSON, 19 real tests "
                        "including two mutation non-tautology tests",
        "L3_not_met": "no integration caller: grep 'ice_gate_power' hits only its own benchmark script "
                      "and docs, no main-solver/platform path invokes decide/gate_verdict (finding F2)",
        "L4_not_met": "tensor_benchmark_cases has 0 rows for anchor_slug='GAP-ICE-GATE-POWER' -> no "
                      "registered referee; got-vs-ref is computed only by the audited object itself "
                      "(finding F1); literature external anchor missing",
        "L5_not_met": "no engineering closed loop and no external experimental reference",
    },
    "findings": [
        {"id": "F1", "severity": "blocking-for-L4 (platform-side)",
         "claim": "the anchor has a registered benchmark case",
         "evidence": "tensor_benchmark_cases WHERE anchor_slug='GAP-ICE-GATE-POWER' -> 0 rows; the "
                     "mission shipped docs/oss/...sql for oss_solutions, not for the benchmark table",
         "remediation": "register the case row with ref 228.6 t/m (#19234 Q5) and tolerance 0.15"},
        {"id": "F2", "severity": "major (integration)",
         "claim": "the criterion is machine-checkable",
         "evidence": "grep -rn ice_gate_power (repo): only benchmarks/ice_gate_power/* and docs; no "
                     "solver/platform caller of decide/gate_verdict",
         "remediation": "call gate_verdict from the ice-chain run/CI with fail-fast"},
        {"id": "F3", "severity": "minor (evidence integrity)",
         "claim": "all doc numbers are program-injected",
         "evidence": "make_doc.py:191 hard-codes 'accept（0.38%）' while the injected deviation is "
                     "-0.44% and no 0.38% quantity exists in gate_evidence.json",
         "remediation": "drop the parenthetical or bind it to a JSON key"},
        {"id": "F4", "severity": "minor (auditability)",
         "claim": "generated tables are readable",
         "evidence": "tables 4-4, 5-1 (x2), 5-2, 5-3 in docs/GAP-ICE-GATE-POWER-round2.md have no "
                     "header row",
         "remediation": "add header rows in make_doc.py"},
        {"id": "F5", "severity": "minor (method statement)",
         "claim": "u=2.9% is required by the sublinear panel convergence",
         "evidence": "judgement-resolution shift is +0.043%/+0.099% (measured here); u originates "
                     "from #19234 route residual 2.9% and the +/-1.93% free-surface band",
         "remediation": "state the causal chain correctly; u is conservative, not forced by panels"},
    ],
    "checks_run": [
        "git show --stat ee385cf3 -> 11 files, +2277 lines, artifacts included",
        "pytest tests/test_ice_gate_power.py -q (in m22319 tree) -> 19 passed, 27.31 s, 0 deselected",
        "python benchmarks/ice_gate_power/run_gate_evidence.py -> 41.2 s, JSON key-identical to commit",
        "python benchmarks/ice_gate_power/make_doc.py -> git status clean (docs regenerated identical)",
        "psql tensor_benchmark_cases WHERE anchor_slug='GAP-ICE-GATE-POWER' -> 0 rows",
        "psql tensor_reports id=20090 -> 2877 chars, 6 sections, 3 figures",
        "psql tensor_dev_missions id=22319 -> succeeded by dev_acceptance, server_quality_audit "
        "gate_valid=false / key_present=false",
        "sha256 chain module + added_mass_bem2d.py identical in main and m22319 (reference side untouched)",
        "python docs/board_audit/m22346/verify_gate_power.py -> independent FV route 225.83 t/m "
        "(omega0 variant) vs audited 225.197 (rel +0.28%); closed-form bracket [161.01, 322.01]; "
        "declared 86.1 is -46.5% below the lower bound; rule re-derivation matches published numbers",
        "NOT RUN: MFS route (auditor's own implementation did not converge), omega->infinity FV "
        "variant, R=240 with h<=0.5, literature table value (Vugts 1968), cluster job cj4035",
    ],
    "inflated_claims": [
        "'criterion is machine-checkable (pass=True)' - self-computed only: no caller (F2) and no "
        "registered benchmark row (F1)",
        "'accept（0.38%）' in the cross-check row - no source quantity (F3)",
        "'sublinear observed order is why u must be 2.9%' - panel residual is 0.04-0.1% (F5)",
        "'route-vs-Richardson <3%' is a same-route self-consistency check, not external validation",
    ],
    "verified_strengths": [
        "real implementation, real tests (mutation non-tautology), bit-identical reproduction, "
        "docs regenerated from JSON (no hand-copied numbers)",
        "negative controls N1-N5 come from chain channels / other routes / a rejected earlier "
        "reference, not from ref*constant (the m22156 failure mode is fixed)",
        "deep-cylinder exact validation -0.59% / +0.98% at depth 8a; independent anchor -0.44%",
        "verdict robust: reject for any admissible reference in [161, 322] t/m and any u <= 0.2375",
    ],
    "auditor_independence_limits": [
        "read-only access to the audited tree; frozen byte-identical copies kept in this audit bundle",
        "the auditor's FV route shares the free-surface mirror model (independence is in the "
        "discretisation, not in the model); omega->infinity variant not independently checked",
        "no literature table value available (no external network)",
    ],
    "disposition": "approve with notes (L2)",
}


def main() -> int:
    ev = json.dumps(EVIDENCE, ensure_ascii=False)
    sql_literal = ev.replace("'", "''")
    sql = f"""
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('{REPO}', '{SCOPE}', {MISSION_ID}, {LEVEL}, '{sql_literal}', 'board-audit');
SELECT id, repo, scope, mission_id, level, verdict_by, created_at
  FROM tensor_maturity_verdicts WHERE mission_id = {MISSION_ID} AND verdict_by = 'board-audit';
"""
    pathlib.Path("/tmp/verdict_m22346.sql").write_text(sql)
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform",
                        "-f", "/tmp/verdict_m22346.sql"], capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(r.stderr[-800:], file=sys.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())