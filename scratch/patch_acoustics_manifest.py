"""Re-emit the fem_structural_acoustics certificate in the canonical envelope.

The node was shipped with a private schema tag (``...-partial.v1``) on the belief
that ``tensorfem.verified-gates.v1`` cannot carry honest failures.  It can: the
envelope records ``passed`` / ``failed_gates`` and the auditor compares them with
the recomputed verdicts (cf. ``benchmarks/verified/fem_thermal``, which ships a
declared FAIL inside v1).  This patch

1. switches the schema tag to the canonical envelope,
2. gives every gate an explicit operator + numeric threshold (the auditor refuses
   to re-check a verdict that only exists in prose),
3. restates three gates as the *deviation* they actually measure
   (B3: |p-2|, B4: |sigma-1|, C5: gap(L8)/gap(L2)) instead of leaving a
   non-comparable raw number under a prose sentence,
4. demotes A2 (coarse-mesh modal diagnostic, never given a threshold) to an
   explicit informational check so it is no longer counted as a pass.

No value is changed, no threshold is relaxed, and the two honest failures
(C3, C4 vs Junger & Feit) stay failures.
"""
from __future__ import annotations

from pathlib import Path

PATH = Path("benchmarks/verified/fem_structural_acoustics/make_report.py")

OLD_BLOCK = '''    gate_list = []
    for name, gate in gates.items():
        value = gate.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            if isinstance(value, list) and all(
                isinstance(item, (int, float)) for item in value
            ):
                # vector-valued gate: keep the worst element as the scalar value
                value = max(abs(item) for item in value)
            else:
                value = float("nan")
        gate_list.append({
            "id": name,
            "quantity": gate.get("note", name),
            "value": float(value),
            "threshold": gate.get("threshold")
            if isinstance(gate.get("threshold"), (int, float)) else None,
            "passed": bool(gate.get("pass")),
        })
'''

NEW_BLOCK = '''    # Explicit operator + threshold per gate.  The auditor recomputes every
    # verdict from (value, operator, threshold); a verdict that only exists in
    # prose is un-recomputable by construction and is rejected (E_PASS_WITHOUT_
    # CRITERION / W_OPERATOR_ABSENT).  Thresholds are the acceptance bands the
    # node already used (3 % platform standard, 5 %, 10 %, 15 % versus
    # Junger & Feit), not values chosen after seeing the result.
    OPERATORS = {
        "F1_pytest": (">=", 12.0),
        "A1_plate_modal_first6": ("<=", 0.03),
        "B1_plate_sigma_low_ka": ("<=", 0.10),
        "B2_plate_sigma_high_ka": ("<=", 0.05),
        "B3_plate_low_ka_exponent": ("<=", 0.05),      # value restated as |p-2|
        "B4_plate_high_ka_plateau": ("<=", 0.05),      # value restated as |s-1|
        "B5_plate_monopole_law": ("<=", 0.03),
        "C1_esm_sphere": ("<=", 0.02),
        "C2_esm_resolution_convergence": ("<=", 0.05),
        "C3_cylinder_sigma_vs_jf_L2": ("<=", 0.15),
        "C4_cylinder_sigma_vs_jf_L8": ("<=", 0.15),
        "C5_cylinder_gap_trend": ("<=", 1.0),          # value restated as gap(L8)/gap(L2)
        "D1_plate_three_way": ("<=", 0.10),
        "D2_physical_bound": ("<=", 1.02),
        "E1_negative_control": ("<=", 1e-30),
        "E2_esm_regularization_sensitivity": ("<=", 0.05),
    }
    # A2 was a coarse-mesh modal diagnostic that never carried a threshold; it is
    # kept as a measurement but no longer reported as a passed gate.
    INFORMATIONAL = {"A2_plate_mesh_convergence"}

    def _restated(name: str, value: float) -> float:
        """Express deviation-style gates as the quantity their band refers to."""
        if name == "B3_plate_low_ka_exponent":
            return abs(value - 2.0)          # sigma ~ (ka)^p, theory p = 2
        if name == "B4_plate_high_ka_plateau":
            return abs(value - 1.0)          # sigma -> 1 (modal-RMS normalisation)
        if name == "C5_cylinder_gap_trend":
            l2 = max(abs(r["relative_gap"]) for r in result["cylinder_L2"])
            l8 = max(abs(r["relative_gap"]) for r in result["cylinder_L8"])
            return l8 / l2                    # < 1 means the gap shrinks with length
        return value

    gate_list = []
    informational = []
    for name, gate in gates.items():
        value = gate.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            if isinstance(value, list) and all(
                isinstance(item, (int, float)) for item in value
            ):
                # vector-valued gate: keep the worst element as the scalar value
                value = max(abs(item) for item in value)
            else:
                value = float("nan")
        operator, threshold = OPERATORS.get(name, (None, None))
        if name in INFORMATIONAL:
            informational.append({
                "id": name,
                "quantity": gate.get("note", name),
                "value": float(value),
                "note": "diagnostic only: no acceptance band was ever declared, "
                        "so it is reported and not counted as a pass",
            })
            continue
        if operator is None:
            raise SystemExit(f"gate {name} has no operator/threshold: refusing to "
                             "emit an un-recomputable verdict")
        value = _restated(name, float(value))
        gate_list.append({
            "id": name,
            "quantity": gate.get("note", name),
            "operator": operator,
            "value": value,
            "threshold": float(threshold),
            "passed": bool(gate.get("pass")),
        })
'''

OLD_SCHEMA = '''    evidence = {
        # NOTE: deliberately *not* "tensorfem.verified-gates.v1": that schema's
        # semantics require every gate to pass.  This node reports two honest
        # failures (C3, C4) and must not be flattened into a pass.
        "schema": "tensorfem.verified-gates-partial.v1",'''

NEW_SCHEMA = '''    evidence = {
        # Canonical envelope: it carries `passed` plus `failed_gates`, so the two
        # honest failures (C3, C4) travel inside the certificate instead of
        # forcing a private schema tag (cf. fem_thermal, declared FAIL in v1).
        "schema": "tensorfem.verified-gates.v1",'''

OLD_TAIL = '''        "passed": not failed_gates,
        "failed_gates": failed_gates,'''

NEW_TAIL = '''        "passed": not failed_gates,
        "failed_gates": failed_gates,
        "informational_checks": informational,'''


def main() -> None:
    text = PATH.read_text()
    for old, new, tag in (
        (OLD_BLOCK, NEW_BLOCK, "gate_list"),
        (OLD_SCHEMA, NEW_SCHEMA, "schema"),
        (OLD_TAIL, NEW_TAIL, "tail"),
    ):
        if text.count(old) != 1:
            raise SystemExit(f"anchor {tag} not found exactly once: {text.count(old)}")
        text = text.replace(old, new)
    PATH.write_text(text)
    print("patched", PATH)


if __name__ == "__main__":
    main()