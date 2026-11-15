"""Generate the acceptance artifacts for GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE.

Produces:
  docs/grillage_ice_ultimate/acceptance_map.json
  docs/grillage_ice_ultimate/acceptance_rule_benchmark.json
  docs/grillage_ice_ultimate/acceptance_combined_cases.json
  docs/grillage_ice_ultimate/acceptance_verification.json
"""
from __future__ import annotations

import json
from pathlib import Path

from tensorfem.grillage_ice_ultimate import (
    case_study_geometry,
    panel_stress_ultimate_map,
    write_ultimate_map,
)
from tensorfem.grillage_ice_verification import (
    combined_case_table,
    run_grillage_verification,
)
from tensorfem.grillage_ice_rule_benchmark import (
    rule_benchmark_table,
    run_rule_benchmark,
    write_rule_benchmark,
)

OUT = Path("docs/grillage_ice_ultimate")
OUT.mkdir(parents=True, exist_ok=True)

geom = case_study_geometry()

# 1. Panel stress -> ultimate load map (acceptance criterion 3)
print("Generating ultimate map...")
mapping = panel_stress_ultimate_map(
    geom,
    axial_ratios=(0.0, 0.15, 0.30, 0.45),
    patch_area_ratios=(0.10, 0.25, 0.50, 1.00),
    hydrostatic_pressures=(0.0, 5.0e4, 1.0e5),
    design_pressure=2.0e6,
    design_provenance="IACS-PC6",
)
write_ultimate_map(mapping, OUT / "acceptance_map.json")
print(f"  Map: {len(mapping['rows'])} rows written")

# 2. Rule benchmark (acceptance criterion 1: literature/rule comparison)
print("Running rule benchmark...")
bench = run_rule_benchmark(geom)
write_rule_benchmark(bench, OUT / "acceptance_rule_benchmark.json")
# Also write the table separately
table = rule_benchmark_table(geom)
(OUT / "acceptance_rule_benchmark_table.json").write_text(
    json.dumps(table, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(f"  Benchmark: {bench['n_checks']} checks, passed={bench['passed']}")
print(f"  Table: {len(table['rows'])} rows")

# 3. Combined case table (acceptance criterion 2: >=3 combined cases)
print("Generating combined case table...")
cases = combined_case_table(geom)
(OUT / "acceptance_combined_cases.json").write_text(
    json.dumps(cases, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(f"  Combined cases: {len(cases)} rows")

# 4. Full verification suite
print("Running verification suite...")
verif = run_grillage_verification()
(OUT / "acceptance_verification.json").write_text(
    json.dumps(verif, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(f"  Verification: {verif['n_checks']} checks, passed={verif['passed']}")

# Summary
print("\n=== ACCEPTANCE SUMMARY ===")
print(f"Criterion 1 (literature/rule comparison <15%): T7 Paik-Thayamballi + T14 IACS Polar Class = {'PASS' if bench['passed'] else 'FAIL'}")
print(f"Criterion 2 (>=3 combined cases): {len(cases)} cases = {'PASS' if len(cases) >= 3 else 'FAIL'}")
print(f"Criterion 3 (machine-readable map): {len(mapping['rows'])} rows = {'PASS' if len(mapping['rows']) > 0 else 'FAIL'}")
print(f"Overall verification: {verif['n_checks']} checks, all passed = {verif['passed']}")
