"""Create and compare versioned 18-degree-hole hemisphere evidence."""
import copy
from tensorfem.benchmark_cases import (
    compare_archives,hemisphere_case,run_case,seal_archive,write_archive,
)

baseline=run_case(hemisphere_case(),tier="quick")
write_archive(baseline,"hemisphere-evidence.json")
candidate=copy.deepcopy(baseline)
candidate["results"][-1]["computed"]*=1.001
candidate=seal_archive(candidate)
comparison=compare_archives(baseline,candidate,drift_tolerance=.01)
print("case/evidence hashes:",baseline["case_hash"],baseline["evidence_hash"])
print("qualification:",baseline["qualification_passed"])
print("mesh pass flags:",[(r["mesh"],r["passed"]) for r in baseline["results"]])
print("drift comparison:",comparison["passed"],comparison["results"][-1])

