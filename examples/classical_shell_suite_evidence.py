"""Build the quick classical-shell suite archive."""
from tensorfem.shell_benchmark_suite import run_classical_shell_suite,write_suite

suite=run_classical_shell_suite(tier="quick")
write_suite(suite,"classical-shell-suite.json")
print("suite hash:",suite["suite_hash"])
for case in suite["cases"]:
    print(case["case"]["case_id"],case["qualification_passed"],
          [(x["mesh"],round(100*x["error"],4),x["passed"]) for x in case["results"]])

