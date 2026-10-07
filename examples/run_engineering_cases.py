"""Run all three end-to-end engineering case packages."""
from pathlib import Path
from tensorfem.engineering_cases import run_case

if __name__=="__main__":
    here=Path(__file__).with_name("engineering_cases");root=Path("engineering_case_jobs")
    for path in sorted(here.glob("*.json")):print(path.name,run_case(path,root))
