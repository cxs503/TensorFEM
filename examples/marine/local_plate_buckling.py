"""Run TensorFEM-only local plate buckling qualification evidence."""
from tensorfem.marine_plate_buckling import run_local_plate_buckling_qualification


if __name__ == "__main__":
    report = run_local_plate_buckling_qualification()
    for case in report["cases"]:
        print(case["case"], "mode", case["mode"], "errors", case["errors"])
