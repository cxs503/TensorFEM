"""Run TensorFEM-only prestress and imperfect plate qualification."""
from tensorfem.marine_plate_postbuckling import run_plate_postbuckling_qualification


if __name__ == "__main__":
    report = run_plate_postbuckling_qualification()
    for row in report["prestress_eigenbuckling"]:
        print(row["case"], "factor=", row["factors"][-1], "error=", row["errors"][-1])
    path = report["imperfect_postbuckling"]
    print("imperfect path amplitude=", path["final_amplitudes"][-1],
          "error=", path["maximum_errors"][-1])
