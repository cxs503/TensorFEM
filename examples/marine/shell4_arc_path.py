"""Run the limited real-discretization Shell4 arc-length evidence."""
from pprint import pprint

from tensorfem.shell4_arc_path import run_shell4_arc_path_evidence


if __name__ == "__main__":
    pprint(run_shell4_arc_path_evidence())
