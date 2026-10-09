"""Run the bounded stiffener-instability and local-pit qualification."""
from pprint import pprint

from tensorfem.marine_stiffener_failure import run_stiffener_failure_qualification


if __name__ == "__main__":
    pprint(run_stiffener_failure_qualification())
