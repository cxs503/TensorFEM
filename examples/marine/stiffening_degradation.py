"""Run the marine stiffening/degradation sensitivity qualification."""
from pprint import pprint

from tensorfem.marine_degradation_qualification import (
    run_marine_degradation_qualification,
)


if __name__ == "__main__":
    pprint(run_marine_degradation_qualification())
