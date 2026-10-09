"""Run discrete stiffener and nonuniform-corrosion screening evidence."""
from pprint import pprint

from tensorfem.marine_discrete_panel import run_discrete_panel_qualification


if __name__ == "__main__":
    pprint(run_discrete_panel_qualification())
