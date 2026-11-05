"""Run the curved-geometry Shell4 geometric-stiffness audit."""
from pprint import pprint

from tensorfem.shell4_curved_buckling import run_shell4_curved_buckling_qualification

pprint(run_shell4_curved_buckling_qualification())