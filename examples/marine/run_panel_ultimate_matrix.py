"""Execute or resume the declared marine-panel mesh/control matrix."""
from __future__ import annotations

import argparse
import json

from tensorfem.marine_panel_execution import execute_panel_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--cache", default="artifacts/marine-panel-ultimate")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--no-resume", action="store_true")
parser.add_argument("--maximum-wall-seconds", type=float)
arguments = parser.parse_args()
report = execute_panel_matrix(arguments.cache, steps=arguments.steps,
                              resume=not arguments.no_resume,
                              maximum_wall_seconds=arguments.maximum_wall_seconds)
print(json.dumps(report, indent=2, sort_keys=True))
