"""Importable builder for the marine benchmark casebook."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_path = Path(__file__).parents[2] / "scripts" / "run_marine_benchmark_report.py"
_spec = spec_from_file_location("_marine_benchmark_report_script", _path)
_module = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
build_report = _module.build_report

__all__ = ["build_report"]
