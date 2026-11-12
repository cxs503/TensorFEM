"""Runner for CASE-SPHERE-PRESSURE (thin wrapper, no case logic here).

The implementation is the production module
:mod:`tensorfem.sphere_pressure_case` (reused production element
:class:`tensorfem.axisym_shell.AxisymShell`; single source of truth for the
reference derivation, the studies, the gates and the figures).  This script only
points the run at the case directory so the committed evidence
(``result.json`` / ``gates_verified.json`` / ``figures/``) stays where the case
card expects it.  The names re-exported below are the module's public case API,
so existing callers (``tests/test_sphere_pressure_membrane_case.py``) keep
importing this script without duplicating any logic.

Run::

    python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py

Equivalent console-script form (writes to the case-pool artifact directory)::

    tensorfem-sphere-pressure --out artifacts/CASE-SPHERE-PRESSURE
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfem.sphere_pressure_case import (  # noqa: E402,F401
    CASE,
    FACETS,
    PRODUCTION_FACETS,
    TOL,
    axisym_case,
    chord_control,
    convergence,
    figures,
    flat_facet_shell4_control,
    gates,
)
from tensorfem.sphere_pressure_case import (  # noqa: E402,F401
    poisson_sensitivity,
    pressure_scaling,
    quick_check,
    reference,
    run_case,
    thickness_sensitivity,
)

if __name__ == "__main__":
    run_case(out_dir=HERE)