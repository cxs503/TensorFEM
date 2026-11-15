# -*- coding: utf-8 -*-
"""GAP-PP-1-ICE-STRUCT-CONTACT-PATCH: small-deflection-valid end-to-end case.

The acceptance end-to-end example (contact -> patch load -> plate stress) must
be reproducible *and* inside the validity of the theory it uses.  The original
demo load (5 MPa peak) sat at w/t ~ 14, outside small-deflection theory; the
valid case (``run_valid_case`` in ``examples/ice_struct_contact/
end_to_end_demo.py``) scales the load to w/t < 0.5 and benchmarks the FEM
chain against the independent Kirchhoff Navier series at the H3-ICE-PLATE-FEM
anchor level (0.339 %).

These tests guard that increment:
* the case is inside the small-deflection regime (w/t < 0.5);
* the FEM-vs-Navier peak-stress error is at the anchor level (same order of
  magnitude, i.e. within a factor of 3 of 0.339 %) and converges with the
  mesh;
* the chain is force-conserving (exact sub-cell splitting);
* the machine-readable non-uniform-vs-uniform peak stress difference is
  present and physically signed (the concentrated patch peaks higher).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "examples" / "ice_struct_contact"))

DEMO_PATH = REPO / "examples" / "ice_struct_contact" / "end_to_end_demo.py"
ARTIFACT = (
    REPO / "benchmarks" / "artifacts" / "GAP-PP-1-end-to-end-demo"
    / "end_to_end_demo.json"
)

H3_ANCHOR_PCT = 0.339
W_OVER_T_LIMIT = 0.5


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("gap_pp1_e2e_demo", DEMO_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def demo():
    return _load_demo_module()


@pytest.fixture(scope="module")
def valid_case(demo):
    # reduced ladder keeps the suite short; the committed artifact carries 16/24/32
    return demo.run_valid_case(plate_nx=24, meshes=(16, 24))


def test_valid_case_within_small_deflection(valid_case):
    """The acceptance example must be inside the small-deflection regime."""
    w_over_t = valid_case["small_deflection_validity"]["peak_deflection_over_thickness"]
    assert w_over_t < W_OVER_T_LIMIT
    assert valid_case["small_deflection_validity"]["valid"] is True
    # and the absolute stress is a meaningful number for structural steel
    assert valid_case["peak_stress_nonuniform"]["sigma_vm_MPa"] < 355.0


def test_valid_case_fem_vs_navier_within_anchor_band(valid_case):
    """FEM-vs-Navier peak bending stress error at the anchor level (live)."""
    nav = valid_case["navier_benchmark"]
    for key in ("nonuniform_fem_vs_navier_pct", "uniform_fem_vs_navier_pct"):
        err = abs(nav[key])
        assert err <= 3.0 * H3_ANCHOR_PCT, f"{key} = {nav[key]} % exceeds 3x anchor"
    assert valid_case["h3_anchor_check"]["same_order_of_magnitude"] is True
    assert valid_case["h3_anchor_check"]["h3_anchor_level_pct"] == H3_ANCHOR_PCT


def test_valid_case_mesh_convergence(valid_case):
    """The FEM error must shrink as the mesh is refined (not a constant)."""
    ladder = valid_case["navier_benchmark"]["mesh_ladder"]
    errs = [abs(row["fem_vs_navier_pct"]) for row in ladder]
    assert errs[-1] < errs[0]
    assert max(errs) < 5.0  # a few percent at the coarsest mesh, not +12.8 %


def test_valid_case_force_conserving(valid_case):
    """Exact sub-cell splitting: applied force == contact force to machine eps."""
    proj = valid_case["projected_patch"]
    assert proj["projection_method"] == "exact_subcell_splitting"
    assert abs(proj["load_balance_rel"]) < 1e-10
    # the sampled contact grid spans +-3 sigma: 99.7 % of the analytic force
    assert abs(proj["force_error_rel"]) < 0.01


def test_valid_case_machine_readable_difference(valid_case):
    """The non-uniform-vs-uniform peak stress difference is machine readable."""
    delta = valid_case["delta_nonuniform_over_uniform_pct"]
    for key in ("sigma_vm", "sigma_x", "deflection"):
        assert key in delta
        assert math.isfinite(delta[key])
    # a concentrated (Gaussian) patch must peak *higher* than the equal-force
    # uniform patch of the same footprint
    assert delta["sigma_x"] > 0.0
    assert delta["sigma_vm"] > 0.0


def test_committed_artifact_carries_valid_case():
    """The committed artifact JSON must carry the valid case (regeneration)."""
    assert ARTIFACT.exists(), "run the demo to regenerate the artifact"
    data = json.loads(ARTIFACT.read_text())
    vc = data["valid_case"]
    assert vc["small_deflection_validity"]["valid"] is True
    assert vc["h3_anchor_check"]["same_order_of_magnitude"] is True
    assert vc["delta_nonuniform_over_uniform_pct"]["sigma_x"] > 0.0
    # the overloaded reference case is retained but explicitly flagged
    assert data["small_deflection_validity"]["valid"] is False
