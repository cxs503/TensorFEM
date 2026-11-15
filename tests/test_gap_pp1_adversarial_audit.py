"""Independent audit of the merged GAP-PP-1 contact→patch→plate deliverable.

These tests do not exercise the chain "as intended" — they pin the *claims* made
in ``docs/VERIFY-GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.md`` against the current HEAD:

* the quoted "FEM vs Navier 0.41 %" is a single-mesh number, so the mesh ladder
  must stay inside the 3 % acceptance bound on the *finest* mesh (the coarse-mesh
  error is a known under-resolution artefact for the highly localised Gaussian
  patch — see ``docs/AUDIT-GAP-PP-1-adversarial-verification.md``);
* the "two independent FEM paths cross-validate each other" claim is checked by
  showing that both paths resolve to the same assembly code (their < 1e-9
  agreement is then code identity, not validation);
* the archived cross-check artifact (mission 20624) is re-run and the drift is
  reported instead of assumed zero;
* the demo's H3-anchor consistency check must be *derived* from the archive, not
  a literal, and the demo must expose its small-deflection validity;
* ``loaded_area`` semantics and the two distinct "force error" quadratures.

Numbers are produced by ``tools/verify_gap_pp1_adversarial.py``; the runtime is
kept to a coarse ladder (nx <= 32) so the module stays a short test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import tensorfem.ice_panel_patch_contact as IC  # noqa: E402
from tools.verify_gap_pp1_adversarial import (  # noqa: E402
    ARCHIVE,
    artefact_reproduction,
    chain_independence,
    crosscheck_convergence,
    gaussian_case,
    load_area_semantics,
    validity_audit,
)


def test_fe_error_converges_toward_the_navier_series():
    """The chain error must stay < 3 % on the finest mesh.

    The Navier series is evaluated *at the FE peak location*, which shifts with
    mesh refinement; for the highly localised Gaussian patch (sigma=0.08 on a
    1.0 m plate) this makes the signed error oscillate (e.g. -4.19 % @16,
    +0.73 % @24, -1.13 % @32) rather than decay strictly monotonically.  The
    meaningful acceptance is therefore that the *finest* mesh error is inside the
    3 % bound; the coarse-mesh under-resolution and the ~1.1 % finest-mesh level
    (just above the 1 % H3-anchor order of magnitude) are recorded as known
    limitations of the highly-localised patch.
    """
    conv = crosscheck_convergence((24, 32))
    for name, summary in conv["summary"].items():
        # the finest mesh is inside the 3 % acceptance bound
        assert summary["finest_abs_error_pct"] < 3.0, (name, summary["errors_pct_by_nx"])
        # force balance of the projection is machine exact on every mesh
    for row in conv["rows"]:
        # ``force_error_rel`` from the ice_panel chain compares the projection with
        # the patch's *force metadata*; for the Gaussian patch that metadata is the
        # analytic 2*pi*sigma^2 force, so the percent-level part is quadrature
        # (finite grid + 3 sigma truncation), not projection error -- see F6 in
        # docs/AUDIT-GAP-PP-1-adversarial-verification.md.
        assert row["projection_force_error_rel"] < 3.0e-2, (row["patch"], row["plate_nx"])
        if row["patch"].startswith("uniform"):
            assert row["projection_force_error_rel"] < 1e-12
        assert row["force_balance_rel"] is None or abs(row["force_balance_rel"]) < 1e-10


def test_the_two_fem_paths_are_not_independent():
    """A shared assembly module means the agreement is tautological.

    The VERIFY doc quotes "双独立 FEM 路径一致性 < 10 %" as an independent
    cross-check; if both paths assemble through the same routines the honest
    statement is "same code path, identical to round-off".
    """
    audit = chain_independence(nx=8)
    assert audit["numeric_gap_rel_at_nx16"] < 1e-9
    assert audit["shared_assembly_routines"], audit
    assert audit["verdict"].startswith("not independent")


def test_archived_artifact_is_rechecked_not_trusted():
    """The mission-20624 artifact must be re-derived; any drift is recorded.

    The archived cross-check used a coarse 16-mesh ladder; the current HEAD
    re-derivation on the same mesh is compared and the *drift* is asserted to be
    finite and recorded (the archived Gaussian-patch row at nx=16 was -0.41 %,
    the current HEAD re-derives -4.19 % — a genuine coarse-mesh drift that the
    audit must surface rather than assume zero).
    """
    if not ARCHIVE.exists():  # pragma: no cover - archive is committed
        pytest.skip("archived verification artifact missing")
    rep = artefact_reproduction(meshes=(16,))
    assert rep["archive_found"] and rep["rows"]
    for row in rep["rows"]:
        assert np.isfinite(row["mine_drift_pct"])
        assert np.isfinite(row["head_mine_vs_navier_pct"])
    # the tool must give a definite reproducibility verdict either way
    assert isinstance(rep["reproducible"], bool)
    if not rep["reproducible"]:
        assert rep["max_abs_mine_drift_pct"] > 1e-6


def test_demo_ratio_is_load_invariant_and_validity_is_reported():
    """w/t of the demo case is far outside small-deflection theory.

    In linear theory the non-uniform/uniform ratio is exactly invariant under a
    load rescale, so a 100x rescale is both a validity fix and a proof that the
    quoted ratio is a linear-model property, not a converged physical number.
    """
    val = validity_audit(plate_nx=8)
    hi, lo = val["rows"]
    assert hi["w_over_t"] > val["w_over_t_limit"]           # demo case invalid
    assert lo["w_over_t"] <= val["w_over_t_limit"]          # rescaled case valid
    assert val["linearity_rel_dev"] < 1e-6                  # ratio is load invariant
    assert "INVALID small-deflection regime" in val["verdict_demo_case"]


def test_loaded_area_and_the_two_force_quadratures():
    """``loaded_area`` is a footprint; projection error and quadrature error differ."""
    audit = load_area_semantics()
    rows = audit["rows"]
    contact = rows["czm_spherical_edge"]
    assert contact["loaded_area_over_pressurised_pct"] == pytest.approx(0.0, abs=1e-6)
    assert contact["projection_vs_discrete_field_rel"] < 1e-12

    one_sided = rows["synthetic_one_sided_half_zero"]
    # a partially open patch: the key overstates the loaded area by ~2x
    assert one_sided["loaded_area_over_pressurised_pct"] == pytest.approx(100.0, rel=1e-6)

    gauss = rows["gaussian_2MPa_sigma0.08"]
    # projection is exact, but the discrete field vs the analytic 2*pi*sigma^2
    # force is percent level -> the VERIFY doc's "< 0.001 % vs analytic" claim
    assert gauss["projection_vs_discrete_field_rel"] < 1e-12
    assert 1.0e-3 < gauss["discrete_field_vs_metadata_rel"] < 3.0e-2
    assert "footprint" in audit["verdict_loaded_area"]


def test_pressurised_area_is_exposed_by_the_projection():
    """The module now reports the pressure-carrying area, not only the footprint."""
    law = IC.default_law()
    patch = IC.czm_contact_patch(
        law=law, ceiling=2.0e6,
        edge={"kind": "spherical", "radius": 1.0, "amplitude": 0.02, "box_half": 0.10},
        total_force=4.0e4, panel=(0.8, 0.7), grid=61,
    )
    _n, _e, _f, info = IC.patch_from_contact(patch, 24, 21, 0.8, 0.7)
    assert "pressurised_area" in info and "footprint_area" in info
    assert info["pressurised_area"] == pytest.approx(info["footprint_area"], rel=1e-12)

    zero_half = patch.pressure.clone()
    zero_half[:, zero_half.shape[1] // 2:] = 0.0
    one_sided = IC.ContactPatch(
        x=patch.x, y=patch.y, pressure=zero_half,
        total_force=float(patch.total_force), contact_area=float(patch.contact_area),
        peak_pressure=float(patch.peak_pressure), centroid=patch.centroid,
        approach=0.0, opening=IC.torch.zeros_like(zero_half), residual=0.0,
        config={"kind": "one_sided"},
    )
    _n, _e, _f, info1 = IC.patch_from_contact(one_sided, 24, 21, 0.8, 0.7)
    assert info1["pressurised_area"] == pytest.approx(0.5 * info1["footprint_area"], rel=1e-9)


def test_demo_h3_check_is_derived_from_the_archive():
    """The acceptance check must be computed, not a literal that cannot fail."""
    sys.path.insert(0, str(ROOT / "examples" / "ice_struct_contact"))
    from end_to_end_demo import h3_anchor_check, run_demo

    check = h3_anchor_check()
    assert check["h3_anchor_level_pct"] is not None
    archived = json.loads(ARCHIVE.read_text())["navier_crosscheck"]
    row = next(r for r in archived["rows"] if r["patch"].startswith("gaussian"))
    assert check["gaussian_patch_fem_vs_navier_pct"] == pytest.approx(row["mine_vs_navier_pct"])
    assert check["same_order_of_magnitude"] == bool(
        abs(row["mine_vs_navier_pct"]) <= 3.0 * check["h3_anchor_level_pct"]
    )

    result = run_demo(plate_nx=8)
    validity = result["small_deflection_validity"]
    assert validity["valid"] == bool(
        validity["peak_deflection_over_thickness"] <= validity["limit"]
    )
    assert validity["valid"] is False  # the shipped demo load is outside the theory
