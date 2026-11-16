"""Reproducibility gate for the ``patch_from_contact`` mission deliverables.

Mission GAP-PP-1-ICE-STRUCT-CONTACT-PATCH (original PR#1121 / m20207, reopened
as m23259) published four documents next to ``tensorfem/patch_from_contact.py``:

* ``docs/fig1_pressure_distributions.png``
* ``docs/fig2_deflection_comparison.png``
* ``docs/fig3_plate_deflection.png``
* ``docs/patch_from_contact_report.json``

None of them was reproducible from the code in the repository: the figures had
no committed generator, and the report still claimed an inverse-distance
projection (removed in m20624) and published a non-uniform-vs-uniform peak
deflection difference of -99.5 % that no longer reproduces at all (the current
code gives a *positive* difference for every uniform reference, because the
legacy uniform reference was not force-matched).

These tests make the artifacts part of the regression surface:

``test_figures_*``       the three PNGs exist, are real PNGs, and are embedded
                         in the report record;
``test_report_*``        schema, mission ids, and no stale projection claim;
``test_force_match_*``   the force-matched uniform patch load used by the
                         generator carries ``p*W*L`` to machine precision for
                         every mesh size / patch-edge alignment (the defect that
                         corrupted the published delta);
``test_legacy_reference_*``
                         the rejected Gauss-sampled reference is measurably
                         non-force-matched -- the negative control;
``test_numbers_*``       the published numbers re-derive from the current code.

Numbers live in ``artifacts/patch_from_contact_report_numbers.json``, written by
``tools/gen_patch_from_contact_report.py`` (nothing in the docs is hand-typed).
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.patch_from_contact import (  # noqa: E402
    ContactPressureData,
    _gaussian_contact_grid,
    contact_pressure_to_plate_load,
    peak_stresses,
)
from tensorfem.plate_mindlin import (  # noqa: E402
    solve_plate_on_winkler_foundation,
    structured_square_mesh,
)
from tools.gen_patch_from_contact_report import (  # noqa: E402
    E,
    K_WINKLER,
    N_CONTACT,
    NU,
    SIDE,
    T,
    integrated_force_gauss,
    navier_ssss_center_deflection,
    step_pressure_fn,
    uniform_patch_load_exact,
)

DOCS = ROOT / "docs"
ART = ROOT / "artifacts" / "patch_from_contact_report_numbers.json"
REPORT = DOCS / "patch_from_contact_report.json"
FIGURES = (
    DOCS / "fig1_pressure_distributions.png",
    DOCS / "fig2_deflection_comparison.png",
    DOCS / "fig3_plate_deflection.png",
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def numbers():
    assert ART.exists(), (
        "artifacts/patch_from_contact_report_numbers.json missing -- run "
        "`python tools/gen_patch_from_contact_report.py`"
    )
    return json.loads(ART.read_text())


@pytest.fixture(scope="module")
def report():
    assert REPORT.exists(), "docs/patch_from_contact_report.json missing"
    return json.loads(REPORT.read_text())


# ---------------------------------------------------------------------------
# Artifact presence / schema
# ---------------------------------------------------------------------------
class TestArtifactsPresent:
    def test_figures_are_png(self):
        for path in FIGURES:
            assert path.exists(), f"{path.name} missing"
            raw = path.read_bytes()
            assert raw[:8] == PNG_MAGIC, f"{path.name} is not a PNG"
            assert len(raw) > 20_000, f"{path.name} looks like a placeholder ({len(raw)} B)"

    def test_figures_are_embedded_in_report(self, report):
        names = [f["name"] for f in report["figures"]]
        assert len(names) == 3
        for fig in report["figures"]:
            raw = base64.b64decode(fig["png_base64"])
            assert raw[:8] == PNG_MAGIC
            assert fig["caption"].strip()

    def test_report_schema(self, report):
        for key in ("title", "summary", "scenario_key", "mission_id", "tags",
                    "markdown", "figures"):
            assert key in report
        assert report["scenario_key"] == "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH"
        assert report["mission_id"] == 20207          # the mission it documents
        assert report["refreshed_by_mission"] == 23259  # this refresh
        assert len(report["markdown"]) > 3000

    def test_report_has_no_stale_projection_claim(self, report):
        md = report["markdown"]
        # the pre-m20624 claim must be gone from the method description
        assert "反距离加权插值 → 板格节点压力" not in md
        assert "精确子单元分裂" in md
        # and the retracted -99.5 % must be flagged as non-reproducible, not
        # presented as a result
        assert "不可复现" in md
        assert "| mild_nonuniform | 947" not in md


# ---------------------------------------------------------------------------
# The force-matching property (the defect that corrupted the published delta)
# ---------------------------------------------------------------------------
class TestForceMatchedUniformReference:
    @pytest.mark.parametrize("nx", [10, 16, 31])
    @pytest.mark.parametrize("patch", [0.3, 0.2, 0.1])
    def test_applied_force_equals_nominal(self, nx, patch):
        """``uniform_patch_load_exact`` carries p*W*L to machine precision.

        ``nx=31`` is deliberately misaligned with the even patch extents, i.e.
        patch edges fall strictly inside elements -- exactly the configuration in
        which the element-Gauss reference loses force.
        """
        nodes, elements = structured_square_mesh(nx, SIDE, dtype=torch.float64)
        pressure = 2.5e5
        vec, info = uniform_patch_load_exact(
            nodes.numpy(), elements.numpy(), pressure,
            (SIDE / 2, SIDE / 2), (patch, patch),
        )
        assert abs(info["force_error_rel"]) < 1e-12
        assert vec.shape[0] == 3 * nodes.shape[0]
        # load only on the transverse (w) dofs
        assert np.all(vec.reshape(-1, 3)[:, 1:] == 0.0)

    def test_overhanging_patch_is_clipped(self):
        """A patch hanging over the plate edge applies only the overlap force."""
        nx = 12
        nodes, elements = structured_square_mesh(nx, SIDE, dtype=torch.float64)
        pressure = 1.0e5
        # patch centred on the edge x=0, half of it outside the plate
        _, info = uniform_patch_load_exact(
            nodes.numpy(), elements.numpy(), pressure, (0.0, SIDE / 2), (0.4, 0.4),
        )
        overlap = 0.2 * 0.4
        assert abs(info["applied_force"] / (pressure * overlap) - 1.0) < 1e-12


class TestLegacyReferenceIsRejected:
    """Negative control: the Gauss-sampled step reference is *not* force-matched."""

    def test_legacy_reference_force_error_is_measurable(self):
        nx, patch = 32, 0.3
        nodes, elements = structured_square_mesh(nx, SIDE, dtype=torch.float64)
        pressure = 1.0e5
        fn = step_pressure_fn(pressure, (SIDE / 2, SIDE / 2), patch / 2, patch / 2)
        applied = integrated_force_gauss(nodes.numpy(), elements.numpy(), fn)
        nominal = pressure * patch * patch
        rel = applied / nominal - 1.0
        # the defect is O(h/patch) per edge -> several percent at nx=32
        assert abs(rel) > 1e-3, (
            "the legacy Gauss-sampled step reference unexpectedly reproduced the "
            f"nominal force ({rel:+.2e}); the negative control is no longer a control"
        )
        # ... while the exact sub-cell reference is at machine precision
        _, info = uniform_patch_load_exact(
            nodes.numpy(), elements.numpy(), pressure,
            (SIDE / 2, SIDE / 2), (patch, patch),
        )
        assert abs(info["force_error_rel"]) < 1e-12
        assert abs(rel) > 100 * abs(info["force_error_rel"])

    def test_published_legacy_errors_are_recorded(self, numbers):
        for row in numbers["configurations"]:
            legacy = row["legacy_uniform_reference"]
            matched = row["uniform_reference"]
            assert abs(matched["force_error_rel"]) < 1e-12
            # the published legacy row is the *rejected* one -> must be off
            assert abs(legacy["force_error_rel"]) > 1e-4
            assert abs(legacy["applied_force_N"] - matched["nominal_force_N"]) > 0.0


# ---------------------------------------------------------------------------
# The published numbers re-derive from the current code
# ---------------------------------------------------------------------------
class TestPublishedNumbersReproduce:
    def test_projection_force_error_is_machine_precision(self, numbers):
        for row in numbers["configurations"]:
            assert row["projection_force_error_rel"] < 1e-12
            assert abs(row["uniform_reference"]["force_error_rel"]) < 1e-12

    def test_deltas_are_positive_and_ordered(self, numbers):
        """Concentration -> larger peak response; delta shrinks with sharpness.

        (The published -99.5 % had the opposite sign; a negative delta here would
        mean the projection/load chain regressed again.)
        """
        rows = numbers["configurations"]
        deltas = [r["delta_force_matched_pct"] for r in rows]
        assert all(d > 0 for d in deltas), deltas
        assert deltas[0] > deltas[1] > deltas[2]

    def test_scaling_identity(self, numbers):
        """The w/t=0.2 rescaling is exact (linear elasticity), verified by re-solve."""
        sc = numbers["scaling_identity_check"]
        assert sc["peak_deflection_rel_error"] < 1e-10
        assert sc["sigma_x_rel_error"] < 1e-10

    def test_navier_anchor_reproduces(self, numbers):
        anchor = numbers["navier_uniform_anchor"]
        d = E * T ** 3 / (12.0 * (1.0 - NU ** 2))
        w_ref = navier_ssss_center_deflection(1.0e4, SIDE, d)
        assert abs(w_ref * 1e3 / anchor["navier_deflection_mm"] - 1.0) < 1e-9
        # the FEM (Mindlin Q4/SRI, nx=32) must be within 1 % of the series
        assert abs(anchor["error_pct"]) < 1.0

    def test_ladder_delta_is_mesh_converged(self, numbers):
        ladder = numbers["mesh_ladder"]
        assert [r["plate_nx"] for r in ladder] == [16, 24, 32]
        d = [r["delta_force_matched_pct"] for r in ladder]
        # coarse-mesh shift smaller than 3 pp over the ladder
        assert abs(d[0] - d[-1]) < 3.0
        for row in ladder:
            assert abs(row["w_over_t"] - numbers["case"]["target_w_over_t"]) < 1e-9

    def test_recompute_mild_case_at_nx16(self, numbers):
        """Independent re-run of the nx=16 mild row against the published numbers."""
        ladder = {r["plate_nx"]: r for r in numbers["mesh_ladder"]}
        published = ladder[16]
        sigma = published["sigma_m"]
        centre = (SIDE / 2.0, SIDE / 2.0)
        pos, pres = _gaussian_contact_grid(
            centre[0], centre[1], sigma, sigma, published["peak_pressure_Pa"],
            SIDE, N_CONTACT,
        )
        contact = ContactPressureData(
            node_positions=pos, normal_pressures=pres,
            contact_area=2.0 * math.pi * sigma * sigma,
        )
        plate, patch = contact_pressure_to_plate_load(
            contact, 16, SIDE, E, NU, T, K_WINKLER,
        )
        w_nu = float(abs(plate.displacement[0::3].numpy()).max()) * 1e3

        nodes, elements = structured_square_mesh(16, SIDE, dtype=torch.float64)
        footprint = (6.0 * sigma, 6.0 * sigma)
        p_uniform = float(patch.total_force) / (footprint[0] * footprint[1])
        vec, info = uniform_patch_load_exact(
            nodes.numpy(), elements.numpy(), p_uniform, centre, footprint,
        )
        plate_uni = solve_plate_on_winkler_foundation(
            16, SIDE, E, NU, T, K_WINKLER, 0.0,
            force_vector=torch.as_tensor(vec, dtype=torch.float64),
        )
        w_uni = float(abs(plate_uni.displacement[0::3].numpy()).max()) * 1e3
        delta = 100.0 * (w_nu / w_uni - 1.0)

        assert abs(w_nu / published["peak_deflection_mm"] - 1.0) < 1e-9
        assert abs(w_uni / published["uniform_reference"]["peak_deflection_mm"] - 1.0) < 1e-9
        assert abs(delta - published["delta_force_matched_pct"]) < 1e-9
        assert abs(info["force_error_rel"]) < 1e-12
        # the peak bending stress must be a real stress, not a deflection proxy
        stress = peak_stresses(plate)
        assert stress["sigma_x_MPa"] > 0.0