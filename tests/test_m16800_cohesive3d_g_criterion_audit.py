"""Independent audit checks for the merged DCB cohesive energy-release criterion.

These assertions are the *reviewer's* invariants (verifier seat m16800), not a
re-run of the author's tests: they pin the three facts that decide whether the
published claim "energy release rate vs analytic < 10 % -> PASS" means what it
says.

1. ``test_published_g_triplet_reproduces_from_raw_curve`` -- the three published
   G numbers are a deterministic function of the committed curve (no hand
   transcription, no hidden preprocessing).
2. ``test_analytic_anchor_is_the_law_area_itself`` -- the "independent analytic
   anchor" G_Ic is, to machine precision, the area under the very
   traction-separation law whose work is being differentiated.  This is what
   makes the criterion a **bookkeeping consistency check** rather than a
   physics validation: it cannot detect an error in G_Ic.
3. ``test_measured_g_is_closer_to_beam_theory_with_repaired_convention`` --
   a cross-merge check that never touches G_Ic: DCB beam theory at the measured
   load and crack front, evaluated with the repaired ``3 P^2 a^2`` coefficient,
   is an order-of-magnitude better match to the measured interface G than the
   pre-repair ``6 P^2 a^2`` coefficient.  This corroborates the m16661
   convention repair from data produced by a different agent (m16698).

Run: PYTHONPATH=src python -m pytest tests/test_m16800_cohesive3d_g_criterion_audit.py -q
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

from tensorfem.cohesive3d import BKCohesive3DLaw
from tensorfem.cohesive3d_dcb import DCBGeometry, beam_serr

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "cohesive3d_dcb_energy_release.json"
sys.path.insert(0, str(ROOT / "outputs" / "verify_m16800"))

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="energy-release artifact not committed"
)


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def audit(data: dict) -> dict:
    import audit_g_criterion as agc

    out = {
        "recompute": agc.recompute_g(ARTIFACT),
        "law": agc.law_area_identity(data["parameters"]),
        "beam": agc.beam_crosscheck(ARTIFACT),
    }
    return out


def test_published_g_triplet_reproduces_from_raw_curve(audit: dict) -> None:
    rep = audit["recompute"]
    for key in ("local", "fit", "global"):
        assert abs(rep["reproduction_rel_dev"][key]) < 1e-9, (
            f"{key}: recomputed {rep['G_' + key + '_recomputed']:.6e} vs "
            f"published {rep['published'][key]:.6e}"
        )
    # the driving quantity of the criterion is strictly positive and bracketed
    assert 250.0 < rep["G_local_recomputed"] < 400.0


def test_analytic_anchor_is_the_law_area_itself(audit: dict) -> None:
    """G_Ic and the measured quantity come from one and the same source."""
    law = audit["law"]
    assert abs(law["rel_dev"]) < 1e-12
    # peak traction of the law equals the declared strength: the law triple
    # (k_n, t_n0, G_Ic) is fully self-consistent, hence circular as a check.
    assert law["peak_traction_Pa"] == pytest.approx(law["declared_strength_Pa"], rel=1e-9)


def test_measured_g_is_closer_to_beam_theory_with_repaired_convention(
    data: dict, audit: dict
) -> None:
    p = data["parameters"]
    geom = DCBGeometry(
        length=p["length_m"],
        width=p["width_m"],
        arm_thickness=p["arm_thickness_m"],
        young_modulus=p["E_Pa"],
        poisson_ratio=p["nu"],
        initial_crack=p["initial_crack_m"],
        nx=p["nx"],
        nz=p["nz"],
    )
    end = audit["beam"]["per_step"][-1]
    g_meas = audit["recompute"]["G_local_recomputed"]
    g_repaired = beam_serr(end["load_N"], end["crack_front_m"], geom, modulus=p["E_Pa"])
    assert g_repaired == pytest.approx(end["G_beam_J_m2"], rel=1e-12)

    g_prerepair = 6.0 * end["load_N"] ** 2 * end["crack_front_m"] ** 2 / (
        p["E_Pa"] * p["width_m"] ** 2 * p["arm_thickness_m"] ** 3
    )
    dev_repaired = abs(g_meas / g_repaired - 1.0)
    dev_prerepair = abs(g_meas / g_prerepair - 1.0)
    assert dev_repaired < 0.5 * dev_prerepair, (
        f"repaired beam G {g_repaired:.1f} (dev {dev_repaired:.3f}) is not clearly "
        f"closer to measured {g_meas:.1f} than pre-repair {g_prerepair:.1f} "
        f"(dev {dev_prerepair:.3f})"
    )
    # Beam theory is an approximation; the measured G must not drift far from it.
    assert 1.0 < g_meas / g_repaired < 1.35