"""Evidence-class audit of the scalar benchmark registry.

Why this module exists
----------------------
``tensorfem.benchmark_registry`` reports a single aggregate ("69 checks, all
inside 3 %").  That aggregate is *not* a statement about the finite-element
kernels: some registered checks compare a discretised solution against a
closed form (real numerical verification), while others compare two independent
closed-form implementations of the same formula (a useful cross-coding check,
but it cannot detect an FE defect), and one is a consistency identity that the
algorithm enforces by construction.

This module makes the distinction machine-visible and fail-closed:

* every registered check id is declared in :data:`EVIDENCE_CLASSES` with its
  class and a one-line basis;
* :func:`audit_evidence_classes` reports per-class counts, the error
  distribution per class, "thin margin" checks (``error/tolerance >= 0.5``) and
  self-comparison risks (algebra-level checks that agree to ``<= 1e-9``);
* an unclassified id is an error, not a silent ``discretized`` default.

Classes
-------
``discretized``
    ``computed`` comes from a mesh / finite-difference / eigenvalue / time-step
    computation of the governing equations.
``trial_space``
    ``computed`` comes from a numerical quadrature, minimisation or contour
    integral over an analytical trial space (Rayleigh-Ritz, Navier series,
    Morison quadrature) - numerical, but without a mesh.
``closed_form``
    ``computed`` is an algebraic closed-form evaluation checked against a second
    closed-form coding of the same physics.
``identity``
    ``computed`` is checked against a consistency identity that the algorithm
    enforces by construction, so the check cannot detect a wrong constitutive
    assumption.

Independent oracle
------------------
:func:`spheroid_equator_displacement_closed_form` re-derives, from membrane
theory alone, the equatorial normal displacement that
``shell.spheroid.displacement_equator`` compares against the in-repo Rayleigh-
Ritz reference.  For a spheroid ``r = a sin(theta)``, ``z = b cos(theta)`` the
cap-axial-equilibrium meridional force is ``N_phi = p a / 2`` and normal
equilibrium gives ``N_theta = p a (1 - a^2/(2 b^2))``, so at the equator
``w = a (N_theta - nu N_phi) / (E t) = (p a^2/(E t)) (1 - a^2/(2b^2) - nu/2)``.
"""
from __future__ import annotations

import math

__all__ = [
    "EVIDENCE_CLASSES",
    "CLASS_ORDER",
    "audit_evidence_classes",
    "classification_basis",
    "spheroid_equator_displacement_closed_form",
    "THIN_MARGIN_THRESHOLD",
]

CLASS_ORDER = ("discretized", "trial_space", "closed_form", "identity")
THIN_MARGIN_THRESHOLD = 0.5

#: ``check id -> (class, basis)``.  Basis is what a reviewer must be able to
#: verify in the source; it is deliberately short and factual.
EVIDENCE_CLASSES = {
    # --- continuum / structural FE against closed form -----------------------
    "truss.bar": ("discretized", "2-node bar FE tip displacement vs u=PL/(EA)"),
    "frame.cantilever": ("discretized", "frame FE tip displacement vs PL^3/(3EI)"),
    "continuum.cook": ("discretized", "Q4 FE membrane tip displacement vs Simo & Rifai 23.96"),
    "beam.timoshenko": ("discretized", "Timoshenko beam FE vs bending+shear closed form"),
    "buckling.euler": ("discretized", "beam FE eigenvalue vs Euler fixed-pinned load"),
    "dynamics.newmark": ("discretized", "Newmark time integration vs u0 cos(omega t)"),
    "dynamics.explicit": ("discretized", "central-difference integration vs u0 cos(omega t)"),
    "solid.hex8": ("discretized", "HEX8 FE axial frequency vs pi/(2L) sqrt(E/rho)"),
    "solid.tet10_patch": ("discretized", "TET10 element energy identity on a constant-strain patch"),
    "plate.mindlin": ("discretized", "Q4 Mindlin FE vs Navier sine-load Mindlin solution"),
    "nonlinear.arch": ("discretized", "two-bar truss continuation vs closed-form limit load"),
    "nonlinear.finite_bar": ("discretized", "finite-strain bar FE/adaptive steps vs closed form"),
    "modal.cantilever": ("discretized", "beam FE modes vs beta1=1.875104"),
    "shell.patch_energy": ("discretized", "Shell4 stiffness energy vs constant-strain analytical energy"),
    "patch.robinson_A": ("discretized", "Q4 patch test A: FE stress vs exact constant field"),
    "patch.robinson_B": ("discretized", "Q4 patch test B: FE stress vs exact constant field"),
    "patch.robinson_C": ("discretized", "Q8 patch test C: FE stress vs exact linear field"),
    "sparse.axial_bar": ("discretized", "sparse assembly+solve tip displacement vs PL/(EA)"),
    "thermal.rod_convection": ("discretized", "1-D FE conduction/convection vs series resistances"),
    "thermal.constrained_bar": ("discretized", "restrained bar FE vs sigma=-E alpha dT"),
    "plasticity.tet4_j2": ("discretized", "TET4 FE uniaxial-strain J2 vs linear-hardening solution"),
    "marine.hull_girder.deflection": ("discretized", "frame FE girder (24 elements) vs 5qL^4/(384EI)"),
    "marine.hull_girder.moment": ("discretized", "frame FE girder end moment vs qL^2/8"),
    "marine.plate_buckling.square": ("discretized", "finite-difference plate eigen vs Navier buckling load"),
    "marine.plate_buckling.long_panel": ("discretized", "finite-difference plate eigen vs Navier buckling load"),
    "marine.plate_buckling.longitudinally_stiffened": ("discretized", "finite-difference orthotropic eigen vs Navier"),
    "marine.hull_ultimate.initial_yield": ("discretized", "midpoint-fibre section integration vs b sy h^2/6"),
    "marine.hull_ultimate.full_plastic": ("discretized", "midpoint-fibre section integration vs b sy h^2/4"),
    "marine.hull_progressive.moment": ("discretized", "fibre section integration vs mirrored-component closed form"),
    "marine.shell4_buckling.navier": ("discretized", "Shell4 initial-stress eigen vs Navier critical load"),
    "marine.imperfect_strip.amplitude": ("discretized", "1280-step plastic strip integration vs refined reference"),
    "marine.imperfect_strip.membrane_force": ("discretized", "1280-step plastic strip integration vs refined reference"),
    "marine.imperfect_strip.external_work": ("discretized", "1280-step plastic strip integration vs refined reference"),
    "shell.spheroid.meridional_pole": ("discretized", "axisym ring FE vs cap-axial statics N=p r/(2 sin phi)"),
    "shell.spheroid.meridional_equator": ("discretized", "axisym ring FE vs cap-axial statics"),
    "shell.spheroid.hoop_pole": ("discretized", "axisym ring FE vs normal equilibrium N=p R2 (1-R2/(2R1))"),
    "shell.spheroid.displacement_equator": ("discretized", "axisym ring FE (256 facets) vs Rayleigh-Ritz membrane reference"),
    "shell.sphere.pressure_sigma_theta": ("discretized", "axisym shell FE vs pR/(2t)"),
    "shell.sphere.pressure_sigma_phi": ("discretized", "axisym shell FE vs pR/(2t)"),
    "shell.sphere.pressure_radial_displacement": ("discretized", "axisym shell FE vs pR^2(1-nu)/(2Et)"),
    "plate.mindlin_thick.center_mx": ("discretized", "Q4 Mindlin FE vs Mindlin-Reissner Navier series"),
    "plate.mindlin_thick.max_mxy": ("discretized", "Q4 Mindlin FE vs Mindlin-Reissner Navier series"),
    "plate.mindlin_thick.center_w": ("discretized", "Q4 Mindlin FE vs Mindlin-Reissner Navier series"),
    "inverse.axial_bar.modulus": ("discretized", "Levenberg-Marquardt inversion of a TrussModel forward solve vs generating E"),
    "plate.foundation.winkler_center_w": ("discretized", "Q4 Mindlin FE on Winkler foundation vs exact FSDT Navier series"),
    "plate.foundation.pasternak_center_w": ("discretized", "Q4 Mindlin FE on Pasternak foundation vs exact FSDT Navier series"),
    "plate.foundation.pasternak_zero_reduces_to_winkler": ("identity", "G->0 reduction compares two identical centre_deflection calls: consistency identity, cannot detect a defect"),
    # --- numerical trial-space / quadrature references -----------------------
    "cohesive.fracture_energy": ("trial_space", "trapezoid quadrature of the bilinear law vs triangular area"),
    "marine.stiffened_panel.navier": ("trial_space", "one-term Ritz + Gauss quadrature vs closed Navier amplitude"),
    "marine.morison.base_shear": ("trial_space", "32-point Morison quadrature vs quarter-phase closed form"),
    "marine.morison.overturning": ("trial_space", "32-point Morison quadrature vs quarter-phase closed form"),
    "marine.prestress_buckling.uniaxial": ("trial_space", "Navier prestress eigenvalue series vs reference factor"),
    "marine.prestress_buckling.biaxial": ("trial_space", "Navier prestress eigenvalue series vs reference factor"),
    "marine.postbuckling.imperfect_path": ("trial_space", "bracketed von Karman-Koiter branch vs reference amplitude"),
    "marine.fracture.numerical_j.plane_stress": ("trial_space", "contour J integral on the Williams field vs K^2/E'"),
    "marine.fracture.numerical_j.plane_strain": ("trial_space", "contour J integral on the Williams field vs K^2/E'"),
    # --- closed-form cross-coding (no discretisation involved) --------------
    "contact.spring": ("closed_form", "penalty-regularised 1-DOF contact algebra on both sides"),
    "marine.hydrostatics.displacement": ("closed_form", "box-barge Archimedes algebra vs same identities written twice"),
    "marine.hydrostatics.gm": ("closed_form", "KB+BM-KG initial-stability algebra vs same identities"),
    "marine.hydrostatics.restoring": ("closed_form", "restoring-moment algebra vs same identities"),
    "marine.fatigue.hot_spot.linear": ("closed_form", "linear extrapolation formula vs hand calculation"),
    "marine.fatigue.hot_spot.quadratic": ("closed_form", "quadratic extrapolation formula vs hand calculation"),
    "marine.fatigue.sn.single_block": ("closed_form", "power-law S-N block life vs hand calculation"),
    "marine.fatigue.miner.multi_block": ("closed_form", "Palmgren-Miner sum vs hand calculation"),
    "marine.fatigue_advanced.hot_spot.path_linear": ("closed_form", "path extrapolation vs hand calculation"),
    "marine.fatigue_advanced.rainflow.triangle_count": ("closed_form", "rainflow counting vs hand calculation"),
    "marine.fatigue_advanced.miner.variable_amplitude": ("closed_form", "variable-amplitude Miner sum vs hand calculation"),
    "marine.fatigue_advanced.paris.m2_closed_form": ("closed_form", "Paris-law integration vs closed form"),
    "marine.fracture.hot_spot": ("closed_form", "affine-field hot-spot algebra vs identity"),
    "marine.fracture.scl_membrane": ("closed_form", "SCL membrane algebra vs identity"),
    "marine.fracture.mode_i_k": ("closed_form", "finite-width Mode-I K formula vs analytical identity"),
    "marine.fracture.mode_i_j": ("closed_form", "K^2/E' conversion vs analytical identity"),
    # --- consistency identities --------------------------------------------
    "plasticity.j2": ("identity", "radial return enforces the yield surface it is checked against"),
}


def classification_basis(check_id: str) -> tuple[str, str]:
    """``(class, basis)`` of one registered check (raises when unclassified)."""
    try:
        return EVIDENCE_CLASSES[check_id]
    except KeyError:  # pragma: no cover - guarded by tests
        raise KeyError(
            f"check {check_id!r} has no declared evidence class; "
            "register it in tensorfem.registry_evidence_audit.EVIDENCE_CLASSES"
        ) from None


def spheroid_equator_displacement_closed_form(
    *, a: float, b: float, thickness: float, young: float, poisson: float, pressure: float
) -> float:
    """Exact membrane-theory equatorial normal displacement of a spheroid.

    Derived from cap axial equilibrium ``N_phi = p r / (2 sin phi)`` (at the
    equator ``r = a``, ``sin phi = 1``) and normal equilibrium
    ``N_phi/R1 + N_theta/R2 = p`` with ``R1 = R2 = a^2/b`` at the equator, i.e.
    ``N_theta = p a (1 - a^2/(2 b^2))``.  Hooke's law for plane stress then
    gives ``w = a (N_theta - nu N_phi) / (E t)``.
    """
    n_phi = pressure * a / 2.0
    n_theta = pressure * a * (1.0 - a * a / (2.0 * b * b))
    return a * (n_theta - poisson * n_phi) / (young * thickness)


def audit_evidence_classes(report: dict) -> dict:
    """Classify the results of ``verification_report()``.

    ``report`` must be the registry report dict (``results`` list).  The return
    value is JSON-serialisable and is what the gate test asserts on.
    """
    rows = report["results"]
    if not rows:
        raise ValueError("empty registry report")
    per_class = {name: [] for name in CLASS_ORDER}
    unclassified = []
    for row in rows:
        cid = row["id"]
        if cid not in EVIDENCE_CLASSES:
            unclassified.append(cid)
            continue
        cls, basis = EVIDENCE_CLASSES[cid]
        per_class[cls].append(
            {
                "id": cid,
                "error": float(row["error"]),
                "tolerance": float(row["tolerance"]),
                "margin": float(row["error"]) / float(row["tolerance"]),
                "basis": basis,
            }
        )

    stale = sorted(set(EVIDENCE_CLASSES) - {row["id"] for row in rows})
    counts = {name: len(items) for name, items in per_class.items()}
    classified = sum(counts.values())

    thin = sorted(
        (
            {"id": item["id"], "class": name, "margin": item["margin"], "error": item["error"]}
            for name, items in per_class.items()
            for item in items
            if item["margin"] >= THIN_MARGIN_THRESHOLD
        ),
        key=lambda d: -d["margin"],
    )
    # A check that agrees to <=1e-9 while both sides are algebra (or while the
    # reference is an identity) cannot detect a numerical defect: it is flagged
    # so that it is never quoted as FE verification.
    algebra_only = sorted(
        item["id"] for name in ("closed_form", "identity") for item in per_class[name]
        if item["error"] <= 1e-9
    )
    return {
        "schema": "tensorfem.registry-evidence-classes/1.0",
        "total_checks": len(rows),
        "classified": classified,
        "unclassified_ids": sorted(unclassified),
        "stale_ids": stale,
        "counts": counts,
        "discretized_share": (counts["discretized"] / len(rows)) if rows else 0.0,
        "min_error": {name: (min((i["error"] for i in items), default=None))
                      for name, items in per_class.items()},
        "max_error": {name: (max((i["error"] for i in items), default=None))
                      for name, items in per_class.items()},
        "thin_margin": thin,
        "algebra_only_ids": algebra_only,
        "policy": "unclassified checks are errors; aggregate 'passed' is not FE coverage",
        "class_basis": {name: [i["id"] for i in items] for name, items in per_class.items()},
    }