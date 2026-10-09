"""Fail-closed qualification gates for deformable--deformable 3-D Hertz FE.

This module does not solve contact.  It records and evaluates evidence from a
real three-dimensional solid/contact solve without allowing an analytical
Hertz calculator, an axisymmetric model, or a rigid indenter to be promoted to
the general 3-D capability by accident.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, Sequence


@dataclass(frozen=True)
class Hertz3DReference:
    effective_modulus: float
    contact_radius: float
    indentation: float
    peak_pressure: float


def deformable_hertz_reference(
    force: float, radius: float, young_sphere: float, poisson_sphere: float,
    young_halfspace: float, poisson_halfspace: float,
) -> Hertz3DReference:
    """Classical small-contact oracle for two elastic bodies."""
    if min(force, radius, young_sphere, young_halfspace) <= 0:
        raise ValueError("force, radius and moduli must be positive")
    if not (-1 < poisson_sphere < .5 and -1 < poisson_halfspace < .5):
        raise ValueError("Poisson ratios must lie in (-1, 0.5)")
    effective = 1.0 / (
        (1 - poisson_sphere**2) / young_sphere
        + (1 - poisson_halfspace**2) / young_halfspace
    )
    radius_contact = (3 * force * radius / (4 * effective)) ** (1 / 3)
    indentation = radius_contact**2 / radius
    peak = 3 * force / (2 * math.pi * radius_contact**2)
    return Hertz3DReference(effective, radius_contact, indentation, peak)


@dataclass(frozen=True)
class Hertz3DRun:
    """Auditable output of one FE run, normalized by the Hertz oracle."""

    contact_cells_across_radius: int
    domain_radii: float
    sphere_solid_elements: int
    halfspace_solid_elements: int
    integrated_force: float
    support_reaction: float
    contact_radius: float
    indentation: float
    peak_pressure: float
    pressure_l2_error: float
    sphere_is_deformable: bool = True
    halfspace_is_deformable: bool = True
    spatial_dimension: int = 3
    pressure_was_integrated: bool = True

    def validate(self) -> "Hertz3DRun":
        if self.contact_cells_across_radius < 1 or self.domain_radii <= 0:
            raise ValueError("positive contact resolution and domain size are required")
        if min(self.sphere_solid_elements, self.halfspace_solid_elements) < 1:
            raise ValueError("both bodies require solid elements")
        values = (self.integrated_force, self.contact_radius, self.indentation,
                  self.peak_pressure)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("finite positive contact outputs are required")
        if not math.isfinite(self.support_reaction) or not math.isfinite(self.pressure_l2_error):
            raise ValueError("finite reaction and pressure error are required")
        return self


@dataclass(frozen=True)
class Hertz3DQualification:
    status: Literal["qualified", "blocked"]
    tolerance: float
    errors: dict[str, float]
    missing_evidence: tuple[str, ...]
    runs: tuple[dict[str, object], ...]


def qualify_deformable_hertz_3d(
    reference_force: float,
    reference: Hertz3DReference,
    mesh_runs: Sequence[Hertz3DRun],
    domain_run: Hertz3DRun | None,
    *,
    tolerance: float = .03,
) -> Hertz3DQualification:
    """Apply strict mesh, physics, equilibrium, and truncation gates.

    Three strictly increasing contact-zone resolutions are mandatory.  The
    domain study must have a larger remote boundary while retaining at least
    the fine run's resolution.  Every reported Hertz quantity, pressure-field
    L2 norm, force balance, and domain sensitivity must be below ``tolerance``.
    """
    if reference_force <= 0 or not (0 < tolerance <= .03):
        raise ValueError("invalid force or qualification tolerance")
    runs = tuple(run.validate() for run in mesh_runs)
    missing: list[str] = []
    if len(runs) < 3:
        missing.append("at least three contact-zone mesh densities")
    resolutions = [r.contact_cells_across_radius for r in runs]
    if len(runs) >= 3 and any(b <= a for a, b in zip(resolutions, resolutions[1:])):
        missing.append("strictly increasing contact-zone resolution")
    for index, run in enumerate(runs):
        if run.spatial_dimension != 3:
            missing.append(f"mesh run {index}: true three-dimensional discretization")
        if not run.sphere_is_deformable or not run.halfspace_is_deformable:
            missing.append(f"mesh run {index}: two deformable solid bodies")
        if not run.pressure_was_integrated:
            missing.append(f"mesh run {index}: integrated surface pressure")

    errors: dict[str, float] = {}
    if runs:
        fine = runs[-1]
        errors = {
            "resultant_force": abs(fine.integrated_force / reference_force - 1),
            "force_balance": abs(fine.support_reaction - fine.integrated_force)
                             / reference_force,
            "contact_radius": abs(fine.contact_radius / reference.contact_radius - 1),
            "indentation": abs(fine.indentation / reference.indentation - 1),
            "peak_pressure": abs(fine.peak_pressure / reference.peak_pressure - 1),
            "pressure_l2": fine.pressure_l2_error,
        }
        if any(value >= tolerance for value in errors.values()):
            missing.append("fine-mesh Hertz outputs and equilibrium below 3%")
        if len(runs) >= 3:
            primary = ("contact_radius", "indentation", "peak_pressure")
            sequences = {
                name: [abs(getattr(r, name) / getattr(reference, name) - 1) for r in runs]
                for name in primary
            }
            if any(any(b >= a for a, b in zip(seq, seq[1:])) for seq in sequences.values()):
                missing.append("monotone convergence of radius, indentation and pressure")

        if domain_run is None:
            missing.append("larger-domain truncation run")
        else:
            large = domain_run.validate()
            if (large.domain_radii <= fine.domain_radii
                    or large.contact_cells_across_radius < fine.contact_cells_across_radius):
                missing.append("domain run with a farther boundary at retained resolution")
            domain_changes = [
                abs(getattr(large, name) / getattr(fine, name) - 1)
                for name in ("integrated_force", "contact_radius", "indentation", "peak_pressure")
            ]
            errors["domain_sensitivity"] = max(domain_changes)
            if errors["domain_sensitivity"] >= tolerance:
                missing.append("domain-size sensitivity below 3%")
    else:
        missing.append("global deformable-to-deformable 3-D contact results")

    missing = list(dict.fromkeys(missing))
    return Hertz3DQualification(
        "blocked" if missing else "qualified", tolerance, errors, tuple(missing),
        tuple(asdict(run) for run in runs),
    )

