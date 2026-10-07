"""Traceable, fail-closed contracts for external nonlinear benchmarks.

The contracts in this module separate an available analytical or published
reference from a qualified TensorFEM finite-element result.  In particular,
having a Hertz formula is not evidence that a deformable 3-D contact solve has
been performed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal


Status = Literal["qualified", "blocked"]


@dataclass(frozen=True)
class ExternalSource:
    title: str
    authors: tuple[str, ...]
    publication: str
    year: int
    pages: str
    doi: str
    url: str
    access_date: str
    reuse_note: str


@dataclass(frozen=True)
class ExternalBenchmarkContract:
    case_id: str
    category: Literal["nonlinear_shell", "contact_3d"]
    source: ExternalSource
    parameters: dict[str, object]
    boundary_conditions: tuple[str, ...]
    reference_quantities: dict[str, object]
    solver_requirements: tuple[str, ...]
    evidence_files: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    status: Status
    tolerance: float = 0.03

    def validate(self) -> "ExternalBenchmarkContract":
        """Validate traceability and prevent an unsupported qualification."""
        if not self.case_id or not self.source.title or not self.source.authors:
            raise ValueError("case and bibliographic identity are required")
        if not self.source.doi.startswith("10."):
            raise ValueError("a DOI is required for these external contracts")
        if self.source.url != "https://doi.org/" + self.source.doi:
            raise ValueError("source URL must be the canonical DOI URL")
        if not (0 < self.tolerance <= 0.03):
            raise ValueError("qualification tolerance must be in (0, 0.03]")
        if not self.parameters or not self.boundary_conditions:
            raise ValueError("parameters and boundary conditions are required")
        if not self.solver_requirements:
            raise ValueError("solver requirements are required")
        if self.status == "qualified":
            if self.missing_evidence:
                raise ValueError("qualified contract cannot have missing evidence")
            if not self.reference_quantities or not self.evidence_files:
                raise ValueError("qualified contract requires oracle and result evidence")
        elif self.status == "blocked":
            if not self.missing_evidence:
                raise ValueError("blocked contract must identify missing evidence")
        else:
            raise ValueError("status must be 'qualified' or 'blocked'")
        return self

    def require_qualified(self) -> "ExternalBenchmarkContract":
        self.validate()
        if self.status != "qualified":
            raise RuntimeError(
                f"external benchmark {self.case_id!r} is blocked: "
                + "; ".join(self.missing_evidence)
            )
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def shallow_spherical_shell_ring_load_contract() -> ExternalBenchmarkContract:
    """Published snap-through candidate; deliberately not a numerical oracle."""
    return ExternalBenchmarkContract(
        case_id="karatas-yuksel-2021-clamped-ring-load",
        category="nonlinear_shell",
        source=ExternalSource(
            title="Snap-through Buckling of Shallow Spherical Shells under Ring Loads",
            authors=("Esra Eylem Karataş", "Receb Faruk Yükseler"),
            publication="Teknik Dergi 32(2)",
            year=2021,
            pages="10695-10716",
            doi="10.18400/tekderg.565095",
            url="https://doi.org/10.18400/tekderg.565095",
            access_date="2026-10-03",
            reuse_note=(
                "Citation and factual parameters only; do not redistribute or "
                "digitize article figures as qualification data without permission."
            ),
        ),
        parameters={
            "sphere_radius": {"value": 254.0, "unit": "mm"},
            "thickness": {"value": 0.3810, "unit": "mm"},
            "depth_parameter_eta": {"value": 0.0618, "unit": "1"},
            "ring_diameter": {"value": 25.4, "unit": "mm"},
            "material": "rigid PVC; complete numerical constants not recoverable",
        },
        boundary_conditions=(
            "clamped circular edge",
            "axisymmetric regularity at the apex",
            "static axisymmetric ring resultant at diameter 25.4 mm",
        ),
        reference_quantities={},
        solver_requirements=(
            "geometrically nonlinear curved-shell discretization",
            "configuration-dependent ring-load residual and consistent tangent",
            "continuation through ascending, descending, and post-snap branches",
            "three meshes and two continuation step sizes",
        ),
        evidence_files=(),
        missing_evidence=(
            "tabulated primary-source limit loads and displacements",
            "complete material constants for the selected published curve",
            "configuration-dependent ring-load residual and consistent tangent",
            "mesh- and arc-size-converged TensorFEM result",
        ),
        status="blocked",
    ).validate()


def large_rotation_pure_bending_contract() -> ExternalBenchmarkContract:
    """Runnable large-rotation shell case with an analytical circular oracle."""
    return ExternalBenchmarkContract(
        case_id="bathe-bolourchi-1979-large-rotation-pure-bending",
        category="nonlinear_shell",
        source=ExternalSource(
            title="Large displacement analysis of three-dimensional beam structures",
            authors=("Klaus-Jürgen Bathe", "Saïd Bolourchi"),
            publication=(
                "International Journal for Numerical Methods in Engineering 14"
            ),
            year=1979,
            pages="961-986",
            doi="10.1002/nme.1620140703",
            url="https://doi.org/10.1002/nme.1620140703",
            access_date="2026-10-03",
            reuse_note=(
                "Citation and independently derived circular-elastica oracle only; "
                "no publisher text, figure, or table is redistributed."
            ),
        ),
        parameters={
            "strip_length": {"value": 10.0, "unit": "length"},
            "strip_width": {"value": 1.0, "unit": "length"},
            "thickness": {"value": 0.1, "unit": "length"},
            "young_modulus": {"value": 1.2e6, "unit": "force/length^2"},
            "poisson_ratio": {"value": 0.0, "unit": "1"},
            "prescribed_end_rotation": {"value": 90.0, "unit": "degree"},
            "mesh_elements": (1, 2, 4),
        },
        boundary_conditions=(
            "root cross-section clamped",
            "cross-section rotation prescribed linearly along the strip",
            "displacement-controlled pure bending to 90 degrees",
        ),
        reference_quantities={
            "tip_x": {"value": 6.366197723675814, "unit": "length"},
            "tip_z": {"value": 6.366197723675814, "unit": "length"},
            "reaction_moment": {"value": 15.707963267948966, "unit": "force*length"},
            "formulae": (
                "R=L/theta; x=R sin(theta); z=R(1-cos(theta)); "
                "M=E b t^3 theta/(12 L)"
            ),
        },
        solver_requirements=(
            "geometrically nonlinear shell-strip equilibrium",
            "accepted intermediate rotations checked against the circular path",
            "monotone 1-, 2-, and 4-element convergence",
            "fine-mesh tip and reaction-moment errors below 3%",
        ),
        evidence_files=(
            "tests/test_large_rotation_shell_benchmark.py",
            "docs/large-rotation-shell-validation.md",
            "src/tensorfem/large_rotation_shell_benchmark.py",
        ),
        missing_evidence=(),
        status="qualified",
    ).validate()


def hertz_sphere_halfspace_contract() -> ExternalBenchmarkContract:
    """3-D deformable-contact candidate with a complete analytical oracle."""
    return ExternalBenchmarkContract(
        case_id="hertz-1882-sphere-elastic-halfspace",
        category="contact_3d",
        source=ExternalSource(
            title="Ueber die Berührung fester elastischer Körper",
            authors=("Heinrich Hertz",),
            publication="Journal für die reine und angewandte Mathematik 92",
            year=1882,
            pages="156-171",
            doi="10.1515/crll.1882.92.156",
            url="https://doi.org/10.1515/crll.1882.92.156",
            access_date="2026-10-03",
            reuse_note=(
                "Bibliographic facts and independently evaluated classical equations; "
                "no source scan or publisher typesetting is redistributed."
            ),
        ),
        parameters={
            "normal_force": {"value": 1000.0, "unit": "N"},
            "sphere_radius": {"value": 0.01, "unit": "m"},
            "sphere_young_modulus": {"value": 210.0e9, "unit": "Pa"},
            "sphere_poisson_ratio": {"value": 0.3, "unit": "1"},
            "halfspace_young_modulus": {"value": 210.0e9, "unit": "Pa"},
            "halfspace_poisson_ratio": {"value": 0.3, "unit": "1"},
        },
        boundary_conditions=(
            "frictionless normal contact between a deformable sphere and half-space",
            "uniform far-field half-space constraint sufficiently remote from contact",
            "symmetry constraints when a half/quarter model is used",
        ),
        reference_quantities={
            "effective_modulus": {"value": 115384615384.61537, "unit": "Pa"},
            "contact_radius": {"value": 0.000402072575858906, "unit": "m"},
            "indentation": {"value": 1.6166235625781572e-05, "unit": "m"},
            "maximum_pressure": {"value": 2953469442.9062705, "unit": "Pa"},
            "formulae": (
                "1/E*=(1-nu1^2)/E1+(1-nu2^2)/E2; "
                "a=(3 F R/(4 E*))^(1/3); delta=a^2/R; p0=3 F/(2 pi a^2)"
            ),
        },
        solver_requirements=(
            "deformable 3-D sphere and half-space solid discretizations",
            "surface-to-surface pressure integration and global equilibrium",
            "domain-size study demonstrating the finite model represents a half-space",
            "at least three contact-zone mesh densities",
            "contact radius, indentation, pressure, and resultant-force comparison",
        ),
        evidence_files=(),
        missing_evidence=(
            "global deformable-to-deformable 3-D contact solve",
            "integrated contact-pressure field and resultant-force audit",
            "half-space domain-size and contact-zone mesh convergence",
        ),
        status="blocked",
    ).validate()


def external_p0_contracts() -> tuple[ExternalBenchmarkContract, ...]:
    return (
        large_rotation_pure_bending_contract(),
        shallow_spherical_shell_ring_load_contract(),
        hertz_sphere_halfspace_contract(),
    )


def write_external_contracts(path: str | Path) -> None:
    """Write deterministic JSON after validating every contract."""
    payload = {
        "schema": "tensorfem.external-benchmark-contract/1.0",
        "contracts": [contract.to_dict() for contract in external_p0_contracts()],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
