"""Three-level analytic verification ladder for the linear-elastic solvers.

Why this module exists
----------------------
The library ships many single-purpose benchmark modules, but it had no single
entry point that answers the most basic question a reviewer asks first: *are
the Euler--Bernoulli beam, the Kirchhoff/Mindlin plate and the modal solver
actually solving the equations they claim to solve, to a stated tolerance?*
This module is that entry point.  It assembles a **ladder** of six checks with
machine-readable acceptance thresholds and returns them in one report object::

    L1  static beam      uniform load on a simply supported beam, 5 sections  < 0.5 %
    L1  static beam      end point load on a cantilever, tip deflection       < 1 %
    L2  static plate     simply supported square, Navier series (11x11)       < 1 %
    L2  static plate     clamped square, Timoshenko table / Ritz reference    < 2 %
    L3  modal            simply supported beam, first 5 frequencies          < 1 %
    L3  modal            simply supported square plate, first 3 frequencies  < 2 %

Every rung compares a **production solver path** against a **closed-form or
tabulated reference computed here**, so a regression in either the element
kernels or the reference formulas shows up as a failed rung:

===========  ===========================================================
rung         production path exercised
===========  ===========================================================
L1 beam      :func:`tensorfem.frame2d.solve_frame_static` (Hermite cubic
             frame element with its consistent load vector)
L2 plate     :func:`tensorfem.plate_mindlin.solve_plate_on_winkler_foundation`
             (Q4 SRI Mindlin element, ``winkler_k = 0``), thin limit
             ``a/h = 100`` where Mindlin degenerates to Kirchhoff
L3 beam      :func:`tensorfem.modal.simply_supported_beam_modes`
L3 plate     :func:`tensorfem.plate_vibration.solve_plate_vibration`
===========  ===========================================================

What the ladder does *not* prove is stated in the limitations of
``docs/basic-verification-ladder.md``: nodal exactness of the cubic beam
element, thin-plate scope of the plate rungs, and the reference truncation
bias of the 11x11 Navier sum (reported explicitly, not hidden).

Sign conventions
----------------
Transverse displacement, rotation ``theta = dw/dx`` and transverse load are
positive in the same direction; the stiffness and mass matrices are those of
:mod:`tensorfem.modal`, where the sign of ``w`` is immaterial because the
matrix is symmetric and no gravity/upward distinction enters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from tensorfem.modal import (
    simply_supported_beam_modes,
    simply_supported_exact_angular_frequency,
)

#: Acceptance thresholds of the ladder, keyed by criterion id.
CRITERIA: dict[str, float] = {
    "L1-SS-UDL": 5.0e-3,
    "L1-CANTILEVER-TIP": 1.0e-2,
    "L2-SS-NAVIER": 1.0e-2,
    "L2-CLAMPED": 2.0e-2,
    "L3-BEAM-MODAL": 1.0e-2,
    "L3-PLATE-MODAL": 2.0e-2,
}

#: Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells* (2nd ed.,
#: 1959), Table 35: ``w_max = 0.00126 q a^4 / D`` for a clamped square plate,
#: ``nu = 0.3``.  Tabulated to three significant digits only.
TIMOSHENKO_CLAMPED_COEFFICIENT = 0.00126


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    """One comparison of a computed quantity against an independent reference."""

    criterion: str
    description: str
    computed: float
    reference: float
    relative_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return abs(self.relative_error) <= self.tolerance

    def as_dict(self) -> dict[str, object]:
        return {
            "criterion": self.criterion,
            "description": self.description,
            "computed": self.computed,
            "reference": self.reference,
            "relative_error": self.relative_error,
            "relative_error_percent": 100.0 * self.relative_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BeamStaticResult:
    """Nodal solution of a uniform Euler--Bernoulli beam under static load."""

    length: float
    elements: int
    bending_stiffness: float
    nodes: torch.Tensor          # (elements+1,)
    deflections: torch.Tensor    # (elements+1,)
    rotations: torch.Tensor      # (elements+1,)

    def deflection(self, x: float) -> float:
        """Hermite interpolation of ``w`` at an arbitrary section ``x``."""
        if not 0.0 <= x <= self.length:
            raise ValueError("x must lie inside the beam")
        le = self.length / self.elements
        element = min(int(x / le), self.elements - 1)
        xi = (x - element * le) / le
        h1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
        h2 = le * (xi - 2.0 * xi**2 + xi**3)
        h3 = 3.0 * xi**2 - 2.0 * xi**3
        h4 = le * (-xi**2 + xi**3)
        return float(
            h1 * self.deflections[element]
            + h2 * self.rotations[element]
            + h3 * self.deflections[element + 1]
            + h4 * self.rotations[element + 1]
        )

    def section_curve(self, samples: int = 201) -> tuple[list[float], list[float]]:
        """Return ``(x, w)`` sampled uniformly along the beam."""
        xs = [self.length * i / (samples - 1) for i in range(samples)]
        return xs, [self.deflection(x) for x in xs]


@dataclass(frozen=True)
class LadderReport:
    """Full result of the three-level verification ladder."""

    checks: tuple[CheckResult, ...]
    details: dict[str, object] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def failed(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def as_dict(self) -> dict[str, object]:
        return {
            "ladder": "TensorFEM linear-elastic analytic verification ladder (W18-N01)",
            "criteria": dict(CRITERIA),
            "checks": [check.as_dict() for check in self.checks],
            "all_passed": self.all_passed,
            "failed_criteria": [check.criterion for check in self.failed()],
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# L1 -- Euler--Bernoulli beam statics
# ---------------------------------------------------------------------------

def euler_bernoulli_static_solve(
    length: float,
    elements: int,
    bending_stiffness: float,
    *,
    uniform_load: float = 0.0,
    tip_load: float = 0.0,
    support: str = "pinned-pinned",
) -> BeamStaticResult:
    """Solve a uniform Euler--Bernoulli beam under a uniform and/or tip load.

    The assembly is the *production* 2-D frame path of
    :func:`tensorfem.frame2d.solve_frame_static` (same element and consistent
    load vector the repo uses for hull girders), driven here with a straight
    horizontal chain of elements, ``E = 1`` and ``I = EI`` so that the element
    carries only the requested bending rigidity; the axial stiffness decouples
    and carries no load.  ``support="pinned-pinned"`` fixes the transverse
    displacement at both ends (axial displacement fixed at the first node only),
    ``support="clamped-free"`` fixes all three degrees of freedom at ``x = 0``.
    The transverse load acts along ``+y`` and the reported deflection is the
    ``+y`` displacement; only magnitudes are compared with the closed form.
    """
    if length <= 0.0 or elements < 1:
        raise ValueError("length and elements must be positive")
    if bending_stiffness <= 0.0:
        raise ValueError("bending_stiffness must be positive")
    if support not in ("pinned-pinned", "clamped-free"):
        raise ValueError("support must be 'pinned-pinned' or 'clamped-free'")

    from tensorfem.frame2d import FrameModel, solve_frame_static

    dtype = torch.float64
    nodes = torch.zeros((elements + 1, 2), dtype=dtype)
    nodes[:, 0] = torch.linspace(0.0, length, elements + 1, dtype=dtype)
    connectivity = torch.stack(
        (torch.arange(elements), torch.arange(1, elements + 1)), dim=1
    ).to(torch.int64)
    forces = torch.zeros(3 * (elements + 1), dtype=dtype)
    forces[3 * elements + 1] = tip_load
    fixed = [1, 3 * elements + 1]
    if support == "clamped-free":
        fixed = [0, 1, 2]
    else:
        fixed.append(0)  # remove the axial rigid-body mode
    model = FrameModel(
        nodes=nodes,
        elements=connectivity,
        young_modulus=torch.tensor(1.0, dtype=dtype),
        area=torch.tensor(1.0, dtype=dtype),
        inertia=torch.tensor(bending_stiffness, dtype=dtype),
        forces=forces,
        fixed_dofs=torch.tensor(sorted(fixed), dtype=torch.int64),
        distributed_load_y=torch.tensor(uniform_load, dtype=dtype),
    )
    solved = solve_frame_static(model)
    return BeamStaticResult(
        length=length,
        elements=elements,
        bending_stiffness=bending_stiffness,
        nodes=nodes[:, 0].clone(),
        deflections=solved.displacement[1::3].clone(),
        rotations=solved.displacement[2::3].clone(),
    )


def beam_interior_error(length: float, elements: int, EI: float, q: float,
                         samples: int = 24) -> float:
    """Max relative error of the uniform-load field at non-nodal sections.

    The cubic Hermite element is nodally exact for the quartic closed form, so
    this is the only place where the discretisation error of the *field* is
    visible.  The sample abscissae are deliberately offset from the nodes of
    any of the meshes used in the study.
    """
    beam = euler_bernoulli_static_solve(
        length, elements, EI, uniform_load=q, support="pinned-pinned"
    )
    xs = [length * (0.05 + 0.9 * (k + 0.37) / samples) for k in range(samples)]
    return max(
        abs(beam.deflection(x) / ss_udl_deflection(x, length, q, EI) - 1.0) for x in xs
    )


def ss_udl_deflection(x: float, length: float, uniform_load: float, EI: float) -> float:
    """Closed-form ``w(x) = q x (L^3 - 2 L x^2 + x^3) / (24 EI)``."""
    if not 0.0 <= x <= length:
        raise ValueError("x must lie inside the beam")
    return (
        uniform_load * x * (length**3 - 2.0 * length * x**2 + x**3) / (24.0 * EI)
    )


def ss_udl_max_deflection(length: float, uniform_load: float, EI: float) -> float:
    """Closed-form mid-span deflection ``5 q L^4 / (384 EI)``."""
    return 5.0 * uniform_load * length**4 / (384.0 * EI)


def cantilever_tip_deflection(length: float, tip_load: float, EI: float) -> float:
    """Closed-form tip deflection ``P L^3 / (3 EI)``."""
    return tip_load * length**3 / (3.0 * EI)


# ---------------------------------------------------------------------------
# L2 -- Kirchhoff plate statics (closed-form / tabulated references)
# ---------------------------------------------------------------------------

def navier_ss_deflection(
    x: float, y: float, side: float, rigidity: float, pressure: float, terms: int = 11
) -> float:
    """Navier double-sine series for a simply supported square plate.

    ``w = 16 q a^4 / (pi^6 D) sum_{m,n odd} sin(m pi x/a) sin(n pi y/a)
    / (m n (m^2 + n^2)^2)`` with ``m, n = 1, 3, ..., terms``.  ``terms = 11``
    is the truncation used by the ladder criterion; larger values are used to
    quantify the truncation bias of that reference.
    """
    if terms < 1 or terms % 2 == 0:
        raise ValueError("terms must be a positive odd number")
    if side <= 0.0 or rigidity <= 0.0:
        raise ValueError("side and rigidity must be positive")
    total = 0.0
    for m in range(1, terms + 1, 2):
        sm = math.sin(m * math.pi * x / side)
        for n in range(1, terms + 1, 2):
            total += (
                sm
                * math.sin(n * math.pi * y / side)
                / (m * n * (m * m + n * n) ** 2)
            )
    return 16.0 * pressure * side**4 / (math.pi**6 * rigidity) * total


def plate_rigidity(young: float, poisson: float, thickness: float) -> float:
    """``D = E h^3 / (12 (1 - nu^2))``."""
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


def clamped_square_coefficient_reference() -> float:
    """Timoshenko's tabulated ``w_max = 0.00126 q a^4 / D`` coefficient."""
    return TIMOSHENKO_CLAMPED_COEFFICIENT


# ---------------------------------------------------------------------------
# Ladder rungs
# ---------------------------------------------------------------------------

def _relative_error(computed: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("reference must be non-zero")
    return computed / reference - 1.0


def l1_beam_rungs(
    *,
    length: float = 2.0,
    young: float = 200.0e9,
    width: float = 0.05,
    height: float = 0.02,
    uniform_load: float = 1.0e4,
    tip_load: float = 1.0e3,
    elements: int = 8,
    sections: Sequence[float] = (0.10, 0.20, 0.30, 0.40, 0.50),
) -> tuple[tuple[CheckResult, ...], dict[str, object]]:
    """L1: static beam rungs (uniform load on 5 sections, cantilever tip load)."""
    inertia = width * height**3 / 12.0
    EI = young * inertia

    ss = euler_bernoulli_static_solve(
        length, elements, EI, uniform_load=uniform_load, support="pinned-pinned"
    )
    section_rows = []
    worst_row: dict[str, float] = {}
    for fraction in sections:
        x = fraction * length
        computed = ss.deflection(x)
        reference = ss_udl_deflection(x, length, uniform_load, EI)
        error = _relative_error(computed, reference)
        if not worst_row or abs(error) > abs(worst_row["relative_error"]):
            worst_row = {"computed": computed, "reference": reference,
                         "relative_error": error, "x": x, "x_over_L": fraction}
        section_rows.append(
            {
                "x_over_L": fraction,
                "x": x,
                "computed": computed,
                "reference": reference,
                "relative_error": error,
            }
        )
    checks = [
        CheckResult(
            "L1-SS-UDL",
            "simply supported beam, uniform load: worst of the 5 section"
            f" deflections (at x/L = {worst_row['x_over_L']:.2f})",
            computed=worst_row["computed"],
            reference=worst_row["reference"],
            relative_error=worst_row["relative_error"],
            tolerance=CRITERIA["L1-SS-UDL"],
        )
    ]

    cantilever = euler_bernoulli_static_solve(
        length, elements, EI, tip_load=tip_load, support="clamped-free"
    )
    tip_computed = cantilever.deflection(length)
    tip_reference = cantilever_tip_deflection(length, tip_load, EI)
    checks.append(
        CheckResult(
            "L1-CANTILEVER-TIP",
            "cantilever beam, end point load: tip deflection",
            computed=tip_computed,
            reference=tip_reference,
            relative_error=_relative_error(tip_computed, tip_reference),
            tolerance=CRITERIA["L1-CANTILEVER-TIP"],
        )
    )
    return tuple(checks), {
        "L1": {
            "model": {
                "length_m": length,
                "young_Pa": young,
                "section_m": [width, height],
                "second_moment_m4": inertia,
                "EI_Nm2": EI,
                "uniform_load_N_per_m": uniform_load,
                "tip_load_N": tip_load,
                "elements": elements,
            },
            "worst_section": {
                "x_over_L": worst_row["x_over_L"],
                "x": worst_row["x"],
                "relative_error": worst_row["relative_error"],
            },
            "sections": section_rows,
            "curves": {
                "x": ss.section_curve()[0],
                "w_fem": ss.section_curve()[1],
                "w_analytic": [
                    ss_udl_deflection(x, length, uniform_load, EI)
                    for x in ss.section_curve()[0]
                ],
            },
        }
    }


def l2_plate_rungs(
    *,
    side: float = 1.0,
    young: float = 200.0e9,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0e3,
    elements: int = 16,
    navier_terms: int = 11,
    converged_terms: int = 401,
) -> tuple[tuple[CheckResult, ...], dict[str, object]]:
    """L2: static plate rungs (SS Navier series, clamped Timoshenko/Ritz)."""
    from tensorfem.plate_mindlin import solve_plate_on_winkler_foundation
    from tensorfem.plate_uniform_load import clamped_uniform_ritz

    D = plate_rigidity(young, poisson, thickness)
    q = pressure
    navier_11 = navier_ss_deflection(
        0.5 * side, 0.5 * side, side, D, q, terms=navier_terms
    )
    navier_converged = navier_ss_deflection(
        0.5 * side, 0.5 * side, side, D, q, terms=converged_terms
    )

    ss = solve_plate_on_winkler_foundation(
        elements, side, young, poisson, thickness, 0.0, q, boundary="simply_supported"
    )
    ss_w = ss.displacement.view(-1, 3)[:, 0]
    ss_centre = _node_value(ss.nodes, ss_w, 0.5 * side, 0.5 * side)

    clamped = solve_plate_on_winkler_foundation(
        elements, side, young, poisson, thickness, 0.0, q, boundary="clamped"
    )
    clamped_w = clamped.displacement.view(-1, 3)[:, 0]
    clamped_centre = _node_value(clamped.nodes, clamped_w, 0.5 * side, 0.5 * side)
    ritz = clamped_uniform_ritz(
        side=side, young=young, poisson=poisson, thickness=thickness, pressure=q
    )
    ritz_centre = ritz.deflection(0.5 * side, 0.5 * side)

    checks = (
        CheckResult(
            "L2-SS-NAVIER",
            "simply supported square plate, uniform pressure: centre deflection"
            f" vs Navier series ({navier_terms}x{navier_terms} odd terms)",
            computed=ss_centre,
            reference=navier_11,
            relative_error=_relative_error(ss_centre, navier_11),
            tolerance=CRITERIA["L2-SS-NAVIER"],
        ),
        CheckResult(
            "L2-CLAMPED",
            "clamped square plate, uniform pressure: centre deflection vs"
            " Timoshenko's tabulated coefficient",
            computed=clamped_centre,
            reference=TIMOSHENKO_CLAMPED_COEFFICIENT * q * side**4 / D,
            relative_error=_relative_error(
                clamped_centre, TIMOSHENKO_CLAMPED_COEFFICIENT * q * side**4 / D
            ),
            tolerance=CRITERIA["L2-CLAMPED"],
        ),
    )
    return checks, {
        "L2": {
            "model": {
                "side_m": side,
                "young_Pa": young,
                "poisson": poisson,
                "thickness_m": thickness,
                "aspect_ratio_a_over_h": side / thickness,
                "pressure_Pa": pressure,
                "rigidity_Nm": D,
                "elements_per_side": elements,
                "element": "tensorfem.plate Q4 Mindlin SRI, winkler_k = 0",
            },
            "references": {
                "navier_11_terms": navier_11,
                "navier_converged_terms": converged_terms,
                "navier_converged": navier_converged,
                "navier_11_relative_bias": _relative_error(navier_11, navier_converged),
                "timoshenko_coefficient": TIMOSHENKO_CLAMPED_COEFFICIENT,
                "timoshenko_centre": TIMOSHENKO_CLAMPED_COEFFICIENT * q * side**4 / D,
                "ritz_centre": ritz_centre,
                "ritz_coefficient": ritz_centre / (q * side**4 / D),
                "clamped_vs_ritz_relative_error": _relative_error(
                    clamped_centre, ritz_centre
                ),
            },
            "fields": {
                "nodes": ss.nodes.tolist(),
                "w_simply_supported": ss_w.tolist(),
                "w_clamped": clamped_w.tolist(),
                "elements": ss.elements.tolist(),
                "side": side,
                "pressure": q,
                "rigidity": D,
            },
        }
    }


def _node_value(nodes: torch.Tensor, values: torch.Tensor, x: float, y: float) -> float:
    index = int(torch.argmin((nodes[:, 0] - x).abs() + (nodes[:, 1] - y).abs()))
    return float(values[index])


def modal_mac(fe_mode: Sequence[float], analytic: Sequence[float]) -> float:
    """Modal assurance criterion ``(phi . psi)^2 / (|phi|^2 |psi|^2)``."""
    a = torch.as_tensor(list(fe_mode), dtype=torch.float64)
    b = torch.as_tensor(list(analytic), dtype=torch.float64)
    denom = (a @ a) * (b @ b)
    if float(denom) <= 0.0:
        raise ValueError("mode shape has zero norm")
    return float((a @ b) ** 2 / denom)


def l3_modal_rungs(
    *,
    beam_length: float = 2.0,
    beam_young: float = 200.0e9,
    beam_width: float = 0.05,
    beam_height: float = 0.02,
    beam_density: float = 7850.0,
    beam_elements: int = 16,
    beam_modes: int = 5,
    plate_side: float = 1.0,
    plate_young: float = 200.0e9,
    plate_poisson: float = 0.3,
    plate_thickness: float = 0.01,
    plate_density: float = 7850.0,
    plate_elements: int = 16,
    plate_modes: int = 3,
) -> tuple[tuple[CheckResult, ...], dict[str, object]]:
    """L3: modal rungs (beam 5 frequencies, simply supported plate 3 modes)."""
    from tensorfem.plate_vibration import solve_plate_vibration

    area = beam_width * beam_height
    inertia = beam_width * beam_height**3 / 12.0
    EI = beam_young * inertia
    rho_a = beam_density * area

    beam = simply_supported_beam_modes(
        beam_length, beam_elements, EI, rho_a, modes=beam_modes
    )
    beam_rows = []
    worst = 0.0
    for mode in range(1, beam_modes + 1):
        computed = float(beam.angular_frequencies[mode - 1])
        reference = simply_supported_exact_angular_frequency(
            beam_length, EI, rho_a, mode=mode
        )
        error = _relative_error(computed, reference)
        worst = max(worst, abs(error))
        nodes = torch.linspace(0.0, beam_length, beam_elements + 1, dtype=torch.float64)
        shape = beam.modes[0::2, mode - 1]
        analytic = torch.sin(mode * math.pi * nodes / beam_length)
        beam_rows.append(
            {
                "mode": mode,
                "computed_rad_per_s": computed,
                "reference_rad_per_s": reference,
                "computed_hz": computed / (2.0 * math.pi),
                "reference_hz": reference / (2.0 * math.pi),
                "relative_error": error,
                "mac": modal_mac(shape, analytic),
                "x": nodes.tolist(),
                "shape": shape.tolist(),
                "shape_analytic": analytic.tolist(),
            }
        )
    checks = [
        CheckResult(
            "L3-BEAM-MODAL",
            "simply supported beam: worst of the first"
            f" {beam_modes} circular frequencies, omega_n = (n pi/L)^2 sqrt(EI/rho A)",
            computed=float(beam.angular_frequencies[-1]),
            reference=simply_supported_exact_angular_frequency(
                beam_length, EI, rho_a, mode=beam_modes
            ),
            relative_error=worst,
            tolerance=CRITERIA["L3-BEAM-MODAL"],
        )
    ]

    D = plate_rigidity(plate_young, plate_poisson, plate_thickness)
    plate = solve_plate_vibration(
        plate_elements,
        plate_elements,
        length=plate_side,
        young=plate_young,
        poisson=plate_poisson,
        thickness=plate_thickness,
        density=plate_density,
        boundary="simple",
        modes=plate_modes,
    )
    # Navier frequency reference, modes sorted ascending by (m^2 + n^2).
    pairs = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
    plate_rows = []
    worst = 0.0
    for index in range(plate_modes):
        m, n = pairs[index]
        reference = (
            math.pi**2
            * math.sqrt(D / (plate_density * plate_thickness))
            * ((m / plate_side) ** 2 + (n / plate_side) ** 2)
        )
        computed = float(plate.angular_frequencies[index])
        error = _relative_error(computed, reference)
        worst = max(worst, abs(error))
        nodes = plate.nodes
        mode_shape = plate.modes[0::3, index]
        analytic = torch.sin(m * math.pi * nodes[:, 0] / plate_side) * torch.sin(
            n * math.pi * nodes[:, 1] / plate_side
        )
        # A square plate has degenerate (m,n)/(n,m) pairs; the FE eigenvector is
        # an arbitrary rotation inside that two-dimensional eigenspace, so the
        # single-shape MAC is not meaningful there.  Report both numbers: the
        # scalar MAC against one basis shape and the fraction of the FE mode
        # captured by the whole degenerate subspace.
        partner = torch.sin(n * math.pi * nodes[:, 0] / plate_side) * torch.sin(
            m * math.pi * nodes[:, 1] / plate_side
        )
        basis = torch.stack([analytic, partner], dim=1)
        coefficients = torch.linalg.lstsq(basis, mode_shape).solution
        projected = basis @ coefficients
        capture = float((projected @ projected) / (mode_shape @ mode_shape))
        plate_rows.append(
            {
                "mode": index + 1,
                "m": m,
                "n": n,
                "computed_rad_per_s": computed,
                "reference_rad_per_s": reference,
                "computed_hz": computed / (2.0 * math.pi),
                "reference_hz": reference / (2.0 * math.pi),
                "relative_error": error,
                "mac": modal_mac(mode_shape, analytic),
                "degenerate_with": None if m == n else [n, m],
                "subspace_capture": capture,
            }
        )
    checks.append(
        CheckResult(
            "L3-PLATE-MODAL",
            "simply supported square plate: worst of the first"
            f" {plate_modes} circular frequencies vs Navier mode (m,n)",
            computed=float(plate.angular_frequencies[-1]),
            reference=plate_rows[-1]["reference_rad_per_s"],
            relative_error=worst,
            tolerance=CRITERIA["L3-PLATE-MODAL"],
        )
    )
    return tuple(checks), {
        "L3": {
            "beam": {
                "model": {
                    "length_m": beam_length,
                    "young_Pa": beam_young,
                    "section_m": [beam_width, beam_height],
                    "density_kg_per_m3": beam_density,
                    "elements": beam_elements,
                },
                "modes": beam_rows,
            },
            "plate": {
                "model": {
                    "side_m": plate_side,
                    "young_Pa": plate_young,
                    "poisson": plate_poisson,
                    "thickness_m": plate_thickness,
                    "density_kg_per_m3": plate_density,
                    "elements_per_side": plate_elements,
                },
                "modes": plate_rows,
            },
        }
    }


# ---------------------------------------------------------------------------
# Full ladder
# ---------------------------------------------------------------------------

def run_verification_ladder(
    *,
    plate_elements: int = 16,
    plate_elements_clamped: int | None = None,
    plate_modal_elements: int = 16,
    beam_elements: int = 16,
    sections: Sequence[float] = (0.10, 0.20, 0.30, 0.40, 0.50),
    progress: Callable[[str], None] | None = None,
) -> LadderReport:
    """Run all six ladder checks and return one report object.

    The plate rungs are the expensive part (dense Q4 assembly and solve), so the
    mesh is exposed: with ``plate_elements = 16`` the whole ladder takes about a
    minute on a laptop, which is the setting used by the test suite.
    """
    def _say(message: str) -> None:
        if progress is not None:
            progress(message)

    _say("L1 beam statics")
    l1_checks, l1_details = l1_beam_rungs(elements=beam_elements, sections=sections)
    _say("L2 plate statics")
    l2_checks, l2_details = l2_plate_rungs(
        elements=plate_elements_clamped or plate_elements
    )
    _say("L3 modal")
    l3_checks, l3_details = l3_modal_rungs(
        beam_elements=beam_elements, plate_elements=plate_modal_elements
    )
    details: dict[str, object] = {}
    details.update(l1_details)
    details.update(l2_details)
    details.update(l3_details)
    return LadderReport(checks=l1_checks + l2_checks + l3_checks, details=details)


__all__ = [
    "CRITERIA",
    "TIMOSHENKO_CLAMPED_COEFFICIENT",
    "BeamStaticResult",
    "CheckResult",
    "LadderReport",
    "beam_interior_error",
    "cantilever_tip_deflection",
    "clamped_square_coefficient_reference",
    "euler_bernoulli_static_solve",
    "l1_beam_rungs",
    "l2_plate_rungs",
    "l3_modal_rungs",
    "modal_mac",
    "navier_ss_deflection",
    "plate_rigidity",
    "run_verification_ladder",
    "ss_udl_deflection",
    "ss_udl_max_deflection",
]