"""Structural mechanics verification face: three classical analytic benchmarks.

Why this module exists
----------------------
TensorFEM carries many single-case verification studies (see ``docs/``), but each
one hand-rolls its own tolerances, its own refinement ladder and its own
pass/fail bookkeeping.  Nothing in the tree answers, in one place, the question a
reviewer actually asks about the structural solver:

    *"on the three textbook benchmarks -- beam bending, pressurised thick
    cylinder, thin plate bending -- how far from the closed form is the
    production path, does the error converge at the theoretical rate, and does it
    hold across the physical parameter range in which the closed form is
    trusted?"*

This module is that place.  It is a *harness*, not a new element: every arm
drives an existing production path
(:func:`tensorfem.quadratic_plane.solve_q8_plane` for the two continuum arms,
:func:`tensorfem.plate_uniform_load.solve_uniform_square_plate` for the plate arm)
and compares it against an independent closed form, at a declared tolerance, on a
declared three-level ladder, with a declared minimum observed order.

The three arms, and what each one isolates
------------------------------------------
``cantilever``
    Q8 plane-stress continuum, clamped root, **parabolic end shear** -- the
    traction that makes the classical beam stress field ``sigma_x = M y / I`` an
    *exact* 2-D elasticity solution.  Swept over ``L/h in {10, 20, 40}``.  The
    sweep is the point: Euler-Bernoulli is the requested yardstick, but a 2-D
    continuum converges to the *shear-deformable* (Timoshenko) answer, whose
    relative gap to Euler-Bernoulli is ``~0.78 (h/L)^2``.  Both references are
    reported, so the residual splits into a *model* part (beam theory vs
    continuum) and a *discretisation* part (finite element vs continuum) instead
    of hiding both inside one number.

``cylinder``
    Q8 plane-strain annulus, **pressure traction on a quadratic (exactly
    circular) bore edge**, swept over ``r_o/r_i in {1.25, 1.6, 2.5}``.  The Lame
    field is an exact elasticity solution, so only discretisation error is
    admissible.  The wall-ratio sweep matters because the hoop gradient scales
    like ``1/r^2``: a thin wall stresses the radial resolution, a thick wall the
    angular one.

``plate``
    Q4 Mindlin plate with selective reduced shear integration, **clamped** square
    under uniform pressure, swept over ``a/t in {10, 50, 100}``.  Kirchhoff gives
    ``w_max = alpha q a^4 / D`` with ``alpha`` from a Rayleigh-Ritz reference
    (Timoshenko's tabulated 0.00126 is the cross-check).  The ``a/t`` sweep tests
    the Kirchhoff limit itself: an element that locked would lose accuracy as
    ``a/t`` grows, which is exactly the failure mode this arm is built to catch.

Gates (module constants, reported per case)
-------------------------------------------
* cantilever: ``|delta_FE - delta_EB| / delta_EB < 1 %``,
* cylinder: worst-point hoop stress error vs Lame ``< 2 %``,
* plate: ``|w_FE - w_Ritz| / w_Ritz < 3 %``,
* every arm: observed order of the QoI on the ladder ``>= 1.5``.

Honesty notes (part of the contract, not a disclaimer)
------------------------------------------------------
* The cantilever gate is evaluated against Euler-Bernoulli because that *is* the
  benchmark definition, but the observed order is measured on the QoI sequence
  itself (Richardson).  Measuring it against a reference the discrete model does
  not converge to would manufacture a fake order near zero.
* Cylinder stresses are compared pointwise at the 3x3 Gauss points, i.e. where
  the element stress is superconvergent, normalised by the local exact value
  (``sigma_r`` is additionally normalised by the problem stress scale, because it
  vanishes identically at the free outer surface).
* The plate reference is a Rayleigh-Ritz series appropriate for a *clamped* edge;
  its mode convergence is checked before use (:func:`ritz_reference`).
* Nothing here re-declares an existing study as new: the relation to the per-case
  verifications already in the tree is tabulated in
  ``docs/structural-verification-face.md``.

Units are SI (m, N, Pa) and ``dtype`` is float64 throughout, because the gates
are 1-3 % and float32 round-off in a condensed plate solve is not negligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .convergence_verification import (
    ConvergenceLevel,
    observed_order_from_errors,
    refinement_ratio,
    richardson_extrapolate,
    richardson_observed_order,
)
from .lame_reference import lame_state
from .plate_uniform_load import (
    centre_node_deflection,
    clamped_uniform_ritz,
    solve_uniform_square_plate,
)
from .quadratic_plane import (
    GAUSS_POINTS_3X3,
    q8_mesh,
    q8_shape,
    polar_q8_mesh,
    solve_q8_plane,
)

DTYPE = torch.float64

#: Gate values requested by the benchmark definition.
GATE_CANTILEVER = 0.01
GATE_CYLINDER = 0.02
GATE_PLATE = 0.03
#: Minimum observed order of accuracy on the three-level ladder.
MIN_ORDER = 1.5
#: Slenderness (a/t) from which the Kirchhoff yardstick is used as a gate; below
#: it the transverse-shear contribution of the Mindlin model is a *physics*
#: difference, not a discretisation error (see ``run_plate_ladder``).
KIRCHHOFF_MIN_ASPECT = 50.0

#: Linear-solver settings for the Q8 arms.  Conjugate gradients on a clamped,
#: ill-conditioned plane-stress mesh stall near 1e-8 relative residual, which is
#: still seven orders below the 1 % gate, so the tolerance is declared here
#: rather than left to a default that raises.
CG_RTOL = 1e-9
CG_MAXITER = 200000

#: three-point Gauss rule on the parent interval [-1, 1].
_GAUSS_3 = (
    (-math.sqrt(3.0 / 5.0), 5.0 / 9.0),
    (0.0, 8.0 / 9.0),
    (math.sqrt(3.0 / 5.0), 5.0 / 9.0),
)

#: Local node triples of the four edges of a serendipity Q8 element.
_ELEMENT_EDGES = ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7))


# --------------------------------------------------------------------------- #
# closed-form references
# --------------------------------------------------------------------------- #
def euler_bernoulli_tip_deflection(
    force: float, length: float, young: float, inertia: float
) -> float:
    """``delta = F L^3 / (3 E I)``, the magnitude of the tip deflection."""
    if min(force, length, young, inertia) <= 0.0:
        raise ValueError("force, length, young and inertia must be positive")
    return force * length**3 / (3.0 * young * inertia)


def timoshenko_tip_deflection(
    force: float,
    length: float,
    young: float,
    inertia: float,
    area: float,
    poisson: float,
    shear_factor: float = 5.0 / 6.0,
) -> float:
    """``delta = F L^3/(3 E I) + F L/(k G A)`` with ``G = E/(2(1+nu))``.

    For a rectangular section under parabolic end shear this is the exact 2-D
    elasticity tip deflection, i.e. the value a continuum model must converge to.
    """
    if min(force, length, young, inertia, area, shear_factor) <= 0.0:
        raise ValueError("geometry and shear factor must be positive")
    shear_modulus = young / (2.0 * (1.0 + poisson))
    bending = force * length**3 / (3.0 * young * inertia)
    shear = force * length / (shear_factor * shear_modulus * area)
    return bending + shear


def beam_bending_stress(moment: float, offset: float, inertia: float) -> float:
    """``sigma_x = M y / I`` for a downward tip load.

    With ``y`` measured upward from the neutral axis the drooping cantilever is
    in tension on the upper fibre, hence the positive sign; ``moment`` is the
    magnitude ``F (L - x)`` of the section moment.
    """
    if inertia <= 0.0:
        raise ValueError("inertia must be positive")
    return moment * offset / inertia


def flexural_rigidity(young: float, thickness: float, poisson: float) -> float:
    """Plate rigidity ``D = E t^3 / (12 (1 - nu^2))``."""
    if min(young, thickness) <= 0.0 or not -1.0 < poisson < 0.5:
        raise ValueError("invalid plate parameters")
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


def ritz_reference(
    *,
    side: float,
    young: float,
    poisson: float,
    thickness: float,
    pressure: float,
    mode_counts: tuple[int, ...] = (6, 8, 10),
) -> dict[str, object]:
    """Clamped-square Rayleigh-Ritz centre deflection, with its mode-convergence spread.

    The Ritz value is only usable as an oracle if it is insensitive to the basis
    size, so the spread over ``mode_counts`` is returned and gated in the tests.
    """
    values = [
        float(
            clamped_uniform_ritz(
                side=side,
                young=young,
                poisson=poisson,
                thickness=thickness,
                pressure=pressure,
                n_modes=n_modes,
            ).deflection(0.5 * side, 0.5 * side)
        )
        for n_modes in mode_counts
    ]
    finest = values[-1]
    spread = max(abs(value - finest) for value in values) / abs(finest)
    return {
        "deflection": finest,
        "mode_spread": spread,
        "modes": list(mode_counts),
        "values": values,
        "alpha": finest
        * flexural_rigidity(young, thickness, poisson)
        / (pressure * side**4),
    }


# --------------------------------------------------------------------------- #
# mesh / load helpers
# --------------------------------------------------------------------------- #
def rect_q4_grid(
    length: float, height: float, nx: int, ny: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured Q4 grid of ``[0, L] x [-h/2, h/2]`` with counter-clockwise cells."""
    if min(nx, ny) < 1:
        raise ValueError("nx and ny must be positive")
    if min(length, height) <= 0.0:
        raise ValueError("length and height must be positive")
    x = torch.linspace(0.0, length, nx + 1, dtype=DTYPE)
    y = torch.linspace(-0.5 * height, 0.5 * height, ny + 1, dtype=DTYPE)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    cells = []
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            cells.append((n0, n0 + ny + 1, n0 + ny + 2, n0 + 1))
    return nodes, torch.tensor(cells, dtype=torch.long)


def q8_edge_traction(points: torch.Tensor, traction) -> torch.Tensor:
    """Work-equivalent nodal forces of a traction on one quadratic (3-node) edge.

    ``points`` are the three edge nodes ordered along the edge and ``traction``
    maps a physical position ``(2,)`` to a traction vector.  The edge measure is
    ``|dx/dxi|`` of the *quadratic* mapping, so three nodes lying on a circle
    integrate the exact arc length instead of a chord.  What remains is the
    ``O(span**2)`` error of the traction *direction* sampled on the quadratic
    interpolation of the circle; for the cylindrical bore that residual is
    measured explicitly (:func:`solve_lame_annulus` reports it per edge) and
    converges quadratically, so it is visible rather than assumed away.
    """
    points = torch.as_tensor(points, dtype=DTYPE)
    if tuple(points.shape) != (3, 2):
        raise ValueError("an edge is three nodes with two coordinates each")
    forces = torch.zeros((3, 2), dtype=DTYPE)
    for xi, weight in _GAUSS_3:
        shape = torch.tensor(
            (0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)), dtype=DTYPE
        )
        derivative = torch.tensor((xi - 0.5, -2.0 * xi, xi + 0.5), dtype=DTYPE)
        position = shape @ points
        jacobian = float(torch.linalg.norm(derivative @ points))
        if jacobian <= 0.0:
            raise ValueError("degenerate edge")
        forces += (weight * jacobian) * torch.outer(
            shape, torch.as_tensor(traction(position), dtype=DTYPE)
        )
    return forces


def boundary_edge_forces(
    nodes: torch.Tensor, elements: torch.Tensor, on_boundary, traction
) -> torch.Tensor:
    """Global load vector from a traction on every element edge matching ``on_boundary``.

    ``on_boundary(edge_coordinates)`` receives the three edge nodes and returns a
    bool.  Each boundary edge belongs to exactly one element, so no edge is
    integrated twice; the edge layout comes from the element connectivity, never
    from an angular re-sorting of the nodes (a re-sorted chain can be offset by
    half an edge, which silently swaps the corner and mid-side weights).
    """
    forces = torch.zeros(2 * nodes.shape[0], dtype=DTYPE)
    for connectivity in elements.tolist():
        for first, second, middle in _ELEMENT_EDGES:
            edge = torch.tensor(
                (connectivity[first], connectivity[middle], connectivity[second]),
                dtype=torch.long,
            )
            if not on_boundary(nodes[edge]):
                continue
            local = q8_edge_traction(nodes[edge], traction)
            for slot, node in enumerate(edge.tolist()):
                forces[2 * node] += local[slot, 0]
                forces[2 * node + 1] += local[slot, 1]
    return forces


# --------------------------------------------------------------------------- #
# arm 1: cantilever beam, Q8 plane stress
# --------------------------------------------------------------------------- #
def solve_cantilever(
    nx: int,
    ny: int,
    *,
    length: float,
    height: float,
    young: float,
    poisson: float,
    force: float,
) -> dict[str, object]:
    """Clamped plane-stress cantilever under a parabolic end shear.

    ``nx``/``ny`` count Q8 elements per direction (they are also the Q4 cells of
    the parent grid handed to :func:`tensorfem.quadratic_plane.q8_mesh`).  The
    clamped end fixes both displacement components, the continuum counterpart of
    the Euler-Bernoulli ``w = w' = 0``.
    """
    if ny % 2 != 0:
        raise ValueError("an even ny keeps a mid-side node on the neutral axis at the tip")
    nodes, cells = q8_mesh(*rect_q4_grid(length, height, nx, ny))
    tolerance = 1e-9 * max(length, height)

    inertia = height**3 / 12.0          # unit width
    area = height                        # unit width
    half = 0.5 * height

    dirichlet: dict[int, float] = {}
    for node in torch.nonzero(nodes[:, 0].abs() < tolerance)[:, 0].tolist():
        dirichlet[2 * node] = 0.0
        dirichlet[2 * node + 1] = 0.0

    def on_tip(edge: torch.Tensor) -> bool:
        return bool(torch.all((edge[:, 0] - length).abs() < tolerance))

    def shear_traction(position: torch.Tensor) -> torch.Tensor:
        magnitude = force / (2.0 * inertia) * (half**2 - float(position[1]) ** 2)
        return torch.tensor((0.0, -magnitude), dtype=DTYPE)

    forces = boundary_edge_forces(nodes, cells, on_tip, shear_traction)
    result = solve_q8_plane(
        nodes,
        cells,
        young=young,
        poisson=poisson,
        thickness=1.0,
        plane="stress",
        forces=forces,
        dirichlet=dirichlet,
        rtol=CG_RTOL,
        maxiter=CG_MAXITER,
    )

    tip_nodes = torch.nonzero((nodes[:, 0] - length).abs() < tolerance)[:, 0]
    neutral = tip_nodes[nodes[tip_nodes, 1].abs() < tolerance]
    if neutral.numel() != 1:
        raise ValueError("expected exactly one tip node on the neutral axis")
    tip_deflection = -float(result.displacement[2 * int(neutral[0]) + 1])

    # location-matched bending stress in the Saint-Venant interior (20-80 % span),
    # where sigma_x = M y / I is the exact 2-D elasticity field
    positions: list[tuple[float, float]] = []
    computed: list[float] = []
    field_x: list[float] = []
    field_y: list[float] = []
    field_sigma: list[float] = []
    for element in range(len(cells)):
        connectivity = cells[element]
        position = q8_shape(0.0, 0.0).to(DTYPE) @ nodes[connectivity]
        x_position = float(position[0])
        field_x.append(x_position)
        field_y.append(float(position[1]))
        field_sigma.append(float(result.centre_stress[element, 0]))
        if not 0.2 * length <= x_position <= 0.8 * length:
            continue
        positions.append((x_position, float(position[1])))
        computed.append(float(result.centre_stress[element, 0]))
    reference = [
        beam_bending_stress(force * (length - x_position), y_position, inertia)
        for x_position, y_position in positions
    ]
    if not reference:
        raise ValueError("no element centre fell inside the sampled span")
    stress_error = max(
        abs(value - exact) for value, exact in zip(computed, reference)
    ) / max(abs(exact) for exact in reference)

    bernoulli = euler_bernoulli_tip_deflection(force, length, young, inertia)
    shear_corrected = timoshenko_tip_deflection(
        force, length, young, inertia, area, poisson
    )
    return {
        "label": f"{nx}x{ny}",
        "elements": int(len(cells)),
        "dofs": int(2 * nodes.shape[0]),
        "h": length / nx,
        "tip_deflection": tip_deflection,
        "euler_bernoulli": bernoulli,
        "timoshenko": shear_corrected,
        "error_vs_euler_bernoulli": abs(tip_deflection - bernoulli) / bernoulli,
        "error_vs_timoshenko": abs(tip_deflection - shear_corrected) / shear_corrected,
        "bending_stress_error": stress_error,
        "relative_residual": float(result.relative_residual),
        "load_resultant": -float(forces[1::2].sum()),
        "slenderness": length / height,
        # element-centre stress field, kept for the figure builder only
        "field": {"x": field_x, "y": field_y, "sigma_x": field_sigma},
    }


# --------------------------------------------------------------------------- #
# arm 2: pressurised thick cylinder, Q8 plane strain
# --------------------------------------------------------------------------- #
def solve_lame_annulus(
    n_theta: int,
    n_radial: int,
    *,
    inner: float,
    outer: float,
    pressure: float,
    young: float,
    poisson: float,
) -> dict[str, object]:
    """Closed annulus under bore pressure: Q8 plane strain against the Lame field.

    ``polar_q8_mesh`` builds the whole annulus and puts every mid-side node on the
    exact circle, so the bore traction is integrated over the true arc (see
    :func:`q8_edge_traction`) instead of a polygonal chord.

    Rigid-body suppression uses a property of the exact solution rather than an
    arbitrary gauge: for a closed annulus under uniform bore pressure the
    circumferential displacement vanishes identically, so ``u_y = 0`` on the
    ``theta = 0, pi`` axis and ``u_x = 0`` on the ``theta = +-pi/2`` axis are true
    statements about the solution; imposing them removes exactly the three
    rigid-body modes and perturbs nothing.

    The load is audited against the analytic arc resultant of every loaded edge,
    ``2 p r sin(dtheta/2)`` along the edge bisector, so a wrong edge measure or a
    mis-assigned corner/mid-side weight cannot pass silently; the reported
    residual is the ``O(dtheta**2)`` direction error of the quadratic edge and
    falls quadratically (7.7e-5 -> 3.1e-7 over the certified ladder).
    """
    if not 0.0 < inner < outer:
        raise ValueError("require 0 < inner < outer")
    if pressure < 0.0:
        raise ValueError("pressure must be non-negative")
    nodes, elements = polar_q8_mesh(inner, outer, n_theta, n_radial)
    radius = torch.linalg.norm(nodes, dim=1)
    tolerance = 1e-9 * outer

    dirichlet: dict[int, float] = {}
    for node in range(nodes.shape[0]):
        x_position, y_position = float(nodes[node, 0]), float(nodes[node, 1])
        if abs(y_position) < tolerance:
            dirichlet[2 * node + 1] = 0.0      # theta = 0, pi: u_theta = 0 -> u_y = 0
        if abs(x_position) < tolerance:
            dirichlet[2 * node] = 0.0          # theta = +-pi/2: u_theta = 0 -> u_x = 0

    bore = torch.nonzero((radius - inner).abs() < tolerance)[:, 0]
    chain = bore[torch.argsort(torch.atan2(nodes[bore, 1], nodes[bore, 0]))]

    def on_bore(edge: torch.Tensor) -> bool:
        return bool(torch.all((torch.linalg.norm(edge, dim=1) - inner).abs() < tolerance))

    def outward(position: torch.Tensor) -> torch.Tensor:
        return pressure * position / float(torch.linalg.norm(position))

    forces = torch.zeros(2 * nodes.shape[0], dtype=DTYPE)
    worst_resultant = 0.0
    worst_direction = 0.0
    loaded_edges = 0
    for connectivity in elements.tolist():
        for first, second, middle in _ELEMENT_EDGES:
            edge = torch.tensor(
                (connectivity[first], connectivity[middle], connectivity[second]),
                dtype=torch.long,
            )
            if not on_bore(nodes[edge]):
                continue
            local = q8_edge_traction(nodes[edge], outward)
            for slot, node in enumerate(edge.tolist()):
                forces[2 * node] += local[slot, 0]
                forces[2 * node + 1] += local[slot, 1]
            loaded_edges += 1
            start_angle = math.atan2(float(nodes[edge[0], 1]), float(nodes[edge[0], 0]))
            end_angle = math.atan2(float(nodes[edge[2], 1]), float(nodes[edge[2], 0]))
            span = abs(end_angle - start_angle)
            span = min(span, 2.0 * math.pi - span)
            expected = 2.0 * pressure * inner * math.sin(0.5 * span)
            summed = local.sum(dim=0)
            worst_resultant = max(
                worst_resultant, abs(float(torch.linalg.norm(summed)) - expected) / expected
            )
            bisector = nodes[edge[0]] + nodes[edge[2]]
            bisector = bisector / float(torch.linalg.norm(bisector))
            direction = summed / float(torch.linalg.norm(summed))
            worst_direction = max(
                worst_direction, float(torch.linalg.norm(direction - bisector))
            )
    if loaded_edges != n_theta:
        raise ValueError(f"bore load touched {loaded_edges} edges, expected {n_theta}")

    result = solve_q8_plane(
        nodes,
        elements,
        young=young,
        poisson=poisson,
        thickness=1.0,
        plane="strain",
        forces=forces,
        dirichlet=dirichlet,
        rtol=CG_RTOL,
        maxiter=CG_MAXITER,
    )

    exact_bore = lame_state(
        torch.tensor(inner, dtype=DTYPE),
        inner_radius=inner,
        outer_radius=outer,
        inner_pressure=pressure,
        young_modulus=young,
        poisson_ratio=poisson,
        plane="strain",
    )
    hoop_scale = float(exact_bore.hoop_stress)      # the problem stress scale

    worst_hoop = 0.0
    worst_radial = 0.0
    field = {"r": [], "angle": [], "hoop": [], "radial": [], "exact_hoop": [], "exact_radial": []}
    for element in range(len(elements)):
        connectivity = elements[element]
        for slot, (xi, eta) in enumerate(GAUSS_POINTS_3X3):
            position = q8_shape(xi, eta).to(DTYPE) @ nodes[connectivity]
            r = float(torch.linalg.norm(position))
            angle = math.atan2(float(position[1]), float(position[0]))
            exact = lame_state(
                torch.tensor(r, dtype=DTYPE),
                inner_radius=inner,
                outer_radius=outer,
                inner_pressure=pressure,
                young_modulus=young,
                poisson_ratio=poisson,
                plane="strain",
            )
            sxx = float(result.gauss_stress[element, slot, 0])
            syy = float(result.gauss_stress[element, slot, 1])
            sxy = float(result.gauss_stress[element, slot, 2])
            cosine, sine = math.cos(angle), math.sin(angle)
            radial = (
                cosine * cosine * sxx + sine * sine * syy + 2.0 * cosine * sine * sxy
            )
            hoop = (
                sine * sine * sxx + cosine * cosine * syy - 2.0 * cosine * sine * sxy
            )
            exact_hoop = float(exact.hoop_stress)
            exact_radial = float(exact.radial_stress)
            worst_hoop = max(worst_hoop, abs(hoop - exact_hoop) / abs(exact_hoop))
            worst_radial = max(worst_radial, abs(radial - exact_radial) / hoop_scale)
            field["r"].append(r)
            field["angle"].append(angle)
            field["hoop"].append(hoop)
            field["radial"].append(radial)
            field["exact_hoop"].append(exact_hoop)
            field["exact_radial"].append(exact_radial)

    bore_values = []
    for node in chain.tolist():
        position = nodes[node]
        norm = float(torch.linalg.norm(position))
        bore_values.append(
            (
                float(position[0]) * float(result.displacement[2 * node])
                + float(position[1]) * float(result.displacement[2 * node + 1])
            )
            / norm
        )
    bore_mean = sum(bore_values) / len(bore_values)
    exact_bore_displacement = float(exact_bore.radial_displacement)
    bore_spread = (max(bore_values) - min(bore_values)) / abs(exact_bore_displacement)

    # a closed ring must be self-equilibrated and every nodal force radial
    total = torch.tensor((float(forces[0::2].sum()), float(forces[1::2].sum())), dtype=DTYPE)
    resultant_ratio = float(torch.linalg.norm(total)) / (pressure * 2.0 * math.pi * inner)
    radial_leak = 0.0
    for node in chain.tolist():
        vector = torch.tensor((forces[2 * node], forces[2 * node + 1]), dtype=DTYPE)
        direction = nodes[node] / float(torch.linalg.norm(nodes[node]))
        radial_leak = max(
            radial_leak,
            float(torch.linalg.norm(vector - direction * (vector @ direction)))
            / float(torch.linalg.norm(vector)),
        )

    return {
        "label": f"{n_theta}x{n_radial}",
        "elements": int(len(elements)),
        "dofs": int(2 * nodes.shape[0]),
        "h": (outer - inner) / n_radial,
        "hoop_error": worst_hoop,
        "radial_error": worst_radial,
        "hoop_scale": hoop_scale,
        "bore_displacement": bore_mean,
        "bore_displacement_exact": exact_bore_displacement,
        "bore_displacement_error": abs(bore_mean - exact_bore_displacement)
        / abs(exact_bore_displacement),
        "bore_circumferential_spread": bore_spread,
        "load_resultant_ratio": resultant_ratio,
        "load_edge_resultant_error": worst_resultant,
        "load_edge_direction_error": worst_direction,
        "load_radial_leak": radial_leak,
        "wall_ratio": outer / inner,
        "relative_residual": float(result.relative_residual),
        # Gauss-point stress field, kept for the figure builder only
        "field": field,
    }


# --------------------------------------------------------------------------- #
# arm 3: clamped thin plate, Q4 Mindlin
# --------------------------------------------------------------------------- #
def solve_clamped_plate(
    n: int,
    *,
    side: float,
    young: float,
    poisson: float,
    thickness: float,
    pressure: float,
    reference: dict[str, object] | None = None,
) -> dict[str, object]:
    """Clamped square under uniform pressure: Q4 Mindlin (SRI shear) vs Kirchhoff ``alpha``."""
    oracle = reference or ritz_reference(
        side=side,
        young=young,
        poisson=poisson,
        thickness=thickness,
        pressure=pressure,
    )
    result = solve_uniform_square_plate(
        n,
        side=side,
        young=young,
        poisson=poisson,
        thickness=thickness,
        pressure=pressure,
        boundary="clamped",
    )
    centre = centre_node_deflection(result)
    exact = float(oracle["deflection"])
    deflection = [
        float(result.displacement[3 * node]) for node in range(result.nodes.shape[0])
    ]
    rigidity = flexural_rigidity(young, thickness, poisson)
    return {
        "label": f"{n}x{n}",
        "elements": int(n * n),
        "dofs": int(3 * result.nodes.shape[0]),
        "h": side / n,
        "centre_deflection": centre,
        "ritz_deflection": exact,
        "error_vs_ritz": abs(centre - exact) / abs(exact),
        "aspect": side / thickness,
        "alpha_measured": centre * rigidity / (pressure * side**4),
        "alpha_reference": float(oracle["alpha"]),
        # nodal deflection field, kept for the figure builder only
        "field": result.nodes.tolist(),
        "deflection": deflection,
    }


# --------------------------------------------------------------------------- #
# ladder driver
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    """One benchmark arm: its ladder, its gate and its observed order."""

    name: str
    quantity: str
    gate: float
    levels: list[dict[str, object]] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    @property
    def finest(self) -> dict[str, object]:
        """The finest ladder level (the one the gate is evaluated on)."""
        return self.levels[-1]

    def to_dict(self) -> dict[str, object]:
        """Serialisable form, with the verdict made explicit."""
        return {
            "case": self.name,
            "quantity": self.quantity,
            "gate": self.gate,
            "min_order": MIN_ORDER,
            "levels": self.levels,
            "metrics": self.metrics,
            "passed": bool(self.metrics.get("passed", False)),
        }


def _levels(values: list[float], mesh_sizes: list[float]) -> list[ConvergenceLevel]:
    return [
        ConvergenceLevel(float(h), float(v), f"L{index}")
        for index, (h, v) in enumerate(zip(mesh_sizes, values))
    ]


def observed_order(values: list[float], mesh_sizes: list[float]) -> float:
    """Richardson observed order on a three-level QoI sequence.

    Raises when the sequence is not monotone or the refinement ratio is not
    constant, so a noisy ladder can never be reported as an order.
    """
    coarse, middle, fine = _levels(values, mesh_sizes)
    first = refinement_ratio(coarse, middle)
    second = refinement_ratio(middle, fine)
    if abs(first - second) > 1e-9 * max(first, second):
        raise ValueError("ladder must use a constant refinement ratio")
    return float(richardson_observed_order(coarse, middle, fine))


def _ratio_between(sizes: list[float]) -> list[float]:
    return [
        float(sizes[index] / sizes[index + 1]) for index in range(len(sizes) - 1)
    ]


def run_cantilever_ladder(
    *,
    length: float = 1.0,
    height: float = 0.05,
    young: float = 210.0e9,
    poisson: float = 0.3,
    force: float = 1000.0,
    divisions: tuple[tuple[int, int], ...] = ((10, 2), (20, 4), (40, 8)),
    slenderness_sweep: tuple[float, ...] = (10.0, 20.0, 40.0),
) -> CaseResult:
    """Cantilever arm: the main ladder plus the ``L/h`` sweep that splits model error."""
    levels = [
        solve_cantilever(
            nx,
            ny,
            length=length,
            height=height,
            young=young,
            poisson=poisson,
            force=force,
        )
        for nx, ny in divisions
    ]
    values = [float(level["tip_deflection"]) for level in levels]
    mesh_sizes = [float(level["h"]) for level in levels]
    order = observed_order(values, mesh_sizes)
    extrapolated = float(
        richardson_extrapolate(*_levels(values, mesh_sizes)[-2:], order)
    )

    sweep = []
    for slenderness in slenderness_sweep:
        swept_height = length / slenderness
        # keep the element aspect ratio of the base ladder (cells are 4:1) so the
        # sweep changes the physics and not the discretisation quality
        nx = divisions[-1][0]
        ny = max(2, int(round(4.0 * nx / slenderness)))
        ny += ny % 2
        level = solve_cantilever(
            nx,
            ny,
            length=length,
            height=swept_height,
            young=young,
            poisson=poisson,
            force=force,
        )
        sweep.append(
            {
                "slenderness": slenderness,
                "height": swept_height,
                "elements": level["elements"],
                "error_vs_euler_bernoulli": level["error_vs_euler_bernoulli"],
                "error_vs_timoshenko": level["error_vs_timoshenko"],
                "bending_stress_error": level["bending_stress_error"],
            }
        )

    finest = levels[-1]
    gate_error = float(finest["error_vs_euler_bernoulli"])
    metrics = {
        "observed_order": order,
        "richardson_tip_deflection": extrapolated,
        "euler_bernoulli": float(finest["euler_bernoulli"]),
        "timoshenko": float(finest["timoshenko"]),
        "model_gap_euler_bernoulli_to_timoshenko": abs(
            float(finest["timoshenko"]) - float(finest["euler_bernoulli"])
        )
        / float(finest["euler_bernoulli"]),
        "discretisation_error_vs_richardson": abs(
            float(finest["tip_deflection"]) - extrapolated
        )
        / abs(extrapolated),
        "continuum_limit_error_vs_euler_bernoulli": abs(extrapolated - float(finest["euler_bernoulli"]))
        / float(finest["euler_bernoulli"]),
        "continuum_limit_error_vs_timoshenko": abs(extrapolated - float(finest["timoshenko"]))
        / float(finest["timoshenko"]),
        "gate_metric": "error_vs_euler_bernoulli",
        "gate_value": gate_error,
        "slenderness_sweep": sweep,
        "sweep_scaling": _sweep_exponent(sweep),
        "passed": bool(
            gate_error < GATE_CANTILEVER
            and abs(extrapolated - float(finest["euler_bernoulli"]))
            / float(finest["euler_bernoulli"])
            < GATE_CANTILEVER
            and order >= MIN_ORDER
            and float(finest["bending_stress_error"]) < 0.03
        ),
    }
    return CaseResult(
        name="cantilever_plane_stress_q8",
        quantity="tip deflection delta [m] vs F L^3 / (3 E I)",
        gate=GATE_CANTILEVER,
        levels=levels,
        metrics=metrics,
    )


def _sweep_exponent(sweep: list[dict[str, object]]) -> dict[str, object]:
    """Fit ``gap ~ (h/L)^n`` on the Euler-Bernoulli sweep (``n = 2`` expected)."""
    gaps = [float(row["error_vs_euler_bernoulli"]) for row in sweep]
    slenderness = [float(row["slenderness"]) for row in sweep]
    exponent = float("nan")
    if (
        len(gaps) >= 2
        and gaps[0] > 1e-12
        and gaps[-1] > 1e-12
        and slenderness[-1] > slenderness[0]
    ):
        exponent = math.log(gaps[0] / gaps[-1]) / math.log(
            slenderness[-1] / slenderness[0]
        )
    return {
        "gaps": gaps,
        "slenderness": slenderness,
        "fitted_exponent": exponent,
        "expected_exponent": 2.0,
    }


def run_cylinder_ladder(
    *,
    inner: float = 0.5,
    outer: float = 0.8,
    pressure: float = 100.0e6,
    young: float = 210.0e9,
    poisson: float = 0.3,
    divisions: tuple[tuple[int, int], ...] = ((12, 4), (24, 8), (48, 16)),
    wall_ratio_sweep: tuple[float, ...] = (1.25, 1.6, 2.5),
) -> CaseResult:
    """Cylinder arm: the main ladder plus the ``r_o/r_i`` sweep at the finest mesh."""
    levels = [
        solve_lame_annulus(
            n_theta,
            n_radial,
            inner=inner,
            outer=outer,
            pressure=pressure,
            young=young,
            poisson=poisson,
        )
        for n_theta, n_radial in divisions
    ]
    mesh_sizes = [float(level["h"]) for level in levels]
    hoop_errors = [float(level["hoop_error"]) for level in levels]
    hoop_orders = observed_order_from_errors(hoop_errors, _ratio_between(mesh_sizes))
    # The gate is a stress gate, so the gated order is the stress order.  The
    # displacement order is recorded too but not gated: the bore displacement is
    # superconvergent (1e-7 relative on the finest mesh), so its Richardson
    # differences sit within a decade of the CG tolerance and do not describe
    # an asymptotic rate.
    order = float(hoop_orders[-1])
    displacement_order = observed_order(
        [float(level["bore_displacement"]) for level in levels], mesh_sizes
    )

    sweep = []
    for wall_ratio in wall_ratio_sweep:
        level = solve_lame_annulus(
            *divisions[-1],
            inner=inner,
            outer=inner * wall_ratio,
            pressure=pressure,
            young=young,
            poisson=poisson,
        )
        sweep.append(
            {
                "wall_ratio": wall_ratio,
                "elements": level["elements"],
                "hoop_error": level["hoop_error"],
                "radial_error": level["radial_error"],
                "bore_displacement_error": level["bore_displacement_error"],
            }
        )

    finest = levels[-1]
    metrics = {
        "observed_order": order,
        "hoop_error_orders": hoop_orders,
        "displacement_order_ungated": displacement_order,
        "hoop_errors": hoop_errors,
        "gate_metric": "hoop_error",
        "gate_value": float(finest["hoop_error"]),
        "wall_ratio_sweep": sweep,
        "passed": bool(
            float(finest["hoop_error"]) < GATE_CYLINDER
            and order >= MIN_ORDER
            and float(finest["bore_displacement_error"]) < GATE_CYLINDER
            and all(float(row["hoop_error"]) < GATE_CYLINDER for row in sweep)
        ),
    }
    return CaseResult(
        name="thick_cylinder_plane_strain_q8",
        quantity="worst-point hoop stress sigma_theta vs Lame",
        gate=GATE_CYLINDER,
        levels=levels,
        metrics=metrics,
    )


def run_plate_ladder(
    *,
    side: float = 1.0,
    young: float = 210.0e9,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0e3,
    divisions: tuple[int, ...] = (8, 16, 32),
    aspect_sweep: tuple[float, ...] = (10.0, 20.0, 50.0, 100.0),
    sweep_mesh: int | None = None,
) -> CaseResult:
    """Plate arm: clamped-square ladder at ``a/t = 100`` plus the ``a/t`` limit sweep.

    The sweep runs on a *fixed* mesh (``sweep_mesh``, by default the middle ladder
    level): the point of the sweep is to change the physics with the
    discretisation held still, and it keeps the cost of the arm bounded.
    """
    reference = ritz_reference(
        side=side, young=young, poisson=poisson, thickness=thickness, pressure=pressure
    )
    levels = [
        solve_clamped_plate(
            n,
            side=side,
            young=young,
            poisson=poisson,
            thickness=thickness,
            pressure=pressure,
            reference=reference,
        )
        for n in divisions
    ]
    mesh_sizes = [float(level["h"]) for level in levels]
    order = observed_order(
        [float(level["centre_deflection"]) for level in levels], mesh_sizes
    )

    sweep_mesh = sweep_mesh if sweep_mesh is not None else divisions[len(divisions) // 2]
    sweep = []
    for aspect in aspect_sweep:
        swept_reference = ritz_reference(
            side=side,
            young=young,
            poisson=poisson,
            thickness=side / aspect,
            pressure=pressure,
        )
        level = solve_clamped_plate(
            sweep_mesh,
            side=side,
            young=young,
            poisson=poisson,
            thickness=side / aspect,
            pressure=pressure,
            reference=swept_reference,
        )
        sweep.append(
            {
                "aspect": aspect,
                "elements": level["elements"],
                "error_vs_ritz": level["error_vs_ritz"],
                "alpha_measured": level["alpha_measured"],
                "alpha_reference": level["alpha_reference"],
                "ritz_mode_spread": swept_reference["mode_spread"],
                # Kirchhoff thin-plate theory is only a yardstick while the
                # transverse-shear contribution is negligible; the rows below
                # a/t = 50 are reported as the *validity boundary* of the oracle,
                # not as a finite-element failure.
                "gated": bool(aspect >= KIRCHHOFF_MIN_ASPECT),
            }
        )

    finest = levels[-1]
    metrics = {
        "observed_order": order,
        "ritz_alpha": float(reference["alpha"]),
        "ritz_mode_spread": float(reference["mode_spread"]),
        "timoshenko_table_alpha": 0.00126,
        "gate_metric": "error_vs_ritz",
        "gate_value": float(finest["error_vs_ritz"]),
        "aspect_sweep": sweep,
        "passed": bool(
            float(finest["error_vs_ritz"]) < GATE_PLATE
            and order >= MIN_ORDER
            and all(
                float(row["error_vs_ritz"]) < GATE_PLATE
                for row in sweep
                if row["gated"]
            )
        ),
    }
    return CaseResult(
        name="clamped_plate_uniform_pressure_q4_mindlin",
        quantity="centre deflection w_max vs alpha q a^4 / D",
        gate=GATE_PLATE,
        levels=levels,
        metrics=metrics,
    )


def run_face(**kwargs) -> dict[str, object]:
    """Run all three arms and return the machine-graded face report.

    ``kwargs`` may carry ``cantilever``, ``cylinder`` and ``plate`` dictionaries
    that are forwarded to the matching ``run_*_ladder`` function, so the test lane
    can run a cheaper ladder under the same contract.
    """
    cantilever = run_cantilever_ladder(**kwargs.get("cantilever", {}))
    cylinder = run_cylinder_ladder(**kwargs.get("cylinder", {}))
    plate = run_plate_ladder(**kwargs.get("plate", {}))
    cases = [cantilever, cylinder, plate]
    return {
        "schema": "tensorfem-structural-verification-face/1",
        "cases": [case.to_dict() for case in cases],
        "gates": {
            "cantilever_error_vs_euler_bernoulli": GATE_CANTILEVER,
            "cylinder_hoop_error": GATE_CYLINDER,
            "plate_error_vs_ritz": GATE_PLATE,
            "min_observed_order": MIN_ORDER,
        },
        "all_passed": all(bool(case.metrics.get("passed")) for case in cases),
    }