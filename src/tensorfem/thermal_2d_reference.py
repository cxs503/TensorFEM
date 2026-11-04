"""Closed-form references and verification studies for the Q4 (2-D) conduction path.

``tensorfem.thermal`` exposes a genuine two-dimensional element (``Q4`` with
bilinear shape functions, 2x2 Gauss integration, volumetric heat source,
consistent capacity and a two-node Robin/convection edge).  Until this module
existed that 2-D path had **no analytic verification**:

* ``tests/test_thermal.py`` only runs a linear-temperature patch test and a
  series-resistance check, both of which a badly assembled 2-D conductivity
  matrix can still pass;
* ``study_q4_strip_consistency`` compares a Q4 strip against the *1-D solver
  of this same code base*, so a shared error in the source convention, the
  capacity distribution or the boundary measure would cancel out;
* the literature anchors of ``thermal_transient_reference`` (Carslaw & Jaeger
  Sec. 3.4 Fourier series) are one-dimensional.

The references below are independent of the solver and close that debt.

``A``  Steady 1-D-in-x heat flow with a volumetric source and a convective
       face (plane wall with uniform generation and surface convection, e.g.
       Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed., Sec. 4.2;
       Incropera & DeWitt Sec. 3.5)::

           -k T'' = q,   T(0) = 0,   -k T'(L) = h (T(L) - T_inf)
           T(x)    = -q x**2/(2k) + A x,   T'(0) = A
           A       = (q L + h q L**2/(2k) + h T_inf) / (k + h L)

       Run on a *2-D* mesh it pins the sign and volumetric scaling of the
       source vector, the out-of-plane ``thickness`` factor and the Robin edge
       integral (measure ``Gamma * t``) at once.  The exact profile is
       representable by the discretisation (constant in ``y``, linear-element
       nodally exact in ``x``), so the anchor is expected at machine precision
       instead of at a percentage.

``B``  Manufactured harmonic fields on a quad mesh:
       ``T = x**2 - y**2`` (quadratic, nodally exact for bilinear elements --
       a machine-precision gate) and ``T = x**3 - 3 x y**2`` (cubic, harmonic,
       *not* representable), whose h-refinement measures the spatial
       convergence of the 2-D conductivity matrix, including the Jacobian and
       the inverse map on non-affine distorted quads.

``C``  Genuine 2-D transient cooling of a rectangle with uniform initial
       temperature and zero temperature on all four edges.  Separability of
       the heat equation gives the closed form as the *product* of two 1-D
       step solutions (Carslaw & Jaeger Sec. 3.4)::

           T(x,y,t) = S(x,t; a) * S(y,t; b),
           S(xi,t; L) = sum_{n odd} 4/(n*pi) sin(n*pi*xi/L)
                        * exp(-n**2*pi**2*alpha*t/L**2)

       Both directions are active, so neither the 2-D capacity matrix nor the
       2-D conductivity matrix can be right by accident.  A non-square domain
       (``b != a``) destroys the symmetry that an accidentally isotropic
       assembly could rely on, and the reflection-symmetry anchor on a square
       domain catches transposed Jacobians.

Every number reported by the studies is computed here; nothing is hand
copied.  ``benchmarks/thermal_2d_reference.py`` writes the evidence JSON, the
report and the figures, and ``tests/test_thermal_2d_reference.py`` recomputes
the gates.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfem.thermal import (
    ThermalModel,
    assemble_thermal,
    solve_steady_thermal,
    solve_transient_thermal,
)
from tensorfem.thermal_transient_reference import exact_slab_step, observed_order

FLOAT = torch.float64


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------
def rect_mesh(nx: int, ny: int, length: float = 1.0, height: float = 1.0,
              distortion: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured Q4 mesh of ``[0,length] x [0,height]`` with ``nx x ny`` elements.

    Node ``(i, j)`` has index ``i*(ny+1) + j``.  ``distortion > 0`` applies the
    deterministic map ``(x, y) -> (x + d, y + d)`` with
    ``d = distortion * length * sin(pi*x/length) * sin(pi*y/height)``, which
    keeps the element Jacobians positive while making the quads non-affine and
    the domain curved (the exact fields are evaluated at the moved node
    coordinates, as the manufactured-solution method requires).
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    if length <= 0.0 or height <= 0.0:
        raise ValueError("length and height must be positive")
    if not 0.0 <= distortion < 0.2:
        raise ValueError("distortion must be in [0, 0.2)")
    xs = torch.linspace(0.0, length, nx + 1, dtype=FLOAT)
    ys = torch.linspace(0.0, height, ny + 1, dtype=FLOAT)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    if distortion:
        d = distortion * length * torch.sin(torch.pi * xx / length) \
            * torch.sin(torch.pi * yy / height)
        xx = xx + d
        yy = yy + d
    nodes = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    conn = []
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            n1 = (i + 1) * (ny + 1) + j
            conn.append([n0, n1, n1 + 1, n0 + 1])
    return nodes, torch.tensor(conn, dtype=torch.long)


def node_index(i: int, j: int, ny: int) -> int:
    """Index of structured node ``(i, j)`` in :func:`rect_mesh`."""
    return i * (ny + 1) + j


def boundary_nodes(nx: int, ny: int) -> torch.Tensor:
    """All boundary node indices of the structured mesh, sorted, no repeats."""
    ids = set()
    for i in range(nx + 1):
        ids.add(node_index(i, 0, ny))
        ids.add(node_index(i, ny, ny))
    for j in range(ny + 1):
        ids.add(node_index(0, j, ny))
        ids.add(node_index(nx, j, ny))
    return torch.tensor(sorted(ids), dtype=torch.long)


def right_edge(nx: int, ny: int) -> torch.Tensor:
    """Two-node edges on the ``x = length`` face, bottom to top."""
    return torch.tensor([[node_index(nx, j, ny), node_index(nx, j + 1, ny)]
                         for j in range(ny)], dtype=torch.long)


# ---------------------------------------------------------------------------
# closed-form references
# ---------------------------------------------------------------------------
def wall_source_slope(length: float = 1.0, conductivity: float = 1.0,
                      source: float = 1.0, coefficient: float = 1.0,
                      ambient: float = 0.0) -> float:
    """Slope ``A = T'(0)`` of the reference ``A`` wall profile.

    With ``T(x) = -q x**2/(2k) + A x`` the convective condition at ``x = L``,
    ``-k T'(L) = h (T(L) - T_inf)``, gives
    ``A = (q L + h q L**2/(2k) + h T_inf) / (k + h L)``.  Note that the naive
    ``T(L) = q L/h`` is only the ``k -> infinity`` limit; keeping the parabolic
    contribution is what makes the closed form agree with the FE solution to
    machine precision (see the module notes).
    """
    if length <= 0.0 or conductivity <= 0.0 or coefficient <= 0.0:
        raise ValueError("length, conductivity and coefficient must be positive")
    return (source * length + coefficient * source * length ** 2 / (2.0 * conductivity)
            + coefficient * ambient) / (conductivity + coefficient * length)


def exact_wall_source_convection(x: torch.Tensor, length: float = 1.0,
                                 conductivity: float = 1.0, source: float = 1.0,
                                 coefficient: float = 1.0,
                                 ambient: float = 0.0) -> torch.Tensor:
    """Reference ``A``: plane wall, uniform generation, one convective face."""
    a = wall_source_slope(length, conductivity, source, coefficient, ambient)
    xi = torch.as_tensor(x, dtype=FLOAT)
    return -source * xi ** 2 / (2.0 * conductivity) + a * xi


def exact_harmonic_quadratic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference ``B1``: ``T = x**2 - y**2`` (harmonic, bilinear-nodally exact)."""
    return torch.as_tensor(x, dtype=FLOAT) ** 2 - torch.as_tensor(y, dtype=FLOAT) ** 2


def exact_harmonic_cubic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference ``B2``: ``T = x**3 - 3 x y**2`` (harmonic, not representable)."""
    xx = torch.as_tensor(x, dtype=FLOAT)
    yy = torch.as_tensor(y, dtype=FLOAT)
    return xx ** 3 - 3.0 * xx * yy ** 2


def exact_rectangle_transient(x: torch.Tensor, y: torch.Tensor, t: float,
                              length: float = 1.0, height: float = 1.0,
                              alpha: float = 1.0,
                              n_terms: int = 400) -> torch.Tensor:
    """Reference ``C``: rectangle, unit initial field, zero edges.

    ``T(x,y,t) = S(x,t; length) * S(y,t; height)`` -- the product of the two
    one-dimensional Carslaw & Jaeger step solutions, which solves the 2-D heat
    equation because ``(d/dt - alpha*laplace)`` factors over the product.
    """
    if t <= 0.0:
        raise ValueError("t must be positive")
    sx = exact_slab_step(x, t, length=length, alpha=alpha, T0=1.0, n_terms=n_terms)
    sy = exact_slab_step(y, t, length=height, alpha=alpha, T0=1.0, n_terms=n_terms)
    return sx * sy


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------
def relative_l2(numeric: torch.Tensor, reference: torch.Tensor) -> float:
    ref = torch.as_tensor(reference, dtype=FLOAT)
    num = torch.as_tensor(numeric, dtype=FLOAT)
    return float(torch.linalg.vector_norm(num - ref) / torch.linalg.vector_norm(ref))


def max_abs(numeric: torch.Tensor, reference: torch.Tensor) -> float:
    return float((torch.as_tensor(numeric, dtype=FLOAT)
                  - torch.as_tensor(reference, dtype=FLOAT)).abs().max())


def _alpha(conductivity: float, density: float, specific_heat: float) -> float:
    return conductivity / (density * specific_heat)


# ---------------------------------------------------------------------------
# reference A: steady wall with source and a convective face
# ---------------------------------------------------------------------------
def wall_model(nx: int, ny: int, length: float, height: float, conductivity: float,
               source: float, coefficient: float, thickness: float,
               ambient: float = 0.0, with_convection: bool = True) -> ThermalModel:
    """Q4 model of the extruded wall, Dirichlet at ``x=0``, Robin at ``x=length``."""
    nodes, conn = rect_mesh(nx, ny, length, height)
    fixed = torch.tensor([node_index(0, j, ny) for j in range(ny + 1)], dtype=torch.long)
    kwargs = {}
    if with_convection:
        kwargs = dict(convection_edges=right_edge(nx, ny),
                      convection_coefficient=torch.tensor(coefficient, dtype=FLOAT),
                      ambient_temperature=torch.tensor(ambient, dtype=FLOAT))
    return ThermalModel(
        nodes=nodes, elements=conn,
        conductivity=torch.tensor(conductivity, dtype=FLOAT),
        density=torch.tensor(7850.0, dtype=FLOAT),
        specific_heat=torch.tensor(465.0, dtype=FLOAT),
        fixed_nodes=fixed,
        fixed_temperatures=torch.zeros(fixed.numel(), dtype=FLOAT),
        heat_source=torch.tensor(source, dtype=FLOAT),
        thickness=torch.tensor(thickness, dtype=FLOAT),
        element_type="q4",
        **kwargs,
    )


def study_source_convection_wall(nx: int = 8, ny: int = 3, length: float = 1.0,
                                 height: float = 0.5, conductivity: float = 45.0,
                                 source: float = 5.0e4, coefficient: float = 500.0,
                                 thickness: float = 0.02,
                                 ambient: float = 0.0) -> dict:
    """Reference ``A`` on a 2-D mesh: extruded wall, source + Robin face."""
    model = wall_model(nx, ny, length, height, conductivity, source, coefficient,
                       thickness, ambient)
    temperature = solve_steady_thermal(model)
    nodes = model.nodes
    exact = exact_wall_source_convection(nodes[:, 0], length, conductivity,
                                         source, coefficient, ambient)

    # --- energy balance, all terms from the solver plus mesh geometry ------
    matrices = assemble_thermal(model)
    residual = matrices.conductivity @ temperature - matrices.heat
    fixed = model.fixed_nodes
    # A negative reaction at a constrained temperature means the constraint has
    # to *remove* heat, so the flux leaving through that face is -sum(R).
    heat_out_fixed = -float(residual[fixed].sum())
    generated = source * length * height * thickness       # exact: bilinear partition of unity
    # independent trapezoidal evaluation of the Robin flux on the x=length face
    # (the profile is constant along y there, so trapezoid is exact)
    h, t = coefficient, thickness
    edge = right_edge(nx, ny)
    edge_len = torch.linalg.vector_norm(nodes[edge[:, 1]] - nodes[edge[:, 0]], dim=1)
    edge_mean = 0.5 * (temperature[edge[:, 0]] + temperature[edge[:, 1]]) - ambient
    robin_outflux = float((h * edge_len * t * edge_mean).sum())
    balance = abs(robin_outflux + heat_out_fixed - generated) / generated

    profile_i = [node_index(i, 0, ny) for i in range(nx + 1)]
    return {
        "anchor": "source_convection_wall_2d",
        "description": "extruded plane wall on a %dx%d Q4 mesh: -k laplace(T) = q, T=0 at "
                       "x=0 and a convective face at x=L (thickness %.3f m)"
                       % (nx, ny, thickness),
        "reference": "T(x) = -q x^2/(2k) + A x,  A = (qL + hqL^2/(2k) + h T_inf)/(k + hL)",
        "relative_l2_error": relative_l2(temperature, exact),
        "max_absolute_error": max_abs(temperature, exact),
        "outlet_temperature": float(temperature[node_index(nx, 0, ny)]),
        "exact_outlet_temperature": float(exact_wall_source_convection(
            torch.tensor([length], dtype=FLOAT), length, conductivity, source,
            coefficient, ambient)[0]),
        "generated_heat": generated,
        "heat_out_fixed_face": heat_out_fixed,
        "robin_outflux": robin_outflux,
        "energy_balance_relative_error": balance,
        "gate_l2": 1.0e-8,
        "gate_energy": 1.0e-8,
        "values": {"nx": nx, "ny": ny, "k": conductivity, "q": source, "h": coefficient,
                   "L": length, "H": height, "t": thickness, "ambient": ambient},
        "profile": {"x": [float(nodes[i, 0]) for i in profile_i],
                    "numeric": [float(temperature[i]) for i in profile_i],
                    "exact": [float(exact[i]) for i in profile_i]},
        "field": {"nodes": nodes.tolist(),
                  "conn": model.elements.tolist(),
                  "temperature": temperature.tolist(),
                  "exact": exact.tolist()},
    }


# ---------------------------------------------------------------------------
# reference B: manufactured harmonic solutions
# ---------------------------------------------------------------------------
def exact_harmonic_smooth(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference ``B3``: ``T = exp(pi*x) * sin(pi*y)`` (harmonic, C-infinity)."""
    xx = torch.as_tensor(x, dtype=FLOAT)
    yy = torch.as_tensor(y, dtype=FLOAT)
    return torch.exp(torch.pi * xx) * torch.sin(torch.pi * yy)


_HARMONICS = {
    "quadratic": exact_harmonic_quadratic,
    "cubic": exact_harmonic_cubic,
    "smooth": exact_harmonic_smooth,
}


def solve_harmonic(n: int, order: str = "cubic", distortion: float = 0.0,
                   conductivity: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the Laplace problem with Dirichlet data from the exact harmonic."""
    if order not in _HARMONICS:
        raise ValueError("order must be one of %s" % ", ".join(sorted(_HARMONICS)))
    field = _HARMONICS[order]
    nodes, conn = rect_mesh(n, n, 1.0, 1.0, distortion=distortion)
    fixed = boundary_nodes(n, n)
    exact_nodes = field(nodes[:, 0], nodes[:, 1])
    model = ThermalModel(
        nodes=nodes, elements=conn,
        conductivity=torch.tensor(conductivity, dtype=FLOAT),
        density=torch.tensor(1.0, dtype=FLOAT),
        specific_heat=torch.tensor(1.0, dtype=FLOAT),
        fixed_nodes=fixed,
        fixed_temperatures=exact_nodes[fixed],
        thickness=torch.tensor(1.0, dtype=FLOAT),
        element_type="q4",
    )
    return solve_steady_thermal(model), exact_nodes


def study_harmonic_exactness(distortion: float = 0.0, levels=(2, 4, 8)) -> dict:
    """Reference ``B1``: the quadratic harmonic must be reproduced nodally exactly."""
    worst = 0.0
    for n in levels:
        temperature, exact = solve_harmonic(n, "quadratic", distortion)
        worst = max(worst, max_abs(temperature, exact))
    return {
        "anchor": "harmonic_quadratic_nodal_exactness",
        "description": "T = x^2 - y^2 on 2..16 Q4 meshes%s; bilinear elements are "
                       "nodally exact for harmonic quadratics on rectangles"
                       % (", distorted %.0f%%" % (100 * distortion) if distortion else ""),
        "reference": "T = x**2 - y**2 (exact Laplacian solution, nodal values)",
        "max_absolute_error": worst,
        "gate_absolute": 1.0e-10,
        "values": {"distortion": distortion, "levels": list(levels)},
    }


_HARMONIC_REFS = {
    "quadratic": "T = x**2 - y**2",
    "cubic": "T = x**3 - 3*x*y**2",
    "smooth": "T = exp(pi*x)*sin(pi*y)",
}


def study_harmonic_convergence(order: str = "smooth", levels=(4, 8, 16, 32),
                               distortion: float = 0.0) -> dict:
    """Reference ``B2``: h-refinement on a harmonic field bilinear cannot represent."""
    errors = []
    for n in levels:
        temperature, exact = solve_harmonic(n, order, distortion)
        errors.append(relative_l2(temperature, exact))
    orders = [observed_order(a, b) for a, b in zip(errors[:-1], errors[1:])]
    return {
        "anchor": "mms_harmonic_" + order + ("_distorted" if distortion else ""),
        "description": "MMS harmonic field (%s, laplace T = 0), Dirichlet data from the exact "
                       "field, Q4 h-refinement%s"
                       % (order, ", mesh distortion %.0f%%" % (100 * distortion) if distortion else ""),
        "reference": _HARMONIC_REFS[order] + " (exact solution of the Laplace problem)",
        "levels": list(levels),
        "errors": errors,
        "orders": orders,
        "observed_order": orders[-1],
        "expected_order": 2.0,
        "gate_order": 1.7,
        "values": {"distortion": distortion},
    }


# ---------------------------------------------------------------------------
# reference C: genuine 2-D transient
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RectangleCase:
    """One 2-D transient solve reduced to its final field."""

    temperature: torch.Tensor
    nodes: torch.Tensor
    nx: int
    ny: int
    n_steps: int
    t_end: float


def run_rectangle(nx: int = 6, ny: int = 3, length: float = 1.0, height: float = 0.5,
                  t_end: float = 0.15, n_steps: int = 60, theta: float = 0.5,
                  conductivity: float = 1.0, density: float = 1.0,
                  specific_heat: float = 1.0,
                  thickness: float = 1.0) -> RectangleCase:
    """Reference ``C``: uniform initial field, zero temperature on all edges."""
    nodes, conn = rect_mesh(nx, ny, length, height)
    fixed = boundary_nodes(nx, ny)
    model = ThermalModel(
        nodes=nodes, elements=conn,
        conductivity=torch.tensor(conductivity, dtype=FLOAT),
        density=torch.tensor(density, dtype=FLOAT),
        specific_heat=torch.tensor(specific_heat, dtype=FLOAT),
        fixed_nodes=fixed,
        fixed_temperatures=torch.zeros(fixed.numel(), dtype=FLOAT),
        thickness=torch.tensor(thickness, dtype=FLOAT),
        element_type="q4",
    )
    initial = torch.ones(nodes.shape[0], dtype=FLOAT)
    initial[fixed] = 0.0
    time = torch.linspace(0.0, t_end, n_steps + 1, dtype=FLOAT)
    result = solve_transient_thermal(model, time, initial, theta=theta)
    return RectangleCase(result.temperature[-1].clone(), nodes, nx, ny, n_steps, t_end)


def study_rectangle_transient(length: float = 1.0, height: float = 0.5,
                              t_end: float = 0.05,
                              steps=((8, 4, 40), (16, 8, 80), (32, 16, 160)),
                              theta: float = 0.5) -> dict:
    """Reference ``C``: 2-D cooling of a 1.0 x 0.5 rectangle, both directions live.

    ``t_end = 0.05`` keeps the field alive (peak ~0.12 of the initial value): at
    ``t_end = 0.15`` the solution has decayed to ~1e-3 and the *relative* L2
    norm then measures rounding-level noise rather than the discretisation, so
    the earlier, longer time window was discarded.
    """
    alpha = _alpha(1.0, 1.0, 1.0)
    errors = []
    for (nx, ny, n_steps) in steps:
        case = run_rectangle(nx, ny, length, height, t_end, n_steps, theta)
        exact = exact_rectangle_transient(case.nodes[:, 0], case.nodes[:, 1], t_end,
                                          length, height, alpha)
        errors.append(relative_l2(case.temperature, exact))
    orders = [observed_order(a, b) for a, b in zip(errors[:-1], errors[1:])]
    return {
        "anchor": "rectangle_transient_product_series",
        "description": "T(x,y,0)=1, T=0 on all edges, %.2fx%.2f rectangle, Fo=%.3f, "
                       "Crank-Nicolson, Q4 h- and dt-refinement" % (length, height, t_end),
        "reference": "T = S(x,t;a) * S(y,t;b), S = Carslaw & Jaeger 1-D step series",
        "levels": [nx for nx, _, _ in steps],
        "steps": [s for _, _, s in steps],
        "errors": errors,
        "orders": orders,
        "observed_order": orders[-1],
        "expected_order": 2.0,
        "gate_error": 0.03,
        "values": {"length": length, "height": height, "t_end": t_end, "theta": theta,
                   "alpha": alpha},
    }


def study_rectangle_maximum_principle(n: int = 16, length: float = 1.0,
                                      height: float = 0.5, t_end: float = 0.05,
                                      n_steps: int = 60, theta: float = 1.0) -> dict:
    """Pointwise bounds: with non-negative data no node may leave ``[0, 1]``."""
    case = run_rectangle(n, n // 2, length, height, t_end, n_steps, theta)
    exact = exact_rectangle_transient(case.nodes[:, 0], case.nodes[:, 1], t_end,
                                      length, height, _alpha(1.0, 1.0, 1.0))
    centre = node_index(n // 2, n // 4, n // 2)
    return {
        "anchor": "rectangle_transient_maximum_principle",
        "description": "%dx%d Q4 mesh, backward Euler, t=%.3f: 0 <= T <= 1 pointwise"
                       % (n, n // 2, t_end),
        "reference": "maximum principle for the heat equation with non-negative data",
        "minimum": float(case.temperature.min()),
        "maximum": float(case.temperature.max()),
        "centre_numeric": float(case.temperature[centre]),
        "centre_exact": float(exact[centre]),
        "relative_l2_error": relative_l2(case.temperature, exact),
        "gate_l2": 0.03,
    }


def study_square_symmetry(n: int = 8, t_end: float = 0.05, n_steps: int = 80) -> dict:
    """On a square domain the field must be symmetric under ``x <-> y``.

    A 2-D assembly that mixes up the two Gauss directions (a classic
    transposed-Jacobian bug) preserves the total energy but breaks this
    symmetry, so it is an independent, cheap discriminator.
    """
    case = run_rectangle(n, n, 1.0, 1.0, t_end, n_steps, theta=0.5)
    grid = case.temperature.reshape(n + 1, n + 1)
    asymmetry = float((grid - grid.transpose(0, 1)).abs().max())
    return {
        "anchor": "square_transient_reflection_symmetry",
        "description": "square domain, uniform initial field, zero edges, %dx%d Q4, t=%.3f"
                       % (n, n, t_end),
        "reference": "T(x,y,t) = T(y,x,t) exactly for this data",
        "max_asymmetry": asymmetry,
        "max_temperature": float(case.temperature.max()),
        "gate_absolute": 1.0e-12,
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def run_all() -> dict:
    """Run every anchor and attach a pass/fail verdict."""
    anchors = [
        study_source_convection_wall(),
        study_harmonic_exactness(),
        study_harmonic_convergence("smooth"),
        study_harmonic_convergence("smooth", distortion=0.06),
        study_rectangle_transient(),
        study_rectangle_maximum_principle(),
        study_square_symmetry(),
    ]
    for anchor in anchors:
        anchor["passed"] = verdict(anchor)
    return {
        "schema": "tensorfem.thermal-2d-q4/1.0",
        "anchors": anchors,
        "all_passed": all(a["passed"] for a in anchors),
    }


def verdict(anchor: dict) -> bool:
    """Pass/fail rule for one anchor (kept explicit so it can be audited)."""
    name = anchor["anchor"]
    if name.startswith("source_convection_wall"):
        return (anchor["relative_l2_error"] < anchor["gate_l2"]
                and anchor["energy_balance_relative_error"] < anchor["gate_energy"])
    if name.startswith("harmonic_quadratic_nodal"):
        return anchor["max_absolute_error"] < anchor["gate_absolute"]
    if name.startswith("mms_harmonic"):
        return (anchor["observed_order"] >= anchor["gate_order"]
                and anchor["errors"][-1] < 0.03)
    if name.startswith("rectangle_transient_product"):
        return (anchor["errors"][-1] < anchor["gate_error"]
                and anchor["observed_order"] >= 1.8)
    if name.startswith("rectangle_transient_maximum"):
        return (anchor["relative_l2_error"] < anchor["gate_l2"]
                and anchor["minimum"] >= -1.0e-12
                and anchor["maximum"] <= 1.0 + 1.0e-12)
    if name.startswith("square_transient_reflection"):
        return anchor["max_asymmetry"] <= anchor["gate_absolute"]
    raise KeyError("no verdict rule for anchor %r" % name)