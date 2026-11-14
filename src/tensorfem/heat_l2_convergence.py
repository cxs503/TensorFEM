"""CASE-FEM-HEAT-L2 -- L2 convergence rate of steady heat conduction on Q4 elements.

Problem class
-------------
Steady conduction without generation-free shortcuts::

    -div(k grad T) = q   in  Omega = (0, Lx) x (0, Ly),   k = 1,
                      T = T_exact   on  dOmega  (Dirichlet, full boundary)

discretised by the bilinear ``Q4`` conduction kernel of :mod:`tensorfem.thermal`
(2x2 Gauss conductivity matrix, element-constant volumetric source, consistent
Dirichlet elimination).

Reference (zero hand-copy)
--------------------------
The reference is a **manufactured solution** (MMS): the exact temperature field
is an analytic expression implemented as code (``HeatFieldCase.temperature``)
and every reference *number* in the studies below is evaluated from it at run
time -- no value is transcribed from a table.

The **target order** is not typed either.  :func:`target_l2_order_from_manual`
parses the a-priori order table that this repository already ships in the
module docstring of :mod:`tensorfem.adaptive_poisson_2d` (the ``||e||_0`` row
reads ``O(h^2)``) and returns the exponent, the verbatim row and its line
number, so a report can quote the manual row it came from.  That table is the
in-repo, version-controlled statement of the formal order for bilinear
quadrilaterals; the external methodology anchors are

* K. Salari & P. Knupp, *Code Verification by the Method of Manufactured
  Solutions*, Sandia National Laboratories SAND2000-1009J (2000) -- the
  observed order of accuracy of a correctly implemented discretisation must
  converge to its formal order;
* P.J. Roache (1994) "Perspective: a method for uniform reporting of grid
  refinement studies", *J. Fluids Eng.* 116, 405-413 -- refinement-ratio
  extrapolation and the GCI safety factor, the same reference that
  :mod:`tensorfem.convergence_verification` is built on.

Field cases (``HeatFieldCase``)
-------------------------------
``sine_generation``   T = sin(pi x) sin(pi y) on the unit square, q = 2 pi^2 T
                      (the classic MMS load; non-harmonic, so the source term
                      of the heat kernel is exercised);
``exp_trig_harmonic`` T = exp(pi x) sin(pi y), q = 0 (Laplace), smooth and
                      non-polynomial;
``cubic_harmonic``    T = x^3 - 3 x y^2, q = 0, harmonic cubic: *not*
                      representable by bilinear Q4, and because the Q4 stencil
                      on a uniform rectangle *is* the 5-point Laplacian its
                      nodal values are exact (a machine-precision gate on the
                      assembled stencil) while its L2 error is pure
                      interpolation error -- an independent baseline;
``sine_aspect2``      T = sin(pi x / 2) sin(pi y) on (0,2)x(0,1),
                      q = 1.25 pi^2 T (non-square Jacobian; kills an
                      accidentally isotropic assembly and a transposed map);
``quadratic_harmonic`` T = x^2 - y^2, q = 0: nodal values exact (the Q4 stencil
                      of a uniform rectangle *is* the 5-point Laplacian, which
                      is exact for quadratics) while the L2 error is pure
                      interpolation error, ratio ``e_FE / e_interp = 1`` -- the
                      "solver adds nothing to interpolation" baseline;
``bilinear_control``  T = x*y, q = 0 -- **control**: harmonic *and* inside the
                      bilinear space, hence solved exactly at every level.  Its
                      ladder carries no signal and no order is claimed for it
                      (the same trap produced err ~ 1e-15 in a point-load FD
                      ladder registered as negative knowledge for this
                      ecosystem).

Specification correction (kept in the record).  The first version of this
study used ``T = x^2 - y^2`` as the representable control and the control gate
failed at 6.6e-3: the bilinear space of a Q4 element is ``span{1, x, y, x*y}``,
so ``x^2 - y^2`` is *not* representable -- the specification, not the solver,
was wrong.  The case was reclassified as ``quadratic_harmonic`` (a nodal-exact
case with a real O(h^2) L2 signal) and ``T = x*y`` took over as the control.

Predeclared acceptance (fixed in this docstring before the ladder was run)
--------------------------------------------------------------------------
``GATE_L2_PAIR = 1.90``       every consecutive pair whose coarse mesh has
                              ``nx >= 8`` must show ``p_L2 >= 1.90``
                              (theory 2.0, i.e. a 5% band);
``GATE_L2_FINEST = (1.95, 2.10)``  the finest-pair L2 order must land inside
                              this window (no runaway super-convergence either);
``GATE_H1_PAIR = 0.90``       supplementary: H1-seminorm order ``>= 0.90``
                              (theory 1.0 for bilinear elements);
``GATE_POWER_LAW = 1.0e-2``   the L2 ladder extrapolated with the *theoretical*
                              order 2 must leave a residual
                              ``|e0| / e_fine <= 1%`` of Roache's form
                              ``e0 = e_fine + (e_fine - e_coarse)/(r^p - 1)``,
                              i.e. the error sequence really is the theoretical
                              power law and not three lucky points;
``GATE_CONTROL_ABS = 1.0e-10``  the representable quadratic control must be
                              exact to this absolute level on *every* level, and
                              no order is claimed for it;
``GATE_FIELD_REL = 3.0e-2``   on the finest mesh the reconstructed field must be
                              within 3% of the analytic field at every element
                              centre, normalised by the peak of ``|T|`` (a
                              pointwise ``|T|`` normalisation is singular where
                              the field crosses zero, e.g. on the diagonal of
                              ``T = x^2 - y^2``); else the convergence rate
                              would be meaningless;
``GATE_QUADRATURE = 1.0e-8``  relative change of every reported error norm when
                              the post-processing Gauss order is raised
                              ``4 -> 6 -> 8`` (no quadrature contamination of
                              the measured order: the exact trap that
                              invalidated an earlier thermal "order 2.0" claim
                              in this repository).

Error norms
-----------
All norms are true domain integrals evaluated by an **independent** Gauss
post-processor (:func:`evaluate_errors`) that rebuilds the bilinear field from
the nodal values with its own shape functions -- it does not reuse the assembly
kernel, so an assembly bug cannot hide inside the norm::

    ||e||_0  = sqrt( int (T_h - T)^2 dO / int T^2 dO )      (relative L2)
    |e|_1    = sqrt( int |grad T_h - grad T|^2 dO /
                     int |grad T|^2 dO )                    (relative H1)

The same post-processor applied to ``T_exact`` sampled at the nodes yields the
L2 error of the **bilinear interpolant**, reported as the "what interpolation
alone can do" baseline of every ladder.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from tensorfem.thermal import ThermalModel, solve_steady_thermal
from tensorfem.thermal_analytic import structured_quad_mesh

FLOAT = torch.float64
SCHEMA = "tensorfem.case-fem-heat-l2/1.0"

# --- predeclared acceptance gates (not tuned after the run) -----------------
GATE_L2_PAIR = 1.90
GATE_L2_FINEST = (1.95, 2.10)
GATE_H1_PAIR = 0.90
GATE_POWER_LAW = 1.0e-2
GATE_CONTROL_ABS = 1.0e-10
GATE_FIELD_REL = 3.0e-2
GATE_QUADRATURE = 1.0e-8

DEFAULT_LEVELS = (4, 8, 16, 32, 64)
DEFAULT_QUAD_ORDER = 4


# ---------------------------------------------------------------------------
# quadrature and Q4 interpolation (independent post-processor)
# ---------------------------------------------------------------------------
def gauss_legendre_1d(order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``order``-point Gauss-Legendre rule on ``[-1, 1]`` (exact to degree 2n-1)."""
    if order < 1:
        raise ValueError("order must be positive")
    n = order
    x = torch.cos(math.pi * (torch.arange(1, n + 1, dtype=FLOAT) - 0.25) / (n + 0.5))
    dp = torch.ones_like(x)
    for _ in range(200):
        p0 = torch.ones_like(x)
        p1 = x.clone()
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1.0)
        dx = p1 / dp
        x = x - dx
        if float(dx.abs().max()) < 1.0e-17:
            break
    w = 2.0 / ((1.0 - x * x) * dp * dp)
    return x, w


def q4_shape(xi: float, eta: float) -> torch.Tensor:
    """Bilinear shape functions at ``(xi, eta)``, CCW from ``(-1, -1)``."""
    return torch.tensor([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    ], dtype=FLOAT)


def q4_shape_grad(xi: float, eta: float) -> torch.Tensor:
    """Reference-space gradients ``dN/dxi`` (4x2) at ``(xi, eta)``."""
    return torch.tensor([
        [-0.25 * (1 - eta), -0.25 * (1 - xi)],
        [0.25 * (1 - eta), -0.25 * (1 + xi)],
        [0.25 * (1 + eta), 0.25 * (1 + xi)],
        [-0.25 * (1 + eta), 0.25 * (1 - xi)],
    ], dtype=FLOAT)


def evaluate_errors(nodes: torch.Tensor, elements: torch.Tensor,
                    nodal_values: torch.Tensor, case: "HeatFieldCase",
                    quad_order: int = DEFAULT_QUAD_ORDER) -> dict[str, float]:
    """True L2 / H1 error norms of a bilinear field against the analytic case.

    ``nodal_values`` may be the FE solution *or* the exact field sampled at the
    nodes; in the latter case the returned norms are the interpolation errors.
    """
    coords = nodes[elements].to(FLOAT)                        # (ne, 4, 2)
    values = nodal_values[elements].to(FLOAT)                 # (ne, 4)
    xi, w1 = gauss_legendre_1d(quad_order)

    err_sq = 0.0
    ref_exact_sq = 0.0
    grad_err_sq = 0.0
    grad_exact_sq = 0.0
    for a in range(quad_order):
        for b in range(quad_order):
            weight = float(w1[a] * w1[b])
            shape = q4_shape(float(xi[a]), float(xi[b]))
            dshape = q4_shape_grad(float(xi[a]), float(xi[b]))
            jac = torch.einsum("ik,eil->ekl", dshape, coords)   # J[e,k,l]=dX_k/dxi_l
            det = torch.linalg.det(jac)
            inv = torch.linalg.inv(jac)
            dshape_x = dshape.unsqueeze(0) @ inv.transpose(1, 2)   # (ne, 4, 2)
            points = torch.einsum("i,eic->ec", shape, coords)
            t_h = torch.einsum("i,ei->e", shape, values)
            t_ex = case.temperature(points)
            g_h = torch.einsum("eic,ei->ec", dshape_x, values)
            g_ex = case.gradient(points)
            measure = det * weight
            err_sq = err_sq + float((measure * (t_h - t_ex) ** 2).sum())
            ref_exact_sq = ref_exact_sq + float((measure * t_ex ** 2).sum())
            grad_err_sq = grad_err_sq + float(
                (measure * ((g_h - g_ex) ** 2).sum(dim=1)).sum())
            grad_exact_sq = grad_exact_sq + float(
                (measure * (g_ex ** 2).sum(dim=1)).sum())

    exact_nodes = case.temperature(nodes.to(FLOAT))
    return {
        "l2_relative": math.sqrt(err_sq / ref_exact_sq),
        "h1_relative": math.sqrt(grad_err_sq / grad_exact_sq),
        "l2_absolute": math.sqrt(err_sq),
        "max_abs_nodal": float((nodal_values.to(FLOAT) - exact_nodes).abs().max()),
    }


# ---------------------------------------------------------------------------
# manufactured fields
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeatFieldCase:
    """One manufactured steady-conduction case on a rectangle."""

    key: str
    description: str
    length: float
    height: float
    temperature: Callable[[torch.Tensor], torch.Tensor]
    gradient: Callable[[torch.Tensor], torch.Tensor]
    source: Callable[[torch.Tensor], torch.Tensor]
    reference: str
    representable: bool = False
    nodal_exact_expected: bool = False
    refinement: str = "isotropic"

    def element_mean_source(self, nodes: torch.Tensor, elements: torch.Tensor,
                            quad_order: int = 8) -> torch.Tensor:
        """Element-averaged volumetric source ``q`` from an 8x8 Gauss integral.

        The heat kernel takes an element-constant source, so the consistent
        element load of the MMS is ``int_elem q dO / int_elem 1 dO``.  The 8x8
        rule integrates these smooth ``q`` to machine precision on every mesh of
        the ladder (``h >= 1/64``), so the source is *not* a hidden
        O(h^2) consistency error of the study.
        """
        coords = nodes[elements].to(FLOAT)
        xi, w1 = gauss_legendre_1d(quad_order)
        total = torch.zeros(coords.shape[0], dtype=FLOAT)
        area = torch.zeros(coords.shape[0], dtype=FLOAT)
        for a in range(quad_order):
            for b in range(quad_order):
                weight = float(w1[a] * w1[b])
                shape = q4_shape(float(xi[a]), float(xi[b]))
                dshape = q4_shape_grad(float(xi[a]), float(xi[b]))
                jac = torch.einsum("ik,eil->ekl", dshape, coords)
                det = torch.linalg.det(jac)
                measure = det * weight
                points = torch.einsum("i,eic->ec", shape, coords)
                total = total + measure * self.source(points)
                area = area + measure
        return total / area


def field_cases() -> dict[str, HeatFieldCase]:
    """The manufactured cases of this verification study (ordered)."""
    cases: dict[str, HeatFieldCase] = {}

    def sine_generation(xy):
        x, y = xy[:, 0], xy[:, 1]
        return torch.sin(math.pi * x) * torch.sin(math.pi * y)

    def sine_generation_grad(xy):
        x, y = xy[:, 0], xy[:, 1]
        return torch.stack((math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y),
                            math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)),
                           dim=1)

    cases["sine_generation"] = HeatFieldCase(
        key="sine_generation",
        description="T = sin(pi x) sin(pi y), q = 2 pi^2 T, T = 0 on the boundary",
        length=1.0, height=1.0,
        temperature=sine_generation,
        gradient=sine_generation_grad,
        source=lambda xy: 2.0 * math.pi ** 2 * sine_generation(xy),
        reference="T = sin(pi x) sin(pi y) (MMS, exact solution of -laplace T = q)",
    )

    def exp_trig(xy):
        x, y = xy[:, 0], xy[:, 1]
        return torch.exp(math.pi * x) * torch.sin(math.pi * y)

    def exp_trig_grad(xy):
        x, y = xy[:, 0], xy[:, 1]
        e = torch.exp(math.pi * x)
        return torch.stack((math.pi * e * torch.sin(math.pi * y),
                            math.pi * e * torch.cos(math.pi * y)), dim=1)

    cases["exp_trig_harmonic"] = HeatFieldCase(
        key="exp_trig_harmonic",
        description="T = exp(pi x) sin(pi y), q = 0 (Laplace), Dirichlet from the exact field",
        length=1.0, height=1.0,
        temperature=exp_trig,
        gradient=exp_trig_grad,
        source=lambda xy: torch.zeros(xy.shape[0], dtype=FLOAT),
        reference="T = exp(pi x) sin(pi y) (harmonic MMS)",
    )

    def cubic(xy):
        x, y = xy[:, 0], xy[:, 1]
        return x ** 3 - 3.0 * x * y ** 2

    def cubic_grad(xy):
        x, y = xy[:, 0], xy[:, 1]
        return torch.stack((3.0 * x ** 2 - 3.0 * y ** 2, -6.0 * x * y), dim=1)

    cases["cubic_harmonic"] = HeatFieldCase(
        key="cubic_harmonic",
        description="T = x^3 - 3 x y^2, q = 0 (harmonic cubic, not representable)",
        length=1.0, height=1.0,
        temperature=cubic,
        gradient=cubic_grad,
        source=lambda xy: torch.zeros(xy.shape[0], dtype=FLOAT),
        reference="T = x^3 - 3 x y^2 (harmonic MMS)",
        nodal_exact_expected=True,
    )

    def sine_aspect(xy):
        x, y = xy[:, 0], xy[:, 1]
        return torch.sin(0.5 * math.pi * x) * torch.sin(math.pi * y)

    def sine_aspect_grad(xy):
        x, y = xy[:, 0], xy[:, 1]
        return torch.stack((0.5 * math.pi * torch.cos(0.5 * math.pi * x)
                            * torch.sin(math.pi * y),
                            math.pi * torch.sin(0.5 * math.pi * x)
                            * torch.cos(math.pi * y)), dim=1)

    cases["sine_aspect2"] = HeatFieldCase(
        key="sine_aspect2",
        description="T = sin(pi x/2) sin(pi y) on (0,2)x(0,1), q = 1.25 pi^2 T",
        length=2.0, height=1.0,
        temperature=sine_aspect,
        gradient=sine_aspect_grad,
        source=lambda xy: 1.25 * math.pi ** 2 * sine_aspect(xy),
        reference="T = sin(pi x/2) sin(pi y) on a 2:1 domain (MMS)",
        refinement="anisotropic-2to1",
    )

    def quadratic(xy):
        x, y = xy[:, 0], xy[:, 1]
        return x ** 2 - y ** 2

    def quadratic_grad(xy):
        x, y = xy[:, 0], xy[:, 1]
        return torch.stack((2.0 * x, -2.0 * y), dim=1)

    cases["quadratic_harmonic"] = HeatFieldCase(
        key="quadratic_harmonic",
        description="T = x^2 - y^2, q = 0 (harmonic quadratic: nodally exact, "
                    "L2 error = interpolation error)",
        length=1.0, height=1.0,
        temperature=quadratic,
        gradient=quadratic_grad,
        source=lambda xy: torch.zeros(xy.shape[0], dtype=FLOAT),
        reference="T = x^2 - y^2 (harmonic quadratic MMS)",
        nodal_exact_expected=True,
    )

    def bilinear(xy):
        x, y = xy[:, 0], xy[:, 1]
        return x * y

    def bilinear_grad(xy):
        x, y = xy[:, 0], xy[:, 1]
        return torch.stack((y, x), dim=1)

    cases["bilinear_control"] = HeatFieldCase(
        key="bilinear_control",
        description="T = x y, q = 0 (CONTROL: harmonic and bilinear-representable)",
        length=1.0, height=1.0,
        temperature=bilinear,
        gradient=bilinear_grad,
        source=lambda xy: torch.zeros(xy.shape[0], dtype=FLOAT),
        reference="T = x y (harmonic, exactly representable control)",
        representable=True,
    )
    return cases


# ---------------------------------------------------------------------------
# mesh + solve
# ---------------------------------------------------------------------------
def mesh_dims(case: HeatFieldCase, level: int) -> tuple[int, int]:
    """Element counts for one refinement level (isotropic or 2:1 aspect)."""
    if level < 1:
        raise ValueError("level must be positive")
    if case.refinement == "anisotropic-2to1":
        return level, max(level // 2, 1)
    return level, level


def solve_case(case: HeatFieldCase, level: int,
               quad_order: int = DEFAULT_QUAD_ORDER) -> dict[str, object]:
    """One MMS solve on a ``level``-refined mesh, with all error norms."""
    nx, ny = mesh_dims(case, level)
    mesh = structured_quad_mesh(nx, ny, lx=case.length, ly=case.height)
    nodes = mesh.nodes.to(FLOAT)
    elements = mesh.elements
    boundary = mesh.boundary_nodes
    source = case.element_mean_source(nodes, elements)
    model = ThermalModel(
        nodes=nodes,
        elements=elements,
        conductivity=torch.tensor(1.0, dtype=FLOAT),
        density=torch.tensor(1.0, dtype=FLOAT),
        specific_heat=torch.tensor(1.0, dtype=FLOAT),
        fixed_nodes=boundary,
        fixed_temperatures=case.temperature(nodes[boundary]),
        heat_source=source,
        thickness=torch.tensor(1.0, dtype=FLOAT),
        element_type="q4",
    )
    temperature = solve_steady_thermal(model)
    errors = evaluate_errors(nodes, elements, temperature, case, quad_order)
    interpolation = evaluate_errors(
        nodes, elements, case.temperature(nodes), case, quad_order)
    h = math.sqrt(case.length * case.height / float(elements.shape[0]))
    # element-centre field comparison: for a rectangle the bilinear field at the
    # centre is the arithmetic mean of the four nodal values.
    centres = nodes[elements].mean(dim=1)
    t_centre = temperature[elements].mean(dim=1)
    t_ref = case.temperature(centres)
    abs_err = (t_centre - t_ref).abs()
    scale = t_ref.abs().clamp_min(torch.finfo(FLOAT).tiny)
    return {
        "level": level,
        "nx": nx,
        "ny": ny,
        "n_nodes": int(nodes.shape[0]),
        "n_elements": int(elements.shape[0]),
        "h": h,
        "errors": errors,
        "interpolation_errors": interpolation,
        "source_max_abs": float(source.abs().max()),
        "field_comparison": {
            "n_points": int(centres.shape[0]),
            "max_abs": float(abs_err.max()),
            "max_rel": float((abs_err / scale).max()),
            "rms_abs": float(torch.sqrt((abs_err ** 2).mean())),
            "ref_max_abs": float(t_ref.abs().max()),
            "relative_to_peak": float(abs_err.max() / t_ref.abs().max().clamp_min(1e-300)),
        },
    }


def _orders(values: Sequence[float], hs: Sequence[float]) -> list[float]:
    return [math.log(a / b) / math.log(hc / hf)
            for a, b, hc, hf in zip(values[:-1], values[1:], hs[:-1], hs[1:])]


def richardson_residual(errors: Sequence[float], hs: Sequence[float],
                        order: float) -> float:
    """``|e0| / e_fine`` of Roache's extrapolation ``e0 = e_f + (e_f-e_c)/(r^p-1)``.

    The discretisation error has the limit 0, so an error ladder that follows the
    theoretical ``O(h^p)`` law extrapolates to 0; the normalised residual is a
    direct measure of how far the ladder is from that power law.
    """
    if order <= 0.0:
        raise ValueError("order must be positive")
    coarse, fine = errors[-2], errors[-1]
    ratio = hs[-2] / hs[-1]
    e0 = fine + (fine - coarse) / (ratio ** order - 1.0)
    return abs(e0) / fine


def _loglog_slope(hs: Sequence[float], errs: Sequence[float]) -> float:
    """Least-squares slope of ``ln e`` against ``ln h`` over the whole ladder."""
    ln_h = [math.log(h) for h in hs]
    ln_e = [math.log(e) for e in errs]
    mean_h = sum(ln_h) / len(ln_h)
    mean_e = sum(ln_e) / len(ln_e)
    num = sum((a - mean_h) * (b - mean_e) for a, b in zip(ln_h, ln_e))
    den = sum((a - mean_h) ** 2 for a in ln_h)
    return num / den


def study_case(case: HeatFieldCase, levels: Sequence[int] = DEFAULT_LEVELS,
               quad_order: int = DEFAULT_QUAD_ORDER,
               target_order: float = 2.0) -> dict[str, object]:
    """The full h-refinement ladder of one case."""
    levels = tuple(levels)
    if len(levels) < 3:
        raise ValueError("at least three levels are required")
    runs = [solve_case(case, level, quad_order) for level in levels]
    hs = [run["h"] for run in runs]
    l2 = [run["errors"]["l2_relative"] for run in runs]
    h1 = [run["errors"]["h1_relative"] for run in runs]
    l2_interp = [run["interpolation_errors"]["l2_relative"] for run in runs]
    return {
        "case": case.key,
        "description": case.description,
        "reference": case.reference,
        "representable": case.representable,
        "nodal_exact_expected": case.nodal_exact_expected,
        "refinement": case.refinement,
        "levels": list(levels),
        "quad_order": quad_order,
        "runs": runs,
        "h": hs,
        "l2_relative": l2,
        "h1_relative": h1,
        "l2_relative_interpolant": l2_interp,
        "l2_orders": _orders(l2, hs),
        "h1_orders": _orders(h1, hs),
        "l2_orders_interpolant": _orders(l2_interp, hs),
        "l2_order_slope": _loglog_slope(hs, l2),
        "h1_order_slope": _loglog_slope(hs, h1),
        "interpolant_ratio": [a / b for a, b in zip(l2, l2_interp)],
        "richardson_residual": {
            "theoretical": richardson_residual(l2, hs, target_order),
            "observed": richardson_residual(l2, hs, max(_orders(l2, hs)[-1], 1e-3)),
        },
    }


# ---------------------------------------------------------------------------
# reference anchors read from the repository manual (zero hand-copy)
# ---------------------------------------------------------------------------
_MANUAL = Path(__file__).with_name("adaptive_poisson_2d.py")
_L2_ROW = re.compile(r"L2 norm.*?O\(h\^(\d)\)")


def target_l2_order_from_manual(manual: Path | None = None) -> dict[str, object]:
    """Parse the a-priori L2 order of Q4 elements out of the in-repo manual table.

    Returns the exponent together with the verbatim row and its ``file``/``line``
    so a report can cite the manual row it came from.  Nothing is hard-coded: if
    the table changes, the target changes with it and the study fails closed.
    """
    path = Path(manual) if manual is not None else _MANUAL
    text = path.read_text(encoding="utf-8")
    match = _L2_ROW.search(text)
    if match is None:
        raise RuntimeError(f"a-priori L2 order row not found in {path}")
    line = text[:match.start()].count("\n") + 1
    row = text.splitlines()[line - 1].strip()
    return {
        "order": float(match.group(1)),
        "row": row,
        "file": str(path.name),
        "line": line,
    }


# ---------------------------------------------------------------------------
# study driver + verdict
# ---------------------------------------------------------------------------
def quadrature_sensitivity(case: HeatFieldCase, level: int,
                           orders: Sequence[int] = (4, 6, 8)) -> dict[str, object]:
    """Relative change of every error norm when the post-processing rule is raised."""
    norms = {}
    for order in orders:
        run = solve_case(case, level, quad_order=order)
        norms[order] = dict(run["errors"])
    base = norms[orders[0]]
    worst = 0.0
    detail = {}
    for order in orders[1:]:
        for name, value in norms[order].items():
            denom = abs(base[name]) if base[name] != 0.0 else 1.0
            rel = abs(value - base[name]) / denom
            detail[f"{name}@{order}"] = rel
            worst = max(worst, rel)
    return {"case": case.key, "level": level, "orders": list(orders),
            "norms": norms, "worst_relative_change": worst, "detail": detail}


def run_all(levels: Sequence[int] = DEFAULT_LEVELS,
            quad_order: int = DEFAULT_QUAD_ORDER,
            quadrature_case: str = "sine_generation",
            quadrature_level: int = 32) -> dict[str, object]:
    """Every case of the study, plus the target order and the controls."""
    cases = field_cases()
    target = target_l2_order_from_manual()
    studies = {key: study_case(case, levels, quad_order,
                               target_order=float(target["order"]))
               for key, case in cases.items()}
    quadrature = quadrature_sensitivity(cases[quadrature_case], quadrature_level)
    evidence = {
        "schema": SCHEMA,
        "problem": ("-div(k grad T) = q, k = 1, Dirichlet on the whole boundary, "
                    "bilinear Q4 conduction (tensorfem.thermal)"),
        "solver": "tensorfem.thermal.solve_steady_thermal (Q4, 2x2 Gauss, dense direct)",
        "target": target,
        "gates": {
            "l2_pair": GATE_L2_PAIR,
            "l2_finest": list(GATE_L2_FINEST),
            "h1_pair": GATE_H1_PAIR,
            "power_law": GATE_POWER_LAW,
            "control_abs": GATE_CONTROL_ABS,
            "field_rel": GATE_FIELD_REL,
            "quadrature": GATE_QUADRATURE,
        },
        "studies": studies,
        "quadrature_sensitivity": quadrature,
    }
    evidence["verdict"] = verdict(evidence)
    return evidence


def verdict(evidence: Mapping[str, object]) -> dict[str, object]:
    """Apply the predeclared gates to an evidence dictionary."""
    target = float(evidence["target"]["order"])  # type: ignore[index]
    studies = evidence["studies"]  # type: ignore[index]
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for key, study in studies.items():
        if study["representable"]:
            worst = max(max(run["errors"]["max_abs_nodal"],
                            run["errors"]["l2_absolute"]) for run in study["runs"])
            check(f"{key}:control_exact", worst <= GATE_CONTROL_ABS,
                  f"representable field, worst absolute error {worst:.3e} over the "
                  f"whole ladder (gate {GATE_CONTROL_ABS:.0e}); no order claimed")
            continue
        pairs = list(zip(study["l2_orders"], study["runs"][:-1]))
        worst = min(p for p, run in pairs if run["nx"] >= 8)
        check(f"{key}:l2_pair", worst >= GATE_L2_PAIR,
              f"min L2 order over pairs with coarse nx>=8 is {worst:.3f} "
              f"(gate >={GATE_L2_PAIR:.2f}, theory {target:.1f})")
        finest = study["l2_orders"][-1]
        lo, hi = GATE_L2_FINEST
        check(f"{key}:l2_finest", lo <= finest <= hi,
              f"finest-pair L2 order {finest:.3f} inside [{lo:.2f}, {hi:.2f}]")
        worst_h1 = min(p for p, run in zip(study["h1_orders"], study["runs"][:-1])
                       if run["nx"] >= 8)
        check(f"{key}:h1_pair", worst_h1 >= GATE_H1_PAIR,
              f"min H1 order {worst_h1:.3f} (gate >={GATE_H1_PAIR:.2f}, theory 1.0)")
        if study["nodal_exact_expected"]:
            worst_nodal = max(run["errors"]["max_abs_nodal"] for run in study["runs"])
            check(f"{key}:nodal_exact", worst_nodal <= GATE_CONTROL_ABS,
                  f"5-point-Laplacian exactness: worst nodal error over the "
                  f"ladder {worst_nodal:.3e} (gate {GATE_CONTROL_ABS:.0e})")
            ratio = study["interpolant_ratio"][-1]
            check(f"{key}:interp_identity", abs(ratio - 1.0) <= 1.0e-6,
                  f"finest-mesh e_FE / e_interpolant = {ratio:.9f} "
                  f"(the solver must add nothing to interpolation error)")
        residual = study["richardson_residual"]["theoretical"]
        check(f"{key}:power_law", residual <= GATE_POWER_LAW,
              f"Richardson residual at the theoretical order {target:.0f}: "
              f"{residual:.3e} of e_fine (gate <={GATE_POWER_LAW:.0e})")
        field = study["runs"][-1]["field_comparison"]
        check(f"{key}:field_3pct", field["relative_to_peak"] <= GATE_FIELD_REL,
              f"finest mesh, element centres: max |error| / max|T| = "
              f"{100 * field['relative_to_peak']:.4f}% "
              f"(gate {100 * GATE_FIELD_REL:.0f}%, peak |T| = "
              f"{field['ref_max_abs']:.4f})")

    quad = evidence["quadrature_sensitivity"]  # type: ignore[index]
    check("quadrature_postprocessing",
          quad["worst_relative_change"] <= GATE_QUADRATURE,
          f"worst relative change of an error norm for 4 -> 6/8 point rules: "
          f"{quad['worst_relative_change']:.2e} (gate <={GATE_QUADRATURE:.0e})")

    failed = [item["name"] for item in checks if not item["passed"]]
    return {"passed": not failed, "checks": checks, "failed": failed}


__all__ = [
    "SCHEMA",
    "GATE_L2_PAIR",
    "GATE_L2_FINEST",
    "GATE_H1_PAIR",
    "GATE_POWER_LAW",
    "GATE_CONTROL_ABS",
    "GATE_FIELD_REL",
    "GATE_QUADRATURE",
    "DEFAULT_LEVELS",
    "DEFAULT_QUAD_ORDER",
    "HeatFieldCase",
    "field_cases",
    "gauss_legendre_1d",
    "q4_shape",
    "q4_shape_grad",
    "evaluate_errors",
    "mesh_dims",
    "solve_case",
    "study_case",
    "richardson_residual",
    "target_l2_order_from_manual",
    "quadrature_sensitivity",
    "run_all",
    "verdict",
]