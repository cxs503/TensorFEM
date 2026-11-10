"""Mesh-level von Karman plate postbuckling traced by Crisfield arc length.

Why this module exists
----------------------
The repository already owns several *plate buckling* evidences, but none of
them traces a postbuckling equilibrium path on a real finite element mesh:

* :mod:`tensorfem.shell4_buckling` / :mod:`tensorfem.stiffened_panel_buckling`
  assemble a linearised bifurcation pencil -- eigenvalue only, no path.
* :mod:`tensorfem.marine_plate_postbuckling` reduces the plate to a *single*
  von Karman--Koiter amplitude equation and integrates that scalar problem; it
  is documented as "not a shell finite element".
* :mod:`tensorfem.shell4_arc_path` exercises the Crisfield solver on a
  one-element shell cantilever and is documented as "not a postbuckling
  qualification".
* :mod:`tensorfem.general_shell_nonlinear` provides a corotational shell, but
  its element tangent is built by repeated reverse-mode differentiation per
  element, which puts a 100+ element plate out of reach of the hundreds of
  assemblies an arc-length trace needs.

This module closes that gap with its own element, so the two claims that matter
can be *separated*: the **discretisation** (a real mesh, many modes, membrane
redistribution) and the **continuation** (Crisfield arc length from
:mod:`tensorfem.arc_length`, reused rather than reimplemented).

Element
-------
A rectangular von Karman plate element with six DOFs per node
``(u, v, w, w_,x, w_,y, w_,xy)``:

* **bending** -- Bogner--Fox--Schmit bicubic Hermite (`C1` conforming, exactly
  reproduces bicubic polynomials);
* **membrane** -- the bilinear `Q4` field on the same four corners;
* **coupling** -- the von Karman strain measure

      eps_x = u,x + (w_t,x^2 - w0,x^2)/2
      eps_y = v,y + (w_t,y^2 - w0,y^2)/2
      g_xy  = u,y + v,x + (w_t,x w_t,y - w0,x w0,y),   w_t = w0 + w,

  so an initial geometric imperfection ``w0`` enters as a stress-free initial
  configuration: curvature of ``w0`` alone carries no energy and membrane strain
  of ``w0`` alone carries no energy.  Curvatures use the added deflection ``w``
  only.

The element tangent is analytic (material ``B^T C B`` plus the initial-stress
``G^T S G`` term), so one assembly of a 16x16 mesh takes milliseconds and a full
arc-length trace runs in seconds.

Reference oracles
-----------------
1. **Linear bifurcation** against the Timoshenko/Galerkin closed form
   ``sigma_cr = k pi^2 D / (b^2 t)`` with ``k = min_m (m b/a + a/(m b))^2``.
2. **Postbuckling path** against the classical one-mode von Karman solution of
   :func:`koiter_load_factor` (assumptions stated there)::

       N/N_cr = (1 - w0/A) + c (A/t)^2
       c = 3 (1 - nu^2) (alpha^4 + beta^4) / (4 (alpha^2 + beta^2)^2)

   with ``A`` the *total* transverse amplitude, imperfection included.
3. **Negative control** -- with no compressive load the assembled tangent must
   still be positive definite (no spurious zero-energy mode), and with a
   *tensile* reference load no positive buckling factor may be reported.

Scope and honesty
-----------------
This is a small-rotation von Karman plate, not a finite-rotation shell.
Yielding, stiffener web local buckling, contact and plasticity are outside the
model; the module reports numbers, not scantlings.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch

from .arc_length import ArcLengthProblem, solve_arc_length

DOF_PER_NODE = 6
U, V, W, WX, WY, WXY = range(DOF_PER_NODE)

_BENDING_LOCAL = (2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23)
_MEMBRANE_LOCAL = (0, 1, 6, 7, 12, 13, 18, 19)

_GAUSS = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
_GAUSS_W = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------
def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _count(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _poisson(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("poisson must be a real number")
    result = float(value)
    if not math.isfinite(result) or not -1.0 < result < 0.5:
        raise ValueError("poisson must be finite and lie between -1 and 0.5")
    return result


def _imperfection_callable(imperfection):
    if imperfection is None:
        return _zero_imperfection
    if not callable(imperfection):
        raise TypeError("imperfection must be None or a callable w0(x, y)")
    return imperfection


# --------------------------------------------------------------------------
# closed-form oracles
# --------------------------------------------------------------------------
def timoshenko_critical_stress(*, a: float, b: float, thickness: float,
                               young: float, poisson: float,
                               half_waves: int | None = None) -> dict[str, float]:
    """Simply supported isotropic plate in uniaxial compression (Timoshenko).

    ``a`` is the loaded length and ``b`` the supported width, so the classical
    coefficient is ``k = min_m (m b/a + a/(m b))^2``.
    """
    a, b = _positive("a", a), _positive("b", b)
    thickness = _positive("thickness", thickness)
    young, poisson = _positive("young", young), _poisson(poisson)
    rigidity = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    candidates = [half_waves] if half_waves else list(range(1, 61))
    m, k = min(((m, (m * b / a + a / (m * b)) ** 2) for m in candidates),
               key=lambda item: item[1])
    return {"half_waves": int(m), "buckling_coefficient": float(k),
            "rigidity": float(rigidity),
            "sigma_cr": float(k * math.pi ** 2 * rigidity / (b ** 2 * thickness)),
            "critical_line_load": float(k * math.pi ** 2 * rigidity / b ** 2)}


def koiter_amplitude_coefficient(*, a: float, b: float, poisson: float,
                                 m: int = 1, n: int = 1) -> float:
    """Coefficient ``c`` of ``N/N_cr = (1 - w0/A) + c (A/t)^2``.

    One-mode von Karman energy (``alpha``/``beta`` are the wave numbers)::

        V = D/2 (A-w0)^2 (alpha^2+beta^2)^2 + E A^4 (alpha^4+beta^4)/(256 t)
            - N A^2 alpha^2 / 8

    Stationarity in ``A`` gives ``N/N_cr = (1 - w0/A) +
    3(1-nu^2)(alpha^4+beta^4) A^2 / (4 (alpha^2+beta^2)^2 t^2)``.  The same
    stationarity as ``A -> 0`` reproduces ``N_cr = D (alpha^2+beta^2)^2/alpha^2``,
    i.e. exactly the Timoshenko value -- the consistency check a reviewer should
    redo before trusting ``c``.
    """
    a, b = _positive("a", a), _positive("b", b)
    poisson = _poisson(poisson)
    m, n = _count("m", m, 0), _count("n", n, 0)
    alpha, beta = max(m, 1) * math.pi / a, max(n, 1) * math.pi / b
    return float(3.0 * (1.0 - poisson ** 2) * (alpha ** 4 + beta ** 4)
                 / (4.0 * (alpha ** 2 + beta ** 2) ** 2))


def koiter_load_factor(*, amplitude: float, thickness: float, imperfection: float,
                       coefficient: float) -> float:
    """``N/N_cr`` of the one-mode von Karman branch at a **total** amplitude.

    Assumptions inherited from the classical reduction: one buckling mode, the
    particular Airy stress function only (zero *mean* membrane stress in the
    unloaded direction), small rotations, elastic material.
    """
    thickness = _positive("thickness", thickness)
    coefficient = _positive("coefficient", coefficient)
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude must be positive and finite")
    if not math.isfinite(imperfection) or imperfection < 0.0:
        raise ValueError("imperfection must be nonnegative and finite")
    return float((1.0 - imperfection / amplitude)
                 + coefficient * (amplitude / thickness) ** 2)


def sine_imperfection(*, a: float, b: float, amplitude: float, m: int = 1,
                      n: int = 1) -> Callable[[float, float], tuple[float, float, float]]:
    """Stress-free sine imperfection ``w0 = A sin(m pi x/a) sin(n pi y/b)``.

    Returns ``(w0, w0_,x, w0_,y)``.  ``w0`` vanishes on all four edges, which is
    the compatibility condition the simply supported boundary constraints impose
    on the transverse field.
    """
    a, b = _positive("a", a), _positive("b", b)
    m, n = _count("m", m), _count("n", n)
    if not math.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    alpha, beta = m * math.pi / a, n * math.pi / b

    def field(x: float, y: float) -> tuple[float, float, float]:
        s, c = math.sin(alpha * x), math.cos(alpha * x)
        t, d = math.sin(beta * y), math.cos(beta * y)
        return (amplitude * s * t, amplitude * alpha * c * t, amplitude * beta * s * d)

    return field


def _zero_imperfection(_x: float, _y: float) -> tuple[float, float, float]:
    return (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# mesh and boundary conditions
# --------------------------------------------------------------------------
def plate_mesh(*, a: float, b: float, nx: int, ny: int,
               dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured rectangular Q4 grid: ``nodes (n,2)``, ``elements (m,4)``."""
    a, b = _positive("a", a), _positive("b", b)
    nx, ny = _count("nx", nx), _count("ny", ny)
    xs = [a * i / nx for i in range(nx + 1)]
    ys = [b * j / ny for j in range(ny + 1)]
    nodes = [[x, y] for y in ys for x in xs]
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elements.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    return (torch.tensor(nodes, dtype=dtype),
            torch.tensor(elements, dtype=torch.long))


def simply_supported_boundary(nodes: torch.Tensor, *, a: float, b: float,
                              tol: float = 1e-9) -> dict[str, object]:
    """Simply supported edges plus the in-plane support of the test rig.

    Out-of-plane: ``w = 0`` on all four edges.  Because ``w`` vanishes
    identically on a straight edge, the slope *along* the edge also vanishes:
    for the bicubic Hermite that means ``w_,x = 0`` on the ``y`` edges and
    ``w_,y = 0`` on the ``x`` edges.

    In-plane: ``u = 0`` on the ``x = 0`` edge (the loaded edge ``x = a`` carries
    the reference force), ``v`` free on every edge except one pinned corner.
    The unloaded edges therefore stay stress free and no rigid in-plane motion
    survives.
    """
    a, b = _positive("a", a), _positive("b", b)
    tol = float(tol)
    fixed: set[int] = set()
    loaded: list[int] = []
    left: list[int] = []
    for index, (x, y) in enumerate(nodes.tolist()):
        is_left, is_right = abs(x) <= tol, abs(x - a) <= tol
        is_bottom, is_top = abs(y) <= tol, abs(y - b) <= tol
        if is_left or is_right or is_bottom or is_top:
            fixed.add(6 * index + W)
        if is_bottom or is_top:
            fixed.add(6 * index + WX)
        if is_left or is_right:
            fixed.add(6 * index + WY)
        if is_left:
            fixed.add(6 * index + U)
            left.append(index)
        if is_right:
            loaded.append(index)
    if not left or not loaded:
        raise ValueError("mesh must have nodes on both x=0 and x=a edges")
    corner = min(range(len(nodes)),
                 key=lambda i: float(nodes[i, 0] ** 2 + nodes[i, 1] ** 2))
    fixed.add(6 * corner + V)
    return {"fixed": sorted(fixed), "loaded_edge": loaded, "left_edge": left,
            "ndof": DOF_PER_NODE * len(nodes), "corner": corner}


def reference_edge_load(nodes: torch.Tensor, boundary: dict[str, object], *,
                        b: float, line_load: float) -> torch.Tensor:
    """Nodal force vector of a uniform line load on the ``x = a`` edge.

    ``line_load`` is the force per unit width, positive in compression.  The
    resultant ``line_load * b`` is lumped with trapezoidal tributary lengths so
    that the applied edge stress is uniform and the resultant is reproduced to
    machine precision (checked below).
    """
    b = _positive("b", b)
    if not math.isfinite(line_load):
        raise ValueError("line_load must be finite")
    loaded = list(boundary["loaded_edge"])
    ys = sorted(nodes[node, 1].item() for node in loaded)
    if len(ys) < 2:
        raise ValueError("the loaded edge needs at least two nodes")
    order = sorted(range(len(loaded)), key=lambda i: float(nodes[loaded[i], 1]))
    loaded = [loaded[i] for i in order]
    force = torch.zeros(int(boundary["ndof"]), dtype=nodes.dtype)
    for position, y in enumerate(ys):
        if position == 0:
            tributary = 0.5 * (ys[1] - ys[0])
        elif position == len(ys) - 1:
            tributary = 0.5 * (ys[-1] - ys[-2])
        else:
            tributary = 0.5 * (ys[position + 1] - ys[position - 1])
        force[6 * loaded[position] + U] = -line_load * tributary
    resultant = float(-force.reshape(-1, DOF_PER_NODE)[:, U].sum())
    if abs(resultant - line_load * b) > 1e-9 * max(abs(line_load * b), 1e-30):
        raise ValueError("edge load lumping does not reproduce the resultant")
    return force


# --------------------------------------------------------------------------
# element preparation
#
# The element kernels below run on ``numpy`` float64 arrays: every operation is
# a handful of doubles, and per-op interpreter overhead dominates the cost at
# this size, so numpy keeps a full arc-length trace interactive.  Torch tensors
# are used at the public boundary because :mod:`tensorfem.arc_length` is written
# in torch.
# --------------------------------------------------------------------------
def _hermite(x: float, length: float):
    """Cubic Hermite bases ``(h1..h4)`` and derivatives 1..2 on ``[0, length]``."""
    u = x / length
    h = (1.0 - 3.0 * u * u + 2.0 * u ** 3,
         x * (1.0 - u) ** 2,
         3.0 * u * u - 2.0 * u ** 3,
         length * (u ** 3 - u ** 2))
    d1 = ((-6.0 * u + 6.0 * u * u) / length,
          (1.0 - u) * (1.0 - 3.0 * u),
          (6.0 * u - 6.0 * u * u) / length,
          3.0 * u * u - 2.0 * u)
    d2 = ((-6.0 + 12.0 * u) / length ** 2,
          (6.0 * u - 4.0) / length,
          (6.0 - 12.0 * u) / length ** 2,
          (6.0 * u - 2.0) / length)
    return h, d1, d2


@dataclass(frozen=True)
class _GaussPoint:
    weight: float                  # quadrature weight * Jacobian
    bending: np.ndarray            # (3,16) [w_xx, w_yy, 2 w_xy]
    membrane: np.ndarray           # (3,8) linear membrane strain operator
    gradient: np.ndarray           # (2,16) [w_x, w_y]
    imperfection: tuple[float, float]


@dataclass(frozen=True)
class _PreparedElement:
    dofs: np.ndarray
    points: tuple[_GaussPoint, ...]
    bending_stiffness: np.ndarray  # (16,16) constant, integrated


def _as_array(tensor, dtype=np.float64) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return np.ascontiguousarray(tensor, dtype=dtype)
    return tensor.detach().cpu().numpy().astype(dtype, copy=False)


def _prepare(nodes, elements,
             imperfection: Callable[[float, float], tuple[float, float, float]],
             *, poisson: float, young: float,
             thickness: float) -> tuple[_PreparedElement, ...]:
    nodes = _as_array(nodes)
    elements = _as_array(elements).astype(np.int64)
    rigidity = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    bending_material = rigidity * np.array(
        [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, (1.0 - poisson) / 2.0]])
    membrane_material = thickness * young / (1.0 - poisson ** 2) * np.array(
        [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, (1.0 - poisson) / 2.0]])
    bending_ids = np.array(_BENDING_LOCAL)
    membrane_ids = np.array(_MEMBRANE_LOCAL)
    x_side = (0, 2, 2, 0)   # corners 0..3 -> left, right, right, left
    y_side = (0, 0, 2, 2)   # corners 0..3 -> bottom, bottom, top, top
    # DOF k of a corner: 0 -> (0,0), 1 -> (d/dx,0), 2 -> (0,d/dy), 3 -> (d/dx,d/dy)
    derivative_pair = ((0, 0), (1, 0), (0, 1), (1, 1))
    prepared = []
    for conn in elements:
        corners = nodes[conn]
        x0, y0 = corners[0]
        x1, y1 = corners[2]
        ae, be = x1 - x0, y1 - y0
        points = []
        for i, gx in enumerate(_GAUSS):
            for j, gy in enumerate(_GAUSS):
                x = x0 + 0.5 * ae * (1.0 + gx)
                y = y0 + 0.5 * be * (1.0 + gy)
                hx = _hermite(x - x0, ae)
                hy = _hermite(y - y0, be)
                bending = np.zeros((3, 16))
                gradient = np.zeros((2, 16))
                for corner in range(4):
                    for k, (a_derivative, b_derivative) in enumerate(derivative_pair):
                        # The 1D basis of a "value" DOF is h1/h3 and of a
                        # "slope" DOF h2/h4, so the basis index is the corner
                        # side plus the derivative flag; ``hx[p][basis]`` is the
                        # p-th derivative of that basis.
                        xs = x_side[corner] + a_derivative
                        ys = y_side[corner] + b_derivative
                        column = 4 * corner + k
                        bending[0, column] = hx[2][xs] * hy[0][ys]
                        bending[1, column] = hx[0][xs] * hy[2][ys]
                        bending[2, column] = 2.0 * hx[1][xs] * hy[1][ys]
                        gradient[0, column] = hx[1][xs] * hy[0][ys]
                        gradient[1, column] = hx[0][xs] * hy[1][ys]
                membrane = np.zeros((3, 8))
                for corner, (sx, sy) in enumerate(((-1.0, -1.0), (1.0, -1.0),
                                                   (1.0, 1.0), (-1.0, 1.0))):
                    dx = 0.25 * sx * (1.0 + sy * gy) * 2.0 / ae
                    dy = 0.25 * sy * (1.0 + sx * gx) * 2.0 / be
                    membrane[0, 2 * corner] = dx
                    membrane[1, 2 * corner + 1] = dy
                    membrane[2, 2 * corner] = dy
                    membrane[2, 2 * corner + 1] = dx
                weight = _GAUSS_W[i] * _GAUSS_W[j] * 0.25 * ae * be
                w0, w0x, w0y = imperfection(float(x), float(y))
                points.append(_GaussPoint(weight, bending, membrane, gradient,
                                          (float(w0x), float(w0y))))
        bending_stiffness = np.zeros((16, 16))
        for point in points:
            bending_stiffness += point.weight * (point.bending.T @ bending_material
                                                 @ point.bending)
        dofs = np.array([6 * node + k for node in conn for k in range(DOF_PER_NODE)])
        prepared.append(_PreparedElement(dofs, tuple(points), bending_stiffness))
    return tuple(prepared)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def assemble(nodes, prepared: tuple[_PreparedElement, ...], dofs, *, young: float,
             poisson: float, thickness: float,
             tangent: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Internal force vector and (optionally) the analytic tangent.

    ``dofs`` may be a torch tensor or a numpy array; the returned dtype follows
    the input when a torch tensor is given.
    """
    young = _positive("young", young)
    thickness = _positive("thickness", thickness)
    poisson = _poisson(poisson)
    values = _as_array(dofs)
    ndof = int(values.size)
    if ndof != DOF_PER_NODE * len(_as_array(nodes)):
        raise ValueError("wrong global DOF vector length")
    membrane_material = thickness * young / (1.0 - poisson ** 2) * np.array(
        [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, (1.0 - poisson) / 2.0]])
    bending_ids = np.array(_BENDING_LOCAL)
    membrane_ids = np.array(_MEMBRANE_LOCAL)
    internal = np.zeros(ndof)
    stiffness = np.zeros((ndof, ndof)) if tangent else None
    for element in prepared:
        local = values[element.dofs]
        bending_dofs = local[bending_ids]
        membrane_dofs = local[membrane_ids]
        force = np.zeros(24)
        force[bending_ids] = element.bending_stiffness @ bending_dofs
        if tangent:
            element_stiffness = np.zeros((24, 24))
            element_stiffness[np.ix_(bending_ids, bending_ids)] = element.bending_stiffness
        else:
            element_stiffness = None
        for point in element.points:
            strain_linear = point.membrane @ membrane_dofs
            gradient = point.gradient @ bending_dofs
            w0x, w0y = point.imperfection
            total_x = gradient[0] + w0x
            total_y = gradient[1] + w0y
            resultants = membrane_material @ np.array((
                strain_linear[0] + 0.5 * (total_x ** 2 - w0x ** 2),
                strain_linear[1] + 0.5 * (total_y ** 2 - w0y ** 2),
                strain_linear[2] + total_x * total_y - w0x * w0y))
            force[membrane_ids] += point.weight * (point.membrane.T @ resultants)
            q0 = total_x * resultants[0] + total_y * resultants[2]
            q1 = total_y * resultants[1] + total_x * resultants[2]
            force[bending_ids] += point.weight * (q0 * point.gradient[0]
                                                  + q1 * point.gradient[1])
            if tangent:
                operator = np.zeros((3, 24))
                operator[:, membrane_ids] = point.membrane
                operator[0, bending_ids] = total_x * point.gradient[0]
                operator[1, bending_ids] = total_y * point.gradient[1]
                operator[2, bending_ids] = (total_y * point.gradient[0]
                                            + total_x * point.gradient[1])
                element_stiffness += point.weight * (operator.T @ membrane_material
                                                     @ operator)
                stress = np.array([[resultants[0], resultants[2]],
                                   [resultants[2], resultants[1]]])
                element_stiffness[np.ix_(bending_ids, bending_ids)] += (
                    point.weight * (point.gradient.T @ stress @ point.gradient))
        internal[element.dofs] += force
        if tangent:
            stiffness[np.ix_(element.dofs, element.dofs)] += element_stiffness
    internal_tensor = torch.from_numpy(internal)
    stiffness_tensor = torch.from_numpy(stiffness) if tangent else None
    if isinstance(dofs, torch.Tensor):
        internal_tensor = internal_tensor.to(dofs.dtype)
        if tangent:
            stiffness_tensor = stiffness_tensor.to(dofs.dtype)
    return internal_tensor, stiffness_tensor


# --------------------------------------------------------------------------
# linear bifurcation
# --------------------------------------------------------------------------
def _free_dofs(boundary: dict[str, object]) -> np.ndarray:
    fixed = set(boundary["fixed"])
    return np.array([index for index in range(int(boundary["ndof"]))
                     if index not in fixed], dtype=np.int64)


def linear_buckling(*, a: float, b: float, thickness: float, young: float,
                    poisson: float, nx: int, ny: int, modes: int = 1,
                    line_load: float = 1.0) -> dict[str, object]:
    """Linearised bifurcation of the simply supported plate.

    The geometric operator is the *difference* ``K_T(u_pre) - K_T(0)`` of the
    assembled nonlinear tangent at unit reference line load, so the prebuckling
    state is the solution of the same element instead of a hand-injected stress
    field.  ``critical_line_load`` is therefore the critical line load in N/m for
    a unit reference load.
    """
    thickness = _positive("thickness", thickness)
    modes = _count("modes", modes)
    nodes, elements = plate_mesh(a=a, b=b, nx=nx, ny=ny)
    boundary = simply_supported_boundary(nodes, a=a, b=b)
    prepared = _prepare(nodes, elements, _zero_imperfection, poisson=_poisson(poisson),
                        young=_positive("young", young), thickness=thickness)
    ndof = int(boundary["ndof"])
    free = _free_dofs(boundary)
    _, base = assemble(nodes, prepared, np.zeros(ndof), young=young, poisson=poisson,
                       thickness=thickness, tangent=True)
    base_free = base.numpy()[np.ix_(free, free)]
    if not math.isfinite(line_load) or line_load == 0.0:
        raise ValueError("line_load must be finite and nonzero")
    load = reference_edge_load(nodes, boundary, b=b, line_load=line_load).numpy()[free]
    preload = np.zeros(ndof)
    preload[free] = np.linalg.solve(base_free, load)
    _, loaded = assemble(nodes, prepared, preload, young=young, poisson=poisson,
                         thickness=thickness, tangent=True)
    geometric = loaded.numpy()[np.ix_(free, free)] - base_free
    geometric = 0.5 * (geometric + geometric.T)
    try:
        cholesky = np.linalg.cholesky(base_free)
    except np.linalg.LinAlgError as error:
        raise ValueError("the base tangent is not positive definite: "
                         "no buckling problem exists") from error
    inverse = np.linalg.solve(cholesky, -geometric)
    mapped = np.linalg.solve(cholesky, inverse.T).T
    mapped = 0.5 * (mapped + mapped.T)
    values, vectors = np.linalg.eigh(mapped)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    # Under tension the whole mapped spectrum is (numerically) non-positive, and
    # round-off leaves spurious ``+1e-23`` eigenvalues whose reciprocal is a
    # meaningless ``1e23`` "buckling factor".  A relative floor keeps the
    # negative control honest instead of spectacular.
    floor = float(np.abs(values).max()) * 1e-10 if values.size else 0.0
    if values[0] <= floor:
        return {"buckling": False, "rationale": "no positive buckling factor",
                "max_mapped_eigenvalue": float(values[0]),
                "eigenvalue_floor": floor,
                "base_positive_definite": True, "n_dofs": ndof,
                "n_free": int(free.size), "mesh": [nx, ny],
                "reference_line_load": line_load, "load_factors": []}
    factors, shapes, residuals = [], [], []
    for index in range(min(modes, int((values > floor).sum()))):
        factor = 1.0 / float(values[index])
        reduced = np.linalg.solve(cholesky.T, vectors[:, index])
        shape = np.zeros(ndof)
        shape[free] = reduced
        left = base_free @ reduced
        right = geometric @ reduced
        residuals.append(float(np.linalg.norm(left + factor * right))
                         / max(float(np.linalg.norm(left)) + factor * float(np.linalg.norm(right)),
                               1e-300))
        factors.append(factor)
        shapes.append(shape)
    mode = shapes[0]
    amplitude = float(np.abs(mode[free]).max())
    mode = mode / amplitude if amplitude > 0.0 else mode
    return {"buckling": True, "critical_line_load": factors[0],
            "critical_stress": factors[0] / thickness, "load_factors": factors,
            "mode": torch.from_numpy(mode), "residuals": residuals,
            "n_dofs": ndof, "n_free": int(free.size), "mesh": [nx, ny],
            "base_positive_definite": True,
            "preload": torch.from_numpy(preload), "nodes": nodes, "free": free}


def base_tangent_spectrum(*, a: float, b: float, thickness: float, young: float,
                          poisson: float, nx: int, ny: int) -> dict[str, object]:
    """Negative control: spectrum of the assembled tangent at zero load."""
    nodes, elements = plate_mesh(a=a, b=b, nx=nx, ny=ny)
    boundary = simply_supported_boundary(nodes, a=a, b=b)
    prepared = _prepare(nodes, elements, _zero_imperfection, poisson=_poisson(poisson),
                        young=_positive("young", young),
                        thickness=_positive("thickness", thickness))
    free = _free_dofs(boundary)
    _, stiffness = assemble(nodes, prepared, np.zeros(int(boundary["ndof"])), young=young,
                            poisson=poisson, thickness=thickness, tangent=True)
    values = np.linalg.eigvalsh(stiffness.numpy()[np.ix_(free, free)])
    return {"min_eigenvalue": float(values.min()), "max_eigenvalue": float(values.max()),
            "positive_definite": bool(values.min() > 0.0),
            "n_free": int(free.size), "mesh": [nx, ny]}


# --------------------------------------------------------------------------
# postbuckling path
# --------------------------------------------------------------------------
def dof_scale(free: np.ndarray, *, a: float, b: float) -> np.ndarray:
    """Per-DOF scaling that makes the arc-length metric amplitude based.

    The cubic Hermite rotation DOF ``w_,x`` scales like ``A k_x`` and the mixed
    DOF ``w_,xy`` like ``A k_x k_y`` relative to the transverse amplitude ``A``,
    so the returned weights are the *physical magnitude per unit amplitude* of
    each DOF family.  Dividing a DOF by its weight makes one arc-length step
    mean one step of physical bending amplitude; without it the arc constraint
    is dominated by the high-order rotational DOFs and the trace stalls.
    """
    a, b = _positive("a", a), _positive("b", b)
    kx, ky = math.pi / a, math.pi / b
    weights = np.ones(int(free.size))
    kinds = np.asarray(free) % DOF_PER_NODE
    weights[kinds == WX] = kx
    weights[kinds == WY] = ky
    weights[kinds == WXY] = kx * ky
    return weights


def _arc_problem(nodes, prepared, boundary, *, young, poisson, thickness,
                 free: np.ndarray, reference_load: torch.Tensor,
                 load_offset: float = 0.0,
                 scale: np.ndarray | None = None) -> ArcLengthProblem:
    """Arc-length problem with an optional preload offset.

    ``internal(u) - (load_offset + mu) * reference_load = 0`` is presented to the
    continuation solver as ``(internal(u) - load_offset * reference_load)
    - mu * reference_load = 0``, so a trace can start from a preloaded
    equilibrium point instead of spending its arc budget on the boring
    prebuckling branch.
    """
    ndof = int(boundary["ndof"])
    weights = (np.ones(int(free.size)) if scale is None
               else np.asarray(scale, dtype=np.float64))
    if weights.shape != (int(free.size),) or not np.all(weights > 0.0):
        raise ValueError("dof scaling must be positive and match the free DOFs")
    reduced_reference = (reference_load[free].numpy() * weights)
    reduced_reference = torch.from_numpy(reduced_reference)

    def response(scaled: torch.Tensor):
        full = np.zeros(ndof)
        full[free] = scaled.detach().cpu().numpy() * weights
        internal, stiffness = assemble(nodes, prepared, full, young=young,
                                       poisson=poisson, thickness=thickness,
                                       tangent=True)
        shifted = internal.numpy()[free] - load_offset * (reduced_reference.numpy() / weights)
        scaled_internal = shifted * weights
        reduced = stiffness.numpy()[np.ix_(free, free)]
        reduced = weights[:, None] * reduced * weights[None, :]
        return (torch.from_numpy(scaled_internal).to(scaled.dtype),
                torch.from_numpy(reduced).to(scaled.dtype))

    return ArcLengthProblem(response, reduced_reference)


def trace_postbuckling(*, a: float, b: float, thickness: float, young: float,
                       poisson: float, nx: int, ny: int,
                       imperfection_amplitude: float = 0.0,
                       steps: int = 60, step_size: float = 1e-3,
                       load_scale: float | None = None, tolerance: float = 1e-6,
                       target_load_increment: float = 0.02,
                       max_iterations: int = 12,
                       reference_line_load: float | None = None,
                       half_waves: int | None = None,
                       start_load_factor: float = 0.0,
                       maximum_step: float | None = None,
                       newton_tolerance: float = 1e-6,
                       newton_iterations: int = 25) -> dict[str, object]:
    """Trace the von Karman equilibrium path of the (imperfect) plate.

    ``reference_line_load`` defaults to the Timoshenko critical line load, so the
    arc-length load factor *is* ``N/N_cr``.  ``imperfection_amplitude`` seeds the
    sine imperfection of mode ``(m, 1)``, with ``m`` the Timoshenko half-wave
    count unless ``half_waves`` is given.
    """
    thickness = _positive("thickness", thickness)
    steps = _count("steps", steps)
    step_size = _positive("step_size", step_size)
    if target_load_increment <= 0.0 or not math.isfinite(target_load_increment):
        raise ValueError("target_load_increment must be positive and finite")
    load_scale = _positive("load_scale", load_scale if load_scale is not None
                           else step_size / target_load_increment)
    closed = timoshenko_critical_stress(a=a, b=b, thickness=thickness, young=young,
                                       poisson=poisson, half_waves=half_waves)
    reference = (closed["critical_line_load"] if reference_line_load is None
                 else _positive("reference_line_load", reference_line_load))
    m = int(closed["half_waves"])
    imperfection = (sine_imperfection(a=a, b=b, amplitude=imperfection_amplitude, m=m)
                    if imperfection_amplitude else _zero_imperfection)
    nodes, elements = plate_mesh(a=a, b=b, nx=nx, ny=ny)
    boundary = simply_supported_boundary(nodes, a=a, b=b)
    prepared = _prepare(nodes, elements, imperfection, poisson=poisson,
                        young=young, thickness=thickness)
    free = _free_dofs(boundary)
    load = reference_edge_load(nodes, boundary, b=b, line_load=reference)
    scale = dof_scale(free, a=a, b=b)
    problem = _arc_problem(nodes, prepared, boundary, young=young, poisson=poisson,
                           thickness=thickness, free=free, reference_load=load,
                           load_offset=start_load_factor, scale=scale)
    coordinates = _as_array(nodes)
    centre = int(np.argmin((coordinates[:, 0] - 0.5 * a) ** 2
                           + (coordinates[:, 1] - 0.5 * b) ** 2))
    reduced = np.zeros(int(free.size))
    newton_residual = None
    if start_load_factor:
        reference_reduced = load.numpy()[free]
        _, base = assemble(nodes, prepared, np.zeros(int(boundary["ndof"])),
                           young=young, poisson=poisson, thickness=thickness,
                           tangent=True)
        base_free = base.numpy()[np.ix_(free, free)]
        reduced = start_load_factor * np.linalg.solve(base_free, reference_reduced)
        for iteration in range(1, newton_iterations + 1):
            internal, stiffness = assemble(nodes, prepared, _expand(reduced, free,
                                                                    int(boundary["ndof"])),
                                           young=young, poisson=poisson,
                                           thickness=thickness, tangent=True)
            residual = (internal.numpy()[free]
                        - start_load_factor * reference_reduced)
            residual_scale = max(float(np.linalg.norm(reference_reduced))
                                 * start_load_factor, 1e-30)
            newton_residual = float(np.linalg.norm(residual)) / residual_scale
            if newton_residual <= newton_tolerance:
                break
            reduced = reduced + np.linalg.solve(
                stiffness.numpy()[np.ix_(free, free)], -residual)
    start = torch.from_numpy(np.ascontiguousarray(reduced / scale))
    result = solve_arc_length(problem, start, steps=steps, step_size=step_size,
                              load_scale=load_scale, tolerance=tolerance,
                              max_iterations=max_iterations,
                              maximum_step=maximum_step or step_size)
    points = []
    for point in result.points:
        full = np.zeros(int(boundary["ndof"]))
        full[free] = point.displacement.detach().cpu().numpy() * scale
        amplitude = float(full[6 * centre + W])
        points.append({"load_factor": float(point.load_factor) + start_load_factor,
                       "increment_factor": float(point.load_factor),
                       "amplitude": amplitude,
                       "amplitude_over_t": amplitude / thickness,
                       "total_amplitude_over_t": (amplitude + imperfection_amplitude) / thickness,
                       "step": int(point.step),
                       "iterations": int(point.iterations),
                       "displacement": full})
    return {"points": points, "converged": bool(result.converged),
            "load_scale": load_scale,
            "reference_line_load": reference, "critical_stress": closed["sigma_cr"],
            "half_waves": m, "centre_node": centre, "mesh": [nx, ny],
            "imperfection_amplitude": imperfection_amplitude,
            "n_free": int(free.size), "free": free, "nodes": nodes,
            "steps": steps, "step_size": step_size, "final_step_size": float(result.step_size),
            "start_load_factor": start_load_factor,
            "start_newton_residual": newton_residual}


def _expand(reduced: np.ndarray, free: np.ndarray, ndof: int) -> np.ndarray:
    full = np.zeros(ndof)
    full[free] = reduced
    return full


def mac(first, second) -> float:
    """Modal assurance criterion ``(a.b)^2 / (|a|^2 |b|^2)`` of two vectors."""
    left = _as_array(first).reshape(-1)
    right = _as_array(second).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("MAC operands must have the same length")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        raise ValueError("MAC operands must be nonzero")
    return float(abs(np.dot(left, right)) ** 2 / denominator ** 2)


def nodal_shape_mac(nodes, field, *, a: float, b: float, m: int = 1,
                    n: int = 1) -> float:
    """MAC of a nodal transverse field against ``sin(m pi x/a) sin(n pi y/b)``.

    The comparison uses the transverse nodal values only (one per mesh node), so
    it is free of any rotation or curvature sign convention.
    """
    coordinates = _as_array(nodes)
    values = _as_array(field).reshape(-1)
    if values.size != coordinates.shape[0]:
        raise ValueError("field must hold one value per node")
    reference = (np.sin(m * math.pi * coordinates[:, 0] / _positive("a", a))
                 * np.sin(n * math.pi * coordinates[:, 1] / _positive("b", b)))
    return mac(values, reference)


def path_points(points: list[dict[str, object]]) -> list[tuple[float, float]]:
    """``(total amplitude / t, load factor)`` pairs of a traced path."""
    return [(float(point["total_amplitude_over_t"]), float(point["load_factor"]))
            for point in points]


def interpolate_load_factor(points: list[dict[str, object]], *,
                            total_amplitude_over_t: float) -> float | None:
    """Load factor of the traced path at a requested total amplitude (linear)."""
    ordered = sorted(path_points(points))
    target = float(total_amplitude_over_t)
    for (x0, y0), (x1, y1) in zip(ordered[:-1], ordered[1:]):
        if x0 <= target <= x1 and x1 > x0:
            return y0 + (y1 - y0) * (target - x0) / (x1 - x0)
    return None


def relative_error(measured: float, reference: float) -> float:
    if not math.isfinite(reference) or reference == 0.0:
        raise ValueError("reference must be finite and nonzero")
    return abs(measured / reference - 1.0)
