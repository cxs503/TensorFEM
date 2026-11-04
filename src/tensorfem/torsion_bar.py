"""Saint-Venant torsion benchmark: closed-form series reference and 3-D FE driver.

Why this module exists
----------------------
Torsion was the one elementary load case that none of the repository's 3-D
solid paths had ever been checked against: every stored ``HEX8``/``HEX20``/
``TET4``/``TET10`` benchmark used axial, bending or pressure loading.  Torsion
is the cheapest discriminating case that a bending benchmark cannot cover,
because the exact solution is *not* a linear displacement field (it warps) and
the shear stress is carried by a harmonic Prandtl stress function with a
maximum on the boundary, i.e. exactly where an isoparametric element is
weakest.  A pure bending check therefore says nothing about it.

Reference side (``torsion_constant`` and friends)
-------------------------------------------------
Saint-Venant's solution for a rectangle ``a`` (long side, ``y``) by ``b``
(short side, ``x``) is written with the normalised Prandtl stress function
``psi`` (``lap psi = -2``, ``psi = 0`` on the boundary, twist rate ``theta``):

    psi = b^2/4 - x^2 - (8 b^2/pi^3) sum_{n odd} (-1)^((n-1)/2)/n^3
                        cos(n pi x / b) cosh(n pi y / b)/cosh(n pi a/(2 b))

    tau_zx = G theta dpsi/dy,      tau_zy = -G theta dpsi/dx
    J      = 2 int psi dA = a b^3/3 - (64 b^4/pi^5) sum_{n odd} tanh(n pi a/(2 b))/n^5
    w      = x y - (8 b^2/pi^3) sum_{n odd} (-1)^((n-1)/2)/n^3
                     sin(n pi x/b) sinh(n pi y/b)/cosh(n pi a/(2 b))

with ``u = theta (-y, x, w)``.  The cosh products are evaluated through
``_cosh_ratio`` so that the series is usable for slender sections where
``cosh(n pi a/(2 b))`` overflows float64 for large ``n``.

Finite-element side (``solve_torsion_bar``)
-------------------------------------------
A ``side x side x length`` prism is meshed either with ``HEX8`` (2x2x2 Gauss,
fully integrated) or ``HEX20`` (3x3x3 Gauss) elements, ``segments`` elements
across the side and ``4*segments`` along the axis.  A pure torque ``T`` is
applied on the far end face as a circumferential linear traction
``t = c (-y, x)`` integrated with the consistent face weights of the element
(Q4: ``A/4`` per corner; Q8 serendipity: ``A/3`` per midside node and
``-A/12`` per corner, which reproduces linear tractions exactly).  ``c`` is
rescaled so the assembled nodal forces carry exactly ``T``.  The near end face
has its two in-plane translations suppressed and one axial dof fixed, so the
section can warp freely.  The twist rate is a least-squares slope of the
per-section rotation over the interior window ``0.25 L <= z <= 0.75 L``, which
removes the two Saint-Venant boundary layers.

The measured twist rate is compared with ``T/(G J)``, the mid-plane shear
stress with the closed form at the element centroids, and the mid-plane axial
displacement (divided by the measured twist rate) with ``w`` after removing the
arbitrary axial datum.  Nothing here compares against a stored snapshot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .hex20 import hex20_stiffness, structured_hex20_mesh
from .solid3d import elasticity_matrix_3d, hex8_stiffness, structured_hex_mesh
from .sparse_core import assemble_coo, solve_sparse_static

__all__ = [
    "TorsionBarResult",
    "boundary_shear_peak",
    "max_shear_stress",
    "prandtl_stress_function",
    "rectangular_torsion_constant",
    "solve_torsion_bar",
    "torsion_shear_stress",
    "torsion_warping_function",
]

_TERMS = 200


def _odd_modes(terms: int, *, dtype=torch.float64, device=None) -> torch.Tensor:
    if terms < 1:
        raise ValueError("terms must be a positive integer")
    return torch.arange(1, 2 * terms, 2, dtype=dtype, device=device)


def _cosh_ratio(u: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """``cosh(u)/cosh(upper)`` evaluated without overflow (``|u| <= upper``)."""
    au = u.abs()
    return torch.exp(au - upper) * (1.0 + torch.exp(-2.0 * au)) / (1.0 + torch.exp(-2.0 * upper))


def _series_arguments(a: float, b: float, terms: int, *, dtype=torch.float64):
    if not (a > 0 and b > 0):
        raise ValueError("section sides must be positive")
    if b > a * (1.0 + 1e-12):
        raise ValueError("expects a >= b (long side first)")
    n = _odd_modes(terms, dtype=dtype)
    sign = (-1.0) ** ((n - 1) / 2)
    upper = n * math.pi * a / (2.0 * b)
    return n, sign, upper


def rectangular_torsion_constant(a: float, b: float, *, terms: int = _TERMS) -> float:
    """Saint-Venant torsion constant ``J`` of an ``a x b`` rectangle (``a >= b``)."""
    n, _, upper = _series_arguments(a, b, terms)
    return a * b**3 / 3.0 - (64.0 * b**4 / math.pi**5) * float(torch.sum(torch.tanh(upper) / n**5))


def prandtl_stress_function(x, y, a: float, b: float, *, terms: int = _TERMS):
    """Normalised Prandtl stress function ``psi`` (``J = 2 int psi dA``)."""
    n, sign, upper = _series_arguments(a, b, terms, dtype=_dtype_of(x, y))
    X, Y = _column(x), _column(y)
    series = torch.sum(
        (sign / n**3)[None, :]
        * torch.cos(n[None, :] * math.pi * X / b)
        * _cosh_ratio(n[None, :] * math.pi * Y / b, upper[None, :]),
        dim=1,
    )
    return b**2 / 4.0 - X[:, 0] ** 2 - (8.0 * b**2 / math.pi**3) * series


def torsion_shear_stress(x, y, a: float, b: float, *, twist_rate: float = 1.0,
                         shear_modulus: float = 1.0, terms: int = _TERMS):
    """Closed-form ``(tau_zx, tau_zy)`` for twist rate ``theta`` and modulus ``G``."""
    n, sign, upper = _series_arguments(a, b, terms, dtype=_dtype_of(x, y))
    X, Y = _column(x), _column(y)
    cosx = torch.cos(n[None, :] * math.pi * X / b)
    sinx = torch.sin(n[None, :] * math.pi * X / b)
    ratio = _cosh_ratio(n[None, :] * math.pi * Y / b, upper[None, :])
    dpsidx = -2.0 * X[:, 0] + (8.0 * b / math.pi**2) * torch.sum((sign / n**2)[None, :] * sinx * ratio, dim=1)
    dpsidy = -(8.0 * b / math.pi**2) * torch.sum(
        (sign / n**2)[None, :] * cosx * ratio * torch.tanh(n[None, :] * math.pi * Y / b), dim=1
    )
    g_theta = shear_modulus * twist_rate
    return g_theta * dpsidy, -g_theta * dpsidx


def torsion_warping_function(x, y, a: float, b: float, *, terms: int = _TERMS):
    """Warping shape ``w`` in ``u_z = theta * w`` (defined up to a constant)."""
    n, sign, upper = _series_arguments(a, b, terms, dtype=_dtype_of(x, y))
    X, Y = _column(x), _column(y)
    ratio = _cosh_ratio(n[None, :] * math.pi * Y / b, upper[None, :])
    series = torch.sum(
        (sign / n**3)[None, :]
        * torch.sin(n[None, :] * math.pi * X / b)
        * ratio
        * torch.tanh(n[None, :] * math.pi * Y / b),
        dim=1,
    )
    return X[:, 0] * Y[:, 0] - (8.0 * b**2 / math.pi**3) * series


def max_shear_stress(a: float, b: float, *, torque: float = 1.0, samples: int = 401,
                     terms: int = _TERMS):
    """Peak ``|tau|`` and its location, scanned on an interior ``samples^2`` grid.

    The exact maximum sits on the boundary at the midpoint of a long side, so
    the grid is shifted off the boundary by half a cell; the reported value is
    therefore a lower bound on the true peak (grid refinement is monotone).
    """
    if samples < 5:
        raise ValueError("samples must be >= 5")
    j = rectangular_torsion_constant(a, b, terms=terms)
    # G*theta = tau_scale; shear modulus cancels in the scan
    step_x = b / samples
    step_y = a / samples
    xs = torch.linspace(-b / 2 + step_x / 2, b / 2 - step_x / 2, samples, dtype=torch.float64)
    ys = torch.linspace(-a / 2 + step_y / 2, a / 2 - step_y / 2, samples, dtype=torch.float64)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    tx, ty = torsion_shear_stress(X.reshape(-1), Y.reshape(-1), a, b, terms=terms)
    magnitude = torch.sqrt(tx**2 + ty**2) * (torque / j)
    index = int(torch.argmax(magnitude))
    return float(magnitude[index]), (float(X.reshape(-1)[index]), float(Y.reshape(-1)[index]))


def boundary_shear_peak(a: float, b: float, *, shear_modulus: float = 1.0,
                        twist_rate: float = 1.0, terms: int = _TERMS):
    """Exact peak ``|tau|`` *on* the lateral surface of an ``a x b`` rectangle.

    The maximum of ``|grad psi|`` of the torsional problem is attained on the
    boundary, so it cannot be approached by any interior sampling: an interior
    grid scan (``max_shear_stress``) returns a monotone lower bound, and an
    element centroid returns a value that is wrong by tens of percent.  On the
    long-side midpoint ``y = +-a/2, x = 0`` the series differentiates in closed
    form (``tau_zx = G theta dpsi/dy``, ``tau_zy = -G theta dpsi/dx = 0`` there,
    which is the traction-free condition of that face)::

        tau_max = G theta (8 b / pi^2)
                  sum_{n odd} (-1)^((n-1)/2) / n^2 * tanh(n pi a / (2 b))

    For a square section this returns ``0.6753 G theta a``, i.e. the Timoshenko
    coefficient ``alpha = 0.208`` of ``tau_max = T/(alpha a b^2)`` to the three
    digits of the table.  The series converges like ``1/n^2`` only while
    ``tanh(n pi a / (2 b))`` saturates quickly, so it is the peak of the *long
    side*; for a strip (``a >> b``) the coefficient approaches
    ``(8/pi^2) * Catalan = 0.7424``, which is the slowly converging thin-strip
    limit and is reported as such rather than as a converged value.
    """
    if not (a > 0 and b > 0):
        raise ValueError("section sides must be positive")
    if b > a * (1.0 + 1e-12):
        raise ValueError("expects a >= b (long side first)")
    n, sign, upper = _series_arguments(a, b, terms)
    coefficient = (8.0 * b / math.pi**2) * float(torch.sum(sign / n**2 * torch.tanh(upper)))
    return {"shear_modulus": shear_modulus, "twist_rate": twist_rate,
            "coefficient": coefficient, "tau_max": shear_modulus * twist_rate * coefficient,
            "location": (0.0, a / 2.0)}


def _dtype_of(*values):
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.dtype
    return torch.float64


def _column(value) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    return tensor[:, None]


@dataclass(frozen=True)
class TorsionBarResult:
    """Measured versus exact Saint-Venant torsion quantities for one mesh."""

    element: str
    segments: int
    n_elements: int
    n_nodes: int
    n_dofs: int
    applied_torque: float
    twist_rate: float
    exact_twist_rate: float
    measured_torsion_constant: float
    exact_torsion_constant: float
    profile_z: torch.Tensor
    profile_theta: torch.Tensor
    centroid_xy: torch.Tensor
    centroid_shear_fe: torch.Tensor
    centroid_shear_exact: torch.Tensor
    shear_l2_relative_error: float
    shear_peak_fe: float
    shear_peak_exact_sampled: float
    shear_peak_exact_global: float
    warping_xy: torch.Tensor
    warping_fe: torch.Tensor
    warping_exact: torch.Tensor
    warping_relative_error: float
    nodes: torch.Tensor | None = None
    cells: torch.Tensor | None = None
    displacement: torch.Tensor | None = None
    modulus: torch.Tensor | None = None

    @property
    def twist_rate_relative_error(self) -> float:
        return abs(self.twist_rate - self.exact_twist_rate) / abs(self.exact_twist_rate)

    @property
    def shear_peak_relative_error(self) -> float:
        """Error of the peak centroid stress against the exact field at the same points.

        The true peak of ``|tau|`` sits on the lateral surface, which a centroid
        sample can never reach; ``shear_peak_exact_global`` carries that value
        and is reported separately instead of being compared with centroids.
        """
        if abs(self.shear_peak_exact_sampled) < 1e-9:
            return math.nan  # a single element per side has its only centroid at tau = 0
        return abs(self.shear_peak_fe - self.shear_peak_exact_sampled) / abs(self.shear_peak_exact_sampled)


def _face_weights(nodes: torch.Tensor, cells: torch.Tensor, *, level: float, side: float,
                  segments: int, quadratic: bool) -> torch.Tensor:
    """Consistent nodal weights (integral of the face shape functions) at ``z == level``.

    For a bilinear quad the weights are ``A/4`` on every corner; for the
    8-node serendipity face they are ``A/3`` on the midside nodes and ``-A/12``
    on the corners.  Both reproduce the integral of any linear function
    exactly, so a linear traction is integrated without quadrature error.
    """
    weights = nodes.new_zeros(len(nodes))
    area = (side / segments) ** 2
    corners = (4, 5, 6, 7)
    mids = (12, 13, 14, 15) if quadratic else ()
    for cell in cells.tolist():
        if abs(float(nodes[cell[4], 2]) - level) > 1e-12:
            continue
        for node in corners:
            weights[cell[node]] += -area / 12.0 if quadratic else area / 4.0
        for node in mids:
            weights[cell[node]] += area / 3.0
    return weights


def solve_torsion_bar(*, element: str = "hex8", segments: int = 4, side: float = 1.0,
                      length: float = 4.0, young: float = 1.0e9, poisson: float = 0.3,
                      torque: float = 1.0, window: tuple[float, float] = (0.25, 0.75),
                      terms: int = _TERMS, rtol: float = 1e-12) -> TorsionBarResult:
    """Twist a ``side x side x length`` prism with a pure end torque and measure it."""
    if segments < 1:
        raise ValueError("segments must be >= 1")
    if element not in ("hex8", "hex20"):
        raise ValueError("element must be 'hex8' or 'hex20'")
    if not 0.0 <= window[0] < window[1] <= 1.0:
        raise ValueError("window must be an increasing sub-interval of [0, 1]")
    quadratic = element == "hex20"
    nz = 4 * segments
    if quadratic:
        nodes, cells = structured_hex20_mesh(side, side, length, segments, segments, nz)
        stiffness = hex20_stiffness
        per_node = 20
    else:
        nodes, cells = structured_hex_mesh(side, side, length, segments, segments, nz)
        stiffness = hex8_stiffness
        per_node = 8
    n_elements = len(cells)
    modulus = elasticity_matrix_3d(nodes.new_tensor(young).expand(n_elements),
                                   nodes.new_tensor(poisson).expand(n_elements))
    ke, b_matrix = stiffness(nodes[cells.long()], modulus)
    element_dofs = torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(n_elements, 3 * per_node).long()
    n_dofs = 3 * len(nodes)
    K = assemble_coo(ke, element_dofs, n_dofs, csr=True)

    xc = nodes[:, 0] - side / 2.0
    yc = nodes[:, 1] - side / 2.0
    loading = torch.where(nodes[:, 2] == length)[0]
    weights = _face_weights(nodes, cells, level=length, side=side, segments=segments, quadratic=quadratic)
    forces = nodes.new_zeros(n_dofs)
    forces[3 * loading] = weights[loading] * (-yc[loading])
    forces[3 * loading + 1] = weights[loading] * xc[loading]
    polar = float(torch.sum(weights[loading] * (xc[loading] ** 2 + yc[loading] ** 2)))
    forces = forces * (torque / polar)
    applied = float(torch.sum(xc[loading] * forces[3 * loading + 1] - yc[loading] * forces[3 * loading]))
    if abs(applied - torque) > 1e-9 * abs(torque):
        raise RuntimeError("end-face traction does not carry the requested torque")

    root = torch.where(nodes[:, 2] == 0)[0]
    fixed = {int(dof): 0.0 for dof in torch.cat((3 * root, 3 * root + 1)).tolist()}
    fixed[3 * int(root[0]) + 2] = 0.0
    solution = solve_sparse_static(K, forces, dirichlet=fixed, rtol=rtol)
    displacement = solution.displacement

    radius2 = xc**2 + yc**2
    planes = torch.unique(nodes[:, 2])
    thetas = nodes.new_zeros(len(planes))
    for index, z in enumerate(planes.tolist()):
        plane = torch.where(nodes[:, 2] == z)[0]
        thetas[index] = torch.sum(displacement[3 * plane] * (-yc[plane]) + displacement[3 * plane + 1] * xc[plane]) / torch.sum(radius2[plane])
    inside = (planes >= window[0] * length) & (planes <= window[1] * length)
    design = torch.stack((planes[inside], torch.ones(int(inside.sum()), dtype=nodes.dtype)), 1)
    twist_rate = float(torch.linalg.lstsq(design, thetas[inside]).solution[0])

    j_exact = rectangular_torsion_constant(side, side, terms=terms)
    exact_rate = torque / (young / (2.0 * (1.0 + poisson)) * j_exact)

    strain = torch.einsum("eij,ej->ei", b_matrix, displacement[element_dofs])
    stress = torch.einsum("eij,ej->ei", modulus, strain)
    centroids = nodes[cells.long()].mean(1)
    hz = length / nz
    slab = (centroids[:, 2] - length / 2.0).abs() < hz
    cx = centroids[slab, 0] - side / 2.0
    cy = centroids[slab, 1] - side / 2.0
    tx, ty = torsion_shear_stress(cx, cy, side, side, twist_rate=twist_rate,
                                  shear_modulus=young / (2.0 * (1.0 + poisson)), terms=terms)
    fe = torch.stack((stress[slab, 5], stress[slab, 4]), 1)   # tau_zx, tau_zy
    exact = torch.stack((tx, ty), 1)
    denominator = float(torch.sqrt(torch.sum(exact**2)))
    l2 = float(torch.sqrt(torch.sum((fe - exact) ** 2))) / denominator if denominator > 1e-9 else math.nan

    peak_fe = float(torch.sqrt(fe[:, 0] ** 2 + fe[:, 1] ** 2).max())
    peak_sampled = float(torch.sqrt(exact[:, 0] ** 2 + exact[:, 1] ** 2).max())
    peak_global, _ = max_shear_stress(side, side, torque=torque, terms=terms)

    mid = torch.where((nodes[:, 2] - length / 2.0).abs() < 1e-12)[0]
    warping_xy = torch.stack((xc[mid], yc[mid]), 1)
    warping_fe = displacement[3 * mid + 2] / twist_rate
    warping_ref = torsion_warping_function(xc[mid], yc[mid], side, side, terms=terms)
    fe_centred = warping_fe - warping_fe.mean()  # axial datum is arbitrary
    ref_centred = warping_ref - warping_ref.mean()
    warping_error = float(torch.sqrt(torch.sum((fe_centred - ref_centred) ** 2)) / torch.sqrt(torch.sum(ref_centred**2)))

    return TorsionBarResult(
        element=element, segments=segments, n_elements=n_elements, n_nodes=len(nodes),
        n_dofs=n_dofs, applied_torque=applied, twist_rate=twist_rate, exact_twist_rate=exact_rate,
        measured_torsion_constant=torque / (young / (2.0 * (1.0 + poisson)) * twist_rate),
        exact_torsion_constant=j_exact, profile_z=planes, profile_theta=thetas,
        centroid_xy=torch.stack((cx, cy), 1), centroid_shear_fe=fe, centroid_shear_exact=exact,
        shear_l2_relative_error=l2, shear_peak_fe=peak_fe,
        shear_peak_exact_sampled=peak_sampled, shear_peak_exact_global=peak_global,
        warping_xy=warping_xy, warping_fe=fe_centred + ref_centred.mean(), warping_exact=warping_ref,
        warping_relative_error=warping_error,
        nodes=nodes, cells=cells, displacement=displacement, modulus=modulus,
    )