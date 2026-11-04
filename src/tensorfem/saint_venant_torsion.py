"""Saint-Venant torsion oracles for the merged HEX8 solid path.

TensorFEM carried no torsion benchmark: a prismatic bar under end torque has a
closed-form shear field for circular, elliptical and rectangular sections, and
the merged ``solid3d`` kernel (``hex8_stiffness`` + ``sparse_core``) was never
checked against one.  Torsion is the classic place where a displacement
formulation loses accuracy (warping is outside the bilinear interpolation
space), so this is an independent check rather than a second look at a bending
case.

Conventions (right handed, bar axis ``z``, section in the ``(x, y)`` plane):

* Prandtl stress function ``phi`` with ``tau_xz = dphi/dy``, ``tau_yz = -dphi/dx``,
  ``laplace(phi) = -2 G theta`` inside and ``phi = 0`` on the boundary;
* ``T = integral(x tau_yz - y tau_xz) dA = 2 integral(phi dA) = G theta J``; the
  middle identity follows from ``phi = 0`` on the boundary and is used here as
  a *self-check* of the oracle, never as its definition;
* every field is the exact Saint-Venant solution (the rectangular section is an
  exponentially convergent series; truncation is bound by ``RectSection`` terms
  and re-checked in the test suite).

Literature anchors reproduced by the tests (Timoshenko & Goodier, *Theory of
Elasticity*, 3rd ed., ch. 11): square section of side ``c`` has
``J = 0.1406 c^4`` and ``tau_max = T / (0.208 c^3)``; an ellipse has
``tau_max = 2T / (pi a b^2)`` at the ends of its minor axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .solid3d import elasticity_matrix_3d, hex8_stiffness
from .sparse_core import assemble_coo, solve_sparse_static

_HALF_PI3 = math.pi ** 3


class _Section:
    """A torsion section carrying an exact Prandtl stress field."""

    name = "section"

    def __init__(self, shear_modulus: float = 1.0, twist: float = 1.0):
        if shear_modulus <= 0.0 or twist <= 0.0:
            raise ValueError("shear modulus and twist must be positive")
        self.g = float(shear_modulus)
        self.theta = float(twist)

    # -- exact solution ---------------------------------------------------- #
    @property
    def j(self) -> float:
        raise NotImplementedError

    @property
    def tau_max(self) -> float:
        raise NotImplementedError

    @property
    def area(self) -> float:
        raise NotImplementedError

    @property
    def scale(self) -> float:
        """Characteristic half-dimension, used to size sampling windows."""
        raise NotImplementedError

    def phi(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def tau(self, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    # -- oracle self-checks ------------------------------------------------ #
    def laplacian_residual(self, x: torch.Tensor, y: torch.Tensor, h: float = 1e-5):
        """``laplace(phi) + 2 G theta`` by central differences."""
        one = torch.ones_like(x)
        f = self.phi
        d2 = (f(x + h * one, y) - 2.0 * f(x, y) + f(x - h * one, y)
              + f(x, y + h * one) - 2.0 * f(x, y) + f(x, y - h * one)) / (h * h)
        return d2 + 2.0 * self.g * self.theta

    def torque_from_phi(self, n: int = 401) -> float:
        """``2 * integral(phi dA)`` by midpoint quadrature over the section."""
        grid, weight = self.quadrature(n)
        inside = self.contains(grid[:, 0], grid[:, 1])
        return float(2.0 * (self.phi(grid[:, 0], grid[:, 1]) * weight * inside).sum())

    def torque_from_stress(self, n: int = 401) -> float:
        """``integral(x tau_yz - y tau_xz) dA`` on the same quadrature."""
        grid, weight = self.quadrature(n)
        x, y = grid[:, 0], grid[:, 1]
        inside = self.contains(x, y)
        txz, tyz = self.tau(x, y)
        return float(((x * tyz - y * txz) * weight * inside).sum())

    def contains(self, x, y):
        raise NotImplementedError

    def quadrature(self, n: int):
        raise NotImplementedError

    def cross_section(self, n: int):
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"{type(self).__name__}({self.name}, J={self.j:.6g}, "
                f"tau_max={self.tau_max:.6g})")


class CircleSection(_Section):
    """Solid circular shaft of radius ``a`` (warping-free)."""

    name = "circle"

    def __init__(self, radius: float = 1.0, shear_modulus: float = 1.0, twist: float = 1.0):
        super().__init__(shear_modulus, twist)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        self.a = float(radius)

    @property
    def j(self):
        return math.pi * self.a ** 4 / 2.0

    @property
    def tau_max(self):
        return self.g * self.theta * self.a

    @property
    def area(self):
        return math.pi * self.a ** 2

    @property
    def scale(self):
        return self.a

    def phi(self, x, y):
        return 0.5 * self.g * self.theta * (self.a ** 2 - x * x - y * y)

    def tau(self, x, y):
        one = torch.ones_like(x)
        return -self.g * self.theta * y * one, self.g * self.theta * x * one

    def contains(self, x, y):
        return (x * x + y * y) <= self.a ** 2

    def quadrature(self, n: int = 401):
        k = torch.linspace(-self.a, self.a, n + 1, dtype=torch.float64)[:-1] + self.a / n
        xx, yy = torch.meshgrid(k, k, indexing="ij")
        weight = torch.full((n * n,), (2.0 * self.a / n) ** 2, dtype=torch.float64)
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), 1), weight

    def cross_section(self, n: int, dtype=torch.float64):
        """Square-to-disk mapped grid; the square boundary maps onto the circle."""
        xi = torch.linspace(-1.0, 1.0, n + 1, dtype=dtype)
        xx, yy = torch.meshgrid(xi, xi, indexing="ij")
        px = xx * torch.sqrt(1.0 - yy * yy / 2.0)
        py = yy * torch.sqrt(1.0 - xx * xx / 2.0)
        nodes = torch.stack((self.a * px.reshape(-1), self.a * py.reshape(-1)), 1)
        return nodes, _structured_quads(n)


class EllipseSection(_Section):
    """Solid elliptical shaft, semi-axes ``a`` (x) and ``b`` (y)."""

    name = "ellipse"

    def __init__(self, a: float = 1.0, b: float = 0.5,
                 shear_modulus: float = 1.0, twist: float = 1.0):
        super().__init__(shear_modulus, twist)
        if a <= 0.0 or b <= 0.0:
            raise ValueError("semi-axes must be positive")
        self.a, self.b = float(max(a, b)), float(min(a, b))

    @property
    def _c(self):
        a, b = self.a, self.b
        return self.g * self.theta * a * a * b * b / (a * a + b * b)

    @property
    def j(self):
        a, b = self.a, self.b
        return math.pi * a ** 3 * b ** 3 / (a * a + b * b)

    @property
    def tau_max(self):
        a, b = self.a, self.b
        return 2.0 * self.g * self.theta * a * a * b / (a * a + b * b)

    @property
    def area(self):
        return math.pi * self.a * self.b

    @property
    def scale(self):
        return self.b

    def phi(self, x, y):
        a, b = self.a, self.b
        return self._c * (1.0 - x * x / (a * a) - y * y / (b * b))

    def tau(self, x, y):
        a, b = self.a, self.b
        one = torch.ones_like(x)
        txz = -2.0 * self.g * self.theta * a * a / (a * a + b * b) * y * one
        tyz = 2.0 * self.g * self.theta * b * b / (a * a + b * b) * x * one
        return txz, tyz

    def contains(self, x, y):
        a, b = self.a, self.b
        return (x * x) / (a * a) + (y * y) / (b * b) <= 1.0

    def quadrature(self, n: int = 401):
        kx = torch.linspace(-self.a, self.a, n + 1, dtype=torch.float64)[:-1] + self.a / n
        ky = torch.linspace(-self.b, self.b, n + 1, dtype=torch.float64)[:-1] + self.b / n
        xx, yy = torch.meshgrid(kx, ky, indexing="ij")
        weight = torch.full_like(xx, (2.0 * self.a / n) * (2.0 * self.b / n))
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), 1), weight.reshape(-1)

    def cross_section(self, n: int, dtype=torch.float64):
        xi = torch.linspace(-1.0, 1.0, n + 1, dtype=dtype)
        xx, yy = torch.meshgrid(xi, xi, indexing="ij")
        px = xx * torch.sqrt(1.0 - yy * yy / 2.0)
        py = yy * torch.sqrt(1.0 - xx * xx / 2.0)
        nodes = torch.stack((self.a * px.reshape(-1), self.b * py.reshape(-1)), 1)
        return nodes, _structured_quads(n)


class RectSection(_Section):
    """Solid rectangular shaft with half-sides ``a >= b`` (Timoshenko series)."""

    name = "rectangle"

    def __init__(self, a: float = 1.0, b: float = 0.5, shear_modulus: float = 1.0,
                 twist: float = 1.0, terms: int = 61):
        super().__init__(shear_modulus, twist)
        if a <= 0.0 or b <= 0.0:
            raise ValueError("half-sides must be positive")
        if terms < 3:
            raise ValueError("at least three series terms are required")
        self.a, self.b = float(max(a, b)), float(min(a, b))
        self.terms = int(terms)
        n = torch.arange(1, 2 * self.terms, 2, dtype=torch.float64)
        self._n = n
        self._sign = torch.tensor([1.0 if i % 2 == 0 else -1.0
                                   for i in range(self.terms)], dtype=torch.float64)
        self._lam = n * math.pi / (2.0 * self.b)

    # -- series helpers ---------------------------------------------------- #
    def _cosh_ratio(self, x):
        """``cosh(lam x) / cosh(lam a)`` without overflow."""
        ax = x.abs()
        lam = self._lam.reshape(-1, *([1] * x.dim()))
        return (torch.exp(lam * (ax - self.a))
                * (1.0 + torch.exp(-2.0 * lam * ax))
                / (1.0 + torch.exp(-2.0 * lam * self.a)))

    def _cosh_ratio_slope(self, x):
        """``d/dx cosh(lam x)/cosh(lam a)`` without overflow."""
        ax = x.abs()
        lam = self._lam.reshape(-1, *([1] * x.dim()))
        magnitude = (lam * torch.exp(lam * (ax - self.a))
                     * (1.0 - torch.exp(-2.0 * lam * ax))
                     / (1.0 + torch.exp(-2.0 * lam * self.a)))
        return torch.sign(x) * magnitude

    def _series(self, x):
        lam = self._lam.reshape(-1, *([1] * x.dim()))
        coef = (self._sign / self._n ** 3).reshape(-1, *([1] * x.dim()))
        return coef, lam

    # -- exact solution ---------------------------------------------------- #
    @property
    def j(self):
        a, b = self.a, self.b
        series = float((torch.tanh(self._lam * a) / self._n ** 5).sum())
        return 16.0 * a * b ** 3 / 3.0 - 1024.0 * b ** 4 / math.pi ** 5 * series

    @property
    def tau_max(self):
        """``|tau_xz(0, b)|``: exact maximum, at the midpoint of the long sides."""
        series = float((1.0 / (self._n ** 2 * torch.cosh(self._lam * self.a))).sum())
        return self.g * self.theta * self.b * (2.0 - 16.0 / math.pi ** 2 * series)

    @property
    def area(self):
        return 4.0 * self.a * self.b

    @property
    def scale(self):
        return self.b

    def phi(self, x, y):
        g, t, b = self.g, self.theta, self.b
        coef, lam = self._series(x)
        series = (coef * self._cosh_ratio(x) * torch.cos(lam * y)).sum(0)
        return g * t * (b * b - y * y) - 32.0 * g * t * b * b / _HALF_PI3 * series

    def tau(self, x, y):
        g, t, b = self.g, self.theta, self.b
        coef, lam = self._series(x)
        one = torch.ones_like(x)
        common = 32.0 * g * t * b * b / _HALF_PI3
        txz = (-2.0 * g * t * y * one
               + common * (coef * self._cosh_ratio(x) * lam * torch.sin(lam * y)).sum(0))
        tyz = common * (coef * self._cosh_ratio_slope(x) * torch.cos(lam * y)).sum(0)
        return txz, tyz

    def contains(self, x, y):
        return (x.abs() <= self.a) & (y.abs() <= self.b)

    def quadrature(self, n: int = 401):
        kx = torch.linspace(-self.a, self.a, n + 1, dtype=torch.float64)[:-1] + self.a / n
        ky = torch.linspace(-self.b, self.b, n + 1, dtype=torch.float64)[:-1] + self.b / n
        xx, yy = torch.meshgrid(kx, ky, indexing="ij")
        weight = torch.full_like(xx, (2.0 * self.a / n) * (2.0 * self.b / n))
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), 1), weight.reshape(-1)

    def cross_section(self, n: int, dtype=torch.float64):
        xi = torch.linspace(-self.a, self.a, n + 1, dtype=dtype)
        yi = torch.linspace(-self.b, self.b, n + 1, dtype=dtype)
        xx, yy = torch.meshgrid(xi, yi, indexing="ij")
        nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), 1)
        return nodes, _structured_quads(n)


class AnnulusCircleSection(_Section):
    """Annular shaft, inner radius ``c`` and outer radius ``a``.

    The annular section is *multiply connected*, so the Prandtl stress function
    needs a circulation condition that a single-valued ``phi`` with ``phi = 0``
    on both circles does **not** satisfy; forcing it there would report a
    torsion constant several times too small.  The exact annular solution is
    instead the warping-free rotation field ``u = (-theta z y, theta z x, 0)``,
    which is simultaneously equilibrium, compatible and traction free on both
    cylindrical surfaces, so the stress is ``tau = G theta r`` and
    ``J = pi (a^4 - c^4) / 2``.  ``phi`` is therefore intentionally not provided
    for this section and the Prandtl self-checks cover the simply connected
    sections only.
    """

    name = "annulus"

    def __init__(self, inner: float = 0.5, outer: float = 1.0,
                 shear_modulus: float = 1.0, twist: float = 1.0):
        super().__init__(shear_modulus, twist)
        if not (0.0 < inner < outer):
            raise ValueError("need 0 < inner radius < outer radius")
        self.c, self.a = float(inner), float(outer)

    @property
    def j(self):
        return math.pi * (self.a ** 4 - self.c ** 4) / 2.0

    @property
    def tau_max(self):
        return self.g * self.theta * self.a

    @property
    def area(self):
        return math.pi * (self.a ** 2 - self.c ** 2)

    @property
    def scale(self):
        return self.c

    def phi(self, x, y):
        raise NotImplementedError(
            "the annular section is multiply connected: use the warping-free "
            "tau = G theta r solution instead of a single-valued phi")

    def tau(self, x, y):
        one = torch.ones_like(x)
        return -self.g * self.theta * y * one, self.g * self.theta * x * one

    def contains(self, x, y):
        r2 = x * x + y * y
        return (r2 >= self.c ** 2) & (r2 <= self.a ** 2)

    def quadrature(self, n: int = 401):
        k = torch.linspace(-self.a, self.a, n + 1, dtype=torch.float64)[:-1] + self.a / n
        xx, yy = torch.meshgrid(k, k, indexing="ij")
        weight = torch.full((n * n,), (2.0 * self.a / n) ** 2, dtype=torch.float64)
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), 1), weight

    def cross_section(self, n: int, dtype=torch.float64):
        """Polar grid: ``n`` angular divisions (multiple of 4), n//4 radial rings."""
        if n % 4:
            raise ValueError("annulus requires an angular count divisible by 4")
        rings = max(2, n // 4)
        angles = torch.linspace(0.0, 2.0 * math.pi, n + 1, dtype=dtype)[:-1]
        radii = torch.linspace(self.c, self.a, rings + 1, dtype=dtype)
        xx = radii.reshape(-1, 1) * torch.cos(angles).reshape(1, -1)
        yy = radii.reshape(-1, 1) * torch.sin(angles).reshape(1, -1)
        nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), 1)
        def idx(i, j):
            return j * n + i % n
        cells = [(idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j))
                 for j in range(rings) for i in range(n)]
        return nodes, torch.tensor(cells, dtype=torch.long)


def _structured_quads(n: int):
    """Counter-clockwise quads of an ``n x n`` grid laid out as ``meshgrid(x, y)``."""
    def idx(i, j):
        return i * (n + 1) + j
    cells = [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))
             for j in range(n) for i in range(n)]
    return torch.tensor(cells, dtype=torch.long)


# --------------------------------------------------------------------------- #
# production-path driver: prismatic HEX8 bar under Saint-Venant torsion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TorsionResult:
    section: str
    n_section: int
    layers: int
    nodes: int
    elements: int
    dofs: int
    torque_exact: float
    torque_end_a: float
    torque_end_b: float
    torsional_rigidity_ratio: float      # T_FE / (G theta J)
    shear_l2_error: float                # mid-span band, element centres
    shear_max_error: float               # pointwise max over the same band
    shear_peak_ratio: float              # recovered max shear / exact tau_max
    displacement_error: float            # vs exact u_z-free kinematics (relative L2)
    extra: dict


def prism_mesh(section: _Section, n: int = 8, layers: int = 4, length: float = 4.0,
               dtype=torch.float64):
    """Prismatic HEX8 mesh of ``section`` between ``z = 0`` and ``z = length``."""
    if n < 2 or layers < 1 or length <= 0.0:
        raise ValueError("need n >= 2, layers >= 1 and length > 0")
    xy, quads = section.cross_section(n)
    xy = xy.to(dtype)
    m = len(xy)
    zs = torch.linspace(0.0, length, layers + 1, dtype=dtype)
    nodes = torch.cat([torch.stack((xy[:, 0], xy[:, 1], z.expand(m)), 1) for z in zs], 0)
    offs = (torch.arange(layers, dtype=torch.long) * m)[:, None]
    hexes = torch.cat([quads + o for o in offs], 0)
    hexes = torch.cat([hexes, hexes + m], 1)
    return nodes, hexes, m, len(zs)


def torsion_run(section: _Section, n: int = 8, layers: int = 4, length: float = 4.0,
                young_modulus: float = 2.0 * (1.0 + 0.3), poisson_ratio: float = 0.3,
                sample_band: float = 0.3, rtol: float = 1e-11) -> TorsionResult:
    """Solve the merged HEX8 path and compare with the exact Saint-Venant field.

    Kinematics: the end faces carry the exact transverse rotation
    ``u_x = -theta z y``, ``u_y = theta z x`` (``z = 0`` gives zero, so the rigid
    body motion is removed); ``u_z`` (warping) is left free everywhere, which is
    what makes the comparison a test of the element instead of a test of a
    prescribed field.  The lateral surface is traction free.
    """
    if young_modulus <= 0.0 or not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("invalid elastic constants")
    shear = young_modulus / (2.0 * (1.0 + poisson_ratio))
    if abs(shear - section.g) > 1e-12 * max(1.0, section.g):
        raise ValueError("section shear modulus must match E/(2(1+nu))")
    nodes, hexes, m, _ = prism_mesh(section, n=n, layers=layers, length=length)
    ne = len(hexes)
    ndof = 3 * len(nodes)
    E = nodes.new_full((ne,), young_modulus)
    nu = nodes.new_full((ne,), poisson_ratio)
    D = elasticity_matrix_3d(E, nu)
    ke, B = hex8_stiffness(nodes[hexes], D)
    edofs = (3 * hexes[:, :, None] + torch.arange(3)).reshape(ne, 24)
    K = assemble_coo(ke, edofs, ndof)
    theta = section.theta
    dirichlet = {}
    end_slices = [torch.arange(m), torch.arange(ndof // 3 - m, ndof // 3)]
    for k, sl in enumerate(end_slices):
        z = 0.0 if k == 0 else length
        xy = nodes[sl]
        ux, uy = -theta * z * xy[:, 1], theta * z * xy[:, 0]
        for pos, node in enumerate(sl.tolist()):
            dirichlet[3 * node] = float(ux[pos])
            dirichlet[3 * node + 1] = float(uy[pos])
    force = nodes.new_zeros(ndof)
    solved = solve_sparse_static(K, force, dirichlet=dirichlet, rtol=rtol)
    u, reaction = solved.displacement, solved.reaction

    centres = nodes[hexes].mean(1)
    strain = torch.einsum("eij,ej->ei", B, u.flatten()[edofs])
    stress = torch.einsum("eij,ej->ei", D, strain)
    txz, tyz = stress[:, 5], stress[:, 4]   # columns: xy, yz, xz
    exz, eyz = section.tau(centres[:, 0], centres[:, 1])
    band = (centres[:, 2] >= 0.5 * length - sample_band * length) & \
           (centres[:, 2] <= 0.5 * length + sample_band * length)
    num = ((txz[band] - exz[band]) ** 2 + (tyz[band] - eyz[band]) ** 2).sum()
    den = (exz[band] ** 2 + eyz[band] ** 2).sum()
    shear_l2 = float(torch.sqrt(num / den))
    ref_peak = float(torch.sqrt(exz[band] ** 2 + eyz[band] ** 2).max())
    got_peak = float(torch.sqrt(txz[band] ** 2 + tyz[band] ** 2).max())
    pointwise = (torch.sqrt((txz[band] - exz[band]) ** 2 + (tyz[band] - eyz[band]) ** 2)
                 / torch.sqrt(exz[band] ** 2 + eyz[band] ** 2).clamp_min(1e-300))

    def end_torque(sl):
        return float((nodes[sl, 0] * reaction[3 * sl + 1] - nodes[sl, 1] * reaction[3 * sl]).sum())

    torque_exact = section.g * theta * section.j
    reference = torch.zeros_like(u)
    reference[0::3] = -theta * nodes[:, 2] * nodes[:, 1]
    reference[1::3] = theta * nodes[:, 2] * nodes[:, 0]
    transverse = torch.zeros_like(u, dtype=torch.bool)
    transverse[0::3] = True
    transverse[1::3] = True
    disp_err = float(torch.linalg.vector_norm(u[transverse] - reference[transverse])
                     / torch.linalg.vector_norm(reference[transverse]))
    return TorsionResult(
        section=section.name, n_section=n, layers=layers, nodes=len(nodes),
        elements=ne, dofs=ndof, torque_exact=torque_exact,
        torque_end_a=end_torque(end_slices[0]), torque_end_b=end_torque(end_slices[1]),
        torsional_rigidity_ratio=(abs(end_torque(end_slices[1]))
                                  + abs(end_torque(end_slices[0]))) / 2.0 / torque_exact,
        shear_l2_error=shear_l2, shear_max_error=float(pointwise.max()),
        shear_peak_ratio=got_peak / ref_peak, displacement_error=disp_err,
        extra={"sample_points": int(band.sum()),
               "warping_max": float(u[2::3].abs().max())},
    )


def default_sections(shear_modulus: float = 1.0, twist: float = 1.0):
    """The sections qualified by the audit script and the test suite."""
    return [RectSection(1.0, 1.0, shear_modulus, twist),
            RectSection(1.0, 0.5, shear_modulus, twist),
            AnnulusCircleSection(0.5, 1.0, shear_modulus, twist)]