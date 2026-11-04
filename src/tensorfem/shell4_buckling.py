"""Initial-stress buckling operators for flat Shell4 meshes.

The geometric matrix is integrated from the bilinear Q4 shape gradients and
acts on the transverse displacement of the existing six-DOF ``shell4``
element.  Stress resultants are positive in compression.  This is a linear
eigenbuckling operator: it does not update stress resultants after buckling.

Two stress descriptions are supported.  The original API takes a single
constant resultant triple ``(Nx, Ny, Nxy)`` per element.  The field API takes
four nodal resultants of shape ``(4, 3)`` that are interpolated bilinearly, so
that a non-uniform pre-buckling membrane state (for example the linear
distribution produced by an in-plane bending or patch edge load) can be
integrated consistently inside every element.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable, Sequence
import torch

from .shell4 import shell4_local_frame, shell4_stiffness

# Local 24-DOF indices of the three bending DOFs (w, theta_x, theta_y) of each
# node, used to assemble a 12-DOF element matrix in bending space only.
_BENDING_LOCAL = (2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22)
_GAUSS = 1.0 / math.sqrt(3.0)
_NATURAL = ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS))
_NODE_NATURAL = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


def _shape_values(gi: int) -> torch.Tensor:
    """Bilinear shape functions of all four nodes at Gauss point ``gi``."""
    xi, eta = _NATURAL[gi]
    return torch.tensor([0.25 * (1 + xi * sx) * (1 + eta * sy) for sx, sy in _NODE_NATURAL],
                        dtype=torch.float64)


@dataclass(frozen=True)
class ShellBucklingResult:
    load_factors: torch.Tensor
    modes: torch.Tensor


def _finite_positive(name: str, value) -> float:
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return x


def _natural_gradients(xi: float, eta: float, dtype, device) -> torch.Tensor:
    return torch.tensor([
        [-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
        [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)],
    ], dtype=dtype, device=device) * 0.25


def shell4_geometric_stiffness(xyz: torch.Tensor, stress_resultants: torch.Tensor) -> torch.Tensor:
    """Return the 24x24 initial-stress matrix for compressional resultants.

    ``stress_resultants`` is either a constant ``(3,)`` triple ``[Nx, Ny, Nxy]``
    or a ``(4, 3)`` array of nodal resultants interpolated with the bilinear
    Q4 shape functions.  ``Nx`` and ``Ny`` are positive compressive force
    resultants per unit length.  The returned positive-semidefinite matrix is
    used as ``K - lambda*Kg``.
    """
    if not isinstance(xyz, torch.Tensor) or not xyz.dtype.is_floating_point:
        raise TypeError("xyz must be a floating-point tensor")
    xy, basis = shell4_local_frame(xyz)
    N = torch.as_tensor(stress_resultants, dtype=xyz.dtype, device=xyz.device)
    if N.shape not in ((3,), (4, 3)) or not bool(torch.all(torch.isfinite(N))):
        raise ValueError("stress_resultants must be a finite (3,) or (4, 3) tensor")
    nodal = (N.reshape(1, 3).expand(4, 3) if N.shape == (3,) else N)
    kg4 = xyz.new_zeros((4, 4))
    for k, (xi, eta) in enumerate(_NATURAL):
        dnat = _natural_gradients(xi, eta, xyz.dtype, xyz.device)
        J = xy.T @ dnat
        detJ = torch.linalg.det(J)
        if bool((detJ <= torch.finfo(xyz.dtype).eps).item()):
            raise ValueError("inverted or degenerate shell element")
        grad = dnat @ torch.linalg.inv(J)
        shape = _shape_values(k).to(dtype=xyz.dtype, device=xyz.device)
        Nloc = shape @ nodal
        Smat = torch.stack((torch.stack((Nloc[0], Nloc[2])), torch.stack((Nloc[2], Nloc[1]))))
        kg4 += grad @ Smat @ grad.T * detJ
    local = xyz.new_zeros((24, 24))
    w = 6 * torch.arange(4, device=xyz.device) + 2
    local[w[:, None], w] = kg4
    transform = xyz.new_zeros((24, 24))
    for node in range(4):
        transform[6*node:6*node+3, 6*node:6*node+3] = basis
        transform[6*node+3:6*node+6, 6*node+3:6*node+6] = basis
    return transform.T @ local @ transform


def rectangular_shell_mesh(a: float, b: float, nx: int, ny: int, *, dtype=torch.float64):
    """Structured flat Shell4 mesh, with counter-clockwise connectivity."""
    a = _finite_positive("a", a); b = _finite_positive("b", b)
    if isinstance(nx, bool) or isinstance(ny, bool) or nx < 2 or ny < 2:
        raise ValueError("nx and ny must be integers >= 2")
    if int(nx) != nx or int(ny) != ny:
        raise ValueError("nx and ny must be integers >= 2")
    xs = torch.linspace(0., a, int(nx)+1, dtype=dtype)
    ys = torch.linspace(0., b, int(ny)+1, dtype=dtype)
    nodes = torch.tensor([(x, y, 0.) for y in ys for x in xs], dtype=dtype)
    conns = []
    for j in range(int(ny)):
        for i in range(int(nx)):
            n = j*(int(nx)+1)+i
            conns.append((n, n+1, n+int(nx)+2, n+int(nx)+1))
    return nodes, torch.tensor(conns, dtype=torch.long)


def linearly_varying_uniaxial_field(psi: float) -> Callable[[float, float], Sequence[float]]:
    """Uniaxial pre-stress ``Nx(y)/N0 = 1 + (psi - 1) y / b``.

    ``psi = 1`` is uniform compression, ``psi = 0`` compresses one edge only,
    ``psi = -1`` is pure in-plane bending.  ``N0`` is the compressive resultant
    at the ``y = 0`` edge, so it is the eigenvalue reported by the solvers.
    """
    psi = float(psi)
    if not math.isfinite(psi):
        raise ValueError("psi must be finite")

    def field(_x: float, y: float) -> Sequence[float]:
        return (1.0 + (psi - 1.0) * float(y), 0.0, 0.0)

    return field


def _solve_bending_eigen(K, G, nodes, a, b, modes):
    """Simply-supported transverse BC, static condensation of rotations."""
    tol = 64*torch.finfo(nodes.dtype).eps
    edge = ((nodes[:, 0].abs() < tol) | ((nodes[:, 0]-a).abs() < tol) |
            (nodes[:, 1].abs() < tol) | ((nodes[:, 1]-b).abs() < tol))
    fixed_w = set((3*torch.nonzero(edge).flatten()).tolist())
    free = torch.tensor([int(i) for i in range(3*len(nodes)) if int(i) not in fixed_w],
                        dtype=torch.long)
    Kr = K[free[:, None], free]; Gr = G[free[:, None], free]
    active = torch.linalg.vector_norm(Gr, dim=0) > 1e-13
    ia = torch.nonzero(active).flatten(); ib = torch.nonzero(~active).flatten()
    recovery = None
    if len(ib):
        recovery = -torch.linalg.solve(Kr[ib[:, None], ib], Kr[ib[:, None], ia])
        Kc = Kr[ia[:, None], ia] + Kr[ia[:, None], ib] @ recovery
    else:
        Kc = Kr[ia[:, None], ia]
    Gc = Gr[ia[:, None], ia]
    # Whiten with the elastic matrix: it stays positive definite even when the
    # pre-stress field changes sign (pure in-plane bending), where the
    # initial-stress matrix is indefinite and cannot be Cholesky factored.
    L = torch.linalg.cholesky(Kc)
    A = torch.linalg.solve_triangular(L, Gc, upper=False)
    A = torch.linalg.solve_triangular(L, A.T, upper=False).T
    mu, vectors = torch.linalg.eigh((A+A.T)/2)
    keep_pos = mu > 1e-12 * float(mu.abs().max())
    mu = mu[keep_pos]; vectors = vectors[:, keep_pos]
    order = torch.argsort(mu, descending=True)[:int(modes)]
    mu = mu[order]; vectors = vectors[:, order]
    if len(mu) < int(modes):
        raise RuntimeError("insufficient positive buckling modes")
    values = 1.0 / mu
    full = nodes.new_zeros((3*len(nodes), len(values)))
    active_modes = torch.linalg.solve_triangular(L.T, vectors, upper=True)
    full[free[ia], :] = active_modes
    if recovery is not None:
        full[free[ib], :] = recovery @ active_modes
    return ShellBucklingResult(values, full)


def simply_supported_plate_buckling(*, a: float, b: float, nx: int, ny: int,
                                    young: float, poisson: float, thickness: float,
                                    ny_ratio: float = 0., modes: int = 1) -> ShellBucklingResult:
    """Solve Shell4 plate buckling for unit ``Nx`` and ``Ny=ny_ratio*Nx``."""
    ratio = float(ny_ratio)
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("ny_ratio must be finite and nonnegative")
    return simply_supported_plate_buckling_field(
        a=a, b=b, nx=nx, ny=ny, young=young, poisson=poisson, thickness=thickness,
        stress_field=lambda _x, _y: (1.0, ratio, 0.0), modes=modes)


def simply_supported_plate_buckling_field(*, a: float, b: float, nx: int, ny: int,
                                          young: float, poisson: float, thickness: float,
                                          stress_field: Callable[[float, float], Sequence[float]],
                                          modes: int = 1) -> ShellBucklingResult:
    """Linear eigenbuckling with a non-uniform pre-stress resultant field.

    ``stress_field(x, y)`` returns ``[Nx, Ny, Nxy]`` per node, positive for
    compression.  The field is interpolated bilinearly inside each element, so
    a linear membrane pre-state obtained from an in-plane bending or patch edge
    load is represented without any piecewise-constant loss.
    """
    E = _finite_positive("young", young); t = _finite_positive("thickness", thickness)
    a = _finite_positive("a", a); b = _finite_positive("b", b)
    nu = float(poisson)
    if not math.isfinite(nu) or not -1 < nu < .5:
        raise ValueError("poisson must be in (-1, 0.5)")
    if isinstance(modes, bool) or modes < 1 or int(modes) != modes:
        raise ValueError("modes must be a positive integer")
    if not callable(stress_field):
        raise TypeError("stress_field must be callable")
    nodes, elements = rectangular_shell_mesh(a, b, nx, ny)
    nodal = torch.tensor([tuple(float(v) for v in stress_field(float(p[0]), float(p[1])))
                          for p in nodes], dtype=nodes.dtype)
    if nodal.shape != (len(nodes), 3) or not bool(torch.all(torch.isfinite(nodal))):
        raise ValueError("stress_field must return finite [Nx, Ny, Nxy] triples")

    K = nodes.new_zeros((3*len(nodes), 3*len(nodes))); G = K.clone()
    bend = torch.tensor(_BENDING_LOCAL, dtype=torch.long)
    for conn in elements:
        xyz = nodes[conn]
        ke = shell4_stiffness(xyz, E, nu, t)[bend[:, None], bend]
        ge = shell4_geometric_stiffness(xyz, nodal[conn])[bend[:, None], bend]
        ids = (3*conn[:, None] + torch.arange(3, dtype=torch.long)).reshape(-1)
        K[ids[:, None], ids] += ke; G[ids[:, None], ids] += ge
    return _solve_bending_eigen(K, G, nodes, a, b, modes)


def navier_plate_load(a, b, young, poisson, thickness, ny_ratio=0.):
    """Exact simply-supported isotropic plate minimum over integer waves."""
    a=_finite_positive("a",a); b=_finite_positive("b",b); E=_finite_positive("young",young); t=_finite_positive("thickness",thickness)
    nu=float(poisson); r=float(ny_ratio)
    D=E*t**3/(12*(1-nu**2)); candidates=[]
    for m in range(1,20):
        for n in range(1,20):
            ax=(m*math.pi/a)**2; ay=(n*math.pi/b)**2
            candidates.append(D*(ax+ay)**2/(ax+r*ay))
    return min(candidates)


def run_shell4_buckling_qualification():
    case=dict(a=1.,b=1.,young=210e9,poisson=.3,thickness=.01,ny_ratio=0.)
    exact=navier_plate_load(**case)
    rows=[]
    for n in (4,8,16):
        value=float(simply_supported_plate_buckling(nx=n,ny=n,**case).load_factors[0])
        rows.append({"mesh":f"{n}x{n}","value":value,"relative_error":abs(value-exact)/exact})
    changes=[abs(rows[i]["value"]-rows[i-1]["value"])/rows[i]["value"] for i in (1,2)]
    # Mixed shell eigenvalues need not approach the limit from one side.
    passed=max(row["relative_error"] for row in rows) < .03 and changes[-1] < .03
    if not passed: raise AssertionError("Shell4 buckling qualification failed")
    return {"schema":"tensorfem.shell4-buckling/1.0","reference":"Navier simply-supported isotropic plate",
            "tolerance":.03,"exact":exact,"meshes":rows,"successive_changes":changes,"passed":passed,
            "boundary":"linear initial-stress eigenbuckling; no nonlinear postbuckling claim"}