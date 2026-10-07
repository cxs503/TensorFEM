"""Initial-stress buckling operators for flat Shell4 meshes.

The geometric matrix is integrated from the bilinear Q4 shape gradients and
acts on the transverse displacement of the existing six-DOF ``shell4``
element.  Stress resultants are positive in compression.  This is a linear
eigenbuckling operator: it does not update stress resultants after buckling.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from .shell4 import shell4_local_frame, shell4_stiffness


@dataclass(frozen=True)
class ShellBucklingResult:
    load_factors: torch.Tensor
    modes: torch.Tensor


def _finite_positive(name: str, value) -> float:
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return x


def shell4_geometric_stiffness(xyz: torch.Tensor, stress_resultants: torch.Tensor) -> torch.Tensor:
    """Return the 24x24 initial-stress matrix for constant local ``[Nx,Ny,Nxy]``.

    ``Nx`` and ``Ny`` are positive compressive force resultants per unit length.
    The returned positive-semidefinite matrix is used as ``K-lambda*Kg``.
    """
    if not isinstance(xyz, torch.Tensor) or not xyz.dtype.is_floating_point:
        raise TypeError("xyz must be a floating-point tensor")
    xy, basis = shell4_local_frame(xyz)
    N = torch.as_tensor(stress_resultants, dtype=xyz.dtype, device=xyz.device)
    if N.shape != (3,) or not bool(torch.all(torch.isfinite(N))):
        raise ValueError("stress_resultants must be finite [Nx, Ny, Nxy]")
    Smat = torch.stack((torch.stack((N[0], N[2])), torch.stack((N[2], N[1]))))
    kg4 = xyz.new_zeros((4, 4))
    g = 1.0 / math.sqrt(3.0)
    natural = ((-g, -g), (g, -g), (g, g), (-g, g))
    for xi, eta in natural:
        dnat = xyz.new_tensor([
            [-(1-eta), -(1-xi)], [(1-eta), -(1+xi)],
            [(1+eta), (1+xi)], [-(1+eta), (1-xi)],
        ]) * 0.25
        J = xy.T @ dnat
        detJ = torch.linalg.det(J)
        if bool((detJ <= torch.finfo(xyz.dtype).eps).item()):
            raise ValueError("inverted or degenerate shell element")
        grad = dnat @ torch.linalg.inv(J)
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


def simply_supported_plate_buckling(*, a: float, b: float, nx: int, ny: int,
                                    young: float, poisson: float, thickness: float,
                                    ny_ratio: float = 0., modes: int = 1) -> ShellBucklingResult:
    """Solve Shell4 plate buckling for unit ``Nx`` and ``Ny=ny_ratio*Nx``."""
    E = _finite_positive("young", young); t = _finite_positive("thickness", thickness)
    a = _finite_positive("a", a); b = _finite_positive("b", b)
    nu = float(poisson); ratio = float(ny_ratio)
    if not math.isfinite(nu) or not -1 < nu < .5: raise ValueError("poisson must be in (-1, 0.5)")
    if not math.isfinite(ratio) or ratio < 0: raise ValueError("ny_ratio must be finite and nonnegative")
    if isinstance(modes, bool) or modes < 1 or int(modes) != modes: raise ValueError("modes must be a positive integer")
    nodes, elements = rectangular_shell_mesh(a, b, nx, ny)
    ndof = 6*len(nodes); K = nodes.new_zeros((ndof, ndof)); G = K.clone()
    for conn in elements:
        ids = torch.stack(tuple(6*conn+k for k in range(6)), 1).reshape(-1)
        ke = shell4_stiffness(nodes[conn], E, nu, t)
        ge = shell4_geometric_stiffness(nodes[conn], nodes.new_tensor([1., ratio, 0.]))
        K[ids[:, None], ids] += ke; G[ids[:, None], ids] += ge
    # Bending subspace avoids irrelevant membrane/drilling null spaces.  A
    # simply-supported edge constrains w while rotations remain natural.
    bending = torch.stack((6*torch.arange(len(nodes))+2, 6*torch.arange(len(nodes))+3,
                           6*torch.arange(len(nodes))+4), 1).reshape(-1)
    tol = 64*torch.finfo(nodes.dtype).eps
    edge = ((nodes[:,0].abs() < tol) | ((nodes[:,0]-a).abs() < tol) |
            (nodes[:,1].abs() < tol) | ((nodes[:,1]-b).abs() < tol))
    fixed_w = set((6*torch.nonzero(edge).flatten()+2).tolist())
    free = torch.tensor([int(i) for i in bending if int(i) not in fixed_w], dtype=torch.long)
    Kr = K[free[:,None], free]; Gr = G[free[:,None], free]
    active = torch.linalg.vector_norm(Gr, dim=0) > 1e-13
    # Static condensation of rotation DOFs onto active transverse DOFs.
    ia = torch.nonzero(active).flatten(); ib = torch.nonzero(~active).flatten()
    recovery = None
    if len(ib):
        recovery = -torch.linalg.solve(Kr[ib[:,None],ib], Kr[ib[:,None],ia])
        Kc = Kr[ia[:,None],ia] + Kr[ia[:,None],ib] @ recovery
    else: Kc = Kr
    Gc = Gr[ia[:,None],ia]
    L = torch.linalg.cholesky(Gc)
    A = torch.linalg.solve_triangular(L, Kc, upper=False)
    A = torch.linalg.solve_triangular(L, A.T, upper=False).T
    values, vectors = torch.linalg.eigh((A+A.T)/2)
    keep = values > 1e-8
    values = values[keep][:int(modes)]; vectors = vectors[:,keep][:,:int(modes)]
    if len(values) < int(modes): raise RuntimeError("insufficient positive buckling modes")
    full = nodes.new_zeros((ndof, len(values)))
    active_modes = torch.linalg.solve_triangular(L.T, vectors, upper=True)
    full[free[ia],:] = active_modes
    if recovery is not None: full[free[ib],:] = recovery @ active_modes
    return ShellBucklingResult(values, full)


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
