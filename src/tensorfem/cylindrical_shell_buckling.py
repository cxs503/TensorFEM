"""Initial-stress (linear eigenbuckling) assembly for cylindrical Shell4 meshes.

The geometric-stiffness operator itself is the planar-facet operator of
:mod:`tensorfem.shell4_buckling`; what this module adds is (a) a ruled
cylindrical facet mesh whose local ``e1`` axis is the generator, so a uniform
axial compression maps to the local resultant ``[Nx, 0, 0]`` of every facet, and
(b) the diaphragm-simply-supported sparse assembly whose lowest eigenvalue is
the axial-compression buckling resultant of the shell.

Scope: linear (bifurcation) eigenbuckling under a *prescribed* unit membrane
resultant.  The prebuckling state is the uniformly compressed membrane field, so
no nonlinear load path, imperfection sensitivity or postbuckling response is
claimed.  A flat-facet cylinder is not an exact cylinder: projecting the radial
displacement of a vertex onto the facet plane injects a spurious in-plane
component, so the reported resultant carries a faceting error that vanishes
only as ``sin(theta/2)**2``.  :func:`run_cylindrical_shell_buckling_qualification`
measures that error instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .shell4 import shell4_stiffness
from .shell4_buckling import shell4_geometric_stiffness


@dataclass(frozen=True)
class CylinderBucklingResult:
    """Load factors ``N_cr`` (force per unit length), modes and mesh."""

    load_factors: torch.Tensor
    modes: torch.Tensor
    nodes: torch.Tensor
    elements: torch.Tensor


def _positive(name: str, value) -> float:
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return x


def _count(name: str, value) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 2:
        raise ValueError(f"{name} must be an integer >= 2")
    return int(value)


def _poisson(value) -> float:
    nu = float(value)
    if not math.isfinite(nu) or not -1.0 < nu < 0.5:
        raise ValueError("poisson must be in (-1, 0.5)")
    return nu


def cylindrical_shell_mesh(
    radius: float, length: float, nz: int, ntheta: int, *, dtype=torch.float64
):
    """Ruled cylindrical facet mesh, axis along ``x``, closed in theta.

    Node ``(i, j)`` sits at ``x = i*length/nz`` and ``theta = j*2*pi/ntheta``.
    Element ``(i, j)`` connects ``(i,j) -> (i+1,j) -> (i+1,j2) -> (i,j2)`` with
    ``j2 = (j+1) % ntheta``.  The first local facet axis is therefore a
    generator, which is what makes ``stress_resultants = [Nx, 0, 0]`` a uniform
    axial compression, and every facet is a planar rectangle.  The theta seam is
    shared, so the mesh has no duplicated nodes.
    """
    R = _positive("radius", radius)
    L = _positive("length", length)
    nz = _count("nz", nz)
    nt = _count("ntheta", ntheta)
    xs = torch.linspace(0.0, L, nz + 1, dtype=dtype)
    th = torch.arange(nt, dtype=dtype) * (2.0 * math.pi / nt)
    nodes = torch.stack(
        (
            xs[:, None].expand(nz + 1, nt).reshape(-1),
            (R * torch.cos(th)).expand(nz + 1, nt).reshape(-1),
            (R * torch.sin(th)).expand(nz + 1, nt).reshape(-1),
        ),
        dim=1,
    ).contiguous()
    conn = []
    for i in range(nz):
        for j in range(nt):
            j2 = (j + 1) % nt
            conn.append((i * nt + j, (i + 1) * nt + j, (i + 1) * nt + j2, i * nt + j2))
    return nodes, torch.tensor(conn, dtype=torch.long)


def donell_axisymmetric_load(
    length: float,
    young: float,
    poisson: float,
    thickness: float,
    radius: float,
    *,
    max_m: int = 400,
) -> float:
    """Exact Donnell axisymmetric buckling resultant for a finite SS cylinder.

    ``N(m) = [D (m*pi/L)**4 + E*t/R**2] / (m*pi/L)**2`` minimised over the axial
    half-wave count ``m``.  The minimiser of the length-independent form is the
    textbook ``E*t**2/(R*sqrt(3*(1-nu**2)))``; for a short cylinder the discrete
    ``m`` restriction is a few tenths of a percent above that asymptote, which
    is why this is the primary benchmark and the asymptote only an anchor.
    """
    L = _positive("length", length)
    E = _positive("young", young)
    t = _positive("thickness", thickness)
    R = _positive("radius", radius)
    nu = _poisson(poisson)
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    hoop = E * t / R**2
    best = None
    for m in range(1, int(max_m) + 1):
        k = m * math.pi / L
        value = (D * k**4 + hoop) / k**2
        if best is None or value < best[0]:
            best = (value, m)
    return float(best[0])


def classical_axial_buckling_resultant(
    young: float, poisson: float, thickness: float, radius: float
) -> float:
    """Textbook asymptotic shell buckling resultant ``E*t**2/(R*sqrt(3*(1-nu**2)))``."""
    E = _positive("young", young)
    t = _positive("thickness", thickness)
    R = _positive("radius", radius)
    nu = _poisson(poisson)
    return E * t**2 / (R * math.sqrt(3.0 * (1.0 - nu**2)))


def _assemble(nodes, elements, young, poisson, thickness, drilling_factor, formulation="shell4"):
    """Sparse global elastic and initial-stress matrices (scipy CSC/CSR).

    ``formulation`` selects the element elastic operator: ``"shell4"`` (the
    historical full-integration operator, the default) or ``"mitc"`` (the
    locking-controlled assumed-shear operator of :mod:`tensorfem.shell4_mitc`).
    The initial-stress operator is the same in both cases: it is the planar
    facet operator of :mod:`tensorfem.shell4_buckling` and does not depend on
    the bending formulation.
    """
    import numpy as np
    from scipy.sparse import coo_matrix

    if formulation not in ("shell4", "mitc"):
        raise ValueError("formulation must be 'shell4' or 'mitc'")
    if formulation == "mitc":
        from .shell4_mitc import shell4_mitc_stiffness as _elastic
    else:
        _elastic = shell4_stiffness

    ndof = 6 * len(nodes)
    unit = nodes.new_tensor([1.0, 0.0, 0.0])
    rows, cols, kv, gv = [], [], [], []
    for el in elements:
        ids = (6 * el[:, None] + torch.arange(6, dtype=torch.long)[None, :]).reshape(-1)
        xyz = nodes[el]
        ke = _elastic(xyz, young, poisson, thickness, drilling_factor=drilling_factor)
        ge = shell4_geometric_stiffness(xyz, unit)
        block = len(ids)
        r = ids.repeat_interleave(block).numpy()
        c = ids.repeat(block).numpy()
        rows.append(r)
        cols.append(c)
        kv.append(ke.reshape(-1).numpy())
        gv.append(ge.reshape(-1).numpy())
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    K = coo_matrix((np.concatenate(kv), (row, col)), shape=(ndof, ndof)).tocsr()
    G = coo_matrix((np.concatenate(gv), (row, col)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()
    G.sum_duplicates()
    return K, G


def _axisymmetric_constraint(nz: int, ntheta: int, *, dtype=torch.float64) -> torch.Tensor:
    """Constraint matrix of the axisymmetric subspace of the faceted cylinder.

    The reduced field has three DOFs per axial ring, ``(u, w, phi)``, where
    ``u`` is the axial translation, ``w`` the radial translation and ``phi``
    the rotation of the cross-section about the local tangent.  Node ``(i, j)``
    at angle ``theta_j`` therefore carries the global Cartesian displacement

    * translation ``(u_i, w_i*cos(theta_j), w_i*sin(theta_j))``,
    * rotation vector ``phi_i * (0, -sin(theta_j), cos(theta_j))``,

    i.e. rotation about the tangent.  The returned matrix ``C`` of shape
    ``(6*(nz+1)*ntheta, 3*(nz+1))`` maps the reduced vector to the full
    6-DOF-per-node field, so the reduced pencil is ``C^T K C w = lambda *
    C^T G C w``.  Two consequences make this subspace the clean one to
    certify: (a) the chord strain of the axisymmetric field is exactly
    ``w/R`` (the chord/arc factors cancel in the edge strain), so the hoop
    membrane term carries no faceting error; (b) the field is independent of
    theta, so the circumferential mesh only fixes the local frames.
    """
    nz = _count("nz", nz)
    nt = _count("ntheta", ntheta)
    nrings = nz + 1
    C = torch.zeros((6 * nrings * nt, 3 * nrings), dtype=dtype)
    th = torch.arange(nt, dtype=dtype) * (2.0 * math.pi / nt)
    cos, sin = torch.cos(th), torch.sin(th)
    for i in range(nrings):
        u_d, w_d, p_d = 3 * i, 3 * i + 1, 3 * i + 2
        for j in range(nt):
            base = 6 * (i * nt + j)
            C[base + 0, u_d] = 1.0
            C[base + 1, w_d] = cos[j]
            C[base + 2, w_d] = sin[j]
            C[base + 4, p_d] = -sin[j]
            C[base + 5, p_d] = cos[j]
    return C


def cylindrical_shell_axial_buckling(
    radius: float,
    length: float,
    *,
    ntheta: int,
    nz: int,
    young: float,
    poisson: float,
    thickness: float,
    modes: int = 1,
    drilling_factor: float = 1e-6,
    formulation: str = "shell4",
    axisymmetric: bool = False,
    dtype=torch.float64,
) -> CylinderBucklingResult:
    """Lowest axial-compression buckling resultant of a diaphragm-SS cylinder.

    Both end rings have all translational DOFs restrained (diaphragm
    simply-supported), which reproduces Donnell's ``u=v=w=0`` ends and lets the
    axial compression enter as the *prescribed* unit membrane resultant rather
    than as a load path.  The pencil is ``K*w = lambda*Kg*w`` with ``K`` positive
    definite on the free DOFs and ``Kg`` positive semi-definite; because ``Kg``
    only touches local transverse displacement it is singular, so the problem is
    solved through its inverse: ``lambda = 1/mu`` with ``mu`` the largest
    generalised eigenvalue of ``(Kg, K)``.  Only ``mu > 0`` is admissible, which
    discards the singular directions instead of inverting them.

    ``formulation`` selects the element elastic operator (``"shell4"`` default,
    bit-for-bit the historical path; ``"mitc"`` the locking-controlled
    assumed-shear operator).  ``axisymmetric=True`` condenses the pencil onto
    the three-DOF-per-ring axisymmetric subspace of
    :func:`_axisymmetric_constraint` before solving, which isolates the
    Donnell-type mode family from the non-axisymmetric facet modes.
    """
    R = _positive("radius", radius)
    L = _positive("length", length)
    nz_i = _count("nz", nz)
    nt_i = _count("ntheta", ntheta)
    E = _positive("young", young)
    t = _positive("thickness", thickness)
    nu = _poisson(poisson)
    if isinstance(modes, bool) or int(modes) != modes or int(modes) < 1:
        raise ValueError("modes must be a positive integer")
    try:
        import numpy as np
        from scipy.sparse.linalg import eigsh
    except ImportError as exc:  # pragma: no cover - scipy is a repository dependency
        raise RuntimeError("cylindrical shell buckling assembly requires scipy") from exc

    nodes, elements = cylindrical_shell_mesh(R, L, nz_i, nt_i, dtype=dtype)
    K, G = _assemble(nodes, elements, E, nu, t, drilling_factor, formulation=formulation)
    ndof = 6 * len(nodes)
    if axisymmetric:
        C = _axisymmetric_constraint(nz_i, nt_i, dtype=dtype)
        Cn = C.numpy()
        # Condense the pencil onto the axisymmetric subspace.  The condensed
        # ``G`` stays singular (it acts on the radial DOFs only), so the solve
        # uses the same shift-invert-about-zero strategy as the full path:
        # the factorised operator is ``K_red^{-1} G_red`` with ``K_red`` SPD.
        K_red = Cn.T @ K.toarray() @ Cn
        G_red = Cn.T @ G.toarray() @ Cn
        # Donnell ends: u and w of the two end rings are restrained, phi free.
        fixed_red = [0, 1, 3 * nz_i, 3 * nz_i + 1]
        mask = np.ones(3 * (nz_i + 1), dtype=bool)
        mask[np.asarray(fixed_red)] = False
        free = np.nonzero(mask)[0]
        Kf = K_red[free][:, free]
        Gf = G_red[free][:, free]
        # Dense symmetric solve: K_red is SPD on the free reduced DOFs, so
        # with K_red = L L^T and w = L^{-T} y the pencil becomes the standard
        # symmetric problem ``A y = mu y`` with ``A = L^{-1} G L^{-T}`` and
        # ``mu = 1/lambda``; the singular directions of G give mu = 0 and are
        # discarded by the positivity filter below.
        Lc = np.linalg.cholesky((Kf + Kf.T) / 2.0)
        LinvT = np.linalg.solve(Lc.T, np.eye(Kf.shape[0]))
        A = np.linalg.solve(Lc, Gf @ LinvT)
        A = (A + A.T) / 2.0
        evals, evecs = np.linalg.eigh(A)
        if not bool((evals > 1e-14).any()):
            raise RuntimeError("no positive buckling load factor found")
        order = np.argsort(evals)[::-1][: int(modes)]
        mus = evals[order]
        values = 1.0 / mus
        y = evecs[:, order]
        red = np.zeros((3 * (nz_i + 1), int(modes)))
        red[free, :] = np.linalg.solve(Lc.T, y)
        full = Cn @ red
    else:
        ring = torch.arange(nt_i, dtype=torch.long)
        end = torch.cat((ring, nz_i * nt_i + ring))
        fixed = (6 * end[:, None] + torch.arange(3, dtype=torch.long)[None, :]).reshape(-1)
        mask = np.ones(ndof, dtype=bool)
        mask[fixed.numpy()] = False
        free = np.nonzero(mask)[0]
        Kf = K[free][:, free].tocsc()
        Gf = G[free][:, free].tocsc()
        k = min(int(modes) + 2, Kf.shape[0] - 2)
        # Shift-invert about sigma=0 keeps the factorised operator positive
        # definite (K is SPD on the free DOFs) while ``Kg`` may stay singular.
        values, vec = eigsh(Kf, k=k, M=Gf, sigma=0.0, which="LM", tol=1e-9)
        order = np.argsort(values)
        values = values[order][: int(modes)]
        vec = vec[:, order][:, : int(modes)]
        full = np.zeros((ndof, int(modes)))
        full[free, :] = vec
    if not np.all(values > 0):
        raise RuntimeError("no positive buckling load factor found")
    return CylinderBucklingResult(
        torch.from_numpy(values).to(dtype), torch.from_numpy(full).to(dtype), nodes, elements
    )


def _faceting_ratio(ntheta: int) -> float:
    """Aspect ratio of the spurious in-plane projection: ``sin(pi/ntheta)**2``."""
    return math.sin(math.pi / _count("ntheta", ntheta)) ** 2


def run_cylindrical_shell_buckling_qualification(*, tolerance: float = 0.03) -> dict:
    """Mesh sweep of the cylinder axial-buckling resultant against Donnell theory.

    Case ``R=1.0 m``, ``t=0.01 m``, ``L=1.2 m``, ``E=200 GPa``, ``nu=0.3``:
    radius-to-thickness 100, length-to-radius 1.2, and the Donnell minimiser is
    ``m=7`` half waves, so the mesh must resolve ~0.17 m in the axial direction.
    Both a circumferential sweep (fixed ``nz``) and an axial sweep (fixed
    ``ntheta``) are run; the circumferential error law is fitted in
    ``sin(pi/ntheta)**2`` so the faceting error can be separated from the
    discretisation error and the extrapolated limit can be compared with theory.
    """
    case = dict(radius=1.0, length=1.2, young=200e9, poisson=0.3, thickness=0.01)
    finite = donell_axisymmetric_load(**case)
    classical = classical_axial_buckling_resultant(
        young=case["young"],
        poisson=case["poisson"],
        thickness=case["thickness"],
        radius=case["radius"],
    )

    def row(nz, nt):
        res = cylindrical_shell_axial_buckling(ntheta=nt, nz=nz, **case)
        value = float(res.load_factors[0])
        return {
            "nz": nz,
            "ntheta": nt,
            "elements": nz * nt,
            "N_cr": value,
            "error_vs_donnell": (value - finite) / finite,
            "error_vs_classical": (value - classical) / classical,
            "faceting_ratio": _faceting_ratio(nt),
        }

    hoop = [row(20, nt) for nt in (16, 24, 32, 48, 64, 96)]
    axial = [row(nz, 96) for nz in (10, 15, 20, 30, 40)]
    xs = [r["faceting_ratio"] for r in hoop]
    ys = [r["error_vs_donnell"] for r in hoop]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx
    finest = hoop[-1]
    changes = [
        abs(axial[i]["N_cr"] - axial[i - 1]["N_cr"]) / axial[i]["N_cr"]
        for i in range(1, len(axial))
    ]
    passed = (
        abs(intercept) < tolerance
        and abs(finest["error_vs_donnell"]) < tolerance
        and changes[-1] < tolerance
    )
    if not passed:
        raise AssertionError(
            "cylindrical shell buckling qualification failed: "
            f"extrapolated offset {intercept:.4%}, "
            f"finest mesh {finest['error_vs_donnell']:.4%}, "
            f"axial change {changes[-1]:.4%}"
        )
    return {
        "schema": "tensorfem.cylindrical-shell-buckling/1.0",
        "case": case,
        "references": {
            "donnell_finite_length": finite,
            "classical_asymptotic": classical,
            "donnell_over_classical": finite / classical,
        },
        "tolerance": tolerance,
        "circumferential_sweep": hoop,
        "axial_sweep": axial,
        "faceting_fit": {
            "slope": slope,
            "intercept": intercept,
            "model": "error = intercept + slope*sin(pi/ntheta)**2",
        },
        "successive_changes": changes,
        "passed": passed,
        "boundary": (
            "linear initial-stress eigenbuckling under a prescribed unit "
            "membrane resultant; flat-facet cylinder; diaphragm simply "
            "supported ends; no nonlinear postbuckling, imperfection "
            "sensitivity or plasticity claim"
        ),
    }
