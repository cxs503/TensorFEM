"""Simply-supported laminated composite plates in bending: three independent solvers.

Why this module exists
----------------------
``composite_laminate_buckling`` covers laminate *stability* with closed-form
CLT/FSDT Navier references but the repository has no laminate **bending**
verification face: no deflection reference, no thick-plate 3-D elasticity
reference, and no interlaminar-shear comparison.  This module supplies the
solvers that such a face needs, all for the same boundary-value problem::

    simply supported (shear-diaphragm / SS1) rectangular laminate, a x b,
    transverse load  q(x, y) = q0 sin(alpha x) sin(beta y),
    alpha = m*pi/a, beta = n*pi/b.

1. :func:`navier_clt` -- classical (Kirchhoff) laminate theory.  For a load with
   a single sine mode the Navier double series degenerates to one term, so this
   is the *exact* thin-plate solution.
2. :func:`navier_fsdt` -- first-order shear deformation theory
   (Mindlin-Reissner): exact 3x3 amplitude solution, with A45 coupling.
3. :class:`LayeredExact3D` -- Pagano-type layered 3-D elasticity solution
   (ply-wise state-space / propagator method, exact through the thickness).
   This is the thick-plate anchor and the only reference that resolves
   interlaminar shear and ply-level stress continuity.
4. :func:`fsdt_plate_fem` -- Q4 Mindlin FSDT plate finite elements, reusing the
   element of :mod:`tensorfem.plate` through ``PlateRigidity``.
5. :func:`layered_hex20_fem` -- 3-D HEX20 finite elements with one stiffness
   matrix per ply, reusing :mod:`tensorfem.hex20`; thick-plate FE path and the
   source of FE interlaminar shear.

Conventions
-----------
* Plate DOFs follow :mod:`tensorfem.plate`: ``(w, theta_x, theta_y)`` with
  ``u = z*theta_x``, ``v = z*theta_y``, ``kappa_x = d(theta_x)/dx``,
  ``kappa_y = d(theta_y)/dy``, ``kappa_xy = d(theta_x)/dy + d(theta_y)/dx``,
  ``gamma_xz = dw/dx + theta_x``, ``gamma_yz = dw/dy + theta_y``.
* SS1 constraints implied by the sine modes: ``w = 0`` on all edges;
  ``theta_y = 0`` on ``x = 0, a`` and ``theta_x = 0`` on ``y = 0, b`` (so the
  tangential in-plane surface displacement vanishes on every edge while the
  normal in-plane displacement is free -- the shear-diaphragm condition).
* 3-D stresses use the Voigt order of :mod:`tensorfem.solid3d`,
  ``(xx, yy, zz, xy, yz, xz)``.  The internal "tensor" order used when a ply
  stiffness is expanded into a fourth-order tensor is
  ``(xx, yy, zz, yz, xz, xy)``.
* Deflection is positive *downwards* and ``q0 > 0`` acts downwards, so the top
  face carries ``sigma_zz = -q0``.
* Lengths in m, stresses in Pa.

References
----------
* Reddy, J.N. (2004) *Mechanics of Laminated Composite Plates and Shells*, 2nd
  ed. -- CLT/FSDT bending of SS laminates and the layered 3-D formulation.
* Pagano, N.J. (1970) *Exact solutions for rectangular bidirectional composites
  and sandwich plates*, J. Composite Materials 4:20-34 -- the layered 3-D
  state-space solution reproduced here.
* Jones, R.M. (1999) *Mechanics of Composite Materials*, 2nd ed. -- ABD and
  transformed reduced stiffnesses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

__all__ = [
    "Ply3D",
    "Laminate",
    "symmetric_layup",
    "ply_stiffness_3d",
    "NavierResult",
    "navier_clt",
    "navier_fsdt",
    "navier_clt_series",
    "plate_fem",
    "LayeredExact3D",
    "fsdt_plate_fem",
    "layered_hex20_fem",
    "convergence_order",
    "gci_study",
    "richardson_extrapolate",
    "PAGANO_MATERIAL",
    "IM7_8552_MATERIAL",
]

#: Voigt ordering used by tensorfem.solid3d is (xx, yy, zz, xy, yz, xz); the
#: tensor order below is (xx, yy, zz, yz, xz, xy), hence this permutation.
FE_ORDER = (0, 1, 2, 5, 3, 4)
TENSOR_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


# ── material ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Ply3D:
    """Orthotropic ply with all nine engineering constants."""

    E1: float
    E2: float
    E3: float
    nu12: float
    nu13: float
    nu23: float
    G12: float
    G13: float
    G23: float

    def __post_init__(self) -> None:
        for name in ("E1", "E2", "E3", "G12", "G13", "G23"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("nu12", "nu13", "nu23"):
            if not -1.0 < float(getattr(self, name)) < 0.5:
                raise ValueError(f"{name} must lie in (-1, 0.5)")
        for nu, ea, eb in ((self.nu12, self.E1, self.E2),
                           (self.nu13, self.E1, self.E3),
                           (self.nu23, self.E2, self.E3)):
            if nu * nu * eb / ea >= 1.0:
                raise ValueError("Poisson pair is thermodynamically inadmissible")


#: Pagano (1970) benchmark set: E1/E2 = 25, G12 = G13 = 0.5 E2, G23 = 0.2 E2,
#: nu12 = nu13 = nu23 = 0.25.  Only ratios matter for the normalised comparison;
#: E2 = 1 GPa keeps stresses O(1) for q0 = 1 Pa.
PAGANO_MATERIAL = Ply3D(E1=25.0e9, E2=1.0e9, E3=1.0e9, nu12=0.25, nu13=0.25,
                        nu23=0.25, G12=0.5e9, G13=0.5e9, G23=0.2e9)

#: IM7/8552 carbon-epoxy: a realistic second ply for the ensemble scan.
IM7_8552_MATERIAL = Ply3D(E1=161.0e9, E2=11.38e9, E3=11.38e9, nu12=0.32,
                          nu13=0.32, nu23=0.436, G12=5.17e9, G13=5.17e9, G23=3.98e9)


def _material_stiffness(material: Ply3D) -> np.ndarray:
    """6x6 orthotropic stiffness in material axes, TENSOR_ORDER."""
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1.0 / material.E1, 1.0 / material.E2, 1.0 / material.E3
    S[0, 1] = S[1, 0] = -material.nu12 / material.E1
    S[0, 2] = S[2, 0] = -material.nu13 / material.E1
    S[1, 2] = S[2, 1] = -material.nu23 / material.E2
    S[3, 3] = 1.0 / material.G23
    S[4, 4] = 1.0 / material.G13
    S[5, 5] = 1.0 / material.G12
    return np.linalg.inv(S)


def _to_fourth(C6: np.ndarray) -> np.ndarray:
    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            v = C6[I, J]
            C4[i, j, k, l] = C4[j, i, k, l] = C4[i, j, l, k] = C4[j, i, l, k] = v
    return C4


def _to_sixth(C4: np.ndarray) -> np.ndarray:
    C6 = np.zeros((6, 6))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            C6[I, J] = C4[i, j, k, l]
    return C6


def ply_stiffness_3d(material: Ply3D, theta_deg: float = 0.0) -> np.ndarray:
    """6x6 ply stiffness in plate axes (TENSOR_ORDER), rotated ``theta_deg`` about z."""
    C4 = _to_fourth(_material_stiffness(material))
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # local axes on global basis
    C4g = np.einsum("ai,bj,ck,dl,abcd->ijkl", R, R, R, R, C4)
    return _to_sixth(C4g)


def to_fe_order(C6_tensor_order: np.ndarray) -> np.ndarray:
    """Reorder a TENSOR_ORDER 6x6 into the solid3d FE order (xx, yy, zz, xy, yz, xz)."""
    return np.asarray(C6_tensor_order)[np.ix_(FE_ORDER, FE_ORDER)]


# ── laminate ────────────────────────────────────────────────────────────────
def symmetric_layup(half_angles: Sequence[float]) -> List[float]:
    """Full top-to-bottom angle list of ``[a, b, ... , b, a]``."""
    return list(half_angles) + list(reversed(half_angles))


@dataclass(frozen=True)
class Laminate:
    """Laminate: full top-to-bottom ply angles, equal ply thickness, one material."""

    angles_deg: Sequence[float]
    ply_thickness: float
    material: Ply3D = PAGANO_MATERIAL

    def __post_init__(self) -> None:
        if not len(self.angles_deg):
            raise ValueError("angles_deg must not be empty")
        if not float(self.ply_thickness) > 0.0:
            raise ValueError("ply_thickness must be positive")

    @property
    def n_plies(self) -> int:
        return len(self.angles_deg)

    @property
    def thickness(self) -> float:
        return self.n_plies * self.ply_thickness

    @property
    def z_interfaces(self) -> List[float]:
        """Ply interfaces bottom to top, midsurface at z = 0."""
        return [-0.5 * self.thickness + k * self.ply_thickness
                for k in range(self.n_plies + 1)]

    @property
    def is_balanced(self) -> bool:
        """True when ``A16 = A26 = 0`` (no extension-shear coupling)."""
        A = self.abd()[0]
        return (abs(A[0, 2]) < 1e-9 * A[0, 0]) and (abs(A[1, 2]) < 1e-9 * A[0, 0])

    @property
    def is_symmetric(self) -> bool:
        angles = [float(a) for a in self.angles_deg]
        return angles == list(reversed(angles))

    def abd(self):
        """Plane-stress ``(A, B, D, As)``; ``As`` indexed ``[xz, yz]`` (A55, A45; A45, A44)."""
        z = self.z_interfaces
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        As = np.zeros((2, 2))
        ip, op = [0, 1, 5], [2, 3, 4]   # xx, yy, xy and zz, yz, xz (TENSOR_ORDER)
        for k, theta in enumerate(self.angles_deg):
            C = ply_stiffness_3d(self.material, theta)
            Q = C[np.ix_(ip, ip)] - C[np.ix_(ip, op)] @ np.linalg.solve(C[np.ix_(op, op)],
                                                                        C[np.ix_(op, ip)])
            z0, z1 = z[k], z[k + 1]
            A += Q * (z1 - z0)
            B += 0.5 * Q * (z1**2 - z0**2)
            D += Q * (z1**3 - z0**3) / 3.0
            G = np.array([[C[4, 4], C[4, 3]], [C[3, 4], C[3, 3]]])
            As += G * (z1 - z0)
        return A, B, D, As

    def d_matrix(self) -> np.ndarray:
        return self.abd()[2]

    def as_matrix(self, shear_correction: float = 5.0 / 6.0) -> np.ndarray:
        return shear_correction * self.abd()[3]

    def incident_ply(self, z: float) -> int:
        zs = self.z_interfaces
        for k in range(self.n_plies):
            if z <= zs[k + 1] + 1e-12:
                return k
        return self.n_plies - 1


# ── Navier reference solutions ──────────────────────────────────────────────
@dataclass(frozen=True)
class NavierResult:
    """Amplitudes of the single excited sine mode."""

    theory: str
    a: float
    b: float
    q0: float
    m: int
    n: int
    W: float
    X: float
    Y: float
    D: np.ndarray
    As: np.ndarray

    @property
    def center_deflection(self) -> float:
        return self.W


def _check_special_orthotropic(D: np.ndarray, tol: float = 1e-6) -> None:
    scale = max(abs(D[0, 0]), abs(D[1, 1]), abs(D[2, 2]), 1e-300)
    if abs(D[0, 2]) > tol * scale or abs(D[1, 2]) > tol * scale:
        raise ValueError("Navier references require a balanced laminate "
                         f"(D16 = D26 = 0); got D16={D[0, 2]:.3e}, D26={D[1, 2]:.3e}")


def navier_clt(laminate: Laminate, a: float, b: float, q0: float = 1.0,
               m: int = 1, n: int = 1) -> NavierResult:
    """Exact Kirchhoff (CLT) solution: ``W = q0 / (D11 a^4 + 2H a^2 b^2 + D22 b^4)`` in wavenumbers."""
    D = laminate.d_matrix()
    _check_special_orthotropic(D)
    al, be = m * math.pi / a, n * math.pi / b
    H = D[0, 1] + 2.0 * D[2, 2]
    kw = D[0, 0] * al**4 + 2.0 * H * al**2 * be**2 + D[1, 1] * be**4
    W = q0 / kw
    return NavierResult("CLT", a, b, q0, m, n, W, -al * W, -be * W, D, np.zeros((2, 2)))


def navier_fsdt(laminate: Laminate, a: float, b: float, q0: float = 1.0,
                m: int = 1, n: int = 1,
                shear_correction: float = 5.0 / 6.0) -> NavierResult:
    """Exact first-order (Mindlin-Reissner) solution: symmetric 3x3 amplitude system."""
    D = laminate.d_matrix()
    _check_special_orthotropic(D)
    As = laminate.as_matrix(shear_correction)
    al, be = m * math.pi / a, n * math.pi / b
    k55, k45, k44 = As[0, 0], As[0, 1], As[1, 1]
    K = np.array([
        [D[0, 0] * al**2 + D[2, 2] * be**2 + k55, (D[0, 1] + D[2, 2]) * al * be + k45,
         k55 * al + k45 * be],
        [(D[0, 1] + D[2, 2]) * al * be + k45, D[2, 2] * al**2 + D[1, 1] * be**2 + k44,
         k45 * al + k44 * be],
        [k55 * al + k45 * be, k45 * al + k44 * be,
         k55 * al**2 + 2.0 * k45 * al * be + k44 * be**2],
    ])
    X, Y, W = np.linalg.solve(K, np.array([0.0, 0.0, q0]))
    return NavierResult("FSDT", a, b, q0, m, n, float(W), float(X), float(Y), D, As)


def _int_ss(m: int, p: int, length: float) -> float:
    """``int_0^L sin(m pi x / L) sin(p pi x / L) dx``."""
    return 0.5 * length if m == p else 0.0


def _int_cc(m: int, p: int, length: float) -> float:
    """``int_0^L cos(m pi x / L) cos(p pi x / L) dx``."""
    return 0.5 * length if m == p else 0.0


def _int_sc(m: int, p: int, length: float) -> float:
    """``int_0^L sin(m pi x / L) cos(p pi x / L) dx`` (nonzero for m +- p odd)."""
    total = 0.0
    for sign in (+1.0, -1.0):
        k = (m + sign * p) * math.pi / length
        if abs(k) < 1e-14:
            continue
        total += (1.0 - math.cos(k * length)) / k
    return 0.5 * total


def navier_clt_series(laminate: Laminate, a: float, b: float, q0: float = 1.0,
                      modes: int = 15) -> dict:
    """Truncated Navier (Fourier-Ritz) CLT solution for a *general* laminate.

    :func:`navier_clt` is exact only for ``D16 = D26 = 0``; with bending-twisting
    coupling the odd derivatives turn ``sin sin`` into ``cos cos`` and the load's
    single (1,1) mode excites a whole family of sine modes.  This routine
    minimises the classical laminate energy in a ``modes x modes`` sine basis,

        Pi = 1/2 int [ D11 w,xx^2 + 2 D12 w,xx w,yy + 4 D66 w,xy^2
                       + 4 D16 w,xx w,xy + 4 D26 w,yy w,xy + D22 w,yy^2 ] - int q w,

    so the centre deflection converges from above as ``modes`` grows, and for
    ``D16 = D26 = 0`` the system is diagonal and reproduces :func:`navier_clt`
    mode for mode.  All overlap integrals are analytic.
    """
    D = laminate.d_matrix()
    M = int(modes)
    al = np.arange(1, M + 1) * math.pi / a          # (M,)
    be = np.arange(1, M + 1) * math.pi / b
    Iss = np.array([[_int_ss(i + 1, j + 1, a) for j in range(M)] for i in range(M)])
    Isc_x = np.array([[_int_sc(i + 1, j + 1, a) for j in range(M)] for i in range(M)])
    Isc_xT = np.array([[_int_sc(j + 1, i + 1, a) for j in range(M)] for i in range(M)])
    Icc = np.array([[_int_cc(i + 1, j + 1, a) for j in range(M)] for i in range(M)])
    Iss_y = np.array([[_int_ss(i + 1, j + 1, b) for j in range(M)] for i in range(M)])
    Isc_y = np.array([[_int_sc(i + 1, j + 1, b) for j in range(M)] for i in range(M)])
    Isc_yT = np.array([[_int_sc(j + 1, i + 1, b) for j in range(M)] for i in range(M)])
    Icc_y = np.array([[_int_cc(i + 1, j + 1, b) for j in range(M)] for i in range(M)])

    am, ap = al[:, None], al[None, :]
    bn, bq = be[:, None], be[None, :]

    def kron2(Ax, By):
        return np.kron(Ax, By)

    K = (D[0, 0] * kron2(am**2 * ap**2 * Iss, Iss_y)
         + D[1, 1] * kron2(Iss, bn**2 * bq**2 * Iss_y)
         + D[0, 1] * (kron2(am**2 * Iss, bq**2 * Iss_y) + kron2(ap**2 * Iss, bn**2 * Iss_y))
         + 4.0 * D[2, 2] * kron2(am * ap * Icc, bn * bq * Icc_y)
         - 2.0 * D[0, 2] * (kron2(am**2 * ap * Isc_x, bq * Isc_y)
                            + kron2(ap**2 * am * Isc_xT, bn * Isc_yT))
         - 2.0 * D[1, 2] * (kron2(ap * Isc_x, bn**2 * bq * Isc_y)
                            + kron2(am * Isc_xT, bq**2 * bn * Isc_yT)))
    F = np.zeros(M * M)
    F[0] = q0 * _int_ss(1, 1, a) * _int_ss(1, 1, b)
    W = np.linalg.solve(K, F)
    Wm = W.reshape(M, M)
    signs_m = np.sin(al * 0.5 * a)
    signs_n = np.sin(be * 0.5 * b)
    centre = float(signs_m @ Wm @ signs_n)
    return {"theory": "CLT-series", "modes": M, "W": centre, "D": D, "W11": float(Wm[0, 0])}


# ── plate finite element with the full ABD bending stiffness ────────────────
def _q4_shape(xi: float, eta: float):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                          [(1 - eta), -(1 + xi)],
                          [(1 + eta), (1 + xi)],
                          [-(1 + eta), (1 - xi)]])
    return N, dN


def _q4_laminate_stiffness(xy: np.ndarray, D: np.ndarray, As: np.ndarray,
                           shear_treatment: str = "mitc") -> np.ndarray:
    """12x12 Mindlin Q4 element stiffness for a laminate with the full 3x3 ``D``.

    Same kinematics as :func:`tensorfem.plate.q4_mindlin_stiffness` but the
    bending block carries the complete ``D`` (so ``D16``/``D26`` bending-twisting
    coupling of angle-ply laminates is retained).  Shear is the Dvorkin-Bathe
    assumed field (``"mitc"``) by default, which has no spurious mechanisms.
    """
    ke = np.zeros((12, 12))
    gp = 1.0 / math.sqrt(3.0)

    def bending_matrix(xi: float, eta: float):
        _, dN_nat = _q4_shape(xi, eta)
        J = dN_nat.T @ xy
        dN = dN_nat @ np.linalg.inv(J).T
        B = np.zeros((3, 12))
        for i in range(4):
            B[0, 3 * i + 1] = dN[i, 0]
            B[1, 3 * i + 2] = dN[i, 1]
            B[2, 3 * i + 1] = dN[i, 1]
            B[2, 3 * i + 2] = dN[i, 0]
        return B, float(np.linalg.det(J))

    for xi in (-gp, gp):
        for eta in (-gp, gp):
            B, detJ = bending_matrix(xi, eta)
            ke += B.T @ D @ B * detJ

    def shear_rows(xi: float, eta: float):
        N, dN_nat = _q4_shape(xi, eta)
        J = dN_nat.T @ xy
        dN = dN_nat @ np.linalg.inv(J).T
        rows = np.zeros((2, 12))
        for i in range(4):
            rows[0, 3 * i] = dN[i, 0]
            rows[0, 3 * i + 1] = N[i]
            rows[1, 3 * i] = dN[i, 1]
            rows[1, 3 * i + 2] = N[i]
        return rows

    if shear_treatment == "mitc":
        bottom, top = shear_rows(0.0, -1.0), shear_rows(0.0, 1.0)
        left, right = shear_rows(-1.0, 0.0), shear_rows(1.0, 0.0)
        for xi in (-gp, gp):
            for eta in (-gp, gp):
                _, detJ = bending_matrix(xi, eta)
                Bs = np.zeros((2, 12))
                Bs[0] = 0.5 * (1 - eta) * bottom[0] + 0.5 * (1 + eta) * top[0]
                Bs[1] = 0.5 * (1 - xi) * left[1] + 0.5 * (1 + xi) * right[1]
                ke += Bs.T @ As @ Bs * detJ
    else:      # selective reduced integration: one-point shear
        Bs = shear_rows(0.0, 0.0)
        _, detJ = bending_matrix(0.0, 0.0)
        ke += Bs.T @ As @ Bs * detJ * 4.0
    return ke


def plate_fem(laminate: Laminate, a: float, q0: float = 1.0, n: int = 8,
              b: float | None = None, m: int = 1, n_mode: int = 1,
              shear_treatment: str = "mitc",
              shear_correction: float = 5.0 / 6.0) -> dict:
    """Q4 Mindlin FSDT plate FE solution for an arbitrary (also angle-ply) laminate."""
    a = float(a)
    b = float(b if b is not None else a)
    D = laminate.d_matrix()
    As = laminate.as_matrix(shear_correction)
    nx, ny = int(n), max(1, int(round(n * b / a)))
    xs = np.linspace(0.0, a, nx + 1)
    ys = np.linspace(0.0, b, ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    ndof = 3 * nodes.shape[0]
    K = np.zeros((ndof, ndof))
    f = np.zeros(ndof)
    al, be = m * math.pi / a, n_mode * math.pi / b
    gp = 1.0 / math.sqrt(3.0)
    for i in range(nx):
        for j in range(ny):
            cell = [idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]]
            xy = nodes[cell]
            dofs = np.array([3 * c + k for c in cell for k in range(3)])
            K[np.ix_(dofs, dofs)] += _q4_laminate_stiffness(xy, D, As, shear_treatment)
            jac = 0.25 * (a / nx) * (b / ny)
            for xi in (-gp, gp):
                for eta in (-gp, gp):
                    N, _ = _q4_shape(xi, eta)
                    pt = N @ xy
                    qv = q0 * math.sin(al * pt[0]) * math.sin(be * pt[1])
                    f[dofs[0::3]] += N * qv * jac
    tol = 1e-12 * max(a, b)
    fixed = []
    for i in range(nodes.shape[0]):
        x0, y0 = nodes[i]
        if abs(x0) < tol or abs(x0 - a) < tol:
            fixed += [3 * i, 3 * i + 2]        # w = 0, tangential rotation = 0
        if abs(y0) < tol or abs(y0 - b) < tol:
            fixed += [3 * i, 3 * i + 1]
    fixed = np.array(sorted(set(fixed)))
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    residual = float(np.linalg.norm(K[free] @ u - f[free]) / max(np.linalg.norm(f), 1e-300))
    cx = int(np.argmin((nodes[:, 0] - 0.5 * a) ** 2 + (nodes[:, 1] - 0.5 * b) ** 2))
    return {"w_center": float(u[3 * cx]), "n_elements": int(nx * ny),
            "n_dofs": int(ndof), "residual": residual, "n": int(n),
            "shear_treatment": shear_treatment}


# ── layered 3-D elasticity (Pagano state space) ─────────────────────────────
def _expm(M: np.ndarray) -> np.ndarray:
    """Matrix exponential of a small dense matrix (scipy Pade, no scaling needed).

    The Pagano state used here is the *derivative* state ``(U, V, W, U', V', W')``,
    whose matrix entries are all O(wavenumber), so a plain Pade exponential is
    accurate; the displacement/stress state ``(U, V, W, S_xz, S_yz, S_zz)`` spans
    20 decades and is only used through the similarity transforms below.
    """
    import scipy.linalg as sla

    return sla.expm(M)


class LayeredExact3D:
    """Exact layered 3-D elasticity solution of the SS1 sine-loaded laminate.

    Pagano (1970) propagator method.  Inside ply ``k`` the derivative state

        y = (U, V, W, U', V', W')

    of the (m, n) mode satisfies ``dy/dz = A_k y`` with ``A_k`` constant, where

        u = U(z) cos(alpha x) sin(beta y),  v = V(z) sin(alpha x) cos(beta y),
        w = W(z) sin(alpha x) sin(beta y)

    and the remaining stress amplitudes follow from the ply stiffness.  Plies are
    chained through the interface continuity matrix ``J_k = T_{k+1} T_k^{-1}``
    (``T_k`` maps the displacement/stress state -- the interface-continuous
    variable -- to the derivative state), and the six traction conditions on the
    two faces determine the six unknowns at the bottom face.

    The face conditions are traction conditions, i.e. they involve the strain
    field, which is why the *derivative* state is the unknown: in the pure
    displacement/stress state the three traction conditions at a face leave a
    near-singular block (the state carries a slowly penetrating mode) and the
    linear solve loses every digit.
    """

    def __init__(self, laminate: Laminate, a: float, b: float, q0: float = 1.0,
                 m: int = 1, n: int = 1) -> None:
        for name, val in (("a", a), ("b", b), ("q0", q0)):
            if not float(val) > 0.0:
                raise ValueError(f"{name} must be positive")
        if m < 1 or n < 1:
            raise ValueError("half-wave numbers must be >= 1")
        for theta in laminate.angles_deg:
            if abs(math.remainder(float(theta), 90.0)) > 1e-9:
                raise ValueError(
                    "the ply-wise state-space (Pagano) solution is exact only for "
                    "specially orthotropic plies (fibres at 0 or 90 deg): for a "
                    f"general angle ply the in-plane and twisting strain bases mix "
                    f"and the 6-ODE reduction is invalid (got theta = {theta} deg)")
        self.laminate = laminate
        self.a, self.b, self.q0, self.m, self.n = float(a), float(b), float(q0), m, n
        self._z = laminate.z_interfaces
        self._C = [ply_stiffness_3d(laminate.material, t) for t in laminate.angles_deg]
        self._T = [self._to_derivative_state(C) for C in self._C]
        self._R = [self._traction_rows(C) for C in self._C]
        self._A = [self._ply_matrix(self._C[k], self._T[k]) for k in range(laminate.n_plies)]
        self._E = [_expm(A * self.laminate.ply_thickness) for A in self._A]
        self._J = [self._T[k + 1] @ np.linalg.inv(self._T[k])
                   for k in range(laminate.n_plies - 1)]
        self._x0 = self._solve_bottom_state()

    # -- internals -----------------------------------------------------------
    @property
    def alpha(self) -> float:
        return self.m * math.pi / self.a

    @property
    def beta(self) -> float:
        return self.n * math.pi / self.b

    def _strain_split(self, y_deriv: np.ndarray):
        """``(in_plane, out_plane)`` strain amplitudes for a derivative state."""
        U, V, W, Up, Vp, Wp = y_deriv
        al, be = self.alpha, self.beta
        in_plane = np.array([-al * U, -be * V, be * U + al * V])
        out_plane = np.array([Wp, Vp + be * W, Up + al * W])   # eps_zz, gamma_yz, gamma_xz
        return in_plane, out_plane

    def _to_derivative_state(self, C: np.ndarray) -> np.ndarray:
        """``6x6`` map from the displacement/stress state to the derivative state."""
        al, be = self.alpha, self.beta
        ip, op = [0, 1, 5], [2, 3, 4]
        Minv = np.linalg.inv(C[np.ix_(op, op)])
        Coi, Cio, Cii = C[np.ix_(op, ip)], C[np.ix_(ip, op)], C[np.ix_(ip, ip)]
        T = np.zeros((6, 6))
        T[:3, :3] = np.eye(3)
        for j in range(6):
            e = np.zeros(6)
            e[j] = 1.0
            in_plane = np.array([-al * e[0], -be * e[1], be * e[0] + al * e[1]])
            out_plane = np.array([e[5], e[4], e[3]])
            azz, ayz, axz = Minv @ (out_plane - Coi @ in_plane)
            T[3, j] = axz - al * e[2]      # U' = gamma_xz - alpha W
            T[4, j] = ayz - be * e[2]      # V' = gamma_yz - beta  W
            T[5, j] = azz                  # W' = epsilon_zz
        return T

    def _traction_rows(self, C: np.ndarray) -> np.ndarray:
        """``3x6`` rows giving ``(tau_yz, tau_xz, sigma_zz)`` from a derivative state."""
        ip, op = [0, 1, 5], [2, 3, 4]
        Cio, Coi, Coo = C[np.ix_(ip, op)], C[np.ix_(op, ip)], C[np.ix_(op, op)]
        rows = np.zeros((3, 6))
        for j in range(6):
            e = np.zeros(6)
            e[j] = 1.0
            in_plane, out_plane = self._strain_split(e)
            S_op = Coi @ in_plane + Coo @ out_plane     # (sigma_zz, tau_yz, tau_xz)
            rows[:, j] = [S_op[1], S_op[2], S_op[0]]
        return rows

    def _ply_matrix(self, C: np.ndarray, T: np.ndarray) -> np.ndarray:
        """``dy/dz = A y`` for the derivative state (built from the stress state)."""
        al, be = self.alpha, self.beta
        ip, op = [0, 1, 5], [2, 3, 4]
        Minv = np.linalg.inv(C[np.ix_(op, op)])
        Coi, Cio, Cii = C[np.ix_(op, ip)], C[np.ix_(ip, op)], C[np.ix_(ip, ip)]
        A_s = np.zeros((6, 6))
        for j in range(6):
            e = np.zeros(6)
            e[j] = 1.0
            in_plane = np.array([-al * e[0], -be * e[1], be * e[0] + al * e[1]])
            out_plane = np.array([e[5], e[4], e[3]])
            azz, ayz, axz = Minv @ (out_plane - Coi @ in_plane)
            A_s[0, j] = axz - al * e[2]
            A_s[1, j] = ayz - be * e[2]
            A_s[2, j] = azz
            sxx, syy, sxy = Cii @ in_plane + Cio @ np.array([azz, ayz, axz])
            A_s[3, j] = be * sxy - al * sxx
            A_s[4, j] = al * sxy - be * syy
            A_s[5, j] = al * e[3] + be * e[4]
        return T @ A_s @ np.linalg.inv(T)

    def _bottom_states(self) -> list:
        """Derivative state at the bottom face of every ply."""
        out = [self._x0]
        for k in range(self.laminate.n_plies - 1):
            out.append(self._J[k] @ (self._E[k] @ out[-1]))
        return out

    def _solve_bottom_state(self) -> np.ndarray:
        y = np.eye(6)                      # identity propagation of the bottom unknown
        for k in range(self.laminate.n_plies):
            y = self._E[k] @ y             # state at the top face of ply k
            if k < self.laminate.n_plies - 1:
                y = self._J[k] @ y
        top = self._R[-1] @ y
        rows = np.vstack([self._R[0], top])
        rhs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -self.q0])
        scale = np.max(np.abs(rows), axis=1)
        return np.linalg.solve(rows / scale[:, None], rhs / scale)

    def _states_at(self, z: float) -> np.ndarray:
        k = self.laminate.incident_ply(float(z))
        bottoms = self._bottom_states()
        return _expm(self._A[k] * (float(z) - self._z[k])) @ bottoms[k]

    # -- fields --------------------------------------------------------------
    def stress_amplitudes(self, z: float) -> np.ndarray:
        """Six stress amplitudes ``(S_xx, S_yy, S_zz, S_yz, S_xz, S_xy)`` at ``z``."""
        k = self.laminate.incident_ply(float(z))
        y = self._states_at(z)
        in_plane, out_plane = self._strain_split(y)
        C = self._C[k]
        ip, op = [0, 1, 5], [2, 3, 4]
        S = np.zeros(6)
        S[ip] = C[np.ix_(ip, ip)] @ in_plane + C[np.ix_(ip, op)] @ out_plane
        S[op] = C[np.ix_(op, ip)] @ in_plane + C[np.ix_(op, op)] @ out_plane
        return S

    def w_profile(self, z=None) -> np.ndarray:
        """Centre deflection (downward positive) at ``z`` (scalar or array)."""
        zs = np.atleast_1d(np.asarray(0.0 if z is None else z, dtype=float))
        return -np.array([self._states_at(float(zz))[2] for zz in zs])

    @property
    def center_deflection(self) -> float:
        return float(self.w_profile(0.0)[0])

    def face_shear(self, z: float, x: float = None, y: float = None) -> float:
        """``tau_xz(x, y, z)``; it peaks at the *edge* ``x = 0, y = b/2``."""
        x = 0.0 if x is None else x
        y = 0.5 * self.b if y is None else y
        amp = self.stress_amplitudes(z)[4]
        return float(amp * math.cos(self.alpha * x) * math.sin(self.beta * y))

    def tau_xz_profile(self, z=None, x: float = None, y: float = None) -> np.ndarray:
        zs = np.atleast_1d(np.asarray(0.0 if z is None else z, dtype=float))
        return np.array([self.face_shear(float(zz), x, y) for zz in zs])

    def interface_shear(self) -> list:
        return [{"z": float(z), "tau_xz": self.face_shear(float(z))}
                for z in self._z[1:-1]]

    def max_interface_shear(self) -> float:
        vals = [abs(item["tau_xz"]) for item in self.interface_shear()]
        return max(vals) if vals else 0.0

    def verification(self, samples: int = 7) -> dict:
        """Independent checks of the state-space solution (all should be ~0)."""
        q = abs(self.q0)
        face = 0.0
        for z, target in ((self._z[0], 0.0), (self._z[-1], -self.q0)):
            k = self.laminate.incident_ply(float(z))
            tr = self._R[k] @ self._states_at(z)
            face = max(face, abs(tr[0]) / q, abs(tr[1]) / q, abs(tr[2] - target) / q)
        jump = 0.0
        for z in self._z[1:-1]:
            lo = self.stress_amplitudes(z - 1e-9)
            hi = self.stress_amplitudes(z + 1e-9)
            jump = max(jump, float(np.max(np.abs(hi[2:] - lo[2:]))) /
                       max(abs(self.q0) * self.a, 1e-300))
        res = 0.0
        for z in np.linspace(self._z[0], self._z[-1], samples + 2)[1:-1]:
            if min(abs(z - zi) for zi in self._z) < 1e-6:
                continue                       # the state derivative jumps at an interface
            k = self.laminate.incident_ply(float(z))
            h = 1e-7
            num = (self._states_at(z + h) - self._states_at(z - h)) / (2 * h)
            dy = self._A[k] @ self._states_at(z)
            res = max(res, float(np.max(np.abs(dy - num))) / (1.0 + float(np.max(np.abs(dy)))))
        return {"face_traction_rel": face, "interface_stress_jump_rel": jump,
                "state_residual": res}


# ── finite-element solvers ──────────────────────────────────────────────────
def _fsdt_rigidity(laminate: Laminate, shear_correction: float = 5.0 / 6.0):
    from .plate import PlateRigidity

    D = laminate.d_matrix()
    _check_special_orthotropic(D)
    As = laminate.as_matrix(shear_correction)
    if abs(As[0, 1]) > 1e-6 * max(As[0, 0], As[1, 1]):
        raise ValueError("PlateRigidity cannot carry A45 coupling (balanced layups only)")
    return PlateRigidity(dx=float(D[0, 0]), dy=float(D[1, 1]), dxy=float(D[0, 1]),
                         dt=float(D[2, 2]), sx=float(As[0, 0]), sy=float(As[1, 1]))


def fsdt_plate_fem(laminate: Laminate, a: float, q0: float = 1.0, n: int = 8,
                   b: float | None = None, m: int = 1, n_mode: int = 1,
                   shear_treatment: str = "mitc",
                   shear_correction: float = 5.0 / 6.0) -> dict:
    """Q4 Mindlin FSDT plate FE solution of the SS1 sine-loaded laminate.

    ``n`` is the number of subdivisions along ``a``; the mesh follows ``b/a``.
    Returns the centre deflection, stresses via the element B matrices and the
    discrete residual of the reduced system.
    """
    import torch

    from .plate import q4_mindlin_stiffness

    dtype = torch.float64
    a = float(a)
    b = float(b if b is not None else a)
    rig = _fsdt_rigidity(laminate, shear_correction)
    nx, ny = int(n), max(1, int(round(n * b / a)))
    xs = torch.linspace(0.0, a, nx + 1, dtype=dtype)
    ys = torch.linspace(0.0, b, ny + 1, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    idx = torch.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    conn = torch.stack([torch.stack((idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]))
                        for j in range(ny) for i in range(nx)])
    ndof = 3 * nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=dtype)
    f = torch.zeros(ndof, dtype=dtype)
    al, be = m * math.pi / a, n_mode * math.pi / b
    gp = 1.0 / math.sqrt(3.0)
    jac = 0.25 * (a / nx) * (b / ny)          # detJ of the affine Q4 map
    for cell in conn:
        xy = nodes[cell]
        dofs = (3 * cell[:, None] + torch.arange(3)).reshape(-1)
        K[dofs[:, None], dofs[None, :]] += q4_mindlin_stiffness(
            xy, rigidity=rig, shear_treatment=shear_treatment)
        for xi in (-gp, gp):
            for eta in (-gp, gp):
                N = 0.25 * torch.stack(((1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                                        (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)))
                pt = (N.unsqueeze(1) * xy).sum(0)
                qv = q0 * math.sin(al * float(pt[0])) * math.sin(be * float(pt[1]))
                f[dofs[0::3]] += N * qv * float(jac)
    fixed = []
    tol = 1e-12 * max(a, b)
    for i in range(nodes.shape[0]):
        x0, y0 = float(nodes[i, 0]), float(nodes[i, 1])
        if abs(x0) < tol or abs(x0 - a) < tol:
            fixed += [3 * i, 3 * i + 2]        # w = 0, theta_y = 0
        if abs(y0) < tol or abs(y0 - b) < tol:
            fixed += [3 * i, 3 * i + 1]        # w = 0, theta_x = 0
    fixed = torch.tensor(sorted(set(fixed)), dtype=torch.long)
    free = torch.tensor([d for d in range(ndof) if d not in set(fixed.tolist())], dtype=torch.long)
    Kff = K[free][:, free]
    u = torch.zeros(ndof, dtype=dtype)
    u[free] = torch.linalg.solve(Kff, f[free])
    residual = float(torch.linalg.norm(K[free] @ u - f[free]) / max(float(torch.linalg.norm(f)), 1e-300))
    # centre node (exact half-grid point exists for even n)
    cx = torch.argmin((nodes[:, 0] - 0.5 * a).abs() + (nodes[:, 1] - 0.5 * b).abs())
    return {"w_center": float(u[3 * int(cx)]), "n_elements": int(len(conn)),
            "n_dofs": int(ndof), "residual": residual,
            "shear_treatment": shear_treatment}


def layered_hex20_fem(laminate: Laminate, a: float, q0: float = 1.0, nx: int = 8,
                      nz_per_ply: int = 2, b: float | None = None, m: int = 1,
                      n_mode: int = 1) -> dict:
    """3-D HEX20 FE solution with one orthotropic stiffness matrix per element.

    The through-thickness mesh is uniform with ``nz_per_ply`` elements inside each
    ply, so ply interfaces always fall on element faces.  Boundary conditions and
    the consistent sine load on the top face mirror
    :func:`tensorfem.plate_3d_exact.solve_hex20_ss_plate`.
    """
    import torch

    from .hex20 import hex20_shape_gradients, hex20_stiffness, structured_hex20_mesh

    a = float(a)
    b = float(b if b is not None else a)
    h = laminate.thickness
    nz = laminate.n_plies * int(nz_per_ply)
    nodes, cells = structured_hex20_mesh(a, b, h, int(nx), int(nx), nz)
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes
    # per-element ply index from the element centroid height
    cent_z = nodes[cells][:, :, 2].mean(dim=1).numpy() - 0.5 * h
    C = np.zeros((len(cells), 6, 6))
    for e in range(len(cells)):
        k = laminate.incident_ply(float(cent_z[e]))
        C[e] = to_fe_order(ply_stiffness_3d(laminate.material, laminate.angles_deg[k]))
    d = torch.from_numpy(C)

    ke, _ = hex20_stiffness(nodes[cells], d)
    ed = torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(len(cells), 60)
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    rows = ed[:, :, None].expand(-1, -1, 60).reshape(-1).numpy()
    cols = ed[:, None, :].expand(-1, 60, -1).reshape(-1).numpy()
    K = sp.coo_matrix((ke.reshape(-1).numpy(), (rows, cols)), shape=(ndof, ndof)).tocsr()

    # consistent load: q0 sin(alpha x) sin(beta y) on the top face (3x3 Gauss per face)
    al, be = m * math.pi / a, n_mode * math.pi / b
    forces = np.zeros(ndof)
    gp = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
    gw = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    top_face = (4, 5, 6, 7, 12, 13, 14, 15)
    eps = 1e-6
    for e, cell in enumerate(cells.tolist()):
        if abs(float(nodes[cell[4], 2]) - h) > 1e-12:
            continue
        xyz = nodes[cell]
        for gi, r in enumerate(gp):
            for gj, s in enumerate(gp):
                N, _ = hex20_shape_gradients(r, s, 1.0, like=xyz)
                N = N.detach()
                xg = float((N * xyz[:, 0]).sum())
                yg = float((N * xyz[:, 1]).sum())

                def pos(rr, ss):
                    Nn, _ = hex20_shape_gradients(rr, ss, 1.0, like=xyz)
                    Nn = Nn.detach()
                    return np.array([float((Nn * xyz[:, 0]).sum()), float((Nn * xyz[:, 1]).sum())])

                dr = (pos(r + eps, s) - pos(r - eps, s)) / (2 * eps)
                ds = (pos(r, s + eps) - pos(r, s - eps)) / (2 * eps)
                area = abs(dr[0] * ds[1] - dr[1] * ds[0])
                wgt = q0 * math.sin(al * xg) * math.sin(be * yg) * area * gw[gi] * gw[gj]
                for node in top_face:
                    forces[3 * cell[node] + 2] -= wgt * float(N[node])

    tol = 1e-9 * max(a, b)
    x_edge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - a).abs() < tol)
    y_edge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - b).abs() < tol)
    xi = torch.nonzero(x_edge).flatten()
    yi = torch.nonzero(y_edge).flatten()
    fixed = torch.cat((3 * xi + 1, 3 * xi + 2, 3 * yi, 3 * yi + 2)).unique().numpy()
    mask = np.ones(ndof, dtype=bool)
    mask[fixed] = False
    free = np.nonzero(mask)[0]
    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), forces[free])
    residual = float(np.linalg.norm(K @ u - forces) / max(np.linalg.norm(forces), 1e-300))

    # nodal stress recovery: average element stresses at shared nodes (batched)
    sig_sum = np.zeros((n_nodes, 6))
    sig_cnt = np.zeros(n_nodes)
    nat = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
           (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
           (0, -1, -1), (1, 0, -1), (0, 1, -1), (-1, 0, -1),
           (0, -1, 1), (1, 0, 1), (0, 1, 1), (-1, 0, 1),
           (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    ue_all = u[ed.numpy()]                                     # (ne, 60)
    ue_t = torch.from_numpy(ue_all)
    xyz_all = nodes[cells]                                     # (ne, 20, 3)
    for ln, (r, s_, t) in enumerate(nat):
        Nn, dNn = hex20_shape_gradients(float(r), float(s_), float(t), like=xyz_all)
        J = torch.einsum("eia,ib->eab", xyz_all, dNn)           # (ne,3,3)
        dN = torch.einsum("ib,ebc->eic", dNn, torch.linalg.inv(J))
        B = torch.zeros((len(cells), 6, 60), dtype=torch.float64)
        for i in range(20):
            B[:, 0, 3 * i] = dN[:, i, 0]
            B[:, 1, 3 * i + 1] = dN[:, i, 1]
            B[:, 2, 3 * i + 2] = dN[:, i, 2]
            B[:, 3, 3 * i] = dN[:, i, 1]
            B[:, 3, 3 * i + 1] = dN[:, i, 0]
            B[:, 4, 3 * i + 1] = dN[:, i, 2]
            B[:, 4, 3 * i + 2] = dN[:, i, 1]
            B[:, 5, 3 * i] = dN[:, i, 2]
            B[:, 5, 3 * i + 2] = dN[:, i, 0]
        eps = torch.einsum("eij,ej->ei", B, ue_t)               # (ne,6)
        sig = torch.einsum("eij,ej->ei", d, eps).numpy()
        nd = cells[:, ln].numpy()
        np.add.at(sig_sum, nd, sig)
        np.add.at(sig_cnt, nd, 1.0)

    sig_mean = np.where(sig_cnt[:, None] > 0,
                        sig_sum / np.maximum(sig_cnt, 1)[:, None], 0.0)
    xyz_np = nodes.numpy()
    cx = int(np.argmin((xyz_np[:, 0] - 0.5 * a) ** 2 + (xyz_np[:, 1] - 0.5 * b) ** 2
                       + (xyz_np[:, 2] - 0.5 * h) ** 2))
    w_center = float(-u[3 * cx + 2])

    def _column(x_target: float, y_target: float):
        sel = np.nonzero((np.abs(xyz_np[:, 0] - x_target) < tol)
                         & (np.abs(xyz_np[:, 1] - y_target) < tol))[0]
        zz = xyz_np[sel, 2] - 0.5 * h
        order = np.argsort(zz)
        return sel[order], zz[order]

    # tau_xz peaks on the edge x = 0 at y = b/2 ; tau_yz peaks at x = a/2, y = 0
    sel_x, z_x = _column(0.0, 0.5 * b)
    sel_y, z_y = _column(0.5 * a, 0.0)
    tau_xz = [(float(z), float(sig_mean[i, 5])) for z, i in zip(z_x, sel_x)]
    tau_yz = [(float(z), float(sig_mean[i, 4])) for z, i in zip(z_y, sel_y)]
    return {"w_center": w_center, "residual": residual,
            "n_dofs": int(ndof), "n_elements": int(len(cells)), "nz": int(nz),
            "nx": int(nx), "tau_xz_profile": tau_xz, "tau_yz_profile": tau_yz,
            "nodes": xyz_np, "u": u, "sigma_nodal": sig_mean}


# ── convergence helpers ─────────────────────────────────────────────────────
def convergence_order(errors: Sequence[float]) -> float:
    """Observed order ``p`` from a three-level halving study, ``p = log2(e1/e2)/log2(e2/e3)``."""
    e = [float(x) for x in errors]
    if len(e) < 3 or min(e) <= 0.0:
        raise ValueError("need three positive errors, finest last")
    return float(math.log2(e[0] / e[1]) / math.log2(e[1] / e[2]))


def _gci_from_errors(e_med: float, e_fine: float, p: float, ratio: float = 2.0,
                     safety: float = 1.25) -> dict:
    """GCI on the two finest levels given the observed order ``p``."""
    return {"p": p, "gci_fine": safety * e_fine / (ratio**p - 1.0),
            "gci_medium": safety * e_med / (ratio**p - 1.0)}


def richardson_extrapolate(values: Sequence[float], ratio: float = 2.0) -> dict:
    """Richardson extrapolation from three mesh levels (``values`` coarse -> fine)."""
    w1, w2, w3 = (float(v) for v in values[-3:])
    d1, d2 = w2 - w1, w3 - w2
    if d1 == 0.0 or d2 == 0.0 or d1 / d2 <= 0.0:
        return {"p": float("nan"), "extrapolated": w3, "uncertainty": abs(d2)}
    p = math.log(d1 / d2) / math.log(ratio)
    ext = w3 + d2 / (ratio**p - 1.0)
    return {"p": float(p), "extrapolated": float(ext), "uncertainty": float(abs(ext - w3))}


def gci_study(deflections: Sequence[float], reference: float, levels: int = 3,
              ratio: float = 2.0, safety: float = 1.25) -> dict:
    """Observed order and Roache GCI from a geometric mesh sequence.

    ``deflections`` runs coarse to fine; ``reference`` is the anchor value (an
    analytic solution, or a Richardson extrapolation of the same sequence).
    The order uses the three finest levels and the ratio ``ratio`` between
    successive levels.
    """
    w = [float(x) for x in deflections][-levels:]
    if len(w) < 3:
        raise ValueError("need at least three mesh levels")
    err = [abs(x - reference) / abs(reference) for x in w]
    p = convergence_order(err)
    return {"levels_used": levels, "ratio": ratio, "deflections": w,
            "errors": err, "p_observed": p,
            "gci_fine_pct": 100.0 * safety * err[-1] / (ratio**p - 1.0),
            "gci_medium_pct": 100.0 * safety * err[-2] / (ratio**p - 1.0)}


