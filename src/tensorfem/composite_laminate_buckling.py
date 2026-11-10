"""Composite laminate buckling: CLT + FSDT for symmetric orthotropic laminates.

Why this module exists
----------------------
The existing plate buckling modules (``plate_buckling_hermite``,
``marine_plate_buckling``, ``plate_bending_buckling``) cover isotropic and
orthotropic plates but not *composite laminates* whose rigidities are derived
from ply-level properties via Classical Laminate Theory (CLT).  This module
closes that gap and extends the plan-40 Kirchhoff isotropic verification into
the anisotropic composite domain (谱系继承: 各向同性 → 各向异性).

Formulation
-----------
1. **CLT ABD matrix**: For a symmetric laminate (B=0), the bending stiffness
   matrix D is computed from ply-level orthotropic properties (E1, E2, nu12,
   G12) transformed to global coordinates and integrated through thickness.

2. **Navier closed-form** (CLT/Kirchhoff): For a simply-supported symmetric
   special-orthotropic laminate under uniaxial compression Nx::

       Nx_cr = min_{m,n} [D11*(mπ/a)⁴ + 2(D12+2D66)*(mπ/a)²(nπ/b)²
                          + D22*(nπ/b)⁴] / (mπ/a)²

3. **FSDT correction** (Reissner-Mindlin): Transverse shear deformation is
   included via a 3×3 eigenvalue problem per (m,n) mode.  The shear stiffnesses
   As44, As55 are computed from the laminate layup with shear correction
   factor κ=5/6.

4. **Negative controls**:
   - Isotropic limit (E1=E2, ν12=ν21, G12=E/2(1+ν)) → isotropic plate result
   - Zero thickness → singular stiffness (raises ValueError)

Scope and conventions
---------------------
* Lengths in m, forces in N, moments in N·m.
* Nx > 0 is compression.
* Simply-supported on all four edges (SS1/Navier type).
* Symmetric laminates only (B=0, no bending-stretching coupling).
* D16/D26 bending-twisting coupling is neglected (special orthotropic assumption).

References
----------
* Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed. Taylor & Francis.
* Reddy, J.N. (2004). Mechanics of Laminated Composite Plates and Shells, 2nd ed.
* Timoshenko & Gere (1961). Theory of Elastic Stability, 2nd ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numpy.linalg import eigh

__all__ = [
    "PlyProperty",
    "LaminateLayup",
    "LaminateStiffness",
    "BucklingResult",
    "compute_ply_transformed_stiffness",
    "compute_laminate_abd",
    "navier_clt_buckling",
    "fsdt_buckling",
    "navier_isotropic_reference",
    "isotropic_limit_check",
    "parametric_ply_angle_sweep",
    "thick_plate_ratio_scan",
    "AS4_9310_E1",
    "AS4_9310_E2",
    "AS4_9310_NU12",
    "AS4_9310_G12",
]

# ── Typical carbon/epoxy properties (AS4/9310) ──────────────────────────────
AS4_9310_E1 = 148.0e9    # Pa, longitudinal modulus
AS4_9310_E2 = 10.5e9     # Pa, transverse modulus
AS4_9310_NU12 = 0.3      # major Poisson's ratio
AS4_9310_G12 = 5.6e9     # Pa, in-plane shear modulus


@dataclass(frozen=True)
class PlyProperty:
    """Orthotropic ply properties in material coordinates."""
    E1: float   # longitudinal modulus [Pa]
    E2: float   # transverse modulus [Pa]
    nu12: float  # major Poisson's ratio
    G12: float  # in-plane shear modulus [Pa]
    G23: float | None = None  # out-of-plane shear modulus [Pa]; default E2/2

    def __post_init__(self):
        for name in ("E1", "E2", "G12"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.nu12, (int, float)) or not (0 < self.nu12 < 0.5):
            raise ValueError("nu12 must be in (0, 0.5)")
        if self.G23 is not None and (not isinstance(self.G23, (int, float)) or self.G23 <= 0):
            raise ValueError("G23 must be positive if provided")

    @property
    def nu21(self) -> float:
        return self.nu12 * self.E2 / self.E1

    @property
    def g23(self) -> float:
        return self.G23 if self.G23 is not None else self.E2 / 2.0


@dataclass(frozen=True)
class LaminateLayup:
    """Symmetric laminate layup definition.

    Parameters
    ----------
    ply_props : PlyProperty
        Orthotropic ply properties (same for all plies).
    angles_deg : list of float
        Ply angles in degrees, top to bottom of half-layup.
        The full layup is the mirror: angles + reversed(angles).
        E.g., [0, 90] → [0, 90, 90, 0] for [0/90]s.
    ply_thickness : float
        Thickness of each ply [m].
    """
    ply_props: PlyProperty
    angles_deg: List[float]
    ply_thickness: float

    def __post_init__(self):
        if self.ply_thickness <= 0:
            raise ValueError("ply_thickness must be positive")
        if not self.angles_deg:
            raise ValueError("angles_deg must not be empty")

    @property
    def full_angles_deg(self) -> List[float]:
        """Full symmetric layup angles (top to bottom)."""
        return list(self.angles_deg) + list(reversed(self.angles_deg))

    @property
    def total_thickness(self) -> float:
        return len(self.full_angles_deg) * self.ply_thickness

    @property
    def n_plies(self) -> int:
        return len(self.full_angles_deg)


@dataclass(frozen=True)
class LaminateStiffness:
    """Laminate stiffness matrices (CLT).

    For a symmetric laminate, B=0 and we only need A and D.
    """
    A: np.ndarray   # (3,3) extensional stiffness [N/m]
    D: np.ndarray   # (3,3) bending stiffness [N·m]
    As: np.ndarray  # (2,2) shear stiffness [N/m]
    h: float        # total thickness [m]
    # Convenience: D components
    @property
    def D11(self) -> float: return self.D[0, 0]
    @property
    def D12(self) -> float: return self.D[0, 1]
    @property
    def D22(self) -> float: return self.D[1, 1]
    @property
    def D16(self) -> float: return self.D[0, 2]
    @property
    def D26(self) -> float: return self.D[1, 2]
    @property
    def D66(self) -> float: return self.D[2, 2]
    @property
    def As44(self) -> float: return self.As[1, 1]  # yz shear
    @property
    def As55(self) -> float: return self.As[0, 0]  # xz shear


def compute_ply_transformed_stiffness(
    ply: PlyProperty, theta_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute transformed reduced stiffness Q_bar and transformed shear Q_s_bar.

    Parameters
    ----------
    ply : PlyProperty
    theta_deg : float
        Ply angle in degrees (0 = fibers along x).

    Returns
    -------
    Q_bar : (3,3) ndarray
        Transformed reduced in-plane stiffness [Pa].
        Indices: 0=xx, 1=yy, 2=xy (engineering shear).
    Q_s_bar : (2,2) ndarray
        Transformed out-of-plane shear stiffness [Pa].
        Indices: 0=xz (55), 1=yz (44).
    """
    E1, E2, nu12, G12 = ply.E1, ply.E2, ply.nu12, ply.G12
    nu21 = ply.nu21
    denom = 1.0 - nu12 * nu21

    # Reduced stiffness in material coords
    Q11 = E1 / denom
    Q12 = nu12 * E2 / denom
    Q22 = E2 / denom
    Q66 = G12

    theta = math.radians(theta_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    c2 = c * c
    s2 = s * s
    c3 = c2 * c
    s3 = s2 * s
    c4 = c2 * c2
    s4 = s2 * s2

    # Transformed reduced stiffness (Jones eq. 2.35)
    Q_bar = np.zeros((3, 3))
    Q_bar[0, 0] = Q11 * c4 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * s4
    Q_bar[1, 1] = Q11 * s4 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * c4
    Q_bar[0, 1] = Q_bar[1, 0] = (Q11 + Q22 - 4 * Q66) * s2 * c2 + Q12 * (s4 + c4)
    Q_bar[0, 2] = Q_bar[2, 0] = (Q11 - Q12 - 2 * Q66) * c3 * s - (Q22 - Q12 - 2 * Q66) * c * s3
    Q_bar[1, 2] = Q_bar[2, 1] = (Q11 - Q12 - 2 * Q66) * c * s3 - (Q22 - Q12 - 2 * Q66) * c3 * s
    Q_bar[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s2 * c2 + Q66 * (s4 + c4)

    # Transformed shear stiffness
    G23 = ply.g23
    G13 = G12  # approximation: G13 ≈ G12
    Q_s_bar = np.zeros((2, 2))
    Q_s_bar[0, 0] = G13 * c2 + G23 * s2   # xz (55)
    Q_s_bar[1, 1] = G13 * s2 + G23 * c2   # yz (44)
    Q_s_bar[0, 1] = Q_s_bar[1, 0] = (G13 - G23) * c * s  # 45 coupling

    return Q_bar, Q_s_bar


def compute_laminate_abd(layup: LaminateLayup) -> LaminateStiffness:
    """Compute CLT ABD matrices for a symmetric laminate.

    For symmetric laminates, B=0. We compute A, D, and As.
    """
    ply = layup.ply_props
    n_full = layup.n_plies
    tp = layup.ply_thickness
    h = layup.total_thickness
    angles = layup.full_angles_deg

    # Interface z-coordinates (mid-plane at z=0)
    z = np.zeros(n_full + 1)
    z[0] = -h / 2.0
    for k in range(n_full):
        z[k + 1] = z[k] + tp

    A = np.zeros((3, 3))
    D = np.zeros((3, 3))
    As = np.zeros((2, 2))

    for k in range(n_full):
        Q_bar, Q_s_bar = compute_ply_transformed_stiffness(ply, angles[k])
        zk1, zk0 = z[k + 1], z[k]

        # A = sum Q_bar * (z_k - z_{k-1})
        A += Q_bar * (zk1 - zk0)

        # D = sum Q_bar * (z_k^3 - z_{k-1}^3) / 3
        D += Q_bar * (zk1**3 - zk0**3) / 3.0

        # As = kappa * sum Q_s_bar * (z_k - z_{k-1})
        As += Q_s_bar * (zk1 - zk0)

    # Shear correction factor (5/6 for homogeneous, used as approximation)
    kappa = 5.0 / 6.0
    As *= kappa

    return LaminateStiffness(A=A, D=D, As=As, h=h)


@dataclass(frozen=True)
class BucklingResult:
    """Buckling result for a laminate plate."""
    N_cr: float           # critical line load [N/m]
    m_cr: int             # longitudinal half-waves
    n_cr: int             # transverse half-waves
    a: float              # plate length [m]
    b: float              # plate width [m]
    theory: str           # "CLT" or "FSDT"


def navier_clt_buckling(
    stiff: LaminateStiffness,
    a: float,
    b: float,
    m_max: int = 30,
    n_max: int = 30,
) -> BucklingResult:
    """Navier closed-form buckling load for symmetric laminate (CLT/Kirchhoff).

    For a simply-supported symmetric special-orthotropic laminate
    (D16=D26=0 or neglected) under uniaxial compression Nx::

        Nx_cr = min_{m,n} [D11 α⁴ + 2(D12+2D66) α² β² + D22 β⁴] / α²

    where α = mπ/a, β = nπ/b.

    Parameters
    ----------
    stiff : LaminateStiffness
    a, b : float
        Plate dimensions [m] (a = length in x, b = width in y).
    m_max, n_max : int
        Maximum half-wave numbers to search.

    Returns
    -------
    BucklingResult
    """
    D11, D12, D22, D66 = stiff.D11, stiff.D12, stiff.D22, stiff.D66

    best_Ncr = float("inf")
    best_m, best_n = 1, 1

    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            alpha = m * math.pi / a
            beta = n * math.pi / b
            alpha2 = alpha * alpha
            beta2 = beta * beta

            numerator = (D11 * alpha2**2
                         + 2.0 * (D12 + 2.0 * D66) * alpha2 * beta2
                         + D22 * beta2**2)
            N_cr_mn = numerator / alpha2

            if N_cr_mn < best_Ncr:
                best_Ncr = N_cr_mn
                best_m, best_n = m, n

    return BucklingResult(
        N_cr=best_Ncr, m_cr=best_m, n_cr=best_n,
        a=a, b=b, theory="CLT"
    )


def fsdt_buckling(
    stiff: LaminateStiffness,
    a: float,
    b: float,
    m_max: int = 30,
    n_max: int = 30,
) -> BucklingResult:
    """FSDT (Reissner-Mindlin) buckling for symmetric laminate.

    Uses Schur complement to solve the generalized eigenvalue problem
    that includes transverse shear deformation.

    Parameters
    ----------
    stiff : LaminateStiffness
    a, b : float
        Plate dimensions [m].
    m_max, n_max : int
        Maximum half-wave numbers.

    Returns
    -------
    BucklingResult
    """
    D11, D12, D22, D66 = stiff.D11, stiff.D12, stiff.D22, stiff.D66
    As55, As44 = stiff.As55, stiff.As44

    best_Ncr = float("inf")
    best_m, best_n = 1, 1

    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            alpha = m * math.pi / a
            beta = n * math.pi / b
            a2 = alpha * alpha
            b2 = beta * beta
            ab = alpha * beta

            # Stiffness matrix A = K_bend + K_shear
            A = np.zeros((3, 3))
            A[0, 0] = D11 * a2 + D66 * b2 + As55
            A[1, 1] = D66 * a2 + D22 * b2 + As44
            A[0, 1] = A[1, 0] = (D12 + D66) * ab
            A[0, 2] = A[2, 0] = As55 * alpha
            A[1, 2] = A[2, 1] = As44 * beta
            A[2, 2] = As55 * a2 + As44 * b2

            # Geometric stiffness B = diag(0, 0, α²)
            # Eigenvalue problem: A * u = Nx * B * u
            # Using Schur complement:
            # Nx = (A[2,2] - A[2,0:2] * A[0:2,0:2]^{-1} * A[0:2,2]) / α²

            A_bb = A[:2, :2]
            A_bw = A[:2, 2]
            A_wb = A[2, :2]

            try:
                phi = np.linalg.solve(A_bb, A_bw)
                Nx_mn = (A[2, 2] - np.dot(A_wb, phi)) / a2
            except np.linalg.LinAlgError:
                continue

            if Nx_mn > 0 and Nx_mn < best_Ncr:
                best_Ncr = Nx_mn
                best_m, best_n = m, n

    return BucklingResult(
        N_cr=best_Ncr, m_cr=best_m, n_cr=best_n,
        a=a, b=b, theory="FSDT"
    )


def navier_isotropic_reference(
    E: float, nu: float, h: float, a: float, b: float,
    m_max: int = 30, n_max: int = 30,
) -> BucklingResult:
    """Analytical Navier buckling for isotropic simply-supported plate.

    Nx_cr = min_{m,n} D * [((m/a)² + (n/b)²)² / (m/a)²] * π²
    where D = E*h³/(12*(1-ν²))
    """
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    best_Ncr = float("inf")
    best_m, best_n = 1, 1

    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            am = m / a
            bn = n / b
            N_cr_mn = D * math.pi**2 * (am**2 + bn**2)**2 / am**2
            if N_cr_mn < best_Ncr:
                best_Ncr = N_cr_mn
                best_m, best_n = m, n

    return BucklingResult(
        N_cr=best_Ncr, m_cr=best_m, n_cr=best_n,
        a=a, b=b, theory="CLT-isotropic"
    )


def isotropic_limit_check(
    E: float = 210e9, nu: float = 0.3, h: float = 0.01,
    a: float = 1.0, b: float = 1.0,
    tol: float = 0.01,
) -> dict:
    """Negative control: isotropic limit of composite laminate.

    Set E1=E2=E, nu12=nu21=nu, G12=E/(2(1+nu)) and verify that
    the CLT laminate buckling matches the isotropic Navier solution.

    Returns dict with both results and relative error.
    """
    G = E / (2.0 * (1.0 + nu))
    ply = PlyProperty(E1=E, E2=E, nu12=nu, G12=G, G23=G)
    # Single-ply "laminate" - use full layup directly to avoid doubling
    # We create a 2-ply symmetric layup [0/0] with total thickness h
    layup = LaminateLayup(ply_props=ply, angles_deg=[0.0], ply_thickness=h / 2.0)
    stiff = compute_laminate_abd(layup)

    res_clt = navier_clt_buckling(stiff, a, b)
    ref = navier_isotropic_reference(E, nu, h, a, b)

    rel_err = abs(res_clt.N_cr - ref.N_cr) / ref.N_cr

    return {
        "composite_clt_Ncr": res_clt.N_cr,
        "isotropic_navier_Ncr": ref.N_cr,
        "relative_error": rel_err,
        "pass": rel_err < tol,
        "tolerance": tol,
    }


def parametric_ply_angle_sweep(
    ply: PlyProperty,
    layup_configs: List[Tuple[str, List[float]]],
    a: float,
    b: float,
    ply_thickness: float = 0.125e-3,
    angles_scan: List[float] | None = None,
) -> dict:
    """Parametric sweep of ply angle for multiple layup configurations.

    Parameters
    ----------
    ply : PlyProperty
    layup_configs : list of (name, half_angles)
        E.g., [("0/90_s", [0, 90]), ("±45_2s", [45, -45]), ...]
    a, b : float
        Plate dimensions [m].
    ply_thickness : float
        Individual ply thickness [m].
    angles_scan : list of float, optional
        Angles to sweep for parametric curve. Default: 0 to 90 in 5° steps.

    Returns
    -------
    dict with sweep results.
    """
    if angles_scan is None:
        angles_scan = list(range(0, 91, 5))

    results = {}

    # Fixed layup comparison
    for name, half_angles in layup_configs:
        layup = LaminateLayup(ply_props=ply, angles_deg=half_angles,
                              ply_thickness=ply_thickness)
        stiff = compute_laminate_abd(layup)
        res_clt = navier_clt_buckling(stiff, a, b)
        res_fsd = fsdt_buckling(stiff, a, b)
        results[name] = {
            "N_cr_CLT": res_clt.N_cr,
            "N_cr_FSDT": res_fsd.N_cr,
            "m_cr": res_clt.m_cr,
            "n_cr": res_clt.n_cr,
            "ratio_FSDT_CLT": res_fsd.N_cr / res_clt.N_cr if res_clt.N_cr > 0 else float("nan"),
            "D11": stiff.D11,
            "D22": stiff.D22,
            "D66": stiff.D66,
            "h": stiff.h,
        }

    # Parametric angle sweep: [θ/-θ]s layup
    sweep_angles = []
    sweep_Ncr_CLT = []
    sweep_Ncr_FSDT = []
    for theta in angles_scan:
        half_angles = [theta, -theta]
        layup = LaminateLayup(ply_props=ply, angles_deg=half_angles,
                              ply_thickness=ply_thickness)
        stiff = compute_laminate_abd(layup)
        res_clt = navier_clt_buckling(stiff, a, b)
        res_fsd = fsdt_buckling(stiff, a, b)
        sweep_angles.append(theta)
        sweep_Ncr_CLT.append(res_clt.N_cr)
        sweep_Ncr_FSDT.append(res_fsd.N_cr)

    results["_sweep"] = {
        "angles": sweep_angles,
        "N_cr_CLT": sweep_Ncr_CLT,
        "N_cr_FSDT": sweep_Ncr_FSDT,
    }

    return results


def thick_plate_ratio_scan(
    ply: PlyProperty,
    half_angles: List[float],
    a: float,
    b: float,
    h_over_a_range: List[float] | None = None,
    ply_thickness_base: float = 0.125e-3,
) -> dict:
    """Scan FSDT/CLT ratio vs. thickness-to-length ratio h/a.

    For thick plates, FSDT should give lower N_cr than CLT due to
    transverse shear flexibility.

    Returns dict with h/a values and ratios.
    """
    if h_over_a_range is None:
        h_over_a_range = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

    ha_values = []
    ratios = []
    Ncr_CLT_list = []
    Ncr_FSDT_list = []

    for ha in h_over_a_range:
        h = ha * a
        n_plies = max(4, int(round(h / ply_thickness_base)))
        # Ensure even number for symmetric layup
        if n_plies % 2 != 0:
            n_plies += 1
        tp = h / n_plies

        # Build symmetric layup by repeating the half_angles pattern
        n_half = n_plies // 2
        # Repeat half_angles to fill n_half plies
        reps = (n_half + len(half_angles) - 1) // len(half_angles)
        full_half = (half_angles * reps)[:n_half]

        layup = LaminateLayup(ply_props=ply, angles_deg=full_half,
                              ply_thickness=tp)
        stiff = compute_laminate_abd(layup)

        res_clt = navier_clt_buckling(stiff, a, b)
        res_fsd = fsdt_buckling(stiff, a, b)

        ha_values.append(ha)
        Ncr_CLT_list.append(res_clt.N_cr)
        Ncr_FSDT_list.append(res_fsd.N_cr)
        ratios.append(res_fsd.N_cr / res_clt.N_cr if res_clt.N_cr > 0 else float("nan"))

    return {
        "h_over_a": ha_values,
        "N_cr_CLT": Ncr_CLT_list,
        "N_cr_FSDT": Ncr_FSDT_list,
        "ratio_FSDT_CLT": ratios,
    }
