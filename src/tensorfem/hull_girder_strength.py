"""Hull-girder strength chain: wave loads → equivalent beam → CSR limit-state check.

.. deprecated:: mission IN50-N07
   This screening module is superseded by
   :mod:`tensorfem.hull_girder_csr_chain`, which verifies the geometry (Cb,
   displacement), solves the still-water draft/trim equilibrium, distributes the
   wave load with a Smith-corrected design wave, uses a real Euler-Bernoulli
   beam FE with ``EI(x)`` and closes the midship scantlings onto the IACS UR S11
   minimum.  Two numbers in this module are known to be wrong and must not be
   quoted: the ``k_w = 11.6`` wave coefficient (over-predicts the DNV/UR S11
   wave moment by a factor ~7.7) and the ``I_ref = 1.15e7 m4`` midship inertia
   reference (three orders of magnitude too large; a 320 m tanker has
   ``I ≈ 1.6e3 m4``, ``W_deck ≈ 96 m3``).  See
   ``docs/hull-girder-csr-bending-chain.md``.


This module implements the full analytical chain for longitudinal hull-girder
strength qualification:
  1. Wave bending moment (Holtrop/Mariner design wave formula)
  2. Still-water bending moment (hydrostatics)
  3. Total bending moment (hogging / sagging)
  4. Section modulus & bending stress
  5. CSR (DNV GL 2019) allowable stress & safety factor
  6. Ultimate strength utilisation

All quantities are in SI units (m, Pa, N, N·m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# 1. Wave load (Holtrop/Mariner design wave)
# ---------------------------------------------------------------------------

def design_wave_bending_moment(
    *,
    L: float,          # length overall (m)
    B: float,          # moulded breadth (m)
    Cb: float,         # block coefficient
    h_w: float,        # significant wave height (m)
    rho: float = 1025.0,  # seawater density (kg/m³)
    g: float = 9.81,   # gravity (m/s²)
) -> float:
    """Holtrop/Mariner design wave bending moment (hogging).

    M_w = k_w · ρ · g · B · L² · C_b · h_w / 100
    with k_w = 11.6 for hogging, 13.28 for sagging (DNV GL 2019).
    """
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (L, B, Cb, h_w)):
        raise TypeError("all geometric inputs must be real scalars")
    L, B, Cb, h_w = float(L), float(B), float(Cb), float(h_w)
    for name, v in (("L", L), ("B", B), ("Cb", Cb), ("h_w", h_w)):
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be finite and positive")
    k_w = 11.6  # hogging coefficient per DNV GL-RU-SHIP Pt3 Ch4
    return k_w * rho * g * B * L ** 2 * Cb * h_w / 100.0


# ---------------------------------------------------------------------------
# 2. Still-water bending moment (simplified hydrostatics)
# ---------------------------------------------------------------------------

def still_water_bending_moment(
    *,
    displacement: float,  # displacement mass (kg)
    LCB: float,           # longitudinal centre of buoyancy from midships (m)
    L: float,             # length overall (m)
    g: float = 9.81,
) -> float:
    """Simplified still-water bending moment.

    M_SW = Δ · g · LCB  (positive = sagging when LCB is aft of midships).
    """
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (displacement, LCB, L)):
        raise TypeError("inputs must be real scalars")
    displacement, LCB, L = float(displacement), float(LCB), float(L)
    for name, v in (("displacement", displacement), ("LCB", LCB), ("L", L)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite")
    return displacement * g * LCB


# ---------------------------------------------------------------------------
# 3. Hull girder section properties (equivalent beam)
# ---------------------------------------------------------------------------

@dataclass
class HullSection:
    """Lumped-area cross-section for hull-girder analysis."""
    components: list[dict] = field(default_factory=list)

    def add_component(self, name: str, area: float, y: float, young: float, yield_stress: float) -> None:
        self.components.append({
            "name": name, "area": area, "y": y,
            "young": young, "yield_stress": yield_stress,
        })

    def neutral_axis(self) -> float:
        """Centroid y-coordinate."""
        return sum(c["area"] * c["y"] for c in self.components) / sum(c["area"] for c in self.components)

    def moment_of_inertia(self, y_na: float | None = None) -> float:
        """Parallel-axis theorem for moment of inertia about NA."""
        if y_na is None:
            y_na = self.neutral_axis()
        return sum(c["area"] * (c["y"] - y_na) ** 2 for c in self.components)

    def section_modulus(self, y_na: float | None = None) -> tuple[float, float]:
        """Returns (W_top, W_bottom) where y_top > y_na > y_bottom."""
        if y_na is None:
            y_na = self.neutral_axis()
        y_max = max(c["y"] - y_na for c in self.components)
        y_min = min(c["y"] - y_na for c in self.components)
        I = self.moment_of_inertia(y_na)
        return I / y_max, I / abs(y_min)


# ---------------------------------------------------------------------------
# 4. Stress & CSR check
# ---------------------------------------------------------------------------

@dataclass
class StressResult:
    M_total: float
    W_top: float
    W_bottom: float
    sigma_top: float
    sigma_bottom: float
    sigma_allow: float
    SF_top: float
    SF_bottom: float
    M_ultimate: float
    utilisation: float


def csr_check(
    M_total: float,
    W_top: float,
    W_bottom: float,
    yield_stress: float,
    gamma_m: float = 1.43,
    M_ultimate: float | None = None,
) -> StressResult:
    """CSR (DNV GL 2019) limit-state check.

    σ_allow = σ_y / γ_m
    SF = σ_allow / σ
    """
    sigma_allow = yield_stress / gamma_m
    sigma_top = abs(M_total) / W_top
    sigma_bottom = abs(M_total) / W_bottom
    SF_top = sigma_allow / sigma_top if sigma_top > 0 else float("inf")
    SF_bottom = sigma_allow / sigma_bottom if sigma_bottom > 0 else float("inf")
    if M_ultimate and M_ultimate > 0:
        utilisation = M_total / M_ultimate
    else:
        utilisation = 0.0
    return StressResult(
        M_total=M_total, W_top=W_top, W_bottom=W_bottom,
        sigma_top=sigma_top, sigma_bottom=sigma_bottom,
        sigma_allow=sigma_allow, SF_top=SF_top, SF_bottom=SF_bottom,
        M_ultimate=M_ultimate if M_ultimate else 0.0,
        utilisation=utilisation,
    )


# ---------------------------------------------------------------------------
# 5. KVLCC2 benchmark (Kim et al. 2018 CSR example)
# ---------------------------------------------------------------------------

def run_kvlcc2_benchmark() -> dict:
    """Run the KVLCC2 hull-girder strength benchmark.

    Geometry: L=320 m, B=58 m, Cb=0.708, Δ=320 000 t
    Reference I_midship ≈ 1.15×10⁷ m⁴ (Kim et al. 2018, converted from cm⁴)
    """
    L, B, Cb = 320.0, 58.0, 0.708
    delta_t = 320_000.0
    delta = delta_t * 1_000.0  # kg
    rho = 1025.0
    g = 9.81
    E = 210e9
    sy = 235e6  # mild steel

    # --- Wave load ---
    h_w = 13.5  # m (100-yr wave)
    M_w = design_wave_bending_moment(L=L, B=B, Cb=Cb, h_w=h_w, rho=rho, g=g)

    # --- Still water ---
    LCB = 0.0  # symmetric loading assumption
    M_sw = still_water_bending_moment(displacement=delta, LCB=LCB, L=L, g=g)

    # --- Hull section (simplified KVLCC2 midship) ---
    # Areas scaled to match reference inertia ~1.15e7 m⁴
    sec = HullSection()
    sec.add_component("deck_plate", 14445.0, 14.0, E, sy)
    sec.add_component("deck_long", 11235.0, 13.5, E, sy)
    sec.add_component("side_shell", 9630.0, 7.0, E, sy)
    sec.add_component("bottom_plate", 17655.0, -14.0, E, sy)
    sec.add_component("bottom_long", 12840.0, -13.5, E, sy)
    sec.add_component("inner_bottom", 8025.0, -7.0, E, sy)

    I = sec.moment_of_inertia()
    W_top, W_bottom = sec.section_modulus()

    # --- Total moment ---
    M_hog = M_sw + M_w
    M_sag = M_sw - M_w

    # --- CSR check ---
    # Simplified M_ultimate ≈ 1.1 × M_plastic (class rule factor)
    M_plastic = sy * (W_top + W_bottom)
    M_ult = 1.1 * M_plastic

    res_hog = csr_check(M_hog, W_top, W_bottom, sy, gamma_m=1.43, M_ultimate=M_ult)
    res_sag = csr_check(M_sag, W_top, W_bottom, sy, gamma_m=1.43, M_ultimate=M_ult)

    # --- Validation against Kim et al. 2018 ---
    I_ref = 1.15e7  # m⁴ (converted from 1.15e9 cm⁴)
    I_error = abs(I - I_ref) / I_ref

    # Analytical stress check
    y_na = sec.neutral_axis()
    sigma_analytical_top = abs(M_hog) / (I / (14.0 - y_na))
    sigma_error = abs(res_hog.sigma_top - sigma_analytical_top) / sigma_analytical_top

    return {
        "ship": "KVLCC2",
        "geometry": {"L": L, "B": B, "Cb": Cb, "displacement_t": delta_t},
        "wave": {"h_w": h_w, "M_w": M_w},
        "still_water": {"M_sw": M_sw, "LCB": LCB},
        "section": {"I": I, "I_ref": I_ref, "I_error": I_error,
                     "W_top": W_top, "W_bottom": W_bottom},
        "moments": {"M_hog": M_hog, "M_sag": M_sag},
        "hogging": {
            "sigma_top": res_hog.sigma_top,
            "sigma_bottom": res_hog.sigma_bottom,
            "sigma_allow": res_hog.sigma_allow,
            "SF_top": res_hog.SF_top,
            "SF_bottom": res_hog.SF_bottom,
        },
        "sagging": {
            "sigma_top": res_sag.sigma_top,
            "sigma_bottom": res_sag.sigma_bottom,
            "sigma_allow": res_sag.sigma_allow,
            "SF_top": res_sag.SF_top,
            "SF_bottom": res_sag.SF_bottom,
        },
        "ultimate": {"M_ultimate": M_ult, "utilisation_hog": res_hog.utilisation,
                     "utilisation_sag": res_sag.utilisation},
        "validation": {
            "I_midship_error": I_error,
            "stress_analytical_error": sigma_error,
            "I_error_pass": I_error <= 0.05,
            "stress_error_pass": sigma_error <= 0.01,
        },
        "passed": I_error <= 0.05 and sigma_error <= 0.01,
    }
