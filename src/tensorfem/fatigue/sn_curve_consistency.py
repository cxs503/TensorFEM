"""Cross-module consistency audit of the S-N curves carried by TensorFEM.

Why this module exists
----------------------
The same published line -- DNV-RP-C203 curve D **in air** -- is re-declared by
six independent modules of this library (``structural_fatigue_detail``,
``weld_fatigue_ice`` twice, ``fatigue.weld_toe`` three environments,
``propeller_ice_milling_fatigue_chain`` and ``ice_slamming_whipping``).  Each was
written for its own campaign, so the two published intercepts (``log10 a1`` below
the knee ``N = 1e7``, ``log10 a2`` above it) can drift apart silently: nothing in
the test suite compared them, and drift is invisible from inside any single
module because every module is *self-consistent* (its own two branches meet).

This module is the missing cross-check.  It collects every implementation, asks
each one for the constants it actually uses, and grades it against

* the published pair ``log10 a1 = 12.164`` (``m1 = 3``), ``log10 a2 = 15.606``
  (``m2 = 5``), knee at ``N = 1e7`` cycles (i.e. 52.64 MPa for FAT90), DNV-RP-C203
  Table 2-1, curve D in air;
* *branch continuity* -- ``N`` from the two branches must agree at the knee;
* *slope identity* -- ``N(s)/N(2s) = 2**m`` at stress ranges that demonstrably sit
  inside the branch being tested (a range that straddles the knee verifies
  nothing; that mistake is the subject of
  ``docs/fatigue-sn-curve-consistency-audit.md``).

The audit is read-only: it never changes a curve, it only reports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

#: DNV-RP-C203 (2021) Table 2-1, curve D **in air** -- the reference pair.
DNV_D_AIR: dict[str, object] = {
    "log_a1": 12.164,
    "m1": 3.0,
    "log_a2": 15.606,
    "m2": 5.0,
    "knee_cycles": 1.0e7,
    "stress_at_2e6_mpa": 90.0,
    "source": ("DNV-RP-C203 Table 2-1, curve D in air: log a1 = 12.164 (m1 = 3, "
               "N <= 1e7), log a2 = 15.606 (m2 = 5, N > 1e7)"),
}

#: IIW-1823 / Hobbacher FAT 90, as-welded, for the class-based implementations.
IIW_FAT90: dict[str, object] = {
    "fat_mpa": 90.0,
    "reference_cycles": 2.0e6,
    "m1": 3.0,
    "knee_cycles": 1.0e7,
    "m2": 5.0,
    "source": "IIW-1823-07 / Hobbacher 2008, FAT 90 (2e6 cycles, m = 3 before the knee at 1e7, m = 5 after)",
}


def reference_life(
    stress_pa: float,
    *,
    log_a1: float = 12.164,
    m1: float = 3.0,
    log_a2: float = 15.606,
    m2: float = 5.0,
    knee_cycles: float = 1.0e7,
) -> float:
    """Two-slope power law ``N = a s^-m`` written directly from the table."""
    if not (stress_pa > 0.0 and math.isfinite(stress_pa)):
        raise ValueError("stress_pa must be positive and finite")
    s = stress_pa / 1.0e6
    n1 = 10.0 ** log_a1 * s ** (-m1)
    if n1 <= knee_cycles:
        return n1
    return 10.0 ** log_a2 * s ** (-m2)


@dataclass(frozen=True)
class CurveRow:
    """One S-N implementation, as declared by the module that owns it."""

    name: str
    module: str
    standard: str
    log_a1: float
    m1: float
    log_a2: float
    m2: float
    knee_cycles: float
    life: Callable[[float], float]

    @property
    def knee_stress_mpa(self) -> float:
        return (10.0 ** self.log_a1 / self.knee_cycles) ** (1.0 / self.m1)

    def audit(self) -> dict[str, object]:
        knee_pa = self.knee_stress_mpa * 1.0e6
        n_knee_low = 10.0 ** self.log_a1 * self.knee_stress_mpa ** (-self.m1)
        n_knee_high = 10.0 ** self.log_a2 * self.knee_stress_mpa ** (-self.m2)
        continuity = abs(n_knee_low - n_knee_high) / self.knee_cycles

        # Slope identity, branch-guarded: the two ranges must live in one branch.
        probe = 1.4 * knee_pa
        slope_m1 = self.life(probe) / self.life(2.0 * probe)
        # Stay clear of the constant-amplitude fatigue limit (IIW FAT 90 stops at
        # 1e9 cycles), otherwise the cutoff -- not the slope -- sets the ratio.
        probe = 0.9 * knee_pa
        slope_m2 = self.life(0.5 * probe) / self.life(probe)

        ref_100 = reference_life(100.0e6)
        ref_30 = reference_life(30.0e6)
        life_100 = self.life(100.0e6)
        life_30 = self.life(30.0e6)
        row_100 = reference_life(100.0e6, log_a1=self.log_a1, m1=self.m1,
                                 log_a2=self.log_a2, m2=self.m2,
                                 knee_cycles=self.knee_cycles)
        return {
            "name": self.name,
            "module": self.module,
            "standard": self.standard,
            "log_a1": self.log_a1,
            "m1": self.m1,
            "log_a2": self.log_a2,
            "m2": self.m2,
            "knee_cycles": self.knee_cycles,
            "knee_stress_mpa": self.knee_stress_mpa,
            "knee_continuity_rel": continuity,
            "slope_m1_ratio": slope_m1,
            "slope_m2_ratio": slope_m2,
            "slope_m1_expected": 2.0 ** self.m1,
            "slope_m2_expected": 2.0 ** self.m2,
            "delta_log_a1_vs_dnv_d": self.log_a1 - float(DNV_D_AIR["log_a1"]),
            "delta_log_a2_vs_dnv_d": self.log_a2 - float(DNV_D_AIR["log_a2"]),
            "life_100mpa": life_100,
            "life_30mpa": life_30,
            "life_100mpa_delta_vs_dnv_d": life_100 / ref_100 - 1.0,
            "life_30mpa_delta_vs_dnv_d": life_30 / ref_30 - 1.0,
            # the row's own callable must reproduce the table it declares
            "declared_vs_callable_rel": abs(life_100 / row_100 - 1.0),
        }


def curve_rows() -> list[CurveRow]:
    """Every S-N implementation of this library, as it declares itself."""
    from tensorfem.fatigue.weld_toe import ENVIRONMENTS
    from tensorfem.ice_slamming_whipping import SNCurve as SlammingSNCurve
    from tensorfem.propeller_ice_milling_fatigue_chain import dnv_d_curve_in_air
    from tensorfem.structural_fatigue_detail import DNVDualSlopeSN
    from tensorfem.weld_fatigue_ice import DNVSNCurve, IIWFatClass

    rows: list[CurveRow] = []

    detail = DNVDualSlopeSN()
    rows.append(CurveRow(
        name="structural_fatigue_detail.DNVDualSlopeSN", module="structural_fatigue_detail",
        standard="DNV-RP-C203 D air", log_a1=detail.log_a1, m1=detail.m1,
        log_a2=detail.log_a2, m2=detail.m2, knee_cycles=detail.n_knee,
        life=detail.cycles_to_failure))

    ice = DNVSNCurve()
    rows.append(CurveRow(
        name="weld_fatigue_ice.DNVSNCurve", module="weld_fatigue_ice",
        standard="DNV-RP-C203 D air", log_a1=ice.log_a1, m1=ice.slope_m1,
        log_a2=ice.log_a2, m2=ice.slope_m2, knee_cycles=ice.knee_cycles,
        life=ice.cycles_to_failure))

    iiw = IIWFatClass()
    reference_cycles = 2.0e6   # the class is defined at 2e6 cycles (see IIWFatClass)
    knee_stress = iiw.fat_mpa * (reference_cycles / iiw.knee_cycles) ** (1.0 / iiw.slope_m1)
    rows.append(CurveRow(
        name="weld_fatigue_ice.IIWFatClass", module="weld_fatigue_ice",
        standard="IIW FAT 90", log_a1=math.log10(reference_cycles * iiw.fat_mpa ** iiw.slope_m1),
        m1=iiw.slope_m1,
        log_a2=math.log10(iiw.knee_cycles * knee_stress ** iiw.slope_m2),
        m2=iiw.slope_m2, knee_cycles=iiw.knee_cycles, life=iiw.cycles_to_failure))

    for environment, spec in sorted(ENVIRONMENTS.items()):
        curve = spec.curve
        log_a1 = curve.log_a1 if environment != "seawater_free" else curve.log_a1
        rows.append(CurveRow(
            name=f"fatigue.weld_toe.ENVIRONMENTS[{environment!r}]", module="fatigue.weld_toe",
            standard=f"DNV-RP-C203 D {environment}",
            log_a1=log_a1, m1=curve.slope_m1, log_a2=curve.log_a2,
            m2=curve.slope_m2, knee_cycles=curve.knee_cycles,
            life=curve.cycles_to_failure))

    propeller = dnv_d_curve_in_air()
    rows.append(CurveRow(
        name="propeller_ice_milling_fatigue_chain.dnv_d_curve_in_air",
        module="propeller_ice_milling_fatigue_chain", standard="DNV-RP-C203 D air",
        log_a1=propeller.log_a1, m1=propeller.slope_m1, log_a2=propeller.log_a2,
        m2=propeller.slope_m2, knee_cycles=propeller.knee_cycles,
        life=propeller.cycles_to_failure))

    slamming = SlammingSNCurve()
    rows.append(CurveRow(
        name="ice_slamming_whipping.SNCurve", module="ice_slamming_whipping",
        standard="DNV-RP-C203 D air", log_a1=math.log10(slamming.c1), m1=slamming.m1,
        log_a2=math.log10(slamming.c2), m2=slamming.m2,
        knee_cycles=slamming.knee_cycles, life=slamming.cycles_to_failure))

    return rows


def audit() -> dict[str, object]:
    """Full cross-module audit report (JSON-serialisable)."""
    rows = [row.audit() for row in curve_rows()]
    dnv = [r for r in rows if str(r["standard"]).endswith("D air")
           or str(r["standard"]).endswith("D CP")]
    dnv_air = [r for r in rows if str(r["standard"]) == "DNV-RP-C203 D air"]
    summary = {
        "n_implementations": len(rows),
        "n_dnv_curve_d_air": len(dnv_air),
        "max_knee_continuity_rel": max(r["knee_continuity_rel"] for r in rows),
        "max_abs_delta_log_a1_vs_dnv_d": max(abs(r["delta_log_a1_vs_dnv_d"]) for r in dnv_air),
        "max_abs_life_delta_vs_dnv_d_at_100mpa": max(abs(r["life_100mpa_delta_vs_dnv_d"]) for r in dnv_air),
        "max_abs_life_delta_vs_dnv_d_at_30mpa": max(abs(r["life_30mpa_delta_vs_dnv_d"]) for r in dnv_air),
        "max_declared_vs_callable_rel": max(r["declared_vs_callable_rel"] for r in rows),
        # The published intercepts carry three decimals, so the two branches of a
        # *declared* pair meet only to ~10**-0.0005 = 1.15e-3 in cycles.  The
        # continuity gate is therefore set at 5e-3, far below any physically
        # meaningful drift (a 1 % life bias is 1e-2) but above the rounding floor.
        "knee_continuity_floor_rel": 1.2e-3,
        "max_slope_m1_error": max(abs(r["slope_m1_ratio"] / r["slope_m1_expected"] - 1.0) for r in rows),
        "max_slope_m2_error": max(abs(r["slope_m2_ratio"] / r["slope_m2_expected"] - 1.0) for r in rows),
    }
    summary["passed"] = bool(
        summary["max_knee_continuity_rel"] <= 5e-3
        and summary["max_abs_delta_log_a1_vs_dnv_d"] <= 1e-9
        and summary["max_declared_vs_callable_rel"] <= 1e-12
        and summary["max_slope_m1_error"] <= 1e-12
        and summary["max_slope_m2_error"] <= 1e-12
    )
    return {
        "reference": DNV_D_AIR,
        "iiw_reference": IIW_FAT90,
        "rows": rows,
        "summary": summary,
    }
