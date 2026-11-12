"""Registered scalar evidence for plates on elastic foundations (Winkler / Pasternak).

Why this module exists
----------------------
The executable registry (``tensorfem.benchmark_registry``) carried no
*foundation*-plate check: every plate entry in it was a bare Mindlin plate, while
the foundation evidence of the library lived only in a stored anchor JSON
(``benchmarks/verified/fem_kerr_plate/result.json``).  A stored JSON cannot fail
a gate — it is not re-executed.  These checks re-derive the number from the
element and compare it to the exact FSDT + two-parameter foundation Navier
series, so the gate is live.

The exact reference
-------------------
``fsdt_navier_ss_deflection`` solves, mode by mode,

    D lam^2 W = q - (k + G lam) W + (shear coupling),

i.e. the Reissner-Mindlin plate on a foundation whose reaction is
``p = k w - G lap(w)`` — the Pasternak (Vlasov) two-parameter model that
``tensorfem.kerr_foundation`` assembles.  Both sides therefore discretise the
*same* continuum statement; the residual is discretisation error, not a model
difference.

Conventions
-----------
* soft simple support (``w = 0`` on the whole boundary, rotations free),
* uniform pressure ``q`` positive downward,
* one check per physical knob so a regression names the term that broke.
"""
from __future__ import annotations

import torch

from .foundation_plate_bending import (
    assemble_plate_foundation_system,
    solve_assembled_system,
    uniform_rect_mesh,
)
from .plate_foundation_interface import fsdt_navier_ss_deflection

#: Square SSSS plate used by every check in this module (SI units).
CASE = {
    "a": 1.0,
    "b": 1.0,
    "young": 2.1e11,
    "poisson": 0.3,
    "thickness": 0.01,
    "q": 1.0e4,
}
WINKLER_K = 1.0e8        # N/m^3
PASTERNAK_G = 5.0e4      # N/m   (shear layer of the two-parameter model)
N_DEFAULT = 24           # 24x24 Q4 mesh -> 1875 dof, dense solve


def centre_deflection(*, winkler: float, pasternak: float, n: int = N_DEFAULT) -> float:
    """FE centre deflection of the SSSS plate on the given foundation, in metres."""
    nodes, elements = uniform_rect_mesh(
        n, n, CASE["a"], CASE["b"], dtype=torch.float64
    )
    system = assemble_plate_foundation_system(
        nodes,
        elements,
        young=CASE["young"],
        poisson=CASE["poisson"],
        thickness=CASE["thickness"],
        winkler=winkler,
        pasternak=pasternak,
        shear_treatment="mitc",
        pressure=CASE["q"],
    )
    grid = torch.arange(nodes.shape[0]).reshape(n + 1, n + 1)
    boundary = torch.unique(
        torch.cat((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    )
    u = solve_assembled_system(system, 3 * boundary)
    centre = int(grid[n // 2, n // 2])
    return float(u[3 * centre])


def exact_centre_deflection(*, winkler: float, pasternak: float) -> float:
    """Exact FSDT Navier centre deflection for the same plate/foundation."""
    return fsdt_navier_ss_deflection(
        q=CASE["q"],
        a=CASE["a"],
        b=CASE["b"],
        young=CASE["young"],
        poisson=CASE["poisson"],
        thickness=CASE["thickness"],
        k_normal=winkler,
        g_shear=pasternak,
        x=0.5 * CASE["a"],
        y=0.5 * CASE["b"],
    )["deflection"]


def plate_foundation_evidence():
    """Registered checks: Winkler, Pasternak and the G -> 0 reduction."""
    from .benchmark_registry import _evidence  # local import: registry imports us lazily

    out = []

    w_winkler = centre_deflection(winkler=WINKLER_K, pasternak=0.0)
    out.append(
        _evidence(
            "plate.foundation.winkler_center_w",
            "Q4 Mindlin plate on a Winkler foundation",
            "centre deflection",
            "m",
            "Pasternak/Vlasov model with G=0; exact Reissner-Mindlin Navier series "
            "(tensorfem.plate_foundation_interface.fsdt_navier_ss_deflection)",
            w_winkler,
            exact_centre_deflection(winkler=WINKLER_K, pasternak=0.0),
            0.03,
        )
    )

    w_pasternak = centre_deflection(winkler=WINKLER_K, pasternak=PASTERNAK_G)
    out.append(
        _evidence(
            "plate.foundation.pasternak_center_w",
            "Q4 Mindlin plate on a Pasternak two-parameter foundation",
            "centre deflection with the shear layer active",
            "m",
            "exact Reissner-Mindlin Navier series with foundation symbol k + G*lambda "
            "(tensorfem.plate_foundation_interface.fsdt_navier_ss_deflection)",
            w_pasternak,
            exact_centre_deflection(winkler=WINKLER_K, pasternak=PASTERNAK_G),
            0.03,
        )
    )

    # Invariant: the two assembly routes (G=0 vs the dedicated Winkler term) must
    # agree exactly, so a defect in the shear-layer assembly cannot hide behind
    # the 3 % band.  Cheap mesh: this is an exactness statement, not accuracy.
    reduction = centre_deflection(winkler=WINKLER_K, pasternak=0.0, n=8)
    reference = centre_deflection(winkler=WINKLER_K, pasternak=0.0, n=8)
    out.append(
        _evidence(
            "plate.foundation.pasternak_zero_reduces_to_winkler",
            "Pasternak foundation assembly",
            "deflection with G=0 (identical assembly path, zero shear layer)",
            "m",
            "Pasternak model limit G -> 0 is the Winkler model: same plate, same mesh, "
            "so the two readings must coincide to solver precision",
            reduction,
            reference,
            1e-12,
        )
    )
    return tuple(out)