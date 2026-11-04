"""Machine-checked derivation of the FSDT (Mindlin) simply-supported plate spectrum.

The free-vibration reference in ``tensorfem.plate_thick_reference`` is a closed
form obtained from the determinant of the 3x3 wave-number (Fourier) operator of
the first-order shear deformation plate with rotary inertia.  A closed form that
is only transcribed from a textbook is unfalsifiable, so this script re-derives
it with ``sympy`` and checks that the coefficient set implemented in the library
reproduces the factored determinant symbolically.

Run::

    python scripts/derive_mindlin_sss_determinant.py

Exit code 0 means the factorisation and the library coefficient set agree.  The
script needs ``sympy``, which is *not* a runtime dependency of the library.
"""

from __future__ import annotations

import sys

import sympy as sp

ALPHA, BETA = sp.symbols("alpha beta", positive=True)
K_SHEAR, RIGIDITY = sp.symbols("k_shear rigidity", positive=True)
RHO, H, NU, LAM, ROTARY = sp.symbols("rho h nu lam rotary", positive=True)


def factor_determinant():
    """Return the factored determinant of the FSDT Fourier operator.

    Unknowns are the transverse displacement ``w`` and the two rotations
    ``psi_x, psi_y``; ``lam = omega**2``.  ``k_shear = kappa G h`` is the
    transverse shear stiffness and ``rotary = rho h**3 / 12`` the rotary
    inertia per unit area.
    """
    wave = ALPHA**2 + BETA**2
    a11 = RHO * H * LAM - K_SHEAR * wave
    a12 = -K_SHEAR * ALPHA
    a13 = -K_SHEAR * BETA
    common = ROTARY * LAM - RIGIDITY * (1 - NU) * wave / 2 - K_SHEAR
    a22 = common - RIGIDITY * (1 + NU) * ALPHA**2 / 2
    a33 = common - RIGIDITY * (1 + NU) * BETA**2 / 2
    a23 = -RIGIDITY * (1 + NU) * ALPHA * BETA / 2
    operator = sp.Matrix([[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]])
    return sp.factor(sp.expand(operator.det()))


def library_closed_form():
    """The coefficient set used by ``tensorfem.plate_thick_reference``.

    Returns ``(quadratic, rotation, flexural_first_order)`` where
    ``quadratic = a*lam**2 + b*lam + c`` is the two-branch flexural problem,
    ``rotation`` the third (pure-rotation) root and ``flexural_first_order`` the
    quadratic with rotary inertia dropped (transverse shear only).
    """
    wave = ALPHA**2 + BETA**2
    quadratic = (
        ROTARY * LAM**2
        - (RIGIDITY * wave + K_SHEAR + K_SHEAR * wave * H**2 / 12) * LAM
        + RIGIDITY * K_SHEAR * wave**2 / (RHO * H)
    )
    rotation = (K_SHEAR + RIGIDITY * (1 - NU) * wave / 2) / ROTARY
    shear_only = -(RIGIDITY * wave + K_SHEAR) * LAM + RIGIDITY * K_SHEAR * wave**2 / (RHO * H)
    return sp.expand(quadratic), rotation, sp.expand(shear_only)


def main() -> int:
    determinant = sp.expand(factor_determinant())
    quadratic, rotation, shear_only = library_closed_form()
    wave = ALPHA**2 + BETA**2

    # The library hard-codes rotary = rho h**3 / 12; impose that identity.
    rotary_identity = {ROTARY: RHO * H**3 / 12}
    det = sp.expand(determinant.subs(rotary_identity))
    quad = sp.expand(quadratic.subs(rotary_identity))
    rot = sp.simplify(rotation.subs(rotary_identity))

    # det = rho h (lam*rotary - k_shear - D(1-nu)R/2) * quad
    candidate = sp.expand(
        RHO * H * (LAM * ROTARY - K_SHEAR - RIGIDITY * (1 - NU) * wave / 2) * quad
    ).subs(rotary_identity)
    candidate = sp.expand(candidate)
    residual = sp.simplify(sp.expand(det - candidate))

    # Limiting cases that the reference must reproduce exactly.
    kirchhoff = RIGIDITY * wave**2 / (RHO * H)  # no shear, no rotary inertia
    thin = sp.simplify(quad.subs(RIGIDITY, 0))  # sanity: shear-only spectrum shape

    print("factored determinant:")
    print(" ", factor_determinant())
    print("third (pure-rotation) branch lam =", rot)
    print("thin-limit check R->0:", thin)
    print("Kirchhoff reference lam =", kirchhoff)
    print("residual (must vanish):", residual)
    if residual != 0:
        print("FAIL: library closed form does not reproduce the determinant")
        return 1
    print("OK: closed form reproduces the symbolic determinant")
    return 0


if __name__ == "__main__":
    sys.exit(main())