"""Compressible baffled-panel radiation impedance: reactance *and* resistance.

Why this module exists
----------------------
Two merged capabilities stop at the in-phase (``k -> 0``) limit and say so:

* :mod:`tensorfem.hydro_added_mass` -- added mass only, ``M_ij = rho A_i A_j /
  (2 pi r_ij)``, i.e. the ``k -> 0`` limit of the baffled-panel operator.
* :mod:`tensorfem.fsi_wet_modal` -- module docstring, scope limits:
  "Fluid is incompressible (``k_0 a << 1`` ...); acoustic radiation damping is
  **not** modelled."

So a wet panel in the library has a frequency-*independent* added mass and no
radiation damping at all.  Physically the fluid reaction on a baffled surface
is the Rayleigh integral with the retarded kernel,

    p(r) = (j omega rho / (2 pi)) * int_S v(r') exp(-j k R) / R dS',
    R = |r - r'|,  k = omega / c,

whose modal (single mode ``v = q_dot phi``) form is the **complex** generalised
impedance

    Z(omega) = (j omega rho / (2 pi)) * int int phi(r) phi(r')
               exp(-j k R) / R dS dS'.

Writing ``Z = R + j X``:

* ``X / omega`` is the frequency-dependent generalised added mass
  (``X/omega -> M_in-phase`` as ``k -> 0``),
* ``R`` is the generalised radiation damping, and the radiated power is
  ``P = (1/2) R |q_dot|^2``.

This module supplies both, with the discretisation deliberately reusing the
panel geometry and mode functions of :mod:`tensorfem.hydro_added_mass`, so the
``k -> 0`` reduction can be checked against an already-verified anchor instead
of against itself.

Verification anchors (see ``benchmarks/plate_radiation_impedance.py``)
----------------------------------------------------------------------
1. **Baffled circular piston, closed form** (Morse & Ingard 1968, Sec. 7.3):
   ``Z/(rho c S) = R1(x) + j X1(x)``, ``x = 2 k a``,
   ``R1 = 1 - 2 J1(x)/x``, ``X1 = 2 H1(x)/x`` (``H1`` = Struve function).
2. **Piston low-frequency limits**: ``X1 -> 8 k a / (3 pi)``, i.e. the added
   mass ``X/omega -> (8/3) rho a^3`` -- identical to
   :func:`tensorfem.hydro_added_mass.piston_radiation_mass`; and
   ``R1 -> (k a)^2 / 2``.
3. **Piston high-frequency limit**: ``R1 -> 1`` (all incident power radiated),
   ``X1 -> 0`` as ``1/(k a)``.
4. **Independent code path for the resistance**:
   ``R(omega) = 2 P_rad`` with ``P_rad`` from
   :func:`tensorfem.acoustic_radiation.rayleigh_radiated_power`, which
   integrates the *real* (sine) kernel with 4-D Gauss-Legendre quadrature --
   a different kernel decomposition, a different quadrature and a different
   singularity treatment from the complex panel operator here.
5. **In-phase reduction**: ``mu(omega -> 0)`` must reproduce
   :func:`tensorfem.hydro_added_mass.panel_added_mass_ratio`.

Discretisation (and its honest error)
-------------------------------------
Panels are flat mid-point cells of area ``A = h^2``; off-diagonal entries use
the point-kernel ``exp(-j k r)/r``.  The diagonal is the **analytic cell
integral**

    int_cell exp(-j k r)/r dA
      ~= 2 h ln(1 + sqrt 2) - (k^2/2)(h^3/6)(sqrt 2 + ln(1 + sqrt 2))
         - j [ k h^2 - k^3 h^4 / 36 ]

(the first two terms are the exact ``int dA/r`` and the ``O(k^2)`` correction
for a square of side ``h``; the imaginary part is the ``sin(k r)/r`` expansion
to ``O(k^3 h^4)``).  Using the square-cell value instead of the equal-area-disc
value used by :mod:`tensorfem.hydro_added_mass` changes the result by ``O(h)``
only -- both are ``O(h)`` errors that vanish under refinement; the convergence
study in the benchmark reports the observed order, it does not assume one.

All quantities SI.  ``scipy`` is optional: it is used for the high-argument
Struve function (``x > 8``) where the direct series loses digits; below that
the module evaluates its own series, which is what the verification compares
against ``scipy.special.struve``.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "struve_h1_series",
    "struve_h1",
    "piston_normalized_impedance",
    "piston_radiation_impedance",
    "self_patch_integral",
    "panel_modal_impedance",
    "disc_piston_impedance_panels",
    "panel_wet_dynamics",
]

_SQRT2_LN = math.log(1.0 + math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Struve function H1 (own series, for the "two independent implementations" check)
# ---------------------------------------------------------------------------

def _bessel_j1(x: float) -> float:
    """``J1(x)``: own series for ``|x| <= 20``, ``scipy`` above.

    Above ``|x| = 20`` the alternating series loses its digits (the largest
    term exceeds the result by many orders), so the optional dependency is used
    instead of returning a silently wrong value.
    """
    x = float(x)
    if abs(x) > 20.0:
        try:  # pragma: no cover - optional-dependency path
            from scipy.special import j1 as _scipy_j1
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "|J1(x)| for |x| > 20 needs scipy; install scipy or use "
                "|x| <= 20"
            ) from exc
        return float(_scipy_j1(x))
    half = 0.5 * x
    term = half  # m = 0: (x/2) / (0! 1!)
    total = term
    m = 0
    while True:
        m += 1
        term *= -(half * half) / (m * (m + 1.0))
        total += term
        if abs(term) < 1e-17 * abs(total):
            break
        if m > 200:  # pragma: no cover
            break
    return total


def struve_h1_series(x: float, n_terms: int = 80) -> float:
    """Struve ``H1(x)`` from its power series (accurate for ``|x| <= 8``).

    ``H1(x) = sum_{m>=0} (-1)^m (x/2)^{2m+2} / (Gamma(m+3/2) Gamma(m+5/2))``.

    The series is alternating with terms that first grow, so digits are lost to
    cancellation as ``x`` grows; :func:`struve_h1` switches to the asymptotic
    form (``scipy``) above ``x = 8``.
    """
    x = float(x)
    total = 0.0
    half = 0.5 * x
    power = half * half  # (x/2)^2
    for m in range(int(n_terms)):
        term = power / (math.gamma(m + 0.5 + 1.0) * math.gamma(m + 1.5 + 1.0))
        total += term if (m % 2 == 0) else -term
        power *= half * half
        if m > 8 and abs(term) < 1e-18 * max(abs(total), 1e-300):
            break
    return total


def struve_h1(x: float) -> float:
    """Struve ``H1(x)``: own series below ``x = 8``, ``scipy`` above.

    Raises ``RuntimeError`` (rather than silently returning a series value that
    has lost its digits) when ``x > 8`` and ``scipy`` is unavailable.
    """
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("x must be finite")
    if abs(x) <= 8.0:
        return struve_h1_series(x)
    try:  # pragma: no cover - only on the optional-dependency path
        from scipy.special import struve as _scipy_struve
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "struve_h1(x) for x > 8 needs scipy (the direct series loses "
            "digits at that argument); install scipy or use x <= 8"
        ) from exc
    return float(_scipy_struve(1, x))


# ---------------------------------------------------------------------------
# Baffled circular piston: exact literature anchor
# ---------------------------------------------------------------------------

def piston_normalized_impedance(ka: float) -> tuple[float, float]:
    """Exact normalised radiation impedance of a baffled circular piston.

    Returns ``(R1, X1)`` with ``Z/(rho c S) = R1 + j X1`` and ``x = 2 ka``::

        R1(x) = 1 - 2 J1(x) / x
        X1(x) = 2 H1(x) / x

    Reference: Morse & Ingard, *Theoretical Acoustics* (1968), Sec. 7.3.
    """
    ka = float(ka)
    if not math.isfinite(ka) or ka < 0.0:
        raise ValueError("ka must be non-negative and finite")
    x = 2.0 * ka
    if x == 0.0:
        return 0.0, 0.0
    j1 = _bessel_j1(x)
    r1 = 1.0 - 2.0 * j1 / x
    x1 = 2.0 * struve_h1(x) / x
    return r1, x1


def piston_radiation_impedance(
    ka: float,
    radius: float,
    rho: float = 1025.0,
    sound_speed: float = 1500.0,
) -> dict[str, float]:
    """Dimensional baffled-piston impedance split into mass and damping.

    ``Z = rho c S (R1 + j X1)``.  The single-frequency added mass is
    ``M_rad = X / omega`` and the damping coefficient is ``C_rad = R``.
    """
    radius = float(radius)
    rho = float(rho)
    sound_speed = float(sound_speed)
    for name, value in (("radius", radius), ("rho", rho),
                        ("sound_speed", sound_speed)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    ka = float(ka)
    if not math.isfinite(ka) or ka < 0.0:
        raise ValueError("ka must be non-negative and finite")
    r1, x1 = piston_normalized_impedance(ka)
    area = math.pi * radius * radius
    scale = rho * sound_speed * area
    omega = ka * sound_speed / radius
    added_mass = x1 * scale / omega if omega > 0.0 else float("inf")
    return {
        "ka": ka,
        "R1": r1,
        "X1": x1,
        "Z_re": r1 * scale,
        "Z_im": x1 * scale,
        "added_mass": added_mass,
        "damping": r1 * scale,
    }


# ---------------------------------------------------------------------------
# Analytic cell integral (the diagonal that makes the panel route usable)
# ---------------------------------------------------------------------------

def self_patch_integral(h: float, k: float) -> complex:
    """``int_cell exp(-j k r) / r dA`` over a square cell of side ``h``.

    Real part: exact ``2 h ln(1 + sqrt 2)`` minus the ``O(k^2)`` correction
    ``(k^2/2)(h^3/6)(sqrt 2 + ln(1 + sqrt 2))``.
    Imaginary part: ``-(k h^2 - k^3 h^4 / 36)`` from ``sin(k r)/r ~ k - k^3 r^2/6``
    with ``int_cell r^2 dA = h^4 / 6``.

    Neglected terms are ``O((k h)^4 h ln h)``: negligible at the panel
    resolutions used here (``k h <= 0.1``).
    """
    h = float(h)
    k = float(k)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("h must be positive and finite")
    if not math.isfinite(k) or k < 0.0:
        raise ValueError("k must be non-negative and finite")
    real = 2.0 * h * _SQRT2_LN - 0.5 * k * k * (h ** 3 / 6.0) * (
        math.sqrt(2.0) + _SQRT2_LN
    )
    imag = -(k * h * h - (k ** 3) * (h ** 4) / 36.0)
    return complex(real, imag)


# ---------------------------------------------------------------------------
# Panel modal radiation impedance
# ---------------------------------------------------------------------------

def _panel_grid(lx: float, ly: float, nx: int, ny: int):
    dx, dy = lx / nx, ly / ny
    xs = dx * (torch.arange(nx, dtype=torch.float64) + 0.5)
    ys = dy * (torch.arange(ny, dtype=torch.float64) + 0.5)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    centres = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    areas = torch.full((centres.shape[0],), dx * dy, dtype=torch.float64)
    return centres, areas, dx, dy


def _kernel_quadratic_form(
    centres: torch.Tensor,
    areas: torch.Tensor,
    weights: torch.Tensor,
    k: float,
    h: float,
    chunk: int = 192,
) -> complex:
    """``sum_ij w_i w_j A_i A_j G_ij`` with the complex Rayleigh kernel.

    Rows are processed in blocks so the dense ``N x N`` matrix is never held in
    memory; the diagonal uses :func:`self_patch_integral` divided by ``h^2``
    (the cell average of the kernel), so the fully-discrete double sum *is* the
    mid-point quadrature of the double surface integral.
    """
    centres = torch.as_tensor(centres, dtype=torch.float64)
    areas = torch.as_tensor(areas, dtype=torch.float64)
    w = torch.as_tensor(weights, dtype=torch.float64)
    n = centres.shape[0]
    if w.shape != (n,):
        raise ValueError("weights must match the number of panels")
    if areas.shape != (n,):
        raise ValueError("areas must match the number of panels")
    k = float(k)
    total = 0.0 + 0.0j
    diag_val = self_patch_integral(h, k) / (h * h)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dx = centres[start:stop, None, 0] - centres[None, :, 0]
        dy = centres[start:stop, None, 1] - centres[None, :, 1]
        dist = torch.sqrt(dx * dx + dy * dy)
        kernel = torch.exp(-1j * k * dist) / dist
        rows = torch.arange(start, stop)
        kernel[torch.arange(stop - start), rows] = diag_val
        block = (
            w[start:stop, None] * w[None, :]
            * areas[start:stop, None] * areas[None, :]
        )
        total += complex(torch.sum(block * kernel).item())
    return total


def _mode_weights(centres: torch.Tensor, lx: float, ly: float,
                  m: int, n: int) -> torch.Tensor:
    from .hydro_added_mass import sine_mode_values

    return sine_mode_values(centres, lx, ly, m, n)


def panel_modal_impedance(
    *,
    lx: float,
    ly: float,
    thickness: float,
    rho_steel: float,
    rho_water: float,
    sound_speed: float,
    frequency: float,
    m: int = 1,
    n: int = 1,
    nx: int = 60,
    ny: int = 40,
) -> dict[str, float]:
    """Complex modal radiation impedance of a baffled simply-supported panel.

    ``v(x) = q_dot sin(m pi x/lx) sin(n pi y/ly)`` on ``[0,lx] x [0,ly]``,
    radiating into a half-space of density ``rho_water`` and sound speed
    ``sound_speed`` through an infinite rigid baffle.

    Returns ``Z = R + jX`` together with the physically meaningful quantities:

    ``added_mass_ratio``
        ``mu = (X/omega) / M_struct`` with ``M_struct = rho_s t lx ly / 4``
        (exact for the sine mode; same normalisation as
        :func:`tensorfem.hydro_added_mass.panel_added_mass_ratio`).
    ``radiation_damping``
        ``R`` [N s/m] in the generalised coordinate, so ``P = 0.5 R |q_dot|^2``.
    ``loss_factor``
        ``eta = R / (omega (M_struct + X/omega))`` -- the fraction of the modal
        energy lost per radian, i.e. the half-power bandwidth of the wet mode.
    """
    lx, ly = float(lx), float(ly)
    thickness = float(thickness)
    rho_steel, rho_water = float(rho_steel), float(rho_water)
    sound_speed = float(sound_speed)
    frequency = float(frequency)
    for name, value in (
        ("lx", lx), ("ly", ly), ("thickness", thickness),
        ("rho_steel", rho_steel), ("rho_water", rho_water),
        ("sound_speed", sound_speed),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be positive and finite")
    nx, ny = int(nx), int(ny)
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")
    if int(m) < 1 or int(n) < 1:
        raise ValueError("mode numbers must be >= 1")

    centres, areas, dx, dy = _panel_grid(lx, ly, nx, ny)
    if abs(dx - dy) > 1e-12 * max(dx, dy):
        raise ValueError("panels must be square for the analytic self-patch term")
    phi = _mode_weights(centres, lx, ly, m, n)

    omega = 2.0 * math.pi * frequency
    k = omega / sound_speed
    integral = _kernel_quadratic_form(centres, areas, phi, k, dx)
    impedance = (1j * omega * rho_water / (2.0 * math.pi)) * integral

    m_struct = rho_steel * thickness * lx * ly / 4.0
    added_mass = impedance.imag / omega
    total_mass = m_struct + added_mass
    # Once the (frequency-dependent) reactance cancels the structural
    # generalised mass, the single-mode loss factor R/(omega (M + X/omega))
    # degenerates (it passes through a pole).  Report it as NaN with an explicit
    # flag rather than returning a meaningless negative number.
    loss_factor = (
        impedance.real / (omega * total_mass) if total_mass > 0.0
        else float("nan")
    )
    return {
        "frequency": frequency,
        "ka": k * math.sqrt(lx * ly / math.pi),
        "k": k,
        "impedance_re": impedance.real,
        "impedance_im": impedance.imag,
        "added_mass": added_mass,
        "added_mass_ratio": added_mass / m_struct,
        "structural_generalised_mass": m_struct,
        "radiation_damping": impedance.real,
        "total_generalised_mass": total_mass,
        "loss_factor": loss_factor,
        "loss_factor_valid": total_mass > 0.0,
        "n_panels": float(centres.shape[0]),
    }


def disc_piston_impedance_panels(
    *,
    radius: float,
    ka: float,
    rho: float = 1025.0,
    sound_speed: float = 1500.0,
    n_cells: int = 40,
) -> dict[str, float]:
    """Same panel operator applied to a *disc* -- comparable with the exact piston.

    This is the key cross-check of the discretisation: the geometry (staircase
    disc from :func:`tensorfem.hydro_added_mass.disc_panels`), the kernel and
    the analytic self-patch term are the ones used for panels, yet the result
    can be compared with the closed form of
    :func:`piston_normalized_impedance`.
    """
    from .hydro_added_mass import disc_panels

    radius = float(radius)
    ka = float(ka)
    rho = float(rho)
    sound_speed = float(sound_speed)
    for name, value in (("radius", radius), ("rho", rho),
                        ("sound_speed", sound_speed)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(ka) or ka < 0.0:
        raise ValueError("ka must be non-negative and finite")
    n_cells = int(n_cells)
    if n_cells < 4:
        raise ValueError("n_cells must be at least 4")

    centres, areas = disc_panels(radius, n_cells)
    h = math.sqrt(float(areas[0]))
    # Staircase cells are squares of side h but the boundary is not aligned, so
    # the mid-point cell area is what the operator sees; report the area error.
    k = ka / radius
    ones = torch.ones(centres.shape[0], dtype=torch.float64)
    integral = _kernel_quadratic_form(centres, areas, ones, k, h)
    omega = k * sound_speed
    n = centres.shape[0]
    area_disc = math.pi * radius * radius
    total_area = float(areas.sum())
    # Piston impedance Z = F/v with F = int_S p dS and uniform v:
    #   p = j omega rho / (2 pi) * int_S v exp(-j k R)/R dS'
    # so Z = j omega rho / (2 pi) * double integral (no area division).
    impedance = (1j * omega * rho / (2.0 * math.pi)) * integral
    scale = rho * sound_speed * area_disc
    return {
        "ka": ka,
        "R1_panels": impedance.real / scale,
        "X1_panels": impedance.imag / scale,
        "n_panels": float(n),
        "area_ratio": total_area / area_disc,
        "h_over_a": h / radius,
    }


def panel_wet_dynamics(
    *,
    lx: float,
    ly: float,
    thickness: float,
    rho_steel: float,
    rho_water: float,
    sound_speed: float,
    youngs_modulus: float,
    poisson: float,
    m: int = 1,
    n: int = 1,
    nx: int = 60,
    ny: int = 40,
    low_frequency_reference: float | None = None,
    max_iterations: int = 40,
    tolerance: float = 1e-10,
) -> dict[str, float]:
    """Wet resonance and radiation loss factor of one panel mode.

    The dry (simply-supported, Kirchhoff-Navier) modal stiffness is

        ``K = D pi^4 [ (m/lx)^2 + (n/ly)^2 ]^2 lx ly / 4``,
        ``D = E t^3 / (12 (1 - nu^2))``

    and the undamped wet frequency solves ``f = f_dry / sqrt(1 + mu(f))`` with
    ``mu(f) = (X(f)/omega) / M_struct`` (fixed point, residual reported).

    ``inphase_model_error`` compares that with the frequency-*independent*
    in-phase prediction obtained from the **same** operator at
    ``low_frequency_reference`` (default ``1e-3 * f_dry``, where ``k h ~ 1e-7``).
    Using the same discretisation for both isolates the *physics* of
    compressibility from the ``O(h)`` discretisation error; the cross-module
    difference against :func:`tensorfem.hydro_added_mass.panel_added_mass_ratio`
    is reported separately in ``mu_inphase_hydro_module`` for exactly that
    reason.
    """
    from .hydro_added_mass import panel_added_mass_ratio

    youngs_modulus = float(youngs_modulus)
    poisson = float(poisson)
    if not math.isfinite(youngs_modulus) or youngs_modulus <= 0.0:
        raise ValueError("youngs_modulus must be positive and finite")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("poisson must lie in (-1, 0.5)")

    lx, ly = float(lx), float(ly)
    thickness = float(thickness)
    rho_steel, rho_water = float(rho_steel), float(rho_water)
    sound_speed = float(sound_speed)

    d_flex = youngs_modulus * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    m_struct = rho_steel * thickness * lx * ly / 4.0
    k_modal = (
        d_flex
        * math.pi ** 4
        * ((m / lx) ** 2 + (n / ly) ** 2) ** 2
        * lx * ly / 4.0
    )
    f_dry = math.sqrt(k_modal / m_struct) / (2.0 * math.pi)

    f_ref = (1e-3 * f_dry if low_frequency_reference is None
             else float(low_frequency_reference))
    if f_ref <= 0.0:
        raise ValueError("low_frequency_reference must be positive")

    def mu_at(freq: float) -> float:
        return panel_modal_impedance(
            lx=lx, ly=ly, thickness=thickness, rho_steel=rho_steel,
            rho_water=rho_water, sound_speed=sound_speed, frequency=freq,
            m=m, n=n, nx=nx, ny=ny,
        )["added_mass_ratio"]

    mu_inphase = mu_at(f_ref)
    f_wet_inphase = f_dry / math.sqrt(1.0 + mu_inphase)

    f_wet = f_wet_inphase
    residual = float("inf")
    iterations = 0
    for _ in range(int(max_iterations)):
        f_new = f_dry / math.sqrt(1.0 + mu_at(f_wet))
        residual = abs(f_new - f_wet) / f_wet
        f_wet = f_new
        iterations += 1
        if residual < tolerance:
            break

    imp = panel_modal_impedance(
        lx=lx, ly=ly, thickness=thickness, rho_steel=rho_steel,
        rho_water=rho_water, sound_speed=sound_speed, frequency=f_wet,
        m=m, n=n, nx=nx, ny=ny,
    )
    hydro = panel_added_mass_ratio(
        lx, ly, thickness, rho_steel, rho_water, nx=nx, ny=ny, m=m, n=n
    )
    return {
        "f_dry": f_dry,
        "f_wet_inphase": f_wet_inphase,
        "f_wet_compressible": f_wet,
        "mu_inphase_same_operator": mu_inphase,
        "mu_inphase_hydro_module": hydro["mu"],
        "mu_at_wet": imp["added_mass_ratio"],
        "inphase_model_error": (f_wet_inphase - f_wet) / f_wet,
        "ka_at_wet": imp["ka"],
        "loss_factor": imp["loss_factor"],
        "radiation_damping": imp["radiation_damping"],
        "iteration_residual": residual,
        "iterations": float(iterations),
    }
