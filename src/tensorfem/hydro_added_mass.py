"""Hydrodynamic added mass of a baffled panel, with exact reference anchors.

Motivation
----------
``tensorfem.added_mass`` adds a *prescribed* mass per unit area

    m_a = c_m * rho_w * L_char

to a plate element.  That is an engineering approximation: the coefficient
``c_m`` is a user input and is not, by itself, derived from the fluid.  This
module supplies the missing reference: the low-frequency (in-phase) added mass
of a planar surface set in an infinite rigid baffle, written as the classical
Rayleigh single-layer integral

    M_ij = rho_w * (1 / (2*pi)) * A_i * A_j / |x_i - x_j|,      i != j
    M_ii = rho_w * A_i * a_i,            a_i = sqrt(A_i / pi)

which is the ``k -> 0`` limit of the baffled-panel radiation operator
(diverging spherical waves, ``G = exp(-j k r) / (2*pi*r)``, ``k -> 0``).
The self term is the exact integral of the kernel over a circular patch of
equal area (``int_patch dA / (2*pi*r) = a`` for a disc of radius ``a``).

Reference anchors used for verification
---------------------------------------
1. **Circular piston in an infinite baffle.**  The low-frequency radiation
   mass is exactly

       M_rad = (8/3) * rho_w * a**3        (Rayleigh 1945, Sec. 302)

   Discretising the disc into flat panels and summing the dipole operator must
   reproduce this value as the mesh is refined.  This is an *analytic*,
   literature-anchored check of the kernel, its normalisation and the
   quadrature.
2. **Single-mode (in-phase) shift.**  For a structure whose response is
   dominated by one mode with a hydrodynamic added-mass ratio

       mu = phi^T M_hydro phi / phi^T M_struct phi

   the wet natural frequency obeys exactly

       f_wet / f_dry = 1 / sqrt(1 + mu).

   Feeding the *equivalent uniform* mass per unit area ``m_a = mu * m_s`` into
   ``added_mass.assemble_added_mass_consistent`` must recover the same ratio,
   which cross-checks the element-level assembly against an independent
   hydrodynamic operator.

All quantities are SI (m, kg, s).  Computations are chunked so that the dense
(N x N) influence matrix can be built for a few thousand panels inside a
modest memory budget.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "piston_radiation_mass",
    "disc_panels",
    "dipole_added_mass_matrix",
    "sine_mode_values",
    "modal_added_mass_ratio",
    "equivalent_added_mass_per_area",
    "wet_frequency_ratio",
    "panel_added_mass_ratio",
]


def _as_tensor(value, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype)


def piston_radiation_mass(radius: float, rho_water: float = 1025.0) -> float:
    """Exact low-frequency radiation (added) mass of a circular piston.

    ``M_rad = (8/3) * rho * a**3`` -- Rayleigh's baffled-piston result, the
    ``ka -> 0`` limit of the piston radiation impedance.

    Parameters
    ----------
    radius : float
        Piston radius [m], must be positive.
    rho_water : float
        Fluid density [kg/m^3], must be positive.
    """
    radius = float(radius)
    rho_water = float(rho_water)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if not math.isfinite(rho_water) or rho_water <= 0.0:
        raise ValueError("rho_water must be positive and finite")
    return 8.0 / 3.0 * rho_water * radius ** 3


def disc_panels(
    radius: float,
    n_cells: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretise a disc into flat square panels on a staircase boundary.

    The disc is inscribed in the square ``[-a, a]^2`` split into
    ``n_cells x n_cells`` equal cells; a cell is kept when its centre lies
    inside the circle.  The discretised area converges to ``pi*a**2`` as
    ``O(1/n_cells)``, which is the leading discretisation error for the
    piston anchor.

    Returns
    -------
    centres : (N, 2) tensor
        Panel centres [m].
    areas : (N,) tensor
        Panel areas [m^2].
    """
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    n_cells = int(n_cells)
    if n_cells < 2:
        raise ValueError("n_cells must be at least 2")
    h = 2.0 * radius / n_cells
    xs = -radius + h * (torch.arange(n_cells, dtype=dtype) + 0.5)
    xx, yy = torch.meshgrid(xs, xs, indexing="xy")
    keep = (xx ** 2 + yy ** 2) <= radius ** 2
    centres = torch.stack((xx[keep], yy[keep]), dim=1)
    areas = torch.full((centres.shape[0],), h * h, dtype=dtype)
    if centres.shape[0] < 4:
        raise ValueError("discretisation produced too few panels")
    return centres, areas


def dipole_added_mass_matrix(
    centres: torch.Tensor,
    areas: torch.Tensor,
    rho_water: float = 1025.0,
    *,
    chunk: int = 256,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Low-frequency baffled-panel added-mass operator (dense, symmetric).

    ``M[i, j] = rho * A_i * A_j / (2*pi*|x_i - x_j|)`` for ``i != j`` and the
    exact self-patch value ``M[i, i] = rho * A_i * sqrt(A_i/pi)``.

    The rows are accumulated in blocks of ``chunk`` panels to keep peak memory
    bounded for a few thousand panels.

    Parameters
    ----------
    centres : (N, 2) tensor
        Panel centres [m].
    areas : (N,) tensor
        Panel areas [m^2].
    rho_water : float
        Fluid density [kg/m^3].
    chunk : int
        Row-block size for the accumulation.
    """
    centres = torch.as_tensor(centres, dtype=dtype)
    areas = torch.as_tensor(areas, dtype=dtype)
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError("centres must have shape (N, 2)")
    n = centres.shape[0]
    if areas.shape != (n,):
        raise ValueError("areas must have shape (N,) matching centres")
    if bool((areas <= 0).any()):
        raise ValueError("all panel areas must be positive")
    rho = float(rho_water)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho_water must be positive and finite")
    chunk = max(1, int(chunk))

    matrix = torch.zeros((n, n), dtype=dtype)
    scale = rho / (2.0 * math.pi)
    # Far-field / near-field part: fill row blocks.
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dx = centres[start:stop, None, 0] - centres[None, :, 0]
        dy = centres[start:stop, None, 1] - centres[None, :, 1]
        dist = torch.sqrt(dx * dx + dy * dy)
        block = areas[start:stop, None] * areas[None, :] / dist
        matrix[start:stop, :] = scale * block
    # Replace the (integrable) singular diagonal by the exact patch value.
    diag = rho * areas * torch.sqrt(areas / math.pi)
    idx = torch.arange(n)
    matrix[idx, idx] = diag
    return matrix


def sine_mode_values(
    centres: torch.Tensor,
    lx: float,
    ly: float,
    m: int = 1,
    n: int = 1,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Nodal values of a simply-supported panel mode ``sin(m pi x/lx) sin(n pi y/ly)``."""
    centres = torch.as_tensor(centres, dtype=dtype)
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError("centres must have shape (N, 2)")
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("panel dimensions must be positive")
    if int(m) < 1 or int(n) < 1:
        raise ValueError("mode numbers must be >= 1")
    return torch.sin(int(m) * math.pi * centres[:, 0] / lx) * torch.sin(
        int(n) * math.pi * centres[:, 1] / ly
    )


def modal_added_mass_ratio(
    matrix: torch.Tensor,
    mode_values: torch.Tensor,
    structural_generalised_mass: float,
) -> float:
    """Added-mass ratio ``mu = phi^T M_hydro phi / phi^T M_struct phi``."""
    matrix = torch.as_tensor(matrix)
    phi = torch.as_tensor(mode_values, dtype=matrix.dtype)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if phi.shape != (matrix.shape[0],):
        raise ValueError("mode_values must match the matrix size")
    m_struct = float(structural_generalised_mass)
    if not math.isfinite(m_struct) or m_struct <= 0.0:
        raise ValueError("structural_generalised_mass must be positive and finite")
    return float(phi @ (matrix @ phi)) / m_struct


def equivalent_added_mass_per_area(mu: float, mass_per_area: float) -> float:
    """Uniform mass per unit area that reproduces the modal ratio ``mu``.

    ``m_a = mu * m_s`` with ``m_s`` the structural mass per unit area.  This is
    the input required by :mod:`tensorfem.added_mass`.
    """
    mu = float(mu)
    mass_per_area = float(mass_per_area)
    if not math.isfinite(mu) or mu < 0.0:
        raise ValueError("mu must be non-negative and finite")
    if not math.isfinite(mass_per_area) or mass_per_area <= 0.0:
        raise ValueError("mass_per_area must be positive and finite")
    return mu * mass_per_area


def wet_frequency_ratio(mu: float) -> float:
    """Exact single-mode in-phase shift ``f_wet / f_dry = 1 / sqrt(1 + mu)``."""
    mu = float(mu)
    if not math.isfinite(mu) or mu < 0.0:
        raise ValueError("mu must be non-negative and finite")
    return 1.0 / math.sqrt(1.0 + mu)


def panel_added_mass_ratio(
    lx: float,
    ly: float,
    thickness: float,
    rho_steel: float,
    rho_water: float,
    *,
    nx: int = 60,
    ny: int = 40,
    m: int = 1,
    n: int = 1,
) -> dict[str, float]:
    """Modal added-mass ratio of a simply-supported rectangular panel in a baffle.

    The panel ``lx x ly`` is discretised into ``nx x ny`` flat panels; the
    structural generalised mass of the sine mode is ``rho_steel * t * lx * ly / 4``
    (exact, since ``int_0^lx sin^2 = lx/2``).

    Returns a dict with the added-mass ratio ``mu``, the equivalent uniform
    added mass per unit area ``added_mass_per_area`` and the exact wet/dry
    frequency ratio ``wet_frequency_ratio``.
    """
    lx, ly = float(lx), float(ly)
    thickness = float(thickness)
    rho_steel, rho_water = float(rho_steel), float(rho_water)
    for name, value in (("lx", lx), ("ly", ly), ("thickness", thickness),
                        ("rho_steel", rho_steel), ("rho_water", rho_water)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    nx, ny = int(nx), int(ny)
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")
    dx, dy = lx / nx, ly / ny
    xs = dx * (torch.arange(nx, dtype=torch.float64) + 0.5)
    ys = dy * (torch.arange(ny, dtype=torch.float64) + 0.5)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    centres = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    areas = torch.full((centres.shape[0],), dx * dy, dtype=torch.float64)
    matrix = dipole_added_mass_matrix(centres, areas, rho_water)
    phi = sine_mode_values(centres, lx, ly, m, n)
    m_struct = rho_steel * thickness * lx * ly / 4.0
    mu = modal_added_mass_ratio(matrix, phi, m_struct)
    mass_per_area = rho_steel * thickness
    return {
        "mu": mu,
        "added_mass_per_area": equivalent_added_mass_per_area(mu, mass_per_area),
        "wet_frequency_ratio": wet_frequency_ratio(mu),
        "structural_generalised_mass": m_struct,
        "n_panels": float(centres.shape[0]),
    }