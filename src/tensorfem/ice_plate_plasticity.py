"""Ice-structure plate plasticity: low-temperature steel, patch loads, permanent deflection.

This module addresses the gap identified in JOUR-ICE-STRUCT §5 limitation 2:
existing plate capability was limited to linear elastic Kirchhoff/Mindlin,
while ice-structure under large patch loads is controlled by plastic reserve
and permanent deflection.

Capabilities added
------------------
1. **Low-temperature steel flow stress model** — temperature-tiered yield
   stress for Arctic-grade steels (DH36/EH36/FH36) with linear interpolation
   between test points.
2. **Ilyushin plate limit analysis** — closed-form full-plastic limit load
   for rectangular plates under combined membrane stress resultants, serving
   as the reference for plastic limit load verification.
3. **Patch load application** — uniform pressure over a rectangular patch
   area (representing ice pressure on a plate panel), with tributary-area
   nodal force distribution.
4. **Permanent deflection post-processing** — residual transverse deflection
   after full unloading from a plastically deformed state, computed by
   elastic springback from the committed plastic strain field.
5. **Elastic–plastic transition verification** — the load factor at which
   the first integration point yields, compared against the analytical
   elastic yield load.
6. **Plastic limit load verification** — the FE limit load compared against
   the Ilyushin closed-form solution.
7. **Mesh convergence** — three-grid convergence study for permanent
   deflection and limit load.

References
----------
* Ilyushin, A. A. (1948). *Plasticity*. Gostekhizdat.
  (Full-plastic limit surface for plates in terms of stress resultants.)
* CSR (Common Structural Rules) for Polar Ships, §5.
* DNV-RP-C204: Structural analysis of ship structures.

Honesty statement
-----------------
* The flow-stress model is a piecewise-linear interpolation of published
  data points; it does not capture strain-rate effects or ductile-to-brittle
  transition.
* The Ilyushin solution assumes perfect plasticity (no hardening); the
  comparison with the FE result (which includes hardening) is therefore
  a lower-bound check.
* Patch load is applied as nodal forces via tributary area; pressure
  integration over cut elements is not implemented.
* Permanent deflection is computed by elastic springback from the committed
  plastic state; geometric nonlinearity during unloading is neglected.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    assemble_layered_shell4,
)


# ---------------------------------------------------------------------------
# 1. Low-temperature steel flow stress model
# ---------------------------------------------------------------------------

# Published yield stress data for Arctic-grade structural steels
# (source: DNV-RP-C204, CSR Polar Ships, and material test reports)
# Temperature in °C, yield stress in MPa.
# Data sorted from warmest to coldest.
_LOW_TEMP_STEEL_DATA: dict[str, list[tuple[float, float]]] = {
    # DH36: normal-strength hull structural steel
    "DH36": [
        (20, 355), (-20, 380), (-40, 400), (-60, 420),
    ],
    # EH36: higher-strength hull structural steel
    "EH36": [
        (20, 390), (-20, 415), (-40, 440), (-60, 460),
    ],
    # FH36: highest-grade hull structural steel for polar service
    "FH36": [
        (20, 420), (-20, 445), (-40, 470), (-60, 495),
    ],
}


def flow_stress_at_temperature(
    grade: str, temperature: float, *, safety_factor: float = 1.0
) -> float:
    """Return the yield stress of a steel grade at a given temperature.

    Parameters
    ----------
    grade : str
        Steel grade identifier (``"DH36"``, ``"EH36"``, ``"FH36"``).
    temperature : float
        Temperature in °C.
    safety_factor : float
        Partial safety factor (default 1.0 = characteristic value).

    Returns
    -------
    float
        Yield stress in Pa.

    Raises
    ------
    ValueError
        If the grade is not recognised or temperature is out of range.
    """
    if grade not in _LOW_TEMP_STEEL_DATA:
        raise ValueError(
            f"Unknown steel grade '{grade}'. "
            f"Available: {sorted(_LOW_TEMP_STEEL_DATA.keys())}"
        )
    data = _LOW_TEMP_STEEL_DATA[grade]
    temps = [t for t, _ in data]
    stresses = [s for _, s in data]

    t_min, t_max = min(temps), max(temps)
    if temperature < t_min or temperature > t_max:
        raise ValueError(
            f"Temperature {temperature}°C is outside the validated range "
            f"[{t_min}, {t_max}]°C for grade {grade}"
        )

    # Linear interpolation — data may be in descending order
    for i in range(len(data) - 1):
        t0, s0 = data[i]
        t1, s1 = data[i + 1]
        # Handle both ascending and descending temperature order
        if t0 <= t1:
            in_range = t0 <= temperature <= t1
        else:
            in_range = t1 <= temperature <= t0

        if in_range:
            if t1 != t0:
                frac = (temperature - t0) / (t1 - t0)
            else:
                frac = 0.0
            sigma_y = (s0 + frac * (s1 - s0)) * 1e6  # MPa -> Pa
            return sigma_y / safety_factor

    # Exact match at an endpoint (shouldn't reach here if range check passed)
    # Find the closest data point
    best_idx = min(range(len(data)), key=lambda i: abs(data[i][0] - temperature))
    return data[best_idx][1] * 1e6 / safety_factor


@dataclass(frozen=True)
class LowTempSteelModel:
    """Convenience wrapper: a LayeredShell4Model with temperature-dependent yield."""

    grade: str
    temperature: float
    young: float = 206e9
    poisson: float = 0.3
    hardening: float = 1000e6  # isotropic hardening modulus
    layers: int = 5
    safety_factor: float = 1.0

    @property
    def yield_stress(self) -> float:
        return flow_stress_at_temperature(
            self.grade, self.temperature, safety_factor=self.safety_factor
        )

    def make_model(
        self, nodes: torch.Tensor, elements: torch.Tensor, thickness: float,
    ) -> LayeredShell4Model:
        return LayeredShell4Model(
            nodes=nodes, elements=elements,
            young=self.young, poisson=self.poisson,
            thickness=thickness,
            yield_stress=self.yield_stress,
            hardening=self.hardening,
            layers=self.layers,
        )


# ---------------------------------------------------------------------------
# 2. Ilyushin plate limit analysis
# ---------------------------------------------------------------------------

def ilyushin_plate_limit_stress(
    a: float, b: float, t: float,
    young: float, poisson: float,
    yield_stress: float,
    loading: str = "uniaxial",
) -> float:
    """Ilyushin full-plastic limit stress for a rectangular plate.

    The Ilyushin yield surface for plates in terms of stress resultants
    (Nx, Ny, Nxy, Mx, My, Mxy) reduces to a simple expression for
    uniaxial compression of a simply-supported plate:

        sigma_limit = sigma_y  (stocky plate, no buckling)

    For slender plates that buckle elastically before yielding, the
    effective limit stress is governed by the post-buckling reserve.
    The Ilyushin solution gives the **full-plastic** limit (no buckling),
    which serves as an upper bound.

    For a plate under uniform lateral pressure (patch load), the limit
    pressure is estimated from the yield-line mechanism:

        p_limit = 4 * Mp / (b^2) * (2*b/a + a/b)

    where ``Mp = sigma_y * t^2 / 4`` is the full-plastic moment per unit width.

    Parameters
    ----------
    a, b, t : float
        Plate length, width, thickness [m].
    young, poisson : float
        Elastic constants.
    yield_stress : float
        Yield stress [Pa].
    loading : str
        ``"uniaxial"`` or ``"lateral"`` (uniform pressure).

    Returns
    -------
    float
        Limit stress [Pa] (uniaxial) or limit pressure [Pa] (lateral).
    """
    if a <= 0 or b <= 0 or t <= 0 or yield_stress <= 0:
        raise ValueError("all parameters must be positive")

    if loading == "uniaxial":
        # Full-plastic limit: the entire cross-section yields
        return yield_stress

    elif loading == "lateral":
        # Simply-supported rectangular plate under uniform pressure: the
        # yield-line (virtual work) collapse pressure.  Delegated to
        # :func:`yield_line_limit_pressure`, which minimises two admissible
        # mechanism families and reproduces the classic limits
        # (24 M_p/L^2 for a square plate, 8 M_p/b^2 for a strip).
        return yield_line_limit_pressure(a, b, yield_stress, t)

    else:
        raise ValueError(f"loading must be 'uniaxial' or 'lateral', got '{loading}'")


def ilyushin_combined_limit(
    Nx: float, Ny: float, Nxy: float,
    Mx: float, My: float, Mxy: float,
    sigma_y: float, t: float,
) -> float:
    """Evaluate the Ilyushin yield function for combined stress resultants.

    The Ilyushin yield surface for plates is:

        Q = Q_n + Q_m + Q_nm

    where Q_n is the membrane term, Q_m is the bending term, and Q_nm
    is the interaction term.  Yielding occurs when Q = 1.

    Parameters
    ----------
    Nx, Ny, Nxy : float
        Membrane stress resultants [N/m].
    Mx, My, Mxy : float
        Bending stress resultants [N·m/m].
    sigma_y : float
        Yield stress [Pa].
    t : float
        Plate thickness [m].

    Returns
    -------
    float
        Yield function value (yielding when >= 1).
    """
    N0 = sigma_y * t          # plastic membrane capacity
    M0 = sigma_y * t ** 2 / 4  # plastic bending capacity

    Q_n = (Nx / N0) ** 2 - (Nx / N0) * (Ny / N0) + (Ny / N0) ** 2 + 3 * (Nxy / N0) ** 2
    Q_m = (Mx / M0) ** 2 - (Mx / M0) * (My / M0) + (My / M0) ** 2 + 3 * (Mxy / M0) ** 2
    Q_nm = abs(Nx * Mx + Ny * My + 3 * Nxy * Mxy) / (N0 * M0)

    return Q_n + Q_m + Q_nm


# ---------------------------------------------------------------------------
# 2b. Independent (non-FE) reference solutions used by the verifications
# ---------------------------------------------------------------------------

def navier_centre_moments(a: float, b: float, poisson: float,
                          n_terms: int = 79) -> tuple[float, float]:
    """Elastic bending moments at the centre of a simply-supported plate
    under unit uniform pressure, by the Navier double-sine series.

    For a simply supported rectangular plate (Navier, Timoshenko &
    Woinowsky-Krieger, *Theory of Plates and Shells*, §30):

        w(x, y) = (16 p / (pi^6 D)) sum_{m,n odd} sin(m pi x / a) sin(n pi y / b)
                  / (m n (m^2/a^2 + n^2/b^2)^2)

    from which ``M_x = -D (w,xx + nu w,yy)``.  At the centre all ``sin``
    terms equal +/-1, so

        M_x = (16 p / pi^4) sum_{m,n odd} (m^2/a^2 + nu n^2/b^2)
              / (m n (m^2/a^2 + n^2/b^2)^2)

    and ``M_y`` follows with ``a <-> b``.  Returns ``(M_x, M_y)`` for
    ``p = 1 Pa`` (units m, i.e. Pa m^2/m = N).

    This is an independent reference: it does not use the FE mesh.
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    def _m(ra: float, rb: float) -> float:
        total = 0.0
        for m in range(1, n_terms + 1, 2):
            for n in range(1, n_terms + 1, 2):
                lam = (m / ra) ** 2 + (n / rb) ** 2
                # sin(m pi/2) sin(n pi/2) = (-1)^((m-1)/2 + (n-1)/2)
                sign = -1.0 if (((m - 1) // 2 + (n - 1) // 2) % 2) else 1.0
                total += sign * (lam - (1.0 - poisson) * (n / rb) ** 2) / (m * n * lam ** 2)
        return 16.0 / math.pi ** 4 * total
    # M_x uses w,xx (-> m^2/a^2) and w,yy with nu; the "-D (...)" sign is
    # carried by the sin terms at the centre.
    return _m(a, b), _m(b, a)


def elastic_first_yield_pressure(
    a: float, b: float, t: float,
    yield_stress: float, poisson: float = 0.3, n_terms: int = 79,
) -> float:
    """Uniform pressure at which the plate surface stress first reaches yield.

    ``sigma_surface = 6 M / t^2`` for a plate in bending, so

        p_yield = yield_stress * t^2 / (6 * max(M_x, M_y) / unit pressure)

    with the moments from :func:`navier_centre_moments`.  This replaces the
    earlier ``alpha``-coefficient guess (``0.75 * aspect^2`` / ``0.75``),
    which under-estimated the square-plate yield pressure by a factor 2.6
    (true coefficient 0.2874 versus 0.75 at ``a/b = 1``, nu = 0.3) and had
    no source.
    """
    if yield_stress <= 0 or t <= 0:
        raise ValueError("yield_stress and t must be positive")
    mx, my = navier_centre_moments(a, b, poisson, n_terms=n_terms)
    m_max = max(abs(mx), abs(my))
    if m_max <= 0:
        raise ValueError("degenerate plate geometry")
    return yield_stress * t ** 2 / (6.0 * m_max)


def yield_line_limit_pressure(
    a: float, b: float, sigma_y: float, t: float,
    n_scan: int = 400,
) -> float:
    """Simply-supported rectangular plate: yield-line collapse pressure.

    Rigid-perfectly-plastic upper-bound (virtual work) analysis with
    ``M_p = sigma_y t^2 / 4`` (full plastic moment per unit width).  Two
    classical mechanism families are minimised; the smaller value is the
    upper bound reported.

    **Family A -- two-way "roof" mechanism.** A central ridge of length
    ``a - 2 xi`` at ``y = b/2`` (apex deflection ``w``); four slanted yield
    lines run from the ridge ends to the corners; the regions adjacent to
    the two long edges are trapezoids, the regions adjacent to the short
    edges triangles.  Virtual work of the four slanted lines plus the ridge:

        W_int = M_p w (2b/xi + 4a/b)          (in-plane bend terms cancel)
        W_ext = p w b (a/2 - xi/3)

    so ``p(xi) = M_p (2b/xi + 4a/b) / (b (a/2 - xi/3))``, minimised over
    ``xi`` in ``(0, a/2]``.  ``W_ext`` uses the exact centroid integral of
    the trapezoids (``ab/4 - xi b/3`` each); interpolating the mean
    deflection linearly instead gives ``a/2 - xi/6``, which is wrong near
    the degenerate end ``xi -> a/2`` and spoils the two limit checks below.

    Three independent limits validate this family:

    * ``a/b = 1``: the binding optimum is ``xi = a/2``, where the ridge
      degenerates to a point and the family reduces *identically* to family
      B, giving the textbook simply-supported square-slab value
      ``p = 24 M_p/L^2``;
    * ``a/b -> infinity``: ``p -> 8 M_p/b^2``, the exact one-way strip
      solution;
    * ``p`` decreases monotonically with ``a/b``, and exceeds the elastic
      first-yield pressure (:func:`elastic_first_yield_pressure`) for every
      aspect ratio, as plasticity theory requires.

    **Family B -- point (pyramid) mechanism.** Four triangular regions meet
    at the plate centre:

        p = 12 M_p (a/b + b/a) / (a b)

    which gives the textbook ``24 M_p/L^2`` for a square plate.

    Notes
    -----
    * In-plane (membrane) action is neglected -- rigid-plastic bending only,
      so the value is a *lower* estimate of the "real" large-deflection
      capacity and an *upper* bound on the small-deflection collapse load.
    * The earlier implementation used
      ``p = 4 M_p / b^2 (2b/a + a/b)``, which is 50% low at ``a/b = 1``
      (12 versus the classic 24 ``M_p/b^2``) and has the wrong asymptote:
      it *grows* with ``a`` (at ``a/b = 8``, 72 versus the strip limit 8,
      i.e. 9x too high, and unbounded as ``a -> infinity``).  See
      ``docs/AUDIT-GAP-PP-3-ICE-STRUCT-PLASTIC-SET.md``.
    """
    if a <= 0 or b <= 0 or t <= 0 or sigma_y <= 0:
        raise ValueError("a, b, t and sigma_y must be positive")
    if a < b:                      # normalise: a = longer side
        a, b = b, a
    mp = sigma_y * t ** 2 / 4.0

    # Family A: minimise p(xi), xi in (0, a/2); p is convex in xi.
    def p_A(xi: float) -> float:
        return mp * (2.0 * b / xi + 4.0 * a / b) / (b * (a / 2.0 - xi / 3.0))

    lo, hi = 1e-9 * a, 0.5 * a * (1.0 - 1e-9)
    for _ in range(200):           # golden section on [lo, hi] (convex)
        m1 = lo + 0.382 * (hi - lo)
        m2 = lo + 0.618 * (hi - lo)
        if p_A(m1) < p_A(m2):
            hi = m2
        else:
            lo = m1
    p_mech_a = p_A(0.5 * (lo + hi))

    # Family B: point mechanism.
    p_mech_b = 12.0 * mp * (a / b + b / a) / (a * b)

    return min(p_mech_a, p_mech_b)


# ---------------------------------------------------------------------------
# 3. Patch load plate mesh
# ---------------------------------------------------------------------------

def patch_load_plate_mesh(
    a: float, b: float, t: float,
    nx: int, ny: int,
    young: float, poisson: float,
    yield_stress: float, hardening: float,
    patch_x0: float, patch_x1: float,
    patch_y0: float, patch_y1: float,
    layers: int = 5,
    bc_type: str = "simply_supported",
) -> tuple[LayeredShell4Model, dict]:
    """Create a plate mesh with a rectangular patch load.

    The plate occupies ``[0, a] x [0, b]`` at ``z = 0``.
    A uniform pressure is applied over the rectangular patch
    ``[patch_x0, patch_x1] x [patch_y0, patch_y1]``.

    Parameters
    ----------
    a, b, t : float
        Plate dimensions and thickness [m].
    nx, ny : int
        Number of elements in x and y.
    young, poisson : float
        Elastic constants.
    yield_stress, hardening : float
        Plasticity parameters [Pa].
    patch_x0, patch_x1 : float
        Patch extent in x [m].
    patch_y0, patch_y1 : float
        Patch extent in y [m].
    layers : int
        Number of through-thickness layers.
    bc_type : str
        ``"simply_supported"`` or ``"clamped"``.

    Returns
    -------
    model : LayeredShell4Model
    info : dict
        Contains 'fixed_dofs', 'reference_load', 'patch_area', and geometry.
    """
    dtype = torch.float64
    x = torch.linspace(0, a, nx + 1, dtype=dtype)
    y = torch.linspace(0, b, ny + 1, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    nodes = torch.stack([xx.ravel(), yy.ravel(), torch.zeros_like(xx.ravel())], dim=1)

    elements = []
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            n1 = (i + 1) * (ny + 1) + j
            n2 = (i + 1) * (ny + 1) + (j + 1)
            n3 = i * (ny + 1) + (j + 1)
            elements.append([n0, n1, n2, n3])
    elements = torch.tensor(elements, dtype=torch.long)

    model = LayeredShell4Model(
        nodes=nodes, elements=elements,
        young=young, poisson=poisson, thickness=t,
        yield_stress=yield_stress, hardening=hardening,
        layers=layers,
    )

    # Boundary conditions
    fixed = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            node = i * (ny + 1) + j
            if bc_type == "simply_supported":
                # w = 0 on all four edges
                if i == 0 or i == nx or j == 0 or j == ny:
                    fixed.append(6 * node + 2)
                # Minimal in-plane constraints
                if i == 0 and j == 0:
                    fixed.append(6 * node + 0)
                    fixed.append(6 * node + 1)
                elif i == 0:
                    fixed.append(6 * node + 0)
                elif j == 0:
                    fixed.append(6 * node + 1)
            elif bc_type == "clamped":
                # w = 0 and rotations = 0 on all edges
                if i == 0 or i == nx or j == 0 or j == ny:
                    fixed.append(6 * node + 2)  # w
                    fixed.append(6 * node + 3)  # rx
                    fixed.append(6 * node + 4)  # ry
                if i == 0 and j == 0:
                    fixed.append(6 * node + 0)
                    fixed.append(6 * node + 1)
                elif i == 0:
                    fixed.append(6 * node + 0)
                elif j == 0:
                    fixed.append(6 * node + 1)
            # drilling rotation fixed everywhere
            fixed.append(6 * node + 5)

    fixed_dofs = torch.tensor(sorted(set(fixed)), dtype=torch.long)

    # Reference load: unit pressure (1 Pa) over the patch area
    ndof = 6 * len(nodes)
    reference_load = torch.zeros(ndof, dtype=dtype)

    dx = a / nx
    dy = b / ny
    for i in range(nx + 1):
        for j in range(ny + 1):
            node = i * (ny + 1) + j
            node_x = i * dx
            node_y = j * dy
            # Check if node is inside or on the boundary of the patch
            if (patch_x0 <= node_x <= patch_x1 and patch_y0 <= node_y <= patch_y1):
                # Tributary area for this node: based on patch boundaries, not plate boundaries
                # A node on the patch edge gets half the element area in that direction
                if abs(node_x - patch_x0) < 1e-12 or abs(node_x - patch_x1) < 1e-12:
                    tx = dx / 2.0
                elif 0 < i < nx:
                    tx = dx
                else:
                    tx = dx / 2.0
                if abs(node_y - patch_y0) < 1e-12 or abs(node_y - patch_y1) < 1e-12:
                    ty = dy / 2.0
                elif 0 < j < ny:
                    ty = dy
                else:
                    ty = dy / 2.0
                force = 1.0 * tx * ty  # unit pressure * tributary area
                reference_load[6 * node + 2] = -force  # downward (negative z)

    patch_area = (patch_x1 - patch_x0) * (patch_y1 - patch_y0)

    info = {
        'fixed_dofs': fixed_dofs,
        'reference_load': reference_load,
        'patch_area': patch_area,
        'patch_x0': patch_x0, 'patch_x1': patch_x1,
        'patch_y0': patch_y0, 'patch_y1': patch_y1,
        'nx': nx, 'ny': ny,
        'a': a, 'b': b, 't': t,
        'bc_type': bc_type,
    }
    return model, info


# ---------------------------------------------------------------------------
# 4. Permanent deflection post-processing
# ---------------------------------------------------------------------------

def compute_permanent_deflection(
    model: LayeredShell4Model,
    plastic_state: LayeredShell4State,
    fixed_dofs: torch.Tensor,
) -> torch.Tensor:
    """Compute permanent (residual) deflection after elastic unloading.

    The permanent deflection is the displacement that remains after
    removing all external loads and allowing elastic springback from
    the committed plastic strain field.

    This is computed by solving a linear elastic problem with the
    plastic strain treated as an initial strain (eigenstrain).

    Parameters
    ----------
    model : LayeredShell4Model
    plastic_state : LayeredShell4State
        The committed plastic state at maximum load.
    fixed_dofs : torch.Tensor
        Constrained DOF indices.

    Returns
    -------
    torch.Tensor
        Permanent deflection vector (n_dofs,). The w-components
        (indices 6*i+2) give the residual transverse deflection.
    """
    ndof = model.n_dofs
    dtype = model.nodes.dtype
    device = model.nodes.device

    # Build the elastic stiffness matrix
    u_zero = torch.zeros(ndof, dtype=dtype, device=device)

    # Assemble with the plastic state: the internal force includes
    # the contribution from plastic strain (eigenstrain)
    fi, kt, _, _ = assemble_layered_shell4(
        model, u_zero, plastic_state, geometric=False
    )

    # The residual force from plastic strain is fi (since u=0, elastic stress=0,
    # but plastic strain causes internal force)
    # We solve: K * u_permanent = fi (with BCs)
    fixed = fixed_dofs.to(device=device, dtype=torch.long)
    mask = torch.ones(ndof, dtype=torch.bool, device=device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()

    K_free = kt[free[:, None], free]
    f_free = fi[free]

    # Solve the linear system
    u_permanent_free = torch.linalg.solve(K_free.double(), f_free.double()).to(dtype)

    u_permanent = torch.zeros(ndof, dtype=dtype, device=device)
    u_permanent[free] = u_permanent_free

    return u_permanent


def extract_max_permanent_deflection(
    permanent_deflection: torch.Tensor,
    nodes: torch.Tensor,
) -> float:
    """Extract the maximum absolute permanent deflection (w-direction)."""
    n_nodes = len(nodes)
    w_dofs = torch.arange(n_nodes, dtype=torch.long) * 6 + 2
    w_vals = permanent_deflection[w_dofs]
    return float(w_vals.abs().max())


# ---------------------------------------------------------------------------
# 4b. Shared incremental driver (used by both verification drivers)
# ---------------------------------------------------------------------------

def _newton_load_control(model, external, u, state, free, max_iter: int = 25):
    """One load step: Newton-Raphson from a committed state.

    Returns ``(u, state, converged)``.  The state is only advanced when the
    residual norm falls below ``1e-8`` relative to the load norm.
    """
    u_trial = u.clone()
    state_trial = state
    ref = max(1.0, float(torch.linalg.vector_norm(external[free])))
    for _ in range(max_iter):
        fi, kt, _, new_state = assemble_layered_shell4(
            model, u_trial, state_trial, geometric=False)
        residual = external - fi
        residual_free = residual[free]
        if float(torch.linalg.vector_norm(residual_free)) < 1e-8 * ref:
            return u_trial, new_state, True
        k_free = kt[free[:, None], free]
        try:
            delta = torch.linalg.solve(
                k_free.double(), residual_free.double()).to(u_trial.dtype)
        except RuntimeError:
            return u_trial, state_trial, False
        if not torch.isfinite(delta).all():
            return u_trial, state_trial, False
        u_trial = u_trial.clone()
        u_trial[free] += delta
        state_trial = new_state
    return u_trial, state_trial, False


def _any_point_yielded(model, state) -> bool:
    """True if any layer of any integration point has plastic strain."""
    for elem_states in state.points:
        for gp in elem_states:
            for layer_state in gp:
                if float(layer_state.alpha) > 0.0:
                    return True
    return False


# ---------------------------------------------------------------------------
# 5. Elastic–plastic transition verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElasticPlasticTransitionResult:
    """Result of elastic-plastic transition verification."""
    case_name: str
    analytical_yield_load: float
    fe_yield_load: float
    relative_error: float
    mesh_info: dict
    passed: bool
    detected: bool = True
    load_resolution: float = 0.0   # absolute resolution of fe_yield_load


def verify_elastic_plastic_transition(
    a: float, b: float, t: float,
    young: float, poisson: float,
    yield_stress: float, hardening: float,
    nx: int = 16, ny: int = 8,
    loading: str = "uniaxial",
    patch_x0: float = 0.0, patch_x1: float = 0.0,
    patch_y0: float = 0.0, patch_y1: float = 0.0,
    refine: int = 14,
) -> ElasticPlasticTransitionResult:
    """Verify the elastic-plastic transition point against analytical yield load.

    The analytical yield load is computed from the elastic stress distribution
    and the yield criterion.  For uniaxial compression of a simply-supported
    plate, the yield load factor is ``sigma_y / sigma_applied``.

    The FE yield load is found by incrementally loading the plate and
    detecting the first load step where any integration point yields.

    Parameters
    ----------
    a, b, t : float
        Plate geometry [m].
    young, poisson : float
        Elastic constants.
    yield_stress, hardening : float
        Plasticity parameters [Pa].
    nx, ny : int
        Mesh resolution.
    loading : str
        ``"uniaxial"`` or ``"lateral"`` (patch load).
    patch_* : float
        Patch load geometry (used when loading="lateral").

    Returns
    -------
    ElasticPlasticTransitionResult
    """
    if loading == "uniaxial":
        from .elastoplastic_plate_buckling import simply_supported_plate_mesh
        model, info = simply_supported_plate_mesh(
            a, b, t, nx, ny, young, poisson, yield_stress, hardening)
        reference_load = info['reference_load']
        fixed_dofs = info['fixed_dofs']
        # Analytical yield load: for uniaxial compression, the applied stress
        # at yield is sigma_y. The reference load applies unit stress (1 Pa).
        analytical_yield_load = yield_stress  # load factor = sigma_y / 1 Pa
    elif loading == "lateral":
        model, info = patch_load_plate_mesh(
            a, b, t, nx, ny, young, poisson, yield_stress, hardening,
            patch_x0, patch_x1, patch_y0, patch_y1)
        reference_load = info['reference_load']
        fixed_dofs = info['fixed_dofs']
        # For lateral pressure the analytical first-yield pressure comes from
        # the Navier (double-sine) elastic solution for the same plate: an
        # independent reference that does not use the FE mesh.  The previous
        # version used a guess alpha = 0.75 * aspect**2 (aspect<=1) / 0.75
        # (aspect>1) with no source; for a square plate that is 2.6x too high
        # (correct coefficient 0.2874 at nu = 0.3, i.e. M = 0.0479 p a^2).
        analytical_yield_load = elastic_first_yield_pressure(
            a, b, t, yield_stress, poisson)
    else:
        raise ValueError(f"loading must be 'uniaxial' or 'lateral', got '{loading}'")

    # Incremental loading to find the yield point
    ndof = model.n_dofs
    fixed = fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    mask = torch.ones(ndof, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()

    u = torch.zeros(ndof, dtype=model.nodes.dtype, device=model.nodes.device)
    state = LayeredShell4State.virgin(model)

    # Load incrementally to bracket the first-yield load, then bisect.
    # NOTE (audit GAP-PP-3-ICE-STRUCT-PLASTIC-SET): the previous version
    # returned the *step* load at which yield was first seen, so the reported
    # error could never exceed half a step (1.5 * analytical / 50 = 3 %).
    # The assertion "error < 5 %" was therefore satisfied by the step grid
    # itself, independently of the FE accuracy.  The bisection below resolves
    # the transition load to ``step / 2**refine`` (~0.02 % with the default).
    n_steps = 50
    step_load = analytical_yield_load * 1.5 / n_steps
    fe_yield_load = None
    last_ok = 0.0
    bracket = None
    for step in range(1, n_steps + 1):
        load_factor = step * step_load
        external = reference_load * load_factor
        u_new, state_new, converged = _newton_load_control(
            model, external, u, state, free, max_iter=25)

        if not converged:
            # Cannot even reach the elastic solution: nothing to report.
            bracket = None
            break
        if _any_point_yielded(model, state_new):
            # Keep the last *elastic* state as the committed state: below yield
            # the response is path independent, so every bisection trial can be
            # restarted from it.  (Committing the yielded state here and then
            # loading it *down* would solve the wrong incremental problem.)
            bracket = (last_ok, load_factor)
            break
        u, state, last_ok = u_new, state_new, load_factor

    if bracket is not None:
        lo, hi = bracket
        for _ in range(refine):
            mid = 0.5 * (lo + hi)
            u_mid, state_mid, conv = _newton_load_control(
                model, reference_load * mid, u, state, free, max_iter=25)
            if conv and _any_point_yielded(model, state_mid):
                hi = mid
            else:
                lo = mid
        fe_yield_load = hi
        step_resolution = step_load / 2.0 ** refine
    else:
        fe_yield_load = None
        step_resolution = step_load

    if fe_yield_load is None:
        # No yield detected: report an explicit failure instead of a number
        # that happens to sit on the reference.
        return ElasticPlasticTransitionResult(
            case_name=f"ep_transition_{loading}_{nx}x{ny}",
            analytical_yield_load=analytical_yield_load,
            fe_yield_load=float('nan'),
            relative_error=float('inf'),
            mesh_info={'nx': nx, 'ny': ny, 'a': a, 'b': b, 't': t,
                       'loading': loading, 'reason': 'no yield detected'},
            passed=False,
            detected=False, load_resolution=step_resolution,
        )

    rel_error = abs(fe_yield_load - analytical_yield_load) / analytical_yield_load

    return ElasticPlasticTransitionResult(
        case_name=f"ep_transition_{loading}_{nx}x{ny}",
        analytical_yield_load=analytical_yield_load,
        fe_yield_load=fe_yield_load,
        relative_error=rel_error,
        mesh_info={'nx': nx, 'ny': ny, 'a': a, 'b': b, 't': t, 'loading': loading},
        passed=rel_error < 0.05,  # 5% tolerance
        detected=True, load_resolution=step_resolution,
    )


# ---------------------------------------------------------------------------
# 6. Plastic limit load verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlasticLimitLoadResult:
    """Result of plastic limit load verification."""
    case_name: str
    ilyushin_limit: float
    fe_limit: float
    relative_error: float
    mesh_info: dict
    passed: bool
    applicable: bool = True      # False = no limit point inside the scan
    reason: str = ""
    load_resolution: float = 0.0  # absolute resolution of fe_limit


def verify_plastic_limit_load(
    a: float, b: float, t: float,
    young: float, poisson: float,
    yield_stress: float, hardening: float,
    nx: int = 16, ny: int = 8,
    loading: str = "uniaxial",
    patch_x0: float = 0.0, patch_x1: float = 0.0,
    patch_y0: float = 0.0, patch_y1: float = 0.0,
    refine: int = 10,
    n_steps: int = 80,
) -> PlasticLimitLoadResult:
    """Verify the plastic limit load against a reference (upper-bound) solution.

    Reference
    ---------
    * ``loading="lateral"``: :func:`yield_line_limit_pressure` (virtual-work
      yield-line collapse pressure of the simply-supported plate, the minimum
      over two admissible mechanism families).
    * ``loading="uniaxial"``: ``sigma_y`` (full-plastic membrane capacity).
      This case has **no limit point** in the present model: the assembly is
      small-strain with ``geometric=False``, so there is no geometric
      softening, and an isotropically hardening material keeps carrying load
      far past ``sigma_y``.  Consequently the "highest converged load step"
      is not a limit load and is not reported as one -- if the scan runs to
      its end without loss of convergence the result is returned with
      ``applicable=False``.  Use
      :func:`verify_elastic_plastic_transition` for the uniaxial case, or
      :mod:`tensorfem.elastoplastic_plate_buckling` (arc-length + geometric
      stiffness) if a genuine limit point is required.

    Method
    ------
    Incremental load control with Newton-Raphson.  The scan brackets the
    limit point between the last converged factor and the first factor at
    which Newton fails, then bisects that bracket ``refine`` times.  The
    resolution of ``fe_limit`` is therefore ``gap / 2**refine`` -- without
    the bisection the reported error could never exceed half a load step
    (with the default ``n_steps = 80`` and a 2x scan range that is 2.5 % of
    the reference), so a "error < 10 %" assertion would have been satisfied
    by the step grid alone, independently of the FE accuracy.

    Honesty statement
    -----------------
    The limit point is detected as a *loss of Newton convergence at fixed
    load* (classic load-control limit-point symptom), not by arc-length
    continuation: there is no branch switching, and no post-limit path is
    traced.  Perfect plasticity under pure bending can fail to converge
    slightly before the true limit load, which biases ``fe_limit`` low.
    """
    if loading == "lateral":
        reference = yield_line_limit_pressure(a, b, yield_stress, t)
        model, info = patch_load_plate_mesh(
            a, b, t, nx, ny, young, poisson, yield_stress, hardening,
            patch_x0, patch_x1, patch_y0, patch_y1)
        scan_top = 2.0 * reference
        applicable_case = True
        reason = ""
    elif loading == "uniaxial":
        from .elastoplastic_plate_buckling import simply_supported_plate_mesh
        reference = yield_stress
        model, info = simply_supported_plate_mesh(
            a, b, t, nx, ny, young, poisson, yield_stress, hardening)
        scan_top = 2.0 * reference
        applicable_case = False
        reason = ("uniaxial small-strain plate without geometric stiffness has "
                  "no limit point short of full plasticity; the scan is "
                  "reported as not applicable")
    else:
        raise ValueError(f"loading must be 'uniaxial' or 'lateral', got '{loading}'")

    reference_load = info['reference_load']
    fixed_dofs = info['fixed_dofs']

    ndof = model.n_dofs
    device = model.nodes.device
    fixed = fixed_dofs.to(device=device, dtype=torch.long)
    mask = torch.ones(ndof, dtype=torch.bool, device=device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()

    u = torch.zeros(ndof, dtype=model.nodes.dtype, device=device)
    state = LayeredShell4State.virgin(model)

    step_load = scan_top / n_steps
    last_ok = 0.0
    bracket = None
    for step in range(1, n_steps + 1):
        load_factor = step * step_load
        u_new, state_new, converged = _newton_load_control(
            model, reference_load * load_factor, u, state, free, max_iter=30)
        if not converged:
            bracket = (last_ok, load_factor)
            break
        u, state = u_new, state_new
        last_ok = load_factor

    if bracket is None:
        # No limit point inside the scan -- do NOT fall back to the reference
        # value: that made the check pass with zero error whenever the FE
        # failed to find (or simply did not reach) a limit point.
        return PlasticLimitLoadResult(
            case_name=f"plastic_limit_{loading}_{nx}x{ny}",
            ilyushin_limit=reference,
            fe_limit=float('nan'),
            relative_error=float('inf'),
            mesh_info={'nx': nx, 'ny': ny, 'a': a, 'b': b, 't': t,
                       'loading': loading, 'reason': 'no limit point in scan'},
            passed=False, applicable=False,
            reason=(reason or "no limit point detected within the scan range "
                    f"[0, {scan_top:.4g}]; the highest converged step "
                    f"({last_ok:.4g}) is not a limit load"),
            load_resolution=step_load,
        )

    lo, hi = bracket
    for _ in range(refine):
        mid = 0.5 * (lo + hi)
        _, _, conv = _newton_load_control(
            model, reference_load * mid, u, state, free, max_iter=30)
        if conv:
            lo = mid
        else:
            hi = mid
    fe_limit = lo
    resolution = (bracket[1] - bracket[0]) / 2.0 ** refine

    rel_error = abs(fe_limit - reference) / reference
    return PlasticLimitLoadResult(
        case_name=f"plastic_limit_{loading}_{nx}x{ny}",
        ilyushin_limit=reference,
        fe_limit=fe_limit,
        relative_error=rel_error,
        mesh_info={'nx': nx, 'ny': ny, 'a': a, 'b': b, 't': t,
                   'loading': loading},
        passed=rel_error < 0.10,  # 10% tolerance
        applicable=applicable_case, reason=reason, load_resolution=resolution,
    )


# ---------------------------------------------------------------------------
# 7. Mesh convergence study
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeshConvergenceResult:
    """Result of a mesh convergence study."""
    case_name: str
    meshes: list[dict]
    quantity: str
    values: list[float]
    relative_errors: list[float]
    convergence_rate: float | None
    passed: bool


def verify_mesh_convergence(
    a: float, b: float, t: float,
    young: float, poisson: float,
    yield_stress: float, hardening: float,
    mesh_configs: Sequence[tuple[int, int]] = ((8, 4), (16, 8), (32, 16)),
    loading: str = "uniaxial",
    quantity: str = "max_deflection",
) -> MeshConvergenceResult:
    """Three-grid convergence study for permanent deflection or limit load.

    Parameters
    ----------
    a, b, t : float
        Plate geometry [m].
    young, poisson : float
        Elastic constants.
    yield_stress, hardening : float
        Plasticity parameters [Pa].
    mesh_configs : sequence of (nx, ny)
        Three mesh resolutions.
    loading : str
        ``"uniaxial"`` or ``"lateral"``.
    quantity : str
        ``"max_deflection"`` or ``"limit_load"``.

    Returns
    -------
    MeshConvergenceResult
    """
    if len(mesh_configs) < 2:
        raise ValueError("at least 2 mesh configurations required")

    meshes = []
    values = []
    relative_errors = []

    for nx, ny in mesh_configs:
        if loading == "uniaxial":
            from .elastoplastic_plate_buckling import simply_supported_plate_mesh
            model, info = simply_supported_plate_mesh(
                a, b, t, nx, ny, young, poisson, yield_stress, hardening)
        else:
            model, info = patch_load_plate_mesh(
                a, b, t, nx, ny, young, poisson, yield_stress, hardening,
                0.25 * a, 0.75 * a, 0.25 * b, 0.75 * b)

        reference_load = info['reference_load']
        fixed_dofs = info['fixed_dofs']

        ndof = model.n_dofs
        fixed = fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
        mask = torch.ones(ndof, dtype=torch.bool, device=model.nodes.device)
        mask[fixed] = False
        free = torch.nonzero(mask).flatten()

        u = torch.zeros(ndof, dtype=model.nodes.dtype, device=model.nodes.device)
        state = LayeredShell4State.virgin(model)

        # Load to a fixed fraction of yield
        target_load = yield_stress * 0.8 if loading == "uniaxial" else yield_stress * t ** 2 / b ** 2
        n_steps = 20
        for step in range(1, n_steps + 1):
            load_factor = step / n_steps * target_load
            external = reference_load * load_factor

            u_trial = u.clone()
            state_trial = state
            for iteration in range(20):
                fi, kt, _, new_state = assemble_layered_shell4(
                    model, u_trial, state_trial, geometric=False)
                residual = external - fi
                residual_free = residual[free]
                res_norm = float(torch.linalg.vector_norm(residual_free))

                if res_norm < 1e-8 * max(1.0, float(torch.linalg.vector_norm(external[free]))):
                    u = u_trial
                    state = new_state
                    break

                K_free = kt[free[:, None], free]
                try:
                    delta = torch.linalg.solve(K_free.double(), residual_free.double()).to(u.dtype)
                except RuntimeError:
                    break
                u_trial = u_trial.clone()
                u_trial[free] += delta
                state_trial = new_state

        if quantity == "max_deflection":
            n_nodes = len(model.nodes)
            w_dofs = torch.arange(n_nodes, dtype=torch.long) * 6 + 2
            value = float(u[w_dofs].abs().max())
        elif quantity == "limit_load":
            # AUDIT FIX: this used to be ``value = target_load``, i.e. the same
            # hardcoded number for every mesh.  Every relative error was then
            # identically zero, so "convergence" was satisfied by construction.
            # The limit load is now actually computed on each grid.
            res = verify_plastic_limit_load(
                a, b, t, young, poisson, yield_stress, hardening,
                nx=nx, ny=ny, loading=loading,
                patch_x0=0.25 * a, patch_x1=0.75 * a,
                patch_y0=0.25 * b, patch_y1=0.75 * b)
            value = res.fe_limit if res.applicable else float("nan")
        else:
            raise ValueError(f"unknown quantity: {quantity}")

        meshes.append({'nx': nx, 'ny': ny, 'n_elements': nx * ny})
        values.append(value)

    # Relative errors against the finest grid.  NOTE: the finest entry is 0 by
    # construction, so it can never be used to test convergence -- the pass
    # criterion below uses the *successive* difference instead (audit fix; the
    # old criterion ``relative_errors[-1] < 0.05`` was trivially true).
    finite = all(isinstance(v, float) and math.isfinite(v) for v in values)
    ref_value = values[-1]
    for v in values:
        if math.isfinite(ref_value) and ref_value != 0 and math.isfinite(v):
            relative_errors.append(abs(v - ref_value) / abs(ref_value))
        else:
            relative_errors.append(float("nan"))

    # Successive-grid relative change (self-convergence), and its ratio.
    successive = []
    for i in range(len(values) - 1):
        denom = abs(values[i + 1]) if values[i + 1] else float("nan")
        successive.append(abs(values[i + 1] - values[i]) / denom)

    convergence_rate = None
    if (finite and len(successive) >= 2 and successive[0] > 0
            and successive[1] > 0 and successive[0] / successive[1] > 0):
        # refinement ratio r = h_i / h_{i+1} (here 2 for the default grids)
        h = [math.sqrt(a * b / (nx * ny)) for nx, ny in mesh_configs]
        ratio = math.log(successive[0] / successive[1]) / math.log(h[0] / h[1])
        convergence_rate = ratio

    passed = (finite and len(successive) >= 2
              and successive[-1] < 0.05
              and values[-1] > values[-2] * 0.0)  # kept explicit for clarity

    return MeshConvergenceResult(
        case_name=f"convergence_{quantity}_{loading}",
        meshes=meshes,
        quantity=quantity,
        values=values,
        relative_errors=relative_errors,
        convergence_rate=convergence_rate,
        passed=passed,
    )
