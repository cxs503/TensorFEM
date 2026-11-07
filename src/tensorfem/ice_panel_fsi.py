"""Elastic hull panel under ice contact loading: two-way flexibility feedback.

Physical scope
--------------
A stiffened-plate **hull panel bay** (rectangular plate, four edges either
clamped at the frames/stringers or simply supported) is loaded by a row of
crushable ice blocks advancing against it.  The ice side is a discrete-element
(DEM) idealisation: independent blocks, each with a unilateral crush spring in
series with the panel compliance, and each block additionally limited by the
bending capacity of the ice sheet that carries it.  The panel is a real
finite-element Mindlin-Reissner plate (:func:`tensorfem.plate.q4_mindlin_stiffness`,
selective reduced integration), condensed once into a contact compliance so the
coupled time integration stays cheap.

Two feedback loops are resolved every time step:

1. **Mechanical (load) feedback.**  Contact penetration is measured from the
   *deformed* panel surface, ``delta_i = d_i - w_i``, with ``d_i`` the ice
   advance and ``w_i`` the panel deflection under the current contact forces.
   A flexible panel therefore reduces the penetration at the points that
   deflect most and stores part of the impact energy as elastic strain, which
   lowers the peak contact load.
2. **Failure-mode feedback.**  The panel reaction is generally non-uniform
   along the contact line (highest where the panel is locally stiffest).  The
   ice sheet is a plate on a buoyancy (Winkler) foundation; solved as a
   periodic strip along the contact line it develops a bending moment
   ``M(y) = -D_ice w''``.  Where ``6M/h^2`` reaches the ice flexural strength
   the ice sheet fails in bending and the local pressure is capped by the
   bending capacity.  A rigid panel produces a uniform reaction
   (``M == 0``), so it fails by crushing; a flexible panel produces a
   non-uniform reaction, so more of the ice fails by bending and the load
   drops.

Honest limitations
------------------
* The ice is *not* a calibrated rate-dependent sea-ice constitutive model:
  crushing is elastic-perfectly-plastic with a velocity-dependent pressure cap,
  and the blocks are uncoupled in translation (a "block row", not a
  self-consistent granular assembly).  There is no ``tensordem`` dependency in
  this environment, so the DEM is implemented here against a small documented
  contract (``IceBlockState``) that a real DEM back end can replace.
* The panel is linear elastic and is treated as **quasi-static** (no panel
  inertia).  :meth:`HullPanel.rayleigh_frequency` reports the fundamental
  frequency so the assumption can be checked against the contact duration.
* No hydrostatic/hydrodynamic added mass and no water pressure on the outer
  shell; the buoyancy foundation appears only in the ice bending model.
* Absolute load magnitudes are scenario-dependent; the certified quantity is
  the *relative* rigid-versus-elastic difference and the energy closure, not
  an experimentally validated ice load.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .plate import q4_mindlin_stiffness, _shape

D = torch.float64


# --------------------------------------------------------------------------
# panel geometry and finite-element model
# --------------------------------------------------------------------------
def rect_mesh(span_x: float, span_y: float, nx: int, ny: int, *,
              dtype=D) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured Q4 mesh on ``[0, span_x] x [0, span_y]``.

    Node ``(i, j)`` lives at index ``j*(nx+1)+i``; element connectivity is
    counter-clockwise, matching :mod:`tensorfem.plate`.
    """
    for name, value in (("span_x", span_x), ("span_y", span_y)):
        if not value > 0:
            raise ValueError(f"{name} must be positive")
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive integers")
    xs = torch.linspace(0.0, span_x, nx + 1, dtype=dtype)
    ys = torch.linspace(0.0, span_y, ny + 1, dtype=dtype)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elems.append((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1))
    return nodes, torch.tensor(elems, dtype=torch.long)


@dataclass(frozen=True)
class PanelConfig:
    """Rectangular hull-panel bay."""

    span_x: float = 0.75
    span_y: float = 2.4
    thickness: float = 0.025
    young: float = 206.0e9
    poisson: float = 0.30
    density: float = 7850.0
    nx: int = 8
    ny: int = 16
    boundary: str = "clamped"          # "clamped" | "simply_supported"

    def validate(self) -> None:
        if self.boundary not in ("clamped", "simply_supported"):
            raise ValueError("boundary must be 'clamped' or 'simply_supported'")
        for name in ("span_x", "span_y", "thickness", "young", "density"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson out of range")
        if self.nx < 4 or self.ny < 4:
            raise ValueError("need at least a 4x4 mesh")

    @property
    def flexural_rigidity(self) -> float:
        return (self.young * self.thickness ** 3
                / (12.0 * (1.0 - self.poisson ** 2)))


class HullPanel:
    """Linear Mindlin plate condensed to a contact compliance operator."""

    def __init__(self, config: PanelConfig, contacts: torch.Tensor):
        config.validate()
        self.config = config
        self.nodes, self.elems = rect_mesh(config.span_x, config.span_y,
                                          config.nx, config.ny)
        self.contacts = torch.as_tensor(contacts, dtype=D)
        if self.contacts.ndim != 2 or self.contacts.shape[1] != 2:
            raise ValueError("contacts must be an (n, 2) array")
        n_node = self.nodes.shape[0]
        ndof = 3 * n_node
        self.ndof = ndof

        K = torch.zeros((ndof, ndof), dtype=D)
        for conn in self.elems:
            xy = self.nodes[conn]
            ke = q4_mindlin_stiffness(xy, config.young, config.poisson,
                                      config.thickness)
            dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2),
                               dim=1).reshape(-1)
            K[dofs[:, None], dofs] += ke
        self.stiffness = K

        self.fixed, self.free = self._boundary_dofs()
        self.K_ff = K[self.free[:, None], self.free]
        self._chol = torch.linalg.cholesky(self.K_ff)

        # consistent mass on the transverse dofs-only, used only for the
        # Rayleigh frequency estimate of the quasi-static assumption
        self.mass_ww = self._consistent_mass_ww()

        self._build_contact_operators()

    # -- boundary conditions ------------------------------------------------
    def _boundary_dofs(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        tol = 1e-9 * max(cfg.span_x, cfg.span_y)
        xedge = (self.nodes[:, 0].abs() < tol) | (
            (self.nodes[:, 0] - cfg.span_x).abs() < tol)
        yedge = (self.nodes[:, 1].abs() < tol) | (
            (self.nodes[:, 1] - cfg.span_y).abs() < tol)
        edge = xedge | yedge
        e_idx = torch.nonzero(edge, as_tuple=False).flatten()
        w_edge = 3 * e_idx
        if cfg.boundary == "clamped":
            fixed = torch.cat((3 * e_idx, 3 * e_idx + 1, 3 * e_idx + 2))
        else:
            fixed = torch.cat((w_edge,
                               3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1,
                               3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2))
        fixed = fixed.unique()
        all_dofs = torch.arange(self.ndof)
        return fixed, all_dofs[~torch.isin(all_dofs, fixed)]

    # -- element lookup -----------------------------------------------------
    def locate(self, x: float, y: float):
        cfg = self.config
        hx = cfg.span_x / cfg.nx
        hy = cfg.span_y / cfg.ny
        i = min(max(int(math.floor(x / hx)), 0), cfg.nx - 1)
        j = min(max(int(math.floor(y / hy)), 0), cfg.ny - 1)
        x0, y0 = i * hx, j * hy
        xi = 2.0 * (x - x0) / hx - 1.0
        eta = 2.0 * (y - y0) / hy - 1.0
        n0 = j * (cfg.nx + 1) + i
        conn = torch.tensor([n0, n0 + 1, n0 + cfg.nx + 2, n0 + cfg.nx + 1])
        return conn, xi, eta

    def _contact_rows(self, conn, xi, eta):
        xy = self.nodes[conn]
        N, dN_nat = _shape(torch.tensor(xi, dtype=D), torch.tensor(eta, dtype=D))
        J = dN_nat.T @ xy                       # d(x)/d(xi)
        dN = dN_nat @ torch.linalg.inv(J).T     # rows: dN/dx, dN/dy
        return conn, N, dN

    def _build_contact_operators(self) -> None:
        n = self.contacts.shape[0]
        P = torch.zeros((n, self.ndof), dtype=D)    # interpolation of w
        L = torch.zeros((self.ndof, n), dtype=D)    # consistent unit loads
        Gx = torch.zeros((n, self.ndof), dtype=D)   # kappa_x
        Gy = torch.zeros((n, self.ndof), dtype=D)   # kappa_y
        for k, (x, y) in enumerate(self.contacts.tolist()):
            conn, xi, eta = self.locate(x, y)
            conn, N, dN = self._contact_rows(conn, xi, eta)
            P[k, 3 * conn] = N
            L[3 * conn, k] = N
            Gx[k, 3 * conn + 1] = dN[:, 0]
            Gy[k, 3 * conn + 2] = dN[:, 1]
        self._P, self._L, self._Gx, self._Gy = P, L, Gx, Gy
        A = self.solve_reduced(L)                   # (ndof_full, n)
        self.compliance = P @ A                     # w = R f
        self.kappa_x = Gx @ A                       # kappa_x = Sx f
        self.kappa_y = Gy @ A
        self._A = A

    def _consistent_mass_ww(self) -> torch.Tensor:
        cfg = self.config
        n_node = self.nodes.shape[0]
        M = torch.zeros((n_node, n_node), dtype=D)
        gp = 1.0 / 3.0 ** 0.5
        for conn in self.elems:
            xy = self.nodes[conn]
            me = torch.zeros((4, 4), dtype=D)
            for xif in (-gp, gp):
                for etaf in (-gp, gp):
                    N, dN_nat = _shape(torch.tensor(xif, dtype=D),
                                       torch.tensor(etaf, dtype=D))
                    J = dN_nat.T @ xy
                    me += cfg.density * cfg.thickness * torch.outer(N, N) \
                        * torch.linalg.det(J)
            M[conn[:, None], conn] += me
        return M

    # -- linear solves ------------------------------------------------------
    def solve_reduced(self, rhs: torch.Tensor) -> torch.Tensor:
        """Solve ``K_ff u_f = rhs_f`` and scatter back to full dof order."""
        rhs = torch.as_tensor(rhs, dtype=D)
        squeeze = rhs.ndim == 1
        if squeeze:
            rhs = rhs[:, None]
        out = torch.zeros((self.ndof, rhs.shape[1]), dtype=D)
        out[self.free] = torch.cholesky_solve(rhs[self.free], self._chol)
        return out[:, 0] if squeeze else out

    # -- coupled response ---------------------------------------------------
    def contact_deflection(self, forces: torch.Tensor) -> torch.Tensor:
        """Panel deflection ``w`` at the contact points for contact forces."""
        return self.compliance @ torch.as_tensor(forces, dtype=D)

    def contact_curvature(self, forces: torch.Tensor):
        f = torch.as_tensor(forces, dtype=D)
        return self.kappa_x @ f, self.kappa_y @ f

    def strain_energy(self, forces: torch.Tensor) -> torch.Tensor:
        f = torch.as_tensor(forces, dtype=D)
        return 0.5 * f @ (self.compliance @ f)

    def displacement_field(self, forces: torch.Tensor) -> torch.Tensor:
        """Full nodal solution ``(n_node, 3)`` for contact forces."""
        u = self.solve_reduced(self._L @ torch.as_tensor(forces, dtype=D))
        return u.reshape(-1, 3)

    def local_stiffness(self) -> torch.Tensor:
        """Diagonal contact stiffness ``1/R_ii`` (N/m at each contact)."""
        return 1.0 / torch.diagonal(self.compliance)

    def rayleigh_frequency(self, forces: torch.Tensor | None = None) -> float:
        """Rayleigh estimate of the fundamental panel frequency (Hz)."""
        n = self.contacts.shape[0]
        f = (torch.ones(n, dtype=D) if forces is None
             else torch.as_tensor(forces, dtype=D))
        u = self.solve_reduced(self._L @ f)
        u = u / torch.max(torch.abs(u))
        k = u @ (self.stiffness @ u)
        w_dofs = u.reshape(-1, 3)[:, 0]
        m = w_dofs @ (self.mass_ww @ w_dofs)
        return float(math.sqrt(k / m) / (2.0 * math.pi))


# --------------------------------------------------------------------------
# plate-theory references for the static verification
# --------------------------------------------------------------------------
def navier_ss_rect_centre(span_x: float, span_y: float, young: float,
                          poisson: float, thickness: float, pressure: float,
                          terms: int = 81) -> float:
    """Kirchhoff simply-supported rectangular plate, uniform load, centre.

    Navier double series; only odd ``m, n`` contribute for a uniform load.
    """
    rigidity = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    total = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            lam = (m / span_x) ** 2 + (n / span_y) ** 2
            total += (math.sin(m * math.pi / 2.0) * math.sin(n * math.pi / 2.0)
                      / (m * n * lam ** 2))
    return 16.0 * pressure / (math.pi ** 6 * rigidity) * total


def mindlin_shear_correction(span_x: float, span_y: float, young: float,
                             poisson: float, thickness: float, pressure: float,
                             shear_correction: float = 5.0 / 6.0) -> float:
    """First-term Navier transverse-shear *increment* for the same case.

    The Mindlin amplification of the first Navier mode is
    ``w * lam * D / (kappa G h)`` with ``lam = pi^2 (1/a^2 + 1/b^2)``, so the
    increment is that fraction of the Kirchhoff centre deflection.
    """
    g = young / (2.0 * (1.0 + poisson))
    rigidity = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    lam = (math.pi / span_x) ** 2 + (math.pi / span_y) ** 2
    fraction = lam * rigidity / (shear_correction * g * thickness)
    return navier_ss_rect_centre(span_x, span_y, young, poisson, thickness,
                                 pressure) * fraction


# --------------------------------------------------------------------------
# ice side
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class IceConfig:
    """First-year ice sheet properties and contact-law parameters."""

    thickness: float = 0.60
    young: float = 5.0e9
    poisson: float = 0.30
    density: float = 917.0
    compressive_strength: float = 2.5e6
    flexural_strength: float = 0.50e6
    water_density: float = 1025.0
    gravity: float = 9.81
    velocity_ref: float = 0.05
    velocity_exponent: float = 0.30
    pressure_area_exponent: float = 0.0
    contact_width: float = 0.10          # patch width along the panel normal
    velocity: float = 0.50               # ice advance rate (m/s)
    gap_amplitude: float = 0.0           # deterministic edge imperfection (m)
    compliance_length: float | None = None  # elastic length scale of the ice

    def validate(self) -> None:
        for name in ("thickness", "young", "density", "compressive_strength",
                     "flexural_strength", "contact_width", "velocity"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.pressure_area_exponent < 0:
            raise ValueError("pressure_area_exponent must be non-negative")
        if self.gap_amplitude < 0:
            raise ValueError("gap_amplitude must be non-negative")

    @property
    def flexural_rigidity(self) -> float:
        return self.young * self.thickness ** 3 / (12.0 * (1.0 - self.poisson ** 2))

    def elastic_length(self) -> float:
        """Length over which the ice sheet shortens elastically under load.

        The sheet transfers the contact force into the far field over the
        buoyancy characteristic length, so the contact elastic stiffness is
        ``E_ice / L_c`` (a compression member of length ``L_c``) rather than
        ``E_ice / h`` (the whole thickness in compression, which overstates the
        ice stiffness by ~``L_c/h``).
        """
        if self.compliance_length is not None:
            return self.compliance_length
        return self.characteristic_length

    @property
    def characteristic_length(self) -> float:
        """``L_c = (D_i / (rho_w g))^(1/4)``."""
        return (self.flexural_rigidity
                / (self.water_density * self.gravity)) ** 0.25

    def crush_pressure(self, velocity=None) -> torch.Tensor:
        """Velocity-dependent crushing pressure cap (ductile/brittle trend)."""
        v = torch.as_tensor(self.velocity if velocity is None else velocity,
                            dtype=D)
        v = torch.clamp(torch.abs(v), min=1.0e-3)
        ratio = torch.clamp((self.velocity_ref / v) ** self.velocity_exponent,
                            0.5, 2.0)
        return self.compressive_strength * ratio

    def pressure_area_factor(self, area: torch.Tensor) -> torch.Tensor:
        ref = self.contact_width * self.contact_width
        return (torch.as_tensor(area, dtype=D) / ref) ** (-self.pressure_area_exponent)


def ice_bending_operator(ice: IceConfig, positions: torch.Tensor,
                         samples_per_patch: int = 8) -> torch.Tensor:
    """Map contact pressures on the ice edge to ice bending stress ``sigma = B p``.

    The ice sheet is a plate on a buoyancy (Winkler) foundation,

    ``D_ice d4w/dy4 + rho_w g w = p(y)``,   ``M = -D_ice d2w/dy2``,
    ``sigma = 6 M / h^2``.

    Along the contact line the sheet is continuous, so the strip is solved
    **periodically** over one panel-bay pitch (the neighbouring bays see the
    same panel reaction - a documented idealisation).  Each DEM block loads the
    sheet as a uniform patch of width ``span_y/n`` (a continuous top-hat, not a
    point load, so the operator is independent of the FFT resolution).  A
    uniform pressure field therefore produces ``w = const``, ``M = 0``,
    ``sigma = 0``: a rigidly reacting panel generates no ice bending, and only
    the *variation* of the panel reaction bends the ice.

    Returns the dense ``n x n`` matrix ``B`` with ``B[i, j]`` the (signed,
    patch-averaged) bending stress at block ``i`` per unit pressure at block
    ``j``; the physical stress is ``sigma_i = |(B p)_i|``.
    """
    positions = torch.as_tensor(positions, dtype=D)
    n = positions.shape[0]
    if n < 2:
        raise ValueError("need at least two contact patches")
    pitch = float(positions[1] - positions[0])
    span = float(positions.max() - positions.min()) + pitch
    if pitch <= 0:
        raise ValueError("contact positions must be strictly increasing")
    sub = max(2, int(samples_per_patch))
    segments = n * sub
    L = span
    dj = ice.flexural_rigidity
    k = ice.water_density * ice.gravity
    kappa = 2.0 * math.pi * torch.fft.fftfreq(segments, d=L / segments, dtype=D)
    response = dj * kappa ** 2 / (dj * kappa ** 4 + k)     # p_hat -> M_hat
    patch = [(torch.arange(i * sub, (i + 1) * sub) % segments) for i in range(n)]
    B = torch.zeros((n, n), dtype=D)
    load = torch.zeros(segments, dtype=D)
    scale = 6.0 / ice.thickness ** 2
    for j in range(n):
        load.zero_()
        load[patch[j]] = 1.0
        moment = torch.fft.ifft(torch.fft.fft(load) * response).real
        # signed patch average keeps the map linear: a uniform pressure field
        # gives exactly zero bending stress
        B[:, j] = torch.tensor([float(torch.mean((scale * moment)[q]))
                                for q in patch], dtype=D)
    return B


def bending_capped_pressure(B: torch.Tensor, pressure: torch.Tensor,
                            flexural_strength: float, iterations: int = 24,
                            relaxation: float = 0.6) -> torch.Tensor:
    """Cap contact pressure where the ice sheet would exceed ``sigma_f``.

    The bending stress is linear in the pressure, ``sigma = B p``, so the cap
    ``p_i <- p_i * sigma_f / sigma_i`` is a fixed-point iteration; the returned
    pressure is the largest field that keeps ``sigma <= sigma_f``.
    """
    q = torch.clamp(torch.as_tensor(pressure, dtype=D), min=0.0).clone()
    if float(torch.max(q)) <= 0.0:
        return q
    for _ in range(iterations):
        sig = torch.abs(B @ q)
        over = sig > flexural_strength
        if not bool(torch.any(over)):
            break
        scale = torch.where(over, flexural_strength / torch.clamp(sig, min=1e-30),
                            torch.ones_like(sig))
        q = torch.clamp(q * ((1.0 - relaxation) + relaxation * scale), min=0.0)
    # final projection: a local cap can leave a residual overshoot because the
    # moment field is nonlocal, so finish with one global scaling
    sig = torch.abs(B @ q)
    worst = float(torch.max(sig))
    if worst > flexural_strength:
        q = q * (flexural_strength / worst)
    return q

# --------------------------------------------------------------------------
# two-way coupled simulation
# --------------------------------------------------------------------------
_PANEL_CACHE: dict = {}


def cached_panel(config: PanelConfig, contacts: torch.Tensor) -> HullPanel:
    """Build (or reuse) the condensed panel for a geometry/contact layout."""
    key = (tuple(sorted(config.__dict__.items())),
           tuple(torch.as_tensor(contacts, dtype=D).flatten().tolist()))
    if key not in _PANEL_CACHE:
        _PANEL_CACHE[key] = HullPanel(config, contacts)
    return _PANEL_CACHE[key]
@dataclass(frozen=True)
class CoupledConfig:
    """Ice-block row against a hull-panel bay."""

    panel: PanelConfig = field(default_factory=PanelConfig)
    ice: IceConfig = field(default_factory=IceConfig)
    blocks: int = 12
    block_mass: float = 233.0        # kg per DEM block (ice edge fragment)
    rigid: bool = False
    steps: int = 400
    dt: float | None = None
    coupling_iterations: int = 10
    bending_cap: bool = True

    def validate(self) -> None:
        self.panel.validate()
        self.ice.validate()
        if self.blocks < 2:
            raise ValueError("need at least two ice blocks")
        if not self.block_mass > 0:
            raise ValueError("block_mass must be positive")
        if self.coupling_iterations < 1:
            raise ValueError("coupling_iterations must be >= 1")

    def contact_positions(self) -> torch.Tensor:
        dy = self.panel.span_y / self.blocks
        y = (torch.arange(self.blocks, dtype=D) + 0.5) * dy
        x = torch.full_like(y, 0.5 * self.panel.span_x)
        return torch.stack((x, y), dim=1)

    def block_area(self) -> float:
        return self.ice.contact_width * self.panel.span_y / self.blocks

    def stable_dt(self) -> float:
        stiffness = self.ice.young * self.block_area() / self.ice.thickness
        omega = math.sqrt(stiffness / self.block_mass)
        return 0.02 * 2.0 * math.pi / omega


def run_ice_panel_case(cfg: CoupledConfig) -> dict:
    """Run one coupled ice-block / panel impact and return the audit record.

    The ice blocks obey an elastic-perfectly-plastic crush law with an
    accumulated plastic penetration (a 1D radial return): loading at
    ``E_ice/h`` up to the velocity-dependent pressure cap ``p_c``, elastic
    unloading from the cap, so crushing dissipates energy instead of bouncing
    the ice back elastically.

    Every time step resolves both feedback loops by fixed-point iteration
    (``coupling_iterations`` sweeps of penetration -> crush pressure -> ice
    bending cap -> panel deflection -> penetration).  Block dynamics use
    symplectic Euler and the panel is quasi-static (no panel inertia); the
    panel's Rayleigh frequency is reported so the assumption can be checked
    against the contact duration.
    """
    cfg.validate()
    contacts = cfg.contact_positions()
    panel = cached_panel(cfg.panel, contacts)
    n = cfg.blocks
    area = cfg.block_area()
    ice = cfg.ice

    B = ice_bending_operator(ice, contacts[:, 1])
    gap = (ice.gap_amplitude
           * torch.cos(2.0 * math.pi * (torch.arange(n, dtype=D) + 0.5) / n)
           if ice.gap_amplitude > 0 else torch.zeros(n, dtype=D))

    compliance = (torch.zeros_like(panel.compliance) if cfg.rigid
                  else panel.compliance)
    dt = cfg.dt if cfg.dt else cfg.stable_dt()
    p_cap = ice.crush_pressure(torch.tensor(ice.velocity, dtype=D)) \
        * ice.pressure_area_factor(torch.tensor(area, dtype=D))
    crush_stiffness = ice.young / ice.elastic_length()

    # exact coupled gain: p = (I + K A R)^-1 K A (d_eff), so the block/panel
    # interaction no longer relies on a divergent fixed point
    coupled = torch.eye(n, dtype=D) + crush_stiffness * area * compliance
    coupled_inv = torch.linalg.inv(coupled)
    gain = crush_stiffness * area

    v0 = ice.velocity
    adv = torch.zeros(n, dtype=D)
    vel = torch.full((n,), v0, dtype=D)
    plastic = torch.zeros(n, dtype=D)
    force = torch.zeros(n, dtype=D)
    w = torch.zeros(n, dtype=D)
    pressure = torch.zeros(n, dtype=D)
    pressure_prev = torch.zeros(n, dtype=D)
    delta_prev = torch.zeros(n, dtype=D)
    crush_prev = torch.zeros(n, dtype=D)

    plastic_work = 0.0
    bent_seen = torch.zeros(n, dtype=torch.bool)
    crush_seen = torch.zeros(n, dtype=torch.bool)
    peak = 0.0
    peak_state = None
    times, total_force, max_deflection = [], [], []
    ke0 = 0.5 * cfg.block_mass * n * v0 ** 2
    steps_done = 0

    for step in range(cfg.steps):
        # 1D radial return from the previous converged state: the ice keeps an
        # irreversible offset whenever the previous pressure sat on a plateau
        # (the crushing cap or the ice-bending limit) instead of the elastic
        # branch p = E_ice * delta_el / h.
        on_plateau = (crush_stiffness
                      * torch.clamp(delta_prev - plastic, min=0.0)
                      > p_cap * (1.0 + 1.0e-12))
        plastic_new = torch.where(
            on_plateau,
            torch.maximum(plastic,
                          torch.clamp(delta_prev - p_cap / crush_stiffness,
                                      min=0.0)),
            plastic)
        plastic_work += float(area * torch.sum(
            torch.minimum(crush_prev, p_cap)
            * torch.clamp(plastic_new - plastic, min=0.0)))
        plastic = plastic_new
        applied = torch.zeros(n, dtype=D)
        for _ in range(cfg.coupling_iterations):
            delta = torch.clamp(adv - w - gap, min=0.0)
            elastic = torch.clamp(delta - plastic, min=0.0)
            # coupled_inv maps penetration to contact force; /area gives pressure
            trial = torch.clamp((coupled_inv @ (gain * elastic)) / area, min=0.0)
            crush_pressure = torch.minimum(trial, p_cap)
            if cfg.bending_cap:
                applied = bending_capped_pressure(B, crush_pressure,
                                                  ice.flexural_strength)
                bent_seen |= (crush_pressure - applied) > 1.0e3
            else:
                applied = crush_pressure
            force = area * applied
            w = torch.zeros(n, dtype=D) if cfg.rigid else compliance @ force
        pressure = applied
        crush_seen |= torch.isclose(crush_pressure, p_cap, rtol=0.0, atol=1.0)
        delta = torch.clamp(adv - w - gap, min=0.0)
        d_delta = torch.clamp(delta - delta_prev, min=0.0)
        pressure_prev = pressure.clone()
        delta_prev = delta.clone()
        crush_prev = crush_pressure.clone()
        steps_done = step + 1

        total = float(torch.sum(force))
        if total > peak:
            peak = total
            peak_state = {"force": force.clone(), "pressure": pressure.clone(),
                          "deflection": w.clone(), "advance": adv.clone(),
                          "delta": delta.clone(), "time": step * dt}
        times.append(step * dt)
        total_force.append(total)
        max_deflection.append(float(torch.max(w)))

        vel = vel - force / cfg.block_mass * dt
        adv = adv + vel * dt
        separated = float(torch.max(force)) == 0.0 and step > 10
        if separated and float(torch.max(torch.abs(vel))) < 1.0e-9:
            break
        if separated and float(torch.min(vel)) > 0.0 and step > 40:
            break

        elastic_end = torch.clamp(delta - plastic, min=0.0)
        elastic_energy = float(area * torch.sum(
            0.5 * crush_stiffness * elastic_end ** 2))

    strain_energy = float(panel.strain_energy(force))
    ke_end = 0.5 * cfg.block_mass * float(torch.sum(vel ** 2))
    block_volume = area * ice.thickness
    bend_energy = float(torch.sum(bent_seen.to(D))) * (
        ice.flexural_strength ** 2 / (2.0 * ice.young) * block_volume)
    residual = (ke0 - ke_end - strain_energy - elastic_energy - plastic_work
                - bend_energy)
    n_bent = int(torch.sum(bent_seen))
    n_crush = int(torch.sum(crush_seen))
    return {
        "config": cfg,
        "dt": dt,
        "steps": steps_done,
        "duration": steps_done * dt,
        "time": torch.tensor(times, dtype=D),
        "total_force": torch.tensor(total_force, dtype=D),
        "max_deflection": torch.tensor(max_deflection, dtype=D),
        "peak_force": peak,
        "peak_state": peak_state,
        "panel": panel,
        "compliance": compliance,
        "bending_operator": B,
        "bending_cap": cfg.bending_cap,
        "bent_blocks": n_bent,
        "crushed_blocks": n_crush,
        "bending_fraction": (n_bent / n if (cfg.bending_cap and not cfg.rigid)
                             else 0.0),
        "crush_energy": plastic_work,
        "elastic_ice_energy": elastic_energy,
        "strain_energy": strain_energy,
        "bend_energy": bend_energy,
        "kinetic_initial": ke0,
        "kinetic_final": ke_end,
        "energy_residual": residual,
        "energy_closure_error": abs(residual) / ke0,
        "panel_frequency": panel.rayleigh_frequency(),
        "contact_area": area,
        "p_cap": float(p_cap),
        "peak_pressure": (float(torch.max(peak_state["pressure"]))
                          if peak_state else 0.0),
        "ice": ice,
        "panel_stiffness": float(1.0 / (panel.compliance.sum() / n ** 2)),
    }


def compare_rigid_elastic(cfg: CoupledConfig) -> dict:
    """Run the rigid-plate and the elastic-plate case with identical forcing."""
    rigid = run_ice_panel_case(CoupledConfig(**{**cfg.__dict__, "rigid": True}))
    elastic = run_ice_panel_case(CoupledConfig(**{**cfg.__dict__, "rigid": False}))
    reduction = (rigid["peak_force"] - elastic["peak_force"]) / rigid["peak_force"]
    return {"rigid": rigid, "elastic": elastic,
            "peak_rigid": rigid["peak_force"],
            "peak_elastic": elastic["peak_force"],
            "load_reduction": reduction,
            "bending_fraction_rigid": rigid["bending_fraction"],
            "bending_fraction_elastic": elastic["bending_fraction"]}


def thickness_sensitivity(cfg: CoupledConfig, thicknesses) -> list[dict]:
    """Peak-load reduction and bending fraction versus plate thickness."""
    out = []
    for t in thicknesses:
        case = CoupledConfig(**{**cfg.__dict__,
                                "panel": PanelConfig(**{**cfg.panel.__dict__,
                                                        "thickness": float(t)})})
        row = compare_rigid_elastic(case)
        out.append({"thickness": float(t),
                    "peak_rigid": row["peak_rigid"],
                    "peak_elastic": row["peak_elastic"],
                    "load_reduction": row["load_reduction"],
                    "bending_fraction_elastic": row["bending_fraction_elastic"],
                    "panel_frequency": row["elastic"]["panel_frequency"],
                    "energy_closure_error": row["elastic"]["energy_closure_error"]})
    return out


# --------------------------------------------------------------------------
# static verification against plate theory
# --------------------------------------------------------------------------
def uniform_pressure_forces(panel: HullPanel, pressure: float) -> torch.Tensor:
    """Consistent nodal load vector for a uniform pressure on the panel."""
    f = torch.zeros(panel.ndof, dtype=D)
    gp = 1.0 / 3.0 ** 0.5
    for conn in panel.elems:
        xy = panel.nodes[conn]
        fe = torch.zeros(4, dtype=D)
        for xif in (-gp, gp):
            for etaf in (-gp, gp):
                N, dN_nat = _shape(torch.tensor(xif, dtype=D),
                                   torch.tensor(etaf, dtype=D))
                J = dN_nat.T @ xy
                fe += N * pressure * torch.linalg.det(J)
        f[3 * conn] += fe
    return f


def uniform_load_centre_deflection(config: PanelConfig,
                                   pressure: float) -> float:
    """Finite-element centre deflection of the bay under a uniform pressure."""
    panel = HullPanel(config, torch.zeros((1, 2), dtype=D))
    f = uniform_pressure_forces(panel, pressure)
    u = panel.solve_reduced(f).reshape(-1, 3)[:, 0]
    i, j = config.nx // 2, config.ny // 2
    return float(abs(u[j * (config.nx + 1) + i]))


def theory_centre_deflection(config: PanelConfig, pressure: float) -> float:
    """Thin-plate reference (Kirchhoff Navier plus Mindlin shear term)."""
    bending = navier_ss_rect_centre(config.span_x, config.span_y, config.young,
                                    config.poisson, config.thickness, pressure)
    shear = mindlin_shear_correction(config.span_x, config.span_y, config.young,
                                     config.poisson, config.thickness, pressure)
    return bending + shear
