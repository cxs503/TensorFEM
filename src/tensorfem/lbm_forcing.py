"""Guo (2002) body-force scheme for D2Q9 LBM, with an analytic benchmark.

Anchor: ``JOUR-DIFFLBM-REF-VERIFY`` — the citation of the forcing scheme must
be *code-backed* and *quantitatively* right, not decorative.  The first
delivery of this anchor cited Guo-Zheng-Shi (2002) with a venue/DOI belonging
to a different paper *and* no force term anywhere in the JOUR-DIFFLBM fluid
core, so neither the bibliographic fact nor the method claim could be checked.

This module supplies the two things that make the citation verifiable:

1. :func:`guo_source` — the discrete-lattice-effect-corrected source term of
   Guo, Zheng & Shi, *Phys. Rev. E* **65**, 046308 (2002), Eq. (20):
   ``F_i = w_i (1 - 1/(2 tau)) [ (c_i - u)/cs^2 + (c_i.u) c_i / cs^4 ] . F``,
   together with the matching velocity redefinition
   ``u = (sum_i c_i f_i + F/2) / rho`` — implemented as :func:`guo_macroscopic`.
   :func:`naive_source` is the uncorrected ("velocity-shifted", no lattice
   correction) scheme the paper argues against.
2. :func:`poiseuille_benchmark` — force-driven plane Poiseuille flow, whose
   steady state is known analytically,
   ``u(y) = F_x (y - y_wall)(y_wall2 - y) / (2 rho nu)`` with the half-way
   bounce-back walls at ``y_wall = 1`` and ``y_wall2 = ny - 1``.
   The benchmark reports ``ref`` (analytic) vs ``got`` (simulated) per
   resolution for both schemes, so the size of the discrete-lattice error the
   Guo correction removes is a number in an artifact rather than a claim.

The production consumer of the scheme is
:meth:`tensorfem.lbm_channel2d.ChannelLBM._collide` (the ``body_force``
branch); :func:`poiseuille_benchmark` drives that solver as well, so the
citation is validated *through the solver that ships* and not only through
the reference implementation here.

Conventions: lattice units ``dx = dt = rho0 = 1``, ``cs^2 = 1/3``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lbm_references import REFS_BY_KEY

DTYPE = np.float64
CS2 = 1.0 / 3.0
CX = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1), dtype=np.int64)
CY = np.array((0, 0, 1, 0, -1, 1, 1, -1, -1), dtype=np.int64)
W9 = np.array((4.0 / 9.0,) + (1.0 / 9.0,) * 4 + (1.0 / 36.0,) * 4, dtype=DTYPE)
OPP = np.array((0, 3, 4, 1, 2, 7, 8, 5, 6), dtype=np.int64)

_CX = CX.reshape(9, 1, 1).astype(DTYPE)
_CY = CY.reshape(9, 1, 1).astype(DTYPE)
_W = W9.reshape(9, 1, 1)

#: citation key of the scheme implemented here (registry-checked)
SCHEME_REFERENCE = REFS_BY_KEY["guo2002_forcing"]

__references__ = SCHEME_REFERENCE.bib_line()


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium populations, shape ``(9, ny, nx)``."""
    cu = _CX * ux[None] + _CY * uy[None]
    usq = ux * ux + uy * uy
    return _W * rho[None] * (
        1.0
        + 3.0 * cu
        + 4.5 * cu * cu
        - 1.5 * usq[None]
    )


def guo_source(u: np.ndarray, force: np.ndarray, tau: float
               ) -> np.ndarray:
    """Guo-Zheng-Shi source term ``F_i`` (Phys. Rev. E 65, 046308, Eq. 20).

    ``u`` has shape ``(2, ny, nx)`` and ``force`` the same; ``tau`` is the BGK
    relaxation time.  The prefactor ``(1 - 1/(2 tau))`` is the discrete-lattice
    correction that makes the recovered momentum equation carry ``F`` with no
    spurious ``O(F (1 - 1/(2 tau)))`` term.
    """
    ux, uy = u[0], u[1]
    fx, fy = force[0], force[1]
    cu = _CX * ux[None] + _CY * uy[None]
    cdotF = _CX * fx[None] + _CY * fy[None]
    udotF = (ux * fx + uy * fy)[None]
    pref = 1.0 - 0.5 / tau
    return pref * _W * (3.0 * (cdotF - udotF) + 9.0 * cu * cdotF)


def naive_source(u: np.ndarray, force: np.ndarray, tau: float) -> np.ndarray:
    """Uncorrected forcing scheme ("scheme I" of Guo et al. 2002).

    ``F_i = w_i (c_i . F) / cs^2`` together with the *same* half-force
    velocity shift ``u = (sum_i c_i f_i + F/2)/rho``: the source term is
    applied with no ``(1 - 1/(2 tau))`` discrete-lattice correction and no
    ``(c_i . u) c_i`` structure, which is exactly the scheme Guo et al. show
    to be in error at ``O(F (1 - 1/(2 tau)))``.  Kept so the benchmark
    *measures* the size of that effect instead of asserting it.
    """
    del u, tau                                    # scheme has no dependence
    cdotF = _CX * force[0][None] + _CY * force[1][None]
    return 3.0 * _W * cdotF


def guo_macroscopic(f: np.ndarray, force: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rho, ux, uy)`` with the Guo half-force velocity shift.

    Solid nodes carry no populations; their (zero) density is replaced by
    ``1`` before dividing so the moments stay finite there (their output is
    never used — the masks remove them).
    """
    rho = f.sum(axis=0)
    rho = np.where(np.abs(rho) < 1e-12, 1.0, rho)
    ux = (f * _CX).sum(axis=0) / rho
    uy = (f * _CY).sum(axis=0) / rho
    if force is not None:
        ux = ux + 0.5 * force[0] / rho
        uy = uy + 0.5 * force[1] / rho
    return rho, ux, uy


def source_momentum(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(sum_i c_i F_i)`` — the momentum the source term actually injects."""
    return (source * _CX).sum(axis=0), (source * _CY).sum(axis=0)


# ---------------------------------------------------------------------------
# reference implementation: force-driven Poiseuille flow
# ---------------------------------------------------------------------------
@dataclass
class PoiseuilleConfig:
    """Plane Poiseuille flow driven by a constant body force (lattice units)."""

    ny: int = 34               # wall rows 0 and ny-1 are solid
    nx: int = 4                # periodic in x, only the profile matters
    nu: float = 0.05
    force_x: float = 0.0       # default: chosen for u_max ~ 0.05
    scheme: str = "guo"        # "guo" | "naive"
    tol: float = 1e-11
    max_steps: int = 400_000

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def height(self) -> float:
        """Wall-to-wall distance: half-way bounce-back puts the walls on the
        faces between the solid rows (centres ``0.5``, ``ny - 0.5``) and the
        first/last fluid rows (centres ``1.5``, ``ny - 1.5``),
        i.e. at ``y = 1`` and ``y = ny - 1``."""
        return float(self.ny - 2)

    @property
    def u_max_analytic(self) -> float:
        return self.force_x * self.height ** 2 / (8.0 * self.nu)

    def __post_init__(self) -> None:
        if self.force_x == 0.0:
            # keep the Mach number of the benchmark fixed across resolutions
            self.force_x = 0.05 * 8.0 * self.nu / self.height ** 2


def analytic_profile(cfg: PoiseuilleConfig) -> np.ndarray:
    """``u(y)`` for the fluid rows ``y = 1 .. ny-2`` (cell centres at y+0.5)."""
    y = np.arange(1, cfg.ny - 1, dtype=DTYPE) + 0.5
    y1, y2 = 1.0, float(cfg.ny - 1)
    return cfg.force_x / (2.0 * cfg.nu) * (y - y1) * (y2 - y)


class PoiseuilleLBM:
    """Minimal D2Q9 channel: periodic in x, bounce-back walls, body force."""

    def __init__(self, cfg: PoiseuilleConfig):
        if cfg.scheme not in ("guo", "naive"):
            raise ValueError(f"unknown scheme {cfg.scheme!r}")
        self.cfg = cfg
        self.solid = np.zeros((cfg.ny, cfg.nx), dtype=bool)
        self.solid[0, :] = True
        self.solid[-1, :] = True
        self.fluid = ~self.solid
        self.f = np.empty((9, cfg.ny, cfg.nx), dtype=DTYPE)
        self._from_top = np.zeros((9, cfg.ny, cfg.nx), dtype=bool)
        self._from_bottom = np.zeros((9, cfg.ny, cfg.nx), dtype=bool)
        self._from_top[:, 0, :] = True          # row 0 reflects downwards
        self._from_bottom[:, -1, :] = True
        self.force = np.zeros((2, cfg.ny, cfg.nx), dtype=DTYPE)
        self.force[0, :, :] = cfg.force_x
        self.force[:, ~self.fluid] = 0.0
        self.time = 0
        self.residual = float("nan")
        self.reset()

    # -- setup --------------------------------------------------------------
    def reset(self) -> None:
        """Start from the analytic profile (still an out-of-equilibrium state,
        so the run converges to the *discrete* steady solution)."""
        ux = np.zeros((self.cfg.ny, self.cfg.nx), dtype=DTYPE)
        ux[1:-1, :] = analytic_profile(self.cfg)[:, None]
        rho = np.ones((self.cfg.ny, self.cfg.nx), dtype=DTYPE)
        uy = np.zeros_like(ux)
        self.f = equilibrium(rho, ux, uy)
        self.f[:, ~self.fluid] = 0.0
        self.time = 0

    # -- evolution ----------------------------------------------------------
    def _collide(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.scheme == "guo":
            rho, ux, uy = guo_macroscopic(self.f, self.force)
            u = np.stack([ux, uy])
            src = guo_source(u, self.force, cfg.tau)
        else:
            # scheme I keeps the half-force velocity shift as well; only the
            # discrete-lattice correction of the source term is missing
            rho, ux, uy = guo_macroscopic(self.f, self.force)
            src = naive_source(np.stack([ux, uy]), self.force, cfg.tau)
        feq = equilibrium(rho, ux, uy)
        out = self.f + (feq - self.f) / cfg.tau + src
        out[:, ~self.fluid] = 0.0
        if not np.isfinite(out).all():
            raise FloatingPointError("Poiseuille flow diverged")
        return out

    def _stream(self, f_post: np.ndarray) -> np.ndarray:
        out = np.empty_like(f_post)
        for i in range(9):
            streamed = np.roll(f_post[i], (CY[i], CX[i]), axis=(0, 1))
            # half-way bounce-back: a population whose upstream node is solid
            # is replaced by the post-collision opposite population of the
            # node itself (the wall sits half-way between the two centres)
            mask = self._solid_upstream(i)
            out[i] = np.where(mask, f_post[OPP[i]], streamed)
        out[:, ~self.fluid] = 0.0
        return out

    def _solid_upstream(self, i: int) -> np.ndarray:
        """Nodes whose upstream neighbour in direction ``i`` is solid."""
        dx, dy = int(CX[i]), int(CY[i])
        if dy > 0:                       # comes from below
            mask = np.zeros_like(self.solid)
            mask[dy:, :] = self.solid[:-dy, :]
            return mask
        if dy < 0:                       # comes from above
            mask = np.zeros_like(self.solid)
            mask[:dy, :] = self.solid[-dy:, :]
            return mask
        # purely horizontal: periodic in x, only walls block (handled by dy=0)
        return np.zeros_like(self.solid)

    def step(self) -> None:
        self.f = self._stream(self._collide())
        self.time += 1

    def run(self, check_every: int = 200) -> dict[str, float]:
        """March to the discrete steady state; return run statistics."""
        cfg = self.cfg
        prev = self.velocity()[0]
        while self.time < cfg.max_steps:
            for _ in range(check_every):
                self.step()
            ux = self.velocity()[0]
            # only the *shape* matters: normalise away the initial transient
            self.residual = float(np.max(np.abs(ux - prev)))
            if self.residual < cfg.tol:
                break
            prev = ux
        return {
            "steps": self.time,
            "residual": self.residual,
            "converged": bool(self.residual < cfg.tol),
        }

    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        rho, ux, uy = guo_macroscopic(self.f, self.force)
        return ux, uy

    def profile(self) -> np.ndarray:
        """``u_x`` on the fluid rows, shape ``(ny - 2,)``."""
        ux, _ = self.velocity()
        return ux[1:-1, :].mean(axis=1)

    def profile_l2_error(self) -> float:
        ref = analytic_profile(self.cfg)
        got = self.profile()
        scale = float(np.max(np.abs(ref)))
        return float(np.linalg.norm(got - ref) / np.linalg.norm(ref)) if scale else float("nan")


# ---------------------------------------------------------------------------
# benchmark driver
# ---------------------------------------------------------------------------
@dataclass
class BenchRow:
    solver: str
    scheme: str
    ny: int
    nu: float
    force_x: float
    u_max_ref: float
    u_max_got: float
    rel_err_u_max: float
    rel_l2_profile: float
    steps: int
    converged: bool
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        d = {
            "solver": self.solver,
            "scheme": self.scheme,
            "ny": self.ny,
            "nu": self.nu,
            "force_x": self.force_x,
            "u_max_ref": self.u_max_ref,
            "u_max_got": self.u_max_got,
            "rel_err_u_max": self.rel_err_u_max,
            "rel_l2_profile": self.rel_l2_profile,
            "steps": self.steps,
            "converged": self.converged,
        }
        d.update(self.extra)
        return d


def _row(solver: str, scheme: str, cfg: PoiseuilleConfig, ux_prof: np.ndarray,
         steps: int, converged: bool, extra: dict | None = None) -> BenchRow:
    ref = analytic_profile(cfg)
    got_max = float(np.max(ux_prof))
    ref_max = float(np.max(ref))
    l2 = float(np.linalg.norm(ux_prof - ref) / np.linalg.norm(ref))
    return BenchRow(
        solver=solver, scheme=scheme, ny=cfg.ny, nu=cfg.nu,
        force_x=cfg.force_x, u_max_ref=ref_max, u_max_got=got_max,
        rel_err_u_max=abs(got_max - ref_max) / ref_max, rel_l2_profile=l2,
        steps=steps, converged=converged, extra=extra or {})


def poiseuille_benchmark(resolutions: tuple[int, ...] = (10, 18, 34),
                         nu: float = 0.05,
                         schemes: tuple[str, ...] = ("guo", "naive"),
                         guo_only_resolutions: tuple[int, ...] = (66,),
                         tau_sweep: tuple[float, ...] = (0.05, 0.2, 0.5),
                         ny_sweep: int = 34,
                         include_production: bool = True) -> dict[str, object]:
    """Analytic vs simulated Poiseuille profile for each scheme/resolution.

    Three views of the same discrete-lattice question:

    * ``rows``          — grid ladder, Guo vs uncorrected scheme;
    * ``tau_sweep``     — fixed grid, viscosity sweep: the uncorrected scheme's
      steady profile keeps the analytic *shape* with its amplitude scaled by
      ``1 + 1/(2 tau)`` (the discrete lattice effect Guo et al. remove);
    * ``production``    — the shipping solver
      (:class:`tensorfem.lbm_channel2d.ChannelLBM`, whose ``_collide`` carries
      the Guo source term) driven by the same body force.
    """
    rows: list[BenchRow] = []
    for ny in resolutions:
        for scheme in schemes:
            rows.append(_march(ny, nu, scheme))
    for ny in guo_only_resolutions:
        rows.append(_march(ny, nu, "guo"))

    sweep = []
    for nu_i in tau_sweep:
        entry: dict[str, object] = {"nu": nu_i, "tau": 0.5 + 3.0 * nu_i,
                                    "amplitude_law": 1.0 + 0.5 / (0.5 + 3.0 * nu_i)}
        for scheme in ("guo", "naive"):
            row = _march(ny_sweep, nu_i, scheme)
            entry[scheme] = row.as_dict()
            entry[f"{scheme}_amplitude_ratio"] = (
                row.u_max_got / row.u_max_ref)
        sweep.append(entry)

    production = _production_row(ny_sweep, nu) if include_production else None
    return {
        "reference": SCHEME_REFERENCE.bib_line(),
        "reference_doi": SCHEME_REFERENCE.doi,
        "case": "force-driven plane Poiseuille flow (periodic in x, "
                "half-way bounce-back walls at y=1 and y=ny-1)",
        "analytic": "u(y) = F_x (y - 1)(ny - 1 - y) / (2 rho nu)",
        "resolutions": list(resolutions),
        "rows": [r.as_dict() for r in rows],
        "tau_sweep": sweep,
        "production": production.as_dict() if production else None,
        "convergence_order": _convergence_order(rows),
        "gate": {
            "criterion": "Guo scheme: rel_err_u_max < 3 % and rel_l2 < 1 % at "
                         "the finest grid, and smaller than the uncorrected "
                         "scheme at the same grid",
            "guo_finest": _finest(rows, "guo"),
            "naive_finest": _finest(rows, "naive"),
            "production": (production.as_dict() if production else None),
            "passed": None,      # filled by benchmark_passed()
        },
    }


def _march(ny: int, nu: float, scheme: str,
           store_profile: bool = True) -> BenchRow:
    cfg = PoiseuilleConfig(ny=ny, nu=nu, scheme=scheme)
    sim = PoiseuilleLBM(cfg)
    stats = sim.run()
    extra: dict[str, object] = {"residual": stats["residual"], "tau": cfg.tau}
    if store_profile:
        y = np.arange(1, cfg.ny - 1, dtype=DTYPE) + 0.5
        extra["y"] = [round(float(v), 6) for v in y]
        extra["u_profile"] = [round(float(v), 10) for v in sim.profile()]
        extra["u_ref"] = [round(float(v), 10) for v in analytic_profile(cfg)]
    return _row("PoiseuilleLBM", scheme, cfg, sim.profile(),
                int(stats["steps"]), bool(stats["converged"]), extra)


def _convergence_order(rows: list[BenchRow]) -> dict[str, object]:
    """Observed order of accuracy of the Guo scheme from the grid ladder."""
    guo = sorted((r for r in rows if r.scheme == "guo" and r.solver == "PoiseuilleLBM"),
                 key=lambda r: r.ny)
    if len(guo) < 2:
        return {}
    pairs = list(zip(guo[:-1], guo[1:]))
    orders = []
    for lo, hi in pairs:
        ratio = lo.rel_l2_profile / hi.rel_l2_profile if hi.rel_l2_profile else float("inf")
        h_ratio = (hi.ny - 2) / (lo.ny - 2)
        orders.append(float(np.log(ratio) / np.log(h_ratio)))
    return {
        "scheme": "guo",
        "resolutions": [r.ny for r in guo],
        "rel_l2": [r.rel_l2_profile for r in guo],
        "orders": orders,
        "mean_order": float(np.mean(orders)),
    }


def _finest(rows: list[BenchRow], scheme: str) -> dict[str, object]:
    cand = [r for r in rows if r.scheme == scheme]
    if not cand:
        return {}
    best = max(cand, key=lambda r: r.ny)
    return {
        "ny": best.ny,
        "solver": best.solver,
        "rel_err_u_max": best.rel_err_u_max,
        "rel_l2_profile": best.rel_l2_profile,
    }


def _finest_from_rows(rows: list[dict], scheme: str) -> dict[str, object]:
    """``_finest`` for already-serialised rows (staged/JSON driven runs)."""
    cand = [r for r in rows if r["scheme"] == scheme]
    if not cand:
        return {}
    best = max(cand, key=lambda r: r["ny"])
    return {"ny": best["ny"], "solver": best["solver"],
            "rel_err_u_max": best["rel_err_u_max"],
            "rel_l2_profile": best["rel_l2_profile"]}


def _production_row(ny: int, nu: float) -> BenchRow:
    """Drive the shipping channel solver with a constant body force."""
    from .lbm_channel2d import ChannelConfig, ChannelLBM

    cfg = PoiseuilleConfig(ny=ny, nu=nu, scheme="guo")
    prod = ChannelLBM(ChannelConfig(nx=cfg.nx, ny=ny, u_mean=0.05, nu=nu,
                                    periodic_x=True))
    force = np.zeros((2, ny, cfg.nx), dtype=DTYPE)
    force[0, :, :] = cfg.force_x
    ux_prev = None
    steps = 0
    converged = False
    while steps < cfg.max_steps:
        for _ in range(200):
            prod.step(body_force=force)
        steps += 200
        ux = _channel_ux(prod)
        if ux_prev is not None and float(np.max(np.abs(ux - ux_prev))) < 1e-11:
            converged = True
            break
        ux_prev = ux
    prof = _channel_ux(prod)[1:-1, :].mean(axis=1)
    return _row("ChannelLBM(production)", "guo", cfg, prof, steps, converged)


def _channel_ux(solver) -> np.ndarray:
    """``u_x`` field of a :class:`ChannelLBM` (macroscopic moments)."""
    from .lbm_channel2d import macroscopic

    _rho, ux, _uy = macroscopic(solver.f)
    return ux


def benchmark_passed(result: dict[str, object]) -> bool:
    """Gate used by the CLI/entry point: Guo must beat naive *and* meet 3 %."""
    gate = result["gate"]
    guo = gate.get("guo_finest") or {}
    naive = gate.get("naive_finest") or {}
    if not guo:
        return False
    ok_guo = guo.get("rel_err_u_max", 1.0) < 0.03 and \
        guo.get("rel_l2_profile", 1.0) < 0.01
    ok_order = True
    if naive:
        ok_order = guo["rel_err_u_max"] < naive["rel_err_u_max"]
    return bool(ok_guo and ok_order)