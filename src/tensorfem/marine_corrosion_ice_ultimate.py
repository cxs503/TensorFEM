"""Time-dependent ultimate strength of polar hull plating under coupled damage.

This module implements the chain that answers three engineering questions for
ice-class hull plating:

1. ``generate_pitting_field`` -- a stochastic pitting field (spatially
   correlated susceptibility, Poisson pit census, Weibull pit depths grown by a
   Melchers power law or a DNV-RP-G101 style wastage law) rasterised into a
   thickness field ``t(x, y)``;
2. ``plate_ultimate_strength`` -- a *field resolved* ultimate strength: the
   elastic buckling load of the realised thickness field is computed by a
   finite-difference generalised eigenproblem (sparse shift-invert) and the
   inelastic collapse stress follows the classical effective-width
   (Faulkner / von Karman) law expressed in the field-resolved slenderness.  Two
   further mechanisms are evaluated and the minimum governs: the net-section
   squash limit of the rasterised field and a parallel (iso-strain) bound built
   from the thinnest patch inside each correlation cell;
3. ``run_corrosion_ice_strength_assessment`` -- the decadal margin history
   ``U(t) = sigma_ice / sigma_u(t)`` with the ice pressure from a pressure-area
   law and the plate membrane stress of the laterally loaded panel, plus the
   inverse problem (allowable wastage and inspection interval for a target
   safety factor and a target reliability fractile).

Every number that leaves this module is either computed by the numerics above,
a documented empirical formula, or an explicitly flagged input constant.  The
empirical anchor is the Paik-type pitting reduction power law
``R(DOP) = 1 - a * DOP**b`` used as a fast pass and as a cross-check of the
field solver.  The primary-source coefficient table of Paik et al. (2003) is
not available offline in this repository: the default pair is a
secondary-literature transcription (``a = 0.33``, ``b = 1``, i.e. about 10 %
loss at ``DOP = 0.3``), the ``provenance`` strings say so, and every report
carries the anchor's coefficient uncertainty as a sensitivity row
(``anchor_sensitivity``) instead of hiding it.

Scope boundary (deliberate): small-deflection plate panel, uniaxial in-plane
compression along the panel length, simply supported edges, ideal
elastic-perfectly-plastic material without hardening, no residual stress, no
stiffener tripping, no crack or brittle-fracture screening (that is a separate
qualification), no hydrodynamic amplification of the ice load, and no
time-dependent pit geometry beyond the documented initiation window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

TOLERANCE = 0.03
PAIK_TOLERANCE = 0.10
DEGENERACY_TOLERANCE = 0.02
MC_CONVERGENCE_TOLERANCE = 0.01
MASS_LOSS_TOLERANCE = 0.05


# --------------------------------------------------------------------------- #
# input provenance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CorrosionLaw:
    """Time-dependent corrosion depth ``d(t) = A * (t - t_i) ** n`` in millimetres."""

    name: str
    coefficient: float
    exponent: float
    initiation_years: float = 0.0
    provenance: str = "transcribed parameters; survey data must replace them"

    def depth_mm(self, years: float) -> float:
        value = _nonnegative("years", years)
        if value <= self.initiation_years:
            return 0.0
        return self.coefficient * (value - self.initiation_years) ** self.exponent


#: Melchers power law for freely corroding immersion steel, ``d`` in mm.
MELCHERS_IMMERSION = CorrosionLaw(
    name="Melchers power law (immersion, mean)",
    coefficient=0.076,
    exponent=0.5,
    provenance=(
        "Melchers (2005) power-law form d = A t^n with the commonly quoted "
        "immersion pair A = 0.076 mm/yr^n, n = 0.5; the fitted values are "
        "climate specific and the paper was not re-read offline"
    ),
)

#: DNV-RP-G101 style linear wastage after coating breakdown (``n = 1``).
DNV_G101_ARCTIC_ICE_BELT = CorrosionLaw(
    name="DNV-RP-G101 style linear wastage (ice belt, coated)",
    coefficient=0.11,
    exponent=1.0,
    initiation_years=8.0,
    provenance=(
        "DNV-RP-G101 tabulated corrosion-rate practice: constant rate after a "
        "coating-breakdown period. The 0.11 mm/yr rate and 8 yr initiation are "
        "a documented placeholder for an ice-belt exposure class, not a code check"
    ),
)


@dataclass(frozen=True)
class SteelGrade:
    """Arctic hull structural steel with a linear yield-stress temperature shift."""

    name: str
    yield_stress_mpa: float
    ultimate_stress_mpa: float
    charpy_joules: float
    charpy_temperature_c: float
    temperature_slope_per_c: float = 0.0015
    provenance: str = ""

    def yield_stress_at(self, temperature_c: float) -> float:
        """Return the yield stress in MPa at ``temperature_c`` (linear shift)."""
        value = _finite("temperature_c", temperature_c)
        shift = 1.0 + self.temperature_slope_per_c * (20.0 - value)
        if shift <= 0.0:
            raise ValueError("temperature correction left the elastic range")
        return self.yield_stress_mpa * shift


#: Minimum-yield arctic grades.  ``temperature_slope_per_c`` is a low-alloy
#: ferritic estimate (about +9 % yield at -40 C); it is flagged, not verified.
ARCTIC_STEELS = {
    "EH36": SteelGrade(
        "EH36", 355.0, 490.0, 34.0, -40.0,
        provenance="minimum specified yield/ultimate for EH36 plate <= 50 mm; "
                   "34 J @ -40 C is the grade toughness criterion",
    ),
    "DH36": SteelGrade(
        "DH36", 355.0, 490.0, 34.0, -20.0,
        provenance="minimum specified yield/ultimate for DH36 plate <= 50 mm; "
                   "34 J @ -20 C is the grade toughness criterion",
    ),
}


@dataclass(frozen=True)
class IceDesignLoad:
    """Local ice pressure with a pressure-area law."""

    reference_pressure_mpa: float = 2.5
    reference_area_m2: float = 1.0
    area_exponent: float = 0.5
    contact_area_m2: float = 1.0
    provenance: str = (
        "pressure-area power law p = p_ref (A / A_ref)^-beta; the default "
        "(2.5 MPa at 1 m2, beta = 0.5) is a documented placeholder for a design "
        "ice pressure, not a project-specific ISO 19906 value"
    )

    def pressure_mpa(self) -> float:
        _positive("reference_pressure_mpa", self.reference_pressure_mpa)
        _positive("reference_area_m2", self.reference_area_m2)
        _positive("contact_area_m2", self.contact_area_m2)
        exponent = _nonnegative("area_exponent", self.area_exponent)
        return self.reference_pressure_mpa * (
            self.contact_area_m2 / self.reference_area_m2
        ) ** (-exponent)


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _positive(name: str, value: float) -> float:
    out = _finite(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _nonnegative(name: str, value: float) -> float:
    out = _finite(name, value)
    if out < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return out


_NUMPY_CACHE = None
_SPARSE_CACHE = None


def _require_numpy():
    """Return numpy, importing it once per process."""
    global _NUMPY_CACHE
    if _NUMPY_CACHE is None:
        try:
            import numpy as numpy_module
        except ImportError as error:  # pragma: no cover - numpy is a declared dev dep
            raise RuntimeError(
                "marine_corrosion_ice_ultimate requires numpy (declared dev dependency)"
            ) from error
        _NUMPY_CACHE = numpy_module
    return _NUMPY_CACHE


def _require_sparse():
    """Return ``(scipy.sparse, scipy.sparse.linalg)``, imported once per process."""
    global _SPARSE_CACHE
    if _SPARSE_CACHE is None:
        try:
            import scipy.sparse as sparse_module
            import scipy.sparse.linalg as linalg_module
        except ImportError as error:  # pragma: no cover - scipy is a declared dev dep
            raise RuntimeError(
                "marine_corrosion_ice_ultimate requires scipy.sparse "
                "(declared dev dependency)"
            ) from error
        _SPARSE_CACHE = (sparse_module, linalg_module)
    return _SPARSE_CACHE


def _trapezoid(values, coordinates):
    numpy = _require_numpy()
    if hasattr(numpy, "trapezoid"):
        return numpy.trapezoid(values, coordinates)
    return numpy.trapz(values, coordinates)  # pragma: no cover - numpy < 2


# --------------------------------------------------------------------------- #
# 1. corrosion kinetics and uniform mass loss
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MassLossResult:
    analytic_kg: float
    discrete_kg: float
    relative_error: float
    time_step_years: float
    thickness_loss_mm: float


def uniform_wastage_mass_loss(
    *, years: float, area_m2: float, density_kg_m3: float, law: CorrosionLaw,
    time_step_years: float = 1.0,
) -> MassLossResult:
    """Integrate a wastage law into lost mass, discrete versus closed form.

    ``d(t) = A (t - t_i)^n`` gives the closed form
    ``m = rho A_s 1e-3 A T^(n+1) / (n+1)`` with ``T = t - t_i``.  The discrete
    route is the annual midpoint sum an inspector would use, so the deviation is
    a numerical-integration check rather than a model comparison.
    """
    duration = _nonnegative("years", years)
    area = _positive("area_m2", area_m2)
    density = _positive("density_kg_m3", density_kg_m3)
    step = _positive("time_step_years", time_step_years)
    if not isinstance(law, CorrosionLaw):
        raise TypeError("law must be a CorrosionLaw")
    if law.exponent <= 0.0:
        raise ValueError("law.exponent must be positive")
    exposure = max(duration - law.initiation_years, 0.0)
    analytic = (
        density * area * 1.0e-3 * law.coefficient * exposure ** (law.exponent + 1.0)
        / (law.exponent + 1.0)
    )
    steps = max(1, int(math.ceil(duration / step)))
    dt = duration / steps
    discrete = 0.0
    for index in range(steps):
        midpoint = (index + 0.5) * dt
        discrete += density * area * 1.0e-3 * law.depth_mm(midpoint) * dt
    error = abs(discrete / analytic - 1.0) if analytic > 0.0 else 0.0
    return MassLossResult(analytic, discrete, error, dt, law.depth_mm(duration))


# --------------------------------------------------------------------------- #
# 2. stochastic pitting field
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PittingSpec:
    """Geometry, intensity and stochastic model of the pitting field."""

    length_m: float
    width_m: float
    thickness_m: float
    pitting_intensity: float = 0.20
    pit_diameter_m: float = 0.030
    depth_ratio: float = 0.50
    depth_model: str = "corrosion_law_volume"
    correlation_length_m: float = 0.30
    weibull_shape: float = 1.20
    clustering_sigma: float = 1.00
    diameter_cov: float = 0.25
    reference_years: float = 25.0
    initiation_window_years: float = 15.0
    seed: int = 0
    grid_x: int = 40
    grid_y: int = 20

    def validated(self) -> "PittingSpec":
        _positive("length_m", self.length_m)
        _positive("width_m", self.width_m)
        _positive("thickness_m", self.thickness_m)
        intensity = _nonnegative("pitting_intensity", self.pitting_intensity)
        if intensity >= 1.0:
            raise ValueError("pitting_intensity must be below one")
        _positive("pit_diameter_m", self.pit_diameter_m)
        depth = _nonnegative("depth_ratio", self.depth_ratio)
        if depth >= 0.98:
            raise ValueError("depth_ratio must stay below through-thickness loss")
        if self.depth_model not in {"corrosion_law_volume", "depth_ratio"}:
            raise ValueError(
                "depth_model must be 'corrosion_law_volume' or 'depth_ratio'"
            )
        _positive("correlation_length_m", self.correlation_length_m)
        _positive("weibull_shape", self.weibull_shape)
        if self.clustering_sigma < 0.0:
            raise ValueError("clustering_sigma must be nonnegative")
        if self.diameter_cov < 0.0:
            raise ValueError("diameter_cov must be nonnegative")
        _positive("reference_years", self.reference_years)
        _nonnegative("initiation_window_years", self.initiation_window_years)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.grid_x < 4 or self.grid_y < 4:
            raise ValueError("grid_x and grid_y must be at least four")
        return self


@dataclass(frozen=True)
class PittingField:
    """A realised pit census rasterised into a thickness field (metres)."""

    years: float
    mean_depth_mm: float
    pit_count: int
    dop_raster: float
    dop_nominal: float
    net_section_ratio: float
    thickness_grid: object
    thickness_effective_grid: object
    cell_thickness_min: object
    correlation_length_m: float
    pit_depths_mm: object
    seed: int


def gaussian_correlated_field(
    *, grid_x: int, grid_y: int, length_m: float, width_m: float,
    correlation_length_m: float, rng,
):
    """Return a unit-variance spatially correlated Gaussian field.

    Spectral synthesis with the Fourier amplitude of the Gaussian covariance
    ``C(r) = exp(-r^2 / l^2)``, i.e. ``H(k) = exp(-pi^2 l^2 |k|^2)``.  The
    realised correlation length is measured back from the field by
    ``realized_correlation_length`` and cross-checked in the tests.
    """
    numpy = _require_numpy()
    nx, ny = int(grid_x), int(grid_y)
    length = _positive("length_m", length_m)
    width = _positive("width_m", width_m)
    corr = _positive("correlation_length_m", correlation_length_m)
    white = rng.standard_normal((ny, nx))
    kx = numpy.fft.fftfreq(nx, d=length / nx)
    ky = numpy.fft.fftfreq(ny, d=width / ny)
    kxx, kyy = numpy.meshgrid(kx, ky)
    filt = numpy.exp(-math.pi**2 * corr**2 * (kxx**2 + kyy**2))
    values = numpy.real(numpy.fft.ifft2(numpy.fft.fft2(white) * filt))
    scale = float(numpy.std(values))
    if scale <= 0.0:
        raise RuntimeError("degenerate correlation field")
    return values / scale


def realized_correlation_length(values, dx: float) -> float:
    """Return the distance where the transverse autocorrelation drops to 1/e."""
    numpy = _require_numpy()
    data = numpy.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("values must be a two dimensional field with four columns")
    centred = data - data.mean()
    variance = float(numpy.mean(centred**2))
    if variance <= 0.0:
        return 0.0
    target = math.exp(-1.0)
    for lag in range(1, data.shape[1] - 1):
        rho = float(numpy.mean(centred[:, lag:] * centred[:, :-lag])) / variance
        if rho <= target:
            return lag * dx
    return (data.shape[1] - 2) * dx


def build_pit_catalogue(spec: PittingSpec, *, rng) -> dict:
    """Draw the pit census: Poisson counts in correlation cells, Weibull depths."""
    numpy = _require_numpy()
    cfg = spec.validated()
    area = cfg.length_m * cfg.width_m
    pit_area = math.pi * (0.5 * cfg.pit_diameter_m) ** 2
    if cfg.pitting_intensity <= 0.0:
        expected = 0.0
    elif cfg.pitting_intensity >= 0.95:
        expected = -math.log(0.05) * area / pit_area
    else:
        # Invert the random-disc coverage relation ``covered = 1 - exp(-A_p/A)``
        # so that the *realised* pitting intensity matches the requested one
        # instead of saturating through overlap.
        expected = -math.log(1.0 - cfg.pitting_intensity) * area / pit_area
    cell = max(cfg.correlation_length_m / 3.0, cfg.pit_diameter_m)
    nx = max(1, int(round(cfg.length_m / cell)))
    ny = max(1, int(round(cfg.width_m / cell)))
    susceptibility = gaussian_correlated_field(
        grid_x=nx, grid_y=ny, length_m=cfg.length_m, width_m=cfg.width_m,
        correlation_length_m=cfg.correlation_length_m, rng=rng,
    )
    sigma = cfg.clustering_sigma
    weight = numpy.exp(sigma * susceptibility - 0.5 * sigma**2)
    weight = weight / float(numpy.mean(weight))
    counts = rng.poisson(expected * weight / (nx * ny)).ravel()
    total = int(numpy.sum(counts))
    xs: list[float] = []
    ys: list[float] = []
    cell_w = cfg.length_m / nx
    cell_h = cfg.width_m / ny
    for index, count in enumerate(counts):
        if count == 0:
            continue
        jy, ix = divmod(int(index), nx)
        xs.extend(((rng.random(count) * cell_w + ix * cell_w) / cfg.length_m).tolist())
        ys.extend(((rng.random(count) * cell_h + jy * cell_h) / cfg.width_m).tolist())
    log_sigma = math.sqrt(math.log(1.0 + cfg.diameter_cov**2))
    diameters = cfg.pit_diameter_m * rng.lognormal(
        mean=-0.5 * log_sigma**2, sigma=log_sigma, size=total,
    )
    depth_factor = rng.weibull(cfg.weibull_shape, size=total) / math.gamma(
        1.0 + 1.0 / cfg.weibull_shape
    )
    initiation = rng.uniform(0.0, cfg.initiation_window_years, size=total)
    return {
        "x": numpy.clip(numpy.asarray(xs, dtype=float), 0.0, 1.0),
        "y": numpy.clip(numpy.asarray(ys, dtype=float), 0.0, 1.0),
        "diameter": diameters,
        "depth_factor": depth_factor,
        "initiation_years": initiation,
        "expected_count": expected,
        "susceptibility": susceptibility,
    }


def generate_pitting_field(
    *, spec: PittingSpec, years: float, law: CorrosionLaw, catalogue=None,
    forced_mean_depth_mm: float | None = None,
) -> PittingField:
    """Rasterise a stochastic pit census into a thickness field.

    The pit census (positions, diameters, depth factors, initiation times) is
    drawn once per ``spec.seed`` and reused for every ``years``, so a growing
    corrosion depth only deepens existing pits and activates new ones.  That is
    what makes a per-realisation strength history monotone in time.
    """
    numpy = _require_numpy()
    cfg = spec.validated()
    duration = _nonnegative("years", years)
    rng = numpy.random.default_rng(cfg.seed)
    cata = catalogue if catalogue is not None else build_pit_catalogue(cfg, rng=rng)
    initiations = numpy.asarray(cata["initiation_years"], dtype=float)
    depth_factors = numpy.asarray(cata["depth_factor"], dtype=float)
    areas = numpy.pi * (numpy.asarray(cata["diameter"], dtype=float) / 2.0) ** 2
    active = initiations <= duration
    exposure = numpy.maximum(0.0, duration - initiations)
    raw = law.coefficient * exposure ** law.exponent
    mean_factor = float(numpy.mean(depth_factors)) if depth_factors.size else 1.0
    if cfg.depth_model == "depth_ratio":
        reference_exposure = max(cfg.reference_years, law.initiation_years)
        mean_initiation = float(numpy.mean(initiations)) if initiations.size else 0.0
        reference_raw = law.coefficient * max(
            reference_exposure - mean_initiation, 0.0
        ) ** law.exponent
        if reference_raw <= 0.0:
            raise ValueError("corrosion law gives no depth at reference_years")
        scale = (cfg.depth_ratio * cfg.thickness_m * 1000.0) / (reference_raw * mean_factor)
        depths = depth_factors * raw * scale
    else:
        # Volume consistency: the corrosion law fixes the mean wastage over the
        # whole surface and the census fixes the pitted area, so the mean depth
        # over the pitted area is wastage / pitted fraction at the same age.
        wastage_mm = law.depth_mm(duration)
        panel_area = cfg.length_m * cfg.width_m
        pitted_area = float(numpy.sum(areas[active])) if bool(active.any()) else 0.0
        pitted_fraction = pitted_area / panel_area
        if pitted_fraction <= 0.0 or wastage_mm <= 0.0:
            depths = numpy.zeros_like(depth_factors)
        else:
            depths = depth_factors * (wastage_mm / (pitted_fraction * mean_factor))
    if forced_mean_depth_mm is not None:
        target = _nonnegative("forced_mean_depth_mm", forced_mean_depth_mm)
        active_mean = float(numpy.mean(depth_factors[active])) if bool(active.any()) else 1.0
        if active_mean <= 0.0:
            raise ValueError("degenerate depth factors")
        depths = numpy.where(active, depth_factors * (target / active_mean), 0.0)
    cap = 0.98 * cfg.thickness_m * 1000.0
    depths = numpy.clip(depths, 0.0, cap)
    depths = numpy.where(active, depths, 0.0)

    fine_dx = cfg.pit_diameter_m / 4.0
    fine_nx = max(8, int(round(cfg.length_m / fine_dx)))
    fine_ny = max(8, int(round(cfg.width_m / fine_dx)))
    fine_dx = cfg.length_m / fine_nx
    fine_dy = cfg.width_m / fine_ny
    coverage = numpy.zeros((fine_ny, fine_nx))
    deepest = numpy.zeros((fine_ny, fine_nx))
    diameters = numpy.asarray(cata["diameter"], dtype=float)
    xs = numpy.asarray(cata["x"], dtype=float) * cfg.length_m
    ys = numpy.asarray(cata["y"], dtype=float) * cfg.width_m
    for index in range(depths.size):
        depth = float(depths[index])
        if depth <= 0.0:
            continue
        radius = 0.5 * float(diameters[index])
        x0, y0 = float(xs[index]), float(ys[index])
        i_lo = max(0, int(math.floor((x0 - radius) / fine_dx)))
        i_hi = min(fine_nx - 1, int(math.ceil((x0 + radius) / fine_dx)))
        j_lo = max(0, int(math.floor((y0 - radius) / fine_dy)))
        j_hi = min(fine_ny - 1, int(math.ceil((y0 + radius) / fine_dy)))
        if i_hi < i_lo or j_hi < j_lo:
            continue
        cx = (numpy.arange(i_lo, i_hi + 1) + 0.5) * fine_dx
        cy = (numpy.arange(j_lo, j_hi + 1) + 0.5) * fine_dy
        mask = ((cx[None, :] - x0) ** 2 + (cy[:, None] - y0) ** 2) <= radius**2
        if not bool(mask.any()):
            continue
        block = coverage[j_lo:j_hi + 1, i_lo:i_hi + 1]
        block[mask] += 1.0
        deep = deepest[j_lo:j_hi + 1, i_lo:i_hi + 1]
        numpy.maximum(deep, numpy.where(mask, depth, 0.0), out=deep)
    thickness_mm = cfg.thickness_m * 1000.0
    removed = float(numpy.sum(deepest)) / deepest.size
    net_ratio = max(1.0e-6, 1.0 - removed / thickness_mm)
    fine_thickness = thickness_mm - numpy.minimum(deepest, cap)

    nx_cells, ny_cells = cfg.grid_x, cfg.grid_y
    cell_thickness = numpy.zeros((ny_cells, nx_cells))
    cell_effective = numpy.zeros((ny_cells, nx_cells))
    cell_min = numpy.zeros((ny_cells, nx_cells))
    x_edges = numpy.linspace(0, fine_nx, nx_cells + 1).astype(int)
    y_edges = numpy.linspace(0, fine_ny, ny_cells + 1).astype(int)
    for j in range(ny_cells):
        for i in range(nx_cells):
            block = fine_thickness[y_edges[j]:y_edges[j + 1], x_edges[i]:x_edges[i + 1]]
            cell_thickness[j, i] = float(numpy.mean(block))
            cell_min[j, i] = float(numpy.min(block))
            # Bending rigidity of a cell whose internal mode shape is smooth: the
            # correct sub-cell average is the arithmetic mean of D ~ t^3 (Voigt),
            # not t^3 of the mean thickness.  The equivalent uniform thickness is
            # therefore (mean t^3)^(1/3).
            cell_effective[j, i] = float(numpy.mean(block**3)) ** (1.0 / 3.0)

    return PittingField(
        years=duration,
        mean_depth_mm=float(numpy.mean(depths[active])) if bool(active.any()) else 0.0,
        pit_count=int(numpy.count_nonzero(active)),
        dop_raster=float(numpy.mean(coverage > 0.0)),
        dop_nominal=cfg.pitting_intensity,
        net_section_ratio=net_ratio,
        thickness_grid=cell_thickness / 1000.0,
        thickness_effective_grid=cell_effective / 1000.0,
        cell_thickness_min=cell_min / 1000.0,
        correlation_length_m=cfg.correlation_length_m,
        pit_depths_mm=depths,
        seed=cfg.seed,
    )# --------------------------------------------------------------------------- #
# 3. field-resolved elastic buckling of a thickness field
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldBucklingResult:
    critical_line_load: float
    critical_stress: float
    relative_error: float
    grid: tuple[int, int]
    mean_thickness_m: float
    mode_shape: object = field(default=None, repr=False, compare=False)


#: ``k pi^2 / (12 (1 - nu^2))`` for the classical simply supported plate, i.e.
#: ``sigma_E / sigma_y = PLATE_K(nu) / beta^2`` with ``k = 4``.
def plate_k(poisson: float) -> float:
    value = _finite("poisson", poisson)
    if not -1.0 < value < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5")
    return 4.0 * math.pi**2 / (12.0 * (1.0 - value**2))


def _interior_index_map(ny: int, nx: int):
    numpy = _require_numpy()
    table = -numpy.ones((ny, nx), dtype=int)
    counter = 0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            table[j, i] = counter
            counter += 1
    return table, counter


def thickness_field_buckling_fd(
    *, thickness, length_m: float, width_m: float, young: float, poisson: float,
    reference_stress: float | None = None,
) -> FieldBucklingResult:
    """Lowest uniaxial buckling stress of a variable-thickness plate (FD route).

    This is the *verification* route: it discretises the interior and factorises
    a shift-invert Lanczos problem, which is accurate but too slow for Monte
    Carlo.  Production calls use the modal ``thickness_field_buckling`` and the
    two routes are cross-checked in ``run_corrosion_ice_strength_assessment``.

    Original docstring follows: lowest uniaxial buckling stress of a plate with
    a variable thickness field.

    ``thickness`` is a nodal array ``(ny, nx)`` that includes the simply
    supported edges (``w = 0`` there).  The discretisation is the second-order
    central difference on the interior nodes -- the same convention as
    ``marine_plate_buckling.finite_difference_uniaxial_buckling`` -- but the
    rigidities are node-wise::

        K_b = Bxx^T W_D Bxx + Byy^T W_D Byy
              + nu (Bxx^T W_D Byy + Byy^T W_D Bxx)
              + 2 (1 - nu) Bxy^T W_D Bxy
        K_g = Bx^T W Bx,   W_D = diag(D_ij dx dy),
        D_ij = E t_ij^3 / (12 (1 - nu^2))

    and the generalised eigenproblem ``K_b v = N K_g v`` is solved by sparse
    shift-invert, so ``critical_line_load`` is the discrete minimum rather than
    an interpolated wave-number formula.
    """
    numpy = _require_numpy()
    sparse, sparse_linalg = _require_sparse()
    young_v = _positive("young", young)
    poisson_v = _finite("poisson", poisson)
    if not -1.0 < poisson_v < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5")
    length = _positive("length_m", length_m)
    width = _positive("width_m", width_m)
    values = numpy.asarray(thickness, dtype=float)
    if values.ndim != 2:
        raise ValueError("thickness must be a two dimensional array")
    ny, nx = values.shape
    if ny < 4 or nx < 4:
        raise ValueError("thickness needs at least a 4x4 nodal grid")
    if numpy.any(values <= 0.0) or not bool(numpy.all(numpy.isfinite(values))):
        raise ValueError("thickness must be finite and positive")
    dx = length / (nx - 1)
    dy = width / (ny - 1)
    table, n = _interior_index_map(ny, nx)
    rows_xx, cols_xx, vals_xx = [], [], []
    rows_yy, cols_yy, vals_yy = [], [], []
    rows_xy, cols_xy, vals_xy = [], [], []
    rows_x, cols_x, vals_x = [], [], []
    def push(rows, cols, vals, p, entries):
        """Append stencil entries, dropping the simply supported boundary nodes.

        Boundary nodes carry ``w = 0``, so their contribution to the interior row
        is zero and the column is simply omitted (this is the standard
        elimination of prescribed zero dofs, not an approximation).
        """
        for column, value in entries:
            if column >= 0:
                rows.append(p)
                cols.append(column)
                vals.append(value)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            p = table[j, i]
            coef = 1.0 / dx**2
            push(rows_xx, cols_xx, vals_xx, p, (
                (table[j, i - 1], coef), (table[j, i], -2.0 * coef),
                (table[j, i + 1], coef),
            ))
            coef = 1.0 / dy**2
            push(rows_yy, cols_yy, vals_yy, p, (
                (table[j - 1, i], coef), (table[j, i], -2.0 * coef),
                (table[j + 1, i], coef),
            ))
            coef = 1.0 / (4.0 * dx * dy)
            push(rows_xy, cols_xy, vals_xy, p, (
                (table[j - 1, i - 1], coef), (table[j - 1, i + 1], -coef),
                (table[j + 1, i - 1], -coef), (table[j + 1, i + 1], coef),
            ))
            coef = 1.0 / (2.0 * dx)
            push(rows_x, cols_x, vals_x, p, (
                (table[j, i - 1], -coef), (table[j, i + 1], coef),
            ))

    def build(rows, cols, vals):
        return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    bxx = build(rows_xx, cols_xx, vals_xx)
    byy = build(rows_yy, cols_yy, vals_yy)
    bxy = build(rows_xy, cols_xy, vals_xy)
    bx = build(rows_x, cols_x, vals_x)
    d_node = young_v * values**3 / (12.0 * (1.0 - poisson_v**2))
    weight_d = sparse.diags((d_node * dx * dy)[1:-1, 1:-1].ravel())
    weight = sparse.diags(numpy.full(n, dx * dy))
    k_bend = (
        bxx.T @ weight_d @ bxx + byy.T @ weight_d @ byy
        + poisson_v * (bxx.T @ weight_d @ byy + byy.T @ weight_d @ bxx)
        + 2.0 * (1.0 - poisson_v) * bxy.T @ weight_d @ bxy
    ).tocsr()
    k_geom = (bx.T @ weight @ bx).tocsr()
    eigenvalues, eigenvectors = sparse_linalg.eigsh(
        k_bend, k=1, M=k_geom, sigma=0.0, which="LM",
    )
    load = float(eigenvalues[0])
    mode = eigenvectors[:, 0]
    mean_thickness = float(numpy.mean(values))
    stress = load / mean_thickness
    relative = 0.0
    if reference_stress is not None:
        relative = abs(stress / _positive("reference_stress", reference_stress) - 1.0)
    return FieldBucklingResult(
        critical_line_load=load,
        critical_stress=stress,
        relative_error=relative,
        grid=(nx - 2, ny - 2),
        mean_thickness_m=mean_thickness,
        mode_shape=mode,
    )


def _resample_field(values, length_m: float, width_m: float, quadrature):
    """Bilinearly resample a thickness field onto the quadrature lattice."""
    numpy = _require_numpy()
    if quadrature is None:
        return values
    qx, qy = int(quadrature[0]), int(quadrature[1])
    if qx < 3 or qy < 3:
        raise ValueError("quadrature must be at least 3x3")
    ny, nx = values.shape
    source_x = numpy.linspace(0.0, length_m, nx)
    source_y = numpy.linspace(0.0, width_m, ny)
    target_x = numpy.linspace(0.0, length_m, qx)
    target_y = numpy.linspace(0.0, width_m, qy)
    intermediate = numpy.array([numpy.interp(target_x, source_x, row)
                                for row in values])
    return numpy.array([numpy.interp(target_y, source_y, intermediate[:, column])
                        for column in range(qx)]).T


def thickness_field_buckling(
    *, thickness, length_m: float, width_m: float, young: float, poisson: float,
    reference_stress: float | None = None, half_waves_x: int = 8,
    half_waves_y: int = 3, quadrature=(33, 9),
) -> FieldBucklingResult:
    """Lowest uniaxial buckling stress of a variable-thickness plate (modal route).

    Fourier-Galerkin projection on ``sin(m pi x / a) sin(n pi y / b)`` with
    ``m <= half_waves_x``, ``n <= half_waves_y``.  The rigidities are integrated
    on the supplied node lattice with trapezoidal weights, so a spatially varying
    ``D_ij = E t_ij^3 / (12 (1 - nu^2))`` enters exactly as a weighted sum::

        K_b[i,j] = int D [ w_xx^i w_xx^j + w_yy^i w_yy^j
                           + nu (w_xx^i w_yy^j + w_yy^i w_xx^j)
                           + 2 (1-nu) w_xy^i w_xy^j ] dx dy
        K_g[i,j] = int w_x^i w_x^j dx dy

    and the discrete minimum of the pencil ``K_b v = N K_g v`` is returned.  For
    a uniform field the exact classical mode is inside the basis, so the result
    is the classical value up to quadrature error only (checked in the tests);
    for a lumpy field the basis must be rich enough for the perturbation, which
    is why ``half_waves_x``/``half_waves_y`` are exposed and a basis-convergence
    study is part of the qualification.
    """
    numpy = _require_numpy()

    young_v = _positive("young", young)
    poisson_v = _finite("poisson", poisson)
    if not -1.0 < poisson_v < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5")
    length = _positive("length_m", length_m)
    width = _positive("width_m", width_m)
    if isinstance(half_waves_x, bool) or not isinstance(half_waves_x, int):
        raise TypeError("half_waves_x must be an integer")
    if isinstance(half_waves_y, bool) or not isinstance(half_waves_y, int):
        raise TypeError("half_waves_y must be an integer")
    if half_waves_x < 1 or half_waves_y < 1:
        raise ValueError("half_waves counts must be positive")
    values = numpy.asarray(thickness, dtype=float)
    if values.ndim != 2:
        raise ValueError("thickness must be a two dimensional array")
    ny, nx = values.shape
    if ny < 3 or nx < 3:
        raise ValueError("thickness needs at least a 3x3 node lattice")
    if numpy.any(values <= 0.0) or not bool(numpy.all(numpy.isfinite(values))):
        raise ValueError("thickness must be finite and positive")
    values = _resample_field(values, length, width, quadrature)
    ny, nx = values.shape
    x = numpy.linspace(0.0, length, nx)
    y = numpy.linspace(0.0, width, ny)
    weights_x = numpy.full(nx, length / (nx - 1))
    weights_y = numpy.full(ny, width / (ny - 1))
    weights_x[0] *= 0.5
    weights_x[-1] *= 0.5
    weights_y[0] *= 0.5
    weights_y[-1] *= 0.5
    weight = weights_y[:, None] * weights_x[None, :]
    rigidity = young_v * values**3 / (12.0 * (1.0 - poisson_v**2))
    modes = [(m, n) for m in range(1, half_waves_x + 1)
             for n in range(1, half_waves_y + 1)]
    shape = (len(modes), ny * nx)
    phi_xx = numpy.empty(shape)
    phi_yy = numpy.empty(shape)
    phi_xy = numpy.empty(shape)
    phi_x = numpy.empty(shape)
    for index, (m, n) in enumerate(modes):
        alpha = m * math.pi / length
        beta = n * math.pi / width
        sx, cx = numpy.sin(alpha * x), numpy.cos(alpha * x)
        sy, cy = numpy.sin(beta * y), numpy.cos(beta * y)
        phi_xx[index] = numpy.outer(sy, -alpha**2 * sx).ravel()
        phi_yy[index] = numpy.outer(-beta**2 * sy, sx).ravel()
        phi_xy[index] = numpy.outer(beta * cy, alpha * cx).ravel()
        phi_x[index] = numpy.outer(sy, alpha * cx).ravel()
    weighted = (rigidity * weight).ravel()

    def projection(rows_a, rows_b):
        return (rows_a * weighted) @ rows_b.T

    k_bend = (
        projection(phi_xx, phi_xx) + projection(phi_yy, phi_yy)
        + poisson_v * (projection(phi_xx, phi_yy) + projection(phi_yy, phi_xx))
        + 2.0 * (1.0 - poisson_v) * projection(phi_xy, phi_xy)
    )
    k_geom = (phi_x * weight.ravel()) @ phi_x.T
    k_bend = 0.5 * (k_bend + k_bend.T)
    k_geom = 0.5 * (k_geom + k_geom.T)
    # Cholesky reduction to a symmetric dense problem.  ``numpy.linalg`` is used
    # deliberately: the scipy.linalg generalised eigensolver is unusable in the
    # qualification environment (it aborts the interpreter), and the reduced
    # problem is small (tens of modes).
    lower = numpy.linalg.cholesky(k_geom)
    reduced = numpy.linalg.solve(
        lower, numpy.linalg.solve(lower, k_bend).T
    ).T
    reduced = 0.5 * (reduced + reduced.T)
    values_eig, reduced_vectors = numpy.linalg.eigh(reduced)
    vectors = numpy.linalg.solve(lower.T, reduced_vectors)
    load = float(values_eig[0])
    mean_thickness = float(numpy.mean(values))
    stress = load / mean_thickness
    relative = 0.0
    if reference_stress is not None:
        relative = abs(stress / _positive("reference_stress", reference_stress) - 1.0)
    return FieldBucklingResult(
        critical_line_load=load,
        critical_stress=stress,
        relative_error=relative,
        grid=(len(modes), len(modes)),
        mean_thickness_m=mean_thickness,
        mode_shape=vectors[:, 0],
    )


def nodal_thickness_field(cell_thickness):
    """Interpolate a cell-centred thickness field onto the plate node lattice."""
    numpy = _require_numpy()
    cells = numpy.asarray(cell_thickness, dtype=float)
    if cells.ndim != 2:
        raise ValueError("cell_thickness must be two dimensional")
    ny, nx = cells.shape
    nodal = numpy.zeros((ny + 1, nx + 1))
    nodal[1:-1, 1:-1] = 0.25 * (
        cells[:-1, :-1] + cells[1:, :-1] + cells[:-1, 1:] + cells[1:, 1:]
    )
    nodal[0, 1:-1] = 0.5 * (cells[0, :-1] + cells[0, 1:])
    nodal[-1, 1:-1] = 0.5 * (cells[-1, :-1] + cells[-1, 1:])
    nodal[1:-1, 0] = 0.5 * (cells[:-1, 0] + cells[1:, 0])
    nodal[1:-1, -1] = 0.5 * (cells[:-1, -1] + cells[1:, -1])
    nodal[0, 0], nodal[0, -1] = cells[0, 0], cells[0, -1]
    nodal[-1, 0], nodal[-1, -1] = cells[-1, 0], cells[-1, -1]
    return nodal


# --------------------------------------------------------------------------- #
# 4. ultimate strength
# --------------------------------------------------------------------------- #
def faulkner_effective_width(beta: float) -> float:
    """Effective-width collapse ratio ``sigma_u / sigma_y`` for a plate in compression."""
    value = _positive("beta", beta)
    if value <= 1.0:
        return 1.0
    return 2.0 / value - 1.0 / value**2


def paik_pitted_reduction(dop: float, *, coefficient: float = 0.33,
                          exponent: float = 1.0) -> float:
    """Paik-type pitting reduction factor ``R(DOP) = 1 - a DOP**b``.

    The default pair reproduces the secondary-literature trend of roughly 10 %
    loss at ``DOP = 0.3``.  The primary table of Paik et al. (2003) is not
    available offline: callers should treat the pair as an anchor with a stated
    uncertainty (``anchor_sensitivity``) rather than as a code value.
    """
    value = _nonnegative("dop", dop)
    a = _nonnegative("coefficient", coefficient)
    b = _positive("exponent", exponent)
    if value > 1.0:
        raise ValueError("dop must not exceed one")
    return max(0.0, 1.0 - a * value**b)


def anchor_sensitivity(dop_values, *, coefficients=(0.20, 0.33, 0.45),
                       exponents=(0.75, 1.0)):
    """Return the anchor band over the documented coefficient uncertainty."""
    rows = []
    for coefficient in coefficients:
        for exponent in exponents:
            rows.append({
                "coefficient": coefficient,
                "exponent": exponent,
                "reduction": [paik_pitted_reduction(d, coefficient=coefficient,
                                                    exponent=exponent)
                              for d in dop_values],
            })
    return rows


@dataclass(frozen=True)
class UltimateStrengthResult:
    sigma_elastic_mpa: float
    sigma_elastic_intact_mpa: float
    beta_field: float
    beta_intact: float
    sigma_global_mpa: float
    sigma_net_section_mpa: float
    sigma_first_patch_mpa: float
    sigma_ultimate_mpa: float
    sigma_intact_mpa: float
    governing: str
    dop_raster: float
    mean_thickness_mm: float
    net_section_ratio: float
    patch_span_used_m: float
    cluster_threshold_m: float
    elastic_ratio: float
    reduction_factor: float


def plate_ultimate_strength(
    *, field_values: PittingField, young: float, poisson: float,
    yield_stress_mpa: float, length_m: float, width_m: float,
    intact_thickness_m: float | None = None,
) -> UltimateStrengthResult:
    """Ultimate compressive strength of a pitted panel.

    Three mechanisms are evaluated and the minimum governs:

    ``field_global_collapse``
        field-resolved elastic buckling stress promoted to inelastic collapse by
        the effective-width law.  The slenderness comes from the *computed*
        buckling stress, ``beta^2 = plate_k(nu) sigma_y / sigma_E``, which is the
        classical relation for the simply supported plate, so ``DOP = 0``
        reproduces the classical intact result by construction.
    ``net_section_squash``
        gross-section yield scaled by the rasterised removed-volume fraction.
    The local pit scale enters as a diagnostic only: ``sigma_first_patch_mpa``
    is the collapse stress of the thinnest patch at the correlation-cell scale
    and ``cluster_threshold_m`` is the patch size at which that patch becomes
    buckling critical instead of squash critical.  Neither is allowed to govern
    the panel load: a buckling patch sheds load into its neighbours, and turning
    ``min(sigma_first_patch, sigma_global)`` into the answer would be a modelled
    assumption about load shedding that this module does not solve.
    """
    numpy = _require_numpy()
    yield_v = _positive("yield_stress_mpa", yield_stress_mpa)
    young_v = _positive("young", young)
    length = _positive("length_m", length_m)
    width = _positive("width_m", width_m)
    if not isinstance(field_values, PittingField):
        raise TypeError("field_values must be a PittingField")
    nodal = nodal_thickness_field(field_values.thickness_effective_grid)
    buckling = thickness_field_buckling(
        thickness=nodal, length_m=length, width_m=width, young=young_v,
        poisson=poisson,
    )
    sigma_elastic = buckling.critical_stress / 1.0e6
    factor = plate_k(poisson)
    beta_field = math.sqrt(factor * yield_v / sigma_elastic)
    sigma_global = yield_v * faulkner_effective_width(beta_field)
    sigma_net = yield_v * field_values.net_section_ratio

    # Diagnostics for the local scale (never governing on their own): the first
    # patch to buckle and the cluster size at which a patch becomes buckling
    # critical rather than squash critical.
    cell_min = numpy.asarray(field_values.cell_thickness_min, dtype=float)
    if numpy.any(cell_min <= 0.0):
        raise ValueError("cell thickness must be positive")
    patch_span = min(max(field_values.correlation_length_m, 1.0e-3),
                     min(length, width))
    beta_cell = (patch_span / cell_min) * math.sqrt(yield_v * 1.0e6 / young_v)
    first_patch = float(yield_v * min(
        faulkner_effective_width(float(value)) for value in beta_cell.ravel()
    ))
    cluster_threshold = float(numpy.min(cell_min)) * math.sqrt(young_v / (yield_v * 1.0e6))

    candidates = {
        "field_global_collapse": sigma_global,
        "net_section_squash": sigma_net,
    }
    governing = min(candidates, key=candidates.get)
    sigma_ultimate = candidates[governing]

    mean_thickness = float(numpy.mean(numpy.asarray(field_values.thickness_grid)))
    if intact_thickness_m is None:
        nominal = float(numpy.max(numpy.asarray(field_values.thickness_grid)))
    else:
        nominal = _positive("intact_thickness_m", intact_thickness_m)
    sigma_elastic_intact = (
        4.0 * math.pi**2 * young_v * nominal**2
        / (12.0 * (1.0 - poisson**2) * width**2)
    ) / 1.0e6
    beta_intact = math.sqrt(factor * yield_v / sigma_elastic_intact)
    sigma_intact = yield_v * faulkner_effective_width(beta_intact)
    return UltimateStrengthResult(
        sigma_elastic_mpa=sigma_elastic,
        sigma_elastic_intact_mpa=sigma_elastic_intact,
        beta_field=beta_field,
        beta_intact=beta_intact,
        sigma_global_mpa=sigma_global,
        sigma_net_section_mpa=sigma_net,
        sigma_first_patch_mpa=first_patch,
        sigma_ultimate_mpa=min(sigma_ultimate, sigma_net),
        sigma_intact_mpa=sigma_intact,
        governing=governing,
        dop_raster=field_values.dop_raster,
        mean_thickness_mm=mean_thickness * 1000.0,
        net_section_ratio=field_values.net_section_ratio,
        patch_span_used_m=patch_span,
        cluster_threshold_m=cluster_threshold,
        elastic_ratio=sigma_elastic / sigma_elastic_intact,
        reduction_factor=min(sigma_ultimate, sigma_net) / sigma_intact,
    )


# --------------------------------------------------------------------------- #
# 5. ice-side membrane stress (Rayleigh-Ritz, clamped circular idealisation)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MembraneStressResult:
    membrane_stress_mpa: float
    centre_deflection_m: float
    deflection_to_thickness: float
    bending_energy_j: float
    membrane_energy_j: float
    external_work_j: float
    energy_residual: float


def _clamped_shape_family(radius, shape: str, count: int, order: int):
    """Clamped trial family ``phi_k`` (``order=0`` values, ``order=1`` slopes)."""
    numpy = _require_numpy()
    rho = radius / radius[-1] if radius[-1] > 0.0 else radius
    stack = []
    for k in range(1, count + 1):
        if shape == "quadratic":
            base = (1.0 - rho**2) ** (k + 1)
            if order == 0:
                stack.append(base)
            else:
                stack.append(-(k + 1) * (1.0 - rho**2) ** k * 2.0 * rho / radius[-1])
        elif shape == "cosine":
            angle = (2 * k - 1) * math.pi * rho
            if order == 0:
                stack.append(0.5 * (1.0 + numpy.cos(angle)))
            else:
                stack.append(-0.5 * (2 * k - 1) * math.pi
                             * numpy.sin(angle) / radius[-1])
        else:
            raise ValueError("shape must be 'quadratic' or 'cosine'")
    return stack


def clamped_plate_membrane_stress(
    *, pressure_pa: float, radius_m: float, thickness_m: float, young: float,
    poisson: float, points: int = 121, shapes: int = 3,
    shape: str = "quadratic",
) -> MembraneStressResult:
    """Membrane stress of a clamped circular plate under uniform pressure.

    Rayleigh-Ritz in a clamped family ``phi_k`` (``quadratic``:
    ``(1 - r^2/a^2)^(k+1)``; ``cosine``: ``0.5 (1 + cos((2k-1) pi r/a))``, an
    independent family used for the cross-check).  The compatibility equation

    ``r u'' + u' - u/r = -0.5 (w'^2 + r (w'^2)') + 0.5 nu w'^2``

    is linear in ``u`` and quadratic in ``w``, so for ``w = sum a_k phi_k`` the
    in-plane displacement is ``u = sum a_k a_l u_kl`` with ``u_kl`` precomputed by
    central differences (``u(0) = 0`` and the free radial edge
    ``u'(a) + nu u(a)/a + 0.5 w'(a)^2 = 0``).  The total potential
    ``Pi(a) = U_b + U_m - W`` is then a quartic polynomial in ``a`` minimised by
    a backtracking gradient descent, which makes ``shapes`` a genuine
    discretisation parameter: the value converges from above as ``shapes`` grows
    and the convergence is reported by the qualification.

    This is the classical large-deflection *plate* idealisation used to turn a
    local ice pressure into a plate membrane stress; it is not a stiffened-panel
    transfer.
    """
    numpy = _require_numpy()
    q = _nonnegative("pressure_pa", pressure_pa)
    a_radius = _positive("radius_m", radius_m)
    t = _positive("thickness_m", thickness_m)
    e = _positive("young", young)
    nu = _finite("poisson", poisson)
    if not -1.0 < nu < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5")
    if isinstance(shapes, bool) or not isinstance(shapes, int):
        raise TypeError("shapes must be an integer")
    if shapes < 1:
        raise ValueError("shapes must be at least one")
    if points < 16:
        raise ValueError("points must be at least sixteen")
    if q == 0.0:
        return MembraneStressResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    stiffness = e * t / (1.0 - nu**2)
    rigidity = e * t**3 / (12.0 * (1.0 - nu**2))
    r = numpy.linspace(0.0, a_radius, points + 1)
    dr = r[1] - r[0]
    safe_r = numpy.where(r > 0.0, r, 1.0)
    phi = _clamped_shape_family(r, shape, shapes, 0)
    dphi = _clamped_shape_family(r, shape, shapes, 1)
    d2phi = numpy.array([numpy.gradient(slope, dr, edge_order=2)
                         for slope in dphi])
    weight = 2.0 * math.pi * r

    # bending form (quadratic in the coefficients)
    k_bend = numpy.zeros((shapes, shapes))
    for i in range(shapes):
        for j in range(shapes):
            integrand = (
                d2phi[i] * d2phi[j]
                + nu * (dphi[i] * d2phi[j] + d2phi[i] * dphi[j]) / safe_r
                + dphi[i] * dphi[j] / safe_r**2
            )
            k_bend[i, j] = 0.5 * rigidity * float(_trapezoid(integrand * weight, r))
    load_vector = q * numpy.array([float(_trapezoid(phi_i * weight, r))
                                   for phi_i in phi])

    # in-plane displacement for every shape pair (linear in u, quadratic in w)
    pairs = []
    n = points + 1
    for i in range(shapes):
        for j in range(i, shapes):
            multiplicity = 1.0 if i == j else 2.0
            product = dphi[i] * dphi[j]
            rhs = -0.5 * (product + r * numpy.gradient(product, dr, edge_order=2)) \
                + 0.5 * nu * product
            matrix = numpy.zeros((n, n))
            vector = numpy.zeros(n)
            matrix[0, 0] = 1.0
            for index in range(1, n - 1):
                ri = r[index]
                matrix[index, index - 1] = ri / dr**2 - 1.0 / (2.0 * dr)
                matrix[index, index] = -2.0 * ri / dr**2 - 1.0 / ri
                matrix[index, index + 1] = ri / dr**2 + 1.0 / (2.0 * dr)
                vector[index] = rhs[index]
            matrix[n - 1, n - 1] = 3.0 / (2.0 * dr) + nu / a_radius
            matrix[n - 1, n - 2] = -4.0 / (2.0 * dr)
            matrix[n - 1, n - 3] = 1.0 / (2.0 * dr)
            u = numpy.linalg.solve(matrix, vector)
            du = numpy.gradient(u, dr, edge_order=2)
            eps_r = multiplicity * (du + 0.5 * product)
            eps_t = multiplicity * numpy.where(r > 0.0, u / safe_r, du)
            pairs.append((i, j, multiplicity, eps_r, eps_t))

    def potential(coefficients):
        w = numpy.zeros_like(r)
        dw = numpy.zeros_like(r)
        for index, value in enumerate(coefficients):
            w = w + value * phi[index]
            dw = dw + value * dphi[index]
        eps_r = numpy.zeros_like(r)
        eps_t = numpy.zeros_like(r)
        for i, j, multiplicity, pair_r, pair_t in pairs:
            factor = coefficients[i] * coefficients[j] * multiplicity
            eps_r = eps_r + factor * pair_r
            eps_t = eps_t + factor * pair_t
        membrane_integrand = eps_r**2 + 2.0 * nu * eps_r * eps_t + eps_t**2
        membrane = 0.5 * stiffness * float(_trapezoid(membrane_integrand * weight, r))
        bend = float(coefficients @ k_bend @ coefficients)
        work = float(coefficients @ load_vector)
        total = bend + membrane - work
        n_r = stiffness * (eps_r + nu * eps_t)
        return {
            "total": total,
            "bend": bend,
            "membrane": membrane,
            "work": work,
            "stress": float(numpy.max(numpy.abs(n_r))) / t,
            "deflection": float(w[0]),
        }

    # start from the linear clamped-plate amplitude, then descend.
    linear = q * a_radius**4 / (64.0 * rigidity)
    coefficients = numpy.zeros(shapes)
    coefficients[0] = min(linear, 6.0 * t)
    state = potential(coefficients)
    step = 0.2 * max(coefficients[0], 1.0e-9)
    for _ in range(4000):
        gradient = numpy.zeros(shapes)
        for index in range(shapes):
            probe = coefficients.copy()
            delta = max(1.0e-9 * max(abs(coefficients[index]), 1.0), 1.0e-12)
            probe[index] += delta
            gradient[index] = (potential(probe)["total"] - state["total"]) / delta
        norm = float(numpy.linalg.norm(gradient))
        if norm < 1.0e-9 * max(abs(state["total"]), 1.0):
            break
        trial = coefficients - step * gradient
        trial = numpy.maximum(trial, 0.0)
        improved = potential(trial)
        if improved["total"] < state["total"]:
            coefficients, state = trial, improved
            step *= 1.6
        else:
            step *= 0.4
            if step < 1.0e-14 * max(abs(coefficients[0]), 1.0e-9):
                break
    best = state
    total = best["bend"] + best["membrane"]
    residual = abs(total - best["work"]) / best["work"] if best["work"] > 0 else 0.0
    return MembraneStressResult(
        membrane_stress_mpa=best["stress"] / 1.0e6,
        centre_deflection_m=best["deflection"],
        deflection_to_thickness=best["deflection"] / t,
        bending_energy_j=best["bend"],
        membrane_energy_j=best["membrane"],
        external_work_j=best["work"],
        energy_residual=residual,
    )


def membrane_stress_curve(*, config: "PanelConfig", thicknesses,
                          shape: str = "quadratic", shapes: int = 3) -> list[dict]:
    """Evaluate the ice-induced membrane stress over a thickness sweep."""
    rows = []
    for thickness in thicknesses:
        result = ice_membrane_stress_mpa(
            config=config, thickness_m=float(thickness), shape=shape, shapes=shapes,
        )
        rows.append({
            "thickness_mm": float(thickness) * 1000.0,
            "membrane_stress_mpa": result.membrane_stress_mpa,
            "deflection_to_thickness": result.deflection_to_thickness,
            "energy_residual": result.energy_residual,
        })
    return rows


def membrane_stress_at(*, curve: list[dict], thickness_m: float) -> float:
    """Interpolate the membrane stress curve in log-log space."""
    numpy = _require_numpy()
    thickness = _positive("thickness_m", thickness_m)
    xs = numpy.log([row["thickness_mm"] for row in curve])
    ys = numpy.log([max(row["membrane_stress_mpa"], 1.0e-12) for row in curve])
    if xs.size == 1:
        return float(math.exp(ys[0]))
    order = numpy.argsort(xs)
    value = float(numpy.interp(math.log(thickness * 1000.0), xs[order], ys[order]))
    return math.exp(value)


# --------------------------------------------------------------------------- #
# 6. panel configuration, decadal history and inverse problem
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PanelConfig:
    """Everything one ice-belt panel needs for the decadal assessment."""

    length_m: float = 3.2
    width_m: float = 0.8
    thickness_m: float = 0.015
    steel_grade: str = "EH36"
    design_temperature_c: float = -40.0
    young_gpa: float = 206.0
    poisson: float = 0.30
    pitting: PittingSpec | None = None
    corrosion_law: CorrosionLaw = MELCHERS_IMMERSION
    ice: IceDesignLoad = field(default_factory=IceDesignLoad)
    panel_radius_m: float = 0.40
    years: tuple = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
    samples: int = 100
    target_safety_factor: float = 1.5
    reliability_fractile: float = 0.05

    def validated(self) -> "PanelConfig":
        _positive("length_m", self.length_m)
        _positive("width_m", self.width_m)
        _positive("thickness_m", self.thickness_m)
        if self.steel_grade not in ARCTIC_STEELS:
            raise ValueError(f"unknown steel_grade {self.steel_grade!r}")
        _positive("young_gpa", self.young_gpa)
        if not -1.0 < self.poisson < 0.5:
            raise ValueError("poisson must lie between -1 and 0.5")
        _positive("panel_radius_m", self.panel_radius_m)
        if isinstance(self.samples, bool) or not isinstance(self.samples, int):
            raise TypeError("samples must be an integer")
        if self.samples < 1:
            raise ValueError("samples must be at least one")
        _positive("target_safety_factor", self.target_safety_factor)
        fractile = _finite("reliability_fractile", self.reliability_fractile)
        if not 0.0 < fractile < 0.5:
            raise ValueError("reliability_fractile must lie in (0, 0.5)")
        if not self.years:
            raise ValueError("years must not be empty")
        previous = -1.0
        for value in self.years:
            current = _nonnegative("years entry", value)
            if current <= previous:
                raise ValueError("years must be strictly increasing")
            previous = current
        return self

    def pitting_spec(self) -> PittingSpec:
        cfg = self.validated()
        if cfg.pitting is not None:
            return cfg.pitting
        return PittingSpec(
            length_m=cfg.length_m, width_m=cfg.width_m,
            thickness_m=cfg.thickness_m,
        )

    def grade(self) -> SteelGrade:
        return ARCTIC_STEELS[self.validated().steel_grade]

    def yield_stress_mpa(self) -> float:
        cfg = self.validated()
        return cfg.grade().yield_stress_at(cfg.design_temperature_c)

    def young_pa(self) -> float:
        return self.validated().young_gpa * 1.0e9


def ice_membrane_stress_mpa(*, config: PanelConfig, thickness_m: float,
                            shape: str = "quadratic") -> MembraneStressResult:
    """Plate membrane stress produced by the design local ice pressure."""
    cfg = config.validated()
    pressure = cfg.ice.pressure_mpa() * 1.0e6
    return clamped_plate_membrane_stress(
        pressure_pa=pressure, radius_m=cfg.panel_radius_m,
        thickness_m=_positive("thickness_m", thickness_m),
        young=cfg.young_pa(), poisson=cfg.poisson, shape=shape,
    )


@dataclass(frozen=True)
class IceDemandResult:
    """Ice-belt in-plane membrane demand from the design ice force."""

    pressure_mpa: float
    ice_force_mn: float
    line_load_mn_per_m: float
    membrane_stress_mpa: float
    spread_edges: float
    yielded: bool
    plate_screening_mpa: float
    plate_screening_w_over_t: float


def ice_demand(*, config: "PanelConfig", thickness_m: float,
               spread_edges: float = 2.0,
               include_plate_screening: bool = False) -> IceDemandResult:
    """In-plane membrane demand produced by the design local ice force.

    Statics idealisation only: the design ice force ``F = p(A) A_contact`` is
    transferred into the surrounding structure as in-plane membrane action along
    ``spread_edges`` panel edges of length ``width_m``, so the membrane line load
    is ``n = F / (spread_edges * width_m)`` and ``sigma_ice = n / t``.  The
    clamped-plate large-deflection solution (``ice_membrane_stress_mpa``) is
    reported alongside as an independent *local* screening idealisation; it
    exceeds yield for a 2.5 MPa patch on a 0.4 m radius, which is exactly why the
    statics transfer -- not the elastic plate stress -- is used for the margin.
    """
    cfg = config.validated()
    edges = _positive("spread_edges", spread_edges)
    thickness = _positive("thickness_m", thickness_m)
    pressure = cfg.ice.pressure_mpa()
    force = pressure * 1.0e6 * cfg.ice.contact_area_m2
    line_load = force / (edges * cfg.width_m)
    stress = line_load / thickness
    yield_stress = cfg.yield_stress_mpa()
    if include_plate_screening:
        screening = ice_membrane_stress_mpa(config=cfg, thickness_m=thickness)
        screening_stress = screening.membrane_stress_mpa
        screening_ratio = screening.deflection_to_thickness
    else:
        screening_stress = 0.0
        screening_ratio = 0.0
    return IceDemandResult(
        pressure_mpa=pressure,
        ice_force_mn=force / 1.0e6,
        line_load_mn_per_m=line_load / 1.0e6,
        membrane_stress_mpa=stress / 1.0e6,
        spread_edges=edges,
        yielded=stress / 1.0e6 > yield_stress,
        plate_screening_mpa=screening_stress,
        plate_screening_w_over_t=screening_ratio,
    )


def calibrate_ice_pressure(
    *, config: PanelConfig | None = None, utilization: float = 0.55,
) -> PanelConfig:
    """Scale the reference ice pressure so the intact panel sits at ``utilization``.

    The design ice pressure of an ice-class panel is a class-rule output, not an
    input this module can invent.  For a worked example the placeholder pressure
    is therefore calibrated so that the intact panel has
    ``sigma_ice / sigma_u(0) = utilization``, which makes the reserve to the
    ``gamma`` design point explicit instead of accidental.
    """
    cfg = (config or PanelConfig()).validated()
    target = _positive("utilization", utilization)
    intact = panel_realization(config=cfg, years=0.0)[0]
    demand = ice_demand(config=cfg, thickness_m=cfg.thickness_m)
    scale = target * intact.sigma_ultimate_mpa / demand.membrane_stress_mpa
    ice = IceDesignLoad(
        reference_pressure_mpa=cfg.ice.reference_pressure_mpa * scale,
        reference_area_m2=cfg.ice.reference_area_m2,
        area_exponent=cfg.ice.area_exponent,
        contact_area_m2=cfg.ice.contact_area_m2,
        provenance=(
            cfg.ice.provenance + "; reference pressure calibrated so the intact "
            f"panel sits at utilization {target:.2f} (worked example only)"
        ),
    )
    return PanelConfig(**{**cfg.__dict__, "ice": ice})


def panel_realization(
    *, config: PanelConfig, years: float, catalogue=None,
) -> UltimateStrengthResult:
    """Ultimate strength of one panel realisation at one age."""
    cfg = config.validated()
    spec = cfg.pitting_spec()
    field_values = generate_pitting_field(
        spec=spec, years=years, law=cfg.corrosion_law, catalogue=catalogue,
    )
    result = plate_ultimate_strength(
        field_values=field_values, young=cfg.young_pa(), poisson=cfg.poisson,
        yield_stress_mpa=cfg.yield_stress_mpa(), length_m=cfg.length_m,
        width_m=cfg.width_m,
    )
    return result, field_values


def _percentile(values, fraction: float) -> float:
    numpy = _require_numpy()
    data = numpy.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("empty sample set")
    return float(numpy.quantile(data, fraction))


def decadal_history(
    *, config: PanelConfig | None = None, samples: int | None = None,
) -> dict:
    """Monte Carlo decadal margin history ``U(t) = sigma_ice(t) / sigma_u(t)``.

    Each realisation draws one pit census (``PittingSpec.seed``), keeps it for
    every age and only grows the pit depths and activates new pits, so a single
    realisation cannot gain strength with age.  ``sigma_ice`` is evaluated with
    the *current* mean thickness, so plate thinning raises the membrane stress
    as well as lowering the collapse stress.
    """
    numpy = _require_numpy()
    cfg = (config or PanelConfig()).validated()
    count = cfg.samples if samples is None else int(samples)
    if count < 1:
        raise ValueError("samples must be at least one")
    spec = cfg.pitting_spec()
    per_year: dict[float, list[dict]] = {float(year): [] for year in cfg.years}
    for seed in range(count):
        realisation = spec
        realisation = PittingSpec(**{**spec.__dict__, "seed": seed})
        rng = numpy.random.default_rng(seed)
        catalogue = build_pit_catalogue(realisation, rng=rng)
        for year in cfg.years:
            year = float(year)
            result, field_values = panel_realization(
                config=cfg, years=year, catalogue=catalogue,
            )
            thickness = float(numpy.mean(numpy.asarray(field_values.thickness_grid)))
            ice = ice_demand(config=cfg, thickness_m=thickness)
            per_year[year].append({
                "sigma_ultimate_mpa": result.sigma_ultimate_mpa,
                "sigma_ice_mpa": ice.membrane_stress_mpa,
                "dop": result.dop_raster,
                "thickness_mm": thickness * 1000.0,
                "reduction_factor": result.reduction_factor,
                "governing": result.governing,
            })
    rows = []
    monotone_violations = 0
    checked = 0
    monotone_by_sample: list[bool] = [True] * count
    for year in cfg.years:
        year = float(year)
        entries = per_year[year]
        sigma_u = [entry["sigma_ultimate_mpa"] for entry in entries]
        sigma_ice = [entry["sigma_ice_mpa"] for entry in entries]
        utilization = [ice / strength for ice, strength in zip(sigma_ice, sigma_u)]
        rows.append({
            "years": year,
            "dop_p50": _percentile([entry["dop"] for entry in entries], 0.5),
            "thickness_mm_p50": _percentile([entry["thickness_mm"] for entry in entries], 0.5),
            "sigma_ultimate_mpa_p50": _percentile(sigma_u, 0.5),
            "sigma_ultimate_mpa_p05": _percentile(sigma_u, cfg.reliability_fractile),
            "sigma_ultimate_mpa_p95": _percentile(sigma_u, 1.0 - cfg.reliability_fractile),
            "sigma_ice_mpa_p50": _percentile(sigma_ice, 0.5),
            "sigma_ice_mpa_p95": _percentile(sigma_ice, 1.0 - cfg.reliability_fractile),
            "utilization_p50": _percentile(utilization, 0.5),
            "utilization_p95": _percentile(utilization, 0.95),
            "reduction_factor_p50": _percentile(
                [entry["reduction_factor"] for entry in entries], 0.5),
            "governing": max(
                {entry["governing"] for entry in entries},
                key=lambda name: sum(entry["governing"] == name for entry in entries),
            ),
        })
    for index in range(count):
        previous = None
        for year in cfg.years:
            value = per_year[float(year)][index]["sigma_ice_mpa"] / per_year[float(year)][index]["sigma_ultimate_mpa"]
            if previous is not None:
                checked += 1
                if value < previous - 1.0e-9 * max(abs(previous), 1.0):
                    monotone_violations += 1
                    monotone_by_sample[index] = False
            previous = value
    return {
        "schema": "tensorfem.corrosion-ice-history/1.0",
        "samples": count,
        "years": [float(year) for year in cfg.years],
        "rows": rows,
        "monotonicity": {
            "samples_checked": count,
            "comparisons": checked,
            "violations": monotone_violations,
            "monotone_sample_fraction": (
                sum(1 for flag in monotone_by_sample if flag) / count
            ),
            "tolerance": 1.0e-9,
        },
    }


def _interp(xs, ys, value: float) -> float:
    """Monotone piecewise linear interpolation with flat extrapolation."""
    xs = list(xs)
    ys = list(ys)
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for left, right, y0, y1 in zip(xs, xs[1:], ys, ys[1:]):
        if left <= value <= right:
            weight = (value - left) / (right - left)
            return y0 + weight * (y1 - y0)
    return ys[-1]


def allowable_service_life(
    *, history: dict, config: PanelConfig | None = None,
) -> dict:
    """Solve ``sigma_u^fractile(t) = gamma sigma_ice^95(t)`` for the first crossing.

    The reliability route uses the configured fractile of the collapse stress
    and the complementary fractile of the ice membrane stress.  It assumes the
    scatter is time invariant (verified by the reported per-year percentiles),
    which is stated as a limitation rather than silently assumed away.
    """
    cfg = (config or PanelConfig()).validated()
    gamma = cfg.target_safety_factor
    years = [row["years"] for row in history["rows"]]
    capacity = [row["sigma_ultimate_mpa_p05"] for row in history["rows"]]
    demand = [gamma * row["sigma_ice_mpa_p95"] for row in history["rows"]]

    def margin(year: float) -> float:
        return _interp(years, capacity, year) - _interp(years, demand, year)

    horizon = years[-1]
    if margin(0.0) <= 0.0:
        return {
            "allowable_years": 0.0,
            "target_safety_factor": gamma,
            "fractile": cfg.reliability_fractile,
            "status": "already_below_target",
            "horizon_years": horizon,
        }
    if margin(horizon) > 0.0:
        return {
            "allowable_years": None,
            "target_safety_factor": gamma,
            "fractile": cfg.reliability_fractile,
            "status": "beyond_horizon",
            "horizon_years": horizon,
        }
    low, high = 0.0, horizon
    for _ in range(80):
        mid = 0.5 * (low + high)
        if margin(mid) > 0.0:
            low = mid
        else:
            high = mid
    return {
        "allowable_years": 0.5 * (low + high),
        "target_safety_factor": gamma,
        "fractile": cfg.reliability_fractile,
        "status": "governed",
        "horizon_years": horizon,
    }


def allowable_pitting_depth(
    *, config: PanelConfig | None = None, design_years: float | None = None,
    sample_seed: int = 0, iterations: int = 22,
) -> dict:
    """Inverse problem: largest mean pit depth that still meets the target margin.

    The design brief asks for the *allowable wastage* and the inspection
    criterion.  The forward model is parametrised by the mean pit depth at the
    design age, so bisection on that depth is well posed for both depth models
    (``corrosion_law_volume`` and ``depth_ratio``), unlike bisection on the rate
    coefficient of a law whose depth is normalised at a reference age.  The
    capacity is taken at the configured reliability fractile (the nearly time
    invariant ``p05/p50`` ratio of the Monte Carlo history) and the demand at the
    complementary ice fractile.
    """
    numpy = _require_numpy()
    cfg = (config or PanelConfig()).validated()
    target = float(cfg.years[-1] if design_years is None else design_years)
    history = decadal_history(config=cfg, samples=max(2, min(cfg.samples, 12)))
    ratio = 1.0
    for row in history["rows"]:
        if abs(row["years"] - target) < 1.0e-9:
            ratio = row["sigma_ultimate_mpa_p05"] / row["sigma_ultimate_mpa_p50"]
    spec = cfg.pitting_spec()
    rng = numpy.random.default_rng(sample_seed)
    realisation = PittingSpec(**{**spec.__dict__, "seed": sample_seed})
    catalogue = build_pit_catalogue(realisation, rng=rng)
    forward = generate_pitting_field(
        spec=realisation, years=target, law=cfg.corrosion_law, catalogue=catalogue,
    )
    nominal_depth = float(forward.mean_depth_mm)

    def margin(depth_mm: float) -> float:
        field_values = generate_pitting_field(
            spec=realisation, years=target, law=cfg.corrosion_law,
            catalogue=catalogue, forced_mean_depth_mm=depth_mm,
        )
        result = plate_ultimate_strength(
            field_values=field_values, young=cfg.young_pa(), poisson=cfg.poisson,
            yield_stress_mpa=cfg.yield_stress_mpa(), length_m=cfg.length_m,
            width_m=cfg.width_m,
        )
        thickness = float(numpy.mean(numpy.asarray(field_values.thickness_grid)))
        demand = ice_demand(config=cfg, thickness_m=thickness)
        return (ratio * result.sigma_ultimate_mpa
                - cfg.target_safety_factor * demand.membrane_stress_mpa)

    low, high = 0.0, 0.97 * cfg.thickness_m * 1000.0
    at_low, at_high = margin(low), margin(high)
    exposure = max(target - cfg.corrosion_law.initiation_years, 1.0e-9)
    base = {
        "design_years": target,
        "target_safety_factor": cfg.target_safety_factor,
        "fractile_ratio_used": ratio,
        "reference_mean_depth_mm": nominal_depth,
        "margin_extrapolated_years": None,
    }
    if at_low <= 0.0:
        return {**base, "status": "target_unreachable_even_intact",
                "allowable_mean_depth_mm": 0.0}
    if at_high > 0.0:
        return {**base, "status": "not_governed", "allowable_mean_depth_mm": high}
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if margin(mid) > 0.0:
            low = mid
        else:
            high = mid
    depth = 0.5 * (low + high)
    return {
        **base,
        "status": "governed",
        "allowable_mean_depth_mm": depth,
        "allowable_depth_ratio": depth / (cfg.thickness_m * 1000.0),
        "allowable_mean_rate_mm_per_year": depth / exposure,
        "reference_rate_mm_per_year": nominal_depth / exposure,
        "depth_margin_factor": depth / nominal_depth if nominal_depth > 0.0 else None,
    }


def inspection_schedule(
    *, history: dict, config: PanelConfig | None = None,
    interval_cap_years: float = 5.0,
) -> list[dict]:
    """Turn the margin history plus the allowable life into an inspection table."""
    cfg = (config or PanelConfig()).validated()
    cap = _positive("interval_cap_years", interval_cap_years)
    life = allowable_service_life(history=history, config=cfg)
    allowable = life["allowable_years"]
    rows = []
    for row in history["rows"]:
        year = row["years"]
        capacity = row["sigma_ultimate_mpa_p05"]
        demand = cfg.target_safety_factor * row["sigma_ice_mpa_p95"]
        margin = capacity / demand
        if allowable is None:
            interval = cap
            action = "routine"
        else:
            remaining = allowable - year
            interval = max(0.0, min(cap, remaining))
            action = "routine" if margin >= 1.0 else "close_out"
            if remaining <= cap:
                action = "renew_or_dock" if margin < 1.0 else "increase_frequency"
        rows.append({
            "years": year,
            "dop_p50": row["dop_p50"],
            "sigma_ultimate_mpa_p50": row["sigma_ultimate_mpa_p50"],
            "sigma_ultimate_mpa_p05": capacity,
            "design_ice_demand_mpa": demand,
            "margin_ratio": margin,
            "utilization_p50": row["utilization_p50"],
            "next_inspection_interval_years": interval,
            "action": action,
        })
    return rows


def paik_cross_check(
    *, config: PanelConfig | None = None, dop_levels=(0.0, 0.075, 0.15, 0.225, 0.30),
    coefficient: float = 0.33, exponent: float = 1.0,
) -> list[dict]:
    """Compare the field solver with the Paik-type anchor at five DOP levels."""
    cfg = (config or PanelConfig()).validated()
    base = cfg.pitting_spec()
    rows = []
    for level in dop_levels:
        spec = PittingSpec(**{
            **base.__dict__,
            "pitting_intensity": float(level),
            "reference_years": 25.0,
            "depth_model": "depth_ratio",
            "depth_ratio": 0.50,
        })
        field_values = generate_pitting_field(
            spec=spec, years=float(cfg.years[-1]), law=cfg.corrosion_law,
        )
        result = plate_ultimate_strength(
            field_values=field_values, young=cfg.young_pa(), poisson=cfg.poisson,
            yield_stress_mpa=cfg.yield_stress_mpa(), length_m=cfg.length_m,
            width_m=cfg.width_m,
        )
        anchor = paik_pitted_reduction(
            result.dop_raster, coefficient=coefficient, exponent=exponent
        )
        rows.append({
            "dop_nominal": float(level),
            "dop_realized": result.dop_raster,
            "sigma_intact_mpa": result.sigma_intact_mpa,
            "sigma_ultimate_mpa": result.sigma_ultimate_mpa,
            "reduction_computed": result.reduction_factor,
            "reduction_anchor": anchor,
            "relative_error": abs(result.reduction_factor / anchor - 1.0),
            "governing": result.governing,
            "sigma_elastic_ratio": result.elastic_ratio,
        })
    return rows


def intact_reference_check(*, config: PanelConfig | None = None) -> dict:
    """Cross-check the DOP = 0 field solver against the plate-buckling sibling node."""
    from .marine_plate_buckling import (
        finite_difference_uniaxial_buckling, isotropic_plate_rigidities,
        navier_uniaxial_buckling_load,
    )

    cfg = (config or PanelConfig()).validated()
    base = cfg.pitting_spec()
    spec = PittingSpec(**{
        **base.__dict__, "pitting_intensity": 0.0, "clustering_sigma": 0.0,
    })
    field_values = generate_pitting_field(
        spec=spec, years=float(cfg.years[-1]), law=cfg.corrosion_law,
    )
    nodal = nodal_thickness_field(field_values.thickness_grid)
    mean_thickness = float(_require_numpy().mean(nodal))
    rigidity_x, rigidity_y, coupling = isotropic_plate_rigidities(
        cfg.young_pa(), cfg.poisson, mean_thickness
    )
    sibling = finite_difference_uniaxial_buckling(
        length=cfg.length_m, width=cfg.width_m, thickness=mean_thickness,
        rigidity_x=rigidity_x, rigidity_y=rigidity_y, coupling_rigidity=coupling,
        grid_x=nodal.shape[1] - 2, grid_y=nodal.shape[0] - 2,
    )
    field = thickness_field_buckling(
        thickness=nodal, length_m=cfg.length_m, width_m=cfg.width_m,
        young=cfg.young_pa(), poisson=cfg.poisson,
    )
    navier, m, n = navier_uniaxial_buckling_load(
        length=cfg.length_m, width=cfg.width_m, rigidity_x=rigidity_x,
        rigidity_y=rigidity_y, coupling_rigidity=coupling,
    )
    return {
        "field_line_load": field.critical_line_load,
        "sibling_line_load": sibling.critical_line_load,
        "navier_line_load": navier,
        "relative_error_vs_sibling": abs(
            field.critical_line_load / sibling.critical_line_load - 1.0),
        "relative_error_vs_navier": abs(field.critical_line_load / navier - 1.0),
        "sibling_relative_error_vs_navier": abs(
            sibling.critical_line_load / navier - 1.0),
        "half_waves": [m, n],
        "grid": list(field.grid),
        "mean_thickness_mm": mean_thickness * 1000.0,
    }


def grid_convergence_check(*, config: PanelConfig | None = None,
                           grids=((20, 10), (30, 15), (40, 20), (60, 30))) -> list[dict]:
    """Re-solve one pitted realisation on a sequence of eigen grids."""
    cfg = (config or PanelConfig()).validated()
    base = cfg.pitting_spec()
    rows = []
    reference = None
    for grid_x, grid_y in grids:
        spec = PittingSpec(**{
            **base.__dict__, "grid_x": int(grid_x), "grid_y": int(grid_y),
            "pitting_intensity": 0.20,
        })
        field_values = generate_pitting_field(
            spec=spec, years=float(cfg.years[-1]), law=cfg.corrosion_law,
        )
        result = plate_ultimate_strength(
            field_values=field_values, young=cfg.young_pa(), poisson=cfg.poisson,
            yield_stress_mpa=cfg.yield_stress_mpa(), length_m=cfg.length_m,
            width_m=cfg.width_m,
        )
        if reference is None:
            reference = result.sigma_ultimate_mpa
        rows.append({
            "grid": [int(grid_x), int(grid_y)],
            "elements_equivalent": int(grid_x) * int(grid_y),
            "sigma_ultimate_mpa": result.sigma_ultimate_mpa,
            "sigma_elastic_mpa": result.sigma_elastic_mpa,
            "relative_to_finest": None,
        })
    finest = rows[-1]["sigma_ultimate_mpa"]
    for row in rows:
        row["relative_to_finest"] = abs(row["sigma_ultimate_mpa"] / finest - 1.0)
    return rows


def monte_carlo_convergence(
    *, config: PanelConfig | None = None, year: float | None = None,
    levels=(500, 1000),
) -> dict:
    """Compare the P50 collapse stress at two sample counts (same seeds)."""
    numpy = _require_numpy()
    cfg = (config or PanelConfig()).validated()
    target = float(cfg.years[-1] if year is None else year)
    spec = cfg.pitting_spec()
    values: list[float] = []
    largest = max(levels)
    for seed in range(largest):
        realisation = PittingSpec(**{**spec.__dict__, "seed": seed})
        rng = numpy.random.default_rng(seed)
        catalogue = build_pit_catalogue(realisation, rng=rng)
        result, _ = panel_realization(config=cfg, years=target, catalogue=catalogue)
        values.append(result.sigma_ultimate_mpa)
    summary = {}
    for level in levels:
        subset = values[:level]
        summary[level] = {
            "p50": _percentile(subset, 0.5),
            "p05": _percentile(subset, 0.05),
            "p95": _percentile(subset, 0.95),
            "mean": float(numpy.mean(subset)),
            "std": float(numpy.std(subset, ddof=1)),
        }
    coarse, fine = levels[0], levels[-1]
    difference = abs(summary[coarse]["p50"] / summary[fine]["p50"] - 1.0)
    return {
        "year": target,
        "levels": summary,
        "p50_relative_difference": difference,
        "passed": difference < MC_CONVERGENCE_TOLERANCE,
    }


def _intact_elastic_stress_mpa(config: "PanelConfig") -> float:
    """Classical elastic buckling stress of the undamaged panel, in MPa."""
    cfg = config.validated()
    return (
        4.0 * math.pi**2 * cfg.young_pa() * cfg.thickness_m**2
        / (12.0 * (1.0 - cfg.poisson**2) * cfg.width_m**2)
    ) / 1.0e6


def run_corrosion_ice_strength_assessment(
    *, config: PanelConfig | None = None, samples: int | None = None,
    mc_levels=(500, 1000),
) -> dict:
    """Assemble the full evidence payload (the platform endpoint body).

    The platform contract is
    ``POST /api/v1/calculations/structure/corrosion_ice_strength`` -> this
    dictionary, serialised as JSON.
    """
    cfg = (config or PanelConfig()).validated()
    history = decadal_history(config=cfg, samples=samples)
    anchor_rows = paik_cross_check(config=cfg)
    mass_rows = []
    for law in (MELCHERS_IMMERSION, DNV_G101_ARCTIC_ICE_BELT):
        loss = uniform_wastage_mass_loss(
            years=float(cfg.years[-1]), area_m2=cfg.length_m * cfg.width_m,
            density_kg_m3=7850.0, law=law, time_step_years=1.0,
        )
        mass_rows.append({
            "law": law.name,
            "coefficient": law.coefficient,
            "exponent": law.exponent,
            "initiation_years": law.initiation_years,
            "analytic_kg": loss.analytic_kg,
            "discrete_kg": loss.discrete_kg,
            "relative_error": loss.relative_error,
            "thickness_loss_mm": loss.thickness_loss_mm,
        })
    degeneracy = intact_reference_check(config=cfg)
    convergence = monte_carlo_convergence(config=cfg, levels=mc_levels)
    grid_rows = grid_convergence_check(config=cfg)
    schedule = inspection_schedule(history=history, config=cfg)
    life = allowable_service_life(history=history, config=cfg)
    dop_levels = [row["dop_realized"] for row in anchor_rows]
    intact_demand = ice_demand(config=cfg, thickness_m=cfg.thickness_m)
    membrane_shapes = {}
    membrane_w_over_t = {}
    for shape in ("quadratic", "cosine"):
        screening = ice_membrane_stress_mpa(
            config=cfg, thickness_m=cfg.thickness_m, shape=shape,
        )
        membrane_shapes[shape] = screening.membrane_stress_mpa
        membrane_w_over_t[shape] = screening.deflection_to_thickness
    criteria = {
        "a_paik_anchor": {
            "tolerance": PAIK_TOLERANCE,
            "max_relative_error": max(row["relative_error"] for row in anchor_rows),
            "passed": max(row["relative_error"] for row in anchor_rows) < PAIK_TOLERANCE,
            "rows": anchor_rows,
            "anchor_sensitivity": anchor_sensitivity(dop_levels),
        },
        "b_uniform_mass_loss": {
            "tolerance": MASS_LOSS_TOLERANCE,
            "max_relative_error": max(row["relative_error"] for row in mass_rows),
            "passed": max(row["relative_error"] for row in mass_rows) < MASS_LOSS_TOLERANCE,
            "rows": mass_rows,
        },
        "c_degeneracy_dop_zero": {
            "tolerance": DEGENERACY_TOLERANCE,
            "relative_error_vs_sibling": degeneracy["relative_error_vs_sibling"],
            "relative_error_vs_navier": degeneracy["relative_error_vs_navier"],
            "passed": max(degeneracy["relative_error_vs_sibling"],
                          degeneracy["relative_error_vs_navier"]) < DEGENERACY_TOLERANCE,
            "detail": degeneracy,
        },
        "d_monotonicity": {
            "samples": history["monotonicity"]["samples_checked"],
            "violations": history["monotonicity"]["violations"],
            "monotone_sample_fraction": history["monotonicity"]["monotone_sample_fraction"],
            "passed": history["monotonicity"]["violations"] == 0,
            "detail": history["monotonicity"],
        },
        "e_monte_carlo_convergence": {
            "tolerance": MC_CONVERGENCE_TOLERANCE,
            "p50_relative_difference": convergence["p50_relative_difference"],
            "passed": convergence["passed"],
            "detail": convergence,
        },
    }
    grid_error = max(row["relative_to_finest"] for row in grid_rows[:-1])
    payload = {
        "schema": "tensorfem.corrosion-ice-ultimate/1.0",
        "endpoint": "/api/v1/calculations/structure/corrosion_ice_strength",
        "panel": {
            "length_m": cfg.length_m,
            "width_m": cfg.width_m,
            "thickness_mm": cfg.thickness_m * 1000.0,
            "aspect_ratio": cfg.length_m / cfg.width_m,
            "steel_grade": cfg.steel_grade,
            "yield_stress_mpa_rt": cfg.grade().yield_stress_mpa,
            "yield_stress_mpa_design": cfg.yield_stress_mpa(),
            "design_temperature_c": cfg.design_temperature_c,
            "young_gpa": cfg.young_gpa,
            "poisson": cfg.poisson,
            "plate_slenderness_beta": math.sqrt(
                plate_k(cfg.poisson) * cfg.yield_stress_mpa()
                / _intact_elastic_stress_mpa(cfg)
            ),
        },
        "corrosion": {
            "law": cfg.corrosion_law.name,
            "provenance": cfg.corrosion_law.provenance,
            "pitting_intensity_nominal": cfg.pitting_spec().pitting_intensity,
            "pit_diameter_mm": cfg.pitting_spec().pit_diameter_m * 1000.0,
            "depth_ratio": cfg.pitting_spec().depth_ratio,
            "correlation_length_m": cfg.pitting_spec().correlation_length_m,
            "weibull_shape": cfg.pitting_spec().weibull_shape,
            "clustering_sigma": cfg.pitting_spec().clustering_sigma,
        },
        "ice": {
            "design_pressure_mpa": cfg.ice.pressure_mpa(),
            "pressure_area_exponent": cfg.ice.area_exponent,
            "contact_area_m2": cfg.ice.contact_area_m2,
            "provenance": cfg.ice.provenance,
            "demand_model": "design ice force spread as in-plane membrane "
                            "action along two panel edges (statics)",
            "membrane_stress_intact_mpa": intact_demand.membrane_stress_mpa,
            "line_load_mn_per_m": intact_demand.line_load_mn_per_m,
            "ice_force_mn": intact_demand.ice_force_mn,
            "yielded_at_design_load": intact_demand.yielded,
            "plate_screening_mpa": membrane_shapes,
            "plate_screening_w_over_t": membrane_w_over_t,
            "plate_screening_family_spread": abs(
                membrane_shapes["quadratic"] / membrane_shapes["cosine"] - 1.0),
        },
        "history": history,
        "inspection": {
            "allowable_life": life,
            "schedule": schedule,
            "target_safety_factor": cfg.target_safety_factor,
        },
        "criteria": criteria,
        "grid_convergence": {
            "rows": grid_rows,
            "max_error_of_coarser_grids": grid_error,
            "passed": grid_error < TOLERANCE,
        },
        "passed": all(entry["passed"] for entry in criteria.values())
                  and grid_error < TOLERANCE,
    }
    return payload


def render_summary_markdown(payload: dict) -> str:
    """Render a short human-readable summary of the payload."""
    panel = payload["panel"]
    lines = [
        f"# Corrosion-ice ultimate strength: {panel['steel_grade']} panel",
        "",
        f"* panel {panel['length_m']:.2f} x {panel['width_m']:.2f} x "
        f"{panel['thickness_mm']:.1f} mm, beta = {panel['plate_slenderness_beta']:.2f}",
        f"* design temperature {panel['design_temperature_c']:.0f} C, "
        f"yield {panel['yield_stress_mpa_design']:.1f} MPa",
        f"* design ice pressure {payload['ice']['design_pressure_mpa']:.2f} MPa, "
        f"intact membrane stress "
        f"{payload['ice']['membrane_stress_intact_mpa']['quadratic']:.1f} MPa",
        "",
        "| year | DOP p50 | sigma_u p50 (MPa) | U p50 | margin |",
        "|---|---|---|---|---|",
    ]
    for row in payload["history"]["rows"]:
        lines.append(
            f"| {row['years']:.0f} | {row['dop_p50']:.3f} | "
            f"{row['sigma_ultimate_mpa_p50']:.1f} | {row['utilization_p50']:.3f} | "
            f"{'-'} |"
        )
    lines.append("")
    for name, entry in payload["criteria"].items():
        lines.append(f"* {name}: passed={entry['passed']}")
    return "\n".join(lines)