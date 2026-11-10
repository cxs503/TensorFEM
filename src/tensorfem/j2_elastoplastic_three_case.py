"""Three-case qualification surface for J2 (von Mises) elastoplasticity.

Why this module
---------------
The constitutive pieces already exist in the tree and are qualified separately:

* :mod:`tensorfem.plasticity` / :mod:`tensorfem.j2_elastoplasticity` -- the
  backward-Euler radial return map (Wilkins 1964) with **linear isotropic
  hardening**, its consistent tangent and a posteriori residuals;
* :mod:`tensorfem.j2_saturation_hardening` -- the implicit (Voce + linear)
  hardening kernel and an *independent* 1-D closed-form oracle;
* :mod:`tensorfem.thick_cylinder_plasticity` -- Hill's plane-strain
  elastic-plastic tube solution and an axisymmetric Q4 model.

What is *not* in the tree is a single surface that asks the classical question
a constitutive model has to answer before it may be used in a strength
assessment: does the same material card reproduce (i) a uniaxial tensile curve
(with the yield point), (ii) the pure-shear yield point and shear curve, and
(iii) the plastic-zone growth of a pressure vessel, each of them against an
oracle that was *not* produced by the code under test?  This module is that
surface.  It adds three things the existing pieces do not have:

1. **One material card, three cases.**  ``E = 210 GPa``, ``nu = 0.3``,
   ``sigma_y = 250 MPa``, ``H = 1000 MPa`` is instantiated once and driven
   through a TET4 displacement-controlled tensile bar, a material-point
   pure-shear path (the thin-wall torsion idealisation, ``t/R = 0.04``) and the
   plane-strain axisymmetric thick-walled cylinder, so a card that is wrong in
   one place cannot hide behind a per-case re-parameterisation.
2. **Independent oracles.**  The uniaxial reference is
   :func:`tensorfem.j2_saturation_hardening.uniaxial_reference` (scalar bisection
   on ``E (eps - alpha) = sigma_y(alpha)``); the shear reference is the closed
   form ``tau = (sigma_y0 + H lambda)/sqrt(3)``; the cylinder reference is Hill
   1950 with the radial displacement obtained by quadrature, not by a plastic
   closed form.  The FE path never sees any of them.
3. **An h-refinement / GCI study of the plastic-zone quantity**, wired to
   :func:`tensorfem.convergence_verification.evaluate_convergence_study`, so the
   discretisation error of the answer (``u_r`` at the bore, ``r_p``) is a
   measured number and not an assumption.

Case definitions (all SI, metres)
--------------------------------
* **A - uniaxial tension**: ``L = 100 mm``, ``A = 10 x 10 mm``, displacement
  control ``eps = 0 .. 5 %``.  The exact solution is *homogeneous*, therefore
  the bar isolates the constitutive integration + incremental driver (a TET4
  can represent the uniform field, so the case has no spatial error to hide
  behind) -- see :func:`uniaxial_tension_fem`.
* **B - pure shear**: thin-wall tube ``L = 100 mm``, ``R = 50 mm``,
  ``t = 2 mm``.  A thin-walled tube carries an essentially uniform shear stress
  through the wall, so the verification object is the material point under the
  strain state ``eps_xy = gamma/2``; ``tau_y = sigma_y/sqrt(3)`` and the plastic
  shear slope ``G H / (3G + H)`` are the oracles.  A TET4 patch test
  (:func:`pure_shear_patch_test`) additionally shows that the *assembled* code
  path reproduces the same homogeneous state.
* **C - thick-walled cylinder**: ``R_i = 50 mm``, ``R_o = 100 mm``,
  ``L = 200 mm``, internal pressure.  The plastic-zone radius ``r_p(p)`` and the
  radial displacement field ``u_r(r)`` are compared with Hill 1950 (perfect
  plasticity, the case that has the closed form) and the same problem is re-run
  with ``H = 1000 MPa`` to report how hardening moves the answer (no closed form
  exists for that branch, so it is verified by mesh refinement and by
  degeneracy to the ``H -> 0`` solution only).

Honesty statement
-----------------
* Infinitesimal strain, isotropic elasticity, J2 yield surface, *linear*
  isotropic hardening, associative flow rule, monotonic radial loading.  No
  kinematic/combined hardening, no rate or thermal effects, no damage, no
  finite strain, no fracture.  ``eps = 5 %`` is at the edge of the
  infinitesimal-strain contract and is reported as such.
* The mission's stated cylinder pressure band ``p = 0 .. 100 MPa`` produces **no**
  plastic zone for this card: first yield is at ``p_y = 108.25 MPa`` and the
  plastic-collapse pressure is ``p_lim = 200.09 MPa``.  The band is therefore
  reported twice: as an elastic-only check over 0..100 MPa
  (:func:`cylinder_elastic_band`) and as five *contained-plastic* levels
  ``p/k in {1.00, 1.10, 1.20, 1.30, 1.35}`` (``144..195 MPa``) where the
  plastic-zone question actually exists.  Deviating from the literal band is a
  physics requirement, not a relaxation of the criterion.
* No experimental (ASTM E8) tensile dataset is part of this repository and the
  worktree has no network access, so the uniaxial comparison is made against the
  analytic material-card curve.  The experimental chord is **not** claimed: it
  is recorded as ``withheld`` in :func:`run_qualification` and in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .convergence_verification import ConvergenceLevel, evaluate_convergence_study
from .j2_elastoplasticity import (
    J2Material,
    pure_shear_yield_point,
    radial_return,
    uniaxial_stress_path,
    yield_probe,
)
from .j2_saturation_hardening import HardeningLaw, uniaxial_reference
from .solid3d import hex_to_tet_mesh, structured_hex_mesh
from .solid_plasticity import PlasticSolidState, Tet4J2Model, assemble_tet4_j2
from .thick_cylinder_plasticity import (
    HillPlaneStrainTube,
    annulus_mesh,
    solve_plastic_tube,
)

Tensor = torch.Tensor

__all__ = [
    "UniaxialSpec",
    "TorsionSpec",
    "CylinderSpec",
    "bilinear_uniaxial_stress",
    "bilinear_shear_stress",
    "uniaxial_tension_material_point",
    "uniaxial_tension_fem",
    "pure_shear_material_point",
    "pure_shear_patch_test",
    "pi_plane_yield_surface",
    "cylinder_elastic_band",
    "thick_wall_cylinder",
    "plastic_pressure_levels",
    "mesh_convergence_gci",
    "run_qualification",
]

_SQRT3 = math.sqrt(3.0)

#: Contained-plastic pressure levels (multiples of the shear yield stress ``k``).
#: ``p/k = 1`` is four times first yield; ``p/k = 1.35`` puts the interface at
#: ``r_p ~ 0.85 b``; ``2 ln(b/a) = 1.386`` would be collapse.
PLASTIC_PRESSURE_RATIOS = (1.00, 1.10, 1.20, 1.30, 1.35)

#: The literal mission band, which is elastic for this material card.
ELASTIC_PRESSURE_BAND = (20.0e6, 40.0e6, 60.0e6, 80.0e6, 100.0e6)


# --------------------------------------------------------------------------- #
# material / geometry cards
# --------------------------------------------------------------------------- #
def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


@dataclass(frozen=True)
class UniaxialSpec:
    """Standard tensile coupon (ASTM E8 proportions, SI units)."""

    length: float = 0.100
    width: float = 0.010
    young: float = 210.0e9
    poisson: float = 0.3
    yield_stress: float = 250.0e6
    hardening: float = 1000.0e6

    def __post_init__(self) -> None:
        for name in ("length", "width", "young", "yield_stress"):
            _require_positive(name, getattr(self, name))
        if self.hardening < 0.0:
            raise ValueError("hardening modulus must be non-negative")
        if not -1.0 < self.poisson < 0.5:
            raise ValueError("poisson ratio must lie in (-1, 0.5)")

    @property
    def area(self) -> float:
        return self.width * self.width

    @property
    def material(self) -> J2Material:
        return J2Material(self.young, self.poisson, self.yield_stress, self.hardening)

    @property
    def yield_strain(self) -> float:
        return self.yield_stress / self.young

    @property
    def tangent_modulus(self) -> float:
        """Elastoplastic tangent ``E H / (E + H)`` of the bilinear card."""
        return self.young * self.hardening / (self.young + self.hardening)

    def bilinear_stress(self, strain: float) -> float:
        return bilinear_uniaxial_stress(strain, self.young, self.yield_stress, self.hardening)


@dataclass(frozen=True)
class TorsionSpec:
    """Thin-walled tube in torsion (``R`` = mid-surface radius, ``t`` = wall)."""

    length: float = 0.100
    radius: float = 0.050
    thickness: float = 0.002
    young: float = 210.0e9
    poisson: float = 0.3
    yield_stress: float = 250.0e6
    hardening: float = 1000.0e6

    def __post_init__(self) -> None:
        for name in ("length", "radius", "thickness", "young", "yield_stress"):
            _require_positive(name, getattr(self, name))
        if self.thickness >= self.radius:
            raise ValueError("thin-wall idealisation requires t << R")
        if self.hardening < 0.0:
            raise ValueError("hardening modulus must be non-negative")

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def shear_yield_stress(self) -> float:
        return self.yield_stress / _SQRT3

    @property
    def shear_yield_strain(self) -> float:
        return self.shear_yield_stress / self.shear_modulus

    @property
    def plastic_shear_modulus(self) -> float:
        """``d tau / d gamma`` on the plastic branch: ``G H / (3G + H)``."""
        g, h = self.shear_modulus, self.hardening
        return g * h / (3.0 * g + h)

    @property
    def polar_modulus(self) -> float:
        """Thin-wall polar section modulus ``W_t = 2 pi R^2 t`` (``tau = T / W_t``)."""
        return 2.0 * math.pi * self.radius ** 2 * self.thickness

    def bilinear_shear(self, gamma: float) -> float:
        return bilinear_shear_stress(gamma, self.shear_modulus, self.yield_stress,
                                     self.hardening)


@dataclass(frozen=True)
class CylinderSpec:
    """Plane-strain thick-walled cylinder (closed ends are *not* modelled)."""

    inner_radius: float = 0.050
    outer_radius: float = 0.100
    length: float = 0.200
    young: float = 210.0e9
    poisson: float = 0.3
    yield_stress: float = 250.0e6
    hardening: float = 0.0

    def __post_init__(self) -> None:
        for name in ("inner_radius", "outer_radius", "length", "young", "yield_stress"):
            _require_positive(name, getattr(self, name))
        if not self.outer_radius > self.inner_radius:
            raise ValueError("outer_radius must exceed inner_radius")
        if self.hardening < 0.0:
            raise ValueError("hardening modulus must be non-negative")

    def hill(self, *, hardening: float | None = None) -> HillPlaneStrainTube:
        if hardening not in (None, 0.0):
            raise ValueError("Hill's closed form exists for perfect plasticity only")
        return HillPlaneStrainTube(
            inner_radius=self.inner_radius,
            outer_radius=self.outer_radius,
            yield_stress=self.yield_stress,
            young=self.young,
            poisson=self.poisson,
        )

    @property
    def shear_yield_stress(self) -> float:
        return self.yield_stress / _SQRT3

    @property
    def pressure_first_yield(self) -> float:
        k = self.shear_yield_stress
        return k * (1.0 - (self.inner_radius / self.outer_radius) ** 2)

    @property
    def pressure_limit(self) -> float:
        return 2.0 * self.shear_yield_stress * math.log(self.outer_radius / self.inner_radius)

    @property
    def hardening_law(self) -> HardeningLaw:
        return HardeningLaw(self.yield_stress, linear=self.hardening)


# --------------------------------------------------------------------------- #
# closed-form uniaxial / shear references
# --------------------------------------------------------------------------- #
def bilinear_uniaxial_stress(strain: float, young: float, yield_stress: float,
                             hardening: float) -> float:
    """Piecewise bilinear ``sigma(eps)`` of the material card.

    Elastic branch ``sigma = E eps`` up to ``eps_y = sigma_y / E``, then
    ``sigma = sigma_y + E_t (eps - eps_y)`` with ``E_t = E H / (E + H)``.
    """
    eps = float(strain)
    eps_y = yield_stress / young
    if eps <= eps_y:
        return young * eps
    tangent = young * hardening / (young + hardening)
    return yield_stress + tangent * (eps - eps_y)


def bilinear_shear_stress(gamma: float, shear_modulus: float, yield_stress: float,
                          hardening: float) -> float:
    """Closed-form ``tau(gamma)`` for a pure-shear path under J2 + linear hardening.

    Yield at ``tau_y = sigma_y/sqrt(3)``, plastic slope ``G H / (3G + H)``; the
    branch is the solution of ``sqrt(3) G (gamma - sqrt(3) lambda) = sigma_y0 + H
    lambda``.
    """
    g, sy, h = float(shear_modulus), float(yield_stress), float(hardening)
    gamma = float(gamma)
    tau_y = sy / _SQRT3
    gamma_y = tau_y / g
    if gamma <= gamma_y:
        return g * gamma
    lam = (_SQRT3 * g * gamma - sy) / (3.0 * g + h)
    return (sy + h * lam) / _SQRT3


# --------------------------------------------------------------------------- #
# case A - uniaxial tension
# --------------------------------------------------------------------------- #
def uniaxial_tension_material_point(spec: UniaxialSpec, strains) -> dict:
    """Uniaxial-**stress** path at the material point, two independent routes.

    Route 1 is the production kernel (:func:`tensorfem.j2_elastoplasticity.
    uniaxial_stress_path` -> radial return + consistent-tangent lateral
    Newton).  Route 2 is the independent scalar oracle of
    :func:`tensorfem.j2_saturation_hardening.uniaxial_reference`.  Route 3 is the
    closed form :func:`bilinear_uniaxial_stress`.
    """
    levels = [float(s) for s in strains]
    if not levels:
        raise ValueError("at least one strain level is required")
    material = spec.material
    path = uniaxial_stress_path(material, levels)
    law = HardeningLaw(spec.yield_stress, linear=spec.hardening)
    rows = []
    for index, eps in enumerate(levels):
        kernel = float(path["stress"][index])
        reference = float(uniaxial_reference(eps, law, spec.young)["stress"])
        closed = bilinear_uniaxial_stress(eps, spec.young, spec.yield_stress,
                                          spec.hardening)
        rows.append({
            "strain": eps,
            "stress_kernel": kernel,
            "stress_oracle": reference,
            "stress_closed_form": closed,
            "alpha": float(path["equivalent_plastic_strain"][index]),
            "lateral_strain": float(path["lateral_strain"][index]),
            "relative_error_oracle": abs(kernel - reference) / reference,
            "relative_error_closed_form": abs(kernel - closed) / closed,
            "elastic": bool(path["elastic"][index]),
        })
    return {
        "case": "A_uniaxial_material_point",
        "levels": rows,
        "max_relative_error_oracle": max(r["relative_error_oracle"] for r in rows),
        "max_relative_error_closed_form": max(r["relative_error_closed_form"] for r in rows),
        "incremental_modulus": path["incremental_modulus"],
        "yield_strain": spec.yield_strain,
        "tangent_modulus": spec.tangent_modulus,
    }


def _solve_displacement_control(model: Tet4J2Model, prescribed: dict[int, float],
                                state: PlasticSolidState, start: Tensor,
                                *, tolerance: float = 1e-10,
                                max_iterations: int = 40) -> tuple[Tensor, Tensor, PlasticSolidState, int, float]:
    """Newton with prescribed dofs; returns ``(u, element stress, state, iters, res)``."""
    n = model.n_dofs
    mask = torch.ones(n, dtype=torch.bool)
    mask[model.fixed_dofs.long()] = False
    pres = torch.tensor(sorted(prescribed), dtype=torch.long)
    mask[pres] = False
    free = torch.arange(n)[mask]
    u = start.clone()
    for dof, value in prescribed.items():
        u[dof] = value
    residual_norm = float("inf")
    for iteration in range(1, max_iterations + 1):
        internal, tangent, stress, trial = assemble_tet4_j2(model, u, state)
        residual = -internal
        residual[pres] = 0.0
        residual_norm = float(torch.linalg.vector_norm(residual[free]))
        scale = max(float(torch.linalg.vector_norm(internal)), 1.0)
        if residual_norm <= tolerance * scale:
            return u, stress, trial, iteration, residual_norm
        if free.numel() == 0:  # pragma: no cover - defensive
            break
        delta = torch.linalg.solve(tangent[free][:, free], residual[free])
        u = u.index_add(0, free, delta)
        for dof, value in prescribed.items():
            u[dof] = value
    raise RuntimeError(
        f"displacement-controlled Newton did not converge (residual {residual_norm:.3e})"
    )


def _bar_model(spec: UniaxialSpec, mesh: tuple[int, int, int]) -> tuple[Tet4J2Model, list[int]]:
    """One-eighth tensile bar with symmetry planes at ``x = y = z = 0``."""
    nx, ny, nz = (int(v) for v in mesh)
    if min(nx, ny, nz) < 1:
        raise ValueError("mesh must have at least one element per direction")
    nodes, hexes = structured_hex_mesh(spec.length, 0.5 * spec.width, 0.5 * spec.width,
                                       nx, ny, nz)
    nodes, elements = hex_to_tet_mesh(nodes, hexes)
    fixed = []
    for node, (x, y, z) in enumerate(nodes.tolist()):
        if abs(x) < 1e-12:
            fixed.append(3 * node)
        if abs(y) < 1e-12:
            fixed.append(3 * node + 1)
        if abs(z) < 1e-12:
            fixed.append(3 * node + 2)
    model = Tet4J2Model(nodes, elements, spec.young, spec.poisson, spec.yield_stress,
                        spec.hardening, torch.tensor(sorted(fixed), dtype=torch.long))
    loaded = [node for node in range(nodes.shape[0])
              if abs(float(nodes[node, 0]) - spec.length) < 1e-12]
    return model, loaded


def uniaxial_tension_fem(spec: UniaxialSpec, strains,
                         *, mesh: tuple[int, int, int] = (8, 1, 1)) -> dict:
    """Displacement-controlled TET4 tensile bar (one-eighth model).

    The exact solution of this bar is *homogeneous* (uniform axial stress, free
    lateral contraction ``eps_yy = eps_zz = -eps_pp/2`` in the plastic range), and
    a linear tetrahedron represents a uniform strain field exactly; the case
    therefore isolates the constitutive update and the incremental driver with
    no spatial error to hide behind.  The axial stress is measured twice: from
    the element stress and from the reaction (load-cell) sum over the loaded
    face, divided by the quarter area.
    """
    levels = [float(s) for s in strains]
    if not levels:
        raise ValueError("at least one strain level is required")
    if any(level <= 0.0 for level in levels):
        raise ValueError("strain levels must be positive (monotonic tension)")
    model, loaded = _bar_model(spec, mesh)
    state = PlasticSolidState.virgin(model)
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype)
    area = 0.25 * spec.area
    rows = []
    for eps in levels:
        prescribed = {3 * node: eps * spec.length for node in loaded}
        u, stress, state, iterations, residual = _solve_displacement_control(
            model, prescribed, state, u)
        sigma_element = float(stress[:, 0].mean())
        internal, _, _, _ = assemble_tet4_j2(model, u, state)
        reaction = sum(float(internal[3 * node]) for node in loaded) / area
        reference = spec.bilinear_stress(eps)
        rows.append({
            "strain": eps,
            "stress_element": sigma_element,
            "stress_reaction": reaction,
            "stress_reference": reference,
            "alpha": float(max(float(s.alpha) for s in state.points)),
            "relative_error_element": abs(sigma_element - reference) / reference,
            "relative_error_reaction": abs(reaction - reference) / reference,
            "newton_iterations": iterations,
            "residual": residual,
        })
    return {
        "case": "A_uniaxial_tet4_bar",
        "mesh": {"elements_axial": int(mesh[0]), "elements_transverse": int(mesh[1]),
                 "n_elements": int(model.elements.shape[0]),
                 "n_nodes": int(model.nodes.shape[0])},
        "levels": rows,
        "max_relative_error_element": max(r["relative_error_element"] for r in rows),
        "max_relative_error_reaction": max(r["relative_error_reaction"] for r in rows),
        "yield_stress_estimate": _yield_stress_from_curve(
            [r["strain"] for r in rows], [r["stress_element"] for r in rows],
            spec.young, spec.tangent_modulus),
        "yield_stress": spec.yield_stress,
    }


def _yield_stress_from_curve(strains, stresses, young: float, tangent: float) -> float:
    """Yield stress re-measured from a computed curve (offset-free bilinear fit).

    On the plastic branch ``sigma = E_t eps + sigma_y (1 - E_t/E)``, so the
    intercept ``b = sigma - E_t eps`` of the plastic points gives
    ``sigma_y = b / (1 - E_t/E)``.  This is the "0.2 %-style" construction
    without hand-copied numbers: the yield point is *read back* from the curve.
    """
    plastic = []
    for eps, sigma in zip(strains, stresses):
        # plastic branch: sigma = E_t eps + b, strictly below the elastic line
        if float(sigma) < young * float(eps) * (1.0 - 1e-6):
            plastic.append(float(sigma) - tangent * float(eps))
    if not plastic:
        raise RuntimeError("no plastic points on the curve; cannot read a yield stress")
    intercept = sum(plastic) / len(plastic)
    return intercept / (1.0 - tangent / young)


# --------------------------------------------------------------------------- #
# case B - pure shear (thin-wall tube torsion idealisation)
# --------------------------------------------------------------------------- #
def pure_shear_material_point(spec: TorsionSpec, shears) -> dict:
    """Material-point pure-shear path plus the assembled TET4 patch test hook.

    The strain state is ``eps_xy = gamma/2`` only; for an isotropic J2 material
    that maps to ``tau_xy`` only, which is the thin-wall tube solution
    (``t/R = 0.04`` makes the through-wall gradient 4 % of ``tau``).
    """
    levels = [float(g) for g in shears]
    if not levels:
        raise ValueError("at least one shear level is required")
    material = J2Material(spec.young, spec.poisson, spec.yield_stress, spec.hardening)
    state = material.virgin_state()
    rows = []
    for gamma in levels:
        strain = torch.zeros((3, 3), dtype=torch.float64)
        strain[0, 1] = strain[1, 0] = 0.5 * gamma
        result = radial_return(strain, material, state)
        state = result.state
        tau = float(result.stress[0, 1])
        reference = bilinear_shear_stress(gamma, spec.shear_modulus, spec.yield_stress,
                                          spec.hardening)
        rows.append({
            "gamma": gamma,
            "tau_kernel": tau,
            "tau_reference": reference,
            "alpha": float(result.state.alpha),
            "equivalent_stress_over_sqrt3": float(result.equivalent_stress) / _SQRT3,
            "plastic": not result.elastic,
            "relative_error": abs(tau - reference) / reference,
            "residual": max(result.residual.values()),
        })
    yield_point = pure_shear_yield_point(material)
    return {
        "case": "B_pure_shear",
        "levels": rows,
        "max_relative_error": max(r["relative_error"] for r in rows),
        "yield_point": yield_point,
        "tau_y": yield_point["tau_y"],
        "tau_y_analytic": yield_point["tau_y_analytic"],
        "tau_y_relative_error": yield_point["tau_relative_error"],
        "gamma_y": yield_point["gamma_y"],
        "plastic_shear_modulus_numeric": _plastic_shear_modulus(
            rows, spec.shear_modulus),
        "plastic_shear_modulus_analytic": spec.plastic_shear_modulus,
        "torque_at_yield": yield_point["tau_y"] * spec.polar_modulus,
        "polar_modulus": spec.polar_modulus,
    }


def _plastic_shear_modulus(rows, shear_modulus: float) -> float:
    """Secant-free plastic slope from the last two plasticised levels."""
    plastic = [r for r in rows if r["plastic"]]
    if len(plastic) < 2:
        raise RuntimeError("need at least two plastic levels to measure a slope")
    a, b = plastic[-2], plastic[-1]
    return (b["tau_kernel"] - a["tau_kernel"]) / (b["gamma"] - a["gamma"])


def pure_shear_patch_test(spec: TorsionSpec, gamma: float,
                          *, mesh: tuple[int, int, int] = (2, 2, 2)) -> dict:
    """TET4 patch test: does the assembled code path reproduce homogeneous shear?

    Every boundary node is driven by the homogeneous field ``u_x = gamma y``,
    ``u_y = u_z = 0``; the interior nodes are free.  Consistency of the
    element then requires the interior stresses to equal the material-point
    answer, i.e. the test detects a mismatch between assembly and constitutive
    integration (it does not test a boundary-value problem -- that is case C).
    """
    size = 0.010
    nx, ny, nz = (int(v) for v in mesh)
    nodes, hexes = structured_hex_mesh(size, size, size, nx, ny, nz)
    nodes, elements = hex_to_tet_mesh(nodes, hexes)
    boundary = []
    for node, (x, y, z) in enumerate(nodes.tolist()):
        on_face = (abs(x) < 1e-12 or abs(y) < 1e-12 or abs(z) < 1e-12
                   or abs(x - size) < 1e-12 or abs(y - size) < 1e-12
                   or abs(z - size) < 1e-12)
        if on_face:
            boundary.append((node, x, y))
    prescribed = {}
    for node, _, y in boundary:
        prescribed[3 * node] = gamma * y
        prescribed[3 * node + 1] = 0.0
        prescribed[3 * node + 2] = 0.0
    model = Tet4J2Model(nodes, elements, spec.young, spec.poisson, spec.yield_stress,
                        spec.hardening, torch.tensor([], dtype=torch.long))
    state = PlasticSolidState.virgin(model)
    u, stress, state, iterations, residual = _solve_displacement_control(
        model, prescribed, state, torch.zeros(model.n_dofs, dtype=nodes.dtype))
    tau = float(stress[:, 3].mean())
    reference = bilinear_shear_stress(gamma, spec.shear_modulus, spec.yield_stress,
                                      spec.hardening)
    return {
        "case": "B_pure_shear_tet4_patch",
        "gamma": gamma,
        "tau_patch": tau,
        "tau_reference": reference,
        "relative_error": abs(tau - reference) / reference,
        "stress_spread": float(stress[:, 3].max() - stress[:, 3].min()),
        "newton_iterations": iterations,
        "residual": residual,
        "n_interior_nodes": int(model.nodes.shape[0] - len(boundary)),
    }


# --------------------------------------------------------------------------- #
# C6 - yield surface in the pi plane
# --------------------------------------------------------------------------- #
def pi_plane_yield_surface(spec: UniaxialSpec | TorsionSpec | CylinderSpec,
                           *, directions: int = 6) -> dict:
    """Probe the yield surface along ``directions`` deviatoric directions.

    The direction lattice is ``cos(theta) s1 + sin(theta) s2`` with
    ``s1 = diag(1, -1/2, -1/2)`` and ``s2 = (0, 1, -1)/sqrt(2)`` sampled at
    ``theta = 2 pi j / directions``.  A von Mises surface must return the same
    onset stress :math:`q = \\sigma_y` for every direction; the test therefore
    measures the pi-plane circularity of the *implemented* return map.
    """
    if directions < 3:
        raise ValueError("at least three directions are required")
    material = J2Material(spec.young, spec.poisson, spec.yield_stress,
                          getattr(spec, "hardening", 0.0))
    s1 = torch.diag(torch.tensor([1.0, -0.5, -0.5], dtype=torch.float64))
    s2 = torch.tensor([[0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, -1.0]], dtype=torch.float64) / math.sqrt(2.0)
    rows = []
    for index in range(directions):
        theta = 2.0 * math.pi * index / directions
        direction = math.cos(theta) * s1 + math.sin(theta) * s2
        probe = yield_probe(material, direction)
        rows.append({
            "theta_deg": math.degrees(theta),
            "equivalent_stress_at_onset": probe["equivalent_stress"],
            "scale": probe["scale"],
            "relative_deviation": probe["yield_surface_relative_error"],
        })
    worst = max(rows, key=lambda row: row["relative_deviation"])
    return {
        "case": "C6_pi_plane_circle",
        "directions": rows,
        "yield_stress": material.yield_stress,
        "max_relative_deviation": worst["relative_deviation"],
        "worst_theta_deg": worst["theta_deg"],
        "circularity": 1.0 - worst["relative_deviation"],
    }


# --------------------------------------------------------------------------- #
# case C - thick-walled cylinder
# --------------------------------------------------------------------------- #
def _tube_solution(spec: CylinderSpec, pressure: float, n_r: int, *, steps: int,
                   grading: float = 0.0):
    tube = spec.hill()
    mesh = annulus_mesh(n_r, inner_radius=spec.inner_radius,
                        outer_radius=spec.outer_radius, grading=grading)
    law = HardeningLaw(spec.yield_stress, linear=spec.hardening)
    return tube, mesh, solve_plastic_tube(tube, mesh, law, pressure, steps=steps)


def _displacement_profile(solution, tube: HillPlaneStrainTube, pressure: float,
                          *, samples: int = 11) -> dict:
    """Pointwise comparison of ``u_r(r)`` on the Gauss-point/nodal radius grid."""
    radii = solution.mesh.nodes[:, 0]
    computed = solution.displacement[:, 0]
    reference = tube.radial_displacement(radii, pressure, samples=20000)
    scale = float(reference.abs().max())
    error = (computed - reference).abs() / scale
    order = torch.argsort(radii)
    picks = torch.linspace(0, radii.numel() - 1, samples).round().long()
    picks = torch.unique(picks)
    return {
        "max_relative_error": float(error.max()),
        "mean_relative_error": float(error.mean()),
        "bore_relative_error": float(error[order[0]]),
        "profile": [
            {"r": float(radii[order[i]]), "u_computed": float(computed[order[i]]),
             "u_reference": float(reference[order[i]]),
             "relative_error": float(error[order[i]])}
            for i in picks.tolist()
        ],
    }


def cylinder_elastic_band(spec: CylinderSpec, pressures=ELASTIC_PRESSURE_BAND,
                          *, n_r: int = 48, grading: float = 0.1) -> dict:
    """The mission's literal ``p = 0 .. 100 MPa`` band: it is *elastic*.

    Reports, per pressure, the maximum equivalent plastic strain (must be 0) and
    the Lamé stress error of the same FE model, so the statement "no plastic zone
    below ``p_y = 108.25 MPa``" is a measured result and not an assumption.
    """
    rows = []
    tube = spec.hill()
    law = HardeningLaw(spec.yield_stress)
    # Radial grading towards the bore: the Lamé sigma_r gradient is 1/r^2, so a
    # uniform mesh resolves the bore value only to a few percent of p.
    mesh = annulus_mesh(n_r, inner_radius=spec.inner_radius,
                        outer_radius=spec.outer_radius, grading=grading)
    for pressure in pressures:
        solution = solve_plastic_tube(tube, mesh, law, float(pressure), steps=4)
        sig_r, sig_t, _, _ = tube.stresses(solution.gauss_radius, float(pressure))
        scale = abs(float(pressure))
        rows.append({
            "pressure": float(pressure),
            "pressure_over_first_yield": float(pressure) / spec.pressure_first_yield,
            "max_equivalent_plastic_strain": float(solution.gauss_alpha.max()),
            "sigma_r_max_relative_error": float(
                (solution.gauss_stress[:, 0] - sig_r).abs().max()) / scale,
            "sigma_theta_max_relative_error": float(
                (solution.gauss_stress[:, 2] - sig_t).abs().max()) / scale,
            "u_bore": float(solution.displacement[0, 0]),
            "u_bore_reference": float(tube.radial_displacement(
                torch.tensor(spec.inner_radius, dtype=torch.float64), float(pressure))),
        })
    for row in rows:
        row["u_bore_relative_error"] = abs(row["u_bore"] - row["u_bore_reference"]) / abs(
            row["u_bore_reference"])
    return {
        "case": "C_elastic_band",
        "first_yield_pressure": spec.pressure_first_yield,
        "collapse_pressure": spec.pressure_limit,
        "levels": rows,
        "max_plastic_strain": max(row["max_equivalent_plastic_strain"] for row in rows),
        "max_stress_relative_error": max(
            max(row["sigma_r_max_relative_error"], row["sigma_theta_max_relative_error"])
            for row in rows),
        "max_u_bore_relative_error": max(row["u_bore_relative_error"] for row in rows),
    }


def thick_wall_cylinder(spec: CylinderSpec, pressures=PLASTIC_PRESSURE_RATIOS,
                        *, mesh_sizes=(16, 32, 64), steps: int = 8,
                        grading: float = 0.0) -> dict:
    """Plastic-zone growth and radial displacement against Hill 1950.

    ``pressures`` are absolute pressures in Pa; the helper
    :func:`plastic_pressure_levels` builds them from ``p/k`` ratios.
    """
    levels = [float(p) for p in pressures]
    if not levels:
        raise ValueError("at least one pressure level is required")
    if len(mesh_sizes) < 2:
        raise ValueError("a mesh-convergence study needs at least two meshes")
    rows = []
    # Hill's closed form is a *perfect-plasticity* solution; with H > 0 it is not
    # the reference, so the oracle comparison columns are reported as ``None``
    # instead of quietly scoring the FE answer against the wrong physics.
    has_oracle = spec.hardening == 0.0
    for pressure in levels:
        if pressure > spec.pressure_limit:
            raise ValueError("pressure above the plastic collapse pressure")
        reference_radius = spec.hill().plastic_radius(pressure) if has_oracle else None
        for n_r in mesh_sizes:
            tube, mesh, solution = _tube_solution(spec, pressure, int(n_r), steps=steps,
                                                 grading=grading)
            estimate = solution.plastic_radius(spec.yield_stress)
            u_bore = float(solution.displacement[0, 0])
            row = {
                "pressure": pressure,
                "pressure_over_k": pressure / spec.shear_yield_stress,
                "n_r": int(n_r),
                "r_p_fem": estimate["lame_fit"],
                "r_p_bracket": estimate["bracket"],
                "r_p_reference": reference_radius,
                "r_p_relative_error": None,
                "yielded_fraction": estimate["plastic_fraction"],
                "u_bore": u_bore,
                "u_bore_reference": None,
                "u_bore_relative_error": None,
                "u_profile_max_relative_error": None,
                "u_profile": None,
                "sigma_r_max_relative_error": None,
                "sigma_theta_max_relative_error": None,
                "newton_iterations": solution.iterations,
                "residual": solution.residual,
            }
            if has_oracle:
                profile = _displacement_profile(solution, tube, pressure)
                u_bore_reference = float(tube.radial_displacement(
                    torch.tensor(spec.inner_radius, dtype=torch.float64), pressure))
                sig_r, sig_t, _, _ = tube.stresses(solution.gauss_radius, pressure)
                row.update({
                    "r_p_relative_error": abs(estimate["lame_fit"] - reference_radius)
                    / reference_radius,
                    "u_bore_reference": u_bore_reference,
                    "u_bore_relative_error": abs(u_bore - u_bore_reference)
                    / abs(u_bore_reference),
                    "u_profile_max_relative_error": profile["max_relative_error"],
                    "u_profile": profile["profile"],
                    "sigma_r_max_relative_error": float(
                        (solution.gauss_stress[:, 0] - sig_r).abs().max()) / pressure,
                    "sigma_theta_max_relative_error": float(
                        (solution.gauss_stress[:, 2] - sig_t).abs().max()) / pressure,
                })
            rows.append(row)
    convergence = {}
    if len(mesh_sizes) >= 3:
        for pressure in levels:
            block = [row for row in rows if row["pressure"] == pressure]
            convergence[f"{pressure:.6e}"] = mesh_convergence_gci(block, spec)
    return {
        "case": "C_thick_wall_cylinder",
        "hardening": spec.hardening,
        "closed_form_available": spec.hardening == 0.0,
        "first_yield_pressure": spec.pressure_first_yield,
        "collapse_pressure": spec.pressure_limit,
        "mesh_sizes": [int(n) for n in mesh_sizes],
        "levels": rows,
        "max_r_p_relative_error": _max_or_none(row["r_p_relative_error"] for row in rows),
        "max_u_bore_relative_error": _max_or_none(
            row["u_bore_relative_error"] for row in rows),
        "max_u_profile_relative_error": _max_or_none(
            row["u_profile_max_relative_error"] for row in rows),
        "convergence": convergence,
    }


def _max_or_none(values):
    values = [value for value in values if value is not None]
    return max(values) if values else None


def plastic_pressure_levels(spec: CylinderSpec, ratios=PLASTIC_PRESSURE_RATIOS):
    """Absolute pressures ``p = ratio * k`` (``k = sigma_y/sqrt(3)``)."""
    return [float(ratio) * spec.shear_yield_stress for ratio in ratios]


def mesh_convergence_gci(rows, spec: CylinderSpec, *,
                         analytic_comparison: bool = True) -> dict:
    """Roache GCI on ``u_r(a)`` over the finest three meshes of one pressure.

    ``mesh_size`` is the uniform radial element size ``(b - a)/n_r``.  The GCI
    quantifies the *discretisation* error only, so the analytic Hill value is
    **not** fed into the checks; it is reported separately as
    ``richardson_relative_offset_vs_analytic`` because the residual offset is a
    property of the classical plane-strain closed form (see the module
    docstring), not of the mesh.
    """
    ordered = sorted(rows, key=lambda row: row["n_r"])
    if len(ordered) < 3:
        raise ValueError("GCI needs three mesh levels")
    selected = ordered[-3:]
    span = spec.outer_radius - spec.inner_radius
    levels = [ConvergenceLevel(span / row["n_r"], row["u_bore"], f"n_r={row['n_r']}")
              for row in selected]
    report = evaluate_convergence_study(levels, qoi="u_r(r=a)")
    if analytic_comparison and spec.hardening == 0.0:
        tube = spec.hill()
        analytic = float(tube.radial_displacement(
            torch.tensor(spec.inner_radius, dtype=torch.float64),
            selected[-1]["pressure"]))
        richardson = float(report.get("richardson_value", float("nan")))
        report["analytic_bore_displacement"] = analytic
        report["richardson_relative_offset_vs_analytic"] = richardson / analytic - 1.0
        report["finest_relative_offset_vs_analytic"] = selected[-1]["u_bore"] / analytic - 1.0
    return report


# --------------------------------------------------------------------------- #
# full qualification
# --------------------------------------------------------------------------- #
def run_qualification(
    *, uniaxial: UniaxialSpec | None = None,
    torsion: TorsionSpec | None = None,
    cylinder: CylinderSpec | None = None,
    uniaxial_strains=None,
    shear_levels=None,
    cylinder_meshes: tuple[int, ...] = (16, 32, 64),
    bar_mesh: tuple[int, int, int] = (8, 1, 1),
    steps: int = 8,
) -> dict:
    """Run cases A/B/C and return criteria C1..C7 with every number attached.

    ``C8`` (regression: the pre-existing elastic FEM tests) is *not* produced
    here -- it is a test-suite property, reported by the benchmark runner.
    """
    uniaxial = uniaxial or UniaxialSpec()
    torsion = torsion or TorsionSpec()
    cylinder = cylinder or CylinderSpec()
    strains = uniaxial_strains or [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    shears = shear_levels or [0.0004, 0.0009, 0.0015, 0.0021, 0.003, 0.006, 0.012,
                              0.025, 0.05]

    point = uniaxial_tension_material_point(uniaxial, strains)
    bar = uniaxial_tension_fem(uniaxial, strains, mesh=bar_mesh)
    shear = pure_shear_material_point(torsion, shears)
    patch = pure_shear_patch_test(torsion, 0.003)
    pi_plane = pi_plane_yield_surface(uniaxial, directions=6)
    elastic_band = cylinder_elastic_band(cylinder)
    tube = thick_wall_cylinder(cylinder, plastic_pressure_levels(cylinder),
                               mesh_sizes=cylinder_meshes, steps=steps)

    gci_reports = tube["convergence"]
    worst_gci = max(
        (report for report in gci_reports.values() if report.get("gci")),
        key=lambda report: report["gci"]["relative_gci"], default=None)

    gci_ok = all(report.get("status") != "blocked" for report in gci_reports.values())
    criteria = {
        "C1_uniaxial_curve": {
            "value": max(bar["max_relative_error_element"], point["max_relative_error_oracle"]),
            "tolerance": 0.05,
            "basis": "TET4 bar + material point vs analytic bilinear card",
            "experimental_dataset": "withheld (no ASTM E8 data in repo, no network)",
        },
        "C2_yield_stress": {
            "value": abs(bar["yield_stress_estimate"] - bar["yield_stress"])
            / bar["yield_stress"],
            "tolerance": 0.02,
            "basis": "yield stress read back from the computed bar curve (plastic-branch intercept)",
        },
        "C3_shear_yield": {
            "value": shear["tau_y_relative_error"],
            "tolerance": 0.02,
            "basis": "tau_y vs sigma_y/sqrt(3), bisection on the implemented return map",
        },
        "C4_plastic_radius": {
            "value": tube["max_r_p_relative_error"],
            "tolerance": 0.05,
            "basis": "r_p(p) at 5 contained-plastic pressure levels vs Hill 1950",
        },
        "C5_radial_displacement": {
            "value": tube["max_u_profile_relative_error"],
            "tolerance": 0.10,
            "basis": "u_r(r) profile vs quadrature oracle (normalised by u_r(a))",
        },
        "C6_pi_plane_circle": {
            "value": pi_plane["max_relative_deviation"],
            "tolerance": 0.01,
            "basis": "6 deviatoric directions, onset q vs sigma_y",
        },
        "C7_gci": {
            "value": None if worst_gci is None else worst_gci["gci"]["relative_gci"],
            "tolerance": 0.05,
            "basis": "Roache GCI on u_r(a) over the three finest meshes; "
                     "value = worst (largest) relative GCI of the five pressures",
            "reports": gci_reports,
            "status": "passed" if (worst_gci is not None and gci_ok) else "blocked",
        },
    }
    for entry in criteria.values():
        tolerance = entry["tolerance"]
        value = entry["value"]
        entry["passed"] = None if (value is None or tolerance is None) else bool(
            value <= tolerance)
    criteria["C1_uniaxial_curve"]["passed"] = bool(
        criteria["C1_uniaxial_curve"]["value"] <= 0.05
        and point["max_relative_error_oracle"] <= 0.05)

    hardening_tube = None
    if uniaxial.hardening > 0.0:
        hardened = CylinderSpec(cylinder.inner_radius, cylinder.outer_radius,
                                cylinder.length, cylinder.young, cylinder.poisson,
                                cylinder.yield_stress, uniaxial.hardening)
        hardening_tube = thick_wall_cylinder(
            hardened, plastic_pressure_levels(hardened),
            mesh_sizes=cylinder_meshes, steps=steps)
    return {
        "schema": "tensorfem.j2_elastoplastic_three_case/1",
        "material_card": {
            "young": uniaxial.young, "poisson": uniaxial.poisson,
            "yield_stress": uniaxial.yield_stress, "hardening": uniaxial.hardening,
            "yield_strain": uniaxial.yield_strain,
            "tangent_modulus": uniaxial.tangent_modulus,
            "shear_modulus": torsion.shear_modulus,
        },
        "case_A_uniaxial": {"material_point": point, "tet4_bar": bar},
        "case_B_shear": {"material_point": shear, "patch_test": patch},
        "case_C_cylinder": {"elastic_band": elastic_band, "plastic": tube,
                            "hardening_reference": hardening_tube},
        "pi_plane": pi_plane,
        "criteria": criteria,
    }