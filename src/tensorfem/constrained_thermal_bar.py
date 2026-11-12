"""Constrained (fully restrained) thermal stress: the ``sigma = E alpha dT`` anchor.

Case ``CASE-THERMAL-BAR`` of the benchmark pool.  Physical background: a bar /
plate that is prevented from expanding is the simplest thermal-stress problem
there is, and it is the *reference* every industrial thermal-stress model is
calibrated against, because the answer is a closed form -- no series, no root
finding, no table:

    axial restraint only (free lateral contraction), linear elastic::

        sigma_xx = -E alpha dT          (plane stress / uniaxial)
        sigma_yy = sigma_zz = 0

    in-plane biaxial restraint (plane stress, lateral z free)::

        sigma_xx = sigma_yy = -E alpha dT / (1 - nu)

    triaxial restraint (all three strains held at zero)::

        sigma_xx = sigma_yy = sigma_zz = -E alpha dT / (1 - 2 nu)

    unrestrained (control): ``sigma = 0``, free expansion ``eps = alpha dT``.

Sign convention: ``dT = T - T_ref > 0`` (heating) produces *compression*
(negative normal stress); cooling produces tension.  All stresses below are
reported with that sign and compared as signed values, so a sign error cannot
hide behind an absolute value.

Provenance of the reference number (zero hand-copying)
------------------------------------------------------
The target value is **not transcribed**.  It is evaluated in this module from
the closed form above and the material properties of
:func:`tensorfem.polar_thermal_stress.steel_from_anchors` (``EH36`` hull steel,
linear regression of the transcribed low-temperature anchor table, evaluated at
``T_ref = 20 degC``: ``E = 206 GPa``, ``alpha = 12.00e-6 /K``,
``nu = 0.30``).  Both the anchor table and the ``source`` string that describes
it travel with the regression object, and the benchmark driver hashes the
table, so the report cites data, not prose.  The closed form itself is the
classical linear-thermoelasticity result (Hooke's law with an added thermal
strain ``eps_th = alpha dT I``), stated in every thermal-stress text
(Timoshenko & Goodier, *Theory of Elasticity*, thermal-stress article; Boley &
Weiner, *Theory of Thermal Stresses*, Ch. 1).  This sandbox has no library
access, so no page number is claimed here that could not be verified offline --
the formula is derived from the constitutive law in ``derivation_residual``
below, which a reader can re-run instead of trusting a citation.

What is exercised (existing TensorFEM capability, not a new solver)
-------------------------------------------------------------------
* 1-D: :func:`tensorfem.thermal.solve_thermoelastic_bar` (sequential
  thermoelasticity on the bar element).
* 3-D: the hex8 thermo-elastic path of :mod:`tensorfem.polar_thermal_stress`
  (``assemble_elasticity`` + ``assemble_thermal_load`` + ``solve_dirichlet``,
  dense reference assembly) driven with a *uniaxial* restraint set, which is a
  configuration that path had not been verified in: the repo's existing thermal
  gates cover the biaxial (``/(1 - nu)``) clamped-edge case and the fully free
  plate, i.e. the two extremes, and nothing in between.

Predeclared acceptance gates (fixed before the numbers were seen)
-----------------------------------------------------------------
* ``TASK_REL_TOL = 3e-2``  -- the pool's acceptance criterion, applied to the
  signed relative error of the governing stress component.
* ``ROUNDOFF_REL_TOL = 1e-8`` -- internal gate for configurations whose exact
  solution is representable by the discretisation (uniform ``dT``, restrained):
  anything above float64 round-off would mean an assembly/BC bug.
* ``ORDER_TOL`` -- observed h-convergence order of the quadratic-
  ``dT`` variant must be within ``0.35`` of the theoretical ``2.0``.
These thresholds are not adjusted after seeing results; a miss is reported as
an honest FAIL in the evidence JSON.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tensorfem.polar_thermal_stress import (
    DTYPE,
    HexMesh,
    PolarSteel,
    _hex8_B,
    _hex8_gauss_data,
    assemble_elasticity,
    assemble_thermal_load,
    isotropic_D,
    solve_dirichlet,
    steel_from_anchors,
    structured_hex_mesh,
)
from tensorfem.thermal import solve_thermoelastic_bar

TASK_REL_TOL = 3.0e-2
ROUNDOFF_REL_TOL = 1.0e-8
ORDER_TOL = 0.35


# ---------------------------------------------------------------------------
# closed forms (the reference, evaluated -- never transcribed)
# ---------------------------------------------------------------------------
def restrained_uniaxial_stress(E: float, alpha: float, dT: float) -> float:
    """``sigma_xx = -E alpha dT`` for axial restraint with free lateral faces."""
    return -float(E) * float(alpha) * float(dT)


def restrained_biaxial_stress(E: float, alpha: float, dT: float, nu: float) -> float:
    """``sigma_xx = sigma_yy = -E alpha dT / (1 - nu)`` (plane stress, z free)."""
    return -float(E) * float(alpha) * float(dT) / (1.0 - float(nu))


def restrained_triaxial_stress(E: float, alpha: float, dT: float, nu: float) -> float:
    """``sigma_xx = sigma_yy = sigma_zz = -E alpha dT / (1 - 2 nu)``."""
    return -float(E) * float(alpha) * float(dT) / (1.0 - 2.0 * float(nu))


def free_expansion_strain(alpha: float, dT: float) -> float:
    """``eps = alpha dT``, the unrestrained thermal expansion (the control case)."""
    return float(alpha) * float(dT)


ANALYTIC_STRESS = {
    "uniaxial": lambda E, alpha, dT, nu: restrained_uniaxial_stress(E, alpha, dT),
    "biaxial": lambda E, alpha, dT, nu: restrained_biaxial_stress(E, alpha, dT, nu),
    "triaxial": lambda E, alpha, dT, nu: restrained_triaxial_stress(E, alpha, dT, nu),
    "free": lambda E, alpha, dT, nu: 0.0,
}


def derivation_residual(E: float, alpha: float, dT: float, nu: float,
                        restraint: str = "uniaxial") -> float:
    """Hooke's law residual of the claimed closed form (should be 0).

    For the restrained directions the total strain is zero, so
    ``eps = (1/E) [sigma - nu (sigma_yy + sigma_zz)] + alpha dT`` must vanish.
    This is the *derivation check*: it verifies the formula the benchmark uses as
    its reference instead of asking the reader to trust a citation.
    """
    s = ANALYTIC_STRESS[restraint](E, alpha, dT, nu)
    if restraint == "uniaxial":
        eps = (s - nu * 0.0) / E + alpha * dT
    elif restraint == "biaxial":
        eps = (s - nu * s) / E + alpha * dT
    elif restraint == "triaxial":
        eps = (s - nu * (2.0 * s)) / E + alpha * dT
    else:
        raise ValueError("free expansion has no restraint residual")
    return float(eps)


def default_steel(name: str = "EH36", T_ref: float = 20.0) -> PolarSteel:
    """Transcribed-anchor hull steel at ``T_ref`` (temperature dependent)."""
    return steel_from_anchors(name, T_ref=T_ref)


def constant_steel(name: str = "EH36", T_ref: float = 20.0) -> PolarSteel:
    """Same anchor values with the temperature slopes switched off.

    The benchmark reference ``sigma = -E alpha dT`` is a *constant-property*
    statement.  Freezing ``dE/dT = dalpha/dT = 0`` at the ``T_ref`` anchor makes
    the closed form and the FEM use exactly the same material numbers, so the
    comparison measures the discretisation and the boundary conditions and not
    the property-evaluation convention.  The temperature-dependent variant is
    kept as a separate case (see :func:`solve_hex`), where the reference is
    evaluated at the element temperature, matching the code's convention.
    """
    from dataclasses import replace

    return replace(steel_from_anchors(name, T_ref=T_ref),
                   dE_dT=0.0, dalpha_dT=0.0, dyield_dT=0.0)


# ---------------------------------------------------------------------------
# 1-D path: reuse of tensorfem.thermal.solve_thermoelastic_bar
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BarResult:
    n_elem: int
    length: float
    dT: float
    sigma: torch.Tensor          # (ne,) axial stress
    displacement: torch.Tensor   # (n+1,)
    reaction: torch.Tensor       # (n+1,)
    exact_sigma: float
    exact_displacement: torch.Tensor


def _bar_mesh(n_elem: int, length: float):
    nodes = torch.linspace(0.0, float(length), n_elem + 1, dtype=DTYPE)[:, None]
    idx = torch.arange(n_elem, dtype=torch.long)
    elements = torch.stack([idx, idx + 1], dim=1)
    return nodes, elements


def solve_bar(n_elem: int, E: float, alpha: float, dT: float, T_ref: float = 20.0,
              length: float = 1.0, temperature: str = "uniform",
              area: float = 1.0, fixed: str = "both") -> BarResult:
    """Fully restrained bar under a ``dT`` field, solved on the 1-D FEM path.

    ``temperature="uniform"``: ``T(x) = T_ref + dT`` (nodally exact, so the FEM
    stress must equal ``-E alpha dT`` to round-off).
    ``temperature="quadratic"``: ``T(x) = T_ref + dT (x/L)**2``; the linear
    elements interpolate ``T`` only to ``O(h^2)``, so this variant has a genuine
    discretisation error and is used for the h-convergence study.  Its exact
    solution is ``sigma = -E alpha dT / 3`` (constant axial force) and
    ``u(x) = alpha dT (x**3/(3 L**2) - x/3)``.
    """
    L = float(length)
    nodes, elements = _bar_mesh(n_elem, L)
    x = nodes[:, 0]
    if temperature == "uniform":
        T = torch.full_like(x, float(T_ref) + float(dT))
        exact_sigma = -E * alpha * dT
        exact_u = torch.zeros_like(x)
    elif temperature == "quadratic":
        T = float(T_ref) + float(dT) * (x / L) ** 2
        exact_sigma = -E * alpha * dT / 3.0
        exact_u = alpha * dT * (x ** 3 / (3.0 * L ** 2) - x / 3.0)
    else:
        raise ValueError("temperature must be 'uniform' or 'quadratic'")
    fixed_nodes = torch.tensor([0, n_elem], dtype=torch.long) if fixed == "both" \
        else torch.tensor([0], dtype=torch.long)
    res = solve_thermoelastic_bar(nodes, elements, E, area, alpha, T, float(T_ref),
                                  fixed_nodes)
    sigma = res.axial_force / area
    return BarResult(n_elem=n_elem, length=L, dT=float(dT), sigma=sigma,
                     displacement=res.displacement, reaction=res.reaction,
                     exact_sigma=float(exact_sigma), exact_displacement=exact_u)


# ---------------------------------------------------------------------------
# 3-D path: hex8 thermo-elasticity with a selectable restraint set
# ---------------------------------------------------------------------------
def _node_index(mesh: HexMesh, xyz) -> int:
    d = ((mesh.nodes - torch.tensor(xyz, dtype=DTYPE)) ** 2).sum(dim=1)
    return int(torch.argmin(d))


def restraint_dofs(mesh: HexMesh, restraint: str) -> torch.Tensor:
    """Dof set for the restraint family (removes exactly the rigid-body modes).

    ``uniaxial``  : ``u_x = 0`` on both ``x`` faces (lateral faces free);
    ``biaxial``   : additionally ``u_y = 0`` on both ``y`` faces;
    ``triaxial``  : all three components held on all six faces;
    ``free``      : ``u_x = 0`` on the ``x = 0`` face only (control, stress-free).

    The residual rigid-body modes of the free directions are removed with the
    minimum number of point constraints, chosen so that no rotational mode is
    left: for the ``y``/``z`` pair two nodes differing in ``z`` carry ``u_y`` and
    one of them carries ``u_z``.
    """
    X = mesh.nodes
    tol = 1.0e-9
    Lx = float(X[:, 0].max())
    Ly = float(X[:, 1].max())
    Lz = float(X[:, 2].max())
    dofs: list[torch.Tensor] = []

    def add(mask, comp):
        n = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if n.numel():
            dofs.append(3 * n + comp)

    add(X[:, 0] <= tol, 0)
    if restraint in ("uniaxial", "biaxial", "triaxial"):
        add(X[:, 0] >= Lx - tol, 0)
    if restraint in ("biaxial", "triaxial"):
        add(X[:, 1] <= tol, 1)
        add(X[:, 1] >= Ly - tol, 1)
    if restraint == "triaxial":
        add(X[:, 2] <= tol, 2)
        add(X[:, 2] >= Lz - tol, 2)
    if restraint in ("uniaxial", "biaxial", "free"):
        # anti-rigid-body constraints for the free lateral directions: the
        # translation pair (u_y, u_z) and the rotation about x (u_y at two nodes
        # that differ in z).  Rotations about y/z move u_x on the x faces, which
        # the face conditions already suppress.
        n1 = _node_index(mesh, (0.0, 0.0, 0.0))
        n2 = _node_index(mesh, (0.0, 0.0, Lz))
        dofs.append(torch.tensor([3 * n1 + 1], dtype=torch.long))
        dofs.append(torch.tensor([3 * n2 + 1], dtype=torch.long))
        dofs.append(torch.tensor([3 * n1 + 2], dtype=torch.long))
    if restraint == "biaxial":
        # u_y is already held on both y faces (translation + rotation about x);
        # only the z translation is left.
        n3 = _node_index(mesh, (0.0, Ly, 0.0))
        dofs.append(torch.tensor([3 * n3 + 2], dtype=torch.long))
    return torch.unique(torch.cat(dofs))


@dataclass(frozen=True)
class HexResult:
    restraint: str
    n_elem: int
    dT: float
    sigma: torch.Tensor         # (ne, 6) Gauss-averaged element stress
    displacement: torch.Tensor  # (n_nodes, 3)
    exact_sigma: float
    exact_lateral: tuple[float, ...]


def solve_hex(mesh: HexMesh, steel: PolarSteel, dT: float, T_ref: float = 20.0,
              restraint: str = "uniaxial", uniform: bool = True) -> HexResult:
    """Uniform-``dT`` hex8 solve with the requested restraint set."""
    if restraint not in ANALYTIC_STRESS:
        raise ValueError(f"unknown restraint {restraint!r}")
    ne = mesh.n_elements
    T_elem = torch.full((ne,), float(T_ref) + float(dT), dtype=DTYPE)
    E_e = steel.E(T_elem)
    alpha_e = torch.clamp(steel.alpha(T_elem), min=0.0)
    K = assemble_elasticity(mesh, E_e, steel.nu)
    f = assemble_thermal_load(mesh, E_e, alpha_e, T_elem, float(T_ref), steel.nu)
    fixed = restraint_dofs(mesh, restraint)
    if fixed.numel():
        u = solve_dirichlet(K, f, fixed, torch.zeros(fixed.numel(), dtype=DTYPE))
    else:  # pragma: no cover - the restraint families always carry constraints
        u = torch.linalg.lstsq(K, f[:, None]).solution.reshape(-1)

    gd = _hex8_gauss_data(mesh)
    B = _hex8_B(gd)
    D = isotropic_D(E_e, steel.nu)
    if D.ndim == 2:
        D = D[None].expand(ne, 6, 6)
    ue = u.reshape(mesh.n_nodes, 3)[mesh.elements].reshape(ne, 24)
    eps = torch.einsum("egki,ei->egk", B, ue)
    a = alpha_e * (T_elem - float(T_ref))
    eps_th = torch.zeros_like(eps)
    eps_th[:, :, 0] = a[:, None]
    eps_th[:, :, 1] = a[:, None]
    eps_th[:, :, 2] = a[:, None]
    sigma = torch.einsum("ekl,egl->egk", D, eps - eps_th).mean(dim=1)

    # the solver evaluates E and alpha at the element temperature, so the
    # reference has to as well; with constant_steel() this is E(T_ref), alpha(T_ref)
    E0 = float(steel.E(float(T_ref) + float(dT)))
    a0 = float(steel.alpha(float(T_ref) + float(dT)))
    s = ANALYTIC_STRESS[restraint](E0, a0, dT, steel.nu)
    lateral = {"uniaxial": (0.0, 0.0), "biaxial": (s, 0.0),
               "triaxial": (s, s), "free": (0.0, 0.0)}[restraint]
    return HexResult(restraint=restraint, n_elem=ne, dT=float(dT), sigma=sigma,
                     displacement=u.reshape(mesh.n_nodes, 3), exact_sigma=s,
                     exact_lateral=tuple(float(v) for v in lateral))


def hex_mesh(n: int, length=(0.20, 0.20, 0.20)) -> HexMesh:
    """Cube ``n^3`` element mesh (the benchmark specimen, SI metres)."""
    return structured_hex_mesh(n, n, n, *length)


# ---------------------------------------------------------------------------
# small helpers used by the study driver
# ---------------------------------------------------------------------------
def rel_error(computed: float, reference: float) -> float:
    """Signed relative error ``(computed - reference) / |reference|``."""
    if reference == 0.0:
        raise ValueError("relative error against a zero reference is undefined; "
                         "use an absolute/normalised measure")
    return float((float(computed) - float(reference)) / abs(float(reference)))


def observed_order(errors, refinement) -> float:
    """Least-squares slope of ``log(error)`` vs ``log(h)``."""
    if len(errors) != len(refinement):
        raise ValueError("errors and refinement must have the same length")
    x = torch.log(torch.tensor([float(h) for h in refinement], dtype=DTYPE))
    y = torch.log(torch.tensor([float(e) for e in errors], dtype=DTYPE))
    n = x.numel()
    b = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x * x).sum() - x.sum() ** 2)
    return float(b)


def material_digest(steel: PolarSteel) -> dict:
    """Everything needed to reproduce the reference value, plus its provenance."""
    import hashlib

    from tensorfem.polar_thermal_stress import MATERIAL_ANCHORS

    table = MATERIAL_ANCHORS[steel.name]
    blob = repr(sorted((k, v) for k, v in table.items() if k != "source")).encode()
    return {
        "name": steel.name,
        "E_ref_Pa": float(steel.E_ref),
        "dE_dT_Pa_per_K": float(steel.dE_dT),
        "alpha_ref_per_K": float(steel.alpha_ref),
        "dalpha_dT_per_K2": float(steel.dalpha_dT),
        "nu": float(steel.nu),
        "T_ref_C": float(steel.T_ref),
        "anchor_source": table.get("source", ""),
        "anchor_table_sha256": hashlib.sha256(blob).hexdigest(),
        "anchor_residual": {
            k: (float(v) if not isinstance(v, dict) else
                {kk: float(vv) for kk, vv in v.items()})
            for k, v in (steel.anchor_residual or {}).items()
        },
    }


__all__ = [
    "TASK_REL_TOL", "ROUNDOFF_REL_TOL", "ORDER_TOL",
    "restrained_uniaxial_stress", "restrained_biaxial_stress",
    "restrained_triaxial_stress", "free_expansion_strain", "ANALYTIC_STRESS",
    "derivation_residual", "default_steel", "constant_steel", "BarResult", "HexResult",
    "solve_bar", "solve_hex", "hex_mesh", "restraint_dofs",
    "rel_error", "observed_order", "material_digest",
]