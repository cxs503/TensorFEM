"""Cross-validation (XVAL) applicability screening for external solver practices.

Anchor: ``XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT`` -- the practice
documented by ``AminZar69/CudaPhaseField`` ("phase-field is *not* a
pseudopotential method") is screened here instead of being cross-validated
numerically.

Why a screen exists at all: the XVAL family is a *verification-debt* tool.  For a
candidate practice borrowed from an open-source solver the default move is "run
both codes on one case and compare".  That move is only legitimate when the two
candidates (a) discretise the same governing PDE class, (b) carry the same
state-variable semantics, (c) share at least one closed-form/literature
reference, and (d) are both implemented in this platform.  When any of these is
violated the honest verdict is **NOT_APPLICABLE** -- not a fabricated "0 % error"
table.  This module makes that verdict a machine object with per-predicate
evidence, so it can be audited and cannot be silently upgraded.

The pairing screened here:

* ``PHASE_FIELD_AT2`` -- the damage/phase-field model implemented in
  :mod:`tensorfem.phase_field_fracture` (variational diffuse-interface, free
  regularisation length ``l``, non-conserved order parameter ``d``).
* ``PSEUDOPOTENTIAL_LBM`` -- Shan-Chen style single-component multiphase lattice
  Boltzmann (kinetic BGK + density-dependent pseudo-force, conserved density,
  interface width *emergent* from the interaction strength / equation of state).

The module also carries the analytic anchor that makes the NOT_APPLICABLE verdict
*useful*: the exact 1-D AT2 equilibrium profile

    d(x) = cosh((L - x)/l) / cosh(L/l)          (d(0) = 1, d'(L) = 0)

together with a driver that reproduces it with the production TensorFEM
phase-field solver.  So the phase-field side is verified against a closed-form
reference (the substitute path the screen demands), and the missing piece is only
the incommensurable LBM side, which is documented rather than faked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .phase_field_fracture import (
    PhaseFieldProblem,
    PhaseFieldSolver,
    structured_quad_mesh,
)

# --------------------------------------------------------------------------------------
# descriptor vocabulary
# --------------------------------------------------------------------------------------
#: Governing-PDE families that can appear in an XVAL pairing.
PDE_CLASSES = (
    "variational_diffuse_interface",   # energy functional, 2nd-order Euler-Lagrange
    "kinetic_bgk_lattice",             # LBM: discrete-velocity BGK transport
    "conservation_law_fv",             # hyperbolic/parabolic FV
    "elliptic_potential",              # scalar/vector potential problem
)

#: What the primary unknown physically is.
STATE_VARIABLES = (
    "non_conserved_order_parameter",   # damage / phase index, source-like evolution
    "conserved_order_parameter",       # Cahn-Hilliard concentration
    "density",                         # LBM / compressible
    "displacement",
)

#: How the diffuse-interface width is fixed.
WIDTH_CONTROLS = (
    "free_parameter",                  # explicit length scale in the functional (l)
    "emergent",                        # implicit in a coupling constant / EOS
    "none",                            # sharp interface assumption
)


@dataclass(frozen=True)
class MethodDescriptor:
    """Machine-readable identity of one candidate method."""

    key: str
    name: str
    pde_class: str
    state_variable: str
    interface_width_control: str
    analytic_benchmarks: tuple[str, ...] = ()
    platform_module: str | None = None
    upstream: str = ""

    def __post_init__(self) -> None:
        if self.pde_class not in PDE_CLASSES:
            raise ValueError(f"unknown pde_class {self.pde_class!r}")
        if self.state_variable not in STATE_VARIABLES:
            raise ValueError(f"unknown state_variable {self.state_variable!r}")
        if self.interface_width_control not in WIDTH_CONTROLS:
            raise ValueError(f"unknown interface_width_control {self.interface_width_control!r}")


# --------------------------------------------------------------------------------------
# the two candidates
# --------------------------------------------------------------------------------------
PHASE_FIELD_AT2 = MethodDescriptor(
    key="phase_field_at2",
    name="AT2 variational phase-field damage (Miehe)",
    pde_class="variational_diffuse_interface",
    state_variable="non_conserved_order_parameter",
    interface_width_control="free_parameter",
    analytic_benchmarks=(
        "1D AT2 equilibrium profile d(x)=cosh((L-x)/l)/cosh(L/l)",
        "1D homogeneous damage d=2Hl/(Gc+2Hl) under uniform history H",
        "AT2 critical stress sigma_c=sqrt(9EGc/(16l)) (uniaxial, plane strain)",
    ),
    platform_module="tensorfem.phase_field_fracture",
)

PSEUDOPOTENTIAL_LBM = MethodDescriptor(
    key="pseudopotential_lbm",
    name="Shan-Chen pseudopotential multiphase LBM",
    pde_class="kinetic_bgk_lattice",
    state_variable="density",
    interface_width_control="emergent",
    analytic_benchmarks=(
        "Maxwell construction on the imposed EOS (no closed-form interface profile)",
    ),
    platform_module=None,
    upstream="AminZar69/CudaPhaseField (note: phase field is not pseudopotential)",
)

KNOWN_METHODS = {m.key: m for m in (PHASE_FIELD_AT2, PSEUDOPOTENTIAL_LBM)}

# --------------------------------------------------------------------------------------
# predicate results
# --------------------------------------------------------------------------------------
#: Predicates that, when failed, forbid a direct numeric cross-validation.
BLOCKING_CODES = ("X1", "X2", "X3", "X4", "X5")


@dataclass(frozen=True)
class PredicateResult:
    code: str
    question: str
    passed: bool
    evidence: str
    blocking: bool = True

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "question": self.question,
            "passed": bool(self.passed),
            "blocking": bool(self.blocking),
            "evidence": self.evidence,
        }


@dataclass
class ScreenVerdict:
    """Outcome of the applicability screen for one pairing."""

    left: str
    right: str
    verdict: str
    predicates: list[PredicateResult] = field(default_factory=list)
    substitute_path: list[str] = field(default_factory=list)

    @property
    def blocking_failures(self) -> list[str]:
        return [p.code for p in self.predicates if p.blocking and not p.passed]

    @property
    def reasons(self) -> list[str]:
        return [f"{p.code}: {p.evidence}" for p in self.predicates if not p.passed]

    def to_dict(self) -> dict:
        return {
            "left": self.left,
            "right": self.right,
            "verdict": self.verdict,
            "blocking_failures": self.blocking_failures,
            "reasons": self.reasons,
            "predicates": [p.to_dict() for p in self.predicates],
            "substitute_path": list(self.substitute_path),
        }


# --------------------------------------------------------------------------------------
# platform inventory (evidence for X5)
# --------------------------------------------------------------------------------------
_PSEUDOPOTENTIAL_TOKENS = (
    "pseudopotential",
    "shan_chen",
    "shanchen",
    "shan-chen",
    "psi_coupling",
)


def platform_implementation_inventory(package_dir: Path | str | None = None) -> dict:
    """Scan the installed package for a pseudopotential-multiphase implementation.

    Returns ``{"package_dir": str, "modules_scanned": int, "pseudopotential_hits":
    [relative paths], "tokens": [...]}``.  An empty hit list is the machine
    evidence for predicate X5 ("the counterpart does not exist here, so no numeric
    cross-validation is possible in this repository").
    """
    root = Path(__file__).resolve().parent if package_dir is None else Path(package_dir)
    hits: list[str] = []
    scanned = 0
    for path in sorted(root.rglob("*.py")):
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:  # pragma: no cover - unreadable file is not a hit
            continue
        # the screen module itself names the tokens on purpose: skip self-matches
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(tok in text for tok in _PSEUDOPOTENTIAL_TOKENS):
            hits.append(str(path.relative_to(root)))
    return {
        "package_dir": str(root),
        "modules_scanned": scanned,
        "pseudopotential_hits": hits,
        "tokens": list(_PSEUDOPOTENTIAL_TOKENS),
    }


# --------------------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------------------
def _pred_x1(a: MethodDescriptor, b: MethodDescriptor) -> PredicateResult:
    ok = a.pde_class == b.pde_class
    return PredicateResult(
        "X1",
        "Do both candidates discretise the same governing PDE class?",
        ok,
        f"pde_class {a.key}={a.pde_class!r} vs {b.key}={b.pde_class!r}"
        + ("" if ok else " -- a variational energy minimisation and a discrete-velocity "
                         "kinetic transport equation have no shared discrete limit, so a "
                         "field-by-field difference has no interpretation"),
    )


def _pred_x2(a: MethodDescriptor, b: MethodDescriptor) -> PredicateResult:
    ok = a.state_variable == b.state_variable
    return PredicateResult(
        "X2",
        "Do both candidates carry the same state-variable semantics?",
        ok,
        f"state_variable {a.key}={a.state_variable!r} vs {b.key}={b.state_variable!r}"
        + ("" if ok else " -- a non-conserved order parameter (source-type evolution, "
                         "irreversibility floor) cannot be mapped onto a conserved mass "
                         "density; there is no common field to subtract"),
    )


def _pred_x3(a: MethodDescriptor, b: MethodDescriptor) -> PredicateResult:
    ok = a.interface_width_control == b.interface_width_control
    return PredicateResult(
        "X3",
        "Is the diffuse-interface width fixed by the same kind of control?",
        ok,
        f"interface_width_control {a.key}={a.interface_width_control!r} vs "
        f"{b.key}={b.interface_width_control!r}"
        + ("" if ok else " -- one width is an explicit functional parameter, the other is "
                         "an output of the coupling constant / EOS; comparing profiles would "
                         "compare a knob with a consequence"),
    )


def _pred_x4(a: MethodDescriptor, b: MethodDescriptor) -> PredicateResult:
    shared = sorted(set(a.analytic_benchmarks) & set(b.analytic_benchmarks))
    # A closed-form reference on *both* sides is the only way an error bar means
    # something; a Maxwell construction is a bulk thermodynamic constraint, not a
    # field-level reference, so it does not count as a shared field benchmark.
    field_refs_a = [s for s in a.analytic_benchmarks if s.startswith("1D")]
    field_refs_b = [s for s in b.analytic_benchmarks if s.startswith("1D")]
    ok = bool(field_refs_a) and bool(field_refs_b)
    return PredicateResult(
        "X4",
        "Is there a shared closed-form field-level reference to measure error against?",
        ok,
        f"shared={shared or 'none'}; "
        f"{a.key} field refs={len(field_refs_a)}, {b.key} field refs={len(field_refs_b)}"
        + ("" if ok else " -- without a common analytic reference any 'error' number would "
                         "measure the difference between two models, not the error of "
                         "either"),
    )


def _pred_x5(a: MethodDescriptor, b: MethodDescriptor, inventory: dict | None) -> PredicateResult:
    inv = platform_implementation_inventory() if inventory is None else inventory
    hits = inv.get("pseudopotential_hits", [])
    missing = [m.key for m in (a, b) if m.platform_module is None]
    # The platform-module field says where the candidate lives; the inventory is the
    # independent cross-check that the borrowed (LBM) side is genuinely absent.
    ok = not missing and not hits
    return PredicateResult(
        "X5",
        "Are both candidates implemented in this platform (so the pair can be run)?",
        ok,
        f"implemented={[m.key for m in (a, b) if m.platform_module]} "
        f"missing={missing}; inventory modules_scanned={inv.get('modules_scanned')} "
        f"pseudopotential_hits={hits}"
        + ("" if ok else " -- one side of the pair has no implementation here; the "
                         "cross-validation cannot be executed, only claimed"),
    )


def screen_applicability(
    a: MethodDescriptor,
    b: MethodDescriptor,
    inventory: dict | None = None,
) -> ScreenVerdict:
    """Decide whether a direct numeric cross-validation of ``a`` against ``b`` is legitimate.

    The verdict is ``APPLICABLE`` only when every blocking predicate passes.  A
    ``NOT_APPLICABLE`` verdict always carries the failed predicate codes and the
    substitute verification path that *should* be walked instead.
    """
    predicates = [_pred_x1(a, b), _pred_x2(a, b), _pred_x3(a, b)]
    if inventory is None:
        inventory = platform_implementation_inventory()
    predicates.append(_pred_x4(a, b))
    predicates.append(_pred_x5(a, b, inventory))

    failures = [p for p in predicates if p.blocking and not p.passed]
    verdict = "APPLICABLE" if not failures else "NOT_APPLICABLE"

    substitute: list[str] = []
    if verdict == "NOT_APPLICABLE":
        for m in (a, b):
            if m.analytic_benchmarks and m.platform_module:
                substitute.append(
                    f"{m.key}: verify against its own closed-form references "
                    f"({'; '.join(m.analytic_benchmarks)})"
                )
        for m in (a, b):
            if m.platform_module is None:
                substitute.append(
                    f"{m.key}: absent from this repository -- no verdict may be claimed "
                    "here; re-open the XVAL only after an implementation lands"
                )
        substitute.append(
            "record the model-level distinction (different PDE class / state variable) as "
            "the deliverable instead of an error table"
        )

    return ScreenVerdict(
        left=a.key,
        right=b.key,
        verdict=verdict,
        predicates=predicates,
        substitute_path=substitute,
    )


# --------------------------------------------------------------------------------------
# analytic anchor: 1-D AT2 equilibrium profile
# --------------------------------------------------------------------------------------
def at2_1d_reference_profile(x: np.ndarray, ell: float, length: float) -> np.ndarray:
    """Exact 1-D AT2 damage profile for ``d(0)=1`` and the natural BC ``d'(L)=0``.

    The homogeneous, strain-free strong form of the AT2 functional
    ``int Gc (d^2/(2*l) + l/2 d'^2) dV`` is ``d = ell^2 d''``; with the two boundary
    conditions above its solution is ``cosh((L-x)/ell)/cosh(L/ell)``, which degenerates
    to the familiar ``exp(-x/ell)`` for ``L >> ell``.
    """
    x = np.asarray(x, dtype=np.float64)
    if ell <= 0.0:
        raise ValueError("ell must be positive")
    return np.cosh((length - x) / ell) / math.cosh(length / ell)


def solve_at2_1d_profile(
    ell: float,
    length: float,
    n_elem: int,
    *,
    E: float = 1.0e3,
    nu: float = 0.3,
    Gc: float = 1.0,
) -> dict:
    """Reproduce :func:`at2_1d_reference_profile` with the production AT2 solver.

    A traction-free strip ``[0, L] x [0, h]`` (``h = L/n_elem``, one element through
    the thickness) is used with *all* displacement DOFs fixed to zero, so the strain
    energy -- hence the history field and the damage driving force -- vanishes
    identically.  The damage sub-problem is then linear, so a single Newton step from
    ``d = 0`` lands on the exact discrete solution (no iteration error contaminates
    the comparison).  ``d = 1`` is prescribed at ``x = 0``.
    """
    if n_elem < 4:
        raise ValueError("n_elem must be >= 4")
    h = length / n_elem
    nodes, elements = structured_quad_mesh(0.0, length, 0.0, h, n_elem, 1)
    problem = PhaseFieldProblem(nodes, elements, E, nu, Gc=Gc, l=ell, thickness=h)

    n_nodes = problem.n_nodes
    fixed = {int(i): 0.0 for i in range(2 * n_nodes)}          # u == 0 everywhere
    crack_nodes = np.where(np.abs(nodes[:, 0]) <= 1e-12)[0]
    solver = PhaseFieldSolver(problem, fixed=fixed)
    solver.fix_damage_nodes(crack_nodes, 1.0)
    record = solver.newton_step(max_iter=8, tol=1e-12, line_search=False)

    x = nodes[:, 0]
    d_num = solver.d.copy()
    d_ref = at2_1d_reference_profile(x, ell, length)

    # L2 error over the whole strip normalised by the reference L2 norm.
    order = np.argsort(x)
    xs, dn, dr = x[order], d_num[order], d_ref[order]
    l2_ref = float(np.sqrt(np.trapezoid(dr ** 2, xs)))
    l2_err = float(np.sqrt(np.trapezoid((dn - dr) ** 2, xs)))
    finite = dr > 1e-12
    nodal_max = float(np.max(np.abs(dn[finite] - dr[finite]) / dr[finite]))

    return {
        "ell": float(ell),
        "length": float(length),
        "n_elem": int(n_elem),
        "h": float(h),
        "h_over_l": float(h / ell),
        "newton_iterations": int(record.iterations),
        "newton_converged": bool(record.converged),
        "residual_norm": float(record.residual_norms[-1]) if record.residual_norms else float("nan"),
        "x": xs,
        "d_numeric": dn,
        "d_reference": dr,
        "l2_relative_error": l2_err / max(l2_ref, 1e-30),
        "nodal_max_relative_error": nodal_max,
    }


def interface_width_ratio(ell: float, length: float, n_elem: int, **kwargs) -> float:
    """Measured ``x(d = 1/e) / ell`` -- the dimensionless width of the numeric profile."""
    result = solve_at2_1d_profile(ell, length, n_elem, **kwargs)
    x = result["x"]
    d = result["d_numeric"]
    target = math.exp(-1.0)
    # d decreases monotonically: walk from x = 0 to the first node below 1/e
    for i in range(1, x.size):
        if d[i] <= target:
            x0, x1 = x[i - 1], x[i]
            d0, d1 = d[i - 1], d[i]
            if d0 == d1:
                return float(x0 / ell)
            x_at = x0 + (d0 - target) * (x1 - x0) / (d0 - d1)
            return float(x_at / ell)
    return float("nan")


def convergence_study(
    ell: float = 0.05,
    length: float = 1.0,
    refinements: tuple[int, ...] = (5, 10, 20, 40, 80),
    **kwargs,
) -> list[dict]:
    """h-refinement ladder on the 1-D AT2 profile (returns compact per-level rows)."""
    rows = []
    for n in refinements:
        res = solve_at2_1d_profile(ell, length, n, **kwargs)
        rows.append(
            {
                "n_elem": int(n),
                "h_over_l": res["h_over_l"],
                "l2_relative_error": res["l2_relative_error"],
                "nodal_max_relative_error": res["nodal_max_relative_error"],
                "newton_iterations": res["newton_iterations"],
                "newton_converged": res["newton_converged"],
            }
        )
    return rows


def observed_order(rows: list[dict], h_over_l_max: float = 1.0) -> float:
    """Least-squares slope of ``log(L2 error)`` against ``log(h)`` (expected ~2).

    Only levels with ``h/ell <= h_over_l_max`` enter the fit: once an element is
    coarser than the regularisation length the discrete problem is *not* in the
    asymptotic range (and at ``h/ell >= 4`` the linear damage solve even fails to
    converge, see :func:`convergence_study`), so including those levels would
    measure pre-asymptotic behaviour rather than the convergence rate.
    """
    sel = [r for r in rows if r["h_over_l"] <= h_over_l_max]
    h = np.array([r["h_over_l"] for r in sel], dtype=np.float64)
    e = np.array([r["l2_relative_error"] for r in sel], dtype=np.float64)
    keep = e > 0.0
    if keep.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(h[keep]), np.log(e[keep]), 1)
    return float(slope)


def solve_at2_homogeneous(
    H: float,
    ell: float = 0.05,
    *,
    Gc: float = 1.0,
    E: float = 1.0e3,
    nu: float = 0.3,
    n_elem: int = 8,
) -> dict:
    """Uniform-history anchor: the AT2 damage saturation law ``d = 2Hl/(Gc + 2Hl)``.

    With a spatially constant driving force ``H`` and vanishing strain the damage
    Euler-Lagrange equation collapses to ``Gc d/ell = 2(1-d) H``, whose solution is the
    algebraic expression above.  The test body is loaded through the history field
    (``solver.history``), i.e. the very irreversibility floor the production path
    uses, so this checks the history branch and not just the Laplace term.
    """
    length = 8.0 * ell
    h = length / n_elem
    nodes, elements = structured_quad_mesh(0.0, length, 0.0, h, n_elem, 1)
    problem = PhaseFieldProblem(nodes, elements, E, nu, Gc=Gc, l=ell, thickness=h)
    solver = PhaseFieldSolver(problem, fixed={int(i): 0.0 for i in range(2 * problem.n_nodes)})
    solver.history = np.full((problem.n_elem, 4), float(H))
    record = solver.newton_step(max_iter=20, tol=1e-13, line_search=False)
    d_num = float(np.mean(solver.d))
    d_ref = 2.0 * H * ell / (Gc + 2.0 * H * ell)
    return {
        "H": float(H),
        "ell": float(ell),
        "Gc": float(Gc),
        "d_numeric": d_num,
        "d_reference": d_ref,
        "relative_error": abs(d_num - d_ref) / max(d_ref, 1e-30),
        "newton_converged": bool(record.converged),
        "newton_iterations": int(record.iterations),
    }


def resolution_threshold(rows: list[dict], rel_tol: float = 0.03) -> dict:
    """Coarsest ``h/ell`` in ``rows`` for which both error metrics are below ``rel_tol``."""
    ok = [
        r for r in rows
        if r["l2_relative_error"] < rel_tol and r["nodal_max_relative_error"] < rel_tol
    ]
    if not ok:
        return {"h_over_l_max": float("nan"), "n_elem_min": None}
    best = max(ok, key=lambda r: r["h_over_l"])
    return {"h_over_l_max": best["h_over_l"], "n_elem_min": best["n_elem"]}


# --------------------------------------------------------------------------------------
# verdict bundle
# --------------------------------------------------------------------------------------
#: Owner precision standard: >15 % is unacceptable; the doc-level self-check uses 3 %.
ACCEPTANCE_REL_TOL = 0.03


def not_applicable_verdict(
    ell: float = 0.05,
    length: float = 1.0,
    n_elem: int = 160,
    refinements: tuple[int, ...] = (5, 10, 20, 40, 80, 160),
) -> dict:
    """Full evidence bundle for the phase-field vs pseudopotential screening."""
    inventory = platform_implementation_inventory()
    verdict = screen_applicability(PHASE_FIELD_AT2, PSEUDOPOTENTIAL_LBM, inventory=inventory)

    reference_run = solve_at2_1d_profile(ell, length, n_elem)
    ladder = convergence_study(ell, length, refinements)
    order = observed_order(ladder)
    homogeneous = [solve_at2_homogeneous(H, ell) for H in (0.5, 2.0, 10.0)]
    widths = []
    for factor in (1.0, 2.0, 4.0):
        ll = ell * factor
        ratio = interface_width_ratio(ll, length * factor, n_elem)
        widths.append({"ell": ll, "l_over_L": ll / (length * factor), "width_ratio": ratio})

    err_l2 = reference_run["l2_relative_error"]
    err_nodal = reference_run["nodal_max_relative_error"]
    same_width = max(w["width_ratio"] for w in widths) / min(
        w["width_ratio"] for w in widths
    )

    return {
        "anchor": "XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT",
        "screen": verdict.to_dict(),
        "inventory": inventory,
        "at2_reference_run": {
            k: v
            for k, v in reference_run.items()
            if k not in ("x", "d_numeric", "d_reference")
        },
        "at2_convergence_ladder": ladder,
        "at2_observed_l2_order": order,
        "at2_resolution_threshold": resolution_threshold(ladder, ACCEPTANCE_REL_TOL),
        "at2_homogeneous_anchor": homogeneous,
        "width_scaling": widths,
        "acceptance": {
            "rel_tol": ACCEPTANCE_REL_TOL,
            "at2_profile_l2_ok": bool(err_l2 < ACCEPTANCE_REL_TOL),
            "at2_profile_nodal_ok": bool(err_nodal < ACCEPTANCE_REL_TOL),
            "l2_order_ok": bool(order > 1.5),
            "at2_homogeneous_ok": bool(
                max(hh["relative_error"] for hh in homogeneous) < ACCEPTANCE_REL_TOL
            ),
            "width_invariant_under_l_scaling": bool(same_width < 1.10),
            "verdict_is_not_applicable": verdict.verdict == "NOT_APPLICABLE",
        },
    }