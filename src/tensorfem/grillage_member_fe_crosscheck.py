"""Independent nonlinear-FE cross-check of the member-governed ice-grillage ultimate.

Why this module exists
----------------------
:mod:`tensorfem.grillage_ice_ultimate` closes the ``GAP-PP-5-ICE-STRUCT-
GRILLAGE-ULTIMATE`` "last jump" (plate-field bending stress -> structural
ultimate) with a **closed-form / empirical** chain.  The plate-field branch has
been cross-checked against an independent 13-point FD biharmonic solver (T11)
and against a bare-plate nonlinear-FE collapse
(:mod:`tensorfem.grillage_ice_fe_crosscheck`).  The **member-governed** branch
(longitudinal stiffener beam-column, effective-width Perry-Robertson /
Johnson-Ostenfeld, secant magnification) has, however, only been checked
against (a) closed-form algebra that shares the same section convention (T12)
and (b) a rigid-plastic grillage collapse that is a *different* limit state
(T13).  It has never been cross-checked against an independent **nonlinear
finite-element** elasto-plastic response of the same member.  That is the
verification debt this module repays.

What it does
------------
For the member-governed scantling of
:func:`tensorfem.grillage_ice_verification.member_governed_geometry`
(25 mm plating, 0.55 m bay, T160x6+50x7 longitudinal stiffener, 2.4 m frame
spacing), it builds a **3D small-strain J2 elasto-plastic solid** of the
longitudinal T-stiffener (web + flange) spanning ``frame_spacing`` between two
transverse frames (simply supported: the end faces release the axial DOF and
fix the transverse DOFs, so the member can buckle in the weak direction).

The model carries an **initial geometric imperfection** of amplitude
``imperfection_ratio * L`` (the same ratio the closed-form chain uses for the
Perry-Robertson reduction), in the first buckling half-sine mode.  The load is
the **axial compression** applied as a uniform end traction; the ice patch and
hydrostatic head are *not* applied in this trace (they enter the closed-form
chain through the bending term ``M * beta_m / Z``, which is verified
separately by T12).  This isolates the **axial** part of the member limit --
the part that the closed-form chain computes with the Perry-Robertson /
Johnson-Ostenfeld empirical column curves -- and cross-checks it against an
independent elasto-plastic FE.

The elasto-plastic path is traced with a transactional Newton driver
(:func:`tensorfem.solid_plasticity.assemble_tet4_j2` + a local solve).  The
**FE peak load factor** ``lam_peak`` is the largest converged multiplier on the
axial reference force; it is the load at which the member actually
yields/buckles in the FE.

The cross-check asserts the **conservatism** of the closed-form member limit:
the closed-form ``longitudinal_stiffener`` utilisation is a *first-yield /
beam-column* criterion, which is a **lower bound** on the true elasto-plastic
peak of the imperfect member.  The FE peak must therefore be **at or above**
the closed-form prediction; an over-prediction would be an unsafe design
formula.  The ratio ``lam_peak / closed_lam`` is the quantified conservatism
margin.

Scope and honesty
-----------------
* The FE model is the **stiffener only** (no attached plate); the comparison
  is against the *stiffener-only* section (``A_g`` / ``I_g``), not the
  effective section (``A_e`` / ``I_e``).  The plate's contribution (Faulkner
  effective width) is verified separately by the T7 gate (Paik-Thayamballi).
  This is a deliberate scoping choice: the stiffener-only column test is the
  cleanest possible "axial last jump" and does not entangle the two load
  branches.
* The FE model is **small-strain** (infinitesimal-strain J2, no finite
  rotation); the closed-form chain is also small-strain, so the comparison is
  apples-to-apples on the strain measure.
* The FE peak is an **elasto-plastic peak** (the member yields and the load
  drops); the closed-form limit is a **first-yield / beam-column** criterion.
  The FE peak is expected to be *at or above* the closed-form limit (the
  closed form is conservative).  A large gap (> 30 %) would indicate that the
  closed form is over-conservative and should be revisited.
* No residual stress, no strain-rate effects, no web tripping -- the same
  scope as the closed-form module.
* The FE model is the **stiffener only**; the transverse frame and the rest of
  the grillage are not modelled (the frame branch is verified separately by
  T6 / T13).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    johnson_ostenfeld,
)
from .grillage_ice_verification import (
    MEMBER_GOVERNED_SCENARIOS,
    member_governed_geometry,
)
from .solid_plasticity import (
    PlasticSolidState,
    Tet4J2Model,
    assemble_tet4_j2,
)

__all__ = [
    "build_stiffener_column_model",
    "stiffener_gross_section",
    "stiffener_column_capacity",
    "axial_reference_load",
    "trace_member_collapse",
    "crosscheck_member_governed",
    "run_member_fe_crosscheck",
    "main",
]


# ---------------------------------------------------------------------------
# 3D solid mesh of the T-stiffener (web + flange)
# ---------------------------------------------------------------------------

def _hex_to_tet(i0, i1, i2, i3, i4, i5, i6, i7):
    """Split one hexahedron into six TET4 (6-tet split, positive orientation).

    The node ordering is the standard one used by :func:`_block`
    (``i0..i3`` bottom face, ``i4..i7`` top face, ``i_k + 4`` above ``i_k``).
    The split uses the body diagonal ``i0 - i7``; the first three tets are
    pre-flipped so that every tet has a positive Jacobian determinant
    (``solid3d._gradient`` rejects non-positive determinants).
    """
    return (
        (i0, i4, i6, i7),
        (i0, i6, i2, i7),
        (i0, i2, i3, i7),
        (i0, i1, i3, i7),
        (i0, i5, i1, i7),
        (i0, i4, i5, i7),
    )


def _block(x0, x1, y0, y1, z0, z1, nx, ny, nz,
           all_nodes: list, all_tets: list) -> None:
    """Append a structured hex block (split into 6-tet) to the global lists."""
    def idx(ix, iy, iz):
        return (iz * (ny + 1) + iy) * (nx + 1) + ix

    base = len(all_nodes)
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                all_nodes.append([
                    x0 + (x1 - x0) * ix / nx,
                    y0 + (y1 - y0) * iy / ny,
                    z0 + (z1 - z0) * iz / nz,
                ])
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                hx = (
                    idx(ix, iy, iz), idx(ix + 1, iy, iz),
                    idx(ix + 1, iy + 1, iz), idx(ix, iy + 1, iz),
                    idx(ix, iy, iz + 1), idx(ix + 1, iy, iz + 1),
                    idx(ix + 1, iy + 1, iz + 1), idx(ix, iy + 1, iz + 1),
                )
                for tet in _hex_to_tet(*hx):
                    all_tets.append(tuple(base + i for i in tet))


def build_stiffener_column_model(
    geom: GrillageGeometry,
    *,
    nx: int = 12,
    n_web_z: int = 2,
    n_flange_z: int = 1,
    n_web_y: int = 2,
    n_flange_y: int = 2,
    with_imperfection: bool = True,
) -> tuple[Tet4J2Model, dict[str, float]]:
    """Assemble the :class:`Tet4J2Model` for the T-stiffener column.

    Coordinate system: ``x`` runs along the stiffener (length ``L =
    frame_spacing``), ``y`` across the web (width ``tw``), ``z`` through the
    thickness.  The web occupies ``z in [-h, 0]`` and the flange
    ``z in [-h - tf, -h]`` (the web top is at ``z = 0``, the flange below).

    Boundary conditions: the two end faces (``x = 0`` and ``x = L``) are
    simply supported -- the axial DOF (``x``) is free, the transverse DOFs
    (``y``, ``z``) are fixed.  This lets the member buckle in ``y`` (about the
    strong axis of the T-section) while preventing rigid-body motion (one
    corner node is additionally fixed in ``x``).
    """
    geom.validate()
    h = geom.stiffener.web_height
    tw = geom.stiffener.web_thickness
    bf = geom.stiffener.flange_width
    tf = geom.stiffener.flange_thickness
    L = geom.frame_spacing

    all_nodes: list[list[float]] = []
    all_tets: list[tuple[int, ...]] = []
    # stiffener web: y in [-tw/2, tw/2], z in [-h, 0]
    _block(0.0, L, -0.5 * tw, 0.5 * tw, -h, 0.0, nx, n_web_y, n_web_z,
           all_nodes, all_tets)
    # stiffener flange: y in [-bf/2, bf/2], z in [-h - tf, -h]
    _block(0.0, L, -0.5 * bf, 0.5 * bf, -h - tf, -h, nx, n_flange_y, n_flange_z,
           all_nodes, all_tets)

    nodes = torch.tensor(all_nodes, dtype=torch.float64)
    elements = torch.tensor(all_tets, dtype=torch.long)

    if with_imperfection:
        amp = geom.imperfection_ratio * L
        x = nodes[:, 0]
        nodes[:, 1] = nodes[:, 1] + amp * torch.sin(math.pi * x / L)

    fixed = []
    for i in range(nodes.shape[0]):
        x = float(nodes[i, 0])
        if x <= 1e-9 or abs(x - L) <= 1e-9:
            fixed.append(3 * i + 1)
            fixed.append(3 * i + 2)
    fixed.append(0)  # kill the rigid-body x-translation
    fixed = torch.tensor(sorted(set(fixed)), dtype=torch.long)

    model = Tet4J2Model(
        nodes, elements, geom.young, geom.poisson,
        geom.yield_stress, 0.0, fixed,  # hardening = 0 (elastic-perfectly-plastic)
    )
    meta = {
        "L_m": L,
        "web_h_m": h,
        "web_t_m": tw,
        "flange_b_m": bf,
        "flange_t_m": tf,
        "nx": nx,
        "n_web_z": n_web_z,
        "n_flange_z": n_flange_z,
        "n_web_y": n_web_y,
        "n_flange_y": n_flange_y,
        "n_nodes": int(nodes.shape[0]),
        "n_tet": int(elements.shape[0]),
        "imperfection_amp_m": float(geom.imperfection_ratio * L),
    }
    return model, meta


# ---------------------------------------------------------------------------
# Stiffener-only section properties and column capacity (independent algebra)
# ---------------------------------------------------------------------------

def stiffener_gross_section(geom: GrillageGeometry) -> dict[str, float]:
    """Gross section properties of the T-stiffener (no plate).

    Independent of ``MemberSection``: the laminae are laid out explicitly
    (web at ``z in [-h, 0]``, flange at ``z in [-h - tf, -h]``) so that a
    defect in the composite-property algebra cannot hide in both places.
    """
    h = geom.stiffener.web_height
    tw = geom.stiffener.web_thickness
    bf = geom.stiffener.flange_width
    tf = geom.stiffener.flange_thickness
    parts = [
        (h * tw, -0.5 * h, tw * h ** 3 / 12.0),
        (bf * tf, -h - 0.5 * tf, bf * tf ** 3 / 12.0),
    ]
    area = sum(a for a, _, _ in parts)
    z_bar = sum(a * z for a, z, _ in parts) / area
    inertia = sum(a * (z - z_bar) ** 2 + i0 for a, z, i0 in parts)
    c_ext = max(max(abs(z - z_bar) for _, z, _ in parts), abs(z_bar))
    return {
        "area": area,
        "inertia": inertia,
        "c_ext": c_ext,
        "z_section": inertia / c_ext,
        "z_bar": z_bar,
    }


def stiffener_column_capacity(geom: GrillageGeometry) -> dict[str, float]:
    """Perry-Robertson / Johnson-Ostenfeld column capacity of the T-stiffener.

    Independent transcription: ``sigma_e = pi^2 E I / (A L^2)``,
    ``eta = e0 * c * A / I`` with ``e0 = imperfection_ratio * L``,
    ``sigma_PR`` from the Perry-Robertson quadratic, ``sigma_JO`` by
    Johnson-Ostenfeld, ``sigma_L = min`` thereof.  The capacity is
    ``sigma_L * A``.
    """
    geom.validate()
    sec = stiffener_gross_section(geom)
    A = sec["area"]
    I = sec["inertia"]
    c = sec["c_ext"]
    L = geom.frame_spacing
    sigma_y = geom.yield_stress
    sigma_e = math.pi ** 2 * geom.young * I / (A * L ** 2)
    e0 = geom.imperfection_ratio * L
    eta = e0 * c * A / I
    s = sigma_y + sigma_e * (1.0 + eta)
    sigma_pr = (s - math.sqrt(max(s * s - 4.0 * sigma_y * sigma_e, 0.0))) / 2.0
    sigma_jo = johnson_ostenfeld(sigma_y, sigma_e)
    sigma_l = min(sigma_pr, sigma_jo)
    return {
        "area": A,
        "inertia": I,
        "c_ext": c,
        "euler_stress": sigma_e,
        "perry_robertson_stress": sigma_pr,
        "johnson_ostenfeld_stress": sigma_jo,
        "limiting_stress": sigma_l,
        "capacity_N": sigma_l * A,
    }


# ---------------------------------------------------------------------------
# Axial reference load
# ---------------------------------------------------------------------------

def axial_reference_load(
    model: Tet4J2Model,
    geom: GrillageGeometry,
    case: CombinedLoadCase,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Nodal load vector for the axial compression ``sigma_x * A_g``.

    The force is applied as a uniform traction on the ``x = L`` face (the
    ``x = 0`` face is the reaction face).  Returns ``(fext, info)``.
    """
    geom.validate()
    case.validate(geom)
    fext = torch.zeros(model.n_dofs, dtype=model.nodes.dtype)
    L = geom.frame_spacing
    sec = stiffener_gross_section(geom)
    A = sec["area"]
    axial_force = case.axial_stress * A
    end_nodes = [i for i in range(model.nodes.shape[0])
                 if abs(float(model.nodes[i, 0]) - L) <= 1e-9]
    if end_nodes and axial_force > 0.0:
        per = axial_force / len(end_nodes)
        for i in end_nodes:
            fext[3 * i] = -per  # compression: force in -x on the x=L face
    info = {
        "axial_force_N": float(axial_force),
        "area_gross_m2": float(A),
        "axial_stress_Pa": float(case.axial_stress),
    }
    return fext, info


# ---------------------------------------------------------------------------
# Elasto-plastic trace (load control on the axial force, sub-incrementation)
# ---------------------------------------------------------------------------

def _newton_step(model: Tet4J2Model, force: torch.Tensor, state: PlasticSolidState,
                 u: torch.Tensor, *, tolerance: float, max_iterations: int):
    """One equilibrium solve at a fixed force; returns (u, state, iters)."""
    fixed = model.fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.arange(model.n_dofs, device=model.nodes.device)[mask]
    scale = max(float(torch.linalg.vector_norm(force[free])), 1.0)
    for iteration in range(1, max_iterations + 1):
        internal, tangent, _stress, trial = assemble_tet4_j2(model, u, state)
        residual = force - internal
        norm = float(torch.linalg.vector_norm(residual[free]))
        if norm <= tolerance * scale:
            return u, trial, iteration
        if free.numel() == 0:
            break
        du = torch.linalg.solve(tangent[free][:, free], residual[free])
        u = u.index_add(0, free, du)
    raise RuntimeError(f"Newton failed to converge after {max_iterations} iterations")


def trace_member_collapse(
    model: Tet4J2Model,
    fext: torch.Tensor,
    *,
    n_steps: int = 60,
    dlam0: float = 0.02,
    max_sub: int = 8,
    tolerance: float = 1e-8,
) -> dict[str, object]:
    """Trace the elasto-plastic path with the axial force as the control load.

    ``force = lam * fext``; ``lam`` is the multiplier on the axial reference
    force.  Load control with automatic sub-incrementation: if a step fails to
    converge, it is halved (up to ``max_sub`` halvings).  The peak load factor
    is the largest converged ``lam``; the path is recorded for the artifact.
    """
    state = PlasticSolidState.virgin(model)
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    L = float(model.nodes[:, 0].max())
    mid = min(range(model.nodes.shape[0]),
              key=lambda i: abs(float(model.nodes[i, 0]) - 0.5 * L))
    y0_mid = float(model.nodes[mid, 1])

    path_lam: list[float] = []
    path_disp: list[float] = []
    peak = 0.0
    failed_at: float | None = None
    lam = 0.0
    step = dlam0
    converged_total = 0
    for _ in range(n_steps):
        target = lam + step
        ok = False
        sub = step
        for _s in range(max_sub):
            try:
                force = target * fext
                u, state, _it = _newton_step(model, force, state, u,
                                             tolerance=tolerance, max_iterations=40)
                lam = target
                ok = True
                break
            except RuntimeError:
                sub = 0.5 * sub
                target = lam + sub
        if not ok:
            failed_at = lam
            break
        peak = max(peak, lam)
        path_lam.append(lam)
        path_disp.append(float(u[3 * mid + 1]) - y0_mid)
        converged_total += 1
        step = min(step * 1.05, dlam0 * 2.0)
    return {
        "lam_peak": float(peak),
        "failed_at": failed_at,
        "converged_steps": converged_total,
        "path_lam": path_lam,
        "path_disp": path_disp,
    }


# ---------------------------------------------------------------------------
# Cross-check driver
# ---------------------------------------------------------------------------

def crosscheck_member_governed(
    geom: GrillageGeometry | None = None,
    *,
    nx: int = 12,
    n_steps: int = 60,
    dlam0: float = 0.02,
) -> dict[str, object]:
    """Run the FE cross-check for the member-governed scenarios (eta > 0).

    For each scenario in :data:`MEMBER_GOVERNED_SCENARIOS` with ``eta > 0``,
    it builds the FE model, traces the elasto-plastic path (axial force as the
    control load), and compares the FE peak load factor against the closed-form
    axial capacity (the load at which the stiffener's axial utilisation
    reaches 1).  Both are expressed as a multiplier on the case's axial
    reference force, so the comparison is direct.
    """
    if geom is None:
        geom = member_governed_geometry()
    geom.validate()

    results = []
    for name, eta, p_ice, lx, ly, p_h in MEMBER_GOVERNED_SCENARIOS:
        if eta <= 0.0:
            continue  # no axial load: the axial-only FE is not applicable
        case = CombinedLoadCase(
            axial_stress=eta * geom.yield_stress,
            ice_pressure=p_ice,
            patch_length=lx,
            patch_width=ly,
            hydrostatic_pressure=p_h,
        ).validate(geom)
        # closed-form axial capacity (the load at which u_axial = 1)
        cap = stiffener_column_capacity(geom)
        axial_capacity = cap["capacity_N"]
        # the case's axial force
        sec = stiffener_gross_section(geom)
        axial_force = case.axial_stress * sec["area"]
        # the closed-form limit as a multiplier on the case's axial force
        closed_lam = axial_capacity / axial_force if axial_force > 0.0 else float("inf")

        model, meta = build_stiffener_column_model(geom, nx=nx)
        fext, info = axial_reference_load(model, geom, case)
        trace = trace_member_collapse(model, fext, n_steps=n_steps, dlam0=dlam0)
        fe_lam = float(trace["lam_peak"])
        results.append({
            "scenario": name,
            "eta": eta,
            "p_ice_case_MPa": p_ice / 1e6,
            "p_h_MPa": p_h / 1e6,
            "fe_lam_peak": fe_lam,
            "fe_failed_at": trace["failed_at"],
            "converged_steps": trace["converged_steps"],
            "closed_form_lam": float(closed_lam),
            "closed_form_axial_capacity_N": float(axial_capacity),
            "case_axial_force_N": float(axial_force),
            "conservatism_ratio": (fe_lam / closed_lam) if closed_lam > 0 else None,
            "info": info,
        })
    return {
        "geometry": {
            "frame_spacing_m": geom.frame_spacing,
            "bay_spacing_m": geom.bay_spacing,
            "plate_thickness_mm": geom.plate_thickness * 1e3,
            "stiffener": (f"T{geom.stiffener.web_height*1e3:.0f}x"
                          f"{geom.stiffener.web_thickness*1e3:.0f}+"
                          f"{geom.stiffener.flange_width*1e3:.0f}x"
                          f"{geom.stiffener.flange_thickness*1e3:.0f}"),
            "imperfection_ratio": geom.imperfection_ratio,
        },
        "mesh": meta,
        "scenarios": results,
    }


def run_member_fe_crosscheck(
    out_dir: str | Path = "outputs/grillage_member_fe_crosscheck",
    *,
    nx: int = 12,
    n_steps: int = 60,
    dlam0: float = 0.02,
) -> dict[str, object]:
    """Run the cross-check and write the machine-readable artifact."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = crosscheck_member_governed(geom=None, nx=nx,
                                        n_steps=n_steps, dlam0=dlam0)
    (out / "member_fe_crosscheck.json").write_text(
        json.dumps(result, indent=2, default=float))
    return result


def main() -> None:
    result = run_member_fe_crosscheck()
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
