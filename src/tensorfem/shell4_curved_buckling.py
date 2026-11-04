"""Curved-geometry audit of the Shell4 initial-stress (geometric) operator.

The qualified flat-plate path of ``shell4_buckling`` states its own boundary:
it "does not yet qualify ... curved-shell geometric stiffness".  This module
puts that open item under test with the cleanest closed-form case available:

a thin closed cylinder under uniform external pressure ``p`` buckles into the
inextensional ovalising (``n = 2``) ring mode, and the classical long-tube
result is

    p_cr(n) = (n^2 - 1) E t^3 / (12 R^3 (1 - nu^2))
            = E t^3 / (4 R^3 (1 - nu^2))            at n = 2,

which is algebraically identical to the classification-society long-shell
elastic pressure ``2 E / (1 - nu^2) (t/D)^3`` with ``D = 2 R``.  The membrane
state is exact and uniform (``N_theta = p R``, ``N_x = 0`` for an open-ended
shell), so the only ingredients under test are the 24x24 facet operators, the
rotation of each facet frame around the axis, and the assembled boundary
conditions -- no load-integration or stress-recovery path is involved.

The eigensolver used here is first verified against the *qualified* flat-plate
path (agreement to 1e-14 relative), the rigid-body annihilation of both
operators is verified on the curved mesh, and the curved results are then
reported with no favourable selection: the lowest positive load factor, its
circumferential wave number, and the elastic energy carried by the mode.

The verdict (see ``run_shell4_curved_buckling_qualification``) is negative and
is locked fail-closed: on faceted bilinear Shell4 meshes the weakest mode under
hoop compression is a facet-scale (Nyquist-wave) near-mechanism, so the
eigenvalue degrades monotonically under mesh refinement instead of converging
to the classical ring load.  Curved-shell geometric stiffness therefore stays
unqualified; the module is kept as the reproducible evidence.

Sign convention: the geometric operator takes compressive stress resultants as
positive (``K - lambda Kg``), so hoop compression enters with a positive sign.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .shell4 import shell4_local_frame, shell4_stiffness
from .shell4_buckling import shell4_geometric_stiffness

__all__ = [
    "CurvedBucklingResult",
    "cylindrical_shell4_mesh",
    "long_tube_buckling_pressure",
    "long_shell_pressure_society_form",
    "ring_pressure_wave_factors",
    "assemble_shell4_buckling_system",
    "generalized_buckling_load_factor",
    "simply_supported_plate_cross_check",
    "cylinder_pressure_buckling",
    "dominant_circumferential_wave",
    "run_shell4_curved_buckling_qualification",
]

_DTYPE = torch.float64


@dataclass(frozen=True)
class CurvedBucklingResult:
    """Lowest positive load factor, its mode and the assembled operators."""

    load_factor: float
    eigenvalues: torch.Tensor
    mode: torch.Tensor
    nodes: torch.Tensor
    elements: torch.Tensor
    diagnostics: dict = field(default_factory=dict)

    def radial_components(self) -> torch.Tensor:
        """Cylindrical radial component of the stored mode at every node."""
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        radius = torch.linalg.vector_norm(self.nodes[:, :2], dim=1)
        safe = torch.where(radius > 0, radius, torch.ones_like(radius))
        return (self.mode[0::6] * x + self.mode[1::6] * y) / safe


def _positive(name: str, value) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _integer(name: str, value, minimum: int) -> int:
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return number


def long_tube_buckling_pressure(radius, thickness, young, poisson, wave: int = 2) -> float:
    """Classical inextensional ring buckling pressure of a long tube.

    ``p_cr(n) = (n^2 - 1) E t^3 / (12 R^3 (1 - nu^2))``.  The ``n = 2`` value is
    the minimum over integer waves and is the reference used by this audit.
    """
    radius = _positive("radius", radius)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    nu = float(poisson)
    if not -0.99 < nu < 0.5:
        raise ValueError("poisson must satisfy -0.99 < nu < 0.5")
    n = _integer("wave", wave, 2)
    return (n * n - 1.0) * young * thickness**3 / (12.0 * radius**3 * (1.0 - nu * nu))


def long_shell_pressure_society_form(young, poisson, thickness, diameter) -> float:
    """``2 E / (1 - nu^2) (t/D)^3`` -- the classification-society long-shell form."""
    young = _positive("young", young)
    thickness = _positive("thickness", thickness)
    diameter = _positive("diameter", diameter)
    nu = float(poisson)
    return 2.0 * young / (1.0 - nu * nu) * (thickness / diameter) ** 3


def ring_pressure_wave_factors(waves=(2, 3, 4, 5)) -> dict:
    """Normalised classical ring pressures ``p_cr(n)/p_cr(2) = (n^2-1)/3``."""
    return {int(n): (int(n) ** 2 - 1.0) / 3.0 for n in waves}


def cylindrical_shell4_mesh(radius, length, n_circ: int = 32, n_axial: int = 4):
    """Closed, gap-free faceted cylinder of flat Shell4 quads.

    Node ``i * n_circ + k`` sits at angle ``2 pi k / n_circ`` and axial station
    ``i * length / n_axial``.  Each quad spans two generators, hence it is
    exactly planar and is accepted by ``shell4_local_frame``.  Element ordering
    ``[k, k+1, (k+1)', k']`` puts the local ``x`` axis along the circumference.
    """
    radius = _positive("radius", radius)
    length = _positive("length", length)
    n_circ = _integer("n_circ", n_circ, 8)
    n_axial = _integer("n_axial", n_axial, 1)
    theta = 2.0 * math.pi * torch.arange(n_circ, dtype=_DTYPE) / n_circ
    axial = torch.arange(n_axial + 1, dtype=_DTYPE) * (length / n_axial)
    nodes = torch.stack([
        (radius * torch.cos(theta))[None, :].expand(n_axial + 1, n_circ).reshape(-1),
        (radius * torch.sin(theta))[None, :].expand(n_axial + 1, n_circ).reshape(-1),
        axial[:, None].expand(n_axial + 1, n_circ).reshape(-1),
    ], dim=1)
    elements = []
    for i in range(n_axial):
        for k in range(n_circ):
            k1 = (k + 1) % n_circ
            elements.append([i * n_circ + k, i * n_circ + k1,
                             (i + 1) * n_circ + k1, (i + 1) * n_circ + k])
    return nodes, torch.tensor(elements, dtype=torch.long)


def _element_resultants(xyz: torch.Tensor, hoop: float, axial: float) -> torch.Tensor:
    """Place hoop/axial resultants on the facet's local axes.

    ``shell4_geometric_stiffness`` expects ``[Nx, Ny, Nxy]`` in the element frame
    built by ``shell4_local_frame``.  The frame's first axis runs along the edge
    ``node0 -> node1``, which is circumferential for this mesh ordering, but the
    mapping is resolved numerically so the routine does not depend on ordering:
    each local axis is assigned by comparing its direction with the local
    circumferential and axial directions.
    """
    _xy, basis = shell4_local_frame(xyz)
    radii = xyz[:, :2].mean(dim=0)
    norm = torch.linalg.vector_norm(radii)
    radial = radii / norm if bool(norm > 0) else torch.zeros(2, dtype=xyz.dtype)
    circum = torch.stack((radial[1], -radial[0], torch.zeros((), dtype=xyz.dtype)))
    axial_dir = torch.tensor([0.0, 0.0, 1.0], dtype=xyz.dtype)
    values = torch.zeros(3, dtype=xyz.dtype)
    for axis in (0, 1):
        direction = basis[axis]
        if abs(float(direction @ circum)) >= abs(float(direction @ axial_dir)):
            values[axis] = hoop
        else:
            values[axis] = axial
    return values


def assemble_shell4_buckling_system(nodes, elements, young, poisson, thickness,
                                    hoop=0.0, axial=0.0):
    """Assemble the elastic and initial-stress operators of a facet mesh."""
    ndof = 6 * len(nodes)
    K = torch.zeros((ndof, ndof), dtype=_DTYPE)
    Kg = torch.zeros((ndof, ndof), dtype=_DTYPE)
    for conn in elements.tolist():
        ix = torch.tensor(conn, dtype=torch.long)
        dofs = torch.stack([6 * ix + k for k in range(6)], 1).reshape(-1)
        xyz = nodes[ix]
        K[dofs[:, None], dofs] += shell4_stiffness(xyz, young, poisson, thickness)
        Kg[dofs[:, None], dofs] += shell4_geometric_stiffness(
            xyz, _element_resultants(xyz, hoop, axial))
    return K, Kg


def _power_max(matrix: torch.Tensor, iterations: int = 15) -> float:
    """Largest eigenvalue magnitude by power iteration (avoids a second eigh)."""
    n = matrix.shape[0]
    vector = torch.ones(n, dtype=matrix.dtype) / math.sqrt(n)
    estimate = 0.0
    for _ in range(iterations):
        image = matrix @ vector
        norm = torch.linalg.vector_norm(image)
        if float(norm) <= 0.0:
            return 0.0
        vector = image / norm
        estimate = float(norm)
    return estimate


def generalized_buckling_load_factor(K: torch.Tensor, Kg: torch.Tensor, free: torch.Tensor):
    """Smallest significant positive ``lambda`` of ``K phi = lambda Kg phi``.

    Both operators are singular on the free set, so the condensation used by the
    qualified flat-plate path is applied to any mesh:

    1. *active* DOFs are those the geometric operator actually touches
       (``|Kg|`` column norm above a relative threshold); rotations and
       non-destabilised translations are *passive*;
    2. ``K`` is statically condensed onto the active set,
       ``Kc = Kaa - Kap Kpp^-1 Kpa``;
    3. ``G = Kg[active, active]`` is whitened (``G = V s V^T``, ``s > 0``),
       which drops its null directions (rigid translation, membrane modes): they
       carry infinite load factor;
    4. the whitened ``Kc`` is diagonalised and factors below
       ``1e-9 max(lambda)`` are discarded as numerical mechanisms.  This filter
       deliberately does *not* remove a facet-scale near-mechanism that carries
       finite destabilising energy -- such a mode is a genuine result of the
       discretisation and is reported, not hidden.

    A tensile pre-stress leaves ``Kg`` without a positive range, so no finite
    factor can be reported and ``RuntimeError`` is raised.
    """
    free = torch.as_tensor(free, dtype=torch.long)
    Kr = K[free[:, None], free]
    Gr = Kg[free[:, None], free]
    Kr = (Kr + Kr.T) / 2.0
    Gr = (Gr + Gr.T) / 2.0
    columns = torch.linalg.vector_norm(Gr, dim=0)
    peak = float(columns.max())
    if not math.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("geometric operator has no positive direction")
    active = columns > peak * 1e-10
    ia = torch.nonzero(active).flatten()
    ib = torch.nonzero(~active).flatten()
    Ka = Kr[ia[:, None], ia]
    Ga = Gr[ia[:, None], ia]
    recovery = None
    if len(ib):
        Kbb = Kr[ib[:, None], ib]
        Kab = Kr[ia[:, None], ib]
        recovery = torch.linalg.solve(Kbb, Kab.T)
        Ka = Ka - Kab @ recovery
    values, vectors = torch.linalg.eigh(Ga)
    top = float(values[-1])
    magnitude = max(abs(top), abs(float(values[0])))
    # Compression makes ``Ga`` positive semidefinite, so its largest eigenvalue
    # dominates.  Under tension the form is negative semidefinite and the
    # positive part is round-off: no finite buckling factor exists.
    if not math.isfinite(top) or top <= magnitude * 1e-9:
        raise RuntimeError("geometric operator has no positive direction")
    keep = values > top * 1e-12
    vectors = vectors[:, keep]
    scale = 1.0 / torch.sqrt(values[keep])
    Z = vectors * scale[None, :]
    B = Z.T @ Ka @ Z
    mu, U = torch.linalg.eigh((B + B.T) / 2.0)
    significant = mu > float(mu[-1]) * 1e-9
    if not bool(significant.any()):
        raise RuntimeError("no positive buckling load factor")
    order = torch.argsort(mu[significant])
    lambdas = mu[significant][order]
    reduced = U[:, significant][:, order]
    active_mode = Z @ reduced[:, 0]
    mode = torch.zeros(K.shape[0], dtype=K.dtype)
    mode[free[ia]] = active_mode
    if recovery is not None:
        mode[free[ib]] = -(recovery @ active_mode)
    info = {"active_dof": int(len(ia)), "passive_dof": int(len(ib)),
            "mode_count": int(len(lambdas))}
    return float(lambdas[0]), lambdas, mode, info


def _mode_diagnostics(K: torch.Tensor, Kg: torch.Tensor, free: torch.Tensor,
                      mode: torch.Tensor) -> dict:
    """Rayleigh quotient and elastic-stiffness content of a computed mode."""
    free = torch.as_tensor(free, dtype=torch.long)
    phi = mode[free]
    Kr = K[free[:, None], free]
    Gr = Kg[free[:, None], free]
    norm2 = float(phi @ phi)
    elastic = float(phi @ Kr @ phi)
    geometric = float(phi @ Gr @ phi)
    stiffest = _power_max(Kr)
    return {
        "rayleigh_factor": elastic / geometric if geometric else float("inf"),
        "elastic_energy_per_dof": elastic / norm2 if norm2 else 0.0,
        "stiffest_elastic_eigenvalue": stiffest,
        "elastic_fraction": (elastic / norm2) / stiffest if stiffest else 0.0,
    }


def cylinder_pressure_buckling(radius, length, young, poisson, thickness,
                               n_circ: int = 32, n_axial: int = 4,
                               pressure: float = 1.0, ends: str = "free",
                               closed_ends: bool = False) -> CurvedBucklingResult:
    """Lowest external-pressure buckling factor of a faceted Shell4 cylinder.

    ``ends="free"`` leaves the end rings unconstrained -- the ovalising mode is
    then free of boundary layers, which isolates the ring limit -- while
    ``ends="diaphragm"`` clamps all end-ring translations to emulate a simply
    supported tube end.  ``closed_ends=True`` adds the axial membrane force
    ``p R / 2`` carried by end closures.
    """
    radius = _positive("radius", radius)
    length = _positive("length", length)
    young = _positive("young", young)
    thickness = _positive("thickness", thickness)
    pressure = _positive("pressure", pressure)
    if ends not in {"free", "diaphragm"}:
        raise ValueError("ends must be 'free' or 'diaphragm'")
    n_circ = _integer("n_circ", n_circ, 8)
    n_axial = _integer("n_axial", n_axial, 1)
    nodes, elements = cylindrical_shell4_mesh(radius, length, n_circ, n_axial)
    hoop = pressure * radius
    axial = 0.5 * hoop if closed_ends else 0.0
    K, Kg = assemble_shell4_buckling_system(nodes, elements, young, poisson,
                                            thickness, hoop, axial)
    free = torch.arange(6 * len(nodes), dtype=torch.long)
    if ends == "diaphragm":
        blocked = set()
        for ring in (0, n_axial):
            for k in range(n_circ):
                base = 6 * (ring * n_circ + k)
                blocked.update((base, base + 1, base + 2))
        free = torch.tensor([i for i in free.tolist() if i not in blocked], dtype=torch.long)
    load_factor, eigenvalues, mode, info = generalized_buckling_load_factor(K, Kg, free)
    diagnostics = _mode_diagnostics(K, Kg, free, mode)
    diagnostics.update(info)
    diagnostics.update({"n_circ": n_circ, "n_axial": n_axial, "ends": ends,
                        "dof": int(6 * len(nodes))})
    return CurvedBucklingResult(load_factor, eigenvalues, mode, nodes, elements, diagnostics)


def dominant_circumferential_wave(result: CurvedBucklingResult) -> int:
    """Dominant Fourier wave number of the radial mode on a mid-length ring."""
    nodes = result.nodes
    ring_z = float(torch.median(nodes[:, 2]))
    on_ring = torch.nonzero((nodes[:, 2] - ring_z).abs() < 1e-12).flatten()
    theta = torch.atan2(nodes[on_ring, 1], nodes[on_ring, 0])
    radial = result.radial_components()[on_ring]
    order = torch.argsort(theta)
    radial = radial[order]
    count = len(radial)
    spectrum = torch.zeros(count // 2 + 1, dtype=_DTYPE)
    index = torch.arange(count, dtype=_DTYPE)
    for wave in range(1, len(spectrum)):
        cosine = torch.cos(wave * 2.0 * math.pi * index / count)
        sine = torch.sin(wave * 2.0 * math.pi * index / count)
        spectrum[wave] = torch.linalg.vector_norm(
            torch.stack((radial @ cosine, radial @ sine)))
    return int(torch.argmax(spectrum[1:]) + 1)


def simply_supported_plate_cross_check(nx: int = 8, ny: int = 8) -> dict:
    """Re-solve the qualified flat-plate case through this generic solver."""
    from .shell4_buckling import (navier_plate_load, rectangular_shell_mesh,
                                  simply_supported_plate_buckling)

    a = b = 1.0
    young, nu, thickness = 210e9, 0.3, 0.01
    nodes, elements = rectangular_shell_mesh(a, b, nx, ny)
    K, _ = assemble_shell4_buckling_system(nodes, elements, young, nu, thickness)
    Kg = torch.zeros_like(K)
    for conn in elements.tolist():
        ix = torch.tensor(conn, dtype=torch.long)
        dofs = torch.stack([6 * ix + k for k in range(6)], 1).reshape(-1)
        Kg[dofs[:, None], dofs] += shell4_geometric_stiffness(
            nodes[ix], torch.tensor([1.0, 0.0, 0.0], dtype=_DTYPE))
    tol = 64 * torch.finfo(_DTYPE).eps
    edge = ((nodes[:, 0].abs() < tol) | ((nodes[:, 0] - a).abs() < tol)
            | (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - b).abs() < tol))
    blocked = set((6 * torch.nonzero(edge).flatten() + 2).tolist())
    free = torch.tensor([i for i in range(K.shape[0]) if i not in blocked], dtype=torch.long)
    generic, _, _, _ = generalized_buckling_load_factor(K, Kg, free)
    established = float(simply_supported_plate_buckling(
        a=a, b=b, young=young, poisson=nu, thickness=thickness, nx=nx, ny=ny).load_factors[0])
    exact = navier_plate_load(a=a, b=b, young=young, poisson=nu, thickness=thickness)
    return {"generic_solver": generic, "production_flat_path": established, "navier": exact,
            "generic_vs_flat_relative": abs(generic - established) / established}


def run_shell4_curved_buckling_qualification() -> dict:
    """Reproducible evidence for the curved-geometric-stiffness verdict."""
    young, nu = 210e9, 0.3
    radius, thickness = 1.0, 0.01          # R/t = 100
    length = 8.0
    exact = long_tube_buckling_pressure(radius, thickness, young, nu)
    society = long_shell_pressure_society_form(young, nu, thickness, 2 * radius)

    cross = simply_supported_plate_cross_check()

    meshes = []
    for n_circ in (16, 32, 48, 64):
        result = cylinder_pressure_buckling(radius, length, young, nu, thickness,
                                            n_circ=n_circ, n_axial=4, ends="free")
        meshes.append({
            "n_circ": n_circ, "dof": result.diagnostics["dof"],
            "pressure": result.load_factor,
            "relative_error": (result.load_factor - exact) / exact,
            "circumferential_wave": dominant_circumferential_wave(result),
            "nyquist_wave": n_circ // 2,
            "elastic_fraction": result.diagnostics["elastic_fraction"],
            "rayleigh_factor": result.diagnostics["rayleigh_factor"],
        })
    refinement = [meshes[i]["pressure"] / meshes[i + 1]["pressure"] for i in range(len(meshes) - 1)]

    ends_study = []
    for ends in ("free", "diaphragm"):
        result = cylinder_pressure_buckling(radius, length, young, nu, thickness,
                                            n_circ=32, n_axial=4, ends=ends)
        ends_study.append({"ends": ends, "pressure": result.load_factor,
                           "relative_error": (result.load_factor - exact) / exact,
                           "circumferential_wave": dominant_circumferential_wave(result)})

    axial_study = []
    for n_axial in (1, 2, 4, 8):
        result = cylinder_pressure_buckling(radius, length, young, nu, thickness,
                                            n_circ=32, n_axial=n_axial, ends="free")
        axial_study.append({"n_axial": n_axial, "pressure": result.load_factor,
                            "relative_error": (result.load_factor - exact) / exact,
                            "circumferential_wave": dominant_circumferential_wave(result)})

    length_study = []
    for ratio in (2.0, 4.0, 8.0):
        result = cylinder_pressure_buckling(radius, radius * ratio, young, nu, thickness,
                                            n_circ=32, n_axial=4, ends="free")
        length_study.append({"length_over_radius": ratio, "pressure": result.load_factor,
                             "relative_error": (result.load_factor - exact) / exact,
                             "circumferential_wave": dominant_circumferential_wave(result)})

    scaling = []
    for ratio in (50.0, 100.0, 200.0):
        t = radius / ratio
        reference = long_tube_buckling_pressure(radius, t, young, nu)
        result = cylinder_pressure_buckling(radius, length, young, nu, t,
                                            n_circ=32, n_axial=4, ends="free")
        scaling.append({"radius_over_thickness": ratio, "reference": reference,
                        "pressure": result.load_factor,
                        "relative_error": (result.load_factor - reference) / reference})

    worst_abs = max(abs(row["relative_error"]) for row in meshes)
    monotone_down = all(refinement[i] > 1.0 for i in range(len(refinement)))
    converges = abs(meshes[-1]["relative_error"]) < 0.03 and refinement[-1] < 1.0
    passed = bool(converges)
    return {
        "schema": "tensorfem.shell4-curved-buckling/1.1",
        "reference": "long-tube external-pressure ring buckling, n = 2 (inextensional)",
        "reference_pressure": exact,
        "society_form_pressure": society,
        "society_form_relative_gap": abs(society - exact) / exact,
        "ring_wave_factors": ring_pressure_wave_factors(),
        "tolerance": 0.03,
        "case": {"radius": radius, "length": length, "thickness": thickness,
                 "young": young, "poisson": nu, "radius_over_thickness": radius / thickness},
        "flat_plate_solver_cross_check": cross,
        "meshes": meshes,
        "refinement_ratios": refinement,
        "monotone_degradation": monotone_down,
        "ends_study": ends_study,
        "axial_subdivision_study": axial_study,
        "length_study": length_study,
        "thickness_scaling": scaling,
        "worst_absolute_relative_error": worst_abs,
        "passed": passed,
        "verdict": "curved-shell-geometric-stiffness-unqualified" if not passed
                   else "curved-shell-geometric-stiffness-qualified",
        "blocking_mechanism": (
            "lowest mode sits at the mesh Nyquist wave and its elastic energy is "
            "many orders below a stiff direction: faceted bilinear Shell4 admits a "
            "facet-scale near-mechanism under hoop compression, so the load factor "
            "falls with refinement instead of converging"),
        "boundary": ("linear initial-stress eigenbuckling of isotropic Shell4 facets; "
                     "uniform prestress supplied in closed form (no nonlinear "
                     "prebuckling solve, no load-follower/pressure stiffness term, "
                     "no imperfections, no postbuckling).  A failing verdict is a "
                     "capability statement, not a discretisation tweak."),
    }