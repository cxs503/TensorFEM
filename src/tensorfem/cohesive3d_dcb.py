"""Coupled 3-D double-cantilever-beam (DCB) delamination driver.

Purpose
-------
``tensorfem.cohesive3d`` verifies the *element* (traction-separation law,
24x24 consistent tangent, patch tests, rigid rotation, mesh objectivity) and
``benchmarks/cohesive3d_delamination.py`` grows a crack with **one element**
through the specimen width, i.e. a plane-stress-like idealisation.  This module
provides a reusable, genuinely three-dimensional driver: an arbitrary number of
HEX8 elements through the width, a pre-meshed zero-thickness COH3D8 ligament
(``tensorfem.cohesive3d``), displacement-controlled Newton iterations with
committed damage histories, and *independent* closed-form references taken from
standard DCB beam theory (including the Williams root-rotation correction used
by ASTM D5528) so that nothing is compared with itself.

Theory anchors (see ``docs/cohesive3d-dcb-3d-verification.md``)
---------------------------------------------------------------
* ASTM D5528 -- mode-I interlaminar fracture toughness of unidirectional
  composites, DCB specimen, corrected beam theory with root rotation.
* Williams (1989), *J. Strain Analysis* 24(4): end-rotation correction
  ``Delta = h sqrt((E / 11G) (3 - 2 (Gamma/(1+Gamma))**2))``,
  ``Gamma = 1.18 sqrt(E/G)``.
* Benzeggagh & Kenane (1996) -- mixed-mode propagation criterion (already in
  ``BKCohesive3DLaw``).
* Ortiz & Pandolfi (1999), Alfano & Crisfield (2001) -- intrinsic
  traction-separation cohesive element formulation with committed history.

Conventions
-----------
SI units (m, Pa, N, J/m^2).  ``x`` is the crack-growth direction, ``y`` the
specimen width, ``z`` the opening direction.  The specimen is clamped at
``x = 0``, loaded at ``x = L`` and the initial (pre-)crack extends from
``x = L - a0`` to the loaded end, so the cohesive ligament occupies
``0 <= x <= L - a0``.  Symmetric opening is prescribed at the loaded end
(``+u_z`` on the upper arm, ``-u_z`` on the lower arm) and the reported load
``P`` is the **total** load (sum of the two arm reactions), which makes

    ``C(a) = delta / P = 4 a**3 / (E b h**3) + a / (k G b h)``

the standard DCB compliance, and ``G_I = 6 P**2 a**2 / (E b**2 h**3)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .cohesive3d import BKCohesive3DLaw, cohesive_hex8_response, interface_area
from .solid3d import elasticity_matrix_3d, hex8_stiffness

__all__ = [
    "DCBGeometry",
    "DCBModel",
    "DCBResult",
    "build_dcb_model",
    "solve_dcb",
    "williams_root_rotation",
    "beam_compliance",
    "beam_peak_load",
    "beam_serr",
    "effective_modulus",
]

D = torch.float64
_AR3 = np.arange(3, dtype=np.int64)

#: shear correction factor of a rectangular section (Timoshenko beam).
SHEAR_FACTOR = 5.0 / 6.0


# --------------------------------------------------------------------------- #
# geometry / mesh
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DCBGeometry:
    """Specimen geometry, adherend material and mesh density.

    ``nx`` elements along the length, ``ny`` across the width, ``nz`` per arm
    through the thickness.  ``initial_crack`` is measured from the loaded end
    ``x = L``.  ``width`` / ``arm_thickness`` control the 3-D stress state:
    ``b/h ~ 1`` is the narrow-beam (plane-stress) limit, ``b/h >= 6`` approaches
    the wide-specimen (plane-strain) limit.
    """

    length: float = 80.0e-3
    width: float = 3.0e-3
    arm_thickness: float = 3.0e-3
    initial_crack: float = 50.0e-3
    young_modulus: float = 70.0e9
    poisson_ratio: float = 0.30
    nx: int = 160
    ny: int = 1
    nz: int = 4
    width_configuration: str = "free"

    def __post_init__(self) -> None:
        if not (0.0 < self.initial_crack < self.length):
            raise ValueError("initial_crack must lie inside (0, length)")
        for name in ("nx", "ny", "nz"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.arm_thickness <= 0.0 or self.width <= 0.0:
            raise ValueError("width and arm_thickness must be positive")
        if self.width_configuration not in ("free", "roller"):
            raise ValueError("width_configuration must be 'free' or 'roller'")

    # -- derived -----------------------------------------------------------
    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def bonded_length(self) -> float:
        return self.length - self.initial_crack

    @property
    def n_bonded(self) -> int:
        """Number of element columns spanned by the cohesive ligament."""
        return max(1, int(round(self.bonded_length / self.dx)))

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))


@dataclass
class DCBModel:
    """Two HEX8 adherends plus a pre-meshed cohesive ligament on ``z = 0``."""

    geometry: DCBGeometry
    law: BKCohesive3DLaw
    nodes: torch.Tensor
    arms: torch.Tensor
    iface: torch.Tensor
    n_lower: int
    fixed_dofs: np.ndarray
    upper_load_dofs: np.ndarray
    lower_load_dofs: np.ndarray
    _solid_stiffness: sp.csr_matrix | None = field(default=None, repr=False)
    _frames: dict | None = field(default=None, repr=False)

    # -- layout helpers ----------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes

    @property
    def n_iface(self) -> int:
        return int(self.iface.shape[0])

    @property
    def free_dofs(self) -> np.ndarray:
        mask = np.ones(self.n_dofs, dtype=bool)
        mask[self.fixed_dofs] = False
        return np.flatnonzero(mask)

    def element_dofs(self, connectivity: torch.Tensor) -> np.ndarray:
        return (3 * connectivity.numpy()[:, :, None] + _AR3).reshape(
            connectivity.shape[0], -1
        )

    # -- assembly ----------------------------------------------------------
    def solid_stiffness(self) -> sp.csr_matrix:
        if self._solid_stiffness is None:
            g = self.geometry
            d = elasticity_matrix_3d(
                torch.tensor([g.young_modulus], dtype=D),
                torch.tensor([g.poisson_ratio], dtype=D),
            )
            ke, _ = hex8_stiffness(self.nodes[self.arms], d)
            dofs = self.element_dofs(self.arms)
            ne = dofs.shape[0]
            rows = np.repeat(dofs, dofs.shape[1], axis=1).reshape(-1)
            cols = np.tile(dofs, (1, dofs.shape[1])).reshape(-1)
            vals = ke.numpy().reshape(-1)
            k = sp.coo_matrix(
                (vals, (rows, cols)), shape=(self.n_dofs, self.n_dofs)
            ).tocsr()
            k.sum_duplicates()
            self._solid_stiffness = k
        return self._solid_stiffness

    # -- interface assembly ------------------------------------------------
    def reference_frames(self):
        """Cache the (identical) reference frame of every interface element.

        The structured ligament is translation invariant, so every COH3D8 has
        the same reference mid-surface frame, ``B`` matrix and Gauss area.
        Computing them once from the first element and checking that all
        elements share the same area is what makes the vectorised assembly
        below exact rather than approximate; the equivalence with the
        per-element :func:`cohesive_hex8_response` loop is asserted in
        ``tests/test_cohesive3d_dcb.py``.
        """
        if self._frames is None:
            from .cohesive3d import mid_surface_frame

            fr, weights, _ = mid_surface_frame(self.nodes[self.iface[0]])
            areas = np.array(
                [float(interface_area(self.nodes[c])) for c in self.iface]
            )
            ref_area = areas[0]
            if np.max(np.abs(areas - ref_area)) > 1e-12 * ref_area:
                raise ValueError(
                    "interface elements are not congruent; the vectorised "
                    "assembly requires one uniform ligament patch"
                )
            rotations, shapes, bmats, garea = [], [], [], []
            for (normal, e1, e2, jac, n), w in zip(fr, weights):
                rotation = torch.stack((normal, e1, e2))
                bm = torch.zeros((3, 24), dtype=D)
                for a in range(4):
                    block = n[a] * rotation
                    bm[:, 3 * a : 3 * a + 3] = -block
                    bm[:, 3 * (a + 4) : 3 * (a + 4) + 3] = block
                rotations.append(rotation)
                shapes.append(n)
                bmats.append(bm)
                garea.append(float(w * jac))
            self._frames = {
                "rotation": torch.stack(rotations),
                "shape": torch.stack(shapes),
                "b": torch.stack(bmats),
                "area": np.array(garea),
                "element_area": areas,
            }
        return self._frames

    def interface_response(self, u: np.ndarray, history: torch.Tensor):
        """Assemble the cohesive internal force and tangent at ``u``.

        Returns ``(f, K, history_new, diagnostics)`` with ``f`` a ``(n_dofs,)``
        numpy vector, ``K`` a ``(n_dofs, n_dofs)`` csr matrix, ``history_new`` a
        ``(n_iface, 4)`` tensor and ``diagnostics`` a dict of per-Gauss-point
        arrays (``separation``, ``traction``, ``damage``, ``area``).
        """
        fr = self.reference_frames()
        uu = torch.as_tensor(u, dtype=D).reshape(self.n_nodes, 3)
        conn = self.iface
        jump = uu[conn[:, 4:]] - uu[conn[:, :4]]                 # (ne, 4, 3)
        rot = fr["rotation"]                                     # (ngp, 3, 3)
        local = torch.einsum("eac,gdc->egad", jump, rot)         # (ne, ngp, 4, 3)
        sep = torch.einsum("ga,egad->egd", fr["shape"], local)   # (ne, ngp, 3)
        hist = history if history is not None else torch.zeros(
            (self.n_iface, 4), dtype=D
        )
        resp = self.law.update(sep.reshape(-1, 3), hist.reshape(-1))
        traction = resp.traction.reshape(self.n_iface, 4, 3)
        tangent = resp.tangent.reshape(self.n_iface, 4, 3, 3)
        history_new = resp.history.reshape(self.n_iface, 4)
        damage = resp.damage.reshape(self.n_iface, 4)
        bmat = fr["b"]                                           # (ngp, 3, 24)
        area = torch.as_tensor(fr["area"], dtype=D)              # (ngp,)
        f_e = torch.einsum("gki,egk,g->ei", bmat, traction, area)
        k_e = torch.einsum("gki,egkl,glj,g->eij", bmat, tangent, bmat, area)
        dofs = self.element_dofs(conn)                           # (ne, 24)
        n_dofs, ne = self.n_dofs, dofs.shape[0]
        f = np.zeros(n_dofs)
        np.add.at(f, dofs.reshape(-1), f_e.reshape(-1))
        rows = np.repeat(dofs, 24, axis=1).reshape(-1)
        cols = np.tile(dofs, (1, 24)).reshape(-1)
        K = sp.coo_matrix(
            (k_e.numpy().reshape(-1), (rows, cols)), shape=(n_dofs, n_dofs)
        ).tocsr()
        K.sum_duplicates()
        diagnostics = {
            "separation": sep.detach().numpy(),
            "traction": traction.detach().numpy(),
            "damage": damage.detach().numpy(),
            "area": np.broadcast_to(fr["area"], (self.n_iface, 4)),
        }
        return f, K, history_new, diagnostics


def build_dcb_model(
    geometry: DCBGeometry | None = None,
    law: BKCohesive3DLaw | None = None,
) -> DCBModel:
    """Build the two-arm HEX8 model with duplicated interface node layers."""
    g = geometry or DCBGeometry()
    law = law or BKCohesive3DLaw(
        normal_stiffness=2.4e14,
        shear_stiffness=1.0e14,
        normal_strength=120.0e6,
        shear_strength=80.0e6,
        mode_i_energy=300.0,
        mode_ii_energy=800.0,
        bk_eta=2.2,
    )
    nx, ny, nz = g.nx, g.ny, g.nz
    xs = np.linspace(0.0, g.length, nx + 1)
    ys = np.linspace(0.0, g.width, ny + 1)
    zl = np.linspace(-g.arm_thickness, 0.0, nz + 1)  # lower arm, top layer = 0
    zu = np.linspace(0.0, g.arm_thickness, nz + 1)  # upper arm, bottom layer = 0

    def block(zs: np.ndarray) -> np.ndarray:
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)

    lower = block(zl)
    upper = block(zu)
    n_lower = lower.shape[0]
    nodes = torch.tensor(np.vstack((lower, upper)), dtype=D)

    def idx(i, j, k, base=0, nz_block=nz):
        return base + (i * (ny + 1) + j) * (nz_block + 1) + k

    def hexes(base: int) -> np.ndarray:
        cells = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    cells.append(
                        (
                            idx(i, j, k, base),
                            idx(i + 1, j, k, base),
                            idx(i + 1, j + 1, k, base),
                            idx(i, j + 1, k, base),
                            idx(i, j, k + 1, base),
                            idx(i + 1, j, k + 1, base),
                            idx(i + 1, j + 1, k + 1, base),
                            idx(i, j + 1, k + 1, base),
                        )
                    )
        return np.array(cells, dtype=np.int64)

    arms = np.vstack((hexes(0), hexes(n_lower)))

    iface = []
    for i in range(g.n_bonded):
        for j in range(ny):
            iface.append(
                (
                    idx(i, j, nz, 0),
                    idx(i + 1, j, nz, 0),
                    idx(i + 1, j + 1, nz, 0),
                    idx(i, j + 1, nz, 0),
                    idx(i, j, 0, n_lower),
                    idx(i + 1, j, 0, n_lower),
                    idx(i + 1, j + 1, 0, n_lower),
                    idx(i, j + 1, 0, n_lower),
                )
            )
    iface_arr = np.array(iface, dtype=np.int64)

    # boundary conditions: clamp the whole x = 0 face of both arms
    clamp = nodes[:, 0] == 0.0
    fixed = (3 * clamp.numpy()[:, None] + _AR3).reshape(-1)
    fixed = np.unique(np.concatenate((fixed, np.zeros(0, dtype=np.int64))))

    loaded = (nodes[:, 0] == g.length).nonzero().reshape(-1).numpy()
    low = loaded[loaded < n_lower]
    up = loaded[loaded >= n_lower]
    lower_load = (3 * low[:, None] + 2).reshape(-1)
    upper_load = (3 * up[:, None] + 2).reshape(-1)
    if g.width_configuration == "roller":  # plane-strain-like: no lateral motion
        lat = np.concatenate([3 * loaded + 1, 3 * loaded])
        fixed = np.unique(np.concatenate((fixed, lat)))

    return DCBModel(
        geometry=g,
        law=law,
        nodes=nodes,
        arms=torch.tensor(arms, dtype=torch.long),
        iface=torch.tensor(iface_arr, dtype=torch.long),
        n_lower=n_lower,
        fixed_dofs=fixed.astype(np.int64),
        upper_load_dofs=upper_load.astype(np.int64),
        lower_load_dofs=lower_load.astype(np.int64),
    )


# --------------------------------------------------------------------------- #
# closed-form references (independent of the FE model)
# --------------------------------------------------------------------------- #
def williams_root_rotation(
    arm_thickness: float,
    young_modulus: float,
    shear_modulus: float,
) -> float:
    """Williams (1989) end-rotation correction ``Delta`` [m] of a DCB arm."""
    ratio = math.sqrt(young_modulus / shear_modulus)
    gamma = 1.18 * ratio
    return arm_thickness * math.sqrt(
        (young_modulus / (11.0 * shear_modulus))
        * (3.0 - 2.0 * (gamma / (1.0 + gamma)) ** 2)
    )


def effective_modulus(geometry: DCBGeometry) -> float:
    """Bending modulus for the *narrow* (plane-stress) DCB beam.

    A narrow specimen (``b/h ~ 1``) has free lateral contraction and is
    described by ``E``; for wide specimens the anticlastic constraint raises the
    effective stiffness towards ``E/(1-nu^2)`` and the caller is expected to
    ask for :meth:`DCBGeometry`'s plate modulus explicitly.
    """
    return geometry.young_modulus


def beam_compliance(
    crack: float,
    geometry: DCBGeometry | None = None,
    *,
    modulus: float | None = None,
    root_rotation: bool = True,
    shear: bool = True,
) -> float:
    """DCB compliance ``delta/P`` for the **total** load.

    ``C = 4 a^3 / (E b h^3) + a / (k G b h)`` with ``a -> a + Delta`` when the
    Williams root rotation is requested.
    """
    g = geometry or DCBGeometry()
    e = modulus if modulus is not None else effective_modulus(g)
    h, b = g.arm_thickness, g.width
    a = crack + (
        williams_root_rotation(h, e, g.shear_modulus) if root_rotation else 0.0
    )
    c = 4.0 * a**3 / (e * b * h**3)
    if shear:
        c += a / (SHEAR_FACTOR * g.shear_modulus * b * h)
    return c


def beam_serr(
    load: float,
    crack: float,
    geometry: DCBGeometry | None = None,
    *,
    modulus: float | None = None,
    root_rotation: bool = True,
) -> float:
    """Mode-I strain energy release rate ``G_I`` for the total load ``P``."""
    g = geometry or DCBGeometry()
    e = modulus if modulus is not None else effective_modulus(g)
    h, b = g.arm_thickness, g.width
    a = crack + (
        williams_root_rotation(h, e, g.shear_modulus) if root_rotation else 0.0
    )
    return 6.0 * load**2 * a**2 / (e * b**2 * h**3)


def beam_peak_load(
    crack: float,
    toughness: float,
    geometry: DCBGeometry | None = None,
    *,
    modulus: float | None = None,
    root_rotation: bool = True,
) -> float:
    """``P`` at which ``G_I == toughness`` (LEFM limit of the CZM peak)."""
    g = geometry or DCBGeometry()
    e = modulus if modulus is not None else effective_modulus(g)
    h, b = g.arm_thickness, g.width
    a = crack + (
        williams_root_rotation(h, e, g.shear_modulus) if root_rotation else 0.0
    )
    return b * h**1.5 * math.sqrt(toughness * e / 6.0) / a


# --------------------------------------------------------------------------- #
# nonlinear solution
# --------------------------------------------------------------------------- #
@dataclass
class DCBResult:
    """History of a displacement-controlled DCB simulation."""

    curve: list[dict]
    damage: np.ndarray          # (n_iface,) committed mean damage
    stats: dict
    model: DCBModel
    displacement: np.ndarray    # (n_dofs,) committed displacement
    history: torch.Tensor       # (n_iface, 4)
    iface_x: np.ndarray         # (n_iface,) element centroid x
    iface_area: np.ndarray      # (n_iface,) element area

    @property
    def peak(self) -> dict:
        return max(self.curve, key=lambda c: c["load"]) if self.curve else {}


def _reaction(fint: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> float:
    """Total load ``P`` from the arm reactions at the loaded end.

    The prescribed opening is symmetric (``+delta/2`` upper, ``-delta/2``
    lower), so each arm carries ``P/2`` and the total load is the sum of the two
    reaction magnitudes.
    """
    return abs(float(fint[upper].sum())) + abs(float(fint[lower].sum()))


def solve_dcb(
    model: DCBModel,
    *,
    delta_max: float,
    n_steps: int = 80,
    max_iter: int = 40,
    tol: float = 1e-9,
    min_step_ratio: float = 1e-4,
    verbose: bool = False,
) -> DCBResult:
    """Displacement-controlled Newton path follower with committed histories.

    ``delta_max`` is the *total* opening (upper ``+delta/2``, lower
    ``-delta/2``).  Steps are halved whenever Newton fails and grown by 20 %
    after an easy convergence, so the damage front can advance through the
    load peak without manual tuning.
    """
    g = model.geometry
    K_solid = model.solid_stiffness()
    n = model.n_dofs
    free = model.free_dofs
    u = np.zeros(n)

    faced = np.concatenate((model.upper_load_dofs, model.lower_load_dofs))
    signs = np.concatenate(
        (
            np.full(model.upper_load_dofs.size, 0.5 / model.upper_load_dofs.size),
            np.full(model.lower_load_dofs.size, -0.5 / model.lower_load_dofs.size),
        )
    )
    prescription = np.zeros(n)
    prescription[faced] = signs

    # interface geometry (for area weighting and the crack-front read-out)
    iface_x = model.nodes[model.iface][:, :, 0].mean(dim=1).numpy()
    areas = []
    for e in range(model.n_iface):
        areas.append(float(interface_area(model.nodes[model.iface[e]])))
    iface_area = np.array(areas)

    history = torch.zeros((model.n_iface, 4), dtype=D)
    curve: list[dict] = []
    committed_damage = np.zeros(model.n_iface)
    scale = 1.0
    delta, step = 0.0, delta_max / max(n_steps, 1)
    work, diss_path, failed_area = 0.0, 0.0, 0.0
    sep_prev = tra_prev = None
    converged_all = True
    iterations_total = 0
    steps_taken = 0

    while delta < delta_max * (1.0 + 1e-12):
        target = min(delta + step, delta_max)
        trial = u + (target - delta) * prescription
        ok, its, history_new, diag, fint = _newton(
            model, K_solid, trial, history, free, max_iter, tol, scale
        )
        iterations_total += its
        if not ok:
            step *= 0.5
            if step < min_step_ratio * delta_max / max(n_steps, 1):
                converged_all = False
                break
            continue
        # accept
        prev_delta = delta
        delta = target
        u = trial
        history = history_new
        steps_taken += 1
        load = _reaction(fint, model.upper_load_dofs, model.lower_load_dofs)
        scale = max(scale, load)
        if not curve:
            work += 0.0
        else:
            work += 0.5 * (curve[-1]["load"] + load) * (delta - prev_delta)
        committed_damage = diag["damage"].mean(axis=1)
        sep_now = diag["separation"]
        tra_now = diag["traction"]
        area_gp = diag["area"]
        if sep_prev is not None:
            diss_path += float(
                np.sum(
                    0.5
                    * (tra_prev + tra_now)
                    * (sep_now - sep_prev)
                    * area_gp[:, :, None]
                )
            )
        sep_prev, tra_prev = sep_now, tra_now
        stuck = committed_damage >= 0.9
        failed_area = float(iface_area[stuck].sum())
        strain_energy = 0.5 * float(u @ (K_solid @ u))
        curve.append(
            {
                "step": steps_taken,
                "disp": float(delta),
                "load": float(load),
                "max_damage": float(committed_damage.max()) if model.n_iface else 0.0,
                "crack_front_m": _crack_front(iface_x, committed_damage, g),
                "strain_energy": strain_energy,
                "external_work": work,
                "interface_work": diss_path,
                "failed_area": failed_area,
                "newton_iterations": its,
            }
        )
        if verbose:
            print(
                f"step {steps_taken:4d} disp={delta:.4e} load={load:.3f} "
                f"Dmax={curve[-1]['max_damage']:.3f} it={its}"
            )
        if step < delta_max / max(n_steps, 1):
            step = min(step * 1.2, delta_max / max(n_steps, 1))

    stats = {
        "converged_all": bool(converged_all),
        "steps": steps_taken,
        "newton_iterations": iterations_total,
        "n_dofs": n,
        "n_iface": model.n_iface,
        "reach_delta": float(delta),
        "final_strain_energy": 0.5 * float(u @ (K_solid @ u)),
        "final_external_work": work,
        "final_interface_work": diss_path,
        "final_failed_area": failed_area,
        "interface_work_per_area": (
            diss_path / failed_area if failed_area > 0.0 else float("nan")
        ),
        "energy_balance_rel": (
            abs(work - 0.5 * float(u @ (K_solid @ u)) - diss_path) / work
            if work > 0.0
            else float("nan")
        ),
    }
    return DCBResult(
        curve=curve,
        damage=committed_damage,
        stats=stats,
        model=model,
        displacement=u,
        history=history,
        iface_x=iface_x,
        iface_area=iface_area,
    )


def _crack_front(iface_x: np.ndarray, damage: np.ndarray, g: DCBGeometry) -> float:
    """Crack length from the loaded end, from the 0.9-damage front."""
    if damage.size == 0:
        return g.initial_crack
    failed = iface_x[damage >= 0.9]
    if failed.size == 0:
        return g.initial_crack
    return float(g.length - failed.min())


def _newton(model, K_solid, u, history, free, max_iter, tol, scale):
    """Newton iterations on the free dofs; returns ``(ok, its, hist, diag, fint)``."""
    n = model.n_dofs
    for it in range(1, max_iter + 1):
        f_coh, K_coh, hist_new, diag = model.interface_response(u, history)
        fint = K_solid @ u + f_coh
        residual = fint[free]
        norm = float(np.linalg.norm(residual, np.inf))
        if norm <= tol * max(scale, 1.0):
            return True, it, hist_new, diag, fint
        K = (K_solid + K_coh)[free][:, free].tocsc()
        try:
            du = spla.splu(K).solve(-residual)
        except RuntimeError:  # singular factorisation -> treat as divergence
            return False, it, history, diag, fint
        if not np.all(np.isfinite(du)):
            return False, it, history, diag, fint
        u[free] += du
    return False, max_iter, history, diag, fint