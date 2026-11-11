#!/usr/bin/env python
"""Specimen-level verification of the 3-D cohesive element: DCB / ENF / MMB.

Run ``python benchmarks/cohesive3d_delamination.py`` (``--quick`` for a fast
smoke run); writes ``benchmarks/cohesive3d_delamination.json`` and figures under
``docs/assets/cohesive3d-delamination/``.

Why this benchmark exists
-------------------------
``benchmarks/cohesive3d_verification.py`` verifies the *element*: the 3-D
traction-separation law, the 24x24 consistent tangent, patch tests, rigid-body
objectivity, mesh objectivity and a two-block series-spring assembly.  What it
does not do is grow a crack: no check there shows that the element reproduces
the *global* fracture behaviour of a delaminating specimen.  This script closes
that gap with three standard specimens built from the production HEX8 solid
(``tensorfem.solid3d.hex8_stiffness``) and the production cohesive element
(``tensorfem.cohesive3d.cohesive_hex8_response``):

==================  =========================================================
``dcb_mode_i``      double cantilever beam (ASTM D5528 idealisation), mode I:
                    ``C = 4a^3/(E b h^3)``, ``G_I = 6P^2 a^2/(E b^2 h^3)``
``enf_mode_ii``     end-notched flexure (ASTM D7905), mode II, crack tip beyond
                    midspan: ``C = (2L^3+3a^3)/(96 E I)``,
                    ``G_II = 9P^2 a^2/(16 E b^2 h^3)``
``mmb_mixed``       mixed-mode bending (ASTM D6671 idealisation): lever + midspan
                    load, Benzeggagh-Kenane criterion on VCCT-resolved modes
``mmb_pure_mode_ii``lever arm ``c = L/3`` must give a pure mode II state (rig
                    statics self-consistency: no literature constant needed)
==================  =========================================================

Three independent numerical routes are used so that no check is circular:

1. linear compliance of the intact specimen vs the beam closed form (validates
   mesh, boundary conditions and the cohesive penalty stiffness);
2. VCCT (Rybicki-Kanninen virtual crack closure) on the linear solution:
   mode-resolved ``G_I``/``G_II`` at the crack front vs the closed forms;
3. cohesive-zone simulation: displacement/force controlled Newton solve with a
   growing damage front; the peak (propagation) load is compared with the load
   predicted by the closed-form ``G_c``, i.e. the computed energy release rate at
   debonding is compared with ``G_c``.

Acceptance (mission): ``|G_sim - G_c| / G_c < 5 %``.

Conventions: SI units (m, Pa, N, J/m^2), small strain, fully integrated HEX8
adherends, one element through the width (plane-stress-like).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.cohesive3d import BKCohesive3DLaw, cohesive_hex8_response  # noqa: E402
from tensorfem.solid3d import (  # noqa: E402
    elasticity_matrix_3d,
    hex8_stiffness,
    structured_hex_mesh,
)

D = torch.float64
torch.set_default_dtype(D)
torch.set_num_threads(1)  # tiny tensors: thread pool overhead dominates
AR3 = torch.arange(3, dtype=torch.long)

# --------------------------------------------------------------------------- #
# Specimen / interface properties
# --------------------------------------------------------------------------- #
LX = 150.0e-3        # specimen length
WIDTH = 12.0e-3      # specimen width (single element)
H_ARM = 3.0e-3       # adherend thickness
E_ADH = 70.0e9       # adherend modulus (aluminium)
NU_ADH = 0.33

KN = 1.0e16          # penalty normal stiffness [Pa/m]
KS = 4.0e15          # penalty shear stiffness [Pa/m]
TN0 = 10.0e6         # normal strength [Pa] (cohesive length ~4.7 mm)
TS0 = 7.0e6          # shear strength [Pa]
GIC = 300.0          # mode I toughness [J/m^2]
GIIC = 800.0         # mode II toughness [J/m^2]
BK_ETA = 2.2

MESH = {"nx": 100, "nz": 3}     # 1.5 mm along x, 1.0 mm per arm
CRACK_DCB = 45.0e-3             # debond length from the loaded mouth
CRACK_ENF = 105.0e-3            # debond length from the left support (> L)
LEVER_RATIO = 0.6               # MMB lever arm c / L


def make_law(tn0=TN0, ts0=TS0, gic=GIC, giic=GIIC) -> BKCohesive3DLaw:
    return BKCohesive3DLaw(KN, KS, tn0, ts0, gic, giic, BK_ETA)


def bk_toughness(g1: float, g2: float, gic=GIC, giic=GIIC, eta=BK_ETA) -> float:
    """Benzeggagh-Kenane mixed-mode toughness."""
    g = g1 + g2
    if g <= 0.0:
        return gic
    return gic + (giic - gic) * (g2 / g) ** eta


def onset_load(unit_g_total: float, g1: float, g2: float) -> float:
    """Load P at which the BK criterion is met, from linear (unit-load) SERR."""
    g_c = bk_toughness(g1, g2)
    return math.sqrt(g_c / unit_g_total)


# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #
@dataclass
class Specimen:
    """Two HEX8 adherends plus a pre-meshed cohesive ligament on the mid-plane."""

    nodes: torch.Tensor
    arms: torch.Tensor
    iface: torch.Tensor
    nx: int
    nz: int
    nxt: int
    n_arm_nodes: int
    dx: float
    i_bond: int

    def up(self, i, j, k):
        return k * 2 * self.nxt + j * self.nxt + i

    def lo(self, i, j, k):
        return self.n_arm_nodes + k * 2 * self.nxt + j * self.nxt + i

    @property
    def n_dofs(self):
        return 3 * self.nodes.shape[0]


def build_specimen(nx: int, nz: int, bonded_start: float, h: float = H_ARM) -> Specimen:
    """Structured HEX8 arms meeting at z=0; cohesive elements for x >= bonded_start."""
    nodes0, cells = structured_hex_mesh(LX, WIDTH, h, nx, 1, nz, dtype=D)
    n_arm = nodes0.shape[0]
    nxt = nx + 1
    lower = nodes0 - torch.tensor([0.0, 0.0, h], dtype=D)
    nodes = torch.cat([nodes0, lower], 0)
    arms = torch.cat([cells, cells + n_arm], 0)

    dx = LX / nx
    i_bond = int(round(bonded_start / dx))
    if i_bond >= nx:
        raise ValueError("no ligament left: bonded_start must be < LX")
    low_top = n_arm + nz * 2 * nxt
    quads = []
    for i in range(i_bond, nx):
        quads.append(
            [
                low_top + i, low_top + i + 1,
                low_top + i + 1 + nxt, low_top + i + nxt,
                i, i + 1, i + 1 + nxt, i + nxt,
            ]
        )
    iface = torch.tensor(quads, dtype=torch.long)
    return Specimen(nodes, arms, iface, nx, nz, nxt, n_arm, dx, i_bond)


# --------------------------------------------------------------------------- #
# Finite-element model: linear adherends + nonlinear cohesive ligament
# --------------------------------------------------------------------------- #
class InterfaceModel:
    def __init__(self, spec: Specimen, law: BKCohesive3DLaw, e=E_ADH, nu=NU_ADH):
        self.spec = spec
        self.law = law
        ne = spec.arms.shape[0]
        d = elasticity_matrix_3d(
            torch.full((ne,), e, dtype=D), torch.full((ne,), nu, dtype=D)
        )
        ke, _ = hex8_stiffness(spec.nodes[spec.arms], d)
        ed = (3 * spec.arms[:, :, None] + AR3).reshape(ne, 24)
        rows = ed[:, :, None].expand(-1, -1, 24).reshape(-1).numpy()
        cols = ed[:, None, :].expand(-1, 24, -1).reshape(-1).numpy()
        self.K_arms = sp.coo_matrix(
            (ke.reshape(-1).numpy(), (rows, cols)), shape=(spec.n_dofs, spec.n_dofs)
        ).tocsc()
        self.iface_dofs = [(3 * c[:, None] + AR3).reshape(-1) for c in spec.iface]
        self.n_iface = spec.iface.shape[0]
        self.damage = np.zeros((self.n_iface, 4))

    def assemble(self, u: torch.Tensor, history: torch.Tensor):
        spec = self.spec
        fint = self.K_arms @ u.numpy()
        disp = u.reshape(-1, 3)
        rows, cols, vals = [], [], []
        new_history = torch.empty_like(history)
        for e, dofs in enumerate(self.iface_dofs):
            conn = spec.iface[e]
            fe, ke, hnew, diag = cohesive_hex8_response(
                spec.nodes[conn], disp[conn], self.law, history[e]
            )
            new_history[e] = hnew
            self.damage[e] = diag["damage"].numpy()
            dn = dofs.numpy()
            fint[dn] += fe.reshape(-1).numpy()
            rows.append(np.repeat(dn, 24))
            cols.append(np.tile(dn, 24))
            vals.append(ke.reshape(-1).numpy())
        K = self.K_arms + sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(spec.n_dofs, spec.n_dofs),
        ).tocsc()
        return fint, K, new_history

    def initial_history(self) -> torch.Tensor:
        return self.law.initial_history((4,)).reshape(1, 4).repeat(self.n_iface, 1)

    def newton(self, free, u, history, fext=None, maxit=40, rtol=1e-9):
        fint = None
        for it in range(maxit):
            fint, K, new_history = self.assemble(u, history)
            R = fint if fext is None else fint - fext
            Rn = R[free]
            scale = max(
                float(np.abs(fext).max()) if fext is not None else 0.0,
                float(np.abs(fint).max()),
            )
            if np.abs(Rn).max() <= rtol * scale:
                return True, it + 1, new_history, fint
            Kff = K.tocsr()[free][:, free].tocsc()
            du = spla.spsolve(Kff, -Rn)
            if not np.all(np.isfinite(du)):
                return False, it + 1, history, fint
            u[torch.from_numpy(free)] += torch.from_numpy(du)
            history = new_history
        return False, maxit, history, fint

    def vcct(self, u: torch.Tensor) -> dict:
        """Rybicki-Kanninen virtual crack closure at the first ligament element."""
        spec = self.spec
        conn = spec.iface[0]
        disp = u.reshape(-1, 3)
        fe, _, _, _ = cohesive_hex8_response(
            spec.nodes[conn], disp[conn], self.law, None
        )
        force = fe[4:, :].sum(dim=0).numpy()
        jump = np.zeros(3)
        for j in (0, 1):
            jump += (disp[spec.up(spec.i_bond, j, 0)] - disp[spec.lo(spec.i_bond, j, spec.nz)]).numpy()
        jump /= 2.0
        g_i = 0.5 * abs(force[2]) * abs(jump[2]) / (WIDTH * spec.dx)
        g_ii = 0.5 * abs(force[0]) * abs(jump[0]) / (WIDTH * spec.dx)
        return {"G_I": float(g_i), "G_II": float(g_ii)}


# --------------------------------------------------------------------------- #
# Boundary conditions
# --------------------------------------------------------------------------- #
def bc_clamp_left(spec: Specimen) -> np.ndarray:
    nodes = []
    for j in (0, 1):
        for k in range(spec.nz + 1):
            nodes += [spec.up(0, j, k), spec.lo(0, j, k)]
    n = torch.tensor(nodes, dtype=torch.long)
    return (3 * n[:, None] + AR3).reshape(-1).numpy()


def dcb_faces(spec: Specimen):
    fixed = bc_clamp_left(spec)
    top = torch.tensor([spec.up(spec.nx, j, k) for j in (0, 1) for k in range(spec.nz + 1)], dtype=torch.long)
    bot = torch.tensor([spec.lo(spec.nx, j, k) for j in (0, 1) for k in range(spec.nz + 1)], dtype=torch.long)
    return fixed, (3 * top + 2).numpy(), (3 * bot + 2).numpy()


def enf_faces(spec: Specimen):
    left = torch.tensor([spec.lo(0, j, 0) for j in (0, 1)], dtype=torch.long)
    right = torch.tensor([spec.lo(spec.nx, j, 0) for j in (0, 1)], dtype=torch.long)
    fixed = np.concatenate([
        (3 * left + 2).numpy(),
        (3 * right + 2).numpy(),
        np.array([3 * int(spec.lo(spec.nx, 0, 0))]),
        (3 * torch.tensor([spec.lo(0, 0, 0), spec.up(0, 0, 0)]) + 1).numpy(),
    ])
    mid = spec.nx // 2
    load = torch.tensor([spec.up(mid, j, spec.nz) for j in (0, 1)], dtype=torch.long)
    return fixed, (3 * load + 2).numpy()


def mmb_faces(spec: Specimen, lever_ratio: float):
    """ASTM D6671 idealisation: rollers at x=0 and x=2L, lever hinge at the mouth.

    Machine load ``P`` down on the lever at arm ``c`` from the midspan fulcrum.
    Lever equilibrium about the fulcrum gives ``U = P c / L`` upward at the
    hinge (top arm, x=0) and ``P + U`` downward at the midspan loading block.
    At ``c = L/3`` the two crack-tip arm loads are equal, i.e. pure mode II --
    which the model reproduces, so the rig statics are checked by the solver.
    """
    left = torch.tensor([spec.lo(0, j, 0) for j in (0, 1)], dtype=torch.long)
    right = torch.tensor([spec.lo(spec.nx, j, 0) for j in (0, 1)], dtype=torch.long)
    fixed = np.concatenate([
        (3 * left + 2).numpy(),
        (3 * right + 2).numpy(),
        np.array([3 * int(spec.lo(spec.nx, 0, 0))]),
        (3 * torch.tensor([spec.lo(0, 0, 0), spec.up(0, 0, 0)]) + 1).numpy(),
    ])
    mouth = torch.tensor([spec.up(0, j, k) for j in (0, 1) for k in range(spec.nz + 1)], dtype=torch.long)
    mid = spec.nx // 2
    block = torch.tensor([spec.up(mid, j, spec.nz) for j in (0, 1)], dtype=torch.long)
    mouth_z = (3 * mouth + 2).numpy()
    block_z = (3 * block + 2).numpy()

    u_hinge = lever_ratio                    # U / P
    f_block = 1.0 + u_hinge                  # (P + U) / P
    w = np.zeros(spec.n_dofs)
    w[mouth_z] = u_hinge / mouth_z.size
    w[block_z] = -f_block / block_z.size
    return fixed, w, mouth_z, block_z


# --------------------------------------------------------------------------- #
# Closed-form anchors
# --------------------------------------------------------------------------- #
def dcb_compliance(a: float, b=WIDTH, h=H_ARM, e=E_ADH) -> float:
    """Two cantilever arms of length a, clamped at the tip: C = 4a^3/(E b h^3)."""
    return 4.0 * a**3 / (e * b * h**3)


def dcb_g(p: float, a: float, b=WIDTH, h=H_ARM, e=E_ADH) -> float:
    return 6.0 * p**2 * a**2 / (e * b**2 * h**3)


def enf_compliance(a: float, half_span: float = LX / 2.0, b=WIDTH, h=H_ARM, e=E_ADH) -> float:
    """ENF with the crack tip beyond midspan: C = (2L^3+3a^3)/(96 E I)."""
    i_arm = b * h**3 / 12.0
    return (2.0 * half_span**3 + 3.0 * a**3) / (96.0 * e * i_arm)


def enf_g(p: float, a: float, b=WIDTH, h=H_ARM, e=E_ADH) -> float:
    return 9.0 * p**2 * a**2 / (16.0 * e * b**2 * h**3)


# --------------------------------------------------------------------------- #
# Linear checks: compliance and VCCT
# --------------------------------------------------------------------------- #
def linear_solution(model: InterfaceModel, fixed: np.ndarray, fext: np.ndarray):
    n = model.spec.n_dofs
    free = np.setdiff1d(np.arange(n), fixed)
    u = torch.zeros(n, dtype=D)
    _, K, _ = model.assemble(u, model.initial_history())
    u[torch.from_numpy(free)] = torch.from_numpy(
        spla.spsolve(K.tocsr()[free][:, free].tocsc(), fext[free])
    )
    return u, free


def check_dcb_linear(mesh=MESH) -> dict:
    nx, nz = mesh["nx"], mesh["nz"]
    spec = build_specimen(nx, nz, LX - CRACK_DCB)
    model = InterfaceModel(spec, make_law())
    fixed, top_z, bot_z = dcb_faces(spec)

    p = 1.0
    fext = np.zeros(spec.n_dofs)
    fext[top_z] = p / (2 * top_z.size)
    fext[bot_z] = -p / (2 * bot_z.size)
    u, _ = linear_solution(model, fixed, fext)
    delta = float(u[torch.from_numpy(top_z)].mean() - u[torch.from_numpy(bot_z)].mean())
    c_fem = delta / p
    c_ref = dcb_compliance(CRACK_DCB)
    v = model.vcct(u)
    g_ref = dcb_g(p, CRACK_DCB)
    return {
        "compliance_fem": c_fem,
        "compliance_ref": c_ref,
        "compliance_rel_error": abs(c_fem - c_ref) / c_ref,
        "G_I_fem": v["G_I"],
        "G_I_ref": g_ref,
        "G_I_rel_error": abs(v["G_I"] - g_ref) / g_ref,
        "G_II_fem": v["G_II"],
        "G_II_over_G_total": v["G_II"] / max(v["G_I"] + v["G_II"], 1e-30),
    }


def check_enf_linear(mesh=MESH) -> dict:
    nx, nz = mesh["nx"], mesh["nz"]
    spec = build_specimen(nx, nz, CRACK_ENF)
    model = InterfaceModel(spec, make_law())
    fixed, load_z = enf_faces(spec)
    p = 1.0
    fext = np.zeros(spec.n_dofs)
    fext[load_z] = -p / load_z.size
    u, _ = linear_solution(model, fixed, fext)
    delta = float(-u[torch.from_numpy(load_z)].mean())
    c_fem = delta / p
    c_ref = enf_compliance(CRACK_ENF)
    v = model.vcct(u)
    g_ref = enf_g(p, CRACK_ENF)
    return {
        "compliance_fem": c_fem,
        "compliance_ref": c_ref,
        "compliance_rel_error": abs(c_fem - c_ref) / c_ref,
        "G_II_fem": v["G_II"],
        "G_II_ref": g_ref,
        "G_II_rel_error": abs(v["G_II"] - g_ref) / g_ref,
        "G_I_fem": v["G_I"],
        "G_I_over_G_total": v["G_I"] / max(v["G_I"] + v["G_II"], 1e-30),
    }


def check_mmb_pure_mode_ii(mesh=MESH) -> dict:
    """At c = L/3 the lever applies equal crack-tip arm loads: pure mode II."""
    nx, nz = mesh["nx"], mesh["nz"]
    spec = build_specimen(nx, nz, CRACK_DCB)
    model = InterfaceModel(spec, make_law())
    fixed, w, _, _ = mmb_faces(spec, 1.0 / 3.0)
    u, _ = linear_solution(model, fixed, w)
    v = model.vcct(u)
    total = v["G_I"] + v["G_II"]
    return {
        "G_I": v["G_I"],
        "G_II": v["G_II"],
        "G_I_over_G_total": v["G_I"] / total if total else 0.0,
        "criterion": "G_I/(G_I+G_II) < 1% at c = L/3",
        "passed": (v["G_I"] / total if total else 0.0) < 0.01,
    }


def mmb_linear_prediction(lever_ratio, mesh=MESH) -> dict:
    nx, nz = mesh["nx"], mesh["nz"]
    spec = build_specimen(nx, nz, CRACK_DCB)
    model = InterfaceModel(spec, make_law())
    fixed, w, mouth_z, _ = mmb_faces(spec, lever_ratio)
    u, _ = linear_solution(model, fixed, w)          # unit machine load P = 1 N
    v = model.vcct(u)
    total = v["G_I"] + v["G_II"]
    p_pred = onset_load(total, v["G_I"], v["G_II"])
    return {
        "lever_ratio": lever_ratio,
        "unit_load_G_I": v["G_I"],
        "unit_load_G_II": v["G_II"],
        "mode_II_fraction_from_VCCT": v["G_II"] / total if total else 0.0,
        "bk_toughness_at_mix": bk_toughness(v["G_I"], v["G_II"]),
        "predicted_onset_load": p_pred,
        "mouth_disp_per_unit_load": float(u[torch.from_numpy(mouth_z)].mean()),
    }


# --------------------------------------------------------------------------- #
# Cohesive-zone simulations
# --------------------------------------------------------------------------- #
def simulate_displacement(case: str, mesh=MESH, strengths=(TN0, TS0), d_max=None,
                          n_steps=160, crack=None) -> dict:
    nx, nz = mesh["nx"], mesh["nz"]
    if case == "dcb":
        spec = build_specimen(nx, nz, LX - (crack or CRACK_DCB))
        fixed, top_z, bot_z = dcb_faces(spec)
    else:
        spec = build_specimen(nx, nz, crack or CRACK_ENF)
        fixed, load_z = enf_faces(spec)
    model = InterfaceModel(spec, make_law(*strengths))
    n = spec.n_dofs
    free = np.setdiff1d(np.arange(n), fixed)
    u = torch.zeros(n, dtype=D)
    history = model.initial_history()

    d_max = d_max or (9.0e-5 if case == "dcb" else 6.0e-5)
    curve, peak = [], {"load": 0.0, "disp": 0.0, "step": -1}
    ok_all = True
    for step in range(1, n_steps + 1):
        amp = d_max * step / n_steps
        u[:] = 0.0
        if case == "dcb":
            u[torch.from_numpy(top_z)] = 0.5 * amp
            u[torch.from_numpy(bot_z)] = -0.5 * amp
        else:
            u[torch.from_numpy(load_z)] = -amp
        ok, its, history, fint = model.newton(free, u, history)
        ok_all &= ok
        if not ok:
            break
        if case == "dcb":
            load = 0.5 * (float(fint[top_z].sum()) - float(fint[bot_z].sum()))
        else:
            load = -float(fint[load_z].sum())
        curve.append({"step": step, "disp": amp, "load": load,
                      "max_damage": float(model.damage.max())})
        if load > peak["load"]:
            peak = {"load": load, "disp": amp, "step": step}
    profile = model.damage.mean(axis=1)
    return {
        "case": case,
        "converged_all": bool(ok_all),
        "peak_load": peak["load"],
        "peak_disp": peak["disp"],
        "peak_step": peak["step"],
        "curve": curve,
        "damage_profile": profile.tolist(),
        "damage_front_fraction": float((profile > 0.5).mean()),
        "n_dofs": n,
        "n_iface": model.n_iface,
        "n_steps": n_steps,
    }


def simulate_mmb(lever_ratio=LEVER_RATIO, mesh=MESH, strengths=(TN0, TS0),
                 p_start=40.0, n_steps=90) -> dict:
    """Force control on the MMB rig; the limit point is where Newton stops."""
    nx, nz = mesh["nx"], mesh["nz"]
    spec = build_specimen(nx, nz, CRACK_DCB)
    model = InterfaceModel(spec, make_law(*strengths))
    fixed, w, mouth_z, block_z = mmb_faces(spec, lever_ratio)
    n = spec.n_dofs
    free = np.setdiff1d(np.arange(n), fixed)
    u = torch.zeros(n, dtype=D)
    history = model.initial_history()

    lam, dlam0, dlam = 0.0, p_start / n_steps, p_start / n_steps
    curve, best = [], {"load": 0.0, "disp": 0.0}
    for _ in range(n_steps * 6):
        trial = lam + dlam
        fext = trial * w
        ok, its, history_new, fint = model.newton(free, u, history, fext)
        if ok:
            lam = trial
            history = history_new
            disp = float(u[torch.from_numpy(mouth_z)].mean())
            curve.append({"load": lam, "disp": disp, "max_damage": float(model.damage.max())})
            best = {"load": lam, "disp": disp}
        else:
            dlam *= 0.5
            if dlam < 1e-3 * dlam0:
                break
    profile = model.damage.mean(axis=1)
    return {
        "case": "mmb",
        "lever_ratio": lever_ratio,
        "limit_load": best["load"],
        "limit_disp": best["disp"],
        "curve": curve,
        "damage_profile": profile.tolist(),
        "damage_front_fraction": float((profile > 0.5).mean()),
        "n_dofs": n,
        "n_iface": model.n_iface,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(quick: bool = False) -> dict:
    t_start = time.time()
    mesh = {"nx": 60, "nz": 2} if quick else dict(MESH)
    n_steps = 20 if quick else 80

    out = {
        "schema": "tensorfem.cohesive3d-delamination/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/cohesive3d_delamination.py",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "model": {
            "specimen": {"length_m": LX, "width_m": WIDTH, "arm_thickness_m": H_ARM,
                         "adherend_E_Pa": E_ADH, "adherend_nu": NU_ADH,
                         "mesh": mesh},
            "interface": {"k_n_Pa_per_m": KN, "k_s_Pa_per_m": KS,
                          "t_n0_Pa": TN0, "t_s0_Pa": TS0,
                          "G_Ic_J_per_m2": GIC, "G_IIc_J_per_m2": GIIC,
                          "bk_eta": BK_ETA},
        },
        "criteria": {},
        "curves": {},
    }

    # 1. linear checks ----------------------------------------------------- #
    dcb_lin = check_dcb_linear(mesh)
    out["criteria"]["dcb_linear_compliance"] = {
        "criterion": "FE compliance vs 4a^3/(E b h^3), rel < 2%",
        "ref": dcb_lin["compliance_ref"], "got": dcb_lin["compliance_fem"],
        "rel_error": dcb_lin["compliance_rel_error"],
        "passed": dcb_lin["compliance_rel_error"] < 0.02,
    }
    out["criteria"]["dcb_linear_vcct"] = {
        "criterion": "VCCT G_I vs 6P^2a^2/(E b^2 h^3), rel < 5%",
        "ref": dcb_lin["G_I_ref"], "got": dcb_lin["G_I_fem"],
        "rel_error": dcb_lin["G_I_rel_error"],
        "spurious_G_II_fraction": dcb_lin["G_II_over_G_total"],
        "passed": dcb_lin["G_I_rel_error"] < 0.05,
    }
    enf_lin = check_enf_linear(mesh)
    out["criteria"]["enf_linear_compliance"] = {
        "criterion": "FE compliance vs (2L^3+3a^3)/(96 E I), rel < 5%",
        "ref": enf_lin["compliance_ref"], "got": enf_lin["compliance_fem"],
        "rel_error": enf_lin["compliance_rel_error"],
        "passed": enf_lin["compliance_rel_error"] < 0.05,
    }
    out["criteria"]["enf_linear_vcct"] = {
        "criterion": "VCCT G_II vs 9P^2a^2/(16 E b^2 h^3), rel < 5%",
        "ref": enf_lin["G_II_ref"], "got": enf_lin["G_II_fem"],
        "rel_error": enf_lin["G_II_rel_error"],
        "spurious_G_I_fraction": enf_lin["G_I_over_G_total"],
        "passed": enf_lin["G_II_rel_error"] < 0.05,
    }

    # 2. mixed-mode rig self-consistency ---------------------------------- #
    pure_ii = check_mmb_pure_mode_ii(mesh)
    out["criteria"]["mmb_pure_mode_ii"] = pure_ii

    # 3. DCB cohesive propagation ----------------------------------------- #
    dcb = simulate_displacement("dcb", mesh, n_steps=n_steps)
    g_dcb = dcb_g(dcb["peak_load"], CRACK_DCB)
    out["criteria"]["dcb_mode_i_debond"] = {
        "criterion": "computed G_I at debonding vs G_Ic, rel < 5%",
        "ref": GIC, "got": g_dcb,
        "rel_error": abs(g_dcb - GIC) / GIC,
        "peak_load_N": dcb["peak_load"],
        "peak_load_ref_N": math.sqrt(GIC * E_ADH * WIDTH**2 * H_ARM**3 / (6.0 * CRACK_DCB**2)),
        "passed": abs(g_dcb - GIC) / GIC < 0.05,
        "converged_all": dcb["converged_all"],
        "damage_front_fraction": dcb["damage_front_fraction"],
    }
    out["curves"]["dcb"] = dcb["curve"]

    # 4. ENF cohesive propagation ----------------------------------------- #
    enf = simulate_displacement("enf", mesh, n_steps=n_steps)
    g_enf = enf_g(enf["peak_load"], CRACK_ENF)
    out["criteria"]["enf_mode_ii_debond"] = {
        "criterion": "computed G_II at debonding vs G_IIc, rel < 5%",
        "ref": GIIC, "got": g_enf,
        "rel_error": abs(g_enf - GIIC) / GIIC,
        "peak_load_N": enf["peak_load"],
        "peak_load_ref_N": math.sqrt(GIIC * 16.0 * E_ADH * WIDTH**2 * H_ARM**3 / (9.0 * CRACK_ENF**2)),
        "passed": abs(g_enf - GIIC) / GIIC < 0.05,
        "converged_all": enf["converged_all"],
        "damage_front_fraction": enf["damage_front_fraction"],
    }
    out["curves"]["enf"] = enf["curve"]

    # 5. MMB mixed mode ---------------------------------------------------- #
    mmb_lin = mmb_linear_prediction(LEVER_RATIO, mesh)
    mmb = simulate_mmb(LEVER_RATIO, mesh)
    p = mmb["limit_load"]
    g1 = mmb_lin["unit_load_G_I"] * p**2
    g2 = mmb_lin["unit_load_G_II"] * p**2
    g_c_mix = bk_toughness(g1, g2)
    out["criteria"]["mmb_mixed_mode_debond"] = {
        "criterion": "BK(G_I,G_II) at the cohesive limit load vs BK G_c, rel < 5%",
        "ref": g_c_mix, "got": g1 + g2,
        "rel_error": abs(g1 + g2 - g_c_mix) / g_c_mix,
        "limit_load_N": p,
        "predicted_onset_load_N": mmb_lin["predicted_onset_load"],
        "mode_II_fraction": mmb_lin["mode_II_fraction_from_VCCT"],
        "G_I": g1, "G_II": g2,
        "passed": abs(g1 + g2 - g_c_mix) / g_c_mix < 0.05,
        "damage_front_fraction": mmb["damage_front_fraction"],
    }
    out["curves"]["mmb"] = mmb["curve"]
    out["damage_profiles"] = {"dcb": dcb["damage_profile"],
                              "enf": enf["damage_profile"],
                              "mmb": mmb["damage_profile"]}

    out["mesh"] = mesh
    out["all_passed"] = all(c.get("passed", True) for c in out["criteria"].values())
    out["runtime_s"] = time.time() - t_start
    return out


def plot(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    labels = {"dcb": ("DCB (mode I)", "C0"), "enf": ("ENF (mode II)", "C1"),
              "mmb": ("MMB (mixed)", "C2")}
    for key, (label, colour) in labels.items():
        curve = out["curves"].get(key) or []
        if not curve:
            continue
        ax.plot([c["disp"] * 1e6 for c in curve], [c["load"] for c in curve],
                colour, marker=".", ms=3, lw=1.2, label=label)
    ax.axhline(out["criteria"]["dcb_mode_i_debond"]["peak_load_N"], ls="--", lw=0.8, color="gray")
    ax.set_xlabel("mouth / midspan displacement [um]")
    ax.set_ylabel("load [N]")
    ax.set_title("cohesive-zone specimen response")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key, (label, colour) in labels.items():
        prof = out["damage_profiles"].get(key) or []
        if not prof:
            continue
        ax.plot(np.linspace(0, 1, len(prof)), prof, colour, lw=1.4, label=label)
    ax.set_xlabel("normalised interface position (mouth -> root)")
    ax.set_ylabel("mean secant damage D")
    ax.set_title("damage field on the cohesive ligament (final state)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    out = run(quick=args.quick)
    dest = ROOT / "benchmarks" / "cohesive3d_delamination.json"
    if not args.quick:
        dest.write_text(json.dumps(out, indent=2))
        plot(out, ROOT / "docs" / "assets" / "cohesive3d-delamination" / "response.png")
    print(json.dumps({k: {"rel_error": v.get("rel_error"), "passed": v.get("passed")}
                      for k, v in out["criteria"].items()}, indent=2))
    print(f"all_passed={out['all_passed']} runtime={out['runtime_s']:.1f}s")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())