"""Sandwich (夹层) panel bending: heterogeneous layups and three independent references.

Why this module exists
----------------------
:mod:`tensorfem.composite_laminate_bending` covers laminates built from *one*
material with *equal* ply thicknesses.  A sandwich panel -- the construction used
for FRP decks, superstructure and ice-belt panels -- is the opposite case: two
thin stiff faces separated by a thick soft core, so ``E_face / E_core`` reaches
1e3..1e4 and the thickness ratio 1:20.  Feeding such a panel into the laminate
code as if it were homogeneous is a modelling error, and the uniform shear
correction ``k = 5/6`` that is right for a homogeneous plate is *not* right for a
sandwich, where practically all transverse shear is carried by the core.

This module supplies

1. :class:`SandwichPanel` -- the panel definition, mapped onto the generalised
   :class:`~tensorfem.composite_laminate_bending.Laminate` of the laminate
   module (per-ply materials and thicknesses), so every existing solver --
   CLT Navier, FSDT Navier, the layered 3-D elasticity (Pagano) solution and the
   layered HEX20 FE -- runs on a sandwich unchanged;
2. :meth:`SandwichPanel.classical` -- the engineering sandwich reference
   (antiplane-core theory, faces as membranes, core carrying transverse shear
   only; Allen 1969, Plantema 1966, Zenkert 1995), i.e. an
   *independent-of-FE* closed form;
3. :func:`deflection_budget` / :func:`accuracy_envelope` -- the comparison
   machinery that turns the three references into a graded verification table
   (the file ``docs/sandwich-panel-bending.md`` is generated from it).

Normalisation and conventions follow the laminate module: ``q0`` acts down,
deflection is positive down, the panel is simply supported with the shear
diaphragm (SS1) condition and the load is the single sine mode
``q0 sin(alpha x) sin(beta y)``, which makes every theory's answer exact for its
own kinematics.

References
----------
* Allen, H.G. (1969) *Analysis and Design of Structural Sandwich Panels*,
  Pergamon -- antiplane-core bending and shear rigidities of a sandwich plate.
* Plantema, F.J. (1966) *Sandwich Construction*, Wiley.
* Zenkert, D. (1995) *An Introduction to Sandwich Construction*, EMAS --
  face-stress and core-shear engineering formulas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .composite_laminate_bending import (
    Laminate,
    Ply3D,
    navier_clt,
    navier_fsdt,
)

__all__ = [
    "SandwichPanel",
    "deflection_budget",
    "accuracy_envelope",
    "ELASTIC_FACE",
    "PVC_CORE",
    "BALSA_CORE",
]

#: Marine E-glass/vinyl-ester face ply, quasi-isotropic equivalent (Pa).
ELASTIC_FACE = Ply3D(E1=30.0e9, E2=30.0e9, E3=30.0e9, nu12=0.30, nu13=0.30, nu23=0.30,
                     G12=11.54e9, G13=11.54e9, G23=11.54e9)
#: H80 PVC foam core (Pa): E = 90 MPa, G = 32 MPa, nu = 0.4.
PVC_CORE = Ply3D(E1=90.0e6, E2=90.0e6, E3=90.0e6, nu12=0.40, nu13=0.40, nu23=0.40,
                 G12=32.0e6, G13=32.0e6, G23=32.0e6)
#: End-grain balsa core (Pa): E = 2.5 GPa along grain.
BALSA_CORE = Ply3D(E1=2.5e9, E2=0.15e9, E3=0.15e9, nu12=0.30, nu13=0.30, nu23=0.30,
                   G12=0.11e9, G13=0.11e9, G23=0.05e9)


@dataclass(frozen=True)
class SandwichPanel:
    """Two faces (``t_face`` each) on a core (``t_core``), all plies at 0 deg.

    ``n_core`` subdivides the core through the thickness.  It changes only the
    discretisation of the through-thickness integration and of the HEX20 mesh,
    never the physics; ``n_core >= 2`` is needed for the FE mesh to resolve a
    bending stress gradient inside the core.
    """

    face: Ply3D = ELASTIC_FACE
    core: Ply3D = PVC_CORE
    t_face: float = 2.0e-3
    t_core: float = 40.0e-3
    n_core: int = 2

    def __post_init__(self) -> None:
        for name in ("t_face", "t_core"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")
        if int(self.n_core) < 1:
            raise ValueError("n_core must be >= 1")

    # -- geometry -----------------------------------------------------------
    @property
    def thickness(self) -> float:
        return 2.0 * self.t_face + self.t_core

    @property
    def d(self) -> float:
        """Distance between the two face mid-planes (the sandwich moment arm)."""
        return self.t_core + self.t_face

    def laminate(self) -> Laminate:
        """The equivalent generalised laminate (faces at 0 deg)."""
        n = int(self.n_core)
        materials = [self.face] + [self.core] * n + [self.face]
        thicknesses = [self.t_face] + [self.t_core / n] * n + [self.t_face]
        return Laminate([0.0] * (n + 2), self.t_face, self.face,
                        ply_materials=materials, ply_thicknesses=thicknesses)

    # -- engineering (antiplane core) reference -----------------------------
    def face_Q(self) -> np.ndarray:
        """Plane-stress reduced stiffness of a face ply in panel axes."""
        from .composite_laminate_bending import ply_stiffness_3d

        C = ply_stiffness_3d(self.face, 0.0)
        ip, op = [0, 1, 5], [2, 3, 4]
        return C[np.ix_(ip, ip)] - C[np.ix_(ip, op)] @ np.linalg.solve(
            C[np.ix_(op, op)], C[np.ix_(op, ip)])

    def core_shear_modulus(self) -> float:
        """In-plane-transverse core shear modulus ``G_xz`` used by the core model."""
        from .composite_laminate_bending import ply_stiffness_3d

        C = ply_stiffness_3d(self.core, 0.0)
        return float(C[4, 4])

    def classic_D(self) -> np.ndarray:
        """Antiplane-core bending rigidity from the faces only (Allen 1969).

        The faces act as membranes at ``+-d/2``, so ``D_ij = Q_ij t_f d^2 / 2``
        (two faces, each ``A_ij = Q_ij t_f`` at lever arm ``d/2``).  The faces'
        own bending and the core in-plane stiffness are neglected -- that is the
        definition of the antiplane-core model, and its error is quantified
        against the layered 3-D solution by :func:`accuracy_envelope`.
        """
        Q = self.face_Q()
        Dm = np.zeros((3, 3))
        idx = [(0, 0), (1, 1), (0, 1), (2, 2)]
        for i, j in idx:
            Dm[i, j] = Q[i, j] * self.t_face * self.d ** 2 / 2.0
        return Dm

    def classic_shear_rigidity(self) -> float:
        """Antiplane-core transverse shear rigidity ``D_Q = G_core d^2 / t_core``."""
        return self.core_shear_modulus() * self.d ** 2 / self.t_core

    def core_only_shear_rigidity(self) -> float:
        """``G_core t_core``: the core's own ``int G dz`` (k = 1, no 5/6 factor)."""
        return self.core_shear_modulus() * self.t_core

    def classical(self, a: float, b: float | None = None, q0: float = 1.0,
                  m: int = 1, n: int = 1) -> dict:
        """Closed-form antiplane-core sandwich solution of the SS1 sine problem.

        Bending deflection from the face-carried rigidity, shear deflection from
        ``D_Q``; the two add (Allen's ``w = w_b + w_s``), which is exact for the
        antiplane-core kinematics and for a single sine mode.
        """
        b = float(b if b is not None else a)
        a = float(a)
        al, be = m * math.pi / a, n * math.pi / b
        D = self.classic_D()
        DQ = self.classic_shear_rigidity()
        denom_b = (D[0, 0] * al**4 + 2.0 * (D[0, 1] + 2.0 * D[2, 2]) * al**2 * be**2
                   + D[1, 1] * be**4)
        w_b = q0 / denom_b
        w_s = q0 / (DQ * (al**2 + be**2))
        Mx = D[0, 0] * al**2 * w_b + D[0, 1] * be**2 * w_b
        My = D[0, 1] * al**2 * w_b + D[1, 1] * be**2 * w_b
        Qx = DQ * al * w_s
        # sagging plate under a downward load: the top face carries N = -M/d, so
        # sigma_xx(top face) = -Mx/(d t_f)  (compression negative, tension positive)
        return {"w_bending": w_b, "w_shear": w_s, "w_center": w_b + w_s,
                "D": D, "D_Q": DQ, "Mx_center": Mx, "My_center": My,
                "Qx_at_edge": Qx,
                "sigma_face_top_x": -Mx / (self.d * self.t_face),
                "tau_core_xz_center": Qx / self.d,
                "alpha": al, "beta": be}

    # -- theory-side references from the laminate module --------------------
    def clt(self, a: float, b: float | None = None, q0: float = 1.0, m: int = 1,
            n: int = 1):
        return navier_clt(self.laminate(), a, b if b is not None else a, q0, m, n)

    def fsdt(self, a: float, b: float | None = None, q0: float = 1.0, m: int = 1,
             n: int = 1, shear_correction: float = 5.0 / 6.0):
        """FSDT Navier solution of the sandwich (uniform shear correction)."""
        return navier_fsdt(self.laminate(), a, b if b is not None else a, q0, m, n,
                           shear_correction=shear_correction)

    def fsdt_core_only_shear(self, a: float, b: float | None = None, q0: float = 1.0,
                             m: int = 1, n: int = 1):
        """FSDT with ``As = G_core t_core`` (no 5/6, faces contribute no shear).

        This is the shear rigidity a sandwich model is *supposed* to use; the
        uniform ``k = 5/6`` applied to the whole stack (what a homogeneous-plate
        code does) is compared against it in :func:`accuracy_envelope`.
        """
        return navier_fsdt(_Rigidity(self.laminate().d_matrix(),
                                     np.array([[self.core_only_shear_rigidity(), 0.0],
                                               [0.0, self.core_only_shear_rigidity()]])),
                           a, b if b is not None else a, q0, m, n, shear_correction=1.0)

    def fsdt_sandwich(self, a: float, b: float | None = None, q0: float = 1.0,
                      m: int = 1, n: int = 1, face_only_bending: bool = False):
        """FSDT with the *sandwich* shear rigidity ``D_Q = G_core d^2 / t_core``.

        This is the shear rigidity the antiplane-core theory (Allen 1969) assigns
        to a sandwich, recovered here as an FSDT input instead of a kinematic
        assumption; ``face_only_bending`` uses the faces-only bending rigidity
        (pure antiplane-core) rather than the full ABD ``D``.
        """
        D = self.classic_D() if face_only_bending else self.laminate().d_matrix()
        DQ = self.classic_shear_rigidity()
        return navier_fsdt(_Rigidity(D, np.array([[DQ, 0.0], [0.0, DQ]])),
                           a, b if b is not None else a, q0, m, n, shear_correction=1.0)

    def exact3d(self, a: float, b: float | None = None, q0: float = 1.0, m: int = 1,
                n: int = 1):
        """Layered 3-D elasticity (Pagano) solution -- the thick-panel referee."""
        from .composite_laminate_bending import LayeredExact3D

        return LayeredExact3D(self.laminate(), a, b if b is not None else a, q0, m, n)

    def hex20(self, a: float, b: float | None = None, q0: float = 1.0, nx: int = 8,
              nz_per_ply: int = 1, m: int = 1, n: int = 1) -> dict:
        """Layered HEX20 3-D FE solution of the same problem."""
        from .composite_laminate_bending import layered_hex20_fem

        return layered_hex20_fem(self.laminate(), a, q0, nx=nx, nz_per_ply=nz_per_ply,
                                 b=b, m=m, n_mode=n)


@dataclass(frozen=True)
class _Rigidity:
    """Duck-typed stand-in for ``Laminate`` when the shear rigidity is prescribed."""

    D: np.ndarray
    As: np.ndarray

    def d_matrix(self) -> np.ndarray:
        return self.D

    def as_matrix(self, shear_correction: float = 5.0 / 6.0) -> np.ndarray:
        return shear_correction * self.As


def _rel(value: float, reference: float) -> float:
    return (value - reference) / reference


def deflection_budget(panel: SandwichPanel, a: float, q0: float = 1.0,
                      include_3d: bool = True, include_hex20: bool = False,
                      nx: int = 8) -> dict:
    """One benchmark point: every model's centre deflection plus the reference.

    The reference is the layered 3-D elasticity solution when ``include_3d``
    (it is the only model that resolves the core shear *and* the face bending
    without a kinematic assumption), otherwise the antiplane-core closed form.
    """
    b = a
    out = {"a_over_h": a / panel.thickness, "t_core": panel.t_core,
           "t_face": panel.t_face, "d_over_t": panel.d / panel.t_core}
    classic = panel.classical(a, b, q0)
    out["classical"] = classic["w_center"]
    out["classical_bending"] = classic["w_bending"]
    out["classical_shear"] = classic["w_shear"]
    out["clt"] = panel.clt(a, b, q0).W
    out["fsdt_k56"] = panel.fsdt(a, b, q0).W
    out["fsdt_core"] = panel.fsdt_core_only_shear(a, b, q0).W
    out["fsdt_sandwich"] = panel.fsdt_sandwich(a, b, q0).W
    if include_3d:
        ex = panel.exact3d(a, b, q0)
        out["exact3d"] = float(ex.center_deflection)
        sig_mid_face = ex.stress_amplitudes(panel.t_core / 2.0 + panel.t_face / 2.0)
        out["exact3d_sigma_face_x"] = float(sig_mid_face[0])
        out["classical_sigma_face_top_x"] = classic["sigma_face_top_x"]
        out["err_sigma_face_classical"] = _rel(classic["sigma_face_top_x"],
                                              out["exact3d_sigma_face_x"])
        tau_exact = float(ex.face_shear(0.0, 0.0, b / 2.0))
        out["exact3d_tau_core_center"] = tau_exact
        out["classical_tau_core_center"] = classic["tau_core_xz_center"]
        out["err_tau_classical"] = _rel(classic["tau_core_xz_center"], tau_exact)
    if include_hex20:
        out["hex20"] = float(panel.hex20(a, b, q0, nx=nx)["w_center"])
    ref = out.get("exact3d", out["classical"])
    out["reference"] = ref
    for key in ("classical", "clt", "fsdt_k56", "fsdt_core", "fsdt_sandwich", "hex20"):
        if key in out:
            out[f"err_{key}"] = _rel(out[key], ref)
    return out


def accuracy_envelope(panel: SandwichPanel, spans: Sequence[float] | None = None,
                      q0: float = 1.0, include_hex20: bool = False,
                      nx: int = 8) -> list[dict]:
    """Sweep the span so the panel is graded from thick (a/h ~ 3) to thin (a/h ~ 60)."""
    if spans is None:
        h = panel.thickness
        spans = [3.0 * h, 5.0 * h, 10.0 * h, 20.0 * h, 40.0 * h, 60.0 * h]
    return [deflection_budget(panel, float(a), q0, include_hex20=include_hex20, nx=nx)
            for a in spans]