"""Quasi-static paraboloid contact on a clamped elastic Q4 Mindlin panel.

The compliant layer transmits compression only. The physical integration grid
and its Q4 work-conjugate transfer do not depend on the plate mesh resolution.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .foundation_plate_bending import assemble_plate_foundation_system, uniform_rect_mesh
from .ice_panel_patch_contact import PanelSolution, _panel_bcs


@dataclass(frozen=True)
class ContactStep:
    panel: PanelSolution
    force_N: float
    contact_area_m2: float
    peak_pressure_Pa: float
    support_N: float
    contact_energy_J: float
    elastic_energy_J: float
    relative_residual: float


class CurvedPanelContact:
    """Rigid parabolic tip, compliant interface and elastic square plate in SI.

    Midpoint integration covers the whole plate. Positive w follows the imposed
    indentation. The stiffness of the interface has units Pa/m. Each solve is
    reversible and memoryless (no ice fracture or impact inertia).
    """

    def __init__(self, mesh, contact_cells, *, side, radius, thickness, young, poisson, stiffness):
        if any(type(n) is not int or n < 2 for n in (mesh, contact_cells)):
            raise ValueError("mesh counts must be integers of at least two")
        if not all(math.isfinite(v) and v > 0 for v in (side, radius, thickness, young, stiffness)):
            raise ValueError("positive finite geometry and stiffness required")
        if not -1 < poisson < 0.5:
            raise ValueError("invalid Poisson ratio")
        self.mesh, self.side = mesh, side
        self.young, self.poisson, self.thickness = young, poisson, thickness
        self.stiffness = stiffness
        self.nodes, self.elements = uniform_rect_mesh(mesh, mesh, side, side)
        rows, cols, vals, _, ndof = assemble_plate_foundation_system(
            self.nodes,
            self.elements,
            young=young,
            poisson=poisson,
            thickness=thickness,
            pressure=0,
            shear_treatment="mitc",
        )
        self.K = coo_matrix(
            (vals.numpy(), (rows.numpy(), cols.numpy())), shape=(ndof, ndof)
        ).tocsr()
        fixed, free = _panel_bcs(self.nodes, side, side, ndof, "clamped")
        self.free = free.numpy()
        self.support = fixed[fixed % 3 == 0].numpy()
        self.Kff = self.K[self.free][:, self.free]
        q = (np.arange(contact_cells) + 0.5) / contact_cells
        x, y = np.meshgrid(q, q)
        x, y = x.ravel(), y.ravel()
        self.z = side**2 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * radius)
        self.area = (side / contact_cells) ** 2
        ix, iy = (mesh * x).astype(int), (mesh * y).astype(int)
        s, t = mesh * x - ix, mesh * y - iy
        shapes = np.column_stack(((1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t))
        faces = self.elements[(iy * mesh + ix)].numpy()
        self.N = coo_matrix(
            (shapes.ravel(), (np.repeat(np.arange(x.size), 4), 3 * faces.ravel())),
            shape=(x.size, ndof),
        ).tocsr()
        self.Nf = self.N[:, self.free]

    def solve(self, indentation):
        """Solve unilateral equilibrium and independently recover clamp reaction."""
        if not math.isfinite(indentation):
            raise ValueError("indentation must be finite")
        u = np.zeros(self.K.shape[0])
        for _ in range(60):
            overlap = indentation - self.z - self.N @ u
            pressure = self.stiffness * np.maximum(overlap, 0)
            nodal = self.N.T @ (pressure * self.area)
            internal = self.K @ u
            residual = (internal - nodal)[self.free]
            error = np.linalg.norm(residual) / max(np.linalg.norm(nodal[self.free]), 1.0)
            if error < 1e-9:
                break
            active = self.Nf[overlap > 0]
            tangent = self.Kff + self.stiffness * self.area * (active.T @ active)
            u[self.free] -= spsolve(tangent.tocsc(), residual)
        else:
            raise RuntimeError("curved contact did not reach free-DOF equilibrium")
        panel = PanelSolution(
            torch.from_numpy(u),
            self.nodes,
            self.elements,
            self.young,
            self.poisson,
            self.thickness,
            "clamped",
            self.side,
            self.side,
            self.mesh,
            self.mesh,
        )
        return ContactStep(
            panel,
            float(pressure.sum() * self.area),
            float(np.count_nonzero(pressure) * self.area),
            float(pressure.max()),
            float((nodal - internal)[self.support].sum()),
            float(self.area * (pressure @ pressure) / (2 * self.stiffness)),
            float(0.5 * u @ internal),
            float(error),
        )
