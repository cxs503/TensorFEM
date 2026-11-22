"""Separate spatial integration, plate resolution, penalty and time sampling."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from tensorfem.curved_panel_contact import CurvedPanelContact

PARAMS = dict(side=1.0, radius=3.0, thickness=0.02, young=2.1e11, poisson=0.3, stiffness=1.0e9)
DEPTH = 0.004
DURATION = 0.1


def run(mesh, cells, stiffness, half_steps):
    model = CurvedPanelContact(mesh, cells, **(PARAMS | {"stiffness": stiffness}))
    # Symmetric triangular displacement. There is no internal history in this
    # elastic model, so the unloading force follows the solved loading branch.
    force = []
    for d in np.linspace(0.0, DEPTH, half_steps + 1):
        step = model.solve(d)
        force.append(step.force_N)
    centre = step.panel.stress_at(PARAMS["side"] / 2, PARAMS["side"] / 2)
    return dict(
        mesh=mesh,
        cells=cells,
        stiffness_Pa_m=stiffness,
        half_steps=half_steps,
        force_N=step.force_N,
        contact_area_m2=step.contact_area_m2,
        centre_vm_MPa=centre["sigma_vm_MPa"],
        station_m=centre["at_m"],
        impulse_Ns=float(trapezoid(force, dx=DURATION / half_steps)),
        loading_force_N=force,
        peak_pressure_Pa=step.peak_pressure_Pa,
        support_N=step.support_N,
        relative_residual=step.relative_residual,
    )


def study():
    # Odd plate meshes share the centre element station exactly.
    base = (31, 120, PARAMS["stiffness"], 8)
    settings = [
        base,
        (15, *base[1:]),
        (47, *base[1:]),
        (31, 60, *base[2:]),
        (31, 240, *base[2:]),
        (31, 120, 0.5e9, 8),
        (31, 120, 2e9, 8),
        (31, 120, 1e9, 16),
    ]
    rows = [run(*cfg) for cfg in settings]
    # Rigid-panel limit of this specific compliant parabolic contact law.
    r = PARAMS["radius"]
    rigid = dict(
        force_N=math.pi * PARAMS["stiffness"] * r * DEPTH**2, area_m2=2 * math.pi * r * DEPTH
    )
    rigid_model = CurvedPanelContact(9, 160, **(PARAMS | {"young": 1e17}))
    step = rigid_model.solve(DEPTH)
    rigid.update(
        numerical_force_N=step.force_N,
        numerical_area_m2=step.contact_area_m2,
        numerical_interface_energy_J=step.contact_energy_J,
        exact_interface_energy_J=rigid["force_N"] * DEPTH / 3,
        young_Pa=1e17,
        mesh=9,
        cells=160,
    )
    return dict(
        parameters=PARAMS,
        indentation_m=DEPTH,
        pulse_duration_s=DURATION,
        rigid_limit=rigid,
        rows=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(study(), indent=2, allow_nan=False) + "\n")
