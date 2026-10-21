"""Cohesive failure criterion for internal faces.

This module implements a tension/shear combined failure criterion for
internal faces of plate elements.  The criterion is parameterised by
tensile strength ``sigma_max`` and shear strength ``tau_max``.

Failure is determined by a quadratic interaction criterion:

    f = (sigma_n / sigma_max)^2 + (tau / tau_max)^2

When f >= 1 the face is considered failed.  Upon failure, the released
energy is G * A where G is the fracture energy (per unit area) and A
is the face area.

Interface contract (frozen for SI-2/DEM parallel development):

    failure_faces.json = {
        "faces": [
            {
                "id": int,
                "area": float,
                "normal": [x, y, z],
                "centroid": [x, y, z],
                "G": float,
                "released_energy": float
            }
        ],
        "units": "SI"
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass(frozen=True)
class FailureCriterion:
    """Tension/shear combined failure criterion.

    Parameters
    ----------
    sigma_max : float
        Tensile strength [Pa].
    tau_max : float
        Shear strength [Pa].
    G : float
        Fracture energy per unit area [J/m^2].
    """
    sigma_max: float
    tau_max: float
    G: float

    def __post_init__(self):
        if self.sigma_max <= 0:
            raise ValueError("sigma_max must be positive")
        if self.tau_max <= 0:
            raise ValueError("tau_max must be positive")
        if self.G <= 0:
            raise ValueError("G must be positive")

    def failure_index(self, sigma_n: float, tau: float) -> float:
        """Compute the failure index f.

        f = (sigma_n / sigma_max)^2 + (tau / tau_max)^2

        Face fails when f >= 1.
        """
        return (sigma_n / self.sigma_max) ** 2 + (tau / self.tau_max) ** 2


def evaluate_face_failure(
    faces: list[dict],
    criterion: FailureCriterion,
) -> list[dict]:
    """Evaluate failure for each face and annotate with G and released_energy.

    Parameters
    ----------
    faces : list of dict
        Face data from ``compute_face_stresses``.
    criterion : FailureCriterion
        Failure criterion parameters.

    Returns
    -------
    failed_faces : list of dict
        Only the faces that have failed, annotated with G and
        released_energy = G * area.
    """
    failed = []
    for face in faces:
        stress = face["stress"]
        normal = face["normal"]

        # Normal stress on the face (tensile positive)
        # sigma_n = sigma . n . n  (projection of stress tensor onto normal)
        # For simplicity, use the normal component of the stress tensor
        # sigma_n = sigma_x * nx^2 + sigma_y * ny^2 + 2*tau_xy*nx*ny
        nx, ny, nz = normal
        sigma_n = (stress["sigma_x"] * nx**2 +
                   stress["sigma_y"] * ny**2 +
                   2 * stress["tau_xy"] * nx * ny)

        # Shear stress magnitude on the face
        # tau = sqrt((sigma_x*nx + tau_xy*ny - sigma_n*nx)^2 +
        #            (tau_xy*nx + sigma_y*ny - sigma_n*ny)^2)
        tx = stress["sigma_x"] * nx + stress["tau_xy"] * ny - sigma_n * nx
        ty = stress["tau_xy"] * nx + stress["sigma_y"] * ny - sigma_n * ny
        tau = (tx**2 + ty**2) ** 0.5

        f_index = criterion.failure_index(sigma_n, tau)

        if f_index >= 1.0:
            released = criterion.G * face["area"]
            failed.append({
                "id": face["id"],
                "area": face["area"],
                "normal": face["normal"],
                "centroid": face["centroid"],
                "G": criterion.G,
                "released_energy": released,
                # Extra diagnostic fields (not in interface contract)
                "_failure_index": f_index,
                "_sigma_n": sigma_n,
                "_tau": tau,
            })

    return failed


def enumerate_failed_faces_analytical(
    n_elements_x: int,
    n_elements_y: int,
    uniform_sigma_n: float,
    uniform_tau: float,
    criterion: FailureCriterion,
) -> tuple[int, float, float]:
    """Analytically enumerate failed faces under uniform stress field.

    For a structured mesh of n_elements_x * n_elements_y quadrilateral
    elements, each element has 4 internal faces.  Under a uniform stress
    field, all faces experience the same stress state, so either all
    fail or none fail.

    Parameters
    ----------
    n_elements_x, n_elements_y : int
        Number of elements in x and y directions.
    uniform_sigma_n : float
        Uniform normal stress [Pa].
    uniform_tau : float
        Uniform shear stress [Pa].
    criterion : FailureCriterion
        Failure criterion parameters.

    Returns
    -------
    n_failed : int
        Number of failed faces.
    total_area : float
        Total area of failed faces.
    total_released_energy : float
        Total released energy = n_failed * G * face_area.
    """
    f_index = criterion.failure_index(uniform_sigma_n, uniform_tau)
    n_total_faces = 4 * n_elements_x * n_elements_y

    if f_index >= 1.0:
        # All faces fail
        # Each face area is side_length * thickness
        # For a unit mesh, each face has area = (1/n_elements) * thickness
        # But we don't know the mesh size here, so we return counts
        return n_total_faces, -1.0, -1.0  # area/energy need mesh info
    else:
        return 0, 0.0, 0.0
