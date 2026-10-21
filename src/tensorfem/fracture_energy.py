"""Fracture energy ledger and mass audit interface.

This module maintains a ledger of released fracture energy and provides
a mass audit interface for tracking energy conservation during cohesive
failure events.

The ledger tracks:
- Cumulative released energy from failed faces
- Individual face failure events
- Energy balance checks (released vs. dissipated)

Interface contract (frozen for SI-2/DEM parallel development):

    The ledger can be serialized to JSON with the following structure:
    {
        "total_released_energy": float,
        "n_failed_faces": int,
        "faces": [
            {
                "id": int,
                "area": float,
                "G": float,
                "released_energy": float
            }
        ],
        "units": "SI"
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FractureEnergyLedger:
    """Ledger for tracking fracture energy release.

    This ledger accumulates released energy from failed faces and
    provides audit interfaces for energy conservation checks.
    """
    total_released_energy: float = 0.0
    n_failed_faces: int = 0
    failed_faces: list = field(default_factory=list)

    def add_failed_face(
        self,
        face_id: int,
        area: float,
        G: float,
        released_energy: float,
    ) -> None:
        """Record a failed face event.

        Parameters
        ----------
        face_id : int
            Unique identifier for the failed face.
        area : float
            Face area [m^2].
        G : float
            Fracture energy per unit area [J/m^2].
        released_energy : float
            Released energy for this face [J].
        """
        if released_energy < 0:
            raise ValueError("released_energy must be non-negative")

        self.total_released_energy += released_energy
        self.n_failed_faces += 1
        self.failed_faces.append({
            "id": face_id,
            "area": area,
            "G": G,
            "released_energy": released_energy,
        })

    def add_failed_faces_batch(self, failed_faces: list[dict]) -> None:
        """Record multiple failed faces from cohesive failure evaluation.

        Parameters
        ----------
        failed_faces : list of dict
            Output from ``evaluate_face_failure``.
        """
        for face in failed_faces:
            self.add_failed_face(
                face_id=face["id"],
                area=face["area"],
                G=face["G"],
                released_energy=face["released_energy"],
            )

    def verify_energy_conservation(
        self,
        expected_released_energy: float,
        tol: float = 1e-12,
    ) -> bool:
        """Verify that total released energy matches expected value.

        Parameters
        ----------
        expected_released_energy : float
            Expected total released energy [J].
        tol : float
            Absolute tolerance for comparison.

        Returns
        -------
        passed : bool
            True if |total - expected| <= tol.
        """
        return abs(self.total_released_energy - expected_released_energy) <= tol

    def verify_energy_sum(
        self,
        tol: float = 1e-12,
    ) -> bool:
        """Verify that total equals sum of individual face energies.

        Parameters
        ----------
        tol : float
            Absolute tolerance for comparison.

        Returns
        -------
        passed : bool
            True if |total - sum(individual)| <= tol.
        """
        individual_sum = sum(f["released_energy"] for f in self.failed_faces)
        return abs(self.total_released_energy - individual_sum) <= tol

    def to_dict(self) -> dict:
        """Serialize ledger to dictionary.

        Returns
        -------
        data : dict
            Ledger data in the interface contract format.
        """
        return {
            "total_released_energy": self.total_released_energy,
            "n_failed_faces": self.n_failed_faces,
            "faces": [
                {
                    "id": f["id"],
                    "area": f["area"],
                    "G": f["G"],
                    "released_energy": f["released_energy"],
                }
                for f in self.failed_faces
            ],
            "units": "SI",
        }

    def to_json(self, path: str) -> None:
        """Write ledger to JSON file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "FractureEnergyLedger":
        """Deserialize ledger from dictionary.

        Parameters
        ----------
        data : dict
            Ledger data in the interface contract format.

        Returns
        -------
        ledger : FractureEnergyLedger
            Reconstructed ledger.
        """
        ledger = cls()
        ledger.total_released_energy = data["total_released_energy"]
        ledger.n_failed_faces = data["n_failed_faces"]
        ledger.failed_faces = [
            {
                "id": f["id"],
                "area": f["area"],
                "G": f["G"],
                "released_energy": f["released_energy"],
            }
            for f in data["faces"]
        ]
        return ledger

    @classmethod
    def from_json(cls, path: str) -> "FractureEnergyLedger":
        """Load ledger from JSON file.

        Parameters
        ----------
        path : str
            Input file path.

        Returns
        -------
        ledger : FractureEnergyLedger
            Loaded ledger.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class MassAuditLedger:
    """Mass audit ledger for tracking energy-mass equivalence.

    This ledger tracks the mass equivalent of released fracture energy
    via E = m * c^2, providing a sanity check for energy conservation.

    Note: In structural mechanics, the mass equivalent of fracture energy
    is typically negligible (~1e-17 kg for 1 J), but this ledger provides
    the interface for completeness.
    """
    c_squared: float = 8.987551787368176e16  # c^2 in m^2/s^2
    total_mass_equivalent: float = 0.0
    n_entries: int = 0

    def add_energy_release(self, energy: float) -> None:
        """Record an energy release event.

        Parameters
        ----------
        energy : float
            Released energy [J].
        """
        if energy < 0:
            raise ValueError("energy must be non-negative")

        mass = energy / self.c_squared
        self.total_mass_equivalent += mass
        self.n_entries += 1

    def verify_mass_conservation(
        self,
        expected_mass: float,
        tol: float = 1e-12,
    ) -> bool:
        """Verify that total mass equivalent matches expected value.

        Parameters
        ----------
        expected_mass : float
            Expected total mass equivalent [kg].
        tol : float
            Absolute tolerance for comparison.

        Returns
        -------
        passed : bool
            True if |total - expected| <= tol.
        """
        return abs(self.total_mass_equivalent - expected_mass) <= tol

    def to_dict(self) -> dict:
        """Serialize ledger to dictionary."""
        return {
            "total_mass_equivalent": self.total_mass_equivalent,
            "n_entries": self.n_entries,
            "c_squared": self.c_squared,
            "units": "SI",
        }
