"""Transient corotational cantilever strip, per metre of out-of-plane span.

The material axis initially points up (+y), and positive transverse displacement
points downstream (+x). Interface stations follow the element chords. The
transpose of this interpolation distributes integrated point forces, preserving
their resultant, moment about the clamp and discrete virtual work.
"""
from __future__ import annotations

import math

import torch

from .elastica_cantilever import _beam_energy


class DynamicStrip2D:
    """Free-ended Euler–Bernoulli strip, clamped at (0, 0), CPU float64.

    Uses the existing large-rotation strain energy, lumped translational and
    cross-sectional rotary inertia, and subcycled symplectic Euler. Forces are
    held fixed over each coupling step. Units are SI, with a one-metre span.
    """

    def __init__(self, *, length: float, thickness: float, young: float,
                 density: float, elements: int):
        if not all(math.isfinite(p) and p > 0 for p in (length, thickness, young, density)):
            raise ValueError("strip dimensions and material properties must be positive and finite")
        if not isinstance(elements, int) or elements < 1:
            raise ValueError("elements must be a positive integer")
        self.length = length
        self.elements = elements
        self.spacing = length / elements
        self.reference = torch.zeros((elements + 1, 2), dtype=torch.float64)
        self.reference[:, 0] = torch.linspace(0, length, elements + 1, dtype=torch.float64)
        self.q = torch.zeros((elements + 1, 3), dtype=torch.float64)
        self.velocity = torch.zeros_like(self.q)
        self.lengths = self.reference[:, 0].diff()
        self.ea = young * thickness
        self.ei = young * thickness**3 / 12
        self.mass = torch.full_like(self.q, density * thickness * self.spacing)
        self.mass[[0, -1]] *= 0.5
        self.mass[:, 2] *= thickness**2 / 12
        wave_speed = math.sqrt(young / density)
        self.max_dt = 0.2 * min(self.spacing, self.spacing**2 / thickness) / wave_speed

    def station_weights(self, arcs) -> torch.Tensor:
        """Shape matrix at undeformed distances from the clamp, in metres."""
        s = torch.as_tensor(arcs, dtype=torch.float64, device="cpu")
        if s.ndim != 1 or not bool(torch.all(torch.isfinite(s) & (s >= 0) & (s <= self.length))):
            raise ValueError("station arcs must be finite and lie on the strip")
        local = s / self.spacing
        left = local.floor().long().clamp(max=self.elements - 1)
        fraction = (local - left).clamp(0, 1)
        weights = s.new_zeros(len(s), self.elements + 1)
        rows = torch.arange(len(s))
        weights[rows, left] = 1 - fraction
        weights[rows, left + 1] = fraction
        return weights

    def sample(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x,y) positions relative to the clamp and (vx,vy) velocities."""
        return (weights @ (self.reference + self.q[:, :2]).flip(1),
                weights @ self.velocity[:, :2].flip(1))

    def advance(self, weights: torch.Tensor, forces_xy: torch.Tensor, dt: float) -> None:
        """Advance by dt seconds under station forces [N for the unit span]."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        f = torch.as_tensor(forces_xy, dtype=torch.float64, device="cpu")
        if f.shape != (len(weights), 2) or not bool(f.isfinite().all()):
            raise ValueError("expected finite (station, 2) integrated forces")
        load = torch.zeros_like(self.q)
        load[:, :2] = weights.T @ f.flip(1)
        count = math.ceil(dt / self.max_dt)
        step = dt / count
        q, v = self.q, self.velocity
        for _ in range(count):
            with torch.enable_grad():
                q_eval = q.detach().requires_grad_(True)
                energy = _beam_energy(self.reference, q_eval, self.ea, self.ei, self.lengths)
                internal, = torch.autograd.grad(energy, q_eval)
            acceleration = (load - internal) / self.mass
            acceleration[0] = 0
            v = v + step * acceleration
            q = q + step * v
        if not bool(torch.isfinite(q).all() & torch.isfinite(v).all()):
            raise FloatingPointError("non-finite structural step")
        self.q, self.velocity = q, v
