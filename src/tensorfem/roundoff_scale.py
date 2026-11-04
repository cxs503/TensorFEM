"""Objective tolerances for "zero to within round-off" assertions.

Some finite-element qualifications assert that a quantity *is zero*, for
example that a consistently warped but stress-free reference configuration
produces no internal force.  Such an assertion cannot be written against an
absolute constant: the assembled value is a sum of ``O(ops)`` floating-point
terms of magnitude ``scale``, so the attainable floor is

    floor ~= ops * eps * scale

with ``eps`` the machine epsilon of the working dtype.  An absolute threshold
therefore encodes the CPU/BLAS build that produced it.  The historical
layered-shell assertion used ``1e-10`` (Pa, or N) on a model whose own term
scale is ``E = 2e5``; the measured floor varied between ``0.9`` and ``3.6``
times ``eps * E`` across coordinate origins (``docs/layered-shell4-roundoff-floor.md``),
so the constant sat inside the noise band and the test was recorded as
machine dependent (issue #8).

This module replaces such constants by a scale-relative bound:

    roundoff_bound(scale) == DEFAULT_OPS * eps * scale

``DEFAULT_OPS = 1024`` keeps a margin of at least 40x over the largest measured
amplification (23x) while staying at least nine orders of magnitude below any
physically meaningful inconsistency, so the check still fails loudly on a real
formulation error.  ``scale`` must be the magnitude of the largest term that
enters the quantity -- ``E`` for a stress, an ``E * t * L`` product for a
force -- never the value under test.
"""
from __future__ import annotations

import math

import torch

#: Operation budget of one assembled quantity.  See the module docstring for
#: how the value is justified by the measured amplification factors.
DEFAULT_OPS = 1024


def machine_epsilon(dtype=torch.float64) -> float:
    """Unit round-off ``eps`` of ``dtype`` as a Python float."""
    return float(torch.finfo(dtype).eps)


def roundoff_bound(scale, dtype=torch.float64, *, ops: int = DEFAULT_OPS) -> float:
    """Absolute tolerance ``ops * eps(dtype) * scale`` for a zero assertion.

    ``scale`` is the magnitude of the largest term entering the tested
    quantity.  It must be finite and strictly positive; a zero or non-finite
    scale means the caller has not identified the term scale and would
    otherwise get a meaningless (infinitely tight or loose) tolerance.
    """
    value = float(scale)
    if not math.isfinite(value):
        raise ValueError("scale must be finite")
    if value <= 0.0:
        raise ValueError("scale must be positive")
    if int(ops) != ops or ops < 1:
        raise ValueError("ops must be an integer >= 1")
    return float(ops) * machine_epsilon(dtype) * value


def roundoff_ratio(value, scale, dtype=torch.float64, *, ops: int = DEFAULT_OPS) -> float:
    """``|value| / roundoff_bound(scale)``; ``<= 1`` means "zero to round-off".

    Reporting the ratio instead of a boolean keeps the margin visible in
    evidence tables: a ratio of ``0.01`` shows how much headroom a build still
    has before the bound would start to depend on the platform.
    """
    bound = roundoff_bound(scale, dtype, ops=ops)
    return abs(float(value)) / bound