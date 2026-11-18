"""Rate-specific, fully plastic spherical indentation of an ice coupon.

The constant bearing pressure multiplies the projected contact area of the
spherical cap (Liefferink et al., Phys. Rev. X 11, 011025). This limiting
law describes monotonic loading; the irreversible work assumes rigid unloading.
"""

import math


def spherical_bearing_load(depth, radius, hardness):
    if not all(math.isfinite(x) for x in (depth, radius, hardness)):
        raise ValueError("finite depth, radius and hardness required")
    if not 0 <= depth <= radius or min(radius, hardness) <= 0:
        raise ValueError("positive radius and hardness, 0 <= depth <= radius required")
    return math.pi * hardness * depth * (2 * radius - depth)


def spherical_bearing_work(depth, radius, hardness):
    """Integral of load with depth, in J for SI inputs."""
    spherical_bearing_load(depth, radius, hardness)
    return math.pi * hardness * depth**2 * (radius - depth / 3)
