"""Third patch: fix element-u assembly and the long/short side targets."""
from __future__ import annotations

import pathlib

path = pathlib.Path("src/tensorfem/torsion_rectangular.py")
text = path.read_text()

old = '''    element_u = displacement[torch.stack(tuple(3 * cell + i for i in range(3)), 0).reshape(24)]'''
new = '''    element_u = displacement[torch.stack(tuple(3 * cell + i for i in range(3)), 1).reshape(24)]'''
assert old in text
text = text.replace(old, new, 1)

old_targets = '''    long_target = (b - b / solve.n_short / 2.0, a / 2.0)
    short_target = (b / 2.0, a - a / solve.n_long / 2.0)
    return SurfacePeak(
        a=a, b=b, segments_short=solve.n_short, segments_long=solve.n_long,
        n_dofs=3 * len(nodes), end_traction=end_traction,
        exact_long=exact_long, exact_short=exact_short,
        direct_long=direct((b, a / 2.0), "long"), direct_short=direct((b / 2.0, a), "short"),
        gauss_long=gap((0.0, 0.0, 0.0), long_target, axis=0),
        gauss_short=gap((0.0, 0.0, 0.0), short_target, axis=1),
        centroid_long=centroid_value(long_target), centroid_short=centroid_value(short_target),
    )'''
new_targets = '''    # The long side is the one of length ``a`` (along y, at y = 0 and y = a); the peak
    # sits at its midpoint (x, y) = (b/2, a), in the module's centred convention
    # (0, +a/2).  The short side is at x = b, midpoint (b, a/2).
    long_target = (b / 2.0, a - a / solve.n_long / 2.0)
    short_target = (b - b / solve.n_short / 2.0, a / 2.0)
    return SurfacePeak(
        a=a, b=b, segments_short=solve.n_short, segments_long=solve.n_long,
        n_dofs=3 * len(nodes), end_traction=end_traction,
        exact_long=exact_long, exact_short=exact_short,
        direct_long=direct((b / 2.0, a), "long"), direct_short=direct((b, a / 2.0), "short"),
        gauss_long=gap((0.0, 0.0, 0.0), long_target, axis=1),
        gauss_short=gap((0.0, 0.0, 0.0), short_target, axis=0),
        centroid_long=centroid_value(long_target), centroid_short=centroid_value(short_target),
    )'''
assert old_targets in text
text = text.replace(old_targets, new_targets, 1)
path.write_text(text)
print("patched")