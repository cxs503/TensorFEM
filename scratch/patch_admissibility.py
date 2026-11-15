"""Apply the path-admissibility patch to tensorfem.plate_patch_plastic_limit."""
import pathlib

P = pathlib.Path("src/tensorfem/plate_patch_plastic_limit.py")
s = P.read_text()

# ---------------------------------------------------------------- 1. PatchPath
old = """    #: ramp stopped early by ``max_deflection_factor`` (flat post-saturation path)
    ramp_truncated: bool = False
"""
new = """    #: ramp stopped early by ``max_deflection_factor`` (flat post-saturation path)
    ramp_truncated: bool = False
    #: why the ramp was truncated (``None`` when the ramp ran to its end)
    truncation_reason: str | None = None
"""
assert s.count(old) == 1
s = s.replace(old, new)

# ------------------------------------------------- 2. admissibility gate helper
anchor = "def limit_load_from_path(path: \"PatchPath\", *,"
gate = '''def section_capacity_ratio_of(path: "PatchPath") -> float | None:
    """Section capacity ``m_p,layers / m_p`` of the path's own model.

    ``None`` when the path carries no model (synthetic paths in the tests).
    """
    model = getattr(path, "model", None)
    if model is None:
        return None
    try:
        return section_moment_capacity(model.thickness, model.yield_stress,
                                      model.layers)["ratio"]
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
        return None


def path_admissibility(path: "PatchPath", *,
                       capacity_ratio: float | None = None,
                       deflection_runaway_ratio: float = 10.0,
                       tolerance: float = 1e-6) -> dict:
    """Physical-admissibility gate for a load-deflection path.

    Why this exists
    ---------------
    :func:`limit_load_from_path` fits two tangent lines to *whatever* it is
    given, including a path whose last step is not an equilibrium state of the
    layer model at all.  Two such states were measured on the anchored
    ice-structure ladder (``artifacts/patch_plate_plastic_ladder_nx12_free.json``,
    queue job ``cj3424``):

    * the peak section-moment invariant exceeded the section capacity
      (``|M|/m_p = 1.077`` at ``n = 4`` layers, whose capacity ratio is exactly
      ``1.0``) -- a converged return mapping cannot leave the section;
    * the centre deflection amplified ``1.9e3`` times in one load step
      (``0.0696 m -> 133.5 m`` on a 20 mm plate, ``max alpha = 35.4``), i.e. the
      Newton solver "converged" onto a non-equilibrium branch.

    On that path the two-line rule still returned ``1.825 MPa`` with a
    plausible 6.8 % criteria spread; removing the bad point makes the same rule
    return a *negative* limit load.  The reported number was therefore produced
    entirely by the inadmissible point -- fail-open.  This gate lets the callers
    (and :func:`limit_load_from_path`) withhold the number instead.

    Returns ``{"admissible": bool, "violations": [str, ...],
    "capacity_ratio": float|None, "max_moment_ratio": float|None}``.
    """
    points = [q for q in path.loading if q.load_factor > 0.0]
    violations: list[str] = []
    capacity = capacity_ratio if capacity_ratio is not None else section_capacity_ratio_of(path)
    peak_moment: float | None = None
    previous: PatchPathPoint | None = None
    for q in points:
        numbers = (q.load_factor, q.centre_deflection, q.max_alpha, q.max_moment_ratio)
        if not all(math.isfinite(v) for v in numbers):
            violations.append(
                f"non-finite step at p = {q.load_factor:.6g}")
            continue
        if q.max_moment_ratio > (capacity if capacity is not None else 1.0) * (1.0 + tolerance):
            violations.append(
                f"section capacity exceeded: |M|/m_p = {q.max_moment_ratio:.6f} > "
                f"capacity {capacity if capacity is not None else 1.0:.6f} "
                f"at p = {q.load_factor:.6g}")
        peak_moment = (q.max_moment_ratio if peak_moment is None
                       else max(peak_moment, q.max_moment_ratio))
        if previous is not None:
            if q.centre_deflection <= previous.centre_deflection:
                violations.append(
                    f"centre deflection does not increase at p = {q.load_factor:.6g} "
                    f"({previous.centre_deflection:.6g} -> {q.centre_deflection:.6g} m)")
            else:
                amplification = ((q.centre_deflection / previous.centre_deflection)
                                 / (q.load_factor / previous.load_factor))
                if amplification > deflection_runaway_ratio:
                    violations.append(
                        "deflection runaway: x%.1f amplification in one load step "
                        "(p = %.6g -> %.6g MPa, w = %.6g -> %.6g m)"
                        % (amplification, previous.load_factor / 1e6,
                           q.load_factor / 1e6, previous.centre_deflection,
                           q.centre_deflection))
        previous = q
    return {"admissible": not violations, "violations": violations,
            "capacity_ratio": capacity, "max_moment_ratio": peak_moment}


'''
assert s.count(anchor) == 1
s = s.replace(anchor, gate + anchor)

# --------------------------------------------------- 3. gate inside the extractor
old = """    points = [q for q in path.loading if q.load_factor > 0.0 and q.centre_deflection > 0.0]
    if len(points) < minimum_points:
        raise ValueError("need at least three loading points with positive deflection")
"""
new = """    points = [q for q in path.loading if q.load_factor > 0.0 and q.centre_deflection > 0.0]
    if len(points) < minimum_points:
        raise ValueError("need at least three loading points with positive deflection")
    admissibility = path_admissibility(path)
"""
assert s.count(old) == 1
s = s.replace(old, new)

old = """    out: dict = {"elastic_slope": k0, "final_slope": k1,
                 "peak_factor": last.load_factor,
                 "peak_deflection": last.centre_deflection,
                 "points": len(points), "measurements": {}}
    measurements = out["measurements"]
"""
new = """    out: dict = {"elastic_slope": k0, "final_slope": k1,
                 "peak_factor": last.load_factor,
                 "peak_deflection": last.centre_deflection,
                 "points": len(points), "measurements": {},
                 "admissibility": admissibility}
    measurements = out["measurements"]
    if not admissibility["admissible"]:
        # Fail-closed: a limit load fitted to a non-equilibrium path is not a
        # limit load.  Withhold every measurement and say why.
        out["limit_load"] = None
        out["spread"] = None
        out["reason"] = "path inadmissible: " + "; ".join(admissibility["violations"])
        return out
"""
assert s.count(old) == 1
s = s.replace(old, new)

# ------------------------------------ 4. per-measurement validity + clamp to range
old = """    values = list(measurements.values())
    out["limit_load"] = measurements.get("intersection_of_tangents") or (
        min(values) if values else None)
"""
new = """    # A limit load must lie strictly inside the ramp: the two-line rule returns
    # a *negative* number as soon as the final branch is a stiffening one (the
    # intersection of two stiffening lines is to the left of the origin), and a
    # measurement above the peak is an extrapolation, not a measurement.
    rejected = {key: value for key, value in measurements.items()
                if not (0.0 < value <= last.load_factor)}
    for key in rejected:
        measurements.pop(key)
    out["rejected_measurements"] = rejected
    values = list(measurements.values())
    if not values:
        out["limit_load"] = None
        out["spread"] = None
        out["reason"] = (
            "no admissible limit-load definition fired: the load-deflection path "
            f"does not flatten (final / elastic slope = {k1 / k0:.3f}; a collapse "
            "branch needs a slope well below the elastic one)")
        return out
    out["limit_load"] = measurements.get("intersection_of_tangents") or min(values)
"""
assert s.count(old) == 1
s = s.replace(old, new)

P.write_text(s)
print("patched", P)