"""Pre-commit admissibility guard in run_patch_plastic_path."""
import pathlib

P = pathlib.Path("src/tensorfem/plate_patch_plastic_limit.py")
s = P.read_text()

old = """    path = PatchPath(model=model, info=info)
    state = LayeredShell4State.virgin(model)
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype)
    centre = _centre_node(info)

    def advance(target: float, current: float, history: list[PatchPathPoint]) -> float | None:
        nonlocal state, u
        stack = [(current, float(target))]
        while stack:
            start, end = stack.pop()
            try:
                step = solve_layered_shell4_increment(
                    model, info["reference_load"], end, info["fixed_dofs"], state, u,
                    tolerance=tolerance, max_iterations=max_iterations,
                    geometric=geometric)
            except RuntimeError:
                if (end - start) < bisection_tolerance * max(abs(end), 1e-12):
                    return None
                mid = 0.5 * (start + end)
                stack.append((mid, end))
                stack.append((start, mid))
                continue
            state, u = step.state, step.displacement
            history.append(PatchPathPoint(
                end, float(u[6 * centre + 2]), float(torch.max(step.alpha)),
                surface_extrapolated_mises(step.stress, model.thickness),
                max_section_moment_ratio(step.stress, model.thickness,
                                         model.yield_stress, model.layers)))
            current = end
        return current
"""

new = """    path = PatchPath(model=model, info=info)
    state = LayeredShell4State.virgin(model)
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype)
    centre = _centre_node(info)
    capacity_ratio = section_moment_capacity(
        model.thickness, model.yield_stress, model.layers)["ratio"]

    def advance(target: float, current: float, history: list[PatchPathPoint]) -> float | None:
        nonlocal state, u
        stack = [(current, float(target))]
        while stack:
            start, end = stack.pop()
            try:
                step = solve_layered_shell4_increment(
                    model, info["reference_load"], end, info["fixed_dofs"], state, u,
                    tolerance=tolerance, max_iterations=max_iterations,
                    geometric=geometric)
            except RuntimeError:
                if (end - start) < bisection_tolerance * max(abs(end), 1e-12):
                    return None
                mid = 0.5 * (start + end)
                stack.append((mid, end))
                stack.append((start, mid))
                continue
            # The state is validated *before* it is committed.  A step that
            # leaves the section (|M|/m_p above the layer capacity) or whose
            # deflection runs away is a Newton artefact, not an equilibrium
            # state; committing it puts a non-physical point into the history,
            # where the limit-load extractor would consume it (measured on the
            # anchored ladder: 0.0696 m -> 133.5 m, |M|/m_p = 1.077; the
            # committed point produced a +20.5 % "limit load").
            w_new = float(step.displacement[6 * centre + 2])
            moment_ratio = max_section_moment_ratio(
                step.stress, model.thickness, model.yield_stress, model.layers)
            rejection = _step_rejection(w_new, moment_ratio, end, history,
                                        capacity_ratio, max_deflection_factor)
            if rejection is not None:
                path.ramp_truncated = True
                path.truncation_reason = rejection
                return current
            state, u = step.state, step.displacement
            history.append(PatchPathPoint(
                end, w_new, float(torch.max(step.alpha)),
                surface_extrapolated_mises(step.stress, model.thickness),
                moment_ratio))
            current = end
        return current
"""
assert s.count(old) == 1
s = s.replace(old, new)

# helper before run_patch_plastic_path
anchor = "def run_patch_plastic_path(a: float, b: float, t: float, *, young: float,"
helper = '''def _step_rejection(w_new: float, moment_ratio: float, load_factor: float,
                    history: list["PatchPathPoint"], capacity_ratio: float,
                    max_deflection_factor: float | None) -> str | None:
    """Reason to refuse committing a converged step, or ``None`` if it is sound.

    The layer model cannot leave the section shell ``|M| <= m_p,layers`` and a
    small-rotation plate cannot gain an order of magnitude of deflection in one
    load step, so either event means the Newton iterate is not an equilibrium
    state of this model.
    """
    if not (math.isfinite(w_new) and math.isfinite(moment_ratio)):
        return "non-finite state in the increment"
    if moment_ratio > capacity_ratio * (1.0 + 1e-6):
        return (f"section capacity exceeded: |M|/m_p = {moment_ratio:.6f} > "
                f"capacity {capacity_ratio:.6f} at p = {load_factor:.6g}")
    if not history:
        return None
    previous = history[-1]
    if w_new <= previous.centre_deflection:
        return (f"centre deflection does not increase at p = {load_factor:.6g} "
                f"({previous.centre_deflection:.6g} -> {w_new:.6g} m)")
    amplification = ((w_new / previous.centre_deflection)
                     / (load_factor / previous.load_factor))
    if amplification > 10.0:
        return ("deflection runaway: x%.1f amplification in one load step "
                "(p = %.6g -> %.6g MPa, w = %.6g -> %.6g m)"
                % (amplification, previous.load_factor / 1e6, load_factor / 1e6,
                   previous.centre_deflection, w_new))
    if max_deflection_factor is not None:
        elastic = load_factor * previous.centre_deflection / previous.load_factor
        if w_new > max_deflection_factor * elastic:
            return ("small-rotation guard: w = %.6g m exceeds %.3g x the elastic "
                    "prediction (%.6g m) at p = %.6g MPa"
                    % (w_new, max_deflection_factor, elastic, load_factor / 1e6))
    return None


'''
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor)

# break the outer load loop on a pre-commit truncation
old = """        reached = advance(target, previous, path.loading)
        if reached is None:
            path.collapse_factor = previous
            path.limit_point_searched = True
            break
        previous = reached
        if max_deflection_factor is not None and len(path.loading) > 1:
"""
new = """        reached = advance(target, previous, path.loading)
        if reached is None:
            path.collapse_factor = previous
            path.limit_point_searched = True
            break
        if path.ramp_truncated:
            # the increment was rejected before commit: stop, do not append
            previous = reached
            break
        previous = reached
        if max_deflection_factor is not None and len(path.loading) > 1:
"""
assert s.count(old) == 1
s = s.replace(old, new)

P.write_text(s)
print("patched path integrator")