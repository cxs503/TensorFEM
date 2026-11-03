

# ---------------------------------------------------------------------------
# Finite beam on a Winkler foundation: general end conditions (Hetényi BVP)
# ---------------------------------------------------------------------------
#
# :func:`finite_beam_series` is exact for simply supported ends only.  The
# routines below close that documented gap with Hetényi's end-conditioning
# solution of the same ODE ``EI w'''' + k_b w = q`` for arbitrary end
# combinations (free / simply supported / clamped / guided).
#
# Working in the dimensionless state
#
#     z(xi) = [w, w'/lam, w''/lam^2, w'''/lam^3],   xi = lam*x,  lam^4 = k_b/(4EI)
#
# the homogeneous ODE becomes the constant-coefficient system
#
#     dz/dxi = A z,   A = [[0,1,0,0],[0,0,1,0],[0,0,0,1],[-4,0,0,0]]
#
# so any state propagates exactly, ``z(xi) = P(xi) z(0)`` with
# ``P(xi) = exp(A xi)``.  ``P`` is elementary: with the four products
#
#     u1 = cosh(xi)cos(xi),  u2 = cosh(xi)sin(xi),
#     u3 = sinh(xi)cos(xi),  u4 = sinh(xi)sin(xi)
#
# the rows of ``exp(A xi)`` follow from ``u1'' = -2u4``, ``u2'' = 2u3``,
# ``u3'' = -2u2``, ``u4'' = 2u1`` and their derivatives; this is the matrix
# form of Hetényi's end-conditioning coefficient table (Hetényi 1946, ch. II).
#
# Because ``P`` contains ``exp(+lam x)``, a single-ended propagation loses
# accuracy once ``lam*L`` is large (the boundary-value system becomes
# exponentially ill-conditioned).  The solver therefore uses *multiple
# shooting*: the beam is subdivided so that every propagation span is at most
# ``max_span_xi / lam``, and the state at every node is an unknown of a banded
# block system.  The conditioning is then independent of ``lam*L``.
#
# Sign conventions agree with the rest of this module: ``w`` positive
# downward, ``q`` positive downward, ``M = -EI w''`` (so that the infinite
# beam gives ``M = P/(4 lam) B(lam x)``).  The shear is ``V = -EI w'''`` so
# that a downward point load ``P`` produces the physical jump ``Delta V = -P``.

# Component indices of the dimensionless state used by each end condition:
#   free            : M = 0, V = 0   -> w''  = 0, w''' = 0
#   simply supported: w = 0, M = 0   -> w    = 0, w''  = 0
#   clamped         : w = 0, w' = 0  -> w    = 0, w'   = 0
#   guided (symm.)  : w' = 0, V = 0  -> w'   = 0, w''' = 0
_END_CONDITION_ROWS: dict[str, tuple[int, int]] = {
    "free": (2, 3),
    "simply_supported": (0, 2),
    "clamped": (0, 1),
    "guided": (1, 3),
}

_END_CONDITION_ALIASES: dict[str, str] = {
    "free": "free", "free-free": "free", "f": "free",
    "simply_supported": "simply_supported", "simply-supported": "simply_supported",
    "simply supported": "simply_supported", "ss": "simply_supported",
    "pinned": "simply_supported", "simple": "simply_supported",
    "clamped": "clamped", "clamped-clamped": "clamped", "fixed": "clamped",
    "c": "clamped",
    "guided": "guided", "symmetry": "guided",
}

#: End conditions accepted by :func:`finite_beam_bvp`.
FINITE_BEAM_END_CONDITIONS: tuple[str, ...] = tuple(_END_CONDITION_ROWS)

_STATE_BASIS_AT_ORIGIN = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 2.0, -2.0, 0.0],
    ],
    dtype=torch.float64,
)
_STATE_BASIS_AT_ORIGIN_INV = torch.linalg.inv(_STATE_BASIS_AT_ORIGIN)


def _normalise_end_condition(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _END_CONDITION_ALIASES:
        raise ValueError(
            f"unsupported end condition {name!r}; expected one of "
            f"{sorted(set(_END_CONDITION_ALIASES.values()))}"
        )
    return _END_CONDITION_ALIASES[key]


def _state_basis_matrix(xi: torch.Tensor) -> torch.Tensor:
    """Rows are the four homogeneous solutions and their xi-derivatives."""
    u1 = torch.cosh(xi) * torch.cos(xi)
    u2 = torch.cosh(xi) * torch.sin(xi)
    u3 = torch.sinh(xi) * torch.cos(xi)
    u4 = torch.sinh(xi) * torch.sin(xi)
    return torch.stack(
        [
            torch.stack([u1, u2, u3, u4]),
            torch.stack([u3 - u2, u1 + u4, u1 - u4, u2 + u3]),
            torch.stack([-2.0 * u4, 2.0 * u3, -2.0 * u2, 2.0 * u1]),
            torch.stack([-2.0 * (u2 + u3), 2.0 * (u1 - u4), -2.0 * (u1 + u4), 2.0 * (u3 - u2)]),
        ]
    )


def beam_state_propagator(xi, *, dtype=torch.float64) -> torch.Tensor:
    """Exact propagator ``P`` with ``z(xi) = P(xi) @ z(0)``.

    ``P(xi) = exp(A xi)`` for the dimensionless Winkler-beam system, evaluated
    in closed form from the ``u1..u4`` products rather than by integrating the
    ODE, so the accuracy is limited only by floating-point round-off.  It is
    compared against ``torch.matrix_exp`` in the unit tests.
    """
    xi_t = torch.as_tensor(xi, dtype=dtype)
    if torch.any(xi_t < 0):
        raise ValueError("xi must be non-negative")
    return _state_basis_matrix(xi_t) @ _STATE_BASIS_AT_ORIGIN_INV.to(dtype)


@dataclass(frozen=True)
class FiniteBeamSolution:
    """Exact field of a finite beam on a Winkler foundation.

    Attributes
    ----------
    x : evaluation abscissae (m).
    deflection : ``w`` positive downward (m).
    slope : ``dw/dx`` (rad).
    moment : ``M = -EI w''`` (N m).
    shear : ``V = -EI w'''`` (N); a downward point load makes ``V`` jump by ``-P``.
    foundation_reaction : ``k_b w`` (N/m), positive downward.
    state_at_origin : dimensionless state at ``x = 0`` (after any jump there).
    end_conditions : (``end_a``, ``end_b``) after alias normalisation.
    system_condition : condition number of the multiple-shooting system.
    nodes : shooting nodes (m).
    """

    x: torch.Tensor
    deflection: torch.Tensor
    slope: torch.Tensor
    moment: torch.Tensor
    shear: torch.Tensor
    foundation_reaction: torch.Tensor
    state_at_origin: torch.Tensor
    end_conditions: tuple[str, str]
    system_condition: float
    nodes: torch.Tensor


def finite_beam_bvp(
    x_eval: torch.Tensor,
    *,
    beam_length: float,
    lam: float,
    k_b: float,
    EI: float,
    point_loads: list[tuple[float, float]] | tuple = (),
    distributed_loads: list[tuple[float, float, float]] | tuple = (),
    end_a: str = "free",
    end_b: str = "free",
    max_span_xi: float = 1.0,
) -> FiniteBeamSolution:
    """Exact finite beam on a Winkler foundation for arbitrary end conditions.

    Solves ``EI w'''' + k_b w = q`` on ``[0, L]`` with two boundary conditions
    at each end using Hetényi's end-conditioning method in matrix form, i.e. the
    exact propagator of the dimensionless state plus a linear boundary-value
    solve.  This is the reference solution that :func:`finite_beam_series`
    could not provide for free or clamped ends.

    Parameters
    ----------
    x_eval : (N,) evaluation abscissae inside ``[0, L]``.
    beam_length : ``L`` (m).
    lam : ``(k_b / 4EI)^(1/4)`` (1/m).
    k_b : foundation coefficient per unit length (N/m^2).
    EI : bending stiffness (N m^2).
    point_loads : list of ``(x, P)``, ``P`` positive downward (N); positions
        must lie strictly inside the span.
    distributed_loads : list of ``(x0, x1, q)`` with ``q`` positive downward
        (N/m); overlapping segments are summed.
    end_a, end_b : end condition at ``x = 0`` / ``x = L``: ``'free'``,
        ``'simply_supported'``, ``'clamped'`` or ``'guided'`` (aliases
        ``'ss'``, ``'pinned'``, ``'fixed'``, ``'symmetry'`` are accepted).
    max_span_xi : largest ``lam * (node spacing)`` used by the multiple
        shooting; the default 1.0 keeps the linear system well conditioned.

    Returns
    -------
    :class:`FiniteBeamSolution`
    """
    L = float(beam_length)
    lam = float(lam)
    k_b = float(k_b)
    EI = float(EI)
    for name, value in (("beam_length", L), ("lam", lam), ("k_b", k_b), ("EI", EI)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    if not math.isfinite(max_span_xi) or max_span_xi <= 0.0:
        raise ValueError(f"max_span_xi must be positive and finite, got {max_span_xi!r}")
    end_a_c = _normalise_end_condition(end_a)
    end_b_c = _normalise_end_condition(end_b)

    dtype = x_eval.dtype
    loads = [(float(x), float(P)) for x, P in point_loads]
    dists = [(float(x0), float(x1), float(q)) for x0, x1, q in distributed_loads]
    for x, _ in loads:
        if not (0.0 < x < L):
            raise ValueError(f"point load at x={x!r} must lie strictly inside (0, {L})")
    for x0, x1, _ in dists:
        if not (0.0 <= x0 < x1 <= L):
            raise ValueError(f"distributed load span ({x0!r}, {x1!r}) must be inside [0, {L}]")
    if torch.any(x_eval < 0) or torch.any(x_eval > L):
        raise ValueError("x_eval must lie inside [0, beam_length]")

    # --- shooting nodes: load stations plus uniform subdivision -------------
    base = sorted({0.0, L} | {x for x, _ in loads} | {x0 for x0, _, _ in dists} | {x1 for _, x1, _ in dists})
    nodes: list[float] = []
    for a, b in zip(base[:-1], base[1:]):
        n = max(1, int(math.ceil(lam * (b - a) / max_span_xi)))
        for j in range(1, n + 1):
            nodes.append(b if j == n else a + (b - a) * j / n)
    n_seg = len(nodes) - 1

    jump = torch.zeros((n_seg + 1, 4), dtype=dtype)
    for x, P in loads:
        idx = min(range(len(nodes)), key=lambda i: abs(nodes[i] - x))
        jump[idx, 3] = jump[idx, 3] + P / (EI * lam ** 3)

    seg_q = []
    for i in range(n_seg):
        mid = 0.5 * (nodes[i] + nodes[i + 1])
        seg_q.append(sum(q for x0, x1, q in dists if x0 <= mid <= x1))
    zp = [torch.tensor([q / k_b, 0.0, 0.0, 0.0], dtype=dtype) for q in seg_q]
    props = [beam_state_propagator(lam * (nodes[i + 1] - nodes[i]), dtype=dtype) for i in range(n_seg)]

    # --- multiple-shooting system: unknowns are the states at the nodes ----
    n_unknown = 4 * (n_seg + 1)
    matrix = torch.zeros((n_unknown, n_unknown), dtype=dtype)
    rhs = torch.zeros(n_unknown, dtype=dtype)
    row = 0
    for comp in _END_CONDITION_ROWS[end_a_c]:
        matrix[row, comp] = 1.0
        row += 1
    for i in range(n_seg):
        block_lo = 4 * i
        nxt = 4 * (i + 1)
        matrix[row:row + 4, block_lo:block_lo + 4] = -props[i]
        matrix[row:row + 4, nxt:nxt + 4] = torch.eye(4, dtype=dtype)
        rhs[row:row + 4] = (-props[i] @ zp[i] + zp[i]) + jump[i + 1]
        row += 4
    for comp in _END_CONDITION_ROWS[end_b_c]:
        matrix[row, 4 * n_seg + comp] = 1.0
        row += 1
    system_condition = float(torch.linalg.cond(matrix))
    solution = torch.linalg.solve(matrix, rhs)
    states = solution.reshape(n_seg + 1, 4)

    # --- evaluate the field -----------------------------------------------
    w = torch.empty_like(x_eval)
    slope = torch.empty_like(x_eval)
    wpp = torch.empty_like(x_eval)
    wppp = torch.empty_like(x_eval)
    node_t = torch.as_tensor(nodes[1:-1], dtype=dtype)
    idx = torch.searchsorted(node_t, x_eval, right=True)
    for e, (xe, k) in enumerate(zip(x_eval.tolist(), idx.tolist())):
        prop = beam_state_propagator(lam * (xe - nodes[k]), dtype=dtype)
        zx = prop @ (states[k] - zp[k]) + zp[k]
        w[e] = zx[0]
        slope[e] = lam * zx[1]
        wpp[e] = lam * lam * zx[2]
        wppp[e] = lam ** 3 * zx[3]
    return FiniteBeamSolution(
        x=x_eval.clone(),
        deflection=w,
        slope=slope,
        moment=-EI * wpp,
        shear=-EI * wppp,
        foundation_reaction=k_b * w,
        state_at_origin=states[0].clone(),
        end_conditions=(end_a_c, end_b_c),
        system_condition=system_condition,
        nodes=torch.as_tensor(nodes, dtype=dtype),
    )


def foundation_reaction_integral(solution: FiniteBeamSolution) -> float:
    """Trapezoidal ``integral k_b w dx`` of a :class:`FiniteBeamSolution`.

    For a beam free at both ends this equals the total applied load, so it is an
    exact global-equilibrium oracle (trapezoidal quadrature is the only error
    source).
    """
    return float(torch.trapezoid(solution.foundation_reaction, solution.x))
