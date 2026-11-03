

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
# so the state propagates exactly, ``z(xi) = P(xi) z(0)`` with
# ``P(xi) = exp(A xi)``.  ``P`` is elementary: with the four products
#
#     u1 = cosh(xi)cos(xi),  u2 = cosh(xi)sin(xi),
#     u3 = sinh(xi)cos(xi),  u4 = sinh(xi)sin(xi)
#
# the columns of ``exp(A xi)`` row-wise are ``[u1,u2,u3,u4]`` and its
# derivatives (``u1'' = -2u4``, ``u2'' = 2u3``, ...), i.e. the matrix form of
# Hetényi's end-conditioning coefficient table (Hetényi 1946, ch. II).
#
# Sign conventions agree with the rest of this module: ``w`` positive
# downward, ``q`` positive downward, ``M = -EI w''`` (so that the infinite
# beam gives ``M = P/(4 lam) B(lam x)``).  The shear is ``V = -EI w'''`` so
# that a downward point load ``P`` produces the physical jump ``Delta V = -P``.

# Component indices of the dimensionless state used by each end condition:
#   free            : M = 0, V = 0            -> w'' = 0, w''' = 0
#   simply supported: w = 0, M = 0            -> w   = 0, w''  = 0
#   clamped         : w = 0, w' = 0           -> w   = 0, w'   = 0
#   guided (symm.)  : w' = 0, V = 0           -> w'  = 0, w''' = 0
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
    """Columns are the four homogeneous solutions and their xi-derivatives."""
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

    ``P(xi) = exp(A xi)`` for the dimensionless Winkler-beam system; it is
    evaluated in closed form from the ``u1..u4`` products, not by integrating
    the ODE, so the accuracy is limited only by float round-off.  The result is
    compared with ``torch.matrix_exp`` in the unit tests.
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
    moment : ``M = -EI w''`` (N m), positive sagging.
    shear : ``V = -EI w'''`` (N), so a downward point load jumps ``V`` by ``-P``.
    foundation_reaction : ``k_b w`` (N/m), positive downward.
    state_at_origin : dimensionless state at ``x = 0``.
    end_conditions : (``end_a``, ``end_b``) after alias normalisation.
    """

    x: torch.Tensor
    deflection: torch.Tensor
    slope: torch.Tensor
    moment: torch.Tensor
    shear: torch.Tensor
    foundation_reaction: torch.Tensor
    state_at_origin: torch.Tensor
    end_conditions: tuple[str, str]


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
) -> FiniteBeamSolution:
    """Exact finite beam on a Winkler foundation for arbitrary end conditions.

    Solves ``EI w'''' + k_b w = q`` on ``[0, L]`` with two boundary conditions
    at each end, by propagating the exact state from ``x = 0`` and solving the
    resulting 4x4 linear system.  This is Hetényi's end-conditioning method in
    matrix form and is the reference solution that
    :func:`finite_beam_series` could not provide for free or clamped ends.

    Parameters
    ----------
    x_eval : (N,) evaluation abscissae inside ``[0, L]``.
    beam_length : ``L`` (m).
    lam : ``(k_b / 4EI)^(1/4)`` (1/m).
    k_b : foundation coefficient per unit length (N/m^2).
    EI : bending stiffness (N m^2).
    point_loads : list of ``(x, P)``, ``P`` positive downward (N).
    distributed_loads : list of ``(x0, x1, q)`` with ``q`` positive downward
        (N/m), segments allowed to overlap (they are summed).
    end_a, end_b : end conditions at ``x = 0`` and ``x = L``; one of
        ``'free'``, ``'simply_supported'``, ``'clamped'``, ``'guided'``
        (aliases such as ``'ss'``, ``'pinned'``, ``'fixed'`` are accepted).

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

    stations = sorted({0.0, L} | {x for x, _ in loads} | {x0 for x0, _, _ in dists} | {x1 for _, x1, _ in dists})
    n_seg = len(stations) - 1
    seg_q = []
    for i in range(n_seg):
        mid = 0.5 * (stations[i] + stations[i + 1])
        seg_q.append(sum(q for x0, x1, q in dists if x0 <= mid <= x1))

    # Propagate the affine map z(x) = P_acc z0 + b_acc station by station.
    eye = torch.eye(4, dtype=dtype)
    p_acc = eye.clone()
    b_acc = torch.zeros(4, dtype=dtype)
    per_station = [(p_acc.clone(), b_acc.clone())]
    for i in range(n_seg):
        if i > 0:
            for x, P in loads:
                if abs(x - stations[i]) <= 1e-12 * max(1.0, L):
                    b_acc = b_acc.clone()
                    b_acc[3] = b_acc[3] + P / (EI * lam ** 3)
        zp = torch.zeros(4, dtype=dtype)
        zp[0] = seg_q[i] / k_b
        prop = beam_state_propagator(lam * (stations[i + 1] - stations[i]), dtype=dtype)
        b_acc = prop @ (b_acc - zp) + zp
        p_acc = prop @ p_acc
        per_station.append((p_acc.clone(), b_acc.clone()))

    # Assemble the 4x4 boundary value system for the state at x = 0.
    rows = []
    rhs = []
    for comp in _END_CONDITION_ROWS[end_a_c]:
        row = torch.zeros(4, dtype=dtype)
        row[comp] = 1.0
        rows.append(row)
        rhs.append(torch.zeros((), dtype=dtype))
    for comp in _END_CONDITION_ROWS[end_b_c]:
        row = p_acc[comp, :]
        rows.append(row.clone())
        rhs.append(-b_acc[comp])
    matrix = torch.stack(rows)
    vector = torch.stack(rhs)
    z0 = torch.linalg.solve(matrix, vector)

    # Evaluate the field on x_eval.
    w = torch.empty_like(x_eval)
    slope = torch.empty_like(x_eval)
    wpp = torch.empty_like(x_eval)
    wppp = torch.empty_like(x_eval)
    idx = torch.searchsorted(torch.tensor(stations[1:-1], dtype=dtype), x_eval, right=True)
    for e, (xe, k) in enumerate(zip(x_eval.tolist(), idx.tolist())):
        p_st, b_st = per_station[k]
        zp = torch.zeros(4, dtype=dtype)
        zp[0] = seg_q[k] / k_b
        state_at = p_st @ z0 + b_st
        prop = beam_state_propagator(lam * (xe - stations[k]), dtype=dtype)
        zx = prop @ (state_at - zp) + zp
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
        state_at_origin=z0,
        end_conditions=(end_a_c, end_b_c),
    )


def foundation_reaction_integral(solution: FiniteBeamSolution) -> float:
    """Trapezoidal ``integral k_b w dx`` of a :class:`FiniteBeamSolution`.

    For a beam that is free at both ends this equals the total applied load, so
    it is an exact global-equilibrium oracle (the trapezoidal quadrature is the
    only error source).
    """
    return float(torch.trapezoid(solution.foundation_reaction, solution.x))
