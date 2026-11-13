"""Rolling-resistance model assessment: four contact-law families with analytic oracles.

Anchor: ``XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS``
(literature/Ai-2011 rolling-model-assessment-4-models, doi:10.1016/j.powtec.2010.11.003).

Role of this module (read this before using it)
-----------------------------------------------
This is TensorFEM's **reference-grade verification surface**, in the same sense as
:mod:`tensorfem.broken_ice_floe_field` (rigid-floe buoyancy/contact oracles) and
:mod:`tensorfem.cattaneo_hertz` (contact closed-form oracles): the *production*
DEM rolling law lives in ``tensordem/rolling_resistance.py``; here we own the
single-particle analytic benchmark any DEM implementation can be checked against,
plus the model family that the production module does not implement.  Reported
quantities are *analytic* or *machine-precision identities*, never digitised
figure readings -- no dataset is in-repo, so figure-level agreement with the
paper is WITHHELD (see ``docs/XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS.md``).

The four families (Ai et al. 2011)
----------------------------------
All four share the saturated torque magnitude

    M_cap = mu_r * F_n * R            (N m;  F_n = m g cos(theta) on an incline)

and differ only in how the resisting torque is generated:

``"cdt"``        constant directional torque (memoryless)

    tau = -M_cap * sign(omega)

    No static branch: the torque magnitude does not vanish as ``omega -> 0``, so a
    roller that has just come to rest is still driven.  The direction is carried
    from the last non-zero spin sample (the usual DEM implementation), so the
    artefact is a limit cycle whose amplitude scales with the step size.

``"viscous"``    velocity-proportional (strictly dissipative, memoryless)

    tau = -(M_cap/omega_ref) * omega

    Rotation at ``|omega| = omega_ref`` reproduces the rolling limit.  No static
    resistance at all: ``tau = 0`` at rest, so the model can never hold a slope
    and its critical-angle threshold is ``theta_c = 0``.

``"hysteretic"`` two-stiffness (bilinear) rolling spring -- Ai's *hysteretic* family

    tau = -k_r * phi ,   phi += scale * dtheta,
    scale = 1 if phi*dtheta >= 0 (loading) else 1/(1-h) (unloading),
    |phi| <= phi_y = M_cap/k_r        (clip; the excess is plastic)

    ``h`` in ``[0, 1)`` is the hysteresis ratio: unloading is ``1/(1-h)`` times
    stiffer, so a load reversal encloses a rate-independent hysteresis loop.
    The elastic clip supplies a static branch.

``"epsd"``       elastic-plastic rolling spring + dashpot (Iwashita & Oda 1998 /
    Jiang et al. 2005, the EPSD2 family)

    loading is purely elastic (``scale = 1``); plastic flow starts when ``|phi|``
    would exceed ``phi_y``; a rolling dashpot ``-c_r * omega`` with
    ``c_r = 2 zeta sqrt(k_r I)`` is added (``zeta = 1`` by default = critical).
    The plastic offset behind ``phi`` is the memory that produces hysteresis.

Implementation note: the two stiffness families are implemented with a single
state variable ``phi`` (elastic rolling strain); the plastic rolling offset
``theta_p = theta_rolled - phi`` is reported for diagnostics.  Ai et al.'s exact
stiffness schedule is not reproduced digit-for-digit (no source equations in
hand); what is reproduced is the *family-level* constitutive behaviour and its
analytic single-particle consequences -- stated explicitly in the assessment doc.

Single-point rigid-roller dynamics (no slip, one DOF)
-----------------------------------------------------
Rolling without slip on a plane inclined at ``theta``: ``omega = v/R`` and the
contact-point balance gives

    m (1 + k) dv/dt = m g sin(theta) + tau/R ,      k = I/(m R^2)

with ``k = 2/5`` (solid sphere), ``1/2`` (solid cylinder), ``1`` (thin ring).

Conventions: ``v > 0`` is downhill, ``omega = v/R > 0``, rolling displacement
``theta_rolled = x/R`` (rad), ``tau > 0`` accelerates the roller (a resisting
torque is negative).  SI units throughout.

References
----------
Ai, J., Chen, J.-F., Rotter, J.M., Ooi, J.Y. (2011). Assessment of rolling
    resistance models in discrete element simulations. Powder Technology 206,
    269-282.
Iwashita, K., Oda, M. (1998). J. Eng. Mech. 124, 285-292.
Jiang, M.J., Yu, H.-S., Harris, D. (2005). Comput. Geotech. 32, 340-357.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

MODELS = ("cdt", "viscous", "hysteretic", "epsd")

#: Radius-of-gyration factors ``k = I/(m R^2)`` for the shapes we benchmark.
INERTIA_FACTOR = {"sphere": 2.0 / 5.0, "cylinder": 0.5, "ring": 1.0}


@dataclass(frozen=True)
class RollingModel:
    """Parameters of one rolling-resistance family acting on one roller.

    Attributes:
        model:            one of :data:`MODELS`.
        mu_r:             dimensionless rolling-resistance coefficient (the
                          ``tan(theta_c)`` of the saturated families).
        radius:           roller radius ``R``, m.
        mass:             roller mass ``m``, kg.
        inertia_factor:   ``k = I/(m R^2)``; see :data:`INERTIA_FACTOR`.
        gravity:          ``g``, m/s^2.
        omega_ref:        reference spin for the viscous family, rad/s.
        yield_angle:      ``phi_y``, the elastic rolling strain at the torque
                          limit; sets ``k_r = M_cap/phi_y``, rad.
        hysteresis_ratio: ``h`` for the hysteretic family, in ``[0, 1)``.
        damping_ratio:    ``zeta`` of the rolling dashpot (hysteretic/epsd).
        omega_tol:        spin below which ``"cdt"`` reuses the previous direction.
    """

    model: str
    mu_r: float
    radius: float
    mass: float
    inertia_factor: float = 0.4
    gravity: float = 9.81
    omega_ref: float = 1.0
    yield_angle: float = 1.0e-3
    hysteresis_ratio: float = 0.25
    damping_ratio: float = 1.0
    omega_tol: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        for name in ("mu_r", "radius", "mass", "inertia_factor", "gravity", "omega_ref",
                     "yield_angle", "hysteresis_ratio", "damping_ratio", "omega_tol"):
            value = getattr(self, name)
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.mu_r < 0.0 or self.radius <= 0.0 or self.mass <= 0.0:
            raise ValueError("mu_r >= 0, radius > 0 and mass > 0 are required")
        if self.inertia_factor <= 0.0 or self.yield_angle <= 0.0:
            raise ValueError("inertia_factor and yield_angle must be positive")
        if not 0.0 <= self.hysteresis_ratio < 1.0:
            raise ValueError("hysteresis_ratio must lie in [0, 1)")
        if self.damping_ratio < 0.0 or self.omega_ref <= 0.0:
            raise ValueError("damping_ratio >= 0 and omega_ref > 0 are required")

    # -- derived quantities -------------------------------------------------
    @property
    def moment_of_inertia(self) -> float:
        return self.inertia_factor * self.mass * self.radius**2

    def normal_force(self, incline_deg: float) -> float:
        """``F_n = m g cos(theta)``, N."""
        return self.mass * self.gravity * math.cos(math.radians(incline_deg))

    def capacity(self, normal_force: float) -> float:
        """Saturated rolling torque ``M_cap = mu_r F_n R``, N m."""
        return self.mu_r * normal_force * self.radius

    def spring_stiffness(self, normal_force: float) -> float:
        """Rolling spring stiffness ``k_r = M_cap/phi_y``, N m/rad."""
        return self.capacity(normal_force) / self.yield_angle

    def damping(self, normal_force: float) -> float:
        """Rolling dashpot ``c_r = 2 zeta sqrt(k_r I)``, N m s/rad."""
        k_r = self.spring_stiffness(normal_force)
        return 2.0 * self.damping_ratio * math.sqrt(k_r * self.moment_of_inertia)


@dataclass
class RollingState:
    """Contact memory carried between steps (empty for the memoryless families)."""

    phi: float = 0.0        # elastic rolling strain, rad
    theta_p: float = 0.0    # plastic (irreversible) rolling offset, rad
    spin_sign: float = 0.0  # last resolved spin direction, used by "cdt"


# ---------------------------------------------------------------------------
# analytic oracles: the numbers every model is checked against
# ---------------------------------------------------------------------------
def critical_incline_angle_deg(mu_r: float) -> float:
    """Critical angle ``theta_c = atan(mu_r)`` of the saturated families, deg.

    Steady rolling of the saturated families requires ``tan(theta) > mu_r``; this
    is the calibration identity ``mu_r = tan(theta_c)`` used to measure rolling
    coefficients from angles of repose.
    """
    return math.degrees(math.atan(mu_r))


def saturated_acceleration(model: RollingModel, incline_deg: float) -> float:
    """Steady (saturated) rolling acceleration ``g(sin - mu_r cos)/(1+k)``, m/s^2.

    Shared by ``"cdt"``, ``"hysteretic"`` and ``"epsd"`` once the torque limit is
    reached: the three families differ in their static/transient branches, not in
    the saturated mean torque.
    """
    theta = math.radians(incline_deg)
    return model.gravity * (math.sin(theta) - model.mu_r * math.cos(theta)) / (
        1.0 + model.inertia_factor
    )


def cdt_stopping_distance(model: RollingModel, v0: float, incline_deg: float = 0.0) -> float:
    """Distance rolled before stopping, ``s = (1+k) v0^2 / (2 g (mu_r cos - sin))``.

    Returns ``inf`` when ``tan(theta) >= mu_r`` (the roller never stops).
    """
    theta = math.radians(incline_deg)
    drive = model.gravity * (model.mu_r * math.cos(theta) - math.sin(theta))
    if drive <= 0.0:
        return math.inf
    return (1.0 + model.inertia_factor) * v0 * v0 / (2.0 * drive)


def viscous_terminal_velocity(model: RollingModel, incline_deg: float) -> float:
    """Terminal velocity ``v_inf = R omega_ref tan(theta)/mu_r`` of the viscous family.

    Non-zero for *any* ``theta > 0``: the viscous family has no static threshold,
    i.e. ``theta_c = 0`` for it.
    """
    theta = math.radians(incline_deg)
    return model.radius * model.omega_ref * math.tan(theta) / model.mu_r


def viscous_time_constant(model: RollingModel, incline_deg: float) -> float:
    """``tau_v = (1+k) m R^2 / C_v`` with ``C_v = mu_r F_n R / omega_ref``, s."""
    c_v = model.capacity(model.normal_force(incline_deg)) / model.omega_ref
    return (1.0 + model.inertia_factor) * model.mass * model.radius**2 / c_v


def static_windup_angle(model: RollingModel, incline_deg: float) -> float:
    """Rolling strain balancing gravity below the limit, ``phi_eq = phi_y tan(theta)/mu_r``.

    Only the spring families (``"hysteretic"``, ``"epsd"``) can reach it: the
    driving torque ``m g sin(theta) R`` is balanced by ``k_r phi_eq`` while
    ``|phi_eq| <= phi_y``, i.e. while ``tan(theta) <= mu_r``.
    """
    theta = math.radians(incline_deg)
    return model.yield_angle * math.tan(theta) / model.mu_r


def damped_terminal_velocity(model: RollingModel, incline_deg: float) -> float:
    """Terminal velocity of a *damped* spring family, ``max(0, (m g sin R - M_cap) R / c_r)``.

    With the rolling dashpot of the EPSD2 family the saturated steady state is not
    the constant acceleration :func:`saturated_acceleration` predicts: the dashpot
    keeps growing with the spin until either the drive is balanced (``v > 0``) or
    the elastic rolling spring holds the roller (``v = 0``).  The switch happens
    exactly at ``tan(theta) = mu_r``, so the critical angle is unchanged.
    """
    c_r = model.damping(model.normal_force(incline_deg))
    if c_r <= 0.0:
        return math.inf if saturated_acceleration(model, incline_deg) > 0.0 else 0.0
    theta = math.radians(incline_deg)
    excess = model.mass * model.gravity * math.sin(theta) * model.radius - model.capacity(
        model.normal_force(incline_deg)
    )
    if excess <= 0.0:
        return 0.0
    return excess * model.radius / c_r


def damped_time_constant(model: RollingModel, incline_deg: float) -> float:
    """``tau_s = (1+k) m R^2 / c_r`` of the damped spring families, s."""
    c_r = model.damping(model.normal_force(incline_deg))
    if c_r <= 0.0:
        return math.inf
    return (1.0 + model.inertia_factor) * model.mass * model.radius**2 / c_r


# ---------------------------------------------------------------------------
# constitutive update: one step of the contact law
# ---------------------------------------------------------------------------
def contact_torque(
    model: RollingModel,
    state: RollingState,
    *,
    omega: float,
    dtheta: float,
    normal_force: float,
) -> tuple[float, RollingState]:
    """Advance the contact law by ``dtheta`` and return ``(tau, new_state)``.

    Args:
        omega:         current rolling spin ``v/R``, rad/s.
        dtheta:        rolling displacement increment this step, rad.
        normal_force:  normal-force magnitude ``F_n``, N (``>= 0``).

    Returns:
        ``(tau, state)`` with ``tau`` the signed torque on the roller, N m
        (negative = resisting).

    A separated contact (``F_n == 0``) returns zero torque and loses its memory,
    matching the ``_deactivate`` convention of ``tensordem.rolling_resistance``.
    """
    if normal_force <= 0.0:
        return 0.0, RollingState()

    cap = model.capacity(normal_force)

    if model.model == "cdt":
        sign = state.spin_sign
        if abs(omega) > model.omega_tol:
            sign = 1.0 if omega > 0.0 else -1.0
        # A roller with no spin history at all gets no torque: the model is
        # direction-blind at rest, which is exactly its defect (documented).
        return -cap * sign, RollingState(spin_sign=sign)

    if model.model == "viscous":
        return -(cap / model.omega_ref) * omega, RollingState()

    # --- spring families ---------------------------------------------------
    k_r = model.spring_stiffness(normal_force)
    phi_y = model.yield_angle
    scale = 1.0
    if model.model == "hysteretic" and state.phi * dtheta < 0.0:
        scale = 1.0 / (1.0 - model.hysteresis_ratio)  # unloading branch
    dphi = scale * dtheta
    phi_trial = state.phi + dphi
    phi = phi_trial
    if phi_trial > phi_y:
        phi = phi_y
    elif phi_trial < -phi_y:
        phi = -phi_y
    theta_p = state.theta_p + (dphi - (phi - state.phi))  # plastic part
    tau = -k_r * phi
    if model.damping_ratio > 0.0:
        tau += -model.damping(normal_force) * omega
    return tau, RollingState(phi=phi, theta_p=theta_p)


# ---------------------------------------------------------------------------
# single-particle driver
# ---------------------------------------------------------------------------
def simulate(
    model: RollingModel,
    *,
    incline_deg: float = 0.0,
    v0: float = 0.0,
    duration: float = 1.0,
    dt: float = 1.0e-6,
    external_torque: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Integrate the no-slip rigid-roller dynamics for one contact law.

    The memoryless families use exact updates (``cdt``: constant acceleration;
    ``viscous``: exponential solution of the linear ODE), the spring families a
    semi-implicit step, so linear analytic oracles are reproduced to machine
    precision.

    Args:
        model:            the contact law.
        incline_deg:      plane inclination, deg (positive = roller rolls downhill).
        v0:               initial velocity, m/s.
        duration, dt:     time window and step, s.
        external_torque:  extra torque about the rolling axis, N m (positive =
                          drives downhill rolling).  Either a constant or a
                          callable ``f(t)`` for driven/reversal tests.

    Returns:
        Dict of float64 tensors: ``t, x, v, omega, theta_rolled, phi, tau, work,
        spring_energy`` -- ``work`` is the *signed* cumulative contact work
        ``integral tau omega dt`` (negative = energy removed from the roller; overa
        periodic steady state it equals the energy dissipated in the contact) and
        ``spring_energy`` the elastic rolling energy ``k_r phi^2/2`` (zero for the
        memoryless families).
    """
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive")
    n_steps = int(math.ceil(duration / dt - 1.0e-12))
    if n_steps < 1:
        raise ValueError("duration must span at least one step")

    theta = math.radians(incline_deg)
    sin_t = math.sin(theta)
    normal_force = model.normal_force(incline_deg)
    inertia = model.mass * (1.0 + model.inertia_factor)
    spring_energy_factor = (
        0.5 * model.spring_stiffness(normal_force)
        if model.model in ("hysteretic", "epsd")
        else 0.0
    )

    # Accumulate in Python floats (torch scalar indexing dominates the runtime
    # otherwise) and hand back float64 tensors.
    t: list[float] = []
    x: list[float] = []
    v: list[float] = []
    phi: list[float] = []
    tau_hist: list[float] = []
    work: list[float] = []

    v_cur = float(v0)
    torque_fn = external_torque if callable(external_torque) else None
    ext_cur = float(torque_fn(0.0)) if torque_fn is not None else float(external_torque)
    # Resolve the initial torque direction of the memoryless "cdt" family: from
    # the spin if there is one, otherwise from the applied (drive) moment, which
    # is the only information available at rest.  This is what makes "cdt"
    # direction-blind at equilibrium (documented defect).
    drive_moment = model.mass * model.gravity * sin_t * model.radius + ext_cur
    if v0 != 0.0:
        spin_sign = 1.0 if v0 > 0.0 else -1.0
    else:
        spin_sign = 1.0 if drive_moment > 0.0 else (-1.0 if drive_moment < 0.0 else 0.0)
    state = RollingState(spin_sign=spin_sign)

    x_cur = 0.0
    work_cur = 0.0
    t.append(0.0)
    x.append(x_cur)
    v.append(v_cur)
    phi.append(state.phi)
    tau_hist.append(0.0)
    work.append(work_cur)

    if model.model == "viscous" and torque_fn is None:
        tau_v = viscous_time_constant(model, incline_deg)
        v_inf = (model.mass * model.gravity * sin_t
                 + ext_cur / model.radius) * tau_v / (1.0 + model.inertia_factor)
        decay = math.exp(-dt / tau_v) if tau_v > 0.0 else 0.0
    else:
        tau_v = 0.0
        v_inf = 0.0
        decay = 0.0

    for n in range(n_steps):
        omega = v_cur / model.radius
        if torque_fn is not None:
            ext_cur = float(torque_fn(t[-1]))
            if model.model == "viscous":
                # A driven memoryless contact needs the general (non-autonomous)
                # update: fall back to the explicit step.
                tau_value, state = contact_torque(
                    model, state, omega=omega, dtheta=omega * dt,
                    normal_force=normal_force,
                )
                accel = (model.mass * model.gravity * sin_t + ext_cur / model.radius) / (
                    1.0 + model.inertia_factor
                ) + tau_value / (inertia * model.radius)
                v_next = v_cur + accel * dt
                dx = 0.5 * (v_cur + v_next) * dt
                work_cur += tau_value * omega * dt
                v_cur = v_next
                x_cur += dx
                t.append(t[-1] + dt)
                x.append(x_cur)
                v.append(v_cur)
                phi.append(state.phi)
                tau_hist.append(tau_value)
                work.append(work_cur)
                continue
        drive_accel = (model.mass * model.gravity * sin_t + ext_cur / model.radius) / (
            1.0 + model.inertia_factor
        )

        if model.model == "viscous":
            v_next = v_inf + (v_cur - v_inf) * decay
            dx = v_inf * dt + (v_cur - v_inf) * tau_v * (1.0 - decay)
            tau_value, state = contact_torque(
                model, state, omega=omega, dtheta=v_next / model.radius * dt,
                normal_force=normal_force,
            )
        else:
            dtheta = omega * dt
            tau_value, state = contact_torque(
                model, state, omega=omega, dtheta=dtheta, normal_force=normal_force,
            )
            accel = drive_accel + tau_value / (inertia * model.radius)
            v_next = v_cur + accel * dt
            dx = 0.5 * (v_cur + v_next) * dt

        work_cur += tau_value * omega * dt
        v_cur = v_next
        x_cur += dx
        t.append(t[-1] + dt)
        x.append(x_cur)
        v.append(v_cur)
        phi.append(state.phi)
        tau_hist.append(tau_value)
        work.append(work_cur)

    t_t = torch.tensor(t, dtype=torch.float64)
    x_t = torch.tensor(x, dtype=torch.float64)
    v_t = torch.tensor(v, dtype=torch.float64)
    phi_t = torch.tensor(phi, dtype=torch.float64)
    return {
        "t": t_t,
        "x": x_t,
        "v": v_t,
        "omega": v_t / model.radius,
        "theta_rolled": x_t / model.radius,
        "phi": phi_t,
        "tau": torch.tensor(tau_hist, dtype=torch.float64),
        "work": torch.tensor(work, dtype=torch.float64),
        "spring_energy": spring_energy_factor * phi_t**2,
    }
