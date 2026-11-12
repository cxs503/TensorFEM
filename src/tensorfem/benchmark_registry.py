"""Traceable executable benchmark registry."""

from dataclasses import asdict, dataclass
import math
import torch


@dataclass(frozen=True)
class BenchmarkEvidence:
    id: str
    capability: str
    quantity: str
    unit: str
    source: str
    computed: float
    reference: float
    error: float
    tolerance: float
    passed: bool

    def to_dict(self):
        return asdict(self)


def _evidence(id, capability, quantity, unit, source, computed, reference, tolerance=0.03):
    if not source.strip():
        raise ValueError("benchmark source is required")
    if not all(math.isfinite(float(v)) for v in (computed, reference)):
        raise ValueError("benchmark values must be finite")
    if reference == 0:
        raise ValueError("zero-reference checks must be registered as invariants")
    if not 0 < tolerance <= 0.03:
        raise ValueError("tolerance must be in (0, 3%]")
    error = abs(float(computed) - float(reference)) / abs(float(reference))
    return BenchmarkEvidence(
        id,
        capability,
        quantity,
        unit,
        source,
        float(computed),
        float(reference),
        error,
        tolerance,
        error < tolerance,
    )


def run_registered_benchmarks():
    from .benchmarks import run_single_bar
    from .frame_benchmarks import cantilever_tip_load
    from .continuum_benchmarks import run_cook_membrane
    from .advanced_beam import exact_tip_deflection, solve_timoshenko_cantilever
    from .advanced_buckling import column_buckling, exact_critical_load
    from .dynamics import newmark_linear
    from .solid3d_benchmarks import longitudinal_bar_frequency_benchmark
    from .plate import mindlin_sine_center_deflection, solve_simply_supported_sine_square
    from .nonlinear_truss import two_bar_shallow_arch_reaction
    from .contact import solve_rigid_plane_contact

    out = []
    r = run_single_bar()
    out.append(
        _evidence(
            "truss.bar", "truss", "tip displacement", "m", "u=PL/(EA)", r.computed, r.reference
        )
    )
    r, e = cantilever_tip_load()
    out.append(
        _evidence(
            "frame.cantilever",
            "frame",
            "tip displacement",
            "m",
            "Euler-Bernoulli v=PL^3/(3EI)",
            r.displacement[-2],
            e[0],
        )
    )
    r = run_cook_membrane(16)
    out.append(
        _evidence(
            "continuum.cook",
            "Q4 continuum",
            r.quantity,
            "length",
            "Simo & Rifai (1990), Int. J. Numer. Meth. Eng., DOI:10.1002/nme.1620290802; value 23.96",
            r.computed,
            r.reference,
        )
    )
    E, nu, b, h, L, P = 210e9, 0.3, 0.1, 0.2, 0.4, -1e4
    G = E / (2 * (1 + nu))
    A = b * h
    second_moment = b * h**3 / 12
    out.append(
        _evidence(
            "beam.timoshenko",
            "Timoshenko beam",
            "deep-beam tip displacement",
            "m",
            "Timoshenko bending-plus-shear solution",
            solve_timoshenko_cantilever(L, 4, E, second_moment, G, A, P).displacement[-2],
            exact_tip_deflection(L, E, second_moment, G, A, P),
        )
    )
    L, EI = 3.0, 1.2e6
    out.append(
        _evidence(
            "buckling.euler",
            "buckling",
            "critical load",
            "N",
            "Euler effective-length solution",
            column_buckling(L, 16, EI, "fixed-pinned").load_factors[0],
            exact_critical_load(L, EI, "fixed-pinned"),
        )
    )
    w = 7.0
    T = 2 * math.pi / w
    dt = T / 100
    t = torch.arange(0.0, T + dt / 2, dt, dtype=torch.float64)
    M = torch.tensor([[2.0]], dtype=torch.float64)
    r = newmark_linear(
        M,
        M * w * w,
        torch.zeros((len(t), 1), dtype=torch.float64),
        t,
        torch.tensor([0.02]),
        torch.tensor([0.0]),
    )
    out.append(
        _evidence(
            "dynamics.newmark",
            "dynamics",
            "one-period displacement",
            "m",
            "SDOF u=u0 cos(omega*t)",
            r.displacement[-1, 0],
            0.02,
        )
    )
    c, r, _ = longitudinal_bar_frequency_benchmark(6)
    out.append(
        _evidence(
            "solid.hex8", "HEX8 solid", "axial frequency", "rad/s", "omega=pi/(2L)sqrt(E/rho)", c, r
        )
    )
    n = 16
    r = solve_simply_supported_sine_square(n, young=1e7, poisson=0.3, thickness=0.01, q0=1.0)
    i = (n // 2) * (n + 1) + n // 2
    out.append(
        _evidence(
            "plate.mindlin",
            "Mindlin plate",
            "centre deflection",
            "m",
            "Navier exact Mindlin sine-load solution including shear",
            r.displacement[3 * i],
            mindlin_sine_center_deflection(1.0, 1e7, 0.3, 0.01, 1.0),
        )
    )
    a, h, E, A = 1.0, 0.2, 2000.0, 0.01
    v = torch.linspace(0.0, 0.35, 10001, dtype=torch.float64)
    y = h / math.sqrt(3)
    ref = E * A * (h * h - y * y) * y / (a * a + h * h) ** 1.5
    out.append(
        _evidence(
            "nonlinear.arch",
            "nonlinear truss",
            "limit load",
            "force",
            "Two-bar arch closed form",
            two_bar_shallow_arch_reaction(a, h, E, A, v).max(),
            ref,
        )
    )
    r = solve_rigid_plane_contact(
        torch.tensor([[1000.0]], dtype=torch.float64),
        torch.tensor([-20.0], dtype=torch.float64),
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([0.01], dtype=torch.float64),
        method="penalty",
        penalty=1e5,
    )
    out.append(
        _evidence(
            "contact.spring",
            "contact",
            "reaction",
            "N",
            "Closed-form penalty spring",
            r.contact_force[0],
            1e5 * 10 / 101000.0,
        )
    )
    from .modal import cantilever_beam_modes, cantilever_exact_angular_frequency

    c = cantilever_beam_modes(1.0, 8, 210e9 * 8e-8, 7850.0 * 4e-4, 1).angular_frequencies[0]
    r = cantilever_exact_angular_frequency(1.0, 210e9 * 8e-8, 7850.0 * 4e-4)
    out.append(
        _evidence(
            "modal.cantilever",
            "beam modal",
            "first angular frequency",
            "rad/s",
            "Euler-Bernoulli cantilever beta1=1.875104",
            c,
            r,
        )
    )
    from .shell4 import shell4_stiffness

    x = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    E, nu, t = 70e9, 0.25, 0.03
    k = shell4_stiffness(x, E, nu, t)
    ex, ey, g = 1.2e-4, -0.4e-4, 0.7e-4
    q = torch.zeros(24, dtype=torch.float64)
    q.reshape(4, 6)[:, 0] = ex * x[:, 0] + 0.5 * g * x[:, 1]
    q.reshape(4, 6)[:, 1] = ey * x[:, 1] + 0.5 * g * x[:, 0]
    s = torch.tensor([ex, ey, g], dtype=torch.float64)
    C = (
        E
        / (1 - nu**2)
        * torch.tensor(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu) / 2]], dtype=torch.float64
        )
    )
    out.append(
        _evidence(
            "shell.patch_energy",
            "flat shell",
            "constant-strain energy",
            "J",
            "Plane-stress constant-strain patch analytical energy",
            0.5 * torch.dot(q, torch.mv(k, q)),
            0.5 * torch.dot(s, torch.mv(C, s)) * t * 2.0,
        )
    )
    out.extend(patch_benchmark_evidence())
    from .cohesive import BilinearCohesiveLaw

    law = BilinearCohesiveLaw(1000.0, 10.0, 0.5)
    openings = torch.linspace(0.0, law.failure_opening, 10001, dtype=torch.float64)
    energy = torch.trapezoid(law.envelope_traction(openings), openings)
    out.append(
        _evidence(
            "cohesive.fracture_energy",
            "cohesive law",
            "traction-separation area",
            "energy/area",
            "Bilinear cohesive triangular-area identity",
            energy,
            law.fracture_energy,
        )
    )
    from .plasticity import J2State, update_j2

    z = torch.zeros((3, 3), dtype=torch.float64)
    stress, state, _ = update_j2(
        torch.diag(torch.tensor([0.004, -0.002, -0.002], dtype=torch.float64)),
        210000.0,
        0.3,
        300.0,
        2000.0,
        J2State(z, torch.tensor(0.0, dtype=torch.float64)),
    )
    dev = stress - torch.trace(stress) / 3 * torch.eye(3, dtype=torch.float64)
    seq = torch.sqrt(1.5 * torch.sum(dev * dev))
    radius = 300.0 + 2000.0 * state.alpha
    out.append(
        _evidence(
            "plasticity.j2",
            "J2 plasticity",
            "yield-surface equivalent stress",
            "stress",
            "J2 radial-return consistency equation",
            seq,
            radius,
        )
    )
    # Industrial-core representatives are intentionally small.  Large-model
    # throughput (including the 100k-DOF sparse case) remains a separate
    # performance benchmark and never slows the default qualification gate.
    from .sparse_core import assemble_coo, solve_sparse_static

    n = 41
    length = 2.0
    area = 0.01
    young = 200e9
    load = 1e5
    ke = (young * area / (length / (n - 1))) * torch.tensor(
        [[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64
    ).repeat(n - 1, 1, 1)
    ed = torch.stack((torch.arange(n - 1), torch.arange(1, n)), 1)
    K = assemble_coo(ke, ed, n)
    f = torch.zeros(n, dtype=torch.float64)
    f[-1] = load
    sr = solve_sparse_static(K, f, dirichlet={0: 0.0}, rtol=1e-12)
    out.append(
        _evidence(
            "sparse.axial_bar",
            "sparse statics",
            "tip displacement",
            "m",
            "u=PL/(EA)",
            sr.displacement[-1],
            load * length / (young * area),
        )
    )

    from .thermal import ThermalModel, solve_steady_thermal

    tm = ThermalModel(
        torch.tensor([[0.0], [2.0]], dtype=torch.float64),
        torch.tensor([[0, 1]]),
        torch.tensor(5.0, dtype=torch.float64),
        torch.tensor(3.0, dtype=torch.float64),
        torch.tensor(7.0, dtype=torch.float64),
        torch.tensor([0]),
        torch.tensor([100.0], dtype=torch.float64),
        thickness=torch.tensor(2.0, dtype=torch.float64),
        convection_edges=torch.tensor([[1, 1]]),
        convection_coefficient=torch.tensor(4.0, dtype=torch.float64),
        ambient_temperature=torch.tensor(20.0, dtype=torch.float64),
    )
    temperature = solve_steady_thermal(tm)
    heat = (100.0 - 20.0) / (2.0 / (5.0 * 2.0) + 1.0 / (4.0 * 2.0))
    exact = 100.0 - heat * 2.0 / (5.0 * 2.0)
    out.append(
        _evidence(
            "thermal.rod_convection",
            "thermal",
            "end temperature",
            "temperature",
            "One-dimensional conduction and convection resistances in series",
            temperature[-1],
            exact,
        )
    )

    from .constrained_thermal_bar import constant_steel, restrained_uniaxial_stress, solve_bar

    _cs = constant_steel("EH36", T_ref=20.0)
    _E = _cs.E(20.0)
    _a = _cs.alpha(20.0)
    _dT = 100.0
    _bar = solve_bar(8, _E, _a, _dT, T_ref=20.0, length=1.0)
    out.append(
        _evidence(
            "thermal.constrained_bar",
            "thermoelasticity",
            "restrained bar axial stress",
            "stress",
            "sigma_xx = -E alpha dT (bar fully restrained at both ends, uniform dT; closed-form linear thermoelasticity)",
            float(_bar.sigma.mean()),
            restrained_uniaxial_stress(_E, _a, _dT),
        )
    )

    from .explicit_dynamics import central_difference

    omega = 20.0
    mass = torch.tensor([[2.0]], dtype=torch.float64)
    stiffness = mass * omega**2
    period = 2 * math.pi / omega
    dt = period / 100
    time = torch.arange(0.0, period + dt / 2, dt, dtype=torch.float64)
    er = central_difference(
        mass,
        stiffness,
        torch.zeros((len(time), 1), dtype=torch.float64),
        time,
        torch.tensor([0.01], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
    )
    out.append(
        _evidence(
            "dynamics.explicit",
            "explicit dynamics",
            "one-period displacement",
            "m",
            "SDOF u=u0 cos(omega*t)",
            er.displacement[-1, 0],
            0.01,
        )
    )

    from .nonlinear_step import StepState, solve_adaptive, total_lagrangian_truss_problem
    from .nonlinear_truss import NonlinearTrussModel

    nm = NonlinearTrussModel(
        torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0, 1]]),
        torch.tensor([200.0], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        torch.tensor([0, 1, 3]),
    )
    problem, free = total_lagrangian_truss_problem(
        nm, torch.tensor([0.0, 0.0, 46.2, 0.0], dtype=torch.float64)
    )
    nr = solve_adaptive(
        problem,
        StepState(0.0, torch.zeros(len(free), dtype=torch.float64)),
        initial_increment=0.2,
        tolerance=1e-11,
    )
    out.append(
        _evidence(
            "nonlinear.finite_bar",
            "adaptive nonlinear step",
            "axial displacement",
            "m",
            "Finite-strain bar closed-form response",
            nr.displacement[0],
            0.1,
        )
    )

    from .quadratic_solid import tet4_to_tet10, tet10_stiffness
    from .solid3d import elasticity_matrix_3d

    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    qnodes = tet4_to_tet10(vertices, torch.tensor([[0, 1, 2, 3]]))[0]
    eps = torch.tensor([1.2e-4, -0.4e-4, 0.7e-4, 0.3e-4, -0.2e-4, 0.5e-4], dtype=torch.float64)
    qu = torch.stack(
        (
            eps[0] * qnodes[:, 0] + eps[3] * qnodes[:, 1] / 2 + eps[5] * qnodes[:, 2] / 2,
            eps[1] * qnodes[:, 1] + eps[3] * qnodes[:, 0] / 2 + eps[4] * qnodes[:, 2] / 2,
            eps[2] * qnodes[:, 2] + eps[5] * qnodes[:, 0] / 2 + eps[4] * qnodes[:, 1] / 2,
        ),
        1,
    ).reshape(-1)
    dc = elasticity_matrix_3d(
        torch.tensor([210e9], dtype=torch.float64), torch.tensor([0.29], dtype=torch.float64)
    )
    qk, _ = tet10_stiffness(qnodes[None], dc)
    exact = 0.5 * torch.dot(eps, torch.mv(dc[0], eps)) / 6
    out.append(
        _evidence(
            "solid.tet10_patch",
            "TET10 solid",
            "affine strain energy",
            "J",
            "Constant-strain tetrahedron energy identity",
            0.5 * torch.dot(qu, torch.mv(qk[0], qu)),
            exact,
        )
    )

    from .solid_plasticity import Tet4J2Model, solve_load_steps

    pm = Tet4J2Model(
        vertices,
        torch.tensor([[0, 1, 2, 3]]),
        200000.0,
        0.3,
        250.0,
        10000.0,
        torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10]),
    )
    pf = torch.zeros(pm.n_dofs, dtype=torch.float64)
    pf[3] = 400.0 / 6
    pr = solve_load_steps(pm, pf, (1.0,))[-1]
    exact = 0.002 + (400.0 - 250.0) / 10000.0
    out.append(
        _evidence(
            "plasticity.tet4_j2",
            "TET4 J2 plasticity",
            "uniaxial strain",
            "strain",
            "Linear-hardening uniaxial stress-strain solution",
            pr.displacement[3],
            exact,
        )
    )
    from .marine_benchmark_suite import run_marine_benchmarks

    out.extend(run_marine_benchmarks())
    # CASE-NAFEMS-LE11 (case pool): spheroidal shell under internal pressure.
    # Reference = closed-form thin-shell statics, recomputed in-process (no transcription).
    from .ellipsoid_shell import ellipsoid_pressure

    r = ellipsoid_pressure(
        a=1.0, b=0.5, thickness=0.01, young=1.0, poisson=0.3, pressure=1.0, facets=256
    )
    out.append(
        _evidence(
            "shell.spheroid.meridional_pole",
            "axisym shell",
            "meridional membrane force at pole",
            "p*a",
            "cap axial statics N=p*r/(2 sin phi)",
            r["N_pole_fe"],
            r["N_pole_ref"],
        )
    )
    out.append(
        _evidence(
            "shell.spheroid.meridional_equator",
            "axisym shell",
            "meridional membrane force at equator",
            "p*a",
            "cap axial statics N=p*r/(2 sin phi)",
            r["N_phi_equator_fe"],
            r["N_phi_equator_ref"],
        )
    )
    out.append(
        _evidence(
            "shell.spheroid.hoop_pole",
            "axisym shell",
            "hoop membrane force at pole",
            "p*a",
            "normal equilibrium N_theta=p*R2*(1-R2/(2*R1))",
            r["N_theta_pole_fe"],
            r["N_theta_pole_ref"],
        )
    )
    out.append(
        _evidence(
            "shell.spheroid.displacement_equator",
            "axisym shell",
            "equatorial normal displacement",
            "p*a^2/(E*t)",
            "Rayleigh-Ritz membrane potential minimisation",
            r["w_equator_fe"],
            r["w_equator_ref"],
        )
    )

    return tuple(out)


def verification_report():
    r = run_registered_benchmarks()
    return {
        "schema_version": 1,
        "qualification_scope": "registered scalar checks; not complete field reports",
        "policy": "error < tolerance <= 0.03; zero references use invariants",
        "passed": all(x.passed for x in r),
        "summary": {"total": len(r), "passed": sum(x.passed for x in r)},
        "results": [x.to_dict() for x in r],
    }


def patch_benchmark_evidence():
    """Robinson patch-test A/B/C as registered scalar evidence (``tensorfem.patch_test``).

    Kept as a separate function so the gate can be exercised on its own: the tests
    call it directly instead of paying for the whole (order 100 s) registry run.
    """
    from .patch_test import SIGMA0, TAU0, double_precision, run_q4_patch, run_q8_patch

    out = []
    # These are the only registered checks whose reference is an *exactness*
    # statement: the interior field of an irregular patch loaded through its
    # boundary must equal the imposed constant/linear field, so the reference is
    # that analytical field at the very Gauss point the number is taken from.
    with double_precision():
        ra = run_q4_patch("A")
        rb = run_q4_patch("B")
        probe = run_q8_patch("C", 3, 0.0).probes[0]
    out.append(
        _evidence(
            "patch.robinson_A",
            "patch test A",
            "constant uniaxial stress reproduction",
            "MPa",
            "Robinson (1987) patch-test criterion on an irregular Q4 patch: sigma_xx = sigma0 exactly",
            ra.probes[0]["sigma_xx_h"],
            SIGMA0,
            1e-10,
        )
    )
    out.append(
        _evidence(
            "patch.robinson_B",
            "patch test B",
            "constant pure-shear stress reproduction",
            "MPa",
            "Robinson (1987) patch-test criterion on an irregular Q4 patch: tau_xy = tau0 exactly",
            rb.probes[0]["tau_xy_h"],
            TAU0,
            1e-10,
        )
    )
    out.append(
        _evidence(
            "patch.robinson_C",
            "patch test C",
            "linear bending stress reproduction",
            "MPa",
            "Robinson (1987) patch-test criterion, case C: sigma_xx = -E*kappa*y at the worst "
            "Gauss point of an affine Q8 patch",
            probe["sigma_xx_h"],
            probe["sigma_xx_exact"],
            1e-10,
        )
    )
    return tuple(out)
