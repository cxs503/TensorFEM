# Thermal and sequential thermoelastic validation

The stable thermal kernel supports LINE2 and Q4 isotropic conduction with a
consistent capacity matrix, nodal heat input, element heat generation,
prescribed temperature and convection. Transient integration uses the theta
method and rejects `theta < 0.5`, so the stable route cannot silently select a
conditionally unstable scheme.

## Analytical benchmarks

| Case | Reference | Recorded error |
|---|---|---:|
| 1-D prescribed end temperatures | linear Fourier solution | < 1e-11 relative |
| 1-D convection at one end | series conduction/convection resistance | < 1e-11 relative |
| Q4 temperature patch | linear temperature field | < 1e-11 relative |
| transient one-element rod | exact solution of the reduced semidiscrete ODE | < 0.2% |
| free uniform thermal expansion | `u=alpha DeltaT L`, zero force | < 1e-11 relative |
| fully restrained bar | `N=-EA alpha DeltaT` | < 1e-11 relative |

The transient reference is independent of the time integrator: after fixing
one end of one LINE2 element, the remaining equation is
`C_ff dT/dt + K_ff T=0`, hence `T=T0 exp(-t/tau)` with
`tau=rho cp L^2/(3k)`. The validation uses backward Euler with 100 increments
per time constant and requires the global error to remain strictly below 3%.

Thermoelastic coupling is sequential in this release: a converged nodal
temperature field is converted to element mean thermal strain and applied as
an equivalent mechanical load. Fully coupled temperature-displacement Newton
iterations are outside this kernel's advertised scope.
