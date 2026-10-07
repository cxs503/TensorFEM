# Cohesive damage validation

## Scope

`BilinearCohesiveLaw` is a small-strain, rate-independent, pure mode-I
traction--separation material-point model. `two_node_interface_response` wraps
that law in a scalar two-node interface with a prescribed crack path. This is
not a claim of automatic crack initiation, crack tracking, mixed-mode fracture,
or general crack propagation capability.

The law uses an initial stiffness `K`, tensile strength `t0`, and fracture
energy `Gc`. Damage begins at `d0=t0/K`; complete separation occurs at
`df=2 Gc/t0`. The monotonic envelope is linear to `(d0,t0)` and then linear to
`(df,0)`. Unloading and reloading use the degraded secant stiffness associated
with the historical maximum tensile opening. Compression retains `K` and does
not advance tensile damage.

## Independent checks

The automated tests use `K=1000`, `t0=10`, and `Gc=0.5`, giving `d0=0.01` and
`df=0.1` exactly.

| Check | Computed | Closed form | Relative error |
|---|---:|---:|---:|
| Peak traction | 10 | 10 | < 1e-12 |
| Failure opening | 0.1 | 0.1 | < 1e-12 |
| Envelope integral | 0.5 | 0.5 | < 1e-8 |

The tests additionally verify irreversible history, secant unloading/reloading,
compression without damage growth, element action--reaction, and the exact
two-node tangent. All advertised quantitative results are therefore below the
project's strict 3% threshold.

## Reference basis

The bilinear cohesive-zone construction and irreversible unloading convention
follow the standard formulation described by Camanho and Dávila, *Mixed-Mode
Decohesion Finite Elements for the Simulation of Delamination in Composite
Materials*, NASA/TM-2002-211737, 2002. The checks above are direct analytical
consequences of the stated piecewise-linear law and do not depend on an
external finite-element result.
