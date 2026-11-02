# TensorFEM — finite-element solver charter

**Status:** repository responsibility and agent-development contract  
**Audience:** maintainers, contributors, autonomous coding agents  
**Canonical ecosystem vision:** [Tensor Ecosystem: Vision for the AI Era](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md)

## Mission

Provide finite-element analysis, mechanics and related numerical methods as an independently usable, verification-first solver in the Tensor ecosystem.

## This repository owns

- Finite-element discretization, element/material/constitutive implementations and solver algorithms supported by the repository
- Solver-local geometry/mesh interfaces, boundary conditions, diagnostics and numerical verification
- Reproducible structural/mechanical benchmarks and clearly stated method limitations

## Explicit non-goals

- Generic agent runtime, platform governance, or cross-domain task orchestration
- Taking ownership of all scientific acceptance policy across domains
- Treating a converged nonlinear solve as sufficient proof of model validity

## Integration contract

Keep solver functionality independently runnable; provide explicit inputs, outputs, units, mesh/material metadata, failure states, and evidence to TensorPhysics. Use shared TensorPlatform services through stable contracts rather than private imports.

## Quality and evidence gates

Test element patch behavior and invariants, convergence/order where appropriate, mesh sensitivity, benchmark agreement, and constitutive assumptions. Distinguish code verification from physical validation and engineering qualification.

## Agent instructions

Before implementing a substantial change, an agent must:

1. Read this charter, the README, the relevant code/tests, and recent PRs.
2. Explain why the change belongs in this repository and identify affected cross-repository contracts.
3. Prefer a stable, versioned public interface over private imports or copied logic.
4. Include executable tests and a reproducible example or benchmark appropriate to the change.
5. Report exactly what was run, what passed/failed, and what was not tested.
6. Distinguish implementation, unit-test, integration, benchmark, validation, and production evidence.
7. Never claim a result is scientifically or engineering-validated unless the relevant domain evidence supports that claim.
8. Avoid adding general platform features here when they belong in TensorPlatform; avoid adding specialist domain rules to TensorPlatform.
9. Keep changes scoped, reviewable, and reversible. Do not equate PR count or workflow completion with quality.

## Current maturity

This charter states intended responsibility; consult actual tests, benchmark records, and evidence before making maturity claims.

This charter describes responsibility and acceptance expectations; it does not override the source code or claim that planned functionality is implemented. Update the README and this document when ownership changes, and resolve conflicting documentation explicitly.
