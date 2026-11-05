# Agent instructions

Before making changes, read the [Tensor ecosystem vision](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md) and this repository's [project charter](docs/PROJECT_CHARTER.md). These documents define ownership boundaries; the source code and tests define what is actually implemented.

## Required development behavior

- State the goal, why it belongs in this repository, and which public or cross-repository contracts it affects.
- Preserve the repository boundary: TensorFEM owns finite-element methods and solver-specific verification. Read docs/PROJECT_CHARTER.md and the ecosystem vision. Keep generic agent/platform services out of this repository and do not claim that convergence alone proves model validity. Include reproducible tests, convergence/mesh studies where appropriate, benchmark evidence, assumptions, and explicit limitations.
- Inspect relevant code, tests, recent changes, and existing benchmark records before editing.
- Prefer stable, versioned public interfaces; avoid private cross-repository imports and duplicated sources of truth.
- Include runnable tests/examples and quantitative evidence appropriate to the change. Report exact commands, results, failures, untested paths, assumptions, and limitations.
- Distinguish implementation, unit tests, integration tests, benchmarks, domain validation, and production deployment evidence. Never claim more than the evidence establishes.
- Keep changes scoped, reviewable, and reversible. Do not optimize for PR count, code volume, or workflow completion alone.

## Lint before you push

CI gates two things on every change (see [docs/ci-lint-gate.md](docs/ci-lint-gate.md)):

```
ruff check src tests scripts examples                    # correctness rules (E9/F63/F7/F81/F82)
python scripts/lint_gate.py --base origin/main           # format ratchet on the files you changed
ruff format <the files you touched>                      # fix the ratchet if it fails
```

A tree-wide `ruff format` is not required: pre-existing format debt in a file you
touch is warned about, not blocked. Do not add `# noqa` or widen
`[tool.ruff.lint] select` to silence a finding; fix it or report it.
