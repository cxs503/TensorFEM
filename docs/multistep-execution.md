# Unified multi-step execution

`StepExecutor` runs an ordered list of named `StepSpec` records.  A
`StepContext` resolves explicit `step.field` dependencies, and every
`StepResult` records fields, elapsed wall time, success, and a typed diagnostic.
Progress callbacks receive start, complete, or failed events.

Qualified adapters currently cover linear truss statics, steady conduction,
beam modal analysis, and the existing sequential thermoelastic bar kernel.
The example passes `heat.temperature` into the expansion step, then also runs
independent structural static and modal steps.  A request for an undeclared
fully coupled thermo-mechanical kernel fails before numerical work; the
framework does not infer or pretend to support coupling.

Successful steps are checkpointed to versioned JSON after each step.  Tensor
dtype and values are retained.  Resume verifies the immutable plan signature,
restores completed fields, and starts at the first unfinished step.  Inputs
(models and matrices) remain supplied by the caller and are not insecurely
deserialized.

`execution_result_db` writes compact step histories, timings, diagnostics, and
field-shape inventory through the unified ResultDB layer.  Multidimensional
bulk modes remain in the `StepResult`; the compact adapter records their shape
rather than flattening them into a misleading scalar history.

```bash
PYTHONWARNINGS=error pytest -q tests/test_step_executor.py
python examples/multistep_project.py
```
