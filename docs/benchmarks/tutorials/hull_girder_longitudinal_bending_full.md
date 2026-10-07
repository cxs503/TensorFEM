# Ship hull-girder longitudinal bending — real Q4 FE benchmark

## Abstract

This public benchmark verifies TensorFEM on an idealised welded-equivalent
hull-girder strip under longitudinal bending. It is deliberately small and
reproducible: the FE result includes reactions, nodal displacement and
element von-Mises stress, and is compared with the Euler–Bernoulli reference.
It is a solver verification case, not a class-rule or production ship model.

## Problem and conditions

The strip is `L=1.0 m`, `H=0.1 m`, thickness `t=0.012 m`, `E=210 GPa`,
`nu=0.30`, with a `P=-100 N` transverse load distributed on the free end.
The section at `x=0` is clamped. SI units are used. The reference is
`v(L)=P L^3/(3 E I)`, `I=t H^3/12`.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/run_hull_girder_fe_benchmark.py \
  --output results/hull-girder-fe.json
PYTHONPATH=src .venv/bin/python scripts/render_benchmark_report.py \
  results/hull-girder-fe.json --output-dir results/hull-girder-report
PYTHONPATH=src .venv/bin/python scripts/render_hull_girder_fe_clouds.py \
  results/hull-girder-fe.json --output-dir results/hull-girder-clouds
```

The first command assembles and solves the actual Q4 continuum model. The
second produces HTML, PDF and DOCX research-report artifacts; the third emits
the nodal displacement and element von-Mises cloud images.

## Results and error gate

The JSON report records both meshes, the FE tip displacement, independent
reference value, reaction balance, complete nodal displacement field and
element stress field. A relative error below 3% is `qualified`; an error at
or above 3%, missing fields, or a failed solve is `blocked`. No result is
promoted by the report renderer.

| mesh | FE tip displacement | reference | relative error | status |
|---|---:|---:|---:|---|
| 20×4 | generated in JSON | Euler–Bernoulli | generated in JSON | gate-derived |
| 40×8 | generated in JSON | Euler–Bernoulli | generated in JSON | gate-derived |

The PNGs are generated from the actual FE fields, not an analytical field.
Use the two mesh levels to inspect convergence; the coarse mesh may remain
`blocked` even when the refined mesh qualifies.

## Interpretation and limitations

This benchmark checks continuum assembly, Dirichlet constraints, load
transfer, reactions, displacement recovery and stress recovery. It does not
represent stiffener geometry, hydroelasticity, nonlinear collapse, welds,
corrosion or class-rule acceptance. Those require separate benchmarks and
must retain the same field and 3% evidence gate.
