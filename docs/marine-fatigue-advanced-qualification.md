# Advanced marine structural-fatigue qualification

This TensorFEM-only qualification covers four calculation primitives:

- surface-path nodal interpolation followed by 0.4t/1.0t linear or
  0.4t/0.9t/1.4t quadratic weld-toe extrapolation;
- ASTM E1049-style four-point rainflow counting, retaining residual half cycles;
- variable-amplitude Palmgren-Miner accumulation against a supplied S-N curve;
- constant-geometry Paris-law crack growth integrated analytically by load block.

All inputs use SI units: stress in Pa, length in m, and cycle count dimensionless.
Inputs must be finite, path coordinates strictly increasing, and physical scale
parameters positive. Rainflow means are reported but no mean-stress correction is
silently applied.

## Independent checks and gate

`run_advanced_fatigue_qualification()` compares production results with four
hand/analytic oracles: a linear path field with a 150 MPa toe intercept, a single
closed triangular cycle, its direct S-N damage, and the exact exponential Paris
solution for exponent `m=2`. Every relative error must be strictly below 3%.

Run the focused checks and example with:

```bash
pytest -q tests/test_marine_fatigue_advanced.py
python examples/marine/advanced_fatigue.py
```

## Boundaries

These routines provide engineering verification evidence, not class-society or
regulatory certification. They do not choose detail categories, S-N curves,
mean-stress corrections, stress concentration factors, inspection intervals, or
fracture toughness. The Paris model assumes constant geometry factor within each
analysis and does not model threshold, retardation, closure, overload, or unstable
fracture. TensorLBM and CFD are neither imported nor executed.
