# Weld hot-spot stress and fatigue qualification

This TensorFEM-only module provides linear and quadratic weld-toe stress
extrapolation, a single-slope double-logarithmic S-N relation, optional explicit
thickness correction, and Palmgren-Miner accumulation over multiple stress blocks.
It neither imports nor executes TensorLBM or CFD code.

All stresses are Pa, distances and thicknesses are m, and cycle counts are
dimensionless. Invalid, non-finite, non-positive, incomplete, or geometrically
degenerate inputs fail closed. Linear and quadratic extrapolation use general
Lagrange interpolation at the weld toe, so standard 0.4t/1.0t and
0.4t/0.9t/1.4t sampling layouts are supported without being hard-coded.

`run_fatigue_qualification()` checks four independently hand-calculated oracles:

- linear extrapolation: 150 MPa;
- quadratic extrapolation: 146 MPa;
- S-N life at 125 MPa: 1,024,000 cycles;
- two-block Miner damage: 0.5.

Every case must have relative error strictly below 3%. The S-N parameters and
thickness exponent are user-selected engineering inputs. These checks demonstrate
numerical consistency only; they are not a class-society rule implementation,
approval, or certification.
