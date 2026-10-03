# Marine benchmark qualification matrix

The marine modules are qualified by executable evidence rather than feature
labels. `tensorfem verify` fails closed if any non-zero reference quantity has
relative error greater than or equal to 3%.

| ID | Application | Independent reference | Additional gate |
|---|---|---|---|
| `marine.hull_girder.deflection` | longitudinal strength screening | simply-supported Euler--Bernoulli `5qL^4/(384EI)` | 4/8/16/32-element sequence and reaction balance |
| `marine.hull_girder.moment` | longitudinal strength screening | simply-supported `qL^2/8` | reaction balance |
| `marine.stiffened_panel.navier` | local stiffened panel | one-term Navier orthotropic plate solution | longitudinal rigidity must exceed transverse rigidity |
| `marine.hydrostatics.displacement` | upright box-barge hydrostatics | Archimedes `LBT` | mass/displacement balance |
| `marine.hydrostatics.gm` | intact initial stability | wall-sided `KB+I/V-KG` | positive/negative stability is retained, not clipped |
| `marine.hydrostatics.restoring` | small-angle heel | `Delta*g*GM*sin(phi)` | input limited to plus/minus 10 degrees |
| `marine.morison.base_shear` | fixed slender pile | integrated Airy/Morison closed form | 4/8/16/32/128-point convergence |
| `marine.morison.overturning` | fixed slender pile | integrated Airy/Morison first moment | same convergence gate |
| `marine.plate_buckling.*` | local plate buckling | continuous Navier eigenvalue | 8/16/32-grid convergence |
| `marine.hull_ultimate.*` | section progressive yielding | rectangular ideal-elastic-plastic closed form | fiber convergence and state checks |
| `marine.fatigue.*` | hot-spot/S-N/Miner primitives | independent hand calculations | strict inputs and non-certification boundary |

The Morison formulation follows the slender-member equation described by
DNV-RP-C205. These cases validate the implemented equations and numerical
integration; they do not qualify diffraction, breaking waves, viscous CFD,
large-angle stability, flooding, fatigue, or class-rule compliance. Real flow
fields remain the responsibility of TensorLBM and cross the solver boundary as
versioned nodal force histories.
