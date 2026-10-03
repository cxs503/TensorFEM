# Multi-element shell-panel ultimate-strength contract

`marine_panel_ultimate_fe` defines the first real finite-element input for a
plate ultimate-strength qualification. It is a square, simply supported Shell4
mesh under uniaxial compression, with a stress-free sine-wave imperfection and
a self-balanced longitudinal residual-stress field at every in-plane and
through-thickness material point.

The required matrix contains 4x4, 8x8 and 12x12 meshes and two arc-length
controls. Qualification requires the peak load, a defined post-peak point,
failure mode, yielded fraction, equilibrium, external work, recoverable energy,
plastic dissipation and the energy residual. Peak mesh and continuation-step
changes must be below 3 percent. The classical simply-supported plate buckling
load and gross-section squash load are independent analytical bounds, not
external experimental evidence.

The runner currently reports **blocked**. The finite-rotation layered Shell4
assembly exists, but the arc-length driver does not yet atomically commit its
`LayeredShell4State` only after an accepted equilibrium point. Running ordinary
load control would not establish a descending path, and replacing the model by
a Koiter, strip or beam reduction would not qualify a multi-element shell FE
implementation. Therefore this module makes no peak-load, post-buckling or
ultimate-strength claim yet.

Run the contract and its tests with:

```bash
PYTHONPATH=src python examples/marine/panel_ultimate_fe_contract.py
PYTHONPATH=src pytest -q tests/test_marine_panel_ultimate_fe.py
```
