# End-to-end engineering case packages

Three small packages exercise input, solver, checkpoint/restart, checksum
replay, ResultDB output, and machine/Markdown verification reports.

| package | qualified quantity | independent reference |
|---|---|---|
| Internally pressurized thick component | inner radial displacement | Lamé plane-strain thick-cylinder solution |
| Thin-wall membrane panel | loaded-edge axial displacement | uniform membrane `u=FL/(EtW)` |
| Two-support contact connector | active contact reaction | Schur-complement two-spring equilibrium |

The first uses the existing axisymmetric solid stiffness kernel with a
consistent cylindrical pressure resultant.  The second uses a multi-element
Shell4 thin panel, not one of the classical curved-shell benchmark wrappers.
The third is deliberately a simplified auditable engineering connector: a
coupled two-DOF support system against an exact rigid stop.  It does not claim
bolt pretension, friction, wear, or general deformable 3-D contact.

Every input is a standalone versioned JSON file under
`examples/engineering_cases`.  A deterministic job directory contains the
copied input, status, solver checkpoint, ResultDB JSON index (and HDF5 body
when the optional dependency exists), VTK field output, validation JSON, and concise Markdown.
Completed reports are checksum-verified on replay.  Restart uses the
job-specific checkpoint and rejects a mismatched job identifier.

All three references are analytical and independent of the numerical kernel;
the common qualification rule is relative error below three percent.

```bash
PYTHONWARNINGS=error pytest -q tests/test_engineering_cases.py
python examples/run_engineering_cases.py
```
