# Abaqus `.inp` keyword gate: the reader may not drop model data silently

Status: **verification debt closed for the import gate** (§6 lists what is still
open). Module: `tensorfem.abaqus_io`. Gate script:
`scripts/audit_abaqus_keyword_gate.py` → `artifacts/abaqus_keyword_gate/audit.json`
+ `keyword_gate.png` (every number below is produced by that script).
Tests: `tests/test_abaqus_io.py` (15).

## 1. Why this case

`read_inp` is the only path from a ship structural deck into `ModelDB`, and every
solver, report and export in TensorFEM reads `ModelDB`. The reader is documented
as a *bounded subset* (10 keywords, 4 element types), which is defensible. What
was not defensible: **any keyword outside that subset was skipped without a
word**, and the skip set includes keywords that carry the model itself.

Measured on the pre-fix code (`docs/model-io.md` declares the subset, nothing
enforced it):

| deck | pre-fix result |
|---|---|
| `*Shell Section, elset=PLATE, material=STEEL` / `0.02,` + `*Step, *Static, *End Step` | read **successfully**, `db.sections == []` — the 20 mm plate thickness vanished |
| `*Include, input=part2.inp` | read **successfully**, the included mesh vanished |
| `*Surface` / `*Tie` | read **successfully**, the contact surface definition vanished |

A deck whose thickness is missing still produces a stiffness matrix, a modal
solve and a stress plot; for a ship panel the plate bending stiffness `D = Et^3/12(1-nu^2)`
would then be whatever the downstream default is. This is the same failure class
as a MOCK result: the number is produced, the physics is gone.

## 2. Method

The gate is an explicit two-list policy in the module:

* `SUPPORTED_KEYWORDS` — the 10 keywords that become model data
  (`NODE, ELEMENT, NSET, ELSET, MATERIAL, ELASTIC, SOLID SECTION, TRUSS SECTION,
  BOUNDARY, CLOAD`);
* `NON_MODEL_KEYWORDS` — declared no-ops that carry no model data for this reader
  (`HEADING, PREPRINT, STEP, END STEP, STATIC, OUTPUT, NODE OUTPUT,
  ELEMENT OUTPUT, NODE PRINT, EL PRINT, CONTROLS, RESTART, SYSTEM,
  PHYSICAL CONSTANTS`, plus the internal `END` terminator);
* everything else raises `ValueError` naming the keyword **and the line number**,
  with an actionable hint for `*INCLUDE` and for any unrecognised `*... SECTION`
  (a section outside the subset loses its thickness/properties, which is exactly
  the silent error being removed).

Hardening that came with the gate: a missing required option (`*NSET` without
`NSET=`, `*SOLID SECTION` without `ELSET=`/`MATERIAL=`, `*MATERIAL` without
`NAME=`) and an unsupported element type now raise a located `ValueError`
instead of `KeyError` / an unlocated message.

`write_inp` is unchanged; only the reader's policy changed, so every existing
round trip is bit-identical (verified).

## 3. Results

### 3.1 Gate decision matrix (22 real keywords, read back from `read_inp`)

| outcome | count | keywords |
|---|---:|---|
| accepted, model data | 8 | NODE, ELEMENT, NSET, ELSET, MATERIAL, SOLID SECTION, BOUNDARY, CLOAD |
| accepted, non-model no-op | 7 | STEP, STATIC, OUTPUT, ELEMENT OUTPUT, NODE PRINT, CONTROLS, RESTART |
| **rejected by the gate** | 7 | INCLUDE, SHELL SECTION, BEAM SECTION, MEMBRANE SECTION, SURFACE, TIE, DISTRIBUTION |

Each candidate is appended to one valid base model and the outcome is classified
from the reader's behaviour (`no-op` = the `ModelDB` is field-by-field unchanged),
not from the source lists — the matrix can therefore catch a keyword that is in a
list but never handled.

### 3.2 Round-trip precision (why the writer emits `%.17g`)

| digits | max relative coordinate error | bit-exact samples |
|---:|---|---:|
| 6 | 1.71e-06 | 2 / 8 |
| 9 | 1.14e-09 | 2 / 8 |
| 12 | 2.80e-12 | 2 / 8 |
| 15 | 2.76e-15 | 2 / 8 |
| **17** | **0.00e+00** | **8 / 8** |

Read → write → read on the verification deck returns field-by-field identical
`ModelDB` (`nodes`, `elements`, `node_sets`, `element_sets`, `materials`,
`sections`, `boundaries`, `loads`), i.e. external IDs, topology and properties
survive the interop boundary exactly.

## 4. Figures

`artifacts/abaqus_keyword_gate/keyword_gate.png`, left: one row per keyword with
its measured outcome (colour = model data / no-op / rejected) and the declared
class in brackets; right: round-trip error versus emitted digits on a log scale,
with the `10^-(d-1)` sampling bound, the float64 roundoff band and open circles
marking bit-exact cases.

## 5. Caveats

* `*STEP` is an accepted no-op: the step name, kind and time increments are **not**
  stored (`ModelDB.steps` keeps its default), and `write_inp` does not emit steps
  or output requests. Temporal analysis data is outside this contract.
* `*INCLUDE` is rejected, not resolved; decks must be expanded first.
* Shell/beam/membrane sections and element types beyond `T2D2/CPS4/CPE4/C3D8`
  remain unsupported — now loudly, previously silently.
* `GENERATE` sets are expanded verbatim: `20, 30, 1` really does require nodes
  21-29 (this bit the author's own fixture before `ModelDB.validate` caught it —
  the validator, not the reader, is what rejects it).

## 6. Error judgement

| Quantity | Baseline | Value | < 3 % ? |
|---|---|---|---|
| Model-bearing keywords silently dropped | pre-fix 3/3 adversarial decks (SHELL SECTION, INCLUDE, SURFACE) | **0/3** after fix | yes (categorical, exact) |
| Gate misclassification over 22 keywords | declared policy | 22/22 agree with measured behaviour | yes (0 disagreements) |
| Coordinate round-trip error | exact decimal value | 0.0 at 17 digits (8/8 bit-exact) | yes |
| Coordinate error at 6 digits (rejected alternative) | exact decimal value | 1.71e-06 | yes, but fails the *bit-exact* contract |

Remaining open: the ordering of `*ELASTIC` rows beyond the first (multi-point
elastic data) is not verified; no independent Abaqus run is available in this
environment, so "the deck means what Abaqus means" is verified only for the
documented subset, by construction and by `ModelDB.validate`, not against Abaqus
itself.
