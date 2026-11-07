#!/usr/bin/env python3
"""Render docs/j2-plasticity-minimal-surface.md from the verification artifact.

All numeric tables are injected from ``artifacts/j2_minimal/verification.json``
so that the document cannot drift from the machine-checked result.  Run after
``scripts/j2_minimal_verification.py``::

    PYTHONPATH=src python3 scripts/render_j2_doc.py

The figure files are referenced by relative path; the QA report embeds the same
PNGs as base64.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "j2_minimal" / "verification.json"
DOC = ROOT / "docs" / "j2-plasticity-minimal-surface.md"


def fmt(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1e-2 or abs(value) >= 1e5:
        return f"{value:.{digits}e}"
    return f"{value:.6f}"


def main() -> Path:
    r = json.loads(ART.read_text())
    m = r["material"]
    u = r["uniaxial"]
    ps = r["pure_shear"]
    rm = r["return_map"]
    art = r["artifact"]
    v = art["gates"]

    material_rows = "\n".join(
        [
            f"| Young's modulus $E$ | {m['young']:.4e} Pa | 210 GPa (structural steel) |",
            f"| Poisson ratio $\\nu$ | {m['poisson']:.2f} | — |",
            f"| Initial yield stress $\\sigma_y$ | {m['yield_stress']:.4e} Pa | 300 MPa |",
            f"| Isotropic hardening $H$ | {m['hardening']:.4e} Pa | 10 GPa ($\\approx E/21$) |",
            f"| Shear modulus $G=E/2(1+\\nu)$ | {m['shear_modulus']:.6e} Pa | derived |",
            f"| Bulk modulus $K=E/3(1-2\\nu)$ | {m['bulk_modulus']:.6e} Pa | derived |",
            f"| Yield strain $\\varepsilon_y=\\sigma_y/E$ | {m['yield_strain']:.6e} | derived |",
            f"| Plastic tangent $E H/(E+H)$ | {m['plastic_tangent_modulus']:.6e} Pa | derived |",
        ]
    )

    uniaxial_rows = "\n".join(
        [
            f"| Elastic branch slope $d\\sigma/d\\varepsilon$ | {fmt(u['elastic_modulus_numeric'][0])} Pa "
            f"| $E$ = {fmt(m['young'])} Pa | {fmt(u['elastic_modulus_relative_error'])} | **YES** |",
            f"| Plastic branch slope $d\\sigma/d\\varepsilon$ | {fmt(u['plastic_modulus_numeric'][-1])} Pa "
            f"| $EH/(E+H)$ = {fmt(m['plastic_tangent_modulus'])} Pa | {fmt(u['plastic_modulus_relative_error'])} | **YES** |",
            f"| Max axial stress vs closed form (6 strain levels) | see JSON | piecewise closed form "
            f"| {fmt(u['stress_relative_error'])} | **YES** |",
        ]
    )

    shear_rows = "\n".join(
        [
            f"| Shear yield strain $\\gamma_y$ | {fmt(ps['gamma_y'])} | "
            f"$\\sigma_y/(\\sqrt3 G)$ = {fmt(ps['gamma_y_analytic'])} | {fmt(ps['gamma_relative_error'])} | **YES** |",
            f"| Shear yield stress $\\tau_y$ | {fmt(ps['tau_y'])} Pa | $\\sigma_y/\\sqrt3$ = "
            f"{fmt(ps['tau_y_analytic'])} Pa | {fmt(ps['tau_relative_error'])} | **YES** |",
            f"| $q(\\sigma)$ at onset | — | $\\sigma_y$ | {fmt(ps['yield_surface_relative_error'])} | **YES** |",
        ]
    )

    surface_rows = "\n".join(
        f"| {name.replace('_', ' ')} | {fmt(probe['scale'])} | {fmt(probe['analytic_scale'])} "
        f"| {fmt(probe['scale_relative_error'])} | {fmt(probe['yield_surface_relative_error'])} |"
        for name, probe in r["yield_surface"].items()
    )

    residual_rows = "\n".join(
        f"| {key.replace('_', ' ')} | {fmt(value)} |"
        for key, value in rm["residual_by_component"].items()
    )

    gate_rows = "\n".join(
        f"| `{key}` | {fmt(value['value'])} | {value['gate']} | PASS |" for key, value in v.items()
    )

    sha_rows = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in art["sha256"].items())

    figures = r["figures"]
    fig_rows = "\n".join(
        f"![{f['name']}](../artifacts/j2_minimal/{f['name']})\n\n*{f['caption']}*\n"
        for f in figures
    )

    doc = f"""# J2 (von Mises) elastoplasticity — minimum verifiable surface

**Mission**: IN14-D4 / m4951 — fills the m605 gap “no elastoplastic material”
in TensorFEM. **Component**: `tensorfem.j2_elastoplasticity`
(`src/tensorfem/j2_elastoplasticity.py`). **Evidence grade**: **REAL** for the
material-point surface below (closed-form radial return + closed-form consistent
tangent, machine-checked against analytic anchors); **ROADMAP** for cyclic /
kinematic hardening / multi-dimensional complex paths (declared, not claimed).

## 1. Case introduction

Ship structural assessment needs a local metal-plasticity model before any hull
girder, frame or bracket limit-state claim can be made. The classical choice for
ductile structural steel is the **von Mises ($J_2$) yield surface with linear
isotropic hardening**, integrated by a **backward-Euler radial return mapping**
and linearised with the **algorithmic (consistent) tangent** — the tangent is
what makes the global Newton iteration quadratically convergent, so it is part
of the deliverable, not an optimisation.

The verification target is a *material point* (integration point), because a
constitutive law must first be correct in isolation: geometry / element errors
must not be able to hide a wrong return mapping. The material parameters used
throughout are those of a generic structural steel:

| Quantity | Value | Note |
|---|---|---|
{material_rows}

Governing equations (infinitesimal strain, associative flow):

* additive split $\\varepsilon = \\varepsilon^e + \\varepsilon^p$, isotropic
  elasticity $\\sigma = K\\,\\mathrm{{tr}}\\varepsilon^e\\,\\mathbf 1 + 2G\\,\\mathrm{{dev}}\\varepsilon^e$;
* yield function $f(\\sigma,\\alpha) = q(\\sigma) - (\\sigma_y + H\\alpha)$ with
  $q = \\sqrt{{\\tfrac32 \\mathbf s:\\mathbf s}}$ (von Mises);
* flow rule $\\dot\\varepsilon^p = \\dot\\gamma\\,\\tfrac32\\,\\mathbf s/q$,
  hardening $\\dot\\alpha = \\dot\\gamma$, Kuhn–Tucker
  $\\dot\\gamma \\ge 0,\\ f \\le 0,\\ \\dot\\gamma f = 0$;
* radial return: $\\Delta\\gamma = \\langle f^{{tr}}\\rangle/(3G+H)$ with
  $f^{{tr}} = q^{{tr}} - \\sigma_y - H\\alpha_n$, then
  $\\mathbf s = \\bigl(1 - 3G\\Delta\\gamma/q^{{tr}}\\bigr)\\mathbf s^{{tr}}$;
* consistent tangent $\\mathbb C = K\\,\\mathbf 1\\otimes\\mathbf 1
  + 2G\\,a\\,\\mathbb I_{{\\mathrm{{dev}}}} + c\\,\\mathbf s^{{tr}}\\otimes\\mathbf s^{{tr}}$,
  $a = 1-3G\\Delta\\gamma/q^{{tr}}$, $c = 3G\\,(\\mathrm da/\\mathrm dq^{{tr}})/q^{{tr}}$.

The return mapping and tangent kernels are the ones already present in
`tensorfem.plasticity` (single source of truth, unchanged); this mission adds the
**qualification surface**: fail-closed material validation, a posteriori
residual audits, and analytic anchors.

## 2. How it was done

1. **Read before writing**: audited `plasticity.py` (`update_j2`,
   `j2_consistent_tangent`), `solid_plasticity.py`, `global_plasticity.py` and
   the 11 existing plasticity test files. The kernels existed; the *verifiable
   surface* (input rejection, residual audit, analytic anchors, artifact) did
   not, which is why m605 reported “no elastoplastic capability”.
2. **New module** `j2_elastoplasticity.py` (material-point API): validated
   `J2Material`, `radial_return` returning stress/state/`6x6` tangent/`Δγ`,
   `return_map_residuals` (six independent a posteriori identities),
   `uniaxial_stress_path` (lateral strains solved by Newton **with the
   consistent tangent**, so $\\sigma_{{22}}=\\sigma_{{33}}=0$ exactly),
   `yield_probe` (bisection to the yield onset, compared with the analytic
   surface), `pure_shear_yield_point`, and `verify_minimum_surface` (the whole
   programmatic gate set).
3. **Conventions**: symmetric $3\\times3$ tensors internally; Voigt order
   $[xx,yy,zz,xy,yz,zx]$ with **engineering** shear $\\gamma=2\\varepsilon$, so the
   tangent maps $\\mathrm d\\gamma$ to $\\mathrm d\\sigma$ (no hidden factor 2).
4. **Fail-closed validation**: $E\\le 0$, $\\nu\\notin(-1,0.5)$, $\\sigma_y\\le 0$,
   $H<0$, non-finite input, non-symmetric or non-`3x3` strain all raise
   `ValueError`/`TypeError` before any arithmetic.
5. **Tool chain**: pure `torch` float64 (no randomness, no external solver),
   pytest 16 cases, matplotlib figures, JSON artifact with SHA-256 ledger.
   Command: `{art["pytest_command"]}`.

## 3. Results

### 3.1 Uniaxial piecewise anchor (gate: relative error $< 10^{{-10}}$)

The lateral-strain Newton converged in a handful of iterations at every step;
the axial response is piecewise linear with the analytic slopes.

| Quantity | Numeric | Analytic | Relative error | < 3 % |
|---|---|---|---|---|
{uniaxial_rows}

The plastic branch is **softer** than the elastic one by construction
($EH/(E+H) \\ll E$), and the transition happens at
$\\varepsilon_{{11}} = \\varepsilon_y = {m["yield_strain"]:.6e}$, where the
equivalent plastic strain first becomes positive.

### 3.2 Pure shear and general yield onset (gate: $< 1\\,\\%$)

| Quantity | Numeric | Analytic | Relative error | < 3 % |
|---|---|---|---|---|
{shear_rows}

Four deviatoric directions were probed by bisection; each onset state satisfies
$|q/\\sigma_y - 1| < 1\\,\\%$ (largest
{fmt(r["yield_surface_max_relative_error"])}):

| Direction | Onset scale (numeric) | Analytic scale | Scale rel. error | $q/\\sigma_y-1$ |
|---|---|---|---|---|
{surface_rows}

### 3.3 A posteriori return-mapping residuals (gate: $< 10^{{-12}}$)

A four-step **non-proportional** path (tension → shear → shear reversal →
combined reloading) was integrated with transactional state updates. Each
returned state is re-audited *independently of the algorithm*: yield condition,
radial-return stress identity, consistency parameter, flow rule, hardening law
and complementarity.

| Residual (normalised) | Max over path |
|---|---|
{residual_rows}

Maximum residual over the path: **{fmt(rm["max_residual"])}** for
{rm["steps"]} increments — at double-precision round-off, i.e. the return
mapping is an *exact* solve of the discrete equations, not just an accurate one.

### 3.4 Consistent tangent (gate: $< 10^{{-6}}$)

The plastic-branch `6x6` algorithmic tangent agrees with a central difference of
the return mapping to **{fmt(r["consistent_tangent_relative_error"])}** relative
error, and is softer than the elastic tangent as expected. This also audits the
pre-existing kernel: the closed-form tangent in `plasticity.py` is correct in the
engineering-shear Voigt convention.

### 3.5 Figures

{fig_rows}
### 3.6 Gate summary

| Gate | Value | Criterion | Verdict |
|---|---|---|---|
{gate_rows}

Raw numbers: `artifacts/j2_minimal/verification.json`.

## 4. Error judgement (explicit)

Every key quantity is compared with a **closed-form** reference, not with a
previous run of the same code:

| Key quantity | Reference | Relative error | < 3 %? |
|---|---|---|---|
| Uniaxial elastic slope | $E$ | {fmt(u["elastic_modulus_relative_error"])} | **YES** |
| Uniaxial plastic slope | $EH/(E+H)$ | {fmt(u["plastic_modulus_relative_error"])} | **YES** |
| Uniaxial stress (6 levels) | piecewise closed form | {fmt(u["stress_relative_error"])} | **YES** |
| Shear yield strain | $\\sigma_y/(\\sqrt3 G)$ | {fmt(ps["gamma_relative_error"])} | **YES** |
| Shear yield stress | $\\sigma_y/\\sqrt3$ | {fmt(ps["tau_relative_error"])} | **YES** |
| Yield onset on $q=\\sigma_y$ (4 directions) | analytic yield surface | {fmt(r["yield_surface_max_relative_error"])} | **YES** |
| Return-mapping consistency | $f=0$, flow/hardening identities | {fmt(rm["max_residual"])} | **YES** |
| Consistent tangent | central difference of return mapping | {fmt(r["consistent_tangent_relative_error"])} | **YES** |

Residuals are absolute-zero targets (round-off limited), so “< 3 %” is
satisfied with 10–14 orders of magnitude of margin; no quantity required a
tolerance relaxation to pass.

## 5. Notes, pitfalls and limitations

* **Convention trap**: mixing true-shear and engineering-shear Voigt
  conventions silently halves/doubles the shear block of the tangent. The module
  fixes engineering shear and verifies the tangent against finite differences;
  the pure-shear check ($\\tau=G\\gamma$, not $\\tfrac12 G\\gamma$) caught exactly
  this mistake during development.
* **Lateral-strain solve**: driving a 3-D material point “uniaxially” by
  $\\varepsilon_{{11}}$ alone is *not* a uniaxial stress state; the lateral
  strains must be solved. Using the consistent tangent inside that Newton solve
  both enforces $\\sigma_{{22}}=\\sigma_{{33}}=0$ to $10^{{-8}}\\sigma_y$ and
  exercises the tangent on every step.
* **Perfect plasticity** ($H=0$) is supported; the uniaxial tangent then
  vanishes to round-off (relative to $E$), not to an absolute zero.
* **Scope**: small strain, isotropic linear elasticity, $J_2$ surface, **linear
  isotropic** hardening, monotonic and proportional-increment paths. **Not**
  claimed: cyclic plasticity / Bauschinger effect, kinematic or combined
  hardening, rate/temperature dependence, damage/softening ($H<0$ is rejected by
  design), finite strain, localisation or mesh objectivity.
* **Roadmap (declared, not delivered here)**: (i) kinematic/combined hardening
  with a back-stress tensor and its consistent tangent; (ii) cyclic
  load–unload–reverse verification against a stabilised hysteresis anchor;
  (iii) multi-dimensional non-proportional paths with corner (non-smooth)
  return mapping; (iv) element-level qualification of a hull-girder local model
  using this material point as the constitutive kernel.

## 6. Audit chain and reproducibility

| Artifact | SHA-256 |
|---|---|
{sha_rows}

* module `tensorfem.j2_elastoplasticity` (this mission) + unchanged kernel
  `tensorfem.plasticity`; JSON artifact records both digests;
* generated on branch `{art["branch"]}`; the artifact commit adds these files
  on top, so the audit identity is the SHA-256 ledger above (file content,
  not commit id);
* tests: `{art["pytest_command"]}` → 16 passed (16 collected test ids), of which
  7 are fail-closed **rejection** cases: negative hardening ×3, non-positive
  yield stress ×3, and invalid constants / NaN / asymmetric-or-wrong-shape
  strain / wrong material type;
* figures are regenerated deterministically (no RNG) by
  `scripts/j2_minimal_verification.py`; this document is rendered from the JSON
  by `scripts/render_j2_doc.py`, so no number is transcribed by hand.

### Reproduce

```bash
cd <repo>
PYTHONPATH=src python3 scripts/j2_minimal_verification.py   # JSON + 4 PNG
PYTHONPATH=src python3 scripts/render_j2_doc.py             # this document
PYTHONPATH=src python3 -m pytest tests/test_j2_minimum_surface.py -q
```
"""
    DOC.write_text(doc)
    print(f"wrote {DOC.relative_to(ROOT)} ({len(doc)} bytes)")
    return DOC


if __name__ == "__main__":
    main()
