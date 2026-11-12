# Anchor `fem_cylinder_radiation` — cylinder modal radiation efficiency

Verification anchor for `tensorfem.struct_acoustics.cylinder_radiation` (verification
debt item #3 of the m14722 census: the module had a pytest gate suite but no
standardised `benchmarks/verified/` anchor with a graded manifest, ladder and
figures — the driver `run.py` landed without any evidence file until m16372).

| oracle | closed form (no hand-copied numbers, all re-derived in the driver) |
|---|---|
| low-`ka` piston law (ring mode `n = 0`) | `sigma -> (pi/2) k a` (Junger & Feit 1987/1992) |
| high-`ka` plateau | `sigma_0 -> 1`, `sigma_n -> 1/2` for `n >= 1` |
| axial cutoff | `k_x > k` ⇒ evanescent, conductance collapses by >15 orders |
| window closure | finite-mode power ∫`G_n(k_x) |V(k_x)|²dk_x` → infinite-cylinder modal value |
| pulsating sphere | `Pi = 2 pi a^2 rho c V^2 (ka)^2/(1+(ka)^2)` (exact Rayleigh) |
| compact body | monopole + dipole asymptote reproduces the sphere at `ka = 0.1` |
| rigid-body / scaling | `v_n = 0` radiates 0; `Pi ~ |V|^2` |
| surface grid | lateral area `2 pi R L`, rigid caps carry zero weight |

Reproduce (single command, ~6 s on 8 threads, no cluster needed):

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_cylinder_radiation/run.py
PYTHONPATH=src python scripts/gen_cylinder_radiation_doc.py      # refresh the doc tables
PYTHONPATH=src pytest tests/test_verified_cylinder_radiation_anchor.py -q
```

Evidence files: `result.json` (every number), `gates_verified.json` (manifest:
per gate the reference formula, value, threshold, verdict + limitations),
`figs/*.png`.  The artifact test re-derives the oracles independently of the
driver, so editing the evidence fails the tests instead of passing silently.

Limitations: air only (fluid loading / added mass neglected); the *prescribed
velocity field* is verified, not the coupled shell eigenvalue problem; the
window closure is a 1-D axial-spectrum quadrature, so it inherits the
infinite-cylinder closure (finite-capped-cylinder end effects are not part of
this anchor — see `docs/reports/fem-structural-acoustics.md` gates C3/C4).
