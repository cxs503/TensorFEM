# Audited 8x8 and 12x12 generation run

The panel generation scheduler advances exactly one point per invocation of
the resumable solver.  Every accepted point publishes its immutable binary
generation first, then its manifest pointer, then a separate event file.  The
event files form a predecessor SHA-256 chain and retain line-search diagnostics
and energy/equilibrium evidence.  A failed or wall-limited point stops the
scheduler without changing the last committed generation.

## 8x8 result

The backtracking path resumed from generation 659 and reached the requested
generation 680.  All 21 new points were accepted on their first attempt:

- terminal load: 1011112.5769 N
- terminal centre deflection: 0.0172207089 m
- terminal yielded fraction: 2.03125%
- attempts/rejections: 21/0
- Newton iterations: 63
- line-search evaluations: 42
- reduced corrections: 0 (all accepted `alpha=1`)
- maximum relative equilibrium norm: `1.27e-12`
- whole-path and incremental energy gates: 21/21 passed
- generation-680 state SHA-256: `2ea12edbca13763718a5c85e3971cefd06c1524caf0c099a9f3bbb568de02047`
- final event SHA-256: `b64dd6bfb2f1a774844ecdd9e0955121645d3a20f79196cfe577525f5180d1fd`

The response is still increasing and no peak or post-peak branch is observed.
Generation 680 is therefore a verified endpoint, not an ultimate-load claim.

## 12x12 result

Generation 25 was migrated to an independent backtracking identity and then
advanced to generation 30:

- migration SHA-256: `e9cc6daa0c3808a4b42a02215cd2acc6b20a8e1b72e676f16b9a87a3824bc092`
- terminal load: 414480.4666 N
- terminal centre deflection: 0.0038043586 m
- attempts/rejections: 5/0
- Newton iterations: 14
- line-search evaluations: 9
- reduced corrections: 0
- maximum relative equilibrium norm: `8.33e-12`
- whole-path and incremental energy gates: 5/5 passed
- generation-30 state SHA-256: `7b5f1c73a233a9e89afb2c5105bbc31193a30e376d5a7ada456c8c130e158290`
- final event SHA-256: `0da1e90cec26ed4948286406aee0d867efa4ba7fad5dd54f0a26d0a573a6a637`

This path also remains pre-buckling and monotonic.  No peak claim is made.

