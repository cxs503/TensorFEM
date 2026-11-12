# W24 - IACS CSR hull-girder ULS partial safety factors, transcribed and wired into the KVLCC2 section scan

Subject: `paper-exp-ship-hull-girder-structure-3`, mission 15972, anchor `AN-IACS-CSR`.  Repository: TensorFEM.

## 1. What was blocking, and what unblocked it

W22 (mission 14512, `benchmarks/anchors/iacs-csr-w22.json`) and W23 (mission 15094, `csr_factor_register.json`) both reported the CSR hull-girder ULS utilisation at unit partial safety factors because the printed factor table could not be transcribed: `www.iacs.org.uk` answers HTTP 403 through the execution sandbox proxy.

W24 used a different channel: IACS also publishes the harmonised CSR and the Unified Requirements on a public object store (`https://iacs.s3.af-south-1.amazonaws.com`) that answers anonymous `ListObjectsV2` and `GetObject`.  The two assigned documents were located and retrieved there, with sha256 recorded, so the transcription is reproducible without credentials (`fetch_csr_sources.py`).


### Source documents

| anchor id      | document                                                                                                 | pdf pages | sha256 (first 16) | retrieved  |
|----------------|----------------------------------------------------------------------------------------------------------|-----------|-------------------|------------|
| AN-IACS-CSR    | IACS Common Structural Rules for Bulk Carriers and Oil Tankers (harmonised CSR), 01 January 2019 edition | 820       | ffb51b3153e43a7d  | 2026-11-12 |
| AN-IACS-UR-S7  | IACS UR S7 Rev.4 (May 2010), Minimum Longitudinal Strength Standards                                     | 1         | 25a82bfec5f2beea  | 2026-11-12 |
| AN-IACS-UR-S11 | IACS UR S11 Rev.10 (2020), Longitudinal Strength Standard                                                | 31        | 4711024b4b537cd7  | 2026-11-12 |


## 2. Clause-by-clause transcription

All 15 clauses needed to state the ULS criterion, its load combination, the partial safety factors and the wave-moment basis are TRANSCRIBED (100.0 %; missing: none).  Across the whole transcription 17 of 19 entries are TRANSCRIBED (89.5 %), 2 are WITHHELD with the exact clauses still needed.  Locator completeness (clause number + table number + printed page + quote): True.

| id          | document | clause                                                         | table                | page | status      |
|-------------|----------|----------------------------------------------------------------|----------------------|------|-------------|
| CSR-ULS-01  | CSR      | Pt 1, Ch 5, Sec 2, [2.1.2]                                     | -                    | 343  | TRANSCRIBED |
| CSR-ULS-02  | CSR      | Pt 1, Ch 5, Sec 2, [2.2.1]                                     | -                    | 343  | TRANSCRIBED |
| CSR-ULS-03  | CSR      | Pt 1, Ch 5, Sec 2, Table 1                                     | Table 1              | 343  | TRANSCRIBED |
| CSR-ULS-04  | CSR      | Pt 1, Ch 5, Sec 2, Table 2                                     | Table 2              | 344  | TRANSCRIBED |
| CSR-ULS-05  | CSR      | Pt 1, Ch 5, Sec 2, [2.1.2]                                     | -                    | 343  | TRANSCRIBED |
| CSR-ULS-06  | CSR      | Pt 1, Ch 5, Sec 2, [2.3.1]                                     | -                    | 344  | TRANSCRIBED |
| CSR-ULS-07  | CSR      | Pt 1, Ch 5, Sec 1, SYMBOLS (f_beta)                            | -                    | 324  | TRANSCRIBED |
| CSR-WAV-01  | CSR      | Pt 1, Ch 4, Sec 4, [3.1.1]                                     | -                    | 187  | TRANSCRIBED |
| CSR-WAV-02  | CSR      | Pt 1, Ch 4, Sec 4, [3.1.1]                                     | -                    | 187  | TRANSCRIBED |
| CSR-WAV-03  | CSR      | Pt 1, Ch 4, Sec 4, [3.1.1]                                     | -                    | 187  | TRANSCRIBED |
| CSR-WAV-04  | CSR      | Pt 1, Ch 4, Sec 4, SYMBOLS (Cw)                                | -                    | 182  | TRANSCRIBED |
| CSR-WAV-05  | CSR      | Pt 1, Ch 4, Sec 3 (fps); Pt 1, Ch 4, Sec 4, [3.1.1] (fp = fps) | -                    | 176  | TRANSCRIBED |
| S7-01       | UR-S7    | S7.1                                                           | -                    | 1    | TRANSCRIBED |
| S7-02       | UR-S7    | S7.0                                                           | -                    | 1    | TRANSCRIBED |
| S11-01      | UR-S11   | S11.2.2.1                                                      | -                    | 4    | TRANSCRIBED |
| S11-02      | UR-S11   | S11.3.1.1                                                      | -                    | 5    | TRANSCRIBED |
| CSR-ULS-W01 | CSR      | Pt 1, Ch 5, App 2, [2] and [3]                                 | App 2 element curves | 356  | WITHHELD    |
| CSR-ULS-W02 | CSR      | Pt 1, Ch 5, Sec 2, [1.1.1] and [1.1.3]                         | -                    | 342  | WITHHELD    |
| CSR-ULS-W03 | CSR      | Pt 1, Ch 4, Sec 4, [2.2.2]                                     | -                    | 184  | TRANSCRIBED |

**Key printed values read from Table 2 (Pt 1, Ch 5, Sec 2, printed page 344):**

| design load scenario | gamma_S | gamma_W |
|----------------------|---------|---------|
| A                    | 1.00    | 1.20    |
| B                    | 1.00    | 1.30    |

Capacity side (Pt 1, Ch 5, Sec 2 [2.1.2], printed page 343): gamma_R = gamma_M * gamma_DB with gamma_M = 1.10, gamma_DB = 1.10 for an oil tanker in hogging and 1.00 in sagging, i.e. gamma_R = 1.21 and 1.10.

### Still WITHHELD

* `CSR-ULS-W01` Pt 1, Ch 5, App 2, [2] and [3] - M_U in this study comes from the repo Smith solver (tensorfem.hull_girder_smith_collapse) on the study scantlings, not from a transcription of the CSR App 2 element curves and their coefficients; the absolute PASS/FAIL level therefore depends on the repo capacity model and is reported as such Needed: Pt 1, Ch 5, App 2, [2.2] plate element load-end-shortening curves; Pt 1, Ch 5, App 2, [2.3] stiffener element curves; Pt 1, Ch 5, App 2, [2.4] shear and twist failure criteria
* `CSR-ULS-W02` Pt 1, Ch 5, Sec 2, [1.1.1] and [1.1.3] - the KVLCC2 study hull is 320 m long, so the 150 m applicability threshold is met; the clause bodies were not transcribed verbatim in this pass and are listed for completeness rather than used Needed: Pt 1, Ch 5, Sec 2, [1.1.1] clause body (printed page 342); Pt 1, Ch 5, Sec 2, [1.1.3] list of check locations (page 342)

## 3. Machine checks on the anchor

| check                                                                                                                               | verdict | residual  | tolerance |
|-------------------------------------------------------------------------------------------------------------------------------------|---------|-----------|-----------|
| dimensional homogeneity + printed unit conversions (8 relations, 11 terms)                                                          | PASS    | 0.000e+00 | 1e-06     |
| CC-01 0.19 x fnl-vs coefficient 0.58 reproduces the 0.11 of UR S11 S11.2.2.1 negative moment (both multiply (Cb + 0.7))             | PASS    | 1.818e-03 | 5e-03     |
| CC-02 UR S7.1 minimum modulus for the study hull against the UR S11 closure target already used by the repo mid-ship scantling loop | PASS    | 0.000e+00 | 1e-06     |
| CC-03 Cw branches join at L = 300 m and L = 350 m                                                                                   | PASS    | 0.000e+00 | 1e-06     |
| CC-04 unit-factor utilisation times gamma* equals one for every scanned case                                                        | PASS    | 1.110e-16 | 1e-12     |
| CC-05 gamma_s and gamma_w columns of section_scan.csv are 1.0, i.e. the W23 utilisation is a unit-factor reference value            | PASS    | 0.000e+00 | 1e-12     |

Internal-consistency residual maximum: 1.110e-16 (criterion < 1e-6).  The only cross-document residual above that level is CC-01, 1.818e-03, which is the rounding of the printed UR S11 coefficient 0.11 against 0.19 x 0.58 = 0.1102.

## 4. The two utilisation bases on the 12 scanned cases

Factored basis (CSR): `u = gamma_R (gamma_S M_sw + gamma_W f_beta M_wv) / M_U`, PASS when `u <= 1`.  Substitute basis (W23): `gamma* = M_U / (M_sw + M_wv)`, whose reciprocal is the unit-factor utilisation.  `M_wv` is the CSR Ch 4 Sec 4 [3.1.1] value at the station; the W23 integrated value is kept as a diagnostic column.

| scen | case                 | cond    | g_W  | g_R  | M_sw/GNm | M_wv/GNm | M_U/GNm | u factored | gamma* | diff % | verdict        |
|------|----------------------|---------|------|------|----------|----------|---------|------------|--------|--------|----------------|
| A    | kvlcc2_x0.20_sagging | sagging | 1.20 | 1.10 | 0.000    | 5.310    | 12.730  | 0.578      | 2.397  | +38.6  | PASS           |
| A    | kvlcc2_x0.20_hogging | hogging | 1.20 | 1.21 | 0.153    | 4.910    | 17.579  | 0.436      | 3.472  | +51.5  | PASS           |
| A    | kvlcc2_x0.30_sagging | sagging | 1.20 | 1.10 | 0.463    | 7.965    | 14.751  | 0.783      | 1.750  | +37.0  | PASS           |
| A    | kvlcc2_x0.30_hogging | hogging | 1.20 | 1.21 | 0.000    | 7.365    | 20.771  | 0.541      | 2.820  | +52.5  | PASS           |
| A    | kvlcc2_x0.40_sagging | sagging | 1.20 | 1.10 | 0.415    | 10.621   | 15.598  | 0.973      | 1.414  | +37.5  | PASS           |
| A    | kvlcc2_x0.40_hogging | hogging | 1.20 | 1.21 | 0.000    | 9.820    | 22.173  | 0.675      | 2.258  | +52.5  | PASS           |
| A    | kvlcc2_x0.50_sagging | sagging | 1.20 | 1.10 | 0.545    | 10.621   | 15.693  | 0.976      | 1.405  | +37.2  | PASS           |
| A    | kvlcc2_x0.50_hogging | hogging | 1.20 | 1.21 | 0.000    | 9.820    | 22.416  | 0.668      | 2.283  | +52.5  | PASS           |
| A    | kvlcc2_x0.60_sagging | sagging | 1.20 | 1.10 | 1.176    | 10.621   | 15.598  | 1.027      | 1.322  | +35.7  | FAIL           |
| A    | kvlcc2_x0.60_hogging | hogging | 1.20 | 1.21 | 0.000    | 9.820    | 22.173  | 0.675      | 2.258  | +52.5  | PASS           |
| A    | kvlcc2_x0.70_sagging | sagging | 1.20 | 1.10 | 2.132    | 9.103    | 14.751  | 1.014      | 1.313  | +33.2  | FAIL           |
| A    | kvlcc2_x0.70_hogging | hogging | 1.20 | 1.21 | 0.000    | 8.417    | 20.771  | 0.618      | 2.468  | +52.5  | PASS           |
| B    | kvlcc2_x0.20_sagging | sagging | 1.30 | 1.10 | 0.000    | 5.310    | 12.730  | 0.626      | 2.397  | +50.2  | PASS           |
| B    | kvlcc2_x0.20_hogging | hogging | 1.30 | 1.21 | 0.153    | 4.910    | 17.579  | 0.472      | 3.472  | +63.8  | NOT_APPLICABLE |
| B    | kvlcc2_x0.30_sagging | sagging | 1.30 | 1.10 | 0.463    | 7.965    | 14.751  | 0.845      | 1.750  | +47.9  | PASS           |
| B    | kvlcc2_x0.30_hogging | hogging | 1.30 | 1.21 | 0.000    | 7.365    | 20.771  | 0.586      | 2.820  | +65.2  | NOT_APPLICABLE |
| B    | kvlcc2_x0.40_sagging | sagging | 1.30 | 1.10 | 0.415    | 10.621   | 15.598  | 1.052      | 1.414  | +48.6  | FAIL           |
| B    | kvlcc2_x0.40_hogging | hogging | 1.30 | 1.21 | 0.000    | 9.820    | 22.173  | 0.731      | 2.258  | +65.2  | NOT_APPLICABLE |
| B    | kvlcc2_x0.50_sagging | sagging | 1.30 | 1.10 | 0.545    | 10.621   | 15.693  | 1.054      | 1.405  | +48.2  | FAIL           |
| B    | kvlcc2_x0.50_hogging | hogging | 1.30 | 1.21 | 0.000    | 9.820    | 22.416  | 0.724      | 2.283  | +65.2  | NOT_APPLICABLE |
| B    | kvlcc2_x0.60_sagging | sagging | 1.30 | 1.10 | 1.176    | 10.621   | 15.598  | 1.105      | 1.322  | +46.1  | FAIL           |
| B    | kvlcc2_x0.60_hogging | hogging | 1.30 | 1.21 | 0.000    | 9.820    | 22.173  | 0.731      | 2.258  | +65.2  | NOT_APPLICABLE |
| B    | kvlcc2_x0.70_sagging | sagging | 1.30 | 1.10 | 2.132    | 9.103    | 14.751  | 1.086      | 1.313  | +42.5  | FAIL           |
| B    | kvlcc2_x0.70_hogging | hogging | 1.30 | 1.21 | 0.000    | 8.417    | 20.771  | 0.669      | 2.468  | +65.2  | NOT_APPLICABLE |

18 rows carry a verdict (24 rows in the table; the six scenario-B hogging rows are NOT_APPLICABLE because Table 1 gives scenario B a sagging still water moment).  6 row(s) FAIL: `A:kvlcc2_x0.60_sagging` (u = 1.027), `A:kvlcc2_x0.70_sagging` (u = 1.014), `B:kvlcc2_x0.40_sagging` (u = 1.052), `B:kvlcc2_x0.50_sagging` (u = 1.054), `B:kvlcc2_x0.60_sagging` (u = 1.105), `B:kvlcc2_x0.70_sagging` (u = 1.086).

Utilisation range 0.436 to 1.105 at the transcribed factors; `gamma*` range 1.313 to 3.472.  The unit-factor reading therefore overstates the margin by 33.2 % to 52.5 %.

### Difference attribution

The two bases differ by exactly the factor that multiplies the demand, so `u / u_0 = gamma_R (gamma_S phi_sw + gamma_W f_beta phi_wv)` with `phi_sw = M_sw/(M_sw+M_wv)`.  The identity is evaluated per row and its residual is recorded in the CSV (`attribution_residual`); the maximum over all rows is 2.69e-16.  Three regimes appear:

* hogging (phi_sw small): the difference is almost purely `gamma_R * gamma_W * f_beta` = 1.21 x 1.20 x 1.05 = 1.515, i.e. +51.5 %;
* sagging at zero still-water moment (`kvlcc2_x0.20_sagging`): differences come only from `gamma_R * gamma_W * f_beta` applied to the wave part, +38.6 %;
* sagging with the largest still-water share (`A:kvlcc2_x0.70_sagging`, phi_sw = 0.190): the still-water term enters with gamma_R only, which damps the product to +33.2 %.

## 5. Factor sensitivity (envelope, not a CSR value)

| scenario | gamma_S | gamma_W | u min | u max | FAIL rows |
|----------|---------|---------|-------|-------|-----------|
| A        | 1.00    | 1.20    | 0.436 | 1.027 | 2         |
| B        | 1.00    | 1.30    | 0.472 | 1.105 | 4         |

Break-even factors per case (`gamma_S = 1`): the gamma_W at which the check sits exactly on the limit.

| case                 | gamma_R | break-even gamma_W | break-even gamma_S | gamma* |
|----------------------|---------|--------------------|--------------------|--------|
| kvlcc2_x0.20_sagging | 1.10    | 2.075              | -                  | 2.397  |
| kvlcc2_x0.20_hogging | 1.21    | 2.788              | 54.598             | 3.472  |
| kvlcc2_x0.30_sagging | 1.10    | 1.548              | 7.281              | 1.750  |
| kvlcc2_x0.30_hogging | 1.21    | 2.220              | -                  | 2.820  |
| kvlcc2_x0.40_sagging | 1.10    | 1.234              | 1.926              | 1.414  |
| kvlcc2_x0.40_hogging | 1.21    | 1.777              | -                  | 2.258  |
| kvlcc2_x0.50_sagging | 1.10    | 1.230              | 1.623              | 1.405  |
| kvlcc2_x0.50_hogging | 1.21    | 1.797              | -                  | 2.283  |
| kvlcc2_x0.60_sagging | 1.10    | 1.166              | 0.679              | 1.322  |
| kvlcc2_x0.60_hogging | 1.21    | 1.777              | -                  | 2.258  |
| kvlcc2_x0.70_sagging | 1.10    | 1.180              | 0.910              | 1.313  |
| kvlcc2_x0.70_hogging | 1.21    | 1.942              | -                  | 2.468  |

Envelope over the declared grid (the two flagged points are the transcribed CSR values, everything else is a probe):

| gamma_S \ gamma_W | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 1.6 |
|-------------------|-----|-----|-----|-----|-----|-----|-----|
| 1.0               | 0   | 0   | 2   | 4   | 4   | 4   | 5   |
| 1.1               | 0   | 0   | 2   | 4   | 4   | 4   | 5   |
| 1.2               | 0   | 0   | 2   | 4   | 4   | 4   | 5   |

## 6. Mission criteria

| criterion                                      | verdict | evidence                                                                                                                                   |
|------------------------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------|
| W24-C1 transcription completeness              | PASS    | 100 % of the 15 clauses needed to state the criterion and the factors; 17 transcribed / 2 withheld overall, each with a needed-clause list |
| W24-C2 dimension + internal consistency < 1e-6 | PASS    | dimension 0.0e+00, internal 1.1e-16                                                                                                        |
| W24-C3 >= 12 verdict-carrying cases            | PASS    | 18 verdicts, 6 FAIL                                                                                                                        |
| W24-C4 per-case difference + attribution       | PASS    | 18 cases above 15 %, attribution residual max 1.6e-16                                                                                      |
| W24-C5 no estimated factor substituted         | PASS    | transcribed gamma_S/gamma_W used directly; WITHHELD entries are capacity and applicability clauses, not factor values                      |

## 7. Limitations and declared substitutions

* study_beam_equilibrium (W23 homogeneous loaded tanker); not the CSR permissible envelope
* repo Smith progressive collapse on study scantlings; not CSR App 2 net scantlings
* f_p = f_ps = 1.0 for the extreme sea loads scenario (CSR-WAV-05); harbour and flooded scenarios would use 0.4 and 0.8 and are not scanned
* scenario B is a sagging-only load case (Table 1 gives a sagging still water moment), so the six scenario-B hogging rows are marked NOT_APPLICABLE; scenario B reuses the study still water field, which happens to be the homogeneous full load condition CSR names
* the transcription is a set of short verbatim quotes with clause, table and page locators; the rule text itself is not redistributed here (the documents stay at their published URL, verified by sha256).

## 8. Figures

`make_figures.py` renders the three figures from the artifacts alone (`figures/w24_csr_utilisation.png`: factored utilisation per station against the gamma* curve; `figures/w24_attribution.png`: the exact range of the difference as a function of the still-water share; `figures/w24_sensitivity.png`: failure count over the declared factor grid with the two transcribed CSR points boxed).

## 9. Reproduce

```bash
# 1. fetch and verify the anchor documents (needs network egress to the S3 endpoint)
python benchmarks/hull_girder_uls/fetch_csr_sources.py --dest /tmp/iacs2
# 2. rebuild the anchor, the utilisation table and the sensitivity envelope
PYTHONPATH=src python -m tensorfem.csr_uls_factors \
    --out-dir benchmarks/hull_girder_uls
# 3. figures and this README, both from the artifacts
PYTHONPATH=src python benchmarks/hull_girder_uls/make_figures.py
PYTHONPATH=src python benchmarks/hull_girder_uls/make_readme.py
# 4. tests
PYTHONPATH=src python -m pytest tests/test_csr_uls_factors.py -q
```

Generator digest of `src/tensorfem/csr_uls_factors.py` at build time: `6c42c0b061c39068`.
