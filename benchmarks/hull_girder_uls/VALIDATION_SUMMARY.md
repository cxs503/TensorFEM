# W23/W24 Hull Girder ULS - Validation Summary

**Mission**: m15094 (W23) + m15972 (W24) matured deliverables  
**PR**: Reopened from closed PR#773 (owner 2026-11-05)  
**Status**: All tests PASS, report published (ID: 17395)

## Deliverables

| File | Lines | Status |
|------|-------|--------|
| `csr_factor_register.json` | 94 | ✅ RESOLVED (factors transcribed from IACS CSR 2019) |
| `csr_utilisation.json` | 34,957 | ✅ Complete (24 cases × 2 scenarios) |
| `grid_convergence.json` | 630 | ✅ All GCI < 5% |
| `section_scan.csv` | 13 | ✅ 12 stations × conditions |
| `docs/assets/w23_hull_girder_uls/*.png` | 5 figures | ✅ Generated from artifacts |

## Test Results

```
tests/test_hull_girder_uls_scan.py: 15 passed
tests/test_csr_uls_factors.py: 21 passed  
tests/test_csr_uls_recheck.py: 15 passed
Total: 51 passed, 0 failed
```

## Key Validation Points

1. **Factor Transcription**: γ_S=1.0, γ_W=1.2(A)/1.3(B), γ_R=1.21(hogging)/1.10(sagging) - transcribed from CSR Pt 1 Ch 5 Sec 2 Table 2, p.344
2. **Dimension Check**: 8 relations, 11 terms, residual 0.0
3. **Internal Consistency**: Max residual 1.1e-16 (criterion < 1e-6)
4. **Grid Convergence**: All cases PASS with GCI < 0.21%
5. **ULS Verdicts**: 18 applicable cases, 6 FAIL (sagging at stations 0.40-0.70 for scenario B, and 0.60-0.70 for scenario A)

## Error Judgement

| Quantity | Reference | Residual | <3%? |
|----------|-----------|----------|------|
| UR S7.1 modulus vs UR S11 target | 96.376 m³ | 0.0 | ✅ |
| Cw branch continuity | 10.75 | 0.0 | ✅ |
| γ* reciprocity | exact | <1e-16 | ✅ |
| Attribution identity | exact | <1e-12 | ✅ |
| 0.19×0.58 vs UR S11 0.11 | 0.11 | 0.18% | ✅ |
| CSR M_wv vs W23 integrated | 10.621 vs 10.121 GNm | 4.7% | ❌ (different models, diagnostic only) |

## Honest Statement

The factor values, load combination, and machine checks are all verified. The absolute PASS/FAIL level depends on two declared substitutions (study M_sw as Msw-U proxy, repo Smith capacity instead of CSR App 2), so the 6 FAIL rows should be read as "this study's capacity model against the CSR factored demand", not as a class compliance verdict.
