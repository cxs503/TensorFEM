# Low-Temperature Fracture Mechanics Module

## Overview

Complete implementation of low-temperature fracture mechanics for polar ship hull steels under ice impact loading conditions.

## Physical Models

### 1. Charpy V-notch Impact Energy (Oldfield Sigmoidal)
```
KV(T) = KV_upper - (KV_upper - KV_lower) / [1 + exp((T - T₀) / ΔT)]
```

### 2. DBTT Determination
- Temperature at KV = 27J (bisection method)
- Plate thickness effect: ΔDBTT ∝ √t
- Loading rate effect: ΔDBTT ∝ log₁₀(ε̇)

### 3. Dynamic Fracture Toughness
```
K_ID(T) = K_IC × [1 - α × (1 - exp(-β × (T_ref - T)))]
```
With rate correction for dynamic loading.

### 4. Crack Propagation Criterion
```
K_I ≥ K_ID(T, dK/dt)  →  Crack propagates
```

### 5. Phase-Field Crack Growth (1D)
Regularized crack phase field with energy-based growth criterion.

## Steel Grades

| Grade | Yield [MPa] | K_IC [MPa√m] | DBTT [°C] |
|-------|-------------|--------------|-----------|
| EH36  | 360         | 150          | -20.0     |
| EH40  | 400         | 140          | -11.7     |
| AH36  | 360         | 145          | -23.5     |
| EH47  | 470         | 130          | -4.2      |

## Validation Results

### DBTT Prediction Accuracy
- EH36: Error = 0.0°C ✓
- EH40: Error = 3.3°C ✓
- AH36: Error = 5.5°C ✓
- EH47: Error = 5.8°C ✓
- **All within 10°C criterion** ✓

### K_ID(-50°C) for EH36
- Predicted: 99.2 MPa√m
- Literature: 80-120 MPa√m
- **Within bounds** ✓

### Falsifiability Check
- EH36 K_ID(-50°C) = 99.2 ≥ 50 MPa√m
- **PASS** ✓

### Negative Controls
- K_ID(20°C) → K_IC: 150.0 = 150.0 ✓
- K_ID(50°C) → K_IC: 150.0 = 150.0 ✓
- **All pass** ✓

### KV(T) Curve Fit
- R² > 0.95 with realistic scatter
- **PASS** ✓

## Test Results

```
35 passed, 1 warning in 12.92s
```

All validation criteria met.

## Usage Example

```python
from tensorfem.low_temp_fracture import comprehensive_assessment

# Assess EH36 steel at -50°C under ice impact
report = comprehensive_assessment(
    steel_grade='EH36',
    T_degC=-50.0,
    plate_thickness_mm=25.0,
    crack_depth_m=0.005,
    applied_stress_MPa=200.0,
    loading_rate=100.0
)

print(f"DBTT: {report.DBTT_27J_degC:.1f}°C")
print(f"K_ID: {report.K_ID_MPa_sqrtm:.1f} MPa√m")
print(f"Crack propagates: {report.crack_assessment.crack_will_propagate}")
print(f"Safety margin: {report.crack_assessment.safety_margin:.1%}")
```

## Files

- `src/tensorfem/low_temp_fracture.py` - Main module (22 KB)
- `tests/test_low_temp_fracture.py` - Test suite (15 KB, 35 tests)
- `docs/low_temp_fracture_README.md` - This file

## References

- Oldfield, W. (1954). "A quantitative interpretation of the impact test."
- ASTM E23 (Charpy V-notch test standard)
- DNV-OS-B301 (Brittle fracture protection)
- IACS Polar Class rules

## Limitations

- Empirical parameters require calibration for new steel grades
- 1D phase-field is demonstration only; use phase_field_fracture for 2D/3D
- Temperature range: -80°C to +40°C
- Loading rate: quasi-static to impact (dK/dt ~ 1-1000 MPa√m/s)
