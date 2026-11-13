# TensorFEM Case Schema JSON & Units Converter

## Overview

This module adds **JSON round-trip serialization** and **multi-unit-system conversion** to the TensorFEM benchmark case schema (`benchmark_cases.py`).

## What was added

### 1. `from_dict()` factory methods

All three dataclasses now support reconstruction from plain dicts (as produced by `json.loads`):

```python
c = Citation.from_dict({"authors": "...", "title": "...", ...})
q = ReferenceQuantity.from_dict({"name": "...", "value": 0.5, ...})
case = BenchmarkCase.from_dict(json.loads(json_text))
```

### 2. `to_json()` / `from_json()` methods

`BenchmarkCase` can now serialise to and deserialise from JSON strings directly:

```python
text = case.to_json()          # compact
text = case.to_json(indent=2)  # pretty
case = BenchmarkCase.from_json(text)
```

Round-trip is **bit-exact**: `BenchmarkCase.from_json(case.to_json()) == case`.

### 3. Units converter

A dimensional-analysis-based converter supporting common FEM unit systems:

| System | Length | Mass | Time | Force | Pressure |
|--------|--------|------|------|-------|----------|
| SI | m | kg | s | N | Pa |
| mm-tonne-s | mm | tonne (1000 kg) | s | N | MPa |
| mm-kg-s | mm | kg | s | N | MPa |
| inch-lbf-s | inch | lb | s | lbf | psi |

**Scalar conversion:**
```python
from tensorfem.benchmark_cases import convert_value
convert_value(1.0, "m", "mm")       # → 1000.0
convert_value(1e6, "Pa", "MPa")     # → 1.0
convert_value(1000, "N", "kN")      # → 1.0
```

**Case-level conversion:**
```python
converted = case.with_units_converted("SI", "mm-tonne-s")
# geometry lengths ×1000, young_modulus ÷1e6, quick_values ×1000
```

The converter uses dimensional exponents (L, M, T) to derive conversion factors for any quantity category (length, mass, force, pressure, density, etc.).

### 4. `convert_case_units()` function

Low-level function that converts a case dict between unit systems, touching:
- `geometry` — length-type fields (radius, thickness, depth, etc.)
- `material` — young_modulus (pressure dim), density
- `quick_values` — scaled by the reference quantity's dimension
- `quantity.value` — scaled by the reference quantity's dimension

Non-numeric fields (case_id, mesh_sequence, boundary_conditions, etc.) are preserved verbatim.

## Design decisions

- **Immutable dataclasses**: All `from_dict` methods return new frozen instances; no in-place mutation.
- **Dimensional analysis**: Conversion factors are derived from (L, M, T) exponents, not hardcoded per-quantity. This ensures consistency across all FEM quantities.
- **Version suffix**: `with_units_converted()` appends `+{from}2{to}` to `case_version` to record the conversion history.
- **Hash preservation**: Round-trip JSON preserves the case hash; tampered JSON fails validation.
- **Unknown units passthrough**: Unrecognised unit tags are returned unchanged (fail-safe, not fail-hard).

## Tests

33 new tests in `tests/test_case_schema_json_units.py`:
- Citation / ReferenceQuantity / BenchmarkCase round-trip
- JSON canonicalisation and tamper detection
- Scalar unit conversion (m↔mm, Pa↔MPa, N↔kN, kg↔tonne, inch↔mm)
- Case-level conversion (geometry, material, quick_values, reference quantity)
- Round-trip identity (SI → mm-tonne-s → SI recovers original values)
- `with_units_converted()` method validation
- Integration: converted cases produce consistent results

All 5 existing `test_benchmark_cases.py` tests continue to pass.
