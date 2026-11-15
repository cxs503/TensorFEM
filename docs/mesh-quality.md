# Mesh Quality Metrics Library

## Overview

`tensorfem.mesh_quality` provides comprehensive mesh quality evaluation for FEM meshes,
covering 7 quality dimensions across 2D and 3D element types.

## Metrics

| Metric | Description | Ideal Value | Element Types |
|--------|-------------|-------------|---------------|
| **aspect_ratio** | Max/min edge length ratio | 1.0 | All |
| **skew** | Max angular deviation from ideal shape | 0° | Tri, Quad, Tet, Hex |
| **non_orthogonality** | Angle between face normal and centroid-to-face vector | 0° | Tet, Hex |
| **jacobian** | Min/max Jacobian determinant at Gauss points | >0, uniform | All |
| **volume** | Physical volume/area of element | — | All |
| **watertightness** | All faces shared by exactly 2 elements | True | 3D |
| **manifoldness** | No face shared by >2 elements | True | All |
| **orientation** | Sign of Jacobian (positive/negative/degenerate) | positive | All |

## Supported Element Types

- **2D**: T2D2 (line), CPS3/CPE3 (triangle), CPS4/CPE4 (quad)
- **3D**: C3D4 (tet4), C3D8 (hex8), C3D10 (tet10), C3D20 (hex20)

## Quick Start

```python
from tensorfem.mesh_quality import evaluate_mesh_quality, compute_aspect_ratio

# Simple 2D mesh
nodes = {1: (0., 0.), 2: (1., 0.), 3: (1., 1.), 4: (0., 1.)}
elements = [(1, "CPS4", [1, 2, 3, 4])]

report = evaluate_mesh_quality(nodes, elements)
print(report.to_dict())
# {'n_elements': 1, 'n_nodes': 4, 'watertight': True, 'manifold': True,
#  'max_aspect_ratio': 1.0, 'max_skew_deg': 0.0, 'min_jacobian': 0.25, ...}
```

## API Reference

### `evaluate_mesh_quality(nodes, elements) -> MeshQualityReport`

Main entry point. Takes a node dictionary and element list, returns aggregate report.

### `compute_aspect_ratio(elem_type, coords) -> float`

Per-element aspect ratio computation.

### `compute_skew(elem_type, coords) -> float`

Per-element skew angle (degrees from ideal).

### `compute_jacobian(elem_type, coords) -> (min, max)`

Jacobian determinant at Gauss points.

### `check_watertight_and_manifold(nodes, elements) -> (bool, bool, violations)`

Topological checks for 3D meshes.

### `evaluate_modeldb_quality(modeldb) -> MeshQualityReport`

Convenience wrapper for ModelDB instances.

## Cross-Validation

All metrics are validated against analytical values for ideal elements:

| Element | Aspect | Skew | Jacobian | Volume |
|---------|--------|------|----------|--------|
| Equilateral triangle | 1.0 | 0° | √3/2 ≈ 0.866 | √3/4 ≈ 0.433 |
| Unit square | 1.0 | 0° | 0.25 | 1.0 |
| Regular tetrahedron | 1.0 | 0° | √2/2 ≈ 0.707 | √2/12 ≈ 0.118 |
| Unit cube | 1.0 | 0° | 0.125 | 1.0 |

## Jacobian Convention

The Jacobian is the determinant of the mapping from the reference element to physical space:
- Triangle: reference area = 2, so J = 2 × physical_area
- Quad: reference area = 4 ([-1,1]²), so J = physical_area / 4
- Tet: reference volume = 6, so J = 6 × physical_volume
- Hex: reference volume = 8 ([-1,1]³), so J = physical_volume / 8

## Acceptance Criteria (GAP-CAD-D1-MESH-QUALITY-LIB)

- ≥3 mesh types tested
- Metric differences vs Gmsh/meshmagick < 1e-6 (or documented explanation)
- Usable as CI gate for mesh quality

## Files

- `src/tensorfem/mesh_quality.py` — Core library
- `tests/test_mesh_quality.py` — 45 test cases
- `examples/mesh_quality_demo.py` — Demonstration script
- `docs/mesh-quality.md` — This document
