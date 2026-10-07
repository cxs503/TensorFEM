# Safe arbitrary-mesh project schema v1

`tensorfem.mesh-project.v1` accepts inline meshes or a relative JSON mesh file
contained within the project directory. It defines node, element and surface
sets, linear-elastic materials, sections, zero displacement constraints, nodal
loads and output requests. Initial static dispatch is limited to verified
`truss2d`, linear `tet4`/`hex8`, and steady thermal `line2`/`q4` kernels.
Structural projects require force/stress/displacement units and permit only
linear-elastic materials with `U/RF/S/E`. Thermal projects instead require
temperature/heat-rate/conductivity units, steady-thermal material properties,
temperature constraints and `TEMP/FLUX`; cross-physics combinations fail.

Unknown fields, element/material types, missing references, duplicate IDs,
invalid topology, uncovered/multiply-sectioned elements, unsupported DOFs and
ambiguous units fail before solve. Absolute paths, parent traversal, symlink
escape and non-JSON mesh references are rejected. The schema contains no code,
module, callable or expression fields and never performs dynamic import.

Normalized project plus referenced mesh content defines the deterministic job
ID. Jobs retain JSON state, checksum replay, full displacement/reaction and
requested stress/strain fields. Existing case-oriented workflows are unchanged.
