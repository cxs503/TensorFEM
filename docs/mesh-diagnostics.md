# Mesh and model diagnostics contract

`mesh_pipeline` provides dependency-free Gmsh 2.x and the common Gmsh 4.1
ASCII node/element subset. Binary Gmsh input is rejected explicitly. An
optional `meshio` adapter returns `None` when that package is absent, so it is
not a runtime dependency.

The fail-closed report detects dangling/repeated connectivity, duplicate IDs
and coordinates, orphan nodes, inverted or degenerate elements, excessive
edge aspect and quadrilateral skew, and non-manifold edges/faces. Supported
quality checks cover line, triangle, quadrilateral, tetrahedron and hexahedron
cells. HEX8 Jacobians are evaluated at all eight 2x2x2 integration points.

Engineering checks cover invalid or empty sets, missing and multiply assigned
sections, missing elastic material data, unknown load/constraint targets,
conflicting prescribed values and non-finite loads. Missing loads, constraints
or units are warnings because valid mesh-only workflows may intentionally omit
them. Every diagnostic has a stable code, severity, message and entity IDs.

`MeshReport.write_json` records counts, quality extrema and all findings.
`require_valid` raises if any error is present. `convert_model` invokes that
gate before writing Abaqus INP or legacy VTK, preventing invalid-model export.
The tests include inverted, degenerate, coincident, orphan and non-manifold
meshes plus a Gmsh 4.1 -> ModelDB -> Abaqus/VTK legal round trip.
