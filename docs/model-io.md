# Engineering model and file-I/O contract

`ModelDB` is solver-neutral and preserves external node/element IDs, sets,
materials, sections, boundary conditions, concentrated loads, steps and output
requests. It rejects dangling connectivity, sets and section references.

Supported Abaqus input keywords are `*NODE`, `*ELEMENT` (`T2D2`, `CPS4`,
`CPE4`, `C3D8`), `*NSET`, `*ELSET`, `*MATERIAL`, `*ELASTIC`, `*SOLID SECTION`,
`*TRUSS SECTION`, `*BOUNDARY` and `*CLOAD`. Keyword and set names are
case-insensitive. This is not advertised as a complete Abaqus parser.
The same supported subset can be written back to INP; a round-trip regression
checks IDs, topology, sets, material data, loads and boundary conditions.

Gmsh v2 ASCII line, quadrilateral, tetrahedron and hexahedron cells can be
imported. VTK legacy unstructured-grid output needs no dependency and writes
point scalar/vector and cell scalar results. HDF5 output is optional: when
`h5py` is absent it returns `False` without failing the analysis.

The end-to-end regression imports a non-contiguously numbered Abaqus T2D2 bar,
adapts it to the verified TensorFEM truss solver and exports displacement and
stress. Its axial displacement is checked against `FL/(EA)` with a strict
relative-error threshold below 3% (observed at floating-point roundoff).
