# Versioned engineering project schema

`tensorfem.project.v1` defines Project, Model, Material, Section, Step, Load,
Constraint and OutputRequest records with explicit named references and units.
The first execution allow-list contains the 18-degree-hole hemisphere, the
MacNeal--Harder pinched cylinder and the large-rotation pure-bending shell.
They dispatch only to statically imported existing kernels.

JSON parsing rejects unknown and missing fields at every level. Case names are
opaque allow-list identifiers: module paths, function names, expressions,
plugins and Python source are never imported or evaluated. Cross references,
procedure compatibility, case parameter sets and the five unit categories are
validated before execution. Jobs retain deterministic IDs, status, timing,
checksum replay and structured failure diagnostics. The older hemisphere-only
workflow remains supported unchanged.
