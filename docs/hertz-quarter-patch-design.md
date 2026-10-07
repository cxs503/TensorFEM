# Hertz quarter-domain local patch

The positive Hertz path uses symmetry planes at `x=0` and `y=0`.  The central
`[0,a] x [0,a]` surface patch has eight uniform cells per contact radius.
Independent transition layers expand to a far boundary at `8a`; the first
depth interval is the same order as the surface spacing and subsequent layers
grow gradually.  Topology gates reject fewer than six cells per radius, a
domain below `6a`, or excessive adjacent spacing growth.

This module freezes the mesh design and quality contract only.  It is not a
Hertz qualification: volume connectivity, symmetry constraints, sparse solid
assembly and Mortar integration must still be connected and pass two mesh
levels below 3% before the formal Hertz evidence schema may be emitted.
