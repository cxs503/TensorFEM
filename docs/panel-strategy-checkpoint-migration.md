# Audited panel checkpoint strategy migration

`migrate_panel_checkpoint_strategy` derives a backtracking checkpoint from an
existing fixed-Newton panel generation without modifying the source.  It
verifies the source manifest and binary SHA-256 values, gives the target a
separate physical-control identity, and persists a provenance hash chain.

The migration compares, without tolerance, displacement, material state,
previous arc increment, load factor, next step size, reference recoverable
energy, cumulative external work, cumulative plastic dissipation, energy
definition, and ledger completeness before and after serialization.  A
failure in any comparison aborts publication.  Existing control-migration
provenance is retained separately from strategy-migration provenance.

## 8x8 generation 658 migration

Source:

- job key: `89ef439f419b310689f2`
- generation: 658
- manifest SHA-256: `c6a2fc3f02dd676b454e32d534d44254a802e43e478f40bca6a50c0c31a2ebdd`
- checkpoint SHA-256: `84898f8776f3989d33d611ce8ed52f46da74fdd95ca38d3117d2736679aaaeb3`
- accepted-state SHA-256: `6cd660bcdcfa5d92cd2b848fa60f9b02e5bba4c3c84c5986130ccc892a492bb8`

Target:

- job key: `30549e9e72230a8c6239`
- strategy: `backtracking`
- migration SHA-256: `b8067e7d8a64f3f27c8a354074762a18e807819a8a58cfb0bde12aac70255f5c`
- mechanical state exact: true
- energy ledger exact: true

The final audited evidence is under
`.qualification/v100-panel-8x8-backtracking-final/`.  With `chunk_size=1` and
a 300-second wall limit, generation 659 was accepted on its first attempt in
about 30 seconds.  It required three Newton iterations and two line-search
evaluations; both accepted full correction (`alpha=1`), so backtracking was
available but not needed for this particular point.

Generation 659 reached 969736.0163 N.  Its relative equilibrium norm was
`4.15e-15`; whole-path and incremental energy gates both passed.  The new
state hash is `996f16afa77ac7b6aac343b0088cac5c05842e6cf7e9149f37df389db4f15ad4`.
The source generation and its hashes remain unchanged.

