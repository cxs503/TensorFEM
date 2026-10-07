# ResultDB v2: steps, frames and integration-point history

## Compatibility

The original `tensorfem.result-db.v1` implementation is unchanged and remains
readable through `read_result_db`. `read_result_db_compatible` detects v1/v2;
with `migrate=True`, a v1 database becomes an in-memory v2 database containing
one `Legacy` step and frame. Migration is explicit and never rewrites the
source files.

## Data model

`ResultDBv2` stores shared topology and an ordered collection of named steps.
Every frame has an integer index, physical time and load factor. Fields are
located at nodes, elements or integration points. `FieldSpec` records physical
unit and ordered components; integration-point tensors use
`(element, integration_point, component)`.

Independent `HistorySeries` objects record step name, time, load factor,
values, unit and components. This supports material-point stress, plastic
hardening variables, energy and probe histories without duplicating complete
frames.

## Storage and lazy queries

The JSON file is a complete inventory containing dataset path, shape, dtype,
location, unit, components and SHA-256 checksum. HDF5 datasets are chunked on
their first dimension. `query_frame_field` opens only one requested
step/frame/location/field and can select IDs and named components.
`query_history_series` reads only a requested time slice.

If `h5py` is absent, writing still emits a valid JSON inventory with
`body.available=false` and returns `False`. Reading field data raises an
actionable optional-dependency error; TensorFEM does not install or silently
substitute a different backend.

## Atomic completion and integrity

The body is first written to a temporary HDF5 path with `complete=false`.
After all datasets flush, the marker changes to true and the body is renamed
to a checksum-derived immutable filename. The JSON summary is written to a
temporary file and atomically published last. Readers require both JSON
`complete=true` and HDF5 `complete=true`; abandoned temporary files or an
interrupted summary fail closed. Every tensor, metadata document and complete
database has a checksum verified by default.

## Plastic load/unload qualification

The automated example advances one small-strain bilinear material point from
zero through plastic loading and unloading. Three frames preserve nodal
displacement plus integration-point `S11` and `alpha`. Separate history series
round-trip time/load/value data. Tests verify:

- two steps and three frames retain ordering and metadata;
- integration-point stress/alpha reproduce bit-for-bit after HDF5 round-trip;
- lazy element-ID/component and history-slice queries return only requested
  data;
- checksum tampering and incomplete publication fail closed;
- v1 data remain readable and migrate explicitly;
- invalid frame ordering or component schemas fail closed.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q tests/test_result_db_v2.py
python examples/plastic_history_result_db.py
```
