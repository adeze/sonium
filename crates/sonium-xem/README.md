# sonium-xem-common

Common types and utilities for BEM/FEM room acoustics simulators.

Shared data structures for numerical acoustics simulation including room geometry definitions, sound source configurations, JSON configuration parsing, and output formatting.

The canonical room-document schema is available from
`sonium_xem::room_config_json_schema()` and
`sonium_xem::room_config_json_schema_string()`. Strict readers should use
`RoomConfig::from_json_checked` (or `from_json_migrated` for legacy documents),
which reject unknown root fields before solver entry. The cross-language v1
fixture is kept at `fixtures/room-config-v1.json`.

The backend-neutral multiphysics envelope has the matching v1 artifact at
`fixtures/multiphysics-problem-v1.json`. Rust-free consumers can validate it
with `io::read_external_multiphysics_document`; the independent Python and
Node readers under `interop/` exercise the same root contract. Run
`just xem-interop` to check both golden envelopes.
The executable compatibility matrix is exposed through
`schema::multiphysics_schema_compatibility_matrix()`: v1 is directly readable
and every other version is rejected until an explicit migration is added.

For release artifacts that include visualization or HRTF output, run
`just xem-external-interop <output-dir>` in an environment with Python
`h5py`. It creates a deterministic VTU/PVD/SOFA set, parses VTU/PVD with
Python's standard-library XML reader, and reads SOFA through HDF5 rather than
the crate's `sofa-reader` dependency. The `xem-compatibility` workflow runs
the same independent-reader smoke on Linux, macOS, and Windows, provisioning
HDF5 explicitly for the Rust SOFA writer. This does not replace the separate
ParaView visual smoke on runners where ParaView is installed.

Tools that do not link the Rust model can use the generic envelope reader:
`sonium_xem::io::read_external_room_document`. It validates the version,
required root members, and root JSON types. For a language-neutral contract
check, `validate_schema_envelope` applies the generated Draft 2020-12 room
schema recursively, including nested source, directivity, boundary, solver,
and convention fields. It remains dependency-free so the same contract is
available to WASM consumers.
Use `read_external_room_document_checked` when the parsed document will be
passed directly into a solver; it combines schema validation with envelope
reading.
