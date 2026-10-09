# MeshLang WASM frontend package

This package is the small browser-facing adapter around the generated
`sonium-roomsim` wasm-bindgen module. It validates the versioned
`meshlangFrontendContract` before exposing parser, grammar, canonical-IR, and
registry-preview, and bounded `meshProjection` artifact operations. `MeshLangEditor` provides the stateful parser
cache with explicit invalidation and telemetry.

The adapter never reads files, invokes Gmsh, or starts a solver. Build the
generated module from the parent crate with `npm run build`; the generated
`pkg/` directory is then included in a package publish.

`loadGeneratedMeshLangFrontend()` imports that package-relative module and
awaits the optional wasm-bindgen initializer before validating the capability
contract. Browser workers and bundlers may inject a zero-argument async module
loader for import maps or hosted assets; the compiler remains free of network
and filesystem access.

The loader rejects null, array, and primitive module results before probing
exports. Initializer failures are reported as typed boundary errors, while
both in-place initialization and returned export namespaces remain supported.

`fetchTrustedPackageJson()` is an opt-in browser transport boundary for host
package registries. It permits only absolute HTTP(S) URLs without credentials,
queries, or fragments, requests JSON without redirects, bounds the response to
8 MiB, and requires a JSON object envelope. The frontend convenience method
`previewWithTrustedPackageFromFetch()` passes that exact response to the Rust
signature/trust-policy verifier; the adapter never treats transport success as
package trust.

`parseProjectionMeshArtifact()` and `validateProjectionMeshArtifact()` mirror
the Rust projection-artifact boundary for browser and optimization clients:
schema version, exact fields, finite positive bounds/vertices, bounded grid cells,
triangle indices, degeneracy, and the 64 MiB UTF-8 payload limit are checked
before a decoded object is consumed.
