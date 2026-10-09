# sonium-xem (lib: `sonium_xem`, version: 0.4.2)

Exchange-model crate: shared, solver-neutral types and validation for BEM/FEM room-acoustics simulators.

## Purpose

Provides shared data structures, serialization, and metadata types used by both `sonium-bem` and `sonium-fem`:
room geometry, sources, directivity, crossovers, boundary configs, JSON schema/envelope validation,
VTU/PVD/SOFA/gallery/chunked-binary I/O, a multiphysics IR, symmetry detection, and a rayon-pool
execution contract.

## Testing

```bash
cargo test -p sonium-xem
cargo check -p sonium-xem && cargo clippy -p sonium-xem
```
