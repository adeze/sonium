# Gotchas

- Sonium is pre-production. Prefer clean forward changes over backward-compatible shims, migration layers, or preserved legacy behavior unless the user explicitly asks for compatibility.
- For generated geometry, keep semantic markers separate from exported surfaces/volumes. Visual QA catches wrong-side caps, accidental marker panels, and missing mechanical clearances that type checks will not.
- Backend-specific CPU/Metal/CUDA branching belongs in `sonium-compute` and solver-level orchestration, not in BEM assembly APIs. BEM may expose algorithmic paths, but should not grow per-backend scheduling logic.
- For CUDA runs, never treat host RAM as the device budget. Preflight and runtime guards must account for free VRAM on the selected device, then take the smaller relevant budget when host-side data also exists.
- Default `cargo test` paths must stay unit-scale. Large sparse/direct-solver scaling cases belong in benches or QA binaries, not ordinary dev/CI tests.
