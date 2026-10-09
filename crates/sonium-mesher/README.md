# sonium-mesher

Mesh generation, partitioning, and I/O for FEM/BEM solvers.

## Installation

```toml
[dependencies]
sonium-mesher = { version = "0.1" }
```

## Modules

- `mesh::generators` — primitive 2D/3D meshes (rectangles, boxes, spheres, tubes, etc.)
- `mesh::tetrahedralize` — 3D tetrahedralization (Bowyer-Watson; see Known Issues)
- `mesh::topology` — face/edge connectivity, `MeshTopology`, boundary detection
- `mesh::refinement` — local element refinement
- `mesh::submesh` — partitioned/sub-mesh views
- `mesh::gmsh` (feature `mshio`) — Gmsh `.msh` reader
- `mesh::meshlang_bridge` (feature `meshlang`) — load `.mlg` files via the MeshLang compiler
- `partition` — graph-based partitioning for domain decomposition (greedy + neighbor-aware)
- `convexhull` — 3D convex hull (quickhull)

## Feature Flags

- `mshio` — Gmsh `.msh` import via the `mshio` crate
- `meshlang` — Compile MeshLang `.mlg` geometry sources through `meshlang-compiler` and load the resulting Gmsh output. Implies `mshio`.

## Mesh I/O

VTU export is provided by the `MeshView` trait from `sonium-xem`, which `Mesh` implements.

## Testing

```bash
cargo test -p sonium-mesher --lib
cargo check -p sonium-mesher && cargo clippy -p sonium-mesher
```

## Known Issues

- Bowyer-Watson tetrahedralization has three interacting bugs that make it unsuitable for non-trivial inputs (boundary cavity faces lose orientation, `in_circumsphere` uses `orient3d` instead of the lifted in-sphere test, and super-tet vertex stripping in `extract_mesh` assumes fixed indices). Use Gmsh-imported meshes for production work.
