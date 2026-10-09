# sonium-mesher (lib: `sonium_mesher`, version: 0.1.2)

Mesh generation, partitioning, and I/O for FEM/BEM solvers.

## Layout

```
src/
  mesh/
    types.rs              Mesh, Element, ElementType, BoundaryFace,
                          element_neighbors cache + invalidate_caches
    topology.rs           MeshTopology, Face/FaceKey, boundary detection
    generators.rs         Primitive 2D/3D meshes + convex_hull_surface
    refinement.rs         Local element refinement (Bey shortest-diagonal tets,
                          uniform 8-way hex subdivision)
    submesh.rs            Partitioned/sub-mesh views
    tetrahedralize.rs     Bowyer-Watson 3D tet kernel + Lloyd / Steiner /
                          Laplacian post-passes + validate_delaunay diagnostic
    quality.rs            Quality metrics (mean ratio, aspect, dihedrals) + histograms
    validate.rs           Orientation / dedup / degenerate / manifoldness / repair
    stl.rs                Binary STL surface writer
    ply.rs                ASCII PLY surface writer
    gmsh.rs               Gmsh .msh reader (feature: mshio)
    gmsh_write.rs         Gmsh .msh v4.1 ASCII writer (feature: mshio)
    meshlang_bridge.rs    Load .mlg geometry through meshlang-compiler
                          (feature: meshlang, implies mshio)
  partition.rs            RCB + greedy + FM local refinement, with O(1)
                          partition_elements via cached inverse index
  convexhull/             3D quickhull (surfaced through mesh::generators)
```

## Key Types

- `Mesh` — node coordinates, element list, boundary faces. Implements `MeshView` from `sonium-xem` for VTU export.
- `MeshTopology` — face/edge connectivity, used by FEM assembly and boundary tagging.
- `BoundaryFace` — preserves original face winding (sorted form is hash-key only, not stored).
- `SubMesh` — slice into a `Mesh` with remapped indices for distributed assembly.

## Features

- `mshio` — Gmsh `.msh` import
- `meshlang` — MeshLang `.mlg` compilation via `meshlang-compiler` (implies `mshio`)

## Testing

```bash
cargo test -p sonium-mesher --lib
cargo test -p sonium-mesher --features mshio
cargo test -p sonium-mesher --features meshlang
cargo check -p sonium-mesher && cargo clippy -p sonium-mesher
```

## Notes

- The 3D Bowyer-Watson tetrahedraliser uses exact `robust` predicates and
  walk-based point location, but is still best-effort for production meshes.
  `tetrahedralize_surface*` adds flip-based constraint recovery plus bounded
  Steiner subdivision for surface facets; on geometry classes it cannot
  recover (coplanar annular end-cap partitions with acute junction angles,
  re-entrant tube-wall edges — see the module docs in `tetrahedralize.rs`)
  it logs an error and returns the unconstrained Delaunay mesh instead of
  panicking. Run [`validate_delaunay`] on any in-house mesh before shipping;
  production meshes should come from Gmsh / meshlang.
- `partition_mesh` seeds with recursive coordinate bisection (RCB) then runs
  Fiduccia-Mattheyses local refinement bounded by `PartitionOptions::imbalance`.
  The greedy fallback (`partition_mesh_adjacency`) is still exported but is
  rarely the right default — RCB + FM usually beats it on graded meshes.
- `compact_mesh` rewrites `BoundaryFace::element_idx` after node compaction.
- All cache fields on `Mesh` (`node_to_elements`, `topology`,
  `element_neighbors`) are cleared together via `Mesh::invalidate_caches()`;
  external mutators that bypass `add_node` / `add_element` must call it
  themselves.
