# sonium-bem ndarray → Vec/DenseMatrix Migration

Map of every public ndarray surface (Array1 / Array2 / ArrayView) in sonium-bem
that was migrated off `oxiblas-ndarray` (re-exported via
`crate::core::array`), grouped by module. This migration is complete —
`oxiblas-ndarray` has been removed from the workspace.

## Replacement type policy (task #5)

| ndarray role                                | replacement                                   | notes                                                              |
|---------------------------------------------|-----------------------------------------------|--------------------------------------------------------------------|
| `Array1<f64>` used as a 3-vector            | `[f64; 3]`                                    | drops heap alloc entirely — element normals, centers, directions   |
| `Array1<f64>` used as shape-function values | `[f64; N]` if N is fixed, else `Vec<f64>`     | tri3 = 3, quad4 = 4; prefer inline arrays                          |
| `Array1<f64>` solver RHS / solution         | `Vec<f64>`                                    |                                                                    |
| `Array1<Complex64>` solver RHS / solution   | `Vec<Complex64>`                              | matches new `gmres`/`cg`/`bicgstab` signatures                     |
| `Array2<f64>` mesh nodes table (N×3)        | `Vec<[f64; 3]>`                               | iteration-friendly, zero-cost index by node id                     |
| `Array2<f64>` element coords (k×3)          | `[[f64; 3]; K]` (K=3 or 4) or `Vec<[f64; 3]>` | small fixed K → inline                                             |
| `Array2<Complex64>` dense BEM matrix        | `sonium_solvers::dense::DenseMatrix<Complex64>` | already used by solver layer; row-major                          |
| `Vec<Array2<Complex64>>` table of matrices  | `Vec<DenseMatrix<Complex64>>`                 | T/S translation tables in FMM                                      |
| `ArrayView2<'_, T>` (read-only matrix view) | `&DenseMatrix<T>` or `(&[T], rows, cols)`     | depends on caller — prefer the typed ref                           |
| `&Array1<T>` argument                       | `&[T]`                                        | matches solver `apply` / `matvec` signatures                       |

`sonium_solvers::dense::DenseMatrix<T>` is the canonical dense type for the new
API. It owns row-major storage and already plugs into `LinearOperator`.

## Hot-path vs metadata boundary

Marked **hot** = inner-loop / per-element / per-matvec. Migration must avoid
introducing per-call allocations. Marked **cold** = called once per problem
(setup / IO / post-processing); allocation cost is irrelevant.

---

## Module: `core/mesh/element.rs` (task #6 — leaf utilities)

All hot — called per element during assembly.

| Symbol                                  | Current                                                  | Target                                                       |
|-----------------------------------------|----------------------------------------------------------|--------------------------------------------------------------|
| `Element::shape_functions(s, t)`        | returns `Array1<f64>`                                    | returns `[f64; 4]` (tri3 pads to 3, quad4 uses 4)            |
| `Element::shape_derivatives(s, t)`      | `(Array1<f64>, Array1<f64>)`                             | `([f64; 4], [f64; 4])`                                       |
| `Element::local_to_global(s, t, nodes)` | `nodes: &Array2<f64>` → `Array1<f64>`                    | `nodes: &[[f64; 3]]` → `[f64; 3]`                            |
| `Element::collocation_points`           | returns `Vec<Array1<f64>>`                               | returns `Vec<[f64; 3]>`                                      |
| `Element::collocation_normals`          | returns `Vec<Array1<f64>>`                               | returns `Vec<[f64; 3]>`                                      |
| `Element::element_coords` (priv)        | returns `Array2<f64>`                                    | returns `Vec<[f64; 3]>` (or smallvec for tri/quad)           |
| `Element::normal_and_jacobian`          | returns `(Array1<f64>, f64)`                             | returns `([f64; 3], f64)`                                    |
| `cross_product(a, b)`                   | `&Array1<f64>, &Array1<f64>` → `Array1<f64>`             | `[f64; 3], [f64; 3]` → `[f64; 3]` (by value, copy is cheap)  |
| `dot_product(a, b)`                     | `&Array1<f64>, &Array1<f64>` → `f64`                     | `[f64; 3], [f64; 3]` → `f64`                                 |
| `normalize(v)`                          | `&Array1<f64>` → `(Array1<f64>, f64)`                    | `[f64; 3]` → `([f64; 3], f64)`                               |
| `compute_element_center(nodes, conn)`   | `&Array2<f64>, &[usize]` → `Array1<f64>`                 | `&[[f64; 3]], &[usize]` → `[f64; 3]`                         |
| `interpolate_normal(node_normals, w)`   | `&Array2<f64>, &[f64]` → `Array1<f64>`                   | `&[[f64; 3]], &[f64]` → `[f64; 3]`                           |

## Module: `core/mesh/barycentric.rs` (task #7)

| Symbol                                       | Current                                          | Target                                       |
|----------------------------------------------|--------------------------------------------------|----------------------------------------------|
| `BarycentricRefinement::refined_nodes`       | `Array2<f64>` field                              | `Vec<[f64; 3]>`                              |
| `refine_from_elements(elements, nodes)`      | `nodes: &Array2<f64>`                            | `nodes: &[[f64; 3]]`                         |

## Module: `core/mesh/generators.rs`, `octree.rs` (task #7)

Internal generators that produce `Array2<f64>` node tables — switch to
`Vec<[f64; 3]>`. Octree pulls coordinates via index, mechanical.

## Module: `core/types.rs` (task #7 — mesh + types)

| Symbol                          | Current                                          | Target                                  |
|---------------------------------|--------------------------------------------------|-----------------------------------------|
| `Element::normal`               | `Array1<f64>` field                              | `[f64; 3]`                              |
| `Element::node_normals`         | `Array2<f64>` field                              | `Vec<[f64; 3]>`                         |
| `Element::center`               | `Array1<f64>` field                              | `[f64; 3]`                              |
| `Mesh::nodes`                   | `Array2<f64>` field                              | `Vec<[f64; 3]>`                         |
| `Mesh::element_nodes(element)`  | returns `Array2<f64>`                            | returns `Vec<[f64; 3]>`                 |
| `Cluster::center`               | `Array1<f64>` field                              | `[f64; 3]`                              |
| `Cluster::new(center)`          | `center: Array1<f64>`                            | `center: [f64; 3]`                      |
| `PressureFieldResult::sphere_coords` | `Array2<f64>`                              | `Vec<[f64; 3]>`                         |
| `IncidentWave::direction`       | `Array1<f64>`                                    | `[f64; 3]`                              |
| `PointSource::position`         | `Array1<f64>`                                    | `[f64; 3]`                              |
| `BemSolution::solution`         | `Array1<Complex64>`                              | `Vec<Complex64>`                        |
| `BemSolution::converged(..)`    | takes `Array1<Complex64>`                        | takes `Vec<Complex64>`                  |
| `BemSolution::not_converged(..)`| takes `Array1<Complex64>`                        | takes `Vec<Complex64>`                  |

## Module: `core/assembly/tbem.rs` (task #8 — TBEM)

| Symbol                        | Current                            | Target                                  |
|-------------------------------|------------------------------------|-----------------------------------------|
| `TbemSystem::matrix`          | `Array2<Complex64>` field          | `DenseMatrix<Complex64>`                |
| `TbemSystem::rhs`             | `Array1<Complex64>` field          | `Vec<Complex64>`                        |
| `get_element_coords(elt, nodes)` | `&Array2<f64>` → `Array2<f64>`  | `&[[f64; 3]]` → `[[f64; 3]; 4]` or `Vec<[f64; 3]>` |

## Module: `core/solver/fmm_interface.rs` (task #8 — solver wrappers)

| Symbol                                 | Current                                                       | Target                                            |
|----------------------------------------|---------------------------------------------------------------|---------------------------------------------------|
| `DenseOperator::new(matrix)`           | `Array2<Complex64>`                                           | `DenseMatrix<Complex64>`                          |
| `SlfmmOperator::rhs() -> &Array1<..>`  | `&Array1<Complex64>`                                          | `&[Complex64]`                                    |
| `MlfmmOperator::rhs() -> &Array1<..>`  | `&Array1<Complex64>`                                          | `&[Complex64]`                                    |
| `DiagonalPreconditioner::from_diagonal`| `Array1<Complex64>`                                           | `Vec<Complex64>`                                  |
| `solve_cgs(op, b, cfg)`                | `b: &Array1<Complex64>`                                       | `b: &[Complex64]`                                 |
| `solve_bicgstab(op, b, cfg)`           | `b: &Array1<Complex64>`                                       | `b: &[Complex64]`                                 |
| `solve_gmres(op, b, cfg)`              | `b: &Array1<Complex64>`                                       | `b: &[Complex64]`                                 |
| `solve_with_ilu(matrix, b, cfg)`       | `&Array2<Complex64>, &Array1<Complex64>`                      | `&DenseMatrix<Complex64>, &[Complex64]`           |
| `solve_tbem_with_ilu(matrix, b, cfg)`  | `&Array2<Complex64>, &Array1<Complex64>`                      | `&DenseMatrix<Complex64>, &[Complex64]`           |
| `gmres_solve_with_ilu(matrix, b, cfg)` | `&Array2<Complex64>, &Array1<Complex64>`                      | `&DenseMatrix<Complex64>, &[Complex64]`           |
| `gmres_solve_with_ilu_operator(..)`    | nearfield `&Array2<Complex64>`, `&Array1<Complex64>`          | `&DenseMatrix<Complex64>, &[Complex64]`           |
| `BemIrSolution::solution`              | `Array1<Complex64>` field                                     | `Vec<Complex64>`                                  |

Once these change, the internal `ndarray_to_dense_matrix` bridge inside
fmm_interface.rs can be deleted.

## Module: `core/assembly/slfmm.rs` (task #9 — SLFMM)

| Symbol                            | Current                                  | Target                                  |
|-----------------------------------|------------------------------------------|-----------------------------------------|
| `SlfmmSystem::t_matrices`         | `Vec<Array2<Complex64>>`                 | `Vec<DenseMatrix<Complex64>>`           |
| `SlfmmSystem::t_vector`           | `Array1<Complex64>`                      | `Vec<Complex64>`                        |
| `SlfmmSystem::s_matrices`         | `Vec<Array2<Complex64>>`                 | `Vec<DenseMatrix<Complex64>>`           |
| `SlfmmSystem::rhs`                | `Array1<Complex64>`                      | `Vec<Complex64>`                        |
| `SlfmmSystem::matvec(x)`          | `&Array1<Complex64>` → `Array1<..>`       | `&[Complex64]` → `Vec<Complex64>`       |
| `SlfmmSystem::matvec_transpose(x)`| `&Array1<Complex64>` → `Array1<..>`       | `&[Complex64]` → `Vec<Complex64>`       |
| `NearFieldBlock::coefficients`    | `Array2<Complex64>`                      | `DenseMatrix<Complex64>`                |
| `DiagonalApproximation::diagonal` | `Array1<Complex64>`                      | `Vec<Complex64>`                        |
| `extract_near_field_matrix()`     | returns `Array2<Complex64>`              | returns `DenseMatrix<Complex64>`        |

## Module: `core/assembly/mlfmm.rs` (task #9 — MLFMM)

Same shape as slfmm.rs:

| Symbol                            | Current                              | Target                                  |
|-----------------------------------|--------------------------------------|-----------------------------------------|
| `MlfmmSystem::multipoles`         | `Vec<Array2<Complex64>>`             | `Vec<DenseMatrix<Complex64>>`           |
| `MlfmmSystem::locals`             | `Vec<Array2<Complex64>>`             | `Vec<DenseMatrix<Complex64>>`           |
| `MlfmmSystem::rhs`                | `Array1<Complex64>`                  | `Vec<Complex64>`                        |
| `MlfmmSystem::matvec(x)`          | `&Array1<Complex64>` → `Array1<..>`   | `&[Complex64]` → `Vec<Complex64>`       |
| `MlfmmSystem::matvec_transpose(x)`| `&Array1<Complex64>` → `Array1<..>`   | `&[Complex64]` → `Vec<Complex64>`       |
| `NearFieldBlock::coefficients`    | `Array2<Complex64>`                  | `DenseMatrix<Complex64>`                |
| `MultipoleExpansion::diagonal`    | `Array1<Complex64>`                  | `Vec<Complex64>`                        |
| `TranslationTable::matrices`      | `Vec<Array2<Complex64>>`             | `Vec<DenseMatrix<Complex64>>`           |

## Module: `core/assembly/hmatrix.rs` (task #9 — H-matrix)

| Symbol                              | Current                              | Target                                  |
|-------------------------------------|--------------------------------------|-----------------------------------------|
| `LowRankBlock::apply(x_local)`      | `&Array1<Complex64>` → `Array1<..>`   | `&[Complex64]` → `Vec<Complex64>`       |
| `LowRankBlock::apply_transpose(..)` | `&Array1<Complex64>` → `Array1<..>`   | `&[Complex64]` → `Vec<Complex64>`       |
| `DenseBlock::matrix`                | `Array2<Complex64>`                  | `DenseMatrix<Complex64>`                |
| `HierarchicalMatrix::matvec(x)`     | `&Array1<Complex64>` → `Array1<..>`   | `&[Complex64]` → `Vec<Complex64>`       |

## Modules: `core/assembly/operators.rs`, `sparse.rs`, `surface_mass.rs` (task #9)

Bulk-rewrite `Array1`/`Array2` to `Vec`/`DenseMatrix`; track per-file in
sub-commits if diffs get large.

## Module: `core/integration/{duffy,regular,singular}.rs` (task #10)

Internal quadrature integrators that consume `&Array2<f64>` element coords and
return contribution `Array1<Complex64>`. Once mesh tables are `Vec<[f64; 3]>`,
these are mechanical. Boundary signatures:

- `pub fn duffy_singular_integrate(coords: &Array2<f64>, …)` → `&[[f64; 3]]`
- `pub fn regular_integrate(coords: &Array2<f64>, …)` → `&[[f64; 3]]`

Return contributions of shape `(k × k)` should become `DenseMatrix<Complex64>`
or, for tri3/quad4, a fixed-size inline matrix.

## Module: `core/incident.rs`, `core/postprocess/pressure.rs` (task #10)

Incident-field generators (plane-wave, point-source) returning
`Array1<Complex64>` → return `Vec<Complex64>`. Pressure post-processor uses
`Array2<f64>` field points → `&[[f64; 3]]`.

## Module: `core/io/vtu.rs`, `io/native.rs`, `io/nc_format.rs` (task #11)

| Symbol                                                | Current                              | Target                                  |
|-------------------------------------------------------|--------------------------------------|-----------------------------------------|
| `write_surface_pressure_vtu(mesh, pressure, …)`       | `&Array1<Complex64>`                 | `&[Complex64]`                          |
| `NativeConfig::load_nodes(base_dir)`                  | `Result<Array2<f64>, _>`             | `Result<Vec<[f64; 3]>, _>`              |
| `load_nc_nodes(path)`                                 | `Result<Array2<f64>, _>`             | `Result<Vec<[f64; 3]>, _>`              |

## Module: `core/bem_solver.rs`, `core/solver/frequency_sweep.rs` (task #11)

| Symbol                                  | Current                              | Target                                  |
|-----------------------------------------|--------------------------------------|-----------------------------------------|
| `BemResult::surface_pressure`           | `Array1<Complex64>`                  | `Vec<Complex64>`                        |
| `BemResult::nodes`                      | `Array2<f64>`                        | `Vec<[f64; 3]>`                         |
| `BemResult::evaluate_pressure_field(p)` | `&Array2<f64>`                       | `&[[f64; 3]]`                           |
| `FrequencySweepResult::solutions`       | `Vec<Array1<Complex64>>`             | `Vec<Vec<Complex64>>`                   |

## Module: `numcalc/{project,runner}.rs`, `room_acoustics/solver.rs`, `speaker/*` (task #11)

Pure consumers of the types above. Once upstream is migrated, these are
mechanical rewrites — no on-disk format changes (mesh2hrtf NC parsing reads
the same bytes; only the Rust shape changes).

## Call-site rewrite patterns (learned from spike)

A trial run on `Element::{normal, center}` + `Cluster::center` + the 3-vec
free functions surfaced 77 cascading errors across 10 files. The cascade is
mostly mechanical but each pattern needs care. Bulk patterns to apply when
re-attempting:

```text
# 3-vec accumulator
let mut v = Array1::zeros(3);     →    let mut v = [0.0_f64; 3];

# 3-vec construction
array![a, b, c]                   →    [a, b, c]
Array1::from_vec(vec![a, b, c])   →    [a, b, c]

# Function arguments (after migrating helpers)
cross_product(&a, &b)             →    cross_product(a, b)
normalize(&v)                     →    normalize(v)

# ndarray ops that disappear with [f64; 3]
v.dot(&w)                         →    v[0]*w[0] + v[1]*w[1] + v[2]*w[2]
                                       (or: dot_product(v, w) after migration)
v.dot(&v).sqrt()                  →    (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt()
&v / scalar                       →    [v[0]/scalar, v[1]/scalar, v[2]/scalar]
v.clone()                         →    v  (arrays are Copy)

# Element field reads — already index-compatible, no change needed
elem.normal[0]                    →    elem.normal[0]    (still works)
&elem.normal                      →    &elem.normal      (now &[f64; 3])

# Element field writes that used Array1::from_vec
elem.normal = Array1::from_vec(vec![x, y, z]);   →   elem.normal = [x, y, z];
```

### Files needing the bulk rewrite (from the spike)

- `core/mesh/element.rs` — already done in the spike, kept as the reference
- `core/mesh/generators.rs` — `compute_element_geometry`: rewrite `elem.center =
  Array1::zeros(3)` and `elem.normal = Array1::from_vec(...)` patterns
- `core/mesh/octree.rs` — uses `Cluster::center` and reads `elem.normal`
- `core/integration/singular.rs` — heaviest; 19 sites with `Array1::zeros(3)`
  accumulators (`diff_poi`, `diff_fsp`, `crd_gp`, `crd_poi`, `dx_ds`, `dx_dt`,
  `result`), `unit_r.dot(&el_norm)` (5×), `&diff_poi / leneg` (2×)
- `core/integration/regular.rs` — same patterns as singular.rs
- `core/integration/duffy.rs` — `cross_product(&dx_dxi, &dx_deta)` and
  `normalize(&diff)`
- `core/assembly/{tbem,slfmm,mlfmm,hmatrix,operators}.rs` — `&elem.normal`
  reads (mostly fine; flag `.dot(&)` and `.clone()`)
- `core/bem_solver.rs` — lines 707/795/875/931/934: `elem.normal.clone()` and
  `normals[[i, j]] = elem.normal[j]` already works
- `core/postprocess/pressure.rs` — `quadrature_geometry` return-tuple reshape
- `numcalc/project.rs` — `elem.center = compute_element_center(...)` and
  `elem.normal = compute_element_normal(...)` work as-is once helpers return
  `[f64; 3]`
- `room_acoustics/solver.rs` — line 1035/1037/1061/1133: `elem.normal[i]` and
  `Cluster::new(Array1::from_vec(vec![x,y,z]))` callers

### Recommended micro-PR order for task #6 / #7

To keep each commit green:

1. **PR 6a**: convert just the 3 pure-math helpers (`cross_product`,
   `dot_product`, `normalize`) to `[f64; 3]` API + every call site. No struct
   field changes. ~17 call sites, all in 5 files (singular.rs, regular.rs,
   duffy.rs, element.rs, octree.rs). The accumulator vars (`diff_*`, `dx_*`)
   become `[f64; 3]` in the same commit.
2. **PR 6b**: convert `interpolate_normal` + `quadrature_geometry` returns
   to `[f64; 3]`. Touches pressure.rs and the integration files that consume
   the tuple.
3. **PR 6c**: convert `compute_element_center`, `compute_element_area`,
   `compute_element_normal` returns + `Element::local_to_global`,
   `Element::normal_and_jacobian`, `Element::collocation_points`,
   `Element::collocation_normals`.
4. **PR 7a**: change `Element::normal` / `Element::center` field types to
   `[f64; 3]` + fix every read site (`elem.normal.clone()` →
   `elem.normal`, `elem.normal.dot(&x)` → explicit dot).
5. **PR 7b**: change `Cluster::center` field + `Cluster::new` signature.
6. **PR 7c**: change `Element::node_normals: Array2<f64>` →
   `Vec<[f64; 3]>` and `Mesh::nodes: Array2<f64>` → `Vec<[f64; 3]>`. This is
   the largest blast radius — every quadrature kernel that does
   `nodes[[idx, j]]` needs rewriting to `nodes[idx][j]`. Plan ~1 day.

Tasks #8–#11 then become cleaner because the data shape is already settled.

## Final cleanup (task #12)

After every public API above is migrated:

1. Delete `crates/sonium-bem/src/core/array.rs` (the `pub use oxiblas_ndarray::{Array1, Array2, ArrayView2}` shim).
2. Remove `oxiblas-ndarray = { workspace = true }` from `crates/sonium-bem/Cargo.toml`.
3. `cargo tree -p sonium-bem -e normal | grep -i ndarray` should return nothing.

## Downstream cleanup (task #13)

Consumers that import sonium-bem types or use ndarray locally at the boundary:

- `sonium-fem` — heaviest downstream user; mostly mesh/quadrature reuse
- `sonium-xem` — config/source/output JSON shapes
- `sonium-sim-room` — orchestrates BEM/FEM, owns RHS construction
- `sonium-sim-speaker` — CEA-2034 output and optimization
- `sonium-sim-pcfan` — optimizer wrapper

## Back to sonium-qa (task #14)

Once the BEM crate compiles without ndarray, the QA boundary shims become
no-ops:

- `let surface_pressure = oxiblas_ndarray::Array1::from_vec(gmres_result.x);` → just `let surface_pressure = gmres_result.x;`
- `&dense_mat` constructed via `ndarray_to_dense_matrix` → already a `DenseMatrix`

Then `oxiblas-ndarray = { workspace = true }` can finally come out of
`crates/sonium-qa/Cargo.toml`.
