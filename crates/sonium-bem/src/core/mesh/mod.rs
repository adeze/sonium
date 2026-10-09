//! Mesh structures and element operations

pub mod barycentric;
pub mod cluster;
pub mod element;
pub mod generators;
pub mod lineage;
pub mod octree;
pub mod quadratic_dof;

pub use barycentric::{BarycentricRefinement, refine_from_elements, refine_surface_mesh};
pub use cluster::*;
pub use element::*;
pub use generators::*;
pub use lineage::{
    BemCacheInvalidation, BemCacheReuse, BemReuseContext, BemReuseDecision, BemReuseError,
    BemReusePolicy, BemReuseRejection, MeshLineage, MeshLineageError, MeshUpdatePlan,
    assess_bem_cache_reuse, classify_mesh_update, relative_frequency_delta,
};
pub use octree::{AABB, Octree, OctreeNode, OctreeStats};
pub use quadratic_dof::{ContinuousQuadraticDofMap, ContinuousQuadraticDofMapError};
