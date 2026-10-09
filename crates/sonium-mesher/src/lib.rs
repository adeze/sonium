//! Mesh generation and partitioning library
//!
//! Provides 3D mesh generation (tetrahedralization), partitioning for distributed-memory FEM,
//! and mesh I/O support.
//!
//! # Modules
//! - `mesh`: Core mesh types and generators
//! - `partition`: Graph-based mesh partitioning for domain decomposition

pub mod convexhull;
pub mod mesh;
pub mod partition;
pub mod room_surface;

pub use room_surface::RoomSurfaceMeshing;
