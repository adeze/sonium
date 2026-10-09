//! VTK Unstructured Grid (.vtu) writer — re-exported from `sonium_xem::io`.
//!
//! The writer itself lives in `sonium-xem` (the crate shared between BEM and
//! FEM) so all sonium binaries write the same format. This module is a thin
//! re-export kept for backwards compatibility; new code should import from
//! `sonium_xem::io::{vtu, pvd}` directly.
//!
//! ```ignore
//! use sonium_fem::io::vtk::VtuWriter;
//!
//! let mut w = VtuWriter::new(&mesh);
//! w.add_point_scalar("pressure", &pressure);
//! w.write("output.vtu")?;
//! ```

pub use sonium_xem::io::pvd::{PvdCollection, write_pvd};
pub use sonium_xem::io::vtu::{CellKind, MeshView, VtuWriter};
