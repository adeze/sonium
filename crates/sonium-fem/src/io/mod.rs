//! Input/Output for FEM meshes and solution data
//!
//! - [`vtk`]: VTK Unstructured Grid (.vtu) writer for ParaView visualization
//! - [`pyvista`]: NumPy `.npy` files for direct PyVista loading

pub mod pyvista;
pub mod vtk;
