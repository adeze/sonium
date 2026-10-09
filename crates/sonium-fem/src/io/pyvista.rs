//! PyVista-compatible output via NumPy `.npy` files
//!
//! Writes mesh geometry and solution fields as `.npy` files that can be
//! loaded directly in Python with `numpy.load()` and assembled into a
//! PyVista `UnstructuredGrid`.
//!
//! Also generates a `load.py` helper script for one-line loading.
//!
//! # Usage (Rust)
//!
//! ```ignore
//! use sonium_fem::io::pyvista::PyVistaWriter;
//!
//! let mut writer = PyVistaWriter::new(&mesh);
//! writer.add_point_scalar("pressure", &pressure);
//! writer.add_point_vector("velocity", &ux, &uy, &uz);
//! writer.write("my_result")?;
//! // Creates: my_result_points.npy, my_result_cells.npy,
//! //          my_result_celltypes.npy, my_result_pressure.npy,
//! //          my_result_velocity.npy, my_result_load.py
//! ```
//!
//! # Usage (Python)
//!
//! ```python
//! # Option 1: run the generated script
//! exec(open("my_result_load.py").read())
//! mesh.plot()
//!
//! # Option 2: manual loading
//! import numpy as np, pyvista as pv
//! points = np.load("my_result_points.npy")
//! cells = np.load("my_result_cells.npy")
//! celltypes = np.load("my_result_celltypes.npy")
//! mesh = pv.UnstructuredGrid(cells, celltypes, points)
//! mesh.point_data["pressure"] = np.load("my_result_pressure.npy")
//! ```

use crate::mesh::{ElementType, Mesh};
use num_complex::Complex64;
use std::io::Write;
use std::path::Path;

fn vtk_cell_type_id(elem_type: ElementType) -> u8 {
    match elem_type {
        ElementType::Triangle => 5,
        ElementType::Quadrilateral => 9,
        ElementType::Tetrahedron => 10,
        ElementType::Hexahedron => 12,
    }
}

enum PointField {
    Scalar { name: String, values: Vec<f64> },
    Vector { name: String, values: Vec<[f64; 3]> },
}

enum CellField {
    Scalar { name: String, values: Vec<f64> },
}

/// Writer that produces NumPy `.npy` files loadable by PyVista.
pub struct PyVistaWriter<'m> {
    mesh: &'m Mesh,
    point_fields: Vec<PointField>,
    cell_fields: Vec<CellField>,
}

impl<'m> PyVistaWriter<'m> {
    pub fn new(mesh: &'m Mesh) -> Self {
        Self {
            mesh,
            point_fields: Vec::new(),
            cell_fields: Vec::new(),
        }
    }

    pub fn add_point_scalar(&mut self, name: &str, values: &[f64]) {
        assert_eq!(values.len(), self.mesh.num_nodes());
        self.point_fields.push(PointField::Scalar {
            name: name.to_string(),
            values: values.to_vec(),
        });
    }

    pub fn add_point_scalar_complex(&mut self, name: &str, values: &[Complex64]) {
        let n = self.mesh.num_nodes();
        assert_eq!(values.len(), n);
        self.add_point_scalar(
            &format!("{name}_magnitude"),
            &values.iter().map(|v| v.norm()).collect::<Vec<_>>(),
        );
        self.add_point_scalar(
            &format!("{name}_real"),
            &values.iter().map(|v| v.re).collect::<Vec<_>>(),
        );
        self.add_point_scalar(
            &format!("{name}_imag"),
            &values.iter().map(|v| v.im).collect::<Vec<_>>(),
        );
    }

    pub fn add_point_vector(&mut self, name: &str, vx: &[f64], vy: &[f64], vz: &[f64]) {
        let n = self.mesh.num_nodes();
        assert_eq!(vx.len(), n);
        assert_eq!(vy.len(), n);
        assert_eq!(vz.len(), n);
        let values: Vec<[f64; 3]> = (0..n).map(|i| [vx[i], vy[i], vz[i]]).collect();
        self.point_fields.push(PointField::Vector {
            name: name.to_string(),
            values,
        });
    }

    pub fn add_cell_scalar(&mut self, name: &str, values: &[f64]) {
        assert_eq!(values.len(), self.mesh.num_elements());
        self.cell_fields.push(CellField::Scalar {
            name: name.to_string(),
            values: values.to_vec(),
        });
    }

    /// Write all `.npy` files and a `load.py` helper script.
    ///
    /// `prefix` is the base name: `"results"` produces `results_points.npy`, etc.
    pub fn write(&self, prefix: &str) -> std::io::Result<()> {
        let dir = Path::new(prefix).parent().unwrap_or(Path::new("."));
        let stem = Path::new(prefix).file_name().unwrap().to_str().unwrap();

        // Points: (n_nodes, 3) float64
        let n = self.mesh.num_nodes();
        let mut pts = Vec::with_capacity(n * 3);
        for node in &self.mesh.nodes {
            pts.push(node.x);
            pts.push(node.y);
            pts.push(node.z);
        }
        write_npy_f64(&dir.join(format!("{stem}_points.npy")), &[n, 3], &pts)?;

        // Cells: VTK-style flat array [npts, p0, p1, .., npts, p0, ...]
        let mut cells: Vec<i64> = Vec::new();
        let mut celltypes: Vec<u8> = Vec::new();
        for elem in &self.mesh.elements {
            let verts = elem.vertices();
            cells.push(verts.len() as i64);
            for &v in verts {
                cells.push(v as i64);
            }
            celltypes.push(vtk_cell_type_id(elem.element_type));
        }
        write_npy_i64(
            &dir.join(format!("{stem}_cells.npy")),
            &[cells.len()],
            &cells,
        )?;
        write_npy_u8(
            &dir.join(format!("{stem}_celltypes.npy")),
            &[celltypes.len()],
            &celltypes,
        )?;

        // Point fields
        let mut field_names: Vec<(String, String, bool)> = Vec::new(); // (name, filename, is_vector)
        for field in &self.point_fields {
            match field {
                PointField::Scalar { name, values } => {
                    let fname = format!("{stem}_{name}.npy");
                    write_npy_f64(&dir.join(&fname), &[values.len()], values)?;
                    field_names.push((name.clone(), fname, false));
                }
                PointField::Vector { name, values } => {
                    let fname = format!("{stem}_{name}.npy");
                    let flat: Vec<f64> = values.iter().flat_map(|v| *v).collect();
                    write_npy_f64(&dir.join(&fname), &[values.len(), 3], &flat)?;
                    field_names.push((name.clone(), fname, true));
                }
            }
        }

        // Cell fields
        let mut cell_field_names: Vec<(String, String)> = Vec::new();
        for field in &self.cell_fields {
            match field {
                CellField::Scalar { name, values } => {
                    let fname = format!("{stem}_{name}.npy");
                    write_npy_f64(&dir.join(&fname), &[values.len()], values)?;
                    cell_field_names.push((name.clone(), fname));
                }
            }
        }

        // Generate load.py
        let py_path = dir.join(format!("{stem}_load.py"));
        let mut py = std::fs::File::create(py_path)?;
        writeln!(py, "\"\"\"Auto-generated PyVista loader for {stem}\"\"\"")?;
        writeln!(py, "import numpy as np")?;
        writeln!(py, "import pyvista as pv")?;
        writeln!(py)?;
        writeln!(py, "points = np.load(\"{stem}_points.npy\")")?;
        writeln!(py, "cells = np.load(\"{stem}_cells.npy\")")?;
        writeln!(py, "celltypes = np.load(\"{stem}_celltypes.npy\")")?;
        writeln!(py, "mesh = pv.UnstructuredGrid(cells, celltypes, points)")?;

        for (name, fname, _is_vector) in &field_names {
            writeln!(py, "mesh.point_data[\"{name}\"] = np.load(\"{fname}\")")?;
        }
        for (name, fname) in &cell_field_names {
            writeln!(py, "mesh.cell_data[\"{name}\"] = np.load(\"{fname}\")")?;
        }

        writeln!(py)?;
        writeln!(py, "if __name__ == \"__main__\":")?;
        if let Some((first_name, _, _)) = field_names.first() {
            writeln!(
                py,
                "    mesh.plot(scalars=\"{first_name}\", show_edges=True)"
            )?;
        } else {
            writeln!(py, "    mesh.plot(show_edges=True)")?;
        }

        Ok(())
    }
}

// ── NumPy .npy format writers ──────────────────────────────────────────

fn write_npy_header(w: &mut impl Write, descr: &str, shape: &[usize]) -> std::io::Result<()> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let inner: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        format!("({})", inner.join(", "))
    };
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );

    // Header must be padded to a multiple of 64 bytes (magic + version + header_len + dict)
    // magic(6) + version(2) + header_len(2) = 10 bytes overhead
    let overhead = 10;
    let pad_len = 64 - ((overhead + dict.len() + 1) % 64); // +1 for trailing \n
    let total_header_len = dict.len() + pad_len + 1; // padded dict + \n

    w.write_all(b"\x93NUMPY")?; // magic
    w.write_all(&[1, 0])?; // version 1.0
    w.write_all(&(total_header_len as u16).to_le_bytes())?;
    w.write_all(dict.as_bytes())?;
    for _ in 0..pad_len {
        w.write_all(b" ")?;
    }
    w.write_all(b"\n")?;
    Ok(())
}

fn write_npy_f64(path: &Path, shape: &[usize], data: &[f64]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    write_npy_header(&mut f, "<f8", shape)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    f.write_all(bytes)
}

fn write_npy_i64(path: &Path, shape: &[usize], data: &[i64]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    write_npy_header(&mut f, "<i8", shape)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    f.write_all(bytes)
}

fn write_npy_u8(path: &Path, shape: &[usize], data: &[u8]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    write_npy_header(&mut f, "|u1", shape)?;
    f.write_all(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_pyvista_writer() {
        let mesh = unit_square_triangles(2);
        let n = mesh.num_nodes();
        let ne = mesh.num_elements();

        let mut writer = PyVistaWriter::new(&mesh);
        writer.add_point_scalar("temperature", &vec![1.0; n]);
        writer.add_cell_scalar("material", &vec![0.0; ne]);

        let dir = std::env::temp_dir().join("sonium_pyvista_test");
        let _ = std::fs::create_dir_all(&dir);
        let prefix = dir.join("test_mesh");
        writer.write(prefix.to_str().unwrap()).unwrap();

        // Verify files exist
        assert!(dir.join("test_mesh_points.npy").exists());
        assert!(dir.join("test_mesh_cells.npy").exists());
        assert!(dir.join("test_mesh_celltypes.npy").exists());
        assert!(dir.join("test_mesh_temperature.npy").exists());
        assert!(dir.join("test_mesh_material.npy").exists());
        assert!(dir.join("test_mesh_load.py").exists());

        // Verify npy header
        let pts_data = std::fs::read(dir.join("test_mesh_points.npy")).unwrap();
        assert_eq!(&pts_data[..6], b"\x93NUMPY");
        assert_eq!(pts_data[6], 1); // version 1.0

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}
