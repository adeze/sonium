//! ASCII PLY (Polygon File Format) writer for surface meshes.
//!
//! Exports the boundary of a 3D mesh, or the surface elements directly when
//! the mesh is already a surface. Same triangulation rules as `stl.rs`: hex
//! faces and quad surface elements are split along diagonal `0-2`.

use super::{ElementType, Mesh};
use std::io::{self, Write};

/// Write the mesh surface as ASCII PLY to `writer`.
///
/// Only triangle elements appear in the body of the file. Quadrilateral
/// boundary faces (e.g. hex boundary quads) are split into two triangles on
/// the `0-2` diagonal. The file ends with a trailing newline so editors and
/// shells that line-buffer pick up the last record cleanly.
pub fn write_ply<W: Write>(mesh: &Mesh, mut writer: W) -> io::Result<()> {
    let nodes: Vec<[f64; 3]> = mesh.nodes.iter().map(|p| [p.x, p.y, p.z]).collect();
    let mut tris: Vec<[usize; 3]> = Vec::new();

    if !mesh.boundaries.is_empty() {
        for face in &mesh.boundaries {
            match face.nodes.len() {
                3 => tris.push([face.nodes[0], face.nodes[1], face.nodes[2]]),
                4 => {
                    tris.push([face.nodes[0], face.nodes[1], face.nodes[2]]);
                    tris.push([face.nodes[0], face.nodes[2], face.nodes[3]]);
                }
                _ => {}
            }
        }
    } else {
        for elem in &mesh.elements {
            let v = elem.vertices();
            match elem.element_type {
                ElementType::Triangle => tris.push([v[0], v[1], v[2]]),
                ElementType::Quadrilateral => {
                    tris.push([v[0], v[1], v[2]]);
                    tris.push([v[0], v[2], v[3]]);
                }
                _ => {}
            }
        }
    }

    writeln!(writer, "ply")?;
    writeln!(writer, "format ascii 1.0")?;
    writeln!(writer, "element vertex {}", nodes.len())?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;
    writeln!(writer, "element face {}", tris.len())?;
    writeln!(writer, "property list uchar int vertex_indices")?;
    writeln!(writer, "end_header")?;
    for n in nodes {
        writeln!(writer, "{} {} {}", n[0], n[1], n[2])?;
    }
    for t in tris {
        writeln!(writer, "3 {} {} {}", t[0], t[1], t[2])?;
    }
    Ok(())
}

/// Write the mesh surface to `path` as ASCII PLY.
pub fn write_ply_file(mesh: &Mesh, path: impl AsRef<std::path::Path>) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut buf = std::io::BufWriter::new(file);
    write_ply(mesh, &mut buf)?;
    buf.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generators::unit_cube_hexahedra;

    #[test]
    fn ply_emits_header_and_correct_face_count() {
        let mesh = unit_cube_hexahedra(1);
        let mut buf = Vec::new();
        write_ply(&mesh, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("ply\n"));
        assert!(s.contains("element vertex 8\n"));
        assert!(s.contains("element face 12\n"));
        // Every face line is "3 i j k".
        let face_lines: Vec<&str> = s.lines().filter(|l| l.starts_with("3 ")).collect();
        assert_eq!(face_lines.len(), 12);
    }
}
