//! Binary STL writer for surface meshes.
//!
//! Emits the triangulated **boundary** of a 3D `Mesh` (or every triangle
//! element when the input is already a surface mesh). Useful for handing FEM
//! / BEM geometry to MeshLab, ParaView, or any STL viewer for visual sanity
//! checks.

use super::{ElementType, Mesh, Point};
use std::io::{self, Write};

/// Extract the surface triangles of `mesh` as `(p0, p1, p2)` triples in world
/// coordinates.
///
/// - For a 3D volume mesh, walks `mesh.boundaries` and triangulates each face
///   (tetra faces are already triangles; hex faces are quads split along
///   diagonal `0-2`).
/// - For a mesh whose elements are already surface triangles, emits each
///   element directly. Quadrilateral surface elements are split along
///   diagonal `0-2`.
///
/// Hex faces and quad surfaces with non-planar vertices will produce two
/// triangles that approximate the original quad. The STL format has no notion
/// of higher-order elements; P2/Q2 mid-side and bubble nodes are ignored.
pub fn surface_triangles(mesh: &Mesh) -> Vec<[Point; 3]> {
    let mut out = Vec::new();

    if !mesh.boundaries.is_empty() {
        for face in &mesh.boundaries {
            match face.nodes.len() {
                3 => out.push([
                    mesh.nodes[face.nodes[0]],
                    mesh.nodes[face.nodes[1]],
                    mesh.nodes[face.nodes[2]],
                ]),
                4 => {
                    // Triangulate quad on diagonal 0-2.
                    let p0 = mesh.nodes[face.nodes[0]];
                    let p1 = mesh.nodes[face.nodes[1]];
                    let p2 = mesh.nodes[face.nodes[2]];
                    let p3 = mesh.nodes[face.nodes[3]];
                    out.push([p0, p1, p2]);
                    out.push([p0, p2, p3]);
                }
                _ => {}
            }
        }
        return out;
    }

    // No detected boundary — fall back to surface elements (Triangle / Quad).
    for elem in &mesh.elements {
        match elem.element_type {
            ElementType::Triangle => {
                let v = elem.vertices();
                out.push([mesh.nodes[v[0]], mesh.nodes[v[1]], mesh.nodes[v[2]]]);
            }
            ElementType::Quadrilateral => {
                let v = elem.vertices();
                let p0 = mesh.nodes[v[0]];
                let p1 = mesh.nodes[v[1]];
                let p2 = mesh.nodes[v[2]];
                let p3 = mesh.nodes[v[3]];
                out.push([p0, p1, p2]);
                out.push([p0, p2, p3]);
            }
            _ => {}
        }
    }
    out
}

/// Write a binary STL representation of the mesh surface to `writer`.
///
/// Format (little-endian throughout):
/// - 80 bytes header (filled with zeros)
/// - u32 triangle count
/// - per triangle: 3 × f32 normal, 9 × f32 vertices, u16 attribute (0)
pub fn write_binary_stl<W: Write>(mesh: &Mesh, mut writer: W) -> io::Result<()> {
    write_binary_stl_triangles(&surface_triangles(mesh), &mut writer)
}

/// Write a binary STL file at `path` containing the mesh surface.
pub fn write_binary_stl_file(mesh: &Mesh, path: impl AsRef<std::path::Path>) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut buf = std::io::BufWriter::new(file);
    write_binary_stl(mesh, &mut buf)?;
    buf.flush()
}

fn write_binary_stl_triangles<W: Write>(tris: &[[Point; 3]], writer: &mut W) -> io::Result<()> {
    writer.write_all(&[0u8; 80])?;
    writer.write_all(&(tris.len() as u32).to_le_bytes())?;
    for tri in tris {
        let normal = facet_normal(tri);
        for c in normal {
            writer.write_all(&(c as f32).to_le_bytes())?;
        }
        for v in tri {
            writer.write_all(&(v.x as f32).to_le_bytes())?;
            writer.write_all(&(v.y as f32).to_le_bytes())?;
            writer.write_all(&(v.z as f32).to_le_bytes())?;
        }
        writer.write_all(&0u16.to_le_bytes())?;
    }
    Ok(())
}

fn facet_normal(tri: &[Point; 3]) -> [f64; 3] {
    let a = (
        tri[1].x - tri[0].x,
        tri[1].y - tri[0].y,
        tri[1].z - tri[0].z,
    );
    let b = (
        tri[2].x - tri[0].x,
        tri[2].y - tri[0].y,
        tri[2].z - tri[0].z,
    );
    let n = (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    );
    let len = (n.0 * n.0 + n.1 * n.1 + n.2 * n.2).sqrt();
    if len > 0.0 {
        [n.0 / len, n.1 / len, n.2 / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generators::unit_cube_hexahedra;

    #[test]
    fn binary_stl_header_and_count_match() {
        let mesh = unit_cube_hexahedra(1);
        // 1 hex has 6 boundary quads → 12 triangles.
        let tris = surface_triangles(&mesh);
        assert_eq!(tris.len(), 12);

        let mut buf = Vec::new();
        write_binary_stl(&mesh, &mut buf).unwrap();
        assert!(buf.len() >= 80 + 4);
        let count = u32::from_le_bytes([buf[80], buf[81], buf[82], buf[83]]);
        assert_eq!(count, 12);
        // 80 + 4 + 12 × (4 × 3 normal + 4 × 9 verts + 2 attr) = 84 + 12·50
        assert_eq!(buf.len(), 84 + 12 * 50);
    }
}
