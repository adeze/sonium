//! Gmsh `.msh` writer (ASCII v4.1).
//!
//! Round-trips P1/Q1 and P2/Q2 sonium meshes back to Gmsh-readable files
//! through the same `mshio`-based reader path that lives in
//! [`super::gmsh`]. The writer emits the v4.1 ASCII flavour (the only version
//! `mshio` accepts) and keeps the `$Entities` block as small as is legal:
//! one geometric entity per distinct cell or boundary marker.
//!
//! ## Higher-order nodes
//!
//! Sonium's P2/Q2 node orderings differ from Gmsh's. When the source mesh is
//! P2/Q2 the writer applies the inverse of the [`p2`](super::p2) permutation
//! tables so the resulting file reads back cleanly through
//! [`super::gmsh::load_gmsh`].
//!
//! ## Markers
//!
//! Cell and boundary markers are written as physical tags on separate entities.
//! Registered names are emitted in `$PhysicalNames`, so numeric markers and
//! dimension-aware names both round-trip through the companion reader.
//!
//! Feature-gated under `mshio` to keep the two Gmsh modules in lockstep.

use super::{Element, ElementType, Mesh};
use std::collections::BTreeMap;
use std::io::{self, Write};

const GMSH_LINE2: u32 = 1;
const GMSH_LINE3: u32 = 8;
const GMSH_TRI3: u32 = 2;
const GMSH_QUA4: u32 = 3;
const GMSH_TET4: u32 = 4;
const GMSH_HEX8: u32 = 5;
const GMSH_TRI6: u32 = 9;
const GMSH_QUA9: u32 = 10;
const GMSH_TET10: u32 = 11;
const GMSH_HEX27: u32 = 12;

fn gmsh_volume_type(element_type: ElementType, p2: bool) -> u32 {
    match (element_type, p2) {
        (ElementType::Triangle, false) => GMSH_TRI3,
        (ElementType::Triangle, true) => GMSH_TRI6,
        (ElementType::Quadrilateral, false) => GMSH_QUA4,
        (ElementType::Quadrilateral, true) => GMSH_QUA9,
        (ElementType::Tetrahedron, false) => GMSH_TET4,
        (ElementType::Tetrahedron, true) => GMSH_TET10,
        (ElementType::Hexahedron, false) => GMSH_HEX8,
        (ElementType::Hexahedron, true) => GMSH_HEX27,
    }
}

/// Build the inverse of a sonium "GMSH_TO_FEM" permutation.
///
/// The tables exported from [`super::p2`] are documented as "GMSH_TO_FEM"
/// but, when read against [`super::p2::reorder_gmsh_to_fem_p2`], the actual
/// semantic is `perm[fem_index] = gmsh_index` — i.e. they read out the
/// Gmsh slot a given FEM slot ends up occupying after reordering. This
/// function returns the inverse: `inv[gmsh_index] = fem_index`, which is
/// what a writer needs to translate FEM-ordered node arrays into Gmsh order.
fn gmsh_to_fem_inverse(perm_fem_to_gmsh: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm_fem_to_gmsh.len()];
    for (fem, &gmsh) in perm_fem_to_gmsh.iter().enumerate() {
        inv[gmsh] = fem;
    }
    inv
}

fn element_nodes_in_gmsh_order(elem: &Element) -> Vec<usize> {
    use super::p2::{HEX27_GMSH_TO_FEM, QUA9_GMSH_TO_FEM, TET10_GMSH_TO_FEM, TRI6_GMSH_TO_FEM};
    if elem.nodes.len() == elem.element_type.num_nodes_p1() {
        return elem.nodes.clone();
    }
    let inv: Vec<usize> = match elem.element_type {
        ElementType::Triangle => gmsh_to_fem_inverse(&TRI6_GMSH_TO_FEM),
        ElementType::Quadrilateral => gmsh_to_fem_inverse(&QUA9_GMSH_TO_FEM),
        ElementType::Tetrahedron => gmsh_to_fem_inverse(&TET10_GMSH_TO_FEM),
        ElementType::Hexahedron => gmsh_to_fem_inverse(&HEX27_GMSH_TO_FEM),
    };
    // `inv[gmsh_pos] = fem_pos`; iterating yields fem positions in gmsh order,
    // which is exactly the output sequence we want.
    inv.iter().map(|&fem_pos| elem.nodes[fem_pos]).collect()
}

fn boundary_gmsh_type(node_count: usize, mesh_dimension: usize) -> Option<u32> {
    match (node_count, mesh_dimension) {
        (2, _) => Some(GMSH_LINE2),
        (3, 2) => Some(GMSH_LINE3),
        (3, 3) => Some(GMSH_TRI3),
        (4, 3) => Some(GMSH_QUA4),
        (6, _) => Some(GMSH_TRI6),
        _ => None,
    }
}

/// Write `mesh` to `writer` as a Gmsh ASCII v4.1 `.msh` file.
pub fn write_gmsh<W: Write>(mesh: &Mesh, mut writer: W) -> io::Result<()> {
    mesh.validate_physical_groups()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let dim = mesh.dimension;
    let bounding_box = compute_bounding_box(mesh);
    let boundary_entity_dim = if dim == 3 { 2 } else { 1 };

    // Collect distinct boundary markers (sorted for deterministic output).
    let mut markers: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (i, b) in mesh.boundaries.iter().enumerate() {
        if boundary_gmsh_type(b.nodes.len(), dim).is_some() {
            markers.entry(b.marker).or_default().push(i);
        }
    }
    let marker_entity_tags: BTreeMap<i32, u32> = markers
        .keys()
        .enumerate()
        .map(|(i, &m)| (m, i as u32 + 1))
        .collect();

    // Group cells by marker. Even an empty mesh gets a synthetic unmarked
    // entity so the node block retains a legal owner.
    let mut cell_markers: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (index, element) in mesh.elements.iter().enumerate() {
        cell_markers.entry(element.marker).or_default().push(index);
    }
    if cell_markers.is_empty() {
        cell_markers.insert(0, Vec::new());
    }
    let cell_entity_tags: BTreeMap<i32, u32> = cell_markers
        .keys()
        .enumerate()
        .map(|(index, &marker)| (marker, index as u32 + 1))
        .collect();
    let node_entity_tag = *cell_entity_tags
        .values()
        .next()
        .expect("synthetic cell entity exists");

    writeln!(writer, "$MeshFormat")?;
    writeln!(writer, "4.1 0 8")?;
    writeln!(writer, "$EndMeshFormat")?;

    write_physical_names(mesh, &mut writer)?;

    // $Entities: one entity per boundary marker and cell marker.
    writeln!(writer, "$Entities")?;
    let counts = entity_counts(
        dim,
        marker_entity_tags.len() as u32,
        cell_entity_tags.len() as u32,
    );
    writeln!(
        writer,
        "{} {} {} {}",
        counts[0], counts[1], counts[2], counts[3]
    )?;
    write_marker_entities(&mut writer, &marker_entity_tags, &bounding_box, dim)?;
    write_cell_entities(&mut writer, &cell_entity_tags, &bounding_box)?;
    writeln!(writer, "$EndEntities")?;

    // $Nodes: single block, all nodes.
    writeln!(writer, "$Nodes")?;
    writeln!(
        writer,
        "1 {} 1 {}",
        mesh.nodes.len(),
        mesh.nodes.len().max(1)
    )?;
    writeln!(writer, "{} {} 0 {}", dim, node_entity_tag, mesh.nodes.len())?;
    for i in 0..mesh.nodes.len() {
        writeln!(writer, "{}", i + 1)?;
    }
    for p in &mesh.nodes {
        writeln!(writer, "{} {} {}", p.x, p.y, p.z)?;
    }
    writeln!(writer, "$EndNodes")?;

    // Split cell blocks by marker × element shape so every block has one
    // owning entity and one Gmsh element type.
    let mut cell_subblocks: Vec<(u32, u32, Vec<usize>)> = Vec::new();
    for (&marker, indices) in &cell_markers {
        let entity_tag = cell_entity_tags[&marker];
        let mut sub_by_type: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for &index in indices {
            let element = &mesh.elements[index];
            let p2 = element.nodes.len() == element.element_type.num_nodes_p2();
            sub_by_type
                .entry(gmsh_volume_type(element.element_type, p2))
                .or_default()
                .push(index);
        }
        for (gmsh_type, element_indices) in sub_by_type {
            cell_subblocks.push((entity_tag, gmsh_type, element_indices));
        }
    }
    // Pre-split boundary blocks by marker × face shape so the header count is
    // correct.
    let mut marker_subblocks: Vec<(u32, u32, u32, Vec<usize>)> = Vec::new(); // (entity_tag, entity_dim, gmsh_ty, indices)
    for (&marker, indices) in &markers {
        let entity_tag = marker_entity_tags[&marker];
        let mut sub_by_type: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for &bi in indices {
            let nodes = &mesh.boundaries[bi].nodes;
            let gmsh_ty = boundary_gmsh_type(nodes.len(), dim).expect("filtered above");
            sub_by_type.entry(gmsh_ty).or_default().push(bi);
        }
        for (gmsh_ty, bidx) in sub_by_type {
            marker_subblocks.push((entity_tag, boundary_entity_dim as u32, gmsh_ty, bidx));
        }
    }

    let total_elements =
        mesh.elements.len() + marker_subblocks.iter().map(|b| b.3.len()).sum::<usize>();
    let num_blocks = cell_subblocks.len() + marker_subblocks.len();

    writeln!(writer, "$Elements")?;
    writeln!(
        writer,
        "{} {} 1 {}",
        num_blocks,
        total_elements,
        total_elements.max(1)
    )?;

    let mut tag = 1usize;
    for (entity_tag, gmsh_ty, elem_indices) in &cell_subblocks {
        writeln!(
            writer,
            "{} {} {} {}",
            dim,
            entity_tag,
            gmsh_ty,
            elem_indices.len()
        )?;
        for &i in elem_indices {
            let elem = &mesh.elements[i];
            let ordered = element_nodes_in_gmsh_order(elem);
            write!(writer, "{tag}")?;
            for n in ordered {
                write!(writer, " {}", n + 1)?;
            }
            writeln!(writer)?;
            tag += 1;
        }
    }
    for (entity_tag, entity_dim, gmsh_ty, bidx) in &marker_subblocks {
        writeln!(
            writer,
            "{} {} {} {}",
            entity_dim,
            entity_tag,
            gmsh_ty,
            bidx.len()
        )?;
        for &bi in bidx {
            let nodes = &mesh.boundaries[bi].nodes;
            write!(writer, "{tag}")?;
            for &n in nodes {
                write!(writer, " {}", n + 1)?;
            }
            writeln!(writer)?;
            tag += 1;
        }
    }
    writeln!(writer, "$EndElements")?;
    Ok(())
}

fn entity_counts(dim: usize, num_marker_entities: u32, num_cell_entities: u32) -> [u32; 4] {
    // [points, curves, surfaces, volumes]
    match dim {
        2 => [0, num_marker_entities, num_cell_entities, 0],
        3 => [0, 0, num_marker_entities, num_cell_entities],
        _ => unreachable!("Mesh::new asserts dimension ∈ {{2,3}}"),
    }
}

fn write_marker_entities<W: Write>(
    writer: &mut W,
    markers: &BTreeMap<i32, u32>,
    bbox: &[f64; 6],
    dim: usize,
) -> io::Result<()> {
    for (&marker, &tag) in markers {
        if dim == 3 {
            // Surface entity: tag, min/max bbox, numPhysicalTags physTag..., numBoundingCurves curveTag...
            writeln!(
                writer,
                "{} {} {} {} {} {} {} 1 {} 0",
                tag, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], marker
            )?;
        } else {
            // Curve entity: tag, min/max bbox, numPhysicalTags physTag..., numBoundingPoints pointTag...
            writeln!(
                writer,
                "{} {} {} {} {} {} {} 1 {} 0",
                tag, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], marker
            )?;
        }
    }
    Ok(())
}

fn write_cell_entities<W: Write>(
    writer: &mut W,
    markers: &BTreeMap<i32, u32>,
    bbox: &[f64; 6],
) -> io::Result<()> {
    for (&marker, &tag) in markers {
        writeln!(
            writer,
            "{} {} {} {} {} {} {} 1 {} 0",
            tag, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], marker
        )?;
    }
    Ok(())
}

fn write_physical_names<W: Write>(mesh: &Mesh, writer: &mut W) -> io::Result<()> {
    if mesh.physical_groups.is_empty() {
        return Ok(());
    }
    writeln!(writer, "$PhysicalNames")?;
    writeln!(writer, "{}", mesh.physical_groups.len())?;
    for group in mesh.physical_groups.iter() {
        let escaped = group.name.replace('\\', "\\\\").replace('"', "\\\"");
        writeln!(writer, "{} {} \"{}\"", group.dimension, group.tag, escaped)?;
    }
    writeln!(writer, "$EndPhysicalNames")
}

fn compute_bounding_box(mesh: &Mesh) -> [f64; 6] {
    if mesh.nodes.is_empty() {
        return [0.0; 6];
    }
    let mut bbox = [
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in &mesh.nodes {
        bbox[0] = bbox[0].min(p.x);
        bbox[1] = bbox[1].min(p.y);
        bbox[2] = bbox[2].min(p.z);
        bbox[3] = bbox[3].max(p.x);
        bbox[4] = bbox[4].max(p.y);
        bbox[5] = bbox[5].max(p.z);
    }
    bbox
}

/// Write `mesh` to `path` as a Gmsh ASCII v4.1 `.msh` file.
pub fn write_gmsh_file(mesh: &Mesh, path: impl AsRef<std::path::Path>) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut buf = std::io::BufWriter::new(file);
    write_gmsh(mesh, &mut buf)?;
    buf.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generators::unit_cube_hexahedra;
    use crate::mesh::gmsh::from_msh_bytes;

    #[test]
    fn round_trip_hex_cube_through_msh() {
        let mesh = unit_cube_hexahedra(1);
        let mut buf = Vec::new();
        write_gmsh(&mesh, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("$MeshFormat"));
        let parsed = from_msh_bytes(s.as_bytes()).expect("re-parsed");
        assert_eq!(parsed.num_nodes(), mesh.num_nodes());
        assert_eq!(
            parsed
                .elements
                .iter()
                .filter(|e| e.element_type == ElementType::Hexahedron)
                .count(),
            1
        );
    }

    #[test]
    fn round_trip_p2_tet_preserves_node_identities() {
        // Build a P1 tet, promote to P2, write to MSH, read back, and verify
        // that every FEM-ordered node lands at the same coordinates. This
        // exercises the `gmsh_to_fem_inverse` path that the P1 round-trip
        // skips.
        use crate::mesh::Point;
        let mut mesh = Mesh::new(3);
        mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
        mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
        mesh.add_node(Point::new_3d(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        let p2 = mesh.to_p2();

        let mut buf = Vec::new();
        write_gmsh(&p2, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let parsed = from_msh_bytes(s.as_bytes()).expect("re-parsed");

        assert_eq!(parsed.num_nodes(), p2.num_nodes());
        assert_eq!(parsed.num_elements(), p2.num_elements());

        let elem = &parsed.elements[0];
        assert_eq!(elem.element_type, ElementType::Tetrahedron);
        assert_eq!(elem.nodes.len(), 10);

        // FEM order: nodes 4..9 are the mid-edges in sonium's edge order
        // (0,1),(0,2),(0,3),(1,2),(1,3),(2,3). The round-trip should keep
        // each mid-edge node at the midpoint of its parent edge.
        let expected_edges: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for (k, &(a, b)) in expected_edges.iter().enumerate() {
            let mid = parsed.nodes[elem.nodes[4 + k]];
            let exp = parsed.nodes[elem.nodes[a]].midpoint(&parsed.nodes[elem.nodes[b]]);
            assert!(
                (mid.x - exp.x).abs() < 1e-12
                    && (mid.y - exp.y).abs() < 1e-12
                    && (mid.z - exp.z).abs() < 1e-12,
                "P2 mid-edge node mismatch for edge {a}-{b}: got {mid:?}, expected {exp:?}"
            );
        }
    }

    #[test]
    fn round_trip_preserves_cell_markers_and_named_physical_groups() {
        use crate::mesh::generators::unit_square_triangles;

        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].marker = 10;
        mesh.elements[1].marker = 20;
        mesh.boundaries[0].marker = 5;
        mesh.register_physical_group(2, 10, "front air").unwrap();
        mesh.register_physical_group(2, 20, "rear \\\"cavity\\\"")
            .unwrap();
        mesh.register_physical_group(1, 5, "driver interface")
            .unwrap();

        let mut buffer = Vec::new();
        write_gmsh(&mesh, &mut buffer).unwrap();
        let text = String::from_utf8(buffer.clone()).unwrap();
        assert!(text.contains("$PhysicalNames"));
        assert!(text.contains("2 10 \"front air\""));
        assert!(text.contains("2 20 \"rear \\\\\\\"cavity\\\\\\\"\""));

        let parsed = from_msh_bytes(&buffer).expect("round-trip mesh");
        assert_eq!(
            parsed
                .elements
                .iter()
                .map(|element| element.marker)
                .collect::<Vec<_>>(),
            vec![10, 20]
        );
        assert_eq!(
            parsed.physical_groups.by_tag(2, 10).unwrap().name,
            "front air"
        );
        assert_eq!(
            parsed
                .physical_groups
                .by_name(2, "rear \\\"cavity\\\"")
                .unwrap()
                .tag,
            20
        );
        assert_eq!(
            parsed
                .boundaries
                .iter()
                .filter(|boundary| boundary.marker == 5)
                .count(),
            1
        );
    }
}
