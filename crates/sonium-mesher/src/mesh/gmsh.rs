//! Gmsh MSH file import
//!
//! Converts a Gmsh `.msh` v4.1 file (parsed by `mshio`) into a [`Mesh`].
//!
//! Physical group tags from Gmsh become boundary markers. Supported element
//! types:
//! - P1/Q1: Tri3, Qua4, Tet4, Hex8 (linear).
//! - P2/Q2: Tri6, Qua9, Tet10, Hex27 (second-order Lagrange). Higher-order
//!   nodes are reordered from Gmsh's convention to sonium's FEM convention via
//!   the permutation tables in [`super::p2`].
//!
//! Boundary elements are matched by their **vertex** node set, so P2 boundary
//! elements (Line3, Tri6, Qua9) are accepted but only their corner vertices
//! are stored on `BoundaryFace.nodes`. If the volume mesh is then promoted
//! via [`Mesh::to_p2`](super::Mesh::to_p2), the boundary faces gain their
//! mid-side / face-centre nodes from the volume promotion.

use std::collections::HashMap;
use std::path::Path;

use super::p2::reorder_gmsh_to_fem_p2;
use super::{BoundaryFace, BoundaryType, ElementType, Mesh, Point};

/// Load a Gmsh `.msh` file and convert it to a [`Mesh`].
///
/// Physical group tags are preserved as boundary markers. If a boundary entity
/// has no physical tag, its entity tag is used as the marker instead.
///
/// Only linear elements are supported: Tri3, Qua4, Tet4, Hex8.
pub fn load_gmsh(path: impl AsRef<Path>) -> Result<Mesh, String> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| format!("Failed to read {}: {e}", path.as_ref().display()))?;
    from_msh_bytes(&bytes)
}

/// Parse MSH bytes and convert to a [`Mesh`].
pub fn from_msh_bytes(bytes: &[u8]) -> Result<Mesh, String> {
    let physical_groups = parse_physical_names(bytes)?;
    let msh = mshio::parse_msh_bytes(bytes).map_err(|e| format!("MSH parse error: {e}"))?;
    let mut mesh = from_msh_file(&msh)?;
    for (dimension, tag, name) in physical_groups {
        mesh.register_physical_group(dimension, tag, name)
            .map_err(|error| format!("invalid $PhysicalNames entry: {error}"))?;
    }
    Ok(mesh)
}

/// Parse the optional ASCII `$PhysicalNames` section.
///
/// `mshio` intentionally does not expose that section, so names are extracted
/// directly before the structural parser runs. Gmsh keeps section headers and
/// physical-name records textual even when later node/element sections are
/// binary; lossy conversion therefore remains safe for this bounded scan.
fn parse_physical_names(bytes: &[u8]) -> Result<Vec<(usize, i32, String)>, String> {
    let text = String::from_utf8_lossy(bytes);
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line.trim() != "$PhysicalNames" {
            continue;
        }
        let count: usize = lines
            .next()
            .ok_or("$PhysicalNames is missing its record count")?
            .trim()
            .parse()
            .map_err(|_| "$PhysicalNames record count is invalid")?;
        let mut groups = Vec::with_capacity(count);
        for _ in 0..count {
            let record = lines
                .next()
                .ok_or("$PhysicalNames ended before all records were read")?;
            groups.push(parse_physical_name_record(record)?);
        }
        if lines.next().map(str::trim) != Some("$EndPhysicalNames") {
            return Err("$PhysicalNames is missing $EndPhysicalNames".into());
        }
        return Ok(groups);
    }
    Ok(Vec::new())
}

fn parse_physical_name_record(line: &str) -> Result<(usize, i32, String), String> {
    let mut fields = line.trim().splitn(3, char::is_whitespace);
    let dimension = fields
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| format!("invalid physical-name dimension in {line:?}"))?;
    let tag = fields
        .next()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .ok_or_else(|| format!("invalid physical-name tag in {line:?}"))?;
    let quoted = fields
        .next()
        .map(str::trim)
        .ok_or_else(|| format!("missing physical name in {line:?}"))?;
    if quoted.len() < 2 || !quoted.starts_with('"') || !quoted.ends_with('"') {
        return Err(format!("physical name is not quoted in {line:?}"));
    }
    let escaped = &quoted[1..quoted.len() - 1];
    let mut name = String::with_capacity(escaped.len());
    let mut chars = escaped.chars();
    while let Some(character) = chars.next() {
        if character == '\\' {
            let escaped_character = chars
                .next()
                .ok_or_else(|| format!("trailing escape in physical name {line:?}"))?;
            match escaped_character {
                '\\' | '"' => name.push(escaped_character),
                other => {
                    name.push('\\');
                    name.push(other);
                }
            }
        } else {
            name.push(character);
        }
    }
    Ok((dimension, tag, name))
}

/// Convert a parsed `MshFile` into a [`Mesh`].
pub fn from_msh_file(msh: &mshio::MshFile<u64, i32, f64>) -> Result<Mesh, String> {
    let nodes_section = msh.data.nodes.as_ref().ok_or("MSH file has no nodes")?;
    let elements_section = msh
        .data
        .elements
        .as_ref()
        .ok_or("MSH file has no elements")?;

    // --- Build node tag → sequential index map ---
    // Gmsh node tags can be sparse (non-sequential). We collect all nodes
    // into a flat Vec and build a tag→index mapping.
    let mut node_coords: Vec<(f64, f64, f64)> = Vec::new();
    let mut tag_to_idx: HashMap<u64, usize> = HashMap::new();

    // Sequential index for all nodes
    let mut global_idx = 0usize;
    for block in &nodes_section.node_blocks {
        if let Some(ref tag_map) = block.node_tags {
            // Sparse: tag_map maps tag→local_idx
            for &tag in tag_map.keys() {
                tag_to_idx.insert(tag, global_idx + tag_map[&tag]);
            }
        }
        for (local_idx, node) in block.nodes.iter().enumerate() {
            if block.node_tags.is_none() {
                // Dense: tags are sequential. Per MSH spec, the first tag in a dense
                // block = min_node_tag + sum of previous blocks' node counts.
                // But mshio doesn't expose per-block first tag directly.
                // We use the global sequential index since we process all blocks in order.
            }
            let _ = local_idx;
            node_coords.push((node.x, node.y, node.z));
        }
        // For dense blocks, compute tags from the sequential range
        if block.node_tags.is_none() {
            // We need the actual node tags. For dense blocks, tags go from some start.
            // The MSH spec says node tags within a block are contiguous.
            // We map them by position within the overall node array.
        }
        global_idx += block.nodes.len();
    }

    // Since mshio's node_tags HashMap is tag→local_index_in_block, we build
    // a comprehensive mapping. For dense blocks (no node_tags), the MSH spec
    // states tags are contiguous, but mshio doesn't expose the starting tag.
    // The simplest correct approach: iterate element node references and
    // map them. But first, let's build from the min/max range.
    //
    // Actually, the cleanest approach: build a tag→(x,y,z) map from all blocks,
    // then assign sequential indices.
    let mut tag_to_coord: HashMap<u64, (f64, f64, f64)> = HashMap::new();
    let min_tag = nodes_section.min_node_tag;

    // For mshio, when node_tags is None, tags are sequential within the block.
    // The global sequence starts at min_node_tag.
    let mut running_tag = min_tag;
    for block in &nodes_section.node_blocks {
        if let Some(ref tag_map) = block.node_tags {
            for (&tag, &local_idx) in tag_map {
                let n = &block.nodes[local_idx];
                tag_to_coord.insert(tag, (n.x, n.y, n.z));
            }
        } else {
            for node in &block.nodes {
                tag_to_coord.insert(running_tag, (node.x, node.y, node.z));
                running_tag += 1;
            }
        }
    }

    // Build sequential node array and tag→index map
    let mut sorted_tags: Vec<u64> = tag_to_coord.keys().copied().collect();
    sorted_tags.sort_unstable();

    node_coords.clear();
    tag_to_idx.clear();
    for (idx, &tag) in sorted_tags.iter().enumerate() {
        tag_to_idx.insert(tag, idx);
        node_coords.push(tag_to_coord[&tag]);
    }

    // --- Determine mesh dimension ---
    let dim = if node_coords.iter().all(|(_, _, z)| z.abs() < 1e-14) {
        2
    } else {
        3
    };

    let mut mesh = Mesh::new(dim);
    for &(x, y, z) in &node_coords {
        mesh.add_node(Point::new_3d(x, y, z));
    }

    // --- Build entity tag → physical tag map for boundary markers ---
    let mut entity_physical: HashMap<(i32, i32), i32> = HashMap::new(); // (dim, entity_tag) → physical_tag
    if let Some(ref entities) = msh.data.entities {
        for pt in &entities.points {
            if let Some(&ptag) = pt.physical_tags.first() {
                entity_physical.insert((0, pt.tag), ptag);
            }
        }
        for curve in &entities.curves {
            if let Some(&ptag) = curve.physical_tags.first() {
                entity_physical.insert((1, curve.tag), ptag);
            }
        }
        for surface in &entities.surfaces {
            if let Some(&ptag) = surface.physical_tags.first() {
                entity_physical.insert((2, surface.tag), ptag);
            }
        }
        for volume in &entities.volumes {
            if let Some(&ptag) = volume.physical_tags.first() {
                entity_physical.insert((3, volume.tag), ptag);
            }
        }
    }

    // --- Process elements ---
    // Gmsh is free to order lower-dimensional entity blocks before volume
    // blocks. Defer boundary ownership resolution until every volume element
    // has been inserted; resolving inline would silently assign `element_idx =
    // 0` when the owning volume has not been seen yet.
    let mut pending_boundaries: Vec<(Vec<usize>, i32)> = Vec::new();

    let volume_dim = dim; // 2D mesh → triangles/quads, 3D → tets/hexes

    for block in &elements_section.element_blocks {
        let entity_dim = block.entity_dim;
        let entity_tag = block.entity_tag;

        // Map element type. `order` is 1 for P1/Q1, 2 for P2/Q2.
        let (sonium_type, is_volume, order) = match block.element_type {
            mshio::ElementType::Tri3 => {
                let is_vol = entity_dim as usize == volume_dim && dim == 2;
                (Some(ElementType::Triangle), is_vol, 1)
            }
            mshio::ElementType::Tri6 => {
                let is_vol = entity_dim as usize == volume_dim && dim == 2;
                (Some(ElementType::Triangle), is_vol, 2)
            }
            mshio::ElementType::Qua4 => {
                let is_vol = entity_dim as usize == volume_dim && dim == 2;
                (Some(ElementType::Quadrilateral), is_vol, 1)
            }
            mshio::ElementType::Qua9 => {
                let is_vol = entity_dim as usize == volume_dim && dim == 2;
                (Some(ElementType::Quadrilateral), is_vol, 2)
            }
            mshio::ElementType::Tet4 => (Some(ElementType::Tetrahedron), true, 1),
            mshio::ElementType::Tet10 => (Some(ElementType::Tetrahedron), true, 2),
            mshio::ElementType::Hex8 => (Some(ElementType::Hexahedron), true, 1),
            mshio::ElementType::Hex27 => (Some(ElementType::Hexahedron), true, 2),
            mshio::ElementType::Lin2 | mshio::ElementType::Lin3 => (None, false, 1),
            mshio::ElementType::Pnt => continue,
            _ => {
                log::warn!(
                    "Unsupported Gmsh element type {:?}, skipping block",
                    block.element_type
                );
                continue;
            }
        };

        if is_volume {
            if let Some(etype) = sonium_type {
                let marker = entity_physical
                    .get(&(entity_dim, entity_tag))
                    .copied()
                    .unwrap_or(0);
                for elem in &block.elements {
                    let raw: Vec<usize> = elem.nodes.iter().map(|&t| tag_to_idx[&t]).collect();
                    let nodes = if order == 2 {
                        reorder_gmsh_to_fem_p2(etype, &raw)
                    } else {
                        raw
                    };
                    mesh.add_element_with_marker(etype, nodes, marker);
                }
            }
        } else {
            // Boundary element — determine marker from physical group
            let marker = entity_physical
                .get(&(entity_dim, entity_tag))
                .copied()
                .unwrap_or(entity_tag); // fallback to entity tag

            // Number of vertices to keep on the boundary. Higher-order Gmsh
            // boundary elements (Lin3, Tri6, Qua9) put corner vertices first,
            // followed by mid-side/centre nodes; drop the higher-order tail to
            // keep `BoundaryFace.nodes` as a vertex-only list. The volume
            // mesh's own permutation handles the higher-order DOFs.
            let n_vertex = match block.element_type {
                mshio::ElementType::Lin2 | mshio::ElementType::Lin3 => 2,
                mshio::ElementType::Tri3 | mshio::ElementType::Tri6 => 3,
                mshio::ElementType::Qua4 | mshio::ElementType::Qua9 => 4,
                _ => 0,
            };

            for elem in &block.elements {
                let raw: Vec<usize> = elem.nodes.iter().map(|&t| tag_to_idx[&t]).collect();
                let nodes: Vec<usize> = if n_vertex == 0 {
                    raw
                } else {
                    raw.into_iter().take(n_vertex).collect()
                };

                pending_boundaries.push((nodes, marker));
            }
        }
    }

    for (nodes, marker) in pending_boundaries {
        let (element_idx, local_idx) = find_parent_face(&mesh, &nodes).ok_or_else(|| {
            format!(
                "boundary node set {:?} does not match any volume element face/edge",
                nodes
            )
        })?;
        mesh.boundaries.push(BoundaryFace {
            nodes,
            boundary_type: BoundaryType::Dirichlet,
            marker,
            element_idx,
            local_idx,
        });
    }

    // If no boundary elements were in the file, auto-detect
    if mesh.boundaries.is_empty() {
        mesh.detect_boundaries();
    }

    Ok(mesh)
}

/// Find the index of a volume element that contains all the given boundary nodes.
fn find_parent_face(mesh: &Mesh, boundary_nodes: &[usize]) -> Option<(usize, usize)> {
    let mut requested = boundary_nodes.to_vec();
    requested.sort_unstable();
    for (element_idx, element) in mesh.elements.iter().enumerate() {
        if mesh.dimension == 3 {
            // Gmsh writes 1D curve elements alongside the volume mesh; a
            // 2-node set owns a tet edge, a 3-node set owns a tet face.
            if requested.len() == 2 {
                for (local_idx, (a, b)) in element.edges().into_iter().enumerate() {
                    let mut candidate = vec![a, b];
                    candidate.sort_unstable();
                    if candidate == requested {
                        return Some((element_idx, local_idx));
                    }
                }
            } else {
                for (local_idx, face) in element.faces().into_iter().enumerate() {
                    let mut candidate = face;
                    candidate.sort_unstable();
                    if candidate == requested {
                        return Some((element_idx, local_idx));
                    }
                }
            }
        } else {
            for (local_idx, (a, b)) in element.edges().into_iter().enumerate() {
                let mut candidate = vec![a, b];
                candidate.sort_unstable();
                if candidate == requested {
                    return Some((element_idx, local_idx));
                }
            }
        }
    }
    None
}
