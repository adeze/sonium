//! Cross-module pipeline tests for `sonium-mesher`.
//!
//! Each test runs a generate → analyze → partition → extract → write chain
//! end to end, mirroring how FEM/BEM callers consume the crate. These live in
//! `tests/` (rather than per-module unit tests) so regressions in module
//! interaction — boundary ownership after partitioning, writer facet counts —
//! are caught in one place.

use sonium_mesher::mesh::{
    quality::per_element_quality, rectangular_mesh_triangles, unit_cube_tetrahedra,
};
use sonium_mesher::partition::{PartitionMesh, partition_mesh_bisection};

/// 2D pipeline: generate → boundaries → quality → partition → extract →
/// STL/PLY writers agree on the surface.
#[test]
fn rectangular_pipeline_partition_and_writers_agree() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4);
    assert_eq!(mesh.num_elements(), 4 * 4 * 2);
    assert!(
        !mesh.boundaries.is_empty(),
        "a closed 2D grid must expose boundary edges"
    );

    let quality = per_element_quality(&mesh);
    assert_eq!(quality.len(), mesh.num_elements());
    assert!(
        quality.iter().all(|q| q.mean_ratio.is_finite()),
        "quality metrics must be finite on a structured grid"
    );

    // Partitioning is deterministic: the same mesh and `k` always assign the
    // same owning partition (stable tie-break on the split axis).
    let first = partition_mesh_bisection(&mesh, 2);
    let second = partition_mesh_bisection(&mesh, 2);
    assert_eq!(first.partition_id, second.partition_id);
    assert!(first.partition_id.iter().all(|&id| id < 2));
    assert!(
        first.partition_id.contains(&0) && first.partition_id.contains(&1),
        "a 4x4 grid split in two must use both partitions"
    );

    let submesh = PartitionMesh::extract_checked(&mesh, &first, 0)
        .expect("extraction from a valid mesh and partition must succeed");
    assert!(submesh.num_elements() > 0);
    assert_eq!(submesh.num_elements(), first.partition_sizes()[0]);
    assert!(
        submesh
            .parent_element_indices
            .iter()
            .all(|&e| e < mesh.num_elements()),
        "extracted parent element indices must reference the source mesh"
    );
    assert!(
        submesh
            .elements
            .iter()
            .flatten()
            .all(|&n| n < submesh.num_nodes()),
        "extracted local node indices must be contiguous"
    );

    // Writers are deterministic and agree on the surface facet count.
    let surface = sonium_mesher::mesh::stl::surface_triangles(&mesh);
    let mut stl_a = Vec::new();
    let mut stl_b = Vec::new();
    sonium_mesher::mesh::stl::write_binary_stl(&mesh, &mut stl_a).unwrap();
    sonium_mesher::mesh::stl::write_binary_stl(&mesh, &mut stl_b).unwrap();
    assert_eq!(
        stl_a, stl_b,
        "STL output must be byte-identical across runs"
    );
    let facet_count = u32::from_le_bytes(stl_a[80..84].try_into().unwrap()) as usize;
    assert_eq!(facet_count, surface.len());
    assert_eq!(stl_a.len(), 84 + facet_count * 50);

    let mut ply_a = Vec::new();
    let mut ply_b = Vec::new();
    sonium_mesher::mesh::ply::write_ply(&mesh, &mut ply_a).unwrap();
    sonium_mesher::mesh::ply::write_ply(&mesh, &mut ply_b).unwrap();
    assert_eq!(
        ply_a, ply_b,
        "PLY output must be byte-identical across runs"
    );
    let header = String::from_utf8_lossy(&ply_a);
    assert!(
        header.contains(&format!("element vertex {}", mesh.num_nodes())),
        "PLY header must declare the mesh vertex count"
    );
}

/// 3D pipeline: generate → topology → quality → partition → extract.
#[test]
fn cube_pipeline_topology_quality_partition() {
    let mut mesh = unit_cube_tetrahedra(2);
    assert!(mesh.num_elements() > 0);

    let topology = mesh.try_topology().expect("generated cube must be valid");
    assert_eq!(topology.element_faces.len(), mesh.num_elements());

    let quality = per_element_quality(&mesh);
    assert!(
        quality.iter().all(|q| q.mean_ratio > 0.0),
        "structured cube tets must have strictly positive quality"
    );

    let partition = partition_mesh_bisection(&mesh, 4);
    assert_eq!(partition.partition_id.len(), mesh.num_elements());
    assert!(partition.partition_id.iter().all(|&id| id < 4));

    let submesh = PartitionMesh::extract_checked(&mesh, &partition, 0)
        .expect("extraction from a valid mesh and partition must succeed");
    assert_eq!(submesh.num_elements(), partition.partition_sizes()[0]);
    assert!(
        submesh
            .parent_element_indices
            .iter()
            .all(|&e| e < mesh.num_elements()),
        "extracted parent element indices must reference the source mesh"
    );
}
