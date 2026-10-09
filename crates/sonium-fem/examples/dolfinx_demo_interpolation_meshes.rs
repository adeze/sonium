//! Interpolation demo - translated from DOLFINx
//!
//! This example demonstrates:
//! - Creating functions on meshes with different element orders
//! - Interpolation between different meshes (tet and hex)
//! - Using existing interpolation infrastructure
//!
//! This matches the DOLFINx demo_interpolation_different_meshes example.

#![allow(non_snake_case)]

use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::{box_mesh_hexahedra, box_mesh_tetrahedra};
use sonium_fem::mesh::interpolation::{create_interpolation_data, interpolate_field};

fn main() {
    println!("=== DOLFINx Demo: Interpolation Different Meshes ===\n");

    // 1. Create two different meshes covering the same domain [0,1]³
    // Tetrahedral mesh (source)
    let mesh_tet = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 8, 8, 8);
    println!(
        "Source (tetrahedral): {} nodes, {} elements",
        mesh_tet.num_nodes(),
        mesh_tet.num_elements()
    );

    // Hexahedral mesh (target) - coarser
    let mesh_hex = box_mesh_hexahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 6, 6, 6);
    println!(
        "Target (hexahedral): {} nodes, {} elements",
        mesh_hex.num_nodes(),
        mesh_hex.num_elements()
    );

    // 2. Define a scalar field on the source mesh
    // f(x,y,z) = sin(πx) * sin(πy) * sin(πz)
    println!("\nDefining source field: f(x,y,z) = sin(πx)sin(πy)sin(πz)");

    let pi = std::f64::consts::PI;
    let source_values: Vec<f64> = mesh_tet
        .nodes
        .iter()
        .map(|n| (pi * n.x).sin() * (pi * n.y).sin() * (pi * n.z).sin())
        .collect();

    let max_src = source_values.iter().cloned().fold(0.0f64, f64::max);
    let min_src = source_values.iter().cloned().fold(0.0f64, f64::min);
    println!("Source field range: [{:.4}, {:.4}]", min_src, max_src);

    // 3. Create interpolation data from tet to hex mesh
    println!("\nCreating interpolation data (tet → hex)...");
    let interp_data = create_interpolation_data(&mesh_hex, &mesh_tet);

    // Count how many target nodes found source elements
    let found_count = interp_data
        .target_to_source
        .iter()
        .filter(|opt| opt.is_some())
        .count();
    println!(
        "Interpolation coverage: {}/{} ({:.1}%)",
        found_count,
        mesh_hex.num_nodes(),
        100.0 * found_count as f64 / mesh_hex.num_nodes() as f64
    );

    // 4. Interpolate to target mesh
    println!("\nInterpolating field...");
    let target_values = interpolate_field(&interp_data, &mesh_tet, &source_values, 0.0);

    let max_tgt = target_values.iter().cloned().fold(0.0f64, f64::max);
    let min_tgt = target_values.iter().cloned().fold(0.0f64, f64::min);
    println!("Interpolated field range: [{:.4}, {:.4}]", min_tgt, max_tgt);

    // 5. Also interpolate from hex back to tet (bidirectional)
    println!("\nCreating reverse interpolation (hex → tet)...");
    let interp_data_rev = create_interpolation_data(&mesh_tet, &mesh_hex);
    let source_rev: Vec<f64> = mesh_hex
        .nodes
        .iter()
        .map(|n| n.x + n.y + n.z) // Simple linear function
        .collect();
    let _tet_from_hex = interpolate_field(&interp_data_rev, &mesh_hex, &source_rev, 0.0);

    // 6. Export interpolated field to VTK
    println!("\nExporting to VTK...");
    let mut writer = VtuWriter::new(&mesh_hex);
    writer.add_point_scalar("interpolated_field", &target_values);

    if writer.write("interpolation_hex.vtu").is_ok() {
        println!("Written to interpolation_hex.vtu");
    }

    // Also export source on tet mesh
    let mut writer_tet = VtuWriter::new(&mesh_tet);
    writer_tet.add_point_scalar("source_field", &source_values);
    if writer_tet.write("interpolation_tet.vtu").is_ok() {
        println!("Written to interpolation_tet.vtu");
    }

    // 7. L2 error (for exact solution, should be small)
    let mut error_sq = 0.0;
    let mut count = 0;
    for (i, &val) in target_values.iter().enumerate() {
        let node = &mesh_hex.nodes[i];
        let exact = (pi * node.x).sin() * (pi * node.y).sin() * (pi * node.z).sin();
        let diff = val - exact;
        error_sq += diff * diff;
        count += 1;
    }
    let l2_error = (error_sq / count as f64).sqrt();
    println!("\nL2 interpolation error: {:.6e}", l2_error);

    println!("\n=== Demo Complete ===");
}
