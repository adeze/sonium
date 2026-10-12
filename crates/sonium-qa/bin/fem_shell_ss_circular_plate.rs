//! Mesh and analytical reference fixture; no solved shell field.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_fem::mesh::circular_shell_mesh_triangles;
use sonium_qa::analytical::{biharmonic_ss_circular_point_center, plate_bending_stiffness};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let mesh = circular_shell_mesh_triangles(0.25, 3, 24);
        let d = plate_bending_stiffness(70e9, 0.33, 0.005);
        let w = biharmonic_ss_circular_point_center(10.0, 0.25, d);
        assert!(mesh.boundaries.iter().any(|b| b.marker == 1) && w > 0.0);
        println!(
            "Shell SS circular plate: rim_edges = {}, w0 = {:.6e}",
            mesh.boundaries.len(),
            w
        );
    });
    reference_result::emit("reference.fem-shell-ss-circular-plate.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
