//! Mesh and analytical reference fixture; no solved shell field.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_fem::mesh::circular_shell_mesh_triangles;
use sonium_qa::analytical::{biharmonic_clamped_circular_center, plate_bending_stiffness};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let mesh = circular_shell_mesh_triangles(0.25, 3, 24);
        let d = plate_bending_stiffness(70e9, 0.33, 0.005);
        let w = biharmonic_clamped_circular_center(500.0, 0.25, d);
        assert!(mesh.num_elements() > 0 && w > 0.0);
        println!(
            "Shell clamped circular plate: elems = {}, w0 = {:.6e}",
            mesh.num_elements(),
            w
        );
    });
    reference_result::emit(
        "reference.fem-shell-clamped-circular-plate.v1",
        result.is_ok(),
    );
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
