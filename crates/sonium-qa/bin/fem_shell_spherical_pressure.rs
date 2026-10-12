//! Mesh and analytical reference fixture; no solved shell field.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_fem::mesh::spherical_shell_surface_mesh_triangles;
use sonium_qa::analytical::shell_spherical_pressure_membrane_stress;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let mesh = spherical_shell_surface_mesh_triangles(0.2, 8, 16);
        let stress = shell_spherical_pressure_membrane_stress(1000.0, 0.2, 0.002);
        assert!(mesh.num_elements() > 0 && (stress - 5.0e4).abs() < 1e-9);
        println!("Shell spherical pressure: membrane stress = {:.6e}", stress);
    });
    reference_result::emit("reference.fem-shell-spherical-pressure.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
