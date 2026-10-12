//! Mesh and analytical reference fixture; no solved shell field.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_fem::mesh::cylindrical_shell_mesh_triangles;
use sonium_qa::analytical::shell_cylindrical_pressure_hoop_stress;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let mesh = cylindrical_shell_mesh_triangles(0.2, 1.0, 24, 4);
        let stress = shell_cylindrical_pressure_hoop_stress(1000.0, 0.2, 0.002);
        assert!(mesh.num_elements() > 0 && (stress - 1.0e5).abs() < 1e-9);
        println!("Shell cylindrical pressure: hoop stress = {:.6e}", stress);
    });
    reference_result::emit(
        "reference.fem-shell-cylindrical-pressure.v1",
        result.is_ok(),
    );
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
