use sonium_fem::mesh::two_cylinder_mesh_triangles;
use sonium_qa::analytical::{EPSILON_0, electrostatics_two_cylinder_capacitance};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let c = electrostatics_two_cylinder_capacitance(0.1, 0.5, EPSILON_0);
        let mesh = two_cylinder_mesh_triangles(
            -1.0,
            1.0,
            -0.6,
            0.6,
            (-0.25, 0.0, 0.1),
            (0.25, 0.0, 0.1),
            20,
            12,
        );
        assert!(c > 0.0);
        assert!(mesh.num_elements() > 0);
        println!(
            "Electrostatics two-cylinder: C' = {:.6e}, elems = {}",
            c,
            mesh.num_elements()
        );
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
