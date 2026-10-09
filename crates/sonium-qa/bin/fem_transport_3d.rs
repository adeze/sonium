use sonium_fem::physics::fluid::transport::TransportDiffusionForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.transport-3d.v1",
        &TransportDiffusionForm {
            diffusivity: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        true,
    );
}
