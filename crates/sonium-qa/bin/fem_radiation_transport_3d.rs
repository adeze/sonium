use sonium_fem::physics::thermal::radiation_transport::StreamingExtinctionForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.radiation-transport-3d.v1",
        &StreamingExtinctionForm {
            direction: [1.0, 0.0, 0.0],
            extinction_coefficient: 1.0,
        },
        false,
    );
}
