use sonium_fem::physics::fluid::porous_flow::DarcyPressureForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.porous-flow-3d.v1",
        &DarcyPressureForm {
            mobility: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        true,
    );
}
