use sonium_fem::physics::fluid::reacting_flow::SpeciesDiffusionForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.reacting-flow-3d.v1",
        &SpeciesDiffusionForm {
            density_diffusivity: 1.0,
        },
        true,
    );
}
