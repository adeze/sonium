use sonium_fem::physics::fluid::phase_field_flow::ChemicalPotentialDiffusionForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.phase-field-flow-3d.v1",
        &ChemicalPotentialDiffusionForm { mobility: 1.0 },
        true,
    );
}
