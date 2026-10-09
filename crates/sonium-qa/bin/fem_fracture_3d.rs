use sonium_fem::physics::solid::fracture::{PhaseFieldDamageForm, PhaseFieldFractureMaterial};

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.fracture-3d.v1",
        &PhaseFieldDamageForm {
            material: PhaseFieldFractureMaterial {
                critical_energy_release_rate: 2.0,
                length_scale: 0.1,
                residual_stiffness: 1.0e-8,
            },
        },
        false,
    );
}
