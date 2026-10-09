use sonium_fem::physics::coupled::conjugate_heat_transfer::ChtConductionForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.conjugate-heat-transfer-3d.v1",
        &ChtConductionForm {
            thermal_conductivity: 1.0,
        },
        true,
    );
}
