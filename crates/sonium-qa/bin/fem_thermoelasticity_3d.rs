use sonium_fem::physics::coupled::thermoelasticity::ThermoelasticDampingForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.thermoelasticity-3d.v1",
        &ThermoelasticDampingForm {
            coupling_coefficient: 1.0,
        },
        false,
    );
}
