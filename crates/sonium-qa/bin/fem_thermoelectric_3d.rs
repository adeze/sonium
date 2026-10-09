use sonium_fem::physics::thermal::thermoelectric::ElectricalConductionForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.thermoelectric-3d.v1",
        &ElectricalConductionForm {
            electrical_conductivity: 1.0,
        },
        true,
    );
}
