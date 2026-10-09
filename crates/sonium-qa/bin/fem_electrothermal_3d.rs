use sonium_fem::physics::coupled::electrothermal::{DcCurrentForm, ElectrothermalMaterial};

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.electrothermal-3d.v1",
        &DcCurrentForm {
            material: ElectrothermalMaterial {
                reference_electrical_conductivity: 1.0,
                resistivity_temperature_coefficient: 0.004,
                reference_temperature: 293.15,
                relative_permeability: 1.0,
                thermal_conductivity: 1.0,
                density: 1.0,
                specific_heat: 1.0,
            },
            temperature: |_, _, _| 293.15,
        },
        true,
    );
}
