use sonium_fem::physics::coupled::battery::BatteryScalarDiffusionForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.battery-3d.v1",
        &BatteryScalarDiffusionForm { coefficient: 1.0 },
        true,
    );
}
