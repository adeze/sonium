use sonium_fem::physics::coupled::mhd::MhdPotentialForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.mhd-3d.v1",
        &MhdPotentialForm {
            electrical_conductivity: 1.0,
        },
        true,
    );
}
