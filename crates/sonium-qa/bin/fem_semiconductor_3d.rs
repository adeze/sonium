use sonium_fem::physics::electromagnetics::semiconductor::CarrierDiffusionForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.semiconductor-3d.v1",
        &CarrierDiffusionForm { diffusivity: 1.0 },
        true,
    );
}
