use sonium_fem::physics::coupled::electrochemistry::NernstPlanckDiffusionForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.electrochemistry-3d.v1",
        &NernstPlanckDiffusionForm { diffusivity: 1.0 },
        true,
    );
}
