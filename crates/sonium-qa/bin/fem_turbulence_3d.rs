use sonium_fem::physics::fluid::turbulence::TurbulenceTransportForm;

fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.turbulence-3d.v1",
        &TurbulenceTransportForm {
            molecular_viscosity: 1.0,
            eddy_viscosity: 0.0,
            sigma: 1.0,
            velocity: [0.0; 3],
            reaction: 0.0,
        },
        true,
    );
}
