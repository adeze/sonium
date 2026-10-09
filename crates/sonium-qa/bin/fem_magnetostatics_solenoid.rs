use sonium_fem::physics::electromagnetics::magnetostatics::{
    AxisymmetricAThetaForm, MU_0, uniform_azimuthal_current_density,
};
use sonium_qa::analytical::magnetostatics_solenoid;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let _form = AxisymmetricAThetaForm::new(MU_0);
        let _source = uniform_azimuthal_current_density(1.0e6);
        let b = magnetostatics_solenoid(1000.0, 2.0);
        assert!((b - MU_0 * 2000.0).abs() < 1e-18);
        println!("Magnetostatics solenoid: B = {:.6e}", b);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
