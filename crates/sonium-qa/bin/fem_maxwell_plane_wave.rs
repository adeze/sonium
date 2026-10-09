use num_complex::Complex64;
use sonium_fem::physics::electromagnetics::maxwell::incident_plane_wave_e;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let k = 2.0 * std::f64::consts::PI;
        let wave = incident_plane_wave_e(
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [1.0, 0.0, 0.0],
            k,
        );
        let e0 = wave(0.0, 0.0, 0.0)[0];
        let e1 = wave(1.0, 0.0, 0.0)[0];
        assert!((e0 - e1).norm() < 1e-12);
        println!(
            "Maxwell plane wave phase closure: |E(0)-E(lambda)| = {:.3e}",
            (e0 - e1).norm()
        );
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
