use sonium_fem::physics::electromagnetics::magnetostatics::PointDipoleSource;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let src = PointDipoleSource::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert_eq!(src.moment[2], 1.0);
        println!(
            "Magnetostatics point dipole source: mz = {:.1}",
            src.moment[2]
        );
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
