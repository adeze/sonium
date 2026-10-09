use sonium_fem::physics::fluid::compressible_flow::{
    ConservativeState, IdealGas, isentropic_area_ratio, local_lax_friedrichs_speed, rusanov_flux,
    sod_shock_tube_states,
};

fn test_euler_flux_and_rusanov_consistency() {
    let gas = IdealGas::AIR;
    let state = ConservativeState::from_primitive(1.2, [30.0, 0.0, 0.0], 101_325.0, gas);
    let physical = state.euler_flux(gas, [1.0, 0.0, 0.0]);
    let numerical = rusanov_flux(state, state, gas, [1.0, 0.0, 0.0]);

    println!(
        "Euler flux: mass={:.4e}, momentum={:.4e}, energy={:.4e}",
        physical[0], physical[1], physical[4]
    );
    for i in 0..5 {
        assert!((physical[i] - numerical[i]).abs() < 1.0e-8);
    }
}

fn test_sod_shock_tube_numerical_flux() {
    let gas = IdealGas::AIR;
    let (left, right) = sod_shock_tube_states(gas);
    let flux = rusanov_flux(left, right, gas, [1.0, 0.0, 0.0]);
    let speed = local_lax_friedrichs_speed(left, right, gas, [1.0, 0.0, 0.0]);

    println!(
        "Sod Rusanov flux: rho={:.4e}, mom={:.4e}, E={:.4e}, speed={:.4e}",
        flux[0], flux[1], flux[4], speed
    );
    assert!(flux[0] > 0.0);
    assert!(flux[1] > 0.0);
    assert!(speed > 1.0);
}

fn test_isentropic_nozzle_area_relation() {
    let gamma = 1.4;
    let throat = isentropic_area_ratio(1.0, gamma);
    let subsonic = isentropic_area_ratio(0.3, gamma);
    let supersonic = isentropic_area_ratio(2.0, gamma);

    println!(
        "Nozzle area ratios: M=1 {:.4e}, M=.3 {:.4e}, M=2 {:.4e}",
        throat, subsonic, supersonic
    );
    assert!((throat - 1.0).abs() < 1.0e-12);
    assert!(subsonic > 1.0);
    assert!(supersonic > 1.0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: compressible Euler flow ===");
        test_euler_flux_and_rusanov_consistency();
        test_sod_shock_tube_numerical_flux();
        test_isentropic_nozzle_area_relation();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
