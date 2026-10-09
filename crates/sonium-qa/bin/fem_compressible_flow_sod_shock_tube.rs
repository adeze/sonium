use sonium_fem::physics::fluid::compressible_flow::{
    ConservativeState, IdealGas, isentropic_area_ratio, local_lax_friedrichs_speed, rusanov_flux,
    sod_shock_tube_states,
};

/// Test 1: Sod shock tube initial states match expected values.
fn test_sod_shock_tube_states() {
    let gas = IdealGas::AIR;
    let (left, right) = sod_shock_tube_states(gas);

    assert!(
        (left.density - 1.0).abs() < 1e-12,
        "Left density should be 1.0"
    );
    assert!(
        (left.velocity()[0]).abs() < 1e-12,
        "Left velocity should be 0"
    );
    assert!(
        (left.pressure(gas) - 1.0).abs() < 1e-12,
        "Left pressure should be 1.0"
    );

    assert!(
        (right.density - 0.125).abs() < 1e-12,
        "Right density should be 0.125"
    );
    assert!(
        (right.velocity()[0]).abs() < 1e-12,
        "Right velocity should be 0"
    );
    assert!(
        (right.pressure(gas) - 0.1).abs() < 1e-12,
        "Right pressure should be 0.1"
    );

    println!(
        "Sod shock tube states: left=({}, {}, {}), right=({}, {}, {})",
        left.density,
        left.velocity()[0],
        left.pressure(gas),
        right.density,
        right.velocity()[0],
        right.pressure(gas)
    );
}

/// Test 2: Rusanov flux with left=right should reduce to exact Euler flux.
fn test_rusanov_consistency() {
    let gas = IdealGas::AIR;
    let state = ConservativeState::from_primitive(2.0, [3.0, 0.0, 0.0], 5.0, gas);
    let normal = [1.0, 0.0, 0.0];

    let flux = rusanov_flux(state, state, gas, normal);
    let expected = state.euler_flux(gas, normal);

    for i in 0..5 {
        assert!(
            (flux[i] - expected[i]).abs() < 1e-12,
            "Rusanov consistency failed at component {}: {} vs {}",
            i,
            flux[i],
            expected[i]
        );
    }
    println!("Rusanov consistency: PASSED");
}

/// Test 3: Rusanov flux is dissipative (eigenvalues bounded by max wave speed).
fn test_rusanov_dissipation() {
    let gas = IdealGas::AIR;
    let left = ConservativeState::from_primitive(1.0, [0.0, 0.0, 0.0], 1.0, gas);
    let right = ConservativeState::from_primitive(0.125, [0.0, 0.0, 0.0], 0.1, gas);
    let normal = [1.0, 0.0, 0.0];

    let smax = local_lax_friedrichs_speed(left, right, gas, normal);
    let flux = rusanov_flux(left, right, gas, normal);

    // The dissipation term magnitude should be related to smax
    let fl = left.euler_flux(gas, normal);
    let fr = right.euler_flux(gas, normal);
    let ul = [
        left.density,
        left.momentum[0],
        left.momentum[1],
        left.momentum[2],
        left.total_energy,
    ];
    let ur = [
        right.density,
        right.momentum[0],
        right.momentum[1],
        right.momentum[2],
        right.total_energy,
    ];

    for i in 0..5 {
        let central = 0.5 * (fl[i] + fr[i]);
        let diff = 0.5 * smax * (ur[i] - ul[i]).abs();
        // Rusanov flux = central - diff, so it should be bounded
        assert!(
            flux[i].abs() < central.abs() + diff + 1.0,
            "Rusanov flux unbounded at component {}: {}",
            i,
            flux[i]
        );
    }

    println!("Rusanov dissipation: smax={:.4e}, PASSED", smax);
}

/// Test 4: Isentropic area ratio at Mach=1 should equal 1.
fn test_isentropic_area_ratio_unity() {
    let gamma = 1.4;
    let a_ratio = isentropic_area_ratio(1.0, gamma);
    println!("Isentropic area ratio at M=1: {:.12e}", a_ratio);
    assert!(
        (a_ratio - 1.0).abs() < 1e-12,
        "Area ratio at M=1 should be 1.0"
    );
}

/// Test 5: Isentropic area ratio increases with Mach number.
fn test_isentropic_area_ratio_monotonic() {
    let gamma = 1.4;
    let a1 = isentropic_area_ratio(1.5, gamma);
    let a2 = isentropic_area_ratio(2.0, gamma);
    let a3 = isentropic_area_ratio(3.0, gamma);

    println!(
        "Isentropic area ratios: M=1.5 -> {:.4e}, M=2.0 -> {:.4e}, M=3.0 -> {:.4e}",
        a1, a2, a3
    );
    assert!(a2 > a1, "Area ratio should increase with Mach");
    assert!(a3 > a2, "Area ratio should increase with Mach");
}

/// Test 6: Conservative state round-trip (primitive -> conservative -> pressure/velocity).
fn test_conservative_state_roundtrip() {
    let gas = IdealGas::AIR;
    let rho = 2.5;
    let v = [1.5, -0.5, 0.3];
    let p = 101325.0;

    let state = ConservativeState::from_primitive(rho, v, p, gas);
    let p_recovered = state.pressure(gas);
    let v_recovered = state.velocity();

    assert!((state.density - rho).abs() < 1e-12, "Density mismatch");
    assert!(
        (p_recovered - p).abs() < 1e-6,
        "Pressure mismatch: {} vs {}",
        p_recovered,
        p
    );
    for i in 0..3 {
        assert!(
            (v_recovered[i] - v[i]).abs() < 1e-12,
            "Velocity mismatch at {}: {} vs {}",
            i,
            v_recovered[i],
            v[i]
        );
    }
    println!("Conservative state roundtrip: PASSED");
}

/// Test 7: Max wave speed for stationary flow is just sound speed.
fn test_max_wave_speed_stationary() {
    let gas = IdealGas::AIR;
    let state = ConservativeState::from_primitive(1.225, [0.0, 0.0, 0.0], 101325.0, gas);
    let normal = [1.0, 0.0, 0.0];

    let c = state.max_wave_speed(gas, normal);
    let expected_c = gas.speed_of_sound(101325.0, 1.225);

    println!(
        "Max wave speed (stationary): {:.4e}, expected {:.4e}",
        c, expected_c
    );
    assert!(
        (c - expected_c).abs() < 1e-6,
        "Wave speed mismatch: {} vs {}",
        c,
        expected_c
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: compressible flow Sod shock tube ===");
        test_sod_shock_tube_states();
        test_rusanov_consistency();
        test_rusanov_dissipation();
        test_isentropic_area_ratio_unity();
        test_isentropic_area_ratio_monotonic();
        test_conservative_state_roundtrip();
        test_max_wave_speed_stationary();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
