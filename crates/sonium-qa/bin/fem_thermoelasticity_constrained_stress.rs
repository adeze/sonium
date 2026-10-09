use sonium_fem::physics::coupled::thermoelasticity::ThermoelasticMaterial;

/// Test constrained thermal stress formulas for various materials.
fn test_constrained_thermal_stress_3d() {
    let test_cases = [
        // (E, nu, alpha, T_ref, T, expected_stress_sign)
        (210e9, 0.3, 12e-6, 293.0, 393.0, 1.0), // steel, heating
        (70e9, 0.33, 23e-6, 293.0, 393.0, 1.0), // aluminum, heating
        (210e9, 0.3, 12e-6, 293.0, 193.0, -1.0), // steel, cooling
        (110e9, 0.31, 17e-6, 300.0, 400.0, 1.0), // titanium, heating
        (3e9, 0.35, 50e-6, 293.0, 343.0, 1.0),  // polymer, heating
    ];

    for (e, nu, alpha, t_ref, t, sign) in &test_cases {
        let material = ThermoelasticMaterial {
            young_modulus: *e,
            poisson_ratio: *nu,
            thermal_expansion: *alpha,
            reference_temperature: *t_ref,
        };
        let delta_t = t - t_ref;
        let stress = material.constrained_thermal_stress_3d(*t);
        let bulk = material.lame_lambda() + 2.0 * material.shear_modulus() / 3.0;
        let expected = 3.0 * bulk * alpha * delta_t;

        println!(
            "Constrained stress 3D (E={:.2e}, nu={:.2}): computed={:.4e}, expected={:.4e}",
            e, nu, stress, expected
        );
        assert!(
            (stress - expected).abs() / expected.abs() < 1e-10,
            "3D constrained stress mismatch: computed={}, expected={}",
            stress,
            expected
        );
        assert!(stress * sign > 0.0, "Stress should have correct sign");
    }
}

/// Test 1D constrained thermal stress: sigma = E * alpha * delta_T.
fn test_constrained_thermal_stress_1d() {
    let test_cases = [
        (210e9, 0.3, 12e-6, 293.0, 393.0),
        (70e9, 0.33, 23e-6, 293.0, 393.0),
        (210e9, 0.3, 12e-6, 293.0, 193.0),
    ];

    for (e, _nu, alpha, t_ref, t) in &test_cases {
        let material = ThermoelasticMaterial {
            young_modulus: *e,
            poisson_ratio: *_nu,
            thermal_expansion: *alpha,
            reference_temperature: *t_ref,
        };
        let delta_t = t - t_ref;
        let stress = material.constrained_thermal_stress_1d(*t);
        let expected = e * alpha * delta_t;

        println!(
            "Constrained stress 1D (E={:.2e}): computed={:.4e}, expected={:.4e}",
            e, stress, expected
        );
        assert!(
            (stress - expected).abs() / expected.abs() < 1e-10,
            "1D constrained stress mismatch: computed={}, expected={}",
            stress,
            expected
        );
    }
}

/// Test Lame parameters against analytical formulas.
fn test_lame_parameters() {
    let e = 210e9;
    let nu = 0.3;
    let material = ThermoelasticMaterial {
        young_modulus: e,
        poisson_ratio: nu,
        thermal_expansion: 12e-6,
        reference_temperature: 293.0,
    };

    let lambda = material.lame_lambda();
    let mu = material.shear_modulus();

    let expected_lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let expected_mu = e / (2.0 * (1.0 + nu));

    println!(
        "Lame parameters: lambda={:.4e} (expected {:.4e}), mu={:.4e} (expected {:.4e})",
        lambda, expected_lambda, mu, expected_mu
    );
    assert!((lambda - expected_lambda).abs() / expected_lambda < 1e-10);
    assert!((mu - expected_mu).abs() / expected_mu < 1e-10);

    // Verify relationship: E = mu * (3*lambda + 2*mu) / (lambda + mu)
    let computed_e = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
    println!("  E from Lame: {:.4e} (expected {:.4e})", computed_e, e);
    assert!((computed_e - e).abs() / e < 1e-10);
}

/// Test that 3D stress = 1D stress / (1 - 2*nu) for isotropic materials.
/// This is because in 3D, the constraint is hydrostatic, so sigma_3d = E*alpha*delta_T / (1 - 2*nu)
/// while in 1D, sigma_1d = E*alpha*delta_T.
fn test_1d_vs_3d_relationship() {
    let e = 210e9;
    let nu = 0.3;
    let alpha = 12e-6;
    let t_ref = 293.0;
    let t = 393.0;
    let _delta_t = t - t_ref;

    let material = ThermoelasticMaterial {
        young_modulus: e,
        poisson_ratio: nu,
        thermal_expansion: alpha,
        reference_temperature: t_ref,
    };

    let stress_1d = material.constrained_thermal_stress_1d(t);
    let stress_3d = material.constrained_thermal_stress_3d(t);

    let expected_ratio = 1.0 / (1.0 - 2.0 * nu);
    let computed_ratio = stress_3d / stress_1d;

    println!(
        "1D vs 3D: sigma_1d={:.4e}, sigma_3d={:.4e}, ratio={:.4e} (expected {:.4e})",
        stress_1d, stress_3d, computed_ratio, expected_ratio
    );
    assert!((computed_ratio - expected_ratio).abs() < 1e-10);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: thermoelasticity constrained stress ===");
        test_constrained_thermal_stress_3d();
        test_constrained_thermal_stress_1d();
        test_lame_parameters();
        test_1d_vs_3d_relationship();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
