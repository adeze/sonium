//! QA: S-N fatigue, rainflow counting, and spotweld force-life.
//!
//! Gates the production fatigue utilities against closed forms:
//! constant-amplitude damage, a two-block Miner sum, Goodman and
//! endurance behavior, the ASTM four-point decomposition, and pulsating
//! spotweld resultants from Worker 2's joint element (read-only).

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::joints::Spotweld;
use sonium_fem::solver::vibration_fatigue::{
    BasquinCurve, ForceLifeCurve, RainflowCycle, miner_damage, rainflow_counts,
    spotweld_force_damage,
};

fn sn_curve() -> BasquinCurve {
    BasquinCurve {
        fatigue_strength: 1000.0,
        exponent: -0.1,
        endurance_limit: None,
        ultimate_tensile: Some(1200.0),
    }
}

fn test_constant_amplitude() {
    // Sine at +-200 MPa, 1001 samples: 500 full range-400 cycles.
    let history: Vec<f64> = (0..1001)
        .map(|i| 200.0 * (2.0 * std::f64::consts::PI * i as f64 / 4.0).sin())
        .collect();
    let cycles = rainflow_counts(&history);
    let curve = sn_curve();
    let damage = miner_damage(&cycles, &curve);
    // Range-400 cycles about zero mean plus residue halves.
    let expected: f64 = cycles
        .iter()
        .map(|c| c.count / curve.cycles_to_failure(c.range / 2.0, c.mean))
        .sum();
    println!(
        "constant amplitude: damage={damage:.6e} cycles={}",
        cycles.len()
    );
    assert!((damage - expected).abs() < 1e-15);
    // Closed form: 250 sine periods at amplitude 200 count 249 full
    // range-400 cycles plus negligible range-200 residue halves.
    let life = curve.cycles_to_failure(200.0, 0.0);
    let full: f64 = cycles
        .iter()
        .filter(|c| c.count == 1.0)
        .map(|c| c.count)
        .sum();
    assert!((full - 249.0).abs() < 1e-9, "full count is {full}");
    assert!((damage - 250.0 / life).abs() / damage < 0.01);
}

fn test_two_block_miner() {
    let curve = sn_curve();
    let cycles = vec![
        RainflowCycle {
            range: 600.0,
            mean: 0.0,
            count: 100.0,
        },
        RainflowCycle {
            range: 300.0,
            mean: 0.0,
            count: 1000.0,
        },
    ];
    let expected =
        100.0 / curve.cycles_to_failure(300.0, 0.0) + 1000.0 / curve.cycles_to_failure(150.0, 0.0);
    let damage = miner_damage(&cycles, &curve);
    println!("two-block miner: {damage:.6e}");
    assert!((damage - expected).abs() < 1e-18);
}

fn test_goodman_and_endurance() {
    let curve = sn_curve();
    let zero_mean = curve.cycles_to_failure(300.0, 0.0);
    let tensile = curve.cycles_to_failure(300.0, 300.0);
    assert!(
        tensile < zero_mean,
        "Goodman must shorten tensile-mean life"
    );
    // Goodman closed form: 300/(1-300/1200) = 400.
    assert!((tensile - curve.cycles_to_failure(400.0, 0.0)).abs() / tensile < 1e-12);
    let limited = BasquinCurve {
        endurance_limit: Some(250.0),
        ..curve
    };
    assert!(limited.cycles_to_failure(200.0, 0.0).is_infinite());
    let cycles = vec![RainflowCycle {
        range: 400.0,
        mean: 0.0,
        count: 1e9,
    }];
    assert_eq!(miner_damage(&cycles, &limited), 0.0);
    println!("goodman/endurance: tensile={tensile:.6e} zero-mean={zero_mean:.6e}");
}

fn test_astm_decomposition() {
    let cycles = rainflow_counts(&[0.0, 1.0, -1.0, 1.0, -1.0, 0.0]);
    let full: Vec<_> = cycles.iter().filter(|c| c.count == 1.0).collect();
    assert_eq!(full.len(), 1);
    assert!((full[0].range - 2.0).abs() < 1e-15);
    let mut halves: Vec<f64> = cycles
        .iter()
        .filter(|c| c.count == 0.5)
        .map(|c| c.range)
        .collect();
    halves.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(halves, vec![1.0, 1.0, 2.0]);
    println!("astm decomposition: 1 full + 3 halves");
}

fn test_spotweld_pulsating() {
    // Pulsating axial resultant 0..2000 N through the joint element.
    let weld = Spotweld::new([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], 1e9, 1e6, 5000.0);
    let history: Vec<f64> = (0..11)
        .map(|i| {
            let d = if i % 2 == 0 { 0.0 } else { 2e-6 };
            weld.check([d, 0.0, 0.0, 0.0, 0.0, 0.0]).0
        })
        .collect();
    let peak = history.iter().cloned().fold(0.0, f64::max);
    assert!((peak - 2000.0).abs() < 1e-6, "resultant peak is {peak}");
    let curve = ForceLifeCurve {
        force_intercept: 10000.0,
        exponent: -0.2,
    };
    curve.validate().unwrap();
    let damage = spotweld_force_damage(&history, &curve);
    // All counted ranges are 2000 N (amplitude 1000): closed-form total.
    let total: f64 = rainflow_counts(&history).iter().map(|c| c.count).sum();
    let expected = total / curve.cycles_to_failure(1000.0);
    println!("spotweld pulsating: damage={damage:.6e} weights={total}");
    assert!((damage - expected).abs() / expected < 1e-12);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_constant_amplitude();
        test_two_block_miner();
        test_goodman_and_endurance();
        test_astm_decomposition();
        test_spotweld_pulsating();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
