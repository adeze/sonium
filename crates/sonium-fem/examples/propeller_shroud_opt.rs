//! Propeller shroud shape optimization
//!
//! 9 control points, all free. Composite objective balances noise reduction
//! against drag, curvature, and volume penalties.
//!
//! Central FD gradient: 2 * 9 = 18 Helmholtz solves per gradient evaluation.
//!
//! Run with: `cargo run -p sonium-fem --release --example propeller_shroud_opt`

use sonium_fem::optimization::shroud_opt::{
    ShroudOptConfig, ShroudProfile, evaluate_composite_objective, run_shroud_optimization,
};
use sonium_fem::physics::acoustics::aeroacoustics::propeller_source::PropellerSource;

fn main() {
    println!("=== Propeller Shroud Shape Optimization ===\n");

    let prop = PropellerSource::drone_5inch(8000.0);
    println!(
        "Propeller: {} blades, {:.0} RPM, BPF = {:.0} Hz, M_tip = {:.3}\n",
        prop.n_blades,
        prop.rpm,
        prop.blade_passing_frequency(),
        prop.tip_mach()
    );

    let r_pipe = 0.070; // propeller clearance radius (tip 63.5mm + 6.5mm gap)
    let l_shroud = 0.15;

    // 5 spline control points — the cubic spline gives a smooth C² profile.
    //
    //   z=0        z=37.5      z=75        z=112.5     z=150
    //   inlet lip  propeller   expansion   diffuser    outlet
    //              zone        chamber     exit
    //
    // Point 1 (z=37.5mm) is the propeller zone — must clear the blades.
    // The propeller sits between points 0 and 1 (z ≈ 20mm); the spline
    // interpolates smoothly through it.
    let n_ctrl = 5;
    let baseline = ShroudProfile::straight(r_pipe, 0.0, l_shroud, n_ctrl);

    let r_prop_clearance = (prop.tip_radius + 0.003) * 1e3;
    println!(
        "Propeller tip: {:.1} mm, clearance radius: {:.1} mm",
        prop.tip_radius * 1e3,
        r_prop_clearance
    );
    println!(
        "Profile: natural cubic spline through {} control points",
        n_ctrl
    );

    let v_straight = baseline.volume();

    println!(
        "\nDesign space: {} control points, {} free (all)",
        n_ctrl,
        baseline.n_free()
    );
    println!(
        "Pipe radius: {:.1} mm, length: {:.0} mm",
        r_pipe * 1e3,
        l_shroud * 1e3
    );
    println!(
        "Propeller zone at z = {:.1} mm (control point 1)",
        baseline.z_positions[1] * 1e3
    );
    println!("Straight duct volume: {:.2} cm³\n", v_straight * 1e6);

    // 30% drag budget, propeller clearance enforced via per-point bound
    let r_prop_min = prop.tip_radius + 0.003; // 66.5mm — hard floor at propeller plane
    let config = ShroudOptConfig {
        propeller: prop,
        n_harmonics: 4,
        nz: 40,
        nr: 8,
        r_min: 0.035, // general: can contract aggressively
        r_max: 0.140, // general: large expansion allowed
        // Point 1 (propeller plane at z≈19mm): must clear the blades
        per_point_bounds: vec![(1, r_prop_min, 0.140)],
        max_iterations: 40,
        fd_step: 2e-4,
        w_drag: 15.0,
        w_curvature: 0.1,
        v_max: v_straight * 2.0,
        w_volume: 1e6,
    };

    println!("Constraints:");
    println!(
        "  r bounds: [{:.0}, {:.0}] mm",
        config.r_min * 1e3,
        config.r_max * 1e3
    );
    println!("  V_max = {:.2} cm³ (150% of straight)", config.v_max * 1e6);
    println!(
        "  w_drag={:.0}, w_curv={:.1}, w_vol={:.0e}\n",
        config.w_drag, config.w_curvature, config.w_volume
    );

    // Baseline evaluation
    let (obj_base, swl_base, drag_base, curv_base, vol_base) =
        evaluate_composite_objective(&baseline, &config);
    println!("Baseline (straight duct):");
    println!("  SWL_A   = {:.1} dB", swl_base);
    println!(
        "  drag    = {:.4} ({:.1} dB)",
        drag_base,
        config.w_drag * drag_base
    );
    println!(
        "  curv    = {:.4} ({:.2} dB)",
        curv_base,
        config.w_curvature * curv_base
    );
    println!("  volume  = {:.2} cm³", vol_base * 1e6);
    println!("  J_total = {:.1} dB\n", obj_base);

    // Run optimizer
    println!(
        "L-BFGS-B: {} max iters, {} solves/gradient (central FD)...\n",
        config.max_iterations,
        2 * baseline.n_free()
    );
    let result = run_shroud_optimization(&baseline, &config);

    // Evaluate result
    let (obj_opt, swl_opt, drag_opt, curv_opt, vol_opt) =
        evaluate_composite_objective(&result.profile, &config);

    println!("Done: {} iterations\n", result.iterations);

    println!(
        "{:>18} {:>12} {:>12} {:>10}",
        "", "Straight", "Optimized", "Delta"
    );
    println!("{:-<55}", "");
    println!(
        "{:>18} {:>12.1} {:>12.1} {:>+10.1}",
        "SWL_A (dB)",
        swl_base,
        swl_opt,
        swl_opt - swl_base
    );
    println!(
        "{:>18} {:>12.2} {:>12.2} {:>+10.2}",
        "Drag (dB)",
        config.w_drag * drag_base,
        config.w_drag * drag_opt,
        config.w_drag * (drag_opt - drag_base)
    );
    println!(
        "{:>18} {:>12.2} {:>12.2} {:>+10.2}",
        "Curvature (dB)",
        config.w_curvature * curv_base,
        config.w_curvature * curv_opt,
        config.w_curvature * (curv_opt - curv_base)
    );
    println!(
        "{:>18} {:>12.2} {:>12.2} {:>+10.2}",
        "Volume (cm³)",
        vol_base * 1e6,
        vol_opt * 1e6,
        (vol_opt - vol_base) * 1e6
    );
    println!(
        "{:>18} {:>12.1} {:>12.1} {:>+10.1}",
        "J_total (dB)",
        obj_base,
        obj_opt,
        obj_opt - obj_base
    );

    // Profile
    println!("\nWall profile:");
    println!(
        "  {:>6} {:>9} {:>9} {:>8}",
        "z(mm)", "init(mm)", "opt(mm)", "delta"
    );
    for i in 0..n_ctrl {
        let z = baseline.z_positions[i] * 1e3;
        let r0 = baseline.r_values[i] * 1e3;
        let r1 = result.profile.r_values[i] * 1e3;
        println!("  {:>6.1} {:>9.2} {:>9.2} {:>+8.2}", z, r0, r1, r1 - r0);
    }

    // Shape stats
    let r_vals = &result.profile.r_values;
    let r_max_v = r_vals.iter().cloned().fold(0.0_f64, f64::max);
    let r_min_v = r_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_i = r_vals
        .iter()
        .position(|&r| (r - r_max_v).abs() < 1e-10)
        .unwrap();
    let min_i = r_vals
        .iter()
        .position(|&r| (r - r_min_v).abs() < 1e-10)
        .unwrap();

    println!(
        "\n  Max r: {:.1} mm at z={:.1} mm (expansion {:.2}x)",
        r_max_v * 1e3,
        baseline.z_positions[max_i] * 1e3,
        r_max_v / r_pipe
    );
    println!(
        "  Min r: {:.1} mm at z={:.1} mm (contraction {:.2}x)",
        r_min_v * 1e3,
        baseline.z_positions[min_i] * 1e3,
        r_min_v / r_pipe
    );
    println!(
        "  Volume change: {:+.0}%",
        (vol_opt / v_straight - 1.0) * 100.0
    );
}
