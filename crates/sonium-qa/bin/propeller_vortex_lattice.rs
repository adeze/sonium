//! Bounded 3-D horseshoe-vortex lattice panel-flow reference QA.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    BladePanel, Point3, VortexLatticeOptions, evaluate_vortex_lattice_reference,
};

const CASE_ID: &str = "propeller.vortex-lattice-reference.v1";

fn panel(root_x: f64, tip_x: f64, chord_m: f64) -> BladePanel {
    BladePanel {
        vertices: [
            Point3 {
                x: root_x,
                y: chord_m * 0.5,
                z: 0.0,
            },
            Point3 {
                x: tip_x,
                y: chord_m * 0.5,
                z: 0.0,
            },
            Point3 {
                x: tip_x,
                y: -chord_m * 0.5,
                z: 0.0,
            },
            Point3 {
                x: root_x,
                y: -chord_m * 0.5,
                z: 0.0,
            },
        ],
        blade_index: 0,
        radius_fraction: 0.5,
        chord_fraction: chord_m,
        area: (tip_x - root_x) * chord_m,
    }
}

fn run() -> Result<(f64, f64, f64, f64), String> {
    let panels = [panel(0.04, 0.065, 0.018), panel(0.065, 0.09, 0.014)];
    let options = VortexLatticeOptions {
        freestream_m_s: Point3 {
            x: 0.0,
            y: 0.0,
            z: 6.0,
        },
        angular_velocity_rad_s: 180.0,
        wake_length_m: 0.8,
        wake_segments: 6,
        ..VortexLatticeOptions::default()
    };
    let result =
        evaluate_vortex_lattice_reference(&panels, options).map_err(|error| error.to_string())?;
    if !result.is_finite() || result.panels.len() != panels.len() {
        return Err("vortex-lattice result is not finite or has wrong panel count".into());
    }
    if result.maximum_normal_residual_m_s > options.residual_tolerance_m_s {
        return Err(format!(
            "normal residual {} exceeds {}",
            result.maximum_normal_residual_m_s, options.residual_tolerance_m_s
        ));
    }
    Ok((
        result.thrust_n,
        result.torque_nm,
        result.maximum_normal_residual_m_s,
        result.maximum_induced_velocity_m_s,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((thrust, torque, residual, induced)) => {
            let result = QaResult::passed(CASE_ID)
                .metric("thrust_n", thrust)
                .metric("torque_nm", torque)
                .metric("maximum_normal_residual_m_s", residual)
                .metric("maximum_induced_velocity_m_s", induced)
                .maximum("maximum_normal_residual_m_s", 1.0e-7)
                .input_checksum("analytic/deterministic-fixture:propeller-vortex-lattice-v1")
                .reference_checksum("analytic/invariant:propeller-vortex-lattice-v1")
                .dimension("two quadrilateral panels; six-segment finite wake; rotating f64 CPU")
                .tolerance("normal residual <= 1e-7 m/s; all loads finite")
                .seed("propeller-vortex-lattice-v1:panels=2:wake=6")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
