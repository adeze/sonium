//! CI-cheap M4 port-noise analytical and finite-record gates.

use std::f64::consts::TAU;
use std::process::ExitCode;

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::spaces::H1Space;
use sonium_sim_speaker_modeling::port_flow::coupled::PortFlowFieldSnapshot;
use sonium_sim_speaker_modeling::{
    AcousticMedium, PortNoiseMethod, PortNoiseObserver, PortNoiseSpec, analyze_port_noise,
    free_turbulence_power_trend,
};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let trend = free_turbulence_power_trend(4.0, 0.1)? / free_turbulence_power_trend(2.0, 0.1)?;
    if (trend - 256.0).abs() > 1.0e-12 {
        return Err("U^8/L^3 trend gate failed".into());
    }

    let mesh = unit_cube_tetrahedra(1);
    let velocity_space = H1Space::new(PolynomialDegree::P1);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let dt = 1.0 / 6_400.0;
    let snapshots = (0..64)
        .map(|sample| {
            let velocity = 2.0 * (TAU * 100.0 * sample as f64 * dt).cos();
            PortFlowFieldSnapshot {
                time_s: (sample + 1) as f64 * dt,
                velocity_dofs_m_per_s: (0..mesh.num_nodes())
                    .flat_map(|_| [velocity, 0.0, 0.0])
                    .collect(),
                pressure_dofs_pa: vec![0.0; mesh.num_nodes()],
            }
        })
        .collect::<Vec<_>>();
    let result = analyze_port_noise(
        &mesh,
        &velocity_space,
        &pressure_space,
        &snapshots,
        AcousticMedium::standard_air(),
        &PortNoiseSpec {
            method: PortNoiseMethod::CompactLighthill,
            frequencies_hz: vec![200.0],
            observers: vec![
                PortNoiseObserver {
                    position_m: [10.0, 0.5, 0.5],
                },
                PortNoiseObserver {
                    position_m: [20.0, 0.5, 0.5],
                },
            ],
            characteristic_source_length_m: 0.01,
            ..PortNoiseSpec::default()
        },
    )?;
    let ratio = result.observers[0].spectrum[0].pressure_rms_pa
        / result.observers[1].spectrum[0].pressure_rms_pa;
    if (ratio - 2.0).abs() > 0.06 {
        return Err(format!("inverse-distance gate failed: ratio={ratio}").into());
    }
    println!("SUCCESS: M4 compact Lighthill scaling and spreading gates passed");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("FAILED: {error}");
            ExitCode::FAILURE
        }
    }
}
