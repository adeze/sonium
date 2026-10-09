//! Export native BEM fields for the GPUI browser prototype.

// Rust guideline compliant 2026-02-21
#[expect(
    dead_code,
    reason = "shared browser module includes playback helpers unused by the exporter"
)]
#[path = "../../../apps/speaker-demo/src/data.rs"]
mod data;

use data::{Dataset, Design, Mesh};
use num_complex::Complex64;
use sonium_sim_speaker::mesh_gen::{MeshRegion, generate_loudspeaker_mesh};
use sonium_sim_speaker::mlg_loader::load_model;
use sonium_sim_speaker::model::TransferPoint;
use sonium_sim_speaker::simulation::try_run_simulation_with_config;
use sonium_sim_speaker::{ExecutionProfile, SolverConfig};
use sonium_sim_speaker_modeling::SpeakerDriverStudy;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let destination = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "apps/speaker-demo/assets/study.json".into());
    let mut model = load_model(include_str!("two_way_speaker.mlg"), "two_way_speaker.mlg")?;
    // A bounded 1.2 kHz study permits a properly resolved mesh and quick replay.
    model.simulation.freq_start = 200.0;
    model.simulation.freq_end = 1200.0;
    model.simulation.freq_points = 12;
    model.simulation.max_mesh_frequency = Some(1200.0);
    let frequencies = model.simulation.frequencies();
    let mut slice = Mesh {
        positions: Vec::new(),
        triangles: Vec::new(),
        regions: Vec::new(),
    };
    // Vertical section: 24 forward-distance samples by 24 height samples.
    const SIDE: usize = 24;
    for y in 0..SIDE {
        for z in 0..SIDE {
            slice.positions.push([
                0.03 + 0.9 * z as f64 / (SIDE - 1) as f64,
                -0.4 + 0.8 * y as f64 / (SIDE - 1) as f64,
                0.0,
            ]);
        }
    }
    for y in 0..SIDE - 1 {
        for z in 0..SIDE - 1 {
            let i = (y * SIDE + z) as u32;
            slice.triangles.extend([
                [i, i + 1, i + SIDE as u32],
                [i + 1, i + SIDE as u32 + 1, i + SIDE as u32],
            ]);
            slice.regions.extend([0.0, 0.0]);
        }
    }
    let front = model.baffle_center()[2];
    let mut receivers: Vec<_> = slice
        .positions
        .iter()
        .map(|p| [0.0, p[1], front + p[0]])
        .collect();
    receivers.push([0.0, 0.0, front + 1.0]);
    let mut designs = Vec::new();
    let mut pressure_range: f64 = 0.0;
    for improved in [false, true] {
        let mut candidate = model.clone();
        if improved {
            candidate.woofer.y_offset = -0.065;
            candidate.tweeter.y_offset = 0.075;
        }
        // First-order complementary crossover, expressed in the BEM convention.
        for driver_index in 0..2 {
            let radius = candidate.drivers()[driver_index].radius;
            let gain = if driver_index == 0 {
                1.0
            } else {
                (candidate.woofer.radius / radius).powi(2)
            };
            let points = frequencies
                .iter()
                .map(|&f| {
                    let s = Complex64::new(0.0, -f / 650.0);
                    let h = if driver_index == 0 {
                        1.0 / (1.0 + s)
                    } else {
                        s / (1.0 + s)
                    };
                    TransferPoint {
                        freq: f,
                        magnitude: 0.001 * gain * h.norm(),
                        phase_deg: h.arg().to_degrees(),
                    }
                })
                .collect();
            candidate.drivers_mut()[driver_index].transfer_function = Some(points);
        }
        let mesh = generate_loudspeaker_mesh(&candidate);
        eprintln!(
            "{}: {} triangles, {} frequencies",
            if improved {
                "Improved candidate"
            } else {
                "Baseline"
            },
            mesh.triangles.len(),
            frequencies.len()
        );
        let result = try_run_simulation_with_config(
            &candidate,
            &receivers,
            &SolverConfig {
                execution_profile: ExecutionProfile::Preview,
                max_frequency_lanes: Some(2),
                ..Default::default()
            },
        )?;
        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        if improved {
            study.moving_system.stiffness_n_per_m *= 1.05;
        }
        let impedance = frequencies
            .iter()
            .map(|&f| {
                let z = study.try_small_signal_impedance(f)?;
                Ok([z.re, z.im])
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let pressure_rms = (0..frequencies.len())
            .map(|fi| {
                result.complex_pressure.row(fi)[..slice.positions.len()]
                    .iter()
                    .map(|p| {
                        pressure_range = pressure_range.max(p.norm());
                        [
                            p.re / std::f64::consts::SQRT_2,
                            p.im / std::f64::consts::SQRT_2,
                        ]
                    })
                    .collect()
            })
            .collect();
        designs.push(Design { impedance_label: "Independent reference driver".into(), tweeter_impedance_ohm: Vec::new(), driver_velocity_peak_m_s:Vec::new(),
            name: if improved { "Improved candidate" } else { "Baseline" }.into(),
            description: if improved { "140 mm driver spacing; woofer stiffness +5%. Candidate for comparison, awaiting measured validation." }
                else { "180 mm driver spacing; first-order 650 Hz crossover; prescribed piston motion." }.into(),
            mesh: Mesh { positions: mesh.nodes, triangles: mesh.triangles.iter().map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect(),
                regions: mesh.region.iter().map(|r| match r { MeshRegion::CabinetWall => 0.0, MeshRegion::WooferAperture => 1.0, MeshRegion::TweeterAperture => 2.0, _ => 3.0 }).collect() },
            pressure_rms,
            on_axis_spl_db: (0..frequencies.len()).map(|fi| {
                20.0 * (result.complex_pressure[[fi, receivers.len() - 1]].norm() / std::f64::consts::SQRT_2 / 20e-6).max(1e-30).log10()
            }).collect(),
            impedance_ohm: impedance,
            directivity_spl_db: Vec::new(),
        });
    }
    let dataset = Dataset { schema_version: 1,
        provenance: "Native CPU f64 BEM preview; mesh sized at 1.2 kHz (lambda/8); prescribed piston velocities; exp(-iωt); cached RMS pressure. Impedance from the independent analytical lumped driver model. No measured validation or structural breakup claim.".into(),
        directivity_angles_deg: Vec::new(), frequencies_hz: frequencies, slice, pressure_range_pa: pressure_range, designs };
    dataset.validate()?;
    if let Some(parent) = std::path::Path::new(&destination).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(destination, serde_json::to_vec(&dataset)?)?;
    Ok(())
}
