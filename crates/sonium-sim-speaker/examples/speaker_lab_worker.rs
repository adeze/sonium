//! Run a validated Speaker Lab project in an isolated native process.

// Rust guideline compliant 2026-02-21
#[expect(
    dead_code,
    reason = "shared browser module includes playback helpers unused by the exporter"
)]
#[path = "../../../../apps/speaker-demo/src/data.rs"]
mod data;

#[expect(
    dead_code,
    reason = "shared browser module includes edit helpers unused by the worker"
)]
#[path = "../../../../apps/speaker-demo/src/project.rs"]
mod project;
use data::{Dataset, Design, Mesh};
use num_complex::Complex64;
use project::{Backend, Project};
use sonium_sim_speaker::mesh_gen::{MeshRegion, generate_loudspeaker_mesh};
use sonium_sim_speaker::mlg_loader::load_model;
use sonium_sim_speaker::model::TransferPoint;
use sonium_sim_speaker::simulation::{SimulationControl, run_simulation_with_control};
use sonium_sim_speaker::{ExecutionProfile, SolverConfig};
use sonium_sim_speaker_modeling::SpeakerDriverStudy;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::var_os("SONIUM_JOB_PARENT").is_some() {
        // Parent pipe EOF also stops jobs if the server crashes or is killed.
        std::thread::spawn(|| {
            let mut input = std::io::stdin().lock();
            let _ = std::io::copy(&mut input, &mut std::io::sink());
            std::process::exit(130);
        });
    }
    let args: Vec<_> = std::env::args().collect();
    let input = args
        .get(1)
        .ok_or("usage: speaker_lab_worker INPUT.json OUTPUT.json")?;
    let destination = args.get(2).ok_or("missing output path")?;
    let project: Project = serde_json::from_slice(&std::fs::read(input)?)?;
    project.validate()?;
    #[cfg(not(feature = "cuda"))]
    if project.backend == Backend::Cuda && !args.iter().any(|a| a == "--validate" || a == "--mesh")
    {
        return Err("This worker was built without CUDA".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(project.cpu_threads)
        .build_global()?;
    let mut model = load_model(include_str!("two_way_speaker.mlg"), "two_way_speaker.mlg")?;
    model.cabinet.height = project.cabinet_m[0];
    model.separator.y_position = 0.05 * project.cabinet_m[0];
    model.separator.thickness = 0.018_f64.min(0.1 * project.cabinet_m[0]);
    model.cabinet.width = project.cabinet_m[1];
    model.cabinet.depth = project.cabinet_m[2];
    model.woofer.radius = project.driver_radius_m[0];
    model.tweeter.radius = project.driver_radius_m[1];
    // Preserve the validated conical profile while scaling it with the aperture.
    if let Some(cone) = &mut model.woofer.cone {
        cone.cone_depth = project.driver_radius_m[0] * (0.008 / 0.065);
        cone.dustcap_radius = project.driver_radius_m[0] * (0.015 / 0.065);
    }
    if let Some(cone) = &mut model.tweeter.cone {
        cone.cone_depth = project.driver_radius_m[1] * (0.003 / 0.013);
        cone.dustcap_radius = project.driver_radius_m[1] * (0.005 / 0.013);
    }
    // Typed T/S handoff is a rigid piston: use its exact projected aperture
    // instead of the prototype cone proxy in the radiation power balance.
    if project.linear.mode != "prescribed" {
        model.woofer.cone = None;
        model.tweeter.cone = None;
    }
    model.simulation.freq_start = project.frequency_start_hz;
    model.simulation.freq_end = project.frequency_end_hz;
    model.simulation.freq_points = project.frequency_points;
    model.simulation.max_mesh_frequency = Some(if project.linear.mode != "prescribed" {
        project.frequency_end_hz.max(600.0)
    } else {
        project.frequency_end_hz
    });
    let frequencies = model.simulation.frequencies();
    for (design, centres) in project.driver_y_m.into_iter().enumerate() {
        let mut candidate = model.clone();
        candidate.woofer.y_offset = centres[0];
        candidate.tweeter.y_offset = centres[1];
        if project.linear.mode != "prescribed" {
            configure_linear(&mut candidate, &project, design, &frequencies)?;
        }
        candidate.validate()?;
    }
    if args.iter().any(|a| a == "--validate") {
        return Ok(());
    }
    if args.iter().any(|a| a == "--mesh") {
        let mut meshes = Vec::new();
        for (design, centres) in project.driver_y_m.into_iter().enumerate() {
            let mut candidate = model.clone();
            candidate.woofer.y_offset = centres[0];
            candidate.tweeter.y_offset = centres[1];
            add_lumped_port_aperture(&mut candidate, &project, design, &frequencies)?;
            let mesh = generate_loudspeaker_mesh(&candidate);
            if mesh.triangles.len() > 2000 {
                return Err("Preview mesh exceeds 2000 triangles".into());
            }
            meshes.push(display_mesh(mesh));
        }
        std::fs::write(destination, serde_json::to_vec(&meshes)?)?;
        return Ok(());
    }
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
                project.slice_distance_m[0]
                    + (project.slice_distance_m[1] - project.slice_distance_m[0]) * z as f64
                        / (SIDE - 1) as f64,
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
    let directivity_angles: Vec<f64> = (-18..=18).map(|i| i as f64 * 5.0).collect();
    for angle in &directivity_angles {
        let theta = angle.to_radians();
        receivers.push([
            0.0,
            project.receiver_distance_m * theta.sin(),
            front + project.receiver_distance_m * theta.cos(),
        ]);
    }
    receivers.push([0.0, 0.0, front + project.receiver_distance_m]);
    let mut designs = Vec::new();
    let mut pressure_range: f64 = 0.0;
    let corrected_frequencies = std::cell::Cell::new(0_usize);
    for improved in [false, true] {
        let mut candidate = model.clone();
        let centres = project.driver_y_m[usize::from(improved)];
        candidate.woofer.y_offset = centres[0];
        candidate.tweeter.y_offset = centres[1];
        if project.linear.mode != "prescribed" {
            configure_linear(
                &mut candidate,
                &project,
                usize::from(improved),
                &frequencies,
            )?;
        } else {
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
                        let s = Complex64::new(0.0, -f / project.crossover_hz);
                        let h = if driver_index == 0 {
                            1.0 / (1.0 + s)
                        } else {
                            s / (1.0 + s)
                        };
                        TransferPoint {
                            freq: f,
                            magnitude: project.piston_velocity_peak_m_s * gain * h.norm(),
                            phase_deg: h.arg().to_degrees(),
                        }
                    })
                    .collect();
                candidate.drivers_mut()[driver_index].transfer_function = Some(points);
            }
        }
        let lumped_models: Vec<_> = if project.linear.mode == "lumped" {
            let mut models = Vec::new();
            for driver in candidate.drivers_mut() {
                let handoff = driver.driver_model.take().ok_or("Missing lumped handoff")?;
                driver.transfer_function = Some(
                    frequencies
                        .iter()
                        .map(|&f| {
                            let v = handoff.velocity_at(f).conj() * std::f64::consts::SQRT_2;
                            TransferPoint {
                                freq: f,
                                magnitude: v.norm(),
                                phase_deg: v.arg().to_degrees(),
                            }
                        })
                        .collect(),
                );
                models.push(handoff);
            }
            models
        } else {
            Vec::new()
        };
        add_lumped_port_aperture(
            &mut candidate,
            &project,
            usize::from(improved),
            &frequencies,
        )?;
        let mesh = generate_loudspeaker_mesh(&candidate);
        normalize_port_flow(&mut candidate, &mesh, &project, usize::from(improved))?;
        eprintln!(
            "{}: {} triangles, {} frequencies",
            if improved { "Candidate" } else { "Baseline" },
            mesh.triangles.len(),
            frequencies.len()
        );
        candidate.validate()?;
        // Keep native preview jobs within a predictable dense-matrix memory budget.
        if mesh.triangles.len() > 2000 {
            return Err(
                "Preview mesh exceeds 2000 triangles; reduce cabinet size or sweep endpoint".into(),
            );
        }
        let control = SimulationControl::new(frequencies.len());
        let progress_control = control.clone();
        let finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progress_finished = std::sync::Arc::clone(&finished);
        let design_index = usize::from(improved);
        let progress = std::thread::spawn(move || {
            while !progress_finished.load(std::sync::atomic::Ordering::Acquire) {
                let p = progress_control.progress();
                println!(
                    "{}",
                    serde_json::json!({"design":design_index,"completed":p.completed,"total":p.total,"frequency_hz":p.frequency_hz})
                );
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
        });
        let config = SolverConfig {
            execution_profile: ExecutionProfile::Preview,
            max_frequency_lanes: Some(project.cpu_threads.min(4)),
            // Deep directivity nulls amplify residual error; retain the original
            // physical parity gate and tighten the solve instead.
            tolerance: 1e-9,
            gpu_reference_correction: false,
            hmatrix_aca_tolerance: 1e-10,
            ..Default::default()
        };
        let outcome = match project.backend {
            Backend::Cpu => {
                run_simulation_with_control(&candidate, &receivers, &config, Some(&control))
            }
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                sonium_sim_speaker::simulation::run_simulation_cuda_on_device_with_control(
                    &candidate,
                    &receivers,
                    &config,
                    0,
                    Some(&control),
                )
                .map(|(result, report)| {
                    corrected_frequencies.set(
                        corrected_frequencies.get()
                            + report.validation.corrected_frequency_indices.len(),
                    );
                    println!(
                        "{}",
                        serde_json::json!({"design":design_index,"gpu_report":report})
                    );
                    result
                })
            }
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("validated before launching"),
        };
        finished.store(true, std::sync::atomic::Ordering::Release);
        progress.join().map_err(|_| "progress reporter failed")?;
        let mut result = outcome?;
        if project.linear.mode == "lumped" {
            // Prescribed CUDA solves omit network diagnostics. Retain the exact
            // peak BEM velocities used for these explicit lumped sources.
            result.driver_velocities_bem = frequencies
                .iter()
                .map(|&f| candidate.velocities_at(f))
                .collect();
        }
        let study = SpeakerDriverStudy::detailed_frequency_domain_driver();

        let reference_impedance = frequencies
            .iter()
            .map(|&f| {
                let z = study.try_small_signal_impedance(f)?;
                Ok([z.re, z.im])
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let impedance = if project.linear.mode == "lumped" {
            frequencies
                .iter()
                .map(|&f| {
                    let z = lumped_models[0].electrical_impedance_at(f);
                    [z.re, z.im]
                })
                .collect()
        } else if project.linear.mode != "prescribed" {
            result
                .driver_electrical_impedances_ohm
                .iter()
                .map(|row| {
                    row[0]
                        .map(|z| [z.re, z.im])
                        .ok_or("Missing loaded woofer impedance")
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            reference_impedance
        };
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
        if project.linear.mode == "lumped" {
            result.driver_electrical_impedances_ohm = frequencies
                .iter()
                .map(|&f| {
                    lumped_models
                        .iter()
                        .map(|m| Some(m.electrical_impedance_at(f)))
                        .collect()
                })
                .collect();
        }
        designs.push(Design {
            name: if improved { "Candidate" } else { "Baseline" }.into(),
            description: if project.linear.mode != "prescribed" {
                format!(
                    "{:.0} mm spacing · {:.3} V RMS · {} rear load · {} active crossover",
                    (centres[0] - centres[1]).abs() * 1000.0,
                    project.linear.voltage_rms_v,
                    project.linear.enclosure[design_index],
                    project.linear.crossover
                )
            } else {
                format!(
                    "{:.0} mm spacing · {:.0} Hz crossover · {:.3} mm/s peak prescribed woofer",
                    (centres[0] - centres[1]).abs() * 1000.0,
                    project.crossover_hz,
                    project.piston_velocity_peak_m_s * 1000.0
                )
            },
            mesh: display_mesh(mesh),
            impedance_label: if project.linear.mode != "prescribed" {
                if project.linear.mode == "lumped" {
                    "Woofer impedance · lumped rear + baffled piston loading"
                } else {
                    "Loaded woofer terminal impedance · BEM self/mutual load"
                }
            } else {
                "Independent reference driver"
            }
            .into(),
            tweeter_impedance_ohm: result
                .driver_electrical_impedances_ohm
                .iter()
                .map(|r| r[1].map(|z| [z.re, z.im]))
                .collect::<Option<Vec<_>>>()
                .unwrap_or_default(),
            driver_velocity_peak_m_s: if project.linear.mode != "prescribed" {
                result
                    .driver_velocities_bem
                    .iter()
                    .map(|r| r.iter().map(|v| v.norm()).collect())
                    .collect()
            } else {
                Vec::new()
            },
            pressure_rms,
            on_axis_spl_db: (0..frequencies.len())
                .map(|fi| {
                    20.0 * (result.complex_pressure[[fi, receivers.len() - 1]].norm()
                        / std::f64::consts::SQRT_2
                        / 20e-6)
                        .max(1e-30)
                        .log10()
                })
                .collect(),
            impedance_ohm: impedance,
            directivity_spl_db: (0..frequencies.len())
                .map(|fi| {
                    result.complex_pressure.row(fi)
                        [slice.positions.len()..slice.positions.len() + directivity_angles.len()]
                        .iter()
                        .map(|p| {
                            20.0 * (p.norm() / std::f64::consts::SQRT_2 / 20e-6)
                                .max(1e-30)
                                .log10()
                        })
                        .collect()
                })
                .collect(),
        });
    }
    let dataset = Dataset {
        schema_version: 1,
        provenance: format!(
            "Native {:?} f64 BEM preview; mesh sized at {:.0} Hz (lambda/8); rigid cabinet; prescribed piston motion with area-compensated tweeter gain; exp(-iωt); RMS pressure. On-axis receiver {:.3} m. Impedance is an independent reference driver, unaffected by cabinet edits. No measured validation or structural breakup claim. CUDA reference operator corrections: {} frequency/design points.",
            project.backend,
            project.frequency_end_hz,
            project.receiver_distance_m,
            corrected_frequencies.get()
        ),
        directivity_angles_deg: directivity_angles,
        frequencies_hz: frequencies,
        slice,
        pressure_range_pa: pressure_range,
        designs,
    };
    let mut dataset = dataset;
    if project.linear.mode != "prescribed" {
        dataset.provenance = format!(
            "Native {:?} f64 exterior BEM with loaded small-signal Re/Le/BL/Mms/Cms/Rms drivers, {} ideal active crossover, {:.3} V RMS; woofer rear load {} / {}, tweeter sealed {:.3} litres. Driver impedances include exterior self/mutual loading. Voltage is RMS; pressure is RMS exp(-iωt). No physical measurement validation. CUDA corrections: {}.",
            project.backend,
            project.linear.crossover,
            project.linear.voltage_rms_v,
            project.linear.enclosure[0],
            project.linear.enclosure[1],
            project.linear.tweeter_volume_litres,
            corrected_frequencies.get()
        );
    }
    if project.linear.mode == "lumped" {
        dataset.provenance = format!(
            "Explicit lumped voltage-driven model: rear enclosure load and analytic baffled-piston front impedance; {} active crossover; {:.3} V RMS. Exterior BEM computes cabinet diffraction from prescribed network velocities. Mutual exterior feedback and interior modes are not included. Vented studies prescribe the lumped port flow on a meshed aperture in the same exterior BEM as the drivers; exterior feedback into the lumped network and internal port flow are not resolved. Pressure is RMS exp(-iωt). No measured validation.",
            project.linear.crossover, project.linear.voltage_rms_v
        );
    }
    dataset.validate()?;
    if let Some(parent) = std::path::Path::new(&destination).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(destination, serde_json::to_vec(&dataset)?)?;
    Ok(())
}

fn display_mesh(mesh: sonium_sim_speaker::mesh_gen::SpeakerMesh) -> Mesh {
    Mesh {
        positions: mesh.nodes,
        triangles: mesh
            .triangles
            .iter()
            .map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
            .collect(),
        regions: mesh
            .region
            .iter()
            .map(|r| match r {
                MeshRegion::CabinetWall => 0.0,
                MeshRegion::WooferAperture => 1.0,
                MeshRegion::TweeterAperture => 2.0,
                _ => 3.0,
            })
            .collect(),
    }
}

fn configure_linear(
    model: &mut sonium_sim_speaker::model::LoudspeakerModel,
    p: &Project,
    design: usize,
    frequencies: &[f64],
) -> Result<(), Box<dyn Error>> {
    use sonium_sim_speaker_modeling::driver_model::DriveTransferPoint;
    use sonium_sim_speaker_modeling::product_design::{CrossoverBranch, CrossoverKind};
    use sonium_sim_speaker_modeling::{DriverModel, EnclosureSpec, PortSpec};
    for i in 0..2 {
        let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
        let d = p.linear.drivers[i];
        study.geometry.diaphragm_radius_m = p.driver_radius_m[i];
        study.geometry.voice_coil_radius_m = study
            .geometry
            .voice_coil_radius_m
            .min(p.driver_radius_m[i] * 0.6);
        study.voice_coil.resistance_ohm = d[0];
        study.voice_coil.inductance_h = d[1];
        study.voice_coil.bl_tm = d[2];
        study.moving_system.moving_mass_kg = d[3];
        study.moving_system.stiffness_n_per_m = 1.0 / d[4];
        study.moving_system.mechanical_resistance_kg_per_s = d[5];
        study.drive_voltage_rms_v = p.linear.voltage_rms_v;
        study.frequency_hz = frequencies.to_vec();
        study.acoustic_medium.density_kg_per_m3 = model.simulation.air_density;
        study.acoustic_medium.sound_speed_m_per_s = model.simulation.speed_of_sound;
        let sealed = |v| EnclosureSpec::Sealed {
            volume_m3: v / 1000.0,
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        };
        study.enclosure = if i == 1 {
            sealed(p.linear.tweeter_volume_litres)
        } else {
            match p.linear.enclosure[design].as_str() {
                "free_air" => EnclosureSpec::FreeAir,
                "sealed" => sealed(p.linear.volume_litres[design]),
                _ => EnclosureSpec::Vented {
                    volume_m3: p.linear.volume_litres[design] / 1000.0,
                    port: PortSpec {
                        radius_m: p.linear.port[0],
                        length_m: p.linear.port[1],
                        end_correction_m: p.linear.port[2],
                        resistance_pa_s_per_m3: p.linear.port[3],
                    },
                    leakage_resistance_pa_s_per_m3: 0.0,
                    absorption_resistance_pa_s_per_m3: 0.0,
                },
            }
        };
        let kind = match (p.linear.crossover.as_str(), i) {
            ("lr2", 0) => CrossoverKind::LinkwitzRiley2LowPass,
            ("lr2", _) => CrossoverKind::LinkwitzRiley2HighPass,
            ("lr4", 0) => CrossoverKind::LinkwitzRiley4LowPass,
            ("lr4", _) => CrossoverKind::LinkwitzRiley4HighPass,
            (_, 0) => CrossoverKind::LowPass,
            (_, _) => CrossoverKind::HighPass,
        };
        let branch = CrossoverBranch {
            kind,
            corner_frequency_hz: p.crossover_hz,
            gain: p.linear.voltage_gain[i],
            delay_s: if i == 1 {
                p.linear.tweeter_delay_ms / 1000.0
            } else {
                0.0
            },
        };
        let mut handoff = DriverModel::from_study(&study)?;
        handoff.drive_transfer = frequencies
            .iter()
            .map(|&f| {
                branch.transfer_at(f).map(|h| DriveTransferPoint {
                    frequency_hz: f,
                    real: h.re,
                    imag: h.im,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        handoff.validate()?;
        model.drivers_mut()[i].transfer_function = None;
        model.drivers_mut()[i].driver_model = Some(handoff);
    }
    Ok(())
}

/// Represent the lumped port flow on an actual aperture in the same exterior
/// operator as the diaphragms, so their spreading and phase agree.
fn add_lumped_port_aperture(
    model: &mut sonium_sim_speaker::model::LoudspeakerModel,
    p: &Project,
    design: usize,
    frequencies: &[f64],
) -> Result<(), Box<dyn Error>> {
    use sonium_sim_speaker_modeling::{AcousticMedium, EnclosureSpec, PortSpec};
    if p.linear.mode != "lumped" || p.linear.enclosure[design] != "vented" {
        return Ok(());
    }
    let medium = AcousticMedium {
        density_kg_per_m3: model.simulation.air_density,
        sound_speed_m_per_s: model.simulation.speed_of_sound,
        ..AcousticMedium::standard_air()
    };
    let port = PortSpec {
        radius_m: p.linear.port[0],
        length_m: p.linear.port[1],
        end_correction_m: p.linear.port[2],
        resistance_pa_s_per_m3: p.linear.port[3],
    };
    let enclosure = EnclosureSpec::Vented {
        volume_m3: p.linear.volume_litres[design] / 1000.0,
        port,
        leakage_resistance_pa_s_per_m3: 0.0,
        absorption_resistance_pa_s_per_m3: 0.0,
    };
    let mut source = model.woofer.clone();
    source.radius = port.radius_m;
    source.x_offset = 0.0;
    source.y_offset = p.linear.port[4];
    source.cone = None;
    source.driver_model = None;
    source.piston_velocity = 0.0;
    source.transfer_function = Some(
        frequencies
            .iter()
            .map(|&f| {
                // Front-positive diaphragm motion increases box volume: outward port
                // flow opposes diaphragm flow below tuning. Both use BEM exp(-iwt).
                let velocity = -enclosure.acoustic_impedance(f, medium)?.conj()
                    / port.acoustic_impedance(f, medium)?.conj()
                    * (p.driver_radius_m[0] / port.radius_m).powi(2)
                    * model.velocities_at(f)[0];
                Ok(TransferPoint {
                    freq: f,
                    magnitude: velocity.norm(),
                    phase_deg: velocity.arg().to_degrees(),
                })
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?,
    );
    model.additional_drivers.push(source);
    Ok(())
}

/// Conserve flow using the actual polygonal aperture areas, including the
/// low-frequency cancellation between the woofer and port.
fn normalize_port_flow(
    model: &mut sonium_sim_speaker::model::LoudspeakerModel,
    mesh: &sonium_sim_speaker::mesh_gen::SpeakerMesh,
    p: &Project,
    design: usize,
) -> Result<(), Box<dyn Error>> {
    if p.linear.mode != "lumped" || p.linear.enclosure[design] != "vented" {
        return Ok(());
    }
    let aperture_area = |region| {
        mesh.triangles
            .iter()
            .zip(&mesh.region)
            .filter(|(_, r)| **r == region)
            .map(|(t, _)| {
                let [a, b, c] = t.map(|i| mesh.nodes[i]);
                let u = std::array::from_fn::<_, 3, _>(|j| b[j] - a[j]);
                let v = std::array::from_fn::<_, 3, _>(|j| c[j] - a[j]);
                ((u[0] * v[1] - u[1] * v[0]).powi(2)
                    + (u[1] * v[2] - u[2] * v[1]).powi(2)
                    + (u[2] * v[0] - u[0] * v[2]).powi(2))
                .sqrt()
                    / 2.0
            })
            .sum::<f64>()
    };
    let woofer = aperture_area(MeshRegion::WooferAperture);
    let port = aperture_area(MeshRegion::DriverAperture(2));
    if woofer <= 0.0 || port <= 0.0 {
        return Err("Missing woofer or port aperture area".into());
    }
    let scale = woofer / port * (p.linear.port[0] / p.driver_radius_m[0]).powi(2);
    for point in model.additional_drivers[0]
        .transfer_function
        .as_mut()
        .ok_or("Missing port drive")?
    {
        point.magnitude *= scale;
        let mach = point.magnitude / model.simulation.speed_of_sound;
        if mach > 0.05 {
            return Err(format!("Port peak Mach {mach:.3} at {:.1} Hz exceeds the supported linear limit 0.05; reduce voltage", point.freq).into());
        }
    }
    Ok(())
}
