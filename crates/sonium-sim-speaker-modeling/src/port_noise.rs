//! M4 one-way aeroacoustic post-processing of M1 port-flow snapshots.
//!
//! Compact sources use the existing Lighthill quadrupole evaluator. A closed
//! fluid boundary uses the stationary permeable FW-H evaluator (3-D patches,
//! or the 2-D contour evaluator for slot meshes, which reports the
//! infinite-line field) when `ka` exceeds the configured compactness limit.
//! Snapshot-to-phasor coupling goes through the shared `snapshot` helper so
//! the DFT convention stays identical across consumers. No flow solve or
//! baffle-scattering correction occurs here.

// Rust guideline compliant 2026-02-21

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{ElementType, Mesh};
use sonium_fem::physics::acoustics::aeroacoustics::{
    LighthillQuadrupole, dft_peak_phasor, fwh_contour_from_series, fwh_patch_from_series,
    lighthill_far_field_pressure, stationary_fwh_pressure, stationary_fwh_pressure_2d,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};

use crate::port_flow::coupled::PortFlowFieldSnapshot;
use crate::port_flow::measure::outward_normal_and_measure;
use crate::{AcousticMedium, ModelingError, require_positive};

pub const PORT_NOISE_SCHEMA_VERSION: u32 = 1;

/// Selection policy for the two spec-defined aeroacoustic routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortNoiseMethod {
    /// Use Lighthill while compact and FW-H otherwise.
    Automatic,
    CompactLighthill,
    PermeableFwh,
}

/// Far-field observer outside the source domain.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PortNoiseObserver {
    pub position_m: [f64; 3],
}

/// M4 analysis controls. Frequencies are finite-record DFT evaluation points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortNoiseSpec {
    pub method: PortNoiseMethod,
    pub frequencies_hz: Vec<f64>,
    pub observers: Vec<PortNoiseObserver>,
    pub characteristic_source_length_m: f64,
    pub compact_ka_max: f64,
    /// Extrusion depth for the CI-cheap 2D slot case.
    pub out_of_plane_depth_m: f64,
    pub reference_pressure_pa: f64,
}

impl Default for PortNoiseSpec {
    fn default() -> Self {
        Self {
            method: PortNoiseMethod::Automatic,
            frequencies_hz: vec![100.0, 200.0, 400.0, 800.0],
            observers: vec![PortNoiseObserver {
                position_m: [1.0, 0.0, 0.0],
            }],
            characteristic_source_length_m: 0.1,
            compact_ka_max: 0.5,
            out_of_plane_depth_m: 0.02,
            reference_pressure_pa: 20.0e-6,
        }
    }
}

impl PortNoiseSpec {
    pub fn validate(&self, sample_interval_s: f64) -> Result<(), ModelingError> {
        require_positive("port_noise.sample_interval_s", sample_interval_s)?;
        require_positive(
            "port_noise.characteristic_source_length_m",
            self.characteristic_source_length_m,
        )?;
        require_positive("port_noise.compact_ka_max", self.compact_ka_max)?;
        require_positive("port_noise.out_of_plane_depth_m", self.out_of_plane_depth_m)?;
        require_positive(
            "port_noise.reference_pressure_pa",
            self.reference_pressure_pa,
        )?;
        if self.frequencies_hz.is_empty()
            || self.frequencies_hz.iter().any(|frequency| {
                !frequency.is_finite() || *frequency <= 0.0 || *frequency >= 0.5 / sample_interval_s
            })
            || self
                .frequencies_hz
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(ModelingError::OutOfRange {
                field: "port_noise.frequencies_hz",
                constraint: "finite, positive, strictly increasing, and below Nyquist",
            });
        }
        if self.observers.is_empty()
            || self
                .observers
                .iter()
                .any(|observer| observer.position_m.iter().any(|value| !value.is_finite()))
        {
            return Err(ModelingError::OutOfRange {
                field: "port_noise.observers",
                constraint: "non-empty finite positions",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PortNoiseSpectrumPoint {
    pub frequency_hz: f64,
    pub method: PortNoiseMethod,
    pub compactness_ka: f64,
    /// Peak phasor using the aeroacoustic modules' `exp(i omega t)` convention.
    pub pressure_peak_pa: [f64; 2],
    pub pressure_rms_pa: f64,
    pub spl_db: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortNoiseObserverResult {
    pub observer: PortNoiseObserver,
    pub spectrum: Vec<PortNoiseSpectrumPoint>,
    pub integrated_noise_floor_db: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortNoiseResult {
    pub schema_version: u32,
    pub provenance: String,
    pub sample_interval_s: f64,
    pub sample_count: usize,
    pub observers: Vec<PortNoiseObserverResult>,
}

/// Versioned production handoff from an M1 field-export job to M4.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortNoiseInput {
    pub schema_version: u32,
    pub mesh: Mesh,
    pub velocity_polynomial_degree: usize,
    pub pressure_polynomial_degree: usize,
    pub snapshots: Vec<PortFlowFieldSnapshot>,
    pub medium: AcousticMedium,
    pub specification: PortNoiseSpec,
}

pub fn analyze_port_noise_input(input: &PortNoiseInput) -> Result<PortNoiseResult, ModelingError> {
    if input.schema_version != PORT_NOISE_SCHEMA_VERSION {
        return Err(ModelingError::OutOfRange {
            field: "port_noise.schema_version",
            constraint: "version 1",
        });
    }
    let degree = |value, field| match value {
        1 => Ok(PolynomialDegree::P1),
        2 => Ok(PolynomialDegree::P2),
        _ => Err(ModelingError::OutOfRange {
            field,
            constraint: "1 or 2",
        }),
    };
    let velocity_space = H1Space::new(degree(
        input.velocity_polynomial_degree,
        "port_noise.velocity_polynomial_degree",
    )?);
    let pressure_space = H1Space::new(degree(
        input.pressure_polynomial_degree,
        "port_noise.pressure_polynomial_degree",
    )?);
    analyze_port_noise(
        &input.mesh,
        &velocity_space,
        &pressure_space,
        &input.snapshots,
        input.medium,
        &input.specification,
    )
}

/// Compare two matched spectra for a grid/time-step convergence gate.
pub fn maximum_relative_spectrum_change(
    coarse: &PortNoiseResult,
    refined: &PortNoiseResult,
) -> Result<f64, ModelingError> {
    if coarse.observers.len() != refined.observers.len()
        || coarse
            .observers
            .iter()
            .zip(&refined.observers)
            .any(|(left, right)| left.spectrum.len() != right.spectrum.len())
    {
        return Err(ModelingError::OutOfRange {
            field: "port_noise.convergence",
            constraint: "matched observer and frequency grids",
        });
    }
    Ok(coarse
        .observers
        .iter()
        .zip(&refined.observers)
        .flat_map(|(left, right)| left.spectrum.iter().zip(&right.spectrum))
        .map(|(left, right)| {
            (left.pressure_rms_pa - right.pressure_rms_pa).abs()
                / right.pressure_rms_pa.max(f64::MIN_POSITIVE)
        })
        .fold(0.0, f64::max))
}

/// Analytical free-turbulence trend requested by the M4 validation plan.
/// This is a relative trend metric, not an absolute acoustic-power model.
pub fn free_turbulence_power_trend(
    velocity_m_per_s: f64,
    length_m: f64,
) -> Result<f64, ModelingError> {
    require_positive("port_noise.velocity_m_per_s", velocity_m_per_s)?;
    require_positive("port_noise.length_m", length_m)?;
    Ok(velocity_m_per_s.powi(8) / length_m.powi(3))
}

/// Transform accepted M1 fields and evaluate the selected free-field analogy.
pub fn analyze_port_noise(
    mesh: &Mesh,
    velocity_space: &H1Space,
    pressure_space: &H1Space,
    snapshots: &[PortFlowFieldSnapshot],
    medium: AcousticMedium,
    spec: &PortNoiseSpec,
) -> Result<PortNoiseResult, ModelingError> {
    medium.validate()?;
    if snapshots.len() < 4 {
        return Err(ModelingError::OutOfRange {
            field: "port_noise.snapshots",
            constraint: "at least four equally spaced accepted fields",
        });
    }
    let sample_interval_s = snapshots[1].time_s - snapshots[0].time_s;
    spec.validate(sample_interval_s)?;
    validate_snapshots(
        mesh,
        velocity_space,
        pressure_space,
        snapshots,
        sample_interval_s,
    )?;

    let source_series = lighthill_tensor_series(
        mesh,
        velocity_space,
        snapshots,
        medium.density_kg_per_m3,
        spec.out_of_plane_depth_m,
    )?;
    let source_centroid = mesh_centroid(mesh);
    let mut observers = Vec::with_capacity(spec.observers.len());
    for &observer in &spec.observers {
        let mut spectrum = Vec::with_capacity(spec.frequencies_hz.len());
        for &frequency_hz in &spec.frequencies_hz {
            let compactness_ka =
                std::f64::consts::TAU * frequency_hz * spec.characteristic_source_length_m
                    / medium.sound_speed_m_per_s;
            let method = match spec.method {
                PortNoiseMethod::Automatic if compactness_ka <= spec.compact_ka_max => {
                    PortNoiseMethod::CompactLighthill
                }
                PortNoiseMethod::Automatic => PortNoiseMethod::PermeableFwh,
                selected => selected,
            };
            let pressure = match method {
                PortNoiseMethod::CompactLighthill => lighthill_at_frequency(
                    source_centroid,
                    &source_series,
                    observer.position_m,
                    sample_interval_s,
                    frequency_hz,
                    medium.sound_speed_m_per_s,
                )?,
                PortNoiseMethod::PermeableFwh if mesh.dimension == 3 => fwh_at_frequency(
                    mesh,
                    snapshots,
                    observer.position_m,
                    sample_interval_s,
                    frequency_hz,
                    medium,
                )?,
                // Slot meshes report the 2-D (infinite-line) FW-H field.
                PortNoiseMethod::PermeableFwh => fwh_2d_at_frequency(
                    mesh,
                    snapshots,
                    [observer.position_m[0], observer.position_m[1]],
                    sample_interval_s,
                    frequency_hz,
                    medium,
                )?,
                PortNoiseMethod::Automatic => unreachable!(),
            };
            let pressure_rms_pa = pressure.norm() / std::f64::consts::SQRT_2;
            let spl_db = 20.0
                * (pressure_rms_pa.max(f64::MIN_POSITIVE) / spec.reference_pressure_pa).log10();
            spectrum.push(PortNoiseSpectrumPoint {
                frequency_hz,
                method,
                compactness_ka,
                pressure_peak_pa: [pressure.re, pressure.im],
                pressure_rms_pa,
                spl_db,
            });
        }
        let integrated_rms = spectrum
            .iter()
            .map(|point| point.pressure_rms_pa.powi(2))
            .sum::<f64>()
            .sqrt();
        observers.push(PortNoiseObserverResult {
            observer,
            spectrum,
            integrated_noise_floor_db: 20.0
                * (integrated_rms.max(f64::MIN_POSITIVE) / spec.reference_pressure_pa).log10(),
        });
    }
    Ok(PortNoiseResult {
        schema_version: PORT_NOISE_SCHEMA_VERSION,
        provenance: "M1UnsteadyFields->Lighthill/FWH->FreeFieldObserver".into(),
        sample_interval_s,
        sample_count: snapshots.len(),
        observers,
    })
}

fn validate_snapshots(
    mesh: &Mesh,
    velocity_space: &H1Space,
    pressure_space: &H1Space,
    snapshots: &[PortFlowFieldSnapshot],
    dt: f64,
) -> Result<(), ModelingError> {
    let n_v = velocity_space.num_global_dofs(mesh) * mesh.dimension;
    let n_p = pressure_space.num_global_dofs(mesh);
    for (index, snapshot) in snapshots.iter().enumerate() {
        if !snapshot.time_s.is_finite()
            || snapshot.velocity_dofs_m_per_s.len() != n_v
            || snapshot.pressure_dofs_pa.len() != n_p
            || snapshot
                .velocity_dofs_m_per_s
                .iter()
                .chain(&snapshot.pressure_dofs_pa)
                .any(|value| !value.is_finite())
            || (index > 0
                && ((snapshot.time_s - snapshots[index - 1].time_s) - dt).abs() > 1.0e-9 * dt)
        {
            return Err(ModelingError::OutOfRange {
                field: "port_noise.snapshots",
                constraint: "finite, dimensionally matched, equally spaced fields",
            });
        }
    }
    Ok(())
}

fn lighthill_tensor_series(
    mesh: &Mesh,
    velocity_space: &H1Space,
    snapshots: &[PortFlowFieldSnapshot],
    density: f64,
    depth: f64,
) -> Result<Vec<[[f64; 3]; 3]>, ModelingError> {
    let mut series = Vec::with_capacity(snapshots.len());
    for snapshot in snapshots {
        let mut tensor = [[0.0; 3]; 3];
        for element_index in 0..mesh.num_elements() {
            let (xi, eta, zeta) = match mesh.elements[element_index].element_type {
                ElementType::Triangle => (1.0 / 3.0, 1.0 / 3.0, 0.0),
                ElementType::Tetrahedron => (0.25, 0.25, 0.25),
                _ => {
                    return Err(ModelingError::OutOfRange {
                        field: "port_noise.mesh",
                        constraint: "triangle or tetrahedron cells",
                    });
                }
            };
            let tab = velocity_space.tabulate(mesh, element_index, xi, eta, zeta);
            let dofs = velocity_space.local_to_global(mesh, element_index);
            let mut velocity = [0.0; 3];
            for (local, &dof) in dofs.iter().enumerate() {
                for (component, slot) in velocity[..mesh.dimension].iter_mut().enumerate() {
                    *slot += tab.value(local)
                        * snapshot.velocity_dofs_m_per_s[dof * mesh.dimension + component];
                }
            }
            let volume = cell_volume(mesh, element_index, depth)?;
            for i in 0..3 {
                for j in 0..3 {
                    tensor[i][j] += density * velocity[i] * velocity[j] * volume;
                }
            }
        }
        series.push(tensor);
    }
    Ok(series)
}

fn lighthill_at_frequency(
    source_position: [f64; 3],
    series: &[[[f64; 3]; 3]],
    observer: [f64; 3],
    dt: f64,
    frequency_hz: f64,
    speed_of_sound: f64,
) -> Result<Complex64, ModelingError> {
    let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let signal = series.iter().map(|sample| sample[i][j]).collect::<Vec<_>>();
            tensor[i][j] = dft_peak_phasor(&signal, dt, frequency_hz).map_err(aero_error)?;
        }
    }
    let source = LighthillQuadrupole::new(source_position, tensor).map_err(aero_error)?;
    lighthill_far_field_pressure(
        &source,
        observer,
        std::f64::consts::TAU * frequency_hz,
        speed_of_sound,
    )
    .map_err(aero_error)
}

fn fwh_at_frequency(
    mesh: &Mesh,
    snapshots: &[PortFlowFieldSnapshot],
    observer: [f64; 3],
    dt: f64,
    frequency_hz: f64,
    medium: AcousticMedium,
) -> Result<Complex64, ModelingError> {
    let mut patches = Vec::with_capacity(mesh.boundaries.len());
    for face in &mesh.boundaries {
        let (normal, area) = outward_normal_and_measure(mesh, face);
        if area <= 0.0 {
            continue;
        }
        let mut pressure_signal = Vec::with_capacity(snapshots.len());
        let mut velocity_signal = Vec::with_capacity(snapshots.len());
        for snapshot in snapshots {
            let inv = 1.0 / face.nodes.len() as f64;
            let pressure = face
                .nodes
                .iter()
                .map(|&node| snapshot.pressure_dofs_pa[node])
                .sum::<f64>()
                * inv;
            let normal_velocity = face
                .nodes
                .iter()
                .map(|&node| {
                    (0..3)
                        .map(|component| {
                            snapshot.velocity_dofs_m_per_s[node * mesh.dimension + component]
                                * normal[component]
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
                * inv;
            pressure_signal.push(pressure);
            velocity_signal.push(normal_velocity);
        }
        let position = face_centroid(mesh, &face.nodes);
        patches.push(
            fwh_patch_from_series(
                position,
                normal,
                area,
                &pressure_signal,
                &velocity_signal,
                dt,
                frequency_hz,
            )
            .map_err(aero_error)?,
        );
    }
    stationary_fwh_pressure(
        &patches,
        observer,
        std::f64::consts::TAU * frequency_hz,
        medium.density_kg_per_m3,
        medium.sound_speed_m_per_s,
    )
    .map(|components| components.total)
    .map_err(aero_error)
}

fn fwh_2d_at_frequency(
    mesh: &Mesh,
    snapshots: &[PortFlowFieldSnapshot],
    observer: [f64; 2],
    dt: f64,
    frequency_hz: f64,
    medium: AcousticMedium,
) -> Result<Complex64, ModelingError> {
    let mut segments = Vec::with_capacity(mesh.boundaries.len());
    for face in &mesh.boundaries {
        let (normal, length) = outward_normal_and_measure(mesh, face);
        if length <= 0.0 {
            continue;
        }
        let mut pressure_signal = Vec::with_capacity(snapshots.len());
        let mut velocity_signal = Vec::with_capacity(snapshots.len());
        for snapshot in snapshots {
            let inv = 1.0 / face.nodes.len() as f64;
            let pressure = face
                .nodes
                .iter()
                .map(|&node| snapshot.pressure_dofs_pa[node])
                .sum::<f64>()
                * inv;
            let normal_velocity = face
                .nodes
                .iter()
                .map(|&node| {
                    (0..mesh.dimension)
                        .map(|component| {
                            snapshot.velocity_dofs_m_per_s[node * mesh.dimension + component]
                                * normal[component]
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
                * inv;
            pressure_signal.push(pressure);
            velocity_signal.push(normal_velocity);
        }
        let centroid = face_centroid(mesh, &face.nodes);
        segments.push(
            fwh_contour_from_series(
                [centroid[0], centroid[1]],
                [normal[0], normal[1]],
                length,
                &pressure_signal,
                &velocity_signal,
                dt,
                frequency_hz,
            )
            .map_err(aero_error)?,
        );
    }
    stationary_fwh_pressure_2d(
        &segments,
        observer,
        std::f64::consts::TAU * frequency_hz,
        medium.density_kg_per_m3,
        medium.sound_speed_m_per_s,
    )
    .map(|components| components.total)
    .map_err(aero_error)
}

fn cell_volume(mesh: &Mesh, element_index: usize, depth: f64) -> Result<f64, ModelingError> {
    let vertices = mesh.elements[element_index].vertices();
    let at = |index: usize| &mesh.nodes[vertices[index]];
    let volume = match mesh.elements[element_index].element_type {
        ElementType::Triangle => {
            let (a, b, c) = (at(0), at(1), at(2));
            0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs() * depth
        }
        ElementType::Tetrahedron => {
            let (a, b, c, d) = (at(0), at(1), at(2), at(3));
            let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
            let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
            let ad = [d.x - a.x, d.y - a.y, d.z - a.z];
            let cross = [
                ac[1] * ad[2] - ac[2] * ad[1],
                ac[2] * ad[0] - ac[0] * ad[2],
                ac[0] * ad[1] - ac[1] * ad[0],
            ];
            (ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]).abs() / 6.0
        }
        _ => 0.0,
    };
    require_positive("port_noise.cell_volume_m3", volume)?;
    Ok(volume)
}

fn mesh_centroid(mesh: &Mesh) -> [f64; 3] {
    let inv = 1.0 / mesh.nodes.len().max(1) as f64;
    let mut centroid = [0.0; 3];
    for point in &mesh.nodes {
        centroid[0] += point.x * inv;
        centroid[1] += point.y * inv;
        centroid[2] += point.z * inv;
    }
    centroid
}

fn face_centroid(mesh: &Mesh, nodes: &[usize]) -> [f64; 3] {
    let inv = 1.0 / nodes.len() as f64;
    let mut centroid = [0.0; 3];
    for &node in nodes {
        centroid[0] += mesh.nodes[node].x * inv;
        centroid[1] += mesh.nodes[node].y * inv;
        centroid[2] += mesh.nodes[node].z * inv;
    }
    centroid
}

fn aero_error(error: impl std::fmt::Display) -> ModelingError {
    ModelingError::ExternalAeroacoustics(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port_flow::analysis::dft_phasor;
    use sonium_fem::mesh::{unit_cube_tetrahedra, unit_square_triangles};
    use std::f64::consts::TAU;

    fn synthetic_snapshots(
        mesh: &Mesh,
        samples: usize,
        dt: f64,
        frequency: f64,
        amplitude: f64,
    ) -> Vec<PortFlowFieldSnapshot> {
        (0..samples)
            .map(|index| {
                let velocity = amplitude * (TAU * frequency * index as f64 * dt).cos();
                PortFlowFieldSnapshot {
                    time_s: (index + 1) as f64 * dt,
                    velocity_dofs_m_per_s: (0..mesh.num_nodes())
                        .flat_map(|_| [velocity, 0.0, 0.0])
                        .collect(),
                    pressure_dofs_pa: vec![0.0; mesh.num_nodes()],
                }
            })
            .collect()
    }

    fn synthetic_snapshots_2d(
        mesh: &Mesh,
        samples: usize,
        dt: f64,
        frequency: f64,
        amplitude: f64,
    ) -> Vec<PortFlowFieldSnapshot> {
        (0..samples)
            .map(|index| {
                let velocity = amplitude * (TAU * frequency * index as f64 * dt).cos();
                PortFlowFieldSnapshot {
                    time_s: (index + 1) as f64 * dt,
                    velocity_dofs_m_per_s: (0..mesh.num_nodes())
                        .flat_map(|_| [velocity, 0.0])
                        .collect(),
                    pressure_dofs_pa: vec![0.0; mesh.num_nodes()],
                }
            })
            .collect()
    }

    #[test]
    fn snapshot_dft_matches_legacy_phasor_convention() {
        // Locks the adoption: shared helper equals dft_phasor().conj().
        let signal: Vec<f64> = (0..64)
            .map(|index| 2.0 * (TAU * 100.0 * index as f64 / 6_400.0 + 0.3).cos())
            .collect();
        let shared = dft_peak_phasor(&signal, 1.0 / 6_400.0, 100.0).unwrap();
        let legacy = dft_phasor(&signal, 1.0 / 6_400.0, 100.0).conj();
        assert!((shared - legacy).norm() < 1e-12);
    }

    #[test]
    fn slot_mesh_uses_planar_fwh_contour_with_cylindrical_spreading() {
        let mesh = unit_square_triangles(2);
        let velocity_space = H1Space::new(PolynomialDegree::P1);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let snapshots = synthetic_snapshots_2d(&mesh, 64, 1.0 / 6_400.0, 100.0, 1.0);
        let spec = PortNoiseSpec {
            method: PortNoiseMethod::PermeableFwh,
            frequencies_hz: vec![100.0],
            observers: vec![
                PortNoiseObserver {
                    position_m: [10.0, 0.5, 0.0],
                },
                PortNoiseObserver {
                    position_m: [40.0, 0.5, 0.0],
                },
            ],
            characteristic_source_length_m: 1.0,
            ..PortNoiseSpec::default()
        };
        let result = analyze_port_noise(
            &mesh,
            &velocity_space,
            &pressure_space,
            &snapshots,
            AcousticMedium::standard_air(),
            &spec,
        )
        .unwrap();
        assert_eq!(
            result.observers[0].spectrum[0].method,
            PortNoiseMethod::PermeableFwh
        );
        // 2-D far field spreads as 1/sqrt(r): 4x distance halves pressure.
        let ratio = result.observers[0].spectrum[0].pressure_rms_pa
            / result.observers[1].spectrum[0].pressure_rms_pa;
        assert!((ratio - 2.0).abs() < 0.12, "ratio={ratio}");
    }

    #[test]
    fn analytical_free_turbulence_trend_is_u8_over_l3() {
        let base = free_turbulence_power_trend(2.0, 0.1).unwrap();
        assert!((free_turbulence_power_trend(4.0, 0.1).unwrap() / base - 256.0).abs() < 1.0e-12);
        assert!((free_turbulence_power_trend(2.0, 0.2).unwrap() / base - 0.125).abs() < 1.0e-12);
    }

    #[test]
    fn compact_lighthill_pressure_has_inverse_distance_spreading() {
        let mesh = unit_cube_tetrahedra(1);
        let velocity_space = H1Space::new(PolynomialDegree::P1);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let snapshots = synthetic_snapshots(&mesh, 64, 1.0 / 6_400.0, 100.0, 2.0);
        let spec = PortNoiseSpec {
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
        };
        let result = analyze_port_noise(
            &mesh,
            &velocity_space,
            &pressure_space,
            &snapshots,
            AcousticMedium::standard_air(),
            &spec,
        )
        .unwrap();
        let ratio = result.observers[0].spectrum[0].pressure_rms_pa
            / result.observers[1].spectrum[0].pressure_rms_pa;
        assert!((ratio - 2.0).abs() < 0.06);
    }

    #[test]
    fn steady_laminar_fields_have_negligible_nonzero_bin_noise() {
        let mesh = unit_cube_tetrahedra(1);
        let velocity_space = H1Space::new(PolynomialDegree::P1);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let snapshots = synthetic_snapshots(&mesh, 64, 1.0 / 6_400.0, 100.0, 0.0);
        let result = analyze_port_noise(
            &mesh,
            &velocity_space,
            &pressure_space,
            &snapshots,
            AcousticMedium::standard_air(),
            &PortNoiseSpec {
                method: PortNoiseMethod::CompactLighthill,
                frequencies_hz: vec![200.0],
                characteristic_source_length_m: 0.01,
                ..PortNoiseSpec::default()
            },
        )
        .unwrap();
        assert!(result.observers[0].spectrum[0].pressure_rms_pa < 1.0e-20);
    }

    #[test]
    fn noncompact_automatic_route_uses_closed_3d_fwh_surface() {
        let mesh = unit_cube_tetrahedra(1);
        let velocity_space = H1Space::new(PolynomialDegree::P1);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let snapshots = synthetic_snapshots(&mesh, 64, 1.0 / 6_400.0, 100.0, 1.0);
        let result = analyze_port_noise(
            &mesh,
            &velocity_space,
            &pressure_space,
            &snapshots,
            AcousticMedium::standard_air(),
            &PortNoiseSpec {
                method: PortNoiseMethod::Automatic,
                frequencies_hz: vec![100.0],
                observers: vec![PortNoiseObserver {
                    position_m: [3.0, 0.5, 0.5],
                }],
                characteristic_source_length_m: 1.0,
                compact_ka_max: 0.5,
                ..PortNoiseSpec::default()
            },
        )
        .unwrap();
        assert_eq!(
            result.observers[0].spectrum[0].method,
            PortNoiseMethod::PermeableFwh
        );
        assert!(result.observers[0].spectrum[0].pressure_rms_pa.is_finite());
    }

    #[test]
    fn m4_versioned_m1_field_handoff_round_trips() {
        let mesh = unit_cube_tetrahedra(1);
        let input = PortNoiseInput {
            schema_version: PORT_NOISE_SCHEMA_VERSION,
            snapshots: synthetic_snapshots(&mesh, 64, 1.0 / 6_400.0, 100.0, 1.0),
            mesh,
            velocity_polynomial_degree: 1,
            pressure_polynomial_degree: 1,
            medium: AcousticMedium::standard_air(),
            specification: PortNoiseSpec {
                method: PortNoiseMethod::CompactLighthill,
                frequencies_hz: vec![200.0],
                characteristic_source_length_m: 0.01,
                ..PortNoiseSpec::default()
            },
        };
        let decoded: PortNoiseInput =
            serde_json::from_slice(&serde_json::to_vec(&input).unwrap()).unwrap();
        let result = analyze_port_noise_input(&decoded).unwrap();
        assert_eq!(result.schema_version, PORT_NOISE_SCHEMA_VERSION);
        assert_eq!(result.sample_count, 64);
        assert!(result.provenance.contains("M1UnsteadyFields"));
    }
}
