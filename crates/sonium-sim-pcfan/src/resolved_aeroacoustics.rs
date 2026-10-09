//! Checked resolved acoustic FEM assembly for the PC-fan workflow.
//!
//! The CFD mesh used by the PC-fan model is a structured hexahedral domain,
//! while the FEM aeroacoustics module consumes [`sonium_fem::mesh::Mesh`].
//! This module is the explicit bridge between those representations.  It
//! assembles the Linearized Euler operator and a harmonic continuity source
//! over the annular rotor volume; it does not silently substitute the reduced
//! pressure model or claim that a complex acoustic solve was performed.

use crate::evaluation::propeller_source_for;
use crate::model::{PcFanCfdDomain, PcFanConfig, PcFanDesignVariables, PcFanFlowSolution};
use num_complex::Complex64;
use sonium_fem::assembly::{TripletMatrix, assemble_linear_checked};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::acoustics::aeroacoustics::{
    LeeSystem, MeanFlow, build_lee_system_uniform,
};
use sonium_fem::physics::acoustics::pressure::ComplexSourceForm;
use sonium_fem::spaces::H1Space;
use std::fmt;

/// Failure while constructing the resolved PC-fan acoustic FEM problem.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanResolvedAeroacousticAssemblyError {
    /// Harmonic indices are one-based because harmonic one is the BPF.
    InvalidHarmonic(usize),
    /// A required physical or geometric quantity was non-finite or invalid.
    InvalidInput(&'static str),
    /// The PC-fan geometry could not be admitted.
    InvalidGeometry(String),
    /// The converted FEM mesh failed connectivity validation.
    InvalidMesh(String),
    /// The generic FEM assembly rejected the validated space or mesh.
    FemAssembly(String),
}

impl fmt::Display for PcFanResolvedAeroacousticAssemblyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHarmonic(harmonic) => {
                write!(formatter, "invalid PC-fan acoustic harmonic {harmonic}")
            }
            Self::InvalidInput(name) => {
                write!(formatter, "invalid PC-fan acoustic assembly input {name}")
            }
            Self::InvalidGeometry(error) => {
                write!(formatter, "invalid PC-fan acoustic geometry: {error}")
            }
            Self::InvalidMesh(error) => {
                write!(formatter, "invalid PC-fan acoustic FEM mesh: {error}")
            }
            Self::FemAssembly(error) => {
                write!(formatter, "PC-fan acoustic FEM assembly failed: {error}")
            }
        }
    }
}

impl std::error::Error for PcFanResolvedAeroacousticAssemblyError {}

/// A resolved PC-fan harmonic Linearized Euler problem before boundary
/// conditions and a complex solve are applied.
///
/// The right-hand side uses the FEM LEE ordering: vector velocity DOFs first,
/// followed by scalar pressure DOFs.  The source is a harmonic volumetric
/// continuity source normalized so its volume integral is `iω p_peak`, where
/// `p_peak` is the existing [`PropellerSource`] annular-inlet pressure
/// amplitude.  This makes the source normalization explicit and testable.
pub struct PcFanResolvedAeroacousticPdeAssembly {
    /// FEM mesh converted from the PC-fan CFD domain.
    pub mesh: Mesh,
    /// Uniform mean flow used by the current LEE assembly.
    pub mean_flow: MeanFlow,
    /// Real-valued FEM blocks assembled from the mean flow.
    pub lee_system: LeeSystem,
    /// Complex harmonic LEE operator.
    pub operator: TripletMatrix<Complex64>,
    /// Complex harmonic RHS in LEE block ordering.
    pub rhs: Vec<Complex64>,
    /// One-based BPF harmonic index.
    pub harmonic: usize,
    /// Harmonic frequency in Hz.
    pub frequency_hz: f64,
    /// Annular source pressure amplitude used for normalization.
    pub source_pressure_amplitude_pa: f64,
    /// Annular source support volume in cubic metres.
    pub source_support_volume_m3: f64,
    /// Provenance for downstream solver/adjointhood checks.
    pub provenance: String,
}

impl PcFanResolvedAeroacousticPdeAssembly {
    /// Number of velocity DOFs in the assembled LEE system.
    pub fn velocity_dof_count(&self) -> usize {
        self.lee_system.n_velocity
    }

    /// Number of pressure DOFs in the assembled LEE system.
    pub fn pressure_dof_count(&self) -> usize {
        self.lee_system.n_pressure
    }

    /// Number of complex operator rows/columns.
    pub fn total_dof_count(&self) -> usize {
        self.lee_system.total_dofs()
    }
}

/// Assemble a checked resolved PC-fan Linearized Euler harmonic problem.
///
/// This is an assembly boundary, not a simulation runner: no acoustic linear
/// solve, QA executable, or full transient simulation is launched here.
pub fn assemble_pc_fan_resolved_aeroacoustic_pde(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    flow: &PcFanFlowSolution,
    harmonic: usize,
) -> Result<PcFanResolvedAeroacousticPdeAssembly, PcFanResolvedAeroacousticAssemblyError> {
    if harmonic == 0 {
        return Err(PcFanResolvedAeroacousticAssemblyError::InvalidHarmonic(
            harmonic,
        ));
    }
    if !flow.axial_velocity_m_s.is_finite()
        || !flow.pressure_rise_pa.is_finite()
        || !flow.airflow_cfm.is_finite()
    {
        return Err(PcFanResolvedAeroacousticAssemblyError::InvalidInput(
            "flow observables",
        ));
    }

    let geometry = config.try_build_geometry(design).map_err(|error| {
        PcFanResolvedAeroacousticAssemblyError::InvalidGeometry(format!("{error:?}"))
    })?;
    let domain = geometry.build_cfd_domain(config);
    let mesh = fem_mesh_from_cfd_domain(&domain)?;

    let hub_radius_m = design.effective_hub_radius_m(config);
    let tip_radius_m = design.effective_tip_radius_m(config);
    let rotor_depth_m = geometry.frame_box.depth_m;
    let source_support_volume_m3 = std::f64::consts::PI
        * (tip_radius_m * tip_radius_m - hub_radius_m * hub_radius_m)
        * rotor_depth_m;
    if !source_support_volume_m3.is_finite() || source_support_volume_m3 <= 0.0 {
        return Err(PcFanResolvedAeroacousticAssemblyError::InvalidInput(
            "rotor source support volume",
        ));
    }

    let source = propeller_source_for(config, design, flow);
    let frequency_hz = source.harmonic_frequency(harmonic);
    let source_pressure_amplitude_pa = source.inlet_pressure_amplitude(harmonic);
    let omega = std::f64::consts::TAU * frequency_hz;
    if !frequency_hz.is_finite()
        || frequency_hz <= 0.0
        || !source_pressure_amplitude_pa.is_finite()
        || source_pressure_amplitude_pa < 0.0
    {
        return Err(PcFanResolvedAeroacousticAssemblyError::InvalidInput(
            "harmonic source",
        ));
    }

    let mean_flow = MeanFlow::air_standard([flow.axial_velocity_m_s, 0.0, 0.0]);
    if !mean_flow.density.is_finite()
        || mean_flow.density <= 0.0
        || !mean_flow.speed_of_sound.is_finite()
        || mean_flow.speed_of_sound <= 0.0
    {
        return Err(PcFanResolvedAeroacousticAssemblyError::InvalidInput(
            "mean flow",
        ));
    }
    let lee_system = build_lee_system_uniform(
        &mesh,
        PolynomialDegree::P2,
        PolynomialDegree::P1,
        &mean_flow,
        3,
    );
    let operator = lee_system.assemble(&mean_flow, omega);

    let source_z0 = -0.5 * rotor_depth_m;
    let source_z1 = 0.5 * rotor_depth_m;
    let source_density_pa_per_s_per_m3 =
        omega * source_pressure_amplitude_pa / source_support_volume_m3;
    let source_form = ComplexSourceForm::new(move |x, y, z| {
        let radius_squared = x * x + y * y;
        let in_rotor_slab = z >= source_z0 && z <= source_z1;
        let in_rotor_annulus = radius_squared >= hub_radius_m * hub_radius_m
            && radius_squared <= tip_radius_m * tip_radius_m;
        if in_rotor_slab && in_rotor_annulus {
            Complex64::new(0.0, source_density_pa_per_s_per_m3)
        } else {
            Complex64::new(0.0, 0.0)
        }
    });
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let pressure_rhs =
        assemble_linear_checked(&source_form, &pressure_space, &mesh, 2).map_err(|error| {
            PcFanResolvedAeroacousticAssemblyError::FemAssembly(format!("{error:?}"))
        })?;
    let mut rhs = vec![Complex64::new(0.0, 0.0); lee_system.n_velocity];
    rhs.extend(pressure_rhs);
    if rhs.len() != lee_system.total_dofs() || rhs.iter().any(|value| !complex_is_finite(*value)) {
        return Err(PcFanResolvedAeroacousticAssemblyError::FemAssembly(
            "assembled RHS has invalid dimensions or values".into(),
        ));
    }

    Ok(PcFanResolvedAeroacousticPdeAssembly {
        mesh,
        mean_flow,
        lee_system,
        operator,
        rhs,
        harmonic,
        frequency_hz,
        source_pressure_amplitude_pa,
        source_support_volume_m3,
        provenance: "fem:pcfan:linearized-euler:annular-harmonic-source:v1".into(),
    })
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn fem_mesh_from_cfd_domain(
    domain: &PcFanCfdDomain,
) -> Result<Mesh, PcFanResolvedAeroacousticAssemblyError> {
    let mut mesh = Mesh::new(3);
    for point in &domain.nodes {
        if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
            return Err(PcFanResolvedAeroacousticAssemblyError::InvalidMesh(
                "non-finite CFD-domain node".into(),
            ));
        }
        mesh.add_node(Point::new_3d(point.x, point.y, point.z));
    }
    for cell in &domain.cells {
        if !cell.volume_m3.is_finite() || cell.volume_m3 <= 0.0 {
            return Err(PcFanResolvedAeroacousticAssemblyError::InvalidMesh(
                "non-positive or non-finite cell volume".into(),
            ));
        }
        mesh.add_element(ElementType::Hexahedron, cell.node_indices.to_vec());
    }
    mesh.validate_connectivity().map_err(|errors| {
        PcFanResolvedAeroacousticAssemblyError::InvalidMesh(format!("{errors:?}"))
    })?;
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{PcFanCfdConfig, PcFanFlowModel, PcFanMeshResolution};

    fn small_fixture() -> (PcFanConfig, PcFanDesignVariables, PcFanFlowSolution) {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 3,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 3,
                cells_y: 3,
                cells_z: 3,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::bench_120mm()
        };
        let design = config.default_design_variables();
        let flow = PcFanFlowSolution {
            airflow_cfm: 20.0,
            axial_velocity_m_s: 2.0,
            pressure_rise_pa: 25.0,
            reynolds_number: 10_000.0,
            mach_number: 0.01,
            disk_solidity: 0.2,
            fixed_hardware: crate::model::PcFanHardwareFlowEffects {
                inlet_open_area_m2: 0.01,
                blockage_area_m2: 0.001,
                blockage_ratio: 0.1,
                motor_blockage_ratio: 0.05,
                strut_blockage_ratio: 0.05,
                frame_lip_loss_factor: 1.0,
                support_wake_loss_factor: 1.0,
                total_loss_factor: 1.0,
            },
        };
        (config, design, flow)
    }

    #[test]
    fn assembly_converts_domain_and_normalizes_annular_source() {
        let (config, design, flow) = small_fixture();
        let assembly = assemble_pc_fan_resolved_aeroacoustic_pde(&config, &design, &flow, 1)
            .expect("small resolved acoustic assembly should succeed");

        assert_eq!(assembly.harmonic, 1);
        assert_eq!(assembly.mesh.elements.len(), 27);
        assert_eq!(
            assembly.velocity_dof_count() + assembly.pressure_dof_count(),
            assembly.rhs.len()
        );
        assert_eq!(assembly.operator.num_rows, assembly.total_dof_count());
        assert_eq!(assembly.operator.num_cols, assembly.total_dof_count());
        assert!(assembly.source_pressure_amplitude_pa.is_finite());
        assert!(assembly.source_support_volume_m3 > 0.0);
        assert!(assembly.rhs.iter().any(|value| value.im.abs() > 0.0));
        assert!(assembly.rhs.iter().all(|value| complex_is_finite(*value)));
        assert_eq!(
            assembly.provenance,
            "fem:pcfan:linearized-euler:annular-harmonic-source:v1"
        );
    }

    #[test]
    fn assembly_rejects_zero_harmonic_and_nonfinite_flow() {
        let (config, design, mut flow) = small_fixture();
        assert!(matches!(
            assemble_pc_fan_resolved_aeroacoustic_pde(&config, &design, &flow, 0),
            Err(PcFanResolvedAeroacousticAssemblyError::InvalidHarmonic(0))
        ));
        flow.pressure_rise_pa = f64::NAN;
        assert!(matches!(
            assemble_pc_fan_resolved_aeroacoustic_pde(&config, &design, &flow, 1),
            Err(PcFanResolvedAeroacousticAssemblyError::InvalidInput(
                "flow observables"
            ))
        ));
    }

    #[test]
    fn converted_mesh_rejects_invalid_cell_volume() {
        let (config, design, flow) = small_fixture();
        let geometry = config.build_geometry(&design);
        let mut domain = geometry.build_cfd_domain(&config);
        domain.cells[0].volume_m3 = 0.0;
        assert!(matches!(
            fem_mesh_from_cfd_domain(&domain),
            Err(PcFanResolvedAeroacousticAssemblyError::InvalidMesh(_))
        ));
        let _ = flow;
    }
}
