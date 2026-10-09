//! Output JSON formatting for room acoustics simulations

use crate::config::{
    BoundaryConfig, MetadataConfig, RoomConfig, RoomSimulation, VisualizationConfig,
};
use crate::conventions::ConventionRegistry;
use crate::geometry::RoomGeometry;
use crate::parallelism::{SLICE_CELLS_PER_THREAD, use_parallel};
use crate::types::{Point3D, RoomMesh, pressure_to_spl};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sonium_compute::HostMatrix;

/// Result of room simulation at one frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyResult {
    /// Frequency (Hz)
    pub frequency: f64,
    /// SPL values at each listening position
    pub spl_at_lp: Vec<f64>,
}

/// Complete simulation results
#[derive(Debug, Clone)]
pub struct SimulationResults {
    /// List of frequencies simulated
    pub frequencies: Vec<f64>,
    /// Frequency response at each listening position
    pub lp_frequency_responses: Vec<Vec<f64>>,
    /// Horizontal slice data (if generated)
    pub horizontal_slice: Option<SliceData>,
    /// Vertical slice data (if generated)
    pub vertical_slice: Option<SliceData>,
}

/// Pressure field data on a 2D slice. `spl` is row-major (shape implied by
/// `x.len()` / `y.len()`).
#[derive(Debug, Clone)]
pub struct SliceData {
    /// X coordinates
    pub x: Vec<f64>,
    /// Y or Z coordinates
    pub y: Vec<f64>,
    /// SPL grid data (row-major).
    pub spl: HostMatrix<f64>,
    /// Frequency of the slice
    pub frequency: f64,
}

/// Provenance and reproducibility metadata embedded in generated datasets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetMetadata {
    /// XEM schema version of the dataset payload.
    pub schema_version: u32,
    /// Build revision when available.
    pub code_revision: Option<String>,
    /// Target operating-system family used to generate the dataset.
    pub platform: String,
    /// Solver/backend label supplied by the producing application.
    pub solver: String,
    /// Numerical tolerances and setup parameters that affect reproducibility.
    pub tolerances: DatasetTolerances,
    /// Shared phasor, frame, normal, and amplitude conventions.
    pub conventions: ConventionRegistry,
    /// Human-readable description and author carried from the input.
    pub description: String,
    /// Author carried from the input.
    pub author: String,
    /// Creation date carried from the input configuration.
    pub date: String,
}

/// Solver settings recorded with a generated dataset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetTolerances {
    /// GMRES relative tolerance.
    pub gmres_tolerance: f64,
    /// GMRES iteration cap.
    pub gmres_max_iter: usize,
    /// FMM expansion order.
    pub fmm_expansion_order: usize,
}

impl DatasetMetadata {
    /// Construct reproducibility metadata from the validated input config.
    pub fn from_config(config: &RoomConfig, solver_name: impl Into<String>) -> Self {
        Self {
            schema_version: config.schema_version.0,
            code_revision: option_env!("GIT_HASH").map(str::to_owned),
            platform: std::env::consts::OS.to_string(),
            solver: solver_name.into(),
            tolerances: DatasetTolerances {
                gmres_tolerance: config.solver.gmres.tolerance,
                gmres_max_iter: config.solver.gmres.max_iter,
                fmm_expansion_order: config.solver.fmm.expansion_order,
            },
            conventions: config.conventions,
            description: config.metadata.description.clone(),
            author: config.metadata.author.clone(),
            date: config.metadata.date.clone(),
        }
    }
}

/// Failure to build output JSON or spatial slices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputError {
    /// `RoomSimulation` was constructed directly with no listening
    /// positions (`RoomConfig::validate` forbids this, but the struct
    /// fields are public, so library code must not index `[0]` blindly).
    NoListeningPositions,
    /// Slice resolution below 2 would divide by `res - 1 == 0` and emit
    /// NaN grids.
    SliceResolutionTooSmall { res: usize },
}

impl std::fmt::Display for OutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoListeningPositions => f.write_str("simulation has no listening positions"),
            Self::SliceResolutionTooSmall { res } => {
                write!(f, "slice resolution {res} must be at least 2")
            }
        }
    }
}

impl std::error::Error for OutputError {}

fn first_listening_position(simulation: &RoomSimulation) -> Result<Point3D, OutputError> {
    simulation
        .listening_positions
        .first()
        .copied()
        .ok_or(OutputError::NoListeningPositions)
}

/// Create output JSON without slices
pub fn create_output_json(
    simulation: &RoomSimulation,
    config: &RoomConfig,
    lp_spl_values: Vec<f64>,
    solver_name: &str,
) -> serde_json::Value {
    try_create_output_json(simulation, config, lp_spl_values, solver_name)
        .expect("create_output_json requires at least one listening position")
}

/// Fallible form of [`create_output_json`] for library callers.
pub fn try_create_output_json(
    simulation: &RoomSimulation,
    config: &RoomConfig,
    lp_spl_values: Vec<f64>,
    solver_name: &str,
) -> Result<serde_json::Value, OutputError> {
    let lp = first_listening_position(simulation)?;
    let (room_width, room_depth, room_height) = simulation.room.dimensions();
    let edges = simulation.room.get_edges();

    Ok(serde_json::json!({
        "room": {
            "type": match simulation.room {
                RoomGeometry::Rectangular(_) => "rectangular",
                RoomGeometry::LShaped(_) => "lshaped",
            },
            "width": room_width,
            "depth": room_depth,
            "height": room_height,
            "edges": edges.iter().map(|(p1, p2)| {
                vec![
                    vec![p1.x, p1.y, p1.z],
                    vec![p2.x, p2.y, p2.z],
                ]
            }).collect::<Vec<_>>(),
        },
        "sources": simulation.sources.iter().map(|s| {
            serde_json::json!({
                "name": s.name,
                "position": [s.position.x, s.position.y, s.position.z],
            })
        }).collect::<Vec<_>>(),
        "listening_position": [lp.x, lp.y, lp.z],
        "frequencies": simulation.frequencies,
        "frequency_response": lp_spl_values,
        "solver": solver_name,
        "metadata": DatasetMetadata::from_config(config, solver_name),
    }))
}

/// Create output JSON with per-source responses
pub fn create_output_json_with_sources(
    simulation: &RoomSimulation,
    config: &RoomConfig,
    lp_spl_values: Vec<f64>,
    source_spl_values: &[Vec<f64>],
    solver_name: &str,
) -> serde_json::Value {
    try_create_output_json_with_sources(
        simulation,
        config,
        lp_spl_values,
        source_spl_values,
        solver_name,
    )
    .expect("create_output_json_with_sources requires at least one listening position")
}

/// Fallible form of [`create_output_json_with_sources`] for library callers.
pub fn try_create_output_json_with_sources(
    simulation: &RoomSimulation,
    config: &RoomConfig,
    lp_spl_values: Vec<f64>,
    source_spl_values: &[Vec<f64>],
    solver_name: &str,
) -> Result<serde_json::Value, OutputError> {
    let lp = first_listening_position(simulation)?;
    let (room_width, room_depth, room_height) = simulation.room.dimensions();
    let edges = simulation.room.get_edges();

    Ok(serde_json::json!({
        "room": {
            "type": match simulation.room {
                RoomGeometry::Rectangular(_) => "rectangular",
                RoomGeometry::LShaped(_) => "lshaped",
            },
            "width": room_width,
            "depth": room_depth,
            "height": room_height,
            "edges": edges.iter().map(|(p1, p2)| {
                vec![
                    vec![p1.x, p1.y, p1.z],
                    vec![p2.x, p2.y, p2.z],
                ]
            }).collect::<Vec<_>>(),
        },
        "sources": simulation.sources.iter().map(|s| {
            serde_json::json!({
                "name": s.name,
                "position": [s.position.x, s.position.y, s.position.z],
            })
        }).collect::<Vec<_>>(),
        "listening_position": [lp.x, lp.y, lp.z],
        "frequencies": simulation.frequencies,
        "frequency_response": lp_spl_values,
        "source_responses": source_spl_values.iter().enumerate().map(|(idx, spl_vals)| {
            serde_json::json!({
                "source_name": simulation.sources[idx].name,
                "source_index": idx,
                "spl": spl_vals,
            })
        }).collect::<Vec<_>>(),
        "solver": solver_name,
        "metadata": DatasetMetadata::from_config(config, solver_name),
    }))
}

/// Trait for computing field pressure at points (implemented by BEM/FEM solvers)
pub trait FieldPressureCalculator {
    /// Compute pressure at field points given the surface solution.
    /// `surface_solution` is a flat (row-major) vector matching the mesh DOFs.
    fn calculate_field_pressure(
        &self,
        mesh: &RoomMesh,
        surface_solution: &[Complex64],
        field_points: &[Point3D],
        wavenumber: f64,
        frequency: f64,
    ) -> Vec<Complex64>;
}

/// Generate spatial slices for visualization
pub fn generate_spatial_slices<F>(
    simulation: &RoomSimulation,
    config: &VisualizationConfig,
    mesh: &RoomMesh,
    solutions: &[(f64, Vec<Complex64>)],
    calculate_pressure: F,
) -> (Vec<serde_json::Value>, Vec<serde_json::Value>)
where
    F: Fn(&RoomMesh, &[Complex64], &[Point3D], f64, f64) -> Vec<Complex64>,
{
    try_generate_spatial_slices(simulation, config, mesh, solutions, calculate_pressure)
        .expect("generate_spatial_slices requires a listening position and resolution >= 2")
}

/// Fallible form of [`generate_spatial_slices`] for library callers.
/// Rejects an empty listener list and resolutions below 2 (which would
/// divide by `res - 1 == 0` and emit NaN grids).
pub fn try_generate_spatial_slices<F>(
    simulation: &RoomSimulation,
    config: &VisualizationConfig,
    mesh: &RoomMesh,
    solutions: &[(f64, Vec<Complex64>)],
    calculate_pressure: F,
) -> Result<(Vec<serde_json::Value>, Vec<serde_json::Value>), OutputError>
where
    F: Fn(&RoomMesh, &[Complex64], &[Point3D], f64, f64) -> Vec<Complex64>,
{
    let lp = first_listening_position(simulation)?;
    let (room_width, room_depth, room_height) = simulation.room.dimensions();

    let res = config.slice_resolution;
    if res < 2 {
        return Err(OutputError::SliceResolutionTooSmall { res });
    }
    let grids = SliceGrids::new(room_width, room_depth, room_height, res, [lp.x, lp.y, lp.z]);
    let wavenumbers: Vec<f64> = solutions
        .iter()
        .map(|(freq, _)| simulation.wavenumber(*freq))
        .collect();

    let mut horizontal_slices = Vec::new();
    let mut vertical_slices = Vec::new();

    for (index, (freq, surface_pressure)) in solutions.iter().enumerate() {
        let (h_slice, v_slice) = slice_pair(
            mesh,
            &grids,
            &calculate_pressure,
            surface_pressure.as_slice(),
            wavenumbers[index],
            *freq,
        );
        horizontal_slices.push(h_slice);
        vertical_slices.push(v_slice);
    }

    Ok((horizontal_slices, vertical_slices))
}

/// Precomputed slice lattice shared by the serial and parallel slice paths.
struct SliceGrids {
    res: usize,
    x_points: Vec<f64>,
    y_points: Vec<f64>,
    z_points: Vec<f64>,
    listener: [f64; 3],
}

impl SliceGrids {
    fn new(
        room_width: f64,
        room_depth: f64,
        room_height: f64,
        res: usize,
        listener: [f64; 3],
    ) -> Self {
        debug_assert!(res >= 2, "slice grids require res >= 2");
        let axis = |span: f64| {
            (0..res)
                .map(|i| i as f64 * span / (res - 1) as f64)
                .collect()
        };
        Self {
            res,
            x_points: axis(room_width),
            y_points: axis(room_depth),
            z_points: axis(room_height),
            listener,
        }
    }

    fn horizontal_points(&self) -> Vec<Point3D> {
        let mut points = Vec::with_capacity(self.res * self.res);
        for &y in &self.y_points {
            for &x in &self.x_points {
                points.push(Point3D::new(x, y, self.listener[2]));
            }
        }
        points
    }

    fn vertical_points(&self) -> Vec<Point3D> {
        let mut points = Vec::with_capacity(self.res * self.res);
        for &z in &self.z_points {
            for &x in &self.x_points {
                points.push(Point3D::new(x, self.listener[1], z));
            }
        }
        points
    }
}

/// Compute one solution's horizontal (XY at LP.z) and vertical (XZ at LP.y)
/// slice JSON pair. Single evaluation core behind both
/// [`try_generate_spatial_slices`] and [`try_generate_spatial_slices_par`].
fn slice_pair<F>(
    mesh: &RoomMesh,
    grids: &SliceGrids,
    calculate_pressure: &F,
    surface_pressure: &[Complex64],
    wavenumber: f64,
    frequency: f64,
) -> (serde_json::Value, serde_json::Value)
where
    F: Fn(&RoomMesh, &[Complex64], &[Point3D], f64, f64) -> Vec<Complex64>,
{
    let res = grids.res;
    let h_pressures = calculate_pressure(
        mesh,
        surface_pressure,
        &grids.horizontal_points(),
        wavenumber,
        frequency,
    );
    let mut h_spl_grid = HostMatrix::<f64>::zeros(res, res);
    for (idx, p) in h_pressures.iter().enumerate() {
        let i = idx % res;
        let j = idx / res;
        h_spl_grid[[j, i]] = pressure_to_spl(*p);
    }
    let h_slice = serde_json::json!({
        "x": grids.x_points,
        "y": grids.y_points,
        "spl": h_spl_grid.as_slice().to_vec(),
        "shape": [res, res],
        "frequency": frequency,
    });

    let v_pressures = calculate_pressure(
        mesh,
        surface_pressure,
        &grids.vertical_points(),
        wavenumber,
        frequency,
    );
    let mut v_spl_grid = HostMatrix::<f64>::zeros(res, res);
    for (idx, p) in v_pressures.iter().enumerate() {
        let i = idx % res;
        let j = idx / res;
        v_spl_grid[[j, i]] = pressure_to_spl(*p);
    }
    let v_slice = serde_json::json!({
        "x": grids.x_points,
        "z": grids.z_points,
        "spl": v_spl_grid.as_slice().to_vec(),
        "shape": [res, res],
        "frequency": frequency,
    });

    (h_slice, v_slice)
}

/// Machine-gated parallel form of [`try_generate_spatial_slices`].
///
/// Parallelizes over solutions with rayon when the machine-adaptive gate
/// ([`use_parallel`] over `solutions.len() × res²` cells at
/// [`SLICE_CELLS_PER_THREAD`] per thread) says pool dispatch pays off;
/// below the gate it runs the identical serial core. Rayon `collect` over
/// the indexed solution iterator preserves solution order, so output
/// matches the serial path element-wise. Requires `F: Sync` since the
/// callback is shared across threads.
pub fn try_generate_spatial_slices_par<F>(
    simulation: &RoomSimulation,
    config: &VisualizationConfig,
    mesh: &RoomMesh,
    solutions: &[(f64, Vec<Complex64>)],
    calculate_pressure: F,
) -> Result<(Vec<serde_json::Value>, Vec<serde_json::Value>), OutputError>
where
    F: Fn(&RoomMesh, &[Complex64], &[Point3D], f64, f64) -> Vec<Complex64> + Sync,
{
    let lp = first_listening_position(simulation)?;
    let (room_width, room_depth, room_height) = simulation.room.dimensions();

    let res = config.slice_resolution;
    if res < 2 {
        return Err(OutputError::SliceResolutionTooSmall { res });
    }
    let grids = SliceGrids::new(room_width, room_depth, room_height, res, [lp.x, lp.y, lp.z]);

    let wavenumbers: Vec<f64> = solutions
        .iter()
        .map(|(freq, _)| simulation.wavenumber(*freq))
        .collect();
    let cells = solutions.len().saturating_mul(res).saturating_mul(res);
    let parallel = use_parallel(cells, SLICE_CELLS_PER_THREAD);
    let pairs = slices_core(
        mesh,
        &grids,
        &wavenumbers,
        solutions,
        &calculate_pressure,
        parallel,
    );
    let (horizontal_slices, vertical_slices): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    Ok((horizontal_slices, vertical_slices))
}

/// Shared evaluation core behind [`try_generate_spatial_slices_par`]:
/// per-solution slice pairs, serially or with rayon over solutions.
///
/// Rayon `collect` over the indexed solution iterator preserves solution
/// order, so both paths emit element-wise identical output. The explicit
/// `parallel` flag exists for parity tests; production callers resolve it
/// through the machine-adaptive gate.
fn slices_core<F>(
    mesh: &RoomMesh,
    grids: &SliceGrids,
    wavenumbers: &[f64],
    solutions: &[(f64, Vec<Complex64>)],
    calculate_pressure: &F,
    parallel: bool,
) -> Vec<(serde_json::Value, serde_json::Value)>
where
    F: Fn(&RoomMesh, &[Complex64], &[Point3D], f64, f64) -> Vec<Complex64> + Sync,
{
    if parallel {
        solutions
            .par_iter()
            .enumerate()
            .map(|(index, (freq, surface_pressure))| {
                slice_pair(
                    mesh,
                    grids,
                    calculate_pressure,
                    surface_pressure.as_slice(),
                    wavenumbers[index],
                    *freq,
                )
            })
            .collect()
    } else {
        solutions
            .iter()
            .enumerate()
            .map(|(index, (freq, surface_pressure))| {
                slice_pair(
                    mesh,
                    grids,
                    calculate_pressure,
                    surface_pressure.as_slice(),
                    wavenumbers[index],
                    *freq,
                )
            })
            .collect()
    }
}

/// Print configuration summary to stdout
pub fn print_config_summary(config: &RoomConfig) {
    println!("\n=== Configuration Summary ===");
    match &config.room {
        crate::config::RoomGeometryConfig::Rectangular {
            width,
            depth,
            height,
        } => {
            println!(
                "Room: Rectangular {:.1}m × {:.1}m × {:.1}m",
                width, depth, height
            );
        }
        crate::config::RoomGeometryConfig::LShaped {
            width1,
            depth1,
            width2,
            depth2,
            height,
        } => {
            println!("Room: L-shaped");
            println!("  Main: {:.1}m × {:.1}m", width1, depth1);
            println!("  Extension: {:.1}m × {:.1}m", width2, depth2);
            println!("  Height: {:.1}m", height);
        }
    }

    println!("\nSources: {}", config.sources.len());
    for source in &config.sources {
        println!(
            "  - {}: ({:.2}, {:.2}, {:.2})",
            source.name, source.position.x, source.position.y, source.position.z
        );
        match &source.crossover {
            crate::config::CrossoverConfig::Lowpass { cutoff_freq, order } => {
                println!("    Lowpass: {:.0}Hz, order {}", cutoff_freq, order)
            }
            crate::config::CrossoverConfig::Highpass { cutoff_freq, order } => {
                println!("    Highpass: {:.0}Hz, order {}", cutoff_freq, order)
            }
            crate::config::CrossoverConfig::Bandpass {
                low_cutoff,
                high_cutoff,
                order,
            } => println!(
                "    Bandpass: {:.0}-{:.0}Hz, order {}",
                low_cutoff, high_cutoff, order
            ),
            _ => {} // Ignore other crossover types
        }
    }

    println!(
        "\nFrequencies: {:.0} Hz to {:.0} Hz ({} points)",
        config.frequencies.min_freq, config.frequencies.max_freq, config.frequencies.num_points
    );

    println!("\nSolver Configuration:");
    println!("  Method: {}", config.solver.method);
    println!(
        "  Mesh resolution: {} elements/meter",
        config.solver.mesh_resolution
    );
    println!(
        "  Adaptive integration: {}",
        config.solver.adaptive_integration
    );

    // Boundary summary
    println!("\nBoundaries:");
    let b = &config.boundaries;
    let format_bc = |s: &crate::config::SurfaceConfig| match s {
        crate::config::SurfaceConfig::Rigid => "Rigid".to_string(),
        crate::config::SurfaceConfig::Absorption { coefficient } => {
            format!("Abs α={:.2}", coefficient)
        }
        crate::config::SurfaceConfig::Impedance { real, imag } => {
            format!("Z={:.1}+{:.1}i", real, imag)
        }
        crate::config::SurfaceConfig::Admittance { real, imag } => {
            format!("Y={:.3}+{:.3}i", real, imag)
        }
        crate::config::SurfaceConfig::PorousLayer {
            model,
            thickness,
            flow_resistivity,
            air_gap,
            ..
        } => {
            let gap = air_gap.unwrap_or(0.0);
            format!(
                "{:?} porous layer, {:.3}m, {:.0} Pa*s/m^2, gap {:.3}m",
                model, thickness, flow_resistivity, gap
            )
        }
    };

    println!("  Default walls: {}", format_bc(&b.walls));
    println!("  Floor:         {}", format_bc(&b.floor));
    println!("  Ceiling:       {}", format_bc(&b.ceiling));
}

/// Create a default room configuration for testing
pub fn create_default_config() -> RoomConfig {
    RoomConfig {
        schema_version: crate::conventions::SchemaVersion::current(),
        room: crate::config::RoomGeometryConfig::Rectangular {
            width: 5.0,
            depth: 4.0,
            height: 2.5,
        },
        sources: vec![crate::config::SourceConfig {
            name: "Main Speaker".to_string(),
            position: crate::config::Point3DConfig {
                x: 2.5,
                y: 0.5,
                z: 1.2,
            },
            amplitude: 1.0,
            directivity: crate::config::DirectivityConfig::Omnidirectional,
            crossover: crate::config::CrossoverConfig::FullRange,
        }],
        listening_positions: vec![crate::config::Point3DConfig {
            x: 2.5,
            y: 2.0,
            z: 1.2,
        }],
        frequencies: crate::config::FrequencyConfig {
            min_freq: 50.0,
            max_freq: 500.0,
            num_points: 20,
            spacing: "logarithmic".to_string(),
        },
        solver: crate::config::SolverConfig::default(),
        boundaries: BoundaryConfig::default(),
        medium: crate::config::AmbientMediumConfig::default(),
        visualization: VisualizationConfig::default(),
        metadata: MetadataConfig::default(),
        conventions: crate::conventions::ConventionRegistry::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_metadata_carries_schema_solver_tolerances_and_conventions() {
        let config = create_default_config();
        config.validate().expect("default config is valid");
        let metadata = DatasetMetadata::from_config(&config, "bem");

        assert_eq!(metadata.schema_version, config.schema_version.0);
        assert_eq!(metadata.solver, "bem");
        assert_eq!(
            metadata.tolerances.gmres_tolerance,
            config.solver.gmres.tolerance
        );
        assert_eq!(
            metadata.tolerances.gmres_max_iter,
            config.solver.gmres.max_iter
        );
        assert_eq!(metadata.conventions, config.conventions);
        assert_eq!(metadata.date, config.metadata.date);

        let encoded = serde_json::to_value(&metadata).expect("metadata serializes");
        assert_eq!(encoded["schema_version"], config.schema_version.0);
        assert_eq!(encoded["solver"], "bem");
        assert!(encoded["tolerances"]["gmres_tolerance"].is_number());
        assert!(encoded["conventions"]["harmonic_time"].is_string());
    }

    fn empty_listener_simulation() -> RoomSimulation {
        let mut sim = RoomSimulation::new(
            RoomGeometry::Rectangular(crate::geometry::RectangularRoom::new(5.0, 4.0, 3.0)),
            vec![],
            vec![],
        );
        sim.listening_positions.clear();
        sim
    }

    #[test]
    fn try_output_json_rejects_empty_listeners() {
        let sim = empty_listener_simulation();
        let config = create_default_config();
        assert_eq!(
            try_create_output_json(&sim, &config, vec![], "bem"),
            Err(OutputError::NoListeningPositions)
        );
        assert_eq!(
            try_create_output_json_with_sources(&sim, &config, vec![], &[], "bem"),
            Err(OutputError::NoListeningPositions)
        );
    }

    #[test]
    fn slices_core_matches_on_both_paths() {
        let config = create_default_config();
        let sim = config
            .to_simulation()
            .expect("default config converts to simulation");
        let mesh = RoomMesh::default();
        let grids = SliceGrids::new(5.0, 4.0, 2.8, 4, [2.5, 2.0, 1.2]);
        let solutions = vec![
            (100.0, vec![Complex64::new(1.0, 0.0); 8]),
            (200.0, vec![Complex64::new(0.0, 1.0); 8]),
        ];
        let wavenumbers: Vec<f64> = solutions
            .iter()
            .map(|(freq, _)| sim.wavenumber(*freq))
            .collect();
        let constant = |_: &RoomMesh, _: &[Complex64], points: &[Point3D], _: f64, _: f64| {
            points
                .iter()
                .map(|p| Complex64::new(p.x + p.y + p.z, 0.0))
                .collect::<Vec<_>>()
        };
        let serial = slices_core(&mesh, &grids, &wavenumbers, &solutions, &constant, false);
        let parallel = slices_core(&mesh, &grids, &wavenumbers, &solutions, &constant, true);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn try_slices_reject_small_resolution() {
        let config = create_default_config();
        let sim = config
            .to_simulation()
            .expect("default config converts to simulation");
        let mesh = RoomMesh::default();
        let mut vis = VisualizationConfig::default();
        vis.slice_resolution = 1;
        assert_eq!(
            try_generate_spatial_slices(&sim, &vis, &mesh, &[], |_, _, _, _, _| vec![]),
            Err(OutputError::SliceResolutionTooSmall { res: 1 })
        );
        vis.slice_resolution = 0;
        assert_eq!(
            try_generate_spatial_slices(&sim, &vis, &mesh, &[], |_, _, _, _, _| vec![]),
            Err(OutputError::SliceResolutionTooSmall { res: 0 })
        );
    }
}
