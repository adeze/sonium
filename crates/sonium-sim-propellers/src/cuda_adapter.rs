//! Bounded native CUDA adapter for propeller panel workloads.
//!
//! This module deliberately exposes one production-connected cell instead of
//! labelling the whole propeller application as GPU-native.  It builds the
//! propeller-owned `BladeMesh`, converts each panel to a constant triangular
//! BEM pair, and invokes the checked Complex64 CUDA pair kernel from
//! `sonium-compute`.  Source-model harmonic evaluation and panelization remain
//! host-side; the artifact says so explicitly.  The device result is therefore
//! a native BEM panel-integral result, not a claim of resolved CFD, wake, or
//! end-to-end acoustic CUDA execution.

use num_complex::Complex64;
use sonium_compute::{
    Backend, BackendError, BemRegularPairInF64, BemRegularPairOutF64, CudaBackend,
    CudaBemRegularPairOperatorC64, CudaFailureInjectionPoint, CudaRuntimeStatsSnapshot, MemoryPool,
    MemoryPoolConfig, OomError, Transpose,
};
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use std::fmt;

use crate::{
    Blade3dConfig, BladeMeshError, EvaluationMode, PropellerDesign, PropellerModelFidelity,
    VortexLatticeError, VortexLatticeOptions, assemble_vortex_lattice_system,
};

/// Versioned identity for the bounded native adapter result.
pub const PROPELLER_CUDA_ADAPTER_SCHEMA_V1: &str = "sonium.propellers.cuda-native-adapter.v1";

/// Versioned identity for the bounded resident multipoint acoustic state.
pub const PROPELLER_CUDA_ACOUSTIC_SWEEP_SCHEMA_V1: &str =
    "sonium.propellers.cuda-native-acoustic-sweep.v1";

/// Versioned identity for the resident production VLM influence action.
pub const PROPELLER_CUDA_VLM_ACTION_SCHEMA_V1: &str =
    "sonium.propellers.cuda-native-vlm-influence-action.v1";

/// Failure returned by the checked propeller CUDA adapter.
#[derive(Debug)]
pub enum PropellerCudaAdapterError {
    /// Propeller mesh input could not be validated or built.
    Mesh(BladeMeshError),
    /// The selected CUDA device could not be initialized.
    Device(OomError),
    /// The owning compute backend rejected or failed the native operation.
    Backend(BackendError),
    /// The selected mesh did not produce a usable panel batch.
    EmptyPanelBatch,
    /// A source parameter cannot produce a finite BEM wave number.
    InvalidWaveNumber,
    /// A multipoint request must contain at least one harmonic.
    EmptyHarmonicBatch,
    /// The production VLM assembly rejected its geometry or options.
    VortexLattice(VortexLatticeError),
}

impl fmt::Display for PropellerCudaAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mesh(error) => write!(formatter, "invalid propeller mesh: {error}"),
            Self::Device(error) => write!(formatter, "CUDA device initialization failed: {error}"),
            Self::Backend(error) => write!(formatter, "CUDA BEM adapter failed: {error}"),
            Self::EmptyPanelBatch => formatter.write_str("propeller mesh produced no panel pairs"),
            Self::InvalidWaveNumber => {
                formatter.write_str("propeller BEM wave number is not finite")
            }
            Self::EmptyHarmonicBatch => {
                formatter.write_str("propeller CUDA multipoint request has no harmonics")
            }
            Self::VortexLattice(error) => write!(formatter, "invalid propeller VLM input: {error}"),
        }
    }
}

impl std::error::Error for PropellerCudaAdapterError {}

impl From<BladeMeshError> for PropellerCudaAdapterError {
    fn from(error: BladeMeshError) -> Self {
        Self::Mesh(error)
    }
}

impl From<OomError> for PropellerCudaAdapterError {
    fn from(error: OomError) -> Self {
        Self::Device(error)
    }
}

impl From<BackendError> for PropellerCudaAdapterError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

impl From<VortexLatticeError> for PropellerCudaAdapterError {
    fn from(error: VortexLatticeError) -> Self {
        Self::VortexLattice(error)
    }
}

/// Native CUDA result for one bounded panel-pair workload.
#[derive(Debug)]
pub struct PropellerCudaPanelResult {
    /// Versioned artifact identity.
    pub schema: &'static str,
    /// Requested and actual backend labels for this result.
    pub requested_backend: &'static str,
    pub actual_backend: &'static str,
    /// Scalar family used by the native kernel.
    pub precision: &'static str,
    /// The application fidelity that supplied panel geometry.
    pub fidelity: PropellerModelFidelity,
    /// Number of generated panels and BEM pair records.
    pub panel_count: usize,
    pub pair_count: usize,
    /// BEM parameters derived from the production propeller source.
    pub harmonic: usize,
    pub wave_number: f64,
    pub harmonic_factor: f64,
    /// CPU reference and native CUDA outputs, retained for the parity gate.
    pub cpu: Vec<BemRegularPairOutF64>,
    pub cuda: Vec<BemRegularPairOutF64>,
    pub max_relative_error: f64,
    /// Native backend counters sampled before the backend is dropped.
    pub runtime: CudaRuntimeStatsSnapshot,
}

/// One production source/panel acoustic state derived from a native CUDA
/// panel solve.  This is a panel-integrated transfer proxy at phase zero; it
/// is not a resolved observer or wake pressure.
#[derive(Debug, Clone, Copy)]
pub struct PropellerCudaAcousticHarmonic {
    pub harmonic: usize,
    pub frequency_hz: f64,
    pub source_pressure_pa: f64,
    pub cpu_panel_pressure: Complex64,
    pub cuda_panel_pressure: Complex64,
    pub max_relative_error: f64,
}

/// Native CUDA result for the bounded production multipoint acoustic cell.
#[derive(Debug)]
pub struct PropellerCudaAcousticSweepResult {
    pub schema: &'static str,
    pub requested_backend: &'static str,
    pub actual_backend: &'static str,
    pub precision: &'static str,
    pub fidelity: PropellerModelFidelity,
    pub panel_count: usize,
    pub pair_count: usize,
    pub harmonics: Vec<PropellerCudaAcousticHarmonic>,
    pub runtime: CudaRuntimeStatsSnapshot,
}

/// Result of applying the real production VLM influence matrix to one
/// circulation vector.  The dense matrix is assembled by
/// `vortex_lattice_reference` and the action itself is dispatched through
/// checked native CUDA f64 GEMV; this does not claim that the host solve,
/// wake construction, or load reconstruction is device-resident.
#[derive(Debug)]
pub struct PropellerCudaVortexLatticeActionResult {
    pub schema: &'static str,
    pub requested_backend: &'static str,
    pub actual_backend: &'static str,
    pub precision: &'static str,
    pub panel_count: usize,
    pub dimension: usize,
    pub cpu: Vec<f64>,
    pub cuda: Vec<f64>,
    pub max_relative_error: f64,
    pub runtime: CudaRuntimeStatsSnapshot,
}

/// Reusable native CUDA session for one production-connected panel topology.
///
/// Mesh construction and pair geometry upload happen once in [`Self::new`].
/// [`Self::try_solve`] may then update the harmonic-derived kernel scalars and
/// publish a new result without reallocating or re-uploading the resident
/// device geometry. The source model remains host-side by design.
pub struct PropellerCudaPanelSession {
    pool: MemoryPool<CudaBackend>,
    operator: CudaBemRegularPairOperatorC64,
    pairs: Vec<BemRegularPairInF64>,
    panel_areas: Vec<f64>,
    panel_count: usize,
    cuda_output: Vec<BemRegularPairOutF64>,
}

/// Reusable native CUDA session for the production propeller VLM influence
/// action.  Geometry and the dense influence matrix are assembled once on the
/// host by the checked production reference, then retained in a CUDA buffer;
/// each action only transfers the changing circulation vector and performs a
/// native f64 GEMV.
pub struct PropellerCudaVortexLatticeSession {
    pool: MemoryPool<CudaBackend>,
    influence: sonium_compute::pool::Buffer<f64, CudaBackend>,
    input: sonium_compute::pool::Buffer<f64, CudaBackend>,
    output: sonium_compute::pool::Buffer<f64, CudaBackend>,
    cpu_influence: Vec<f64>,
    panel_count: usize,
    dimension: usize,
    published_output: Vec<f64>,
}

impl PropellerCudaVortexLatticeSession {
    /// Assemble and upload one production BladeMesh VLM influence system.
    pub fn new(
        design: &PropellerDesign,
        mesh_config: &Blade3dConfig,
        options: VortexLatticeOptions,
        cuda_device: usize,
    ) -> Result<Self, PropellerCudaAdapterError> {
        let backend = CudaBackend::new(cuda_device)?;
        let mesh = mesh_config.try_build_mesh(design)?;
        let system = assemble_vortex_lattice_system(&mesh.panels, options)?;
        let dimension = system.dimension;
        let pool = MemoryPool::new(backend, MemoryPoolConfig::default())?;
        let mut influence = pool.try_alloc(system.influence.len())?;
        pool.backend()
            .try_copy_h2d(&system.influence, influence.raw_mut())?;
        let input = pool.try_alloc(dimension)?;
        let output = pool.try_alloc(dimension)?;
        Ok(Self {
            pool,
            influence,
            input,
            output,
            cpu_influence: system.influence,
            panel_count: mesh.panels.len(),
            dimension,
            published_output: vec![0.0; dimension],
        })
    }

    /// Number of rows/columns in the resident production influence matrix.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of resident production BladeMesh panels.
    pub fn panel_count(&self) -> usize {
        self.panel_count
    }

    /// Snapshot the backend's checked runtime ledger.
    pub fn runtime_stats(&self) -> CudaRuntimeStatsSnapshot {
        self.pool.backend().runtime_stats()
    }

    /// Return the last successfully published action output.
    pub fn last_published_output(&self) -> &[f64] {
        &self.published_output
    }

    /// Arm a one-shot checked CUDA failure for lifecycle evidence.
    pub fn inject_failure(&self, point: CudaFailureInjectionPoint) {
        self.pool.backend().inject_failure(point);
    }

    /// Recover the resident session after a reusable checked CUDA failure.
    pub fn recover_after_error(&self) -> Result<(), PropellerCudaAdapterError> {
        self.pool.backend().recover_after_error()?;
        Ok(())
    }

    /// Apply the resident production influence matrix to a circulation vector.
    ///
    /// Validation occurs before device work.  The host publication is
    /// transactional: GEMV, synchronization, and D2H must all succeed before
    /// the new result replaces the prior output.
    pub fn try_apply(
        &mut self,
        circulation: &[f64],
    ) -> Result<PropellerCudaVortexLatticeActionResult, PropellerCudaAdapterError> {
        if circulation.len() != self.dimension {
            return Err(PropellerCudaAdapterError::Backend(
                BackendError::InvalidArgument {
                    operation: "propeller_vlm_gemv",
                    reason: format!(
                        "circulation length {} does not match VLM dimension {}",
                        circulation.len(),
                        self.dimension
                    ),
                },
            ));
        }
        if circulation.iter().any(|value| !value.is_finite()) {
            return Err(PropellerCudaAdapterError::Backend(
                BackendError::InvalidArgument {
                    operation: "propeller_vlm_gemv",
                    reason: "circulation vector contains a non-finite value".into(),
                },
            ));
        }

        let earlier = self.pool.backend().begin_runtime_stats();
        self.pool
            .backend()
            .try_copy_h2d(circulation, self.input.raw_mut())?;
        self.pool.backend().try_gemv(
            Transpose::NoTrans,
            self.dimension,
            self.dimension,
            1.0_f64,
            self.influence.raw(),
            self.dimension,
            self.input.raw(),
            0.0_f64,
            self.output.raw_mut(),
        )?;
        self.pool.backend().try_synchronize()?;
        let mut cuda = vec![0.0; self.dimension];
        self.pool
            .backend()
            .try_copy_d2h(self.output.raw(), &mut cuda)?;
        if cuda.iter().any(|value| !value.is_finite()) {
            return Err(PropellerCudaAdapterError::Backend(
                BackendError::OperationFailed {
                    operation: "propeller_vlm_gemv",
                    reason: "CUDA influence action returned a non-finite value".into(),
                },
            ));
        }
        let cpu = self
            .cpu_influence
            .chunks_exact(self.dimension)
            .map(|row| {
                row.iter()
                    .zip(circulation)
                    .map(|(coefficient, value)| coefficient * value)
                    .sum()
            })
            .collect::<Vec<_>>();
        let max_relative_error = scalar_vector_relative_error(&cuda, &cpu);
        self.published_output.clone_from(&cuda);
        Ok(PropellerCudaVortexLatticeActionResult {
            schema: PROPELLER_CUDA_VLM_ACTION_SCHEMA_V1,
            requested_backend: "cuda",
            actual_backend: "cuda",
            precision: "f64",
            panel_count: self.panel_count,
            dimension: self.dimension,
            cpu,
            cuda,
            max_relative_error,
            runtime: self.pool.backend().runtime_stats_since(earlier),
        })
    }
}

impl PropellerCudaPanelSession {
    /// Build a reusable session and upload the production mesh's pair
    /// geometry to CUDA. The initial harmonic only validates the kernel
    /// parameters; it is not evaluated until [`Self::try_solve`].
    pub fn new(
        design: &PropellerDesign,
        mesh_config: &Blade3dConfig,
        harmonic: usize,
        cuda_device: usize,
    ) -> Result<Self, PropellerCudaAdapterError> {
        let backend = CudaBackend::new(cuda_device)?;
        Self::new_with_backend(design, mesh_config, harmonic, backend)
    }

    /// Build a session over an already-created CUDA backend.
    ///
    /// This constructor is the narrow driver-shim seam used by lifecycle
    /// evidence: callers can arm a checked allocation failure before the
    /// resident operator is constructed, while production callers should use
    /// [`Self::new`] so device selection remains explicit.
    pub fn new_with_backend(
        design: &PropellerDesign,
        mesh_config: &Blade3dConfig,
        harmonic: usize,
        backend: CudaBackend,
    ) -> Result<Self, PropellerCudaAdapterError> {
        let (wave_number, harmonic_factor) = wave_parameters(&design.source, harmonic)?;
        let mesh = mesh_config.try_build_mesh(design)?;
        let pairs = panel_pairs(&mesh);
        if pairs.is_empty() {
            return Err(PropellerCudaAdapterError::EmptyPanelBatch);
        }

        if !backend.has_bem_regular_pair_c64() {
            return Err(PropellerCudaAdapterError::Backend(
                BackendError::Unsupported {
                    backend: "cuda",
                    operation: "bem_regular_pair_c64_batch",
                    scalar: "Complex64",
                },
            ));
        }
        let pool = MemoryPool::new(backend, MemoryPoolConfig::default())?;
        let operator = CudaBemRegularPairOperatorC64::new(
            pool.clone(),
            pairs.clone(),
            wave_number,
            harmonic_factor,
        )?;
        let cuda_output = vec![BemRegularPairOutF64::default(); pairs.len()];
        let panel_areas = pairs.iter().map(pair_area).collect();
        Ok(Self {
            pool,
            operator,
            pairs,
            panel_areas,
            panel_count: mesh.panels.len(),
            cuda_output,
        })
    }

    /// Number of resident panel-pair descriptors.
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// Snapshot the owning backend's counters and resident byte ledger.
    pub fn runtime_stats(&self) -> CudaRuntimeStatsSnapshot {
        self.pool.backend().runtime_stats()
    }

    /// Return the last host-publication destination used by the session.
    ///
    /// A failed checked operation never publishes a new
    /// [`PropellerCudaPanelResult`]. This read-only snapshot is exposed for
    /// lifecycle evidence to verify that injected pre-dispatch faults leave
    /// the prior destination unchanged before it is discarded or retried.
    pub fn last_published_output(&self) -> &[BemRegularPairOutF64] {
        &self.cuda_output
    }

    /// Arm a one-shot checked failure for acceptance tests. Recoverable
    /// transfer/launch failures can be cleared with [`Self::recover_after_error`].
    pub fn inject_failure(&self, point: CudaFailureInjectionPoint) {
        self.pool.backend().inject_failure(point);
    }

    /// Clear an armed one-shot failure without changing the pending error.
    pub fn clear_failure(&self) {
        self.pool.backend().clear_failure();
    }

    /// Recover the session after a reusable checked CUDA failure.
    pub fn recover_after_error(&self) -> Result<(), PropellerCudaAdapterError> {
        self.pool.backend().recover_after_error()?;
        Ok(())
    }

    /// Evaluate a host-side source update against the resident panel
    /// geometry. The returned runtime snapshot is a per-solve delta, making
    /// warm allocation/transfer/launch bounds directly checkable.
    pub fn try_solve(
        &mut self,
        design: &PropellerDesign,
        harmonic: usize,
    ) -> Result<PropellerCudaPanelResult, PropellerCudaAdapterError> {
        let (wave_number, harmonic_factor) = wave_parameters(&design.source, harmonic)?;
        let earlier = self.pool.backend().begin_runtime_stats();
        let cpu = sonium_compute::bem_cpu::regular_pair_f64_batch(
            &self.pairs,
            wave_number,
            harmonic_factor,
        );
        self.operator.try_apply_into_host_with_parameters(
            &mut self.cuda_output,
            wave_number,
            harmonic_factor,
        )?;
        let runtime = self.pool.backend().runtime_stats_since(earlier);
        let max_relative_error = max_relative_error(&self.cuda_output, &cpu);
        Ok(PropellerCudaPanelResult {
            schema: PROPELLER_CUDA_ADAPTER_SCHEMA_V1,
            requested_backend: "cuda",
            actual_backend: "cuda",
            precision: "f64 / Complex64",
            fidelity: PropellerModelFidelity::PanelGeometrySurrogate,
            panel_count: self.panel_count,
            pair_count: self.pairs.len(),
            harmonic,
            wave_number,
            harmonic_factor,
            cpu,
            cuda: self.cuda_output.clone(),
            max_relative_error,
            runtime,
        })
    }

    /// Evaluate several source harmonics while retaining the panel geometry
    /// and CUDA workspace.  The complete request is validated before any
    /// device operation.  Publication is transactional: if a later harmonic
    /// fails, the prior output destination is restored and no partial sweep
    /// result is returned.
    pub fn try_solve_multipoint(
        &mut self,
        design: &PropellerDesign,
        harmonics: &[usize],
    ) -> Result<PropellerCudaAcousticSweepResult, PropellerCudaAdapterError> {
        if harmonics.is_empty() {
            return Err(PropellerCudaAdapterError::EmptyHarmonicBatch);
        }
        for &harmonic in harmonics {
            wave_parameters(&design.source, harmonic)?;
        }

        let output_before = self.cuda_output.clone();
        let earlier = self.pool.backend().begin_runtime_stats();
        let mut states = Vec::with_capacity(harmonics.len());
        for &harmonic in harmonics {
            let result = match self.try_solve(design, harmonic) {
                Ok(result) => result,
                Err(error) => {
                    self.cuda_output = output_before;
                    return Err(error);
                }
            };
            let source_pressure_pa = design
                .spectrum(harmonic, EvaluationMode::FastSurrogate)
                .harmonics
                .last()
                .map_or(f64::NAN, |entry| entry.pressure_pa);
            let cpu_panel_pressure =
                weighted_panel_pressure(&result.cpu, &self.panel_areas, source_pressure_pa);
            let cuda_panel_pressure =
                weighted_panel_pressure(&result.cuda, &self.panel_areas, source_pressure_pa);
            let max_relative_error = relative_error(cuda_panel_pressure, cpu_panel_pressure);
            if !source_pressure_pa.is_finite() || !max_relative_error.is_finite() {
                self.cuda_output = output_before;
                return Err(PropellerCudaAdapterError::InvalidWaveNumber);
            }
            states.push(PropellerCudaAcousticHarmonic {
                harmonic,
                frequency_hz: design.source.harmonic_frequency(harmonic),
                source_pressure_pa,
                cpu_panel_pressure,
                cuda_panel_pressure,
                max_relative_error,
            });
        }

        Ok(PropellerCudaAcousticSweepResult {
            schema: PROPELLER_CUDA_ACOUSTIC_SWEEP_SCHEMA_V1,
            requested_backend: "cuda",
            actual_backend: "cuda",
            precision: "f64 / Complex64",
            fidelity: PropellerModelFidelity::PanelGeometrySurrogate,
            panel_count: self.panel_count,
            pair_count: self.pairs.len(),
            harmonics: states,
            runtime: self.pool.backend().runtime_stats_since(earlier),
        })
    }
}

/// Evaluate the production propeller panel geometry through the native CUDA
/// regular Complex64 BEM pair kernel.
///
/// This is intentionally a bounded adapter API.  It does not claim that the
/// CPU source model, blade-element loads, wake, or observer propagation are
/// device-resident.  The checked compute operation itself uses CUDA device
/// buffers for pair geometry and output and returns typed failures for missing
/// devices, unsupported kernels, malformed input, or driver errors.
pub fn evaluate_cuda_panel_bem(
    design: &PropellerDesign,
    mesh_config: &Blade3dConfig,
    harmonic: usize,
    cuda_device: usize,
) -> Result<PropellerCudaPanelResult, PropellerCudaAdapterError> {
    if harmonic == 0 {
        return Err(PropellerCudaAdapterError::InvalidWaveNumber);
    }
    let mesh = mesh_config.try_build_mesh(design)?;
    let pairs = panel_pairs(&mesh);
    if pairs.is_empty() {
        return Err(PropellerCudaAdapterError::EmptyPanelBatch);
    }

    let source = &design.source;
    let source_frequency = source.harmonic_frequency(harmonic);
    let wave_number = 2.0 * std::f64::consts::PI * source_frequency / source.speed_of_sound;
    let harmonic_factor = harmonic as f64;
    if !wave_number.is_finite() || !harmonic_factor.is_finite() {
        return Err(PropellerCudaAdapterError::InvalidWaveNumber);
    }

    let cpu = sonium_compute::bem_cpu::regular_pair_f64_batch(&pairs, wave_number, harmonic_factor);
    let backend = CudaBackend::new(cuda_device)?;
    if !backend.has_bem_regular_pair_c64() {
        return Err(PropellerCudaAdapterError::Backend(
            BackendError::Unsupported {
                backend: "cuda",
                operation: "bem_regular_pair_c64_batch",
                scalar: "Complex64",
            },
        ));
    }
    backend.reset_runtime_stats();
    let cuda = backend.try_bem_regular_pair_c64_batch(&pairs, wave_number, harmonic_factor)?;
    let runtime = backend.runtime_stats();
    let max_relative_error = max_relative_error(&cuda, &cpu);

    Ok(PropellerCudaPanelResult {
        schema: PROPELLER_CUDA_ADAPTER_SCHEMA_V1,
        requested_backend: "cuda",
        actual_backend: "cuda",
        precision: "f64 / Complex64",
        fidelity: PropellerModelFidelity::PanelGeometrySurrogate,
        panel_count: mesh.panels.len(),
        pair_count: pairs.len(),
        harmonic,
        wave_number,
        harmonic_factor,
        cpu,
        cuda,
        max_relative_error,
        runtime,
    })
}

fn wave_parameters(
    source: &PropellerSource,
    harmonic: usize,
) -> Result<(f64, f64), PropellerCudaAdapterError> {
    if harmonic == 0 {
        return Err(PropellerCudaAdapterError::InvalidWaveNumber);
    }
    let source_frequency = source.harmonic_frequency(harmonic);
    let wave_number = 2.0 * std::f64::consts::PI * source_frequency / source.speed_of_sound;
    let harmonic_factor = harmonic as f64;
    if !wave_number.is_finite() || !harmonic_factor.is_finite() {
        return Err(PropellerCudaAdapterError::InvalidWaveNumber);
    }
    Ok((wave_number, harmonic_factor))
}

fn panel_pairs(mesh: &crate::BladeMesh) -> Vec<BemRegularPairInF64> {
    mesh.panels
        .iter()
        .filter_map(|panel| {
            let [v0, v1, v2, _] = panel.vertices;
            let edge1 = v1.sub(v0);
            let edge2 = v2.sub(v0);
            let raw_normal = edge1.cross(edge2);
            let normal_length = raw_normal.norm();
            if !normal_length.is_finite() || normal_length <= f64::EPSILON {
                return None;
            }
            let normal = Point3Ext::scale(raw_normal, 1.0 / normal_length);
            let center = Point3Ext::scale(Point3Ext::add(Point3Ext::add(v0, v1), v2), 1.0 / 3.0);
            // Keep the source point just outside the panel so the regular
            // kernel's non-singular contract remains valid even for a
            // one-panel test mesh.
            let offset = panel.area.sqrt().max(1.0e-6) * 1.0e-3;
            let source_point = Point3Ext::add(center, Point3Ext::scale(normal, offset));
            Some(BemRegularPairInF64::from_triangle(
                source_point.into_array(),
                normal.into_array(),
                v0.into_array(),
                v1.into_array(),
                v2.into_array(),
            ))
        })
        .collect()
}

fn max_relative_error(actual: &[BemRegularPairOutF64], expected: &[BemRegularPairOutF64]) -> f64 {
    if actual.len() != expected.len() || actual.is_empty() {
        return f64::INFINITY;
    }
    actual
        .iter()
        .zip(expected)
        .flat_map(|(got, want)| {
            [
                relative_error(got.g, want.g),
                relative_error(got.h, want.h),
                relative_error(got.ht, want.ht),
                relative_error(got.e, want.e),
            ]
        })
        .fold(0.0, f64::max)
}

fn pair_area(pair: &BemRegularPairInF64) -> f64 {
    let e1 = [
        pair.v1[0] - pair.v0[0],
        pair.v1[1] - pair.v0[1],
        pair.v1[2] - pair.v0[2],
    ];
    let e2 = [
        pair.v2[0] - pair.v0[0],
        pair.v2[1] - pair.v0[1],
        pair.v2[2] - pair.v0[2],
    ];
    0.5 * (e1[1] * e2[2] - e1[2] * e2[1])
        .hypot((e1[2] * e2[0] - e1[0] * e2[2]).hypot(e1[0] * e2[1] - e1[1] * e2[0]))
}

fn weighted_panel_pressure(
    output: &[BemRegularPairOutF64],
    panel_areas: &[f64],
    source_pressure_pa: f64,
) -> Complex64 {
    let total_area: f64 = panel_areas.iter().sum();
    let transfer = output
        .iter()
        .zip(panel_areas)
        .map(|(value, area)| value.g * *area)
        .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value)
        / total_area.max(f64::MIN_POSITIVE);
    transfer * source_pressure_pa
}

fn relative_error(actual: Complex64, expected: Complex64) -> f64 {
    if [actual.re, actual.im, expected.re, expected.im]
        .into_iter()
        .any(|value| !value.is_finite())
    {
        return f64::INFINITY;
    }
    (actual - expected).norm() / expected.norm().max(1.0)
}

fn scalar_vector_relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    if actual.len() != expected.len() || actual.is_empty() {
        return f64::INFINITY;
    }
    actual
        .iter()
        .zip(expected)
        .map(|(got, want)| (got - want).abs() / want.abs().max(1.0))
        .fold(0.0, f64::max)
}

trait Point3Ext {
    fn sub(self, other: Self) -> Self;
    fn cross(self, other: Self) -> Self;
    fn norm(self) -> f64;
    fn add(self, other: Self) -> Self;
    fn scale(self, factor: f64) -> Self;
    fn into_array(self) -> [f64; 3];
}

impl Point3Ext for crate::Point3 {
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    fn scale(self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }

    fn into_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}
