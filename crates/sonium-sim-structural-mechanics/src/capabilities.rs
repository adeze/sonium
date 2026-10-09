//! Structural execution provenance and backend capability declarations.
//!
//! The lower-level compute crate has operation-level backend capabilities, but
//! structural studies currently dispatch through the CPU FEM reference path.
//! This module makes that boundary explicit in portable result schemas. It
//! reports integration status, not a runtime hardware probe.

use serde::{Deserialize, Serialize};

/// Schema version for structural backend and execution metadata.
pub const STRUCTURAL_CAPABILITY_SCHEMA_VERSION: u32 = 1;

/// Backend name recorded in structural result provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructuralBackend {
    Cpu,
    Cuda,
    Metal,
    Wasm,
}

impl StructuralBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Wasm => "wasm",
        }
    }
}

/// Precision policy exposed by a structural execution path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructuralPrecisionPolicy {
    F64,
    F32,
    MixedF32F64,
}

impl StructuralPrecisionPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::MixedF32F64 => "mixed-f32-f64",
        }
    }
}

/// Whether a backend is the trusted reference path, an accelerator path, or
/// only a declared unsupported extension point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructuralExecutionMode {
    CpuReference,
    Accelerator,
    Unsupported,
}

/// Structural study families represented by the capability catalogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructuralAnalysisFamily {
    Static,
    Axisymmetric,
    Viscoelastic,
    Shell,
    Modal,
    Harmonic,
    Thermal,
    Solid3D,
    Composite,
    Buckling,
    Transient,
    RandomVibration,
    Fatigue,
    Reliability,
    Contact,
    Plasticity,
    Hyperelastic,
    PhaseFieldDamage,
    ShapeOptimization,
    TopologyOptimization,
    Uncertainty,
    ReducedOrder,
}

/// Capability declaration for one structural backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralBackendCapability {
    pub schema_version: u32,
    pub backend: StructuralBackend,
    /// Integration status in this crate; this is not a device availability probe.
    pub integrated: bool,
    pub execution_mode: StructuralExecutionMode,
    pub precision_policies: Vec<StructuralPrecisionPolicy>,
    pub analysis_families: Vec<StructuralAnalysisFamily>,
    pub limitations: Vec<String>,
}

impl StructuralBackendCapability {
    /// Declared support (integration status x family x precision lists), not
    /// a runtime device probe: `capability_verified` records that the
    /// declaration was reviewed, not that a device answered. Callers needing
    /// a live-device guarantee must probe the backend themselves.
    pub fn supports(
        &self,
        family: StructuralAnalysisFamily,
        precision: StructuralPrecisionPolicy,
    ) -> bool {
        self.integrated
            && self.analysis_families.contains(&family)
            && self.precision_policies.contains(&precision)
    }
}

/// Execution metadata attached to portable structural result reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralExecutionProvenance {
    pub schema_version: u32,
    pub backend: StructuralBackend,
    pub execution_mode: StructuralExecutionMode,
    pub precision_policy: StructuralPrecisionPolicy,
    pub capability_id: String,
    pub capability_verified: bool,
    pub limitations: Vec<String>,
}

impl StructuralExecutionProvenance {
    /// Provenance used by all currently implemented structural FEM studies.
    pub fn cpu_reference() -> Self {
        Self {
            schema_version: STRUCTURAL_CAPABILITY_SCHEMA_VERSION,
            backend: StructuralBackend::Cpu,
            execution_mode: StructuralExecutionMode::CpuReference,
            precision_policy: StructuralPrecisionPolicy::F64,
            capability_id: "sonium-structural.cpu-reference.f64.v1".to_string(),
            capability_verified: true,
            limitations: vec![
                "This CPU reference provenance uses the CPU FEM reference path only; it makes no claim about the optional CUDA P1-bracket slice (see cuda_static_bracket_reference) and claims no Metal or WASM structural dispatch.".to_string(),
                "f64 is the only precision policy exposed by this structural study crate.".to_string(),
            ],
        }
    }

    /// Provenance for the bounded CUDA P1/f64 plane-stress bracket slice.
    #[cfg(feature = "cuda")]
    pub fn cuda_static_bracket_reference() -> Self {
        Self {
            schema_version: STRUCTURAL_CAPABILITY_SCHEMA_VERSION,
            backend: StructuralBackend::Cuda,
            execution_mode: StructuralExecutionMode::Accelerator,
            precision_policy: StructuralPrecisionPolicy::F64,
            capability_id: "sonium-structural.cuda.f64.static-bracket-p1.v1".to_string(),
            capability_verified: true,
            limitations: vec![
                "CUDA support is limited to P1 triangular, small-strain, isotropic, linear plane-stress static bracket analysis with homogeneous root constraints.".to_string(),
                "FEM element assembly and constraint construction are deterministic CPU/f64 operations; the constrained CSR operator, point-Jacobi GMRES forward and transpose-adjoint solves, and true-residual products execute on CUDA without host fallback.".to_string(),
                "The only supported design derivative is uniform thickness compliance. P2, plane strain, modal, buckling, nonlinear, contact, topology, mixed precision, and general design derivatives remain unsupported.".to_string(),
            ],
        }
    }

    /// Provenance for the assembled CPU/f64 axisymmetric plane-strain cylinder path.
    pub fn cpu_axisymmetric_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.axisymmetric-cylinder.v1".to_string();
        provenance.limitations.extend([
            "Axisymmetric execution is limited to a meridional P1 triangular thick-cylinder FEM solve with isotropic elasticity and an explicitly constrained plane-strain axial displacement.".to_string(),
            "The integrated reference compares pressure loading, displacement, and meridional radial/hoop stress against the closed-form Lamé thick-cylinder solution; general axisymmetric, nonlinear, contact, thermal, and external-solver parity are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the assembled CPU/f64 mixed u-p Taylor-Hood path.
    pub fn cpu_mixed_up_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.mixed-up-taylor-hood.v1".to_string();
        provenance.limitations.extend([
            "Mixed u-p execution is limited to 2D Taylor-Hood P2-P1 direct-sparse solves with full Dirichlet data; Neumann traction assembly and iterative saddle solvers are not claimed.".to_string(),
            "Poisson's ratio is admitted on (0, 1/2]; the exactly incompressible limit pins one pressure DOF and needs a fully constrained displacement boundary.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the load-stepped CPU/f64 corotational beam path.
    pub fn cpu_corotational_beam_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.corotational-beam.v1".to_string();
        provenance.limitations.extend([
            "Corotational beam execution is limited to 2D Euler-Bernoulli cantilevers with clamped-root Dirichlet data, tip axial/transverse loads, and dense direct Newton corrections.".to_string(),
            "Load control cannot pass limit points; buckling is characterized below the critical load (beam-column response, Southwell extrapolation), not traced through post-buckling.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the assembled CPU/f64 bolted-flange contact path.
    pub fn cpu_bolted_flange_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.bolted-flange-contact.v1".to_string();
        provenance.limitations.extend([
            "Bolted-flange execution is limited to linear P1 tetrahedral flange/base blocks with a single central axial bolt spring, prescribed pretension shortening, and node-to-surface augmented-Lagrangian contact.".to_string(),
            "Gaskets, thread engagement, bending bolts, multi-bolt patterns, and post-separation flange rotation beyond small-strain kinematics are not claimed.".to_string(),
            "Quasi-static gross sliding past Coulomb onset is not claimed: the validated shear envelope covers stick response and the onset bracket.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the assembled CPU/f64 body-fitted bracket shape path.
    pub fn cpu_bracket_shape_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.bracket-shape.v1".to_string();
        provenance.limitations.extend([
            "Shape execution is limited to the assembled linear P1 plane-stress bracket with transactional body-fitted mesh motion and static/modal reanalysis.".to_string(),
            "Compliance/volume, displacement, centroid-stress, first-bending-frequency, and bounded linearized-buckling target objectives are supported; geometry-level analytic shape derivatives, general 3D shape optimization, and nonlinear/contact behavior are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the integrated CPU/f64 matrix-free quadratic-solid path.
    ///
    /// This is still the trusted CPU reference implementation, but it is kept
    /// distinct from the assembled path so reports can identify the operator
    /// representation and its applicability limits.
    pub fn cpu_matrix_free_p2_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.matrix-free-p2.v1".to_string();
        provenance.limitations.extend([
            "Matrix-free execution is integrated for affine tetrahedral 3D elasticity with P2 H1 displacement space.".to_string(),
            "The matrix-free preconditioner is diagonal FGMRES; AMG is available only on the assembled operator path.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the integrated CPU/f64 flat triangular shell path.
    pub fn cpu_shell_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.shell-tri3-cst-dkt.v1".to_string();
        provenance.limitations.extend([
            "Shell execution is limited to flat three-node triangular CST membrane plus DKT bending elements with drilling stabilization.".to_string(),
            "The integrated shell study is linear small-strain static analysis with homogeneous clamping and pressure loading.".to_string(),
            "Anisotropic, viscoelastic, nonlinear, contact, fracture, and GPU/WASM shell dispatch are not implemented here.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the integrated CPU/f64 complex-stiffness plane-stress path.
    pub fn cpu_viscoelastic_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.complex64.viscoelastic-2d.v1".to_string();
        provenance.limitations.extend([
            "Viscoelastic execution is limited to a rectangular P1 plane-stress mesh with isotropic structural loss factor C* = (1 + i*eta) C.".to_string(),
            "The integrated study uses a homogeneous x-minimum clamp and a right/top corner point load; it is not a time-domain hereditary material model.".to_string(),
            "Anisotropic, frequency-dependent, nonlinear, contact, fracture, and GPU/WASM viscoelastic dispatch are not implemented here.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the integrated real 2D FEM topology path.
    pub fn cpu_fem_topology_2d_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.fem-topology-2d.v1".to_string();
        provenance.limitations.extend([
            "Topology execution is limited to deterministic P1 triangular plane-stress FEM on a rectangular design domain with density-scaled stiffness and consistent mass.".to_string(),
            "The 2D optimizer supports compliance, first-frequency, and compressive linearized-buckling objectives; all retain volume, stress aggregation, connectivity, minimum feature, overhang, modal, projected/body-fitted reanalysis, and equilibrium gates.".to_string(),
            "Frequency sensitivities use assembled generalized-eigenvalue differentiation; buckling sensitivities use bounded central finite differences of the re-solved initial-stress factor and are not nonlinear post-buckling derivatives.".to_string(),
            "CUDA, Metal, WASM, external-solver topology parity, and large-scale distributed optimization are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for shared-operator multi-load FEM topology reanalysis.
    pub fn cpu_fem_topology_2d_multi_load_reference() -> Self {
        let mut provenance = Self::cpu_fem_topology_2d_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.fem-topology-2d-multi-load.v1".to_string();
        provenance.limitations.extend([
            "Multi-load execution reuses one constrained sparse factorization to verify multiple load cases on the accepted single-primary-load topology design; it does not yet optimize the envelope objective across cases.".to_string(),
            "CUDA, Metal, WASM, external-solver parity, and distributed multi-load optimization are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the shared-load exact max-compliance topology path.
    pub fn cpu_fem_topology_2d_envelope_reference() -> Self {
        let mut provenance = Self::cpu_fem_topology_2d_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.fem-topology-2d-envelope.v1".to_string();
        provenance.limitations.extend([
            "Envelope execution optimizes the exact maximum compliance over named load cases with an active-case subgradient; tied cases use their averaged subgradient and the objective is nonsmooth there.".to_string(),
            "One constrained sparse factorization is reused for all load-case RHS solves at each density evaluation; CUDA, Metal, WASM, external-solver parity, and general 3D topology are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for real 2D FEM topology morphology/uncertainty reanalysis.
    pub fn cpu_fem_topology_2d_robust_reference() -> Self {
        let mut provenance = Self::cpu_fem_topology_2d_envelope_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.fem-topology-2d-robust.v1".to_string();
        provenance.limitations.extend([
            "Robust execution applies graph-based element erosion and dilation after the exact load-envelope optimization, then re-solves every named morphology/material/load scenario with assembled P1 FEM.".to_string(),
            "The uncertainty model is multiplicative isotropic load, Young's-modulus, and thickness scaling; it is not a calibrated probabilistic distribution or a general robust 3D optimizer.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the assembled P1 3D solid shared-factorization path.
    pub fn cpu_solid3d_multi_load_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.solid3d-multi-load.v1".to_string();
        provenance.limitations.extend([
            "Multi-load execution is limited to assembled linear P1 tetrahedral elasticity with a common fixed boundary and x-maximum traction face; one constrained sparse factorization is reused for the case RHS solves.".to_string(),
            "Quadratic/matrix-free, shell, nonlinear, contact, external-solver parity, CUDA, Metal, WASM, and distributed execution are not claimed by this capability.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the bounded assembled linear-tetrahedral 3D SIMP path.
    pub fn cpu_solid3d_topology_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.solid3d-topology.v1".to_string();
        provenance.limitations.extend([
            "3D topology uses assembled linear tetrahedra with an exact maximum-compliance load-case subgradient and one constrained sparse factorization per density iterate.".to_string(),
            "Topology uses a centroid-based 3D density filter, Heaviside projection continuation, graph-distance erosion/dilation, active-element body-fitted static reanalysis, an assembled first-frequency objective/gate, and an optional linearized x-compressive buckling objective/gate. Frequency sensitivities use assembled generalized-eigenvalue differentiation; buckling sensitivities use bounded central finite differences of the re-solved initial-stress factor.".to_string(),
            "Its geometric matrix uses the full negative stress tensor when it is positive-semidefinite and reports a sigma_xx fallback for mixed-sign stress states; nonlinear material/contact, post-buckling, general indefinite-stress geometric stiffness, and accelerator dispatch remain outside this bounded reference path.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the assembled flat-shell shared-factorization path.
    /// Provenance for the 3-D robust morphology/uncertainty reanalysis path.
    pub fn cpu_solid3d_robust_topology_reference() -> Self {
        let mut provenance = Self::cpu_solid3d_topology_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.solid3d-robust-topology.v1".to_string();
        provenance.limitations.extend([
            "Robust execution reuses one nominal assembled 3-D SIMP design and re-solves nominal, graph-eroded, and graph-dilated morphologies for explicit multiplicative load and Young's-modulus scenarios.".to_string(),
            "The uncertainty envelope is a deterministic static compliance gate, not a calibrated probabilistic distribution, nonlinear manufacturing process, or external-solver parity claim.".to_string(),
        ]);
        provenance
    }

    pub fn cpu_shell_multi_load_reference() -> Self {
        let mut provenance = Self::cpu_shell_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.shell-tri3-cst-dkt-multi-load.v1".to_string();
        provenance.limitations.extend([
            "Multi-load execution is limited to flat P1 triangular CST membrane plus DKT bending with one common clamp and pressure cases assembled as separate RHS vectors.".to_string(),
            "Curved/nonlinear/contact shell behavior, shell stress recovery, quadratic or matrix-free block solves, external parity, CUDA, Metal, and WASM execution are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the residual-certified modal Galerkin beam ROM path.
    pub fn cpu_rom_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.structural-rom-beam.v1".to_string();
        provenance.limitations.extend([
            "ROM execution is limited to linear Euler/Timoshenko beam static queries with a reference modal basis and nearby Young's-modulus scaling.".to_string(),
            "A full-order snapshot is admitted when the exact reduced residual exceeds the configured tolerance; the result reports basis enrichment and both residuals.".to_string(),
            "Nonlinear, contact, topology, distributed, GPU/WASM, and unvalidated cross-solver ROM parity are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the bounded Basquin/Miner fatigue evidence path.
    pub fn cpu_fatigue_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.fatigue-evidence.v1".to_string();
        provenance.limitations.extend([
            "Fatigue execution is a scalar stress-life block calculation using Basquin and Miner accumulation; it does not recover stresses from a structural FEM load history.".to_string(),
            "Mean-stress correction, multiaxial damage, crack growth, and validated material S-N data are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the bounded P1 plane-stress FEM fatigue path.
    pub fn cpu_fem_fatigue_2d_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.fem-fatigue-2d.v1".to_string();
        provenance.limitations.extend([
            "FEM fatigue execution uses two P1 plane-stress load states per block and a conservative maximum-element stress amplitude.".to_string(),
            "Basquin/Miner accumulation is a scalar S-N evidence model; mean-stress correction, multiaxial damage, crack growth, and calibrated material data are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the bounded unilateral penalty contact evidence path.
    pub fn cpu_contact_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.contact-evidence.v1".to_string();
        provenance.limitations.extend([
            "Contact execution is a one-degree-of-freedom frictionless penalty/Newton reference; it is not a surface-to-surface FEM contact solve.".to_string(),
            "Friction, large-deformation contact, mortar constraints, and 3D contact parity are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the actual P1 tetrahedral hyperelastic surface-contact
    /// reference backed by Sonium FEM's augmented-Lagrangian solver.
    pub fn cpu_hyperelastic_3d_contact_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.hyperelastic-3d-contact-evidence.v1".to_string();
        provenance.limitations.extend([
            "Contact execution is displacement-controlled P1 tetrahedral Neo-Hookean opposing-face contact; frictional contact, self-contact, and product-scale contact are not claimed."
                .to_string(),
            "The reference validates Newton/Uzawa engagement and bounded penetration diagnostics; it is not general product-scale contact or post-buckling parity."
                .to_string(),
        ]);
        provenance
    }

    /// Provenance for the bounded force-controlled opposing-face contact path.
    pub fn cpu_hyperelastic_3d_force_contact_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.hyperelastic-3d-force-contact.v1".to_string();
        provenance.limitations.extend([
            "Force-controlled contact is limited to two disconnected P1 tetrahedral Neo-Hookean blocks with a guided upper external face and a distributed normal load."
                .to_string(),
            "The reference validates monotone load continuation, Newton/Uzawa convergence, and bounded penetration diagnostics; friction, self-contact, post-buckling, calibrated materials, and product-scale parity are not claimed."
                .to_string(),
        ]);
        provenance
    }

    /// Provenance for the bounded prescribed 3-D Coulomb contact reference.
    pub fn cpu_hyperelastic_3d_friction_contact_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.hyperelastic-3d-friction-contact.v1".to_string();
        provenance.limitations.extend([
            "Prescribed two-block P1 tetrahedral contact isolates node-to-surface Coulomb traction and force conservation; it is not a solved product-scale frictional model.".to_string(),
            "Projection derivatives, wear, self-contact dynamics, calibrated friction data, and accelerator parity are not claimed.".to_string(),
        ]);
        provenance
    }

    /// Provenance for the bounded P1 tetrahedral force-controlled reference.
    pub fn cpu_hyperelastic_3d_force_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.hyperelastic-3d-force.v1".to_string();
        provenance.limitations.extend([
            "Force control uses a guided upper face and a distributed load vector on a small P1 tetrahedral block.".to_string(),
            "The reference is monotone static CPU evidence; force-controlled contact, post-buckling, calibrated materials, and product-scale parity are not claimed.".to_string(),
        ]);
        provenance
    }

    pub fn cpu_reliability_reference() -> Self {
        let mut provenance = Self::cpu_reference();
        provenance.capability_id =
            "sonium-structural.cpu-reference.f64.reliability-evidence.v1".to_string();
        provenance.limitations.extend([
            "Reliability execution is empirical sample counting over supplied load/resistance pairs; it is not a probabilistic FEM uncertainty propagation or calibrated distribution model.".to_string(),
            "Correlation, sampling convergence, FORM/SORM validation, and certified probability bounds are not claimed.".to_string(),
        ]);
        provenance
    }
}

/// Return the structural backend capability catalogue.
pub fn structural_backend_capabilities() -> Vec<StructuralBackendCapability> {
    vec![
        StructuralBackendCapability {
            schema_version: STRUCTURAL_CAPABILITY_SCHEMA_VERSION,
            backend: StructuralBackend::Cpu,
            integrated: true,
            execution_mode: StructuralExecutionMode::CpuReference,
            precision_policies: vec![StructuralPrecisionPolicy::F64],
            analysis_families: vec![
                StructuralAnalysisFamily::Static,
                StructuralAnalysisFamily::Axisymmetric,
                StructuralAnalysisFamily::Viscoelastic,
                StructuralAnalysisFamily::Shell,
                StructuralAnalysisFamily::Modal,
                StructuralAnalysisFamily::Harmonic,
                StructuralAnalysisFamily::Thermal,
                StructuralAnalysisFamily::Solid3D,
                StructuralAnalysisFamily::Composite,
                StructuralAnalysisFamily::Buckling,
                StructuralAnalysisFamily::Transient,
                StructuralAnalysisFamily::RandomVibration,
                StructuralAnalysisFamily::Fatigue,
                StructuralAnalysisFamily::Reliability,
                StructuralAnalysisFamily::Contact,
                StructuralAnalysisFamily::Plasticity,
                StructuralAnalysisFamily::Hyperelastic,
                StructuralAnalysisFamily::PhaseFieldDamage,
                StructuralAnalysisFamily::ShapeOptimization,
                StructuralAnalysisFamily::TopologyOptimization,
                StructuralAnalysisFamily::Uncertainty,
                StructuralAnalysisFamily::ReducedOrder,
            ],
            limitations: vec![
                "Parallelism is provided by the CPU-side FEM and solver layers; this crate does not select a device backend.".to_string(),
                "Analysis-family entries denote bounded reference or reduced executions available in this crate; family presence is not a claim of general 3D, nonlinear, fatigue, contact, or external-solver fidelity.".to_string(),
            ],
        },
        cuda_backend_capability(),
        unsupported_backend(
            StructuralBackend::Metal,
            "Metal compute kernels exist below the structural layer, but structural assembly and study dispatch are not wired to Metal.",
        ),
        unsupported_backend(
            StructuralBackend::Wasm,
            "WASM compute aliases exist below the structural layer; this crate exposes portable reduced previews for WASM visualization, but does not claim a WASM structural solve.",
        ),
    ]
}

#[cfg(feature = "cuda")]
fn cuda_backend_capability() -> StructuralBackendCapability {
    StructuralBackendCapability {
        schema_version: STRUCTURAL_CAPABILITY_SCHEMA_VERSION,
        backend: StructuralBackend::Cuda,
        integrated: true,
        execution_mode: StructuralExecutionMode::Accelerator,
        precision_policies: vec![StructuralPrecisionPolicy::F64],
        analysis_families: vec![StructuralAnalysisFamily::Static],
        limitations: StructuralExecutionProvenance::cuda_static_bracket_reference().limitations,
    }
}

#[cfg(not(feature = "cuda"))]
fn cuda_backend_capability() -> StructuralBackendCapability {
    unsupported_backend(
        StructuralBackend::Cuda,
        "The bounded P1/f64 static bracket CUDA slice is available only when this crate is built with the `cuda` feature.",
    )
}

fn unsupported_backend(
    backend: StructuralBackend,
    limitation: &str,
) -> StructuralBackendCapability {
    StructuralBackendCapability {
        schema_version: STRUCTURAL_CAPABILITY_SCHEMA_VERSION,
        backend,
        integrated: false,
        execution_mode: StructuralExecutionMode::Unsupported,
        precision_policies: Vec::new(),
        analysis_families: Vec::new(),
        limitations: vec![limitation.to_string()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_distinguishes_reference_and_unwired_backends() {
        let catalogue = structural_backend_capabilities();
        assert_eq!(catalogue.len(), 4);
        let cpu = catalogue
            .iter()
            .find(|capability| capability.backend == StructuralBackend::Cpu)
            .unwrap();
        assert!(cpu.integrated);
        assert_eq!(cpu.execution_mode, StructuralExecutionMode::CpuReference);
        assert!(cpu.supports(
            StructuralAnalysisFamily::Modal,
            StructuralPrecisionPolicy::F64
        ));
        assert!(cpu.supports(
            StructuralAnalysisFamily::Viscoelastic,
            StructuralPrecisionPolicy::F64
        ));
        assert!(cpu.supports(
            StructuralAnalysisFamily::Axisymmetric,
            StructuralPrecisionPolicy::F64
        ));
        assert!(
            cpu.limitations
                .iter()
                .any(|limitation| limitation.contains("bounded reference or reduced"))
        );
        assert!(!cpu.supports(
            StructuralAnalysisFamily::Modal,
            StructuralPrecisionPolicy::F32
        ));

        for backend in [
            StructuralBackend::Cuda,
            StructuralBackend::Metal,
            StructuralBackend::Wasm,
        ] {
            let capability = catalogue
                .iter()
                .find(|capability| capability.backend == backend)
                .unwrap();
            assert!(!capability.integrated);
            assert_eq!(
                capability.execution_mode,
                StructuralExecutionMode::Unsupported
            );
            assert!(capability.precision_policies.is_empty());
            assert!(!capability.limitations.is_empty());
        }
    }

    #[test]
    fn cpu_reference_provenance_is_versioned_and_explicit() {
        let provenance = StructuralExecutionProvenance::cpu_reference();
        assert_eq!(
            provenance.schema_version,
            STRUCTURAL_CAPABILITY_SCHEMA_VERSION
        );
        assert_eq!(provenance.backend, StructuralBackend::Cpu);
        assert_eq!(provenance.precision_policy, StructuralPrecisionPolicy::F64);
        assert!(provenance.capability_verified);
        assert!(!provenance.capability_id.is_empty());
        assert!(
            provenance
                .limitations
                .iter()
                .any(|limitation| limitation.contains("CUDA"))
        );
    }

    #[test]
    fn matrix_free_reference_provenance_identifies_path_and_limits() {
        let provenance = StructuralExecutionProvenance::cpu_matrix_free_p2_reference();
        assert_eq!(
            provenance.capability_id,
            "sonium-structural.cpu-reference.f64.matrix-free-p2.v1"
        );
        assert!(
            provenance
                .limitations
                .iter()
                .any(|limitation| limitation.contains("diagonal FGMRES"))
        );
    }

    #[test]
    fn bracket_shape_reference_provenance_identifies_objective_scope() {
        let provenance = StructuralExecutionProvenance::cpu_bracket_shape_reference();
        assert_eq!(
            provenance.capability_id,
            "sonium-structural.cpu-reference.f64.bracket-shape.v1"
        );
        assert!(
            provenance
                .limitations
                .iter()
                .any(|limitation| { limitation.contains("first-bending-frequency") })
        );
        assert!(
            provenance
                .limitations
                .iter()
                .any(|limitation| { limitation.contains("general 3D shape optimization") })
        );
    }
}
