//! Mesh lineage and conservative operator-cache invalidation rules.
//!
//! BEM operators encode both mesh topology and geometry.  Reusing a cluster
//! tree, compressed matrix, preconditioner, or recycled Krylov space after an
//! unclassified mesh update can silently solve the wrong boundary problem.
//! This module gives callers a deterministic snapshot and an explicit plan for
//! every supported update class.

use thiserror::Error;

use crate::core::bem_solver::{AssemblyMethod, BemPreconditioner, SolverMethod};
use crate::core::conventions::BemConvention;
use crate::core::types::Mesh;
use crate::core::types::{PhysicsParams, PhysicsParamsError};

/// Deterministic identity of a BEM mesh at one point in its lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshLineage {
    /// Hash of node count, element connectivity, active/evaluation status, and
    /// DOF addressing. Equal values are necessary before an operator topology
    /// can be reused.
    pub topology_fingerprint: u64,
    /// Hash of all node-coordinate bit patterns. This is intentionally exact:
    /// geometric reuse is decided separately by an explicit displacement
    /// tolerance rather than by a lossy hash comparison.
    pub geometry_fingerprint: u64,
    /// Number of mesh nodes represented by the snapshot.
    pub node_count: usize,
    /// Number of mesh elements represented by the snapshot.
    pub element_count: usize,
}

/// Solver and formulation inputs that determine whether BEM state may be
/// reused.  This is intentionally a value snapshot, not a cache: current
/// solvers do not retain an H/FMM cache between calls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BemReuseContext {
    /// Geometry and topology identity of the discretisation.
    pub mesh: MeshLineage,
    /// Frequency of the assembled operator in hertz.
    pub frequency_hz: f64,
    /// Sound speed used to form the wavenumber.
    pub speed_of_sound: f64,
    /// Density used in pressure and impedance scaling.
    pub density: f64,
    /// Green-function sign from the physical convention.
    pub harmonic_factor: f64,
    /// Interior/exterior orientation marker.
    pub tau: f64,
    /// Explicit exchange convention, including the time sign and normal
    /// orientation that are not all represented by scalar physics fields.
    pub convention: BemConvention,
    /// Matrix assembly/compression backend.
    pub assembly_method: AssemblyMethod,
    /// Linear solve method.
    pub solver_method: SolverMethod,
    /// Preconditioner selection.
    pub preconditioner: BemPreconditioner,
    /// Requested iterative residual tolerance.
    pub solver_tolerance: f64,
    /// Burton--Miller coupling scale used to assemble the operator.
    pub beta_scale: f64,
}

impl BemReuseContext {
    /// Capture all inputs that must agree before a BEM cache can be used.
    pub fn capture(
        mesh: &Mesh,
        physics: &PhysicsParams,
        convention: BemConvention,
        assembly_method: AssemblyMethod,
        solver_method: SolverMethod,
        preconditioner: BemPreconditioner,
        solver_tolerance: f64,
        beta_scale: f64,
    ) -> Result<Self, BemReuseError> {
        validate_coordinates(mesh, "current").map_err(|error| match error {
            MeshLineageError::NonFiniteCoordinate { which, node, axis } => {
                BemReuseError::NonFiniteCoordinate { which, node, axis }
            }
            MeshLineageError::InvalidTolerance => {
                unreachable!("coordinate validation has no tolerance")
            }
        })?;
        physics.validate()?;
        if !solver_tolerance.is_finite() || solver_tolerance <= 0.0 {
            return Err(BemReuseError::InvalidSolverTolerance);
        }
        if !beta_scale.is_finite() || beta_scale <= 0.0 {
            return Err(BemReuseError::InvalidBetaScale);
        }
        if !matches!(convention.time_sign, -1 | 1)
            || !matches!(convention.spatial_sign, -1 | 1)
            || f64::from(convention.spatial_sign) != physics.harmonic_factor
        {
            return Err(BemReuseError::IncompatibleConvention);
        }
        Ok(Self {
            mesh: MeshLineage::capture(mesh),
            frequency_hz: physics.frequency,
            speed_of_sound: physics.speed_of_sound,
            density: physics.density,
            harmonic_factor: physics.harmonic_factor,
            tau: physics.tau,
            convention,
            assembly_method,
            solver_method,
            preconditioner,
            solver_tolerance,
            beta_scale,
        })
    }
}

/// Quantitative policy for reusing frequency-dependent BEM solver state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BemReusePolicy {
    /// Largest permitted `|f_new - f_old| / max(f_new, f_old)` for retaining
    /// frequency-dependent interpolation, preconditioner, or recycled Krylov
    /// state. `0.0` permits only an exact frequency match.
    pub max_relative_frequency_delta: f64,
}

impl Default for BemReusePolicy {
    fn default() -> Self {
        Self {
            max_relative_frequency_delta: 0.0,
        }
    }
}

/// Individual BEM cache components eligible for reuse by a caller-owned cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BemCacheReuse {
    /// Reuse the geometry-only cluster tree.
    pub cluster_tree: bool,
    /// Reuse admissibility buckets derived from that tree.
    pub admissibility_buckets: bool,
    /// Reuse interpolation/translation data.
    pub interpolation_operators: bool,
    /// Reuse the configured preconditioner.
    pub preconditioner: bool,
    /// Reuse a recycled Krylov/deflation subspace.
    pub recycled_krylov: bool,
}

impl BemCacheReuse {
    const fn none() -> Self {
        Self {
            cluster_tree: false,
            admissibility_buckets: false,
            interpolation_operators: false,
            preconditioner: false,
            recycled_krylov: false,
        }
    }

    const fn geometry_only() -> Self {
        Self {
            cluster_tree: true,
            admissibility_buckets: true,
            interpolation_operators: true,
            preconditioner: false,
            recycled_krylov: false,
        }
    }

    const fn all() -> Self {
        Self {
            cluster_tree: true,
            admissibility_buckets: true,
            interpolation_operators: true,
            preconditioner: true,
            recycled_krylov: true,
        }
    }
}

/// Why a BEM cache decision did not retain every component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BemReuseRejection {
    /// Mesh geometry or topology changed beyond the caller's mesh policy.
    MeshChanged,
    /// The assembly backend changed.
    AssemblyMethodChanged,
    /// Medium, formulation, or exchange convention changed.
    PhysicsOrConventionChanged,
    /// Solver, preconditioner, tolerance, or Burton--Miller scale changed.
    SolverConfigurationChanged,
    /// Adjacent frequencies exceed the configured quantitative bound.
    FrequencyDeltaExceeded {
        /// Measured normalized frequency difference.
        relative_delta: f64,
        /// Policy limit supplied by the caller.
        maximum: f64,
    },
}

/// Explicit eligibility result for one prior/current BEM solve pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BemReuseDecision {
    /// Components a caller-owned cache may retain.
    pub reuse: BemCacheReuse,
    /// Components that must be rebuilt before the next solve.
    pub invalidation: BemCacheInvalidation,
    /// `None` only when every component is eligible for reuse.
    pub rejection: Option<BemReuseRejection>,
}

/// Errors raised while constructing or comparing a BEM reuse identity.
#[derive(Debug, Error, PartialEq)]
pub enum BemReuseError {
    /// The physical inputs could not describe a valid BEM operator.
    #[error("invalid BEM physics for cache reuse: {0}")]
    Physics(#[from] PhysicsParamsError),
    /// The solve tolerance is not a finite positive number.
    #[error("BEM cache-reuse solver tolerance must be finite and positive")]
    InvalidSolverTolerance,
    /// The Burton--Miller scale is not a finite positive number.
    #[error("BEM cache-reuse Burton--Miller scale must be finite and positive")]
    InvalidBetaScale,
    /// The physical sign and full exchange convention disagree.
    #[error("BEM cache-reuse convention is malformed or disagrees with physics")]
    IncompatibleConvention,
    /// The frequency threshold is not a finite non-negative number.
    #[error("BEM cache-reuse frequency threshold must be finite and non-negative")]
    InvalidFrequencyTolerance,
    /// A mesh-update plan was not produced for the supplied identities.
    #[error("BEM cache-reuse mesh-update plan does not match the supplied contexts")]
    MeshPlanMismatch,
    /// A coordinate is not usable as a geometric state.
    #[error("mesh {which} node {node} coordinate {axis} is non-finite")]
    NonFiniteCoordinate {
        /// Whether the bad coordinate came from the previous or current mesh.
        which: &'static str,
        /// Node index.
        node: usize,
        /// Coordinate axis.
        axis: usize,
    },
}

impl MeshLineage {
    /// Capture a deterministic snapshot of a mesh's topology and geometry.
    pub fn capture(mesh: &Mesh) -> Self {
        Self {
            topology_fingerprint: topology_fingerprint(mesh),
            geometry_fingerprint: geometry_fingerprint(mesh),
            node_count: mesh.nodes.len(),
            element_count: mesh.elements.len(),
        }
    }
}

/// Explicit operator-cache actions required after a mesh update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BemCacheInvalidation {
    /// Rebuild the cluster tree and spatial admissibility buckets.
    pub rebuild_cluster_tree: bool,
    /// Recompute near/far admissibility buckets while retaining the cluster
    /// hierarchy itself.
    pub rebuild_admissibility_buckets: bool,
    /// Recompute frequency-dependent interpolation or translation data.
    pub rebuild_interpolation_operators: bool,
    /// Rebuild compressed or dense operator entries.
    pub rebuild_operator: bool,
    /// Rebuild any geometry-dependent preconditioner.
    pub rebuild_preconditioner: bool,
    /// Discard recycled Krylov/deflation spaces.
    pub discard_recycled_krylov: bool,
    /// Rebuild DOF indexing and block layouts.
    pub rebuild_dof_layout: bool,
}

impl BemCacheInvalidation {
    const fn none() -> Self {
        Self {
            rebuild_cluster_tree: false,
            rebuild_admissibility_buckets: false,
            rebuild_interpolation_operators: false,
            rebuild_operator: false,
            rebuild_preconditioner: false,
            discard_recycled_krylov: false,
            rebuild_dof_layout: false,
        }
    }

    const fn geometry() -> Self {
        Self {
            rebuild_cluster_tree: true,
            rebuild_admissibility_buckets: true,
            rebuild_interpolation_operators: true,
            rebuild_operator: true,
            rebuild_preconditioner: true,
            discard_recycled_krylov: true,
            rebuild_dof_layout: false,
        }
    }

    const fn topology() -> Self {
        Self {
            rebuild_dof_layout: true,
            ..Self::geometry()
        }
    }

    const fn frequency() -> Self {
        Self {
            rebuild_cluster_tree: false,
            rebuild_admissibility_buckets: true,
            rebuild_interpolation_operators: true,
            rebuild_operator: true,
            rebuild_preconditioner: true,
            discard_recycled_krylov: true,
            rebuild_dof_layout: false,
        }
    }

    const fn solver() -> Self {
        Self {
            rebuild_cluster_tree: false,
            rebuild_admissibility_buckets: false,
            rebuild_interpolation_operators: false,
            rebuild_operator: false,
            rebuild_preconditioner: true,
            discard_recycled_krylov: true,
            rebuild_dof_layout: false,
        }
    }

    const fn operator_only() -> Self {
        Self {
            rebuild_cluster_tree: false,
            rebuild_admissibility_buckets: false,
            rebuild_interpolation_operators: false,
            rebuild_operator: true,
            rebuild_preconditioner: false,
            discard_recycled_krylov: false,
            rebuild_dof_layout: false,
        }
    }
}

/// Classification of a transition between two BEM meshes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeshUpdatePlan {
    /// Coordinates are identical or differ by no more than the caller's
    /// explicit exact-reuse tolerance; all cache components are valid.
    Unchanged {
        /// Snapshots of the identical mesh state.
        lineage: MeshLineage,
        /// Cache operations required by the update.
        invalidation: BemCacheInvalidation,
    },
    /// Connectivity and DOF layout are identical, but geometric displacement
    /// exceeded the caller's exact-reuse tolerance.
    GeometryChanged {
        /// Prior mesh identity.
        previous: MeshLineage,
        /// New mesh identity.
        current: MeshLineage,
        /// Largest Euclidean displacement of a corresponding node.
        max_node_displacement: f64,
        /// Conservative cache operations required by the update.
        invalidation: BemCacheInvalidation,
    },
    /// Connectivity, element activity, DOF addressing, or mesh cardinality
    /// changed, so no BEM operator state may be reused.
    TopologyChanged {
        /// Prior mesh identity.
        previous: MeshLineage,
        /// New mesh identity.
        current: MeshLineage,
        /// Cache operations required by the update.
        invalidation: BemCacheInvalidation,
    },
}

impl MeshUpdatePlan {
    /// Return the explicit cache operations required before another solve.
    pub const fn invalidation(self) -> BemCacheInvalidation {
        match self {
            Self::Unchanged { invalidation, .. }
            | Self::GeometryChanged { invalidation, .. }
            | Self::TopologyChanged { invalidation, .. } => invalidation,
        }
    }
}

/// Errors raised while classifying a mesh update.
#[derive(Debug, Error, PartialEq)]
pub enum MeshLineageError {
    /// The tolerance cannot safely classify an exact versus geometric update.
    #[error("mesh geometry-reuse tolerance must be finite and non-negative")]
    InvalidTolerance,
    /// A coordinate is not usable as a geometric state.
    #[error("mesh {which} node {node} coordinate {axis} is non-finite")]
    NonFiniteCoordinate {
        /// Whether the bad coordinate came from the previous or current mesh.
        which: &'static str,
        /// Node index.
        node: usize,
        /// Coordinate axis.
        axis: usize,
    },
}

/// Classify a BEM mesh transition and state exactly which cached data must be
/// invalidated before the next assembly or solve.
///
/// A topology-preserving deformation always rebuilds spatial and numeric
/// caches once it exceeds `exact_reuse_tolerance`: a cluster partition that is
/// merely plausible after deformation is not a proof that its approximation or
/// preconditioner remains valid. Callers that want to accept motion below a
/// physical tolerance must explicitly set that tolerance and retain it in
/// their provenance.
pub fn classify_mesh_update(
    previous_mesh: &Mesh,
    current_mesh: &Mesh,
    exact_reuse_tolerance: f64,
) -> Result<MeshUpdatePlan, MeshLineageError> {
    if !exact_reuse_tolerance.is_finite() || exact_reuse_tolerance < 0.0 {
        return Err(MeshLineageError::InvalidTolerance);
    }
    validate_coordinates(previous_mesh, "previous")?;
    validate_coordinates(current_mesh, "current")?;

    let previous = MeshLineage::capture(previous_mesh);
    let current = MeshLineage::capture(current_mesh);
    if previous.topology_fingerprint != current.topology_fingerprint {
        return Ok(MeshUpdatePlan::TopologyChanged {
            previous,
            current,
            invalidation: BemCacheInvalidation::topology(),
        });
    }

    let max_node_displacement = previous_mesh
        .nodes
        .iter()
        .zip(&current_mesh.nodes)
        .map(|(previous, current)| {
            previous
                .iter()
                .zip(current)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0_f64, f64::max);
    if current.geometry_fingerprint == previous.geometry_fingerprint
        || max_node_displacement <= exact_reuse_tolerance
    {
        return Ok(MeshUpdatePlan::Unchanged {
            lineage: current,
            invalidation: BemCacheInvalidation::none(),
        });
    }
    Ok(MeshUpdatePlan::GeometryChanged {
        previous,
        current,
        max_node_displacement,
        invalidation: BemCacheInvalidation::geometry(),
    })
}

/// Decide which caller-owned BEM caches may be reused for the next solve.
///
/// This is deliberately stricter than a similarity heuristic. Mesh changes
/// are accepted only when [`classify_mesh_update`] says they are within its
/// explicit exact-reuse tolerance; medium, convention, assembly, and solver
/// changes must match exactly. A changed frequency always rebuilds numerical
/// operator entries, but may retain frequency-sensitive state only when its
/// relative distance meets `policy`.
///
/// The result is an eligibility contract. The present BEM solver does not yet
/// own a persistent H/FMM cache, so callers must still record post-reuse
/// residual/accuracy evidence when they materialize a cache from this plan.
pub fn assess_bem_cache_reuse(
    previous: BemReuseContext,
    current: BemReuseContext,
    mesh_plan: MeshUpdatePlan,
    policy: BemReusePolicy,
) -> Result<BemReuseDecision, BemReuseError> {
    if !policy.max_relative_frequency_delta.is_finite() || policy.max_relative_frequency_delta < 0.0
    {
        return Err(BemReuseError::InvalidFrequencyTolerance);
    }
    if !mesh_plan_matches(previous.mesh, current.mesh, mesh_plan) {
        return Err(BemReuseError::MeshPlanMismatch);
    }

    if !matches!(mesh_plan, MeshUpdatePlan::Unchanged { .. }) {
        return Ok(BemReuseDecision {
            reuse: BemCacheReuse::none(),
            invalidation: mesh_plan.invalidation(),
            rejection: Some(BemReuseRejection::MeshChanged),
        });
    }
    if previous.assembly_method != current.assembly_method {
        return Ok(BemReuseDecision {
            reuse: BemCacheReuse::none(),
            invalidation: BemCacheInvalidation::geometry(),
            rejection: Some(BemReuseRejection::AssemblyMethodChanged),
        });
    }
    if !same_physics_and_convention(previous, current) {
        return Ok(BemReuseDecision {
            reuse: BemCacheReuse::none(),
            invalidation: BemCacheInvalidation::geometry(),
            rejection: Some(BemReuseRejection::PhysicsOrConventionChanged),
        });
    }
    if !same_solver_configuration(previous, current) {
        return Ok(BemReuseDecision {
            reuse: BemCacheReuse::geometry_only(),
            invalidation: BemCacheInvalidation::solver(),
            rejection: Some(BemReuseRejection::SolverConfigurationChanged),
        });
    }

    let relative_delta = relative_frequency_delta(previous.frequency_hz, current.frequency_hz);
    if relative_delta == 0.0 {
        return Ok(BemReuseDecision {
            reuse: BemCacheReuse::all(),
            invalidation: BemCacheInvalidation::none(),
            rejection: None,
        });
    }
    if relative_delta <= policy.max_relative_frequency_delta {
        return Ok(BemReuseDecision {
            reuse: BemCacheReuse::all(),
            invalidation: BemCacheInvalidation::operator_only(),
            rejection: None,
        });
    }
    Ok(BemReuseDecision {
        reuse: BemCacheReuse::geometry_only(),
        invalidation: BemCacheInvalidation::frequency(),
        rejection: Some(BemReuseRejection::FrequencyDeltaExceeded {
            relative_delta,
            maximum: policy.max_relative_frequency_delta,
        }),
    })
}

/// Symmetric relative distance for two positive BEM frequencies.
pub fn relative_frequency_delta(previous_hz: f64, current_hz: f64) -> f64 {
    (current_hz - previous_hz).abs() / previous_hz.abs().max(current_hz.abs())
}

fn mesh_plan_matches(previous: MeshLineage, current: MeshLineage, plan: MeshUpdatePlan) -> bool {
    match plan {
        MeshUpdatePlan::Unchanged { lineage, .. } => {
            previous.topology_fingerprint == current.topology_fingerprint && current == lineage
        }
        MeshUpdatePlan::GeometryChanged {
            previous: planned_previous,
            current: planned_current,
            ..
        }
        | MeshUpdatePlan::TopologyChanged {
            previous: planned_previous,
            current: planned_current,
            ..
        } => previous == planned_previous && current == planned_current,
    }
}

fn same_physics_and_convention(previous: BemReuseContext, current: BemReuseContext) -> bool {
    previous.speed_of_sound.to_bits() == current.speed_of_sound.to_bits()
        && previous.density.to_bits() == current.density.to_bits()
        && previous.harmonic_factor.to_bits() == current.harmonic_factor.to_bits()
        && previous.tau.to_bits() == current.tau.to_bits()
        && previous.convention == current.convention
}

fn same_solver_configuration(previous: BemReuseContext, current: BemReuseContext) -> bool {
    previous.solver_method == current.solver_method
        && previous.preconditioner == current.preconditioner
        && previous.solver_tolerance.to_bits() == current.solver_tolerance.to_bits()
        && previous.beta_scale.to_bits() == current.beta_scale.to_bits()
}

fn validate_coordinates(mesh: &Mesh, which: &'static str) -> Result<(), MeshLineageError> {
    for (node, coordinate) in mesh.nodes.iter().enumerate() {
        for (axis, value) in coordinate.iter().enumerate() {
            if !value.is_finite() {
                return Err(MeshLineageError::NonFiniteCoordinate { which, node, axis });
            }
        }
    }
    Ok(())
}

fn topology_fingerprint(mesh: &Mesh) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    mix(&mut hash, mesh.nodes.len() as u64);
    mix(&mut hash, mesh.elements.len() as u64);
    for element in &mesh.elements {
        mix(&mut hash, element.connectivity.len() as u64);
        for &node in &element.connectivity {
            mix(&mut hash, node as u64);
        }
        mix(&mut hash, u64::from(element.property.is_evaluation()));
        mix(&mut hash, element.dof_addresses.len() as u64);
        for &dof in &element.dof_addresses {
            mix(&mut hash, dof as u64);
        }
    }
    hash
}

fn geometry_fingerprint(mesh: &Mesh) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for coordinate in &mesh.nodes {
        for value in coordinate {
            mix(&mut hash, value.to_bits());
        }
    }
    hash
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn mix(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(FNV_PRIME);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::conventions::ACOUSTIC_CONVENTION;
    use crate::core::mesh::generators::generate_sphere_mesh;

    fn reuse_context(mesh: &Mesh, frequency: f64) -> BemReuseContext {
        let physics =
            PhysicsParams::try_new(frequency, 343.0, 1.21, false).expect("test physics is valid");
        BemReuseContext::capture(
            mesh,
            &physics,
            ACOUSTIC_CONVENTION,
            AssemblyMethod::Hmatrix,
            SolverMethod::Gmres,
            BemPreconditioner::Jacobi,
            1.0e-6,
            4.0,
        )
        .expect("test context is valid")
    }

    #[test]
    fn unchanged_mesh_reuses_all_bem_cache_state() {
        let mesh = generate_sphere_mesh(0.1, 4, 8);
        let plan = classify_mesh_update(&mesh, &mesh, 0.0).expect("valid mesh lineage");
        assert!(matches!(plan, MeshUpdatePlan::Unchanged { .. }));
        assert_eq!(plan.invalidation(), BemCacheInvalidation::none());
    }

    #[test]
    fn geometry_change_rebuilds_numeric_and_spatial_state_but_not_dof_layout() {
        let previous = generate_sphere_mesh(0.1, 4, 8);
        let mut current = previous.clone();
        current.nodes[0][0] += 1.0e-3;
        let plan = classify_mesh_update(&previous, &current, 1.0e-6).expect("valid deformation");
        match plan {
            MeshUpdatePlan::GeometryChanged {
                max_node_displacement,
                invalidation,
                ..
            } => {
                assert!(max_node_displacement >= 1.0e-3);
                assert!(invalidation.rebuild_cluster_tree);
                assert!(invalidation.rebuild_operator);
                assert!(invalidation.rebuild_preconditioner);
                assert!(invalidation.discard_recycled_krylov);
                assert!(!invalidation.rebuild_dof_layout);
            }
            other => panic!("expected geometry invalidation, got {other:?}"),
        }
    }

    #[test]
    fn topology_change_rebuilds_every_bem_cache_component() {
        let previous = generate_sphere_mesh(0.1, 4, 8);
        let mut current = previous.clone();
        current.elements[0].connectivity.swap(0, 1);
        let plan = classify_mesh_update(&previous, &current, 1.0).expect("valid topology change");
        match plan {
            MeshUpdatePlan::TopologyChanged { invalidation, .. } => {
                assert!(invalidation.rebuild_cluster_tree);
                assert!(invalidation.rebuild_operator);
                assert!(invalidation.rebuild_preconditioner);
                assert!(invalidation.discard_recycled_krylov);
                assert!(invalidation.rebuild_dof_layout);
            }
            other => panic!("expected topology invalidation, got {other:?}"),
        }
    }

    #[test]
    fn lineage_rejects_non_finite_geometry_and_invalid_tolerance() {
        let mesh = generate_sphere_mesh(0.1, 4, 8);
        assert_eq!(
            classify_mesh_update(&mesh, &mesh, -1.0),
            Err(MeshLineageError::InvalidTolerance)
        );
        let mut malformed = mesh.clone();
        malformed.nodes[0][1] = f64::NAN;
        assert!(matches!(
            classify_mesh_update(&mesh, &malformed, 0.0),
            Err(MeshLineageError::NonFiniteCoordinate {
                which: "current",
                node: 0,
                axis: 1,
            })
        ));
    }

    #[test]
    fn cache_reuse_requires_a_matching_mesh_plan_and_exact_context() {
        let mesh = generate_sphere_mesh(0.1, 4, 8);
        let context = reuse_context(&mesh, 100.0);
        let plan = classify_mesh_update(&mesh, &mesh, 0.0).expect("valid mesh plan");
        let decision = assess_bem_cache_reuse(context, context, plan, BemReusePolicy::default())
            .expect("matching identity is valid");
        assert_eq!(decision.reuse, BemCacheReuse::all());
        assert_eq!(decision.invalidation, BemCacheInvalidation::none());
        assert_eq!(decision.rejection, None);

        let other_mesh = generate_sphere_mesh(0.2, 4, 8);
        let other_context = reuse_context(&other_mesh, 100.0);
        assert_eq!(
            assess_bem_cache_reuse(context, other_context, plan, BemReusePolicy::default()),
            Err(BemReuseError::MeshPlanMismatch)
        );
    }

    #[test]
    fn nearby_frequency_retains_frequency_state_but_rebuilds_operator_entries() {
        let mesh = generate_sphere_mesh(0.1, 4, 8);
        let previous = reuse_context(&mesh, 100.0);
        let current = reuse_context(&mesh, 105.0);
        let plan = classify_mesh_update(&mesh, &mesh, 0.0).expect("valid mesh plan");
        let decision = assess_bem_cache_reuse(
            previous,
            current,
            plan,
            BemReusePolicy {
                max_relative_frequency_delta: 0.1,
            },
        )
        .expect("valid reuse comparison");

        assert_eq!(decision.reuse, BemCacheReuse::all());
        assert!(decision.invalidation.rebuild_operator);
        assert!(!decision.invalidation.rebuild_cluster_tree);
        assert!(!decision.invalidation.rebuild_preconditioner);
        assert_eq!(decision.rejection, None);
    }

    #[test]
    fn explicitly_tolerated_geometry_motion_matches_its_reuse_plan() {
        let previous_mesh = generate_sphere_mesh(0.1, 4, 8);
        let mut current_mesh = previous_mesh.clone();
        current_mesh.nodes[0][0] += 1.0e-7;
        let plan = classify_mesh_update(&previous_mesh, &current_mesh, 1.0e-6)
            .expect("tolerated motion is classified");
        assert!(matches!(plan, MeshUpdatePlan::Unchanged { .. }));
        let decision = assess_bem_cache_reuse(
            reuse_context(&previous_mesh, 100.0),
            reuse_context(&current_mesh, 100.0),
            plan,
            BemReusePolicy::default(),
        )
        .expect("the exact-reuse tolerance must propagate to the cache decision");
        assert_eq!(decision.reuse, BemCacheReuse::all());
    }

    #[test]
    fn far_frequency_and_solver_changes_discard_krylov_and_preconditioner() {
        let mesh = generate_sphere_mesh(0.1, 4, 8);
        let previous = reuse_context(&mesh, 100.0);
        let current = reuse_context(&mesh, 150.0);
        let plan = classify_mesh_update(&mesh, &mesh, 0.0).expect("valid mesh plan");
        let far = assess_bem_cache_reuse(
            previous,
            current,
            plan,
            BemReusePolicy {
                max_relative_frequency_delta: 0.1,
            },
        )
        .expect("valid reuse comparison");
        assert_eq!(far.reuse, BemCacheReuse::geometry_only());
        assert!(matches!(
            far.rejection,
            Some(BemReuseRejection::FrequencyDeltaExceeded { .. })
        ));
        assert!(far.invalidation.rebuild_preconditioner);
        assert!(far.invalidation.discard_recycled_krylov);

        let mut changed_solver = current;
        changed_solver.solver_tolerance = 1.0e-8;
        let solver = assess_bem_cache_reuse(
            current,
            changed_solver,
            plan,
            BemReusePolicy {
                max_relative_frequency_delta: 1.0,
            },
        )
        .expect("valid reuse comparison");
        assert_eq!(solver.reuse, BemCacheReuse::geometry_only());
        assert_eq!(
            solver.rejection,
            Some(BemReuseRejection::SolverConfigurationChanged)
        );
    }

    #[test]
    fn reuse_context_and_policy_reject_malformed_inputs() {
        let mesh = generate_sphere_mesh(0.1, 4, 8);
        let physics =
            PhysicsParams::try_new(100.0, 343.0, 1.21, false).expect("test physics is valid");
        let mut incompatible_convention = ACOUSTIC_CONVENTION;
        incompatible_convention.spatial_sign = -1;
        assert_eq!(
            BemReuseContext::capture(
                &mesh,
                &physics,
                incompatible_convention,
                AssemblyMethod::Tbem,
                SolverMethod::Gmres,
                BemPreconditioner::None,
                1.0e-6,
                4.0,
            ),
            Err(BemReuseError::IncompatibleConvention)
        );

        let context = reuse_context(&mesh, 100.0);
        let plan = classify_mesh_update(&mesh, &mesh, 0.0).expect("valid mesh plan");
        assert_eq!(
            assess_bem_cache_reuse(
                context,
                context,
                plan,
                BemReusePolicy {
                    max_relative_frequency_delta: f64::NAN,
                },
            ),
            Err(BemReuseError::InvalidFrequencyTolerance)
        );
    }
}
