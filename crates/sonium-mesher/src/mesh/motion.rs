//! Transactional nodal mesh motion for ALE and shape-optimization workflows.
//!
//! This module moves an existing mesh without changing connectivity, element
//! markers, boundary markers, or physical groups. It is deliberately a nodal
//! displacement primitive: elasticity/RBF/harmonic field construction and
//! remeshing policy belong to higher-level optimizers. Every checked update is
//! admitted against connectivity, orientation, and a caller-supplied quality
//! barrier before replacing the mesh.

use super::quality::{QualityFailure, QualityThresholds, mesh_quality_report};
use super::types::{Mesh, Point};
use super::validate::{MeshValidationError, try_check_orientation, validate_mesh_admission};
use std::collections::BTreeSet;
use std::fmt;

/// A nodal displacement increment in the mesh coordinate frame.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshMotion {
    displacements: Vec<[f64; 3]>,
}

/// Failure while constructing or applying a [`MeshMotion`].
#[derive(Debug, Clone, PartialEq)]
pub enum MeshMotionError {
    /// The source mesh failed checked admission before motion was attempted.
    InvalidMesh(MeshValidationError),
    /// The displacement count does not match the mesh node count.
    NodeCount { expected: usize, actual: usize },
    /// A displacement or resulting coordinate is not finite.
    NonFinite { node: usize },
    /// A planar two-dimensional mesh received an out-of-plane displacement.
    OutOfPlaneDisplacement { node: usize, displacement_z: f64 },
    /// A requested displacement-to-velocity interval was invalid.
    InvalidTimeStep(f64),
    /// Harmonic extension options cannot produce a converged update.
    InvalidHarmonicOptions {
        max_iterations: usize,
        tolerance: f64,
    },
    /// Harmonic extension needs at least one prescribed node.
    MissingHarmonicConstraints,
    /// A harmonic constraint vector did not match the mesh node count.
    HarmonicConstraintCount { expected: usize, actual: usize },
    /// A mesh node has no topological neighbors for harmonic extension.
    IsolatedNode { node: usize },
    /// Harmonic iteration exhausted its configured limit.
    HarmonicNotConverged { iterations: usize, residual: f64 },
    /// The proposed geometry contains inverted or degenerate elements.
    InvalidElements(Vec<usize>),
    /// The proposed mesh violates the requested quality barrier.
    Quality(QualityFailure),
}

impl fmt::Display for MeshMotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMesh(error) => write!(f, "mesh motion source mesh is invalid: {error}"),
            Self::NodeCount { expected, actual } => {
                write!(
                    f,
                    "mesh motion has {actual} displacements, expected {expected}"
                )
            }
            Self::NonFinite { node } => write!(f, "mesh motion produced a non-finite node {node}"),
            Self::OutOfPlaneDisplacement {
                node,
                displacement_z,
            } => write!(
                f,
                "planar 2-D mesh motion has out-of-plane displacement {displacement_z} at node {node}"
            ),
            Self::InvalidTimeStep(time_step) => {
                write!(
                    f,
                    "mesh-motion time step must be finite and positive, got {time_step}"
                )
            }
            Self::InvalidHarmonicOptions {
                max_iterations,
                tolerance,
            } => write!(
                f,
                "harmonic mesh-motion options require max_iterations > 0 and finite positive tolerance, got iterations={max_iterations}, tolerance={tolerance}"
            ),
            Self::MissingHarmonicConstraints => {
                f.write_str("harmonic mesh motion requires at least one prescribed node")
            }
            Self::HarmonicConstraintCount { expected, actual } => write!(
                f,
                "harmonic mesh motion has {actual} node constraints, expected {expected}"
            ),
            Self::IsolatedNode { node } => {
                write!(
                    f,
                    "harmonic mesh motion has isolated unconstrained node {node}"
                )
            }
            Self::HarmonicNotConverged {
                iterations,
                residual,
            } => write!(
                f,
                "harmonic mesh motion did not converge after {iterations} iterations (residual {residual})"
            ),
            Self::InvalidElements(elements) => {
                write!(
                    f,
                    "mesh motion inverted or degenerated elements {elements:?}"
                )
            }
            Self::Quality(error) => write!(f, "mesh motion violated quality barrier: {error}"),
        }
    }
}

impl std::error::Error for MeshMotionError {}

/// Summary returned after a successful transactional mesh update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshMotionReport {
    /// Number of nodes whose displacement was non-zero.
    pub moved_nodes: usize,
    /// Largest Euclidean nodal displacement.
    pub max_displacement: f64,
    /// Minimum mean-ratio quality after motion.
    pub min_mean_ratio: f64,
    /// Minimum aspect-quality metric after motion.
    pub min_aspect_quality: f64,
    /// Smallest element-measure ratio after motion.
    ///
    /// For an affine P1 element this is the exact absolute Jacobian
    /// determinant ratio. Higher-order elements report the ratio of their
    /// integrated measures.
    pub min_element_measure_ratio: f64,
    /// Largest element-measure ratio after motion.
    pub max_element_measure_ratio: f64,
}

/// Deterministic Jacobi-solver settings for [`MeshMotion::harmonic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicMotionOptions {
    /// Maximum Jacobi iterations before failing closed.
    pub max_iterations: usize,
    /// Maximum infinity-norm nodal displacement update.
    pub tolerance: f64,
}

impl Default for HarmonicMotionOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            tolerance: 1.0e-12,
        }
    }
}

/// Convergence information from a harmonic displacement extension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicMotionReport {
    /// Number of Jacobi iterations performed.
    pub iterations: usize,
    /// Infinity norm of the final nodal update.
    pub residual: f64,
    /// Number of prescribed (fixed) nodal displacement values.
    pub constrained_nodes: usize,
}

impl MeshMotion {
    /// Construct a displacement field after checking all values are finite.
    pub fn new(displacements: Vec<[f64; 3]>) -> Result<Self, MeshMotionError> {
        for (node, displacement) in displacements.iter().enumerate() {
            if displacement.iter().any(|value| !value.is_finite()) {
                return Err(MeshMotionError::NonFinite { node });
            }
        }
        Ok(Self { displacements })
    }

    /// Return the displacement field without transferring ownership.
    pub fn displacements(&self) -> &[[f64; 3]] {
        &self.displacements
    }

    /// Convert this displacement increment to a nodal mesh-velocity field.
    ///
    /// The caller supplies the physical increment duration. The method has no
    /// mesh side effects and is suitable for ALE assembly after the matching
    /// checked coordinate update has been accepted.
    pub fn velocities(&self, time_step: f64) -> Result<Vec<[f64; 3]>, MeshMotionError> {
        if !time_step.is_finite() || time_step <= 0.0 {
            return Err(MeshMotionError::InvalidTimeStep(time_step));
        }
        Ok(self
            .displacements
            .iter()
            .map(|displacement| {
                [
                    displacement[0] / time_step,
                    displacement[1] / time_step,
                    displacement[2] / time_step,
                ]
            })
            .collect())
    }

    /// Extend prescribed nodal displacements harmonically over a mesh.
    ///
    /// Each `Some` entry fixes that node's displacement. Each `None` entry is
    /// iteratively set to the arithmetic mean of its sorted topological
    /// neighbors, solving the graph-Laplacian (harmonic) extension with a
    /// deterministic Jacobi iteration. The result is only a displacement field;
    /// apply it through [`Self::apply_checked_with_quality`] to admit geometry,
    /// orientation, and quality before committing a mesh update.
    ///
    /// This provides a portable, dependency-free baseline for body-fitted shape
    /// motion. It does not replace an elasticity or RBF formulation where their
    /// physical stiffness or boundary-layer behavior is required.
    pub fn harmonic(
        mesh: &Mesh,
        prescribed: &[Option<[f64; 3]>],
        options: HarmonicMotionOptions,
    ) -> Result<(Self, HarmonicMotionReport), MeshMotionError> {
        validate_mesh_admission(mesh).map_err(MeshMotionError::InvalidMesh)?;
        if options.max_iterations == 0 || !options.tolerance.is_finite() || options.tolerance <= 0.0
        {
            return Err(MeshMotionError::InvalidHarmonicOptions {
                max_iterations: options.max_iterations,
                tolerance: options.tolerance,
            });
        }
        if prescribed.len() != mesh.num_nodes() {
            return Err(MeshMotionError::HarmonicConstraintCount {
                expected: mesh.num_nodes(),
                actual: prescribed.len(),
            });
        }

        let mut constrained_nodes = 0;
        for (node, displacement) in prescribed.iter().enumerate() {
            if let Some(displacement) = displacement {
                if displacement.iter().any(|value| !value.is_finite()) {
                    return Err(MeshMotionError::NonFinite { node });
                }
                if mesh.dimension == 2 && displacement[2] != 0.0 {
                    return Err(MeshMotionError::OutOfPlaneDisplacement {
                        node,
                        displacement_z: displacement[2],
                    });
                }
                constrained_nodes += 1;
            }
        }
        if constrained_nodes == 0 {
            return Err(MeshMotionError::MissingHarmonicConstraints);
        }

        let mut adjacency = vec![BTreeSet::new(); mesh.num_nodes()];
        for element in &mesh.elements {
            for (left, right) in element.edges() {
                adjacency[left].insert(right);
                adjacency[right].insert(left);
            }
        }
        for (node, neighbors) in adjacency.iter().enumerate() {
            if prescribed[node].is_none() && neighbors.is_empty() {
                return Err(MeshMotionError::IsolatedNode { node });
            }
        }

        let mut current = prescribed
            .iter()
            .map(|value| value.unwrap_or([0.0, 0.0, 0.0]))
            .collect::<Vec<_>>();
        let mut next = current.clone();
        let mut residual = f64::INFINITY;
        for iterations in 1..=options.max_iterations {
            residual = 0.0;
            for node in 0..mesh.num_nodes() {
                if let Some(displacement) = prescribed[node] {
                    next[node] = displacement;
                    continue;
                }
                let mut mean = [0.0; 3];
                for &neighbor in &adjacency[node] {
                    for component in 0..3 {
                        mean[component] += current[neighbor][component];
                    }
                }
                let inverse_degree = 1.0 / adjacency[node].len() as f64;
                for component in 0..3 {
                    mean[component] *= inverse_degree;
                    residual = residual.max((mean[component] - current[node][component]).abs());
                }
                next[node] = mean;
            }
            std::mem::swap(&mut current, &mut next);
            if residual <= options.tolerance {
                return Ok((
                    Self::new(current)?,
                    HarmonicMotionReport {
                        iterations,
                        residual,
                        constrained_nodes,
                    },
                ));
            }
        }
        Err(MeshMotionError::HarmonicNotConverged {
            iterations: options.max_iterations,
            residual,
        })
    }

    /// Apply a motion transactionally with permissive quality limits.
    ///
    /// Use [`Self::apply_checked_with_quality`] in production to specify a
    /// physics/order-specific quality barrier.
    pub fn apply_checked(&self, mesh: &mut Mesh) -> Result<MeshMotionReport, MeshMotionError> {
        self.apply_checked_with_quality(mesh, QualityThresholds::default())
    }

    /// Apply a motion transactionally and enforce a hard post-motion quality
    /// barrier.
    ///
    /// The source mesh is admitted before any motion. The input mesh is
    /// unchanged on every error path. Successful updates preserve all topology
    /// and marker vectors, invalidate derived caches, and report the resulting
    /// quality extrema for remesh-trigger policy.
    pub fn apply_checked_with_quality(
        &self,
        mesh: &mut Mesh,
        thresholds: QualityThresholds,
    ) -> Result<MeshMotionReport, MeshMotionError> {
        validate_mesh_admission(mesh).map_err(MeshMotionError::InvalidMesh)?;
        if self.displacements.len() != mesh.nodes.len() {
            return Err(MeshMotionError::NodeCount {
                expected: mesh.nodes.len(),
                actual: self.displacements.len(),
            });
        }

        let mut candidate = mesh.clone();
        let original_measures: Vec<f64> = (0..mesh.num_elements())
            .map(|element| mesh.element_measure(element))
            .collect();
        let mut moved_nodes = 0;
        let mut max_displacement = 0.0_f64;
        for (node, (point, displacement)) in candidate
            .nodes
            .iter_mut()
            .zip(self.displacements.iter())
            .enumerate()
        {
            if candidate.dimension == 2 && displacement[2] != 0.0 {
                return Err(MeshMotionError::OutOfPlaneDisplacement {
                    node,
                    displacement_z: displacement[2],
                });
            }
            let moved = [
                point.x + displacement[0],
                point.y + displacement[1],
                point.z + displacement[2],
            ];
            if moved.iter().any(|value| !value.is_finite()) {
                return Err(MeshMotionError::NonFinite { node });
            }
            let norm = displacement
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                moved_nodes += 1;
            }
            max_displacement = max_displacement.max(norm);
            *point = Point::new_3d(moved[0], moved[1], moved[2]);
        }

        let invalid_elements =
            try_check_orientation(&candidate).map_err(MeshMotionError::InvalidMesh)?;
        if !invalid_elements.is_empty() {
            return Err(MeshMotionError::InvalidElements(invalid_elements));
        }
        let quality = mesh_quality_report(&candidate, 16);
        quality
            .validate(thresholds)
            .map_err(MeshMotionError::Quality)?;

        let mut min_element_measure_ratio = f64::INFINITY;
        let mut max_element_measure_ratio = 0.0_f64;
        for (element, original_measure) in original_measures.iter().copied().enumerate() {
            let moved_measure = candidate.element_measure(element);
            if !original_measure.is_finite()
                || original_measure <= 0.0
                || !moved_measure.is_finite()
                || moved_measure <= 0.0
            {
                return Err(MeshMotionError::InvalidElements(vec![element]));
            }
            let ratio = moved_measure / original_measure;
            min_element_measure_ratio = min_element_measure_ratio.min(ratio);
            max_element_measure_ratio = max_element_measure_ratio.max(ratio);
        }
        if original_measures.is_empty() {
            min_element_measure_ratio = 1.0;
            max_element_measure_ratio = 1.0;
        }

        candidate.invalidate_caches();
        *mesh = candidate;
        Ok(MeshMotionReport {
            moved_nodes,
            max_displacement,
            min_mean_ratio: quality.min_mean_ratio,
            min_aspect_quality: quality.min_aspect_quality,
            min_element_measure_ratio,
            max_element_measure_ratio,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{BoundaryType, ElementType};

    fn triangle_mesh() -> Mesh {
        let mut mesh = Mesh::new(2);
        mesh.add_node(Point::new_2d(0.0, 0.0));
        mesh.add_node(Point::new_2d(1.0, 0.0));
        mesh.add_node(Point::new_2d(0.0, 1.0));
        mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
        mesh.detect_boundaries();
        mesh.set_boundary_condition(BoundaryType::Dirichlet, 7, |_| true);
        mesh
    }

    fn assert_mesh_unchanged(actual: &Mesh, original: &Mesh) {
        assert_eq!(actual.nodes, original.nodes);
        assert_eq!(actual.dimension, original.dimension);
        assert_eq!(actual.elements.len(), original.elements.len());
        for (actual, original) in actual.elements.iter().zip(&original.elements) {
            assert_eq!(actual.element_type, original.element_type);
            assert_eq!(actual.nodes, original.nodes);
            assert_eq!(actual.id, original.id);
            assert_eq!(actual.parent_id, original.parent_id);
            assert_eq!(actual.level, original.level);
            assert_eq!(actual.marker, original.marker);
        }
        assert_eq!(actual.boundaries.len(), original.boundaries.len());
        for (actual, original) in actual.boundaries.iter().zip(&original.boundaries) {
            assert_eq!(actual.nodes, original.nodes);
            assert_eq!(actual.boundary_type, original.boundary_type);
            assert_eq!(actual.marker, original.marker);
            assert_eq!(actual.element_idx, original.element_idx);
            assert_eq!(actual.local_idx, original.local_idx);
        }
    }

    #[test]
    fn valid_motion_preserves_topology_markers_and_reports_quality() {
        let mut mesh = triangle_mesh();
        let motion = MeshMotion::new(vec![[0.1, 0.0, 0.0]; 3]).expect("finite motion");
        let report = motion.apply_checked(&mut mesh).expect("valid motion");
        assert_eq!(report.moved_nodes, 3);
        assert!((report.max_displacement - 0.1).abs() < 1e-12);
        assert!(report.min_mean_ratio.is_finite());
        assert!(report.min_aspect_quality.is_finite());
        assert!((report.min_element_measure_ratio - 1.0).abs() < 1e-12);
        assert!((report.max_element_measure_ratio - 1.0).abs() < 1e-12);
        assert_eq!(mesh.elements.len(), 1);
        assert_eq!(mesh.elements[0].marker, 0);
        assert!(mesh.boundaries.iter().all(|boundary| boundary.marker == 7));
        assert!((mesh.nodes[0].x - 0.1).abs() < 1e-12);
    }

    #[test]
    fn affine_scaling_matches_analytic_area_jacobian() {
        let mut mesh = triangle_mesh();
        let original_area = mesh.element_measure(0);
        let scale_x = 1.25;
        let motion = MeshMotion::new(vec![
            [0.0, 0.0, 0.0],
            [scale_x - 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        .expect("finite affine motion");
        let report = motion
            .apply_checked(&mut mesh)
            .expect("valid affine motion");
        let area = mesh.element_measure(0);
        assert!((area - scale_x * original_area).abs() < 1e-12);
        assert!(
            try_check_orientation(&mesh)
                .expect("moved mesh admission")
                .is_empty()
        );
        assert!((report.min_element_measure_ratio - scale_x).abs() < 1e-12);
        assert!((report.max_element_measure_ratio - scale_x).abs() < 1e-12);
    }

    #[test]
    fn velocities_match_displacements_and_reject_invalid_time_steps() {
        let motion =
            MeshMotion::new(vec![[0.2, -0.1, 0.0], [0.0, 0.0, 0.4]]).expect("finite motion");
        assert_eq!(
            motion.velocities(0.2).expect("valid time step"),
            vec![[1.0, -0.5, 0.0], [0.0, 0.0, 2.0]]
        );
        assert!(matches!(
            motion.velocities(0.0),
            Err(MeshMotionError::InvalidTimeStep(0.0))
        ));
    }

    #[test]
    fn harmonic_motion_reproduces_an_affine_boundary_displacement() {
        let mesh = crate::mesh::unit_square_triangles(2);
        let prescribed = mesh
            .nodes
            .iter()
            .map(|point| {
                if point.x == 0.0 || point.x == 1.0 || point.y == 0.0 || point.y == 1.0 {
                    Some([0.2 * point.x, 0.0, 0.0])
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let (motion, report) = MeshMotion::harmonic(
            &mesh,
            &prescribed,
            HarmonicMotionOptions {
                max_iterations: 1_000,
                tolerance: 1e-13,
            },
        )
        .expect("harmonic affine extension");
        assert!(report.iterations > 0);
        assert!(report.residual <= 1e-13);
        assert_eq!(report.constrained_nodes, 8);
        for (point, displacement) in mesh.nodes.iter().zip(motion.displacements()) {
            assert!((displacement[0] - 0.2 * point.x).abs() < 1e-11);
            assert_eq!(displacement[1], 0.0);
            assert_eq!(displacement[2], 0.0);
        }
    }

    #[test]
    fn harmonic_motion_rejects_unconstrained_or_unconverged_fields() {
        let mesh = crate::mesh::unit_square_triangles(1);
        assert!(matches!(
            MeshMotion::harmonic(
                &mesh,
                &vec![None; mesh.num_nodes()],
                HarmonicMotionOptions::default()
            ),
            Err(MeshMotionError::MissingHarmonicConstraints)
        ));
        let prescribed = vec![Some([0.0, 0.0, 0.0]), None, None, Some([0.2, 0.0, 0.0])];
        assert!(matches!(
            MeshMotion::harmonic(
                &mesh,
                &prescribed,
                HarmonicMotionOptions {
                    max_iterations: 1,
                    tolerance: 1e-30,
                },
            ),
            Err(MeshMotionError::HarmonicNotConverged { .. })
        ));
    }

    #[test]
    fn inverted_or_out_of_plane_motion_is_transactional() {
        let mut mesh = triangle_mesh();
        let original = mesh.clone();
        let inverted = MeshMotion::new(vec![[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            .expect("finite motion");
        assert!(matches!(
            inverted.apply_checked(&mut mesh),
            Err(MeshMotionError::InvalidElements(_))
        ));
        assert_mesh_unchanged(&mesh, &original);

        let out_of_plane = MeshMotion::new(vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]])
            .expect("finite motion");
        assert!(matches!(
            out_of_plane.apply_checked(&mut mesh),
            Err(MeshMotionError::OutOfPlaneDisplacement { node: 1, .. })
        ));
        assert_mesh_unchanged(&mesh, &original);
    }

    #[test]
    fn quality_barrier_is_transactional() {
        let mut mesh = triangle_mesh();
        let original = mesh.clone();
        let motion = MeshMotion::new(vec![[0.0, 0.0, 0.0], [-0.9, 0.0, 0.0], [0.0, 0.0, 0.0]])
            .expect("finite motion");
        let thresholds = QualityThresholds {
            min_mean_ratio: 0.5,
            ..QualityThresholds::default()
        };
        assert!(matches!(
            motion.apply_checked_with_quality(&mut mesh, thresholds),
            Err(MeshMotionError::Quality(
                QualityFailure::BelowThreshold { .. }
            ))
        ));
        assert_mesh_unchanged(&mesh, &original);
    }

    #[test]
    fn malformed_motion_is_rejected_before_mesh_mutation() {
        let mut mesh = triangle_mesh();
        let original = mesh.clone();
        let short_motion = MeshMotion::new(vec![[0.0, 0.0, 0.0]; 2]).expect("finite field");
        assert!(matches!(
            short_motion.apply_checked(&mut mesh),
            Err(MeshMotionError::NodeCount { .. })
        ));
        assert_mesh_unchanged(&mesh, &original);
        assert!(matches!(
            MeshMotion::new(vec![[f64::NAN, 0.0, 0.0]; 3]),
            Err(MeshMotionError::NonFinite { node: 0 })
        ));
    }
}
