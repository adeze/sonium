//! DOF location utilities for boundary condition application
//!
//! Provides functions to locate DOFs by topology (boundary facets) or
//! geometry (coordinate predicates), analogous to DOLFINx's
//! `locate_dofs_topological` and `locate_dofs_geometrical`.

use crate::mesh::Mesh;
use crate::spaces::{FunctionSpace, SpaceValidationError};

/// Failure while locating geometric boundary DOFs.
#[derive(Debug, Clone, PartialEq)]
pub enum DofLocationError {
    /// The function space or mesh violates its global DOF contract.
    InvalidSpace(SpaceValidationError),
    /// The space does not expose physical coordinates for its DOFs.
    UnsupportedCoordinates,
    /// The coordinate list does not match the declared global DOF count.
    CoordinateCount {
        /// Declared global DOF count.
        expected: usize,
        /// Number of coordinates returned by the space.
        actual: usize,
    },
    /// A supplied DOF coordinate is non-finite.
    NonFiniteCoordinate {
        /// DOF with the invalid coordinate.
        dof: usize,
    },
    /// A plane query used an axis outside x/y/z.
    InvalidAxis {
        /// Requested coordinate axis.
        axis: usize,
    },
    /// A plane coordinate value was non-finite.
    InvalidValue {
        /// Requested coordinate value.
        value: f64,
    },
    /// A geometric tolerance was non-finite or negative.
    InvalidTolerance {
        /// Requested geometric tolerance.
        tolerance: f64,
    },
}

impl std::fmt::Display for DofLocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSpace(error) => write!(f, "invalid function space: {error}"),
            Self::UnsupportedCoordinates => {
                write!(f, "function space does not expose physical DOF coordinates")
            }
            Self::CoordinateCount { expected, actual } => {
                write!(f, "DOF coordinate count {actual} does not match {expected}")
            }
            Self::NonFiniteCoordinate { dof } => {
                write!(f, "DOF {dof} has a non-finite physical coordinate")
            }
            Self::InvalidAxis { axis } => write!(f, "axis {axis} is outside x/y/z"),
            Self::InvalidValue { value } => write!(f, "plane value must be finite (got {value})"),
            Self::InvalidTolerance { tolerance } => {
                write!(
                    f,
                    "tolerance must be finite and non-negative (got {tolerance})"
                )
            }
        }
    }
}

impl std::error::Error for DofLocationError {}

/// Locate DOFs on mesh entities of a given topological dimension and marker.
///
/// For `dim = mesh.dimension - 1` (facets), returns DOFs on boundary
/// faces/edges with the specified marker. For scalar H1 spaces, these
/// are the node indices on the boundary.
///
/// # Arguments
/// - `mesh`: the computational mesh
/// - `space`: the function space
/// - `boundary_marker`: the boundary tag to match
pub fn locate_dofs_topological<S: FunctionSpace>(
    mesh: &Mesh,
    space: &S,
    boundary_marker: i32,
) -> Vec<usize> {
    space.boundary_dofs(mesh, boundary_marker)
}

/// Locate DOFs whose physical coordinates satisfy a geometric predicate.
///
/// For nodal H1 spaces, including P2 edge/face/interior nodes, the predicate
/// is applied at each DOF coordinate. Vector components share the coordinate
/// of their underlying scalar DOF.
///
/// # Arguments
/// - `mesh`: the computational mesh
/// - `space`: the function space
/// - `predicate`: `(x, y, z) -> bool`, returns true for DOFs to include
///
/// # Example
/// ```ignore
/// // Find all DOFs on the left boundary (x ≈ 0)
/// let dofs = locate_dofs_geometrical(&mesh, &space, |x, _y, _z| x.abs() < 1e-10);
/// ```
pub fn locate_dofs_geometrical<S, F>(mesh: &Mesh, space: &S, predicate: F) -> Vec<usize>
where
    S: FunctionSpace,
    F: Fn(f64, f64, f64) -> bool,
{
    try_locate_dofs_geometrical(mesh, space, predicate).unwrap_or_default()
}

/// Checked variant of [`locate_dofs_geometrical`].
pub fn try_locate_dofs_geometrical<S, F>(
    mesh: &Mesh,
    space: &S,
    predicate: F,
) -> Result<Vec<usize>, DofLocationError>
where
    S: FunctionSpace,
    F: Fn(f64, f64, f64) -> bool,
{
    let n_dofs = space.num_global_dofs(mesh);
    if n_dofs == 0 || mesh.num_nodes() == 0 {
        return Ok(Vec::new());
    }
    space
        .validate(mesh)
        .map_err(DofLocationError::InvalidSpace)?;
    let coordinates = space
        .dof_coordinates(mesh)
        .ok_or(DofLocationError::UnsupportedCoordinates)?;
    if coordinates.len() != n_dofs {
        return Err(DofLocationError::CoordinateCount {
            expected: n_dofs,
            actual: coordinates.len(),
        });
    }
    let dofs = coordinates
        .into_iter()
        .enumerate()
        .map(|(dof, point)| {
            if point.iter().any(|value| !value.is_finite()) {
                return Err(DofLocationError::NonFiniteCoordinate { dof });
            }
            Ok((dof, point))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter_map(|(dof, point)| predicate(point[0], point[1], point[2]).then_some(dof))
        .collect::<Vec<_>>();
    Ok(dofs)
}

/// Locate DOFs on a specific coordinate plane within a tolerance.
///
/// Convenience for common cases like "all DOFs where x = 0" or "z = 1.5".
///
/// `axis`: 0 = x, 1 = y, 2 = z.
pub fn locate_dofs_on_plane<S: FunctionSpace>(
    mesh: &Mesh,
    space: &S,
    axis: usize,
    value: f64,
    tolerance: f64,
) -> Vec<usize> {
    try_locate_dofs_on_plane(mesh, space, axis, value, tolerance).unwrap_or_default()
}

/// Checked variant of [`locate_dofs_on_plane`].
pub fn try_locate_dofs_on_plane<S: FunctionSpace>(
    mesh: &Mesh,
    space: &S,
    axis: usize,
    value: f64,
    tolerance: f64,
) -> Result<Vec<usize>, DofLocationError> {
    if axis > 2 {
        return Err(DofLocationError::InvalidAxis { axis });
    }
    if !value.is_finite() {
        return Err(DofLocationError::InvalidValue { value });
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(DofLocationError::InvalidTolerance { tolerance });
    }
    try_locate_dofs_geometrical(mesh, space, |x, y, z| {
        let coord = match axis {
            0 => x,
            1 => y,
            2 => z,
            _ => unreachable!("axis was validated above"),
        };
        (coord - value).abs() < tolerance
    })
}

/// Locate DOFs inside a bounding box.
#[allow(clippy::too_many_arguments)]
pub fn locate_dofs_in_box<S: FunctionSpace>(
    mesh: &Mesh,
    space: &S,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
) -> Vec<usize> {
    locate_dofs_geometrical(mesh, space, |x, y, z| {
        x >= x_min && x <= x_max && y >= y_min && y <= y_max && z >= z_min && z <= z_max
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{H1Space, HcurlSpace, VectorH1Space};

    #[test]
    fn test_locate_dofs_geometrical_left_boundary() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);

        let left_dofs = locate_dofs_geometrical(&mesh, &space, |x, _y, _z| x.abs() < 1e-10);

        // Left boundary of unit square: x=0, nodes at y = 0, 0.25, 0.5, 0.75, 1.0
        assert_eq!(left_dofs.len(), 5);
        for &dof in &left_dofs {
            assert!(mesh.nodes[dof].x.abs() < 1e-10);
        }
    }

    #[test]
    fn test_locate_dofs_on_plane() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);

        let bottom = locate_dofs_on_plane(&mesh, &space, 1, 0.0, 1e-10);
        assert_eq!(bottom.len(), 5); // y=0 on 4-division unit square
    }

    #[test]
    fn test_locate_dofs_in_box() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);

        // Corner box: [0, 0.3] × [0, 0.3]
        let corner = locate_dofs_in_box(&mesh, &space, 0.0, 0.3, 0.0, 0.3, -1.0, 1.0);

        // Should contain nodes at (0,0), (0.25,0), (0,0.25), (0.25,0.25)
        assert!(corner.len() >= 4);
        for &dof in &corner {
            assert!(mesh.nodes[dof].x <= 0.3 + 1e-10);
            assert!(mesh.nodes[dof].y <= 0.3 + 1e-10);
        }
    }

    #[test]
    fn test_locate_dofs_topological_matches_space() {
        let mut mesh = unit_square_triangles(4);
        mesh.detect_boundaries();
        let space = H1Space::new(PolynomialDegree::P1);

        // Marker 0 should give all boundary nodes
        let boundary_dofs = locate_dofs_topological(&mesh, &space, 0);
        // All 4 edges of unit square have nodes
        assert!(boundary_dofs.len() >= 4 * 4); // at least 16 boundary nodes
    }

    #[test]
    fn test_locate_p2_vector_dofs_includes_edge_midpoints() {
        let mesh = unit_square_triangles(1);
        let space = VectorH1Space::new(PolynomialDegree::P2, 2);

        let left_dofs = locate_dofs_geometrical(&mesh, &space, |x, _y, _z| x.abs() < 1e-12);

        // Two vertex locations plus the P2 midpoint, each with x/y components.
        assert_eq!(left_dofs.len(), 6);
        assert!(left_dofs.iter().all(|&dof| {
            space
                .dof_coordinate(&mesh, dof)
                .is_some_and(|point| point[0].abs() < 1e-12)
        }));
    }

    #[test]
    fn checked_geometric_location_reports_unsupported_space() {
        let mesh = unit_square_triangles(1);
        let space = HcurlSpace::new();
        assert_eq!(
            try_locate_dofs_geometrical(&mesh, &space, |_, _, _| true),
            Err(DofLocationError::UnsupportedCoordinates)
        );
        // The compatibility wrapper remains non-panicking and returns no
        // coordinates when the space has no nodal geometry.
        assert!(locate_dofs_geometrical(&mesh, &space, |_, _, _| true).is_empty());
    }

    #[test]
    fn checked_plane_location_validates_query_contract() {
        let mesh = unit_square_triangles(1);
        let space = H1Space::new(PolynomialDegree::P1);
        assert_eq!(
            try_locate_dofs_on_plane(&mesh, &space, 3, 0.0, 1.0),
            Err(DofLocationError::InvalidAxis { axis: 3 })
        );
        assert!(matches!(
            try_locate_dofs_on_plane(&mesh, &space, 0, f64::NAN, 1.0),
            Err(DofLocationError::InvalidValue { .. })
        ));
        assert!(matches!(
            try_locate_dofs_on_plane(&mesh, &space, 0, 0.0, -1.0),
            Err(DofLocationError::InvalidTolerance { .. })
        ));
    }
}
