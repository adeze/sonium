//! Shared topology, orientation, and Piola conventions for FEM spaces.
//!
//! `MeshTopology` owns global entity identity and canonical edge/face
//! ordering. A [`FunctionSpace`](super::FunctionSpace) owns only its local
//! basis-to-global-DOF map. H(curl) and H(div) implementations must use the
//! local orientation signs supplied by that topology when applying their
//! covariant and contravariant mappings.
//!
//! The mapping helpers below are pointwise. For a curved element, evaluate
//! the geometry Jacobian at the same quadrature point as the reference basis;
//! the helpers deliberately do not cache or approximate that Jacobian.

use std::fmt;

/// Version of the shared FEM topology/Piola contract.
pub const FEM_TOPOLOGY_PIOLA_CONTRACT_VERSION: u32 = 1;

/// The owner of global mesh entities and their canonical ordering.
pub const TOPOLOGY_OWNER: &str = "MeshTopology";

/// The owner of local basis and global-DOF maps.
pub const DOF_MAP_OWNER: &str = "FunctionSpace";

/// Global entity kinds used when describing topology ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Vertex,
    Edge,
    Face,
    Cell,
}

/// Stable identity of an entity in the mesh topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId {
    pub kind: EntityKind,
    pub index: usize,
}

/// Errors from an invalid local/global orientation relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrientationError {
    EmptyEntity,
    LengthMismatch,
    NotTheSameEntity,
}

impl fmt::Display for OrientationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEntity => write!(f, "an oriented entity cannot be empty"),
            Self::LengthMismatch => write!(f, "oriented entities have different lengths"),
            Self::NotTheSameEntity => write!(f, "local and canonical entities do not match"),
        }
    }
}

impl std::error::Error for OrientationError {}

/// Return the sign relating a local directed edge to its canonical edge.
pub fn edge_orientation_sign(
    canonical: [usize; 2],
    local: [usize; 2],
) -> Result<i8, OrientationError> {
    if local == canonical {
        Ok(1)
    } else if local == [canonical[1], canonical[0]] {
        Ok(-1)
    } else {
        Err(OrientationError::NotTheSameEntity)
    }
}

/// Return the sign relating a local cyclic face ordering to its canonical
/// ordering. Rotations preserve orientation; reversed rotations negate it.
pub fn face_orientation_sign(canonical: &[usize], local: &[usize]) -> Result<i8, OrientationError> {
    if canonical.is_empty() || local.is_empty() {
        return Err(OrientationError::EmptyEntity);
    }
    if canonical.len() != local.len() {
        return Err(OrientationError::LengthMismatch);
    }

    let n = canonical.len();
    for shift in 0..n {
        if (0..n).all(|i| local[i] == canonical[(i + shift) % n]) {
            return Ok(1);
        }
        if (0..n).all(|i| local[i] == canonical[(shift + n - i) % n]) {
            return Ok(-1);
        }
    }
    Err(OrientationError::NotTheSameEntity)
}

/// Errors from a pointwise Piola map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiolaError {
    NonFiniteJacobian,
    SingularJacobian,
    NonFiniteResult,
}

impl fmt::Display for PiolaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteJacobian => write!(f, "Piola Jacobian contains a non-finite value"),
            Self::SingularJacobian => write!(f, "Piola Jacobian is singular"),
            Self::NonFiniteResult => write!(f, "Piola map produced a non-finite value"),
        }
    }
}

impl std::error::Error for PiolaError {}

fn check_2x2(j: [[f64; 2]; 2]) -> Result<f64, PiolaError> {
    if !j.iter().flatten().all(|value| value.is_finite()) {
        return Err(PiolaError::NonFiniteJacobian);
    }
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    if !det.is_finite() {
        return Err(PiolaError::NonFiniteJacobian);
    }
    if det.abs() <= f64::EPSILON {
        return Err(PiolaError::SingularJacobian);
    }
    Ok(det)
}

/// Covariant H(curl) Piola map in two dimensions: `J^{-T} v_ref`.
pub fn covariant_piola_2d(
    jacobian: [[f64; 2]; 2],
    reference: [f64; 2],
) -> Result<[f64; 2], PiolaError> {
    let det = check_2x2(jacobian)?;
    let result = [
        (jacobian[1][1] * reference[0] - jacobian[1][0] * reference[1]) / det,
        (-jacobian[0][1] * reference[0] + jacobian[0][0] * reference[1]) / det,
    ];
    if result.iter().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(PiolaError::NonFiniteResult)
    }
}

/// Contravariant H(div) Piola map in two dimensions: `J v_ref / det(J)`.
pub fn contravariant_piola_2d(
    jacobian: [[f64; 2]; 2],
    reference: [f64; 2],
) -> Result<[f64; 2], PiolaError> {
    let det = check_2x2(jacobian)?;
    let result = [
        (jacobian[0][0] * reference[0] + jacobian[0][1] * reference[1]) / det,
        (jacobian[1][0] * reference[0] + jacobian[1][1] * reference[1]) / det,
    ];
    if result.iter().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(PiolaError::NonFiniteResult)
    }
}

/// Two-dimensional curl/divergence scaling associated with the Piola maps.
pub fn curl_piola_2d(jacobian: [[f64; 2]; 2], reference: f64) -> Result<f64, PiolaError> {
    let det = check_2x2(jacobian)?;
    let result = reference / det;
    result
        .is_finite()
        .then_some(result)
        .ok_or(PiolaError::NonFiniteResult)
}

/// Two-dimensional divergence scaling associated with the contravariant map.
pub fn divergence_piola_2d(jacobian: [[f64; 2]; 2], reference: f64) -> Result<f64, PiolaError> {
    curl_piola_2d(jacobian, reference)
}

fn check_3x3(j: [[f64; 3]; 3]) -> Result<f64, PiolaError> {
    if !j.iter().flatten().all(|value| value.is_finite()) {
        return Err(PiolaError::NonFiniteJacobian);
    }
    let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
    if !det.is_finite() {
        return Err(PiolaError::NonFiniteJacobian);
    }
    if det.abs() <= f64::EPSILON {
        return Err(PiolaError::SingularJacobian);
    }
    Ok(det)
}

/// Covariant H(curl) Piola map in three dimensions: `J^{-T} v_ref`.
pub fn covariant_piola_3d(
    jacobian: [[f64; 3]; 3],
    reference: [f64; 3],
) -> Result<[f64; 3], PiolaError> {
    let det = check_3x3(jacobian)?;
    let cofactors = [
        [
            jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1],
            jacobian[1][2] * jacobian[2][0] - jacobian[1][0] * jacobian[2][2],
            jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0],
        ],
        [
            jacobian[0][2] * jacobian[2][1] - jacobian[0][1] * jacobian[2][2],
            jacobian[0][0] * jacobian[2][2] - jacobian[0][2] * jacobian[2][0],
            jacobian[0][1] * jacobian[2][0] - jacobian[0][0] * jacobian[2][1],
        ],
        [
            jacobian[0][1] * jacobian[1][2] - jacobian[0][2] * jacobian[1][1],
            jacobian[0][2] * jacobian[1][0] - jacobian[0][0] * jacobian[1][2],
            jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0],
        ],
    ];
    let result = [
        (cofactors[0][0] * reference[0]
            + cofactors[0][1] * reference[1]
            + cofactors[0][2] * reference[2])
            / det,
        (cofactors[1][0] * reference[0]
            + cofactors[1][1] * reference[1]
            + cofactors[1][2] * reference[2])
            / det,
        (cofactors[2][0] * reference[0]
            + cofactors[2][1] * reference[1]
            + cofactors[2][2] * reference[2])
            / det,
    ];
    if result.iter().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(PiolaError::NonFiniteResult)
    }
}

/// Contravariant H(div) Piola map in three dimensions: `J v_ref / det(J)`.
pub fn contravariant_piola_3d(
    jacobian: [[f64; 3]; 3],
    reference: [f64; 3],
) -> Result<[f64; 3], PiolaError> {
    let det = check_3x3(jacobian)?;
    let result = [
        (jacobian[0][0] * reference[0]
            + jacobian[0][1] * reference[1]
            + jacobian[0][2] * reference[2])
            / det,
        (jacobian[1][0] * reference[0]
            + jacobian[1][1] * reference[1]
            + jacobian[1][2] * reference[2])
            / det,
        (jacobian[2][0] * reference[0]
            + jacobian[2][1] * reference[1]
            + jacobian[2][2] * reference[2])
            / det,
    ];
    if result.iter().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(PiolaError::NonFiniteResult)
    }
}

/// Three-dimensional curl/divergence scaling associated with the Piola maps.
pub fn curl_piola_3d(jacobian: [[f64; 3]; 3], reference: [f64; 3]) -> Result<[f64; 3], PiolaError> {
    // The curl of a covariantly mapped H(curl) field follows the
    // contravariant Piola map: `curl_phys = J curl_ref / det(J)`.
    contravariant_piola_3d(jacobian, reference)
}

/// Three-dimensional divergence scaling associated with the contravariant map.
pub fn divergence_piola_3d(jacobian: [[f64; 3]; 3], reference: f64) -> Result<f64, PiolaError> {
    let det = check_3x3(jacobian)?;
    let result = reference / det;
    result
        .is_finite()
        .then_some(result)
        .ok_or(PiolaError::NonFiniteResult)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_signs_are_defined_by_canonical_entity_order() {
        assert_eq!(edge_orientation_sign([2, 5], [2, 5]), Ok(1));
        assert_eq!(edge_orientation_sign([2, 5], [5, 2]), Ok(-1));
        assert_eq!(face_orientation_sign(&[1, 2, 3, 4], &[3, 4, 1, 2]), Ok(1));
        assert_eq!(face_orientation_sign(&[1, 2, 3, 4], &[1, 4, 3, 2]), Ok(-1));
        assert_eq!(
            face_orientation_sign(&[1, 2, 3], &[1, 2, 4]),
            Err(OrientationError::NotTheSameEntity)
        );
    }

    #[test]
    fn affine_2d_maps_match_covariant_and_contravariant_definitions() {
        let jacobian = [[2.0, 0.0], [0.0, 4.0]];
        assert_eq!(covariant_piola_2d(jacobian, [2.0, 4.0]), Ok([1.0, 1.0]));
        assert_eq!(contravariant_piola_2d(jacobian, [2.0, 4.0]), Ok([0.5, 2.0]));
        assert_eq!(curl_piola_2d(jacobian, 8.0), Ok(1.0));
        assert_eq!(divergence_piola_2d(jacobian, 8.0), Ok(1.0));
    }

    #[test]
    fn affine_3d_maps_and_singular_rejection_are_explicit() {
        let jacobian = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        assert_eq!(
            covariant_piola_3d(jacobian, [2.0, 3.0, 4.0]),
            Ok([1.0, 1.0, 1.0])
        );
        assert_eq!(
            contravariant_piola_3d(jacobian, [2.0, 3.0, 4.0]),
            Ok([1.0 / 6.0, 3.0 / 8.0, 2.0 / 3.0])
        );
        assert_eq!(
            curl_piola_3d(jacobian, [2.0, 3.0, 4.0]),
            Ok([1.0 / 6.0, 3.0 / 8.0, 2.0 / 3.0])
        );
        assert_eq!(
            covariant_piola_3d(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0, 0.0, 0.0]
            ),
            Ok([1.0, 0.0, 0.0])
        );
        assert_eq!(
            contravariant_piola_3d(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [1.0; 3]
            ),
            Err(PiolaError::SingularJacobian)
        );
    }
}
