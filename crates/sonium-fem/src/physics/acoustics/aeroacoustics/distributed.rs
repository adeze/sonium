//! Distributed Lighthill sources integrated over FEM mesh cells.
//!
//! Non-compact sources are summed from compact cells: each cell contributes
//! its centroid tensor moment (stress × measure) through the point or
//! planar quadrupole evaluator. Two-dimensional meshes additionally offer
//! an explicit compact-depth extrusion to 3-D cells, which documents the
//! `d ≪ λ` approximation instead of hiding it in consumer code.

use num_complex::Complex64;

use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature::QuadratureRule;
use crate::spaces::FunctionSpace;
use crate::spaces::dg::DGSpace;

use super::error::AeroacousticsError;
use super::lighthill::{LighthillQuadrupole, lighthill_far_field_pressure, lighthill_pressure};
use super::lighthill_2d_axi::{
    LighthillQuadrupole2d, lighthill_far_field_pressure_2d, lighthill_pressure_2d,
};

/// Centroid and quadrature measure of one mesh cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellGeometry {
    /// Vertex-averaged centroid `[m]`.
    pub centroid: [f64; 3],
    /// Quadrature measure: volume `[m³]` in 3-D, area `[m²]` in 2-D.
    pub measure: f64,
}

/// Computes centroid and measure of one mesh cell.
///
/// The centroid averages element vertices; the measure integrates the
/// Jacobian determinant with a first-order rule, which is exact for affine
/// cells. Works for triangular, quadrilateral, tetrahedral, and hexahedral
/// cells.
///
/// # Errors
///
/// Returns [`AeroacousticsError::InvalidSourceCell`] for an out-of-range
/// element or a degenerate (non-positive or non-finite) measure.
pub fn cell_centroid_and_measure(
    mesh: &Mesh,
    element: usize,
) -> Result<CellGeometry, AeroacousticsError> {
    let entry = mesh
        .elements
        .get(element)
        .ok_or(AeroacousticsError::InvalidSourceCell {
            index: element,
            detail: "element index out of range",
        })?;
    let vertices = entry.vertices();
    if vertices.is_empty() {
        return Err(AeroacousticsError::InvalidSourceCell {
            index: element,
            detail: "element has no vertices",
        });
    }
    let mut centroid = [0.0; 3];
    for &node in vertices {
        let point = mesh
            .nodes
            .get(node)
            .ok_or(AeroacousticsError::InvalidSourceCell {
                index: element,
                detail: "element references a missing node",
            })?;
        centroid[0] += point.x;
        centroid[1] += point.y;
        centroid[2] += point.z;
    }
    let count = vertices.len() as f64;
    centroid = [
        centroid[0] / count,
        centroid[1] / count,
        centroid[2] / count,
    ];
    let rule = QuadratureRule::new(entry.element_type, 1);
    let space = DGSpace::new(PolynomialDegree::P1);
    let measure: f64 = rule
        .iter()
        .map(|point| {
            point.weight
                * space
                    .tabulate(
                        mesh,
                        element,
                        point.coords[0],
                        point.coords[1],
                        point.coords[2],
                    )
                    .det_j
        })
        .sum();
    if !measure.is_finite() || measure <= 0.0 {
        return Err(AeroacousticsError::InvalidSourceCell {
            index: element,
            detail: "cell measure is not positive and finite",
        });
    }
    Ok(CellGeometry { centroid, measure })
}

/// One compact 3-D cell of a distributed quadrupole source.
#[derive(Debug, Clone, Copy)]
pub struct VolumeQuadrupoleCell {
    /// Cell centroid `[m]`.
    pub position: [f64; 3],
    /// Cell volume `[m³]`.
    pub volume: f64,
    /// Harmonic volume-averaged stress `[Pa]`.
    pub stress: [[Complex64; 3]; 3],
}

/// Sums distributed 3-D quadrupole cells at one observer.
///
/// Each cell radiates as a compact point quadrupole with moment
/// `stress × volume`. Set `far_field` to keep only radiating terms.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid parameters, cells, or a
/// coincident observer.
pub fn sum_quadrupole_field(
    cells: &[VolumeQuadrupoleCell],
    observer: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
    far_field: bool,
) -> Result<Complex64, AeroacousticsError> {
    let mut total = Complex64::new(0.0, 0.0);
    for (index, cell) in cells.iter().enumerate() {
        if !cell.volume.is_finite() || cell.volume <= 0.0 {
            return Err(AeroacousticsError::InvalidSourceCell {
                index,
                detail: "cell volume is not positive and finite",
            });
        }
        let moment = cell.stress.map(|row| row.map(|entry| entry * cell.volume));
        let source = LighthillQuadrupole::new(cell.position, moment)?;
        total += if far_field {
            lighthill_far_field_pressure(&source, observer, angular_frequency, speed_of_sound)?
        } else {
            lighthill_pressure(&source, observer, angular_frequency, speed_of_sound)?
        };
    }
    Ok(total)
}

/// One compact 2-D cell of a distributed planar quadrupole source.
#[derive(Debug, Clone, Copy)]
pub struct AreaQuadrupoleCell {
    /// Cell centroid `[m]` in the plane.
    pub position: [f64; 2],
    /// Cell area `[m²]`.
    pub area: f64,
    /// Harmonic area-averaged planar stress `[Pa]`.
    pub stress: [[Complex64; 2]; 2],
}

/// Sums distributed planar quadrupole cells at one observer.
///
/// Each cell radiates through the exact or far-field Hankel kernel with
/// moment `stress × area`. The result is the true 2-D (infinite-line)
/// field; see [`extrude_cells_to_3d`] for finite-depth slots.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid parameters, cells, or a
/// coincident observer.
pub fn sum_quadrupole_field_2d(
    cells: &[AreaQuadrupoleCell],
    observer: [f64; 2],
    angular_frequency: f64,
    speed_of_sound: f64,
    far_field: bool,
) -> Result<Complex64, AeroacousticsError> {
    let mut total = Complex64::new(0.0, 0.0);
    for (index, cell) in cells.iter().enumerate() {
        if !cell.area.is_finite() || cell.area <= 0.0 {
            return Err(AeroacousticsError::InvalidSourceCell {
                index,
                detail: "cell area is not positive and finite",
            });
        }
        let moment = cell.stress.map(|row| row.map(|entry| entry * cell.area));
        let source = LighthillQuadrupole2d::new(cell.position, moment)?;
        total += if far_field {
            lighthill_far_field_pressure_2d(&source, observer, angular_frequency, speed_of_sound)?
        } else {
            lighthill_pressure_2d(&source, observer, angular_frequency, speed_of_sound)?
        };
    }
    Ok(total)
}

/// Extrudes planar cells into 3-D cells of finite depth.
///
/// Multiplies each area by `depth_m` and embeds the planar stress in the
/// 3-D tensor's in-plane block (out-of-plane components vanish). Valid for
/// acoustically compact depths (`d ≪ λ`) observed near the mid-plane; deep
/// slots need the native 2-D field or a resolved 3-D source instead.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for a non-positive depth or invalid cells.
pub fn extrude_cells_to_3d(
    cells: &[AreaQuadrupoleCell],
    depth_m: f64,
) -> Result<Vec<VolumeQuadrupoleCell>, AeroacousticsError> {
    if !depth_m.is_finite() || depth_m <= 0.0 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "depth_m",
            value: depth_m,
        });
    }
    cells
        .iter()
        .enumerate()
        .map(|(index, cell)| {
            if !cell.area.is_finite() || cell.area <= 0.0 {
                return Err(AeroacousticsError::InvalidSourceCell {
                    index,
                    detail: "cell area is not positive and finite",
                });
            }
            let mut stress = [[Complex64::new(0.0, 0.0); 3]; 3];
            for (row, cell_row) in stress.iter_mut().zip(cell.stress.iter()).take(2) {
                for (entry, cell_entry) in row.iter_mut().zip(cell_row.iter()).take(2) {
                    *entry = *cell_entry;
                }
            }
            Ok(VolumeQuadrupoleCell {
                position: [cell.position[0], cell.position[1], 0.0],
                volume: cell.area * depth_m,
                stress,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_geometry_matches_closed_form_measures() {
        let triangles = crate::mesh::unit_square_triangles(1);
        let tri = cell_centroid_and_measure(&triangles, 0).unwrap();
        assert!((tri.measure - 0.5).abs() < 1e-12);
        assert!(tri.centroid[0] >= 0.0 && tri.centroid[0] <= 1.0);

        // Unit-cube tets tile volume 1 exactly.
        let tets = crate::mesh::unit_cube_tetrahedra(1);
        let total: f64 = (0..tets.elements.len())
            .map(|element| cell_centroid_and_measure(&tets, element).unwrap().measure)
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cell_geometry_rejects_bad_elements() {
        let mesh = crate::mesh::unit_square_triangles(1);
        assert!(matches!(
            cell_centroid_and_measure(&mesh, 99),
            Err(AeroacousticsError::InvalidSourceCell { .. })
        ));
    }

    fn planar_stress() -> [[Complex64; 2]; 2] {
        let mut stress = [[Complex64::new(0.0, 0.0); 2]; 2];
        stress[0][0] = Complex64::new(1.0, 0.5);
        stress
    }

    #[test]
    fn distributed_sums_match_point_evaluators() {
        let planar = AreaQuadrupoleCell {
            position: [0.0, 0.0],
            area: 2.0,
            stress: planar_stress(),
        };
        let summed =
            sum_quadrupole_field_2d(std::slice::from_ref(&planar), [10.0, 0.0], 10.0, 1.0, false)
                .unwrap();
        let direct = lighthill_pressure_2d(
            &LighthillQuadrupole2d::new(
                [0.0, 0.0],
                planar_stress().map(|row| row.map(|entry| entry * 2.0)),
            )
            .unwrap(),
            [10.0, 0.0],
            10.0,
            1.0,
        )
        .unwrap();
        assert!((summed - direct).norm() < 1e-14);

        let extruded = extrude_cells_to_3d(std::slice::from_ref(&planar), 0.02).unwrap();
        assert!((extruded[0].volume - 0.04).abs() < 1e-14);
        assert_eq!(extruded[0].stress[2][2], Complex64::new(0.0, 0.0));
        assert_eq!(extruded[0].stress[0][0], planar.stress[0][0]);
        let summed_3d = sum_quadrupole_field(&extruded, [10.0, 0.0, 0.0], 10.0, 1.0, true).unwrap();
        assert!(summed_3d.norm().is_finite() && summed_3d.norm() > 0.0);
    }

    #[test]
    fn distributed_sums_reject_degenerate_cells() {
        let bad = AreaQuadrupoleCell {
            position: [0.0, 0.0],
            area: 0.0,
            stress: planar_stress(),
        };
        assert!(
            sum_quadrupole_field_2d(std::slice::from_ref(&bad), [1.0, 0.0], 1.0, 1.0, false)
                .is_err()
        );
        assert!(extrude_cells_to_3d(std::slice::from_ref(&bad), 0.02).is_err());
    }
}
