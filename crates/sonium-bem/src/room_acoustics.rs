//! Room Acoustics Simulator using BEM
//!
//! This module implements a 3D room acoustics simulator for calculating sound pressure
//! levels (SPL) at listening positions from directional sources with frequency-dependent
//! radiation patterns.

// Room acoustics is still experimental; allow missing docs for now
#![allow(missing_docs)]

mod config;
mod solver;

pub use config::*;
pub use solver::*;

// Re-export common types from math-xem
pub use sonium_xem::{
    BoundaryConfig, CrossoverFilter, Directivity, DirectivityGrid, FrequencyResult, LShapedRoom,
    ListeningPosition, Point3D, RectangularRoom, RoomGeometry, RoomMesh, RoomSimulation,
    SimulationResults, SliceData, Source, SurfaceElement, constants, log_space, pressure_to_spl,
    wavenumber,
};
// Re-export output utilities
pub use sonium_xem::{
    create_default_config, create_output_json, create_output_json_with_sources,
    generate_spatial_slices, print_config_summary,
};

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    #[test]
    fn test_rectangular_room() {
        let room = RectangularRoom::new(5.0, 4.0, 3.0);
        assert_eq!(room.width, 5.0);
        assert_eq!(room.depth, 4.0);
        assert_eq!(room.height, 3.0);
    }

    #[test]
    fn test_omnidirectional_pattern() {
        let grid = DirectivityGrid::omnidirectional();
        // Should be 1.0 in all directions
        assert!((grid.interpolate(0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((grid.interpolate(PI / 2.0, PI) - 1.0).abs() < 1e-6);
        assert!((grid.interpolate(PI, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_space() {
        let freqs = log_space(20.0, 20000.0, 200);
        assert_eq!(freqs.len(), 200);
        assert!((freqs[0] - 20.0).abs() < 1e-6);
        assert!((freqs[199] - 20000.0).abs() < 1e-6);
        // Check logarithmic spacing
        assert!(freqs[1] / freqs[0] > 1.0);
    }

    #[test]
    fn test_point_distance() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(3.0, 4.0, 0.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn fastmath_phasors_preserve_room_matrix_residual_and_shape_perturbation() {
        let mesh = RoomMesh {
            nodes: vec![
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(1.0, 0.0, 0.0),
                Point3D::new(0.0, 1.0, 0.0),
                Point3D::new(0.0, 0.0, 0.75),
                Point3D::new(1.0, 0.0, 0.75),
                Point3D::new(0.0, 1.0, 0.75),
            ],
            elements: vec![
                SurfaceElement::triangle(0, 1, 2),
                SurfaceElement::triangle(3, 4, 5),
            ],
        };
        let k = 2.3;
        let matrix = build_bem_matrix(&mesh, k);
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix[[0, 0]], Complex64::new(0.5, 0.0));

        // The two panel centers are separated only in z. This independently
        // reconstructs both off-diagonal Green derivatives with libm and
        // proves that the fast exp(i·kr) path did not change matrix entries.
        let r = 0.75;
        let phase = k * r;
        let reference_phase = Complex64::new(phase.cos(), phase.sin());
        let panel_area = 0.5;
        let scale = panel_area / (4.0 * PI * r * r);
        let reference_01 = Complex64::new(0.0, phase) - 1.0;
        let reference_01 = reference_01 * reference_phase * scale;
        let reference_10 = Complex64::new(0.0, phase) - 1.0;
        let reference_10 = reference_10 * reference_phase * scale * -1.0;
        assert!((matrix[[0, 1]] - reference_01).norm() < 2.0e-12);
        assert!((matrix[[1, 0]] - reference_10).norm() < 2.0e-12);

        let source = Source::omnidirectional(Point3D::new(0.25, 0.25, 1.5), 1.0);
        let derivative_rhs = calculate_incident_field_derivative_parallel(
            &mesh,
            std::slice::from_ref(&source),
            k,
            125.0,
        );
        assert!(
            derivative_rhs
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        // The legacy direct solve uses the pressure-source collocation RHS;
        // compare its solution against that same RHS rather than mixing in the
        // separate Neumann-derivative helper.
        let rhs = calculate_incident_field(&mesh, std::slice::from_ref(&source), k, 125.0);
        let solution = solve_bem_system_direct(
            &mesh,
            &[Source::omnidirectional(Point3D::new(0.25, 0.25, 1.5), 1.0)],
            k,
            125.0,
        )
        .expect("small fastmath-backed room system should solve");
        let residual = (0..2)
            .map(|row| {
                let predicted = (0..2)
                    .map(|col| matrix[[row, col]] * solution[col])
                    .sum::<Complex64>();
                (predicted - rhs[row]).norm_sqr()
            })
            .sum::<f64>()
            .sqrt();
        assert!(residual < 2.0e-10, "room residual = {residual:e}");

        // A small shape perturbation exercises the same phasor path through
        // the geometry-dependent matrix and keeps the finite-difference
        // sensitivity finite and stable.
        let mut perturbed = mesh.clone();
        let step = 1.0e-6;
        for node in &mut perturbed.nodes[3..] {
            node.z += step;
        }
        let perturbed_matrix = build_bem_matrix(&perturbed, k);
        let shape_delta = (perturbed_matrix[[0, 1]] - matrix[[0, 1]]) / step;
        assert!(shape_delta.re.is_finite() && shape_delta.im.is_finite());
        assert!(shape_delta.norm() > 1.0e-8);
    }
}
