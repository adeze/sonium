use num_complex::Complex64;
use sonium_compute::HostMatrix;
use sonium_sim_speaker::{
    DIRECTIVITY_REFERENCE_SCHEMA_VERSION, DirectivityReference, DirectivityReferencePoint,
    DirectivityReferenceTolerances, correlate_directivity_reference,
};
use sonium_xem::ComplexDirectivityGrid;

#[test]
fn public_directivity_reference_boundary_correlates_a_normalized_phase_grid() {
    let grid = ComplexDirectivityGrid::try_new(
        vec![0.0, 90.0, 180.0, 270.0],
        vec![0.0, 90.0, 180.0],
        vec![100.0],
        vec![HostMatrix::from_row_major_vec(
            3,
            4,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.5, 0.5),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, -0.5),
                Complex64::new(0.5, 0.5),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )],
    )
    .expect("normalized phase grid");
    let reference = DirectivityReference {
        schema_version: DIRECTIVITY_REFERENCE_SCHEMA_VERSION,
        provenance: "public-directivity-reference-test.v1".into(),
        points: vec![
            DirectivityReferencePoint {
                frequency_hz: 100.0,
                vertical_angle_deg: 90.0,
                horizontal_angle_deg: 90.0,
                normalized_pressure: Complex64::new(1.0, 0.0),
            },
            DirectivityReferencePoint {
                frequency_hz: 100.0,
                vertical_angle_deg: 90.0,
                horizontal_angle_deg: 0.0,
                normalized_pressure: Complex64::new(0.5, 0.5),
            },
        ],
    };
    let report = correlate_directivity_reference(
        &grid,
        &reference,
        DirectivityReferenceTolerances {
            max_rmse_relative_error: 1.0e-12,
            max_relative_error: 1.0e-12,
            max_absolute_error: 1.0e-12,
            absolute_floor: 1.0e-6,
        },
    )
    .expect("public directivity correlation");
    assert!(report.passed);
    assert_eq!(report.samples, 2);
}
