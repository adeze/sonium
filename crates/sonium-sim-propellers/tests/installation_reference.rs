use sonium_sim_propellers::{
    ImageReflectionCoefficient, InstallationReferenceError, Point3, RotatingDipoleReference,
    evaluate_ground_image_directivity,
};

fn reference() -> RotatingDipoleReference {
    RotatingDipoleReference::try_from_equatorial_amplitude(2, 240.0, 1.0, 2.0, 1.2, 343.0)
        .expect("valid public installation dipole")
}

fn source() -> Point3 {
    Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.25,
    }
}

fn observer() -> Point3 {
    Point3 {
        x: 1.0,
        y: 0.2,
        z: 0.8,
    }
}

#[test]
fn public_installation_reference_reports_coherent_direct_and_image_paths() {
    let result = evaluate_ground_image_directivity(
        reference(),
        source(),
        observer(),
        ImageReflectionCoefficient::rigid(),
    )
    .expect("public ground image reference");
    assert!(result.is_finite());
    assert!(result.direct_distance_m > 0.0);
    assert!(result.image_distance_m > result.direct_distance_m);
    assert_eq!(
        result.total_pressure_pa,
        sonium_sim_propellers::RotatingDipoleComplexPressure {
            real_pa: result.direct_pressure_pa.real_pa + result.reflected_pressure_pa.real_pa,
            imaginary_pa: result.direct_pressure_pa.imaginary_pa
                + result.reflected_pressure_pa.imaginary_pa,
        }
    );
}

#[test]
fn public_installation_reference_zero_reflection_is_free_field() {
    let result = evaluate_ground_image_directivity(
        reference(),
        source(),
        observer(),
        ImageReflectionCoefficient::try_new(0.0, 0.0).unwrap(),
    )
    .expect("public zero-reflection image reference");
    assert_eq!(result.total_pressure_pa, result.direct_pressure_pa);
    assert_eq!(result.reflected_pressure_pa.real_pa, 0.0);
    assert_eq!(result.reflected_pressure_pa.imaginary_pa, 0.0);
}

#[test]
fn public_installation_reference_rejects_plane_and_reflection_boundaries() {
    assert!(matches!(
        ImageReflectionCoefficient::try_new(0.8, 0.8),
        Err(InstallationReferenceError::InvalidReflection { .. })
    ));
    assert!(matches!(
        evaluate_ground_image_directivity(
            reference(),
            source(),
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            ImageReflectionCoefficient::rigid(),
        ),
        Err(InstallationReferenceError::InvalidPoint { .. })
    ));
}
