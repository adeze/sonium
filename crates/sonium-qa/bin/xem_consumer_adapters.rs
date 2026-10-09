//! Release evidence for typed XEM acoustic-medium consumption.
//!
//! This case crosses the public XEM -> BEM, XEM -> FEM, and XEM -> BEM
//! material adapters.  It deliberately checks both harmonic signs and typed
//! failure paths so a consumer cannot silently fall back to raw compatibility
//! scalars or the process-wide default convention.

use num_complex::Complex64;
use sonium_bem::core::BemProblem;
use sonium_bem::materials::AmbientMedium;
use sonium_bem::{PhysicsParams, PhysicsParamsError};
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::{FemConvention, FemConventionError};
use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::ExchangePhasorBridge;
use sonium_sim_speaker_modeling::conventions::ConventionError;
use sonium_xem::{
    AcousticMedium, ComplexStorage, ConventionRegistry, CoordinateFrame, Domain, DomainKind,
    FieldKind, FieldSpec, HarmonicConvention, MultiphysicsProblem, ScalarType, StorageOrder,
    TensorLayout,
};

const CASE_ID: &str = "xem.consumer-adapters.v1";

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0)
}

fn pressure_exchange(dof_count: usize) -> MultiphysicsProblem {
    MultiphysicsProblem {
        id: format!("xem-consumer-pressure-{dof_count}"),
        conventions: ConventionRegistry::default(),
        domains: vec![Domain {
            id: "acoustics".into(),
            label: "shared acoustic domain".into(),
            kind: DomainKind::Acoustics,
        }],
        fields: vec![FieldSpec {
            id: "pressure".into(),
            name: "complex acoustic pressure".into(),
            domain_id: "acoustics".into(),
            kind: FieldKind::ComplexScalar,
            units: Some("Pa".into()),
            layout: TensorLayout {
                scalar_type: ScalarType::ComplexF64,
                shape: vec![dof_count],
                strides: vec![1],
                byte_offset: 0,
                order: StorageOrder::RowMajor,
                complex_storage: ComplexStorage::Interleaved,
                coordinate_frame: None,
            },
        }],
        ..MultiphysicsProblem::default()
    }
}

fn run() -> Result<QaResult, String> {
    let medium = AcousticMedium::try_new(1_000.0, 343.0, 1.2)
        .map_err(|error| format!("construct canonical XEM medium: {error}"))?;
    let expected_wavenumber =
        2.0 * std::f64::consts::PI * medium.frequency.get() / medium.speed_of_sound.get();

    let bem = PhysicsParams::try_new_medium(medium.clone(), false)
        .map_err(|error| format!("BEM typed medium adapter: {error}"))?;
    bem.validate()
        .map_err(|error| format!("BEM adapted parameters failed validation: {error}"))?;
    let fem = FemConvention::from_medium(&medium)
        .map_err(|error| format!("FEM typed medium adapter: {error}"))?;
    let ambient = AmbientMedium::try_from_xem(medium.density, medium.speed_of_sound)
        .map_err(|error| format!("BEM material typed adapter: {error}"))?;
    let speaker_bridge = ExchangePhasorBridge::from_xem(medium.conventions)
        .map_err(|error| format!("speaker XEM phasor adapter: {error}"))?;
    let speaker_input = Complex64::new(1.25, -0.75);
    let speaker_round_trip =
        speaker_bridge.from_modeling_rms(speaker_bridge.to_modeling_rms(speaker_input));
    let speaker_round_trip_error = (speaker_round_trip - speaker_input).norm();

    // Exercise the actual backend admission constructors in addition to the
    // typed scalar adapters above.  These calls are deliberately small and
    // retain the XEM field/convention contract all the way to the production
    // BEM and legacy FEM problem objects.
    let bem_base = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let bem_exchange = pressure_exchange(bem_base.mesh.nodes.len());
    let bem_admitted = BemProblem::from_xem(
        &bem_exchange,
        "pressure",
        bem_base.mesh.clone(),
        bem_base.physics.clone(),
        bem_base.incident_field.clone(),
        bem_base.bc_type,
        bem_base.use_burton_miller,
    )
    .map_err(|error| format!("BEM XEM problem admission: {error}"))?;
    let fem_mesh = sonium_fem::mesh::unit_square_triangles(1);
    let fem_exchange = pressure_exchange(fem_mesh.num_nodes());
    let fem_admitted = HelmholtzProblem::assemble_from_xem(
        &fem_exchange,
        "pressure",
        &fem_mesh,
        sonium_fem::basis::PolynomialDegree::P1,
        Complex64::new(1.0, 0.0),
        |_, _, _| Complex64::new(1.0, 0.0),
    )
    .map_err(|error| format!("FEM XEM Helmholtz admission: {error}"))?;
    let backend_admission_count = u64::from(
        bem_admitted.physics.harmonic_factor == bem.harmonic_factor
            && fem_admitted.num_dofs() == fem_mesh.num_nodes(),
    );

    let default_wavenumber_error = relative_error(bem.wave_number, expected_wavenumber);
    if bem.harmonic_factor != 1.0
        || fem.time_sign != -1
        || fem.spatial_sign != 1
        || !ambient.characteristic_impedance().is_finite()
        || speaker_round_trip_error > 1.0e-14
        || backend_admission_count != 1
    {
        return Err(format!(
            "canonical adapter mismatch: BEM sign={}, FEM signs=({}, {}), Z={}",
            bem.harmonic_factor,
            fem.time_sign,
            fem.spatial_sign,
            ambient.characteristic_impedance()
        ));
    }

    let mut plus_medium = medium.clone();
    plus_medium.conventions.harmonic_time = HarmonicConvention::ExpPlusIwt;
    let plus_bem = PhysicsParams::try_new_medium(plus_medium.clone(), false)
        .map_err(|error| format!("BEM plus-i medium adapter: {error}"))?;
    let plus_fem = FemConvention::from_medium(&plus_medium)
        .map_err(|error| format!("FEM plus-i medium adapter: {error}"))?;
    let robin = plus_fem.outgoing_robin_coefficient(Complex64::new(2.0, 0.0));
    if plus_bem.harmonic_factor != -1.0
        || (plus_fem.time_sign, plus_fem.spatial_sign) != (1, -1)
        || robin != Complex64::new(0.0, 2.0)
    {
        return Err(format!(
            "plus-i convention mismatch: BEM sign={}, FEM signs=({}, {}), robin={robin:?}",
            plus_bem.harmonic_factor, plus_fem.time_sign, plus_fem.spatial_sign
        ));
    }

    let mut invalid_normals = medium.clone();
    invalid_normals.conventions.outward_normals = false;
    let bem_rejected_inward = matches!(
        PhysicsParams::try_new_medium(invalid_normals.clone(), false),
        Err(PhysicsParamsError::InvalidConvention { .. })
    );
    let fem_rejected_inward = matches!(
        FemConvention::from_medium(&invalid_normals),
        Err(FemConventionError::InwardNormals)
    );
    let speaker_rejected_inward = matches!(
        ExchangePhasorBridge::from_xem(invalid_normals.conventions),
        Err(ConventionError::InwardExchangeNormals)
    );

    let mut invalid_frame = medium;
    invalid_frame.conventions.coordinate_frame = CoordinateFrame::SourceLocalRightHanded;
    let bem_rejected_frame = matches!(
        PhysicsParams::try_new_medium(invalid_frame.clone(), false),
        Err(PhysicsParamsError::InvalidConvention { .. })
    );
    let fem_rejected_frame = matches!(
        FemConvention::from_medium(&invalid_frame),
        Err(FemConventionError::NonWorldCoordinateFrame)
    );
    let speaker_rejected_frame = matches!(
        ExchangePhasorBridge::from_xem(invalid_frame.conventions),
        Err(ConventionError::UnsupportedExchangeCoordinateFrame)
    );
    let rejection_count = [
        bem_rejected_inward,
        fem_rejected_inward,
        speaker_rejected_inward,
        bem_rejected_frame,
        fem_rejected_frame,
        speaker_rejected_frame,
    ]
    .into_iter()
    .filter(|rejected| *rejected)
    .count();
    if rejection_count != 6 {
        return Err(format!(
            "typed convention rejection coverage incomplete: {rejection_count}/4"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("adapter_count", 4.0)
        .metric("default_wavenumber_relative_error", default_wavenumber_error)
        .metric("default_bem_spatial_sign", bem.harmonic_factor)
        .metric("plus_bem_spatial_sign", plus_bem.harmonic_factor)
        .metric("characteristic_impedance_pa_s_per_m", ambient.characteristic_impedance())
        .metric("speaker_phasor_round_trip_error", speaker_round_trip_error)
        .metric("backend_admission_count", backend_admission_count as f64)
        .metric("typed_rejection_count", rejection_count as f64)
        .maximum("default_wavenumber_relative_error", 1.0e-14)
        .maximum("speaker_phasor_round_trip_error", 1.0e-14)
        .minimum("adapter_count", 4.0)
        .minimum("backend_admission_count", 1.0)
        .minimum("typed_rejection_count", 6.0)
        .input_checksum("analytic/deterministic-fixture:xem-consumer-adapters-v1")
        .reference("independent typed-quantity recomputation and convention-sign matrix")
        .reference_checksum("analytic/deterministic-fixture")
        .dimension("one typed acoustic medium; default and conjugate harmonic conventions")
        .formulation("XEM -> BEM PhysicsParams, XEM -> FEM FemConvention/HelmholtzProblem, BEM material, and speaker phasor adapters")
        .tolerance("wavenumber and phasor round-trip error <= 1e-14; six invalid convention paths rejected")
        .seed("xem-consumer-adapters:sign-and-rejection-matrix:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("XEM consumer adapter result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("XEM consumer adapter failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
