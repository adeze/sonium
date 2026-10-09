//! Regression tests for `reviews/spark-sonium-xem-20260904.md` §2 items:
//! boundary validation, ambient-medium propagation, fallible output-dir
//! twins, idempotent pool installation, size-gated parallel slices, and
//! fallible multiphysics accessors.

use num_complex::Complex64;
use sonium_xem::io::{
    read_external_multiphysics_document, try_binary_output_dir, try_sonium_output_root,
};
use sonium_xem::runtime::ensure_global_rayon_pool;
use sonium_xem::{
    AmbientMediumConfig, OutputError, ParallelPolicy, Point3D, RoomGeometry, RoomMesh,
    RoomSimulation, SLICE_CELLS_PER_THREAD, SurfaceConfig, VisualizationConfig,
    create_default_config, try_generate_spatial_slices, try_generate_spatial_slices_par,
    use_parallel,
};

#[test]
fn boundary_validation_rejects_unphysical_surfaces() {
    let mut config = create_default_config();
    config.validate().expect("default config is valid");

    config.boundaries.floor = SurfaceConfig::Absorption { coefficient: 5.0 };
    assert!(
        config.validate().is_err(),
        "absorption 5.0 must fail Rust-side validation"
    );

    config.boundaries.floor = SurfaceConfig::Impedance {
        real: f64::NAN,
        imag: 0.0,
    };
    assert!(
        config.validate().is_err(),
        "NaN impedance must fail Rust-side validation"
    );

    config.boundaries.floor = SurfaceConfig::PorousLayer {
        thickness: 0.0,
        flow_resistivity: 10_000.0,
        air_gap: None,
        ambient: AmbientMediumConfig::default(),
        model: sonium_xem::PorousModelConfig::DelanyBazley,
    };
    assert!(
        config.validate().is_err(),
        "zero-thickness porous layer must fail Rust-side validation"
    );
}

#[test]
fn to_simulation_propagates_ambient_medium() {
    let mut config = create_default_config();
    config.medium.speed_of_sound = 344.0;
    let simulation = config.to_simulation().expect("custom medium is valid");
    assert_eq!(
        simulation.speed_of_sound, 344.0,
        "to_simulation must propagate the ambient medium, not hardcode 343"
    );
    let expected = 2.0 * std::f64::consts::PI * 100.0 / 344.0;
    assert!(
        (simulation.wavenumber(100.0) - expected).abs() < 1e-12,
        "wavenumber must derive from the propagated medium"
    );
}

#[test]
fn output_dir_twins_agree_and_honor_env_override() {
    let tmp = tempfile::tempdir().expect("tempdir");
    // SAFETY: single-threaded test process for env mutation.
    unsafe {
        std::env::set_var("SONIUM_OUTPUT_DIR", tmp.path());
    }
    let root = try_sonium_output_root().expect("env override resolves");
    assert_eq!(root, tmp.path());
    assert_eq!(
        sonium_xem::io::sonium_output_root(),
        root,
        "bare and fallible roots must agree"
    );
    let dir = try_binary_output_dir("crate", "bin").expect("creatable");
    assert!(dir.is_dir());
    assert_eq!(
        sonium_xem::io::binary_output_dir("crate", "bin"),
        dir,
        "bare and fallible binary dirs must agree"
    );
    unsafe {
        std::env::remove_var("SONIUM_OUTPUT_DIR");
    }
}

#[test]
fn pool_installation_is_idempotent() {
    let first = ensure_global_rayon_pool(0, None).expect("first install");
    let second = ensure_global_rayon_pool(0, None).expect("second install");
    assert!(first >= 1 && second >= 1);
}

fn constant_pressure(
    _mesh: &RoomMesh,
    surface: &[Complex64],
    points: &[Point3D],
    _k: f64,
    _freq: f64,
) -> Vec<Complex64> {
    let scale = surface.first().copied().unwrap_or(Complex64::new(1.0, 0.0));
    points
        .iter()
        .map(|p| scale * Complex64::new(p.x + p.y + p.z + 1.0, 0.0))
        .collect()
}

fn simulation_with_res(res: usize) -> (RoomSimulation, VisualizationConfig) {
    let sim = create_default_config()
        .to_simulation()
        .expect("default config converts");
    let vis = VisualizationConfig {
        slice_resolution: res,
        ..VisualizationConfig::default()
    };
    (sim, vis)
}

#[test]
fn parallel_slices_match_serial_through_the_shared_core() {
    // The gated entry point agrees with the always-serial twin on small
    // inputs, whatever this machine's gate decides.
    let (sim, vis) = simulation_with_res(4);
    let mesh = RoomMesh::default();
    let solutions = vec![(100.0, vec![Complex64::new(1.0, 0.0); 8])];
    let serial = try_generate_spatial_slices(&sim, &vis, &mesh, &solutions, constant_pressure)
        .expect("serial slices");
    let par = try_generate_spatial_slices_par(&sim, &vis, &mesh, &solutions, constant_pressure)
        .expect("gated slices");
    assert_eq!(par, serial);

    // Rejection paths agree on both entry points.
    let bad_vis = VisualizationConfig {
        slice_resolution: 1,
        ..VisualizationConfig::default()
    };
    assert_eq!(
        try_generate_spatial_slices_par(&sim, &bad_vis, &mesh, &solutions, constant_pressure),
        Err(OutputError::SliceResolutionTooSmall { res: 1 })
    );
}

#[test]
fn slice_gate_is_machine_adaptive() {
    // The gate scales with thread count instead of using a fixed constant:
    // an 8-thread policy parallelizes at 8× the grain, a serial policy
    // never does, and `SONIUM_SERIAL`-style forcing is expressible without
    // environment mutation.
    let eight = ParallelPolicy::with_threads(8);
    let grain = SLICE_CELLS_PER_THREAD;
    assert!(eight.decide(8 * grain, grain));
    assert!(!eight.decide(8 * grain - 1, grain));
    assert!(!ParallelPolicy::serial().decide(usize::MAX, grain));
    assert!(!ParallelPolicy::with_threads(1).decide(usize::MAX, grain));
    // The convenience gate agrees with an env-resolved policy on this
    // machine for a representative above-grain workload.
    let threads = ParallelPolicy::from_env().threads().max(1);
    let big = threads.saturating_mul(grain).saturating_mul(2);
    assert_eq!(
        use_parallel(big, grain),
        ParallelPolicy::from_env().decide(big, grain)
    );
}

#[test]
fn multiphysics_fallible_accessors_agree_with_bare_ones() {
    let text = include_str!("../fixtures/multiphysics-problem-v1.json");
    let document = read_external_multiphysics_document(text).expect("fixture is valid");
    assert_eq!(document.try_id(), Ok(document.id()));
    assert_eq!(
        document.try_domains().map(|d| d.len()),
        Ok(document.domains().len())
    );
    assert_eq!(
        document.try_fields_array().map(|f| f.len()),
        Ok(document.fields_array().len())
    );
    assert_eq!(document.try_id(), Ok("cross-language-multiphysics-v1"));
}

#[test]
fn default_simulation_uses_rectangular_room() {
    let sim = create_default_config()
        .to_simulation()
        .expect("default config converts");
    assert!(matches!(sim.room, RoomGeometry::Rectangular(_)));
}
