//! FEM VTU/PVD output-path contract.
//!
//! This binary must contain only this test: it drives VTU output through the
//! process-global `SONIUM_ROOM_VTU_DIR` variable, which is unsound to share
//! with sibling tests in one process. Process isolation (one integration
//! binary) is what makes the output directory exclusive.

// Rust guideline compliant 2026-02-21

use sonium_sim_room::fem_runner::{FemMeshConfig, run_fem_full_domain_with_mesh_report};
use sonium_xem::{Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source};

#[test]
fn fem_vtu_output_path_writes_parseable_mesh_and_source_collection() {
    let output_dir = tempfile::tempdir().expect("temporary VTU directory");
    // SAFETY: no other test runs in this binary, so no other thread reads or
    // modifies the process environment concurrently with this solve.
    unsafe { std::env::set_var("SONIUM_ROOM_VTU_DIR", output_dir.path()) };

    let simulation = RoomSimulation::with_frequencies(
        RoomGeometry::Rectangular(RectangularRoom::new(1.0, 1.0, 1.0)),
        vec![Source::omnidirectional(Point3D::new(0.25, 0.25, 0.25), 1.0)],
        vec![Point3D::new(0.75, 0.75, 0.75)],
        100.0,
        100.0,
        1,
    );

    let config = FemMeshConfig {
        modal_damping_ratio: 0.01,
        ..FemMeshConfig::default()
    };
    run_fem_full_domain_with_mesh_report(&simulation, &config)
        .expect("one-frequency FEM solve should succeed");

    let mesh_path = output_dir.path().join("fem_room_mesh.vtu");
    let pvd_path = output_dir.path().join("fem_source_Source.pvd");
    assert!(mesh_path.is_file(), "missing FEM mesh artifact");
    assert!(pvd_path.is_file(), "missing FEM source collection artifact");
    sonium_xem::io::vtu::validate_vtu_file(&mesh_path).expect("VTU must parse");
    assert_eq!(
        sonium_xem::io::pvd::validate_pvd_file(&pvd_path).expect("PVD must validate"),
        1
    );

    unsafe { std::env::remove_var("SONIUM_ROOM_VTU_DIR") };
}
