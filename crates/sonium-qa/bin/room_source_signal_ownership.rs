//! Compile-only RoomSim source-signal ownership contract.
//!
//! This fixture exercises setup validation only. It intentionally does not
//! launch a FEM solve, QA gate, or simulator workload.

use num_complex::Complex64;
use sonium_compute::HostMatrix;
use sonium_sim_room::fem_runner::{FemMeshConfig, validate_source_signal_ownership};
use sonium_xem::{
    ComplexDirectivityGrid, CrossoverFilter, Directivity, Point3D, RectangularRoom, RoomGeometry,
    RoomSimulation, Source,
};

fn main() {
    let grid = ComplexDirectivityGrid::try_new(
        vec![0.0, 90.0, 180.0, 270.0],
        vec![0.0, 90.0, 180.0],
        vec![500.0],
        vec![HostMatrix::from_elem(3, 4, Complex64::new(1.0, 0.0))],
    )
    .expect("compile-only complex directivity grid must be valid");
    let source = Source::new(
        Point3D::new(1.0, 1.0, 1.0),
        Directivity::ComplexGrid(grid),
        1.0,
    )
    .with_crossover(CrossoverFilter::Lowpass {
        cutoff_freq: 500.0,
        order: 2,
    });
    let simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(4.0, 3.0, 2.0)),
        vec![source],
        vec![Point3D::new(2.0, 1.0, 1.0)],
    );

    let default_config = FemMeshConfig::default();
    assert!(validate_source_signal_ownership(&simulation, &default_config).is_err());

    let mut chained_config = default_config;
    chained_config.allow_chained_complex_crossover = true;
    validate_source_signal_ownership(&simulation, &chained_config)
        .expect("explicit chained-crossover opt-in must pass");
    println!("SUCCESS");
}
