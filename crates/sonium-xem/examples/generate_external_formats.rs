//! Generate a minimal, deterministic VTU/PVD/SOFA artifact set for external
//! reader release smoke tests.
//!
//! The companion `interop/read_external_formats.py` deliberately does not
//! depend on this crate: it validates the emitted XML with ElementTree and the
//! SOFA HDF5 file with h5py.

use sonium_xem::{
    Point3D, RoomMesh, SurfaceElement,
    io::{
        PvdCollection, VtuWriter,
        sofa::{CoordinateSystem, SimpleFreeFieldHrtf, write_simple_free_field_hrtf},
    },
};
use std::{env, fs, path::Path};

fn tiny_mesh() -> RoomMesh {
    RoomMesh {
        nodes: vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ],
        elements: vec![
            SurfaceElement::triangle(0, 1, 2),
            SurfaceElement::triangle(0, 2, 3),
        ],
    }
}

fn write_fixture(output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;

    let mesh = tiny_mesh();
    let mut vtu = VtuWriter::new(&mesh);
    vtu.try_add_point_scalar("temperature", &[1.0, 2.0, 3.0, 4.0])?;
    vtu.try_add_cell_scalar("region", &[7.0, 8.0])?;
    vtu.write(output_dir.join("frame_000.vtu"))?;

    let mut pvd = PvdCollection::new(output_dir.join("series.pvd"));
    pvd.try_push(0.0, "frame_000.vtu")?;
    pvd.finish()?;

    let real = [1.0, 2.0, 3.0, 4.0];
    let imag = [0.1, 0.2, 0.3, 0.4];
    let frequencies = [500.0, 1000.0];
    let source_position = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    let receiver_position = [0.0, 0.0, 0.0];
    let hrtf = SimpleFreeFieldHrtf {
        real: &real,
        imag: &imag,
        frequencies: &frequencies,
        source_position: &source_position,
        source_coords: CoordinateSystem::Cartesian,
        receiver_position: &receiver_position,
        receiver_coords: CoordinateSystem::Cartesian,
        listener_position: None,
        title: Some("Sonium external-format smoke"),
        measurements: 2,
        receivers: 1,
    };
    write_simple_free_field_hrtf(output_dir.join("minimal.sofa"), &hrtf)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let Some(output_dir) = arguments.next() else {
        return Err(format!(
            "usage: {} <output-directory>",
            Path::new(&program).display()
        )
        .into());
    };
    if arguments.next().is_some() {
        return Err("expected exactly one output directory".into());
    }
    write_fixture(Path::new(&output_dir))
}
