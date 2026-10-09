//! Computed shoebox-room solver modal benchmark.
//!
//! This uses the assembled FEM stiffness and mass matrices on a rigid
//! rectangular room and evaluates Rayleigh quotients for analytical Neumann
//! mode shapes. It verifies that the discretized Helmholtz operator places
//! low-frequency room modes near their closed-form shoebox frequencies.

use sonium_fem::assembly::{assemble_mass, assemble_stiffness};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_xem::io::binary_output_dir;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
struct ShoeboxRoom {
    width: f64,
    depth: f64,
    height: f64,
    sound_speed: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ModeIndex {
    nx: u32,
    ny: u32,
    nz: u32,
}

impl ShoeboxRoom {
    fn analytical_frequency(self, mode: ModeIndex) -> f64 {
        let kx = mode.nx as f64 / self.width;
        let ky = mode.ny as f64 / self.depth;
        let kz = mode.nz as f64 / self.height;
        0.5 * self.sound_speed * (kx * kx + ky * ky + kz * kz).sqrt()
    }

    fn mode_value(self, mode: ModeIndex, p: &Point) -> f64 {
        let x = (mode.nx as f64 * PI * p.x / self.width).cos();
        let y = (mode.ny as f64 * PI * p.y / self.depth).cos();
        let z = (mode.nz as f64 * PI * p.z / self.height).cos();
        x * y * z
    }
}

fn shoebox_mesh(room: ShoeboxRoom, nx: usize, ny: usize, nz: usize) -> Mesh {
    let mut mesh = Mesh::new(3);
    let dx = room.width / nx as f64;
    let dy = room.depth / ny as f64;
    let dz = room.height / nz as f64;

    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.add_node(Point::new_3d(i as f64 * dx, j as f64 * dy, k as f64 * dz));
            }
        }
    }

    let node =
        |i: usize, j: usize, k: usize| -> usize { k * (ny + 1) * (nx + 1) + j * (nx + 1) + i };

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let v0 = node(i, j, k);
                let v1 = node(i + 1, j, k);
                let v2 = node(i, j + 1, k);
                let v3 = node(i + 1, j + 1, k);
                let v4 = node(i, j, k + 1);
                let v5 = node(i + 1, j, k + 1);
                let v6 = node(i, j + 1, k + 1);
                let v7 = node(i + 1, j + 1, k + 1);

                mesh.add_element(ElementType::Tetrahedron, vec![v0, v1, v3, v7]);
                mesh.add_element(ElementType::Tetrahedron, vec![v0, v1, v7, v5]);
                mesh.add_element(ElementType::Tetrahedron, vec![v0, v5, v7, v4]);
                mesh.add_element(ElementType::Tetrahedron, vec![v0, v3, v2, v7]);
                mesh.add_element(ElementType::Tetrahedron, vec![v0, v2, v6, v7]);
                mesh.add_element(ElementType::Tetrahedron, vec![v0, v6, v4, v7]);
            }
        }
    }

    mesh
}

fn quadratic_form(rows: &[usize], cols: &[usize], values: &[f64], u: &[f64]) -> f64 {
    rows.iter()
        .zip(cols)
        .zip(values)
        .map(|((&row, &col), &value)| u[row] * value * u[col])
        .sum()
}

fn computed_frequency(room: ShoeboxRoom, mesh: &Mesh, mode: ModeIndex) -> Result<f64, String> {
    let stiffness = assemble_stiffness(mesh, PolynomialDegree::P1);
    let mass = assemble_mass(mesh, PolynomialDegree::P1);
    let u: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|p| room.mode_value(mode, p))
        .collect();

    let ku = quadratic_form(&stiffness.rows, &stiffness.cols, &stiffness.values, &u);
    let mu = quadratic_form(&mass.rows, &mass.cols, &mass.values, &u);
    if mu <= 0.0 {
        return Err(format!(
            "mode {mode:?} produced non-positive mass norm {mu}"
        ));
    }

    let omega_h = (ku / mu).sqrt() * room.sound_speed;
    Ok(omega_h / (2.0 * PI))
}

fn validate_mode(
    room: ShoeboxRoom,
    mesh: &Mesh,
    mode: ModeIndex,
    max_relative_error: f64,
) -> Result<(), String> {
    let computed = computed_frequency(room, mesh, mode)?;
    let analytical = room.analytical_frequency(mode);
    let rel = (computed - analytical).abs() / analytical;

    println!(
        "  mode ({}, {}, {}): FEM {:.4} Hz vs analytical {:.4} Hz, rel error {:.2}%",
        mode.nx,
        mode.ny,
        mode.nz,
        computed,
        analytical,
        rel * 100.0
    );

    if rel > max_relative_error {
        return Err(format!(
            "mode {mode:?} relative error {:.2}% exceeds {:.2}%",
            rel * 100.0,
            max_relative_error * 100.0
        ));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let room = ShoeboxRoom {
        width: 5.0,
        depth: 4.0,
        height: 2.5,
        sound_speed: 343.0,
    };
    let mesh = shoebox_mesh(room, 8, 6, 4);

    println!(
        "Computed shoebox FEM modal benchmark: {} nodes, {} tetrahedra",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let _ = VtuWriter::new(&mesh).write(dir.join("shoebox_mesh.vtu"));
    validate_mode(
        room,
        &mesh,
        ModeIndex {
            nx: 1,
            ny: 0,
            nz: 0,
        },
        0.08,
    )?;
    validate_mode(
        room,
        &mesh,
        ModeIndex {
            nx: 0,
            ny: 1,
            nz: 0,
        },
        0.08,
    )?;
    validate_mode(
        room,
        &mesh,
        ModeIndex {
            nx: 0,
            ny: 0,
            nz: 1,
        },
        0.12,
    )?;
    validate_mode(
        room,
        &mesh,
        ModeIndex {
            nx: 1,
            ny: 1,
            nz: 0,
        },
        0.10,
    )?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAILED: {error}");
        println!("FAILED");
        std::process::exit(1);
    }

    println!("SUCCESS");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computed_shoebox_modes_track_analytic_frequencies() {
        run().expect("shoebox solver modes should stay near analytic rigid-room modes");
    }
}
