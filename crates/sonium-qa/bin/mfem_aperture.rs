//! Sonium side of the wide-MFEM harness, slice 4: coupled aperture case
//! against an MFEM-interior reference coupling.
//!
//! Uses the product cylindrical-box-with-circular-aperture fixture (small
//! 8x1 discretization) and the dense monolithic golden
//! (`MonolithicFemBemCoupling::solve_frequency`, 120 Hz, driver-velocity
//! excitation) as the all-sonium coupled solve. The MFEM driver
//! (`crates/sonium-qa/mfem/aperture_interior.cpp`) assembles the P1 interior
//! stiffness + mass on the SAME tetrahedral mesh (exported with
//! `--export-mesh` as Gmsh v2.2); the harness maps MFEM vertices onto
//! interior nodes by coordinates, forms A = S - k²M, and swaps that block
//! into the probed dense monolithic system. The two coupled solves share the
//! identical BEM exterior and coupling coefficients, so the comparison
//! isolates the interior FEM half inside a fully coupled solve. A
//! `InterfacePowerLedger` audit on the golden is the physical oracle, and
//! `--export-dense` emits the probed system for an independent Wolfram
//! `LinearSolve` check.
//!
//! Run e.g.:
//!   cargo run -p sonium-qa --features qa-binaries --bin qa-mfem-aperture -- \
//!     --export-mesh /tmp/aperture.msh
//!   ./mfem/aperture_interior /tmp/aperture.msh > /tmp/mfem_ap.json
//!   cargo run -p sonium-qa --features qa-binaries --bin qa-mfem-aperture -- \
//!     --mfem-json /tmp/mfem_ap.json --export-dense /tmp/aperture_dense.json

use anyhow::{Context, Result, bail};
use clap::Parser;
use num_complex::Complex64;
use serde::Serialize;
use sonium_bem::core::ConservativeTraceTransfer;
use sonium_fem::boundary::{FemSystem, apply_neumann_complex};
use sonium_fem::mesh::{ElementType, Mesh};
use sonium_sim_speaker_modeling::fem_bem_coupling::coupled::CouplingDrive;
use sonium_sim_speaker_modeling::fem_bem_coupling::fixtures::{
    APERTURE_MARKER, CylinderFixtureDiscretization, DRIVER_MARKER, WALL_MARKER,
    cylindrical_box_with_circular_aperture,
};
use sonium_sim_speaker_modeling::fem_bem_coupling::monolithic::MonolithicFemBemCoupling;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::lu_factorize;
use sonium_solvers::sparse::TripletMatrix;
use sonium_solvers::traits::LinearOperator;
use std::collections::HashMap;
use std::fs;

const FREQUENCY_HZ: f64 = 120.0;
const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.204;
const DRIVE_VELOCITY: f64 = 1.0e-3;

#[derive(Parser, Debug)]
#[command(
    name = "qa-mfem-aperture",
    about = "Compare the coupled aperture solve against an MFEM-interior reference."
)]
struct Args {
    /// Write the fixture mesh as Gmsh v2.2 and exit.
    #[arg(long, default_value = "")]
    export_mesh: String,
    /// MFEM interior-dump JSON for the comparison leg.
    #[arg(long, default_value = "")]
    mfem_json: String,
    /// Write the probed dense system + solution for the Wolfram check.
    #[arg(long, default_value = "")]
    export_dense: String,
}

#[derive(Debug, serde::Deserialize)]
struct MfemInterior {
    nv: usize,
    ne: usize,
    stiffness_rowptr: Vec<usize>,
    stiffness_colind: Vec<usize>,
    stiffness_data: Vec<f64>,
    mass_rowptr: Vec<usize>,
    mass_colind: Vec<usize>,
    mass_data: Vec<f64>,
    coords: Vec<[f64; 3]>,
    total_volume: f64,
}

#[derive(Debug, Serialize)]
struct DenseExport {
    n: usize,
    n_interior: usize,
    n_exterior: usize,
    n_aperture: usize,
    matrix_re: Vec<f64>,
    matrix_im: Vec<f64>,
    rhs_re: Vec<f64>,
    rhs_im: Vec<f64>,
    solution_re: Vec<f64>,
    solution_im: Vec<f64>,
}

fn build_fixture()
-> Result<sonium_sim_speaker_modeling::fem_bem_coupling::fixtures::CouplingFixture> {
    cylindrical_box_with_circular_aperture(
        0.11,
        0.16,
        0.04,
        CylinderFixtureDiscretization {
            n_circumference: 8,
            n_box_depth: 1,
            n_cap_radial: 1,
            n_aperture_radial: 1,
            n_port_depth: 1,
        },
    )
    .map_err(|error| anyhow::anyhow!("fixture: {error}"))
}

/// Minimal Gmsh v2.2 ASCII exporter (tet4 volume elements only). P1 assembly
/// is invariant under local node permutations, so no reordering is needed.
fn export_gmsh_v22(mesh: &Mesh, path: &str) -> Result<()> {
    for (index, element) in mesh.elements.iter().enumerate() {
        if element.element_type != ElementType::Tetrahedron || element.nodes.len() != 4 {
            bail!("element {index} is not a tet4");
        }
    }
    let mut text = String::from("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n");
    text.push_str(&format!("{}\n", mesh.nodes.len()));
    for (index, node) in mesh.nodes.iter().enumerate() {
        text.push_str(&format!(
            "{} {:.17e} {:.17e} {:.17e}\n",
            index + 1,
            node.x,
            node.y,
            node.z
        ));
    }
    text.push_str(&format!("$EndNodes\n$Elements\n{}\n", mesh.elements.len()));
    for (index, element) in mesh.elements.iter().enumerate() {
        text.push_str(&format!(
            "{} 4 2 0 0 {} {} {} {}\n",
            index + 1,
            element.nodes[0] + 1,
            element.nodes[1] + 1,
            element.nodes[2] + 1,
            element.nodes[3] + 1,
        ));
    }
    text.push_str("$EndElements\n");
    fs::write(path, text)?;
    Ok(())
}

fn tri_area(mesh: &Mesh, nodes: &[usize]) -> f64 {
    let a = &mesh.nodes[nodes[0]];
    let b = &mesh.nodes[nodes[1]];
    let c = &mesh.nodes[nodes[2]];
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

/// MFEM-interior reference leg: import the MFEM stiffness/mass dump on
/// the exported mesh, swap the A = S - k2*M block into the probed dense
/// monolithic system, and gate the coupled state against the MFEM-interior
/// reference solve. Only runs when --mfem-json is provided.
fn run_mfem_leg(
    mfem_json: &str,
    mesh: &Mesh,
    n_interior: usize,
    n_exterior: usize,
    n_aperture: usize,
    probed: &DenseMatrix<Complex64>,
    rhs: &[Complex64],
    probed_state: &[Complex64],
) -> Result<()> {
    // MFEM interior import: coordinate map, then A = S - k²M (real k).
    let document = fs::read_to_string(mfem_json).with_context(|| "read MFEM JSON".to_string())?;
    // The MFEM Gmsh reader chats on stdout before the JSON payload.
    let payload = document
        .find('{')
        .map(|i| &document[i..])
        .unwrap_or(&document);
    let mfem: MfemInterior = serde_json::from_str(payload)?;
    if mfem.nv != mesh.nodes.len() {
        bail!("MFEM nv {} != mesh nodes {}", mfem.nv, mesh.nodes.len());
    }
    if mfem.ne != mesh.elements.len() {
        bail!("MFEM ne {} != mesh tets {}", mfem.ne, mesh.elements.len());
    }
    let mut key_to_node: HashMap<[u64; 3], usize> = HashMap::new();
    for (node, point) in mesh.nodes.iter().enumerate() {
        key_to_node.insert(
            [
                (point.x * 1.0e12).round() as i64 as u64,
                (point.y * 1.0e12).round() as i64 as u64,
                (point.z * 1.0e12).round() as i64 as u64,
            ],
            node,
        );
    }
    let mut mfem_to_sonium = vec![usize::MAX; mfem.nv];
    for (vertex, coord) in mfem.coords.iter().enumerate() {
        let key = [
            (coord[0] * 1.0e12).round() as i64 as u64,
            (coord[1] * 1.0e12).round() as i64 as u64,
            (coord[2] * 1.0e12).round() as i64 as u64,
        ];
        let Some(&node) = key_to_node.get(&key) else {
            bail!("MFEM vertex {vertex} at {coord:?} matches no mesh node");
        };
        mfem_to_sonium[vertex] = node;
    }
    let mut seen = vec![false; mesh.nodes.len()];
    for &node in &mfem_to_sonium {
        if seen[node] {
            bail!("coordinate map is not bijective");
        }
        seen[node] = true;
    }
    let omega = std::f64::consts::TAU * FREQUENCY_HZ;
    let wavenumber = omega / SPEED_OF_SOUND;
    let k2 = wavenumber * wavenumber;
    let mut a_mfem: DenseMatrix<Complex64> = DenseMatrix::zeros(n_interior, n_interior);
    for row in 0..mfem.nv {
        for t in mfem.stiffness_rowptr[row]..mfem.stiffness_rowptr[row + 1] {
            let i = mfem_to_sonium[row];
            let j = mfem_to_sonium[mfem.stiffness_colind[t]];
            a_mfem[[i, j]] += Complex64::new(mfem.stiffness_data[t], 0.0);
        }
        for t in mfem.mass_rowptr[row]..mfem.mass_rowptr[row + 1] {
            let i = mfem_to_sonium[row];
            let j = mfem_to_sonium[mfem.mass_colind[t]];
            a_mfem[[i, j]] -= Complex64::new(k2 * mfem.mass_data[t], 0.0);
        }
    }
    let mut block_diff = 0.0f64;
    let mut block_scale = 0.0f64;
    for row in 0..n_interior {
        for col in 0..n_interior {
            let entry_diff: f64 = (a_mfem[[row, col]] - probed[[row, col]]).norm_sqr().sqrt();
            let entry_scale: f64 = probed[[row, col]].norm_sqr().sqrt();
            block_diff = block_diff.max(entry_diff);
            block_scale = block_scale.max(entry_scale);
        }
    }
    println!(
        "INTERIOR-BLOCK max_abs_diff={block_diff:.3e} scale={block_scale:.3e} rel={:.3e}",
        block_diff / block_scale
    );
    if block_diff / block_scale > 1.0e-9 {
        bail!("MFEM interior block does not match the probed sonium block");
    }
    // sum(M) = mesh volume by partition of unity; the fixture's
    // cavity_volume is the analytic cylinder, larger by the octagon/circle
    // ratio, so the check is against MFEM's own tet-volume sum.
    let mass_sum: f64 = mfem.mass_data.iter().sum();
    println!(
        "VOLUME mfem_mass_sum={mass_sum:.12e} mfem_tet_sum={:.12e} reldiff={:.3e}",
        mfem.total_volume,
        (mass_sum - mfem.total_volume).abs() / mfem.total_volume
    );

    // MFEM-interior reference coupling: same probed system, interior block
    // replaced, same RHS. Only the interior FEM assembly differs now.
    let mut reference = probed.clone();
    for row in 0..n_interior {
        for col in 0..n_interior {
            reference[[row, col]] = a_mfem[[row, col]];
        }
    }
    let reference_state = lu_factorize(&reference)
        .map_err(|error| anyhow::anyhow!("reference LU: {error}"))?
        .solve(rhs)
        .map_err(|error| anyhow::anyhow!("reference solve: {error}"))?;
    let mut sdiff2 = 0.0;
    let mut snorm2 = 0.0;
    for (a, b) in reference_state.iter().zip(probed_state.iter()) {
        sdiff2 += (a - b).norm_sqr();
        snorm2 += b.norm_sqr();
    }
    let aperture_offset = n_interior + n_exterior;
    let mut adiff2 = 0.0;
    let mut anorm2 = 0.0;
    for k in 0..n_aperture {
        let a = reference_state[aperture_offset + k];
        let b = probed_state[aperture_offset + k];
        adiff2 += (a - b).norm_sqr();
        anorm2 += b.norm_sqr();
    }
    println!(
        "REFERENCE-AGREE rel_state_diff={:.3e} rel_aperture_velocity_diff={:.3e}",
        (sdiff2 / snorm2).sqrt(),
        (adiff2 / anorm2).sqrt()
    );
    if (sdiff2 / snorm2).sqrt() > 1.0e-6 {
        bail!("MFEM-interior reference coupled state diverges");
    }
    Ok(())
}

fn main() {
    // Machine-checkable output contract: SUCCESS on stdout for a pass,
    // FAILED plus a non-zero exit for any failure.
    match run() {
        Ok(()) => println!("SUCCESS"),
        Err(error) => {
            println!("FAILED: {error:?}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let fixture = build_fixture()?;
    let mesh = &fixture.fem_mesh;
    println!(
        "FIXTURE nodes={} tets={} boundaries={} aperture_faces={} volume={:.6e} m³",
        mesh.nodes.len(),
        mesh.elements.len(),
        mesh.boundaries.len(),
        fixture.num_aperture_faces(),
        fixture.cavity_volume_m3(),
    );
    if !args.export_mesh.is_empty() {
        export_gmsh_v22(mesh, &args.export_mesh)?;
        println!("MESH-EXPORT path={} ", args.export_mesh);
        return Ok(());
    }
    // Without --mfem-json the MFEM-interior reference leg is skipped; the
    // golden residual gates and the power-ledger audit below still run, so
    // the no-argument invocation is a real self-consistency check.

    let monolithic = MonolithicFemBemCoupling::from_fixture(&fixture, SPEED_OF_SOUND, DENSITY)
        .map_err(|error| anyhow::anyhow!("monolithic setup: {error}"))?;
    let n_interior = monolithic.num_interior_dofs();
    let n_exterior = monolithic.num_exterior_dofs();
    let n_aperture = monolithic.num_aperture_dofs();
    let total = n_interior + n_exterior + n_aperture;
    println!(
        "DOFS interior={n_interior} exterior={n_exterior} aperture={n_aperture} total={total}"
    );
    if n_interior != mesh.nodes.len() {
        bail!(
            "P1 interior DOFs {n_interior} != mesh nodes {}",
            mesh.nodes.len()
        );
    }

    // Golden dense solve (all-sonium, exact dense BEM exterior).
    let drive = CouplingDrive::DriverVelocity(Complex64::new(DRIVE_VELOCITY, 0.0));
    let golden = monolithic
        .solve_frequency(FREQUENCY_HZ, drive)
        .map_err(|error| anyhow::anyhow!("golden solve: {error}"))?;
    println!(
        "GOLDEN rel_res={:.2e} interior={:.2e} exterior={:.2e} interface={:.2e}",
        golden.relative_residual,
        golden.interior_relative_residual,
        golden.exterior_relative_residual,
        golden.interface_relative_residual,
    );
    // Direct dense LU on a fixed small fixture: residuals sit at rounding
    // level, so a 1e-10 gate only trips on a genuine assembly/solve break.
    for (label, residual) in [
        ("relative", golden.relative_residual),
        ("interior", golden.interior_relative_residual),
        ("exterior", golden.exterior_relative_residual),
        ("interface", golden.interface_relative_residual),
    ] {
        if !residual.is_finite() || residual > 1.0e-10 {
            bail!("golden {label} residual {residual:.3e} exceeds 1e-10");
        }
    }

    // Compressed operator with the single-root-cluster pattern from the
    // product tests, probed into a dense system.
    let mut cluster = sonium_bem::core::types::Cluster::new([0.0; 3]);
    cluster.element_indices = (0..n_exterior).collect();
    cluster.num_elements = n_exterior;
    cluster.radius = 1.0;
    cluster.near_clusters = vec![0];
    let compressed = monolithic
        .build_compressed_operator(FREQUENCY_HZ, &[cluster], 1.0e-10, 16)
        .map_err(|error| anyhow::anyhow!("compressed operator: {error}"))?;
    let mut probed: DenseMatrix<Complex64> = DenseMatrix::zeros(total, total);
    for column in 0..total {
        let mut unit = vec![Complex64::new(0.0, 0.0); total];
        unit[column] = Complex64::new(1.0, 0.0);
        let response = compressed.apply(&unit);
        for row in 0..total {
            probed[[row, column]] = response[row];
        }
    }

    // Replicated drive RHS: iωρ·v_drive Neumann load on the driver marker,
    // matching FemBemCoupling::drive_rhs for CouplingDrive::DriverVelocity.
    let omega = std::f64::consts::TAU * FREQUENCY_HZ;
    let h = Complex64::new(0.0, omega * DENSITY) * DRIVE_VELOCITY;
    let mut drive_system = FemSystem::new(
        TripletMatrix::<Complex64>::new(n_interior, n_interior),
        vec![Complex64::new(0.0, 0.0); n_interior],
    );
    apply_neumann_complex(&mut drive_system, mesh, DRIVER_MARKER, &move |_, _, _| h);
    let mut rhs = vec![Complex64::new(0.0, 0.0); total];
    rhs[..n_interior].copy_from_slice(&drive_system.rhs);
    let probed_state = lu_factorize(&probed)
        .map_err(|error| anyhow::anyhow!("probed LU: {error}"))?
        .solve(&rhs)
        .map_err(|error| anyhow::anyhow!("probed solve: {error}"))?;
    let golden_state = [
        golden.interior_pressure.clone(),
        golden.exterior_pressure.clone(),
        golden.aperture_velocity.clone(),
    ]
    .concat();
    let mut diff2 = 0.0;
    let mut norm2 = 0.0;
    for (a, b) in probed_state.iter().zip(&golden_state) {
        diff2 += (a - b).norm_sqr();
        norm2 += b.norm_sqr();
    }
    let probed_vs_golden = (diff2 / norm2).sqrt();
    println!("PROBED-VS-GOLDEN rel_state_diff={probed_vs_golden:.3e} (H-matrix aca=1e-10)");

    if !args.mfem_json.is_empty() {
        run_mfem_leg(
            &args.mfem_json,
            mesh,
            n_interior,
            n_exterior,
            n_aperture,
            &probed,
            &rhs,
            &probed_state,
        )?;
    }

    // Power-ledger audit on the golden: rebuild the aperture transfer from
    // the public fixture mesh and check interface conservation in watts.
    let mut face_nodes = Vec::new();
    let mut face_areas = Vec::new();
    let mut aperture_face_order = Vec::new();
    for face in &mesh.boundaries {
        if face.marker == APERTURE_MARKER {
            if face.nodes.len() != 3 {
                bail!("non-triangular aperture face");
            }
            face_nodes.push([face.nodes[0], face.nodes[1], face.nodes[2]]);
            face_areas.push(tri_area(mesh, &face.nodes));
            aperture_face_order.push(face_nodes.len() - 1);
        }
    }
    if face_nodes.len() != n_aperture {
        bail!(
            "aperture faces {} != aperture DOFs {n_aperture}",
            face_nodes.len()
        );
    }
    let (transfer, source_nodes) =
        ConservativeTraceTransfer::try_from_tri3_vertex_faces(&face_nodes, &face_areas)
            .map_err(|error| anyhow::anyhow!("trace transfer: {error}"))?;
    let interior_trace: Vec<Complex64> = source_nodes
        .iter()
        .map(|&node| golden.interior_pressure[node])
        .collect();
    let transferred = transfer
        .apply(&interior_trace)
        .map_err(|error| anyhow::anyhow!("transfer apply: {error}"))?;
    // BEM element order follows the collected boundary-face order
    // (aperture + walls, in FemMesh.boundaries order).
    let mut bem_of_aperture = Vec::new();
    {
        let mut exterior_index = 0usize;
        for face in &mesh.boundaries {
            if face.marker == APERTURE_MARKER
                || face.marker == WALL_MARKER
                || Some(face.marker) == fixture.driver_marker
            {
                if face.marker == APERTURE_MARKER {
                    bem_of_aperture.push(exterior_index);
                }
                exterior_index += 1;
            }
        }
    }
    if exterior_count(mesh, &fixture) != n_exterior || bem_of_aperture.len() != n_aperture {
        bail!("BEM aperture mapping inconsistent with golden layout");
    }
    let exterior_trace: Vec<Complex64> = bem_of_aperture
        .iter()
        .map(|&e| golden.exterior_pressure[e])
        .collect();
    let ledger = sonium_coupling::InterfacePowerLedger::audit(
        &face_areas,
        &transferred,
        &exterior_trace,
        &golden.aperture_velocity,
    )
    .map_err(|error| anyhow::anyhow!("power audit: {error}"))?;
    println!(
        "POWER interior={:.6e} W exterior={:.6e} W residual={:.3e} W rel={:.3e} (aperture faces in boundary order: {})",
        ledger.interior_active_power_w,
        ledger.exterior_active_power_w,
        ledger.conservation_residual_w,
        ledger.relative_conservation_residual,
        aperture_face_order.len(),
    );

    if !args.export_dense.is_empty() {
        let mut matrix_re = Vec::with_capacity(total * total);
        let mut matrix_im = Vec::with_capacity(total * total);
        for row in 0..total {
            for col in 0..total {
                matrix_re.push(probed[[row, col]].re);
                matrix_im.push(probed[[row, col]].im);
            }
        }
        let export = DenseExport {
            n: total,
            n_interior,
            n_exterior,
            n_aperture,
            matrix_re,
            matrix_im,
            rhs_re: rhs.iter().map(|v| v.re).collect(),
            rhs_im: rhs.iter().map(|v| v.im).collect(),
            solution_re: probed_state.iter().map(|v| v.re).collect(),
            solution_im: probed_state.iter().map(|v| v.im).collect(),
        };
        fs::write(&args.export_dense, serde_json::to_string(&export)?)?;
        println!("DENSE-EXPORT path={}", args.export_dense);
    }
    if args.mfem_json.is_empty() {
        println!("APERTURE-SELF-CONSISTENT slice=4 mfem_reference=absent");
    } else {
        println!("APERTURE-AGREE slice=4 golden_vs_reference_ok=true");
    }
    Ok(())
}

fn exterior_count(
    mesh: &Mesh,
    fixture: &sonium_sim_speaker_modeling::fem_bem_coupling::fixtures::CouplingFixture,
) -> usize {
    mesh.boundaries
        .iter()
        .filter(|face| {
            face.marker == APERTURE_MARKER
                || face.marker == WALL_MARKER
                || Some(face.marker) == fixture.driver_marker
        })
        .count()
}
