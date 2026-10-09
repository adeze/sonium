//! Interpolation and IO demo
//!
//! Translated from DOLFINx cpp/demo/interpolation-io/main.cpp
//!
//! Demonstrates:
//! - Interpolating scalar functions into P1 Lagrange space
//! - Interpolating vector fields using H(curl) Nedelec space
//! - Complex-valued field interpolation
//! - VTK output for all fields
//!
//! DOLFINx uses Nedelec N2 (degree 2) and outputs to VTX (ADIOS2).
//! We use Nedelec N1 (degree 1) and output to VTU (ParaView).

use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::spaces::{FunctionSpace, H1Space, HcurlSpace};

fn main() {
    println!("=== DOLFINx Demo: Interpolation and IO ===\n");

    // --- 1. Create mesh ---
    let mesh = unit_square_triangles(32);
    let pi = std::f64::consts::PI;
    println!(
        "Mesh: {} nodes, {} elements\n",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    // --- 2. Scalar Lagrange interpolation ---
    // Interpolate f(x,y) = sin(2πx) into P1 space
    println!("1. Scalar P1 Lagrange interpolation");
    let space_p1 = H1Space::new(PolynomialDegree::P1);
    let n_scalar = space_p1.num_global_dofs(&mesh);
    println!("   DOFs: {n_scalar}");

    let scalar_field: Vec<f64> = mesh.nodes.iter().map(|n| (2.0 * pi * n.x).sin()).collect();

    let mut w1 = VtuWriter::new(&mesh);
    w1.add_point_scalar("f", &scalar_field);
    match w1.write("interpolation_scalar.vtu") {
        Ok(_) => println!("   Wrote interpolation_scalar.vtu"),
        Err(e) => println!("   VTK error: {e}"),
    }

    // --- 3. H(curl) Nedelec interpolation ---
    // DOLFINx: Nedelec N2 degree 2. We use N1 (lowest-order).
    // Vector field: E = [sin(πy), cos(πx)]
    println!("\n2. H(curl) Nedelec interpolation");
    let space_hcurl = HcurlSpace::new(); // N1, 1 DOF per edge
    let n_hcurl = space_hcurl.num_global_dofs(&mesh);
    println!("   Nedelec N1 DOFs: {n_hcurl} (one per edge)");

    // Interpolate E = [sin(πy), cos(πx)] into edge DOFs.
    // For Nedelec, the DOF is the tangential line integral along each edge:
    //   DOF_e = ∫_e E · t ds
    // For lowest-order, evaluate at edge midpoint × edge length:
    //   DOF_e ≈ E(midpoint) · tangent × length
    let topo = sonium_fem::mesh::topology::MeshTopology::build(&mesh);
    let mut edge_values = vec![0.0; n_hcurl];

    for (edge_idx, &(n0, n1)) in topo.edges.iter().enumerate() {
        let p0 = &mesh.nodes[n0];
        let p1 = &mesh.nodes[n1];
        let mx = 0.5 * (p0.x + p1.x);
        let my = 0.5 * (p0.y + p1.y);
        let tx = p1.x - p0.x;
        let ty = p1.y - p0.y;
        let _len = (tx * tx + ty * ty).sqrt();

        // E at midpoint
        let ex = (pi * my).sin();
        let ey = (pi * mx).cos();

        // Tangential integral: E · t × length (already unit tangent × length = tangent vector)
        edge_values[edge_idx] = ex * tx + ey * ty;
    }

    // Project edge values to nodes for VTK visualization
    // Reconstruct E at each node by averaging contributions from adjacent edges
    let mut ex_nodes = vec![0.0; mesh.num_nodes()];
    let mut ey_nodes = vec![0.0; mesh.num_nodes()];
    let mut node_edge_count = vec![0u32; mesh.num_nodes()];

    for (edge_idx, &(n0, n1)) in topo.edges.iter().enumerate() {
        let p0 = &mesh.nodes[n0];
        let p1 = &mesh.nodes[n1];
        let tx = p1.x - p0.x;
        let ty = p1.y - p0.y;
        let len = (tx * tx + ty * ty).sqrt();
        if len < 1e-15 {
            continue;
        }

        // Reconstruct vector from edge DOF: E ≈ (DOF / len²) * tangent
        let scale = edge_values[edge_idx] / (len * len);
        let evx = scale * tx;
        let evy = scale * ty;

        for &ni in &[n0, n1] {
            ex_nodes[ni] += evx;
            ey_nodes[ni] += evy;
            node_edge_count[ni] += 1;
        }
    }
    for i in 0..mesh.num_nodes() {
        if node_edge_count[i] > 0 {
            ex_nodes[i] /= node_edge_count[i] as f64;
            ey_nodes[i] /= node_edge_count[i] as f64;
        }
    }

    let ez_nodes = vec![0.0; mesh.num_nodes()];
    let mut w2 = VtuWriter::new(&mesh);
    w2.add_point_vector("E_nedelec", &ex_nodes, &ey_nodes, &ez_nodes);

    // Also write the exact field for comparison
    let ex_exact: Vec<f64> = mesh.nodes.iter().map(|n| (pi * n.y).sin()).collect();
    let ey_exact: Vec<f64> = mesh.nodes.iter().map(|n| (pi * n.x).cos()).collect();
    w2.add_point_vector("E_exact", &ex_exact, &ey_exact, &ez_nodes);

    match w2.write("interpolation_nedelec.vtu") {
        Ok(_) => println!("   Wrote interpolation_nedelec.vtu"),
        Err(e) => println!("   VTK error: {e}"),
    }

    // --- 4. Complex-valued field ---
    println!("\n3. Complex-valued field interpolation");
    let complex_field: Vec<Complex64> = mesh
        .nodes
        .iter()
        .map(|n| Complex64::new((pi * n.x).sin(), (pi * n.y).cos()))
        .collect();

    let mut w3 = VtuWriter::new(&mesh);
    w3.add_point_scalar_complex("psi", &complex_field);
    match w3.write("interpolation_complex.vtu") {
        Ok(_) => println!("   Wrote interpolation_complex.vtu (magnitude, real, imag)"),
        Err(e) => println!("   VTK error: {e}"),
    }

    // --- 5. Summary ---
    println!("\n--- Summary ---");
    println!("Scalar P1:      {} DOFs, f(x,y) = sin(2πx)", n_scalar);
    println!("Nedelec N1:     {} DOFs, E = [sin(πy), cos(πx)]", n_hcurl);
    println!("Complex P1:     {} DOFs, ψ = sin(πx) + i·cos(πy)", n_scalar);

    println!("\n=== Demo Complete ===");
}
