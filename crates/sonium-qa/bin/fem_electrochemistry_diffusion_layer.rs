use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::{BilinearForm, LinearForm};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::coupled::electrochemistry::{
    ButlerVolmerBoundaryFluxForm, ButlerVolmerKinetics, ButlerVolmerTangentForm, IonicSpecies,
    NernstPlanckDiffusionForm,
};
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn test_electrochemistry_diffusion_layer() {
    let l = 1.0;
    let h = 0.1;
    let nx = 40;
    let ny = 2;
    let diffusivity = 1.0;
    let c0 = 1.0;
    let c_l = 0.0;

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let species = IonicSpecies::from_diffusivity(diffusivity, 1.0, 298.15);
    let form = NernstPlanckDiffusionForm {
        diffusivity: species.diffusivity,
    };

    let matrix = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let rhs = vec![0.0; n_dofs];

    let mut system = FemSystem::new(matrix, rhs);
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| c0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| c_l, None);

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES did not converge");

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let exact_c = c0 + (c_l - c0) * (node.x / l);
        let err = (result.x[i] - exact_c).abs();
        if err > max_err {
            max_err = err;
        }
    }

    // Check flux constancy: J = -D * dc/dx should be constant
    let exact_flux = diffusivity * (c0 - c_l) / l;
    let mut max_flux_err: f64 = 0.0;
    for element in &mesh.elements {
        if element.nodes.len() == 3 {
            let n0 = element.nodes[0];
            let n1 = element.nodes[1];
            let n2 = element.nodes[2];
            let x0 = mesh.nodes[n0].x;
            let y0 = mesh.nodes[n0].y;
            let x1 = mesh.nodes[n1].x;
            let y1 = mesh.nodes[n1].y;
            let x2 = mesh.nodes[n2].x;
            let y2 = mesh.nodes[n2].y;

            let area2 = ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs();
            if area2 < 1e-14 {
                continue;
            }
            let inv_area2 = 1.0 / area2;

            let dc_dx = inv_area2
                * ((result.x[n1] - result.x[n0]) * (y2 - y0)
                    - (result.x[n2] - result.x[n0]) * (y1 - y0));
            let flux = -diffusivity * dc_dx;
            let flux_err = (flux - exact_flux).abs();
            if flux_err > max_flux_err {
                max_flux_err = flux_err;
            }
        }
    }

    println!(
        "Electrochemistry diffusion layer: max_c_err={:.2e}, max_flux_err={:.2e}",
        max_err, max_flux_err
    );
    assert!(
        max_err < 1e-10,
        "Concentration error too large: {:.2e}",
        max_err
    );
    assert!(
        max_flux_err < 1e-10,
        "Flux error too large: {:.2e}",
        max_flux_err
    );
}

fn test_butler_volmer_polarization_curve() {
    let kinetics = ButlerVolmerKinetics {
        exchange_current_density: 2.0,
        alpha_anodic: 0.5,
        alpha_cathodic: 0.5,
        temperature: 298.15,
    };
    let eta = 0.025;
    let current = kinetics.current_density(eta);
    let reverse = kinetics.current_density(-eta);
    let resistance = kinetics.linearized_charge_transfer_resistance();
    let tab = BasisTabulation {
        n_dofs: 1,
        spatial_dim: 1,
        values: vec![1.0],
        derivatives: vec![0.0],
        hessians: vec![],
        det_j: 1.0,
        physical_coords: [0.0; 3],
    };
    let mut flux = sonium_fem::forms::LocalVector::zeros(1);
    ButlerVolmerBoundaryFluxForm {
        kinetics,
        overpotential: eta,
        stoichiometric_coefficient: 1.0,
    }
    .integrate_point(&tab, 1.0, &mut flux);
    let mut tangent = sonium_fem::forms::LocalMatrix::zeros(1);
    ButlerVolmerTangentForm {
        kinetics,
        overpotential: eta,
    }
    .integrate_point(&tab, &tab, 1.0, &mut tangent);

    println!(
        "Butler-Volmer: i={:.4e}, reverse={:.4e}, Rct={:.4e}, flux={:.4e}, tangent={:.4e}",
        current,
        reverse,
        resistance,
        flux.values[0],
        tangent.get(0, 0)
    );
    assert!(current > 0.0);
    assert!((current + reverse).abs() < 1.0e-12);
    assert!(resistance > 0.0);
    assert!(flux.values[0] > 0.0);
    assert!(tangent.get(0, 0) > 0.0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: electrochemistry diffusion layer ===");
        test_electrochemistry_diffusion_layer();
        test_butler_volmer_polarization_curve();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
