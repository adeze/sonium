use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::BilinearForm;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::solid::elasticity::{IsotropicMaterial, PlaneAssumption};
use sonium_fem::physics::solid::fracture::{
    DamageHistoryRhsForm, DegradedElasticityForm, PhaseFieldDamageForm, PhaseFieldFractureMaterial,
    damage_history_tangent,
};
use sonium_fem::spaces::{BasisTabulation, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn exact_damage_1d(x: f64, l: f64, gc: f64, ell: f64, history: f64) -> f64 {
    let ratio = l / ell;
    let d_inf = 2.0 * history * ell / gc;
    if ratio > 50.0 {
        return d_inf * (1.0 - (-(l - x.abs()) / ell).exp());
    }
    d_inf * (1.0 - (x.abs() / ell).cosh() / ratio.cosh())
}

fn test_fracture_damage_profile() {
    let l = 1.0;
    let h = 0.1;
    let nx = 160;
    let ny = 2;
    let gc = 2.0;
    let ell = 0.05;
    let history = 1.0;

    let mesh = rectangular_mesh_triangles(-l, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);

    let material = PhaseFieldFractureMaterial {
        critical_energy_release_rate: gc,
        length_scale: ell,
        residual_stiffness: 0.0,
    };

    let damage_form = PhaseFieldDamageForm { material };
    let history_form = DamageHistoryRhsForm {
        history_energy_density: history,
    };

    let matrix = assemble_bilinear(&damage_form, &space, &space, &mesh, 2);
    let rhs = assemble_linear(&history_form, &space, &mesh, 2);

    let mut system = FemSystem::new(matrix, rhs);
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 0.0, None);

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
    let mut max_val: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let exact = exact_damage_1d(node.x, l, gc, ell, history);
        let err = (result.x[i] - exact).abs();
        max_err = max_err.max(err);
        max_val = max_val.max(exact);
    }

    println!(
        "Fracture damage profile: max_err={:.4e}, max_val={:.4e}, rel_err={:.4e}",
        max_err,
        max_val,
        max_err / max_val.max(1e-12)
    );
    assert!(
        max_err / max_val.max(1e-12) < 0.02,
        "Damage profile error too large: {:.4e}",
        max_err / max_val
    );
}

fn test_degraded_elasticity_and_history_tangent() {
    let material = PhaseFieldFractureMaterial {
        critical_energy_release_rate: 2.0,
        length_scale: 0.05,
        residual_stiffness: 1.0e-6,
    };
    let tab = BasisTabulation {
        n_dofs: 4,
        spatial_dim: 2,
        values: vec![0.0; 4],
        derivatives: vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        hessians: vec![],
        det_j: 1.0,
        physical_coords: [0.0; 3],
    };
    let elastic = IsotropicMaterial::new(10.0, 0.25);
    let mut k_intact = sonium_fem::forms::LocalMatrix::zeros(4);
    DegradedElasticityForm {
        elastic_material: elastic,
        fracture_material: material,
        damage: 0.0,
        plane: Some(PlaneAssumption::PlaneStress),
    }
    .integrate_point(&tab, &tab, 1.0, &mut k_intact);
    let mut k_damaged = sonium_fem::forms::LocalMatrix::zeros(4);
    DegradedElasticityForm {
        elastic_material: elastic,
        fracture_material: material,
        damage: 0.75,
        plane: Some(PlaneAssumption::PlaneStress),
    }
    .integrate_point(&tab, &tab, 1.0, &mut k_damaged);
    let tangent = damage_history_tangent(material, 1.0);

    println!(
        "Fracture coupling: intact={:.4e}, damaged={:.4e}, tangent={:.4e}",
        k_intact.get(0, 0),
        k_damaged.get(0, 0),
        tangent
    );
    assert!(k_damaged.get(0, 0) < k_intact.get(0, 0));
    assert!(tangent > material.critical_energy_release_rate / material.length_scale);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: fracture damage profile ===");
        test_fracture_damage_profile();
        test_degraded_elasticity_and_history_tangent();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
