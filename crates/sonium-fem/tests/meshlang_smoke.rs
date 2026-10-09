#![cfg(feature = "meshlang")]

use num_complex::Complex64;
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::mesh_from_mlg_str;
use sonium_fem::solver::{SolverConfig, SolverType, solve};

#[test]
fn meshlang_geometry_reaches_a_fem_helmholtz_solve_when_gmsh_is_configured() {
    if std::env::var_os("GMSH_PATH").is_none() && std::env::var_os("MESHLANG_GMSH").is_none() {
        assert_ne!(
            std::env::var("MESHLANG_REQUIRE_GMSH").ok().as_deref(),
            Some("1"),
            "Gmsh is required for this CI execution lane"
        );
        return;
    }

    let source = r#"
        geometry acoustic_domain {
            let fluid = Box(1.0, 1.0, 1.0)
        }
        mesh for acoustic_domain {
            min_size: 0.3
            max_size: 0.6
            order: 1
        }
        physics {
            group fluid: volume = acoustic_domain.fluid
            group walls: neumann = acoustic_domain.fluid.faces
        }
        export { format: msh4 order: 1 filename: "fem-smoke.msh" }
    "#;

    let mesh = mesh_from_mlg_str(source, "fem-smoke.mlg")
        .expect("MeshLang -> Gmsh -> FEM mesh pipeline failed");
    assert_eq!(mesh.dimension, 3);
    assert!(mesh.physical_groups.by_name(3, "fluid").is_some());
    assert!(mesh.physical_groups.by_name(2, "walls").is_some());

    let problem = HelmholtzProblem::assemble(
        &mesh,
        PolynomialDegree::P1,
        Complex64::new(0.7, 0.05),
        |_, _, _| Complex64::new(1.0, 0.0),
    );
    assert_eq!(problem.num_dofs(), mesh.num_nodes());
    assert!(problem.matrix.nnz() > 0);

    let solution = solve(
        &problem,
        &SolverConfig {
            solver_type: SolverType::Direct,
            ..SolverConfig::default()
        },
    )
    .expect("FEM Helmholtz smoke solve failed");
    assert!(solution.converged);
    assert_eq!(solution.values.len(), mesh.num_nodes());
    assert!(
        solution
            .values
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}
