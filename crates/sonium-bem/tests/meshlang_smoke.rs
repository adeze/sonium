#![cfg(feature = "meshlang")]

use num_complex::Complex64;
use sonium_bem::core::bem_solver::BoundaryConditionType;
use sonium_bem::core::{
    BemProblem, BemSolver, BoundaryCondition, Element, ElementProperty, ElementType, IncidentField,
    Mesh, PhysicsParams, SolverMethod,
};
use sonium_mesher::mesh::mesh_from_mlg_str;

#[test]
fn meshlang_geometry_reaches_a_bem_scattering_solve_when_gmsh_is_configured() {
    if std::env::var_os("GMSH_PATH").is_none() && std::env::var_os("MESHLANG_GMSH").is_none() {
        assert_ne!(
            std::env::var("MESHLANG_REQUIRE_GMSH").ok().as_deref(),
            Some("1"),
            "Gmsh is required for this CI execution lane"
        );
        return;
    }

    let source = r#"
        geometry scatterer {
            let body = Box(0.2, 0.2, 0.2) |> translate(-0.1, -0.1, -0.1)
        }
        mesh for scatterer {
            min_size: 0.08
            max_size: 0.12
            order: 1
        }
        physics {
            group solid: volume = scatterer.body
            group boundary: neumann = scatterer.body.faces
        }
        export { format: msh4 order: 1 filename: "bem-smoke.msh" }
    "#;

    let volume_mesh = mesh_from_mlg_str(source, "bem-smoke.mlg")
        .expect("MeshLang -> Gmsh -> BEM source mesh pipeline failed");
    let nodes: Vec<[f64; 3]> = volume_mesh
        .nodes
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect();
    let mut elements = Vec::with_capacity(volume_mesh.boundaries.len());
    for boundary in &volume_mesh.boundaries {
        assert_eq!(boundary.nodes.len(), 3, "BEM smoke mesh must be triangular");
        let connectivity = boundary.nodes.clone();
        let a = nodes[connectivity[0]];
        let b = nodes[connectivity[1]];
        let c = nodes[connectivity[2]];
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let cross_norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        assert!(cross_norm > 0.0, "degenerate boundary triangle");
        let normal = [
            cross[0] / cross_norm,
            cross[1] / cross_norm,
            cross[2] / cross_norm,
        ];
        let center = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        let dof = elements.len();
        elements.push(Element {
            connectivity,
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal,
            node_normals: vec![normal; 3],
            center,
            area: 0.5 * cross_norm,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: boundary.marker.max(0) as usize,
            dof_addresses: vec![dof],
        });
    }
    assert!(!elements.is_empty());

    let node_count = nodes.len();
    let element_count = elements.len();
    let mut mesh = Mesh {
        nodes,
        elements,
        external_node_numbers: (1..=node_count as i32).collect(),
        external_element_numbers: (1..=element_count as i32).collect(),
        num_boundary_nodes: node_count,
        num_evaluation_nodes: 0,
        num_boundary_elements: element_count,
        num_evaluation_elements: 0,
        symmetry_planes: [0, 0, 0],
        symmetry_coordinates: [0.0, 0.0, 0.0],
        num_reflections: 0,
        node_to_elements: Vec::new(),
    };
    mesh.build_adjacency();

    let problem = BemProblem {
        mesh,
        physics: PhysicsParams::new(50.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: false,
    };
    let solution = BemSolver::new()
        .with_solver_method(SolverMethod::Direct)
        .solve(&problem)
        .expect("BEM scattering smoke solve failed");
    assert_eq!(solution.surface_pressure.len(), problem.mesh.num_dofs());
    assert!(
        solution
            .surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}
