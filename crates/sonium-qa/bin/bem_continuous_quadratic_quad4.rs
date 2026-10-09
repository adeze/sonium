//! Release-backed CPU evidence for globally continuous quadratic Quad4 BEM.
//!
//! The fixture is a closed one-metre cube made from six conforming Quad4
//! panels.  It exercises the production continuous-Q2 TBEM path, including
//! shared vertex/edge numbering and a finite direct solve.  It does not claim
//! higher-order convergence, large-problem performance, or external parity.

use sonium_bem::core::bem_solver::BoundaryConditionType;
use sonium_bem::core::types::{Element, ElementProperty, ElementType, Mesh};
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolver, IncidentField, PhysicsParams, SolverMethod,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "bem.continuous-quadratic-quad4.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    faces: usize,
    global_dofs: usize,
    local_dof_slots: usize,
    shared_dof_slots: usize,
    max_pressure_norm: f64,
}

fn quad(connectivity: Vec<usize>, normal: [f64; 3], center: [f64; 3]) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal,
        node_normals: vec![normal; 4],
        center,
        area: 1.0,
        boundary_condition: sonium_bem::core::types::BoundaryCondition::Velocity(vec![]),
        group: 0,
        dof_addresses: Vec::new(),
    }
}

fn cube_mesh() -> Mesh {
    let a = 0.5;
    Mesh {
        nodes: vec![
            [-a, -a, -a],
            [a, -a, -a],
            [a, a, -a],
            [-a, a, -a],
            [-a, -a, a],
            [a, -a, a],
            [a, a, a],
            [-a, a, a],
        ],
        elements: vec![
            // Outward-oriented bottom, top, -X, +X, -Y, and +Y faces.
            quad(vec![0, 3, 2, 1], [0.0, 0.0, -1.0], [0.0, 0.0, -a]),
            quad(vec![4, 5, 6, 7], [0.0, 0.0, 1.0], [0.0, 0.0, a]),
            quad(vec![0, 4, 7, 3], [-1.0, 0.0, 0.0], [-a, 0.0, 0.0]),
            quad(vec![1, 2, 6, 5], [1.0, 0.0, 0.0], [a, 0.0, 0.0]),
            quad(vec![0, 1, 5, 4], [0.0, -1.0, 0.0], [0.0, -a, 0.0]),
            quad(vec![3, 7, 6, 2], [0.0, 1.0, 0.0], [0.0, a, 0.0]),
        ],
        external_node_numbers: Vec::new(),
        external_element_numbers: Vec::new(),
        num_boundary_nodes: 8,
        num_evaluation_nodes: 0,
        num_boundary_elements: 6,
        num_evaluation_elements: 0,
        symmetry_planes: [0; 3],
        symmetry_coordinates: [0.0; 3],
        num_reflections: 0,
        node_to_elements: Vec::new(),
    }
}

fn run() -> Result<Metrics, String> {
    let mesh = cube_mesh();
    let map = mesh
        .continuous_quadratic_dof_map()
        .map_err(|error| format!("continuous Quad4 topology rejected cube: {error}"))?;

    // Eight vertices + twelve unique edges + six face-interior Q2 DOFs.
    if map.num_dofs() != 26 {
        return Err(format!(
            "unexpected closed-cube continuous Quad4 DOF count: {}",
            map.num_dofs()
        ));
    }

    let problem = BemProblem {
        mesh,
        physics: PhysicsParams::new(100.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };
    problem
        .validate()
        .map_err(|error| format!("closed Quad4 BEM fixture validation failed: {error}"))?;

    let solver = BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .with_tolerance(1.0e-8)
        .with_max_iterations(200);
    solver
        .validate()
        .map_err(|error| format!("continuous Quad4 solver validation failed: {error}"))?;
    let solution = solver
        .solve(&problem)
        .map_err(|error| format!("continuous Quad4 production solve failed: {error}"))?;

    let local_dof_slots = solution
        .elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .map(|element| element.dof_addresses.len())
        .sum::<usize>();
    if local_dof_slots != 54 {
        return Err(format!(
            "unexpected local continuous Quad4 slot count: {local_dof_slots}"
        ));
    }
    let shared_dof_slots = local_dof_slots.saturating_sub(map.num_dofs());
    if shared_dof_slots != 28 {
        return Err(format!(
            "unexpected shared continuous Quad4 slot count: {shared_dof_slots}"
        ));
    }

    let max_pressure_norm = solution
        .surface_pressure
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    if !max_pressure_norm.is_finite()
        || solution
            .surface_pressure
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("continuous Quad4 solution contains non-finite pressure".into());
    }

    Ok(Metrics {
        faces: 6,
        global_dofs: map.num_dofs(),
        local_dof_slots,
        shared_dof_slots,
        max_pressure_norm,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("faces", metrics.faces as f64)
                .metric("global_dofs", metrics.global_dofs as f64)
                .metric("local_dof_slots", metrics.local_dof_slots as f64)
                .metric("shared_dof_slots", metrics.shared_dof_slots as f64)
                .metric("max_pressure_norm", metrics.max_pressure_norm)
                .minimum("faces", 6.0)
                .minimum("global_dofs", 26.0)
                .minimum("shared_dof_slots", 28.0)
                .reference("independent closed-cube Q2 vertex/edge/interior DOF count and finite production solve")
                .dimension("1 m closed cube; six conforming Quad4 surface panels")
                .formulation("CPU TBEM globally continuous quadratic surface unknowns on a closed six-panel Quad4 cube")
                .tolerance("global DOFs = 26; local Q2 slots = 54; shared slots = 28; all pressure coefficients finite")
                .seed("bem-continuous-quadratic-quad4:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent closed-cube Q2 vertex/edge/interior DOF count and finite production solve")
                .dimension("1 m closed cube; six conforming Quad4 surface panels")
                .formulation("CPU TBEM globally continuous quadratic surface unknowns on a closed six-panel Quad4 cube")
                .tolerance("global DOFs = 26; local Q2 slots = 54; shared slots = 28; all pressure coefficients finite")
                .seed("bem-continuous-quadratic-quad4:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            eprintln!("BEM CONTINUOUS QUADRATIC QUAD4 FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
