//! Independent CPU reference for the production RT0/H(div) assembly path.
//!
//! The mass reference integrates the affine RT0 polynomials from exact
//! reference-triangle moments and applies the Piola metric independently of
//! FEM quadrature. The divergence reference uses only the unit-flux identity
//! `div(N_i) = sign_i / |K|`. A shared-edge fixture with reversed winding
//! checks that topology orientation is retained by both assembled operators.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked, assemble_mixed_checked};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{DivergenceDivergenceForm, HdivDivergenceForm, VectorMassForm};
use sonium_fem::mesh::{ElementType, Mesh, MeshTopology, Point};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt0-reference.v1";
const MASS_TOLERANCE: f64 = 2.0e-11;
const DIVERGENCE_TOLERANCE: f64 = 2.0e-10;
const CONSERVATION_TOLERANCE: f64 = 2.0e-11;

// Exact moments of [1, xi, eta] on the reference triangle.
const AFFINE_MOMENTS: [[f64; 3]; 3] = [
    [1.0 / 12.0, 1.0 / 24.0, 1.0 / 6.0],
    [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0],
];

// Coefficients in (xi, eta, 1) for the edge order (v0,v1), (v1,v2),
// (v2,v0): [xi, eta-1], [xi, eta], [xi-1, eta].
const RT0_TRIANGLE_COEFFICIENTS: [[[f64; 3]; 2]; 3] = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
];

fn aggregate(matrix: &TripletMatrix<f64>) -> BTreeMap<(usize, usize), f64> {
    let mut entries = BTreeMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *entries.entry((row, column)).or_insert(0.0) += value;
    }
    entries
}

fn affine_component_inner(lhs: &[f64; 3], rhs: &[f64; 3]) -> f64 {
    lhs.iter()
        .enumerate()
        .map(|(row, &lhs_value)| {
            rhs.iter()
                .enumerate()
                .map(|(column, &rhs_value)| lhs_value * rhs_value * AFFINE_MOMENTS[row][column])
                .sum::<f64>()
        })
        .sum()
}

fn triangle_geometry(mesh: &Mesh, element: usize) -> ([[f64; 2]; 2], f64) {
    let vertices = mesh.elements[element].vertices();
    let p0 = &mesh.nodes[vertices[0]];
    let p1 = &mesh.nodes[vertices[1]];
    let p2 = &mesh.nodes[vertices[2]];
    let jacobian = [[p1.x - p0.x, p2.x - p0.x], [p1.y - p0.y, p2.y - p0.y]];
    let determinant = (jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]).abs();
    let metric = [
        [
            jacobian[0][0] * jacobian[0][0] + jacobian[1][0] * jacobian[1][0],
            jacobian[0][0] * jacobian[0][1] + jacobian[1][0] * jacobian[1][1],
        ],
        [
            jacobian[0][1] * jacobian[0][0] + jacobian[1][1] * jacobian[1][0],
            jacobian[0][1] * jacobian[0][1] + jacobian[1][1] * jacobian[1][1],
        ],
    ];
    (metric, determinant)
}

fn reference_local_mass(mesh: &Mesh, element: usize) -> [[f64; 3]; 3] {
    let (metric, determinant) = triangle_geometry(mesh, element);
    let topology = MeshTopology::build(mesh);
    let signs = topology.element_edge_signs(element);
    let mut local = [[0.0; 3]; 3];

    for (row, row_coefficients) in RT0_TRIANGLE_COEFFICIENTS.iter().enumerate() {
        for (column, column_coefficients) in RT0_TRIANGLE_COEFFICIENTS.iter().enumerate() {
            let mut value = 0.0;
            for row_component in 0..2 {
                for column_component in 0..2 {
                    value += metric[row_component][column_component]
                        * affine_component_inner(
                            &row_coefficients[row_component],
                            &column_coefficients[column_component],
                        );
                }
            }
            local[row][column] = signs[row] as f64 * signs[column] as f64 * value / determinant;
        }
    }
    local
}

fn reference_global_mass(mesh: &Mesh) -> BTreeMap<(usize, usize), f64> {
    let space = HdivSpace::new();
    let mut reference = BTreeMap::new();
    for element in 0..mesh.num_elements() {
        let local = reference_local_mass(mesh, element);
        let dofs = space.local_to_global(mesh, element);
        for (row, &global_row) in dofs.iter().enumerate() {
            for (column, &global_column) in dofs.iter().enumerate() {
                *reference.entry((global_row, global_column)).or_insert(0.0) += local[row][column];
            }
        }
    }
    reference
}

fn reference_global_divergence(mesh: &Mesh) -> BTreeMap<(usize, usize), f64> {
    let space = HdivSpace::new();
    let topology = MeshTopology::build(mesh);
    let mut reference = BTreeMap::new();
    for element in 0..mesh.num_elements() {
        let (_, determinant) = triangle_geometry(mesh, element);
        let area = determinant / 2.0;
        let signs = topology.element_edge_signs(element);
        let dofs = space.local_to_global(mesh, element);
        for (row, &global_row) in dofs.iter().enumerate() {
            for (column, &global_column) in dofs.iter().enumerate() {
                let value = signs[row] as f64 * signs[column] as f64 / area;
                *reference.entry((global_row, global_column)).or_insert(0.0) += value;
            }
        }
    }
    reference
}

fn max_map_error(
    actual: &BTreeMap<(usize, usize), f64>,
    expected: &BTreeMap<(usize, usize), f64>,
) -> f64 {
    actual
        .keys()
        .chain(expected.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|key| {
            (actual.get(key).copied().unwrap_or(0.0) - expected.get(key).copied().unwrap_or(0.0))
                .abs()
        })
        .fold(0.0, f64::max)
}

fn canonical_facet_matrix(
    mesh: &Mesh,
    matrix: &BTreeMap<(usize, usize), f64>,
) -> BTreeMap<(Vec<usize>, Vec<usize>), f64> {
    let topology = MeshTopology::build(mesh);
    let facet_key = |dof: usize| {
        let (a, b) = topology.edges[dof];
        if a < b { vec![a, b] } else { vec![b, a] }
    };
    let mut canonical = BTreeMap::new();
    for (&(row, column), &value) in matrix {
        *canonical
            .entry((facet_key(row), facet_key(column)))
            .or_insert(0.0) += value;
    }
    canonical
}

fn max_canonical_error(
    actual: &BTreeMap<(Vec<usize>, Vec<usize>), f64>,
    expected: &BTreeMap<(Vec<usize>, Vec<usize>), f64>,
) -> f64 {
    actual
        .keys()
        .chain(expected.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|key| {
            (actual.get(key).copied().unwrap_or(0.0) - expected.get(key).copied().unwrap_or(0.0))
                .abs()
        })
        .fold(0.0, f64::max)
}

fn shared_triangle_mesh_with_reversed_winding() -> Mesh {
    let mut mesh = Mesh::new(2);
    for (x, y) in [(0.15, -0.2), (1.4, 0.25), (0.35, 1.3), (1.8, 1.7)] {
        mesh.add_node(Point { x, y, z: 0.0 });
    }
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh.add_element(ElementType::Triangle, vec![1, 3, 2]);
    mesh.elements[1].nodes.reverse();
    mesh.invalidate_caches();
    mesh
}

fn assembled_mass(mesh: &Mesh) -> Result<BTreeMap<(usize, usize), f64>, String> {
    let space = HdivSpace::new();
    space.validate(mesh).map_err(|error| error.to_string())?;
    let assembled = assemble_bilinear_checked(
        &VectorMassForm { coefficient: 1.0 },
        &space,
        &space,
        mesh,
        7,
    )
    .map_err(|error| error.to_string())?;
    Ok(aggregate(&assembled))
}

fn assembled_divergence(mesh: &Mesh) -> Result<BTreeMap<(usize, usize), f64>, String> {
    let space = HdivSpace::new();
    let assembled = assemble_bilinear_checked(&DivergenceDivergenceForm, &space, &space, mesh, 4)
        .map_err(|error| error.to_string())?;
    Ok(aggregate(&assembled))
}

fn mixed_conservation_error(mesh: &Mesh) -> Result<f64, String> {
    let flux = HdivSpace::new();
    let pressure = H1Space::new(PolynomialDegree::P1);
    let system = assemble_mixed_checked(
        &VectorMassForm { coefficient: 1.0 },
        &HdivDivergenceForm,
        &flux,
        &pressure,
        mesh,
        4,
    )
    .map_err(|error| error.to_string())?;

    let mut expected = vec![0.0; flux.num_global_dofs(mesh)];
    for element in 0..mesh.num_elements() {
        let dofs = flux.local_to_global(mesh, element);
        let quadrature = QuadratureRule::new(mesh.elements[element].element_type, 4);
        for point in quadrature.iter() {
            let tabulation = flux.tabulate(mesh, element, point.xi(), point.eta(), point.zeta());
            for (local, &global) in dofs.iter().enumerate() {
                expected[global] -= point.weight * tabulation.det_j * tabulation.divergence(local);
            }
        }
    }

    let actual = aggregate(&system.block_01);
    let mut row_sums = vec![0.0; expected.len()];
    for (&(row, _), &value) in &actual {
        row_sums[row] += value;
    }
    Ok(row_sums
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max))
}

fn permuted_mesh(mut mesh: Mesh) -> Mesh {
    for element in &mut mesh.elements {
        if element.element_type == ElementType::Triangle {
            element.nodes.rotate_left(1);
        }
    }
    mesh.invalidate_caches();
    mesh
}

fn run() -> Result<(f64, f64, f64, f64, usize, u64), String> {
    let started = Instant::now();
    let mesh = shared_triangle_mesh_with_reversed_winding();
    let mass = assembled_mass(&mesh)?;
    let mass_reference = reference_global_mass(&mesh);
    let mass_error = max_map_error(&mass, &mass_reference);

    let divergence = assembled_divergence(&mesh)?;
    let divergence_reference = reference_global_divergence(&mesh);
    let divergence_error = max_map_error(&divergence, &divergence_reference);
    let conservation_error = mixed_conservation_error(&mesh)?;

    let permuted = permuted_mesh(mesh.clone());
    let permutation_error = max_canonical_error(
        &canonical_facet_matrix(&mesh, &divergence),
        &canonical_facet_matrix(&permuted, &assembled_divergence(&permuted)?),
    );

    if mass_error > MASS_TOLERANCE {
        return Err(format!("RT0 mass reference error {mass_error:.3e}"));
    }
    if divergence_error > DIVERGENCE_TOLERANCE {
        return Err(format!(
            "RT0 divergence reference error {divergence_error:.3e}"
        ));
    }
    if conservation_error > CONSERVATION_TOLERANCE {
        return Err(format!(
            "RT0 mixed conservation error {conservation_error:.3e}"
        ));
    }
    if permutation_error > DIVERGENCE_TOLERANCE {
        return Err(format!(
            "RT0 orientation permutation error {permutation_error:.3e}"
        ));
    }

    Ok((
        mass_error,
        divergence_error,
        conservation_error,
        permutation_error,
        mass.len(),
        started.elapsed().as_millis() as u64,
    ))
}

fn main() {
    match run() {
        Ok((
            mass_error,
            divergence_error,
            conservation_error,
            permutation_error,
            entries,
            elapsed_ms,
        )) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_mass_reference_error", mass_error)
                .metric("max_divergence_reference_error", divergence_error)
                .metric("max_mixed_conservation_error", conservation_error)
                .metric("max_orientation_permutation_error", permutation_error)
                .metric("assembled_mass_entries", entries as f64)
                .maximum("max_mass_reference_error", MASS_TOLERANCE)
                .maximum("max_divergence_reference_error", DIVERGENCE_TOLERANCE)
                .maximum("max_mixed_conservation_error", CONSERVATION_TOLERANCE)
                .maximum("max_orientation_permutation_error", DIVERGENCE_TOLERANCE)
                .reference("independent affine RT0 moments, Piola metric, and unit-flux divergence identity")
                .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt0-reference-v1")
                .input_checksum("mesh:shared-triangle-reversed-winding:v1")
                .dimension("2D triangular H(div)/RT0 mass, divergence, and P1 mixed coupling")
                .formulation("shared-edge topology signs, exact affine moments, and signed flux conservation")
                .tolerance("mass <= 2e-11; divergence/orientation <= 2e-10; mixed conservation <= 2e-11")
                .seed("fem-hdiv-rt0-reference:shared-triangle-v1")
                .timing_ms("hdiv_rt0_reference", elapsed_ms)
                .elapsed_ms(elapsed_ms);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("FEM H(div)/RT0 REFERENCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
