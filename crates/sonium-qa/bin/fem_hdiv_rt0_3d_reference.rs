//! Independent CPU reference for the three-dimensional RT0/H(div) path.
//!
//! The reference integrates the affine tetrahedral and tensor-product
//! hexahedral RT0 polynomials from exact reference-cell moments. It applies
//! the contravariant Piola metric and topology face signs independently of the
//! production basis and quadrature, then checks the production mixed coupling
//! and invariance under a tetrahedral local-node permutation.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked, assemble_mixed_checked};
use sonium_fem::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use sonium_fem::forms::{DivergenceDivergenceForm, HdivDivergenceForm, VectorMassForm};
use sonium_fem::mesh::{Face, Mesh, MeshTopology, unit_cube_hexahedra, unit_cube_tetrahedra};
use sonium_fem::spaces::{FunctionSpace, H1Space, HdivSpace};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hdiv-rt0-3d-reference.v1";
const MASS_TOLERANCE: f64 = 5.0e-11;
const DIVERGENCE_TOLERANCE: f64 = 5.0e-10;
const CONSERVATION_TOLERANCE: f64 = 5.0e-11;

// Moments of [xi, eta, zeta, 1] over the reference tetrahedron
// {xi,eta,zeta >= 0, xi+eta+zeta <= 1}.
const TETRA_MOMENTS: [[f64; 4]; 4] = [
    [1.0 / 60.0, 1.0 / 120.0, 1.0 / 120.0, 1.0 / 24.0],
    [1.0 / 120.0, 1.0 / 60.0, 1.0 / 120.0, 1.0 / 24.0],
    [1.0 / 120.0, 1.0 / 120.0, 1.0 / 60.0, 1.0 / 24.0],
    [1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 6.0],
];

// Moments of [xi, eta, zeta, 1] over [-1,1]^3.
const HEXA_MOMENTS: [[f64; 4]; 4] = [
    [8.0 / 3.0, 0.0, 0.0, 0.0],
    [0.0, 8.0 / 3.0, 0.0, 0.0],
    [0.0, 0.0, 8.0 / 3.0, 0.0],
    [0.0, 0.0, 0.0, 8.0],
];

const TETRA_BASIS: [[[f64; 4]; 3]; 4] = [
    [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, -2.0],
    ],
    [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, -2.0],
        [0.0, 0.0, 2.0, 0.0],
    ],
    [
        [2.0, 0.0, 0.0, -2.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
    ],
    [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
    ],
];

const HEXA_BASIS: [[[f64; 4]; 3]; 6] = [
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.125, -0.125],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.125, 0.125],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.125, 0.0, -0.125],
        [0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.125, 0.0, 0.125],
        [0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.125, 0.0, 0.0, -0.125],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.125, 0.0, 0.0, 0.125],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
];

fn aggregate(matrix: &TripletMatrix<f64>) -> BTreeMap<(usize, usize), f64> {
    let mut entries = BTreeMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *entries.entry((row, column)).or_insert(0.0) += value;
    }
    entries
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

fn metric_inner(lhs: &[f64; 4], rhs: &[f64; 4], moments: &[[f64; 4]; 4]) -> f64 {
    lhs.iter()
        .enumerate()
        .map(|(row, &lhs_value)| {
            rhs.iter()
                .enumerate()
                .map(|(column, &rhs_value)| lhs_value * rhs_value * moments[row][column])
                .sum::<f64>()
        })
        .sum()
}

fn metric(mesh: &Mesh, element: usize) -> ([[f64; 3]; 3], f64) {
    let elem = &mesh.elements[element];
    let shape = evaluate_shape(elem.element_type, PolynomialDegree::P1, 0.0, 0.0, 0.0);
    let vertices = elem.vertices();
    let coords: Vec<[f64; 3]> = vertices
        .iter()
        .map(|&node| {
            let point = mesh.nodes[node];
            [point.x, point.y, point.z]
        })
        .collect();
    let jacobian = Jacobian::from_3d(&shape.gradients, shape.grad_dim, &coords);
    let mut gram = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            gram[row][column] = (0..3)
                .map(|physical| jacobian.matrix[physical][row] * jacobian.matrix[physical][column])
                .sum();
        }
    }
    (gram, jacobian.det.abs())
}

fn reference_local_mass(
    mesh: &Mesh,
    element: usize,
    basis: &[[[f64; 4]; 3]],
    moments: &[[f64; 4]; 4],
) -> Vec<Vec<f64>> {
    let (gram, determinant) = metric(mesh, element);
    let topology = MeshTopology::build(mesh);
    let signs = topology.element_face_signs(element);
    let mut local = vec![vec![0.0; basis.len()]; basis.len()];
    for (row, row_basis) in basis.iter().enumerate() {
        for (column, column_basis) in basis.iter().enumerate() {
            let mut value = 0.0;
            for component_row in 0..3 {
                for component_column in 0..3 {
                    value += gram[component_row][component_column]
                        * metric_inner(
                            &row_basis[component_row],
                            &column_basis[component_column],
                            moments,
                        );
                }
            }
            local[row][column] = signs[row] as f64 * signs[column] as f64 * value / determinant;
        }
    }
    local
}

fn reference_global_mass(
    mesh: &Mesh,
    basis: &[[[f64; 4]; 3]],
    moments: &[[f64; 4]; 4],
) -> BTreeMap<(usize, usize), f64> {
    let space = HdivSpace::new();
    let topology = MeshTopology::build(mesh);
    let mut reference = BTreeMap::new();
    for element in 0..mesh.num_elements() {
        let local = reference_local_mass(mesh, element, basis, moments);
        let dofs = space.local_to_global(mesh, element);
        assert_eq!(dofs.len(), basis.len());
        for (row, &global_row) in dofs.iter().enumerate() {
            for (column, &global_column) in dofs.iter().enumerate() {
                *reference.entry((global_row, global_column)).or_insert(0.0) += local[row][column];
            }
        }
    }
    assert_eq!(topology.num_faces(), space.num_global_dofs(mesh));
    reference
}

fn reference_global_divergence(
    mesh: &Mesh,
    local_divergence: f64,
) -> BTreeMap<(usize, usize), f64> {
    let space = HdivSpace::new();
    let topology = MeshTopology::build(mesh);
    let mut reference = BTreeMap::new();
    for element in 0..mesh.num_elements() {
        let (_, determinant) = metric(mesh, element);
        let signs = topology.element_face_signs(element);
        let dofs = space.local_to_global(mesh, element);
        for (row, &global_row) in dofs.iter().enumerate() {
            for (column, &global_column) in dofs.iter().enumerate() {
                let value =
                    signs[row] as f64 * signs[column] as f64 * local_divergence / determinant;
                *reference.entry((global_row, global_column)).or_insert(0.0) += value;
            }
        }
    }
    reference
}

fn assembled_mass(mesh: &Mesh) -> Result<BTreeMap<(usize, usize), f64>, String> {
    let space = HdivSpace::new();
    space.validate(mesh).map_err(|error| error.to_string())?;
    let matrix = assemble_bilinear_checked(
        &VectorMassForm { coefficient: 1.0 },
        &space,
        &space,
        mesh,
        7,
    )
    .map_err(|error| error.to_string())?;
    Ok(aggregate(&matrix))
}

fn assembled_divergence(mesh: &Mesh) -> Result<BTreeMap<(usize, usize), f64>, String> {
    let space = HdivSpace::new();
    let matrix = assemble_bilinear_checked(&DivergenceDivergenceForm, &space, &space, mesh, 5)
        .map_err(|error| error.to_string())?;
    Ok(aggregate(&matrix))
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
        5,
    )
    .map_err(|error| error.to_string())?;

    let topology = MeshTopology::build(mesh);
    let mut expected = vec![0.0; flux.num_global_dofs(mesh)];
    for element in 0..mesh.num_elements() {
        let dofs = flux.local_to_global(mesh, element);
        let signs = topology.element_face_signs(element);
        for (local, &global) in dofs.iter().enumerate() {
            expected[global] -= signs[local] as f64;
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

fn canonical_faces(
    mesh: &Mesh,
    matrix: &BTreeMap<(usize, usize), f64>,
) -> BTreeMap<(Vec<usize>, Vec<usize>), f64> {
    let topology = MeshTopology::build(mesh);
    let key = |index: usize| {
        let mut nodes = match topology.face(index) {
            Face::Triangle(nodes) => nodes.to_vec(),
            Face::Quad(nodes) => nodes.to_vec(),
        };
        nodes.sort_unstable();
        nodes
    };
    let mut canonical = BTreeMap::new();
    for (&(row, column), &value) in matrix {
        *canonical.entry((key(row), key(column))).or_insert(0.0) += value;
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

fn permuted_tetra_mesh(mut mesh: Mesh) -> Mesh {
    for element in &mut mesh.elements {
        element.nodes.rotate_left(1);
    }
    // The generator stores boundary records in the original local ordering;
    // this fixture exercises interior/global face numbering, so discard those
    // records rather than retaining stale ownership metadata.
    mesh.boundaries.clear();
    mesh.invalidate_caches();
    mesh
}

fn run() -> Result<(f64, f64, f64, f64, usize, u64), String> {
    let started = Instant::now();
    let tetra = unit_cube_tetrahedra(1);
    let hexa = unit_cube_hexahedra(1);

    let tetra_mass = assembled_mass(&tetra)?;
    let hexa_mass = assembled_mass(&hexa)?;
    let mass_error = max_map_error(
        &tetra_mass,
        &reference_global_mass(&tetra, &TETRA_BASIS, &TETRA_MOMENTS),
    )
    .max(max_map_error(
        &hexa_mass,
        &reference_global_mass(&hexa, &HEXA_BASIS, &HEXA_MOMENTS),
    ));

    let tetra_divergence = assembled_divergence(&tetra)?;
    let hexa_divergence = assembled_divergence(&hexa)?;
    let divergence_error =
        max_map_error(&tetra_divergence, &reference_global_divergence(&tetra, 6.0)).max(
            max_map_error(
                &hexa_divergence,
                &reference_global_divergence(&hexa, 1.0 / 8.0),
            ),
        );

    let conservation_error =
        mixed_conservation_error(&tetra)?.max(mixed_conservation_error(&hexa)?);
    let permuted = permuted_tetra_mesh(tetra.clone());
    let permutation_error = max_canonical_error(
        &canonical_faces(&tetra, &tetra_divergence),
        &canonical_faces(&permuted, &assembled_divergence(&permuted)?),
    );

    if mass_error > MASS_TOLERANCE {
        return Err(format!("3-D RT0 mass reference error {mass_error:.3e}"));
    }
    if divergence_error > DIVERGENCE_TOLERANCE {
        return Err(format!(
            "3-D RT0 divergence reference error {divergence_error:.3e}"
        ));
    }
    if conservation_error > CONSERVATION_TOLERANCE {
        return Err(format!(
            "3-D RT0 mixed conservation error {conservation_error:.3e}"
        ));
    }
    if permutation_error > DIVERGENCE_TOLERANCE {
        return Err(format!(
            "3-D RT0 face permutation error {permutation_error:.3e}"
        ));
    }

    Ok((
        mass_error,
        divergence_error,
        conservation_error,
        permutation_error,
        tetra_mass.len() + hexa_mass.len(),
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
                .metric("max_face_permutation_error", permutation_error)
                .metric("assembled_mass_entries", entries as f64)
                .maximum("max_mass_reference_error", MASS_TOLERANCE)
                .maximum("max_divergence_reference_error", DIVERGENCE_TOLERANCE)
                .maximum("max_mixed_conservation_error", CONSERVATION_TOLERANCE)
                .maximum("max_face_permutation_error", DIVERGENCE_TOLERANCE)
                .reference("independent tetrahedron/simplex and tensor-product hexahedron RT0 moment integrals")
                .reference_checksum("analytic/deterministic-fixture:fem-hdiv-rt0-3d-reference-v1")
                .input_checksum("mesh:unit-cube-tetra-and-hexa-v1")
                .dimension("3D tetrahedral and hexahedral H(div)/RT0 mass, divergence, and P1 mixed coupling")
                .formulation("exact affine reference-cell moments, Piola metric, global face signs, and mixed conservation")
                .tolerance("mass <= 5e-11; divergence/permutation <= 5e-10; mixed conservation <= 5e-11")
                .seed("fem-hdiv-rt0-3d-reference:unit-cube-v1")
                .timing_ms("hdiv_rt0_3d_reference", elapsed_ms)
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
            eprintln!("FEM 3-D H(div)/RT0 REFERENCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
