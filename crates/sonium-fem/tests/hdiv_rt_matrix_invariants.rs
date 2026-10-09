//! Production-assembly invariants for the CPU RT0 H(div) family.
//!
//! The reference is algebraic: every RT0 basis function has unit signed
//! facet flux, so its divergence is `sign / |K|`.  The assembled
//! divergence-divergence matrix must therefore be the signed element-volume
//! sum below, independently of quadrature-point tabulation.

use std::collections::BTreeMap;

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked, assemble_mixed_checked};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{DivergenceDivergenceForm, HdivDivergenceForm, VectorMassForm};
use sonium_fem::mesh::{ElementType, Mesh, MeshTopology};
use sonium_fem::spaces::{FunctionSpace, H1Space, HdivSpace, SpaceValidationError};

fn aggregate(matrix: &TripletMatrix<f64>) -> BTreeMap<(usize, usize), f64> {
    let mut entries = BTreeMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *entries.entry((row, column)).or_insert(0.0) += value;
    }
    entries
}

fn triangle_area(mesh: &Mesh, element: usize) -> f64 {
    let vertices = mesh.elements[element].vertices();
    let a = &mesh.nodes[vertices[0]];
    let b = &mesh.nodes[vertices[1]];
    let c = &mesh.nodes[vertices[2]];
    0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs()
}

fn tetrahedron_volume(mesh: &Mesh, element: usize) -> f64 {
    let vertices = mesh.elements[element].vertices();
    let a = &mesh.nodes[vertices[0]];
    let b = &mesh.nodes[vertices[1]];
    let c = &mesh.nodes[vertices[2]];
    let d = &mesh.nodes[vertices[3]];
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let ad = [d.x - a.x, d.y - a.y, d.z - a.z];
    let determinant = ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
        - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]);
    determinant.abs() / 6.0
}

fn element_measure(mesh: &Mesh, element: usize) -> f64 {
    match mesh.elements[element].element_type {
        ElementType::Triangle => triangle_area(mesh, element),
        ElementType::Tetrahedron => tetrahedron_volume(mesh, element),
        element_type => panic!("RT0 reference does not cover {element_type:?}"),
    }
}

fn reference_divergence_matrix(mesh: &Mesh, space: &HdivSpace) -> BTreeMap<(usize, usize), f64> {
    let topology = MeshTopology::build(mesh);
    let mut reference = BTreeMap::new();
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(mesh, element);
        let signs = if mesh.elements[element].element_type.dimension() == 2 {
            topology.element_edge_signs(element)
        } else {
            topology.element_face_signs(element)
        };
        let inverse_measure = 1.0 / element_measure(mesh, element);
        for (local_row, &row) in dofs.iter().enumerate() {
            for (local_column, &column) in dofs.iter().enumerate() {
                let value = signs[local_row] as f64 * signs[local_column] as f64 * inverse_measure;
                *reference.entry((row, column)).or_insert(0.0) += value;
            }
        }
    }
    reference
}

fn max_map_residual(
    actual: &BTreeMap<(usize, usize), f64>,
    expected: &BTreeMap<(usize, usize), f64>,
) -> f64 {
    actual
        .keys()
        .chain(expected.keys())
        .collect::<std::collections::BTreeSet<_>>()
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
        let mut nodes = if mesh.dimension == 2 {
            let (a, b) = topology.edges[dof];
            vec![a, b]
        } else {
            topology.faces[dof].nodes()
        };
        nodes.sort_unstable();
        nodes
    };

    let mut canonical = BTreeMap::new();
    for (&(row, column), &value) in matrix {
        *canonical
            .entry((facet_key(row), facet_key(column)))
            .or_insert(0.0) += value;
    }
    canonical
}

fn max_canonical_residual(
    actual: &BTreeMap<(Vec<usize>, Vec<usize>), f64>,
    expected: &BTreeMap<(Vec<usize>, Vec<usize>), f64>,
) -> f64 {
    actual
        .keys()
        .chain(expected.keys())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|key| {
            (actual.get(key).copied().unwrap_or(0.0) - expected.get(key).copied().unwrap_or(0.0))
                .abs()
        })
        .fold(0.0, f64::max)
}

fn permute_local_vertices(mut mesh: Mesh) -> Mesh {
    for element in &mut mesh.elements {
        match element.element_type {
            ElementType::Triangle => element.nodes.rotate_left(1),
            ElementType::Tetrahedron => {
                // A three-cycle preserves the cell orientation while changing
                // the local face/edge numbering presented to the basis.
                let old = element.nodes.clone();
                element.nodes[0] = old[1];
                element.nodes[1] = old[2];
                element.nodes[2] = old[0];
            }
            _ => {}
        }
    }
    for boundary in &mut mesh.boundaries {
        let element = &mesh.elements[boundary.element_idx];
        boundary.nodes = if mesh.dimension == 2 {
            let (a, b) = element.edges()[boundary.local_idx];
            vec![a, b]
        } else {
            element.faces()[boundary.local_idx].clone()
        };
    }
    mesh
}

fn assemble_reference_checked(mesh: &Mesh) -> BTreeMap<(usize, usize), f64> {
    let space = HdivSpace::new();
    space
        .validate(mesh)
        .expect("RT0 topology must pass checked admission");
    let assembled = assemble_bilinear_checked(&DivergenceDivergenceForm, &space, &space, mesh, 4)
        .expect("checked RT0 divergence assembly must succeed");
    aggregate(&assembled)
}

fn assert_matrix_contract(mesh: &Mesh, label: &str) -> BTreeMap<(usize, usize), f64> {
    let space = HdivSpace::new();
    let actual = assemble_reference_checked(mesh);
    let expected = reference_divergence_matrix(mesh, &space);
    let residual = max_map_residual(&actual, &expected);
    assert!(
        residual < 2.0e-10,
        "{label} RT0 divergence matrix reference residual {residual:.3e}"
    );

    let symmetry = actual
        .iter()
        .map(|(&(row, column), &value)| {
            (value - actual.get(&(column, row)).copied().unwrap_or(0.0)).abs()
        })
        .fold(0.0, f64::max);
    assert!(
        symmetry < 2.0e-12,
        "{label} RT0 divergence matrix symmetry residual {symmetry:.3e}"
    );

    let n = space.num_global_dofs(mesh);
    let probe: Vec<f64> = (0..n).map(|dof| 0.37 + 0.083 * dof as f64).collect();
    let energy = actual
        .iter()
        .map(|(&(row, column), &value)| value * probe[row] * probe[column])
        .sum::<f64>();
    assert!(
        energy > 1.0e-10,
        "{label} RT0 divergence energy was non-positive: {energy:.3e}"
    );

    actual
}

#[test]
fn production_rt0_divergence_matrix_matches_signed_flux_reference() {
    for (label, mesh) in [
        ("triangles", sonium_fem::mesh::unit_square_triangles(2)),
        ("tetrahedra", sonium_fem::mesh::unit_cube_tetrahedra(1)),
    ] {
        let original = assert_matrix_contract(&mesh, label);
        let permuted_mesh = permute_local_vertices(mesh.clone());
        let permuted = assert_matrix_contract(&permuted_mesh, &format!("permuted {label}"));
        // Topology numbering is assigned by first occurrence, so a local
        // permutation may renumber global facets. Compare by canonical facet
        // node sets rather than by incidental integer DOF labels.
        let original = canonical_facet_matrix(&mesh, &original);
        let permuted = canonical_facet_matrix(&permuted_mesh, &permuted);
        let permutation_residual = max_canonical_residual(&original, &permuted);
        assert!(
            permutation_residual < 2.0e-10,
            "{} RT0 matrix changed under orientation-preserving local permutation: {:.3e}",
            label,
            permutation_residual
        );
    }
}

#[test]
fn checked_rt0_mixed_coupling_conserves_signed_facet_flux() {
    for mesh in [
        sonium_fem::mesh::unit_square_triangles(1),
        sonium_fem::mesh::unit_cube_tetrahedra(1),
    ] {
        let flux = HdivSpace::new();
        let pressure = H1Space::new(PolynomialDegree::P1);
        let system = assemble_mixed_checked(
            &VectorMassForm { coefficient: 1.0 },
            &HdivDivergenceForm,
            &flux,
            &pressure,
            &mesh,
            4,
        )
        .expect("checked mixed RT0 assembly must admit both spaces");
        assert_eq!(system.block_01.num_rows, flux.num_global_dofs(&mesh));
        assert_eq!(system.block_01.num_cols, pressure.num_global_dofs(&mesh));

        let topology = MeshTopology::build(&mesh);
        let mut expected_row_sums = vec![0.0; flux.num_global_dofs(&mesh)];
        for element in 0..mesh.num_elements() {
            let dofs = flux.local_to_global(&mesh, element);
            let signs = if mesh.elements[element].element_type.dimension() == 2 {
                topology.element_edge_signs(element)
            } else {
                topology.element_face_signs(element)
            };
            for (local, &global) in dofs.iter().enumerate() {
                // H1 P1 is a partition of unity, so summing the pressure
                // columns leaves -∫ div(phi_global) = -Σ signed facet flux.
                expected_row_sums[global] -= signs[local] as f64;
            }
        }

        let actual = aggregate(&system.block_01);
        let mut actual_row_sums = vec![0.0; expected_row_sums.len()];
        for (&(row, _), &value) in &actual {
            actual_row_sums[row] += value;
        }
        let max_residual = actual_row_sums
            .iter()
            .zip(&expected_row_sums)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            max_residual < 2.0e-11,
            "RT0 mixed divergence row-sum conservation residual {max_residual:.3e}"
        );
    }
}

#[test]
fn checked_rt0_mixed_assembly_rejects_malformed_connectivity() {
    let mut mesh = sonium_fem::mesh::unit_square_triangles(1);
    mesh.elements[0].nodes.truncate(2);
    let result = assemble_mixed_checked(
        &VectorMassForm { coefficient: 1.0 },
        &HdivDivergenceForm,
        &HdivSpace::new(),
        &H1Space::new(PolynomialDegree::P1),
        &mesh,
        2,
    );
    assert!(matches!(
        result,
        Err(SpaceValidationError::InvalidMesh {
            element: 0,
            reason: "element has too few node indices"
        })
    ));
}
