//! Full checked mixed-system assembly and block-operator consistency.

use sonium_fem::assembly::{MixedSystem, assemble_mixed_full_checked};
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::spaces::{BasisTabulation, DGSpace, H1Space};
use sonium_solvers::LinearOperator;

#[derive(Debug, Clone, Copy)]
struct WeightedMass(f64);

impl BilinearForm<f64> for WeightedMass {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let factor = self.0 * weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, factor * test.value(i) * trial.value(j));
            }
        }
    }
}

fn apply_both(system: &MixedSystem<f64>, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let block = system.to_block_operator().apply(x);
    let mono = system.to_monolithic().to_csr().apply(x);
    (block, mono)
}

#[test]
fn full_mixed_assembly_keeps_all_four_blocks_and_matches_monolithic_apply() {
    let mesh = unit_square_triangles(2);
    let space_0 = H1Space::new(sonium_fem::basis::PolynomialDegree::P1);
    let space_1 = DGSpace::p0();
    let system = assemble_mixed_full_checked(
        &WeightedMass(1.0),
        &WeightedMass(2.0),
        &WeightedMass(3.0),
        &WeightedMass(4.0),
        &space_0,
        &space_1,
        &mesh,
        2,
    )
    .expect("valid mixed spaces should assemble");

    assert_eq!(system.block_00.num_rows, system.n_dofs_0);
    assert_eq!(system.block_00.num_cols, system.n_dofs_0);
    assert_eq!(system.block_01.num_rows, system.n_dofs_0);
    assert_eq!(system.block_01.num_cols, system.n_dofs_1);
    assert_eq!(system.block_10.num_rows, system.n_dofs_1);
    assert_eq!(system.block_10.num_cols, system.n_dofs_0);
    assert_eq!(system.block_11.num_rows, system.n_dofs_1);
    assert_eq!(system.block_11.num_cols, system.n_dofs_1);
    assert!(
        system.block_11.nnz() > 0,
        "lower-right block must be assembled"
    );

    let x: Vec<f64> = (0..system.n_dofs_0 + system.n_dofs_1)
        .map(|index| 0.25 + 0.13 * index as f64)
        .collect();
    let (block_result, monolithic_result) = apply_both(&system, &x);
    assert_eq!(block_result.len(), monolithic_result.len());
    let max_error = block_result
        .iter()
        .zip(&monolithic_result)
        .map(|(block, mono)| (block - mono).abs())
        .fold(0.0, f64::max);
    assert!(max_error < 1.0e-12, "block/monolithic error {max_error}");
}

#[test]
fn full_mixed_assembly_rejects_malformed_topology_before_scatter() {
    let mut mesh = unit_square_triangles(1);
    mesh.elements[0].nodes.pop();
    let space_0 = H1Space::new(sonium_fem::basis::PolynomialDegree::P1);
    let space_1 = DGSpace::p0();
    let result = assemble_mixed_full_checked(
        &WeightedMass(1.0),
        &WeightedMass(2.0),
        &WeightedMass(3.0),
        &WeightedMass(4.0),
        &space_0,
        &space_1,
        &mesh,
        2,
    );
    assert!(result.is_err());
}
