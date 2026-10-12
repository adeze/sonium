//! Sparse H(curl) solves with exact transverse boundary traces and true residuals.

use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::mesh::{Mesh, topology::MeshTopology};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_solvers::{CsrMatrix, direct::SparseLuFactorization};
use std::collections::BTreeSet;

#[derive(Debug)]
pub(super) struct EdgeSolution {
    pub(super) values: Vec<Complex64>,
    pub(super) residual: f64,
    pub(super) free_dofs: usize,
}

// Cartesian rectangular meshes only: E=(0,E_y(x)), with exact edge
// integrals on prescribed faces. The unmodified operator determines the
// residual after lifting; constrained rows are checked separately.
pub(super) fn solve_transverse(
    mesh: &Mesh,
    operator: &CsrMatrix<Complex64>,
    markers: &[i32],
    exact_y: impl Fn(f64) -> Complex64,
) -> Result<EdgeSolution, String> {
    let space = HcurlSpace::new();
    let count = operator.num_rows;
    let topology = MeshTopology::build(mesh);
    let mut prescribed = BTreeSet::new();
    for &marker in markers {
        prescribed.extend(space.boundary_dofs(mesh, marker));
    }
    let free_dofs = count - prescribed.len();
    if free_dofs == 0 {
        return Err("H(curl) field has no free edge unknowns".into());
    }
    let mut boundary = vec![Complex64::new(0.0, 0.0); count];
    for &edge in &prescribed {
        let (a, b) = topology.edges[edge];
        let start = mesh.nodes[a];
        let end = mesh.nodes[b];
        // Global H(curl) DOFs are oriented line integrals from low to high
        // node ID. On this Cartesian mesh, prescribed nonzero edges are
        // vertical, so this integral is exact without boundary quadrature.
        boundary[edge] = exact_y(start.x) * (end.y - start.y);
    }
    // Scale the physical block before inserting unit Dirichlet diagonals.
    // SI conductivity/curl entries otherwise dwarf those diagonals and
    // trigger the sparse solver's relative pivot safeguard.
    let operator_scale = operator
        .values
        .iter()
        .map(|v| v.norm())
        .fold(0.0_f64, f64::max);
    let mut matrix = TripletMatrix::new(count, count);
    let mut rhs = vec![Complex64::new(0.0, 0.0); count];
    for row in 0..count {
        if prescribed.contains(&row) {
            matrix.push(row, row, Complex64::new(1.0, 0.0));
            rhs[row] = boundary[row];
        } else {
            for (column, raw_value) in operator.row_entries(row) {
                let value = raw_value / operator_scale;
                if prescribed.contains(&column) {
                    rhs[row] -= value * boundary[column];
                } else {
                    matrix.push(row, column, value);
                }
            }
        }
    }
    let factor = SparseLuFactorization::factorize(&matrix.to_csr())
        .map_err(|error| format!("H(curl) factorization: {error}"))?;
    let solution = factor
        .solve(&rhs)
        .map_err(|error| format!("H(curl) solve: {error}"))?;
    if solution
        .iter()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err("nonfinite H(curl) field solution".into());
    }
    // Recompute the physical unmodified operator residual on free DOFs.
    let mut residual_squared = 0.0;
    let mut scale_squared = 0.0;
    for row in 0..count {
        if prescribed.contains(&row) {
            continue;
        }
        let mut action = Complex64::new(0.0, 0.0);
        let mut scale = 0.0;
        for (column, value) in operator.row_entries(row) {
            action += value * solution[column];
            scale += (value * solution[column]).norm();
        }
        residual_squared += action.norm_sqr();
        scale_squared += scale * scale;
    }
    let residual = (residual_squared / scale_squared).sqrt();
    for &edge in &prescribed {
        if (solution[edge] - boundary[edge]).norm() > 1e-12 * boundary[edge].norm().max(1.0) {
            return Err("H(curl) boundary trace was not enforced".into());
        }
    }
    Ok(EdgeSolution {
        values: solution,
        residual,
        free_dofs,
    })
}
