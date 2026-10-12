//! Common production-form assembly for longitudinal piezoelectric rod gates.

use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_mass};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::rectangular_mesh_triangles;
use sonium_fem::physics::coupled::piezoelectric::{
    PiezoCouplingForm, PiezoDielectricForm, PiezoMaterial, PiezoStiffnessForm,
};
use sonium_fem::spaces::H1Space;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::SparseLuFactorization;
use sonium_solvers::eigen::symmetric_eigen;

#[derive(Debug)]
pub(super) struct Rod {
    pub nodes: usize,
    pub length: f64,
    pub phi_scale: f64,
    pub stiffness: Vec<f64>,
    pub mass: Vec<f64>,
    pub symmetry_error: f64,
}

#[derive(Debug)]
pub(super) struct RodModes {
    pub eigenvalues: Vec<f64>,
    pub displacements: DenseMatrix<f64>,
    pub potentials: Vec<Vec<f64>>,
}

pub(super) fn assemble(n: usize, length: f64, material: &PiezoMaterial) -> Result<Rod, String> {
    const WIDTH: f64 = 0.001;
    let mesh = rectangular_mesh_triangles(0.0, length, 0.0, WIDTH, n, 1);
    let space = H1Space::new(PolynomialDegree::P1);
    // The longitudinal coordinate is rotated onto x. The d33 coefficients
    // retain their poling-direction meaning, with derivatives along x.
    let mut mechanical = PiezoStiffnessForm::d33(material);
    mechanical.axis = 0;
    let mut inverse = PiezoCouplingForm::inverse_d33(material);
    inverse.mech_axis = 0;
    inverse.elec_axis = 0;
    let mut direct = PiezoCouplingForm::direct_d33(material);
    direct.mech_axis = 0;
    direct.elec_axis = 0;
    let mut electrical = PiezoDielectricForm::d33(material);
    electrical.axis = 0;
    let blocks = [
        assemble_bilinear(&mechanical, &space, &space, &mesh, 2),
        assemble_bilinear(&inverse, &space, &space, &mesh, 2),
        assemble_bilinear(&direct, &space, &space, &mesh, 2),
        assemble_bilinear(&electrical, &space, &space, &mesh, 2),
    ];
    let nodes = n + 1;
    let count = 2 * nodes;
    let c = material.stiffness[2][2];
    let epsilon = material.permittivity[2][2];
    let phi_scale = length * (c / epsilon).sqrt();
    let work_scale = c * WIDTH * length;
    let mut stiffness = vec![0.0; count * count];
    // Galerkin restriction ties both transverse rows to each cross section.
    // This is the declared 1D rod subspace, rather than a singular 2D field
    // with uncontrolled transverse modes and no transverse derivative term.
    let section: Vec<usize> = mesh
        .nodes
        .iter()
        .map(|p| (p.x * n as f64 / length).round() as usize)
        .collect();
    for (block, ((row_offset, col_offset), scale)) in blocks.iter().zip([
        ((0, 0), length * length / work_scale),
        ((0, nodes), length * phi_scale / work_scale),
        ((nodes, 0), length * phi_scale / work_scale),
        ((nodes, nodes), phi_scale * phi_scale / work_scale),
    ]) {
        for entry in 0..block.values.len() {
            let row = row_offset + section[block.rows[entry]];
            let col = col_offset + section[block.cols[entry]];
            stiffness[row * count + col] += scale * block.values[entry];
        }
    }
    let raw_mass = assemble_mass(&mesh, PolynomialDegree::P1);
    let mut mass = vec![0.0; nodes * nodes];
    for entry in 0..raw_mass.values.len() {
        mass[section[raw_mass.rows[entry]] * nodes + section[raw_mass.cols[entry]]] +=
            raw_mass.values[entry] / (WIDTH * length);
    }
    let mut symmetry_error = 0.0_f64;
    for row in 0..count {
        for col in 0..count {
            symmetry_error = symmetry_error
                .max((stiffness[row * count + col] - stiffness[col * count + row]).abs());
        }
    }
    if !symmetry_error.is_finite() || symmetry_error > 1e-10 {
        return Err(format!(
            "coupled rod stiffness is not symmetric: {symmetry_error:.3e}"
        ));
    }
    Ok(Rod {
        nodes,
        length,
        phi_scale,
        stiffness,
        mass,
        symmetry_error,
    })
}

impl Rod {
    pub fn action(&self, values: &[f64]) -> Vec<f64> {
        let count = 2 * self.nodes;
        (0..count)
            .map(|row| {
                (0..count)
                    .map(|col| self.stiffness[row * count + col] * values[col])
                    .sum()
            })
            .collect()
    }

    pub fn solve(&self, load: &[f64], boundary: &[(usize, f64)]) -> Result<Vec<f64>, String> {
        let count = 2 * self.nodes;
        let mut fixed = vec![None; count];
        for &(index, value) in boundary {
            fixed[index] = Some(value);
        }
        let mut rhs = load.to_vec();
        let mut matrix = TripletMatrix::new(count, count);
        for row in 0..count {
            if let Some(value) = fixed[row] {
                matrix.push(row, row, 1.0);
                rhs[row] = value;
            } else {
                for (col, prescribed) in fixed.iter().enumerate() {
                    let value = self.stiffness[row * count + col];
                    if let Some(known) = prescribed {
                        rhs[row] -= value * known;
                    } else if value != 0.0 {
                        matrix.push(row, col, value);
                    }
                }
            }
        }
        let factor =
            SparseLuFactorization::factorize(&matrix.to_csr()).map_err(|e| e.to_string())?;
        let result = factor.solve(&rhs).map_err(|e| e.to_string())?;
        if result.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite coupled piezoelectric solution".into());
        }
        Ok(result)
    }

    // Electrically insulating/open electrodes enforce zero normal D through
    // the natural electrical equations. Only phi(0)=0 fixes the gauge; every
    // other electrical DOF, including the second electrode, is solved.
    pub fn open_circuit_modes(&self) -> Result<RodModes, String> {
        let n = self.nodes;
        let count = 2 * n;
        let mut electric = TripletMatrix::new(n - 1, n - 1);
        for i in 1..n {
            for j in 1..n {
                let value = self.stiffness[(n + i) * count + n + j];
                if value != 0.0 {
                    electric.push(i - 1, j - 1, value);
                }
            }
        }
        let factor =
            SparseLuFactorization::factorize(&electric.to_csr()).map_err(|e| e.to_string())?;
        let mut response = vec![vec![0.0; n]; n];
        let mut effective = DenseMatrix::zeros(n, n);
        for column in 0..n {
            let rhs: Vec<f64> = (1..n)
                .map(|row| -self.stiffness[(n + row) * count + column])
                .collect();
            let phi = factor.solve(&rhs).map_err(|e| e.to_string())?;
            response[column][1..].copy_from_slice(&phi);
            for row in 0..n {
                effective[(row, column)] = self.stiffness[row * count + column]
                    + (1..n)
                        .map(|j| self.stiffness[row * count + n + j] * phi[j - 1])
                        .sum::<f64>();
            }
        }
        let mass = DenseMatrix::from_row_major_vec(n, n, self.mass.clone());
        let (mass_eigenvalues, mass_vectors) =
            symmetric_eigen(&mass, n).map_err(|e| e.to_string())?;
        if mass_eigenvalues.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err("rod mass is not positive definite".into());
        }
        let whitening = DenseMatrix::from_fn(n, n, |i, j| {
            mass_vectors[(i, j)] / mass_eigenvalues[j].sqrt()
        });
        let transformed = whitening.matmul_transa(&effective.matmul(&whitening));
        let (eigenvalues, eigenvectors) =
            symmetric_eigen(&transformed, n).map_err(|e| e.to_string())?;
        let modes = whitening.matmul(&eigenvectors);
        let potentials: Vec<Vec<f64>> = (0..n)
            .map(|mode| {
                (0..n)
                    .map(|row| {
                        (0..n)
                            .map(|col| response[col][row] * modes[(col, mode)])
                            .sum()
                    })
                    .collect()
            })
            .collect();
        Ok(RodModes {
            eigenvalues,
            displacements: modes,
            potentials,
        })
    }
}
