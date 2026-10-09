//! Production FEM operator invariants for release QA.
//!
//! The checks use the same assembled Helmholtz stiffness/mass path used by
//! frequency-domain applications.  They are independent of a linear solver:
//! symmetry, constant-field conservation, and mass normalization are checked
//! directly from the CSR coefficients.

use sonium_fem::assembly::HelmholtzAssembler;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.operator-invariants.v1";
const TOLERANCE: f64 = 5.0e-13;

#[derive(Debug)]
struct Metrics {
    max_stiffness_row_sum: f64,
    max_mass_symmetry: f64,
    max_stiffness_symmetry: f64,
    mass_total: f64,
    stiffness_energy: f64,
}

fn csr_value(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    row: usize,
    col: usize,
) -> f64 {
    let start = row_ptrs[row];
    let end = row_ptrs[row + 1];
    col_indices[start..end]
        .iter()
        .zip(&values[start..end])
        .find_map(|(&candidate, &value)| (candidate == col).then_some(value))
        .unwrap_or(0.0)
}

fn max_symmetry_error(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    rows: usize,
) -> f64 {
    let mut max_error: f64 = 0.0;
    for row in 0..rows {
        for index in row_ptrs[row]..row_ptrs[row + 1] {
            let col = col_indices[index];
            let transpose = csr_value(row_ptrs, col_indices, values, col, row);
            max_error = max_error.max((values[index] - transpose).abs());
        }
    }
    max_error
}

fn operator_metrics(
    row_ptrs: &[usize],
    col_indices: &[usize],
    k_values: &[f64],
    m_values: &[f64],
    rows: usize,
) -> Result<Metrics, String> {
    if rows == 0 || row_ptrs.len() != rows + 1 {
        return Err("production assembler returned an empty or malformed CSR pattern".into());
    }
    let mut max_stiffness_row_sum: f64 = 0.0;
    let mut mass_total = 0.0;
    let mut mass_row_sums = vec![0.0; rows];
    for row in 0..rows {
        let mut stiffness_sum = 0.0;
        for index in row_ptrs[row]..row_ptrs[row + 1] {
            stiffness_sum += k_values[index];
            mass_row_sums[row] += m_values[index];
            mass_total += m_values[index];
        }
        max_stiffness_row_sum = max_stiffness_row_sum.max(stiffness_sum.abs());
    }

    let max_stiffness_symmetry = max_symmetry_error(row_ptrs, col_indices, k_values, rows);
    let max_mass_symmetry = max_symmetry_error(row_ptrs, col_indices, m_values, rows);

    // A deterministic non-constant vector exercises the assembled energy
    // rather than only the nullspace identity.  The stiffness form must be
    // non-negative for this scalar diffusion operator.
    let probe: Vec<f64> = (0..rows)
        .map(|index| ((index + 1) as f64).sin() + 0.25 * ((index + 3) as f64).cos())
        .collect();
    let mut stiffness_energy = 0.0;
    for row in 0..rows {
        let mut action = 0.0;
        for index in row_ptrs[row]..row_ptrs[row + 1] {
            action += k_values[index] * probe[col_indices[index]];
        }
        stiffness_energy += probe[row] * action;
    }

    let finite = max_stiffness_row_sum.is_finite()
        && max_stiffness_symmetry.is_finite()
        && max_mass_symmetry.is_finite()
        && mass_total.is_finite()
        && stiffness_energy.is_finite()
        && mass_row_sums.iter().all(|value| value.is_finite());
    if !finite {
        return Err("assembled operator invariants contained a non-finite value".into());
    }
    Ok(Metrics {
        max_stiffness_row_sum,
        max_mass_symmetry,
        max_stiffness_symmetry,
        mass_total,
        stiffness_energy,
    })
}

fn validate_metrics(metrics: &Metrics) -> Result<(), String> {
    if !metrics.max_stiffness_row_sum.is_finite()
        || !metrics.max_stiffness_symmetry.is_finite()
        || !metrics.max_mass_symmetry.is_finite()
        || !metrics.mass_total.is_finite()
        || !metrics.stiffness_energy.is_finite()
    {
        return Err("assembled operator invariants contained a non-finite value".into());
    }
    if metrics.max_stiffness_row_sum > TOLERANCE {
        return Err(format!(
            "stiffness constant-field conservation failed: max row sum {:.3e}",
            metrics.max_stiffness_row_sum
        ));
    }
    if metrics.max_stiffness_symmetry > TOLERANCE || metrics.max_mass_symmetry > TOLERANCE {
        return Err(format!(
            "assembled operator symmetry failed: stiffness {:.3e}, mass {:.3e}",
            metrics.max_stiffness_symmetry, metrics.max_mass_symmetry
        ));
    }
    if (metrics.mass_total - 1.0).abs() > TOLERANCE {
        return Err(format!(
            "consistent mass normalization failed: total mass {:.16e}",
            metrics.mass_total
        ));
    }
    if metrics.stiffness_energy <= 0.0 {
        return Err(format!(
            "stiffness energy must be positive for the probe vector: {:.3e}",
            metrics.stiffness_energy
        ));
    }

    Ok(())
}

fn run() -> Result<Metrics, String> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4);
    let assembler = HelmholtzAssembler::new(&mesh, PolynomialDegree::P1);
    let metrics = operator_metrics(
        &assembler.row_ptrs,
        &assembler.col_indices,
        &assembler.k_values,
        &assembler.m_values,
        assembler.num_rows,
    )?;
    validate_metrics(&metrics)?;
    Ok(metrics)
}

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_stiffness_row_sum", metrics.max_stiffness_row_sum)
                .metric("max_mass_symmetry", metrics.max_mass_symmetry)
                .metric("max_stiffness_symmetry", metrics.max_stiffness_symmetry)
                .metric("mass_total", metrics.mass_total)
                .metric("mass_total_error", (metrics.mass_total - 1.0).abs())
                .metric("stiffness_energy", metrics.stiffness_energy)
                .maximum("max_stiffness_row_sum", TOLERANCE)
                .maximum("max_mass_symmetry", TOLERANCE)
                .maximum("max_stiffness_symmetry", TOLERANCE)
                .maximum("mass_total_error", TOLERANCE)
                .minimum("stiffness_energy", 0.0)
                .seed("rectangular-triangles-4x4:p1:probe-sin-cos");
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
            eprintln!("FEM OPERATOR INVARIANTS FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stiffness_sign_mutation_is_rejected_by_the_production_invariant_gate() {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4);
        let assembler = HelmholtzAssembler::new(&mesh, PolynomialDegree::P1);
        let mut sign_mutated_stiffness = assembler.k_values.clone();
        for value in &mut sign_mutated_stiffness {
            *value = -*value;
        }
        let metrics = operator_metrics(
            &assembler.row_ptrs,
            &assembler.col_indices,
            &sign_mutated_stiffness,
            &assembler.m_values,
            assembler.num_rows,
        )
        .expect("mutated CSR remains structurally valid");
        let error = validate_metrics(&metrics).expect_err("negative stiffness energy must fail");
        assert!(error.contains("stiffness energy"), "{error}");
    }
}
