/// Compute J^T W J (n_params x n_params).
pub(super) fn jtw_j(jac: &[Vec<f64>], w: &[f64]) -> Vec<Vec<f64>> {
    let n_params = jac.first().map_or(0, |row| row.len());
    let n_res = jac.len();
    let mut result = vec![vec![0.0; n_params]; n_params];

    for j in 0..n_params {
        for k in j..n_params {
            let mut val = 0.0;
            for i in 0..n_res {
                val += w[i] * jac[i][j] * jac[i][k];
            }
            result[j][k] = val;
            result[k][j] = val;
        }
    }

    result
}

/// Compute J^T W r (n_params vector).
pub(super) fn jtw_r(jac: &[Vec<f64>], w: &[f64], r: &[f64]) -> Vec<f64> {
    let n_params = jac.first().map_or(0, |row| row.len());
    let n_res = jac.len();
    let mut result = vec![0.0; n_params];

    for j in 0..n_params {
        let mut val = 0.0;
        for i in 0..n_res {
            val += w[i] * jac[i][j] * r[i];
        }
        result[j] = val;
    }

    result
}
