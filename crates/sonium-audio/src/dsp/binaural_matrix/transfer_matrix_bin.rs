use num_complex::Complex64;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_direct::dense_svd;

/// Frequency-domain transfer matrix for one listener/head position.
///
/// `values` is row-major with shape `num_ears x num_speakers`.
/// For binaural room correction `num_ears` is normally 2.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferMatrixBin {
    pub num_ears: usize,
    pub num_speakers: usize,
    pub values: Vec<Complex64>,
}

impl TransferMatrixBin {
    pub fn new(num_ears: usize, num_speakers: usize, values: Vec<Complex64>) -> Self {
        assert_eq!(values.len(), num_ears * num_speakers);
        Self {
            num_ears,
            num_speakers,
            values,
        }
    }

    pub(super) fn as_matrix(&self) -> DenseMatrix<Complex64> {
        DenseMatrix::from_row_major_vec(self.num_ears, self.num_speakers, self.values.clone())
    }
}

/// Per-position reconstruction errors for a row-major correction matrix.
pub fn position_errors(
    positions: &[TransferMatrixBin],
    correction: &[Complex64],
    target: &[Complex64],
) -> Result<Vec<f64>, String> {
    let first = positions
        .first()
        .ok_or_else(|| "at least one transfer matrix position is required".to_string())?;
    if correction.len() != first.num_speakers * first.num_ears {
        return Err(format!(
            "correction has {} entries, expected {}",
            correction.len(),
            first.num_speakers * first.num_ears
        ));
    }
    if target.len() != first.num_ears * first.num_ears {
        return Err(format!(
            "target has {} entries, expected {}",
            target.len(),
            first.num_ears * first.num_ears
        ));
    }
    let f =
        DenseMatrix::from_row_major_vec(first.num_speakers, first.num_ears, correction.to_vec());
    let target_matrix =
        DenseMatrix::from_row_major_vec(first.num_ears, first.num_ears, target.to_vec());
    let mut errors = Vec::with_capacity(positions.len());
    for matrix in positions {
        if matrix.num_ears != first.num_ears || matrix.num_speakers != first.num_speakers {
            return Err("all transfer matrices must have the same shape".to_string());
        }
        let delivered = matrix.as_matrix().matmul(&f);
        let mut error = 0.0;
        let mut count = 0usize;
        for row in 0..first.num_ears {
            for col in 0..first.num_ears {
                error += (delivered[(row, col)] - target_matrix[(row, col)]).norm_sqr();
                count += 1;
            }
        }
        errors.push(if count == 0 {
            0.0
        } else {
            error / count as f64
        });
    }
    Ok(errors)
}

/// Estimate the condition number using singular values.
///
/// Computed through the f64 SVD of the real block embedding
/// `[[Re, -Im], [Im, Re]]`, whose spectrum duplicates each singular value
/// twice (leaving the max/min ratio unchanged).
pub fn condition_number(matrix: &TransferMatrixBin) -> f64 {
    let h = matrix.as_matrix();
    let (m, n) = h.shape();
    if m == 0 || n == 0 {
        return f64::INFINITY;
    }
    let emb = DenseMatrix::from_fn(2 * m, 2 * n, |i, j| {
        let z = h[(i % m, j % n)];
        match (i < m, j < n) {
            (true, true) => z.re,
            (true, false) => -z.im,
            (false, true) => z.im,
            (false, false) => z.re,
        }
    });
    let Ok(svd) = dense_svd(&emb) else {
        return f64::INFINITY;
    };
    let mut min_sv = f64::INFINITY;
    let mut max_sv = 0.0;
    for sv in svd.singular_values.iter().copied() {
        if sv > max_sv {
            max_sv = sv;
        }
        if sv > 0.0 && sv < min_sv {
            min_sv = sv;
        }
    }
    if min_sv.is_finite() && min_sv > 0.0 {
        max_sv / min_sv
    } else {
        f64::INFINITY
    }
}
