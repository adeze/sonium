//! Matrix Market download and parsing for real-world sparse matrix benchmarks.
//!
//! Downloads matrices from the University of Florida Sparse Matrix Collection
//! via the Matrix Market API, parses them into CSR format.
//!
//! Use with the sonium-benchmark `--petsc` flag to compare accuracy against
//! PETSc reference solutions on real engineering matrices.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};

/// A curated Matrix Market matrix for benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatrixMarketId {
    /// Collection/Group name (e.g., "Boeing", "HB", "NFSEM")
    pub collection: &'static str,
    /// Matrix name within the collection
    pub name: &'static str,
    /// Optional direct URL; defaults to NIST Matrix Market
    pub url: Option<&'static str>,
    /// Short human-readable description
    pub description: &'static str,
    /// Expected problem size
    pub expected_size: &'static str,
}

impl MatrixMarketId {
    /// Full URL to the .mtx or .mtx.gz file.
    pub fn url(&self) -> String {
        if let Some(u) = self.url {
            u.to_string()
        } else {
            format!(
                "https://math.nist.gov/pub/MatrixMarket2/{}/{}.mtx.gz",
                self.collection, self.name
            )
        }
    }

    /// Short display name for bench output tables.
    pub fn display_name(&self) -> String {
        format!("{}/{}", self.collection, self.name)
    }
}

/// Standard curated matrices for solver accuracy benchmarking.
///
/// All are real-valued square matrices from established engineering
/// and scientific computing test collections.
pub fn standard_matrices() -> Vec<MatrixMarketId> {
    vec![
        MatrixMarketId {
            collection: "Boeing",
            name: "bcsstk15",
            url: Some("https://math.nist.gov/pub/MatrixMarket2/Boeing/bcsstk15.mtx.gz"),
            description: "Dynamic stiffness matrix (structural dynamics)",
            expected_size: "~23K DOF",
        },
        MatrixMarketId {
            collection: "HB",
            name: "sherman1",
            url: Some("https://math.nist.gov/pub/MatrixMarket2/HB/sherman1.mtx.gz"),
            description: "Oil reservoir simulation (least squares / parabolic PDE)",
            expected_size: "~3K DOF",
        },
        MatrixMarketId {
            collection: "NFSEM",
            name: "beatdrop",
            url: Some("https://math.nist.gov/pub/MatrixMarket2/NFSEM/beatdrop.mtx.gz"),
            description: "Finite element model (acoustic / BEM)",
            expected_size: "~13K DOF",
        },
        MatrixMarketId {
            collection: "Pothen",
            name: "shipsec1",
            url: Some("https://math.nist.gov/pub/MatrixMarket2/Pothen/shipsec1.mtx.gz"),
            description: "Navier-Stokes constraint matrix",
            expected_size: "~14K DOF",
        },
        MatrixMarketId {
            collection: "Williams",
            name: "cont-201",
            url: Some("https://math.nist.gov/pub/MatrixMarket2/Williams/cont-201.mtx.gz"),
            description: "Continuum 2D/3D PDE",
            expected_size: "~8K DOF",
        },
    ]
}

/// Fetch and parse a Matrix Market matrix into CSR format.
///
/// Downloads (or loads from cache) the gzipped or plain MTX file and
/// parses the coordinate format into a `CsrMatrix<f64>`.
pub fn fetch_and_parse(
    mtx_id: &MatrixMarketId,
) -> Result<(sonium_solvers::CsrMatrix<f64>, usize), MatrixMarketError> {
    let url = mtx_id.url();
    log::info!("Fetching {} from {}", mtx_id.display_name(), url);

    let resp = ureq::get(&url)
        .call()
        .map_err(|e| MatrixMarketError::NetworkError(e.to_string()))?;

    let reader = resp.into_reader();

    let reader: Box<dyn BufRead> = if url.ends_with(".gz") {
        let decoder = flate2::read::GzDecoder::new(reader);
        Box::new(BufReader::new(decoder))
    } else {
        Box::new(BufReader::new(reader))
    };

    parse_matrix_market(reader)
}

/// Parse a Matrix Market coordinate file from any `BufRead` source.
pub fn parse_matrix_market<R: BufRead>(
    mut reader: R,
) -> Result<(sonium_solvers::CsrMatrix<f64>, usize), MatrixMarketError> {
    // Read lines until we hit the header (first non-comment, non-blank)
    let mut header_line = String::new();
    loop {
        header_line.clear();
        let bytes = reader
            .read_line(&mut header_line)
            .map_err(MatrixMarketError::IoError)?;
        if bytes == 0 {
            return Err(MatrixMarketError::ParseError("Unexpected EOF".into()));
        }
        let trimmed = header_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        break;
    }

    let parts: Vec<&str> = header_line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(MatrixMarketError::ParseError(format!(
            "Invalid MTX header: {}",
            header_line.trim()
        )));
    }

    let num_rows: usize = parts[0]
        .parse()
        .map_err(|e| MatrixMarketError::ParseError(format!("Invalid row count: {e}")))?;
    let num_cols: usize = parts[1]
        .parse()
        .map_err(|e| MatrixMarketError::ParseError(format!("Invalid col count: {e}")))?;
    let _nnz: usize = parts[2]
        .parse()
        .map_err(|e| MatrixMarketError::ParseError(format!("Invalid nnz: {e}")))?;

    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

    loop {
        let mut line = String::new();
        let bytes = reader
            .read_line(&mut line)
            .map_err(MatrixMarketError::IoError)?;
        if bytes == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            continue;
        }

        let i: usize = fields[0]
            .parse()
            .map_err(|e| MatrixMarketError::ParseError(format!("Invalid row index: {e}")))?;
        let j: usize = fields[1]
            .parse()
            .map_err(|e| MatrixMarketError::ParseError(format!("Invalid col index: {e}")))?;
        let v: f64 = fields[2]
            .parse()
            .map_err(|e| MatrixMarketError::ParseError(format!("Invalid value: {e}")))?;

        // Convert from 1-based (MTX) to 0-based indexing
        triplets.push((i.saturating_sub(1), j.saturating_sub(1), v));
    }

    log::info!(
        "Parsed {}x{} with {} nonzeros",
        num_rows,
        num_cols,
        triplets.len()
    );

    let nnz = triplets.len();
    let matrix = sonium_solvers::CsrMatrix::from_triplets(num_rows, num_cols, triplets);
    Ok((matrix, nnz))
}

/// Standard test RHS: Gaussian random vector with fixed seed.
///
/// Uses a simple LCG for reproducibility without adding a rand dependency.
pub fn standard_rhs(n: usize) -> Vec<f64> {
    let seed = 0xdeadbeef_u64;
    let mut state = seed;

    // LCG: same family as many established RNGs (e.g., Java's java.util.Random)
    fn next_f64(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // SplitMix64-like bit mixing for better randomness
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        // Normalize to [0, 1) then use Box-Muller for Gaussian
        (z >> 12) as f64 * (1.0 / (2.0_f64.powi(52)))
    }

    (0..n)
        .map(|_| {
            let u1 = next_f64(&mut state).max(1e-30);
            let u2 = next_f64(&mut state);
            let mag = (-2.0 * u1.ln()).sqrt();
            mag * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

/// Errors from Matrix Market fetch/parse operations.
#[derive(Debug, thiserror::Error)]
#[allow(clippy::enum_variant_names)]
pub enum MatrixMarketError {
    #[error("Network error fetching matrix: {0}")]
    NetworkError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_url_default() {
        let id = MatrixMarketId {
            collection: "HB",
            name: "sherman1",
            url: None,
            description: "test",
            expected_size: "~3K",
        };
        assert!(id.url().contains("sherman1"));
        assert!(id.url().ends_with(".mtx.gz"));
    }

    #[test]
    fn test_id_url_explicit() {
        let id = MatrixMarketId {
            collection: "HB",
            name: "sherman1",
            url: Some("https://example.com/my.mtx"),
            description: "test",
            expected_size: "~3K",
        };
        assert_eq!(id.url(), "https://example.com/my.mtx");
    }

    #[test]
    fn test_standard_rhs_reproducible() {
        let a = standard_rhs(200);
        let b = standard_rhs(200);
        assert_eq!(a.len(), 200);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-30, "RHS not reproducible");
        }
    }

    #[test]
    fn test_standard_rhs_normal_distribution() {
        let rhs = standard_rhs(10_000);
        let mean: f64 = rhs.iter().sum::<f64>() / rhs.len() as f64;
        let var: f64 = rhs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rhs.len() as f64;
        // Should be approximately N(0, 1)
        assert!((mean).abs() < 0.1, "Mean {} too far from 0", mean);
        assert!((var - 1.0).abs() < 0.1, "Variance {} too far from 1", var);
    }
}
