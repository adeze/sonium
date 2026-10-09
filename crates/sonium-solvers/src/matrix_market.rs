//! Checked Matrix Market coordinate parsing for solver validation workloads.

use std::io::BufRead;

use num_complex::Complex64;
use sonium_solvers_core::traits::{AccuracyStatus, SolverErrorKind, SolverFailure, SolverStatus};
use sonium_solvers_sparse::CsrMatrix;
use thiserror::Error;

/// Parsed Matrix Market metadata alongside the canonical CSR matrix.
#[derive(Debug, Clone)]
pub struct ParsedMatrixMarket {
    /// Canonical zero-based CSR matrix.
    pub matrix: CsrMatrix<f64>,
    /// Number of entries declared in the coordinate stream, before symmetry expansion.
    pub declared_entries: usize,
    /// Number of stored CSR values after symmetry expansion and duplicate coalescing.
    pub stored_nonzeros: usize,
}

/// Parsed complex Matrix Market metadata alongside canonical CSR storage.
#[derive(Debug, Clone)]
pub struct ParsedComplexMatrixMarket {
    /// Canonical zero-based CSR matrix.
    pub matrix: CsrMatrix<Complex64>,
    /// Number of entries declared in the coordinate stream, before symmetry expansion.
    pub declared_entries: usize,
    /// Number of stored CSR values after symmetry expansion and duplicate coalescing.
    pub stored_nonzeros: usize,
}

/// Typed failures from checked Matrix Market parsing.
#[derive(Debug, Error)]
pub enum MatrixMarketError {
    /// The input stream could not be read.
    #[error("Matrix Market I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The Matrix Market banner, dimensions, or coordinate entries are invalid.
    #[error("Matrix Market parse error: {0}")]
    Parse(String),
}

impl SolverFailure for MatrixMarketError {
    fn error_kind(&self) -> SolverErrorKind {
        SolverErrorKind::OperationFailed
    }

    fn status(&self) -> SolverStatus {
        SolverStatus::NumericalError
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }
}

#[derive(Clone, Copy)]
enum ValueKind {
    Real,
    Integer,
    Pattern,
    Complex,
}

#[derive(Clone, Copy)]
enum Symmetry {
    General,
    Symmetric,
    Hermitian,
}

fn next_data_line<R: BufRead>(reader: &mut R) -> Result<String, MatrixMarketError> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(MatrixMarketError::Parse("unexpected EOF".into()));
        }
        let trimmed = line.trim();
        if !trimmed.is_empty()
            && (!trimmed.starts_with('%') || trimmed.starts_with("%%MatrixMarket"))
        {
            return Ok(trimmed.to_owned());
        }
    }
}

/// Parse a Matrix Market coordinate stream into validated CSR storage.
///
/// Both the standard `%%MatrixMarket matrix coordinate ...` banner and the
/// historical size-only form are accepted. Real, integer, and pattern values
/// are supported; symmetric and real-valued Hermitian matrices are expanded
/// before duplicate entries are coalesced by the CSR constructor.
pub fn parse_matrix_market<R: BufRead>(
    mut reader: R,
) -> Result<ParsedMatrixMarket, MatrixMarketError> {
    let first = next_data_line(&mut reader)?;
    let (value_kind, symmetry, dimensions_line) = if first.starts_with("%%MatrixMarket") {
        let banner: Vec<_> = first.split_whitespace().collect();
        if banner.len() != 5
            || !banner[1].eq_ignore_ascii_case("matrix")
            || !banner[2].eq_ignore_ascii_case("coordinate")
        {
            return Err(MatrixMarketError::Parse(format!(
                "unsupported banner: {first}"
            )));
        }
        let value_kind = match banner[3].to_ascii_lowercase().as_str() {
            "real" => ValueKind::Real,
            "integer" => ValueKind::Integer,
            "pattern" => ValueKind::Pattern,
            "complex" => ValueKind::Complex,
            other => {
                return Err(MatrixMarketError::Parse(format!(
                    "unsupported value kind: {other}"
                )));
            }
        };
        let symmetry = match banner[4].to_ascii_lowercase().as_str() {
            "general" => Symmetry::General,
            "symmetric" => Symmetry::Symmetric,
            "hermitian" if matches!(value_kind, ValueKind::Real | ValueKind::Integer) => {
                Symmetry::Hermitian
            }
            other => {
                return Err(MatrixMarketError::Parse(format!(
                    "unsupported symmetry: {other}"
                )));
            }
        };
        if matches!(value_kind, ValueKind::Complex) {
            return Err(MatrixMarketError::Parse(
                "complex values require parse_complex_matrix_market".into(),
            ));
        }
        (value_kind, symmetry, next_data_line(&mut reader)?)
    } else {
        (ValueKind::Real, Symmetry::General, first)
    };

    let dimensions: Vec<_> = dimensions_line.split_whitespace().collect();
    if dimensions.len() != 3 {
        return Err(MatrixMarketError::Parse(format!(
            "expected three dimensions, got: {dimensions_line}"
        )));
    }
    let rows: usize = dimensions[0]
        .parse()
        .map_err(|e| MatrixMarketError::Parse(format!("invalid row count: {e}")))?;
    let cols: usize = dimensions[1]
        .parse()
        .map_err(|e| MatrixMarketError::Parse(format!("invalid column count: {e}")))?;
    let declared_entries: usize = dimensions[2]
        .parse()
        .map_err(|e| MatrixMarketError::Parse(format!("invalid entry count: {e}")))?;
    if rows == 0 || cols == 0 {
        return Err(MatrixMarketError::Parse(
            "dimensions must be non-zero".into(),
        ));
    }
    if !matches!(symmetry, Symmetry::General) && rows != cols {
        return Err(MatrixMarketError::Parse(
            "symmetric and Hermitian matrices must be square".into(),
        ));
    }
    let capacity = if matches!(symmetry, Symmetry::General) {
        declared_entries
    } else {
        declared_entries.saturating_mul(2)
    };
    let mut triplets = Vec::with_capacity(capacity);
    for _ in 0..declared_entries {
        let line = next_data_line(&mut reader)?;
        let fields: Vec<_> = line.split_whitespace().collect();
        let required = if matches!(value_kind, ValueKind::Pattern) {
            2
        } else {
            3
        };
        if fields.len() != required {
            return Err(MatrixMarketError::Parse(format!(
                "expected {required} fields, got {}",
                fields.len()
            )));
        }
        let row: usize = fields[0]
            .parse()
            .map_err(|e| MatrixMarketError::Parse(format!("invalid row index: {e}")))?;
        let col: usize = fields[1]
            .parse()
            .map_err(|e| MatrixMarketError::Parse(format!("invalid column index: {e}")))?;
        if row == 0 || row > rows || col == 0 || col > cols {
            return Err(MatrixMarketError::Parse(format!(
                "entry ({row}, {col}) outside {rows}x{cols} matrix"
            )));
        }
        let value = if matches!(value_kind, ValueKind::Pattern) {
            1.0
        } else {
            fields[2]
                .parse::<f64>()
                .map_err(|e| MatrixMarketError::Parse(format!("invalid value: {e}")))?
        };
        if !value.is_finite() {
            return Err(MatrixMarketError::Parse("values must be finite".into()));
        }
        triplets.push((row - 1, col - 1, value));
        if !matches!(symmetry, Symmetry::General) && row != col {
            triplets.push((col - 1, row - 1, value));
        }
    }
    let matrix = CsrMatrix::try_from_triplets(rows, cols, triplets)
        .map_err(|error| MatrixMarketError::Parse(error.to_string()))?;
    let stored_nonzeros = matrix.values.len();
    Ok(ParsedMatrixMarket {
        matrix,
        declared_entries,
        stored_nonzeros,
    })
}

/// Parse a standard complex Matrix Market coordinate stream.
///
/// Complex entries use four fields (`row column real imaginary`). Real,
/// integer, and pattern banners are accepted as lossless complex values as
/// well. Symmetric entries are mirrored unchanged; Hermitian entries are
/// mirrored with complex conjugation. Historical size-only streams are not
/// accepted because their scalar field is ambiguous.
pub fn parse_complex_matrix_market<R: BufRead>(
    mut reader: R,
) -> Result<ParsedComplexMatrixMarket, MatrixMarketError> {
    let first = next_data_line(&mut reader)?;
    if !first.starts_with("%%MatrixMarket") {
        return Err(MatrixMarketError::Parse(
            "complex Matrix Market input requires a standard banner".into(),
        ));
    }
    let banner: Vec<_> = first.split_whitespace().collect();
    if banner.len() != 5
        || !banner[1].eq_ignore_ascii_case("matrix")
        || !banner[2].eq_ignore_ascii_case("coordinate")
    {
        return Err(MatrixMarketError::Parse(format!(
            "unsupported banner: {first}"
        )));
    }
    let value_kind = match banner[3].to_ascii_lowercase().as_str() {
        "real" => ValueKind::Real,
        "integer" => ValueKind::Integer,
        "pattern" => ValueKind::Pattern,
        "complex" => ValueKind::Complex,
        other => {
            return Err(MatrixMarketError::Parse(format!(
                "unsupported value kind: {other}"
            )));
        }
    };
    let symmetry = match banner[4].to_ascii_lowercase().as_str() {
        "general" => Symmetry::General,
        "symmetric" if !matches!(value_kind, ValueKind::Pattern) => Symmetry::Symmetric,
        "hermitian" if !matches!(value_kind, ValueKind::Pattern) => Symmetry::Hermitian,
        other => {
            return Err(MatrixMarketError::Parse(format!(
                "unsupported symmetry: {other}"
            )));
        }
    };
    let dimensions_line = next_data_line(&mut reader)?;
    let dimensions: Vec<_> = dimensions_line.split_whitespace().collect();
    if dimensions.len() != 3 {
        return Err(MatrixMarketError::Parse(format!(
            "expected three dimensions, got: {dimensions_line}"
        )));
    }
    let rows: usize = dimensions[0]
        .parse()
        .map_err(|e| MatrixMarketError::Parse(format!("invalid row count: {e}")))?;
    let cols: usize = dimensions[1]
        .parse()
        .map_err(|e| MatrixMarketError::Parse(format!("invalid column count: {e}")))?;
    let declared_entries: usize = dimensions[2]
        .parse()
        .map_err(|e| MatrixMarketError::Parse(format!("invalid entry count: {e}")))?;
    if rows == 0 || cols == 0 {
        return Err(MatrixMarketError::Parse(
            "dimensions must be non-zero".into(),
        ));
    }
    if !matches!(symmetry, Symmetry::General) && rows != cols {
        return Err(MatrixMarketError::Parse(
            "symmetric and Hermitian matrices must be square".into(),
        ));
    }
    let capacity = if matches!(symmetry, Symmetry::General) {
        declared_entries
    } else {
        declared_entries.saturating_mul(2)
    };
    let mut triplets = Vec::with_capacity(capacity);
    for _ in 0..declared_entries {
        let line = next_data_line(&mut reader)?;
        let fields: Vec<_> = line.split_whitespace().collect();
        let required = match value_kind {
            ValueKind::Pattern => 2,
            ValueKind::Complex => 4,
            ValueKind::Real | ValueKind::Integer => 3,
        };
        if fields.len() != required {
            return Err(MatrixMarketError::Parse(format!(
                "expected {required} fields, got {}",
                fields.len()
            )));
        }
        let row: usize = fields[0]
            .parse()
            .map_err(|e| MatrixMarketError::Parse(format!("invalid row index: {e}")))?;
        let col: usize = fields[1]
            .parse()
            .map_err(|e| MatrixMarketError::Parse(format!("invalid column index: {e}")))?;
        if row == 0 || row > rows || col == 0 || col > cols {
            return Err(MatrixMarketError::Parse(format!(
                "entry ({row}, {col}) outside {rows}x{cols} matrix"
            )));
        }
        let value = match value_kind {
            ValueKind::Pattern => Complex64::new(1.0, 0.0),
            ValueKind::Real | ValueKind::Integer => Complex64::new(
                fields[2]
                    .parse::<f64>()
                    .map_err(|e| MatrixMarketError::Parse(format!("invalid value: {e}")))?,
                0.0,
            ),
            ValueKind::Complex => Complex64::new(
                fields[2]
                    .parse::<f64>()
                    .map_err(|e| MatrixMarketError::Parse(format!("invalid real value: {e}")))?,
                fields[3].parse::<f64>().map_err(|e| {
                    MatrixMarketError::Parse(format!("invalid imaginary value: {e}"))
                })?,
            ),
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MatrixMarketError::Parse("values must be finite".into()));
        }
        triplets.push((row - 1, col - 1, value));
        if !matches!(symmetry, Symmetry::General) && row != col {
            let mirrored = if matches!(symmetry, Symmetry::Hermitian) {
                value.conj()
            } else {
                value
            };
            triplets.push((col - 1, row - 1, mirrored));
        }
    }
    let matrix = CsrMatrix::try_from_triplets(rows, cols, triplets)
        .map_err(|error| MatrixMarketError::Parse(error.to_string()))?;
    let stored_nonzeros = matrix.values.len();
    Ok(ParsedComplexMatrixMarket {
        matrix,
        declared_entries,
        stored_nonzeros,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_general_real_and_size_only_forms() {
        let standard = b"%%MatrixMarket matrix coordinate real general\n% comment\n2 2 3\n1 1 2\n1 2 -1\n2 2 3\n";
        let parsed = parse_matrix_market(Cursor::new(standard)).expect("standard parse");
        assert_eq!(parsed.declared_entries, 3);
        assert_eq!(parsed.stored_nonzeros, 3);
        assert_eq!(parsed.matrix.matvec(&[1.0, 2.0]), vec![0.0, 6.0]);

        let size_only = b"2 2 1\n1 2 4\n";
        let parsed = parse_matrix_market(Cursor::new(size_only)).expect("size-only parse");
        assert_eq!(parsed.matrix.values.len(), 1);
    }

    #[test]
    fn expands_symmetric_and_pattern_entries() {
        let symmetric = b"%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n1 1 2\n1 2 -1\n";
        let parsed = parse_matrix_market(Cursor::new(symmetric)).expect("symmetric parse");
        assert_eq!(parsed.stored_nonzeros, 3);
        assert_eq!(parsed.matrix.matvec(&[1.0, 2.0]), vec![0.0, -1.0]);

        let pattern = b"%%MatrixMarket matrix coordinate pattern general\n2 2 2\n1 2\n2 1\n";
        let parsed = parse_matrix_market(Cursor::new(pattern)).expect("pattern parse");
        assert_eq!(parsed.matrix.matvec(&[2.0, 3.0]), vec![3.0, 2.0]);
    }

    #[test]
    fn rejects_malformed_count_bounds_and_non_finite_values() {
        for source in [
            b"%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n".as_slice(),
            b"%%MatrixMarket matrix coordinate real general\n2 2 1\n0 1 1\n".as_slice(),
            b"%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 NaN\n".as_slice(),
            b"%%MatrixMarket matrix array real general\n2 2 1\n1 1 1\n".as_slice(),
            b"%%MatrixMarket matrix coordinate real symmetric\n2 3 1\n1 1 1\n".as_slice(),
        ] {
            assert!(parse_matrix_market(Cursor::new(source)).is_err());
        }
    }

    #[test]
    fn parses_complex_general_and_hermitian_entries() {
        let general = b"%%MatrixMarket matrix coordinate complex general\n2 2 3\n1 1 1 2\n1 2 3 -4\n2 2 5 0\n";
        let parsed = parse_complex_matrix_market(Cursor::new(general)).expect("complex parse");
        assert_eq!(parsed.declared_entries, 3);
        assert_eq!(parsed.stored_nonzeros, 3);
        assert_eq!(
            parsed
                .matrix
                .matvec(&[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)]),
            vec![Complex64::new(5.0, -5.0), Complex64::new(10.0, 0.0)]
        );

        let hermitian = b"%%MatrixMarket matrix coordinate complex hermitian\n2 2 3\n1 1 2 0\n1 2 1 2\n2 2 3 0\n";
        let parsed = parse_complex_matrix_market(Cursor::new(hermitian)).expect("Hermitian parse");
        assert_eq!(parsed.stored_nonzeros, 4);
        assert_eq!(
            parsed
                .matrix
                .matvec(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]),
            vec![Complex64::new(0.0, 1.0), Complex64::new(1.0, 1.0)]
        );
    }

    #[test]
    fn rejects_complex_malformed_and_ambiguous_inputs() {
        for source in [
            b"%%MatrixMarket matrix coordinate complex general\n2 2 1\n1 1 1\n".as_slice(),
            b"%%MatrixMarket matrix coordinate complex general\n2 2 1\n1 1 NaN 0\n".as_slice(),
            b"2 2 1\n1 1 1 0\n".as_slice(),
            b"%%MatrixMarket matrix coordinate complex hermitian\n2 3 1\n1 1 1 0\n".as_slice(),
        ] {
            assert!(parse_complex_matrix_market(Cursor::new(source)).is_err());
        }
        let real_complex_banner =
            b"%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1 0\n";
        assert!(parse_matrix_market(Cursor::new(real_complex_banner)).is_err());
    }
}
