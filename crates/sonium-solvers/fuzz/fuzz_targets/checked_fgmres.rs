#![no_main]

use libfuzzer_sys::fuzz_target;
use sonium_solvers::sparse::FnOperator;
use sonium_solvers::{try_fgmres, FgmresConfig};

fuzz_target!(|bytes: &[u8]| {
    let n = 1 + bytes.first().copied().unwrap_or_default() as usize % 8;
    let mut diagonal = vec![1.0; n];
    let mut off_diagonal = vec![0.0; n.saturating_sub(1)];
    for (index, value) in diagonal.iter_mut().enumerate() {
        *value += bytes.get(index + 1).copied().unwrap_or_default() as f64 / 255.0;
    }
    for (index, value) in off_diagonal.iter_mut().enumerate() {
        *value = 0.05 * bytes.get(index + n + 1).copied().unwrap_or_default() as f64 / 255.0;
    }

    let operator = FnOperator::new(n, n, move |x: &[f64]| {
        let mut y = vec![0.0; n];
        for row in 0..n {
            y[row] += diagonal[row] * x[row];
            if row > 0 {
                y[row] += off_diagonal[row - 1] * x[row - 1];
            }
            if row + 1 < n {
                y[row] += off_diagonal[row] * x[row + 1];
            }
        }
        y
    });
    let rhs = (0..n)
        .map(|index| bytes.get(index + 2 * n).copied().unwrap_or_default() as f64 / 127.5 - 1.0)
        .collect::<Vec<_>>();
    let config = FgmresConfig {
        max_iterations: 32,
        restart: n.min(8),
        tolerance: 1.0e-8,
        ..FgmresConfig::default()
    };

    // Errors are valid outcomes for malformed/degenerate generated inputs;
    // a panic is the defect this target is intended to expose.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = try_fgmres(&operator, &rhs, &config);
    }));
    assert!(result.is_ok(), "checked FGMRES panicked on fuzz input");
});
