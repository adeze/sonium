//! Levi13 test function

use crate::test_functions::functions::levy_n13::levy_n13;

/// Lévi N.13 function (alias for levy_n13 for compatibility)
pub fn levi13(x: &[f64]) -> f64 {
    levy_n13(x)
}
