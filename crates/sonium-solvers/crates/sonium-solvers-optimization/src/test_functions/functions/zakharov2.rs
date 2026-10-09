//! Zakharov2 test function

use crate::test_functions::functions::zakharov::zakharov;

/// Zakharov function variant (2D specific)
pub fn zakharov2(x: &[f64]) -> f64 {
    zakharov(x)
}
