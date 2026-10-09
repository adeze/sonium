//! Sparse sources injected into the simulation grid.

use num_complex::Complex64;

use crate::array::Array3;

use super::HelmholtzDomain;

/// A single point/region source in physical-grid coordinates.
#[derive(Debug, Clone)]
pub struct Source {
    /// Physical-grid index `[i, j, k]` (before boundary padding).
    pub position: [usize; 3],
    /// Complex amplitude.
    pub value: Complex64,
}

/// Convenience constructor matching `wavesim_py.utilities.point_source`.
///
/// `position_um` is interpreted in the same length unit as `pixel_size`. Each
/// component is rounded to the nearest grid index.
pub fn point_source(position_um: [f64; 3], pixel_size: f64, value: Complex64) -> Source {
    let to_idx = |x: f64| (x / pixel_size).round().max(0.0) as usize;
    Source {
        position: [
            to_idx(position_um[0]),
            to_idx(position_um[1]),
            to_idx(position_um[2]),
        ],
        value,
    }
}

/// Build the padded source array used by the iteration loop.
pub(crate) fn assemble(
    domain: &HelmholtzDomain,
    sources: &[Source],
    pad: [(usize, usize); 3],
) -> Array3<Complex64> {
    let shape = domain.padded_shape();
    let mut out = Array3::<Complex64>::zeros(shape);
    for s in sources {
        let i = s.position[0] + pad[0].0;
        let j = s.position[1] + pad[1].0;
        let k = s.position[2] + pad[2].0;
        if i < shape.0 && j < shape.1 && k < shape.2 {
            out[[i, j, k]] += s.value;
        }
    }
    out
}
