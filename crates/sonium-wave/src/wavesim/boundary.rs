//! Absorbing boundary: pad permittivity with a linear ramp on `Im(ε)`.
//!
//! Matches `wavesim_py.utilities.add_absorbing_boundaries` exactly: edge values
//! are repeated, then linearly interpolated towards `1j · strength` along each
//! padded axis with profile `(arange(1, w+1) - 0.21) / (w + 0.66)`.

use num_complex::Complex64;

use crate::array::{Array3, for_each_in_slab};

const ABSORBER_STRENGTH: f64 = 1.0;

/// Pad `permittivity` with absorbing boundaries; returns the padded array.
///
/// `pad[d] = (lo, hi)` controls each side. Periodic axes use `(0, 0)`.
pub fn add_absorbing_boundary(
    permittivity: &Array3<Complex64>,
    pad: [(usize, usize); 3],
) -> Array3<Complex64> {
    let (nx, ny, nz) = permittivity.shape();
    let new_shape = (
        nx + pad[0].0 + pad[0].1,
        ny + pad[1].0 + pad[1].1,
        nz + pad[2].0 + pad[2].1,
    );
    let mut out = Array3::<Complex64>::zeros(new_shape);

    // Edge-extend.
    for i in 0..new_shape.0 {
        for j in 0..new_shape.1 {
            for k in 0..new_shape.2 {
                let si = nearest(i, pad[0].0, nx);
                let sj = nearest(j, pad[1].0, ny);
                let sk = nearest(k, pad[2].0, nz);
                out[[i, j, k]] = permittivity[[si, sj, sk]];
            }
        }
    }

    // Apply linear-ramp absorber to each padded edge slab independently.
    let target = Complex64::new(0.0, ABSORBER_STRENGTH);
    let axis_dims = [new_shape.0, new_shape.1, new_shape.2];
    for axis in 0..3 {
        let (lo, hi) = pad[axis];
        // Low side
        if lo > 0 {
            for r in 0..lo {
                let weight = profile_weight(lo - r, lo);
                lerp_slab(&mut out, axis, r, weight, target);
            }
        }
        // High side
        if hi > 0 {
            for r in 0..hi {
                let weight = profile_weight(r + 1, hi);
                let pos = axis_dims[axis] - hi + r;
                lerp_slab(&mut out, axis, pos, weight, target);
            }
        }
    }

    out
}

#[inline]
fn nearest(idx: usize, lo: usize, n: usize) -> usize {
    if idx < lo {
        0
    } else if idx >= lo + n {
        n.saturating_sub(1)
    } else {
        idx - lo
    }
}

/// Weight factor in `lerp(orig, target, weight)` for a cell `pos_in_pad` cells
/// from the physical edge in a pad of width `width`. Matches the wavesim_py
/// formula `(arange(1, w+1) - 0.21) / (w + 0.66)`.
fn profile_weight(pos_in_pad: usize, width: usize) -> f64 {
    (pos_in_pad as f64 - 0.21) / (width as f64 + 0.66)
}

/// `out[axis_index = pos] = (1-w)·out[axis_index = pos] + w·target`.
fn lerp_slab(out: &mut Array3<Complex64>, axis: usize, pos: usize, weight: f64, target: Complex64) {
    for_each_in_slab(out, axis, pos, |v| {
        *v = *v * (1.0 - weight) + target * weight;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_preserves_physical_region() {
        let eps = Array3::<Complex64>::from_elem((4, 1, 1), Complex64::new(1.0, 0.0));
        let padded = add_absorbing_boundary(&eps, [(2, 3), (0, 0), (0, 0)]);
        assert_eq!(padded.shape(), (9, 1, 1));
        for i in 2..6 {
            assert_eq!(padded[[i, 0, 0]], Complex64::new(1.0, 0.0));
        }
    }

    #[test]
    fn linear_ramp_matches_wavesim_py_formula() {
        // width=4, pos_in_pad in {1,2,3,4}: profile = (n-0.21)/(4.66)
        let eps = Array3::<Complex64>::from_elem((4, 1, 1), Complex64::new(1.0, 0.0));
        let padded = add_absorbing_boundary(&eps, [(0, 4), (0, 0), (0, 0)]);
        for r in 0..4 {
            let weight = (r as f64 + 1.0 - 0.21) / 4.66;
            let expected =
                Complex64::new(1.0, 0.0) * (1.0 - weight) + Complex64::new(0.0, 1.0) * weight;
            let got = padded[[4 + r, 0, 0]];
            assert!(
                (got - expected).norm() < 1e-12,
                "r={r} got={got} exp={expected}"
            );
        }
    }
}
