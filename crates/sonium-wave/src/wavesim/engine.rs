//! FFT plan cache and 3D in-place FFT helpers used by [`super::HelmholtzDomain`].
//!
//! The wavesim_py engine layer exposes a `WaveArray` wrapper; in Rust we just
//! use a 3D complex array directly. The caching wrapper keeps one
//! forward and one inverse plan per axis length so the iteration loop never
//! reallocates twiddle factors.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use num_complex::Complex64;
use rayon::prelude::*;
use sonium_fft::rustfft_compat::{Fft, FftPlanner};

use crate::array::{Array3, for_each_lane_mut};
use crate::parallelism::{FFT_LANES_PER_THREAD, POINTWISE_CELLS_PER_THREAD, use_parallel};

/// Pointwise complex scale, serially or with rayon. Both paths apply
/// identical operations in identical order.
fn normalize(data: &mut [Complex64], scale: f64, parallel: bool) {
    if parallel {
        data.par_iter_mut().for_each(|v| *v /= scale);
    } else {
        for v in data.iter_mut() {
            *v /= scale;
        }
    }
}

/// Cache of forward / inverse FFT plans, keyed by transform length.
#[derive(Default)]
pub struct FftPlanCache {
    forward: Mutex<HashMap<usize, Arc<dyn Fft<f64>>>>,
    inverse: Mutex<HashMap<usize, Arc<dyn Fft<f64>>>>,
}

impl FftPlanCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn plan(&self, len: usize, inverse: bool) -> Arc<dyn Fft<f64>> {
        let map = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        let mut guard = map.lock().expect("fft plan cache poisoned");
        guard
            .entry(len)
            .or_insert_with(|| {
                let mut planner = FftPlanner::<f64>::new();
                if inverse {
                    planner.plan_fft_inverse(len)
                } else {
                    planner.plan_fft_forward(len)
                }
            })
            .clone()
    }

    /// In-place forward FFT along every axis (3D).
    pub fn fftn(&self, x: &mut Array3<Complex64>) {
        self.transform_along_all_axes(x, false);
    }

    /// In-place inverse FFT along every axis (3D), normalized by `1/N` total
    /// to match numpy/scipy convention.
    pub fn ifftn(&self, x: &mut Array3<Complex64>) {
        self.transform_along_all_axes(x, true);
        let n = x.len();
        if n > 0 {
            let parallel = use_parallel(n, POINTWISE_CELLS_PER_THREAD);
            normalize(x.as_mut_slice(), n as f64, parallel);
        }
    }

    /// Apply forward or inverse FFT along every axis, in place.
    ///
    /// Lanes are independent; the machine-adaptive gate parallelizes the
    /// lane loop when the lane count merits pool dispatch. Axis 2 is
    /// contiguous and transforms in place; axes 0/1 gather into a scratch
    /// buffer (serial streaming copies), transform lanes in parallel with
    /// one scratch allocation total, and scatter back.
    fn transform_along_all_axes(&self, x: &mut Array3<Complex64>, inverse: bool) {
        let shape = x.shape();
        let axis_lens = [shape.0, shape.1, shape.2];
        for (axis, &len) in axis_lens.iter().enumerate() {
            if len <= 1 {
                continue;
            }
            let plan = self.plan(len, inverse);
            let lanes = match axis {
                0 => shape.1 * shape.2,
                1 => shape.0 * shape.2,
                _ => shape.0 * shape.1,
            };
            transform_axis(
                x,
                axis,
                len,
                &plan,
                use_parallel(lanes, FFT_LANES_PER_THREAD),
            );
        }
    }
}

/// Transform one axis, serially or in parallel over independent lanes.
///
/// Both paths apply `plan` to identical lane contents in identical order;
/// the parallel path only changes which thread runs each lane.
fn transform_axis(
    x: &mut Array3<Complex64>,
    axis: usize,
    len: usize,
    plan: &Arc<dyn Fft<f64>>,
    parallel: bool,
) {
    if !parallel {
        for_each_lane_mut(x, axis, |lane| {
            plan.process(lane);
        });
        return;
    }
    if axis == 2 {
        x.as_mut_slice()
            .par_chunks_mut(len)
            .for_each(|lane| plan.process(lane));
        return;
    }
    let (nx, ny, nz) = x.shape();
    let data = x.as_mut_slice();
    let n_lanes = if axis == 0 { ny * nz } else { nx * nz };
    // Contiguous scratch: one extra grid transiently. Gather + transform are
    // fused per lane (each worker owns disjoint chunks); scatter is a serial
    // streaming copy because strided stores cannot be split borrow-safely.
    let mut scratch = vec![Complex64::new(0.0, 0.0); n_lanes * len];
    let shared: &[Complex64] = &data[..];
    scratch
        .par_chunks_mut(len)
        .enumerate()
        .for_each(|(lane, slot)| {
            for (p, target) in slot.iter_mut().enumerate() {
                *target = shared[lane_source(axis, lane, p, ny, nz)];
            }
            plan.process(slot);
        });
    for (lane, chunk) in scratch.chunks(len).enumerate() {
        for (p, value) in chunk.iter().enumerate() {
            data[lane_source(axis, lane, p, ny, nz)] = *value;
        }
    }
}

/// Flat index of lane element `p` for a strided (axis 0/1) lane.
///
/// Layout is `(i * ny + j) * nz + k`: axis-0 lanes fix `(j, k)`, axis-1
/// lanes fix `(i, k)`, both packed as `outer * nz + k`.
fn lane_source(axis: usize, lane: usize, p: usize, ny: usize, nz: usize) -> usize {
    let outer = lane / nz;
    let k = lane % nz;
    if axis == 0 {
        (p * ny + outer) * nz + k
    } else {
        debug_assert_eq!(axis, 1);
        (outer * ny + p) * nz + k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_roundtrip_3d() {
        let cache = FftPlanCache::new();
        let mut a = Array3::<Complex64>::from_fn((4, 5, 3), |i, j, k| {
            Complex64::new((i + 2 * j + 3 * k) as f64, (i as f64) * 0.5)
        });
        let original = a.clone();
        cache.fftn(&mut a);
        cache.ifftn(&mut a);
        for (got, want) in a.as_slice().iter().zip(original.as_slice().iter()) {
            assert!((got.re - want.re).abs() < 1e-10);
            assert!((got.im - want.im).abs() < 1e-10);
        }
    }

    #[test]
    fn fft_roundtrip_1d_singleton_axes() {
        let cache = FftPlanCache::new();
        let mut a = Array3::<Complex64>::from_fn((16, 1, 1), |i, _, _| {
            Complex64::new((i as f64).sin(), (i as f64 * 0.3).cos())
        });
        let original = a.clone();
        cache.fftn(&mut a);
        cache.ifftn(&mut a);
        for (got, want) in a.as_slice().iter().zip(original.as_slice().iter()) {
            assert!((got.re - want.re).abs() < 1e-10);
            assert!((got.im - want.im).abs() < 1e-10);
        }
    }

    fn lane_test_grid() -> Array3<Complex64> {
        Array3::<Complex64>::from_fn((9, 7, 8), |i, j, k| {
            Complex64::new(
                ((3 * i + 5 * j + 7 * k) as f64 * 0.37).sin(),
                ((i + 2 * j + 4 * k) as f64 * 0.53).cos(),
            )
        })
    }

    #[test]
    fn parallel_lanes_match_serial_bitwise() {
        // Each lane is an independent FFT of identical input, so serial and
        // parallel paths must agree bitwise on every axis (including the
        // strided gather/scatter path on axes 0/1).
        let cache = FftPlanCache::new();
        for axis in 0..3 {
            let len = [9, 7, 8][axis];
            for inverse in [false, true] {
                let plan = cache.plan(len, inverse);
                let mut serial = lane_test_grid();
                let mut parallel = serial.clone();
                transform_axis(&mut serial, axis, len, &plan, false);
                transform_axis(&mut parallel, axis, len, &plan, true);
                assert_eq!(
                    serial.as_slice(),
                    parallel.as_slice(),
                    "axis {axis} inverse={inverse} differs"
                );
            }
        }
    }

    #[test]
    fn normalize_matches_on_both_paths() {
        let mut serial = vec![Complex64::new(3.0, -4.0); 1024];
        let mut parallel = serial.clone();
        normalize(&mut serial, 2.5, false);
        normalize(&mut parallel, 2.5, true);
        assert_eq!(serial, parallel);
        assert!((serial[0].re - 1.2).abs() < 1e-15);
    }
}
