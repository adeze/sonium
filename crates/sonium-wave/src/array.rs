//! Array facade for WaveSim internals.
//!
//! Re-exports the dense host containers from `sonium-compute` and adds
//! small helpers for axis-wise iteration / slabs that the wavesim modules
//! rely on. `Array1<T>` is just a `Vec<T>`; we expose constructors that
//! match the patterns used elsewhere in the crate.

use num_traits::Zero;
pub use sonium_compute::host_array::{HostMatrix as Array2, HostTensor3 as Array3};

/// 1D contiguous container — a plain `Vec<T>` under the hood.
pub type Array1<T> = Vec<T>;

/// Bounds/axis failures returned by checked array helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayBoundsError {
    InvalidAxis {
        axis: usize,
    },
    PositionOutOfBounds {
        axis: usize,
        position: usize,
        shape: (usize, usize, usize),
    },
    CropOutOfBounds {
        lower: (usize, usize, usize),
        extent: (usize, usize, usize),
        shape: (usize, usize, usize),
    },
}

impl std::fmt::Display for ArrayBoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAxis { axis } => write!(f, "Array3 axis {axis} is outside 0..=2"),
            Self::PositionOutOfBounds {
                axis,
                position,
                shape,
            } => write!(
                f,
                "Array3 position {position} is outside axis {axis} of shape {shape:?}"
            ),
            Self::CropOutOfBounds {
                lower,
                extent,
                shape,
            } => write!(
                f,
                "Array3 crop lower={lower:?} extent={extent:?} exceeds shape {shape:?}"
            ),
        }
    }
}

impl std::error::Error for ArrayBoundsError {}

/// Construct a length-`n` `Array1` filled with `T::zero()`.
#[inline]
pub fn array1_zeros<T: Clone + Zero>(n: usize) -> Array1<T> {
    vec![T::zero(); n]
}

/// Construct a length-`n` `Array1` filled with `value`.
#[inline]
pub fn array1_from_elem<T: Clone>(n: usize, value: T) -> Array1<T> {
    vec![value; n]
}

/// Linear row-major offset for a `(nx, ny, nz)`-shaped tensor.
#[inline]
pub fn lin_idx(shape: (usize, usize, usize), i: usize, j: usize, k: usize) -> usize {
    debug_assert!(i < shape.0 && j < shape.1 && k < shape.2);
    (i * shape.1 + j) * shape.2 + k
}

/// Checked counterpart to [`lin_idx`] for user-provided coordinates.
pub fn try_lin_idx(
    shape: (usize, usize, usize),
    i: usize,
    j: usize,
    k: usize,
) -> Result<usize, ArrayBoundsError> {
    if i >= shape.0 {
        return Err(ArrayBoundsError::PositionOutOfBounds {
            axis: 0,
            position: i,
            shape,
        });
    }
    if j >= shape.1 {
        return Err(ArrayBoundsError::PositionOutOfBounds {
            axis: 1,
            position: j,
            shape,
        });
    }
    if k >= shape.2 {
        return Err(ArrayBoundsError::PositionOutOfBounds {
            axis: 2,
            position: k,
            shape,
        });
    }
    Ok(lin_idx(shape, i, j, k))
}

/// Apply `op` to every cell of `tensor` whose `axis` coordinate equals `pos`.
///
/// Replaces ndarray's `slice_mut(s![pos, .., ..])` / `s![.., pos, ..]` /
/// `s![.., .., pos]` patterns used by the absorbing-boundary lerp.
///
/// # Panics
///
/// Panics on an out-of-bounds slab. Prefer [`try_for_each_in_slab`] where
/// the index is user-controlled.
pub fn for_each_in_slab<T, F>(tensor: &mut Array3<T>, axis: usize, pos: usize, op: F)
where
    F: FnMut(&mut T),
{
    try_for_each_in_slab(tensor, axis, pos, op).expect("Array3 slab index out of bounds");
}

/// Checked counterpart to [`for_each_in_slab`].
pub fn try_for_each_in_slab<T, F>(
    tensor: &mut Array3<T>,
    axis: usize,
    pos: usize,
    mut op: F,
) -> Result<(), ArrayBoundsError>
where
    F: FnMut(&mut T),
{
    let (nx, ny, nz) = tensor.shape();
    let shape = (nx, ny, nz);
    let axis_len = match axis {
        0 => nx,
        1 => ny,
        2 => nz,
        _ => return Err(ArrayBoundsError::InvalidAxis { axis }),
    };
    if pos >= axis_len {
        return Err(ArrayBoundsError::PositionOutOfBounds {
            axis,
            position: pos,
            shape,
        });
    }
    let data = tensor.as_mut_slice();
    match axis {
        0 => {
            let base = pos * ny * nz;
            for v in &mut data[base..base + ny * nz] {
                op(v);
            }
        }
        1 => {
            for i in 0..nx {
                let base = (i * ny + pos) * nz;
                for v in &mut data[base..base + nz] {
                    op(v);
                }
            }
        }
        2 => {
            for i in 0..nx {
                for j in 0..ny {
                    let off = (i * ny + j) * nz + pos;
                    op(&mut data[off]);
                }
            }
        }
        _ => unreachable!("axis checked above"),
    }
    Ok(())
}

/// Visit every fibre along `axis`, providing a `&mut [T]` buffer for the
/// caller to copy into / process. The buffer length equals the axis length.
///
/// Used by the in-place 3D FFT to apply a 1D plan along each axis.
///
/// # Panics
///
/// Panics on an invalid lane axis. Prefer [`try_for_each_lane_mut`] where
/// the axis is user-controlled.
pub fn for_each_lane_mut<T, F>(tensor: &mut Array3<T>, axis: usize, visit: F)
where
    T: Clone,
    F: FnMut(&mut [T]),
{
    try_for_each_lane_mut(tensor, axis, visit).expect("Array3 lane axis out of bounds");
}

/// Checked counterpart to [`for_each_lane_mut`].
pub fn try_for_each_lane_mut<T, F>(
    tensor: &mut Array3<T>,
    axis: usize,
    mut visit: F,
) -> Result<(), ArrayBoundsError>
where
    T: Clone,
    F: FnMut(&mut [T]),
{
    let (nx, ny, nz) = tensor.shape();
    if axis > 2 {
        return Err(ArrayBoundsError::InvalidAxis { axis });
    }
    let data = tensor.as_mut_slice();
    match axis {
        0 => {
            let len = nx;
            let mut buf: Vec<T> = Vec::with_capacity(len);
            for j in 0..ny {
                for k in 0..nz {
                    buf.clear();
                    for i in 0..nx {
                        buf.push(data[(i * ny + j) * nz + k].clone());
                    }
                    visit(&mut buf);
                    for (i, v) in buf.iter().enumerate() {
                        data[(i * ny + j) * nz + k] = v.clone();
                    }
                }
            }
        }
        1 => {
            let len = ny;
            let mut buf: Vec<T> = Vec::with_capacity(len);
            for i in 0..nx {
                for k in 0..nz {
                    buf.clear();
                    for j in 0..ny {
                        buf.push(data[(i * ny + j) * nz + k].clone());
                    }
                    visit(&mut buf);
                    for (j, v) in buf.iter().enumerate() {
                        data[(i * ny + j) * nz + k] = v.clone();
                    }
                }
            }
        }
        2 => {
            // Contiguous along axis 2 — no copy required.
            for i in 0..nx {
                for j in 0..ny {
                    let base = (i * ny + j) * nz;
                    visit(&mut data[base..base + nz]);
                }
            }
        }
        _ => unreachable!("axis checked above"),
    }
    Ok(())
}

/// Crop `tensor` to `[lo.0..lo.0+ext.0, lo.1..lo.1+ext.1, lo.2..lo.2+ext.2]`.
///
/// # Panics
///
/// Panics on an out-of-bounds crop. Prefer [`try_crop3`] where the bounds
/// are user-controlled.
pub fn crop3<T: Clone>(
    tensor: &Array3<T>,
    lo: (usize, usize, usize),
    ext: (usize, usize, usize),
) -> Array3<T> {
    try_crop3(tensor, lo, ext).expect("Array3 crop out of bounds")
}

/// Checked counterpart to [`crop3`].
pub fn try_crop3<T: Clone>(
    tensor: &Array3<T>,
    lo: (usize, usize, usize),
    ext: (usize, usize, usize),
) -> Result<Array3<T>, ArrayBoundsError> {
    let (nx_in, ny_in, nz_in) = tensor.shape();
    let shape = (nx_in, ny_in, nz_in);
    if lo.0 > nx_in
        || lo.1 > ny_in
        || lo.2 > nz_in
        || ext.0 > nx_in - lo.0
        || ext.1 > ny_in - lo.1
        || ext.2 > nz_in - lo.2
    {
        return Err(ArrayBoundsError::CropOutOfBounds {
            lower: lo,
            extent: ext,
            shape,
        });
    }
    let src = tensor.as_slice();
    let mut out: Vec<T> = Vec::with_capacity(ext.0 * ext.1 * ext.2);
    for i in 0..ext.0 {
        for j in 0..ext.1 {
            let base = ((lo.0 + i) * ny_in + (lo.1 + j)) * nz_in + lo.2;
            out.extend_from_slice(&src[base..base + ext.2]);
        }
    }
    Ok(Array3::from_row_major_vec(ext, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_each_in_slab_axes() {
        let mut t = Array3::<i32>::from_fn((2, 3, 4), |i, j, k| (i * 100 + j * 10 + k) as i32);
        for_each_in_slab(&mut t, 0, 1, |v| *v += 1000);
        assert_eq!(t[(1, 0, 0)], 100 + 1000);
        assert_eq!(t[(0, 0, 0)], 0);

        for_each_in_slab(&mut t, 1, 2, |v| *v = -1);
        for i in 0..2 {
            for k in 0..4 {
                assert_eq!(t[(i, 2, k)], -1);
            }
        }
    }

    #[test]
    fn for_each_lane_mut_axes() {
        let mut t = Array3::<i32>::from_fn((2, 3, 4), |i, j, k| (i * 100 + j * 10 + k) as i32);
        for axis in 0..3 {
            let expect_len = t.shape();
            let want = match axis {
                0 => expect_len.0,
                1 => expect_len.1,
                _ => expect_len.2,
            };
            for_each_lane_mut(&mut t, axis, |lane| {
                assert_eq!(lane.len(), want);
                for v in lane.iter_mut() {
                    *v *= 2;
                }
            });
        }
    }

    #[test]
    fn crop3_basic() {
        let t = Array3::<i32>::from_fn((4, 4, 4), |i, j, k| (i * 100 + j * 10 + k) as i32);
        let c = crop3(&t, (1, 1, 1), (2, 2, 2));
        assert_eq!(c.shape(), (2, 2, 2));
        assert_eq!(c[(0, 0, 0)], 111);
        assert_eq!(c[(1, 1, 1)], 222);
    }

    #[test]
    fn checked_array_helpers_report_bounds_without_panicking() {
        let shape = (2, 3, 4);
        assert_eq!(try_lin_idx(shape, 1, 2, 3), Ok(23));
        assert!(matches!(
            try_lin_idx(shape, 2, 0, 0),
            Err(ArrayBoundsError::PositionOutOfBounds { axis: 0, .. })
        ));

        let mut tensor = Array3::<i32>::from_fn(shape, |i, j, k| (i + j + k) as i32);
        assert!(matches!(
            try_for_each_in_slab(&mut tensor, 3, 0, |_| {}),
            Err(ArrayBoundsError::InvalidAxis { axis: 3 })
        ));
        assert!(matches!(
            try_for_each_in_slab(&mut tensor, 1, 3, |_| {}),
            Err(ArrayBoundsError::PositionOutOfBounds { axis: 1, .. })
        ));
        assert!(matches!(
            try_for_each_lane_mut(&mut tensor, 3, |_| {}),
            Err(ArrayBoundsError::InvalidAxis { axis: 3 })
        ));
        assert!(matches!(
            try_crop3(&tensor, (1, 2, 3), (2, 1, 1)),
            Err(ArrayBoundsError::CropOutOfBounds { .. })
        ));
        assert!(matches!(
            try_crop3(&tensor, (0, 0, 0), (3, 1, 1)),
            Err(ArrayBoundsError::CropOutOfBounds { .. })
        ));
    }
}
