//! Adapter from the host-side [`LinearOperator`] trait to the
//! backend-side [`BackendOperator`] trait, for `CpuBackend`.
//!
//! The CPU backend's `Buffer<T>` is just a `Vec<T>` wrapper; the bridge
//! materialises an `Vec<T>` view from the buffer's slice, invokes the
//! host operator's `apply`, and writes the result back. This is the
//! glue that lets the BEM `LinearOperator` family ([`DenseOperator`],
//! `SlfmmOperator`, `MlfmmOperator`) drive the new
//! [`gmres_ir_backend_preconditioned`](sonium_compute::gmres_ir::gmres_ir_backend_preconditioned)
//! pipeline.
//!
//! # Cost
//!
//! Each `apply` allocates two `Vec<T>` temporaries (one for the
//! input copy, one for the output) at host precision. For BEM at 10M
//! unknowns × Complex64 that's 160 MB of churn per matvec. Tolerable
//! for correctness validation; for production the bridge should be
//! supplemented with a `LinearOperator::apply_into_slice` API to elide
//! the Array1 round-trip. That's a follow-up.
//!
//! # Why CpuBackend only
//!
//! The host `LinearOperator::apply` takes `&[T]` — a CPU-resident
//! ndarray. Forwarding from a `MetalBuffer` or `CudaBuffer` would
//! require an extra copy_d2h / copy_h2d cycle, which defeats the
//! GPU-offload point. For Metal/cudarc, the right path is a native
//! device-side operator (e.g. an FMM tree built directly on the
//! device), not this adapter.

use std::marker::PhantomData;

use sonium_compute::cpu::{CpuBackend, HostBuffer};
use sonium_compute::gmres::BackendOperator;
use sonium_compute::scalar::{BufferElement, DeviceScalar};

use sonium_solvers_core::traits::{ComplexField, LinearOperator};

/// Wraps a host-side [`LinearOperator<T>`] as a backend-side
/// [`BackendOperator<CpuBackend, T>`].
///
/// `T` must satisfy both [`DeviceScalar`] (for the backend interface)
/// and [`ComplexField`] (for the host operator). All four scalar types
/// (`f32`, `f64`, `Complex32`, `Complex64`) implement both.
pub struct HostLinearOperator<O, T>
where
    O: LinearOperator<T> + Send + Sync,
    T: ComplexField + DeviceScalar + BufferElement,
{
    op: O,
    _phantom: PhantomData<T>,
}

impl<O, T> HostLinearOperator<O, T>
where
    O: LinearOperator<T> + Send + Sync,
    T: ComplexField + DeviceScalar + BufferElement,
{
    pub fn new(op: O) -> Self {
        Self {
            op,
            _phantom: PhantomData,
        }
    }

    /// Borrow the wrapped host operator (e.g. for direct host-side use).
    pub fn inner(&self) -> &O {
        &self.op
    }
}

impl<O, T> BackendOperator<CpuBackend, T> for HostLinearOperator<O, T>
where
    O: LinearOperator<T> + Send + Sync,
    T: ComplexField + DeviceScalar + BufferElement,
{
    fn n(&self) -> usize {
        self.op.num_rows()
    }

    fn apply(&self, _backend: &CpuBackend, x: &HostBuffer<T>, y: &mut HostBuffer<T>) {
        let mut y_arr = vec![<T as num_traits::Zero>::zero(); self.op.num_rows()];
        self.op.apply_into(x.as_slice(), &mut y_arr);
        y.as_mut_slice().copy_from_slice(&y_arr);
    }
}

#[cfg(test)]
// Fixtures intentionally use the panicking convenience constructors on tiny
// fixed matrices; production paths must use the `try_*` variants.
#[allow(deprecated)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_compute::Backend;
    use sonium_compute::gmres::{
        BackendCsr, GmresBackendConfig, IdentityBackendPreconditioner, gmres_backend_preconditioned,
    };
    use sonium_solvers_sparse::CsrMatrix;

    /// Minimal `LinearOperator<Complex64>` impl backed by a CSR matrix
    /// — exactly the shape sonium-bem's `DenseOperator` and FMM
    /// operators present (apply only).
    struct SimpleCsrOperator {
        matrix: CsrMatrix<Complex64>,
    }

    impl LinearOperator<Complex64> for SimpleCsrOperator {
        fn num_rows(&self) -> usize {
            self.matrix.num_rows()
        }
        fn num_cols(&self) -> usize {
            self.matrix.num_cols()
        }
        fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
            self.matrix.matvec(x)
        }
        fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y.copy_from_slice(&self.matrix.matvec(x));
        }
        fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
            self.matrix.matvec_transpose(x)
        }
        fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y.copy_from_slice(&self.matrix.matvec_transpose(x));
        }
    }

    #[test]
    fn host_adapter_apply_matches_direct_csr() {
        // Build a small complex tridiagonal as both a host LinearOperator
        // (via SimpleCsrOperator) and a BackendCsr. A matvec via the
        // adapter must agree with the BackendCsr matvec to floating-
        // point precision — that's the round-trip correctness check.
        let backend = CpuBackend::new();
        let n = 32;
        let mut row_ptr = vec![0_i32; n + 1];
        let mut col_idx: Vec<i32> = Vec::new();
        let mut values: Vec<Complex64> = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push(i as i32 - 1);
                values.push(Complex64::new(-1.0, 0.0));
            }
            col_idx.push(i as i32);
            values.push(Complex64::new(2.0, 0.05));
            if i + 1 < n {
                col_idx.push(i as i32 + 1);
                values.push(Complex64::new(-1.0, 0.0));
            }
            row_ptr[i + 1] = values.len() as i32;
        }
        let bem_op = SimpleCsrOperator {
            matrix: CsrMatrix::from_raw_parts(
                n,
                n,
                row_ptr.iter().map(|&v| v as usize).collect(),
                col_idx.iter().map(|&v| v as usize).collect(),
                values.clone(),
            ),
        };
        let adapter = HostLinearOperator::new(bem_op);
        let backend_csr = BackendCsr::<CpuBackend, Complex64>::from_host(
            &backend, n, n, &row_ptr, &col_idx, &values,
        );

        let x_host: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(((i as f64) * 0.1).sin(), ((i as f64) * 0.07).cos()))
            .collect();
        let mut x_buf = backend.raw_alloc::<Complex64>(n).unwrap();
        backend.copy_h2d(&x_host, &mut x_buf);

        let mut y_adapter = backend.raw_alloc::<Complex64>(n).unwrap();
        let mut y_csr = backend.raw_alloc::<Complex64>(n).unwrap();
        adapter.apply(&backend, &x_buf, &mut y_adapter);
        backend_csr.apply(&backend, &x_buf, &mut y_csr);

        let mut a_host = vec![Complex64::new(0.0, 0.0); n];
        let mut c_host = vec![Complex64::new(0.0, 0.0); n];
        backend.copy_d2h(&y_adapter, &mut a_host);
        backend.copy_d2h(&y_csr, &mut c_host);
        for i in 0..n {
            assert!(
                (a_host[i] - c_host[i]).norm() < 1.0e-12,
                "row {i}: adapter {} vs csr {}",
                a_host[i],
                c_host[i]
            );
        }
    }

    /// End-to-end: drive `gmres_backend_preconditioned` with a host
    /// `LinearOperator<Complex64>` via the adapter — the IR pipeline
    /// can now consume any operator that implements `LinearOperator`,
    /// not just `BackendCsr`.
    #[test]
    fn gmres_via_host_adapter_converges() {
        let backend = CpuBackend::new();
        let n = 64;
        let mut row_ptr = vec![0_i32; n + 1];
        let mut col_idx: Vec<i32> = Vec::new();
        let mut values: Vec<Complex64> = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push(i as i32 - 1);
                values.push(Complex64::new(-1.0, 0.0));
            }
            col_idx.push(i as i32);
            values.push(Complex64::new(2.0, 0.05));
            if i + 1 < n {
                col_idx.push(i as i32 + 1);
                values.push(Complex64::new(-1.0, 0.0));
            }
            row_ptr[i + 1] = values.len() as i32;
        }
        let bem_op = SimpleCsrOperator {
            matrix: CsrMatrix::from_raw_parts(
                n,
                n,
                row_ptr.iter().map(|&v| v as usize).collect(),
                col_idx.iter().map(|&v| v as usize).collect(),
                values.clone(),
            ),
        };
        let adapter = HostLinearOperator::new(bem_op);

        let b_host: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(((i as f64) + 1.0) * 0.137, 0.0))
            .collect();
        let mut b_buf = backend.raw_alloc::<Complex64>(n).unwrap();
        backend.copy_h2d(&b_host, &mut b_buf);
        let mut out = backend.raw_alloc::<Complex64>(n).unwrap();

        let cfg = GmresBackendConfig::<Complex64> {
            max_iterations: 200,
            tolerance: 1.0e-10,
        };
        let s = gmres_backend_preconditioned::<CpuBackend, _, _, Complex64>(
            &backend,
            &adapter,
            &IdentityBackendPreconditioner,
            &b_buf,
            None,
            &mut out,
            &cfg,
        )
        .expect("solve");
        assert!(
            s.converged,
            "GMRES via host adapter must converge; final={} after {} iters",
            s.final_residual, s.iterations
        );
    }
}
