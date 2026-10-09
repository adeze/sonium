//! Backend-aware Jacobi/Schur preconditioner for saddle-point systems.
//!
//! This mirrors the CPU upper-triangular saddle-point preconditioner with
//! diagonal block solves:
//!
//! ```text
//! y_p = S_diag^{-1} r_p
//! y_u = A_diag^{-1} (r_u - B y_p)
//! ```
//!
//! The block solves can use polynomial Jacobi/Richardson or Chebyshev
//! approximations.
//! Everything is embedded as full-size CSR matrices so `apply` stays entirely
//! in existing backend primitives: SpMV, scale, and axpy. No device slicing
//! or triangular solves are needed.

use crate::gmres::{BackendCsr, BackendPreconditioner};
use crate::operator::{DeviceCsr, PersistentDeviceCsr, validate_csr_inputs};
use crate::{
    Backend, DeviceBufferLike, DeviceScalar, HostCsrRef, PreconditionerFailureInjectionPoint,
};
use std::sync::Mutex;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PressureSchurBackendKind {
    #[default]
    Polynomial,
    Chebyshev,
    External,
}

/// Upper-triangular Jacobi/Schur preconditioner over backend buffers.
pub struct SaddlePointJacobiSchurBackend<'a, B: Backend, T: DeviceScalar = f64> {
    a: PersistentDeviceCsr<B, T>,
    velocity_block: PersistentDeviceCsr<B, T>,
    pressure_schur: PersistentDeviceCsr<B, T>,
    pressure_preconditioner: Option<&'a dyn BackendPreconditioner<B, f64>>,
    pressure_mask: PersistentDeviceCsr<B, T>,
    velocity_diag_inv: PersistentDeviceCsr<B, T>,
    pressure_diag_inv: PersistentDeviceCsr<B, T>,
    n: usize,
    velocity_degree: usize,
    velocity_damping: T,
    pressure_degree: usize,
    pressure_damping: T,
    pressure_kind: PressureSchurBackendKind,
    pressure_chebyshev_degree: usize,
    pressure_chebyshev_eigenvalue_ratio: f64,
    pressure_lambda_max: f64,
    workspace: Mutex<SaddlePointWorkspace<B>>,
    runtime_diagnostics: crate::DevicePreconditionerDiagnostics,
}

struct SaddlePointWorkspace<B: Backend> {
    result: B::Buffer<f64>,
    y_pressure: B::Buffer<f64>,
    modified_rhs: B::Buffer<f64>,
    residual: B::Buffer<f64>,
    correction: B::Buffer<f64>,
    d_new: B::Buffer<f64>,
    external_input: B::Buffer<f64>,
    external_output: B::Buffer<f64>,
    external_masked: B::Buffer<f64>,
}

impl<B: Backend> SaddlePointWorkspace<B> {
    fn new(backend: &B, len: usize) -> crate::BackendResult<Self> {
        Ok(Self {
            result: backend
                .raw_alloc::<f64>(len)
                .map_err(crate::BackendError::from)?,
            y_pressure: backend
                .raw_alloc::<f64>(len)
                .map_err(crate::BackendError::from)?,
            modified_rhs: backend
                .raw_alloc::<f64>(len)
                .map_err(crate::BackendError::from)?,
            residual: backend
                .raw_alloc::<f64>(len)
                .map_err(crate::BackendError::from)?,
            correction: backend
                .raw_alloc::<f64>(len)
                .map_err(crate::BackendError::from)?,
            d_new: backend
                .raw_alloc::<f64>(len)
                .map_err(crate::BackendError::from)?,
            external_input: backend
                .raw_alloc::<f64>(len)
                .map_err(crate::BackendError::from)?,
            external_output: backend
                .raw_alloc::<f64>(len)
                .map_err(crate::BackendError::from)?,
            external_masked: backend
                .raw_alloc::<f64>(len)
                .map_err(crate::BackendError::from)?,
        })
    }

    fn ensure_len(&mut self, backend: &B, len: usize) -> crate::BackendResult<()> {
        for buffer in [
            &mut self.result,
            &mut self.y_pressure,
            &mut self.modified_rhs,
            &mut self.residual,
            &mut self.correction,
            &mut self.d_new,
        ] {
            if buffer.len() != len {
                *buffer = backend
                    .raw_alloc::<f64>(len)
                    .map_err(crate::BackendError::from)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SaddlePointJacobiSchurBackendConfig {
    pub velocity_block_size: usize,
    pub velocity_degree: usize,
    pub velocity_damping: f64,
    pub pressure_degree: usize,
    pub pressure_damping: f64,
    pub pressure_kind: PressureSchurBackendKind,
    pub pressure_chebyshev_degree: usize,
    pub pressure_chebyshev_eigenvalue_ratio: f64,
}

impl<'a, B: Backend> SaddlePointJacobiSchurBackend<'a, B, f64> {
    /// Build from raw host CSR parts.
    ///
    /// Panics on construction failure; production paths must use
    /// [`Self::try_from_csr_host_parts`] instead.
    #[deprecated(note = "panics on construction failure; use try_from_csr_host_parts")]
    pub fn from_csr_host_parts(
        backend: &B,
        a: &'a BackendCsr<B, f64>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f64],
        pressure_schur: HostCsrRef<'_, f64>,
        config: SaddlePointJacobiSchurBackendConfig,
    ) -> Self {
        Self::try_from_csr_host_parts_with_pressure_preconditioner(
            backend,
            a,
            row_ptr_host,
            col_idx_host,
            values_host,
            pressure_schur,
            config,
            None,
        )
        .expect("saddle-point backend preconditioner construction failed")
    }

    /// Checked variant of [`Self::from_csr_host_parts`].
    pub fn try_from_csr_host_parts(
        backend: &B,
        a: &'a BackendCsr<B, f64>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f64],
        pressure_schur: HostCsrRef<'_, f64>,
        config: SaddlePointJacobiSchurBackendConfig,
    ) -> crate::BackendResult<Self> {
        Self::try_from_csr_host_parts_with_pressure_preconditioner(
            backend,
            a,
            row_ptr_host,
            col_idx_host,
            values_host,
            pressure_schur,
            config,
            None,
        )
    }

    /// Build from raw host CSR parts with an optional full-size pressure
    /// preconditioner. The external preconditioner must operate on a matrix
    /// where velocity rows are identity and pressure rows contain the embedded
    /// Schur approximation.
    ///
    /// Panics on construction failure; production paths must use
    /// [`Self::try_from_csr_host_parts_with_pressure_preconditioner`]
    /// instead.
    #[allow(clippy::too_many_arguments)]
    #[deprecated(
        note = "panics on construction failure; use try_from_csr_host_parts_with_pressure_preconditioner"
    )]
    pub fn from_csr_host_parts_with_pressure_preconditioner(
        backend: &B,
        a: &'a BackendCsr<B, f64>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f64],
        pressure_schur: HostCsrRef<'_, f64>,
        config: SaddlePointJacobiSchurBackendConfig,
        pressure_preconditioner: Option<&'a dyn BackendPreconditioner<B, f64>>,
    ) -> Self {
        Self::try_from_csr_host_parts_with_pressure_preconditioner(
            backend,
            a,
            row_ptr_host,
            col_idx_host,
            values_host,
            pressure_schur,
            config,
            pressure_preconditioner,
        )
        .expect("saddle-point backend preconditioner construction failed")
    }

    /// Checked variant of
    /// [`Self::from_csr_host_parts_with_pressure_preconditioner`].
    #[allow(clippy::too_many_arguments)]
    pub fn try_from_csr_host_parts_with_pressure_preconditioner(
        backend: &B,
        a: &'a BackendCsr<B, f64>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f64],
        pressure_schur: HostCsrRef<'_, f64>,
        config: SaddlePointJacobiSchurBackendConfig,
        pressure_preconditioner: Option<&'a dyn BackendPreconditioner<B, f64>>,
    ) -> crate::BackendResult<Self> {
        let velocity_block_size = config.velocity_block_size;
        if a.m != a.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_constructor",
                reason: "saddle-point backend preconditioner needs square A".into(),
            });
        }
        if a.m > i32::MAX as usize {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_constructor",
                reason: format!("matrix dimension {} exceeds i32 CSR index range", a.m),
            });
        }
        validate_csr_inputs(a.m, a.n, row_ptr_host, col_idx_host, values_host)?;
        HostCsrRef::try_new(
            pressure_schur.num_rows,
            pressure_schur.num_cols,
            pressure_schur.row_ptrs,
            pressure_schur.col_indices,
            pressure_schur.values,
        )
        .map_err(|error| crate::BackendError::InvalidArgument {
            operation: "saddle_preconditioner_constructor",
            reason: error.to_string(),
        })?;
        if velocity_block_size > a.m {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_constructor",
                reason: "velocity block must fit inside matrix".into(),
            });
        }
        let pressure_size = a.m - velocity_block_size;
        if pressure_schur.num_rows != pressure_size || pressure_schur.num_cols != pressure_size {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_constructor",
                reason: format!(
                    "Schur dimensions {}/{} do not match pressure block size {pressure_size}",
                    pressure_schur.num_rows, pressure_schur.num_cols
                ),
            });
        }

        let mut velocity_diag = vec![0.0_f64; a.m];
        let (velocity_block_row_ptr, velocity_block_col_idx, velocity_block_values) =
            extract_embedded_velocity_block(
                a.m,
                row_ptr_host,
                col_idx_host,
                values_host,
                velocity_block_size,
            );
        let (pressure_schur_row_ptr, pressure_schur_col_idx, pressure_schur_values) =
            embed_pressure_schur(a.m, velocity_block_size, pressure_schur);
        for row in 0..velocity_block_size {
            let lo = row_ptr_host[row] as usize;
            let hi = row_ptr_host[row + 1] as usize;
            for k in lo..hi {
                if col_idx_host[k] as usize == row {
                    velocity_diag[row] = values_host[k];
                    break;
                }
            }
        }

        let mut pressure_diag = vec![0.0_f64; a.m];
        for row in 0..pressure_schur.num_rows {
            let lo = pressure_schur.row_ptrs[row];
            let hi = pressure_schur.row_ptrs[row + 1];
            for k in lo..hi {
                if pressure_schur.col_indices[k] == row {
                    pressure_diag[velocity_block_size + row] = pressure_schur.values[k];
                    break;
                }
            }
        }
        let pressure_lambda_max = estimate_lambda_max_host_f64(
            pressure_schur.num_rows,
            pressure_schur.row_ptrs,
            pressure_schur.col_indices,
            pressure_schur.values,
            &pressure_diag[velocity_block_size..],
        )
        .max(f64::EPSILON);

        let matrix_m = a.m;
        let matrix_n = a.n;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::HierarchyUpload,
        )?;
        let a = PersistentDeviceCsr::try_clone_from_device_csr(backend, a)?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        let velocity_diag_inv = PersistentDeviceCsr::try_from_device_csr(
            backend,
            DeviceCsr::try_from_host_inverse_diagonal(backend, &velocity_diag)?,
        )?;
        let pressure_diag_inv = PersistentDeviceCsr::try_from_device_csr(
            backend,
            DeviceCsr::try_from_host_inverse_diagonal(backend, &pressure_diag)?,
        )?;
        let pressure_mask = upload_pressure_mask(backend, matrix_m, velocity_block_size)?;
        let velocity_block = PersistentDeviceCsr::try_from_device_csr(
            backend,
            BackendCsr::try_from_host(
                backend,
                matrix_m,
                matrix_n,
                &velocity_block_row_ptr,
                &velocity_block_col_idx,
                &velocity_block_values,
            )?,
        )?;
        let pressure_schur = PersistentDeviceCsr::try_from_device_csr(
            backend,
            BackendCsr::try_from_host(
                backend,
                matrix_m,
                matrix_n,
                &pressure_schur_row_ptr,
                &pressure_schur_col_idx,
                &pressure_schur_values,
            )?,
        )?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        let workspace = SaddlePointWorkspace::new(backend, matrix_m)?;
        let runtime_diagnostics = crate::DevicePreconditionerDiagnostics::default();
        runtime_diagnostics.record_setup();

        Ok(Self {
            a,
            velocity_block,
            pressure_schur,
            pressure_preconditioner,
            pressure_mask,
            velocity_diag_inv,
            pressure_diag_inv,
            n: matrix_m,
            velocity_degree: config.velocity_degree.max(1),
            velocity_damping: config.velocity_damping,
            pressure_degree: config.pressure_degree.max(1),
            pressure_damping: config.pressure_damping,
            pressure_kind: config.pressure_kind,
            pressure_chebyshev_degree: config.pressure_chebyshev_degree.max(1),
            pressure_chebyshev_eigenvalue_ratio: config
                .pressure_chebyshev_eigenvalue_ratio
                .max(1.0),
            pressure_lambda_max,
            workspace: Mutex::new(workspace),
            runtime_diagnostics,
        })
    }

    fn try_apply_pressure_schur(
        &self,
        backend: &B,
        r: &B::Buffer<f64>,
        y_pressure: &mut B::Buffer<f64>,
        residual: &mut B::Buffer<f64>,
        correction: &mut B::Buffer<f64>,
        d_new: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        match self.pressure_kind {
            PressureSchurBackendKind::Polynomial => {
                self.try_apply_pressure_polynomial(backend, r, y_pressure, residual, correction)?;
            }
            PressureSchurBackendKind::Chebyshev => {
                self.try_apply_pressure_chebyshev(
                    backend, r, y_pressure, residual, correction, d_new,
                )?;
            }
            PressureSchurBackendKind::External => {
                if let Some(preconditioner) = self.pressure_preconditioner {
                    preconditioner.try_apply(backend, r, y_pressure)?;
                    self.pressure_mask
                        .try_apply(backend, y_pressure, correction)?;
                    backend.try_copy_d2d(correction, y_pressure)?;
                } else {
                    self.try_apply_pressure_polynomial(
                        backend, r, y_pressure, residual, correction,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn try_apply_pressure_polynomial(
        &self,
        backend: &B,
        r: &B::Buffer<f64>,
        y_pressure: &mut B::Buffer<f64>,
        residual: &mut B::Buffer<f64>,
        correction: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        self.pressure_diag_inv.try_apply(backend, r, y_pressure)?;
        for _ in 1..self.pressure_degree {
            self.pressure_schur
                .try_apply(backend, y_pressure, residual)?;
            backend.try_scale::<f64>(-1.0, residual)?;
            backend.try_axpy::<f64>(1.0, r, residual)?;
            self.pressure_diag_inv
                .try_apply(backend, residual, correction)?;
            backend.try_axpy::<f64>(self.pressure_damping, correction, y_pressure)?;
        }
        Ok(())
    }

    fn try_apply_pressure_chebyshev(
        &self,
        backend: &B,
        r: &B::Buffer<f64>,
        y_pressure: &mut B::Buffer<f64>,
        residual: &mut B::Buffer<f64>,
        correction: &mut B::Buffer<f64>,
        d_new: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        if self.pressure_lambda_max <= 0.0 || self.pressure_chebyshev_eigenvalue_ratio <= 1.0 {
            self.try_apply_pressure_polynomial(backend, r, y_pressure, residual, correction)?;
            return Ok(());
        }

        let lambda_min = self.pressure_lambda_max / self.pressure_chebyshev_eigenvalue_ratio;
        let theta = 0.5 * (self.pressure_lambda_max + lambda_min);
        let delta = 0.5 * (self.pressure_lambda_max - lambda_min);
        if theta <= 0.0 || delta <= 0.0 {
            self.try_apply_pressure_polynomial(backend, r, y_pressure, residual, correction)?;
            return Ok(());
        }

        let sigma = theta / delta;
        let mut rho = 1.0 / sigma;

        self.pressure_diag_inv.try_apply(backend, r, correction)?;
        backend.try_scale::<f64>(1.0 / theta, correction)?;
        backend.try_copy_d2d(correction, y_pressure)?;

        for _ in 1..self.pressure_chebyshev_degree {
            let rho_new = 1.0 / (2.0 * sigma - rho);
            let coeff_old = rho_new * rho;
            let coeff_res = 2.0 * rho_new / delta;

            self.pressure_schur
                .try_apply(backend, y_pressure, residual)?;
            backend.try_scale::<f64>(-1.0, residual)?;
            backend.try_axpy::<f64>(1.0, r, residual)?;
            self.pressure_diag_inv.try_apply(backend, residual, d_new)?;

            backend.try_scale::<f64>(coeff_old, correction)?;
            backend.try_axpy::<f64>(coeff_res, d_new, correction)?;
            backend.try_axpy::<f64>(1.0, correction, y_pressure)?;
            rho = rho_new;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_apply_pressure_schur_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<f64>,
        y_pressure: &mut B::Buffer<f64>,
        residual: &mut B::Buffer<f64>,
        correction: &mut B::Buffer<f64>,
        d_new: &mut B::Buffer<f64>,
        external_input: &mut B::Buffer<f64>,
        external_output: &mut B::Buffer<f64>,
        external_masked: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        match self.pressure_kind {
            PressureSchurBackendKind::Polynomial => {
                self.try_apply_pressure_polynomial_colmajor(
                    backend, rhs, r, y_pressure, residual, correction,
                )?;
            }
            PressureSchurBackendKind::Chebyshev => {
                self.try_apply_pressure_chebyshev_colmajor(
                    backend, rhs, r, y_pressure, residual, correction, d_new,
                )?;
            }
            PressureSchurBackendKind::External => {
                if let Some(preconditioner) = self.pressure_preconditioner {
                    // The default col-major adapter is allowed to stage through the host.
                    // Apply one column at a time through persistent device buffers so a
                    // composed pressure preconditioner remains device-resident as well.
                    for column in 0..rhs {
                        let view = crate::BufferView::contiguous(column * self.n, self.n);
                        backend.try_copy_views_d2d(
                            r,
                            view,
                            external_input,
                            crate::BufferView::contiguous(0, self.n),
                        )?;
                        preconditioner.try_apply(backend, external_input, external_output)?;
                        self.pressure_mask
                            .try_apply(backend, external_output, external_masked)?;
                        backend.try_copy_views_d2d(
                            external_masked,
                            crate::BufferView::contiguous(0, self.n),
                            y_pressure,
                            view,
                        )?;
                    }
                } else {
                    self.try_apply_pressure_polynomial_colmajor(
                        backend, rhs, r, y_pressure, residual, correction,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_apply_pressure_polynomial_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<f64>,
        y_pressure: &mut B::Buffer<f64>,
        residual: &mut B::Buffer<f64>,
        correction: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        self.pressure_diag_inv
            .try_apply_colmajor(backend, rhs, r, y_pressure)?;
        for _ in 1..self.pressure_degree {
            self.pressure_schur
                .try_apply_colmajor(backend, rhs, y_pressure, residual)?;
            backend.try_scale::<f64>(-1.0, residual)?;
            backend.try_axpy::<f64>(1.0, r, residual)?;
            self.pressure_diag_inv
                .try_apply_colmajor(backend, rhs, residual, correction)?;
            backend.try_axpy::<f64>(self.pressure_damping, correction, y_pressure)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_apply_pressure_chebyshev_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<f64>,
        y_pressure: &mut B::Buffer<f64>,
        residual: &mut B::Buffer<f64>,
        correction: &mut B::Buffer<f64>,
        d_new: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        if self.pressure_lambda_max <= 0.0 || self.pressure_chebyshev_eigenvalue_ratio <= 1.0 {
            self.try_apply_pressure_polynomial_colmajor(
                backend, rhs, r, y_pressure, residual, correction,
            )?;
            return Ok(());
        }

        let lambda_min = self.pressure_lambda_max / self.pressure_chebyshev_eigenvalue_ratio;
        let theta = 0.5 * (self.pressure_lambda_max + lambda_min);
        let delta = 0.5 * (self.pressure_lambda_max - lambda_min);
        if theta <= 0.0 || delta <= 0.0 {
            self.try_apply_pressure_polynomial_colmajor(
                backend, rhs, r, y_pressure, residual, correction,
            )?;
            return Ok(());
        }

        let sigma = theta / delta;
        let mut rho = 1.0 / sigma;

        self.pressure_diag_inv
            .try_apply_colmajor(backend, rhs, r, correction)?;
        backend.try_scale::<f64>(1.0 / theta, correction)?;
        backend.try_copy_d2d(correction, y_pressure)?;

        for _ in 1..self.pressure_chebyshev_degree {
            let rho_new = 1.0 / (2.0 * sigma - rho);
            let coeff_old = rho_new * rho;
            let coeff_res = 2.0 * rho_new / delta;

            self.pressure_schur
                .try_apply_colmajor(backend, rhs, y_pressure, residual)?;
            backend.try_scale::<f64>(-1.0, residual)?;
            backend.try_axpy::<f64>(1.0, r, residual)?;
            self.pressure_diag_inv
                .try_apply_colmajor(backend, rhs, residual, d_new)?;

            backend.try_scale::<f64>(coeff_old, correction)?;
            backend.try_axpy::<f64>(coeff_res, d_new, correction)?;
            backend.try_axpy::<f64>(1.0, correction, y_pressure)?;
            rho = rho_new;
        }
        Ok(())
    }
}

impl<B: Backend> BackendPreconditioner<B, f64> for SaddlePointJacobiSchurBackend<'_, B, f64> {
    fn apply(&self, backend: &B, r: &B::Buffer<f64>, z: &mut B::Buffer<f64>) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "saddle-point preconditioner application failed: {error}"
            );
        }
    }

    fn try_apply(
        &self,
        backend: &B,
        r: &B::Buffer<f64>,
        z: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        let result = self.try_apply_untracked(backend, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_untracked(
        &self,
        backend: &B,
        r: &B::Buffer<f64>,
        z: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        if r.len() != self.n || z.len() != self.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_apply",
                reason: format!(
                    "expected input/output length {}, got {}/{}",
                    self.n,
                    r.len(),
                    z.len()
                ),
            });
        }

        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "saddle_preconditioner_apply",
                    reason: "workspace mutex poisoned".into(),
                })?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        workspace.ensure_len(backend, self.n)?;
        let SaddlePointWorkspace {
            result,
            y_pressure,
            modified_rhs,
            residual,
            correction,
            d_new,
            external_input: _,
            external_output: _,
            external_masked: _,
        } = &mut *workspace;

        self.try_apply_pressure_schur(backend, r, y_pressure, residual, correction, d_new)?;

        self.a.try_apply(backend, y_pressure, modified_rhs)?;
        backend.try_scale::<f64>(-1.0, modified_rhs)?;
        backend.try_axpy::<f64>(1.0, r, modified_rhs)?;
        self.velocity_diag_inv
            .try_apply(backend, modified_rhs, result)?;
        backend.try_scale::<f64>(self.velocity_damping, result)?;

        for _ in 1..self.velocity_degree {
            self.velocity_block.try_apply(backend, result, residual)?;
            backend.try_scale::<f64>(-1.0, residual)?;
            backend.try_axpy::<f64>(1.0, modified_rhs, residual)?;
            self.velocity_diag_inv
                .try_apply(backend, residual, correction)?;
            backend.try_axpy::<f64>(self.velocity_damping, correction, result)?;
        }

        backend.try_axpy::<f64>(1.0, y_pressure, result)?;
        backend.try_copy_d2d(result, z)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, r: &B::Buffer<f64>, z: &mut B::Buffer<f64>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "saddle-point multi-RHS preconditioner application failed: {error}"
            );
        }
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<f64>,
        z: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        let result = self.try_apply_colmajor_untracked(backend, rhs, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_colmajor_untracked(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<f64>,
        z: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        let len = self
            .n
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_apply_colmajor",
                reason: format!("workspace length overflow: n={}, rhs={rhs}", self.n),
            })?;
        if r.len() != len || z.len() != len {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_apply_colmajor",
                reason: format!(
                    "expected input/output length {len}, got {}/{}",
                    r.len(),
                    z.len()
                ),
            });
        }
        if len == 0 {
            return Ok(());
        }

        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "saddle_preconditioner_apply_colmajor",
                    reason: "workspace mutex poisoned".into(),
                })?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        workspace.ensure_len(backend, len)?;
        let SaddlePointWorkspace {
            result,
            y_pressure,
            modified_rhs,
            residual,
            correction,
            d_new,
            external_input,
            external_output,
            external_masked,
        } = &mut *workspace;

        self.try_apply_pressure_schur_colmajor(
            backend,
            rhs,
            r,
            y_pressure,
            residual,
            correction,
            d_new,
            external_input,
            external_output,
            external_masked,
        )?;

        self.a
            .try_apply_colmajor(backend, rhs, y_pressure, modified_rhs)?;
        backend.try_scale::<f64>(-1.0, modified_rhs)?;
        backend.try_axpy::<f64>(1.0, r, modified_rhs)?;
        self.velocity_diag_inv
            .try_apply_colmajor(backend, rhs, modified_rhs, result)?;
        backend.try_scale::<f64>(self.velocity_damping, result)?;

        for _ in 1..self.velocity_degree {
            self.velocity_block
                .try_apply_colmajor(backend, rhs, result, residual)?;
            backend.try_scale::<f64>(-1.0, residual)?;
            backend.try_axpy::<f64>(1.0, modified_rhs, residual)?;
            self.velocity_diag_inv
                .try_apply_colmajor(backend, rhs, residual, correction)?;
            backend.try_axpy::<f64>(self.velocity_damping, correction, result)?;
        }

        backend.try_axpy::<f64>(1.0, y_pressure, result)?;
        backend.try_copy_d2d(result, z)
    }

    fn diagnostics(&self) -> crate::DevicePreconditionerDiagnosticsSnapshot {
        self.runtime_diagnostics.snapshot()
    }

    fn try_apply_transpose(
        &self,
        _backend: &B,
        _r: &B::Buffer<f64>,
        _z: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "saddle_preconditioner_transpose",
            scalar: "f64",
        };
        self.runtime_diagnostics.record_transpose_failure(&error);
        Err(error)
    }

    fn try_apply_hermitian(
        &self,
        _backend: &B,
        _r: &B::Buffer<f64>,
        _z: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "saddle_preconditioner_hermitian",
            scalar: "f64",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }

    fn try_apply_transpose_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<f64>,
        _z: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "saddle_preconditioner_transpose_colmajor",
            scalar: "f64",
        };
        self.runtime_diagnostics.record_transpose_failure(&error);
        Err(error)
    }

    fn try_apply_hermitian_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<f64>,
        _z: &mut B::Buffer<f64>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "saddle_preconditioner_hermitian_colmajor",
            scalar: "f64",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }
}

fn extract_embedded_velocity_block<T: Copy>(
    n: usize,
    row_ptr_host: &[i32],
    col_idx_host: &[i32],
    values_host: &[T],
    velocity_block_size: usize,
) -> (Vec<i32>, Vec<i32>, Vec<T>) {
    let mut row_ptr = vec![0_i32; n + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        if row < velocity_block_size {
            let lo = row_ptr_host[row] as usize;
            let hi = row_ptr_host[row + 1] as usize;
            for k in lo..hi {
                let col = col_idx_host[k] as usize;
                if col < velocity_block_size {
                    col_idx.push(col_idx_host[k]);
                    values.push(values_host[k]);
                }
            }
        }
        row_ptr[row + 1] = col_idx.len() as i32;
    }
    (row_ptr, col_idx, values)
}

fn embed_pressure_schur(
    n: usize,
    velocity_block_size: usize,
    schur: HostCsrRef<'_, f64>,
) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut row_ptr = vec![0_i32; n + 1];
    let mut col_idx = Vec::with_capacity(schur.nnz());
    let mut values = Vec::with_capacity(schur.nnz());
    for row in 0..n {
        if row >= velocity_block_size {
            let pressure_row = row - velocity_block_size;
            let lo = schur.row_ptrs[pressure_row];
            let hi = schur.row_ptrs[pressure_row + 1];
            for k in lo..hi {
                col_idx.push((velocity_block_size + schur.col_indices[k]) as i32);
                values.push(schur.values[k]);
            }
        }
        row_ptr[row + 1] = col_idx.len() as i32;
    }
    (row_ptr, col_idx, values)
}

pub fn embed_pressure_schur_with_velocity_identity(
    n: usize,
    velocity_block_size: usize,
    schur: HostCsrRef<'_, f64>,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut row_ptr = vec![0_usize; n + 1];
    let mut col_idx = Vec::with_capacity(schur.nnz() + velocity_block_size);
    let mut values = Vec::with_capacity(schur.nnz() + velocity_block_size);
    for row in 0..n {
        if row < velocity_block_size {
            col_idx.push(row);
            values.push(1.0);
        } else {
            let pressure_row = row - velocity_block_size;
            let lo = schur.row_ptrs[pressure_row];
            let hi = schur.row_ptrs[pressure_row + 1];
            for k in lo..hi {
                col_idx.push(velocity_block_size + schur.col_indices[k]);
                values.push(schur.values[k]);
            }
        }
        row_ptr[row + 1] = col_idx.len();
    }
    (row_ptr, col_idx, values)
}

fn upload_pressure_mask<B: Backend>(
    backend: &B,
    n: usize,
    velocity_block_size: usize,
) -> crate::BackendResult<PersistentDeviceCsr<B, f64>> {
    let _n_i32 = i32::try_from(n).map_err(|_| crate::BackendError::InvalidArgument {
        operation: "pressure_mask_upload",
        reason: format!("matrix dimension {n} exceeds i32 CSR index range"),
    })?;
    let values_host: Vec<f64> = (0..n)
        .map(|row| if row < velocity_block_size { 0.0 } else { 1.0 })
        .collect();
    PersistentDeviceCsr::try_from_host_diagonal(backend, &values_host)
}

fn estimate_lambda_max_host_f64(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[f64],
    diag: &[f64],
) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut v: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut w = vec![0.0; n];
    let mut lambda = 1.0;
    for _ in 0..20 {
        for i in 0..n {
            let mut acc = 0.0;
            for k in row_ptr[i]..row_ptr[i + 1] {
                acc += values[k] * v[col_idx[k]];
            }
            w[i] = if diag[i].abs() > f64::EPSILON {
                acc / diag[i]
            } else {
                acc
            };
        }

        let mut num = 0.0;
        let mut den = 0.0;
        for (&vi, &wi) in v.iter().zip(w.iter()) {
            num += vi * wi;
            den += vi * vi;
        }
        if den > 0.0 {
            lambda = (num / den).abs();
        }

        let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 0.0 {
            for i in 0..n {
                v[i] = w[i] / nrm;
            }
        }
    }
    (lambda * 1.05).max(1.0)
}

#[cfg(test)]
// Fixtures intentionally use the panicking convenience constructors on
// tiny fixed matrices; production paths must use the `try_*` variants.
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::gmres::BackendCsr;
    use crate::{Backend, CpuBackend, PointJacobiBackend};

    #[test]
    fn backend_apply_matches_upper_triangular_jacobi_schur() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 2, 4, 5];
        let col_idx = vec![0, 2, 1, 2, 2];
        let values = vec![4.0_f64, 1.0, 4.0, 1.0, 0.1];
        let a = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 1,
                velocity_damping: 1.0,
                pressure_degree: 1,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 5,
                pressure_chebyshev_eigenvalue_ratio: 30.0,
            },
        );

        let mut r = backend.raw_alloc::<f64>(3).unwrap();
        backend.copy_h2d(&[1.0_f64, 1.0, 1.0], &mut r);
        let mut z = backend.raw_alloc::<f64>(3).unwrap();
        precond.apply(&backend, &r, &mut z);

        let mut host = vec![0.0_f64; 3];
        backend.copy_d2h(&z, &mut host);
        assert!((host[0] - 0.0).abs() < 1.0e-12);
        assert!((host[1] - 0.0).abs() < 1.0e-12);
        assert!((host[2] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn backend_from_csr_host_parts_matches_matrix_adapter() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 2, 4, 5];
        let col_idx = vec![0, 2, 1, 2, 2];
        let values = vec![4.0_f64, 1.0, 4.0, 1.0, 0.1];
        let a = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];
        let config = SaddlePointJacobiSchurBackendConfig {
            velocity_block_size: 2,
            velocity_degree: 1,
            velocity_damping: 1.0,
            pressure_degree: 1,
            pressure_damping: 1.0,
            pressure_kind: PressureSchurBackendKind::Polynomial,
            pressure_chebyshev_degree: 5,
            pressure_chebyshev_eigenvalue_ratio: 30.0,
        };
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
            config,
        );

        let mut r = backend.raw_alloc::<f64>(3).unwrap();
        backend.copy_h2d(&[1.0_f64, 1.0, 1.0], &mut r);
        let mut z = backend.raw_alloc::<f64>(3).unwrap();
        precond.apply(&backend, &r, &mut z);

        let mut host = vec![0.0_f64; 3];
        backend.copy_d2h(&z, &mut host);
        assert!((host[0] - 0.0).abs() < 1.0e-12);
        assert!((host[1] - 0.0).abs() < 1.0e-12);
        assert!((host[2] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn stronger_pressure_schur_polynomial_reduces_pressure_coupling_error() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 1, 2, 3, 4];
        let col_idx = vec![0, 1, 2, 3];
        let values = vec![4.0_f64, 4.0, 0.1, 0.1];
        let a = BackendCsr::from_host(&backend, 4, 4, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 2, 4];
        let schur_col_indices = vec![0_usize, 1, 0, 1];
        let schur_values = vec![2.0_f64, 0.5, 0.5, 2.0];
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(2, 2, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 1,
                velocity_damping: 1.0,
                pressure_degree: 2,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 5,
                pressure_chebyshev_eigenvalue_ratio: 30.0,
            },
        );

        let mut r = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&[0.0_f64, 0.0, 1.0, 0.0], &mut r);
        let mut z = backend.raw_alloc::<f64>(4).unwrap();
        precond.apply(&backend, &r, &mut z);

        let mut host = vec![0.0_f64; 4];
        backend.copy_d2h(&z, &mut host);
        assert!((host[2] - 0.5).abs() < 1.0e-12);
        assert!((host[3] + 0.125).abs() < 1.0e-12);
    }

    #[test]
    fn backend_apply_colmajor_matches_column_apply() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 3, 6, 7, 8];
        let col_idx = vec![0, 1, 2, 0, 1, 3, 2, 3];
        let values = vec![4.0_f64, 0.5, 1.0, 0.25, 3.0, -0.5, 0.1, 0.1];
        let a = BackendCsr::from_host(&backend, 4, 4, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 2, 4];
        let schur_col_indices = vec![0_usize, 1, 0, 1];
        let schur_values = vec![2.0_f64, 0.5, 0.5, 2.0];
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(2, 2, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 2,
                velocity_damping: 0.9,
                pressure_degree: 2,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 5,
                pressure_chebyshev_eigenvalue_ratio: 30.0,
            },
        );
        let r_host = vec![1.0_f64, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75];
        let mut r = backend.raw_alloc::<f64>(r_host.len()).unwrap();
        backend.copy_h2d(&r_host, &mut r);
        let mut z_multi = backend.raw_alloc::<f64>(r_host.len()).unwrap();

        precond.apply_colmajor(&backend, 2, &r, &mut z_multi);

        let mut expected = vec![0.0_f64; r_host.len()];
        for col in 0..2 {
            let range = col * 4..(col + 1) * 4;
            let mut r_col = backend.raw_alloc::<f64>(4).unwrap();
            backend.copy_h2d(&r_host[range.clone()], &mut r_col);
            let mut z_col = backend.raw_alloc::<f64>(4).unwrap();
            precond.try_apply(&backend, &r_col, &mut z_col).unwrap();
            backend.copy_d2h(&z_col, &mut expected[range]);
        }

        let mut actual = vec![0.0_f64; r_host.len()];
        backend.copy_d2h(&z_multi, &mut actual);
        for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1.0e-12,
                "index {index}: got {got} want {want}; actual={actual:?} expected={expected:?}"
            );
        }

        let diagnostics = precond.diagnostics();
        assert_eq!(diagnostics.apply_count, 3);
        assert_eq!(diagnostics.apply_failures, 0);

        let mut r_failure = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&r_host[..4], &mut r_failure);
        let sentinel = vec![7.0_f64; 4];
        let mut z_failure = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&sentinel, &mut z_failure);
        CpuBackend::inject_pending_error();
        let error = precond
            .try_apply(&backend, &r_failure, &mut z_failure)
            .expect_err("injected backend failure must reach the checked boundary");
        assert!(matches!(error, crate::BackendError::OperationFailed { .. }));
        let mut after_failure = vec![0.0_f64; 4];
        backend.copy_d2h(&z_failure, &mut after_failure);
        assert_eq!(after_failure, sentinel);

        let diagnostics = precond.diagnostics();
        assert_eq!(diagnostics.apply_count, 3);
        assert_eq!(diagnostics.apply_failures, 1);
        assert_eq!(
            diagnostics.first_failure_stage,
            Some(crate::DevicePreconditionerFailureStage::Launch)
        );
    }

    #[test]
    fn external_pressure_block_apply_does_not_stage_through_host() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 3, 6, 7, 8];
        let col_idx = vec![0, 1, 2, 0, 1, 3, 2, 3];
        let values = vec![4.0_f64, 0.5, 1.0, 0.25, 3.0, -0.5, 0.1, 0.1];
        let a = BackendCsr::from_host(&backend, 4, 4, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 2, 4];
        let schur_col_indices = vec![0_usize, 1, 0, 1];
        let schur_values = vec![2.0_f64, 0.5, 0.5, 2.0];
        let external = PointJacobiBackend::from_diagonal(&backend, &[1.0, 1.0, 2.0, 2.0]);
        let config = SaddlePointJacobiSchurBackendConfig {
            velocity_block_size: 2,
            velocity_degree: 1,
            velocity_damping: 1.0,
            pressure_degree: 1,
            pressure_damping: 1.0,
            pressure_kind: PressureSchurBackendKind::External,
            pressure_chebyshev_degree: 5,
            pressure_chebyshev_eigenvalue_ratio: 30.0,
        };
        let precond =
            SaddlePointJacobiSchurBackend::from_csr_host_parts_with_pressure_preconditioner(
                &backend,
                &a,
                &row_ptr,
                &col_idx,
                &values,
                HostCsrRef::new(2, 2, &schur_row_ptrs, &schur_col_indices, &schur_values),
                config,
                Some(&external),
            );
        let r_host = vec![1.0_f64, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75];
        let mut r = backend.raw_alloc::<f64>(r_host.len()).unwrap();
        backend.copy_h2d(&r_host, &mut r);
        let mut expected = vec![0.0_f64; r_host.len()];
        for column in 0..2 {
            let range = column * 4..(column + 1) * 4;
            let mut r_col = backend.raw_alloc::<f64>(4).unwrap();
            backend.copy_h2d(&r_host[range.clone()], &mut r_col);
            let mut z_col = backend.raw_alloc::<f64>(4).unwrap();
            precond.try_apply(&backend, &r_col, &mut z_col).unwrap();
            backend.copy_d2h(&z_col, &mut expected[range]);
        }

        let mut z = backend.raw_alloc::<f64>(r_host.len()).unwrap();
        backend.reset_transfer_stats();
        precond.try_apply_colmajor(&backend, 2, &r, &mut z).unwrap();
        let device_only = backend.transfer_stats();
        assert_eq!(device_only.h2d_operations, 0);
        assert_eq!(device_only.d2h_operations, 0);

        let mut actual = vec![0.0_f64; r_host.len()];
        backend.copy_d2h(&z, &mut actual);
        for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1.0e-12,
                "index {index}: got {got} want {want}"
            );
        }
    }

    #[test]
    fn checked_constructor_rejects_malformed_velocity_csr() {
        let backend = CpuBackend::new();
        let valid_row_ptr = vec![0, 1, 2, 3];
        let valid_col_idx = vec![0, 1, 2];
        let valid_values = vec![1.0_f64; 3];
        let a = BackendCsr::from_host(
            &backend,
            3,
            3,
            &valid_row_ptr,
            &valid_col_idx,
            &valid_values,
        );
        let malformed_row_ptr = vec![0, 2, 1, 3];
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];

        let result = SaddlePointJacobiSchurBackend::try_from_csr_host_parts(
            &backend,
            &a,
            &malformed_row_ptr,
            &valid_col_idx,
            &valid_values,
            HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 1,
                velocity_damping: 1.0,
                pressure_degree: 1,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 2,
                pressure_chebyshev_eigenvalue_ratio: 2.0,
            },
        );

        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn checked_apply_rejects_wrong_buffer_lengths() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![1.0_f64; 3];
        let a = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 1,
                velocity_damping: 1.0,
                pressure_degree: 1,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 2,
                pressure_chebyshev_eigenvalue_ratio: 2.0,
            },
        );
        let r = backend.raw_alloc::<f64>(2).unwrap();
        let mut z = backend.raw_alloc::<f64>(3).unwrap();

        let result = precond.try_apply(&backend, &r, &mut z);
        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn checked_colmajor_rejects_workspace_overflow() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![1.0_f64; 3];
        let a = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 1,
                velocity_damping: 1.0,
                pressure_degree: 1,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 2,
                pressure_chebyshev_eigenvalue_ratio: 2.0,
            },
        );
        let r = backend.raw_alloc::<f64>(0).unwrap();
        let mut z = backend.raw_alloc::<f64>(0).unwrap();

        let result = precond.try_apply_colmajor(&backend, usize::MAX, &r, &mut z);
        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }
}
