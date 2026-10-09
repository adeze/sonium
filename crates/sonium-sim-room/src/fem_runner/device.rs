//! Persistent device-side helpers used by the room FEM GPU runners.
//!
//! Source directivity and complex amplitudes are still evaluated by the
//! application because they are model-specific.  Once those coefficients are
//! available, the sparse source basis and the listener interpolation remain
//! resident and all accumulation/extraction happens through backend kernels.

use anyhow::{Context, Result, anyhow};
use sonium_compute::{
    Backend, BufferElement, DeviceBufferLike, DeviceCsr, DeviceOperator, DeviceScalar,
};

/// Device-resident sparse source scatter.
///
/// The matrix has one column per source-basis entry and one row per mesh DOF.
/// Its values are the real scatter weights promoted to `T`. Production room
/// coefficients already include their source-basis weights, so that path
/// supplies unit-valued scatter entries to avoid applying a weight twice.
pub(crate) struct DeviceRhsAssembler<B: Backend, T: DeviceScalar> {
    basis: DeviceCsr<B, T>,
    coefficients: B::Buffer<T>,
}

/// Device-resident sparse source scatter for a batch of right-hand sides.
///
/// Rows are laid out as `rhs * n_dofs`, with one contiguous block of rows per
/// source.  A single SpMV therefore assembles all source RHS vectors into the
/// same column-major layout consumed by block Krylov solvers. The production
/// room path supplies unit-valued scatter entries because its coefficients
/// already include source-basis weights.
#[cfg(any(feature = "cuda", test))]
pub(crate) struct DeviceRhsBlockAssembler<B: Backend, T: DeviceScalar> {
    basis: DeviceCsr<B, T>,
    coefficients: B::Buffer<T>,
    n_dofs: usize,
    rhs: usize,
}

#[cfg(any(feature = "cuda", test))]
struct ListenerBlockCache<B: Backend, T: DeviceScalar>
where
    T::Real: BufferElement,
{
    n_dofs: usize,
    rhs: usize,
    indices: B::Buffer<i32>,
    weights: B::Buffer<T::Real>,
    output: B::Buffer<T>,
}

#[cfg(any(feature = "cuda", test))]
impl<B: Backend, T: DeviceScalar> DeviceRhsBlockAssembler<B, T> {
    pub(crate) fn new(
        backend: &B,
        n_dofs: usize,
        entries: &[Vec<(usize, T::Real)>],
    ) -> Result<Self> {
        anyhow::ensure!(
            !entries.is_empty(),
            "block source basis needs at least one RHS"
        );
        let rhs = entries.len();
        let total_entries = entries.iter().map(Vec::len).sum::<usize>();
        let mut row_entries: Vec<Vec<(i32, T)>> = vec![Vec::new(); n_dofs * rhs];
        let mut column_offset = 0usize;
        for (source_idx, source_entries) in entries.iter().enumerate() {
            for &(node, weight) in source_entries {
                anyhow::ensure!(
                    node < n_dofs,
                    "source basis node {node} exceeds {n_dofs} DOFs"
                );
                let column = i32::try_from(column_offset)
                    .context("source basis exceeds i32 column range")?;
                row_entries[source_idx * n_dofs + node].push((column, T::from_real(weight)));
                column_offset += 1;
            }
        }

        debug_assert_eq!(column_offset, total_entries);
        let mut row_ptr = vec![0_i32; n_dofs * rhs + 1];
        for (row, values) in row_entries.iter().enumerate() {
            row_ptr[row + 1] = row_ptr[row]
                .checked_add(i32::try_from(values.len()).context("source basis row too large")?)
                .context("source basis exceeds i32 nnz range")?;
        }
        let nnz = row_ptr[n_dofs * rhs] as usize;
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for row in row_entries {
            for (column, value) in row {
                col_idx.push(column);
                values.push(value);
            }
        }

        let basis = DeviceCsr::try_from_host(
            backend,
            n_dofs * rhs,
            total_entries,
            &row_ptr,
            &col_idx,
            &values,
        )
        .map_err(|error| anyhow!(error))?;
        let coefficients = backend
            .raw_alloc::<T>(total_entries)
            .map_err(|error| anyhow!(error))?;

        Ok(Self {
            basis,
            coefficients,
            n_dofs,
            rhs,
        })
    }

    /// Compute all source RHS vectors into one column-major device buffer.
    pub(crate) fn assemble(&mut self, backend: &B, coefficients: &[T], rhs: &mut B::Buffer<T>) {
        assert_eq!(coefficients.len(), self.coefficients.len());
        assert_eq!(rhs.len(), self.n_dofs * self.rhs);
        backend.copy_h2d(coefficients, &mut self.coefficients);
        self.basis.apply(backend, &self.coefficients, rhs);
    }
}

impl<B: Backend, T: DeviceScalar> DeviceRhsAssembler<B, T> {
    pub(crate) fn new(backend: &B, n_dofs: usize, entries: &[(usize, T::Real)]) -> Result<Self> {
        let mut row_ptr = vec![0_i32; n_dofs + 1];
        for &(node, _) in entries {
            anyhow::ensure!(
                node < n_dofs,
                "source basis node {node} exceeds {n_dofs} DOFs"
            );
        }
        row_ptr.fill(0);
        for &(node, _) in entries {
            row_ptr[node + 1] += 1;
        }
        for row in 1..=n_dofs {
            row_ptr[row] += row_ptr[row - 1];
        }
        let mut ordered_cols = vec![0_i32; entries.len()];
        let mut ordered_values = vec![T::zero(); entries.len()];
        let mut cursor = row_ptr[..n_dofs].to_vec();
        for (column, &(node, weight)) in entries.iter().enumerate() {
            let slot = cursor[node] as usize;
            ordered_cols[slot] =
                i32::try_from(column).context("source basis exceeds i32 column range")?;
            ordered_values[slot] = T::from_real(weight);
            cursor[node] += 1;
        }
        let mut row_ptr_dev = backend
            .raw_alloc::<i32>(row_ptr.len())
            .map_err(|error| anyhow!(error))?;
        let mut col_idx_dev = backend
            .raw_alloc::<i32>(ordered_cols.len())
            .map_err(|error| anyhow!(error))?;
        let mut values_dev = backend
            .raw_alloc::<T>(ordered_values.len())
            .map_err(|error| anyhow!(error))?;
        let coefficients = backend
            .raw_alloc::<T>(entries.len())
            .map_err(|error| anyhow!(error))?;
        backend.copy_h2d(&row_ptr, &mut row_ptr_dev);
        backend.copy_h2d(&ordered_cols, &mut col_idx_dev);
        backend.copy_h2d(&ordered_values, &mut values_dev);

        Ok(Self {
            basis: DeviceCsr {
                m: n_dofs,
                n: entries.len(),
                nnz: ordered_values.len(),
                row_ptr: row_ptr_dev,
                col_idx: col_idx_dev,
                values: values_dev,
            },
            coefficients,
        })
    }

    /// Compute `rhs <- basis * coefficients` without downloading a nodal RHS.
    pub(crate) fn assemble(&mut self, backend: &B, coefficients: &[T], rhs: &mut B::Buffer<T>) {
        assert_eq!(coefficients.len(), self.coefficients.len());
        assert_eq!(rhs.len(), self.basis.m);
        backend.copy_h2d(coefficients, &mut self.coefficients);
        self.basis.apply(backend, &self.coefficients, rhs);
    }

    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.coefficients.len()
    }
}

/// Persistent four-point listener extraction buffers.
pub(crate) struct DeviceListenerExtraction<B: Backend, T: DeviceScalar>
where
    T::Real: BufferElement,
{
    indices: B::Buffer<i32>,
    weights: B::Buffer<T::Real>,
    output: B::Buffer<T>,
    #[cfg(any(feature = "cuda", test))]
    indices_host: Vec<i32>,
    #[cfg(any(feature = "cuda", test))]
    weights_host: Vec<T::Real>,
    #[cfg(any(feature = "cuda", test))]
    block_cache: Option<ListenerBlockCache<B, T>>,
}

impl<B: Backend, T: DeviceScalar> DeviceListenerExtraction<B, T>
where
    T::Real: BufferElement,
{
    pub(crate) fn new(backend: &B, nodes: &[[usize; 4]], weights: &[[T::Real; 4]]) -> Result<Self> {
        anyhow::ensure!(
            nodes.len() == weights.len(),
            "listener node/weight count mismatch"
        );
        let flat_indices: Vec<i32> = nodes
            .iter()
            .flat_map(|quad| quad.iter().copied())
            .map(|node| i32::try_from(node).map_err(|_| anyhow!("listener node exceeds i32 range")))
            .collect::<Result<_>>()?;
        let flat_weights: Vec<T::Real> = weights
            .iter()
            .flat_map(|quad| quad.iter().copied())
            .collect();
        let mut indices = backend
            .raw_alloc::<i32>(flat_indices.len())
            .map_err(|error| anyhow!(error))?;
        let mut weights = backend
            .raw_alloc::<T::Real>(flat_weights.len())
            .map_err(|error| anyhow!(error))?;
        let output = backend
            .raw_alloc::<T>(nodes.len())
            .map_err(|error| anyhow!(error))?;
        backend.copy_h2d(&flat_indices, &mut indices);
        backend.copy_h2d(&flat_weights, &mut weights);
        Ok(Self {
            indices,
            weights,
            output,
            #[cfg(any(feature = "cuda", test))]
            indices_host: flat_indices,
            #[cfg(any(feature = "cuda", test))]
            weights_host: flat_weights,
            #[cfg(any(feature = "cuda", test))]
            block_cache: None,
        })
    }

    /// Evaluate all listeners into the persistent device output buffer.
    pub(crate) fn evaluate_device<'a>(
        &'a mut self,
        backend: &B,
        values: &B::Buffer<T>,
    ) -> &'a B::Buffer<T> {
        backend.gather4_weighted(values, &self.indices, &self.weights, &mut self.output);
        &self.output
    }

    /// Evaluate and copy only the listener values to the host.
    pub(crate) fn evaluate_to_host(&mut self, backend: &B, values: &B::Buffer<T>) -> Vec<T> {
        let output_len = self.output.len();
        let output = self.evaluate_device(backend, values);
        let mut host = vec![T::zero(); output_len];
        backend.copy_d2h(output, &mut host);
        host
    }

    /// Evaluate a column-major block of nodal vectors with one persistent
    /// gather launch and copy only the listener values back to the host.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn evaluate_colmajor_to_host(
        &mut self,
        backend: &B,
        values: &B::Buffer<T>,
        n_dofs: usize,
        rhs: usize,
    ) -> Vec<Vec<T>> {
        assert_eq!(values.len(), n_dofs * rhs);
        if rhs == 0 {
            return Vec::new();
        }
        let cache = if let Some(cache) = &mut self.block_cache {
            cache
        } else {
            let mut indices = Vec::with_capacity(self.indices_host.len() * rhs);
            let mut weights = Vec::with_capacity(self.weights_host.len() * rhs);
            for column in 0..rhs {
                let offset =
                    i32::try_from(column * n_dofs).expect("block listener offset exceeds i32");
                indices.extend(self.indices_host.iter().map(|&index| index + offset));
                weights.extend_from_slice(&self.weights_host);
            }
            let mut indices_dev = backend
                .raw_alloc::<i32>(indices.len())
                .expect("block listener index allocation failed");
            let mut weights_dev = backend
                .raw_alloc::<T::Real>(weights.len())
                .expect("block listener weight allocation failed");
            let output_dev = backend
                .raw_alloc::<T>(self.output.len() * rhs)
                .expect("block listener output allocation failed");
            backend.copy_h2d(&indices, &mut indices_dev);
            backend.copy_h2d(&weights, &mut weights_dev);
            self.block_cache = Some(ListenerBlockCache {
                n_dofs,
                rhs,
                indices: indices_dev,
                weights: weights_dev,
                output: output_dev,
            });
            self.block_cache
                .as_mut()
                .expect("block listener cache was just initialized")
        };
        assert_eq!(cache.n_dofs, n_dofs);
        assert_eq!(cache.rhs, rhs);
        backend.gather4_weighted(values, &cache.indices, &cache.weights, &mut cache.output);
        let mut output = vec![T::zero(); self.output.len() * rhs];
        backend.copy_d2h(&cache.output, &mut output);
        output
            .chunks_exact(self.output.len())
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.output.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_compute::CpuBackend;

    #[test]
    fn rhs_basis_accumulates_on_device_for_arbitrary_entry_order() {
        let backend = CpuBackend::new();
        let mut assembler = DeviceRhsAssembler::<CpuBackend, Complex64>::new(
            &backend,
            3,
            &[(2, 0.25), (0, 0.5), (2, 0.75)],
        )
        .unwrap();
        assert_eq!(assembler.entry_count(), 3);
        let mut rhs = backend.raw_alloc::<Complex64>(3).unwrap();
        assembler.assemble(
            &backend,
            &[
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            &mut rhs,
        );
        let mut host = [Complex64::new(0.0, 0.0); 3];
        backend.copy_d2h(&rhs, &mut host);
        assert_eq!(host[0], Complex64::new(0.5, 0.0));
        assert_eq!(host[1], Complex64::new(0.0, 0.0));
        assert_eq!(host[2], Complex64::new(3.5, 0.0));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_preweighted_rhs_scatter_matches_cpu_reference() {
        use sonium_compute::{Backend, CudaBackend};

        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_some() {
                    panic!("CUDA is required for the RHS scatter regression: {error}");
                }
                eprintln!("no CUDA device — skipping RHS scatter regression: {error}");
                return;
            }
        };

        // The room runner applies each source-basis weight while building its
        // coefficient vector. The device matrix must therefore contain unit
        // scatter values, including when several entries target one node.
        let nodes = [(2, 1.0_f64), (0, 1.0_f64), (2, 1.0_f64)];
        let coefficients = [
            Complex64::new(0.5, -0.25),
            Complex64::new(1.0, 0.75),
            Complex64::new(-1.5, 0.125),
        ];
        let mut assembler =
            DeviceRhsAssembler::<CudaBackend, Complex64>::new(&backend, 3, &nodes).unwrap();
        let mut rhs = backend.raw_alloc::<Complex64>(3).unwrap();
        assembler.assemble(&backend, &coefficients, &mut rhs);
        backend.synchronize().unwrap();

        let mut uploaded = [Complex64::new(0.0, 0.0); 3];
        backend.copy_d2h(&assembler.coefficients, &mut uploaded);
        backend.synchronize().unwrap();
        assert_eq!(uploaded, coefficients);

        let mut actual = [Complex64::new(0.0, 0.0); 3];
        backend.copy_d2h(&rhs, &mut actual);
        backend.synchronize().unwrap();
        let mut expected = [Complex64::new(0.0, 0.0); 3];
        expected[2] = coefficients[0] + coefficients[2];
        expected[0] = coefficients[1];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (*actual - expected).norm() <= 1.0e-12,
                "CUDA RHS scatter mismatch: actual={actual:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn listeners_are_uploaded_once_and_extracted_without_full_solution_copy() {
        let backend = CpuBackend::new();
        let mut listeners = DeviceListenerExtraction::<CpuBackend, Complex64>::new(
            &backend,
            &[[0, 1, 2, 3], [3, 2, 1, 0]],
            &[[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
        assert_eq!(listeners.len(), 2);
        let mut values = backend.raw_alloc::<Complex64>(4).unwrap();
        backend.copy_h2d(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            &mut values,
        );
        assert_eq!(
            listeners.evaluate_to_host(&backend, &values),
            vec![Complex64::new(2.5, 0.0), Complex64::new(4.0, 0.0)]
        );
    }

    #[test]
    fn block_rhs_assembler_preserves_source_column_layout() {
        let backend = CpuBackend::new();
        let mut assembler = DeviceRhsBlockAssembler::<CpuBackend, Complex64>::new(
            &backend,
            3,
            &[vec![(2, 0.25), (0, 0.5), (2, 0.75)], vec![(1, 1.0)]],
        )
        .unwrap();
        let mut rhs = backend.raw_alloc::<Complex64>(6).unwrap();
        assembler.assemble(
            &backend,
            &[
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            &mut rhs,
        );
        let mut host = [Complex64::new(0.0, 0.0); 6];
        backend.copy_d2h(&rhs, &mut host);
        assert_eq!(
            host,
            [
                Complex64::new(0.5, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.5, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]
        );
    }

    #[test]
    fn block_listener_extraction_matches_per_column_results() {
        let backend = CpuBackend::new();
        let mut listeners = DeviceListenerExtraction::<CpuBackend, Complex64>::new(
            &backend,
            &[[0, 1, 2, 3]],
            &[[0.25, 0.25, 0.25, 0.25]],
        )
        .unwrap();
        let mut values = backend.raw_alloc::<Complex64>(8).unwrap();
        backend.copy_h2d(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            &mut values,
        );
        assert_eq!(
            listeners.evaluate_colmajor_to_host(&backend, &values, 4, 2),
            vec![
                vec![Complex64::new(2.5, 0.0)],
                vec![Complex64::new(2.5, 0.0)],
            ]
        );
    }
}
