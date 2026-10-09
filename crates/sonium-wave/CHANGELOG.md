# Changelog

## Unreleased

### Fixed

- `run_richardson` reuses one scratch buffer across iterations instead of
  cloning a full-grid `domain.medium(&x)` per step; operation order and
  numerics are unchanged.
- Domain construction shares a single FFT planner across the three
  `laplace_kernel_axis` calls; new `laplace_kernel_axis_with_planner` exposes
  the planner-reusing core for repeated constructions.

### Added

- `tests/wavesim_convergence.rs`: 1D homogeneous point-source solve vs the
  `analytical_1d` discrete-kernel reference (residual + relative error).
- `tests/kernel_consistency.rs`: adjoint-consistency
  (`dg_dnx == -dg_dny` for shared normals) and coincidence policy
  (bare-kernel infinite sentinel vs checked-`try_*` rejection).
- Machine-adaptive rayon gates (`parallelism::ParallelPolicy`, replacing
  any fixed threshold thinking): FFT lane loops and pointwise passes
  (medium application, propagator divide, IFFT normalization) parallelize
  when `work_items >= items_per_thread × available_parallelism`, with
  `SONIUM_WAVE_PAR_THRESHOLD` override and `SONIUM_SERIAL` escape hatch.
  Strided axes gather into one scratch buffer, transform lanes in parallel,
  and scatter back; serial and parallel paths agree bitwise
  (`parallel_lanes_match_serial_bitwise`, `pointwise_kernels_match_on_both_paths`).

## 0.4.1

### Fixed

- Division by zero at `ka = 0` in `compute_rigid_sphere_coefficients`, `oscillating_sphere_3d`,
  and `impedance_sphere_scattering_3d`. Added guards that return the correct low-frequency limit.
- Phase sign mismatch in `damped_wave_1d`: implementation computed `exp(-αx)·exp(+ikx)` but the
  documented formula is `exp(-(α+ik)x) = exp(-αx)·exp(-ikx)`. Fixed to match the documented
  convention.
- Added `speed_of_sound` parameter to `impedance_sphere_scattering_3d` so the ka≈0 guard can
  compute the correct frequency for the returned `AnalyticalSolution`.
- Added clarifying comment on the Giebermann continued-fraction `gnu` formula in
  `spherical_hankel_first_kind` noting the convention matches NC_SphericalHankel.
