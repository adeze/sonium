# Changelog

## Unreleased

### Added

- Non-compact port noise on 2-D slot meshes: `PermeableFwh` now
  evaluates the planar FW-H contour (`stationary_fwh_pressure_2d`)
  instead of rejecting 2-D meshes, reporting the infinite-line field
  (cylindrical `1/√r` spreading, covered by a two-observer test).

### Changed

- Snapshot-to-phasor coupling goes through the shared
  `sonium-fem` `snapshot` helper (`dft_peak_phasor`,
  `fwh_patch_from_series`): sample-identical numerics to the legacy
  `dft_phasor(...).conj()` path, locked by a cross-convention test,
  with fail-closed errors replacing silent zeros on bad input.
