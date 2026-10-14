//! Reusable interior/exterior (FEM/BEM) coupling orchestration.
//!
//! This crate sits above `sonium-fem` and `sonium-bem` and owns the numeric
//! coupling seam — never product geometry, patch registries, driver networks,
//! or measurement acceptance, which stay in the simulation crates. There is no
//! FEM↔BEM dependency cycle: both libraries are dependencies here, and
//! interface-trace schemas live in `sonium-xem`.
//!
//! - [`monolithic`]: product-independent `[p_interior, p_exterior, v_aperture]`
//!   acoustic operator with compressed exterior blocks, field-split
//!   preconditioner, and dense-reference solution record.
//! - [`power`]: interface power ledger — radiated/active power computed
//!   independently on each side of the aperture plus the conservation audit
//!   that ties them together.
//! - [`trace`]: validated aperture-trace schema (topology, areas, measures,
//!   convention) exchanged between interior and exterior builders.
//! - [`acoustic_rom`]: product-independent interior modal basis (rigid-wall
//!   modes + Craig–Bampton attachment modes, static correction, reduced
//!   solve/adjoint) consumed by coupled sweep drivers.
//!
//! Research note (SIM-PERF-20261013.md section 2, gap 5): FEM–BEM coupling
//! is not wired as a preconditioner inside any demo sweep; the demos
//! precondition the interior (FEM) and exterior (BEM) operators
//! separately. Using the coupled operator as a demo preconditioner
//! remains future work.

pub mod acoustic_rom;
pub mod monolithic;
pub mod power;
pub mod trace;

pub use acoustic_rom::{
    AcousticRom, AcousticRomConfig, RomError, RomFrequencySolution, StaticCorrection,
};
pub use monolithic::{
    CompressedMonolithicFemBemOperator, CompressedMonolithicFieldSplitPreconditioner,
    CompressedMonolithicSolution, DEFAULT_MONOLITHIC_MAX_DOFS, MonolithicCoupledSolution,
    MonolithicCouplingError, build_pressure_port_field_split_preconditioner,
    build_velocity_mode_hmatrix,
};
pub use power::{InterfacePowerLedger, PowerLedgerError};
pub use trace::{ApertureFace, ApertureTrace, TraceSchemaError};
