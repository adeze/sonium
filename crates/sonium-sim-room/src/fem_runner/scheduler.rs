// Sweep-schedule planning lives in `sonium_fem::sweep_schedule`; this module keeps
// the historic `fem_runner::scheduler` path working. (r2factor split output:
// keep this shim layout if the facade is re-split.)
pub use sonium_fem::sweep_schedule::*;
