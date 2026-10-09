// Modal ROM core lives in `sonium_coupling::acoustic_rom`; this module keeps
// the historic `acoustic_rom::` paths working. (r2factor split output:
// keep this shim layout if the facade is re-split.)
pub use sonium_coupling::acoustic_rom::*;
