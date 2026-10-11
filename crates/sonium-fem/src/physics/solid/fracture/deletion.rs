//! Element deletion policy and erosion energy accounting.
//!
//! Turns pointwise failure flags into element deletion verdicts and tracks
//! where the plastic and internal energy goes when elements erode. The
//! deletion set itself stays in [`ErosionSet`](super::ductile::ErosionSet)
//! (stable ids, no renumbering); this module adds the card-level policy
//! and the energy ledger on top.
//!
//! - Solid policy mirrors `/FAIL/TAB1` `Ifail_so`: delete the element when
//!   any integration point fails (= 1), else zero the failed-point
//!   deviatoric stress (= 2) (Reference Guide `/FAIL/TAB1` fields).
//! - Shell policy mirrors `Ifail_sh`: one failed point deletes (= 1),
//!   all layers must fail (= 2), and = 3 additionally keeps broken-layer
//!   stresses until the whole element ruptures. `P_thickfail` replaces the
//!   all-layers rule by a failed-thickness ratio (comment 7): rupture
//!   trips once adjacent failed layers from one skin exceed the ratio.
//! - The ledger records per-element plastic work and internal energy each
//!   step; eroded elements move their totals to the eroded column, so the
//!   books always balance: total = active + eroded. SI units throughout.
//!
//! # Example
//!
//! ```rust
//! use sonium_fem::physics::solid::fracture::deletion::{DeletionPolicy, ErosionLedger};
//!
//! let policy = DeletionPolicy::solid_delete();
//! assert!(policy.solid_deleted(&[false, true]));
//! let mut ledger = ErosionLedger::new(2);
//! ledger.record(0, 1.0e-6, 2.0e8, 2.1e8, false);
//! ledger.record(1, 1.0e-6, 3.0e8, 3.2e8, true);
//! let totals = ledger.totals();
//! assert!((totals.eroded_plastic - 300.0).abs() < 1e-9);
//! ```

// Rust guideline compliant 2026-02-21

/// Solid failure action mirroring `Ifail_so` (`/FAIL/TAB1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolidFailAction {
    /// Delete the element when any point fails (`Ifail_so = 1`).
    Delete,
    /// Zero failed-point deviatoric stress, keep element (`Ifail_so = 2`).
    ZeroDeviatoric,
}

/// Shell failure action mirroring `Ifail_sh` (`/FAIL/TAB1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellFailAction {
    /// One failed point deletes the element (`Ifail_sh = 1`).
    AnyPoint,
    /// All layers must fail; broken layers zero stress (`Ifail_sh = 2`).
    AllLayers,
    /// All layers must fail; stresses kept meanwhile (`Ifail_sh = 3`).
    AllLayersKeepStress,
}

/// Element deletion policy from failure-card flags.
#[derive(Debug, Clone, Copy)]
pub struct DeletionPolicy {
    /// Solid point-to-element reduction rule.
    pub solid: SolidFailAction,
    /// Shell point-to-element reduction rule.
    pub shell: ShellFailAction,
    /// Failed-thickness ratio replacing the all-layers rule (shells).
    ///
    /// Zero disables the ratio (pure flag semantics); positive values
    /// delete once adjacent failed layers from one skin exceed it
    /// (Reference Guide `/FAIL/TAB1` comment 7).
    pub thick_fail: f64,
}

impl DeletionPolicy {
    /// Solid policy deleting on any failed point.
    pub fn solid_delete() -> Self {
        Self {
            solid: SolidFailAction::Delete,
            shell: ShellFailAction::AnyPoint,
            thick_fail: 0.0,
        }
    }

    /// Shell policy needing every layer plus a thickness ratio.
    ///
    /// Mirrors the ELEM_SAMP deck (`Ifail_sh = 3`, `Thickfail = 0.4`).
    pub fn shell_layers() -> Self {
        Self {
            solid: SolidFailAction::ZeroDeviatoric,
            shell: ShellFailAction::AllLayersKeepStress,
            thick_fail: 0.4,
        }
    }

    /// Reduces solid point flags to an element verdict.
    ///
    /// Returns true when the policy deletes the element: any failed
    /// point with [`SolidFailAction::Delete`], never with
    /// [`SolidFailAction::ZeroDeviatoric`] (stress-side handling).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::deletion::DeletionPolicy;
    ///
    /// let policy = DeletionPolicy::solid_delete();
    /// assert!(!policy.solid_deleted(&[false, false]));
    /// assert!(policy.solid_deleted(&[false, true]));
    /// ```
    pub fn solid_deleted(&self, point_failed: &[bool]) -> bool {
        match self.solid {
            SolidFailAction::Delete => point_failed.iter().any(|flag| *flag),
            SolidFailAction::ZeroDeviatoric => false,
        }
    }

    /// Reduces shell layer flags to an element verdict.
    ///
    /// With a positive `thick_fail` ratio and layer thicknesses, rupture
    /// trips once adjacent failed layers from either skin exceed the
    /// ratio of total thickness (comment 7). Without thicknesses the
    /// flag decides: any point (= 1) or every layer (= 2, = 3).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::fracture::deletion::DeletionPolicy;
    ///
    /// let mut policy = DeletionPolicy::shell_layers();
    /// let layers = [true, true, false, false, false];
    /// let thick = [0.2, 0.2, 0.2, 0.2, 0.2];
    /// assert!(!policy.shell_deleted(&layers, Some(&thick)));
    /// policy.thick_fail = 0.4;
    /// let broken = [true, true, true, false, false];
    /// assert!(policy.shell_deleted(&broken, Some(&thick)));
    /// ```
    pub fn shell_deleted(&self, layer_failed: &[bool], layer_thickness: Option<&[f64]>) -> bool {
        if layer_failed.is_empty() {
            return false;
        }
        if self.thick_fail > 0.0
            && let Some(thick) = layer_thickness
            && thick.len() == layer_failed.len()
        {
            return thick_ratio_tripped(layer_failed, thick, self.thick_fail);
        }
        match self.shell {
            ShellFailAction::AnyPoint => layer_failed.iter().any(|flag| *flag),
            ShellFailAction::AllLayers | ShellFailAction::AllLayersKeepStress => {
                layer_failed.iter().all(|flag| *flag)
            }
        }
    }
}

/// Tests the failed-thickness ratio from either skin.
///
/// Sums adjacent failed layers inward from each skin over the total
/// thickness; trips when either run strictly exceeds the ratio.
fn thick_ratio_tripped(failed: &[bool], thick: &[f64], ratio: f64) -> bool {
    let total: f64 = thick.iter().sum();
    if total <= 0.0 {
        return failed.iter().all(|flag| *flag);
    }
    let run_from = |range: Vec<usize>| {
        let mut sum = 0.0;
        for i in range {
            if !failed[i] {
                break;
            }
            sum += thick[i].max(0.0);
        }
        sum / total
    };
    let forward: Vec<usize> = (0..failed.len()).collect();
    let backward: Vec<usize> = (0..failed.len()).rev().collect();
    run_from(forward) > ratio || run_from(backward) > ratio
}

/// Per-element energy books split by erosion state.
///
/// Accumulates plastic work and internal energy (J) per element id, kept
/// in the active column until the element erodes, then moved wholesale
/// to the eroded column. Recording is idempotent per step: later records
/// for the same element overwrite the earlier ones, so drivers record
/// cumulative totals and the ledger never double-counts.
#[derive(Debug, Clone)]
pub struct ErosionLedger {
    /// Cumulative plastic work per element in J.
    plastic: Vec<f64>,
    /// Cumulative internal energy per element in J.
    internal: Vec<f64>,
    /// Erosion flags, true once eroded (irreversible).
    eroded: Vec<bool>,
}

impl ErosionLedger {
    /// Creates a ledger for `num_elements` live elements.
    pub fn new(num_elements: usize) -> Self {
        Self {
            plastic: vec![0.0; num_elements],
            internal: vec![0.0; num_elements],
            eroded: vec![false; num_elements],
        }
    }

    /// Counts the elements.
    pub fn len(&self) -> usize {
        self.eroded.len()
    }

    /// Reports whether the ledger holds no elements.
    pub fn is_empty(&self) -> bool {
        self.eroded.is_empty()
    }

    /// Records cumulative energies for one element.
    ///
    /// Takes the element id, reference volume (m^3), cumulative plastic
    /// work density and internal energy density (J/m^3), and whether the
    /// element is eroded after this step. Erosion is irreversible: once
    /// recorded eroded, the element stays in the eroded column.
    ///
    /// # Panics
    ///
    /// Panics when `element` is out of range.
    pub fn record(
        &mut self,
        element: usize,
        volume: f64,
        plastic_density: f64,
        internal_density: f64,
        eroded: bool,
    ) {
        self.plastic[element] = volume.max(0.0) * plastic_density.max(0.0);
        self.internal[element] = volume.max(0.0) * internal_density.max(0.0);
        self.eroded[element] |= eroded;
    }

    /// Reports whether an element has eroded.
    ///
    /// # Panics
    ///
    /// Panics when `element` is out of range.
    pub fn is_eroded(&self, element: usize) -> bool {
        self.eroded[element]
    }

    /// Sums the energy books into active/eroded totals.
    pub fn totals(&self) -> LedgerTotals {
        let mut out = LedgerTotals::default();
        for ((plastic, internal), eroded) in self
            .plastic
            .iter()
            .zip(self.internal.iter())
            .zip(self.eroded.iter())
        {
            if *eroded {
                out.eroded_plastic += plastic;
                out.eroded_internal += internal;
            } else {
                out.active_plastic += plastic;
                out.active_internal += internal;
            }
        }
        out
    }

    /// Residual of the conservation identity (always zero).
    ///
    /// Returns `(active + eroded) - grand`, which vanishes by
    /// construction; drivers assert it to catch ledger misuse. Takes the
    /// expected grand totals of plastic work and internal energy (J).
    pub fn conservation_residual(&self, grand_plastic: f64, grand_internal: f64) -> (f64, f64) {
        let totals = self.totals();
        (
            totals.active_plastic + totals.eroded_plastic - grand_plastic,
            totals.active_internal + totals.eroded_internal - grand_internal,
        )
    }
}

/// Active/eroded energy totals in J.
#[derive(Debug, Clone, Copy, Default)]
pub struct LedgerTotals {
    /// Plastic work held by active elements.
    pub active_plastic: f64,
    /// Plastic work carried away by eroded elements.
    pub eroded_plastic: f64,
    /// Internal energy held by active elements.
    pub active_internal: f64,
    /// Internal energy carried away by eroded elements.
    pub eroded_internal: f64,
}

impl LedgerTotals {
    /// Sums both columns of plastic work.
    pub fn grand_plastic(&self) -> f64 {
        self.active_plastic + self.eroded_plastic
    }

    /// Sums both columns of internal energy.
    pub fn grand_internal(&self) -> f64 {
        self.active_internal + self.eroded_internal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_policy_splits_delete_and_zeroing() {
        let delete = DeletionPolicy::solid_delete();
        assert!(delete.solid_deleted(&[true]));
        assert!(!delete.solid_deleted(&[false, false]));
        let zero = DeletionPolicy {
            solid: SolidFailAction::ZeroDeviatoric,
            ..delete
        };
        assert!(!zero.solid_deleted(&[true, true]));
    }

    #[test]
    fn shell_flags_reduce_without_thickness() {
        let any = DeletionPolicy::solid_delete();
        assert!(any.shell_deleted(&[false, true], None));
        assert!(!any.shell_deleted(&[false, false], None));
        assert!(!any.shell_deleted(&[], None));
        let all = DeletionPolicy {
            shell: ShellFailAction::AllLayers,
            thick_fail: 0.0,
            ..any
        };
        assert!(!all.shell_deleted(&[true, false], None));
        assert!(all.shell_deleted(&[true, true], None));
    }

    #[test]
    fn thick_ratio_counts_adjacent_skins_only() {
        let policy = DeletionPolicy::shell_layers();
        let thick = [0.2, 0.2, 0.2, 0.2, 0.2];
        // Two adjacent skins layers = 0.4, not strictly above: survives.
        assert!(!policy.shell_deleted(&[true, true, false, false, false], Some(&thick)));
        // Three from the skin = 0.6: ruptures.
        assert!(policy.shell_deleted(&[true, true, true, false, false], Some(&thick)));
        // Same count split across a sound layer: the run breaks.
        assert!(!policy.shell_deleted(&[true, true, false, true, true], Some(&thick)));
        // Either skin may carry the run.
        assert!(policy.shell_deleted(&[false, false, true, true, true], Some(&thick)));
        // Length mismatch falls back to flag semantics (all-layers here).
        assert!(policy.shell_deleted(&[true, true], Some(&thick[..1])));
        assert!(!policy.shell_deleted(&[true, false], Some(&thick[..1])));
    }

    #[test]
    fn ledger_splits_and_conserves() {
        let mut ledger = ErosionLedger::new(3);
        assert_eq!(ledger.len(), 3);
        assert!(!ledger.is_empty());
        ledger.record(0, 2.0e-6, 1.0e8, 1.1e8, false);
        ledger.record(1, 1.0e-6, 3.0e8, 3.2e8, true);
        // Re-record overwrites without double counting.
        ledger.record(0, 2.0e-6, 1.5e8, 1.6e8, false);
        let totals = ledger.totals();
        assert!((totals.active_plastic - 300.0).abs() < 1e-9);
        assert!((totals.eroded_plastic - 300.0).abs() < 1e-9);
        assert!((totals.active_internal - 320.0).abs() < 1e-9);
        assert!((totals.eroded_internal - 320.0).abs() < 1e-9);
        assert!(!ledger.is_eroded(0));
        assert!(ledger.is_eroded(1));
        // Erosion never heals through later records.
        ledger.record(1, 1.0e-6, 3.0e8, 3.2e8, false);
        assert!(ledger.is_eroded(1));
        let (rp, ri) = ledger.conservation_residual(600.0, 640.0);
        assert!(rp.abs() < 1e-9 && ri.abs() < 1e-9);
    }

    #[test]
    fn ledger_clamps_degenerate_inputs() {
        let mut ledger = ErosionLedger::new(1);
        ledger.record(0, -1.0, -5.0, -7.0, false);
        let totals = ledger.totals();
        assert_eq!(totals.grand_plastic(), 0.0);
        assert_eq!(totals.grand_internal(), 0.0);
    }
}
