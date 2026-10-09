//! Bounded FEM stress-life fatigue evidence for the 2D plane-stress family.
//!
//! Each block is evaluated from two full-density P1 FEM load states.  The
//! maximum element amplitude of the recovered stress difference feeds a
//! Basquin/Miner calculation.  This keeps the structural part authoritative
//! while making the remaining S-N assumptions explicit in the result
//! provenance.
//!
//! Studies may carry a [`PublishedSnCurve`] (Eurocode 3 detail categories)
//! instead of the legacy single-slope Basquin parameters.  The published
//! path works on the stress range (twice the recovered amplitude) with the
//! code's bilinear knee and cut-off; ranges below the cut-off stress are
//! non-damaging.

use serde::{Deserialize, Serialize};

use crate::fem_topology::solve_fem_static_stress_state;
use crate::{
    FemTopology2DGeometry, FemTopology2DMaterial, FemTopology2DStudy,
    StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption,
};

/// One minimum/maximum FEM load pair and its cycle count.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FemFatigue2DLoadBlock {
    pub minimum_force_n: [f64; 2],
    pub maximum_force_n: [f64; 2],
    pub applied_cycles: f64,
}

/// Reference life N_C of EN 1993-1-9 detail categories, in cycles.
///
/// The detail category stress range is defined at this life (clause 7,
/// Figure 7.1). Changing it rescales every published allowable life.
pub const EUROCODE3_REFERENCE_CYCLES: f64 = 2.0e6;

/// Knee life N_D (constant-amplitude fatigue limit) of EN 1993-1-9, cycles.
///
/// The inverse slope changes from 3 to 5 at this life. Changing it moves
/// the knee stress of every published curve.
pub const EUROCODE3_KNEE_CYCLES: f64 = 5.0e6;

/// Cut-off life N_L of EN 1993-1-9, in cycles.
///
/// Stress ranges below the corresponding cut-off stress do not
/// participate in fatigue damage. Changing it moves the damage cut-off.
pub const EUROCODE3_CUTOFF_CYCLES: f64 = 1.0e8;

/// Inverse slope above the knee for EN 1993-1-9 steel details.
///
/// Applies to stress ranges at or above the knee (constant-amplitude
/// fatigue limit) stress.
pub const EUROCODE3_SLOPE_ABOVE_KNEE: f64 = 3.0;

/// Inverse slope between knee and cut-off for EN 1993-1-9 steel details.
///
/// Applies to stress ranges strictly below the knee stress and at or
/// above the cut-off stress.
pub const EUROCODE3_SLOPE_BELOW_KNEE: f64 = 5.0;

/// Published stress-life curve with knee and cut-off.
///
/// The curve maps a nominal stress range to an allowable life with the
/// EN 1993-1-9 bilinear shape: inverse slope `slope_above_knee` from
/// `reference_cycles` to `knee_cycles`, then `slope_below_knee` to
/// `cutoff_cycles`, and infinite life below the cut-off stress. Only
/// the detail category is material-specific; the lives and slopes are
/// the code values above.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PublishedSnCurve {
    /// Detail-category stress range at `reference_cycles`, in Pa.
    pub detail_category_stress_range_pa: f64,
    /// Life at the detail category (N_C), in cycles.
    pub reference_cycles: f64,
    /// Knee life (constant-amplitude fatigue limit, N_D), in cycles.
    pub knee_cycles: f64,
    /// Cut-off life (N_L), in cycles.
    pub cutoff_cycles: f64,
    /// Inverse slope at or above the knee stress.
    pub slope_above_knee: f64,
    /// Inverse slope below the knee stress down to cut-off.
    pub slope_below_knee: f64,
}

impl PublishedSnCurve {
    /// Eurocode 3 detail category from its MPa value.
    ///
    /// `detail_category_mpa` is the nominal stress range in MPa at
    /// 2e6 cycles (for example 160.0 for plain material, 71.0 for a
    /// transverse butt weld, 36.0 for a cruciform joint). The standard
    /// table runs 160 down to 36; any positive value is accepted so
    /// interpolated assessments stay representable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_sim_structural_mechanics::PublishedSnCurve;
    ///
    /// let curve = PublishedSnCurve::eurocode3(160.0).unwrap();
    /// assert_eq!(curve.allowable_cycles(160.0e6), 2.0e6);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfiguration` when the category is non-finite
    /// or not positive.
    pub fn eurocode3(detail_category_mpa: f64) -> Result<Self, StructuralMechanicsError> {
        if !detail_category_mpa.is_finite() || detail_category_mpa <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "published_sn_curve.detail_category_mpa",
                value: detail_category_mpa.to_string(),
            });
        }
        Ok(Self {
            detail_category_stress_range_pa: detail_category_mpa * 1.0e6,
            reference_cycles: EUROCODE3_REFERENCE_CYCLES,
            knee_cycles: EUROCODE3_KNEE_CYCLES,
            cutoff_cycles: EUROCODE3_CUTOFF_CYCLES,
            slope_above_knee: EUROCODE3_SLOPE_ABOVE_KNEE,
            slope_below_knee: EUROCODE3_SLOPE_BELOW_KNEE,
        })
    }

    /// Knee (constant-amplitude fatigue limit) stress range, in Pa.
    ///
    /// Follows the above-knee branch down to `knee_cycles`.
    pub fn knee_stress_range_pa(&self) -> f64 {
        self.detail_category_stress_range_pa
            * (self.reference_cycles / self.knee_cycles).powf(1.0 / self.slope_above_knee)
    }

    /// Cut-off stress range in Pa.
    ///
    /// Follows the below-knee branch down to `cutoff_cycles`. Ranges
    /// below this stress are non-damaging.
    pub fn cutoff_stress_range_pa(&self) -> f64 {
        self.knee_stress_range_pa()
            * (self.knee_cycles / self.cutoff_cycles).powf(1.0 / self.slope_below_knee)
    }

    /// Allowable life for a nominal stress range in Pa.
    ///
    /// Returns infinity for non-positive ranges and for ranges below
    /// the cut-off stress. The knee stress itself belongs to the
    /// above-knee branch; both branches agree there by construction.
    pub fn allowable_cycles(&self, stress_range_pa: f64) -> f64 {
        if !stress_range_pa.is_finite() || stress_range_pa <= 0.0 {
            return f64::INFINITY;
        }
        if stress_range_pa >= self.knee_stress_range_pa() {
            self.reference_cycles
                * (self.detail_category_stress_range_pa / stress_range_pa)
                    .powf(self.slope_above_knee)
        } else if stress_range_pa >= self.cutoff_stress_range_pa() {
            self.knee_cycles
                * (self.knee_stress_range_pa() / stress_range_pa).powf(self.slope_below_knee)
        } else {
            f64::INFINITY
        }
    }
}

/// Configuration for the P1 plane-stress FEM fatigue reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemFatigue2DStudy {
    pub name: String,
    pub geometry: FemTopology2DGeometry,
    pub material: FemTopology2DMaterial,
    pub reference_stress_pa: f64,
    pub reference_cycles: f64,
    pub sn_slope: f64,
    /// Published S-N curve; overrides the single-slope parameters above.
    ///
    /// When set, each block's allowable life comes from this curve fed
    /// with the stress range (twice the recovered amplitude). When
    /// `None`, the legacy single-slope Basquin law on the amplitude
    /// applies.
    pub sn_curve: Option<PublishedSnCurve>,
    pub load_blocks: Vec<FemFatigue2DLoadBlock>,
    pub damage_limit: f64,
}

impl Default for FemFatigue2DStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl FemFatigue2DStudy {
    pub fn introductory() -> Self {
        let topology = FemTopology2DStudy::introductory();
        Self {
            name: "p1-plane-stress-fem-fatigue".to_string(),
            geometry: topology.geometry,
            material: topology.material,
            reference_stress_pa: 100.0e6,
            reference_cycles: 1.0e6,
            sn_slope: 3.0,
            sn_curve: None,
            load_blocks: vec![
                FemFatigue2DLoadBlock {
                    minimum_force_n: [0.0, -40.0],
                    maximum_force_n: [0.0, -120.0],
                    applied_cycles: 100_000.0,
                },
                FemFatigue2DLoadBlock {
                    minimum_force_n: [0.0, -20.0],
                    maximum_force_n: [0.0, -80.0],
                    applied_cycles: 50_000.0,
                },
            ],
            damage_limit: 1.0,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("fem_fatigue.reference_stress_pa", self.reference_stress_pa),
            ("fem_fatigue.reference_cycles", self.reference_cycles),
            ("fem_fatigue.sn_slope", self.sn_slope),
            ("fem_fatigue.damage_limit", self.damage_limit),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.geometry.nx == 0
            || self.geometry.ny == 0
            || !self.geometry.length_m.is_finite()
            || !self.geometry.height_m.is_finite()
            || !self.geometry.thickness_m.is_finite()
            || self.geometry.length_m <= 0.0
            || self.geometry.height_m <= 0.0
            || self.geometry.thickness_m <= 0.0
            || !self.material.youngs_modulus_pa.is_finite()
            || !self.material.poisson_ratio.is_finite()
            || !self.material.density_kg_m3.is_finite()
            || self.material.youngs_modulus_pa <= 0.0
            || self.material.density_kg_m3 <= 0.0
            || self.material.poisson_ratio <= -1.0
            || self.material.poisson_ratio >= 0.5
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_fatigue.geometry_material",
                value: "non-finite or physically invalid geometry/material".to_string(),
            });
        }
        if let Some(curve) = &self.sn_curve {
            for (field, value) in [
                (
                    "fem_fatigue.detail_category_stress_range_pa",
                    curve.detail_category_stress_range_pa,
                ),
                ("fem_fatigue.sn_reference_cycles", curve.reference_cycles),
                ("fem_fatigue.sn_knee_cycles", curve.knee_cycles),
                ("fem_fatigue.sn_cutoff_cycles", curve.cutoff_cycles),
                ("fem_fatigue.sn_slope_above_knee", curve.slope_above_knee),
                ("fem_fatigue.sn_slope_below_knee", curve.slope_below_knee),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    return Err(StructuralMechanicsError::InvalidConfiguration {
                        field,
                        value: value.to_string(),
                    });
                }
            }
            if !(curve.reference_cycles < curve.knee_cycles
                && curve.knee_cycles < curve.cutoff_cycles)
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "fem_fatigue.sn_curve_lives",
                    value: format!(
                        "reference {} knee {} cutoff {} (must be strictly increasing)",
                        curve.reference_cycles, curve.knee_cycles, curve.cutoff_cycles
                    ),
                });
            }
        }
        if self.load_blocks.is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_fatigue.load_blocks",
                value: "empty".to_string(),
            });
        }
        for block in &self.load_blocks {
            if !block.applied_cycles.is_finite() || block.applied_cycles < 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "fem_fatigue.applied_cycles",
                    value: block.applied_cycles.to_string(),
                });
            }
            if block
                .minimum_force_n
                .into_iter()
                .chain(block.maximum_force_n)
                .any(|value| !value.is_finite())
            {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field: "fem_fatigue.load_blocks",
                    value: "non-finite force".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// FEM and fatigue evidence for one load block.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FemFatigue2DBlockResult {
    pub minimum_force_n: [f64; 2],
    pub maximum_force_n: [f64; 2],
    pub stress_amplitude_pa: f64,
    pub applied_cycles: f64,
    pub allowable_cycles: f64,
    pub damage_increment: f64,
    pub cumulative_damage: f64,
    pub solver_residual: f64,
    pub equilibrium_residual_n: f64,
}

/// Result from the P1 plane-stress FEM fatigue reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemFatigue2DResult {
    pub study: FemFatigue2DStudy,
    pub blocks: Vec<FemFatigue2DBlockResult>,
    pub total_damage: f64,
    pub failed: bool,
    pub failure_cycle: Option<f64>,
    pub max_stress_amplitude_pa: f64,
    pub converged: bool,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Run two assembled FEM states per fatigue block and accumulate Miner damage.
pub fn run_fem_fatigue_2d(
    study: &FemFatigue2DStudy,
) -> Result<FemFatigue2DResult, StructuralMechanicsError> {
    study.validate()?;
    let mut cumulative_damage = 0.0;
    let mut cumulative_cycles = 0.0;
    let mut failure_cycle = None;
    let mut max_stress_amplitude_pa: f64 = 0.0;
    let mut converged = true;
    let mut blocks = Vec::with_capacity(study.load_blocks.len());

    for block in &study.load_blocks {
        let minimum = solve_load_state(study, block.minimum_force_n)?;
        let maximum = solve_load_state(study, block.maximum_force_n)?;
        let stress_amplitude_pa = maximum
            .element_stress
            .iter()
            .zip(&minimum.element_stress)
            .map(|(high, low)| stress_difference_amplitude(*high, *low))
            .fold(0.0, f64::max);
        let allowable_cycles = match &study.sn_curve {
            // Published curves are range-based: the recovered von Mises
            // amplitude is half the stress range.
            Some(curve) => curve.allowable_cycles(2.0 * stress_amplitude_pa),
            None => {
                if stress_amplitude_pa == 0.0 {
                    f64::INFINITY
                } else {
                    study.reference_cycles
                        * (study.reference_stress_pa / stress_amplitude_pa).powf(study.sn_slope)
                }
            }
        };
        let damage_increment = if allowable_cycles.is_infinite() {
            0.0
        } else {
            block.applied_cycles / allowable_cycles
        };
        let previous_damage = cumulative_damage;
        cumulative_damage += damage_increment;
        cumulative_cycles += block.applied_cycles;
        if failure_cycle.is_none()
            && previous_damage < study.damage_limit
            && cumulative_damage >= study.damage_limit
            && damage_increment > 0.0
        {
            failure_cycle = Some(
                cumulative_cycles
                    - block.applied_cycles * (cumulative_damage - study.damage_limit)
                        / damage_increment,
            );
        }
        let solver_residual = minimum.solver_residual.max(maximum.solver_residual);
        let equilibrium_residual_n = minimum
            .free_equilibrium_residual_n
            .max(minimum.force_balance_residual_n)
            .max(maximum.free_equilibrium_residual_n)
            .max(maximum.force_balance_residual_n);
        converged &= solver_residual.is_finite() && equilibrium_residual_n.is_finite();
        max_stress_amplitude_pa = max_stress_amplitude_pa.max(stress_amplitude_pa);
        blocks.push(FemFatigue2DBlockResult {
            minimum_force_n: block.minimum_force_n,
            maximum_force_n: block.maximum_force_n,
            stress_amplitude_pa,
            applied_cycles: block.applied_cycles,
            allowable_cycles,
            damage_increment,
            cumulative_damage,
            solver_residual,
            equilibrium_residual_n,
        });
    }

    Ok(FemFatigue2DResult {
        study: study.clone(),
        blocks,
        total_damage: cumulative_damage,
        failed: cumulative_damage >= study.damage_limit,
        failure_cycle,
        max_stress_amplitude_pa,
        converged,
        model_assumption: StructuralModelAssumption::FatiguePlaneStress2D,
        execution: StructuralExecutionProvenance::cpu_fem_fatigue_2d_reference(),
    })
}

fn solve_load_state(
    study: &FemFatigue2DStudy,
    force_n: [f64; 2],
) -> Result<crate::fem_topology::FemStaticStressState, StructuralMechanicsError> {
    let mut topology = FemTopology2DStudy::introductory();
    topology.geometry = study.geometry;
    topology.material = study.material;
    topology.edge_force_n = force_n;
    if force_n[0].abs() <= f64::EPSILON && force_n[1].abs() <= f64::EPSILON {
        let element_count = study.geometry.nx * study.geometry.ny * 2;
        return Ok(crate::fem_topology::FemStaticStressState {
            element_stress: vec![[0.0; 3]; element_count],
            solver_residual: 0.0,
            free_equilibrium_residual_n: 0.0,
            force_balance_residual_n: 0.0,
        });
    }
    solve_fem_static_stress_state(&topology)
}

fn stress_difference_amplitude(high: [f64; 3], low: [f64; 3]) -> f64 {
    let stress = [high[0] - low[0], high[1] - low[1], high[2] - low[2]];
    0.5 * (stress[0] * stress[0] - stress[0] * stress[1]
        + stress[1] * stress[1]
        + 3.0 * stress[2] * stress[2])
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fem_fatigue_recovers_stress_amplitude_and_damage() {
        let mut study = FemFatigue2DStudy::introductory();
        study.geometry.nx = 4;
        study.geometry.ny = 1;
        study.load_blocks = vec![FemFatigue2DLoadBlock {
            minimum_force_n: [0.0, -20.0],
            maximum_force_n: [0.0, -80.0],
            applied_cycles: 10_000.0,
        }];
        let result = run_fem_fatigue_2d(&study).expect("FEM fatigue reference should converge");

        assert!(result.converged);
        assert!(result.max_stress_amplitude_pa > 0.0);
        assert!(result.total_damage > 0.0);
        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::FatiguePlaneStress2D
        );
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.fem-fatigue-2d.v1"
        );
        assert!(result.blocks[0].equilibrium_residual_n < 1.0e-7);
    }

    #[test]
    fn fem_fatigue_rejects_invalid_blocks() {
        let mut study = FemFatigue2DStudy::introductory();
        study.load_blocks[0].applied_cycles = -1.0;
        assert!(matches!(
            run_fem_fatigue_2d(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "fem_fatigue.applied_cycles",
                ..
            })
        ));
    }
}
