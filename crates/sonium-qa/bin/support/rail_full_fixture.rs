//! Explicit hollow steel rail and contact fixture for full progressive crush.
//!
//! Retains the original nominal length,outer section,elastic constants,density
//! and crusher speed. The original elastic solid-column leg stays separate.
//! This named extension adds a two-millimeter wall,small geometric trigger,
//! explicit Voce plasticity and sixty-percent stroke between clamped grips.
//! Fixture construction alone does not prove progressive folding or parity.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    RadBoundaryCondition, RadExtendedCase, RadExtra, RadGrnodSurfs, RadInterType7, RadInterface,
    RadMaterial, RadMesh, RadNodeSet, RadPlasticMaterial, RadTabulatedHardening, RadThRequest,
    contact_surface::{RadSurfSegments, brick_exterior, write_contact_starter},
};
use std::collections::BTreeMap;

/// Full hollow rail fixture identifier,distinct from elastic compression.
pub const NAME: &str = "rail_crush_full_plastic";

/// Full plastic rail with the original physical two-platen boundary conditions.
pub const PLATEN_NAME: &str = "rail_crush_full_platens";

/// Construct full-stroke crushing between all-body frictionless platens.
///
/// Retains the gripped control's hollow mesh, imperfection, material and
/// full duration. Restores the original fixed anvil and moving crusher,
/// removes transverse clamps, and makes every structural node eligible
/// for either unilateral wall. The massless driver belongs to neither
/// wall group. Shared wall eligibility must be admitted by an actual
/// reference run; no fold or native-parity result follows from rendering.
///
/// # Errors
/// Rejects invalid refinement or physical/contact fixture validation.
pub fn full_platen_case(
    factor: usize,
) -> Result<(RadExtendedCase, Vec<RadSurfSegments>, RadExtra), Box<dyn std::error::Error>> {
    let (mut case, surfaces, extra) = full_case(factor)?;
    let driver = case.base.walls[0].driver_node;
    case.base.name = PLATEN_NAME.into();
    case.base.title = "full plastic hollow steel rail between all-body frictionless platens".into();
    case.base.bcs.clear();
    case.base.node_sets = vec![RadNodeSet {
        name: "all structural rail nodes for both physical platens".into(),
        nodes: case.base.th_nodes.clone(),
    }];
    case.base.walls = super::rail_crush_case().walls;
    for wall in &mut case.base.walls {
        wall.secondary_set = Some(0);
        if wall.velocity_m_per_s.is_some() {
            wall.driver_node = driver;
        }
    }
    case.validate()?;
    write_contact_starter(&case, &surfaces, &extra)?;
    Ok((case, surfaces, extra))
}

/// Construct the full-stroke hollow rail with explicit self-contact topology.
///
/// Root and moving grip transverse translations are clamped; the frictionless
/// moving plane loads tip nodes only, representing an end grip rather than
/// an infinite platen acting on every future fold. Gauge material is free.
/// No erosion,mass scaling or measured-alloy claim is introduced. Numerical
/// self-contact is defined on all true solid boundary faces.
///
/// # Errors
/// Rejects refinement outside1..=4,invalid physical fixture or contact cards.
pub fn full_case(
    factor: usize,
) -> Result<(RadExtendedCase, Vec<RadSurfSegments>, RadExtra), Box<dyn std::error::Error>> {
    if !(1..=4).contains(&factor) {
        return Err("rail refinement in1..=4 required".into());
    }
    let mut base = super::rail_crush_case();
    let axial = 20 * factor;
    let cross = 8 * factor;
    let mut grid = Vec::with_capacity(cross + 1);
    for i in 0..=cross {
        let value = if i <= factor {
            -0.025 + 0.002 * i as f64 / factor as f64
        } else if i <= 7 * factor {
            -0.023 + 0.046 * (i - factor) as f64 / (6 * factor) as f64
        } else {
            0.023 + 0.002 * (i - 7 * factor) as f64 / factor as f64
        };
        grid.push(value);
    }
    let mut mesh = RadMesh {
        nodes: Vec::new(),
        bricks: Vec::new(),
    };
    let mut ids = BTreeMap::new();
    for i in 0..axial {
        for j in 0..cross {
            for k in 0..cross {
                if j >= factor && j < 7 * factor && k >= factor && k < 7 * factor {
                    continue;
                }
                let corners = [
                    (i, j, k),
                    (i + 1, j, k),
                    (i + 1, j + 1, k),
                    (i, j + 1, k),
                    (i, j, k + 1),
                    (i + 1, j, k + 1),
                    (i + 1, j + 1, k + 1),
                    (i, j + 1, k + 1),
                ];
                let mut brick = [0; 8];
                for (slot, key) in corners.into_iter().enumerate() {
                    brick[slot] = *ids.entry(key).or_insert_with(|| {
                        let x = 0.2 * key.0 as f64 / axial as f64;
                        // Smooth small inward trigger; exactly zero at both
                        // grips. Amplitude40um is2% of physical wall thickness.
                        let trigger = (std::f64::consts::PI * x / 0.2).sin().powi(2)
                            * (-0.5 * ((x - 0.17) / 0.01).powi(2)).exp();
                        let scale = 1.0 - 0.00004 / 0.025 * trigger;
                        mesh.nodes
                            .push([x, grid[key.1] * scale, grid[key.2] * scale]);
                        mesh.nodes.len()
                    });
                }
                mesh.bricks.push(brick);
            }
        }
    }
    let mut root = Vec::new();
    let mut tip = Vec::new();
    for (key, &id) in &ids {
        if key.0 == 0 {
            root.push(id);
        }
        if key.0 == axial {
            tip.push(id);
        }
    }
    let original_nodes = (1..=mesh.nodes.len()).collect::<Vec<_>>();
    let driver = mesh.add_node([0.2, 0.0, 0.0]);
    base.name = NAME.into();
    base.title = "full plastic hollow steel rail between end grips".into();
    base.mesh = mesh;
    base.node_sets[0].nodes = root;
    base.node_sets[1].nodes = tip;
    base.bcs = vec![
        RadBoundaryCondition {
            name: "fixed root grip".into(),
            set: 0,
            lock: [true; 6],
        },
        RadBoundaryCondition {
            name: "moving grip transverse clamp".into(),
            set: 1,
            lock: [false, true, true, true, true, true],
        },
    ];
    base.walls.remove(0);
    base.walls[0].driver_node = Some(driver);
    base.walls[0].secondary_set = Some(1);
    base.stop_time_s = 1.2;
    base.th_period_s = 0.005;
    base.anim_period_s = 0.12;
    for f in &mut base.functions {
        f.points = vec![(0.0, 1.0), (1.2, 1.0)];
    }
    base.th_nodes = original_nodes;
    base.probes.clear();
    let mut case = RadExtendedCase::wrap_elastic(base);
    let m = case.base.material;
    case.material = RadMaterial::Plastic(RadPlasticMaterial::Tabulated(RadTabulatedHardening {
        youngs_modulus_pa: m.youngs_modulus_pa,
        poisson_ratio: m.poisson_ratio,
        density_kg_per_m3: m.density_kg_per_m3,
        points: (0..=800)
            .map(|i| {
                let ep = i as f64 * 0.005;
                (ep, 250e6 + 300e6 * (1.0 - (-5.0 * ep).exp()))
            })
            .collect(),
    }));
    case.th_brick = RadThRequest::brick_stress((1..=case.base.mesh.bricks.len()).collect());
    case.validate()?;
    let surface = brick_exterior(&case, 100, "outer and cavity rail contact faces".into())?;
    let mut contact = RadInterType7::basic(300, "rail self contact", 200, 100);
    // Constant0.5mm gap is smaller than the2mm physical wall,so opposing
    // undeformed skins do not start in contact. Actual startup still needs
    // independent engine admission and a zero-reference-force native check.
    contact.gapmin_m = 0.0005;
    let mut extra = RadExtra::default();
    extra.grnods.push(RadGrnodSurfs {
        id: 200,
        title: "all rail surface nodes".into(),
        surfs: vec![100],
    });
    extra.interfaces.push(RadInterface::Type7(contact));
    let surfaces = vec![surface];
    write_contact_starter(&case, &surfaces, &extra)?;
    Ok((case, surfaces, extra))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_physical_platens_cover_every_future_fold_without_grip_clamps() {
        let (control, _, _) = full_case(1).unwrap();
        let (case, surfaces, extra) = full_platen_case(1).unwrap();
        assert_eq!(case.base.mesh.nodes, control.base.mesh.nodes);
        assert_eq!(case.base.mesh.bricks, control.base.mesh.bricks);
        assert_eq!(case.base.stop_time_s, 1.2);
        assert!(case.base.bcs.is_empty());
        assert_eq!(case.base.walls.len(), 2);
        assert_eq!(case.base.walls[0].normal, [1.0, 0.0, 0.0]);
        assert_eq!(case.base.walls[0].velocity_m_per_s, None);
        assert_eq!(case.base.walls[1].normal, [-1.0, 0.0, 0.0]);
        assert_eq!(case.base.walls[1].velocity_m_per_s, Some([-0.1, 0.0, 0.0]));
        let driver = case.base.walls[1].driver_node.unwrap();
        for wall in &case.base.walls {
            let ids = &case.base.node_sets[wall.secondary_set.unwrap()].nodes;
            assert_eq!(ids, &case.base.th_nodes);
            assert!(!ids.contains(&driver));
            assert!(
                case.base
                    .mesh
                    .bricks
                    .iter()
                    .flatten()
                    .all(|n| ids.contains(n))
            );
        }
        assert!(
            write_contact_starter(&case, &surfaces, &extra)
                .unwrap()
                .contains("/RWALL/PLANE/2/1")
        );
    }

    #[test]
    fn hollow_rail_retains_nominal_inputs_and_full_stroke_with_true_cavity() {
        let original = super::super::rail_crush_case();
        let (case, surfaces, extra) = full_case(1).unwrap();
        assert_eq!(
            case.base.material.youngs_modulus_pa,
            original.material.youngs_modulus_pa
        );
        assert_eq!(
            case.base.material.poisson_ratio,
            original.material.poisson_ratio
        );
        assert_eq!(
            case.base.material.density_kg_per_m3,
            original.material.density_kg_per_m3
        );
        assert_eq!(case.base.mesh.bricks.len(), 560);
        assert_eq!(case.base.stop_time_s, 1.2);
        assert_eq!(case.base.walls[0].velocity_m_per_s, Some([-0.1, 0.0, 0.0]));
        assert!((case.base.stop_time_s * 0.1 / 0.2 - 0.6).abs() < 1e-14);
        let driver = case.base.walls[0].driver_node.unwrap();
        assert!(!case.base.th_nodes.contains(&driver));
        assert!(case.base.mesh.bricks.iter().flatten().all(|n| *n != driver));
        for brick in &case.base.mesh.bricks {
            let c: [f64; 3] = std::array::from_fn(|a| {
                brick
                    .iter()
                    .map(|n| case.base.mesh.nodes[n - 1][a])
                    .sum::<f64>()
                    / 8.0
            });
            assert!(c[1].abs() > 0.022 || c[2].abs() > 0.022);
        }
        assert!(surfaces[0].quads.len() > 560);
        assert!(
            write_contact_starter(&case, &surfaces, &extra)
                .unwrap()
                .contains("/INTER/TYPE7/300")
        );
        assert!(full_case(0).is_err());
        assert!(full_case(5).is_err());
    }
}
