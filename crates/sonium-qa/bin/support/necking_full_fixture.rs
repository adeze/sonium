//! Complete tensile deformation extending the original elastic bar fixture.
//!
//! Retains original nominal geometry, elastic constants, density, root clamp
//! and full one-second duration. Adds a smooth one-percent area imperfection,
//! an explicit rate-independent Voce hardening curve and a pulling grip.
//! This is a matched numerical constitutive model, not measured alloy data.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    RadBoundaryCondition, RadExtendedCase, RadMaterial, RadPlasticMaterial, RadProbe,
    RadTabulatedHardening, RadThRequest, RadWall, cantilever_brick_mesh,
};

pub(crate) const NAME: &str = "necking_bar_full_plastic";
pub(crate) const SPEED: f64 = 0.04;

/// Build the full one-second plastic tensile bar with a deterministic imperfection.
///
/// The original elastic force-controlled subcase remains a separate requirement.
/// Both grips hold transverse motion; a tip-only moving plane controls axial
/// extension. The gauge section contracts freely. No failure, erosion, mass
/// scaling or material softening is introduced.
///
/// # Errors
/// Rejects refinement outside1..=4 or an invalid generated physical fixture.
pub(crate) fn full_case(factor: usize) -> Result<RadExtendedCase, Box<dyn std::error::Error>> {
    if !(1..=4).contains(&factor) {
        return Err("necking refinement factor must be in1..=4".into());
    }
    let mut base = super::necking_bar_case();
    let (mut mesh, root, tip, center) =
        cantilever_brick_mesh(10 * factor, 2 * factor, 2 * factor, 0.1, 0.01, 0.01);
    // Subtract the endpoint Gaussian value so both physical grip areas retain
    // their exact original size while the middle area is one percent smaller.
    let endpoint = (-12.5_f64).exp();
    for point in &mut mesh.nodes {
        let profile =
            ((-0.5 * ((point[0] - 0.05) / 0.01).powi(2)).exp() - endpoint) / (1.0 - endpoint);
        let scale = (1.0 - 0.01 * profile).sqrt();
        point[1] *= scale;
        point[2] *= scale;
    }
    let nodes = (1..=mesh.num_nodes()).collect::<Vec<_>>();
    let driver = mesh.add_node([0.1, 0.0, 0.0]);
    base.name = NAME.into();
    base.title = "original nominal steel bar: full plastic tensile deformation".into();
    base.mesh = mesh;
    base.node_sets[0].nodes = root;
    base.node_sets[1].nodes = tip;
    // Symmetric transverse grip constraints keep boundary contraction from
    // replacing gauge-section necking. Axial pull is still imposed by contact.
    base.bcs.push(RadBoundaryCondition {
        name: "pulling grip transverse clamp".into(),
        set: 1,
        lock: [false, true, true, true, true, true],
    });
    base.cloads.clear();
    base.walls = vec![RadWall {
        name: "displacement-controlled pulling grip".into(),
        normal: [1.0, 0.0, 0.0],
        point_m: [0.1, 0.0, 0.0],
        friction: 0.0,
        velocity_m_per_s: Some([SPEED, 0.0, 0.0]),
        driver_node: Some(driver),
        secondary_set: Some(1),
        search_distance_m: 0.0,
    }];
    base.th_nodes = nodes;
    base.probes = [format!("DX/{center}"), "FNX/RWALL/1".into()]
        .into_iter()
        .map(|channel| RadProbe {
            label: channel.clone(),
            channel,
            expected: 0.0,
            abs_tol: 0.0,
            rel_tol: 0.0,
        })
        .collect();
    let mut case = RadExtendedCase::wrap_elastic(base);
    let m = case.base.material;
    // Use the same explicit Voce parameters as the existing slender-bar native
    // necking reference:250MPa yield,300MPa saturation increment,b=5.
    // The curve extends beyond the intended plastic strain without softening.
    case.material = RadMaterial::Plastic(RadPlasticMaterial::Tabulated(RadTabulatedHardening {
        youngs_modulus_pa: m.youngs_modulus_pa,
        poisson_ratio: m.poisson_ratio,
        density_kg_per_m3: m.density_kg_per_m3,
        points: (0..=400)
            .map(|i| {
                let ep = i as f64 * 0.005;
                (ep, 250e6 + 300e6 * (1.0 - (-5.0 * ep).exp()))
            })
            .collect(),
    }));
    case.th_brick = RadThRequest::brick_stress((1..=case.base.mesh.bricks.len()).collect());
    case.validate()?;
    Ok(case)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plastic_leg_retains_original_nominal_geometry_material_clamp_and_duration() {
        let old = super::super::necking_bar_case();
        let new = full_case(1).unwrap();
        assert_eq!(old.stop_time_s, new.base.stop_time_s);
        assert_eq!(new.base.stop_time_s, 1.0);
        assert_eq!(old.bcs[0].lock, new.base.bcs[0].lock);
        assert_eq!(old.mesh.bricks, new.base.mesh.bricks);
        let RadMaterial::Plastic(RadPlasticMaterial::Tabulated(m)) = new.material else {
            panic!("expected explicit Voce hardening");
        };
        assert_eq!(old.material.youngs_modulus_pa, m.youngs_modulus_pa);
        assert_eq!(old.material.poisson_ratio, m.poisson_ratio);
        assert_eq!(old.material.density_kg_per_m3, m.density_kg_per_m3);
        assert!(m.points.windows(2).all(|p| p[1].1 >= p[0].1));
        for (a, b) in old.mesh.nodes.iter().zip(&new.base.mesh.nodes) {
            assert_eq!(a[0], b[0]);
            if a[0] == 0.0 || a[0] == 0.1 {
                assert_eq!(a, b);
            }
        }
        assert_eq!(new.base.walls[0].secondary_set, Some(1));
        assert_eq!(new.base.bcs[1].set, 1);
        assert_eq!(new.base.bcs[1].lock, [false, true, true, true, true, true]);
        assert_eq!(new.base.walls[0].normal, [1.0, 0.0, 0.0]);
        assert_eq!(new.base.walls[0].velocity_m_per_s, Some([SPEED, 0.0, 0.0]));
        assert!(new.base.cloads.is_empty());
    }

    #[test]
    fn refinement_keeps_all_original_bar_nodes_separate_from_the_massless_grip_driver() {
        for factor in 1..=4 {
            let case = full_case(factor).unwrap();
            let driver = case.base.walls[0].driver_node.unwrap();
            assert!(!case.base.th_nodes.contains(&driver));
            assert!(!case.base.node_sets[1].nodes.contains(&driver));
            assert_eq!(case.base.mesh.bricks.len(), 40 * factor.pow(3));
            assert_eq!(
                case.base.mesh.nodes.len(),
                (10 * factor + 1) * (2 * factor + 1).pow(2) + 1
            );
        }
        assert!(full_case(0).is_err());
    }
}
