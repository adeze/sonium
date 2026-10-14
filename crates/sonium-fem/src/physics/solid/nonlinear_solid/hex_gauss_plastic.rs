//! Eight-point objective H8 plasticity with projected volumetric strain.
//!
//! Deviatoric strain is evaluated at all eight Gauss points; its volumetric
//! part is replaced by the volume average (B-bar). The force uses the same
//! projection. Fully integrated shear supplies elastic hourglass stiffness;
//! no viscous hourglass term is required. This is a separate formulation from
//! the existing one-point explicit H8 and does not change its behavior.

// Rust guideline compliant 2026-02-21

use super::{
    HexExplicitError, HexExplicitParams, HexForceOut, HexPlasticOut, ObjectivePlasticMaterial,
    hex_jacobian, hex_quadrature, hex_shape_derivatives, physical_gradients,
};
use crate::physics::solid::plasticity::update::voigt_equivalent_increment_c1;
use sonium_compute::explicit_hourglass::bulk_viscosity_q;

#[derive(Debug, Clone)]
struct GaussPoint {
    gradients: [[f64; 3]; 8],
    weight: f64,
}

/// Current H8 geometry at all eight Gauss points.
#[derive(Debug, Clone)]
pub struct HexGaussPlasticGeom {
    points: [GaussPoint; 8],
    /// Integrated current element volume in cubic meters.
    pub volume: f64,
    /// Smallest current physical edge length in meters.
    pub char_length: f64,
}

impl HexGaussPlasticGeom {
    /// Prepare geometry and reject inverted or collapsed integration points.
    ///
    /// # Errors
    /// Rejects nonfinite coordinates, nonpositive Gauss Jacobians, invalid
    /// integrated volume, or collapsed/nonfinite physical edges.
    pub fn new(coordinates: [[f64; 3]; 8]) -> Result<Self, HexExplicitError> {
        if coordinates.iter().flatten().any(|v| !v.is_finite()) {
            return Err(HexExplicitError::new("Gauss H8 coordinates must be finite"));
        }
        let mut points = Vec::with_capacity(8);
        let mut volume = 0.0;
        for (natural, weight) in hex_quadrature() {
            let derivatives = hex_shape_derivatives(&natural);
            let (_, determinant, inverse) = hex_jacobian(&coordinates, &derivatives);
            if !determinant.is_finite() || determinant <= 0.0 {
                return Err(HexExplicitError::new(
                    "Gauss H8 needs eight positive Jacobians",
                ));
            }
            let gradients = physical_gradients(&inverse, &derivatives);
            if gradients.iter().flatten().any(|v| !v.is_finite()) {
                return Err(HexExplicitError::new("Gauss H8 gradients must be finite"));
            }
            let weight = weight * determinant;
            volume += weight;
            points.push(GaussPoint { gradients, weight });
        }
        let edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ];
        let char_length = edges
            .iter()
            .map(|[a, b]| {
                (0..3)
                    .map(|i| (coordinates[*a][i] - coordinates[*b][i]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        if !volume.is_finite() || volume <= 0.0 || !char_length.is_finite() || char_length <= 0.0 {
            return Err(HexExplicitError::new(
                "Gauss H8 volume and edges must be positive finite",
            ));
        }
        Ok(Self {
            points: points
                .try_into()
                .map_err(|_| HexExplicitError::new("eight Gauss points required"))?,
            volume,
            char_length,
        })
    }
}

/// Material histories for all eight H8 integration points.
#[derive(Debug, Clone)]
pub struct HexGaussPlasticState<S> {
    /// Histories in tensor-product Gauss quadrature order.
    pub points: [S; 8],
}

impl<S: Clone> HexGaussPlasticState<S> {
    /// Initialize every integration point from one material state.
    pub fn new(initial: S) -> Self {
        Self {
            points: std::array::from_fn(|_| initial.clone()),
        }
    }

    /// Interpolate material features to the natural element center.
    ///
    /// The eight tensor-product values have equal interpolation weights at
    /// the natural center. This is an observation, not an erosion criterion;
    /// force evaluation reports maxima separately for conservative admission.
    pub fn center_features<M: ObjectivePlasticMaterial<State = S>>(
        &self,
        material: &M,
    ) -> (f64, f64, f64) {
        self.points
            .iter()
            .map(|s| material.erosion_features(s))
            .fold((0.0, 0.0, 0.0), |a, b| {
                (a.0 + b.0 / 8.0, a.1 + b.1 / 8.0, a.2 + b.2 / 8.0)
            })
    }
}

/// Advance objective B-bar material histories and integrate current forces.
///
/// Uses engineering shear in C1 order. Positive bulk viscosity is handled
/// by the same shock kernel as the one-point formulation; viscous hourglass
/// stabilization is omitted because shear is fully integrated. State commits
/// only after every point and the resulting forces validate. The returned
/// stress interpolates to the natural center; erosion features are maxima.
///
/// # Errors
/// Rejects invalid timestep/velocities, nonfinite kinematics, material
/// outputs, damage/features, shock pressure, or force accumulation.
pub fn hex_gauss_plastic_forces<M: ObjectivePlasticMaterial>(
    geom: &HexGaussPlasticGeom,
    state: &mut HexGaussPlasticState<M::State>,
    material: &M,
    velocities: &[f64; 24],
    dt: f64,
    params: &HexExplicitParams,
) -> Result<HexPlasticOut, HexExplicitError> {
    if !dt.is_finite() || dt <= 0.0 || velocities.iter().any(|v| !v.is_finite()) {
        return Err(HexExplicitError::new(
            "Gauss H8 requires finite velocity and positive finite dt",
        ));
    }
    let gradients: [[[f64; 3]; 3]; 8] = std::array::from_fn(|q| {
        std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                (0..8)
                    .map(|n| velocities[3 * n + a] * geom.points[q].gradients[n][b])
                    .sum()
            })
        })
    });
    let average_trace = gradients
        .iter()
        .zip(&geom.points)
        .map(|(g, p)| (g[0][0] + g[1][1] + g[2][2]) * p.weight)
        .sum::<f64>()
        / geom.volume;
    if !average_trace.is_finite() {
        return Err(HexExplicitError::new("nonfinite B-bar volumetric rate"));
    }
    let bulk_q = bulk_viscosity_q(
        params.density,
        params.sound_speed,
        geom.char_length,
        params.bulk_c0,
        params.bulk_c1,
        average_trace,
    );
    let mut trial = state.clone();
    let mut stresses = [[0.0; 6]; 8];
    let mut mean_stress = [0.0; 6];
    let mut mean_dep = 0.0;
    let mut maximum_rate = 0.0_f64;
    let mut features = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut damage_increment = 0.0_f64;
    let mut yielding = false;
    for (q, g) in gradients.iter().enumerate() {
        let correction = (average_trace - g[0][0] - g[1][1] - g[2][2]) / 3.0;
        let increment = [
            (g[0][0] + correction) * dt,
            (g[1][1] + correction) * dt,
            (g[2][2] + correction) * dt,
            (g[0][1] + g[1][0]) * dt,
            (g[1][2] + g[2][1]) * dt,
            (g[2][0] + g[0][2]) * dt,
        ];
        let spin = [
            0.5 * (g[2][1] - g[1][2]),
            0.5 * (g[0][2] - g[2][0]),
            0.5 * (g[1][0] - g[0][1]),
        ];
        let magnitude = spin[0].hypot(spin[1]).hypot(spin[2]);
        let angle = 0.5 * magnitude * dt;
        if increment.iter().any(|v| !v.is_finite()) || !angle.is_finite() {
            return Err(HexExplicitError::new(
                "nonfinite objective Gauss H8 increment",
            ));
        }
        let rotation = if magnitude == 0.0 {
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        } else {
            // Normalize with the overflow-safe hypot norm before the shared
            // Rodrigues routine squares its axis components.
            let axis = spin.map(|v| v / magnitude);
            let matrix = crate::physics::solid::joints::rotation_matrix(axis, angle);
            std::array::from_fn(|i| matrix[i / 3][i % 3])
        };
        if magnitude != 0.0 {
            material.rotate_state(&mut trial.points[q], &rotation);
        }
        let update = if material.is_rate_dependent() {
            material.update_with_rate(&trial.points[q], &increment, dt)
        } else {
            material.update(&trial.points[q], &increment)
        };
        if update.stress.iter().any(|v| !v.is_finite())
            || !update.delta_equiv_plastic_strain.is_finite()
        {
            return Err(HexExplicitError::new("nonfinite Gauss material response"));
        }
        material.commit(&mut trial.points[q], &update);
        if magnitude != 0.0 {
            material.rotate_state(&mut trial.points[q], &rotation);
        }
        let stress = M::centroid_stress(&trial.points[q]);
        let rate = voigt_equivalent_increment_c1(&increment) / dt;
        let damage = material.accumulate_explicit_damage(
            &mut trial.points[q],
            &stress,
            update.delta_equiv_plastic_strain,
            rate,
        );
        let point_features = material.erosion_features(&trial.points[q]);
        if stress.iter().any(|v| !v.is_finite())
            || [
                rate,
                damage,
                point_features.0,
                point_features.1,
                point_features.2,
            ]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err(HexExplicitError::new(
                "nonfinite committed Gauss material features",
            ));
        }
        stresses[q] = stress;
        for (mean, s) in mean_stress.iter_mut().zip(stress) {
            *mean += s / 8.0;
        }
        mean_dep += update.delta_equiv_plastic_strain / 8.0;
        maximum_rate = maximum_rate.max(rate);
        damage_increment = damage_increment.max(damage);
        features = (
            features.0.max(point_features.0),
            features.1.max(point_features.1),
            features.2.max(point_features.2),
        );
        yielding |= update.yielding;
    }
    let pressure = stresses
        .iter()
        .zip(&geom.points)
        .map(|(s, p)| (s[0] + s[1] + s[2]) * p.weight / 3.0)
        .sum::<f64>()
        / geom.volume;
    let mut forces = [0.0; 24];
    for (point, stress) in geom.points.iter().zip(stresses) {
        let local_pressure = (stress[0] + stress[1] + stress[2]) / 3.0;
        let n = [
            stress[0] - local_pressure + pressure + bulk_q,
            stress[1] - local_pressure + pressure + bulk_q,
            stress[2] - local_pressure + pressure + bulk_q,
        ];
        for (node, g) in point.gradients.iter().enumerate() {
            forces[3 * node] += (n[0] * g[0] + stress[3] * g[1] + stress[5] * g[2]) * point.weight;
            forces[3 * node + 1] +=
                (stress[3] * g[0] + n[1] * g[1] + stress[4] * g[2]) * point.weight;
            forces[3 * node + 2] +=
                (stress[5] * g[0] + stress[4] * g[1] + n[2] * g[2]) * point.weight;
        }
    }
    if forces.iter().any(|v| !v.is_finite()) || !bulk_q.is_finite() {
        return Err(HexExplicitError::new(
            "nonfinite integrated Gauss H8 forces",
        ));
    }
    *state = trial;
    Ok(HexPlasticOut {
        base: HexForceOut {
            forces,
            hg_power: 0.0,
            bulk_q,
            dep: mean_dep,
            yielding,
            stress: mean_stress,
            strain_rate: maximum_rate,
        },
        damage_increment,
        features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::nonlinear_solid::{
        HexExplicitGeom, HexExplicitState, HexMesh, NeoHookean, hex_plastic_forces_objective,
        q1_element_residual,
    };
    use crate::physics::solid::plasticity::law_metals::{Law36, Law36State, RateCurve};

    fn material() -> Law36 {
        Law36::new(
            200e9,
            0.3,
            vec![RateCurve::new(0.0, vec![0.0, 1.0], vec![1e20, 1e20], 1.0).unwrap()],
            0.0,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        )
    }

    #[test]
    fn affine_uniaxial_patch_has_analytic_stress_and_face_tractions() {
        let mesh = HexMesh::unit_cube(1.0);
        let coords = mesh.element_coords(0);
        let geom = HexGaussPlasticGeom::new(coords).unwrap();
        let mut state = HexGaussPlasticState::new(Law36State::new());
        let dt = 1e-3;
        let strain = 1e-6;
        let velocity = std::array::from_fn(|i| {
            let n = i / 3;
            let a = i % 3;
            coords[n][a] * strain / dt * if a == 0 { 1.0 } else { -0.3 }
        });
        let out = hex_gauss_plastic_forces(
            &geom,
            &mut state,
            &material(),
            &velocity,
            dt,
            &HexExplicitParams::new(7800.0, 5000.0),
        )
        .unwrap();
        assert!((out.base.stress[0] - 200e3).abs() < 1e-6);
        assert!(out.base.stress[1..].iter().all(|v| v.abs() < 1e-6));
        for (n, p) in coords.iter().enumerate() {
            assert!((out.base.forces[3 * n] - (p[0] - 0.5) * 100e3).abs() < 1e-6);
        }
        assert_eq!(out.features.0, 0.0);
    }

    #[test]
    fn resting_hourglass_mode_has_physical_shear_stiffness() {
        let mesh = HexMesh::unit_cube(1.0);
        let coords = mesh.element_coords(0);
        let mut deformed = coords;
        let displacement: [f64; 24] = std::array::from_fn(|i| {
            if i % 3 == 2 {
                1e-6 * (2.0 * coords[i / 3][0] - 1.0) * (2.0 * coords[i / 3][1] - 1.0)
            } else {
                0.0
            }
        });
        for (i, p) in deformed.iter_mut().enumerate() {
            p[2] += displacement[3 * i + 2];
        }
        let velocity = displacement.map(|v| v / 1e-3);
        let params = HexExplicitParams::new(7800.0, 5000.0);
        let mat = material();
        let mut state = HexGaussPlasticState::new(Law36State::new());
        hex_gauss_plastic_forces(
            &HexGaussPlasticGeom::new(coords).unwrap(),
            &mut state,
            &mat,
            &velocity,
            1e-3,
            &params,
        )
        .unwrap();
        let force = hex_gauss_plastic_forces(
            &HexGaussPlasticGeom::new(deformed).unwrap(),
            &mut state,
            &mat,
            &[0.0; 24],
            1e-3,
            &params,
        )
        .unwrap()
        .base
        .forces;
        let reference = q1_element_residual(
            &NeoHookean::from_young(200e9, 0.3, 3),
            &mesh,
            0,
            &displacement,
        );
        let error = force
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            / reference.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            error < 1e-5,
            "independent hyperelastic linear-limit force error {error}"
        );
        assert!(
            force
                .iter()
                .zip(displacement)
                .map(|(f, u)| f * u)
                .sum::<f64>()
                > 0.0
        );
        let mut old = HexExplicitState::new(Law36State::new());
        hex_plastic_forces_objective(
            &HexExplicitGeom::new(coords).unwrap(),
            &mut old,
            &mat,
            &velocity,
            1e-3,
            &params,
        )
        .unwrap();
        let old = hex_plastic_forces_objective(
            &HexExplicitGeom::new(deformed).unwrap(),
            &mut old,
            &mat,
            &[0.0; 24],
            1e-3,
            &params,
        )
        .unwrap();
        assert!(old.base.forces.iter().all(|v| v.abs() < 1e-8));
    }

    #[test]
    fn rigid_spin_rotates_all_histories_without_plastic_work() {
        let coords = HexMesh::unit_cube(1.0).element_coords(0);
        let geom = HexGaussPlasticGeom::new(coords).unwrap();
        let mut initial = Law36State::new();
        initial.stress[0] = 100e6;
        let mut state = HexGaussPlasticState::new(initial);
        let velocity = std::array::from_fn(|i| match i % 3 {
            0 => -coords[i / 3][1],
            1 => coords[i / 3][0],
            _ => 0.0,
        });
        let out = hex_gauss_plastic_forces(
            &geom,
            &mut state,
            &material(),
            &velocity,
            0.2,
            &HexExplicitParams::new(7800.0, 5000.0),
        )
        .unwrap();
        let (s, c) = 0.2_f64.sin_cos();
        let expected = [100e6 * c * c, 100e6 * s * s, 0.0, 100e6 * s * c, 0.0, 0.0];
        for point in &state.points {
            let error = point
                .stress
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / 100e6;
            assert!(error < 1e-12, "rigid-spin stress error {error}");
            assert_eq!(point.equiv_plastic_strain, 0.0);
        }
        assert_eq!(out.base.dep, 0.0);
    }

    #[test]
    fn inverted_geometry_rejected_and_failed_step_does_not_commit() {
        let mut coords = HexMesh::unit_cube(1.0).element_coords(0);
        let geom = HexGaussPlasticGeom::new(coords).unwrap();
        for p in &mut coords {
            p[2] = -p[2];
        }
        assert!(HexGaussPlasticGeom::new(coords).is_err());
        let mut state = HexGaussPlasticState::new(Law36State::new());
        let before = state.points[0].stress;
        assert!(
            hex_gauss_plastic_forces(
                &geom,
                &mut state,
                &material(),
                &[0.0; 24],
                f64::NAN,
                &HexExplicitParams::new(7800.0, 5000.0)
            )
            .is_err()
        );
        assert_eq!(state.points[0].stress, before);
    }
}
