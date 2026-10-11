//! Displacement-based fiber beam element for 3-D frames.
//!
//! This module wires [`FiberSection`](super::fiber::FiberSection) into the
//! 3-D beam assembly: each element evaluates fiber-section response at Gauss
//! points along its axis and integrates section resultants back to nodal
//! forces. Axial strain is linear along the element and curvatures follow
//! Euler-Bernoulli Hermite interpolation, so the elastic tangent reproduces
//! the prismatic [`BeamElement3D`](super::element3d::BeamElement3D) stiffness
//! exactly. Torsion stays linear elastic (`GJ/L`); transverse shear is
//! neglected (Euler-Bernoulli kinematics), which suits slender elastoplastic
//! members such as the plastic-hinge QA.
//!
//! History protocol: [`FiberBeamForm3D`] stores committed fiber states and
//! committed section deformations. Residual and tangent evaluations are pure
//! trial evaluations against the committed state, so Newton iterations never
//! mutate history; the driver calls [`FiberBeamForm3D::commit`] once per
//! converged load step.

// Rust guideline compliant 2026-02-21

use super::element3d::{BeamElement3D, transform_12x12};
use super::fiber::{FiberCompatibleModel, FiberResult, FiberSection};
use crate::forms::NonlinearElementForm;

/// Fiber-discretized 3-D beam element with Gauss integration along the axis.
#[derive(Debug, Clone)]
pub struct FiberBeamElement<M: FiberCompatibleModel> {
    /// Straight reference element fixing the frame, length, and torsion.
    pub beam: BeamElement3D,
    /// Fiber section shared by every Gauss point (states stay per point).
    pub section: FiberSection<M>,
    /// Gauss point count along the axis (1 to 4).
    pub n_gauss: usize,
}

impl<M: FiberCompatibleModel> FiberBeamElement<M> {
    /// Build a fiber beam element from a reference beam and section.
    ///
    /// # Panics
    ///
    /// Panics for a Gauss count outside 1 to 4.
    #[must_use]
    pub fn new(beam: BeamElement3D, section: FiberSection<M>, n_gauss: usize) -> Self {
        assert!(
            (1..=4).contains(&n_gauss),
            "fiber beam needs 1 to 4 Gauss points"
        );
        Self {
            beam,
            section,
            n_gauss,
        }
    }
}

/// Committed history of one fiber beam element.
#[derive(Debug, Clone)]
pub struct FiberBeamState<S: Clone> {
    /// Committed fiber states per Gauss point, fiber order inside.
    pub fiber_states: Vec<Vec<S>>,
    /// Committed `(e0, kappa_y, kappa_z)` per Gauss point.
    pub section_deformations: Vec<[f64; 3]>,
}

/// Nonlinear 3-D frame form with fiber-section material response.
///
/// Implements [`NonlinearElementForm<f64>`] with 6 DOFs per node
/// (`[ux, uy, uz, rx, ry, rz]`), directly assemblable alongside linear beam
/// analysis drivers.
#[derive(Debug, Clone)]
pub struct FiberBeamForm3D<M: FiberCompatibleModel> {
    /// Fiber beam elements.
    pub elements: Vec<FiberBeamElement<M>>,
    /// Committed history per element.
    pub states: Vec<FiberBeamState<M::State>>,
    /// Total number of global DOFs.
    pub n_dofs: usize,
}

impl<M: FiberCompatibleModel> FiberBeamForm3D<M> {
    /// Build a form with fresh history from a fiber-state constructor.
    #[must_use]
    pub fn new(
        elements: Vec<FiberBeamElement<M>>,
        make_state: impl Fn() -> M::State,
        n_dofs: usize,
    ) -> Self {
        let states = elements
            .iter()
            .map(|elem| {
                let fiber_states = (0..elem.n_gauss)
                    .map(|_| elem.section.initial_states(&make_state))
                    .collect();
                FiberBeamState {
                    fiber_states,
                    section_deformations: vec![[0.0; 3]; elem.n_gauss],
                }
            })
            .collect();
        Self {
            elements,
            states,
            n_dofs,
        }
    }

    /// Gauss stations and weights on `[0, 1]` for the element.
    fn gauss_points(n_gauss: usize) -> Vec<(f64, f64)> {
        match n_gauss {
            1 => vec![(0.5, 1.0)],
            2 => {
                let a = 0.5 - 0.288_675_134_594_812_9;
                vec![(a, 0.5), (1.0 - a, 0.5)]
            }
            3 => vec![
                (0.112_701_665_379_258_3, 0.277_777_777_777_777_8),
                (0.5, 0.444_444_444_444_444_4),
                (0.887_298_334_620_741_7, 0.277_777_777_777_777_8),
            ],
            4 => vec![
                (0.069_431_844_202_973_7, 0.173_927_422_568_726_9),
                (0.330_009_478_207_571_9, 0.326_072_577_431_273_1),
                (0.669_990_521_792_428_1, 0.326_072_577_431_273_1),
                (0.930_568_155_797_026_2, 0.173_927_422_568_726_9),
            ],
            _ => panic!("fiber beam needs 1 to 4 Gauss points"),
        }
    }

    /// Element displacement vector rotated into the local frame.
    fn local_displacements(&self, elem_idx: usize, u: &[f64]) -> [f64; 12] {
        let elem = &self.elements[elem_idx];
        let dofs = elem.beam.global_dofs();
        let mut local = [0.0; 12];
        // Local component p of block b: axes row p dotted with global block.
        for block in 0..4 {
            for p in 0..3 {
                let mut sum = 0.0;
                for c in 0..3 {
                    sum += elem.beam.axes[p * 3 + c] * u[dofs[block * 3 + c]];
                }
                local[block * 3 + p] = sum;
            }
        }
        local
    }

    /// Strain-displacement rows for `(e0, kappa_y, kappa_z)` at `xi`.
    ///
    /// Axial strain is linear; curvatures are Hermite second derivatives
    /// with `ry = -dw/dx` matching the prismatic element sign convention.
    fn strain_rows(length: f64, xi: f64) -> [[f64; 12]; 3] {
        let l2 = length * length;
        let h0 = (-6.0 + 12.0 * xi) / l2;
        let h1 = (-4.0 + 6.0 * xi) / length;
        let h2 = (6.0 - 12.0 * xi) / l2;
        let h3 = (-2.0 + 6.0 * xi) / length;
        let mut e0 = [0.0; 12];
        e0[0] = -1.0 / length;
        e0[6] = 1.0 / length;
        // kappa_z from (uy1, rz1, uy2, rz2).
        let mut kz = [0.0; 12];
        kz[1] = h0;
        kz[5] = h1;
        kz[7] = h2;
        kz[11] = h3;
        // kappa_y from (uz1, -ry1, uz2, -ry2).
        let mut ky = [0.0; 12];
        ky[2] = h0;
        ky[4] = -h1;
        ky[8] = h2;
        ky[10] = -h3;
        [e0, ky, kz]
    }

    /// Trial fiber results at every Gauss point for a global displacement.
    fn trial_results(&self, elem_idx: usize, u: &[f64]) -> Vec<FiberResult> {
        let elem = &self.elements[elem_idx];
        let local = self.local_displacements(elem_idx, u);
        let mut out = Vec::with_capacity(elem.n_gauss);
        for (g, (xi, _)) in Self::gauss_points(elem.n_gauss).iter().enumerate() {
            let rows = Self::strain_rows(elem.beam.length, *xi);
            let mut def = [0.0; 3];
            for r in 0..3 {
                def[r] = rows[r].iter().zip(local.iter()).map(|(b, v)| b * v).sum();
            }
            let committed = self.states[elem_idx].section_deformations[g];
            let result = elem.section.integrate(
                &self.states[elem_idx].fiber_states[g],
                def[0] - committed[0],
                def[1] - committed[1],
                def[2] - committed[2],
            );
            out.push(result);
        }
        out
    }

    /// Section resultants `(N, My, Mz)` per Gauss point at `u`.
    ///
    /// Trial evaluation against the committed state; used by QA drivers to
    /// inspect hinge moments without committing.
    #[must_use]
    pub fn resultants_for(&self, elem_idx: usize, u: &[f64]) -> Vec<[f64; 3]> {
        self.trial_results(elem_idx, u)
            .iter()
            .map(|r| [r.n, r.my, r.mz])
            .collect()
    }

    /// Commit the trial state at `u` as the new converged history.
    pub fn commit(&mut self, u: &[f64]) {
        for elem_idx in 0..self.elements.len() {
            let results = self.trial_results(elem_idx, u);
            let local = self.local_displacements(elem_idx, u);
            let elem = &self.elements[elem_idx];
            for (g, (xi, _)) in Self::gauss_points(elem.n_gauss).iter().enumerate() {
                let rows = Self::strain_rows(elem.beam.length, *xi);
                for (r, row) in rows.iter().enumerate() {
                    self.states[elem_idx].section_deformations[g][r] =
                        row.iter().zip(local.iter()).map(|(b, v)| b * v).sum();
                }
                let states = &mut self.states[elem_idx].fiber_states[g];
                elem.section.commit(states, &results[g]);
            }
        }
    }

    /// Internal force vector for one element in global coordinates.
    fn internal_forces(&self, elem_idx: usize, u: &[f64]) -> [f64; 12] {
        let elem = &self.elements[elem_idx];
        let local = self.local_displacements(elem_idx, u);
        let results = self.trial_results(elem_idx, u);
        let mut f = [0.0; 12];
        for (g, (xi, w)) in Self::gauss_points(elem.n_gauss).iter().enumerate() {
            let rows = Self::strain_rows(elem.beam.length, *xi);
            let forces = [results[g].n, results[g].my, results[g].mz];
            let scale = w * elem.beam.length;
            for r in 0..3 {
                for i in 0..12 {
                    f[i] += scale * rows[r][i] * forces[r];
                }
            }
        }
        // Linear elastic torsion on the rx DOFs.
        let gj_over_l = elem.beam.section.gj() / elem.beam.length;
        f[3] += gj_over_l * (local[3] - local[9]);
        f[9] += gj_over_l * (local[9] - local[3]);
        rotate_vector_to_global(&elem.beam.axes, &f)
    }

    /// Tangent stiffness for one element in global coordinates.
    fn tangent_stiffness(&self, elem_idx: usize, u: &[f64]) -> [f64; 144] {
        let elem = &self.elements[elem_idx];
        let results = self.trial_results(elem_idx, u);
        let mut k = [0.0; 144];
        for (g, (xi, w)) in Self::gauss_points(elem.n_gauss).iter().enumerate() {
            let rows = Self::strain_rows(elem.beam.length, *xi);
            let scale = w * elem.beam.length;
            for r in 0..3 {
                for c in 0..3 {
                    let tangent = results[g].tangent[r * 3 + c];
                    for i in 0..12 {
                        for j in 0..12 {
                            k[i * 12 + j] += scale * rows[r][i] * tangent * rows[c][j];
                        }
                    }
                }
            }
        }
        let gj_over_l = elem.beam.section.gj() / elem.beam.length;
        k[3 * 12 + 3] += gj_over_l;
        k[3 * 12 + 9] -= gj_over_l;
        k[9 * 12 + 3] -= gj_over_l;
        k[9 * 12 + 9] += gj_over_l;
        transform_12x12(&elem.beam.axes, &k)
    }
}

impl<M> NonlinearElementForm<f64> for FiberBeamForm3D<M>
where
    M: FiberCompatibleModel + Send + Sync,
    M::State: Send + Sync,
{
    fn num_elements(&self) -> usize {
        self.elements.len()
    }

    fn num_local_dofs(&self, _elem_idx: usize) -> usize {
        12
    }

    fn local_to_global(&self, elem_idx: usize) -> Vec<usize> {
        self.elements[elem_idx].beam.global_dofs().to_vec()
    }

    fn num_global_dofs(&self) -> usize {
        self.n_dofs
    }

    fn element_residual(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        self.internal_forces(elem_idx, global_solution).to_vec()
    }

    fn element_tangent(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        self.tangent_stiffness(elem_idx, global_solution).to_vec()
    }
}

/// Rotate a local 12-vector (four 3-blocks) into global coordinates.
fn rotate_vector_to_global(axes: &[f64; 9], local: &[f64; 12]) -> [f64; 12] {
    let mut global = [0.0; 12];
    for block in 0..4 {
        for row in 0..3 {
            let mut sum = 0.0;
            for p in 0..3 {
                sum += axes[p * 3 + row] * local[block * 3 + p];
            }
            global[block * 3 + row] = sum;
        }
    }
    global
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::beam::{Fiber, Section3D, UniaxialJ2Adapter};
    use crate::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};

    fn steel_adapter() -> UniaxialJ2Adapter {
        UniaxialJ2Adapter::new(J2Material::new(
            200.0e9,
            0.3,
            250.0e6,
            HardeningLaw::LinearIsotropic { modulus: 1.0e9 },
        ))
    }

    fn test_form() -> FiberBeamForm3D<UniaxialJ2Adapter> {
        let beam = BeamElement3D::new(
            [0, 1],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            Section3D::rectangular(200.0e9, 0.3, 0.02, 0.01, 7800.0, false),
            [0.0, 1.0, 0.0],
        );
        let section = FiberSection::new(steel_adapter(), Fiber::grid(0.02, 0.01, 12, 12));
        let elem = FiberBeamElement::new(beam, section, 3);
        FiberBeamForm3D::new(vec![elem], PlasticState::new_3d, 12)
    }

    #[test]
    fn elastic_tangent_matches_prismatic_element() {
        let form = test_form();
        let u = vec![0.0; 12];
        let tangent = form.element_tangent(0, &u);
        let expected = form.elements[0].beam.stiffness_local();
        // Fiber midpoint discretization resolves EA/EI to a couple of
        // percent on this 12x12 grid.
        let scale = expected.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        for i in 0..144 {
            assert!(
                (tangent[i] - expected[i]).abs() / scale < 0.03,
                "entry {i} differs"
            );
        }
    }

    #[test]
    fn small_load_stays_elastic_and_linear() {
        let form = test_form();
        // Tip-load the free end through the residual: linear response means
        // doubling the displacement doubles the internal forces.
        let mut u = vec![0.0; 12];
        u[7] = 1.0e-5;
        let f1 = form.element_residual(0, &u);
        for v in u.iter_mut() {
            *v *= 2.0;
        }
        let f2 = form.element_residual(0, &u);
        for i in 0..12 {
            assert!((f2[i] - 2.0 * f1[i]).abs() < 1.0e-9 * f2[i].abs().max(1.0e-12));
        }
        // No fiber yields at this load level.
        let results = form.trial_results(0, &u);
        assert!(
            results
                .iter()
                .flat_map(|r| r.updates.iter())
                .all(|update| !update.yielding)
        );
    }

    #[test]
    fn overload_yields_and_commits() {
        let mut form = test_form();
        // Bend far past first yield (yield curvature 0.125/m here).
        let mut u = vec![0.0; 12];
        u[5] = 0.5;
        u[11] = -0.5;
        let results = form.trial_results(0, &u);
        assert!(
            results
                .iter()
                .flat_map(|r| r.updates.iter())
                .any(|update| update.yielding)
        );
        form.commit(&u);
        assert!(
            form.states[0]
                .fiber_states
                .iter()
                .flat_map(|point| point.iter())
                .any(|state| state.equiv_plastic_strain > 0.0)
        );
    }
}
