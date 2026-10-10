//! Layered through-thickness shell sections.
//!
//! Through-thickness integration over a stack of plies where each ply
//! consumes its constitutive law through the frozen C1 stress-update
//! contract. The integrator is generic over the model trait, so the
//! finite-strain J2 model (Worker 4, Sync 1) plugs in as a new model type
//! with no element-code change.
//!
//! Elastic and orthotropic plies couple to the existing composites CLT
//! data: [`LayeredSection::from_laminate`] reproduces the laminate A/B/D
//! matrices exactly. Small-strain J2 plies delegate to the canonical C1
//! implementation on [`J2Material`](super::super::plasticity::J2Material).

// Rust guideline compliant 2026-02-21

use crate::physics::solid::composites::{
    DamageState, DegradationRule, FailureCriterion, Laminate, OrthotropicPly, PlyStrengths,
    assess_ply_failure, degraded_ply, laminate_to_material_stress,
};
use crate::physics::solid::plasticity::update::{StressUpdate, StressUpdateModel};
use crate::physics::solid::plasticity::{J2Material, PlasticState};

/// In-plane Voigt slots in C1 ordering (xx, yy, xy).
const KEEP: [usize; 3] = [0, 1, 3];
/// Condensed Voigt slots in C1 ordering (zz, yz, zx).
const CONDENSED: [usize; 3] = [2, 4, 5];

/// Statically condense a C1 6x6 tangent to the plane-stress 3x3.
///
/// Condenses out zz/yz/zx, keeping xx/yy/xy. For isotropic elasticity this
/// reproduces the plane-stress Q matrix exactly.
pub fn condense_plane_stress(tangent: &[f64; 36]) -> [f64; 9] {
    let mut c_cc = [0.0; 9];
    let mut c_kc = [0.0; 9];
    let mut c_ck = [0.0; 9];
    let mut c_kk = [0.0; 9];
    for (i, &ci) in CONDENSED.iter().enumerate() {
        for (j, &cj) in CONDENSED.iter().enumerate() {
            c_cc[i * 3 + j] = tangent[ci * 6 + cj];
        }
    }
    for (i, &ki) in KEEP.iter().enumerate() {
        for (j, &cj) in CONDENSED.iter().enumerate() {
            c_kc[i * 3 + j] = tangent[ki * 6 + cj];
        }
    }
    for (i, &ci) in CONDENSED.iter().enumerate() {
        for (j, &kj) in KEEP.iter().enumerate() {
            c_ck[i * 3 + j] = tangent[ci * 6 + kj];
        }
    }
    for (i, &ki) in KEEP.iter().enumerate() {
        for (j, &kj) in KEEP.iter().enumerate() {
            c_kk[i * 3 + j] = tangent[ki * 6 + kj];
        }
    }
    let inv = invert_3x3(&c_cc);
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut correction = 0.0;
            for p in 0..3 {
                let mut inv_ck = 0.0;
                for q in 0..3 {
                    inv_ck += inv[p * 3 + q] * c_ck[q * 3 + j];
                }
                correction += c_kc[i * 3 + p] * inv_ck;
            }
            out[i * 3 + j] = c_kk[i * 3 + j] - correction;
        }
    }
    out
}

/// Invert a row-major 3x3 matrix.
fn invert_3x3(m: &[f64; 9]) -> [f64; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let inv_det = 1.0 / det;
    [
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ]
}

/// Isotropic linear elastic C1 model.
#[derive(Debug, Clone, Copy)]
pub struct ElasticIsotropicModel {
    /// Young's modulus in Pa.
    pub youngs_modulus: f64,
    /// Poisson's ratio.
    pub poisson_ratio: f64,
}

/// Committed state of an elastic layer point.
#[derive(Debug, Clone)]
pub struct ElasticState {
    /// Committed stress, C1 Voigt ordering.
    pub stress: [f64; 6],
}

impl ElasticIsotropicModel {
    /// Full 6x6 isotropic tangent in C1 Voigt ordering.
    pub fn tangent_6x6(&self) -> [f64; 36] {
        let e = self.youngs_modulus;
        let nu = self.poisson_ratio;
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let g = e / (2.0 * (1.0 + nu));
        let l2g = lambda + 2.0 * g;
        let mut c = [0.0; 36];
        for i in 0..3 {
            for j in 0..3 {
                c[i * 6 + j] = if i == j { l2g } else { lambda };
            }
        }
        c[3 * 6 + 3] = g;
        c[4 * 6 + 4] = g;
        c[5 * 6 + 5] = g;
        c
    }
}

impl StressUpdateModel for ElasticIsotropicModel {
    type State = ElasticState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        let c = self.tangent_6x6();
        let mut stress = state.stress;
        for i in 0..6 {
            for j in 0..6 {
                stress[i] += c[i * 6 + j] * d_eps[j];
            }
        }
        StressUpdate {
            stress,
            tangent: c,
            delta_equiv_plastic_strain: 0.0,
            yielding: false,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
    }
}

/// Orthotropic linear elastic C1 ply model (plane-stress Q-bar).
///
/// The in-plane block is the transformed reduced stiffness; transverse
/// entries are dummy-but-invertible so plane-stress condensation returns
/// Q-bar exactly. Built from [`Laminate`] plies by
/// [`LayeredSection::from_laminate`].
#[derive(Debug, Clone, Copy)]
pub struct OrthotropicPlyModel {
    /// Transformed reduced stiffness, row-major 3x3 in Pa.
    pub qbar: [f64; 9],
    /// Transverse Young's modulus stand-in in Pa.
    pub transverse_modulus: f64,
    /// Transverse shear modulus stand-in in Pa.
    pub transverse_shear: f64,
}

/// Committed state of an orthotropic layer point.
#[derive(Debug, Clone)]
pub struct OrthotropicState {
    /// Committed stress, C1 Voigt ordering.
    pub stress: [f64; 6],
}

impl StressUpdateModel for OrthotropicPlyModel {
    type State = OrthotropicState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        let mut c = [0.0; 36];
        for i in 0..3 {
            for j in 0..3 {
                c[KEEP[i] * 6 + KEEP[j]] = self.qbar[i * 3 + j];
            }
        }
        c[2 * 6 + 2] = self.transverse_modulus;
        c[4 * 6 + 4] = self.transverse_shear;
        c[5 * 6 + 5] = self.transverse_shear;
        let mut stress = state.stress;
        for i in 0..6 {
            for j in 0..6 {
                stress[i] += c[i * 6 + j] * d_eps[j];
            }
        }
        StressUpdate {
            stress,
            tangent: c,
            delta_equiv_plastic_strain: 0.0,
            yielding: false,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
    }
}

/// Small-strain J2 C1 adapter over the existing return mapping.
///
/// Single return-mapping call per update; the section drives the transverse
/// strain iteration (see [`LayerModel::requires_plane_stress_iteration`]) so
/// shell layers see sigma_zz = 0. Converts between C1 Voigt ordering
/// (xx,yy,zz,xy,yz,zx) and the `plasticity` ordering (xx,yy,zz,yz,xz,xy).
#[derive(Debug, Clone, Copy)]
pub struct SmallStrainJ2Model {
    /// J2 material parameters.
    pub material: J2Material,
}

/// Committed state of a J2 layer point.
#[derive(Debug, Clone)]
pub struct J2LayerState {
    /// Plasticity state (stress in `plasticity` Voigt ordering).
    pub plastic: PlasticState,
}

impl J2LayerState {
    /// Virgin state.
    pub fn new() -> Self {
        Self {
            plastic: PlasticState::new_3d(),
        }
    }
}

impl Default for J2LayerState {
    fn default() -> Self {
        Self::new()
    }
}

impl StressUpdateModel for SmallStrainJ2Model {
    type State = J2LayerState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        self.material.update(&state.plastic, d_eps)
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        self.material.commit(&mut state.plastic, update)
    }
}

/// Progressive-damage orthotropic ply C1 model.
///
/// The trial update is linear elastic with the committed damage (pure), so
/// the section's static condensation stays exact. Commit advances the
/// damage variables from the trial stress rotated to material coordinates
/// (operator split: damage lags one commit, the standard explicit damage
/// update). Failure assessment uses the Hashin or Puck criterion.
#[derive(Debug, Clone, Copy)]
pub struct DegradingPlyModel {
    /// Undamaged ply properties, angle, and thickness.
    pub ply: OrthotropicPly,
    /// Ply allowables.
    pub strengths: PlyStrengths,
    /// Failure criterion driving damage.
    pub criterion: FailureCriterion,
    /// Stiffness degradation rule past initiation.
    pub rule: DegradationRule,
    /// Transverse Young's modulus stand-in in Pa.
    pub transverse_modulus: f64,
    /// Transverse shear modulus stand-in in Pa.
    pub transverse_shear: f64,
}

/// Committed state of a degrading ply point.
#[derive(Debug, Clone)]
pub struct DegradingPlyState {
    /// Committed stress, C1 Voigt ordering.
    pub stress: [f64; 6],
    /// Committed damage variables.
    pub damage: DamageState,
}

impl DegradingPlyState {
    /// Virgin state.
    pub fn new() -> Self {
        Self {
            stress: [0.0; 6],
            damage: DamageState::virgin(),
        }
    }
}

impl Default for DegradingPlyState {
    fn default() -> Self {
        Self::new()
    }
}

impl StressUpdateModel for DegradingPlyModel {
    type State = DegradingPlyState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        let degraded = degraded_ply(&self.ply, &state.damage);
        let q = degraded.transformed_stiffness();
        let mut c = [0.0; 36];
        for i in 0..3 {
            for j in 0..3 {
                c[KEEP[i] * 6 + KEEP[j]] = q[i][j];
            }
        }
        c[2 * 6 + 2] = self.transverse_modulus;
        c[4 * 6 + 4] = self.transverse_shear;
        c[5 * 6 + 5] = self.transverse_shear;
        let mut stress = state.stress;
        for i in 0..6 {
            for j in 0..6 {
                stress[i] += c[i * 6 + j] * d_eps[j];
            }
        }
        StressUpdate {
            stress,
            tangent: c,
            delta_equiv_plastic_strain: 0.0,
            yielding: false,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
        let laminate = [update.stress[0], update.stress[1], update.stress[3]];
        let material = laminate_to_material_stress(laminate, self.ply.angle_rad);
        let assessment = assess_ply_failure(material, &self.strengths, self.criterion);
        state.damage = state
            .damage
            .advance(assessment.fiber, assessment.matrix, self.rule);
    }
}

/// C1 model usable as a shell ply.
///
/// Blanket capability flag on top of the frozen C1 shape: path-dependent
/// models need the section to iterate the transverse strain so the trial
/// stress satisfies sigma_zz = 0, while linear models are handled exactly
/// by static condensation.
pub trait LayerModel: StressUpdateModel<StrainIncrement = [f64; 6]> {
    /// Whether the section must iterate transverse strain for plane stress.
    fn requires_plane_stress_iteration(&self) -> bool {
        false
    }
}

impl LayerModel for ElasticIsotropicModel {}
impl LayerModel for OrthotropicPlyModel {}
impl LayerModel for DegradingPlyModel {}
impl LayerModel for SmallStrainJ2Model {
    fn requires_plane_stress_iteration(&self) -> bool {
        true
    }
}

/// Dynamic layer material: any single-ply model behind one enum.
///
/// Mixed stacks (elastic + J2 plies) use `LayeredSection<AnyLayerModel>`.
#[derive(Debug, Clone, Copy)]
pub enum AnyLayerModel {
    /// Isotropic elastic ply.
    Elastic(ElasticIsotropicModel),
    /// Orthotropic elastic ply.
    Orthotropic(OrthotropicPlyModel),
    /// Progressive-damage orthotropic ply.
    Degrading(DegradingPlyModel),
    /// Small-strain J2 ply.
    J2(SmallStrainJ2Model),
}

/// Dynamic committed state matching [`AnyLayerModel`].
#[derive(Debug, Clone)]
pub enum AnyLayerState {
    /// Elastic state.
    Elastic(ElasticState),
    /// Orthotropic state.
    Orthotropic(OrthotropicState),
    /// Degrading state.
    Degrading(DegradingPlyState),
    /// J2 state.
    J2(J2LayerState),
}

impl AnyLayerState {
    /// Virgin state matching a model.
    pub fn virgin(model: &AnyLayerModel) -> Self {
        match model {
            AnyLayerModel::Elastic(_) => Self::Elastic(ElasticState { stress: [0.0; 6] }),
            AnyLayerModel::Orthotropic(_) => {
                Self::Orthotropic(OrthotropicState { stress: [0.0; 6] })
            }
            AnyLayerModel::Degrading(_) => Self::Degrading(DegradingPlyState::new()),
            AnyLayerModel::J2(_) => Self::J2(J2LayerState::new()),
        }
    }
}

impl StressUpdateModel for AnyLayerModel {
    type State = AnyLayerState;
    type StrainIncrement = [f64; 6];

    fn update(&self, state: &Self::State, d_eps: &Self::StrainIncrement) -> StressUpdate {
        match (self, state) {
            (Self::Elastic(m), AnyLayerState::Elastic(s)) => m.update(s, d_eps),
            (Self::Orthotropic(m), AnyLayerState::Orthotropic(s)) => m.update(s, d_eps),
            (Self::Degrading(m), AnyLayerState::Degrading(s)) => m.update(s, d_eps),
            (Self::J2(m), AnyLayerState::J2(s)) => m.update(s, d_eps),
            _ => panic!("layer model/state mismatch: programming error"),
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        match (self, state) {
            (Self::Elastic(m), AnyLayerState::Elastic(s)) => m.commit(s, update),
            (Self::Orthotropic(m), AnyLayerState::Orthotropic(s)) => m.commit(s, update),
            (Self::Degrading(m), AnyLayerState::Degrading(s)) => m.commit(s, update),
            (Self::J2(m), AnyLayerState::J2(s)) => m.commit(s, update),
            _ => panic!("layer model/state mismatch: programming error"),
        }
    }
}

impl LayerModel for AnyLayerModel {
    fn requires_plane_stress_iteration(&self) -> bool {
        match self {
            Self::Elastic(m) => m.requires_plane_stress_iteration(),
            Self::Orthotropic(m) => m.requires_plane_stress_iteration(),
            Self::Degrading(m) => m.requires_plane_stress_iteration(),
            Self::J2(m) => m.requires_plane_stress_iteration(),
        }
    }
}

/// Condense a 6-component trial stress to plane-stress in-plane values.
///
/// Applies sigma_ps = sigma_keep - C_kc C_cc^-1 sigma_cond with the matching
/// tangent, exact for linear models and consistent for iterated J2 trials.
pub fn condense_stress(tangent: &[f64; 36], stress: &[f64; 6]) -> [f64; 3] {
    let mut c_cc = [0.0; 9];
    let mut c_kc = [0.0; 9];
    for (i, &ci) in CONDENSED.iter().enumerate() {
        for (j, &cj) in CONDENSED.iter().enumerate() {
            c_cc[i * 3 + j] = tangent[ci * 6 + cj];
        }
    }
    for (i, &ki) in KEEP.iter().enumerate() {
        for (j, &cj) in CONDENSED.iter().enumerate() {
            c_kc[i * 3 + j] = tangent[ki * 6 + cj];
        }
    }
    let inv = invert_3x3(&c_cc);
    let mut out = [0.0; 3];
    for i in 0..3 {
        let mut correction = 0.0;
        for p in 0..3 {
            let mut inv_s = 0.0;
            for q in 0..3 {
                inv_s += inv[p * 3 + q] * stress[CONDENSED[q]];
            }
            correction += c_kc[i * 3 + p] * inv_s;
        }
        out[i] = stress[KEEP[i]] - correction;
    }
    out
}

/// One ply of a layered section.
#[derive(Debug, Clone)]
pub struct Layer<M> {
    /// Ply thickness in m.
    pub thickness: f64,
    /// Ply mass density in kg/m^3.
    pub density: f64,
    /// Ply constitutive model.
    pub model: M,
}

/// Through-thickness resultant evaluation.
#[derive(Debug, Clone)]
pub struct LayeredResultants {
    /// Membrane tangent, row-major 3x3.
    pub d_mm: [f64; 9],
    /// Membrane-bending coupling tangent, row-major 3x3.
    pub d_mb: [f64; 9],
    /// Bending tangent, row-major 3x3.
    pub d_bb: [f64; 9],
    /// Membrane forces [Nxx, Nyy, Nxy].
    pub membrane_force: [f64; 3],
    /// Bending moments [Mxx, Myy, Mxy].
    pub moment: [f64; 3],
}

/// Layered shell section with C1 ply models.
#[derive(Debug, Clone)]
pub struct LayeredSection<M: LayerModel> {
    /// Plies from bottom (-t/2) to top (+t/2).
    pub layers: Vec<Layer<M>>,
    /// Gauss points per ply (2 or 3).
    pub points_per_layer: usize,
    /// Committed states, flattened [layer][point].
    states: Vec<M::State>,
    /// Committed total strain per point, C1 ordering, same flattening.
    totals: Vec<[f64; 6]>,
    /// Accumulated equivalent plastic strain per point, same flattening.
    plastic: Vec<f64>,
}

impl<M: LayerModel> LayeredSection<M> {
    /// Create from plies with the default 2-point Gauss rule per ply.
    pub fn new(layers: Vec<Layer<M>>, virgin: impl Fn(&M) -> M::State) -> Self {
        Self::with_points(layers, 2, virgin)
    }

    /// Create from plies with an explicit per-ply Gauss rule (2 or 3).
    pub fn with_points(
        layers: Vec<Layer<M>>,
        points_per_layer: usize,
        virgin: impl Fn(&M) -> M::State,
    ) -> Self {
        assert!(
            points_per_layer == 2 || points_per_layer == 3,
            "only 2- or 3-point Gauss per ply"
        );
        assert!(!layers.is_empty(), "layered section needs at least one ply");
        let n_points = layers.len() * points_per_layer;
        let states = layers
            .iter()
            .flat_map(|layer| (0..points_per_layer).map(|_| virgin(&layer.model)))
            .collect();
        Self {
            layers,
            points_per_layer,
            states,
            totals: vec![[0.0; 6]; n_points],
            plastic: vec![0.0; n_points],
        }
    }

    /// Total thickness in m.
    pub fn total_thickness(&self) -> f64 {
        self.layers.iter().map(|layer| layer.thickness).sum()
    }

    /// Mass per unit area in kg/m^2.
    pub fn mass_per_area(&self) -> f64 {
        self.layers
            .iter()
            .map(|layer| layer.thickness * layer.density)
            .sum()
    }

    /// Ply mid-surface offsets from the section mid-surface.
    fn ply_windows(&self) -> Vec<(f64, f64)> {
        let total = self.total_thickness();
        let mut z0 = -total / 2.0;
        let mut windows = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let z1 = z0 + layer.thickness;
            windows.push((z0, z1));
            z0 = z1;
        }
        windows
    }

    /// Trial update at one ply point from the committed totals.
    ///
    /// Returns the trial update plus the total strain it was evaluated at.
    /// Models needing plane-stress iteration get a Newton loop on the
    /// transverse total strain; linear models evaluate directly and rely on
    /// exact static condensation.
    fn trial_point(
        &self,
        layer: &Layer<M>,
        state_idx: usize,
        in_plane: &[f64; 3],
    ) -> (StressUpdate, [f64; 6]) {
        let committed = self.totals[state_idx];
        if !layer.model.requires_plane_stress_iteration() {
            let total = [in_plane[0], in_plane[1], 0.0, in_plane[2], 0.0, 0.0];
            let d_eps = [
                total[0] - committed[0],
                total[1] - committed[1],
                total[2] - committed[2],
                total[3] - committed[3],
                total[4] - committed[4],
                total[5] - committed[5],
            ];
            let update = layer.model.update(&self.states[state_idx], &d_eps);
            return (update, total);
        }
        // Transverse Newton iteration toward sigma_zz = 0.
        let mut total_zz = committed[2];
        let mut update = layer.model.update(&self.states[state_idx], &[0.0; 6]);
        for _ in 0..25 {
            let total = [in_plane[0], in_plane[1], total_zz, in_plane[2], 0.0, 0.0];
            let d_eps = [
                total[0] - committed[0],
                total[1] - committed[1],
                total[2] - committed[2],
                total[3] - committed[3],
                total[4] - committed[4],
                total[5] - committed[5],
            ];
            update = layer.model.update(&self.states[state_idx], &d_eps);
            if update.stress[2].abs() < 1e-9 {
                return (update, total);
            }
            let c_zz = update.tangent[2 * 6 + 2];
            if c_zz.abs() < 1e-30 {
                return (update, total);
            }
            total_zz -= update.stress[2] / c_zz;
        }
        let total = [in_plane[0], in_plane[1], total_zz, in_plane[2], 0.0, 0.0];
        (update, total)
    }

    /// Trial resultants from committed states (read-only).
    pub fn resultants(&self, eps_membrane: &[f64; 3], kappa: &[f64; 3]) -> LayeredResultants {
        let rule = gauss_rule(self.points_per_layer);
        let windows = self.ply_windows();
        let mut d_mm = [0.0; 9];
        let mut d_mb = [0.0; 9];
        let mut d_bb = [0.0; 9];
        let mut membrane = [0.0; 3];
        let mut moment = [0.0; 3];
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (z0, z1) = windows[layer_idx];
            let half = (z1 - z0) / 2.0;
            let mid = (z0 + z1) / 2.0;
            for (point, &(zeta, weight)) in rule.iter().enumerate() {
                let z = mid + half * zeta;
                let in_plane = [
                    eps_membrane[0] + z * kappa[0],
                    eps_membrane[1] + z * kappa[1],
                    eps_membrane[2] + z * kappa[2],
                ];
                let state_idx = layer_idx * self.points_per_layer + point;
                let (update, _) = self.trial_point(layer, state_idx, &in_plane);
                let condensed = condense_plane_stress(&update.tangent);
                let stress = condense_stress(&update.tangent, &update.stress);
                let wt = weight * half;
                for i in 0..3 {
                    membrane[i] += wt * stress[i];
                    moment[i] += wt * z * stress[i];
                }
                for i in 0..3 {
                    for j in 0..3 {
                        let c = condensed[i * 3 + j];
                        d_mm[i * 3 + j] += wt * c;
                        d_mb[i * 3 + j] += wt * z * c;
                        d_bb[i * 3 + j] += wt * z * z * c;
                    }
                }
            }
        }
        LayeredResultants {
            d_mm,
            d_mb,
            d_bb,
            membrane_force: membrane,
            moment,
        }
    }

    /// Commit a converged membrane/curvature state into all ply points.
    pub fn commit(&mut self, eps_membrane: &[f64; 3], kappa: &[f64; 3]) {
        let rule = gauss_rule(self.points_per_layer);
        let windows = self.ply_windows();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (z0, z1) = windows[layer_idx];
            let half = (z1 - z0) / 2.0;
            let mid = (z0 + z1) / 2.0;
            for (point, &(zeta, _)) in rule.iter().enumerate() {
                let z = mid + half * zeta;
                let in_plane = [
                    eps_membrane[0] + z * kappa[0],
                    eps_membrane[1] + z * kappa[1],
                    eps_membrane[2] + z * kappa[2],
                ];
                let state_idx = layer_idx * self.points_per_layer + point;
                let (update, total) = self.trial_point(layer, state_idx, &in_plane);
                self.plastic[state_idx] += update.delta_equiv_plastic_strain;
                layer.model.commit(&mut self.states[state_idx], &update);
                self.totals[state_idx] = total;
            }
        }
    }

    /// Maximum accumulated equivalent plastic strain over all ply points.
    pub fn max_equiv_plastic_strain(&self) -> f64 {
        self.plastic.iter().fold(0.0, |a, &b| a.max(b))
    }

    /// Number of ply points with nonzero accumulated plastic strain.
    pub fn yielded_point_count(&self) -> usize {
        self.plastic.iter().filter(|&&p| p > 0.0).count()
    }

    /// Committed point states, flattened [layer][point].
    pub fn point_states(&self) -> &[M::State] {
        &self.states
    }
}

impl LayeredSection<AnyLayerModel> {
    /// Build an elastic layered section from CLT laminate data.
    ///
    /// Each ply becomes an orthotropic C1 ply with the transformed reduced
    /// stiffness; `densities` supplies per-ply mass density in kg/m^3.
    pub fn from_laminate(laminate: &Laminate, densities: &[f64]) -> Self {
        assert_eq!(laminate.plies.len(), densities.len(), "one density per ply");
        let layers = laminate
            .plies
            .iter()
            .zip(densities.iter())
            .map(|(ply, &density)| {
                let q = ply.transformed_stiffness();
                Layer {
                    thickness: ply.thickness,
                    density,
                    model: AnyLayerModel::Orthotropic(OrthotropicPlyModel {
                        qbar: [
                            q[0][0], q[0][1], q[0][2], q[1][0], q[1][1], q[1][2], q[2][0], q[2][1],
                            q[2][2],
                        ],
                        transverse_modulus: ply.e2,
                        transverse_shear: ply.g12,
                    }),
                }
            })
            .collect();
        Self::new(layers, AnyLayerState::virgin)
    }

    /// Build a progressive-damage layered section from ply data.
    ///
    /// Each ply becomes a degrading C1 ply sharing one failure criterion
    /// and degradation rule; `strengths` and `densities` supply per-ply
    /// allowables and mass density in kg/m^3.
    pub fn from_degrading_laminate(
        plies: &[OrthotropicPly],
        strengths: &[PlyStrengths],
        criterion: FailureCriterion,
        rule: DegradationRule,
        densities: &[f64],
    ) -> Self {
        assert_eq!(plies.len(), strengths.len(), "one strength set per ply");
        assert_eq!(plies.len(), densities.len(), "one density per ply");
        let layers = plies
            .iter()
            .zip(strengths.iter())
            .zip(densities.iter())
            .map(|((ply, strengths), &density)| Layer {
                thickness: ply.thickness,
                density,
                model: AnyLayerModel::Degrading(DegradingPlyModel {
                    ply: *ply,
                    strengths: *strengths,
                    criterion,
                    rule,
                    transverse_modulus: ply.e2,
                    transverse_shear: ply.g12,
                }),
            })
            .collect();
        Self::new(layers, AnyLayerState::virgin)
    }
}

/// Gauss-Legendre rule on [-1, 1].
fn gauss_rule(n: usize) -> Vec<(f64, f64)> {
    match n {
        2 => {
            let a = 0.577_350_269_189_625_8;
            vec![(-a, 1.0), (a, 1.0)]
        }
        3 => {
            let a = 0.774_596_669_241_483_4;
            vec![(-a, 5.0 / 9.0), (0.0, 8.0 / 9.0), (a, 5.0 / 9.0)]
        }
        _ => panic!("only 2- or 3-point Gauss per ply"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::composites::OrthotropicPly;
    use crate::physics::solid::plasticity::HardeningLaw;

    #[test]
    fn condensation_reproduces_plane_stress_q() {
        let model = ElasticIsotropicModel {
            youngs_modulus: 200e9,
            poisson_ratio: 0.3,
        };
        let condensed = condense_plane_stress(&model.tangent_6x6());
        let e = 200e9;
        let nu = 0.3;
        let q11 = e / (1.0 - nu * nu);
        let q12 = nu * q11;
        let g = e / (2.0 * (1.0 + nu));
        let expected = [q11, q12, 0.0, q12, q11, 0.0, 0.0, 0.0, g];
        for i in 0..9 {
            assert!(
                (condensed[i] - expected[i]).abs() / expected[i].abs().max(1.0) < 1e-12,
                "slot {i}: {} vs {}",
                condensed[i],
                expected[i]
            );
        }
    }

    #[test]
    fn isotropic_layer_matches_classical_resultants() {
        let layer = Layer {
            thickness: 0.01,
            density: 7800.0,
            model: AnyLayerModel::Elastic(ElasticIsotropicModel {
                youngs_modulus: 200e9,
                poisson_ratio: 0.3,
            }),
        };
        let section = LayeredSection::new(vec![layer], AnyLayerState::virgin);
        let res = section.resultants(&[1e-3, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let e = 200e9;
        let nu = 0.3;
        let q11 = e / (1.0 - nu * nu);
        assert!((res.d_mm[0] - q11 * 0.01).abs() / (q11 * 0.01) < 1e-12);
        assert!((res.membrane_force[0] - q11 * 0.01 * 1e-3).abs() / (q11 * 0.01 * 1e-3) < 1e-12);
        // Symmetric single ply: no membrane-bending coupling.
        assert!(res.d_mb.iter().all(|v| v.abs() < 1e-6));
        let res_b = section.resultants(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        let d11 = q11 * 0.01_f64.powi(3) / 12.0;
        assert!((res_b.d_bb[0] - d11).abs() / d11 < 1e-12);
    }

    #[test]
    fn laminate_section_reproduces_clt_abd() {
        let plies = vec![
            OrthotropicPly {
                e1: 140e9,
                e2: 10e9,
                g12: 5e9,
                nu12: 0.25,
                thickness: 0.001,
                angle_rad: 0.0,
            },
            OrthotropicPly {
                e1: 140e9,
                e2: 10e9,
                g12: 5e9,
                nu12: 0.25,
                thickness: 0.001,
                angle_rad: std::f64::consts::FRAC_PI_2,
            },
        ];
        let laminate = Laminate {
            plies: plies.clone(),
        };
        let abd = laminate.stiffness_abd();
        let section = LayeredSection::from_laminate(&laminate, &[1600.0, 1600.0]);
        let res = section.resultants(&[1e-3, 2e-4, 0.0], &[0.5, -0.25, 0.0]);
        for i in 0..3 {
            for j in 0..3 {
                let scale = abd.a[i][j].abs().max(1.0);
                assert!(
                    (res.d_mm[i * 3 + j] - abd.a[i][j]).abs() / scale < 1e-12,
                    "A[{i}][{j}]"
                );
                let scale_b = abd.b[i][j].abs().max(1.0);
                assert!(
                    (res.d_mb[i * 3 + j] - abd.b[i][j]).abs() / scale_b < 1e-12,
                    "B[{i}][{j}]"
                );
                let scale_d = abd.d[i][j].abs().max(1.0);
                assert!(
                    (res.d_bb[i * 3 + j] - abd.d[i][j]).abs() / scale_d < 1e-12,
                    "D[{i}][{j}]"
                );
            }
        }
        // Unsymmetric cross-ply must show membrane-bending coupling.
        assert!(res.d_mb.iter().any(|v| v.abs() > 1.0));
    }

    #[test]
    fn j2_layer_yields_and_hardens() {
        let layer = Layer {
            thickness: 0.01,
            density: 7800.0,
            model: AnyLayerModel::J2(SmallStrainJ2Model {
                material: J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect),
            }),
        };
        let section = LayeredSection::new(vec![layer], AnyLayerState::virgin);
        // Below yield: membrane force follows the elastic line.
        let small = section.resultants(&[1e-4, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let e = 200e9;
        let q11 = e / (1.0 - 0.09);
        assert!((small.membrane_force[0] - q11 * 0.01 * 1e-4).abs() / (q11 * 0.01 * 1e-4) < 1e-9);
        // Above yield with constrained transverse strain (eps_yy = 0), the
        // von Mises plane-stress response saturates at 2/sqrt(3) sigma_y
        // (plastic incompressibility forces sigma_yy = sigma_xx / 2).
        let big = section.resultants(&[0.05, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let cap = 2.0 / 3.0_f64.sqrt() * 250e6 * 0.01;
        assert!(
            big.membrane_force[0] < 0.3 * q11 * 0.01 * 0.05,
            "plastic force {} exceeds elastic fraction",
            big.membrane_force[0]
        );
        assert!(
            (big.membrane_force[0] - cap).abs() / cap < 0.05,
            "plastic force {} vs cap {cap}",
            big.membrane_force[0]
        );
    }

    #[test]
    fn mixed_stack_commits_without_error_growth() {
        let layers = vec![
            Layer {
                thickness: 0.005,
                density: 7800.0,
                model: AnyLayerModel::Elastic(ElasticIsotropicModel {
                    youngs_modulus: 200e9,
                    poisson_ratio: 0.3,
                }),
            },
            Layer {
                thickness: 0.005,
                density: 7800.0,
                model: AnyLayerModel::J2(SmallStrainJ2Model {
                    material: J2Material::new(
                        200e9,
                        0.3,
                        250e6,
                        HardeningLaw::LinearIsotropic { modulus: 1e9 },
                    ),
                }),
            },
        ];
        let mut section = LayeredSection::new(layers, AnyLayerState::virgin);
        section.commit(&[1e-4, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let res = section.resultants(&[1e-4, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert!(res.membrane_force.iter().all(|v| v.is_finite()));
        assert!(res.d_mm.iter().all(|v| v.is_finite()));
    }

    fn carbon_ply(angle_rad: f64) -> OrthotropicPly {
        OrthotropicPly {
            e1: 140e9,
            e2: 10e9,
            g12: 5e9,
            nu12: 0.25,
            thickness: 0.001,
            angle_rad,
        }
    }

    fn carbon_strengths() -> PlyStrengths {
        PlyStrengths {
            xt: 1.95e9,
            xc: 1.48e9,
            yt: 4.8e7,
            yc: 2.0e8,
            s12: 7.9e7,
            s23: 7.9e7,
        }
    }

    fn degrading_layer(angle_rad: f64, rule: DegradationRule) -> Layer<AnyLayerModel> {
        let ply = carbon_ply(angle_rad);
        Layer {
            thickness: ply.thickness,
            density: 1600.0,
            model: AnyLayerModel::Degrading(DegradingPlyModel {
                ply,
                strengths: carbon_strengths(),
                criterion: FailureCriterion::Hashin,
                rule,
                transverse_modulus: ply.e2,
                transverse_shear: ply.g12,
            }),
        }
    }

    #[test]
    fn degrading_ply_fails_suddenly_in_transverse_tension() {
        let mut section = LayeredSection::new(
            vec![degrading_layer(0.0, DegradationRule::sudden())],
            AnyLayerState::virgin,
        );
        // Below initiation: virgin stiffness and zero damage.
        section.commit(&[0.0, 1e-3, 0.0], &[0.0, 0.0, 0.0]);
        let pre = section.resultants(&[0.0, 1e-3, 0.0], &[0.0, 0.0, 0.0]);
        let nu21 = 0.25 * 10e9 / 140e9;
        let q22 = 10e9 / (1.0 - 0.25 * nu21);
        assert!((pre.d_mm[4] - q22 * 0.001).abs() / (q22 * 0.001) < 1e-9);
        // Past transverse initiation: matrix damage saturates, E2 collapses
        // to residual while E1 is untouched.
        section.commit(&[0.0, 8e-3, 0.0], &[0.0, 0.0, 0.0]);
        let post = section.resultants(&[0.0, 8e-3, 0.0], &[0.0, 0.0, 0.0]);
        assert!(post.d_mm[4] / (q22 * 0.001) < 1e-4);
        // E1 itself is untouched; Q11 shifts slightly via Poisson coupling
        // (nu21 follows the degraded E2).
        let q11 = 140e9 / (1.0 - 0.25 * nu21);
        assert!((post.d_mm[0] - q11 * 0.001).abs() / (q11 * 0.001) < 1e-2);
        // Shear follows matrix damage.
        assert!(post.d_mm[8] / (5e9 * 0.001) < 1e-4);
    }

    #[test]
    fn degrading_ply_fails_in_fiber_tension() {
        let mut section = LayeredSection::new(
            vec![degrading_layer(0.0, DegradationRule::sudden())],
            AnyLayerState::virgin,
        );
        section.commit(&[2e-2, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let post = section.resultants(&[2e-2, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let nu21 = 0.25 * 10e9 / 140e9;
        let q11 = 140e9 / (1.0 - 0.25 * nu21);
        assert!(post.d_mm[0] / (q11 * 0.001) < 1e-4);
    }

    #[test]
    fn degrading_ply_rotates_strengths_to_material_axes() {
        // A 90-degree ply under laminate x-tension loads the transverse
        // material direction, so matrix (not fiber) damage must trigger.
        let mut section = LayeredSection::new(
            vec![degrading_layer(
                std::f64::consts::FRAC_PI_2,
                DegradationRule::sudden(),
            )],
            AnyLayerState::virgin,
        );
        section.commit(&[8e-3, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let post = section.resultants(&[8e-3, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let nu21 = 0.25 * 10e9 / 140e9;
        let q22 = 10e9 / (1.0 - 0.25 * nu21);
        // Laminate x-stiffness was E2-driven and collapses.
        assert!(post.d_mm[0] / (q22 * 0.001) < 1e-4);
    }

    #[test]
    fn degrading_ply_gradual_path_is_monotone() {
        let mut section = LayeredSection::new(
            vec![degrading_layer(0.0, DegradationRule::gradual())],
            AnyLayerState::virgin,
        );
        let mut prev = f64::INFINITY;
        for step in 1..=10 {
            let strain = [0.0, step as f64 * 1e-3, 0.0];
            section.commit(&strain, &[0.0, 0.0, 0.0]);
            let res = section.resultants(&strain, &[0.0, 0.0, 0.0]);
            assert!(res.d_mm[4] <= prev);
            assert!(res.d_mm[4] > 0.0);
            prev = res.d_mm[4];
        }
        // Exposure ~2 at the last step: d ~ 0.5, stiffness roughly halved.
        let nu21 = 0.25 * 10e9 / 140e9;
        let q22 = 10e9 / (1.0 - 0.25 * nu21);
        let ratio = prev / (q22 * 0.001);
        assert!(
            (ratio - 0.5).abs() < 0.15,
            "gradual stiffness ratio {ratio}"
        );
    }
}
