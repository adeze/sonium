//! Finite-strain J2 plasticity with multiplicative split.
//!
//! This module implements von Mises plasticity at finite strains using the
//! multiplicative split `F = Fe Fp` of the deformation gradient into elastic
//! and plastic parts. The stress update integrates a hypoelastic law with
//! the Green-Naghdi objective rate: each increment polar-decomposes the
//! relative deformation gradient `f = R U`, rotates stress and back-stress
//! by `R`, applies the small-strain radial return with the logarithmic
//! strain increment `ln U`, and advances `Fp` with the exponential map.
//!
//! The Green-Naghdi rate is skew-symmetric in the polar rotation spin, so a
//! superposed rigid rotation rotates the Cauchy stress exactly and produces
//! no spurious stress oscillation in simple shear (unlike the Jaumann rate).
//! Isotropic, kinematic, and combined hardening match the small-strain model
//! because both paths share [`return_mapping_3d`](super::return_mapping).
//!
//! # Strain increment convention
//!
//! The [`StressUpdateModel`] strain increment is the incremental (relative)
//! deformation gradient `f = F_new Fn_inv` in row-major order, with `f = I`
//! for a zero increment. Callers holding total deformation gradients form
//! `f` before calling `update`; the history state only stores `Fp`.

// Rust guideline compliant 2026-02-21

use super::material::{HardeningLaw, J2Material, PlasticState};
use super::return_mapping::return_mapping_3d;
use super::update::{StressUpdate, StressUpdateModel, internal_to_c1_6, von_mises_c1};

/// Identity 3x3 in row-major order.
const IDENTITY_3X3: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// Multiplies two row-major 3x3 matrices.
fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    c
}

/// Transposes a row-major 3x3 matrix.
fn mat3_transpose(a: &[f64; 9]) -> [f64; 9] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

/// Determinant of a row-major 3x3 matrix.
fn mat3_det(a: &[f64; 9]) -> f64 {
    a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6])
}

/// Maximum absolute entry of a 3x3 matrix.
fn mat3_max_abs(a: &[f64; 9]) -> f64 {
    a.iter().fold(0.0, |m, &x| m.max(x.abs()))
}

/// Eigendecomposition of a symmetric 3x3 via cyclic Jacobi rotations.
///
/// Returns eigenvalues and the orthonormal eigenvector matrix `v` stored by
/// columns, so that `a = V diag(values) V^T`.
fn sym_eigen_3x3(a: &[f64; 9]) -> ([f64; 3], [f64; 9]) {
    let mut d = *a;
    let mut v = IDENTITY_3X3;
    // Twelve sweeps drive the off-diagonal below 1e-15 for the
    // well-conditioned stretch tensors seen in incremental updates.
    for _ in 0..12 {
        let mut off = 0.0;
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            off += d[p * 3 + q] * d[p * 3 + q];
        }
        if off < 1e-30 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = d[p * 3 + q];
            if apq * apq < 1e-32 {
                continue;
            }
            let app = d[p * 3 + p];
            let aqq = d[q * 3 + q];
            let theta = (aqq - app) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            for k in 0..3 {
                let dpk = d[p * 3 + k];
                let dqk = d[q * 3 + k];
                d[p * 3 + k] = c * dpk - s * dqk;
                d[q * 3 + k] = s * dpk + c * dqk;
            }
            for k in 0..3 {
                let dkp = d[k * 3 + p];
                let dkq = d[k * 3 + q];
                d[k * 3 + p] = c * dkp - s * dkq;
                d[k * 3 + q] = s * dkp + c * dkq;
            }
            for k in 0..3 {
                let vkp = v[k * 3 + p];
                let vkq = v[k * 3 + q];
                v[k * 3 + p] = c * vkp - s * vkq;
                v[k * 3 + q] = s * vkp + c * vkq;
            }
        }
    }
    ([d[0], d[4], d[8]], v)
}

/// Rebuilds a symmetric 3x3 from eigenvectors and mapped eigenvalues.
///
/// Computes `V diag(f(values)) V^T` for a scalar function `f`.
fn sym_apply_3x3(v: &[f64; 9], values: &[f64; 3], f: impl Fn(f64) -> f64) -> [f64; 9] {
    let mapped = [f(values[0]), f(values[1]), f(values[2])];
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = v[i * 3] * mapped[0] * v[j * 3]
                + v[i * 3 + 1] * mapped[1] * v[j * 3 + 1]
                + v[i * 3 + 2] * mapped[2] * v[j * 3 + 2];
        }
    }
    out
}

/// Polar decomposition of a 3x3 with positive determinant.
///
/// Returns the rotation `r` plus the eigenvalues and eigenvectors of the
/// right Cauchy-Green tensor `f^T f`, so callers can form `U`, `U^-1`, and
/// `ln U` without a second eigendecomposition.
///
/// # Panics
///
/// Panics when `f` has a non-positive determinant, which cannot happen for
/// a physical incremental deformation gradient.
fn polar_decompose(f: &[f64; 9]) -> ([f64; 9], [f64; 3], [f64; 9]) {
    assert!(
        mat3_det(f) > 0.0,
        "polar decomposition needs det(f) > 0 for a physical increment"
    );
    let ft = mat3_transpose(f);
    let c = mat3_mul(&ft, f);
    let (values, v) = sym_eigen_3x3(&c);
    // Clamp tiny eigenvalues from roundoff; C = f^T f is positive definite.
    let clamped = values.map(|l| l.max(1e-30));
    let u_inv = sym_apply_3x3(&v, &clamped, |l| 1.0 / l.sqrt());
    let r = mat3_mul(f, &u_inv);
    (r, clamped, v)
}

/// Matrix exponential of a 3x3 via scaling and squaring.
///
/// Accurate to roundoff for the small flow increments of the exponential
/// map; also robust for larger arguments through repeated squaring.
fn mat3_exp(a: &[f64; 9]) -> [f64; 9] {
    let norm = mat3_max_abs(a);
    let mut squarings = 0;
    let mut scaled = *a;
    while mat3_max_abs(&scaled) > 0.5 {
        for x in scaled.iter_mut() {
            *x *= 0.5;
        }
        squarings += 1;
    }
    if norm == 0.0 {
        return IDENTITY_3X3;
    }
    // Taylor series of exp(scaled), converged below 1e-16.
    let mut result = IDENTITY_3X3;
    let mut term = IDENTITY_3X3;
    for k in 1..32 {
        term = mat3_mul(&term, &scaled);
        let inv_k = 1.0 / k as f64;
        let mut delta: f64 = 0.0;
        for i in 0..9 {
            term[i] *= inv_k;
            result[i] += term[i];
            delta = delta.max(term[i].abs());
        }
        if delta < 1e-16 {
            break;
        }
    }
    for _ in 0..squarings {
        result = mat3_mul(&result, &result);
    }
    result
}

/// Converts a C1 Voigt 6-vector to a full symmetric 3x3 tensor.
///
/// The shear entries are tensor components (stress-like); callers holding
/// engineering shear strains must halve them before calling this helper.
fn voigt_c1_to_tensor(v: &[f64; 6]) -> [f64; 9] {
    [v[0], v[3], v[5], v[3], v[1], v[4], v[5], v[4], v[2]]
}

/// Converts a symmetric 3x3 tensor to a C1 Voigt 6-vector.
///
/// When `engineering_shear` is set, shear entries are doubled (strain-like);
/// otherwise they are plain tensor components (stress-like).
fn tensor_to_voigt_c1(t: &[f64; 9], engineering_shear: bool) -> [f64; 6] {
    let s = if engineering_shear { 2.0 } else { 1.0 };
    [
        t[0],
        t[4],
        t[8],
        s * 0.5 * (t[1] + t[3]),
        s * 0.5 * (t[5] + t[7]),
        s * 0.5 * (t[6] + t[2]),
    ]
}

/// Rotates a symmetric tensor stored in C1 Voigt order by `r`.
///
/// Returns `R T R^T` in C1 Voigt order with stress-like shear entries.
fn rotate_voigt_c1(v: &[f64; 6], r: &[f64; 9]) -> [f64; 6] {
    let t = voigt_c1_to_tensor(v);
    let rt = mat3_transpose(r);
    tensor_to_voigt_c1(&mat3_mul(&mat3_mul(r, &t), &rt), false)
}

/// Voigt index pairs for C1 order `[xx, yy, zz, xy, yz, zx]`.
const VOIGT_PAIRS_C1: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)];

/// Isotropic elastic tangent in C1 Voigt order, row-major 6x6.
fn elastic_tangent_c1(k: f64, g: f64) -> [f64; 36] {
    let lambda = k - 2.0 * g / 3.0;
    let mut c = [0.0; 36];
    for i in 0..3 {
        for j in 0..3 {
            c[i * 6 + j] = if i == j { lambda + 2.0 * g } else { lambda };
        }
    }
    for i in 3..6 {
        c[i * 6 + i] = g;
    }
    c
}

/// Rotates a C1 row-major 6x6 tangent by a rotation matrix.
///
/// Expands the tangent to its fourth-order tensor, applies the rotation on
/// all four legs, and contracts back to Voigt form. The expansion uses the
/// identity `C_{v(I)v(J)} = T_{IJ}` with minor symmetries, which holds
/// exactly for engineering-strain Voigt tangents.
fn rotate_tangent_c1(tangent: &[f64; 36], r: &[f64; 9]) -> [f64; 36] {
    // Expand to C[i][j][k][l] with minor symmetries.
    let mut c = [0.0; 81];
    for (idx_i, &(a, b)) in VOIGT_PAIRS_C1.iter().enumerate() {
        for (idx_j, &(d, e)) in VOIGT_PAIRS_C1.iter().enumerate() {
            let value = tangent[idx_i * 6 + idx_j];
            c[((a * 3 + b) * 3 + d) * 3 + e] = value;
            c[((b * 3 + a) * 3 + d) * 3 + e] = value;
            c[((a * 3 + b) * 3 + e) * 3 + d] = value;
            c[((b * 3 + a) * 3 + e) * 3 + d] = value;
        }
    }
    // Rotate: C'_ijkl = R_im R_jn R_ko R_lp C_mnop.
    let mut rotated = [0.0; 81];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    let mut sum = 0.0;
                    for m in 0..3 {
                        for n in 0..3 {
                            let rijn = r[i * 3 + m] * r[j * 3 + n];
                            for o in 0..3 {
                                for p in 0..3 {
                                    sum += rijn
                                        * r[k * 3 + o]
                                        * r[l * 3 + p]
                                        * c[((m * 3 + n) * 3 + o) * 3 + p];
                                }
                            }
                        }
                    }
                    rotated[((i * 3 + j) * 3 + k) * 3 + l] = sum;
                }
            }
        }
    }
    // Contract back to Voigt form.
    let mut out = [0.0; 36];
    for (idx_i, &(a, b)) in VOIGT_PAIRS_C1.iter().enumerate() {
        for (idx_j, &(d, e)) in VOIGT_PAIRS_C1.iter().enumerate() {
            out[idx_i * 6 + idx_j] = rotated[((a * 3 + b) * 3 + d) * 3 + e];
        }
    }
    out
}

/// Finite-strain J2 elastoplastic material.
///
/// Wraps the small-strain [`J2Material`] parameters (elastic moduli, yield
/// stress, hardening law) with a Green-Naghdi objective stress update and
/// multiplicative plasticity. All hardening laws supported by [`J2Material`]
/// behave identically to the small-strain model in the limit of small
/// incremental stretches.
///
/// The stress, tangent, plastic strain, and `Fp` transport exactly under
/// arbitrary incremental rotations. The kinematic back-stress increment is
/// evaluated in the rotated configuration, but its spin transport assumes
/// small incremental rotations (the co-rotational element regime): C1's
/// `commit` receives no rotation, so a large single-step rigid rotation
/// leaves a kinematic back-stress behind by that rotation. Isotropic
/// hardening is exact under all rotations. Callers driving large rotations
/// in one step with kinematic hardening should substep the rotation.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::{
///     FiniteStrainJ2Material, FiniteStrainState, StressUpdateModel,
/// };
///
/// let material = FiniteStrainJ2Material::mild_steel();
/// let mut state = FiniteStrainState::new();
/// // One percent uniaxial stretch of the incremental gradient.
/// let f = [1.01, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// let update = material.update(&state, &f);
/// material.commit(&mut state, &update);
/// assert!(update.stress[0] > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FiniteStrainJ2Material {
    /// Small-strain parameters shared with the infinitesimal model.
    pub base: J2Material,
}

impl FiniteStrainJ2Material {
    /// Creates a finite-strain J2 material from elastic and hardening data.
    ///
    /// Takes Young's modulus in Pa, Poisson's ratio, initial yield stress
    /// in Pa, and the hardening law shared with [`J2Material`].
    pub fn new(e: f64, nu: f64, sigma_y: f64, hardening: HardeningLaw) -> Self {
        Self {
            base: J2Material::new(e, nu, sigma_y, hardening),
        }
    }

    /// Mild steel: E=210 GPa, nu=0.3, sigma_y=250 MPa, H=1 GPa.
    pub fn mild_steel() -> Self {
        Self {
            base: J2Material::mild_steel(),
        }
    }

    /// Aluminum 6061-T6: E=70 GPa, nu=0.33, sigma_y=276 MPa.
    pub fn aluminum_6061() -> Self {
        Self {
            base: J2Material::aluminum_6061(),
        }
    }

    /// Shear modulus G = E / (2(1+nu)).
    pub fn shear_modulus(&self) -> f64 {
        self.base.shear_modulus()
    }

    /// Bulk modulus K = E / (3(1-2nu)).
    pub fn bulk_modulus(&self) -> f64 {
        self.base.bulk_modulus()
    }
}

/// History state of one finite-strain material point.
///
/// Stores the plastic deformation gradient `Fp` of the multiplicative split
/// `F = Fe Fp` together with the Cauchy stress, back-stress, and accumulated
/// plastic strain. The total deformation gradient is owned by the caller,
/// which forms the incremental gradient `f = F_new Fn_inv` per step.
#[derive(Debug, Clone)]
pub struct FiniteStrainState {
    /// Plastic deformation gradient, row-major 3x3.
    pub fp: [f64; 9],
    /// Cauchy stress in C1 Voigt order (Pa).
    pub stress: [f64; 6],
    /// Back-stress in C1 Voigt order (Pa).
    pub back_stress: [f64; 6],
    /// Accumulated equivalent plastic strain.
    pub equiv_plastic_strain: f64,
    /// Whether the last committed step was yielding.
    pub yielding: bool,
}

impl FiniteStrainState {
    /// Creates a virgin state with `Fp = I` and zero stress.
    pub fn new() -> Self {
        Self {
            fp: IDENTITY_3X3,
            stress: [0.0; 6],
            back_stress: [0.0; 6],
            equiv_plastic_strain: 0.0,
            yielding: false,
        }
    }
}

impl Default for FiniteStrainState {
    fn default() -> Self {
        Self::new()
    }
}

impl StressUpdateModel for FiniteStrainJ2Material {
    type State = FiniteStrainState;
    type StrainIncrement = [f64; 9];

    fn update(&self, state: &Self::State, f: &Self::StrainIncrement) -> StressUpdate {
        // Green-Naghdi update: polar-decompose f = R U, rotate the stress
        // and back-stress by R, then run the shared radial return with the
        // logarithmic strain increment d = ln U.
        let (r, eigen_c, eigen_v) = polar_decompose(f);
        let d_tensor = sym_apply_3x3(&eigen_v, &eigen_c, |l| 0.5 * l.ln());
        let d_c1 = tensor_to_voigt_c1(&d_tensor, true);

        // A numerically zero strain increment (pure rotation to roundoff)
        // must transport stress without re-yielding: the committed state
        // already satisfies f <= 0, so any positive trial value here is
        // roundoff. Skipping the return mapping keeps rotation holds exact.
        // Genuine steps below 1e-14 carry stress changes under 1e-5 Pa.
        if d_c1.iter().fold(0.0, |m: f64, &x| m.max(x.abs())) < 1e-14 {
            let elastic = elastic_tangent_c1(self.bulk_modulus(), self.shear_modulus());
            return StressUpdate {
                stress: rotate_voigt_c1(&state.stress, &r),
                tangent: rotate_tangent_c1(&elastic, &r),
                delta_equiv_plastic_strain: 0.0,
                yielding: false,
            };
        }

        let rotated_stress = rotate_voigt_c1(&state.stress, &r);
        let rotated_back = rotate_voigt_c1(&state.back_stress, &r);

        let mut trial = PlasticState::new_3d();
        trial
            .stress
            .copy_from_slice(&super::update::c1_to_internal_6(&rotated_stress));
        trial
            .back_stress
            .copy_from_slice(&super::update::c1_to_internal_6(&rotated_back));
        trial.equiv_plastic_strain = state.equiv_plastic_strain;
        trial.yielding = state.yielding;

        let d_internal = super::update::c1_to_internal_6(&d_c1);
        let legacy = return_mapping_3d(&self.base, &trial, &d_internal);

        let stress_in: [f64; 6] = legacy.stress.as_slice().try_into().unwrap_or_else(|_| {
            panic!("return_mapping_3d must return 6 stress components");
        });
        let tangent_in: [f64; 36] = legacy.tangent.as_slice().try_into().unwrap_or_else(|_| {
            panic!("return_mapping_3d must return a 6x6 tangent");
        });
        let stress = internal_to_c1_6(&stress_in);
        // The return mapping ran in the rotated configuration, so its stress
        // is already the spatial Cauchy stress; only the tangent still needs
        // rotation back into the spatial frame.
        let tangent = rotate_tangent_c1(&super::update::internal_to_c1_36(&tangent_in), &r);

        StressUpdate {
            stress,
            tangent,
            delta_equiv_plastic_strain: legacy.delta_equiv_plastic_strain,
            yielding: legacy.yielding,
        }
    }

    fn commit(&self, state: &mut Self::State, update: &StressUpdate) {
        state.stress = update.stress;
        state.equiv_plastic_strain += update.delta_equiv_plastic_strain;
        state.yielding = update.yielding;

        if update.yielding && update.delta_equiv_plastic_strain > 0.0 {
            // Prager back-stress rule, matching the small-strain commit.
            let c = self.base.kinematic_hardening_modulus();
            let q = von_mises_c1(&update.stress);
            if c > 0.0 && q > 1e-15 {
                let p = (update.stress[0] + update.stress[1] + update.stress[2]) / 3.0;
                let scale = (2.0 / 3.0) * c * update.delta_equiv_plastic_strain / q;
                for i in 0..6 {
                    let normal = if i < 3 {
                        update.stress[i] - p
                    } else {
                        update.stress[i]
                    };
                    state.back_stress[i] += scale * normal;
                }
            }

            // Multiplicative update Fp_new = exp(Delta_gamma N) Fp with the
            // associative flow direction N = 1.5 dev(xi) / q. N is traceless,
            // so det(Fp) = 1 is preserved exactly (isochoric plastic flow).
            let p = (update.stress[0] + update.stress[1] + update.stress[2]) / 3.0;
            let pa = (state.back_stress[0] + state.back_stress[1] + state.back_stress[2]) / 3.0;
            let xi: [f64; 6] = std::array::from_fn(|i| {
                let s = if i < 3 {
                    update.stress[i] - p
                } else {
                    update.stress[i]
                };
                let a = if i < 3 {
                    state.back_stress[i] - pa
                } else {
                    state.back_stress[i]
                };
                s - a
            });
            // xi is already deviatoric; von_mises_c1 removes any mean anyway.
            let q_xi = von_mises_c1(&xi);
            if q_xi > 1e-15 {
                let scale = 1.5 * update.delta_equiv_plastic_strain / q_xi;
                let flow = [
                    scale * xi[0],
                    scale * xi[3],
                    scale * xi[5],
                    scale * xi[3],
                    scale * xi[1],
                    scale * xi[4],
                    scale * xi[5],
                    scale * xi[4],
                    scale * xi[2],
                ];
                state.fp = mat3_mul(&mat3_exp(&flow), &state.fp);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_diff_6(a: &[f64; 6], b: &[f64; 6]) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0, |m, (&x, &y)| m.max((x - y).abs()))
    }

    #[test]
    fn test_identity_increment_is_noop() {
        let material = FiniteStrainJ2Material::mild_steel();
        let mut state = FiniteStrainState::new();
        // Prestress the state so a wrong rotation would show up.
        state.stress = [100.0e6, -50.0e6, 0.0, 25.0e6, 0.0, 0.0];
        let before = state.clone();

        let update = material.update(&state, &IDENTITY_3X3);
        assert!(!update.yielding);
        assert_eq!(update.delta_equiv_plastic_strain, 0.0);
        assert!(max_diff_6(&update.stress, &before.stress) < 1e-6);

        material.commit(&mut state, &update);
        assert!((mat3_det(&state.fp) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_pure_rotation_rotates_stress() {
        let material = FiniteStrainJ2Material::mild_steel();
        let mut state = FiniteStrainState::new();
        state.stress = [100.0e6, -50.0e6, 20.0e6, 25.0e6, -10.0e6, 5.0e6];
        state.back_stress = [10.0e6, -5.0e6, 0.0, 2.0e6, 0.0, 0.0];

        // 90-degree rotation about z: stress must rotate exactly, no yielding.
        let f = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let update = material.update(&state, &f);
        assert!(!update.yielding);
        assert_eq!(update.delta_equiv_plastic_strain, 0.0);

        let expected = rotate_voigt_c1(&state.stress, &f);
        assert!(
            max_diff_6(&update.stress, &expected) < 1e-3,
            "rotation objectivity failed: {update:?}"
        );
        // Von Mises stress is rotation invariant.
        assert!((von_mises_c1(&update.stress) - von_mises_c1(&state.stress)).abs() < 1e-3);
    }

    #[test]
    fn test_small_strain_parity() {
        let small = J2Material::new(
            200e9,
            0.3,
            250e6,
            HardeningLaw::LinearIsotropic { modulus: 1e9 },
        );
        let finite = FiniteStrainJ2Material::new(
            200e9,
            0.3,
            250e6,
            HardeningLaw::LinearIsotropic { modulus: 1e9 },
        );
        // Ten small uniaxial stretch steps, total ~1% (into plasticity).
        // The lateral contraction matches the small-strain Poisson value so
        // both models see the same physical loading (ln(1+x) ~ x).
        let mut s_small = PlasticState::new_3d();
        let mut s_finite = FiniteStrainState::new();
        for _ in 0..10 {
            let d_eps = [0.001, -0.0003, -0.0003, 0.0, 0.0, 0.0];
            let u = small.update(&s_small, &d_eps);
            small.commit(&mut s_small, &u);

            let f = [1.001, 0.0, 0.0, 0.0, 0.9997, 0.0, 0.0, 0.0, 0.9997];
            let v = finite.update(&s_finite, &f);
            finite.commit(&mut s_finite, &v);
        }
        assert!(s_small.yielding && s_finite.yielding);
        let small_c1 = internal_to_c1_6(&s_small.stress.as_slice().try_into().unwrap());
        let diff = max_diff_6(&small_c1, &s_finite.stress);
        let scale = von_mises_c1(&s_finite.stress);
        assert!(
            diff / scale < 0.02,
            "finite/small strain mismatch: {diff:.3e} vs {scale:.3e}"
        );
        assert!(
            (s_small.equiv_plastic_strain - s_finite.equiv_plastic_strain).abs()
                / s_finite.equiv_plastic_strain
                < 0.05,
            "plastic strain mismatch"
        );
    }

    #[test]
    fn test_kinematic_parity() {
        let small = J2Material::new(
            200e9,
            0.3,
            250e6,
            HardeningLaw::LinearKinematic { modulus: 5e9 },
        );
        let finite = FiniteStrainJ2Material::new(
            200e9,
            0.3,
            250e6,
            HardeningLaw::LinearKinematic { modulus: 5e9 },
        );
        let mut s_small = PlasticState::new_3d();
        let mut s_finite = FiniteStrainState::new();
        for _ in 0..10 {
            let d_eps = [0.001, -0.0003, -0.0003, 0.0, 0.0, 0.0];
            let u = small.update(&s_small, &d_eps);
            small.commit(&mut s_small, &u);
            let f = [1.001, 0.0, 0.0, 0.0, 0.9997, 0.0, 0.0, 0.0, 0.9997];
            let v = finite.update(&s_finite, &f);
            finite.commit(&mut s_finite, &v);
        }
        let small_back: [f64; 6] = s_small.back_stress.as_slice().try_into().unwrap();
        let small_back_c1 = internal_to_c1_6(&small_back);
        let diff = max_diff_6(&small_back_c1, &s_finite.back_stress);
        let scale = von_mises_c1(&s_finite.stress);
        assert!(
            diff / scale < 0.05,
            "back-stress mismatch: {diff:.3e} vs {scale:.3e}"
        );
        assert!(
            s_finite.back_stress[0] > 0.0,
            "tension needs positive back-stress"
        );
    }

    #[test]
    fn test_plastic_flow_is_isochoric() {
        let material = FiniteStrainJ2Material::mild_steel();
        let mut state = FiniteStrainState::new();
        // Large shear: F = [1 g 0; 0 1 0; 0 0 1] accumulated in steps.
        for step in 1..=20 {
            let g_prev = 0.05 * (step - 1) as f64;
            let g_new = 0.05 * step as f64;
            // Incremental gradient from the previous total gradient.
            let f = [1.0, g_new - g_prev, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            let update = material.update(&state, &f);
            material.commit(&mut state, &update);
        }
        assert!(state.yielding);
        assert!(
            (mat3_det(&state.fp) - 1.0).abs() < 1e-12,
            "det(Fp) = {}",
            mat3_det(&state.fp)
        );
    }

    #[test]
    fn test_elastic_tangent_matches_hooke() {
        let material = FiniteStrainJ2Material::mild_steel();
        let state = FiniteStrainState::new();
        let update = material.update(&state, &IDENTITY_3X3);
        assert!(!update.yielding);
        // C1 elastic tangent: C00 = lambda + 2G, C01 = lambda, C33 = G.
        let g = material.shear_modulus();
        let lambda = material.bulk_modulus() - 2.0 * g / 3.0;
        assert!((update.tangent[0] - (lambda + 2.0 * g)).abs() < 1.0);
        assert!((update.tangent[1] - lambda).abs() < 1.0);
        assert!((update.tangent[3 * 6 + 3] - g).abs() / g < 1e-12);
    }

    #[test]
    fn test_tangent_symmetry_after_rotation() {
        let material = FiniteStrainJ2Material::mild_steel();
        let state = FiniteStrainState::new();
        // Shear plus 30-degree rotation: tangent must stay symmetric.
        let angle = std::f64::consts::FRAC_PI_6;
        let (s, c) = angle.sin_cos();
        let f = [c, -s + 0.001, 0.0, s, c, 0.0, 0.0, 0.0, 1.0];
        let update = material.update(&state, &f);
        for i in 0..6 {
            for j in (i + 1)..6 {
                let diff = (update.tangent[i * 6 + j] - update.tangent[j * 6 + i]).abs();
                let scale = update.tangent[i * 6 + i].abs().max(1.0);
                assert!(diff / scale < 1e-9, "tangent not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_mat3_exp_rotation_generator() {
        // exp of a skew generator is the corresponding rotation.
        let angle = 0.3;
        let w = [0.0, -angle, 0.0, angle, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r = mat3_exp(&w);
        let (s, c) = angle.sin_cos();
        let expected = [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0];
        for i in 0..9 {
            assert!(
                (r[i] - expected[i]).abs() < 1e-12,
                "exp mismatch at {i}: {} vs {}",
                r[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_polar_decomposition_round_trip() {
        let f = [1.0, 0.4, 0.1, 0.0, 1.1, -0.2, 0.05, 0.0, 0.9];
        let (r, values, v) = polar_decompose(&f);
        let u = sym_apply_3x3(&v, &values, |l| l.sqrt());
        let rebuilt = mat3_mul(&r, &u);
        for i in 0..9 {
            assert!((rebuilt[i] - f[i]).abs() < 1e-12, "polar mismatch at {i}");
        }
        // R must be proper orthogonal.
        let rrt = mat3_mul(&r, &mat3_transpose(&r));
        for i in 0..9 {
            assert!(
                (rrt[i] - IDENTITY_3X3[i]).abs() < 1e-12,
                "R not orthogonal at {i}"
            );
        }
        assert!((mat3_det(&r) - 1.0).abs() < 1e-12);
    }
}
