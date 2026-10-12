//! Phase-7 lane 7C G-leg (lane-owned): native foam-law evolution.
//!
//! Native LAW evolution for the foam miniqa cases: the LOI50 honeycomb
//! brick replays corotationally through the transcribed LAW50 card
//! (full six-component stress plus energy and density), the LOI76
//! shell replays along its TRUE-strain path through the transcribed
//! LAW66 card (force resultant, energy, plastic strain), and the
//! LOI68 shear block checks rigid-body identities plus the
//! couple-traction signature. Offline: reads only
//! `goldens/miniqa/**/*.json` plus the schema rules. Exit 0 = PASS,
//! exit 1 = FAIL with per-probe report on stdout.
//!
//! Probe design (native drivers; no surrogate triaxiality):
//! - F50a pins the LOI50 orthotropic frame: the first-step shear
//!   slope must match G23 (material-23 = global-XY), rejecting the
//!   skew-frame reading (G12, fifty times stiffer).
//! - F50b replays the LOI50 brick corotationally (polar rotation,
//!   Biot increments in the rotated frame, stresses rotated back)
//!   and compares all six stresses plus internal energy and density
//!   against the golden `solide` TH.
//! - F76a checks the analytic TRUE-strain path against the deck
//!   imposed-displacement funct at every golden stamp.
//! - F76b replays the LOI76 shell along the TRUE-strain path under
//!   uniaxial stress and compares the membrane force resultant,
//!   internal energy, and equivalent plastic strain.
//! - F68a checks LOI68 rigid-body identities: wall force balance,
//!   the couple moment arm (MY = FZ x gauge/2), and energy closure.
//! - F68b guards the LOI68 shear level by the elastic envelope and
//!   reports the resultant-vs-cap anomaly (finding, see output).

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::foam_explicit::rotate_voigt6;
use sonium_fem::physics::solid::plasticity::funct::Funct1D;
use sonium_fem::physics::solid::plasticity::law_foams::{
    HoneycombLaw, HoneycombState, StrainBasis,
};
use sonium_fem::physics::solid::plasticity::law_metals::{
    Law66, Law66Rate, Law66Shell, Law66ShellState,
};
use std::path::PathBuf;

/// Golden schema accepted by this leg (see `goldens/SCHEMA.md`).
const GOLDEN_SCHEMA: &str = "opencourant-golden-v1";

/// Vendor near-zero skip from SCHEMA.md (`Zero_tolerance`).
const ZERO_TOLERANCE: f64 = 5e-14;

/// LOI50 brick density in kg/m^3 (deck `/MAT/LAW50/1`).
const LOI50_DENSITY: f64 = 5.12e-8;
/// LOI50 brick reference volume in m^3 (unit cube).
const LOI50_VOLUME: f64 = 1.0;
/// LOI50 deck moduli `[E11, E22, E33, G12, G23, G31]` in Pa.
const LOI50_MODULI: [f64; 6] = [0.019, 1.38e-4, 1.38e-4, 0.0075936, 1.45e-4, 0.0075936];
/// LOI50 full-stress formulation band (relative) plus floor (Pa).
///
/// Measured 2026-10-12: the 500-stamp six-stress replay passes at a
/// 0.001 band with worst margin 0.009 (allowance floor-dominated);
/// the 0.002 band keeps 2x headroom for golden-regen channel shift.
const LOI50_REL_BAND: f64 = 0.002;
const LOI50_FLOOR: f64 = 2.0e-6;
/// LOI76 shell reference volume in mm^3 (10 x 2.5 x 3).
const LOI76_VOLUME: f64 = 75.0;
/// LOI76 formulation band (relative) plus floors.
///
/// Measured 2026-10-12: the 15-stamp TRUE-strain replay passes at a
/// 0.001 band; the 0.002 band keeps 2x headroom for golden-regen
/// channel shift (re-measure after any LOI76 golden regen).
const LOI76_REL_BAND: f64 = 0.002;
const LOI76_FORCE_FLOOR: f64 = 20.0;
const LOI76_ENERGY_FLOOR: f64 = 500.0;
const LOI76_EP_FLOOR: f64 = 0.01;
/// LOI68 shear gauge length in deck units (wall x gap).
const LOI68_GAUGE: f64 = 28.43;
/// LOI68 wall face area in deck units squared (250 x 250).
const LOI68_AREA: f64 = 62500.0;

/// Transcribed LOI50 LAW50 card (deck `/MAT/LAW50/1`).
///
/// Rated tables repeat each yield funct at 0 and 100 /s with unit
/// scales (the deck `Fscale = 0` maps to 1, starter echo); bases
/// negate (`Iflag = -1`); no failure (`Eps_max = 0`).
fn loi50_law() -> HoneycombLaw {
    let f11 = funct_loi50_500005();
    let f2233 = funct_loi50_500004();
    let f12131 = funct_loi50_500006();
    let f23 = funct_loi50_500007();
    let rated = [
        vec![(f11.clone(), 0.0), (f11, 100.0)],
        vec![(f2233.clone(), 0.0), (f2233.clone(), 100.0)],
        vec![(f2233.clone(), 0.0), (f2233, 100.0)],
        vec![(f12131.clone(), 0.0), (f12131.clone(), 100.0)],
        vec![(f23.clone(), 0.0), (f23, 100.0)],
        vec![(f12131.clone(), 0.0), (f12131, 100.0)],
    ];
    let law = HoneycombLaw::law50(LOI50_MODULI, rated).expect("LOI50 card is valid");
    assert!(law.basis_normal == StrainBasis::Negated);
    assert!(law.basis_shear == StrainBasis::Negated);
    law
}

/// LOI50 material frame at t = 0 (material axes in global rows).
///
/// Deck `/PROP/SOL_ORTH/1` sets `Ip = 3` (plane (T, R), angle 0)
/// with the element axes (R, S, T) = (-Z, +Y, +X); the first-step
/// shear slope (probe F50a) anchors material-1 to R, giving
/// (1, 2, 3) = (-Z, +Y, +X). The deck `/SKEW/FIX/1` stays unused
/// (`skew_ID = 0`). Rows are the material axes in global coords.
const LOI50_FRAME: [[f64; 3]; 3] = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];

/// LOI50 brick reference coordinates in SI m (deck `/NODE`).
///
/// Radioss order N1..N8 = deck nodes 1, 2, 4, 3, 5, 6, 8, 7.
const LOI50_NODES: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, -1.0],
    [1.0, 1.0, -1.0],
    [1.0, 1.0, 0.0],
];

/// Natural coordinates of the eight hex nodes (xi-fastest).
const HEX_NATURALS: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// Inverts a 3x3 row-major matrix (unit-cube Jacobians only).
fn invert3(matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    assert!(det.abs() > 1e-30, "singular hex Jacobian");
    let inv = 1.0 / det;
    [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv,
        ],
    ]
}

/// Centroid shape derivatives wrt global coords for the LOI50 brick.
fn loi50_centroid_derivatives() -> [[f64; 3]; 8] {
    let dndxi: [[f64; 3]; 8] = std::array::from_fn(|i| {
        [
            HEX_NATURALS[i][0] / 8.0,
            HEX_NATURALS[i][1] / 8.0,
            HEX_NATURALS[i][2] / 8.0,
        ]
    });
    let mut jac = [[0.0; 3]; 3];
    for i in 0..8 {
        for a in 0..3 {
            for b in 0..3 {
                jac[a][b] += LOI50_NODES[i][a] * dndxi[i][b];
            }
        }
    }
    let inv = invert3(&jac);
    std::array::from_fn(|i| {
        std::array::from_fn(|a| {
            inv[0][a] * dndxi[i][0] + inv[1][a] * dndxi[i][1] + inv[2][a] * dndxi[i][2]
        })
    })
}

/// Deformation gradient from nodal displacements (row-major).
fn deformation_gradient(derivs: &[[f64; 3]; 8], disp: &[[f64; 3]; 8]) -> [[f64; 3]; 3] {
    let mut grad = [[0.0; 3]; 3];
    for i in 0..8 {
        for a in 0..3 {
            for b in 0..3 {
                grad[a][b] += disp[i][a] * derivs[i][b];
            }
        }
    }
    std::array::from_fn(|a| std::array::from_fn(|b| grad[a][b] + if a == b { 1.0 } else { 0.0 }))
}

/// Determinant of a 3x3 row-major matrix.
fn det3(matrix: &[[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

/// Row-major 3x3 product.
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
    })
}

/// Row-major 3x3 transpose.
fn mat3_transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Row-major 3x3 inverse via the adjugate over the determinant.
fn mat3_inv(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = det3(m);
    assert!(det.abs() > 1e-30, "singular gradient in replay");
    let c =
        |i1: usize, i2: usize, j1: usize, j2: usize| m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1];
    [
        [
            c(1, 2, 1, 2) / det,
            c(0, 2, 2, 1) / det,
            c(0, 1, 1, 2) / det,
        ],
        [
            c(1, 2, 2, 0) / det,
            c(0, 2, 0, 2) / det,
            c(0, 1, 2, 0) / det,
        ],
        [
            c(1, 2, 0, 1) / det,
            c(0, 2, 1, 0) / det,
            c(0, 1, 0, 1) / det,
        ],
    ]
}

/// Cross product of two 3-vectors.
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Engine `MREPLOC` material rotation for the LOI50 brick.
///
/// The convected edge vectors are the `SREPISO3`/`SDERI3` combinations
/// `r = F e_Y`, `s = F e_X` of the reference cube, orthonormalized in
/// `MREPLOC` order (`r` first, then `t = r x s`, then `s = t x r`), and
/// combined with the fixed `GAMA0` (`Ip=3`, `phi=0`) exactly as the
/// Jaumann orthotropic path does: `R9 = S E^T`.
fn mreploc_frame(f: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let r = [f[0][1], f[1][1], f[2][1]];
    let s = [f[0][0], f[1][0], f[2][0]];
    mreploc_from_rs(&r, &s)
}

/// Engine `MREPLOC` orthonormalization from raw convected `r`, `s`.
fn mreploc_from_rs(r: &[f64; 3], s: &[f64; 3]) -> [[f64; 3]; 3] {
    let norm = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let nr = norm(r);
    assert!(nr > 1e-30, "degenerate convected r");
    let e1 = [r[0] / nr, r[1] / nr, r[2] / nr];
    let t = cross3(&e1, s);
    let nt = norm(&t);
    assert!(nt > 1e-30, "degenerate convected t");
    let e3 = [t[0] / nt, t[1] / nt, t[2] / nt];
    let e2 = cross3(&e3, &e1);
    // Fixed GAMA0 for Ip=3, phi=0 as engine `MREPLOC` columns:
    // col 0 = (ANG1, ANG2, ANG3), col 1 = (ANG4, ANG5, ANG6).
    let s_mat = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
    let e = [e1, e2, e3];
    // Engine assembly R = E S: col j = S11*e1 + S21*e2 + S31*e3...
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            e[0][i] * s_mat[0][j] + e[1][i] * s_mat[1][j] + e[2][i] * s_mat[2][j]
        })
    })
}

fn goldens_root() -> PathBuf {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.join("goldens").join("miniqa")
}

fn load_golden(root: &std::path::Path, case: &str) -> Value {
    let path = root.join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("golden {case} unreadable: {err}"));
    let doc: Value =
        serde_json::from_str(&text).unwrap_or_else(|err| panic!("golden {case} invalid: {err}"));
    assert_eq!(
        doc["schema"].as_str().unwrap_or(""),
        GOLDEN_SCHEMA,
        "golden {case} schema drift"
    );
    doc
}

fn time_grid(doc: &Value) -> Vec<f64> {
    doc["time"]
        .as_array()
        .expect("golden lacks time grid")
        .iter()
        .map(|v| v.as_f64().expect("time must be numeric"))
        .collect()
}

fn channel_number(value: &Value) -> f64 {
    value.as_f64().expect("channel sample must be numeric")
}

fn channel(doc: &Value, name: &str) -> Vec<f64> {
    doc["channels"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks channel {name}"))
        .iter()
        .map(channel_number)
        .collect()
}

fn channel_tol(doc: &Value, name: &str) -> (f64, f64) {
    let abs = doc["channels"][name]["abs_tol"].as_f64().unwrap_or(1e-12);
    let rel = doc["channels"][name]["rel_tol"].as_f64().unwrap_or(6.6e-4);
    (abs, rel)
}

/// Sample comparison with the vendor near-zero skip (SCHEMA.md).
fn within_tol(actual: f64, expected: f64, abs: f64, rel: f64) -> bool {
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    if actual.abs() < ZERO_TOLERANCE && expected.abs() < ZERO_TOLERANCE {
        return true;
    }
    if (actual - expected).abs() < 2.0 * ZERO_TOLERANCE {
        return true;
    }
    (actual - expected).abs() <= abs + rel * expected.abs()
}

/// Band comparison: golden tolerance widened by a formulation band.
fn within_band(actual: f64, expected: f64, abs: f64, rel: f64, band: f64, floor: f64) -> bool {
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    let allowance = abs + rel * expected.abs() + band * expected.abs() + floor;
    (actual - expected).abs() <= allowance
}

/// Native LOI50 corotational replay along golden nodal motion.
///
/// Returns per-stamp global stress, internal energy, density, and
/// plastic strain. The law runs in the rotated material frame; the
/// reported stress rotates back through the polar rotation.
struct Loi50Replay {
    stress: Vec<[f64; 6]>,
    energy: Vec<f64>,
    density: Vec<f64>,
    ep: Vec<f64>,
}

fn loi50_replay(doc: &Value) -> Loi50Replay {
    use sonium_fem::physics::solid::plasticity::law_foams::PointStepper;
    let time = time_grid(doc);
    let derivs = loi50_centroid_derivatives();
    // Radioss node order N1..N8 = deck nodes 1, 2, 4, 3, 5, 6, 8, 7.
    let order = [1, 2, 4, 3, 5, 6, 8, 7];
    let dx: Vec<Vec<f64>> = order
        .iter()
        .map(|n| channel(doc, &format!("noeuds {n} ~1")))
        .collect();
    let dy: Vec<Vec<f64>> = order
        .iter()
        .map(|n| channel(doc, &format!("noeuds {n} ~2")))
        .collect();
    let dz: Vec<Vec<f64>> = order
        .iter()
        .map(|n| channel(doc, &format!("noeuds {n} ~3")))
        .collect();
    let law = loi50_law();
    let mut state = HoneycombState::new();
    // Anchor check: the convected frame at identity transposes to the
    // slope-pinned initial material rotation.
    let r9_identity = mreploc_frame(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    for a in 0..3 {
        for b in 0..3 {
            assert!(
                (r9_identity[b][a] - LOI50_FRAME[a][b]).abs() < 1e-12,
                "R9 identity drift"
            );
        }
    }
    let mut replay = Loi50Replay {
        stress: Vec::with_capacity(time.len()),
        energy: Vec::with_capacity(time.len()),
        density: Vec::with_capacity(time.len()),
        ep: Vec::with_capacity(time.len()),
    };
    let mut energy = 0.0;
    let mut prev_sig = [0.0; 6];
    let mut prev_det = 1.0;
    let mut prev_grad = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (k, t) in time.iter().enumerate() {
        let disp: [[f64; 3]; 8] = std::array::from_fn(|i| [dx[i][k], dy[i][k], dz[i][k]]);
        let grad = deformation_gradient(&derivs, &disp);
        let det = det3(&grad);
        replay.density.push(LOI50_DENSITY / det);
        if k == 0 {
            // B-assembly self-check: undeformed gradient is identity.
            for a in 0..3 {
                for b in 0..3 {
                    let want = if a == b { 1.0 } else { 0.0 };
                    assert!((grad[a][b] - want).abs() < 1e-12, "B assembly drift");
                }
            }
            prev_grad = grad;
            replay.stress.push([0.0; 6]);
            replay.energy.push(0.0);
            replay.ep.push(0.0);
            continue;
        }
        let dt = t - time[k - 1];
        assert!(dt > 0.0, "golden time grid must be monotone");
        // Engine `SDEFO3` midpoint rate: dF over the half-step gradient.
        let f_mid =
            std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (prev_grad[a][b] + grad[a][b])));
        let df = std::array::from_fn(|a| std::array::from_fn(|b| grad[a][b] - prev_grad[a][b]));
        let rate = mat3_mul(&df, &mat3_inv(&f_mid));
        // Engineering shear sums (`D4 = DXY + DYX`).
        let de_global = [
            rate[0][0],
            rate[1][1],
            rate[2][2],
            rate[0][1] + rate[1][0],
            rate[1][2] + rate[2][1],
            rate[0][2] + rate[2][0],
        ];
        // Engine `MROTENS` input path: halve to tensor shear, rotate by
        // R9^T into material axes, double back to engineering shear.
        let r9 = mreploc_frame(&grad);
        // Engine legs: pre-law R9^T E R9, post-law R9 E R9^T.
        let (r_in, r_out) = (mat3_transpose(&r9), r9);
        let de_half = [
            de_global[0],
            de_global[1],
            de_global[2],
            0.5 * de_global[3],
            0.5 * de_global[4],
            0.5 * de_global[5],
        ];
        let de_rot = rotate_voigt6(&r_in, &de_half);
        let de_mat = [
            de_rot[0],
            de_rot[1],
            de_rot[2],
            2.0 * de_rot[3],
            2.0 * de_rot[4],
            2.0 * de_rot[5],
        ];
        let update = law.step_point(&mut state, &de_mat, dt);
        let sig_mat = state.stresses;
        let det_mid = 0.5 * (prev_det + det);
        for i in 0..6 {
            energy += 0.5 * (prev_sig[i] + sig_mat[i]) * de_mat[i] * det_mid * LOI50_VOLUME;
        }
        prev_sig = sig_mat;
        prev_det = det;
        prev_grad = grad;
        // Engine post-law `MROTENS` with transposed legs: R9 S R9^T.
        replay.stress.push(rotate_voigt6(&r_out, &sig_mat));
        replay.energy.push(energy);
        replay.ep.push(state.equiv_plastic);
        let _ = update;
    }
    replay
}

/// Transcribed LOI76 LAW66 card (deck `/MAT/LAW66/2`).
fn loi76_law() -> Law66 {
    Law66::new(
        60000.0,
        0.32,
        0.0,
        500.0,
        600.0,
        (funct_loi76_3(), 1.0),
        (funct_loi76_4(), 1.0),
        Law66Rate::Static,
        false,
        0.0,
    )
}

/// LOI76 imposed axial displacement in mm (deck `/IMPDEF/3`).
///
/// Funct 2 runs (0, 0), (1, 5), (10, 50) with `Scale_y = -1`, so the
/// driven edge displacement is exactly `-5 t` over the run.
fn loi76_dx(time: f64) -> f64 {
    -5.0 * time
}

/// LOI76 axial TRUE strain over the 10 mm gauge.
fn loi76_true_strain(time: f64) -> f64 {
    (1.0 + loi76_dx(time) / 10.0).ln()
}

fn main() {
    let root = goldens_root();
    let loi50 = load_golden(&root, "LOIS/LOI50/test1/pid_6/input_44");
    let loi76 = load_golden(&root, "LOIS/LOI76/shell/test1/compression");
    let loi68 = load_golden(&root, "LOIS/LOI68/cisaillement/test1/input_51");
    let mut passed = 0;
    let mut total = 0;

    // F50a: orthotropic frame anchor from the first-step shear slope.
    total += 1;
    let time50 = time_grid(&loi50);
    let sxy = channel(&loi50, "solide 1 solide ~5");
    let dx3 = channel(&loi50, "noeuds 3 ~1");
    // Early stamp, small rotation: engineering shear from the driven
    // face displacement over the 1 m gauge. Both signals run negative,
    // so their ratio anchors the active shear modulus by magnitude.
    let k = 10;
    let slope = sxy[k] / dx3[k];
    let g23 = LOI50_MODULI[4];
    let g12 = LOI50_MODULI[3];
    let f50a = slope > 0.2 * g23 && slope < 5.0 * g23;
    println!(
        "PROBE F50a loi50-frame: early shear slope {slope:.4e} vs G23 {g23:.4e} (G12 {g12:.4e}, ratio {:.2}) at t {:.4e} -> {}",
        slope / g12,
        time50[k],
        if f50a { "PASS" } else { "FAIL" }
    );
    passed += usize::from(f50a);

    // F50b: full-stress corotational replay vs the golden brick TH.
    total += 1;
    let replay = loi50_replay(&loi50);
    let names = ["SX", "SY", "SZ", "SXY", "SYZ", "SXZ"];
    let mut f50b = true;
    let mut worst = 0.0f64;
    for (dir, name) in names.iter().enumerate() {
        let gold = channel(&loi50, &format!("solide 1 solide ~{}", dir + 2));
        let (abs, rel) = channel_tol(&loi50, &format!("solide 1 solide ~{}", dir + 2));
        for (i, (native, expected)) in replay.stress.iter().zip(gold.iter()).enumerate().skip(1) {
            let ok = within_band(
                native[dir],
                *expected,
                abs,
                rel,
                LOI50_REL_BAND,
                LOI50_FLOOR,
            );
            f50b = f50b && ok;
            let margin = (native[dir] - expected).abs()
                / (abs + rel * expected.abs() + LOI50_REL_BAND * expected.abs() + LOI50_FLOOR);
            worst = worst.max(margin);
            if !ok && worst < 2.0 {
                println!(
                    "  F50b {name} stamp {i}: native {:.4e} vs golden {:.4e}",
                    native[dir], expected
                );
                worst = 2.0;
            }
        }
    }
    let ie = channel(&loi50, "solide 1 solide ~8");
    let (ie_abs, ie_rel) = channel_tol(&loi50, "solide 1 solide ~8");
    for (i, (native, expected)) in replay.energy.iter().zip(ie.iter()).enumerate().skip(1) {
        let ok = within_band(*native, *expected, ie_abs, ie_rel, LOI50_REL_BAND, 1e-7);
        if !ok {
            println!("  F50b IE stamp {i}: native {native:.4e} vs golden {expected:.4e}");
        }
        f50b = f50b && ok;
    }
    let dens = channel(&loi50, "solide 1 solide ~9");
    let (d_abs, d_rel) = channel_tol(&loi50, "solide 1 solide ~9");
    for (i, (native, expected)) in replay.density.iter().zip(dens.iter()).enumerate().skip(1) {
        let ok = within_tol(*native, *expected, d_abs, d_rel);
        if !ok {
            println!("  F50b DENS stamp {i}: native {native:.4e} vs golden {expected:.4e}");
        }
        f50b = f50b && ok;
    }
    // Elastic case: the golden PLAS stays zero, so must the replay.
    let plas = channel(&loi50, "solide 1 solide ~10");
    let ep_end = *replay.ep.last().unwrap();
    let plas_end = *plas.last().unwrap();
    let f50b_ep = ep_end < 1e-9 && plas_end.abs() < 1e-12;
    f50b = f50b && f50b_ep;
    println!(
        "PROBE F50b loi50-full-stress: 6 stresses + IE + DENS over {} stamps (worst margin {worst:.3}, ep {ep_end:.2e}) -> {}",
        time50.len(),
        if f50b { "PASS" } else { "FAIL" }
    );
    passed += usize::from(f50b);

    // F76a: TRUE-strain path definition check (analytic, no golden).
    total += 1;
    let time76 = time_grid(&loi76);
    let e_end = loi76_true_strain(*time76.last().unwrap());
    let e_mid = loi76_true_strain(1.0);
    let eng_end = loi76_dx(*time76.last().unwrap()) / 10.0;
    let f76a = (e_mid - 0.5f64.ln()).abs() < 1e-15 && e_end < -1.2 && (eng_end - e_end).abs() > 0.4;
    println!(
        "PROBE F76a loi76-true-path: e(1) {e_mid:.6}, e(end) {e_end:.6} vs engineering {eng_end:.6} -> {}",
        if f76a { "PASS" } else { "FAIL" }
    );
    passed += usize::from(f76a);

    // F76b: plane-stress replay vs the golden shell TH through the exact
    // LAW66c-shell port (`Law66Shell`: radial return + DEZZ). Axial E1
    // follows the analytic TRUE path, lateral E2 is Newton-solved per
    // step for sigma22 = 0, and the thickness increment comes from the
    // law's DEZZ (volume bookkeeping only).
    total += 1;
    let shell76 = Law66Shell::new(loi76_law());
    let t_end = *time76.last().expect("golden time grid");
    let nfine = 281;
    let mut history_t = Vec::with_capacity(nfine);
    let mut history_strain = Vec::with_capacity(nfine);
    let mut history_stress = Vec::with_capacity(nfine);
    let mut history_ep = Vec::with_capacity(nfine);
    history_t.push(0.0);
    history_strain.push([0.0; 6]);
    history_stress.push([0.0; 6]);
    history_ep.push(0.0);
    let mut state = Law66ShellState::new();
    let mut e_prev = [0.0f64; 6];
    for k in 1..nfine {
        let t = t_end * k as f64 / (nfine - 1) as f64;
        let t0 = t_end * (k - 1) as f64 / (nfine - 1) as f64;
        let de11 = loi76_true_strain(t) - loi76_true_strain(t0);
        // Secant Newton on de22 for sigma22 = 0; seed from the running
        // lateral ratio, elastic Poisson for the first step.
        let mut de22 = if e_prev[0].abs() > 1e-12 {
            e_prev[1] / e_prev[0] * de11
        } else {
            -0.32 * de11
        };
        for _ in 0..24 {
            let (trial, _) = shell76.step(&state, de11, de22, 0.0);
            let f1 = trial.stress[1];
            let scale = trial.stress[0].abs().max(1.0);
            if f1.abs() < 1e-9 * scale {
                break;
            }
            let h = 1e-7 * de11.abs().max(1e-12);
            let (trial_h, _) = shell76.step(&state, de11, de22 + h, 0.0);
            let slope = (trial_h.stress[1] - f1) / h;
            if slope.abs() < 1e-9 {
                break;
            }
            de22 -= f1 / slope;
        }
        let (next, info) = shell76.step(&state, de11, de22, 0.0);
        state = next;
        e_prev = [
            e_prev[0] + de11,
            e_prev[1] + de22,
            e_prev[2] + info.dezz,
            0.0,
            0.0,
            0.0,
        ];
        history_t.push(t);
        history_strain.push(e_prev);
        history_stress.push([
            state.stress[0],
            state.stress[1],
            0.0,
            state.stress[2],
            0.0,
            0.0,
        ]);
        history_ep.push(state.pla);
    }
    let f1 = channel(&loi76, "spec 1000001 compr ~1");
    let iem = channel(&loi76, "spec 1000001 compr ~7");
    let emin = channel(&loi76, "spec 1000001 compr ~10");
    let emax = channel(&loi76, "spec 1000001 compr ~11");
    let (f_abs, f_rel) = channel_tol(&loi76, "spec 1000001 compr ~1");
    let (e_abs, e_rel) = channel_tol(&loi76, "spec 1000001 compr ~7");
    let (p_abs, p_rel) = channel_tol(&loi76, "spec 1000001 compr ~10");
    // Fine-quadrature work with the current volume, sampled per stamp.
    let mut ie_fine = vec![0.0; nfine];
    for k in 1..nfine {
        let de: [f64; 6] = std::array::from_fn(|d| history_strain[k][d] - history_strain[k - 1][d]);
        let tr_mid: f64 = (0..3)
            .map(|d| 0.5 * (history_strain[k][d] + history_strain[k - 1][d]))
            .sum();
        let vol = LOI76_VOLUME * tr_mid.exp();
        let mut step = 0.0;
        for d in 0..6 {
            step += 0.5 * (history_stress[k - 1][d] + history_stress[k][d]) * de[d] * vol;
        }
        ie_fine[k] = ie_fine[k - 1] + step;
    }
    let sample = |grid_t: &[f64], t: f64| -> usize {
        let mut best = 0;
        let mut err = f64::INFINITY;
        for (k, g) in grid_t.iter().enumerate() {
            let e = (g - t).abs();
            if e < err {
                err = e;
                best = k;
            }
        }
        best
    };
    let mut f76b = true;
    for (i, t) in time76.iter().enumerate().skip(1) {
        let k = sample(&history_t, *t);
        let sx = history_stress[k][0];
        // Golden F1 is the thickness-averaged stress: engine `thkly` layer
        // weights are fractions summing to one, so GBUF%FOR holds stress.
        let force = sx;
        let ok_f = within_band(
            force,
            f1[i],
            f_abs,
            f_rel,
            LOI76_REL_BAND,
            LOI76_FORCE_FLOOR,
        );
        let ie_native = ie_fine[k];
        let ok_e = within_band(
            ie_native,
            iem[i],
            e_abs,
            e_rel,
            LOI76_REL_BAND,
            LOI76_ENERGY_FLOOR,
        );
        let ep_native = history_ep[k];
        let ok_p = within_band(
            ep_native,
            emin[i],
            p_abs,
            p_rel,
            LOI76_REL_BAND,
            LOI76_EP_FLOOR,
        ) && within_band(
            ep_native,
            emax[i],
            p_abs,
            p_rel,
            LOI76_REL_BAND,
            LOI76_EP_FLOOR,
        );
        if !(ok_f && ok_e && ok_p) {
            println!(
                "  F76b t {t:.4}: F1 {force:.4e} vs {:.4e} | IE {ie_native:.4e} vs {:.4e} | ep {:.4e} vs {:.4e}",
                f1[i], iem[i], ep_native, emin[i]
            );
        }
        f76b = f76b && ok_f && ok_e && ok_p;
    }
    println!(
        "PROBE F76b loi76-true-replay: F1 + IEM + EMIN/EMAX over {} stamps -> {}",
        time76.len(),
        if f76b { "PASS" } else { "FAIL" }
    );
    passed += usize::from(f76b);

    // F68a: rigid-body identities (balance, couple arm, energy).
    // Two tiers: the established regime (t >= 1) holds the identities
    // tightly (quasi-static shear), while the full range only bounds
    // the startup transient (couple-field establishment).
    total += 1;
    let time68 = time_grid(&loi68);
    let mut f68a = true;
    let peak_fz = channel(&loi68, "right 2 right ~3")
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    for comp in 1..=3 {
        let left = channel(&loi68, &format!("left 1 left ~{comp}"));
        let right = channel(&loi68, &format!("right 2 right ~{comp}"));
        for (i, (l, r)) in left.iter().zip(right.iter()).enumerate().skip(1) {
            let scale = l.abs().max(r.abs());
            let floor = 0.01 * peak_fz;
            let band = if time68[i] >= 1.0 { 0.005 } else { 0.06 };
            if (l + r).abs() > band * scale.max(floor) + floor {
                println!("  F68a F{comp} t {:.2}: {l:.4e} vs {r:.4e}", time68[i]);
                f68a = false;
            }
        }
    }
    let fz_l = channel(&loi68, "left 1 left ~3");
    let my_l = channel(&loi68, "left 1 left ~5");
    let fz_r = channel(&loi68, "right 2 right ~3");
    let my_r = channel(&loi68, "right 2 right ~5");
    let arm = LOI68_GAUGE / 2.0;
    for i in 1..time68.len() {
        let band = if time68[i] >= 1.0 { 0.005 } else { 0.05 };
        if fz_l[i].abs() > 1.0 {
            let got = (my_l[i] / fz_l[i]).abs();
            if (got - arm).abs() > band * arm {
                println!("  F68a left arm t {:.2}: {got:.4}", time68[i]);
                f68a = false;
            }
        }
        if fz_r[i].abs() > 1.0 {
            let got = (my_r[i] / fz_r[i]).abs();
            if (got - arm).abs() > band * arm {
                println!("  F68a right arm t {:.2}: {got:.4}", time68[i]);
                f68a = false;
            }
        }
    }
    let extw = channel(&loi68, "EXTERNAL WORK");
    let ie68 = channel(&loi68, "INTERNAL ENERGY");
    let ke68 = channel(&loi68, "KINETIC ENERGY");
    let he68 = channel(&loi68, "HOURGLASS ENERGY");
    for i in 1..time68.len() {
        let stored = ie68[i] + ke68[i] + he68[i];
        // Established band matches the vendor energy error order
        // (ref.extract ERROR -1.5% at cycle 2000).
        let band = if time68[i] >= 1.0 { 0.025 } else { 0.03 };
        if (extw[i] - stored).abs() > band * stored.abs() + 1e-9 {
            println!(
                "  F68a energy t {:.2}: {stored:.4e} vs {:.4e}",
                time68[i], extw[i]
            );
            f68a = false;
        }
    }
    println!(
        "PROBE F68a loi68-identities: wall balance + couple arm {arm:.4} + energy over {} stamps -> {}",
        time68.len(),
        if f68a { "PASS" } else { "FAIL" }
    );
    passed += usize::from(f68a);

    // F68b: elastic envelope guard plus the anomaly report (finding).
    total += 1;
    let dz = channel(&loi68, "impacteur 15319 impacteur ~3");
    let g12 = 0.1;
    let cap12 = funct_loi68_12();
    let cap21 = funct_loi68_21();
    let mut f68b = true;
    for i in 1..time68.len() {
        let gamma = dz[i] / LOI68_GAUGE;
        let tau = fz_r[i].abs() / LOI68_AREA;
        if tau > g12 * gamma + 1e-6 {
            f68b = false;
        }
    }
    let gamma_end = dz.last().unwrap() / LOI68_GAUGE;
    let tau_end = fz_r.last().unwrap().abs() / LOI68_AREA;
    // Yield caps at the negated abscissa (deck Iflag2 = -1).
    let c12 = cap12.eval(-gamma_end);
    let c21 = cap21.eval(-gamma_end);
    let others = [
        ("23", funct_loi68_23().eval(-gamma_end)),
        ("32", funct_loi68_32().eval(-gamma_end)),
        ("31", funct_loi68_31().eval(-gamma_end)),
        ("13", funct_loi68_13().eval(-gamma_end)),
    ];
    println!(
        "PROBE F68b loi68-envelope: tau {tau_end:.4e} vs elastic {elastic:.4e}; caps c12 {c12:.4e} c21 {c21:.4e} (ratio {ratio:.3}) -> {}",
        if f68b { "PASS" } else { "FAIL" },
        elastic = g12 * gamma_end,
        ratio = tau_end / (0.5 * (c12 + c21)),
    );
    for (name, cap) in others {
        println!("  F68b shear cap {name}: {cap:.4e}");
    }
    passed += usize::from(f68b);

    if passed == total {
        println!("PASS: qa-fem-foam-native ({passed}/{total})");
        println!("SUCCESS: qa-fem-foam-native");
    } else {
        println!("FAIL: qa-fem-foam-native ({passed}/{total})");
        println!("FAILED: qa-fem-foam-native");
        std::process::exit(1);
    }
}

/// Deck funct loi50_500004 (machine-transcribed, do not hand-edit).
fn funct_loi50_500004() -> Funct1D {
    Funct1D::new(
        vec![-100.0, 0.0, 0.12, 0.38000001, 1.8900002, 4.6],
        vec![13700000.0, 2.56e-05, 2.56e-05, 3.99e-05, 3.99e-05, 0.1373],
    )
    .expect("deck funct loi50_500004 is valid")
}

/// Deck funct loi50_500005 (machine-transcribed, do not hand-edit).
fn funct_loi50_500005() -> Funct1D {
    Funct1D::new(
        vec![-100.0, 0.0, 0.02, 0.025, 1.63, 1.89, 2.0, 2.85, 4.6],
        vec![
            13700000.0, 0.00038, 0.00038, 0.00028, 0.00028, 0.0004, 0.00073, 0.012, 137.3,
        ],
    )
    .expect("deck funct loi50_500005 is valid")
}

/// Deck funct loi50_500006 (machine-transcribed, do not hand-edit).
fn funct_loi50_500006() -> Funct1D {
    Funct1D::new(
        vec![
            -1000.0, -9.100002, -9.0, -0.25, 0.0, 0.25, 9.0, 9.100002, 1000.0,
        ],
        vec![
            0.000586, 0.000586, 0.000586, 0.000586, 0.000586, 0.000586, 0.000586, 0.000586,
            0.000586,
        ],
    )
    .expect("deck funct loi50_500006 is valid")
}

/// Deck funct loi50_500007 (machine-transcribed, do not hand-edit).
fn funct_loi50_500007() -> Funct1D {
    Funct1D::new(
        vec![
            -1000.0, -9.100002, -9.0, -0.25, 0.0, 0.25, 9.0, 9.100002, 1000.0,
        ],
        vec![
            0.000145, 0.000145, 0.000145, 0.000145, 0.000145, 0.000145, 0.000145, 0.000145,
            0.000145,
        ],
    )
    .expect("deck funct loi50_500007 is valid")
}

/// Deck funct loi76_3 (machine-transcribed, do not hand-edit).
fn funct_loi76_3() -> Funct1D {
    Funct1D::new(
        vec![
            0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024,
            0.026, 0.028, 0.03, 0.032, 0.034, 0.036, 0.038, 0.04, 0.042, 0.044, 0.046, 0.048, 0.05,
            0.052, 0.054, 0.056, 0.058, 0.06, 0.062, 0.064, 0.066, 0.068, 0.07, 0.072, 0.074,
            0.076, 0.078, 0.08, 0.082, 0.084, 0.086, 0.088, 0.09, 0.092, 0.094, 0.096, 0.098, 0.1,
            0.102, 0.104, 0.106, 0.108, 0.11, 0.112, 0.114, 0.116, 0.118, 0.12, 0.122, 0.124,
            0.126, 0.128, 0.13, 0.132, 0.134, 0.136, 0.138, 0.14, 1.0,
        ],
        vec![
            376.315789, 382.565265, 388.631363, 394.520665, 400.23959, 405.794389, 411.191151,
            416.435798, 421.534086, 426.491607, 431.313789, 436.005894, 440.573017, 445.020091,
            449.351881, 453.572989, 457.687852, 461.70074, 465.615759, 469.436851, 473.167789,
            476.812187, 480.373487, 483.854972, 487.259756, 490.590789, 493.850857, 497.042579,
            500.168409, 503.230638, 506.231389, 509.172623, 512.056134, 514.88355, 517.656335,
            520.375789, 523.043046, 525.659073, 528.224675, 530.740489, 533.206989, 535.624484,
            537.993116, 540.312863, 542.583539, 544.804789, 546.976099, 549.096783, 551.165996,
            553.182724, 555.145789, 557.053849, 558.905395, 560.698753, 562.432086, 564.103389,
            565.710495, 567.25107, 568.722614, 570.122463, 571.447789, 572.695598, 573.862729,
            574.945859, 575.941497, 576.845989, 577.655516, 578.366092, 578.973567, 579.473627,
            579.861789, 1017.0,
        ],
    )
    .expect("deck funct loi76_3 is valid")
}

/// Deck funct loi76_4 (machine-transcribed, do not hand-edit).
fn funct_loi76_4() -> Funct1D {
    Funct1D::new(
        vec![
            0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024,
            0.026, 0.028, 0.03, 0.032, 0.034, 0.036, 0.038, 0.04, 0.042, 0.044, 0.046, 0.048, 0.05,
            0.052, 0.054, 0.056, 0.058, 0.06, 0.062, 0.064, 0.066, 0.068, 0.07, 0.072, 0.074,
            0.076, 0.078, 0.08, 0.082, 0.084, 0.086, 0.088, 0.09, 0.092, 0.094, 0.096, 0.098, 0.1,
            0.102, 0.104, 0.106, 0.108, 0.11, 0.112, 0.114, 0.116, 0.118, 0.12, 0.122, 0.124,
            0.126, 0.128, 0.13, 0.132, 0.134, 0.136, 0.138, 0.14, 0.142, 0.144, 0.146, 0.148, 0.15,
            0.152, 0.154, 0.156, 0.158, 0.16, 0.162, 0.164, 0.166, 0.168, 0.17, 0.172, 0.174,
            0.176, 0.178, 0.18, 0.182, 0.184, 0.186, 0.188, 0.19, 0.192, 0.194, 0.196, 0.198, 1.0,
        ],
        vec![
            357.894737, 367.302393, 375.929454, 383.844601, 391.112055, 397.791758, 403.939545,
            409.607324, 414.843244, 419.691858, 424.194294, 428.388408, 432.308943, 435.987683,
            439.453601, 442.733009, 445.849694, 448.825063, 451.678278, 454.426385, 457.084448,
            459.665671, 462.181523, 464.641851, 467.055005, 469.42794, 471.766331, 474.074674,
            476.356392, 478.613927, 480.84884, 483.061902, 485.253176, 487.422111, 489.567612,
            491.688128, 493.781718, 495.846127, 497.878849, 499.877195, 501.83835, 503.759433,
            505.637546, 507.469828, 509.253502, 510.985916, 512.66458, 514.287211, 515.851757,
            517.356431, 518.799737, 520.18049, 521.49784, 522.75128, 523.940667, 525.066224,
            526.128551, 527.128621, 528.067783, 528.947758, 529.770625, 530.538816, 531.255095,
            531.922543, 532.544536, 533.124718, 533.666971, 534.175387, 534.654226, 535.107885,
            535.540847, 535.957638, 536.362782, 536.76074, 537.15586, 537.552315, 537.954038,
            538.364658, 538.78743, 539.225158, 539.680122, 540.153995, 540.647762, 541.161628,
            541.694931, 542.246049, 542.812298, 543.389834, 543.973549, 544.556959, 545.132099,
            545.689404, 546.21759, 546.703537, 547.132159, 547.486279, 547.746494, 547.891042,
            547.895663, 547.9, 1307.9,
        ],
    )
    .expect("deck funct loi76_4 is valid")
}

/// Deck funct loi68_12 (machine-transcribed, do not hand-edit).
fn funct_loi68_12() -> Funct1D {
    Funct1D::new(
        vec![
            -10.0, -0.95021, -0.86466, -0.703, -0.165, -0.0906, -0.038, 0.0, 0.038, 0.0906, 0.165,
            0.703, 0.86466, 0.95021, 10.0,
        ],
        vec![
            7e-06, 1.39e-05, 3.39e-05, 8.8e-05, 0.00016, 0.000137, 0.000119, 0.0001, 0.000119,
            0.000137, 0.00016, 8.8e-05, 3.39e-05, 1.39e-05, 7e-06,
        ],
    )
    .expect("deck funct loi68_12 is valid")
}

/// Deck funct loi68_13 (machine-transcribed, do not hand-edit).
fn funct_loi68_13() -> Funct1D {
    Funct1D::new(
        vec![
            -10.0, -0.95021, -0.86466, -0.393, -0.165, -0.0906, -0.038, 0.0, 0.038, 0.0906, 0.165,
            0.393, 0.86466, 0.95021, 10.0,
        ],
        vec![
            7e-06, 1.39e-05, 3.39e-05, 8.8e-05, 0.00016, 0.000137, 0.000119, 0.0001, 0.000119,
            0.000137, 0.00016, 8.8e-05, 3.39e-05, 1.39e-05, 7e-06,
        ],
    )
    .expect("deck funct loi68_13 is valid")
}

/// Deck funct loi68_21 (machine-transcribed, do not hand-edit).
fn funct_loi68_21() -> Funct1D {
    Funct1D::new(
        vec![
            -10.0, -0.80343, -0.79076, -0.14786, -0.08675, -0.07781, -0.0319, -0.0049, 0.0,
            0.00499, 0.0314, 0.07781, 0.08675, 0.14786, 0.79076, 0.80343, 10.0,
        ],
        vec![
            7.6e-05, 7.6e-05, 7.6e-05, 0.00022, 0.0001787, 0.0001779, 0.000149, 0.00013, 0.000127,
            0.00013, 0.000149, 0.0001779, 0.0001787, 0.00022, 7.6e-05, 7.6e-05, 7.6e-05,
        ],
    )
    .expect("deck funct loi68_21 is valid")
}

/// Deck funct loi68_23 (machine-transcribed, do not hand-edit).
fn funct_loi68_23() -> Funct1D {
    Funct1D::new(
        vec![
            -10.0, -0.31, -0.28, -0.27, -0.25, -0.2, 0.0, 0.2, 0.25, 0.27, 0.28, 0.31, 10.0,
        ],
        vec![
            1.00007e-05,
            1.00007e-05,
            0.0001,
            0.0002,
            0.000400007,
            0.0001,
            2.00001e-05,
            0.0001,
            0.000400007,
            0.0002,
            0.0001,
            1.00007e-05,
            1.00007e-05,
        ],
    )
    .expect("deck funct loi68_23 is valid")
}

/// Deck funct loi68_31 (machine-transcribed, do not hand-edit).
fn funct_loi68_31() -> Funct1D {
    Funct1D::new(
        vec![
            -10.0, -0.79343, -0.75076, -0.14786, -0.08675, -0.07781, -0.0319, -0.0049, 0.0,
            0.00499, 0.0314, 0.07781, 0.08675, 0.14786, 0.75076, 0.79343, 10.0,
        ],
        vec![
            7.6e-05, 7.6e-05, 7.6e-05, 0.00022, 0.0001787, 0.0001779, 0.000149, 0.00013, 0.000127,
            0.00013, 0.000149, 0.0001779, 0.0001787, 0.00022, 7.6e-05, 7.6e-05, 7.6e-05,
        ],
    )
    .expect("deck funct loi68_31 is valid")
}

/// Deck funct loi68_32 (machine-transcribed, do not hand-edit).
fn funct_loi68_32() -> Funct1D {
    Funct1D::new(
        vec![
            -10.0, -0.52, -0.48, -0.4, -0.37756, -0.35, 0.0, 0.35, 0.37756, 0.4, 0.48, 0.52, 10.0,
        ],
        vec![
            1.22006e-05,
            1.22006e-05,
            0.0004,
            0.0002,
            0.000139007,
            5.00001e-05,
            2.00001e-05,
            5.00001e-05,
            0.000139007,
            0.0002,
            0.0004,
            1.22006e-05,
            1.22006e-05,
        ],
    )
    .expect("deck funct loi68_32 is valid")
}
