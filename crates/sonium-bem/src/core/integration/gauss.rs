//! Gauss-Legendre quadrature rules

// Allow excessive precision for high-precision mathematical constants
#![allow(clippy::excessive_precision)]

use std::f64::consts::PI;

/// Maximum supported Gauss-Legendre order
pub const N_GAUORDER: usize = 80;

/// Gauss-Legendre abscissas and weights for various orders
///
/// Returns (points, weights) where points are in [-1, 1]
pub fn gauss_legendre(order: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(
        (1..=N_GAUORDER).contains(&order),
        "Order must be 1..{}",
        N_GAUORDER
    );

    match order {
        1 | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 32 => {
            let (x, w) = gauss_legendre_raw(order);
            (x.to_vec(), w.to_vec())
        }
        _ => gauss_legendre_generated(order),
    }
}

/// Generate a Gauss-Legendre rule when an exact checked-in table is not
/// available.
///
/// Callers use the order as part of convergence/admission decisions. Rounding
/// an unsupported order to a neighbouring table would make distinct requested
/// orders identical and could therefore produce a false zero error estimate.
/// Newton refinement of the Legendre roots is deterministic for the bounded
/// `1..=N_GAUORDER` range and preserves the advertised order exactly.
fn gauss_legendre_generated(order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut points = vec![0.0; order];
    let mut weights = vec![0.0; order];
    let half = order.div_ceil(2);
    let order_f64 = order as f64;

    for index in 0..half {
        let mut root = (PI * (index as f64 + 0.75) / (order_f64 + 0.5)).cos();

        for _ in 0..32 {
            let mut p_prev = 1.0;
            let mut p = root;
            for degree in 2..=order {
                let p_next = ((2 * degree - 1) as f64 * root * p - (degree - 1) as f64 * p_prev)
                    / degree as f64;
                p_prev = p;
                p = p_next;
            }
            let derivative = order_f64 * (root * p - p_prev) / (root * root - 1.0);
            let next = root - p / derivative;
            if (next - root).abs() <= 4.0 * f64::EPSILON * next.abs().max(1.0) {
                root = next;
                break;
            }
            root = next;
        }

        // Recompute the derivative at the converged root for the weight.
        let mut p_prev = 1.0;
        let mut p = root;
        for degree in 2..=order {
            let p_next =
                ((2 * degree - 1) as f64 * root * p - (degree - 1) as f64 * p_prev) / degree as f64;
            p_prev = p;
            p = p_next;
        }
        let derivative = order_f64 * (root * p - p_prev) / (root * root - 1.0);
        let weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        points[index] = -root;
        points[order - 1 - index] = root;
        weights[index] = weight;
        weights[order - 1 - index] = weight;
    }

    (points, weights)
}

/// Raw Gauss-Legendre points and weights
fn gauss_legendre_raw(n: usize) -> (&'static [f64], &'static [f64]) {
    match n {
        1 => (&GL1_X, &GL1_W),
        2 => (&GL2_X, &GL2_W),
        3 => (&GL3_X, &GL3_W),
        4 => (&GL4_X, &GL4_W),
        5 => (&GL5_X, &GL5_W),
        6 => (&GL6_X, &GL6_W),
        7 => (&GL7_X, &GL7_W),
        8 => (&GL8_X, &GL8_W),
        10 => (&GL10_X, &GL10_W),
        12 => (&GL12_X, &GL12_W),
        16 => (&GL16_X, &GL16_W),
        20 => (&GL20_X, &GL20_W),
        32 => (&GL32_X, &GL32_W),
        _ => unreachable!("only checked-in Gauss-Legendre table orders are dispatched here"),
    }
}

/// Triangle quadrature points (xi, eta, weight)
///
/// Returns vector of (xi, eta, weight) tuples for the reference triangle
/// with vertices at (0,0), (1,0), (0,1). Weights are scaled so they sum
/// to 0.5 (the area of the reference triangle).
pub fn triangle_quadrature(order: usize) -> Vec<(f64, f64, f64)> {
    // The raw tables have weights that sum to 1.0 (unit simplex convention).
    // We scale by 0.5 to get weights that sum to the reference triangle area.
    const AREA_SCALE: f64 = 0.5;
    match order {
        1 => GAUCORWEI_TR1
            .iter()
            .map(|&[x, y, w]| (x, y, w * AREA_SCALE))
            .collect(),
        2 => GAUCORWEI_TR4
            .iter()
            .map(|&[x, y, w]| (x, y, w * AREA_SCALE))
            .collect(),
        3 => GAUCORWEI_TR7
            .iter()
            .map(|&[x, y, w]| (x, y, w * AREA_SCALE))
            .collect(),
        4 => GAUCORWEI_TR13
            .iter()
            .map(|&[x, y, w]| (x, y, w * AREA_SCALE))
            .collect(),
        _ => {
            assert!(
                (1..=N_GAUORDER).contains(&order),
                "triangle quadrature order must be 1..={N_GAUORDER}"
            );
            // Duffy map [0,1]² → {(xi,eta): xi>=0, eta>=0, xi+eta<=1}:
            // xi=u, eta=(1-u)v, |J|=1-u. This gives a genuine scalable
            // high-order rule instead of silently reusing the fixed 13-point
            // table for every request above order four.
            let (points, weights) = gauss_legendre(order);
            let mut result = Vec::with_capacity(points.len() * points.len());
            for (&u_raw, &u_weight_raw) in points.iter().zip(&weights) {
                let u = 0.5 * (u_raw + 1.0);
                let u_weight = 0.5 * u_weight_raw;
                for (&v_raw, &v_weight_raw) in points.iter().zip(&weights) {
                    let v = 0.5 * (v_raw + 1.0);
                    let v_weight = 0.5 * v_weight_raw;
                    result.push((u, (1.0 - u) * v, u_weight * v_weight * (1.0 - u)));
                }
            }
            result
        }
    }
}

/// Quad quadrature points (tensor product of 1D Gauss-Legendre)
///
/// Returns vector of (xi, eta, weight) tuples for [-1,1]²
pub fn quad_quadrature(order: usize) -> Vec<(f64, f64, f64)> {
    let (points, weights) = gauss_legendre(order);
    let mut result = Vec::with_capacity(order * order);

    for (i, &xi) in points.iter().enumerate() {
        for (j, &eta) in points.iter().enumerate() {
            result.push((xi, eta, weights[i] * weights[j]));
        }
    }

    result
}

/// Unit sphere quadrature for FMM
///
/// Uses Gauss-Legendre in theta direction and uniform in phi
pub fn unit_sphere_quadrature(n_theta: usize, n_phi: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
    use std::f64::consts::PI;

    let (cos_theta, weights_theta) = gauss_legendre(n_theta);
    let delta_phi = 2.0 * PI / n_phi as f64;

    let n_points = n_theta * n_phi;
    let mut coords = Vec::with_capacity(n_points);
    let mut weights = Vec::with_capacity(n_points);

    for (i, &ct) in cos_theta.iter().enumerate() {
        let st = (1.0 - ct * ct).sqrt();
        for j in 0..n_phi {
            let phi = delta_phi * j as f64;
            coords.push([st * phi.cos(), st * phi.sin(), ct]);
            weights.push(weights_theta[i] * delta_phi / (4.0 * PI));
        }
    }

    (coords, weights)
}

// Gauss-Legendre abscissas and weights
// Ported from NC_IntegrationConstants.h
static GL1_X: [f64; 1] = [0.0];
static GL1_W: [f64; 1] = [2.0];

static GL2_X: [f64; 2] = [-0.5773502691896257, 0.5773502691896257];
static GL2_W: [f64; 2] = [1.0, 1.0];

static GL3_X: [f64; 3] = [-0.7745966692414834, 0.0, 0.7745966692414834];
static GL3_W: [f64; 3] = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556];

static GL4_X: [f64; 4] = [
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526,
];
static GL4_W: [f64; 4] = [
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538,
];

static GL5_X: [f64; 5] = [
    -0.9061798459386640,
    -0.5384693101056831,
    0.0,
    0.5384693101056831,
    0.9061798459386640,
];
static GL5_W: [f64; 5] = [
    0.2369268850561891,
    0.4786286704993665,
    0.5688888888888889,
    0.4786286704993665,
    0.2369268850561891,
];

static GL6_X: [f64; 6] = [
    -0.9324695142031521,
    -0.6612093864662645,
    -0.2386191860831969,
    0.2386191860831969,
    0.6612093864662645,
    0.9324695142031521,
];
static GL6_W: [f64; 6] = [
    0.1713244923791704,
    0.3607615730481386,
    0.4679139345726910,
    0.4679139345726910,
    0.3607615730481386,
    0.1713244923791704,
];

static GL7_X: [f64; 7] = [
    -0.9491079123427585,
    -0.7415311855993945,
    -0.4058451513773972,
    0.0,
    0.4058451513773972,
    0.7415311855993945,
    0.9491079123427585,
];
static GL7_W: [f64; 7] = [
    0.1294849661688697,
    0.2797053914892766,
    0.3818300505051189,
    0.4179591836734694,
    0.3818300505051189,
    0.2797053914892766,
    0.1294849661688697,
];

static GL8_X: [f64; 8] = [
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
];
static GL8_W: [f64; 8] = [
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
];

static GL10_X: [f64; 10] = [
    -0.9739065285171717,
    -0.8650633666889845,
    -0.6794095682990244,
    -0.4333953941292472,
    -0.1488743389816312,
    0.1488743389816312,
    0.4333953941292472,
    0.6794095682990244,
    0.8650633666889845,
    0.9739065285171717,
];
static GL10_W: [f64; 10] = [
    0.0666713443086881,
    0.1494513491505806,
    0.2190863625159820,
    0.2692667193099963,
    0.2955242247147529,
    0.2955242247147529,
    0.2692667193099963,
    0.2190863625159820,
    0.1494513491505806,
    0.0666713443086881,
];

static GL12_X: [f64; 12] = [
    -0.9815606342467192,
    -0.9041172563704749,
    -0.7699026741943047,
    -0.5873179542866175,
    -0.3678314989981802,
    -0.1252334085114689,
    0.1252334085114689,
    0.3678314989981802,
    0.5873179542866175,
    0.7699026741943047,
    0.9041172563704749,
    0.9815606342467192,
];
static GL12_W: [f64; 12] = [
    0.0471753363865118,
    0.1069393259953184,
    0.1600783285433462,
    0.2031674267230659,
    0.2334925365383548,
    0.2491470458134028,
    0.2491470458134028,
    0.2334925365383548,
    0.2031674267230659,
    0.1600783285433462,
    0.1069393259953184,
    0.0471753363865118,
];

static GL16_X: [f64; 16] = [
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878318,
    -0.7554044083550030,
    -0.6178762444026438,
    -0.4580167776572274,
    -0.2816035507792589,
    -0.0950125098376374,
    0.0950125098376374,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.7554044083550030,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
];
static GL16_W: [f64; 16] = [
    0.0271524594117541,
    0.0622535239386479,
    0.0951585116824928,
    0.1246289712555339,
    0.1495959888165767,
    0.1691565193950025,
    0.1826034150449236,
    0.1894506104550685,
    0.1894506104550685,
    0.1826034150449236,
    0.1691565193950025,
    0.1495959888165767,
    0.1246289712555339,
    0.0951585116824928,
    0.0622535239386479,
    0.0271524594117541,
];

static GL20_X: [f64; 20] = [
    -0.9931285991850949,
    -0.9639719272779138,
    -0.9122344282513259,
    -0.8391169718222188,
    -0.7463319064601508,
    -0.6360536807265150,
    -0.5108670019508271,
    -0.3737060887154195,
    -0.2277858511416451,
    -0.0765265211334973,
    0.0765265211334973,
    0.2277858511416451,
    0.3737060887154195,
    0.5108670019508271,
    0.6360536807265150,
    0.7463319064601508,
    0.8391169718222188,
    0.9122344282513259,
    0.9639719272779138,
    0.9931285991850949,
];
static GL20_W: [f64; 20] = [
    0.0176140071391521,
    0.0406014298003869,
    0.0626720483341091,
    0.0832767415767048,
    0.1019301198172404,
    0.1181945319615184,
    0.1316886384491766,
    0.1420961093183820,
    0.1491729864726037,
    0.1527533871307258,
    0.1527533871307258,
    0.1491729864726037,
    0.1420961093183820,
    0.1316886384491766,
    0.1181945319615184,
    0.1019301198172404,
    0.0832767415767048,
    0.0626720483341091,
    0.0406014298003869,
    0.0176140071391521,
];

// Order-32 Gauss–Legendre — abscissas and weights on (−1, 1).
// Values from NIST DLMF Table 18.5.1 / Abramowitz–Stegun Table 25.4,
// to 15 significant digits. Sum of weights = 2 (verified by the
// `gl32_weights_sum_to_two` test). Used by `unit_sphere_quadrature`
// when the MLFMM expansion order outranges the GL20 cap (Lebedev 11+
// is the longer-term replacement — see `core::integration::lebedev`).
static GL32_X: [f64; 32] = [
    -0.997263861849481564,
    -0.985611511545268336,
    -0.964762255587506430,
    -0.934906075937739690,
    -0.896321155766052124,
    -0.849367613732569971,
    -0.794483795967942406,
    -0.732182118740289680,
    -0.663044266930215201,
    -0.587715757240762329,
    -0.506899908932229390,
    -0.421351276130635345,
    -0.331868602282127650,
    -0.239287362252137075,
    -0.144471961582796493,
    -0.048307665687738316,
    0.048307665687738316,
    0.144471961582796493,
    0.239287362252137075,
    0.331868602282127650,
    0.421351276130635345,
    0.506899908932229390,
    0.587715757240762329,
    0.663044266930215201,
    0.732182118740289680,
    0.794483795967942406,
    0.849367613732569971,
    0.896321155766052124,
    0.934906075937739690,
    0.964762255587506430,
    0.985611511545268336,
    0.997263861849481564,
];
static GL32_W: [f64; 32] = [
    0.007018610009470097,
    0.016274394730905671,
    0.025392065309262059,
    0.034273862913021434,
    0.042835898022226681,
    0.050998059262376176,
    0.058684093478535547,
    0.065822222776361847,
    0.072345794108848506,
    0.078193895787070306,
    0.083311924226946755,
    0.087652093004403811,
    0.091173878695763885,
    0.093844399080804566,
    0.095638720079274859,
    0.096540088514727801,
    0.096540088514727801,
    0.095638720079274859,
    0.093844399080804566,
    0.091173878695763885,
    0.087652093004403811,
    0.083311924226946755,
    0.078193895787070306,
    0.072345794108848506,
    0.065822222776361847,
    0.058684093478535547,
    0.050998059262376176,
    0.042835898022226681,
    0.034273862913021434,
    0.025392065309262059,
    0.016274394730905671,
    0.007018610009470097,
];

// Triangle quadrature rules (from NC_IntegrationConstants.h)
// Format: [xi, eta, weight]

static GAUCORWEI_TR1: [[f64; 3]; 1] = [[0.333333333333333, 0.333333333333333, 1.0]];

static GAUCORWEI_TR4: [[f64; 3]; 4] = [
    [0.333333333333333, 0.333333333333333, -0.5625],
    [0.6, 0.2, 0.520833333333333],
    [0.2, 0.6, 0.520833333333333],
    [0.2, 0.2, 0.520833333333333],
];

static GAUCORWEI_TR7: [[f64; 3]; 7] = [
    [0.333333333333333, 0.333333333333333, 0.225],
    [0.797426985353087, 0.101286507323456, 0.125939180544827],
    [0.101286507323456, 0.797426985353087, 0.125939180544827],
    [0.101286507323456, 0.101286507323456, 0.125939180544827],
    [0.470142064105115, 0.059715871789770, 0.132394152788506],
    [0.059715871789770, 0.470142064105115, 0.132394152788506],
    [0.470142064105115, 0.470142064105115, 0.132394152788506],
];

static GAUCORWEI_TR13: [[f64; 3]; 13] = [
    [0.333333333333333, 0.333333333333333, -0.149570044467682],
    [0.260345966079040, 0.260345966079040, 0.175615257433208],
    [0.260345966079040, 0.479308067841920, 0.175615257433208],
    [0.479308067841920, 0.260345966079040, 0.175615257433208],
    [0.065130102902216, 0.065130102902216, 0.053347235608838],
    [0.065130102902216, 0.869739794195568, 0.053347235608838],
    [0.869739794195568, 0.065130102902216, 0.053347235608838],
    [0.638444188569810, 0.048690315425316, 0.077113760890257],
    [0.048690315425316, 0.638444188569810, 0.077113760890257],
    [0.638444188569810, 0.312865496004874, 0.077113760890257],
    [0.312865496004874, 0.638444188569810, 0.077113760890257],
    [0.048690315425316, 0.312865496004874, 0.077113760890257],
    [0.312865496004874, 0.048690315425316, 0.077113760890257],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_legendre_2() {
        let (x, w) = gauss_legendre(2);
        assert_eq!(x.len(), 2);
        assert!((x[0] + 0.5773502691896257).abs() < 1e-10);
        assert!((w[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gauss_weights_sum() {
        // Sum of weights should be 2 (integral of 1 over [-1,1])
        for n in [2, 4, 6, 8, 10, 12, 16, 20, 32] {
            let (_, w) = gauss_legendre(n);
            let sum: f64 = w.iter().sum();
            assert!((sum - 2.0).abs() < 1e-10, "n={}: sum={}", n, sum);
        }
    }

    #[test]
    fn generated_rules_preserve_requested_order_and_normalization() {
        for order in [3, 5, 7, 33, 48, N_GAUORDER] {
            let (points, weights) = gauss_legendre(order);
            assert_eq!(points.len(), order, "order {order} must not be rounded");
            assert_eq!(weights.len(), order, "order {order} must not be rounded");
            assert!(points.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(points.iter().all(|point| point.is_finite()));
            assert!(
                weights
                    .iter()
                    .all(|weight| *weight > 0.0 && weight.is_finite())
            );
            let sum: f64 = weights.iter().sum();
            assert!((sum - 2.0).abs() < 2.0e-14, "order={order}: sum={sum:.17}");
        }
    }

    #[test]
    fn generated_order_33_integrates_degree_64_polynomial() {
        // GL32 is exact only through degree 63. This sentinel prevents a
        // future fallback to the nearest checked-in table from making an
        // adjacent-order convergence comparison vacuous.
        let (points, weights) = gauss_legendre(33);
        let approximate: f64 = points
            .iter()
            .zip(&weights)
            .map(|(&point, &weight)| weight * point.powi(64))
            .sum();
        let exact = 2.0 / 65.0;
        assert!((approximate - exact).abs() < 2.0e-14);
    }

    #[test]
    fn test_gl32_integrates_high_degree_polynomial_exactly() {
        // GL of order n integrates polynomials of degree ≤ 2n-1 exactly,
        // so GL32 must reproduce ∫_{-1}^{1} x^{2k} dx = 2/(2k+1) for k ≤ 31.
        // Probe at k = 10, 20, 30.
        let (x, w) = gauss_legendre(32);
        for &k in &[10_usize, 20, 30] {
            let exact = 2.0 / (2.0 * k as f64 + 1.0);
            let q: f64 = x
                .iter()
                .zip(&w)
                .map(|(&xi, &wi)| wi * xi.powi(2 * k as i32))
                .sum();
            assert!(
                (q - exact).abs() < 1e-12,
                "GL32 ∫ x^{} should be {exact:.15}, got {q:.15}",
                2 * k
            );
        }
    }

    #[test]
    fn test_triangle_quadrature() {
        let tri7 = triangle_quadrature(3);
        assert_eq!(tri7.len(), 7);

        // Weights should sum to 0.5 (area of reference triangle)
        let sum: f64 = tri7.iter().map(|&(_, _, w)| w).sum();
        assert!((sum - 0.5).abs() < 1e-10);
    }

    #[test]
    fn high_order_triangle_quadrature_does_not_silently_reuse_the_13_point_rule() {
        let rule = triangle_quadrature(12);
        assert_eq!(rule.len(), 12 * 12);
        assert!((rule.iter().map(|point| point.2).sum::<f64>() - 0.5).abs() < 1e-14);

        // Integral over the unit triangle of x^10 y^8 is
        // 10! 8! / (10 + 8 + 2)!.
        let computed: f64 = rule
            .iter()
            .map(|&(x, y, weight)| weight * x.powi(10) * y.powi(8))
            .sum();
        let exact = (1..=10).product::<u64>() as f64 * (1..=8).product::<u64>() as f64
            / (1..=20).product::<u64>() as f64;
        assert!((computed - exact).abs() < 1e-15);
    }

    #[test]
    fn test_quad_quadrature() {
        let quad4 = quad_quadrature(2);
        assert_eq!(quad4.len(), 4);

        // Weights should sum to 4 (area of [-1,1]²)
        let sum: f64 = quad4.iter().map(|&(_, _, w)| w).sum();
        assert!((sum - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_quadrature() {
        let (coords, weights) = unit_sphere_quadrature(4, 8);
        assert_eq!(coords.len(), 32);
        assert_eq!(weights.len(), 32);

        // All points should be on unit sphere
        for c in &coords {
            let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            assert!((r - 1.0).abs() < 1e-10);
        }

        // Weights should sum to 1 (normalized)
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01); // Approximate due to discretization
    }
}
