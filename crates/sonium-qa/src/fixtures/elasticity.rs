//! Elasticity fixtures for integration tests

pub fn cantilever_beam_mesh(
    length: f64,
    _height: f64,
    n_elements: usize,
) -> (Vec<[f64; 2]>, Vec<[usize; 2]>) {
    let n_nodes = n_elements + 1;
    let h = length / (n_elements as f64);

    let mut nodes = vec![[0.0; 2]; n_nodes];
    for i in 0..n_nodes {
        nodes[i][0] = (i as f64) * h;
        nodes[i][1] = 0.0;
    }

    let mut elements = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        elements.push([i, i + 1]);
    }

    (nodes, elements)
}

pub fn analytical_cantilever_tip_deflection(
    length: f64,
    height: f64,
    pressure: f64,
    youngs_modulus: f64,
) -> f64 {
    let inertia = height.powi(3) / 12.0;
    (pressure * length.powi(4)) / (8.0 * youngs_modulus * inertia)
}

pub fn analytical_beam_modes(length: f64, n_modes: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    (1..=n_modes)
        .map(|n| {
            let n_f = n as f64;
            let a = (2.0 * n_f - 1.0) * PI / (2.0 * length);
            a.powi(2)
        })
        .collect()
}
