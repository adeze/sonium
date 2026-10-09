use rand::Rng;
use rand::seq::SliceRandom;

pub(crate) fn init_latin_hypercube<R: Rng + ?Sized>(
    n: usize,
    npop: usize,
    lower: &[f64],
    upper: &[f64],
    is_free: &[bool],
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let mut samples = vec![vec![0.0; n]; npop];
    // For each dimension, create stratified samples and permute
    for j in 0..n {
        if !is_free[j] {
            // fixed variable
            for row in samples.iter_mut() {
                row[j] = 0.0;
            }
            continue;
        }
        let mut vals = Vec::with_capacity(npop);
        for k in 0..npop {
            let u: f64 = rng.random::<f64>();
            vals.push(((k as f64) + u) / (npop as f64));
        }
        vals.shuffle(rng);
        for i in 0..npop {
            samples[i][j] = vals[i];
        }
    }
    // Scale to [lower, upper]
    for row in samples.iter_mut() {
        for j in 0..n {
            if is_free[j] {
                row[j] = lower[j] + row[j] * (upper[j] - lower[j]);
            } else {
                row[j] = lower[j];
            }
        }
    }
    samples
}
