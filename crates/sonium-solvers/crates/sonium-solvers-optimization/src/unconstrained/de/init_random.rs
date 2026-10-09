use rand::Rng;

pub(crate) fn init_random<R: Rng + ?Sized>(
    n: usize,
    npop: usize,
    lower: &[f64],
    upper: &[f64],
    is_free: &[bool],
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let mut pop = vec![vec![0.0; n]; npop];
    for row in pop.iter_mut() {
        for j in 0..n {
            if is_free[j] {
                let u: f64 = rng.random::<f64>();
                row[j] = lower[j] + u * (upper[j] - lower[j]);
            } else {
                row[j] = lower[j];
            }
        }
    }
    pop
}
