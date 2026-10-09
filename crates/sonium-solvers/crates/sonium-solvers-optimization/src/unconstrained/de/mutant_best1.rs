use rand::Rng;

use super::distinct_indices::distinct_indices_into;

pub(crate) fn mutant_best1_into<R: Rng + ?Sized>(
    out: &mut [f64],
    i: usize,
    pop: &[Vec<f64>],
    best_idx: usize,
    f: f64,
    rng: &mut R,
) {
    let mut idxs = [0usize; 2];
    distinct_indices_into(i, 2, pop.len(), rng, &mut idxs);
    let r0 = idxs[0];
    let r1 = idxs[1];

    let out_slice = out;
    let best = &pop[best_idx];
    let x0 = &pop[r0];
    let x1 = &pop[r1];
    for j in 0..out_slice.len() {
        out_slice[j] = best[j] + f * (x0[j] - x1[j]);
    }
}
