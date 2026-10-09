use rand::Rng;

use super::distinct_indices::distinct_indices_into;

pub(crate) fn mutant_rand2_into<R: Rng + ?Sized>(
    out: &mut [f64],
    i: usize,
    pop: &[Vec<f64>],
    f: f64,
    rng: &mut R,
) {
    let mut idxs = [0usize; 5];
    distinct_indices_into(i, 5, pop.len(), rng, &mut idxs);
    let r0 = idxs[0];
    let r1 = idxs[1];
    let r2 = idxs[2];
    let r3 = idxs[3];
    let r4 = idxs[4];

    let out_slice = out;
    let x0 = &pop[r0];
    let x1 = &pop[r1];
    let x2 = &pop[r2];
    let x3 = &pop[r3];
    let x4 = &pop[r4];
    for j in 0..out_slice.len() {
        out_slice[j] = x0[j] + f * (x1[j] + x2[j] - x3[j] - x4[j]);
    }
}
