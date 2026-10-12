// Temporary probe (deleted before finishing).
use sonium_fem::physics::solid::plasticity::law_metals::{Law62, Law62State};
use sonium_fem::physics::solid::plasticity::update::StressUpdateModel;
#[test]
fn probe_law62_drift() {
    let law = Law62::rubber_prony();
    let dt = 1.0e-3;
    // shear: exact hencky3 = 0.2
    let mut state = Law62State::new(law.prony.len());
    for _ in 0..100 {
        let d = [0.0, 0.0, 0.0, 0.002, 0.0, 0.0];
        let u = law.update_with_rate(&state, &d, dt);
        law.commit(&mut state, &u);
    }
    println!("shear hencky3 = {:.17e} (exact 0.2), drift = {:.3e}", state.hencky[3], state.hencky[3] - 0.2);
    println!("shear sig3 = {:.17e} br3 = {:.17e}", state.stress[3], state.branches[0][3]);
    // vol: exact hencky0 = 0.01
    let mut state = Law62State::new(law.prony.len());
    for _ in 0..100 {
        let d = [0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0];
        let u = law.update_with_rate(&state, &d, dt);
        law.commit(&mut state, &u);
    }
    println!("vol hencky0 = {:.17e} (exact 0.01), drift = {:.3e}", state.hencky[0], state.hencky[0] - 0.01);
    println!("vol sig0 = {:.17e}", state.stress[0]);
    // uni with lateral solve: track sum of axial increments vs hencky0
    let mut state = Law62State::new(law.prony.len());
    let de = 0.002;
    let mut sum_ax = 0.0;
    for _ in 0..100 {
        let probe = |dl: f64| law.update_with_rate(&state, &[de, dl, dl, 0.0, 0.0, 0.0], dt).stress[1];
        let (mut lo, mut hi) = (-de, 0.0);
        let mut blo = probe(lo); let mut bhi = probe(hi);
        for _ in 0..20 {
            if blo < 0.0 && bhi > 0.0 { break; }
            let w = (hi - lo).abs(); lo -= w; hi += w;
            blo = probe(lo); bhi = probe(hi);
        }
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            if probe(mid) > 0.0 { hi = mid; } else { lo = mid; }
        }
        let dl = 0.5 * (lo + hi);
        sum_ax += de;
        let u = law.update_with_rate(&state, &[de, dl, dl, 0.0, 0.0, 0.0], dt);
        law.commit(&mut state, &u);
    }
    println!("uni hencky0 = {:.17e} sum_ax = {:.17e} drift = {:.3e}", state.hencky[0], sum_ax, state.hencky[0] - sum_ax);
    println!("uni hencky1 = {:.17e} sig0 = {:.17e}", state.hencky[1], state.stress[0]);
}
