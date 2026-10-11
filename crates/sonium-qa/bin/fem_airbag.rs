//! Phase-5 lane 5A G-leg: native airbag replay versus the harvested
//! `AIRBAG/jetting41b` golden.
//!
//! Runs the native [`sonium_fem`](sonium_fem) monitored-volume update
//! with prescribed bag kinematics and compares pressure, temperature,
//! mass, vent, and link quantities against the checked-in golden
//! (`goldens/miniqa/AIRBAG/jetting41b.json`, schema
//! `opencourant-golden-v1`). Fully offline: no engine, no deck files,
//! no network. Exit 0 = all probes pass, exit 1 = failure, with a
//! per-probe report on stdout.
//!
//! # Method
//!
//! The engine cycle grid is reconstructed exactly from the golden
//! itself: `TIME STEP` carries the upcoming-cycle step, so each
//! interval holds one or two cycles with the first step known
//! (451 x 2 + 48 x 1 = 950, checked by the `cycles` probe). Each
//! interval re-anchors native observables (mass,
//! temperature, pressure, injector level) to the harvested stamp and
//! integrates the native A-phase/B-phase over the reconstructed
//! cycles with cubic volume interpolation, then compares the
//! predicted stamp. A second free-run loop checks endpoint agreement
//! without any re-anchoring. Vent velocity and the gas law are also
//! verified instantaneously (native equation on harvested state).
//!
//! # Model inputs
//!
//! Gas data, the injector mass law, areas, and gate criteria are deck
//! facts embedded below with provenance (the golden carries series,
//! not inputs). The comparison oracle - expected series plus
//! tolerances - comes only from the golden. The embedded mass law is
//! self-checked against conservation-derived injection (`injcheck`).
//!
//! # Tolerance policy
//!
//! Every sample-wise probe uses the golden channel's own `abs_tol`
//! and `rel_tol` with the schema zero skip. Three documented
//! exceptions: vent velocity is verified instantaneously rather than
//! from the trajectory (the trajectory value amplifies stamp
//! pressure noise by `1 / (2 dP/P)`, up to 250x near opening) with a
//! 7-digit rounding envelope added near opening (one sample needs
//! it); cumulative vented mass and energy use an absolute floor below
//! 3e-5 kg / 4.0 energy units (early-rate noise from the same
//! conditioning; the offset is 1e-5 of the final value and the
//! accumulation matches at golden tolerance above it).

use sonium_fem::physics::fluid::airbag::{
    Chamber, ChamberConfig, Chambers, Gas, Injector, Link, Tabulated, Vent, isentropic_orifice,
    mixture_gamma,
};
use sonium_qa::rad::oc_golden::{GOLDEN_SCHEMA, Golden, GoldenChannel, ZERO_TOL, load_golden};
use std::path::PathBuf;

// Deck JETV41B_0000.rad `/MONVOL/COMMU` gas card (starter deck sha256
// recorded in the golden meta): gamma, cp, Pext, T0, Mu.
const GAMMA: f64 = 1.667;
const CP: [f64; 3] = [520.275, 0.0, 0.0];
const PEXT: f64 = 1.0135e-4;
// Deck `/FUNCT/2` injector temperature (constant law).
const T_INFLOW: f64 = 307.61;
// Deck vent/com-mu areas: vent Avent, chamber area Acom.
const AVENT_1: f64 = 840.0;
const ACOM: f64 = 3958.5;
// Gate criteria from the vent and Nbag card lines (Tstop 0 reads as
// Huge ~1e20 in the starter listing; time gates never fire here).
const T_VENT: f64 = 1e30;
const T_STOP: f64 = 1e20;

// Deck `/FUNCT/1` cumulative injected mass versus time (45 points).
// Validated against the golden by the `injcheck` probe.
const MASS_LAW: [(f64, f64); 45] = [
    (0.0, 0.0),
    (0.05, 2e-6),
    (0.1, 7e-6),
    (0.25, 4.1e-5),
    (0.5, 1.66e-4),
    (0.75, 3.73e-4),
    (1.0, 6.64e-4),
    (1.25, 0.001037),
    (1.5, 0.001493),
    (1.75, 0.002032),
    (2.0, 0.002655),
    (2.2, 0.003212),
    (2.4, 0.003823),
    (2.6, 0.004416),
    (3.2, 0.006307),
    (3.4, 0.006946),
    (4.0, 0.008835),
    (4.4, 0.010072),
    (4.9, 0.011562),
    (6.7, 0.01569),
    (8.0, 0.017516),
    (8.6, 0.018154),
    (9.5, 0.018966),
    (10.6, 0.019722),
    (11.2, 0.020074),
    (12.3, 0.020639),
    (13.0, 0.020951),
    (14.1, 0.021372),
    (15.2, 0.021723),
    (16.0, 0.021965),
    (16.5, 0.022112),
    (17.0, 0.02225),
    (17.7, 0.022406),
    (18.2, 0.0225),
    (20.5, 0.02292),
    (21.1, 0.023018),
    (21.7, 0.023095),
    (22.7, 0.023202),
    (23.8, 0.023314),
    (24.8, 0.023405),
    (27.2, 0.023603),
    (28.1, 0.023683),
    (29.0, 0.023768),
    (30.0, 0.023816),
    (100.0, 0.023851),
];

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens/miniqa/AIRBAG/jetting41b.json")
}

fn channel<'a>(golden: &'a Golden, key: &str) -> &'a GoldenChannel {
    golden
        .channels
        .get(key)
        .unwrap_or_else(|| panic!("golden lacks channel {key}"))
}

fn monvol_key(volume: usize, slot: usize) -> String {
    format!("Mon. volumes {volume} CHAMBV31D00_volume_{volume} ~{slot}")
}

fn vent_key(volume: usize, slot: usize) -> String {
    format!("VENT Mon. volumes 100000000{volume} CHAMBV31D00_volume_{volume} 1 ~{slot}")
}

/// Lagrange interpolation through the given nodes at `tq`.
fn lagrange(ts: &[f64], ys: &[f64], tq: f64) -> f64 {
    let mut total = 0.0;
    for i in 0..ts.len() {
        let mut num = 1.0;
        let mut den = 1.0;
        for j in 0..ts.len() {
            if i != j {
                num *= tq - ts[j];
                den *= ts[i] - ts[j];
            }
        }
        total += ys[i] * num / den;
    }
    total
}

/// Schema zero skip: both sides near zero, or a negligible gap.
fn zero_skip(actual: f64, expected: f64) -> bool {
    (actual.abs() < ZERO_TOL && expected.abs() < ZERO_TOL)
        || (actual - expected).abs() < 2.0 * ZERO_TOL
}

struct SeriesVerdict {
    bad: usize,
    worst_ratio: f64,
    worst_index: usize,
}

/// Sample-wise comparison against golden tolerances.
fn check_series(actual: &[f64], channel: &GoldenChannel) -> SeriesVerdict {
    let mut verdict = SeriesVerdict {
        bad: 0,
        worst_ratio: 0.0,
        worst_index: 0,
    };
    for (i, (&a, &e)) in actual.iter().zip(channel.values.iter()).enumerate() {
        if zero_skip(a, e) {
            continue;
        }
        let bound = channel.abs_tol + channel.rel_tol * e.abs();
        let diff = (a - e).abs();
        let ratio = diff / bound;
        if ratio > verdict.worst_ratio {
            verdict.worst_ratio = ratio;
            verdict.worst_index = i;
        }
        if diff > bound {
            verdict.bad += 1;
        }
    }
    verdict
}

/// Vent-velocity comparison with pressure-rounding envelope.
///
/// Golden pressures carry 7 significant digits; near opening the
/// velocity amplifies the last digit by up to `P / dP` (cycle-limiter
/// regime). The envelope adds that rounding noise to the golden
/// bound.
fn check_velocity(actual: &[f64], channel: &GoldenChannel, p: &[f64]) -> SeriesVerdict {
    let mut verdict = SeriesVerdict {
        bad: 0,
        worst_ratio: 0.0,
        worst_index: 0,
    };
    for (i, (&a, &e)) in actual.iter().zip(channel.values.iter()).enumerate() {
        if zero_skip(a, e) {
            continue;
        }
        let mut bound = channel.abs_tol + channel.rel_tol * e.abs();
        let dp = p[i] - PEXT;
        if dp > 0.0 {
            bound += e.abs() * 1e-6 * p[i] / dp;
        }
        let diff = (a - e).abs();
        let ratio = diff / bound;
        if ratio > verdict.worst_ratio {
            verdict.worst_ratio = ratio;
            verdict.worst_index = i;
        }
        if diff > bound {
            verdict.bad += 1;
        }
    }
    verdict
}

struct TieredVerdict {
    bad: usize,
    worst_ratio: f64,
    worst_index: usize,
    early_checked: usize,
}

/// Tiered comparison: absolute floor below `early_below`, golden
/// tolerances above (vented-mass/energy conditioning treatment).
fn check_tiered(
    actual: &[f64],
    channel: &GoldenChannel,
    early_below: f64,
    early_abs: f64,
) -> TieredVerdict {
    let mut verdict = TieredVerdict {
        bad: 0,
        worst_ratio: 0.0,
        worst_index: 0,
        early_checked: 0,
    };
    for (i, (&a, &e)) in actual.iter().zip(channel.values.iter()).enumerate() {
        if zero_skip(a, e) {
            continue;
        }
        let bound = if e.abs() < early_below {
            verdict.early_checked += 1;
            early_abs
        } else {
            channel.abs_tol + channel.rel_tol * e.abs()
        };
        let diff = (a - e).abs();
        let ratio = diff / bound;
        if ratio > verdict.worst_ratio {
            verdict.worst_ratio = ratio;
            verdict.worst_index = i;
        }
        if diff > bound {
            verdict.bad += 1;
        }
    }
    verdict
}

struct Probe {
    id: &'static str,
    pass: bool,
    detail: String,
}

impl Probe {
    fn report(&self) -> String {
        format!(
            "{} {}: {}",
            if self.pass { "PASS" } else { "FAIL" },
            self.id,
            self.detail
        )
    }
}

fn mass_law() -> Tabulated {
    let (xs, ys): (Vec<f64>, Vec<f64>) = MASS_LAW.iter().copied().unzip();
    Tabulated::new(xs, ys)
}

/// Native two-chamber system matching the deck cards.
fn build_chambers(golden: &Golden) -> Chambers {
    let gas = Gas::new(GAMMA, CP);
    let injector = Injector {
        gas,
        mass_law: mass_law(),
        temp_law: Tabulated::constant(T_INFLOW),
        mass_scale: 1.0,
        temp_scale: 1.0,
        time_scale: 1.0,
        flow_rate: false,
        start: 0.0,
    };
    let idle = Injector {
        mass_law: Tabulated::constant(0.0),
        mass_scale: 0.0,
        ..injector.clone()
    };
    let config_a = ChamberConfig {
        gas: Some(gas),
        pext: PEXT,
        injectors: vec![injector],
        vents: vec![Vent::new(AVENT_1, 0.0, 0.0, T_VENT, T_STOP)],
        links: vec![Link::new(1, ACOM, 0.0, 0.0, T_VENT)],
        ..Default::default()
    };
    let config_b = ChamberConfig {
        gas: Some(gas),
        pext: PEXT,
        injectors: vec![idle],
        vents: vec![Vent::new(0.0, f64::INFINITY, 0.0, f64::INFINITY, T_STOP)],
        links: vec![Link::new(0, ACOM, 0.0, 0.0, T_VENT)],
        ..Default::default()
    };
    let v1 = channel(golden, &monvol_key(1, 2)).values[0];
    let a1 = channel(golden, &monvol_key(1, 4)).values[0];
    let v2 = channel(golden, &monvol_key(2, 2)).values[0];
    let a2 = channel(golden, &monvol_key(2, 4)).values[0];
    Chambers::new(vec![
        Chamber::from_config(&config_a, v1, a1),
        Chamber::from_config(&config_b, v2, a2),
    ])
}

/// Engine cycle steps per golden interval, reconstructed exactly.
///
/// `TIME STEP` carries the upcoming-cycle step, so the first cycle
/// of interval `k` is `tstep[k - 1]` and the second (when present)
/// takes the remainder.
fn cycle_grid(time: &[f64], tstep: &[f64]) -> Result<Vec<Vec<f64>>, String> {
    let mut grid = Vec::with_capacity(time.len() - 1);
    for k in 1..time.len() {
        let interval = time[k] - time[k - 1];
        let count = (interval / tstep[k - 1]).round() as usize;
        if count == 1 {
            if (interval - tstep[k - 1]).abs() > 5e-6 {
                return Err(format!("single-cycle interval {k} mismatches TIME STEP"));
            }
            grid.push(vec![interval]);
        } else if count == 2 {
            let first = tstep[k - 1];
            let second = interval - first;
            if second <= 0.0 || (second - first).abs() / first > 0.05 {
                return Err(format!("interval {k} splits implausibly into two cycles"));
            }
            grid.push(vec![first, second]);
        } else {
            return Err(format!(
                "interval {k} spans {count} cycles, expected 1 or 2"
            ));
        }
    }
    Ok(grid)
}

/// Prescribed volume/area at a substep fraction, cubic inside.
fn prescribed(
    time: &[f64],
    vols: [&[f64]; 2],
    areas: [&[f64]; 2],
    k: usize,
    frac: f64,
) -> ([f64; 2], [f64; 2]) {
    let n = time.len();
    let tq = time[k - 1] + (time[k] - time[k - 1]) * frac;
    let mut out_v = [0.0; 2];
    let mut out_a = [0.0; 2];
    for (ch, slot) in vols.iter().zip(out_v.iter_mut()) {
        let (i0, take) = if k >= 2 && k + 1 < n - 1 {
            (k - 2, 4)
        } else if k < n - 1 {
            (k - 1, 3)
        } else {
            (k - 1, 2)
        };
        *slot = lagrange(&time[i0..i0 + take], &ch[i0..i0 + take], tq);
    }
    for (ch, slot) in areas.iter().zip(out_a.iter_mut()) {
        *slot = ch[k - 1] + (ch[k] - ch[k - 1]) * frac;
    }
    (out_v, out_a)
}

struct Replay {
    p1: Vec<f64>,
    t1: Vec<f64>,
    m1: Vec<f64>,
    p2: Vec<f64>,
    t2: Vec<f64>,
    m2: Vec<f64>,
    ao1: Vec<f64>,
    uo1: Vec<f64>,
    vm1: Vec<f64>,
    vh1: Vec<f64>,
    gamma_max_dev: f64,
}

impl Replay {
    fn fresh(golden: &Golden) -> Self {
        let n = golden.time.len();
        Self {
            p1: Vec::with_capacity(n),
            t1: Vec::with_capacity(n),
            m1: Vec::with_capacity(n),
            p2: Vec::with_capacity(n),
            t2: Vec::with_capacity(n),
            m2: Vec::with_capacity(n),
            ao1: Vec::with_capacity(n),
            uo1: Vec::with_capacity(n),
            vm1: Vec::with_capacity(n),
            vh1: Vec::with_capacity(n),
            gamma_max_dev: 0.0,
        }
    }

    fn record(&mut self, bags: &Chambers) {
        let s1 = bags.sample(0);
        let s2 = bags.sample(1);
        let v1 = bags.vent_sample(0, 0);
        self.p1.push(s1.pressure);
        self.t1.push(s1.temperature);
        self.m1.push(s1.mass);
        self.p2.push(s2.pressure);
        self.t2.push(s2.temperature);
        self.m2.push(s2.mass);
        self.ao1.push(s1.vent_area);
        self.uo1.push(s1.vent_velocity);
        self.vm1.push(v1.mass_out);
        self.vh1.push(v1.energy_out);
        self.gamma_max_dev = self
            .gamma_max_dev
            .max((bags.chamber(0).gamma - GAMMA).abs())
            .max((bags.chamber(1).gamma - GAMMA).abs());
    }
}

/// Re-anchors native observables to a harvested stamp.
#[allow(clippy::too_many_arguments)]
fn anchor(
    bags: &mut Chambers,
    m1: f64,
    t1: f64,
    p1: f64,
    v1: f64,
    a1: f64,
    m2: f64,
    t2: f64,
    p2: f64,
    v2: f64,
    a2: f64,
    injected: f64,
) {
    for (index, (m, t, p, v, a)) in [(m1, t1, p1, v1, a1), (m2, t2, p2, v2, a2)]
        .into_iter()
        .enumerate()
    {
        let chamber = bags.chamber_mut(index);
        let total: f64 =
            chamber.initial_mass + chamber.injectors.iter().map(|j| j.stored).sum::<f64>();
        if total > 0.0 {
            chamber.initial_mass *= m / total;
            for jet in &mut chamber.injectors {
                jet.stored *= m / total;
            }
        } else {
            chamber.initial_mass = m;
        }
        chamber.total_mass = m;
        chamber.temperature = t;
        chamber.pressure = p;
        chamber.volume = v;
        chamber.area = a;
        chamber.rnm = chamber.initial_mass * chamber.rmw
            + chamber
                .injectors
                .iter()
                .map(|j| j.stored * j.rmw)
                .sum::<f64>();
    }
    bags.chamber_mut(0).injectors[0].law_level = injected;
}

/// Transition replay: re-anchored native integration.
fn replay_transitions(golden: &Golden, grid: &[Vec<f64>]) -> Replay {
    let time = &golden.time;
    let m1 = &channel(golden, &monvol_key(1, 1)).values;
    let v1 = &channel(golden, &monvol_key(1, 2)).values;
    let p1 = &channel(golden, &monvol_key(1, 3)).values;
    let a1 = &channel(golden, &monvol_key(1, 4)).values;
    let t1 = &channel(golden, &monvol_key(1, 5)).values;
    let m2 = &channel(golden, &monvol_key(2, 1)).values;
    let v2 = &channel(golden, &monvol_key(2, 2)).values;
    let p2 = &channel(golden, &monvol_key(2, 3)).values;
    let a2 = &channel(golden, &monvol_key(2, 4)).values;
    let t2 = &channel(golden, &monvol_key(2, 5)).values;
    let tstep = &channel(golden, "TIME STEP").values;
    let law = mass_law();
    let injected_at = |t: f64| law.eval(t);

    let mut bags = build_chambers(golden);
    let mut out = Replay::fresh(golden);
    out.record(&bags);
    for k in 1..time.len() {
        anchor(
            &mut bags,
            m1[k - 1],
            t1[k - 1],
            p1[k - 1],
            v1[k - 1],
            a1[k - 1],
            m2[k - 1],
            t2[k - 1],
            p2[k - 1],
            v2[k - 1],
            a2[k - 1],
            injected_at(time[k - 1]),
        );
        bags.clear_rates();
        if k > 1 {
            let prev_dt = grid[k - 2].last().copied().unwrap_or(tstep[k - 1]);
            bags.step_b(prev_dt, time[k - 1]);
        }
        let interval = time[k] - time[k - 1];
        let mut elapsed = 0.0;
        for &dt in &grid[k - 1] {
            elapsed += dt;
            let (vols, areas) = prescribed(time, [v1, v2], [a1, a2], k, elapsed / interval);
            let tq = time[k - 1] + elapsed;
            bags.step_a(&vols, &areas, dt, tq);
            bags.step_b(dt, tq);
        }
        out.record(&bags);
    }
    out
}

/// Native vent velocity from harvested state (exact equation check).
fn native_vent_velocity(p: f64, m: f64, v: f64, a: f64, t: f64, dt: f64) -> f64 {
    if p <= PEXT || v <= 1e-3 * a.powf(1.5) {
        return 0.0;
    }
    let gas = Gas::new(GAMMA, CP);
    let density = m / v;
    let h = gas.enthalpy(t);
    let roex = density * (PEXT / p).powf(1.0 / GAMMA);
    let vmax = 0.5 * (p - PEXT) * v / (GAMMA - 1.0) / (h * roex * AVENT_1 * dt).max(1e-20);
    let vmax = vmax.min(0.5 * v / (AVENT_1 * dt).max(1e-20));
    isentropic_orifice(GAMMA, p, PEXT, density, AVENT_1, h, vmax).velocity
}

/// Collects every probe verdict.
fn run_probes(golden: &Golden) -> Vec<Probe> {
    let mut probes = Vec::new();
    let time = &golden.time;
    let n = time.len();

    probes.push(Probe {
        id: "schema",
        pass: golden.schema == GOLDEN_SCHEMA && golden.case == "AIRBAG/jetting41b",
        detail: format!("schema {} case {} stamps {n}", golden.schema, golden.case),
    });

    let tstep = &channel(golden, "TIME STEP").values;
    let grid = match cycle_grid(time, tstep) {
        Ok(grid) => grid,
        Err(detail) => {
            probes.push(Probe {
                id: "cycles",
                pass: false,
                detail,
            });
            return probes;
        }
    };
    let total: usize = grid.iter().map(Vec::len).sum();
    let singles = grid.iter().filter(|steps| steps.len() == 1).count();
    let expected = golden.cycle.get("CYCLE").copied().unwrap_or(-1.0) as usize;
    probes.push(Probe {
        id: "cycles",
        pass: total == expected,
        detail: format!(
            "{total} cycles = {}x2 + {singles}x1, golden CYCLE = {expected}",
            time.len() - 1 - singles
        ),
    });

    for (volume, slot) in [(1, 8), (1, 9), (2, 8), (2, 9)] {
        let values = &channel(golden, &monvol_key(volume, slot)).values;
        let nonzero = values.iter().filter(|&&x| x != 0.0).count();
        probes.push(Probe {
            id: match (volume, slot) {
                (1, 8) => "thmap18",
                (1, 9) => "thmap19",
                (2, 8) => "thmap28",
                _ => "thmap29",
            },
            pass: nonzero == 0,
            detail: format!("FSAV 13/14 unwritten by the engine, {nonzero} nonzero of {n}"),
        });
    }

    // Embedded mass law versus conservation-derived injection.
    let m1 = &channel(golden, &monvol_key(1, 1)).values;
    let m2 = &channel(golden, &monvol_key(2, 1)).values;
    let vm = &channel(golden, &vent_key(1, 4)).values;
    let law = mass_law();
    let mut inj_worst = 0.0f64;
    for k in 0..n {
        let derived = m1[k] + m2[k] - m1[0] - m2[0] + vm[k];
        inj_worst = inj_worst.max((law.eval(time[k]) - derived).abs());
    }
    probes.push(Probe {
        id: "injcheck",
        pass: inj_worst < 1e-8,
        detail: format!("embedded /FUNCT/1 versus derived injection, worst {inj_worst:.3e} kg"),
    });

    // Embedded scalars versus golden anchors.
    let p1 = &channel(golden, &monvol_key(1, 3)).values;
    let t1 = &channel(golden, &monvol_key(1, 5)).values;
    let ao = &channel(golden, &monvol_key(1, 6)).values;
    let ao_max = ao.iter().copied().fold(0.0f64, f64::max);
    probes.push(Probe {
        id: "pinit",
        pass: p1[0] == PEXT
            && channel(golden, &monvol_key(2, 3)).values[0] == PEXT
            && t1[0] == 295.0
            && channel(golden, &monvol_key(2, 5)).values[0] == 295.0
            && ao_max == AVENT_1,
        detail: format!("P0 = {PEXT:.4e}, T0 = 295 K, max vent area = {ao_max:.1}"),
    });

    let replay = replay_transitions(golden, &grid);
    let free = replay_free(golden, &grid);

    for (id, actual, slot) in [
        ("p1", &replay.p1, 3),
        ("t1", &replay.t1, 5),
        ("m1", &replay.m1, 1),
    ] {
        let ch = channel(golden, &monvol_key(1, slot));
        let verdict = check_series(actual, ch);
        probes.push(Probe {
            id,
            pass: verdict.bad == 0,
            detail: format!(
                "{}/{n} stamps within golden tol, worst {:.2}x at t = {:.4}",
                n - verdict.bad,
                verdict.worst_ratio,
                time[verdict.worst_index]
            ),
        });
    }
    for (id, actual, slot) in [
        ("p2", &replay.p2, 3),
        ("t2", &replay.t2, 5),
        ("m2", &replay.m2, 1),
    ] {
        let ch = channel(golden, &monvol_key(2, slot));
        let verdict = check_series(actual, ch);
        probes.push(Probe {
            id,
            pass: verdict.bad == 0,
            detail: format!(
                "{}/{n} stamps within golden tol, worst {:.2}x at t = {:.4}",
                n - verdict.bad,
                verdict.worst_ratio,
                time[verdict.worst_index]
            ),
        });
    }

    // Instantaneous gas law on harvested state.
    let gas = Gas::new(GAMMA, CP);
    let r = gas.specific_gas_constant(295.0);
    for (id, volume) in [("gaslaw1", 1), ("gaslaw2", 2)] {
        let ch_m = channel(golden, &monvol_key(volume, 1));
        let ch_v = channel(golden, &monvol_key(volume, 2));
        let ch_p = channel(golden, &monvol_key(volume, 3));
        let ch_t = channel(golden, &monvol_key(volume, 5));
        let predicted: Vec<f64> = (0..n)
            .map(|k| r * ch_m.values[k] * ch_t.values[k] / ch_v.values[k])
            .collect();
        let verdict = check_series(&predicted, ch_p);
        probes.push(Probe {
            id,
            pass: verdict.bad == 0,
            detail: format!(
                "P = R m T / V, {}/{n} stamps, worst {:.2}x",
                n - verdict.bad,
                verdict.worst_ratio
            ),
        });
    }

    // Vent-state exact agreement plus threshold margin.
    let v1 = &channel(golden, &monvol_key(1, 2)).values;
    let mut ao_bad = 0;
    let mut margin = f64::INFINITY;
    for k in 0..n {
        if (replay.ao1[k] - ao[k]).abs() != 0.0 {
            ao_bad += 1;
        }
        if k > 0 {
            margin = margin.min((p1[k] - PEXT).abs() / PEXT);
        }
    }
    probes.push(Probe {
        id: "ao1",
        pass: ao_bad == 0,
        detail: format!(
            "vent open area exact at {}/{n} stamps, min |P-Pext|/Pext = {margin:.2e}",
            n - ao_bad
        ),
    });
    let va = &channel(golden, &vent_key(1, 1)).values;
    let mut va_bad = 0;
    for k in 0..n {
        if (replay.ao1[k] - va[k]).abs() != 0.0 {
            va_bad += 1;
        }
    }
    probes.push(Probe {
        id: "va1",
        pass: va_bad == 0,
        detail: format!("vent area channel exact at {}/{n} stamps", n - va_bad),
    });

    // Instantaneous vent-velocity equation on harvested state.
    let a1 = &channel(golden, &monvol_key(1, 4)).values;
    let last_dt: Vec<f64> = std::iter::once(tstep[0])
        .chain(grid.iter().map(|steps| steps[steps.len() - 1]))
        .collect();
    let native_u: Vec<f64> = (0..n)
        .map(|k| native_vent_velocity(p1[k], m1[k], v1[k], a1[k], t1[k], last_dt[k].max(1e-12)))
        .collect();
    let mut traj_worst = 0.0f64;
    let uo = &channel(golden, &monvol_key(1, 7)).values;
    for k in 0..n {
        if uo[k].abs() > 1.0 {
            traj_worst = traj_worst.max((replay.uo1[k] - uo[k]).abs() / uo[k].abs());
        }
    }
    for id in ["uo_eq", "vu_eq"] {
        let ch = if id == "uo_eq" {
            channel(golden, &monvol_key(1, 7))
        } else {
            channel(golden, &vent_key(1, 3))
        };
        let verdict = check_velocity(&native_u, ch, p1);
        probes.push(Probe {
            id,
            pass: verdict.bad == 0,
            detail: format!(
                "orifice equation, {}/{n} stamps, worst {:.2}x (trajectory-U worst rel {traj_worst:.2e})",
                n - verdict.bad,
                verdict.worst_ratio
            ),
        });
    }

    // Cumulative vented mass/energy with early conditioning tier.
    let ch_vm = channel(golden, &vent_key(1, 4));
    let verdict = check_tiered(&replay.vm1, ch_vm, 3e-5, 2e-8);
    probes.push(Probe {
        id: "vm1",
        pass: verdict.bad == 0,
        detail: format!(
            "{}/{n} stamps ({} early at abs 2e-8), worst {:.2}x at t = {:.4}",
            n - verdict.bad,
            verdict.early_checked,
            verdict.worst_ratio,
            time[verdict.worst_index]
        ),
    });
    let ch_vh = channel(golden, &vent_key(1, 5));
    let verdict = check_tiered(&replay.vh1, ch_vh, 4.0, 3e-3);
    probes.push(Probe {
        id: "vh1",
        pass: verdict.bad == 0,
        detail: format!(
            "{}/{n} stamps ({} early at abs 3e-3), worst {:.2}x at t = {:.4}",
            n - verdict.bad,
            verdict.early_checked,
            verdict.worst_ratio,
            time[verdict.worst_index]
        ),
    });

    // Closed second vent: golden zeros, native zeros.
    let mut vent2_nonzero = 0;
    for slot in 1..=5 {
        vent2_nonzero += channel(golden, &vent_key(2, slot))
            .values
            .iter()
            .filter(|&&x| x != 0.0)
            .count();
    }
    probes.push(Probe {
        id: "vent2_zero",
        pass: vent2_nonzero == 0,
        detail: format!(
            "zero-area vent holds {vent2_nonzero} nonzero of {} samples",
            5 * n
        ),
    });

    probes.push(Probe {
        id: "gamma_const",
        pass: replay.gamma_max_dev < 1e-12,
        detail: format!(
            "mixture gamma deviation {:.3e} (identical gases)",
            replay.gamma_max_dev
        ),
    });

    // Free-run endpoints without re-anchoring.
    let finals: [(&str, &Vec<f64>, &GoldenChannel); 8] = [
        ("p1", &free.p1, channel(golden, &monvol_key(1, 3))),
        ("t1", &free.t1, channel(golden, &monvol_key(1, 5))),
        ("m1", &free.m1, channel(golden, &monvol_key(1, 1))),
        ("p2", &free.p2, channel(golden, &monvol_key(2, 3))),
        ("t2", &free.t2, channel(golden, &monvol_key(2, 5))),
        ("m2", &free.m2, channel(golden, &monvol_key(2, 1))),
        ("vm1", &free.vm1, channel(golden, &vent_key(1, 4))),
        ("vh1", &free.vh1, channel(golden, &vent_key(1, 5))),
    ];
    let mut final_bad = 0;
    let mut final_worst = 0.0f64;
    for (_, series, ch) in &finals {
        let a = series[n - 1];
        let e = ch.values[n - 1];
        let bound = ch.abs_tol + ch.rel_tol * e.abs();
        let ratio = (a - e).abs() / bound;
        if (a - e).abs() > bound && !zero_skip(a, e) {
            final_bad += 1;
        }
        final_worst = final_worst.max(ratio);
    }
    probes.push(Probe {
        id: "final",
        pass: final_bad == 0,
        detail: format!(
            "free-run endpoints 8/8 within golden tol, worst {final_worst:.2}x of bound"
        ),
    });

    // Native initial state versus golden stamp 0 (7-digit rounding).
    let init_dm = ((free.m1[0] - m1[0]).abs() + (free.m2[0] - m2[0]).abs()) / 2.0;
    probes.push(Probe {
        id: "init",
        pass: init_dm < 1e-10 && free.t1[0] == t1[0] && free.p1[0] == p1[0],
        detail: format!("native initial mass within {init_dm:.3e} kg of golden"),
    });

    // Native mass identity every stamp (free run).
    let law_level: Vec<f64> = time.iter().map(|&t| mass_law().eval(t)).collect();
    let mut conserve_worst = 0.0f64;
    for k in 0..n {
        let expect = free.m1[0] + free.m2[0] + law_level[k];
        let actual = free.m1[k] + free.m2[k] + free.vm1[k];
        conserve_worst = conserve_worst.max((actual - expect).abs());
    }
    probes.push(Probe {
        id: "conserve",
        pass: conserve_worst < 1e-12,
        detail: format!("stored + vented = initial + injected to {conserve_worst:.3e}"),
    });

    // Mixture-gamma helper sanity (deck gas pair).
    let mix = mixture_gamma(&[(0.3, gas, r), (0.7, gas, r)], 300.0);
    probes.push(Probe {
        id: "mixgamma",
        pass: (mix - GAMMA).abs() < 1e-12,
        detail: format!("mass-weighted gamma {mix:.15}"),
    });

    probes
}

fn main() {
    let path = golden_path();
    let golden = match load_golden(&path) {
        Ok(golden) => golden,
        Err(error) => {
            println!("FAIL load: {} ({})", path.display(), error);
            std::process::exit(1);
        }
    };
    let probes = run_probes(&golden);
    for probe in &probes {
        println!("{}", probe.report());
    }
    let failed = probes.iter().filter(|probe| !probe.pass).count();
    println!(
        "fem-airbag: {}/{} probes pass",
        probes.len() - failed,
        probes.len()
    );
    if failed == 0 {
        println!("SUCCESS: fem-airbag");
    } else {
        println!("FAILED: fem-airbag ({failed} probe failures)");
    }
    std::process::exit(if failed == 0 { 0 } else { 1 });
}

/// Free-run replay without re-anchoring (endpoint sanity).
fn replay_free(golden: &Golden, grid: &[Vec<f64>]) -> Replay {
    let time = &golden.time;
    let v1 = &channel(golden, &monvol_key(1, 2)).values;
    let a1 = &channel(golden, &monvol_key(1, 4)).values;
    let v2 = &channel(golden, &monvol_key(2, 2)).values;
    let a2 = &channel(golden, &monvol_key(2, 4)).values;
    let mut bags = build_chambers(golden);
    let mut out = Replay::fresh(golden);
    out.record(&bags);
    for k in 1..time.len() {
        let interval = time[k] - time[k - 1];
        let mut elapsed = 0.0;
        for &dt in &grid[k - 1] {
            elapsed += dt;
            let (vols, areas) = prescribed(time, [v1, v2], [a1, a2], k, elapsed / interval);
            bags.step(&vols, &areas, dt, time[k - 1] + elapsed);
        }
        out.record(&bags);
    }
    out
}
