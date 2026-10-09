//! MLFMM matvec vs dense parity audit, per `bem-step1.md` §1.4.
//!
//! For each `kD ∈ {1, 10, 50}` this binary builds the *same* Burton-Miller
//! BEM system two ways — `Tbem` (dense, ground truth) and `Mlfmm` (multi-level
//! fast multipole) — and compares matvec on a deterministic complex input.
//! The relative L² difference is the gap between what the production
//! solver actually computes and what the dense reference would.
//!
//! Output is a 3-row table. The exit decision is downstream of this audit:
//! - Worst-case rel err ≤ 1e-3 across all three `kD`s → finish wideband M2L
//!   (a small precision tightening, doc Path A).
//! - Worst-case rel err in (1e-3, 1e-1] → translation order is too coarse;
//!   tune `ClusterLevel::expansion_terms` at construction or revisit the
//!   diagonal-form M2L (doc Path A still, more invasive).
//! - Worst-case rel err > 1e-1 → MLFMM is too far gone to fix in scope;
//!   recommend the kifmm-rs FFI escape hatch (doc Path B).
//!
//! `kD = 50` is gated behind `MLFMM_AUDIT_HEAVY=1` because it builds a
//! ~5120² dense reference (≈420 MB, ≈10–60 s to assemble + matvec).

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{build_cluster_tree, build_mlfmm_system_stabilized};
use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
use std::f64::consts::PI;
use std::time::Instant;

const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;

fn freq_for_ka(ka: f64, radius: f64) -> f64 {
    ka * SPEED_OF_SOUND / (2.0 * PI * radius)
}

fn deterministic_complex_vector(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let x = (i + 1) as f64;
            Complex64::new((0.37 * x).sin(), (0.19 * x).cos())
        })
        .collect()
}

fn relative_l2(reference: &[Complex64], candidate: &[Complex64]) -> f64 {
    let diff_norm: f64 = reference
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let ref_norm: f64 = reference.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    diff_norm / ref_norm.max(1e-30)
}

struct AuditCase {
    label: &'static str,
    ka: f64,
    subdivisions: usize,
    elements_per_leaf: usize,
}

struct AuditResult {
    label: &'static str,
    ka: f64,
    n_dofs: usize,
    n_levels: usize,
    elements_per_wavelength: f64,
    dense_assembly_s: f64,
    mlfmm_assembly_s: f64,
    dense_matvec_s: f64,
    mlfmm_matvec_s: f64,
    rel_err_full: f64,
    /// Far-field contribution magnitude relative to dense ‖A x‖. Tells us
    /// how much of the dense matvec actually comes from far interactions
    /// — if this is small, near-field-only is "good enough" by accident.
    far_share: f64,
    /// Near-field-only matvec error relative to dense restricted to the
    /// same near-cluster pairs (built by zeroing the far MLFMM components).
    /// If this is large the near-field assembler is broken; if it's tiny
    /// the bug is purely in the M2L / upward / downward / evaluate stack.
    rel_err_near_only: f64,
    finite: bool,
}

fn run_case(case: &AuditCase) -> AuditResult {
    let radius = 1.0;
    let frequency = freq_for_ka(case.ka, radius);
    let k = 2.0 * PI * frequency / SPEED_OF_SOUND;
    let physics = PhysicsParams::new(frequency, SPEED_OF_SOUND, DENSITY, false);

    let mesh = generate_icosphere_mesh(radius, case.subdivisions);
    let mut elements = mesh.elements.clone();
    for (i, e) in elements.iter_mut().enumerate() {
        e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        e.dof_addresses = vec![i];
    }
    let n = elements.len();

    let wavelength = 2.0 * PI / k;
    let h = (4.0 * PI * radius * radius / n as f64).sqrt();

    // Same stabilisation convention for both assemblers — sign = -1.0,
    // beta_scale = 1.0 — matches `test_slfmm_matvec_vs_tbem` so the matvec
    // outputs are directly comparable.
    let t0 = Instant::now();
    let tbem = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, -1.0, 1.0);
    let dense_assembly_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let cluster_tree = build_cluster_tree(&elements, case.elements_per_leaf, &physics);
    let mlfmm =
        build_mlfmm_system_stabilized(&elements, &mesh.nodes, cluster_tree, &physics, -1.0, 1.0);
    let mlfmm_assembly_s = t1.elapsed().as_secs_f64();

    let x = deterministic_complex_vector(n);

    let t2 = Instant::now();
    let y_dense = tbem.matrix.matvec(&x);
    let dense_matvec_s = t2.elapsed().as_secs_f64();

    let t3 = Instant::now();
    let y_mlfmm = mlfmm.matvec(&x);
    let mlfmm_matvec_s = t3.elapsed().as_secs_f64();

    let y_near = mlfmm.near_field_matvec(&x);
    let y_far: Vec<Complex64> = y_mlfmm
        .iter()
        .zip(y_near.iter())
        .map(|(a, b)| a - b)
        .collect();

    let dense_norm: f64 = y_dense.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    let far_norm: f64 = y_far.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    let far_share = far_norm / dense_norm.max(1e-30);

    // Near-only reference: dense matvec minus what *should* be the far-field
    // share. We don't have a clean "dense restricted to near pairs" here
    // without recomputing, so the closest comparable quantity is
    // ‖y_dense - y_near‖ / ‖y_dense‖, which equals the relative error
    // *attributable to the missing far-field plus any near-field bug*.
    // This is enough to decide whether the near-field assembler alone
    // matches the dense reference (when the far-field contribution is
    // small) — not a perfect isolation, but the quickest signal we can
    // get without re-instrumenting the assembler.
    let rel_err_near_only = relative_l2(&y_dense, &y_near);

    let finite = y_mlfmm.iter().all(|z| z.re.is_finite() && z.im.is_finite());
    let rel_err_full = if finite {
        relative_l2(&y_dense, &y_mlfmm)
    } else {
        f64::NAN
    };

    AuditResult {
        label: case.label,
        ka: case.ka,
        n_dofs: n,
        n_levels: mlfmm.num_levels,
        elements_per_wavelength: wavelength / h,
        dense_assembly_s,
        mlfmm_assembly_s,
        dense_matvec_s,
        mlfmm_matvec_s,
        rel_err_full,
        far_share,
        rel_err_near_only,
        finite,
    }
}

fn print_header() {
    println!(
        "{:>10}  {:>5}  {:>6}  {:>6}  {:>6}  {:>9}  {:>9}  {:>9}  {:>9}  {:>11}  {:>11}  {:>11}",
        "case",
        "ka",
        "DOFs",
        "levels",
        "elem/λ",
        "dense_A(s)",
        "mlfmm_A(s)",
        "dense_M(s)",
        "mlfmm_M(s)",
        "rel err",
        "far share",
        "near-only"
    );
    println!("  {}", "-".repeat(132));
}

fn print_row(r: &AuditResult) {
    println!(
        "{:>10}  {:>5.1}  {:>6}  {:>6}  {:>6.1}  {:>9.2}  {:>9.2}  {:>9.4}  {:>9.4}  {:>11}  {:>11.3e}  {:>11.3e}",
        r.label,
        r.ka,
        r.n_dofs,
        r.n_levels,
        r.elements_per_wavelength,
        r.dense_assembly_s,
        r.mlfmm_assembly_s,
        r.dense_matvec_s,
        r.mlfmm_matvec_s,
        if r.finite {
            format!("{:.3e}", r.rel_err_full)
        } else {
            "NON-FINITE".to_string()
        },
        r.far_share,
        r.rel_err_near_only,
    );
}

fn classify(worst: f64) -> &'static str {
    if !worst.is_finite() {
        "FAIL: MLFMM produced NaN/Inf — assembly or matvec is broken"
    } else if worst <= 1e-3 {
        "PASS: MLFMM matches dense to ≤ 1e-3 — finish wideband M2L only (Path A)"
    } else if worst <= 1e-1 {
        "DEGRADED: 1e-3 < err ≤ 1e-1 — translation order/diagonal-form M2L need work (Path A, invasive)"
    } else {
        "FAIL: err > 1e-1 — MLFMM beyond easy fix; recommend kifmm-rs FFI (Path B)"
    }
}

fn diagnose(results: &[AuditResult]) -> Vec<&'static str> {
    let mut findings = Vec::new();
    for r in results {
        if !r.finite {
            findings
                .push("non-finite output — likely numerical overflow in M2L translation operator");
            continue;
        }
        // Near-field bug indicator: far contribution is small (cluster tree
        // mostly near, no significant M2L work happening) yet near-only is
        // still large. Means near-field block coverage is incomplete.
        if r.far_share < 1e-2 && r.rel_err_near_only > 0.5 {
            findings.push(
                "near-field block coverage incomplete — small problem has near-only ≫ 0 with negligible far share"
            );
        }
        // Far-field bug indicator: far contribution magnitude grossly
        // exceeds the dense reference. Diagonal-form M2L producing garbage
        // (wrong sphere quadrature, missing 1/(4π), bad translation phase).
        if r.far_share > 10.0 {
            findings.push(
                "far-field translation diverges — far_share ≫ 1 means M2L outputs are orders of magnitude off"
            );
        }
    }
    findings.sort_unstable();
    findings.dedup();
    findings
}

fn main() {
    println!("=== MLFMM matvec vs dense parity audit ===");
    println!("Per bem-step1.md §1.4. Both assemblers use sign=-1.0, beta_scale=1.0.");
    println!();
    print_header();

    let mut worst_err: f64 = 0.0;
    let mut any_nonfinite = false;
    let mut results = Vec::new();

    let always: &[AuditCase] = &[
        AuditCase {
            label: "kD=1",
            ka: 1.0,
            subdivisions: 2, // ~320 elements — dense is 320² ≈ 1.6 MB
            elements_per_leaf: 16,
        },
        AuditCase {
            label: "kD=10",
            ka: 10.0,
            subdivisions: 3, // ~1280 elements — dense is 1280² ≈ 26 MB
            elements_per_leaf: 16,
        },
    ];

    for case in always {
        let r = run_case(case);
        print_row(&r);
        if !r.finite {
            any_nonfinite = true;
        } else {
            worst_err = worst_err.max(r.rel_err_full);
        }
        results.push(r);
    }

    let heavy_enabled = std::env::var("MLFMM_AUDIT_HEAVY").as_deref() == Ok("1");
    if heavy_enabled {
        let heavy = AuditCase {
            label: "kD=50",
            ka: 50.0,
            subdivisions: 4, // ~5120 elements — dense is 5120² ≈ 420 MB
            elements_per_leaf: 32,
        };
        let r = run_case(&heavy);
        print_row(&r);
        if !r.finite {
            any_nonfinite = true;
        } else {
            worst_err = worst_err.max(r.rel_err_full);
        }
        results.push(r);
    } else {
        println!("  (kD=50 skipped — set MLFMM_AUDIT_HEAVY=1 to run; ~5120² dense reference)");
    }

    let worst_observable = if any_nonfinite { f64::NAN } else { worst_err };
    println!();
    println!(
        "Worst observed relative L² error: {}",
        if worst_observable.is_finite() {
            format!("{:.3e}", worst_observable)
        } else {
            "NON-FINITE".to_string()
        }
    );
    let findings = diagnose(&results);
    if !findings.is_empty() {
        println!("Diagnosis:");
        for f in &findings {
            println!("  - {f}");
        }
    }
    println!("Decision: {}", classify(worst_observable));

    // Exit code: 0 only when MLFMM is finite and within Path-A territory.
    // The "DEGRADED" / "FAIL" branches return 1 so a CI consumer can gate
    // on this binary directly.
    let pass = worst_observable.is_finite() && worst_observable <= 1e-3;
    if pass {
        println!("\nSUCCESS");
    } else {
        println!("\nFAILED");
        std::process::exit(1);
    }
}
