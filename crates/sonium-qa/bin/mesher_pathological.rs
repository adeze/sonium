//! Deterministic pathological mesher corpus.
//!
//! This is deliberately a release-facing case rather than a unit-test-only
//! fixture.  Invalid point clouds must fail before the kernel with stable
//! diagnostics, while difficult constrained surfaces must either recover or
//! report the missing constraints as a typed failure.

use sonium_mesher::mesh::tetrahedralize::{
    SurfaceFacet, TetrahedralizationError, TetrahedralizeConfig, tetrahedralize_points_checked,
    tetrahedralize_surface_with_facets_result,
};
use sonium_mesher::mesh::types::Point;
use sonium_mesher::mesh::validate::{check_manifold, check_orientation};
use sonium_qa::QaResult;
use std::time::Instant;

#[derive(Debug)]
struct Metrics {
    invalid_cases: usize,
    deterministic_invalid_cases: usize,
    sliver_nodes: usize,
    sliver_elements: usize,
    sliver_min_quality: f64,
    sliver_count: usize,
    hard_surface_failures: usize,
    hard_surface_missing_edges: usize,
    hard_surface_missing_facets: usize,
    hard_surface_recovery_rounds: usize,
    fuzz_smoke_cases: usize,
    fuzz_smoke_rejections: usize,
    fuzz_smoke_panics: usize,
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed("mesher.pathological-corpus.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("mesher-pathological-v1")
                .metric("invalid_cases", metrics.invalid_cases as f64)
                .metric(
                    "deterministic_invalid_cases",
                    metrics.deterministic_invalid_cases as f64,
                )
                .metric("sliver_nodes", metrics.sliver_nodes as f64)
                .metric("sliver_elements", metrics.sliver_elements as f64)
                .metric("sliver_min_quality", metrics.sliver_min_quality)
                .metric("sliver_count", metrics.sliver_count as f64)
                .metric(
                    "hard_surface_failures",
                    metrics.hard_surface_failures as f64,
                )
                .metric(
                    "hard_surface_missing_edges",
                    metrics.hard_surface_missing_edges as f64,
                )
                .metric(
                    "hard_surface_missing_facets",
                    metrics.hard_surface_missing_facets as f64,
                )
                .metric(
                    "hard_surface_recovery_rounds",
                    metrics.hard_surface_recovery_rounds as f64,
                )
                .metric("fuzz_smoke_cases", metrics.fuzz_smoke_cases as f64)
                .metric(
                    "fuzz_smoke_rejections",
                    metrics.fuzz_smoke_rejections as f64,
                )
                .metric("fuzz_smoke_panics", metrics.fuzz_smoke_panics as f64)
                .minimum("invalid_cases", 5.0)
                .minimum("deterministic_invalid_cases", 5.0)
                .minimum("sliver_nodes", 4.0)
                .minimum("sliver_elements", 1.0)
                .minimum("fuzz_smoke_cases", 64.0)
                .maximum("fuzz_smoke_panics", 0.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            match result.to_json_line() {
                Ok(json) => println!("QA_RESULT: {json}"),
                Err(error) => {
                    eprintln!("failed to serialize QA_RESULT: {error}");
                    println!("FAILED");
                    std::process::exit(1);
                }
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed("mesher.pathological-corpus.v1", &error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("mesher-pathological-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("MESHER PATHOLOGICAL CORPUS FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<Metrics, String> {
    let config = TetrahedralizeConfig {
        max_smooth_iterations: 0,
        add_interior_points: false,
        lloyd_iterations: 0,
        preserve_surface: true,
        max_ratio: 1.5,
    };

    let (fuzz_smoke_cases, fuzz_smoke_rejections, fuzz_smoke_panics) =
        deterministic_fuzz_smoke(&config);
    if fuzz_smoke_panics != 0 {
        return Err(format!(
            "deterministic mesher fuzz smoke observed {fuzz_smoke_panics} panic(s)"
        ));
    }

    let invalid_inputs = vec![
        Vec::new(),
        vec![Point::new_3d(0.0, 0.0, 0.0); 3],
        vec![
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 0.0),
        ],
        vec![
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(1.0, 1.0, 0.0),
        ],
        vec![
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(f64::NAN, 0.0, 1.0),
        ],
    ];
    let mut deterministic_invalid_cases = 0;
    for points in &invalid_inputs {
        let first = tetrahedralize_points_checked(points, &config)
            .expect_err("pathological point cloud must be rejected");
        let second = tetrahedralize_points_checked(points, &config)
            .expect_err("pathological point cloud must be rejected repeatedly");
        if first == second {
            deterministic_invalid_cases += 1;
        }
    }

    // A near-coplanar cloud is valid input, but should remain observable as a
    // low-quality/sliver-heavy output rather than being silently discarded.
    let sliver_points = vec![
        Point::new_3d(0.0, 0.0, 0.0),
        Point::new_3d(1.0, 0.0, 0.0),
        Point::new_3d(0.0, 1.0, 0.0),
        Point::new_3d(0.25, 0.25, 1.0e-10),
        Point::new_3d(0.0, 0.0, 1.0),
    ];
    let sliver = tetrahedralize_points_checked(&sliver_points, &config)
        .map_err(|error| format!("near-coplanar cloud unexpectedly failed: {error}"))?;
    if sliver.mesh.elements.is_empty()
        || !sliver.quality_min.is_finite()
        || !check_orientation(&sliver.mesh).is_empty()
        || !check_manifold(&sliver.mesh).is_manifold()
    {
        return Err("near-coplanar cloud produced an invalid mesh".into());
    }

    let (surface_nodes, surface_facets) = hard_annular_surface();
    let hard_surface =
        tetrahedralize_surface_with_facets_result(&surface_nodes, &surface_facets, &config);
    let (hard_surface_failures, hard_surface_missing_edges, hard_surface_missing_facets, rounds) =
        match hard_surface {
            Ok(result) => {
                if !result.validity.is_valid() {
                    return Err("hard surface reported success with invalid mesh".into());
                }
                (0, 0, 0, result.constraint_recovery.map_or(0, |d| d.rounds))
            }
            Err(TetrahedralizationError::ConstraintRecoveryFailed { diagnostics }) => (
                1,
                diagnostics.missing_edges.len(),
                diagnostics.missing_facets.len(),
                diagnostics.rounds,
            ),
            Err(error) => {
                return Err(format!(
                    "hard surface returned an unexpected typed error: {error}"
                ));
            }
        };

    Ok(Metrics {
        invalid_cases: invalid_inputs.len(),
        deterministic_invalid_cases,
        sliver_nodes: sliver.mesh.nodes.len(),
        sliver_elements: sliver.mesh.elements.len(),
        sliver_min_quality: sliver.quality_min,
        sliver_count: sliver.num_slivers,
        hard_surface_failures,
        hard_surface_missing_edges,
        hard_surface_missing_facets,
        hard_surface_recovery_rounds: rounds,
        fuzz_smoke_cases,
        fuzz_smoke_rejections,
        fuzz_smoke_panics,
    })
}

fn deterministic_fuzz_smoke(config: &TetrahedralizeConfig) -> (usize, usize, usize) {
    let mut rejections = 0;
    let mut panics = 0;
    for seed in 0_u64..64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(17);
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };
        let mut points = (0..5)
            .map(|_| Point::new_3d(next(), next(), next()))
            .collect::<Vec<_>>();
        if seed % 11 == 0 {
            points[4] = points[0];
        }
        if seed % 17 == 0 {
            points[3] = Point::new_3d(f64::NAN, next(), next());
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tetrahedralize_points_checked(&points, config)
        }));
        match outcome {
            Ok(Ok(_)) => {}
            Ok(Err(_)) => rejections += 1,
            Err(_) => panics += 1,
        }
    }
    (64, rejections, panics)
}

fn hard_annular_surface() -> (Vec<Point>, Vec<SurfaceFacet>) {
    let n = 4;
    let mut nodes = Vec::new();
    for z in [0.0, 0.3] {
        for k in 0..n {
            let angle = k as f64 * std::f64::consts::FRAC_PI_2;
            nodes.push(Point::new_3d(angle.cos(), angle.sin(), z));
        }
        for k in 0..n {
            let angle = (k as f64 + 0.5) * std::f64::consts::FRAC_PI_2;
            nodes.push(Point::new_3d(0.5 * angle.cos(), 0.5 * angle.sin(), z));
        }
    }
    let bo = |k: usize| k % n;
    let bi = |k: usize| n + k % n;
    let to = |k: usize| 2 * n + k % n;
    let ti = |k: usize| 3 * n + k % n;

    let mut facets = Vec::new();
    for k in 0..n {
        facets.push(SurfaceFacet {
            nodes: [to(k), to(k + 1), ti(k + 1)],
            marker: Some(1),
        });
        facets.push(SurfaceFacet {
            nodes: [to(k), ti(k + 1), ti(k)],
            marker: Some(1),
        });
        facets.push(SurfaceFacet {
            nodes: [bo(k), bi(k + 1), bo(k + 1)],
            marker: Some(2),
        });
        facets.push(SurfaceFacet {
            nodes: [bo(k), bi(k), bi(k + 1)],
            marker: Some(2),
        });
        facets.push(SurfaceFacet {
            nodes: [bo(k), bo(k + 1), to(k + 1)],
            marker: Some(3),
        });
        facets.push(SurfaceFacet {
            nodes: [bo(k), to(k + 1), to(k)],
            marker: Some(3),
        });
        facets.push(SurfaceFacet {
            nodes: [bi(k), ti(k + 1), bi(k + 1)],
            marker: Some(4),
        });
        facets.push(SurfaceFacet {
            nodes: [bi(k), ti(k), ti(k + 1)],
            marker: Some(4),
        });
    }
    (nodes, facets)
}
