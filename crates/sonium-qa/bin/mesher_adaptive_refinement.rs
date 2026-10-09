//! Release-facing adaptive-refinement and recovery contract.
//!
//! The gate intentionally exercises the checked production API rather than
//! reproducing refinement logic in QA. It proves that an estimator can refine
//! a deterministic local subset, that malformed estimator data is rejected
//! transactionally, and that the resulting mesh still satisfies the geometric
//! invariants required by FEM/BEM consumers.

use sonium_mesher::mesh::generators::unit_square_triangles;
use sonium_mesher::mesh::quality::{QualityThresholds, mesh_quality_report};
use sonium_mesher::mesh::refinement::adaptive_refine;
use sonium_mesher::mesh::validate::{check_manifold, check_orientation};
use sonium_qa::QaResult;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
struct Metrics {
    baseline_nodes: usize,
    baseline_elements: usize,
    refined_nodes: usize,
    refined_elements: usize,
    marked_elements: usize,
    new_nodes: usize,
    new_elements: usize,
    removed_elements: usize,
    min_mean_ratio: f64,
    min_aspect_quality: f64,
    invalid_elements: usize,
    non_manifold_faces: usize,
    inverted_elements: usize,
    deterministic_repeat: bool,
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed("mesher.adaptive-refinement.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("mesher-adaptive-refinement-v1")
                .metric("baseline_nodes", metrics.baseline_nodes as f64)
                .metric("baseline_elements", metrics.baseline_elements as f64)
                .metric("refined_nodes", metrics.refined_nodes as f64)
                .metric("refined_elements", metrics.refined_elements as f64)
                .metric("marked_elements", metrics.marked_elements as f64)
                .metric("new_nodes", metrics.new_nodes as f64)
                .metric("new_elements", metrics.new_elements as f64)
                .metric("removed_elements", metrics.removed_elements as f64)
                .metric("min_mean_ratio", metrics.min_mean_ratio)
                .metric("min_aspect_quality", metrics.min_aspect_quality)
                .metric("invalid_elements", metrics.invalid_elements as f64)
                .metric("non_manifold_faces", metrics.non_manifold_faces as f64)
                .metric("inverted_elements", metrics.inverted_elements as f64)
                .metric(
                    "deterministic_repeat",
                    if metrics.deterministic_repeat {
                        1.0
                    } else {
                        0.0
                    },
                )
                .minimum("marked_elements", 1.0)
                .minimum("new_nodes", 1.0)
                .minimum("new_elements", 1.0)
                .minimum("min_mean_ratio", f64::EPSILON)
                .maximum("invalid_elements", 0.0)
                .maximum("non_manifold_faces", 0.0)
                .maximum("inverted_elements", 0.0)
                .minimum("deterministic_repeat", 1.0)
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
            if let Ok(json) = QaResult::failed("mesher.adaptive-refinement.v1", &error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("mesher-adaptive-refinement-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("MESHER ADAPTIVE REFINEMENT FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<Metrics, String> {
    let mut mesh = unit_square_triangles(2);
    mesh.validate_connectivity()
        .map_err(|error| format!("baseline mesh is invalid: {error:?}"))?;
    let baseline_nodes = mesh.nodes.len();
    let baseline_elements = mesh.elements.len();
    let original = mesh.clone();

    // Select a deterministic local subset.  The values are deliberately
    // generated from the element index so this gate is independent of hash
    // iteration order and does not encode a particular child numbering.
    let errors: Vec<f64> = (0..baseline_elements)
        .map(|index| if index % 5 == 0 { 1.0 } else { 0.0 })
        .collect();
    let marked_elements = errors.iter().filter(|&&error| error > 0.5).count();
    let refinement = adaptive_refine(&mut mesh, &errors, 0.5)
        .map_err(|error| format!("checked adaptive refinement failed: {error}"))?;
    if refinement.removed_elements.len() != marked_elements {
        return Err(format!(
            "refinement removed {} elements but estimator marked {marked_elements}",
            refinement.removed_elements.len()
        ));
    }
    if mesh.nodes.len() <= baseline_nodes || mesh.elements.len() <= baseline_elements {
        return Err("adaptive refinement did not increase mesh resolution".into());
    }

    let quality = mesh_quality_report(&mesh, 8);
    quality
        .validate(QualityThresholds {
            min_mean_ratio: f64::EPSILON,
            min_aspect_quality: 0.0,
            min_dihedral: 0.0,
            max_dihedral: std::f64::consts::PI,
            reject_invalid_elements: true,
        })
        .map_err(|error| format!("refined mesh failed quality gate: {error}"))?;
    let inverted = check_orientation(&mesh);
    let manifold = check_manifold(&mesh);
    if !inverted.is_empty() || !manifold.is_manifold() {
        return Err(format!(
            "refined mesh topology invalid: {} inverted, {} non-manifold faces",
            inverted.len(),
            manifold.non_manifold_faces.len()
        ));
    }

    // A malformed estimator must fail before mutation.  This protects the
    // checked boundary from accidentally consuming a partial update.
    let before_bad = mesh.clone();
    let before_bad_bytes = serde_json::to_vec(&before_bad)
        .map_err(|error| format!("failed to serialize pre-rejection mesh: {error}"))?;
    let malformed = vec![0.0; mesh.elements.len() - 1];
    if adaptive_refine(&mut mesh, &malformed, 0.5).is_ok() {
        return Err("malformed estimator input was not rejected transactionally".into());
    }
    let unchanged_after_rejection = serde_json::to_vec(&mesh)
        .map(|bytes| bytes == before_bad_bytes)
        .unwrap_or(false);
    if !unchanged_after_rejection {
        return Err("malformed estimator input mutated the mesh".into());
    }
    let refined_bytes = serde_json::to_vec(&mesh)
        .map_err(|error| format!("failed to serialize refined mesh: {error}"))?;

    // Replaying the same estimator on the same baseline must produce identical
    // serialized mesh and result metadata, which catches nondeterministic
    // midpoint/child ordering in adaptation.
    let mut repeated = original;
    let repeated_result = adaptive_refine(&mut repeated, &errors, 0.5)
        .map_err(|error| format!("repeated adaptive refinement failed: {error}"))?;
    let deterministic_repeat = serde_json::to_vec(&repeated)
        .map(|bytes| bytes == refined_bytes)
        .unwrap_or(false)
        && repeated_result.new_nodes == refinement.new_nodes
        && repeated_result.new_elements == refinement.new_elements
        && repeated_result.removed_elements == refinement.removed_elements;
    if !deterministic_repeat {
        return Err("repeated adaptive refinement changed result ordering".into());
    }

    Ok(Metrics {
        baseline_nodes,
        baseline_elements,
        refined_nodes: mesh.nodes.len(),
        refined_elements: mesh.elements.len(),
        marked_elements,
        new_nodes: refinement.new_nodes.len(),
        new_elements: refinement.new_elements.len(),
        removed_elements: refinement.removed_elements.len(),
        min_mean_ratio: quality.min_mean_ratio,
        min_aspect_quality: quality.min_aspect_quality,
        invalid_elements: quality.invalid_elements,
        non_manifold_faces: manifold.non_manifold_faces.len(),
        inverted_elements: inverted.len(),
        deterministic_repeat,
    })
}
