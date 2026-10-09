//! Release-facing reversible mesh-adaptation contract.
//!
//! The gate uses the production history-backed refinement/coarsening APIs and
//! checks sibling admission, geometric conservation, boundary restoration,
//! deterministic replay, and transactional rejection of incomplete families.

use sonium_mesher::mesh::generators::{compact_unused_nodes_checked, unit_square_triangles};
use sonium_mesher::mesh::refinement::{
    coarsen_elements_checked, refine_elements_with_history_checked,
};
use sonium_mesher::mesh::validate::{check_manifold, check_orientation};
use sonium_qa::QaResult;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
struct Metrics {
    baseline_nodes: usize,
    baseline_elements: usize,
    refined_nodes: usize,
    refined_elements: usize,
    compacted_nodes: usize,
    removed_orphan_nodes: usize,
    restored_elements: usize,
    removed_elements: usize,
    boundary_count: usize,
    area_relative_error: f64,
    deterministic_repeat: bool,
}

fn mesh_area(mesh: &sonium_mesher::mesh::Mesh) -> f64 {
    (0..mesh.num_elements())
        .map(|index| mesh.element_measure(index))
        .sum()
}

fn mesh_signature(mesh: &sonium_mesher::mesh::Mesh) -> Result<Vec<u8>, String> {
    serde_json::to_vec(mesh).map_err(|error| format!("mesh serialization failed: {error}"))
}

fn run() -> Result<Metrics, String> {
    let mut mesh = unit_square_triangles(2);
    mesh.detect_boundaries();
    mesh.validate_connectivity()
        .map_err(|error| format!("baseline mesh is invalid: {error:?}"))?;
    let baseline_nodes = mesh.nodes.len();
    let baseline_elements = mesh.elements.len();
    let baseline_area = mesh_area(&mesh);
    let baseline_boundaries = mesh.boundaries.len();

    let (refinement, history) = refine_elements_with_history_checked(&mut mesh, &[0])
        .map_err(|error| format!("checked reversible refinement failed: {error}"))?;
    let refined_nodes = mesh.nodes.len();
    let refined_elements = mesh.elements.len();
    if refinement.new_elements.len() != 4 || refinement.removed_elements.len() != 1 {
        return Err("triangle refinement did not produce one complete four-child family".into());
    }
    if (mesh_area(&mesh) - baseline_area).abs() > 1.0e-12 {
        return Err("refinement changed total area".into());
    }

    let parent_id = history
        .parents
        .first()
        .ok_or_else(|| "reversible refinement returned no parent history".to_owned())?
        .element
        .id;
    let coarsening = coarsen_elements_checked(&mut mesh, &history, &[parent_id])
        .map_err(|error| format!("checked coarsening failed: {error}"))?;
    let area_relative_error = (mesh_area(&mesh) - baseline_area).abs() / baseline_area;
    if coarsening.restored_elements.len() != 1
        || coarsening.removed_elements.len() != refinement.new_elements.len()
        || mesh.num_elements() != baseline_elements
        || mesh.boundaries.len() != baseline_boundaries
        || area_relative_error > 1.0e-12
    {
        return Err(format!(
            "coarsening conservation/topology mismatch: restored={}, removed={}, elements={}, boundaries={}, area_error={area_relative_error:.3e}",
            coarsening.restored_elements.len(),
            coarsening.removed_elements.len(),
            mesh.num_elements(),
            mesh.boundaries.len(),
        ));
    }
    if !check_orientation(&mesh).is_empty() || !check_manifold(&mesh).is_manifold() {
        return Err("coarsened mesh failed orientation/manifold checks".into());
    }
    let compaction = compact_unused_nodes_checked(&mut mesh)
        .map_err(|error| format!("post-coarsening node compaction failed: {error}"))?;
    let compacted_nodes = mesh.nodes.len();
    if compaction.removed_nodes == 0 || compacted_nodes != baseline_nodes {
        return Err(format!(
            "coarsened orphan-node compaction mismatch: removed={}, nodes={compacted_nodes}, baseline={baseline_nodes}",
            compaction.removed_nodes,
        ));
    }

    // A second run must produce identical final connectivity and provenance.
    let mut repeated = unit_square_triangles(2);
    repeated.detect_boundaries();
    let (_, repeated_history) = refine_elements_with_history_checked(&mut repeated, &[0])
        .map_err(|error| format!("repeated refinement failed: {error}"))?;
    let repeated_parent = repeated_history.parents[0].element.id;
    coarsen_elements_checked(&mut repeated, &repeated_history, &[repeated_parent])
        .map_err(|error| format!("repeated coarsening failed: {error}"))?;
    compact_unused_nodes_checked(&mut repeated)
        .map_err(|error| format!("repeated post-coarsening compaction failed: {error}"))?;
    let deterministic_repeat = mesh_signature(&mesh)? == mesh_signature(&repeated)?;
    if !deterministic_repeat {
        return Err("reversible adaptation changed serialized ordering on replay".into());
    }

    // Incomplete sibling families must fail without mutating the candidate.
    let mut malformed = unit_square_triangles(1);
    malformed.boundaries.clear();
    let (_, malformed_history) = refine_elements_with_history_checked(&mut malformed, &[0])
        .map_err(|error| format!("malformed fixture refinement failed: {error}"))?;
    let malformed_parent = malformed_history.parents[0].element.id;
    let child = malformed
        .elements
        .iter()
        .position(|element| element.parent_id == Some(malformed_parent))
        .ok_or_else(|| "malformed fixture has no child".to_owned())?;
    malformed.elements.remove(child);
    let before = mesh_signature(&malformed)?;
    if coarsen_elements_checked(&mut malformed, &malformed_history, &[malformed_parent]).is_ok() {
        return Err("incomplete sibling family was accepted".into());
    }
    if mesh_signature(&malformed)? != before {
        return Err("incomplete-family rejection mutated the mesh".into());
    }

    Ok(Metrics {
        baseline_nodes,
        baseline_elements,
        refined_nodes,
        refined_elements,
        compacted_nodes,
        removed_orphan_nodes: compaction.removed_nodes,
        restored_elements: coarsening.restored_elements.len(),
        removed_elements: coarsening.removed_elements.len(),
        boundary_count: mesh.boundaries.len(),
        area_relative_error,
        deterministic_repeat,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed("mesher.coarsening.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("mesher-coarsening-v1")
                .metric("baseline_nodes", metrics.baseline_nodes as f64)
                .metric("baseline_elements", metrics.baseline_elements as f64)
                .metric("refined_nodes", metrics.refined_nodes as f64)
                .metric("refined_elements", metrics.refined_elements as f64)
                .metric("compacted_nodes", metrics.compacted_nodes as f64)
                .metric("removed_orphan_nodes", metrics.removed_orphan_nodes as f64)
                .metric("restored_elements", metrics.restored_elements as f64)
                .metric("removed_elements", metrics.removed_elements as f64)
                .metric("boundary_count", metrics.boundary_count as f64)
                .metric("area_relative_error", metrics.area_relative_error)
                .metric(
                    "deterministic_repeat",
                    if metrics.deterministic_repeat {
                        1.0
                    } else {
                        0.0
                    },
                )
                .minimum("restored_elements", 1.0)
                .minimum("removed_elements", 1.0)
                .minimum("removed_orphan_nodes", 1.0)
                .maximum("area_relative_error", 1.0e-12)
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
            if let Ok(json) = QaResult::failed("mesher.coarsening.v1", &error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("mesher-coarsening-v1")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("MESHER COARSENING FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
