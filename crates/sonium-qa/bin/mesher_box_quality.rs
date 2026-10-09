//! Structured mesher contract for a deterministic box tetrahedralization.

use sonium_mesher::mesh::tetrahedralize::tetrahedralize_box_with_result;
use sonium_mesher::mesh::validate::{check_manifold, check_orientation};
use sonium_mesher::partition::{PartitionOptions, build_element_graph, partition_mesh};
use sonium_qa::QaResult;
use std::time::Instant;

#[derive(Debug)]
struct MeshQaMetrics {
    nodes: usize,
    elements: usize,
    quality_min: f64,
    quality_max: f64,
    num_slivers: usize,
    invalid_elements: usize,
    non_manifold_faces: usize,
    partition_min_size: usize,
    partition_max_size: usize,
    partition_edge_cut: usize,
    recovery_rounds: usize,
    recovery_steiner_points: usize,
    recovery_operations: usize,
    recovery_missing_edges: usize,
    recovery_missing_facets: usize,
    deterministic_repeat: f64,
    edge_length_histogram: Vec<usize>,
    element_measure_histogram: Vec<usize>,
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed("mesher.box-quality.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("nodes", metrics.nodes as f64)
                .metric("elements", metrics.elements as f64)
                .metric("quality_min", metrics.quality_min)
                .metric("quality_max", metrics.quality_max)
                .metric("num_slivers", metrics.num_slivers as f64)
                .metric("invalid_elements", metrics.invalid_elements as f64)
                .metric("non_manifold_faces", metrics.non_manifold_faces as f64)
                .metric("partition_min_size", metrics.partition_min_size as f64)
                .metric("partition_max_size", metrics.partition_max_size as f64)
                .metric("partition_edge_cut", metrics.partition_edge_cut as f64)
                .metric("recovery_rounds", metrics.recovery_rounds as f64)
                .metric(
                    "recovery_steiner_points",
                    metrics.recovery_steiner_points as f64,
                )
                .metric("recovery_operations", metrics.recovery_operations as f64)
                .metric(
                    "recovery_missing_edges",
                    metrics.recovery_missing_edges as f64,
                )
                .metric(
                    "recovery_missing_facets",
                    metrics.recovery_missing_facets as f64,
                );
            let mut record = record
                .metric("deterministic_repeat", metrics.deterministic_repeat)
                .minimum("deterministic_repeat", 1.0);
            for (index, count) in metrics.edge_length_histogram.iter().enumerate() {
                record = record.metric(format!("edge_length_bin_{index}"), *count as f64);
            }
            for (index, count) in metrics.element_measure_histogram.iter().enumerate() {
                record = record.metric(format!("element_measure_bin_{index}"), *count as f64);
            }
            let record = record
                .minimum("quality_min", f64::EPSILON)
                .maximum("non_manifold_faces", 0.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed("mesher.box-quality.v1", error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("MESHER QUALITY FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<MeshQaMetrics, String> {
    let result = tetrahedralize_box_with_result(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3);
    let repeat = tetrahedralize_box_with_result(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3);
    let first_mesh = serde_json::to_vec(&result.mesh)
        .map_err(|error| format!("failed to serialize first deterministic mesh: {error}"))?;
    let repeat_mesh = serde_json::to_vec(&repeat.mesh)
        .map_err(|error| format!("failed to serialize repeated deterministic mesh: {error}"))?;
    if first_mesh != repeat_mesh {
        return Err("repeated box tetrahedralization changed serialized mesh ordering".into());
    }
    if result.mesh.nodes.is_empty() || result.mesh.elements.is_empty() {
        return Err("box tetrahedralization returned an empty mesh".into());
    }
    let inverted = check_orientation(&result.mesh);
    if !inverted.is_empty() {
        return Err(format!("{} inverted elements", inverted.len()));
    }
    let manifold = check_manifold(&result.mesh);
    if !manifold.is_manifold() {
        return Err(format!(
            "{} non-manifold faces",
            manifold.non_manifold_faces.len()
        ));
    }
    if !result.quality_min.is_finite() || result.quality_min <= 0.0 {
        return Err(format!("invalid minimum quality {}", result.quality_min));
    }
    let diagnostics = result.diagnostics(8);
    let graph = build_element_graph(&result.mesh);
    let partition = partition_mesh(
        &result.mesh,
        &PartitionOptions {
            num_partitions: 2,
            refinement: 0,
            ..PartitionOptions::default()
        },
    );
    let sizes = partition.partition_sizes();
    let recovery = diagnostics.constraint_recovery.as_ref();
    let recovery_operations = recovery.map_or(0, |recovery| {
        let operations = recovery.operations;
        operations.flip23
            + operations.flip32
            + operations.flip44
            + operations.retriangulate_nm
            + operations.edge_splits
            + operations.facet_splits
            + operations.facet_tiling_adoptions
            + operations.steiner_suppressions
    });
    Ok(MeshQaMetrics {
        nodes: result.mesh.nodes.len(),
        elements: result.mesh.elements.len(),
        quality_min: result.quality_min,
        quality_max: result.quality_max,
        num_slivers: result.num_slivers,
        invalid_elements: diagnostics.quality.invalid_elements,
        non_manifold_faces: manifold.non_manifold_faces.len(),
        partition_min_size: sizes.iter().copied().min().unwrap_or(0),
        partition_max_size: sizes.iter().copied().max().unwrap_or(0),
        partition_edge_cut: partition.edge_cut(&graph),
        recovery_rounds: recovery.map_or(0, |recovery| recovery.rounds),
        recovery_steiner_points: recovery.map_or(0, |recovery| recovery.steiner_points),
        recovery_operations,
        recovery_missing_edges: recovery.map_or(0, |recovery| recovery.missing_edges.len()),
        recovery_missing_facets: recovery.map_or(0, |recovery| recovery.missing_facets.len()),
        deterministic_repeat: 1.0,
        edge_length_histogram: diagnostics.quality.edge_lengths.bins,
        element_measure_histogram: diagnostics.quality.element_measures.bins,
    })
}
