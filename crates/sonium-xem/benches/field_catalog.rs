//! Audit `reviews/audit-sonium-xem-20260921.md`: field-catalog cost.
//!
//! The audit asks for allocation/serialization benchmarks on large field
//! catalogs. This bench measures `MultiphysicsProblem::validate` (the
//! per-admission cost every consumer pays) and `serde_json` serialization
//! (the exchange cost) over synthetic acoustics catalogs. It is a planning
//! input for admission budgets, not a throughput claim.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sonium_xem::{
    ComplexStorage, Domain, DomainKind, FieldKind, FieldSpec, MultiphysicsProblem, Region,
    ScalarType, StorageOrder, TensorLayout,
};

fn catalog(num_fields: usize) -> MultiphysicsProblem {
    let fields = (0..num_fields)
        .map(|index| FieldSpec {
            id: format!("pressure-{index:05}"),
            name: "complex acoustic pressure".into(),
            domain_id: "acoustics".into(),
            kind: FieldKind::ComplexScalar,
            units: Some("Pa".into()),
            layout: TensorLayout {
                scalar_type: ScalarType::ComplexF64,
                shape: vec![8],
                strides: vec![1],
                byte_offset: 0,
                order: StorageOrder::RowMajor,
                complex_storage: ComplexStorage::Interleaved,
                coordinate_frame: None,
            },
        })
        .collect();
    MultiphysicsProblem {
        id: "bench-field-catalog-v1".into(),
        domains: vec![Domain {
            id: "acoustics".into(),
            label: "bounded acoustic domain".into(),
            kind: DomainKind::Acoustics,
        }],
        regions: vec![Region {
            id: "air".into(),
            domain_id: "acoustics".into(),
            label: None,
            parent_geometry_entity: None,
            material_law_id: None,
        }],
        fields,
        ..MultiphysicsProblem::default()
    }
}

fn bench_catalog(criterion: &mut Criterion) {
    for num_fields in [256usize, 2048, 8192] {
        let problem = catalog(num_fields);
        problem
            .validate()
            .expect("bench catalog must pass admission");
        criterion.bench_with_input(
            BenchmarkId::new("validate", num_fields),
            &problem,
            |bencher, problem| bencher.iter(|| problem.validate()),
        );
        criterion.bench_with_input(
            BenchmarkId::new("serialize_json", num_fields),
            &problem,
            |bencher, problem| bencher.iter(|| serde_json::to_string(problem)),
        );
    }
}

criterion_group!(benches, bench_catalog);
criterion_main!(benches);
