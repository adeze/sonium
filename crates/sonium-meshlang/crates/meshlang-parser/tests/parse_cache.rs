use meshlang_parser::{MAX_CACHED_FILES, PARSE_JSON_SCHEMA_VERSION, ParseCache, ParseOptions};
use serde_json::Value;

const SOURCE: &str = "geometry model { let body = Box(1.0, 1.0, 1.0) }";

#[test]
fn cached_json_reuses_typed_report_and_preserves_envelope() {
    let mut cache = ParseCache::default();
    let first: Value = serde_json::from_str(
        &cache
            .parse_json(SOURCE, "cache.mlg", ParseOptions::default())
            .expect("first cached parse"),
    )
    .expect("first JSON envelope");
    let second: Value = serde_json::from_str(
        &cache
            .parse_json(SOURCE, "cache.mlg", ParseOptions::default())
            .expect("cache hit parse"),
    )
    .expect("second JSON envelope");

    assert_eq!(first, second);
    assert_eq!(first["parse_schema"], PARSE_JSON_SCHEMA_VERSION);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.len(), 1);
}

#[test]
fn cache_invalidation_forces_recovery_again_without_losing_other_entries() {
    let mut cache = ParseCache::default();
    let options = ParseOptions::default();
    cache.parse_json(SOURCE, "one.mlg", options).unwrap();
    cache.parse_json(SOURCE, "two.mlg", options).unwrap();
    assert!(cache.invalidate("one.mlg"));
    assert!(!cache.invalidate("missing.mlg"));

    cache.parse_json(SOURCE, "one.mlg", options).unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.stats().misses, 3);
    assert_eq!(cache.stats().hits, 0);
}

#[test]
fn file_map_is_capacity_bounded() {
    let mut cache = ParseCache::default();
    let options = ParseOptions::default();
    for index in 0..(MAX_CACHED_FILES + 4) {
        cache.parse(SOURCE, &format!("file-{index}.mlg"), options);
    }
    assert!(cache.len() <= MAX_CACHED_FILES);
    assert!(cache.stats().evictions >= 1);
}

#[test]
fn cache_options_are_part_of_the_deterministic_key() {
    let mut cache = ParseCache::default();
    let mut strict = ParseOptions::default();
    strict.max_diagnostics = 1;
    cache
        .parse_json(SOURCE, "options.mlg", ParseOptions::default())
        .unwrap();
    cache.parse_json(SOURCE, "options.mlg", strict).unwrap();
    assert_eq!(cache.stats().misses, 2);
    assert_eq!(cache.stats().hits, 0);
}
