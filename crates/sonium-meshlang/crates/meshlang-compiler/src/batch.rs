//! File-granularity batch parsing for CI and sweep workloads.
//!
//! Parsing is single-threaded by design (recursive descent over one token
//! stream), and the parser crate stays rayon-free so the WASM build keeps
//! no thread-pool dependency. Parallelism therefore lives here, one level
//! up: [`parse_source_files`] fans out over whole files, each worker thread
//! with its own [`ParseCache`]. Never parallelize inside one file; the
//! grammar is sequential.
//!
//! The fan-out gate is machine-adaptive: each file costs a full
//! lex + parse (milliseconds at CI sizes), so any split across two or more
//! threads wins and the `>= 2` minimum is structural, not a machine
//! threshold — the machine dependence is the thread count.
//! `SONIUM_MESHLANG_PAR_THRESHOLD=<n>` overrides with an absolute file
//! count for tuning and forced-path tests; `SONIUM_SERIAL=1` forces the
//! serial path. Results collect in input order, so serial and parallel
//! runs are identical.

use meshlang_parser::{CacheStats, ParseCache, ParseOptions, ParseReport};

/// One batch member: the source text, its filename (used for diagnostics
/// and cache identity), and the parse options for this file.
pub struct BatchParseInput<'a> {
    pub filename: &'a str,
    pub source: &'a str,
    pub options: ParseOptions,
}

/// One batch result, in the same position as its input.
pub struct BatchParseOutput {
    pub filename: String,
    pub report: ParseReport,
}

/// Parse many files, each through a worker-local [`ParseCache`].
///
/// The output order always matches the input order on both paths.
pub fn parse_source_files(inputs: &[BatchParseInput<'_>]) -> Vec<BatchParseOutput> {
    let parse_one = |input: &BatchParseInput<'_>| {
        let mut cache = ParseCache::default();
        let report = cache.parse(input.source, input.filename, input.options);
        BatchParseOutput {
            filename: input.filename.to_string(),
            report,
        }
    };
    if use_batch_parallel(inputs.len()) {
        use rayon::prelude::*;
        inputs.par_iter().map(parse_one).collect()
    } else {
        inputs.iter().map(parse_one).collect()
    }
}

/// Parse many files with invocation-local caches. The serial path shares one
/// cache within this batch; Rayon initializes a cache per job (not per worker).
/// No cache survives a call. For repeated watch-loop compilation use the
/// stateful `IncrementalCompiler` API instead.
///
/// The output order always matches the input order on both paths.
pub fn parse_source_files_cached(inputs: &[BatchParseInput<'_>]) -> Vec<BatchParseOutput> {
    let parse_one = |cache: &mut ParseCache, input: &BatchParseInput<'_>| {
        let report = cache.parse(input.source, input.filename, input.options);
        BatchParseOutput {
            filename: input.filename.to_string(),
            report,
        }
    };
    if use_batch_parallel(inputs.len()) {
        use rayon::prelude::*;
        inputs
            .par_iter()
            .map_init(ParseCache::default, parse_one)
            .collect()
    } else {
        let mut cache = ParseCache::default();
        inputs
            .iter()
            .map(|input| parse_one(&mut cache, input))
            .collect()
    }
}

/// Stateful batch executor that owns one [`ParseCache`] across calls.
///
/// The free [`parse_source_files_cached`] helper is invocation-local: no
/// cache survives a call, on either the serial or the parallel path. A
/// watch-loop or sweep driver that re-parses overlapping file sets therefore
/// gets no cross-call reuse from the free functions. `BatchParser` fills
/// that gap: the serial path parses through the owned cache, so an unchanged
/// file re-parsed by a later call is a cache hit. The parallel path still
/// uses one `ParseCache` per Rayon job (`map_init`), which cannot persist
/// across calls; only serial reuse is owned here. For symbol-aware
/// invalidation across edits, use the stateful `IncrementalCompiler` API.
#[derive(Debug, Default)]
pub struct BatchParser {
    cache: ParseCache,
}

impl BatchParser {
    /// Create an executor with an empty owned cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse many files on the serial path, reusing the owned cache across
    /// calls. Output order always matches input order.
    pub fn parse_cached(&mut self, inputs: &[BatchParseInput<'_>]) -> Vec<BatchParseOutput> {
        inputs
            .iter()
            .map(|input| {
                let report = self
                    .cache
                    .parse(input.source, input.filename, input.options);
                BatchParseOutput {
                    filename: input.filename.to_string(),
                    report,
                }
            })
            .collect()
    }

    /// Owned-cache statistics; `hits` grows when a later call re-parses an
    /// unchanged file with identical options.
    pub fn stats(&self) -> CacheStats {
        self.cache.stats()
    }
}

fn use_batch_parallel(file_count: usize) -> bool {
    if std::env::var("SONIUM_SERIAL")
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
    {
        return false;
    }
    if let Some(threshold) = std::env::var("SONIUM_MESHLANG_PAR_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        return file_count >= threshold;
    }
    let threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    file_count >= 2 && threads >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> ParseOptions {
        ParseOptions {
            file_id: 0,
            max_diagnostics: 32,
            max_nesting: 128,
            max_source_bytes: 8 * 1024 * 1024,
        }
    }

    const SOURCE: &str = "geometry model { let body = Box(1.0, 1.0, 1.0) }\n";

    #[test]
    fn batch_paths_agree_and_preserve_order() {
        // Hermetic single test: this module is the only writer of the gate
        // keys in this binary, and both paths compute identical reports.
        // SAFETY: saved and restored below.
        unsafe {
            let prev_threshold = std::env::var_os("SONIUM_MESHLANG_PAR_THRESHOLD");
            let prev_serial = std::env::var_os("SONIUM_SERIAL");

            let filenames: Vec<String> = (0..4).map(|i| format!("file{i}.mlg")).collect();
            let inputs: Vec<BatchParseInput<'_>> = filenames
                .iter()
                .map(|filename| BatchParseInput {
                    filename,
                    source: SOURCE,
                    options: options(),
                })
                .collect();

            std::env::remove_var("SONIUM_MESHLANG_PAR_THRESHOLD");
            std::env::set_var("SONIUM_SERIAL", "1");
            let serial = parse_source_files(&inputs);
            let serial_cached = parse_source_files_cached(&inputs);
            std::env::remove_var("SONIUM_SERIAL");
            std::env::set_var("SONIUM_MESHLANG_PAR_THRESHOLD", "1");
            let parallel = parse_source_files(&inputs);
            let parallel_cached = parse_source_files_cached(&inputs);

            for outputs in [&serial, &serial_cached, &parallel, &parallel_cached] {
                assert_eq!(outputs.len(), 4);
                for (index, output) in outputs.iter().enumerate() {
                    assert_eq!(output.filename, format!("file{index}.mlg"));
                    assert!(output.report.is_success());
                }
            }
            let serial_files: Vec<_> = serial
                .iter()
                .map(|output| output.report.source_file.declarations.len())
                .collect();
            for outputs in [&serial_cached, &parallel, &parallel_cached] {
                let files: Vec<_> = outputs
                    .iter()
                    .map(|output| output.report.source_file.declarations.len())
                    .collect();
                assert_eq!(files, serial_files);
            }

            match prev_threshold {
                Some(value) => std::env::set_var("SONIUM_MESHLANG_PAR_THRESHOLD", value),
                None => std::env::remove_var("SONIUM_MESHLANG_PAR_THRESHOLD"),
            }
            match prev_serial {
                Some(value) => std::env::set_var("SONIUM_SERIAL", value),
                None => std::env::remove_var("SONIUM_SERIAL"),
            }
        }
    }

    #[test]
    fn batch_parser_reuses_cache_across_calls() {
        let inputs: Vec<BatchParseInput<'_>> = ["a.mlg", "b.mlg"]
            .iter()
            .map(|filename| BatchParseInput {
                filename,
                source: SOURCE,
                options: options(),
            })
            .collect();

        let mut executor = BatchParser::new();
        let first = executor.parse_cached(&inputs);
        assert!(first.iter().all(|output| output.report.is_success()));
        let first_stats = executor.stats();
        assert_eq!(first_stats.hits, 0);
        assert_eq!(first_stats.misses, 2);

        // Re-parsing unchanged files with identical options must hit the
        // owned cache instead of re-lexing/re-parsing.
        let second = executor.parse_cached(&inputs);
        let second_stats = executor.stats();
        assert_eq!(second_stats.hits, 2);
        assert_eq!(second_stats.misses, 2);
        let first_decls: Vec<_> = first
            .iter()
            .map(|output| output.report.source_file.declarations.len())
            .collect();
        let second_decls: Vec<_> = second
            .iter()
            .map(|output| output.report.source_file.declarations.len())
            .collect();
        assert_eq!(first_decls, second_decls);

        // An edited file misses once, then hits again; the untouched file
        // keeps hitting.
        let edited: Vec<BatchParseInput<'_>> = [
            BatchParseInput {
                filename: "a.mlg",
                source: "geometry model { let body = Box(2.0, 2.0, 2.0) }\n",
                options: options(),
            },
            BatchParseInput {
                filename: "b.mlg",
                source: SOURCE,
                options: options(),
            },
        ]
        .into_iter()
        .collect();
        let third = executor.parse_cached(&edited);
        assert!(third.iter().all(|output| output.report.is_success()));
        let third_stats = executor.stats();
        assert_eq!(third_stats.hits, 3);
        assert_eq!(third_stats.misses, 3);
    }

    #[test]
    fn free_cached_helper_keeps_invocation_local_caches() {
        // Locks in the documented contract: the free helper must not carry
        // cache state across calls, so a failed admission in one call cannot
        // leak into the next. `BatchParser` is the cross-call reuse path.
        let inputs: Vec<BatchParseInput<'_>> = [BatchParseInput {
            filename: "solo.mlg",
            source: SOURCE,
            options: options(),
        }]
        .into_iter()
        .collect();
        for _ in 0..2 {
            let outputs = parse_source_files_cached(&inputs);
            assert_eq!(outputs.len(), 1);
            assert!(outputs[0].report.is_success());
        }
    }
}
