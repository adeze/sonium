use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{DirectFactorization, SparseLuFactorization, lu_factorize};
use sonium_solvers::sparse::CsrMatrix;
use std::env;
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const RESIDUAL_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug)]
struct Config {
    output: PathBuf,
    dense_sizes: Vec<usize>,
    sparse_grids: Vec<usize>,
    repeats: usize,
}

#[derive(Debug)]
struct DenseResult {
    n: usize,
    sonium_ms: f64,
    sonium_factor_ms: f64,
    sonium_solve_ms: f64,
    accelerate_ms: Option<f64>,
    sonium_residual: f64,
    accelerate_residual: Option<f64>,
    factor_nonzeros: usize,
    factor_memory_bytes: usize,
    modeled_peak_memory_bytes: usize,
    peak_rss_bytes: Option<u64>,
}

#[derive(Debug)]
struct SparseResult {
    grid: usize,
    n: usize,
    nnz: usize,
    simplicial_ms: f64,
    simplicial_factor_ms: f64,
    simplicial_solve_ms: f64,
    supernodal_ms: f64,
    supernodal_factor_ms: f64,
    supernodal_solve_ms: f64,
    supernodal_reused_ms: f64,
    supernodal_reused_factor_ms: f64,
    supernodal_reused_solve_ms: f64,
    superlu_ms: Option<f64>,
    max_residual: f64,
    simplicial_factor_nonzeros: usize,
    supernodal_factor_nonzeros: usize,
    supernodal_reused_factor_nonzeros: usize,
    simplicial_factor_memory_bytes: usize,
    supernodal_factor_memory_bytes: usize,
    supernodal_reused_factor_memory_bytes: usize,
    modeled_peak_memory_bytes: usize,
    peak_rss_bytes: Option<u64>,
}

/// Peak resident set size of this process, if the platform exposes it.
///
/// Linux reads `/proc/self/status` (`VmHWM`, the high-water mark, so it is a
/// true peak rather than a point sample). Other platforms report `None` and
/// render as "n/a".
fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|contents| {
                contents.lines().find_map(|line| {
                    let rest = line.strip_prefix("VmHWM:")?.trim();
                    let kilobytes: u64 = rest.split_whitespace().next()?.parse().ok()?;
                    Some(kilobytes.saturating_mul(1024))
                })
            })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("direct-solver-perf: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args(env::args().skip(1))?;
    let dense = config
        .dense_sizes
        .iter()
        .map(|&n| benchmark_dense(n, config.repeats))
        .collect::<Result<Vec<_>, _>>()?;
    let sparse = config
        .sparse_grids
        .iter()
        .map(|&grid| benchmark_sparse(grid, config.repeats))
        .collect::<Result<Vec<_>, _>>()?;
    let report = render_report(&config, &dense, &sparse);

    if let Some(parent) = config.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let temporary = config.output.with_extension("md.tmp");
    std::fs::write(&temporary, report)
        .map_err(|error| format!("cannot write {}: {error}", temporary.display()))?;
    std::fs::rename(&temporary, &config.output)
        .map_err(|error| format!("cannot replace {}: {error}", config.output.display()))?;
    println!("wrote {}", config.output.display());
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config, String> {
    let mut output = PathBuf::from("perf/PERF.md");
    let mut dense_sizes = vec![64, 128, 256, 512];
    let mut sparse_grids = vec![8, 12, 16, 20, 28];
    let mut repeats = 3;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("missing value after {argument}"))
        };
        match argument.as_str() {
            "--output" => output = PathBuf::from(value()?),
            "--dense-sizes" => dense_sizes = parse_sizes(&value()?)?,
            "--sparse-grids" => sparse_grids = parse_sizes(&value()?)?,
            "--repeats" => {
                repeats = value()?
                    .parse()
                    .map_err(|_| "--repeats must be a positive integer".to_string())?;
                if repeats == 0 {
                    return Err("--repeats must be a positive integer".into());
                }
            }
            "--help" | "-h" => {
                return Err("usage: direct-solver-perf [--output perf/PERF.md] \
                     [--dense-sizes 64,128,...] [--sparse-grids 8,12,...] \
                     [--repeats 3]"
                    .into());
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(Config {
        output,
        dense_sizes,
        sparse_grids,
        repeats,
    })
}

fn parse_sizes(value: &str) -> Result<Vec<usize>, String> {
    let sizes = value
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid size: {item}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if sizes.is_empty() || sizes.contains(&0) {
        return Err("sizes must be a non-empty list of positive integers".into());
    }
    Ok(sizes)
}

fn benchmark_dense(n: usize, repeats: usize) -> Result<DenseResult, String> {
    eprintln!("dense n={n}");
    let matrix = dense_matrix(n);
    let expected: Vec<f64> = (0..n).map(|i| (i as f64 * 0.017).cos()).collect();
    let rhs = dense_matvec(&matrix, &expected);

    let factor = lu_factorize(&matrix).map_err(|error| format!("Sonium dense n={n}: {error}"))?;
    let solution = factor
        .solve(&rhs)
        .map_err(|error| format!("Sonium dense solve n={n}: {error}"))?;
    let sonium_residual = dense_relative_residual(&matrix, &solution, &rhs);
    require_residual("Sonium dense", n, sonium_residual)?;
    let sonium_ms = median_ms(repeats, || {
        let factor = lu_factorize(black_box(&matrix)).map_err(|error| error.to_string())?;
        let solution = factor
            .solve(black_box(&rhs))
            .map_err(|error| error.to_string())?;
        black_box(solution);
        Ok(())
    })?;
    let sonium_factor_ms = median_ms(repeats, || {
        black_box(lu_factorize(black_box(&matrix)).map_err(|error| error.to_string())?);
        Ok(())
    })?;
    let sonium_solve_ms = median_ms(repeats, || {
        black_box(
            factor
                .solve(black_box(&rhs))
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    })?;

    #[cfg(has_accelerate)]
    let (accelerate_ms, accelerate_residual) = {
        let solution = accelerate::solve(&matrix, &rhs)?;
        let residual = dense_relative_residual(&matrix, &solution, &rhs);
        require_residual("Accelerate", n, residual)?;
        let milliseconds = median_ms(repeats, || {
            black_box(accelerate::solve(black_box(&matrix), black_box(&rhs))?);
            Ok(())
        })?;
        (Some(milliseconds), Some(residual))
    };
    #[cfg(not(has_accelerate))]
    let (accelerate_ms, accelerate_residual) = (None, None);

    let factor_nonzeros = factor.factor_nonzeros();
    let factor_memory_bytes = factor.factor_memory_bytes();
    let modeled_peak_memory_bytes = dense_modeled_peak_memory_bytes(n, factor_memory_bytes);

    Ok(DenseResult {
        n,
        sonium_ms,
        sonium_factor_ms,
        sonium_solve_ms,
        accelerate_ms,
        sonium_residual,
        accelerate_residual,
        factor_nonzeros,
        factor_memory_bytes,
        modeled_peak_memory_bytes,
        peak_rss_bytes: peak_rss_bytes(),
    })
}

fn benchmark_sparse(grid: usize, repeats: usize) -> Result<SparseResult, String> {
    let matrix = convection_diffusion(grid);
    let n = matrix.num_rows;
    eprintln!("sparse grid={grid}x{grid}, n={n}, nnz={}", matrix.nnz());
    let expected: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).sin()).collect();
    let rhs = matrix.matvec(&expected);

    let simplicial = SparseLuFactorization::factorize(&matrix)
        .map_err(|error| format!("Sonium simplicial n={n}: {error}"))?;
    let solution = simplicial
        .solve(&rhs)
        .map_err(|error| format!("Sonium simplicial solve n={n}: {error}"))?;
    let simplicial_residual = sparse_relative_residual(&matrix, &solution, &rhs);
    require_residual("Sonium sparse simplicial", n, simplicial_residual)?;

    let supernodal = SparseLuFactorization::factorize_supernodal(&matrix)
        .map_err(|error| format!("Sonium supernodal n={n}: {error}"))?;
    let solution = supernodal
        .solve(&rhs)
        .map_err(|error| format!("Sonium supernodal solve n={n}: {error}"))?;
    let supernodal_residual = sparse_relative_residual(&matrix, &solution, &rhs);
    require_residual("Sonium sparse supernodal", n, supernodal_residual)?;

    let symbolic = SparseLuFactorization::analyze_supernodal(&matrix)
        .map_err(|error| format!("Sonium symbolic analysis n={n}: {error}"))?;
    let supernodal_reused =
        SparseLuFactorization::factorize_supernodal_with_symbolic(&matrix, &symbolic, 0.1)
            .map_err(|error| format!("Sonium reused-symbolic n={n}: {error}"))?;
    let reused_solution = supernodal_reused
        .solve(&rhs)
        .map_err(|error| format!("Sonium reused-symbolic solve n={n}: {error}"))?;
    let reused_residual = sparse_relative_residual(&matrix, &reused_solution, &rhs);
    require_residual("Sonium reused-symbolic", n, reused_residual)?;
    let simplicial_ms = median_ms(repeats, || {
        let factor = SparseLuFactorization::factorize(black_box(&matrix))
            .map_err(|error| error.to_string())?;
        black_box(
            factor
                .solve(black_box(&rhs))
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    })?;
    let simplicial_factor_ms = median_ms(repeats, || {
        black_box(
            SparseLuFactorization::factorize(black_box(&matrix))
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    })?;
    let simplicial_solve_ms = median_ms(repeats, || {
        black_box(
            simplicial
                .solve(black_box(&rhs))
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    })?;
    let supernodal_ms = median_ms(repeats, || {
        let factor = SparseLuFactorization::factorize_supernodal(black_box(&matrix))
            .map_err(|error| error.to_string())?;
        black_box(
            factor
                .solve(black_box(&rhs))
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    })?;
    let supernodal_factor_ms = median_ms(repeats, || {
        black_box(
            SparseLuFactorization::factorize_supernodal(black_box(&matrix))
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    })?;
    let supernodal_solve_ms = median_ms(repeats, || {
        black_box(
            supernodal
                .solve(black_box(&rhs))
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    })?;
    let supernodal_reused_ms = median_ms(repeats, || {
        let factor = SparseLuFactorization::factorize_supernodal_with_symbolic(
            black_box(&matrix),
            black_box(&symbolic),
            0.1,
        )
        .map_err(|error| error.to_string())?;
        black_box(
            factor
                .solve(black_box(&rhs))
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    })?;
    let supernodal_reused_factor_ms = median_ms(repeats, || {
        black_box(
            SparseLuFactorization::factorize_supernodal_with_symbolic(
                black_box(&matrix),
                black_box(&symbolic),
                0.1,
            )
            .map_err(|error| error.to_string())?,
        );
        Ok(())
    })?;
    let supernodal_reused_solve_ms = median_ms(repeats, || {
        black_box(
            supernodal_reused
                .solve(black_box(&rhs))
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    })?;

    #[cfg(has_superlu)]
    let (superlu_ms, superlu_residual) = {
        let prepared = superlu::PreparedMatrix::new(&matrix)?;
        let solution = prepared.solve(&rhs)?;
        let residual = sparse_relative_residual(&matrix, &solution, &rhs);
        require_residual("SuperLU", n, residual)?;
        let milliseconds = median_ms(repeats, || {
            black_box(prepared.solve(black_box(&rhs))?);
            Ok(())
        })?;
        (Some(milliseconds), residual)
    };
    #[cfg(not(has_superlu))]
    let (superlu_ms, superlu_residual) = (None, 0.0);

    let simplicial_factor_nonzeros = simplicial.factor_nonzeros();
    let supernodal_factor_nonzeros = supernodal.factor_nonzeros();
    let supernodal_reused_factor_nonzeros = supernodal_reused.factor_nonzeros();
    let simplicial_factor_memory_bytes = simplicial_factor_nonzeros * std::mem::size_of::<f64>();
    let supernodal_factor_memory_bytes = supernodal_factor_nonzeros * std::mem::size_of::<f64>();
    let supernodal_reused_factor_memory_bytes =
        supernodal_reused_factor_nonzeros * std::mem::size_of::<f64>();
    let modeled_peak_memory_bytes = sparse_modeled_peak_memory_bytes(
        n,
        matrix.nnz(),
        simplicial_factor_memory_bytes
            .max(supernodal_factor_memory_bytes)
            .max(supernodal_reused_factor_memory_bytes),
    );

    Ok(SparseResult {
        grid,
        n,
        nnz: matrix.nnz(),
        simplicial_ms,
        simplicial_factor_ms,
        simplicial_solve_ms,
        supernodal_ms,
        supernodal_factor_ms,
        supernodal_solve_ms,
        supernodal_reused_ms,
        supernodal_reused_factor_ms,
        supernodal_reused_solve_ms,
        superlu_ms,
        max_residual: simplicial_residual
            .max(supernodal_residual)
            .max(reused_residual)
            .max(superlu_residual),
        simplicial_factor_nonzeros,
        supernodal_factor_nonzeros,
        supernodal_reused_factor_nonzeros,
        simplicial_factor_memory_bytes,
        supernodal_factor_memory_bytes,
        supernodal_reused_factor_memory_bytes,
        modeled_peak_memory_bytes,
        peak_rss_bytes: peak_rss_bytes(),
    })
}

fn median_ms(
    repeats: usize,
    mut operation: impl FnMut() -> Result<(), String>,
) -> Result<f64, String> {
    operation()?;
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        operation()?;
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    Ok(median(&mut samples))
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) * 0.5
    } else {
        samples[middle]
    }
}

fn dense_matrix(n: usize) -> DenseMatrix<f64> {
    DenseMatrix::from_fn(n, n, |row, column| {
        let value = (((row * 37 + column * 17 + 11) % 101) as f64 - 50.0) / 101.0;
        if row == column {
            value + n as f64
        } else {
            value
        }
    })
}

fn convection_diffusion(grid: usize) -> CsrMatrix<f64> {
    let n = grid * grid;
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut column_indices = Vec::with_capacity(5 * n);
    let mut values = Vec::with_capacity(5 * n);
    row_ptrs.push(0);
    for row in 0..grid {
        for column in 0..grid {
            let index = row * grid + column;
            if row > 0 {
                column_indices.push(index - grid);
                values.push(-1.1);
            }
            if column > 0 {
                column_indices.push(index - 1);
                values.push(-1.2);
            }
            column_indices.push(index);
            values.push(4.01);
            if column + 1 < grid {
                column_indices.push(index + 1);
                values.push(-0.8);
            }
            if row + 1 < grid {
                column_indices.push(index + grid);
                values.push(-0.9);
            }
            row_ptrs.push(column_indices.len());
        }
    }
    CsrMatrix::from_raw_parts(n, n, row_ptrs, column_indices, values)
}

fn dense_matvec(matrix: &DenseMatrix<f64>, x: &[f64]) -> Vec<f64> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[[row, column]] * x[column])
                .sum()
        })
        .collect()
}

fn dense_relative_residual(matrix: &DenseMatrix<f64>, x: &[f64], rhs: &[f64]) -> f64 {
    relative_residual(&dense_matvec(matrix, x), rhs)
}

fn sparse_relative_residual(matrix: &CsrMatrix<f64>, x: &[f64], rhs: &[f64]) -> f64 {
    relative_residual(&matrix.matvec(x), rhs)
}

fn relative_residual(actual: &[f64], expected: &[f64]) -> f64 {
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    let scale = expected.iter().map(|value| value.abs()).fold(0.0, f64::max);
    error / scale.max(f64::MIN_POSITIVE)
}

fn require_residual(backend: &str, n: usize, residual: f64) -> Result<(), String> {
    if residual <= RESIDUAL_TOLERANCE {
        Ok(())
    } else {
        Err(format!(
            "{backend} failed correctness gate at n={n}: relative residual {residual:e}"
        ))
    }
}

fn dense_modeled_peak_memory_bytes(n: usize, factor_memory_bytes: usize) -> usize {
    let scalar_bytes = std::mem::size_of::<f64>();
    n.saturating_mul(n)
        .saturating_mul(scalar_bytes)
        .saturating_add(factor_memory_bytes)
        .saturating_add(n.saturating_mul(3).saturating_mul(scalar_bytes))
}

fn sparse_modeled_peak_memory_bytes(n: usize, nnz: usize, factor_memory_bytes: usize) -> usize {
    let scalar_bytes = std::mem::size_of::<f64>();
    let csr_bytes = (n.saturating_add(1).saturating_add(nnz))
        .saturating_mul(std::mem::size_of::<usize>())
        .saturating_add(nnz.saturating_mul(scalar_bytes));
    csr_bytes
        .saturating_add(factor_memory_bytes)
        .saturating_add(n.saturating_mul(3).saturating_mul(scalar_bytes))
}

fn render_report(config: &Config, dense: &[DenseResult], sparse: &[SparseResult]) -> String {
    let mut report = String::new();
    writeln!(report, "# Direct Solver Performance\n").unwrap();
    writeln!(
        report,
        "_Generated {} from commit `{}`._\n",
        command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        git_revision()
    )
    .unwrap();
    writeln!(report, "## Platform\n").unwrap();
    writeln!(report, "| Property | Value |\n|---|---|").unwrap();
    writeln!(
        report,
        "| OS | {} |",
        markdown_cell(&command_output("uname", &["-srv"]))
    )
    .unwrap();
    writeln!(report, "| Architecture | {} |", env::consts::ARCH).unwrap();
    writeln!(report, "| CPU | {} |", markdown_cell(&cpu_name())).unwrap();
    writeln!(report, "| Memory | {} |", markdown_cell(&memory_size())).unwrap();
    writeln!(
        report,
        "| Logical CPUs | {} |",
        std::thread::available_parallelism().map_or(1, usize::from)
    )
    .unwrap();
    writeln!(
        report,
        "| Rust | {} |\n",
        markdown_cell(&command_output("rustc", &["--version"]))
    )
    .unwrap();

    writeln!(report, "## Backend availability\n").unwrap();
    writeln!(report, "| Backend | Status | Workload |\n|---|---|---|").unwrap();
    writeln!(report, "| Sonium blocked LU | available | dense |").unwrap();
    writeln!(
        report,
        "| Apple Accelerate | {} | dense |",
        if cfg!(has_accelerate) {
            "available"
        } else {
            "unavailable on this platform"
        }
    )
    .unwrap();
    writeln!(report, "| Sonium sparse simplicial | available | sparse |").unwrap();
    writeln!(report, "| Sonium sparse supernodal | available | sparse |").unwrap();
    writeln!(
        report,
        "| SuperLU | {} | sparse |",
        if cfg!(has_superlu) {
            option_env!("SONIUM_SUPERLU_VERSION")
                .map(|version| format!("available ({version})"))
                .unwrap_or_else(|| "available".into())
        } else {
            "unavailable (pkg-config could not find superlu)".into()
        }
    )
    .unwrap();
    writeln!(
        report,
        "| CUDA | adapter pending | dense and sparse |\n\
         | ROCm | adapter pending | dense and sparse |\n"
    )
    .unwrap();

    writeln!(report, "## Dense LU: factorization and solve\n").unwrap();
    writeln!(
        report,
        "| N | Sonium blocked (ms) | Sonium factor (ms) | Sonium solve (ms) | Accelerate (ms) | Sonium / Accelerate | Sonium residual | Accelerate residual | Sonium factor nnz | Sonium factor bytes | Modeled peak bytes | Peak RSS bytes |\n\
         |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    .unwrap();
    for row in dense {
        writeln!(
            report,
            "| {} | {:.3} | {:.3} | {:.3} | {} | {} | {:.2e} | {} | {} | {} | {} | {} |",
            row.n,
            row.sonium_ms,
            row.sonium_factor_ms,
            row.sonium_solve_ms,
            optional_ms(row.accelerate_ms),
            optional_ratio(row.accelerate_ms.map(|reference| row.sonium_ms / reference)),
            row.sonium_residual,
            optional_scientific(row.accelerate_residual),
            row.factor_nonzeros,
            row.factor_memory_bytes,
            row.modeled_peak_memory_bytes,
            optional_bytes(row.peak_rss_bytes)
        )
        .unwrap();
    }
    writeln!(report).unwrap();

    writeln!(
        report,
        "## Sparse LU: 2D nonsymmetric convection-diffusion\n"
    )
    .unwrap();
    writeln!(
        report,
        "| Grid | N | NNZ | Simplicial (ms) | Simplicial factor (ms) | Simplicial solve (ms) | Supernodal (ms) | Supernodal factor (ms) | Supernodal solve (ms) | Reused symbolic (ms) | Reused factor (ms) | Reused solve (ms) | SuperLU (ms) | Simplicial / SuperLU | Supernodal / SuperLU | Simplicial factor nnz | Supernodal factor nnz | Reused factor nnz | Simplicial factor bytes | Supernodal factor bytes | Reused factor bytes | Modeled peak bytes | Peak RSS bytes | Max residual |\n\
         |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    .unwrap();
    for row in sparse {
        writeln!(
            report,
            "| {0}×{0} | {1} | {2} | {3:.3} | {4:.3} | {5:.3} | {6:.3} | {7:.3} | {8:.3} | {9:.3} | {10:.3} | {11:.3} | {12} | {13} | {14} | {15} | {16} | {17} | {18} | {19} | {20} | {21} | {22} | {23:.2e} |",
            row.grid,
            row.n,
            row.nnz,
            row.simplicial_ms,
            row.simplicial_factor_ms,
            row.simplicial_solve_ms,
            row.supernodal_ms,
            row.supernodal_factor_ms,
            row.supernodal_solve_ms,
            row.supernodal_reused_ms,
            row.supernodal_reused_factor_ms,
            row.supernodal_reused_solve_ms,
            optional_ms(row.superlu_ms),
            optional_ratio(
                row.superlu_ms
                    .map(|reference| row.simplicial_ms / reference)
            ),
            optional_ratio(
                row.superlu_ms
                    .map(|reference| row.supernodal_ms / reference)
            ),
            row.simplicial_factor_nonzeros,
            row.supernodal_factor_nonzeros,
            row.supernodal_reused_factor_nonzeros,
            row.simplicial_factor_memory_bytes,
            row.supernodal_factor_memory_bytes,
            row.supernodal_reused_factor_memory_bytes,
            row.modeled_peak_memory_bytes,
            optional_bytes(row.peak_rss_bytes),
            row.max_residual
        )
        .unwrap();
    }
    writeln!(report).unwrap();

    writeln!(report, "## Methodology\n").unwrap();
    writeln!(
        report,
        "- Times are medians of {} measured runs after one warm-up run.",
        config.repeats
    )
    .unwrap();
    writeln!(
        report,
        "- Each measured operation includes one factorization and one right-hand-side solve; \
         the factor/solve columns split the two stages (solve reuses the timed factorization's factor)."
    )
    .unwrap();
    writeln!(
        report,
        "- The reused-symbolic column excludes Sonium ordering and symbolic analysis; all other sparse columns are one-shot."
    )
    .unwrap();
    writeln!(
        report,
        "- Factor nnz is the stored numeric factor count; factor bytes use scalar width and exclude allocator overhead."
    )
    .unwrap();
    writeln!(
        report,
        "- Modeled peak bytes include the input matrix, largest reported factor, and three scalar work vectors; this is an explicit lower-bound model, not OS RSS."
    )
    .unwrap();
    writeln!(
        report,
        "- Peak RSS bytes is the process high-water mark (`VmHWM` on Linux, `n/a` elsewhere), sampled after each size; it covers the harness, allocator overhead, and all backends."
    )
    .unwrap();
    writeln!(
        report,
        "- Every backend must pass a relative infinity-norm residual gate of `{RESIDUAL_TOLERANCE:.0e}` before timing."
    )
    .unwrap();
    writeln!(
        report,
        "- Ratios above 1.0 mean Sonium is slower than the platform reference; below 1.0 means faster."
    )
    .unwrap();
    writeln!(
        report,
        "- Regenerate with `scripts/direct_solver_perf.sh`. Override sizes with `DIRECT_PERF_DENSE_SIZES`, `DIRECT_PERF_SPARSE_GRIDS`, and `DIRECT_PERF_REPEATS`."
    )
    .unwrap();
    report
}

fn optional_ms(value: Option<f64>) -> String {
    value.map_or_else(|| "—".into(), |value| format!("{value:.3}"))
}

fn optional_ratio(value: Option<f64>) -> String {
    value.map_or_else(|| "—".into(), |value| format!("{value:.2}×"))
}

fn optional_scientific(value: Option<f64>) -> String {
    value.map_or_else(|| "—".into(), |value| format!("{value:.2e}"))
}

fn optional_bytes(value: Option<u64>) -> String {
    value.map_or_else(|| "n/a".into(), |value| value.to_string())
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn cpu_name() -> String {
    if cfg!(target_os = "macos") {
        let cpu = command_output("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"]);
        if cpu == "unknown" {
            mac_hardware_field("Chip").unwrap_or(cpu)
        } else {
            cpu
        }
    } else {
        command_output(
            "sh",
            &["-c", "lscpu | sed -n 's/^Model name:[[:space:]]*//p'"],
        )
    }
}

fn memory_size() -> String {
    let bytes = if cfg!(target_os = "macos") {
        command_output("/usr/sbin/sysctl", &["-n", "hw.memsize"])
            .parse::<u64>()
            .ok()
    } else {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|contents| {
                contents
                    .lines()
                    .find(|line| line.starts_with("MemTotal:"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|kilobytes| kilobytes.parse::<u64>().ok())
                    .map(|kilobytes| kilobytes * 1024)
            })
    };
    bytes.map_or_else(
        || {
            if cfg!(target_os = "macos") {
                mac_hardware_field("Memory").unwrap_or_else(|| "unknown".into())
            } else {
                "unknown".into()
            }
        },
        |bytes| format!("{:.1} GiB", bytes as f64 / 1024_f64.powi(3)),
    )
}

fn mac_hardware_field(label: &str) -> Option<String> {
    let output = Command::new("/usr/sbin/system_profiler")
        .arg("SPHardwareDataType")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let prefix = format!("{label}:");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix(&prefix))
        .map(str::trim)
        .map(str::to_string)
}

fn git_revision() -> String {
    let mut revision = command_output("git", &["rev-parse", "--short", "HEAD"]);
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .is_ok_and(|output| output.status.success() && !output.stdout.is_empty());
    if dirty {
        revision.push_str("-dirty");
    }
    revision
}

fn markdown_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

#[cfg(has_accelerate)]
mod accelerate {
    use sonium_solvers::dense::DenseMatrix;

    unsafe extern "C" {
        fn dgetrf_(
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            pivots: *mut i32,
            info: *mut i32,
        );
        fn dgetrs_(
            trans: *const u8,
            n: *const i32,
            nrhs: *const i32,
            a: *const f64,
            lda: *const i32,
            pivots: *const i32,
            b: *mut f64,
            ldb: *const i32,
            info: *mut i32,
        );
    }

    pub fn solve(matrix: &DenseMatrix<f64>, rhs: &[f64]) -> Result<Vec<f64>, String> {
        let n = matrix.nrows();
        let order = i32::try_from(n).map_err(|_| "matrix exceeds LAPACK i32 ABI")?;
        let mut column_major = vec![0.0; n * n];
        for row in 0..n {
            for column in 0..n {
                column_major[column * n + row] = matrix[[row, column]];
            }
        }
        let mut pivots = vec![0_i32; n];
        let mut solution = rhs.to_vec();
        let mut info = 0_i32;
        unsafe {
            dgetrf_(
                &order,
                &order,
                column_major.as_mut_ptr(),
                &order,
                pivots.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("Accelerate dgetrf failed with info={info}"));
        }
        let trans = b'N';
        let one = 1_i32;
        unsafe {
            dgetrs_(
                &trans,
                &order,
                &one,
                column_major.as_ptr(),
                &order,
                pivots.as_ptr(),
                solution.as_mut_ptr(),
                &order,
                &mut info,
            );
        }
        if info == 0 {
            Ok(solution)
        } else {
            Err(format!("Accelerate dgetrs failed with info={info}"))
        }
    }
}

#[cfg(has_superlu)]
mod superlu {
    use sonium_solvers::sparse::CsrMatrix;
    use std::ffi::c_void;
    use std::ptr::NonNull;

    unsafe extern "C" {
        fn sonium_superlu_prepare(
            n: i32,
            nnz: i32,
            row_ptrs: *const i32,
            column_indices: *const i32,
            values: *const f64,
        ) -> *mut c_void;
        fn sonium_superlu_destroy(matrix: *mut c_void);
        fn sonium_superlu_solve(
            matrix: *const c_void,
            rhs: *const f64,
            solution: *mut f64,
            factor_nonzeros: *mut i64,
        ) -> i32;
    }

    pub struct PreparedMatrix {
        matrix: NonNull<c_void>,
        n: usize,
    }

    impl PreparedMatrix {
        pub fn new(matrix: &CsrMatrix<f64>) -> Result<Self, String> {
            let row_ptrs: Vec<i32> = matrix
                .row_ptrs
                .iter()
                .map(|&value| i32::try_from(value).map_err(|_| "SuperLU index exceeds i32 ABI"))
                .collect::<Result<_, _>>()?;
            let column_indices: Vec<i32> = matrix
                .col_indices
                .iter()
                .map(|&value| i32::try_from(value).map_err(|_| "SuperLU index exceeds i32 ABI"))
                .collect::<Result<_, _>>()?;
            let n = i32::try_from(matrix.num_rows).map_err(|_| "SuperLU order exceeds i32 ABI")?;
            let nnz = i32::try_from(matrix.nnz()).map_err(|_| "SuperLU NNZ exceeds i32 ABI")?;
            let prepared = unsafe {
                sonium_superlu_prepare(
                    n,
                    nnz,
                    row_ptrs.as_ptr(),
                    column_indices.as_ptr(),
                    matrix.values.as_ptr(),
                )
            };
            Ok(Self {
                matrix: NonNull::new(prepared)
                    .ok_or_else(|| "SuperLU matrix preparation ran out of memory".to_string())?,
                n: matrix.num_rows,
            })
        }

        pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, String> {
            if rhs.len() != self.n {
                return Err(format!(
                    "SuperLU RHS length mismatch: expected {}, got {}",
                    self.n,
                    rhs.len()
                ));
            }
            let mut solution = vec![0.0; rhs.len()];
            let mut factor_nonzeros = 0_i64;
            let info = unsafe {
                sonium_superlu_solve(
                    self.matrix.as_ptr(),
                    rhs.as_ptr(),
                    solution.as_mut_ptr(),
                    &mut factor_nonzeros,
                )
            };
            if info == 0 {
                std::hint::black_box(factor_nonzeros);
                Ok(solution)
            } else {
                Err(format!("SuperLU dgssv failed with info={info}"))
            }
        }
    }

    impl Drop for PreparedMatrix {
        fn drop(&mut self) {
            unsafe {
                sonium_superlu_destroy(self.matrix.as_ptr());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_size_lists() {
        assert_eq!(parse_sizes("8, 12,20").unwrap(), vec![8, 12, 20]);
        assert!(parse_sizes("8,0").is_err());
        assert!(parse_sizes("").is_err());
    }

    #[test]
    fn median_handles_even_and_odd_samples() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn generated_report_keeps_future_backend_slots() {
        let config = Config {
            output: "perf/PERF.md".into(),
            dense_sizes: vec![],
            sparse_grids: vec![],
            repeats: 3,
        };
        let report = render_report(&config, &[], &[]);
        assert!(report.contains("| CUDA | adapter pending |"));
        assert!(report.contains("| ROCm | adapter pending |"));
        assert!(report.contains("Sonium factor nnz"));
        assert!(report.contains("Modeled peak bytes"));
        assert!(report.contains("scripts/direct_solver_perf.sh"));
    }

    #[test]
    fn generated_report_contains_split_stage_and_rss_columns() {
        let config = Config {
            output: "perf/PERF.md".into(),
            dense_sizes: vec![],
            sparse_grids: vec![],
            repeats: 3,
        };
        let dense = vec![DenseResult {
            n: 4,
            sonium_ms: 0.1,
            sonium_factor_ms: 0.08,
            sonium_solve_ms: 0.02,
            accelerate_ms: None,
            sonium_residual: 1e-16,
            accelerate_residual: None,
            factor_nonzeros: 16,
            factor_memory_bytes: 128,
            modeled_peak_memory_bytes: 352,
            peak_rss_bytes: Some(123456),
        }];
        let report = render_report(&config, &dense, &[]);
        assert!(report.contains("Sonium factor (ms)"));
        assert!(report.contains("Sonium solve (ms)"));
        assert!(report.contains("Peak RSS bytes"));
        assert!(report.contains("| 4 | 0.100 | 0.080 | 0.020 |"));
        assert!(report.contains("| 123456 |"));
    }

    #[test]
    fn modeled_memory_metrics_include_factor_and_work_vectors() {
        assert_eq!(dense_modeled_peak_memory_bytes(4, 128), 352);
        assert_eq!(sparse_modeled_peak_memory_bytes(4, 12, 144), 472);
    }

    #[test]
    fn modeled_memory_metrics_saturate_on_overflow() {
        assert_eq!(
            dense_modeled_peak_memory_bytes(usize::MAX, usize::MAX),
            usize::MAX
        );
        assert_eq!(
            sparse_modeled_peak_memory_bytes(usize::MAX, usize::MAX, usize::MAX),
            usize::MAX
        );
    }
}
