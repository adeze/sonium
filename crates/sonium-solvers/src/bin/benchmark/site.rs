//! Static website generator for benchmark results.
//!
//! Reads JSON result files from a directory and produces a single `index.html`
//! with Chart.js-based interactive charts.

use std::path::Path;

use crate::results::BenchmarkReport;

/// Generate a static benchmark website from JSON result files.
pub fn generate_site(json_dir: &Path, output_dir: &Path) {
    // Read all JSON files
    let mut reports: Vec<BenchmarkReport> = Vec::new();

    if json_dir.is_file() {
        // Single file
        let content = std::fs::read_to_string(json_dir).expect("Failed to read JSON file");
        let report: BenchmarkReport = serde_json::from_str(&content).expect("Failed to parse JSON");
        reports.push(report);
    } else {
        // Directory of files
        for entry in std::fs::read_dir(json_dir).expect("Failed to read directory") {
            let entry = entry.expect("Failed to read dir entry");
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                let content = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
                match serde_json::from_str::<BenchmarkReport>(&content) {
                    Ok(report) => reports.push(report),
                    Err(e) => eprintln!("Skipping {}: {e}", path.display()),
                }
            }
        }
    }

    if reports.is_empty() {
        eprintln!("No valid JSON result files found in {}", json_dir.display());
        return;
    }

    // Serialize all reports as a JSON array for embedding
    let data_json = serde_json::to_string(&reports).expect("Failed to serialize reports");

    // Generate HTML
    let html = TEMPLATE.replace("/*__DATA__*/", &format!("const REPORTS = {data_json};"));

    // Write output
    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");
    let index_path = output_dir.join("index.html");
    std::fs::write(&index_path, html).expect("Failed to write index.html");
    eprintln!("Wrote {}", index_path.display());
}

const TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sonium Solvers — Benchmark Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 2rem; }
  h1 { color: #58a6ff; margin-bottom: 0.5rem; }
  h2 { color: #79c0ff; margin: 2rem 0 1rem; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; }
  h3 { color: #d2a8ff; margin: 1.5rem 0 0.5rem; }
  .meta { color: #8b949e; margin-bottom: 2rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 2rem; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; }
  canvas { max-height: 400px; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: 0.85rem; }
  th, td { padding: 0.4rem 0.8rem; text-align: right; border-bottom: 1px solid #21262d; }
  th { color: #58a6ff; font-weight: 600; position: sticky; top: 0; background: #161b22; }
  td:first-child, th:first-child { text-align: left; }
  .pass { color: #3fb950; } .fail { color: #f85149; } .warn { color: #d29922; }
  .filter-bar { margin: 1rem 0; }
  .filter-bar select { background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
                        padding: 0.4rem; border-radius: 4px; margin-right: 0.5rem; }
</style>
</head>
<body>
<h1>Sonium Solvers — Benchmark Results</h1>

<script>
/*__DATA__*/

// If no data embedded, show placeholder
if (typeof REPORTS === 'undefined') {
  document.body.innerHTML += '<p>No data. Run sonium-benchmark --generate-site to embed results.</p>';
}

document.addEventListener('DOMContentLoaded', () => {
  if (typeof REPORTS === 'undefined' || REPORTS.length === 0) return;

  // Platform info
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.innerHTML = REPORTS.map(r =>
    `<b>${r.platform.name}</b>: ${r.platform.cpu_model} (${r.platform.total_cores} cores) — ` +
    `sonium ${r.sonium_version} — ${r.timestamp.substring(0, 10)}`
  ).join('<br>');
  document.body.appendChild(meta);

  // Collect all sonium runs across platforms
  const allRuns = [];
  REPORTS.forEach(report => {
    report.runs.forEach(run => {
      allRuns.push({ ...run, platform: report.platform.name });
    });
  });

  // Group by solver/precond/matrix for charts
  const soniumRuns = allRuns.filter(r => r.engine === 'sonium');
  const petscRuns = allRuns.filter(r => r.engine === 'petsc');

  // --- Solver Timing Chart ---
  const h2timing = document.createElement('h2');
  h2timing.textContent = 'Solver Timing (ms)';
  document.body.appendChild(h2timing);

  const grid = document.createElement('div');
  grid.className = 'grid';
  document.body.appendChild(grid);

  // Group runs by matrix+size
  const groups = {};
  soniumRuns.forEach(r => {
    const key = `${r.matrix} n=${r.n}`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(r);
  });

  // Create a chart per group (limit to first 8 groups)
  Object.entries(groups).slice(0, 8).forEach(([key, runs]) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<h3>${key}</h3><canvas></canvas>`;
    grid.appendChild(card);

    const labels = runs.map(r => `${r.solver}/${r.preconditioner}`);
    const setupData = runs.map(r => r.setup_ms);
    const solveData = runs.map(r => r.solve_ms);

    new Chart(card.querySelector('canvas'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Setup', data: setupData, backgroundColor: '#238636' },
          { label: 'Solve', data: solveData, backgroundColor: '#1f6feb' },
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#c9d1d9' } } },
        scales: {
          x: { stacked: true, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
          y: { stacked: true, ticks: { color: '#8b949e' }, grid: { color: '#21262d' },
               title: { display: true, text: 'Time (ms)', color: '#8b949e' } }
        }
      }
    });
  });

  // --- Results Table ---
  const h2table = document.createElement('h2');
  h2table.textContent = 'All Results';
  document.body.appendChild(h2table);

  const table = document.createElement('table');
  table.innerHTML = `
    <thead><tr>
      <th>Engine</th><th>Solver</th><th>Precond</th><th>Matrix</th><th>Scalar</th>
      <th>N</th><th>RHS</th><th>Setup (ms)</th><th>Solve (ms)</th><th>Solve/RHS (ms)</th>
      <th>Total (ms)</th><th>Iters</th><th>Iters/RHS</th><th>Residual</th><th>OK</th><th>B/nnz</th>
      <th>SpMV GB/s</th><th>K-SpMV</th><th>K-Dots</th><th>K-Axpy</th><th>Setup %</th><th>Limit</th><th>Reuse</th>
      <th>Break-even RHS</th><th>RHS@4 (ms)</th><th>RHS@16 (ms)</th>
    </tr></thead>
    <tbody></tbody>
  `;
  document.body.appendChild(table);

  const tbody = table.querySelector('tbody');
  allRuns.forEach(r => {
    const cls = r.converged ? 'pass' : 'fail';
    tbody.innerHTML += `<tr>
      <td>${r.engine}</td><td>${r.solver}</td><td>${r.preconditioner}</td>
      <td>${r.matrix}</td><td>${r.scalar}</td><td>${r.n.toLocaleString()}</td>
      <td>${r.rhs_count ?? 1}</td>
      <td>${r.setup_ms.toFixed(2)}</td><td>${r.solve_ms.toFixed(2)}</td>
      <td>${formatOptionalNumber(r.mean_solve_ms_per_rhs, 2)}</td>
      <td>${r.total_ms.toFixed(2)}</td><td>${r.iterations}</td>
      <td>${formatOptionalNumber(r.mean_iterations_per_rhs, 1)}</td>
      <td>${r.residual.toExponential(2)}</td>
      <td class="${cls}">${r.converged ? '✓' : '✗'}</td>
      <td>${formatOptionalNumber(r.bytes_per_nnz, 1)}</td>
      <td>${formatOptionalNumber(r.estimated_spmv_gb_s, 1)}</td>
      <td>${formatOptionalInteger(r.krylov_spmv_count)}</td>
      <td>${formatOptionalInteger(r.krylov_dot_reductions)}</td>
      <td>${formatOptionalInteger(r.krylov_axpy_passes)}</td>
      <td>${formatOptionalPercent(r.setup_fraction)}</td>
      <td>${r.time_limited_by ?? '-'}</td>
      <td>${r.setup_reuse_scope ?? '-'}</td>
      <td>${formatOptionalNumber(r.setup_reuse_break_even_rhs, 1)}</td>
      <td>${formatOptionalNumber(r.amortized_total_ms_per_rhs_4, 2)}</td>
      <td>${formatOptionalNumber(r.amortized_total_ms_per_rhs_16, 2)}</td>
    </tr>`;
  });

  // --- Validation Summary ---
  const hasValidation = REPORTS.some(r => r.validation.total_comparisons > 0);
  if (hasValidation) {
    const h2val = document.createElement('h2');
    h2val.textContent = 'PETSc Validation';
    document.body.appendChild(h2val);

    REPORTS.forEach(report => {
      if (report.validation.total_comparisons === 0) return;
      const v = report.validation;
      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = `
        <h3>${report.platform.name}</h3>
        <p><span class="pass">${v.passed} passed</span> /
           <span class="fail">${v.failed} failed</span> /
           <span class="warn">${v.warnings} warnings</span>
           out of ${v.total_comparisons} comparisons</p>
      `;
      if (v.failures.length > 0) {
        let ftable = '<table><thead><tr><th>Solver</th><th>Precond</th><th>Matrix</th><th>N</th><th>Reason</th></tr></thead><tbody>';
        v.failures.forEach(f => {
          ftable += `<tr><td>${f.solver}</td><td>${f.preconditioner}</td><td>${f.matrix}</td><td>${f.n}</td><td class="fail">${f.reason}</td></tr>`;
        });
        ftable += '</tbody></table>';
        div.innerHTML += ftable;
      }
      document.body.appendChild(div);
    });
  }
});

function formatOptionalNumber(value, digits) {
  return typeof value === 'number' ? value.toFixed(digits) : '-';
}

function formatOptionalInteger(value) {
  return Number.isInteger(value) ? value.toLocaleString() : '-';
}

function formatOptionalPercent(value) {
  return typeof value === 'number' ? `${(value * 100).toFixed(0)}%` : '-';
}
</script>
</body>
</html>"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_displays_solver_perf_columns() {
        assert!(TEMPLATE.contains("<th>B/nnz</th>"));
        assert!(TEMPLATE.contains("formatOptionalNumber(r.estimated_spmv_gb_s, 1)"));
        assert!(TEMPLATE.contains("formatOptionalInteger(r.krylov_spmv_count)"));
        assert!(TEMPLATE.contains("r.time_limited_by ?? '-'"));
        assert!(TEMPLATE.contains("r.rhs_count ?? 1"));
        assert!(TEMPLATE.contains("r.mean_solve_ms_per_rhs"));
        assert!(TEMPLATE.contains("r.setup_reuse_scope ?? '-'"));
        assert!(TEMPLATE.contains("r.amortized_total_ms_per_rhs_16"));
    }
}
