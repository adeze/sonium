//! Native Linux/x86_64 FEM4 evidence.
//!
//! This producer intentionally uses only the portable FEM public API.  It
//! exercises the H1 space, affine reference kernels, generic assembly, and
//! checked Dirichlet elimination on the four supported low-order cell
//! families.  The analytical manufactured field and constant-field
//! invariants are independent references; no aarch64 result is synthesized.
//!
//! The test is run by `just test-fem-native-x86` with
//! `SONIUM_FEM_DETERMINISTIC=1` and writes a versioned JSON artifact under
//! `target/sonium-unix-fem4`.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use serde_json::{Value, json};
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::{FemSystem, try_apply_homogeneous_dirichlet_dofs};
use sonium_fem::forms::{LaplacianForm, MassForm};
use sonium_fem::mesh::{ElementType, Mesh};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};

const SCHEMA: &str = "sonium-fem-native-x86-evidence/v1";
const DETERMINISTIC_ENV: &str = "SONIUM_FEM_DETERMINISTIC";

#[derive(Clone, Copy, Debug)]
enum CellFamily {
    Triangle2d,
    Quadrilateral2d,
    Tetrahedron3d,
    Hexahedron3d,
}

impl CellFamily {
    const ALL: [Self; 4] = [
        Self::Triangle2d,
        Self::Quadrilateral2d,
        Self::Tetrahedron3d,
        Self::Hexahedron3d,
    ];

    fn mesh(self, refinement: usize) -> Mesh {
        match self {
            Self::Triangle2d => sonium_fem::mesh::unit_square_triangles(refinement),
            Self::Quadrilateral2d => sonium_fem::mesh::unit_square_quads(refinement),
            Self::Tetrahedron3d => sonium_fem::mesh::unit_cube_tetrahedra(refinement),
            Self::Hexahedron3d => sonium_fem::mesh::unit_cube_hexahedra(refinement),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Triangle2d => "triangle-2d",
            Self::Quadrilateral2d => "quadrilateral-2d",
            Self::Tetrahedron3d => "tetrahedron-3d",
            Self::Hexahedron3d => "hexahedron-3d",
        }
    }

    fn element_type(self) -> ElementType {
        match self {
            Self::Triangle2d => ElementType::Triangle,
            Self::Quadrilateral2d => ElementType::Quadrilateral,
            Self::Tetrahedron3d => ElementType::Tetrahedron,
            Self::Hexahedron3d => ElementType::Hexahedron,
        }
    }

    fn dimension(self) -> usize {
        match self {
            Self::Triangle2d | Self::Quadrilateral2d => 2,
            Self::Tetrahedron3d | Self::Hexahedron3d => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct InterpolationError {
    l2: f64,
    gradient: f64,
}

/// A smooth cubic reference field.  P2/Q2 interpolation therefore has a
/// nonzero error while retaining an exact value and gradient at every point.
fn manufactured(point: [f64; 3]) -> (f64, [f64; 3]) {
    let [x, y, z] = point;
    let value = x.powi(3) + 0.5 * y.powi(3) + 0.25 * z.powi(3) + 0.2 * x * y * z;
    let gradient = [
        3.0 * x * x + 0.2 * y * z,
        1.5 * y * y + 0.2 * x * z,
        0.75 * z * z + 0.2 * x * y,
    ];
    (value, gradient)
}

fn interpolation_error(family: CellFamily, refinement: usize) -> InterpolationError {
    let mesh = family.mesh(refinement);
    let space = H1Space::new(PolynomialDegree::P2);
    space
        .validate(&mesh)
        .unwrap_or_else(|error| panic!("{} space validation failed: {error}", family.name()));
    let coordinates = space
        .dof_coordinates(&mesh)
        .unwrap_or_else(|| panic!("{} must expose P2 coordinates", family.name()));
    let coefficients: Vec<f64> = coordinates
        .iter()
        .map(|&point| manufactured(point).0)
        .collect();

    let mut l2_squared = 0.0;
    let mut gradient_squared = 0.0;
    for element_idx in 0..mesh.num_elements() {
        let element_type = mesh.elements[element_idx].element_type;
        let quadrature = QuadratureRule::new(element_type, 7);
        let local_dofs = space.local_to_global(&mesh, element_idx);
        for point in quadrature.iter() {
            let tabulation =
                space.tabulate(&mesh, element_idx, point.xi(), point.eta(), point.zeta());
            let mut value = 0.0;
            let mut gradient = [0.0; 3];
            for (local, &global) in local_dofs.iter().enumerate() {
                value += coefficients[global] * tabulation.value(local);
                for (direction, component) in
                    gradient.iter_mut().enumerate().take(tabulation.spatial_dim)
                {
                    *component += coefficients[global] * tabulation.grad(local, direction);
                }
            }
            let (exact_value, exact_gradient) = manufactured(tabulation.physical_coords);
            let weight = point.weight * tabulation.det_j;
            l2_squared += weight * (value - exact_value).powi(2);
            gradient_squared += weight
                * gradient
                    .iter()
                    .zip(exact_gradient)
                    .take(tabulation.spatial_dim)
                    .map(|(&actual, expected)| (actual - expected).powi(2))
                    .sum::<f64>();
        }
    }

    InterpolationError {
        l2: l2_squared.sqrt(),
        gradient: gradient_squared.sqrt(),
    }
}

fn aggregate_triplets(matrix: &TripletMatrix<f64>) -> Vec<((usize, usize), f64)> {
    let mut entries = std::collections::BTreeMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *entries.entry((row, column)).or_insert(0.0) += value;
    }
    entries.into_iter().collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct AssemblyReport {
    element_matrix_triplets: usize,
    constant_residual: f64,
    assembled_matrix_entries: usize,
    symmetry_error: f64,
    mass_error: f64,
    constrained_dofs: usize,
    bc_contract_error: f64,
}

fn assembly_and_boundary_report(family: CellFamily) -> AssemblyReport {
    let mesh = family.mesh(1);
    let space = H1Space::new(PolynomialDegree::P2);
    let stiffness = assemble_bilinear_checked(&LaplacianForm, &space, &space, &mesh, 7)
        .unwrap_or_else(|error| panic!("{} stiffness assembly failed: {error}", family.name()));
    let mass = assemble_bilinear_checked(&MassForm, &space, &space, &mesh, 7)
        .unwrap_or_else(|error| panic!("{} mass assembly failed: {error}", family.name()));
    let element_matrix_triplets = stiffness.values.len();
    assert!(
        stiffness.values.iter().all(|value| value.is_finite()),
        "{} produced a non-finite element matrix entry",
        family.name()
    );
    let stiffness_entries = aggregate_triplets(&stiffness);
    let mass_entries = aggregate_triplets(&mass);
    let n = space.num_global_dofs(&mesh);

    let mut symmetry_error: f64 = 0.0;
    for &((row, column), value) in &stiffness_entries {
        let transpose = stiffness_entries
            .binary_search_by_key(&(column, row), |(index, _)| *index)
            .ok()
            .map(|index| stiffness_entries[index].1)
            .unwrap_or(0.0);
        symmetry_error = symmetry_error.max((value - transpose).abs());
    }
    let mass_total: f64 = mass_entries.iter().map(|(_, value)| value).sum();

    // Apply all boundary markers in one checked transaction so shared corner
    // DOFs are de-duplicated.  This is the same public BC path used by FEM
    // applications, without involving a solver backend.
    let markers: BTreeSet<i32> = mesh
        .boundaries
        .iter()
        .map(|boundary| boundary.marker)
        .collect();
    assert!(
        !markers.is_empty(),
        "{} has no boundary markers",
        family.name()
    );
    let mut boundary_dofs = Vec::new();
    for marker in markers {
        boundary_dofs.extend(space.boundary_dofs(&mesh, marker));
    }
    boundary_dofs.sort_unstable();
    boundary_dofs.dedup();
    let mut system = FemSystem::new(stiffness, vec![0.0; n]);
    // Use the checked true-residual API against the constant reference field,
    // before BC elimination changes the operator.  For a Laplacian this is a
    // conservation/null-space residual and is independent of any solver.
    let constant_residual = system
        .residual_report(&vec![1.0; n], 0.0, 0.0)
        .unwrap_or_else(|error| panic!("{} residual report failed: {error}", family.name()))
        .absolute_residual;
    try_apply_homogeneous_dirichlet_dofs(&mut system, &boundary_dofs)
        .unwrap_or_else(|error| panic!("{} checked BC failed: {error}", family.name()));
    assert!(
        !system.dirichlet_dofs.is_empty(),
        "{} has no constrained DOFs",
        family.name()
    );

    let mut bc_contract_error: f64 = 0.0;
    for ((&row, &column), &value) in system
        .matrix
        .rows
        .iter()
        .zip(&system.matrix.cols)
        .zip(&system.matrix.values)
    {
        let constrained_row = system.dirichlet_dofs.contains(&row);
        let constrained_column = system.dirichlet_dofs.contains(&column);
        if constrained_row || constrained_column {
            let expected = if constrained_row && row == column {
                1.0
            } else {
                0.0
            };
            bc_contract_error = bc_contract_error.max((value - expected).abs());
        }
    }
    for &dof in &system.dirichlet_dofs {
        bc_contract_error = bc_contract_error.max(system.rhs[dof].abs());
    }

    AssemblyReport {
        element_matrix_triplets,
        constant_residual,
        assembled_matrix_entries: stiffness_entries.len(),
        symmetry_error,
        mass_error: (mass_total - 1.0).abs(),
        constrained_dofs: system.dirichlet_dofs.len(),
        bc_contract_error,
    }
}

fn refinement_rate(coarse: f64, fine: f64) -> f64 {
    assert!(coarse.is_finite() && fine.is_finite() && coarse > fine && fine > 0.0);
    (coarse / fine).log2()
}

fn read_proc_value(key: &str) -> Option<String> {
    let contents = fs::read_to_string("/proc/cpuinfo").ok()?;
    contents.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name.trim() == key).then(|| value.trim().to_owned())
    })
}

fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|version| version.trim().to_owned())
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| "unavailable".to_owned())
}

#[cfg(target_arch = "x86_64")]
fn runtime_features() -> Value {
    json!({
        "avx2": std::is_x86_feature_detected!("avx2"),
        "fma": std::is_x86_feature_detected!("fma"),
        "avx512f": std::is_x86_feature_detected!("avx512f"),
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn runtime_features() -> Value {
    json!({ "avx2": false, "fma": false, "avx512f": false })
}

fn compile_features() -> Value {
    json!({
        "avx2": cfg!(target_feature = "avx2"),
        "fma": cfg!(target_feature = "fma"),
        "avx512f": cfg!(target_feature = "avx512f"),
    })
}

fn output_path() -> PathBuf {
    let directory = std::env::var_os("SONIUM_FEM_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/sonium-unix-fem4"));
    directory.join("native-x86_64-fem4-v1.json")
}

#[test]
fn native_x86_fem4_reference_evidence_v1() {
    // Opt-in evidence producer: skip under the general suite, run only via
    // the dedicated recipe that sets the deterministic-mode flag.
    if std::env::var(DETERMINISTIC_ENV).as_deref() != Ok("1") {
        eprintln!("skipping native FEM4 evidence: set SONIUM_FEM_DETERMINISTIC=1 to produce it");
        return;
    }
    assert_eq!(
        std::env::consts::OS,
        "linux",
        "FEM4 producer requires Linux"
    );
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "FEM4 producer requires x86_64"
    );

    let mut cases = Vec::new();
    for family in CellFamily::ALL {
        let coarse = interpolation_error(family, 1);
        let fine = interpolation_error(family, 2);
        let assembly = assembly_and_boundary_report(family);
        let l2_rate = refinement_rate(coarse.l2, fine.l2);
        let gradient_rate = refinement_rate(coarse.gradient, fine.gradient);

        // These are deliberately conservative independent-reference budgets;
        // the existing full convergence tests carry the tighter asymptotic
        // thresholds.  A failed gate prevents an artifact from being emitted.
        assert!(
            l2_rate > 2.0,
            "{} L2 rate {l2_rate:.3} is too low",
            family.name()
        );
        assert!(
            gradient_rate > 1.0,
            "{} gradient rate {gradient_rate:.3} is too low",
            family.name()
        );
        assert!(assembly.constant_residual < 2.0e-11);
        assert!(assembly.symmetry_error < 2.0e-12);
        assert!(assembly.mass_error < 2.0e-11);
        assert!(assembly.bc_contract_error == 0.0);

        let mesh = family.mesh(1);
        let space = H1Space::new(PolynomialDegree::P2);
        cases.push(json!({
            "cell_family": family.name(),
            "element_type": format!("{:?}", family.element_type()),
            "dimension": family.dimension(),
            "topology": {
                "nodes": mesh.num_nodes(),
                "elements": mesh.num_elements(),
                "global_dofs": space.num_global_dofs(&mesh),
                "boundary_markers": mesh.boundaries.iter().map(|boundary| boundary.marker).collect::<BTreeSet<_>>(),
            },
            "interpolation": {
                "reference": "analytic-cubic-value-and-gradient",
                "coarse_l2": coarse.l2,
                "fine_l2": fine.l2,
                "l2_rate": l2_rate,
                "coarse_gradient": coarse.gradient,
                "fine_gradient": fine.gradient,
                "gradient_rate": gradient_rate,
                "budgets": {"l2_rate_min": 2.0, "gradient_rate_min": 1.0},
            },
            "assembly": {
                "element_matrix_triplets": assembly.element_matrix_triplets,
                "assembled_matrix_entries": assembly.assembled_matrix_entries,
                "constant_field_residual_l2": assembly.constant_residual,
                "symmetry_max_abs": assembly.symmetry_error,
                "mass_domain_abs_error": assembly.mass_error,
                "budgets": {"constant_field_residual_l2_max": 2.0e-11, "symmetry_max_abs": 2.0e-12, "mass_domain_abs_error": 2.0e-11},
            },
            "boundary_conditions": {
                "kind": "homogeneous-dirichlet",
                "constrained_dofs": assembly.constrained_dofs,
                "destination_preserved_contract_max_abs": assembly.bc_contract_error,
                "budget": 0.0,
            },
            "status": "pass",
        }));
    }

    let artifact = json!({
        "schema": SCHEMA,
        "artifact_version": 1,
        "status": "pass",
        "scope": "native-linux-x86_64",
        "target": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "cpu_model": read_proc_value("model name").unwrap_or_else(|| "unavailable".to_owned()),
            "microcode": read_proc_value("microcode").unwrap_or_else(|| "unavailable".to_owned()),
            "compiler": rustc_version(),
            "compile_target_features": compile_features(),
            "runtime_features": runtime_features(),
        },
        "deterministic_mode": {
            "requested": true,
            "environment": DETERMINISTIC_ENV,
            "rayon_threads": std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_owned()),
        },
        "dispatch": {
            "selected_path": "portable-rust-reference",
            "simd_kernel": "not-used-by-sonium-fem-core",
            "vendor_library": "none-linked",
            "provenance": "public-fem-space-assembly-boundary-APIs",
        },
        "cross_architecture": {
            "status": "waived",
            "comparison": "not-run",
            "reason": "User-authorized Arm64 waiver; no aarch64 result is fabricated or treated as a pass.",
        },
        "reference_gates": [
            "analytic cubic interpolation value/gradient",
            "constant-field stiffness residual and unit-domain mass",
            "symmetric assembled operator",
            "checked homogeneous Dirichlet row/column elimination",
        ],
        "cases": cases,
    });

    let path = output_path();
    fs::create_dir_all(path.parent().expect("artifact has a parent")).expect("create evidence dir");
    let encoded = serde_json::to_vec_pretty(&artifact).expect("serialize FEM4 artifact");
    fs::write(&path, encoded).expect("write FEM4 artifact");
    eprintln!("native FEM4 evidence: {}", path.display());
}
