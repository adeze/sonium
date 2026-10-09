//! Required NVIDIA BEM capability and checked-assembly gate.
//!
//! This is intentionally an assembly/product gate, not a claim that CUDA
//! implements every BEM phase.  Unsupported and host-staged cells are emitted
//! in the machine-readable result so a release runner cannot mistake a CPU
//! fallback for device coverage.

use num_complex::Complex64;
use sonium_bem::compute_scattered_field_checked;
use sonium_bem::compute_scattered_field_cuda_checked;
use sonium_bem::constant_panel_power;
use sonium_bem::core::assembly::tbem::CudaTbemAssemblyError;
use sonium_bem::core::assembly::{
    CudaTbemAssemblyWorkspace, build_tbem_system_stabilized_wavenumber_derivative,
    build_tbem_system_stabilized_with_beta_simd, build_tbem_system_with_beta_cuda_checked,
};
use sonium_bem::core::integration::{QUASI_SINGULAR_THRESHOLD, regular_integration_fixed_order};
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::{
    BoundaryCondition, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_bem::core::{BemProblem, BemSolver};
use sonium_compute::{
    Backend, BackendError, BemRegularPairInF64, BemRegularPairOutF64, CapabilityState, CudaBackend,
    CudaFailureInjectionPoint, CudaRuntimeStatsSnapshot, MemoryPool, MemoryPoolConfig, Transpose,
};
use sonium_qa::{
    QaExecutionCounts, QaFailureStage, QaResult, cuda_hardware_provenance, cuda_runtime_evidence,
    required_cuda_backend, write_qa_artifact,
};
use sonium_solvers::dense::DenseMatrix;
use sonium_wave::analytical::{pulsating_sphere_3d, sphere_scattering_3d};
use std::f64::consts::PI;
use std::time::Instant;

const CASE_ID: &str = "bem.cuda-capability-and-assembly-matrix.v1";
const MATRIX_PARITY_TOLERANCE: f64 = 1.0e-3;
const PRODUCT_PARITY_TOLERANCE: f64 = 1.0e-3;
const INNER_PRODUCT_TOLERANCE: f64 = 1.0e-12;
const SOLVE_RESIDUAL_TOLERANCE: f64 = 1.0e-8;
// The analytic CPU derivative uses the higher-order scalar quadrature while
// this CUDA product gate intentionally exercises the fixed four-point CUDA
// panel rule.  The budget therefore includes the documented quadrature
// discretisation delta as well as floating-point error.
const DERIVATIVE_PARITY_TOLERANCE: f64 = 5.0e-3;
const SOURCE_DERIVATIVE_PARITY_TOLERANCE: f64 = 5.0e-6;
const SOURCE_DERIVATIVE_STEP: f64 = 1.0e-6;
const SPHERE_CUDA_FIELD_TOLERANCE: f64 = 1.0e-10;
const SPHERE_MIE_TOLERANCE: f64 = 3.0e-1;
const RADIATION_CUDA_FIELD_TOLERANCE: f64 = 1.0e-10;
const RADIATION_REFERENCE_TOLERANCE: f64 = 3.0e-1;
const RADIATION_POWER_TOLERANCE: f64 = 3.0e-1;

type RigidSphereFieldMetrics = (f64, f64, usize, u64, usize, usize, usize);
type RadiationFieldMetrics = (f64, f64, f64, f64, f64, usize, u64, usize, usize, usize);
type CudaSolveMetrics = (f64, f64, u64, u64, u64, usize, usize);
type Fixture = (Vec<Element>, Vec<[f64; 3]>, PhysicsParams);

fn failed_cuda(error: String, actual_backend: &str, stage: QaFailureStage) -> QaResult {
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend("cuda")
        .actual_backend(actual_backend)
        .backend_failed("cuda", error)
        .failure_stage(stage)
        .hardware_provenance(cuda_hardware_provenance())
}

fn state_name(state: CapabilityState) -> &'static str {
    match state {
        CapabilityState::Native => "native",
        CapabilityState::VendorNative => "vendor_native",
        CapabilityState::HostSharedFallback => "host_shared_fallback",
        CapabilityState::HostStagedFallback => "host_staged_fallback",
        CapabilityState::Unsupported => "unsupported",
    }
}

fn record_scalar_capabilities(
    result: &mut QaResult,
    phase: &str,
    capabilities: sonium_compute::ScalarCapabilities,
) {
    for (scalar, state) in [
        ("f32", capabilities.f32),
        ("f64", capabilities.f64),
        ("c32", capabilities.c32),
        ("c64", capabilities.c64),
    ] {
        result.environment.insert(
            format!("capability.bem.{phase}.{scalar}"),
            state_name(state).into(),
        );
    }
}

fn record_capabilities(mut result: QaResult, backend: &CudaBackend) -> QaResult {
    let capabilities = backend.capabilities();
    // CUDA admission is intentionally explicit about dimensional coverage:
    // this checked seam currently owns only the 3-D Tri3 constant-panel
    // formulation.  A missing 2-D or axisymmetric cell is recorded as
    // unsupported rather than being mistaken for a passed CUDA run.
    for (dimension, state) in [
        ("2d", "unsupported"),
        ("axisymmetric", "unsupported"),
        ("3d", "supported"),
    ] {
        result.environment.insert(
            format!("capability.bem.dimension.{dimension}"),
            state.into(),
        );
    }
    // Keep the element/order boundary explicit as well.  The checked CUDA
    // assembly seam is deliberately Tri3/constant-panel only; these entries
    // prevent a future consumer from inferring Quad or higher-order coverage
    // merely because a generic capability field is present.
    for (cell, state) in [
        ("element.tri3.constant.c64", "native"),
        ("element.tri3.higher_order.c64", "unsupported"),
        ("element.quad.constant.c64", "unsupported"),
        ("element.quad.higher_order.c64", "unsupported"),
    ] {
        result
            .environment
            .insert(format!("capability.bem.{cell}"), state.into());
    }
    result.environment.insert(
        "capability.bem.regular_pair_c32".into(),
        state_name(capabilities.bem.regular_pair_c32).into(),
    );
    result.environment.insert(
        "capability.bem.regular_pair_c64".into(),
        state_name(capabilities.bem.regular_pair_c64).into(),
    );
    result.environment.insert(
        "capability.bem.source_position_x_derivative.c64".into(),
        state_name(backend.bem_source_x_derivative_capability()).into(),
    );
    result.environment.insert(
        "capability.bem.source_position_y_derivative.c64".into(),
        state_name(backend.bem_source_y_derivative_capability()).into(),
    );
    record_scalar_capabilities(&mut result, "near_pair", capabilities.bem.near_pair);
    record_scalar_capabilities(&mut result, "singular_pair", capabilities.bem.singular_pair);
    record_scalar_capabilities(
        &mut result,
        "hypersingular_pair",
        capabilities.bem.hypersingular_pair,
    );
    record_scalar_capabilities(
        &mut result,
        "field_evaluation",
        capabilities.bem.field_evaluation,
    );
    record_scalar_capabilities(&mut result, "hmatrix_setup", capabilities.bem.hmatrix_setup);
    record_scalar_capabilities(&mut result, "hmatrix", capabilities.bem.hmatrix);
    record_scalar_capabilities(&mut result, "fmm_setup", capabilities.bem.fmm_setup);
    record_scalar_capabilities(&mut result, "fmm", capabilities.bem.fmm);
    // Physical-product cells are deliberately separate from kernel
    // capability cells.  A native 3-D pair kernel is not evidence for a
    // solved cylinder, axisymmetric, piston/duct, or transmission product.
    // Keep every unimplemented cell explicit in the JSON matrix so a release
    // consumer cannot promote an unavailable result to a pass.
    for (cell, state) in [
        ("3d_operator", "validated"),
        ("3d_rigid_sphere_mie", "not_run"),
        ("3d_radiation", "not_run"),
        ("3d_mesh_convergence", "not_run"),
        ("3d_transmission", "unsupported"),
        ("2d_cylinder", "unsupported"),
        ("axisymmetric_mie", "unsupported"),
        ("piston", "unsupported"),
        ("duct", "unsupported"),
        ("3d_wavenumber_derivative", "not_run"),
        ("3d_geometry_derivative", "unsupported"),
        ("3d_material_derivative", "unsupported"),
        ("3d_source_derivative", "unsupported"),
        ("3d_source_y_derivative", "not_run"),
    ] {
        result
            .environment
            .insert(format!("physical_gate.bem.{cell}"), state.into());
    }
    result
}

fn relative_matrix_error(lhs: &[Complex64], rhs: &[Complex64]) -> Result<f64, String> {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return Err(format!(
            "matrix product length mismatch: {} != {}",
            lhs.len(),
            rhs.len()
        ));
    }
    let numerator = lhs
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = rhs.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= f64::MIN_POSITIVE {
        return Err("matrix product contains a non-finite or zero reference".into());
    }
    Ok(numerator / denominator)
}

fn dotc(lhs: &[Complex64], rhs: &[Complex64]) -> Result<Complex64, String> {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return Err(format!(
            "inner-product length mismatch: {} != {}",
            lhs.len(),
            rhs.len()
        ));
    }
    let value: Complex64 = lhs
        .iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| lhs.conj() * rhs)
        .sum();
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err("inner product is non-finite".into());
    }
    Ok(value)
}

fn relative_inner_product_error(
    forward: &[Complex64],
    rhs: &[Complex64],
    lhs: &[Complex64],
    adjoint: &[Complex64],
) -> Result<f64, String> {
    let left = dotc(forward, rhs)?;
    let right = dotc(lhs, adjoint)?;
    let denominator = left.norm().max(right.norm()).max(f64::MIN_POSITIVE);
    let error = (left - right).norm() / denominator;
    if error.is_finite() {
        Ok(error)
    } else {
        Err("inner-product error is non-finite".into())
    }
}

fn dense_product(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    probe: &[Complex64],
    mode: ProductMode,
) -> Vec<Complex64> {
    let n = matrix.nrows();
    (0..n)
        .map(|row| {
            (0..n)
                .map(|column| {
                    let entry = match mode {
                        ProductMode::Forward => matrix[[row, column]],
                        ProductMode::Transpose => matrix[[column, row]],
                        ProductMode::Hermitian => matrix[[column, row]].conj(),
                    };
                    entry * probe[column]
                })
                .sum()
        })
        .collect()
}

/// Exercise the CUDA field path on the same exterior rigid-sphere product
/// used by the CPU/Mie validation suite.  The solve remains an independent
/// CPU reference for the boundary trace; only the exterior field integral is
/// dispatched to CUDA here, which keeps this gate focused on accelerator
/// field correctness rather than conflating it with the separate assembly
/// fixture above.
fn rigid_sphere_cuda_field_gate(
    pool: &MemoryPool<CudaBackend>,
) -> Result<RigidSphereFieldMetrics, String> {
    let radius = 0.1_f64;
    let wave_number = 0.35 / radius;
    let speed_of_sound = 343.0_f64;
    let density = 1.21_f64;
    let frequency = wave_number * speed_of_sound / (2.0 * PI);
    let problem = BemProblem::try_rigid_sphere_scattering_custom(
        radius,
        frequency,
        speed_of_sound,
        density,
        4,
        8,
    )
    .map_err(|error| format!("rigid-sphere CUDA gate admission failed: {error:?}"))?;

    let theta = [0.25_f64, 0.7, 1.15, 1.6, 2.05, 2.5, 2.9];
    let evaluation_radius = 1.2_f64;
    let points: Vec<[f64; 3]> = theta
        .iter()
        .map(|&angle| {
            [
                evaluation_radius * angle.sin(),
                0.0,
                evaluation_radius * angle.cos(),
            ]
        })
        .collect();

    let started = Instant::now();
    let solution = BemSolver::new()
        .solve(&problem)
        .map_err(|error| format!("rigid-sphere CPU reference solve failed: {error:?}"))?;
    let cpu_total: Vec<Complex64> = solution
        .evaluate_pressure_field(&points)
        .into_iter()
        .map(|field| field.p_total)
        .collect();

    pool.backend().reset_runtime_stats();
    let cuda_scattered = compute_scattered_field_cuda_checked(
        pool,
        &points,
        &solution.elements,
        &solution.nodes,
        &solution.surface_pressure,
        None,
        &solution.physics,
        solution.basis_order,
    )
    .map_err(|error| format!("rigid-sphere CUDA field evaluation failed: {error}"))?;
    let incident = solution
        .incident_field
        .evaluate_pressure(&points, &solution.physics);
    let cuda_total: Vec<Complex64> = cuda_scattered
        .iter()
        .zip(incident.iter())
        .map(|(scattered, incident)| *scattered + *incident)
        .collect();

    let mie = sphere_scattering_3d(
        solution.physics.wave_number,
        radius,
        40,
        vec![evaluation_radius],
        theta.to_vec(),
    );
    let cuda_cpu_error = relative_matrix_error(&cuda_total, &cpu_total)?;
    let mie_error = relative_matrix_error(&cpu_total, &mie.pressure)?;
    let stats = pool.backend().runtime_stats();
    if stats.kernel_launches == 0
        || stats.transfer.h2d_operations == 0
        || stats.transfer.d2h_operations == 0
    {
        return Err(format!(
            "rigid-sphere CUDA field gate did not prove device work: kernels={}, h2d={}, d2h={}",
            stats.kernel_launches, stats.transfer.h2d_operations, stats.transfer.d2h_operations
        ));
    }
    if !cuda_cpu_error.is_finite() || !mie_error.is_finite() {
        return Err("rigid-sphere CUDA/Mie gate produced a non-finite error".into());
    }
    Ok((
        cuda_cpu_error,
        mie_error,
        solution.elements.len(),
        started.elapsed().as_millis() as u64,
        stats.kernel_launches,
        stats.transfer.h2d_operations,
        stats.transfer.d2h_operations,
    ))
}

/// Exercise the CUDA field path with a prescribed normal-velocity trace and
/// compare the radiated pressure against the independent isotropic pulsating
/// sphere solution.  This is the supported 3-D radiation cell; no incident
/// field is added to the scattered-pressure result.
fn pulsating_sphere_cuda_field_gate(
    pool: &MemoryPool<CudaBackend>,
) -> Result<RadiationFieldMetrics, String> {
    // The power observable is more sensitive to boundary discretisation than
    // the point-field parity check.  Use the same 1,280-panel resolution as
    // the established pulsating-sphere validation so the physical gate does
    // not turn a coarse-mesh error into a CUDA failure.
    pulsating_sphere_cuda_field_gate_at_subdivision(pool, 3, true)
}

fn pulsating_sphere_cuda_field_gate_at_subdivision(
    pool: &MemoryPool<CudaBackend>,
    subdivisions: usize,
    check_power: bool,
) -> Result<RadiationFieldMetrics, String> {
    let radius = 0.1_f64;
    let wave_number = 0.35 / radius;
    let speed_of_sound = 343.0_f64;
    let density = 1.21_f64;
    let velocity_amplitude = 1.0_f64;
    let physics = PhysicsParams::try_new(
        wave_number * speed_of_sound / (2.0 * PI),
        speed_of_sound,
        density,
        false,
    )
    .map_err(|error| format!("pulsating-sphere CUDA gate physics failed: {error}"))?;
    let mesh = generate_icosphere_mesh(radius, subdivisions);
    let mut elements = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.boundary_condition =
            BoundaryCondition::Velocity(vec![Complex64::new(velocity_amplitude, 0.0)]);
        element.dof_addresses = vec![index];
    }
    let beta = physics.burton_miller_beta();
    let started = Instant::now();
    let system =
        build_tbem_system_stabilized_with_beta_simd(&elements, &mesh.nodes, &physics, 1.0, beta);
    let surface_pressure = sonium_solvers::direct::lu_solve(&system.matrix, &system.rhs)
        .map_err(|error| format!("pulsating-sphere CPU reference solve failed: {error}"))?;
    let radii = [0.13_f64, 0.2, 0.35, 0.7];
    let points: Vec<[f64; 3]> = radii.map(|radius| [0.0, 0.0, radius]).to_vec();
    let velocity = vec![Complex64::new(velocity_amplitude, 0.0); elements.len()];
    let (radiation_power, reference_power, power_error) = if check_power {
        let panel_areas: Vec<f64> = elements.iter().map(|element| element.area).collect();
        let radiation_power = constant_panel_power(&panel_areas, &surface_pressure, &velocity)
            .map_err(|error| format!("pulsating-sphere power contract failed: {error}"))?;
        let ka = physics.wave_number * radius;
        let reference_power = 0.5
            * (4.0 * PI * radius * radius)
            * density
            * speed_of_sound
            * velocity_amplitude
            * velocity_amplitude
            * (ka * ka / (ka * ka + 1.0));
        let power_error =
            (radiation_power - reference_power).abs() / reference_power.max(f64::MIN_POSITIVE);
        if !radiation_power.is_finite()
            || radiation_power <= 0.0
            || !power_error.is_finite()
            || power_error > RADIATION_POWER_TOLERANCE
        {
            return Err(format!(
                "pulsating-sphere radiation power {radiation_power:.6e} / reference {reference_power:.6e} has relative error {power_error:.6e}, tolerance {RADIATION_POWER_TOLERANCE:.6e}"
            ));
        }
        (radiation_power, reference_power, power_error)
    } else {
        (0.0, 0.0, 0.0)
    };
    let cpu_scattered = compute_scattered_field_checked(
        &points,
        &elements,
        &mesh.nodes,
        &surface_pressure,
        Some(&velocity),
        &physics,
        sonium_bem::BasisOrder::Constant,
    )
    .map_err(|error| format!("pulsating-sphere CPU field reference failed: {error}"))?;

    pool.backend().reset_runtime_stats();
    let cuda_scattered = compute_scattered_field_cuda_checked(
        pool,
        &points,
        &elements,
        &mesh.nodes,
        &surface_pressure,
        Some(&velocity),
        &physics,
        sonium_bem::BasisOrder::Constant,
    )
    .map_err(|error| format!("pulsating-sphere CUDA field evaluation failed: {error}"))?;
    let reference = pulsating_sphere_3d(
        physics.wave_number,
        radius,
        velocity_amplitude,
        speed_of_sound,
        density,
        radii.to_vec(),
    );
    let cpu_values = cpu_scattered;
    let cuda_cpu_error = relative_matrix_error(&cuda_scattered, &cpu_values)?;
    let reference_error = relative_matrix_error(&cpu_values, &reference.pressure)?;
    let stats = pool.backend().runtime_stats();
    if stats.kernel_launches == 0
        || stats.transfer.h2d_operations == 0
        || stats.transfer.d2h_operations == 0
    {
        return Err(format!(
            "pulsating-sphere CUDA field gate did not prove device work: kernels={}, h2d={}, d2h={}",
            stats.kernel_launches, stats.transfer.h2d_operations, stats.transfer.d2h_operations
        ));
    }
    if !cuda_cpu_error.is_finite() || !reference_error.is_finite() {
        return Err("pulsating-sphere CUDA/reference gate produced a non-finite error".into());
    }
    Ok((
        cuda_cpu_error,
        reference_error,
        radiation_power,
        reference_power,
        power_error,
        elements.len(),
        started.elapsed().as_millis() as u64,
        stats.kernel_launches,
        stats.transfer.h2d_operations,
        stats.transfer.d2h_operations,
    ))
}

fn pulsating_sphere_mesh_convergence_gate(
    pool: &MemoryPool<CudaBackend>,
    fine_reference_error: f64,
) -> Result<(f64, f64, usize, u64), String> {
    let (_, coarse_reference_error, _, _, _, coarse_panel_count, coarse_elapsed, _, _, _) =
        pulsating_sphere_cuda_field_gate_at_subdivision(pool, 1, false)?;
    if !coarse_reference_error.is_finite()
        || !fine_reference_error.is_finite()
        || fine_reference_error >= coarse_reference_error
    {
        return Err(format!(
            "pulsating-sphere mesh refinement did not reduce reference error: coarse={coarse_reference_error:.6e}, fine={fine_reference_error:.6e}"
        ));
    }
    Ok((
        coarse_reference_error,
        coarse_reference_error - fine_reference_error,
        coarse_panel_count,
        coarse_elapsed,
    ))
}

fn fixture() -> Result<Fixture, String> {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 10.0],
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 10.0],
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [1.5, 1.0, 0.0],
    ];
    let element = |connectivity: Vec<usize>, center_z: f64, dof: usize| Element {
        connectivity,
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 3],
        center: [1.0 / 3.0, 1.0 / 3.0, center_z],
        area: 0.5,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: vec![dof],
    };
    let elements = vec![
        element(vec![0, 1, 2], 0.0, 0),
        // Center distance / sqrt(area) ≈ 2.12, deliberately inside the
        // quasi-singular CUDA quadrature route (< 3.0).
        element(vec![6, 7, 8], 0.0, 1),
        element(vec![3, 4, 5], 10.0, 2),
    ];
    let physics = PhysicsParams::try_new(600.0, 343.0, 1.21, false)
        .map_err(|error| format!("fixed CUDA fixture physics is invalid: {error:?}"))?;
    Ok((elements, nodes, physics))
}

fn resident_regular_pairs(elements: &[Element], nodes: &[[f64; 3]]) -> Vec<BemRegularPairInF64> {
    let mut pairs = Vec::new();
    for (source_index, source) in elements.iter().enumerate() {
        if source.property.is_evaluation() {
            continue;
        }
        for (field_index, field) in elements.iter().enumerate() {
            if field.property.is_evaluation() || source_index == field_index {
                continue;
            }
            let dx = field.center[0] - source.center[0];
            let dy = field.center[1] - source.center[1];
            let dz = field.center[2] - source.center[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            let ratio = distance / field.area.sqrt().max(1.0e-30);
            if ratio < QUASI_SINGULAR_THRESHOLD {
                continue;
            }
            if field.connectivity.len() != 3 {
                continue;
            }
            let vertices = field
                .connectivity
                .iter()
                .map(|&index| nodes[index])
                .collect::<Vec<_>>();
            pairs.push(BemRegularPairInF64::from_triangle(
                source.center,
                source.normal,
                vertices[0],
                vertices[1],
                vertices[2],
            ));
        }
    }
    pairs
}

fn cpu_regular_pair_value(
    pair: &BemRegularPairInF64,
    physics: &PhysicsParams,
) -> BemRegularPairOutF64 {
    let coords = DenseMatrix::from_row_major_vec(
        3,
        3,
        vec![
            pair.v0[0], pair.v0[1], pair.v0[2], pair.v1[0], pair.v1[1], pair.v1[2], pair.v2[0],
            pair.v2[1], pair.v2[2],
        ],
    );
    let value = regular_integration_fixed_order(
        &pair.source_point,
        &pair.source_normal,
        &coords,
        ElementType::Tri3,
        physics,
        2,
    );
    BemRegularPairOutF64 {
        g: value.g_integral,
        h: value.dg_dn_integral,
        ht: value.dg_dnx_integral,
        e: value.d2g_dnxdny_integral,
    }
}

fn cpu_source_x_centered_difference(
    pair: &BemRegularPairInF64,
    physics: &PhysicsParams,
    step: f64,
) -> BemRegularPairOutF64 {
    let mut plus = *pair;
    plus.source_point[0] += step;
    let mut minus = *pair;
    minus.source_point[0] -= step;
    let plus = cpu_regular_pair_value(&plus, physics);
    let minus = cpu_regular_pair_value(&minus, physics);
    let central = |plus: Complex64, minus: Complex64| (plus - minus) / (2.0 * step);
    BemRegularPairOutF64 {
        g: central(plus.g, minus.g),
        h: central(plus.h, minus.h),
        ht: central(plus.ht, minus.ht),
        e: central(plus.e, minus.e),
    }
}

fn cpu_source_y_centered_difference(
    pair: &BemRegularPairInF64,
    physics: &PhysicsParams,
    step: f64,
) -> BemRegularPairOutF64 {
    let mut plus = *pair;
    plus.source_point[1] += step;
    let mut minus = *pair;
    minus.source_point[1] -= step;
    let plus = cpu_regular_pair_value(&plus, physics);
    let minus = cpu_regular_pair_value(&minus, physics);
    let central = |plus: Complex64, minus: Complex64| (plus - minus) / (2.0 * step);
    BemRegularPairOutF64 {
        g: central(plus.g, minus.g),
        h: central(plus.h, minus.h),
        ht: central(plus.ht, minus.ht),
        e: central(plus.e, minus.e),
    }
}

fn relative_pair_output_error(
    actual: &[BemRegularPairOutF64],
    expected: &[BemRegularPairOutF64],
) -> Result<f64, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(format!(
            "pair derivative length mismatch: {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    let channels = |value: &BemRegularPairOutF64| [value.g, value.h, value.ht, value.e];
    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for (actual, expected) in actual.iter().zip(expected) {
        for (actual, expected) in channels(actual).into_iter().zip(channels(expected)) {
            numerator += (actual - expected).norm_sqr();
            denominator += expected.norm_sqr();
        }
    }
    let numerator = numerator.sqrt();
    let denominator = denominator.sqrt();
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= f64::MIN_POSITIVE {
        return Err("source derivative output is non-finite or has a zero reference".into());
    }
    Ok(numerator / denominator)
}

struct SourceDerivativeGate {
    initial_error: f64,
    warm_error: f64,
    retry_error: f64,
    initial: CudaRuntimeStatsSnapshot,
    warm: CudaRuntimeStatsSnapshot,
    failure_typed: bool,
    output_preserved: bool,
    teardown_live_bytes_before: usize,
    teardown_live_bytes_after: usize,
    teardown_released_bytes: usize,
    active_pairs: usize,
}

fn source_x_derivative_gate(
    backend: &CudaBackend,
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
) -> Result<SourceDerivativeGate, String> {
    let pairs = resident_regular_pairs(elements, nodes);
    if pairs.is_empty() {
        return Err("source-position derivative fixture has no resident regular pairs".into());
    }
    let expected: Vec<BemRegularPairOutF64> = pairs
        .iter()
        .map(|pair| cpu_source_x_centered_difference(pair, physics, SOURCE_DERIVATIVE_STEP))
        .collect();
    let allocation_before_workspace = backend.live_allocation_bytes();
    let mut workspace = CudaTbemAssemblyWorkspace::new_with_regular_pair_channels(
        elements, nodes, physics, backend,
    )
    .map_err(|error| format!("construct source derivative workspace: {error}"))?;

    let first_baseline = backend.begin_runtime_stats();
    let initial = workspace
        .assemble_source_x_derivative(elements, nodes, physics, backend)
        .map_err(|error| format!("native source x derivative first action failed: {error}"))?;
    let initial_stats = backend.runtime_stats_since(first_baseline);
    let initial_error = relative_pair_output_error(&initial, &expected)?;

    let warm_baseline = backend.begin_runtime_stats();
    let warm = workspace
        .assemble_source_x_derivative(elements, nodes, physics, backend)
        .map_err(|error| format!("native source x derivative warm action failed: {error}"))?;
    let warm_stats = backend.runtime_stats_since(warm_baseline);
    let warm_error = relative_pair_output_error(&warm, &expected)?;

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = workspace.assemble_source_x_derivative(elements, nodes, physics, backend);
    let failure_typed = matches!(
        failure,
        Err(CudaTbemAssemblyError::Backend(BackendError::Launch { .. }))
    );
    if failure_typed {
        backend.clear_failure();
        backend
            .recover_after_error()
            .map_err(|error| format!("source x derivative recovery failed: {error}"))?;
    }
    // Launch injection is checked before the kernel is encoded and before
    // the explicit D2H publication.  Read the resident buffer back to prove
    // that the previously published warm result remains intact, rather than
    // inferring preservation from the error type alone.
    let output_preserved = if failure_typed {
        workspace
            .read_source_x_derivative_output(backend)
            .map(|after_failure| {
                relative_pair_output_error(&after_failure, &warm).is_ok_and(|error| error == 0.0)
            })
            .map_err(|error| {
                format!("source x derivative output-preservation readback failed: {error}")
            })?
    } else {
        false
    };
    let retry = workspace
        .assemble_source_x_derivative(elements, nodes, physics, backend)
        .map_err(|error| format!("native source x derivative retry failed: {error}"))?;
    let retry_error = relative_pair_output_error(&retry, &expected)?;

    let teardown_live_bytes_before = backend.live_allocation_bytes();
    drop(workspace);
    let teardown_live_bytes_after = backend.live_allocation_bytes();
    let teardown_released_bytes =
        teardown_live_bytes_before.saturating_sub(teardown_live_bytes_after);
    if teardown_live_bytes_before < allocation_before_workspace
        || teardown_live_bytes_after != allocation_before_workspace
    {
        return Err(format!(
            "source derivative workspace teardown changed unrelated live bytes: before={teardown_live_bytes_before}, after={teardown_live_bytes_after}, baseline={allocation_before_workspace}"
        ));
    }

    Ok(SourceDerivativeGate {
        initial_error,
        warm_error,
        retry_error,
        initial: initial_stats,
        warm: warm_stats,
        failure_typed,
        output_preserved,
        teardown_live_bytes_before,
        teardown_live_bytes_after,
        teardown_released_bytes,
        active_pairs: pairs.len(),
    })
}

fn source_y_derivative_gate(
    backend: &CudaBackend,
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
) -> Result<SourceDerivativeGate, String> {
    let pairs = resident_regular_pairs(elements, nodes);
    if pairs.is_empty() {
        return Err("source-position y derivative fixture has no resident regular pairs".into());
    }
    let expected: Vec<BemRegularPairOutF64> = pairs
        .iter()
        .map(|pair| cpu_source_y_centered_difference(pair, physics, SOURCE_DERIVATIVE_STEP))
        .collect();
    let allocation_before_workspace = backend.live_allocation_bytes();
    let mut workspace = CudaTbemAssemblyWorkspace::new_with_regular_pair_channels(
        elements, nodes, physics, backend,
    )
    .map_err(|error| format!("construct source y derivative workspace: {error}"))?;

    let first_baseline = backend.begin_runtime_stats();
    let initial = workspace
        .assemble_source_y_derivative(elements, nodes, physics, backend)
        .map_err(|error| format!("native source y derivative first action failed: {error}"))?;
    let initial_stats = backend.runtime_stats_since(first_baseline);
    let initial_error = relative_pair_output_error(&initial, &expected)?;

    let warm_baseline = backend.begin_runtime_stats();
    let warm = workspace
        .assemble_source_y_derivative(elements, nodes, physics, backend)
        .map_err(|error| format!("native source y derivative warm action failed: {error}"))?;
    let warm_stats = backend.runtime_stats_since(warm_baseline);
    let warm_error = relative_pair_output_error(&warm, &expected)?;

    backend.inject_failure(CudaFailureInjectionPoint::Launch);
    let failure = workspace.assemble_source_y_derivative(elements, nodes, physics, backend);
    let failure_typed = matches!(
        failure,
        Err(CudaTbemAssemblyError::Backend(BackendError::Launch { .. }))
    );
    if failure_typed {
        backend.clear_failure();
        backend
            .recover_after_error()
            .map_err(|error| format!("source y derivative recovery failed: {error}"))?;
    }
    let output_preserved = if failure_typed {
        workspace
            .read_source_y_derivative_output(backend)
            .map(|after_failure| {
                relative_pair_output_error(&after_failure, &warm).is_ok_and(|error| error == 0.0)
            })
            .map_err(|error| {
                format!("source y derivative output-preservation readback failed: {error}")
            })?
    } else {
        false
    };
    let retry = workspace
        .assemble_source_y_derivative(elements, nodes, physics, backend)
        .map_err(|error| format!("native source y derivative retry failed: {error}"))?;
    let retry_error = relative_pair_output_error(&retry, &expected)?;

    let teardown_live_bytes_before = backend.live_allocation_bytes();
    drop(workspace);
    let teardown_live_bytes_after = backend.live_allocation_bytes();
    let teardown_released_bytes =
        teardown_live_bytes_before.saturating_sub(teardown_live_bytes_after);
    if teardown_live_bytes_before < allocation_before_workspace
        || teardown_live_bytes_after != allocation_before_workspace
    {
        return Err(format!(
            "source y derivative workspace teardown changed unrelated live bytes: before={teardown_live_bytes_before}, after={teardown_live_bytes_after}, baseline={allocation_before_workspace}"
        ));
    }

    Ok(SourceDerivativeGate {
        initial_error,
        warm_error,
        retry_error,
        initial: initial_stats,
        warm: warm_stats,
        failure_typed,
        output_preserved,
        teardown_live_bytes_before,
        teardown_live_bytes_after,
        teardown_released_bytes,
        active_pairs: pairs.len(),
    })
}

fn device_product(
    backend: &CudaBackend,
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    probe: &[Complex64],
    mode: ProductMode,
) -> Result<Vec<Complex64>, String> {
    let n = matrix.nrows();
    let mut matrix_device = backend
        .raw_alloc::<Complex64>(n.checked_mul(n).ok_or("matrix allocation overflows")?)
        .map_err(|error| format!("allocate CUDA product matrix: {error}"))?;
    let mut probe_device = backend
        .raw_alloc::<Complex64>(n)
        .map_err(|error| format!("allocate CUDA product input: {error}"))?;
    let mut output_device = backend
        .raw_alloc::<Complex64>(n)
        .map_err(|error| format!("allocate CUDA product output: {error}"))?;
    let zeros = vec![Complex64::new(0.0, 0.0); n];
    backend
        .try_copy_h2d(matrix.as_slice(), &mut matrix_device)
        .map_err(|error| format!("upload CUDA product matrix: {error}"))?;
    backend
        .try_copy_h2d(probe, &mut probe_device)
        .map_err(|error| format!("upload CUDA product input: {error}"))?;
    backend
        .try_copy_h2d(&zeros, &mut output_device)
        .map_err(|error| format!("upload CUDA product output: {error}"))?;
    let operation = match mode {
        ProductMode::Forward => Transpose::NoTrans,
        ProductMode::Transpose => Transpose::Trans,
        ProductMode::Hermitian => Transpose::ConjTrans,
    };
    backend
        .try_gemv(
            operation,
            n,
            n,
            Complex64::new(1.0, 0.0),
            &matrix_device,
            n,
            &probe_device,
            Complex64::new(0.0, 0.0),
            &mut output_device,
        )
        .map_err(|error| format!("CUDA product dispatch failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("CUDA product synchronization failed: {error}"))?;
    let mut output = vec![Complex64::new(0.0, 0.0); n];
    backend
        .try_copy_d2h(&output_device, &mut output)
        .map_err(|error| format!("read CUDA product output: {error}"))?;
    Ok(output)
}

fn dense_residual(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
    mode: ProductMode,
) -> Result<f64, String> {
    if solution.len() != matrix.nrows() || rhs.len() != matrix.nrows() {
        return Err("CUDA solve residual dimension mismatch".into());
    }
    let residual: Vec<Complex64> = (0..matrix.nrows())
        .map(|row| {
            let value: Complex64 = (0..matrix.ncols())
                .map(|column| {
                    let entry = match mode {
                        ProductMode::Forward => matrix[[row, column]],
                        ProductMode::Transpose => matrix[[column, row]],
                        ProductMode::Hermitian => matrix[[column, row]].conj(),
                    };
                    entry * solution[column]
                })
                .sum();
            value - rhs[row]
        })
        .collect();
    let numerator = residual.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    let denominator = rhs.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= f64::MIN_POSITIVE {
        return Err("CUDA solve residual is non-finite or has a zero RHS".into());
    }
    Ok(numerator / denominator)
}

fn cuda_dense_solve_gate(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    rhs: &[Complex64],
) -> Result<CudaSolveMetrics, String> {
    let backend = required_cuda_backend()?;
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(64 << 20))
        .map_err(|error| format!("CUDA solve pool allocation failed: {error}"))?;
    let backend = pool.backend();
    let n = matrix.nrows();
    if matrix.ncols() != n || rhs.len() != n || n == 0 {
        return Err("CUDA solve gate requires a non-empty square system".into());
    }
    let column_major: Vec<Complex64> = (0..n)
        .flat_map(|column| (0..n).map(move |row| matrix[[row, column]]))
        .collect();
    let setup_started = Instant::now();
    let lu = backend
        .dense_lu_c64_from_column_major(n, &column_major)
        .map_err(|error| format!("CUDA Complex64 LU setup failed: {error}"))?;
    let setup_ms = setup_started.elapsed().as_millis() as u64;

    let mut rhs_device = backend
        .raw_alloc::<Complex64>(n)
        .map_err(|error| format!("CUDA solve RHS allocation failed: {error}"))?;
    let mut output_device = backend
        .raw_alloc::<Complex64>(n)
        .map_err(|error| format!("CUDA solve output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(rhs, &mut rhs_device)
        .map_err(|error| format!("upload CUDA solve RHS: {error}"))?;
    let solve_started = Instant::now();
    backend
        .dense_lu_c64_solve(&lu, &rhs_device, &mut output_device, Transpose::NoTrans)
        .map_err(|error| format!("CUDA Complex64 forward solve failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("CUDA forward solve synchronization failed: {error}"))?;
    let solve_ms = solve_started.elapsed().as_millis() as u64;
    let mut solution = vec![Complex64::new(0.0, 0.0); n];
    backend
        .try_copy_d2h(&output_device, &mut solution)
        .map_err(|error| format!("read CUDA forward solution: {error}"))?;
    let residual_error = dense_residual(matrix, &solution, rhs, ProductMode::Forward)?;

    let adjoint_started = Instant::now();
    backend
        .dense_lu_c64_solve(&lu, &rhs_device, &mut output_device, Transpose::ConjTrans)
        .map_err(|error| format!("CUDA Complex64 adjoint solve failed: {error}"))?;
    backend
        .try_synchronize()
        .map_err(|error| format!("CUDA adjoint solve synchronization failed: {error}"))?;
    let adjoint_ms = adjoint_started.elapsed().as_millis() as u64;
    let mut adjoint_solution = vec![Complex64::new(0.0, 0.0); n];
    backend
        .try_copy_d2h(&output_device, &mut adjoint_solution)
        .map_err(|error| format!("read CUDA adjoint solution: {error}"))?;
    let adjoint_residual_error =
        dense_residual(matrix, &adjoint_solution, rhs, ProductMode::Hermitian)?;
    let stats = backend.runtime_stats();
    Ok((
        residual_error,
        adjoint_residual_error,
        setup_ms,
        solve_ms,
        adjoint_ms,
        stats.transfer.h2d_operations,
        stats.transfer.d2h_operations,
    ))
}

#[derive(Clone, Copy)]
enum ProductMode {
    Forward,
    Transpose,
    Hermitian,
}

fn run(backend: &CudaBackend) -> Result<QaResult, String> {
    let capabilities = backend.capabilities();
    if !matches!(
        capabilities.bem.regular_pair_c64,
        CapabilityState::Native | CapabilityState::VendorNative
    ) {
        return Err(format!(
            "required CUDA Complex64 regular-pair capability is {}; this gate does not accept a host fallback",
            state_name(capabilities.bem.regular_pair_c64)
        ));
    }
    if !matches!(
        backend.bem_source_x_derivative_capability(),
        CapabilityState::Native | CapabilityState::VendorNative
    ) {
        return Err(format!(
            "required CUDA Complex64 source-position derivative capability is {}",
            state_name(backend.bem_source_x_derivative_capability())
        ));
    }
    if !matches!(
        backend.bem_source_y_derivative_capability(),
        CapabilityState::Native | CapabilityState::VendorNative
    ) {
        return Err(format!(
            "required CUDA Complex64 source-position y derivative capability is {}",
            state_name(backend.bem_source_y_derivative_capability())
        ));
    }

    let (elements, nodes, physics) = fixture()?;
    let beta = physics.burton_miller_beta();

    let cpu_started = Instant::now();
    let cpu = build_tbem_system_stabilized_with_beta_simd(&elements, &nodes, &physics, 1.0, beta);
    let cpu_elapsed = cpu_started.elapsed().as_millis() as u64;

    backend.reset_runtime_stats();
    let cuda_started = Instant::now();
    let cuda = build_tbem_system_with_beta_cuda_checked(&elements, &nodes, &physics, beta, backend)
        .map_err(|error| format!("checked CUDA assembly failed: {error}"))?;
    let cuda_elapsed = cuda_started.elapsed().as_millis() as u64;

    let matrix_error = relative_matrix_error(cpu.matrix.as_slice(), cuda.matrix.as_slice())?;
    if matrix_error > MATRIX_PARITY_TOLERANCE {
        return Err(format!(
            "CUDA/CPU matrix parity {matrix_error:.6e} exceeds {MATRIX_PARITY_TOLERANCE:.6e}"
        ));
    }

    // Verify the phase, kernel channels, and Burton--Miller k dependence with
    // an independent CPU analytic derivative. The resident workspace
    // exercises the native CUDA derivative channels directly; host code only
    // reconstructs the public TBEM matrix and compares it to the CPU f64
    // reference.
    let derivative_started = Instant::now();
    let mut derivative_workspace =
        CudaTbemAssemblyWorkspace::new(&elements, &nodes, &physics, backend)
            .map_err(|error| format!("construct resident CUDA derivative workspace: {error}"))?;
    derivative_workspace
        .assemble(&elements, &nodes, &physics, beta, backend)
        .map_err(|error| format!("checked CUDA derivative base assembly failed: {error}"))?;
    let cuda_derivative = derivative_workspace
        .assemble_wavenumber_derivative(
            &elements,
            &nodes,
            &physics,
            beta,
            physics.burton_miller_beta_wavenumber_derivative(),
            physics.pressure_factor_wavenumber_derivative(),
            1.0,
            backend,
        )
        .map_err(|error| format!("checked CUDA analytic derivative assembly failed: {error}"))?;
    let cpu_derivative =
        build_tbem_system_stabilized_wavenumber_derivative(&elements, &nodes, &physics, 1.0, 1.0);
    let derivative_error = relative_matrix_error(
        cuda_derivative.matrix.as_slice(),
        cpu_derivative.matrix.as_slice(),
    )?;
    let derivative_elapsed = derivative_started.elapsed().as_millis() as u64;
    if derivative_error > DERIVATIVE_PARITY_TOLERANCE {
        return Err(format!(
            "CUDA/CPU wavenumber derivative parity {derivative_error:.6e} exceeds {DERIVATIVE_PARITY_TOLERANCE:.6e}"
        ));
    }

    // The source-coordinate gate uses an independent CPU centered difference
    // of the fixed-order reference kernel.  The CUDA derivative itself is
    // evaluated from the production resident regular-pair workspace, with
    // warm replay and one-shot typed launch-failure recovery.
    let source_derivative_started = Instant::now();
    let source_derivative = source_x_derivative_gate(backend, &elements, &nodes, &physics)?;
    let source_derivative_elapsed = source_derivative_started.elapsed().as_millis() as u64;
    if source_derivative.initial_error > SOURCE_DERIVATIVE_PARITY_TOLERANCE
        || source_derivative.warm_error > SOURCE_DERIVATIVE_PARITY_TOLERANCE
        || source_derivative.retry_error > SOURCE_DERIVATIVE_PARITY_TOLERANCE
    {
        return Err(format!(
            "CUDA/CPU source x derivative parity {:.6e}/{:.6e}/{:.6e} exceeds {:.6e}",
            source_derivative.initial_error,
            source_derivative.warm_error,
            source_derivative.retry_error,
            SOURCE_DERIVATIVE_PARITY_TOLERANCE
        ));
    }
    if !source_derivative.failure_typed || !source_derivative.output_preserved {
        return Err("source x derivative launch failure was not typed and transactional".into());
    }
    if source_derivative.initial.kernel_launches == 0
        || source_derivative.warm.kernel_launches == 0
        || source_derivative.initial.allocation_operations != 0
        || source_derivative.warm.allocation_operations != 0
        || source_derivative.initial.transfer.h2d_operations != 0
        || source_derivative.warm.transfer.h2d_operations != 0
        || source_derivative.initial.transfer.d2h_operations == 0
        || source_derivative.warm.transfer.d2h_operations == 0
    {
        return Err(format!(
            "source x derivative residency counters invalid: first={:?}, warm={:?}",
            source_derivative.initial, source_derivative.warm
        ));
    }

    let source_y_derivative_started = Instant::now();
    let source_y_derivative = source_y_derivative_gate(backend, &elements, &nodes, &physics)?;
    let source_y_derivative_elapsed = source_y_derivative_started.elapsed().as_millis() as u64;
    if source_y_derivative.initial_error > SOURCE_DERIVATIVE_PARITY_TOLERANCE
        || source_y_derivative.warm_error > SOURCE_DERIVATIVE_PARITY_TOLERANCE
        || source_y_derivative.retry_error > SOURCE_DERIVATIVE_PARITY_TOLERANCE
    {
        return Err(format!(
            "CUDA/CPU source y derivative parity {:.6e}/{:.6e}/{:.6e} exceeds {:.6e}",
            source_y_derivative.initial_error,
            source_y_derivative.warm_error,
            source_y_derivative.retry_error,
            SOURCE_DERIVATIVE_PARITY_TOLERANCE
        ));
    }
    if !source_y_derivative.failure_typed || !source_y_derivative.output_preserved {
        return Err("source y derivative launch failure was not typed and transactional".into());
    }
    if source_y_derivative.initial.kernel_launches == 0
        || source_y_derivative.warm.kernel_launches == 0
        || source_y_derivative.initial.allocation_operations != 0
        || source_y_derivative.warm.allocation_operations != 0
        || source_y_derivative.initial.transfer.h2d_operations != 0
        || source_y_derivative.warm.transfer.h2d_operations != 0
        || source_y_derivative.initial.transfer.d2h_operations == 0
        || source_y_derivative.warm.transfer.d2h_operations == 0
    {
        return Err(format!(
            "source y derivative residency counters invalid: first={:?}, warm={:?}",
            source_y_derivative.initial, source_y_derivative.warm
        ));
    }

    let probe: Vec<Complex64> = (0..cpu.num_dofs)
        .map(|index| Complex64::new(0.25 + index as f64 * 0.003, -0.1 + index as f64 * 0.002))
        .collect();
    let dual_probe: Vec<Complex64> = (0..cpu.num_dofs)
        .map(|index| Complex64::new(-0.17 + index as f64 * 0.005, 0.08 - index as f64 * 0.004))
        .collect();
    let products_started = Instant::now();
    let mut product_error = 0.0_f64;
    for mode in [
        ProductMode::Forward,
        ProductMode::Transpose,
        ProductMode::Hermitian,
    ] {
        let cpu_product = dense_product(&cpu.matrix, &probe, mode);
        let cuda_product = device_product(backend, &cuda.matrix, &probe, mode)?;
        product_error = product_error.max(relative_matrix_error(&cuda_product, &cpu_product)?);
    }
    let products_elapsed = products_started.elapsed().as_millis() as u64;
    if product_error > PRODUCT_PARITY_TOLERANCE {
        return Err(format!(
            "CUDA device forward/transpose/Hermitian product parity {product_error:.6e} exceeds {PRODUCT_PARITY_TOLERANCE:.6e}"
        ));
    }
    let cuda_forward_probe = device_product(backend, &cuda.matrix, &probe, ProductMode::Forward)?;
    let cuda_adjoint_dual =
        device_product(backend, &cuda.matrix, &dual_probe, ProductMode::Hermitian)?;
    let inner_product_error =
        relative_inner_product_error(&cuda_forward_probe, &dual_probe, &probe, &cuda_adjoint_dual)?;
    if inner_product_error > INNER_PRODUCT_TOLERANCE {
        return Err(format!(
            "CUDA forward/Hermitian inner-product error {inner_product_error:.6e} exceeds {INNER_PRODUCT_TOLERANCE:.6e}"
        ));
    }

    let solve_rhs: Vec<Complex64> = (0..cpu.num_dofs)
        .map(|index| Complex64::new(0.5 + index as f64 * 0.07, -0.3 + index as f64 * 0.05))
        .collect();
    let (
        solve_residual_error,
        adjoint_solve_residual_error,
        solve_setup_ms,
        solve_ms,
        adjoint_solve_ms,
        solve_h2d_operations,
        solve_d2h_operations,
    ) = cuda_dense_solve_gate(&cuda.matrix, &solve_rhs)?;
    if solve_residual_error > SOLVE_RESIDUAL_TOLERANCE
        || adjoint_solve_residual_error > SOLVE_RESIDUAL_TOLERANCE
    {
        return Err(format!(
            "CUDA Complex64 solve residuals {solve_residual_error:.6e}/{adjoint_solve_residual_error:.6e} exceed {SOLVE_RESIDUAL_TOLERANCE:.6e}"
        ));
    }

    let field_points = vec![[0.35, 0.2, 2.5], [-0.4, 0.6, 7.0]];
    let field_pressure = vec![
        Complex64::new(1.0, -0.2),
        Complex64::new(-0.35, 0.6),
        Complex64::new(0.2, -0.15),
    ];
    let field_velocity = vec![
        Complex64::new(0.05, 0.02),
        Complex64::new(-0.03, 0.01),
        Complex64::new(0.01, -0.02),
    ];
    let cpu_field_started = Instant::now();
    let cpu_field = compute_scattered_field_checked(
        &field_points,
        &elements,
        &nodes,
        &field_pressure,
        Some(&field_velocity),
        &physics,
        sonium_bem::BasisOrder::Constant,
    )
    .map_err(|error| format!("CPU field reference failed: {error}"))?;
    let cpu_field_elapsed = cpu_field_started.elapsed().as_millis() as u64;

    // The field operator owns reusable device state, so use a pool-backed
    // CUDA instance and measure it independently from the assembly stream.
    let field_backend = required_cuda_backend()?;
    let field_pool = MemoryPool::new(field_backend, MemoryPoolConfig::with_cap(64 << 20))
        .map_err(|error| format!("CUDA field pool allocation failed: {error}"))?;
    field_pool.backend().reset_runtime_stats();
    let cuda_field_started = Instant::now();
    let cuda_field = compute_scattered_field_cuda_checked(
        &field_pool,
        &field_points,
        &elements,
        &nodes,
        &field_pressure,
        Some(&field_velocity),
        &physics,
        sonium_bem::BasisOrder::Constant,
    )
    .map_err(|error| format!("checked CUDA field evaluation failed: {error}"))?;
    let cuda_field_elapsed = cuda_field_started.elapsed().as_millis() as u64;
    let field_error = relative_matrix_error(&cuda_field, &cpu_field)?;
    if field_error > PRODUCT_PARITY_TOLERANCE {
        return Err(format!(
            "CUDA/CPU field parity {field_error:.6e} exceeds {PRODUCT_PARITY_TOLERANCE:.6e}"
        ));
    }
    let field_stats = field_pool.backend().runtime_stats();
    if field_stats.kernel_launches == 0
        || field_stats.transfer.h2d_operations == 0
        || field_stats.transfer.d2h_operations == 0
    {
        return Err(format!(
            "CUDA field evaluation did not prove device work: kernels={}, h2d={}, d2h={}",
            field_stats.kernel_launches,
            field_stats.transfer.h2d_operations,
            field_stats.transfer.d2h_operations
        ));
    }
    let (
        sphere_cuda_cpu_error,
        sphere_mie_error,
        sphere_panel_count,
        sphere_elapsed,
        sphere_kernel_launches,
        sphere_h2d_operations,
        sphere_d2h_operations,
    ) = rigid_sphere_cuda_field_gate(&field_pool)?;
    if sphere_cuda_cpu_error > SPHERE_CUDA_FIELD_TOLERANCE {
        return Err(format!(
            "rigid-sphere CUDA/CPU field parity {sphere_cuda_cpu_error:.6e} exceeds {SPHERE_CUDA_FIELD_TOLERANCE:.6e}"
        ));
    }
    if sphere_mie_error > SPHERE_MIE_TOLERANCE {
        return Err(format!(
            "rigid-sphere CPU/Mie field parity {sphere_mie_error:.6e} exceeds {SPHERE_MIE_TOLERANCE:.6e}"
        ));
    }
    let (
        radiation_cuda_cpu_error,
        radiation_reference_error,
        radiation_power,
        radiation_reference_power,
        radiation_power_error,
        radiation_panel_count,
        radiation_elapsed,
        radiation_kernel_launches,
        radiation_h2d_operations,
        radiation_d2h_operations,
    ) = pulsating_sphere_cuda_field_gate(&field_pool)?;
    if radiation_cuda_cpu_error > RADIATION_CUDA_FIELD_TOLERANCE {
        return Err(format!(
            "pulsating-sphere CUDA/CPU field parity {radiation_cuda_cpu_error:.6e} exceeds {RADIATION_CUDA_FIELD_TOLERANCE:.6e}"
        ));
    }
    if radiation_reference_error > RADIATION_REFERENCE_TOLERANCE {
        return Err(format!(
            "pulsating-sphere BEM/reference field parity {radiation_reference_error:.6e} exceeds {RADIATION_REFERENCE_TOLERANCE:.6e}"
        ));
    }
    let (
        radiation_coarse_reference_error,
        radiation_mesh_convergence_improvement,
        radiation_coarse_panel_count,
        radiation_coarse_elapsed,
    ) = pulsating_sphere_mesh_convergence_gate(&field_pool, radiation_reference_error)?;

    let stats = backend.runtime_stats();
    if stats.kernel_launches == 0
        || stats.transfer.h2d_operations == 0
        || stats.transfer.d2h_operations == 0
    {
        return Err(format!(
            "CUDA assembly did not prove device work: kernels={}, h2d={}, d2h={}",
            stats.kernel_launches, stats.transfer.h2d_operations, stats.transfer.d2h_operations
        ));
    }

    let mut result = QaResult::passed(CASE_ID)
        .requested_backend("cuda")
        .actual_backend("cuda")
        .backend_passed("cuda")
        .reference("checked f64 CPU TBEM assembly, centered CPU source-x/source-y finite differences, CPU/Mie sphere references, and CUDA products")
        .reference_checksum("analytic/deterministic-fixture:bem-cuda-tri3-regular-pair-v1")
        .input_checksum("analytic/deterministic-fixture:bem-cuda-tri3-regular-pair-v1")
        .dimension("3D separated Tri3 constant-panel CUDA assembly fixture")
        .formulation(
            "CUDA Complex64 checked Tri3 assembly and field evaluation with device regular, quasi-singular, Duffy bulk, regularised hypersingular, source-collocation x/y derivatives, scattering, and prescribed-velocity radiation paths; host publishes pair results",
        )
        .tolerance("matrix/product/regular-field <= 1e-3; sphere CUDA-vs-CPU <= 1e-10; Mie/radiation reference <= 3e-1; wavenumber derivative <= 5e-3; source x/y derivatives <= 5e-6")
        .metric("matrix_parity_relative_error", matrix_error)
        .metric("product_parity_relative_error", product_error)
        .metric("inner_product_error", inner_product_error)
        .metric("cuda_solve_residual_relative_error", solve_residual_error)
        .metric(
            "cuda_adjoint_solve_residual_relative_error",
            adjoint_solve_residual_error,
        )
        .metric("cuda_solve_h2d_operations", solve_h2d_operations as f64)
        .metric("cuda_solve_d2h_operations", solve_d2h_operations as f64)
        .metric("wavenumber_derivative_relative_error", derivative_error)
        .metric(
            "source_position_x_derivative_relative_error",
            source_derivative.initial_error,
        )
        .metric(
            "source_position_x_derivative_warm_relative_error",
            source_derivative.warm_error,
        )
        .metric(
            "source_position_x_derivative_retry_relative_error",
            source_derivative.retry_error,
        )
        .metric(
            "source_position_y_derivative_relative_error",
            source_y_derivative.initial_error,
        )
        .metric(
            "source_position_y_derivative_warm_relative_error",
            source_y_derivative.warm_error,
        )
        .metric(
            "source_position_y_derivative_retry_relative_error",
            source_y_derivative.retry_error,
        )
        .metric(
            "source_position_x_derivative_active_pairs",
            source_derivative.active_pairs as f64,
        )
        .metric(
            "source_position_x_derivative_initial_kernel_launches",
            source_derivative.initial.kernel_launches as f64,
        )
        .metric(
            "source_position_x_derivative_warm_kernel_launches",
            source_derivative.warm.kernel_launches as f64,
        )
        .metric(
            "source_position_x_derivative_initial_allocations",
            source_derivative.initial.allocation_operations as f64,
        )
        .metric(
            "source_position_x_derivative_warm_allocations",
            source_derivative.warm.allocation_operations as f64,
        )
        .metric(
            "source_position_x_derivative_initial_h2d_operations",
            source_derivative.initial.transfer.h2d_operations as f64,
        )
        .metric(
            "source_position_x_derivative_warm_h2d_operations",
            source_derivative.warm.transfer.h2d_operations as f64,
        )
        .metric(
            "source_position_x_derivative_initial_d2h_operations",
            source_derivative.initial.transfer.d2h_operations as f64,
        )
        .metric(
            "source_position_x_derivative_warm_d2h_operations",
            source_derivative.warm.transfer.d2h_operations as f64,
        )
        .metric(
            "source_position_x_derivative_failure_typed",
            if source_derivative.failure_typed { 1.0 } else { 0.0 },
        )
        .metric(
            "source_position_x_derivative_output_preserved",
            if source_derivative.output_preserved { 1.0 } else { 0.0 },
        )
        .metric(
            "source_position_x_derivative_teardown_released_bytes",
            source_derivative.teardown_released_bytes as f64,
        )
        .metric(
            "source_position_x_derivative_teardown_live_bytes_before",
            source_derivative.teardown_live_bytes_before as f64,
        )
        .metric(
            "source_position_x_derivative_teardown_live_bytes_after",
            source_derivative.teardown_live_bytes_after as f64,
        )
        .metric(
            "source_position_y_derivative_active_pairs",
            source_y_derivative.active_pairs as f64,
        )
        .metric(
            "source_position_y_derivative_initial_kernel_launches",
            source_y_derivative.initial.kernel_launches as f64,
        )
        .metric(
            "source_position_y_derivative_warm_kernel_launches",
            source_y_derivative.warm.kernel_launches as f64,
        )
        .metric(
            "source_position_y_derivative_initial_allocations",
            source_y_derivative.initial.allocation_operations as f64,
        )
        .metric(
            "source_position_y_derivative_warm_allocations",
            source_y_derivative.warm.allocation_operations as f64,
        )
        .metric(
            "source_position_y_derivative_initial_h2d_operations",
            source_y_derivative.initial.transfer.h2d_operations as f64,
        )
        .metric(
            "source_position_y_derivative_warm_h2d_operations",
            source_y_derivative.warm.transfer.h2d_operations as f64,
        )
        .metric(
            "source_position_y_derivative_initial_d2h_operations",
            source_y_derivative.initial.transfer.d2h_operations as f64,
        )
        .metric(
            "source_position_y_derivative_warm_d2h_operations",
            source_y_derivative.warm.transfer.d2h_operations as f64,
        )
        .metric(
            "source_position_y_derivative_failure_typed",
            if source_y_derivative.failure_typed { 1.0 } else { 0.0 },
        )
        .metric(
            "source_position_y_derivative_output_preserved",
            if source_y_derivative.output_preserved { 1.0 } else { 0.0 },
        )
        .metric(
            "source_position_y_derivative_teardown_released_bytes",
            source_y_derivative.teardown_released_bytes as f64,
        )
        .metric(
            "source_position_y_derivative_teardown_live_bytes_before",
            source_y_derivative.teardown_live_bytes_before as f64,
        )
        .metric(
            "source_position_y_derivative_teardown_live_bytes_after",
            source_y_derivative.teardown_live_bytes_after as f64,
        )
        .metric("field_parity_relative_error", field_error)
        .metric("rigid_sphere_cuda_cpu_field_relative_error", sphere_cuda_cpu_error)
        .metric("rigid_sphere_mie_relative_error", sphere_mie_error)
        .metric("rigid_sphere_panel_count", sphere_panel_count as f64)
        .metric(
            "rigid_sphere_cuda_kernel_launches",
            sphere_kernel_launches as f64,
        )
        .metric(
            "rigid_sphere_cuda_h2d_operations",
            sphere_h2d_operations as f64,
        )
        .metric(
            "rigid_sphere_cuda_d2h_operations",
            sphere_d2h_operations as f64,
        )
        .metric(
            "pulsating_sphere_cuda_cpu_field_relative_error",
            radiation_cuda_cpu_error,
        )
        .metric(
            "pulsating_sphere_reference_relative_error",
            radiation_reference_error,
        )
        .metric(
            "pulsating_sphere_coarse_reference_relative_error",
            radiation_coarse_reference_error,
        )
        .metric(
            "pulsating_sphere_mesh_convergence_improvement",
            radiation_mesh_convergence_improvement,
        )
        .metric(
            "pulsating_sphere_coarse_panel_count",
            radiation_coarse_panel_count as f64,
        )
        .metric(
            "pulsating_sphere_radiation_power_relative_error",
            radiation_power_error,
        )
        .metric("pulsating_sphere_radiation_power", radiation_power)
        .metric(
            "pulsating_sphere_reference_power",
            radiation_reference_power,
        )
        .metric("pulsating_sphere_panel_count", radiation_panel_count as f64)
        .metric(
            "pulsating_sphere_cuda_kernel_launches",
            radiation_kernel_launches as f64,
        )
        .metric(
            "pulsating_sphere_cuda_h2d_operations",
            radiation_h2d_operations as f64,
        )
        .metric(
            "pulsating_sphere_cuda_d2h_operations",
            radiation_d2h_operations as f64,
        )
        .metric("dofs", cpu.num_dofs as f64)
        .metric("cuda_kernel_launches", stats.kernel_launches as f64)
        .metric("cuda_h2d_operations", stats.transfer.h2d_operations as f64)
        .metric("cuda_d2h_operations", stats.transfer.d2h_operations as f64)
        .metric(
            "cuda_synchronization_calls",
            stats.synchronization_calls as f64,
        )
        .metric(
            "cuda_allocation_high_water_bytes",
            stats.allocation_high_water_bytes as f64,
        )
        .metric(
            "field_cuda_kernel_launches",
            field_stats.kernel_launches as f64,
        )
        .metric(
            "field_cuda_h2d_operations",
            field_stats.transfer.h2d_operations as f64,
        )
        .metric(
            "field_cuda_d2h_operations",
            field_stats.transfer.d2h_operations as f64,
        )
        .metric(
            "field_cuda_allocation_high_water_bytes",
            field_stats.allocation_high_water_bytes as f64,
        )
        .maximum("matrix_parity_relative_error", MATRIX_PARITY_TOLERANCE)
        .maximum("product_parity_relative_error", PRODUCT_PARITY_TOLERANCE)
        .maximum("inner_product_error", INNER_PRODUCT_TOLERANCE)
        .maximum("cuda_solve_residual_relative_error", SOLVE_RESIDUAL_TOLERANCE)
        .maximum(
            "cuda_adjoint_solve_residual_relative_error",
            SOLVE_RESIDUAL_TOLERANCE,
        )
        .maximum(
            "wavenumber_derivative_relative_error",
            DERIVATIVE_PARITY_TOLERANCE,
        )
        .maximum(
            "source_position_x_derivative_relative_error",
            SOURCE_DERIVATIVE_PARITY_TOLERANCE,
        )
        .maximum(
            "source_position_x_derivative_warm_relative_error",
            SOURCE_DERIVATIVE_PARITY_TOLERANCE,
        )
        .maximum(
            "source_position_x_derivative_retry_relative_error",
            SOURCE_DERIVATIVE_PARITY_TOLERANCE,
        )
        .maximum(
            "source_position_y_derivative_relative_error",
            SOURCE_DERIVATIVE_PARITY_TOLERANCE,
        )
        .maximum(
            "source_position_y_derivative_warm_relative_error",
            SOURCE_DERIVATIVE_PARITY_TOLERANCE,
        )
        .maximum(
            "source_position_y_derivative_retry_relative_error",
            SOURCE_DERIVATIVE_PARITY_TOLERANCE,
        )
        .maximum("field_parity_relative_error", PRODUCT_PARITY_TOLERANCE)
        .maximum(
            "rigid_sphere_cuda_cpu_field_relative_error",
            SPHERE_CUDA_FIELD_TOLERANCE,
        )
        .maximum("rigid_sphere_mie_relative_error", SPHERE_MIE_TOLERANCE)
        .maximum(
            "pulsating_sphere_cuda_cpu_field_relative_error",
            RADIATION_CUDA_FIELD_TOLERANCE,
        )
        .maximum(
            "pulsating_sphere_reference_relative_error",
            RADIATION_REFERENCE_TOLERANCE,
        )
        .maximum(
            "pulsating_sphere_radiation_power_relative_error",
            RADIATION_POWER_TOLERANCE,
        )
        .minimum("pulsating_sphere_mesh_convergence_improvement", 0.0)
        .timing_ms("cpu_assembly", cpu_elapsed)
        .timing_ms("cuda_assembly", cuda_elapsed)
        .timing_ms("cuda_products", products_elapsed)
        .timing_ms("cuda_solve_setup", solve_setup_ms)
        .timing_ms("cuda_solve", solve_ms)
        .timing_ms("cuda_adjoint_solve", adjoint_solve_ms)
        .timing_ms("cuda_wavenumber_derivative", derivative_elapsed)
        .timing_ms("cuda_source_position_x_derivative", source_derivative_elapsed)
        .timing_ms("cuda_source_position_y_derivative", source_y_derivative_elapsed)
        .timing_ms("cpu_field", cpu_field_elapsed)
        .timing_ms("cuda_field", cuda_field_elapsed)
        .timing_ms("cuda_rigid_sphere_mie_gate", sphere_elapsed)
        .timing_ms("cuda_pulsating_sphere_radiation_gate", radiation_elapsed)
        .timing_ms(
            "cuda_pulsating_sphere_coarse_convergence_gate",
            radiation_coarse_elapsed,
        )
        .memory_bytes(stats.allocation_high_water_bytes as u64)
        .elapsed_ms(
            cpu_elapsed
                .saturating_add(cuda_elapsed)
                .saturating_add(products_elapsed)
                .saturating_add(solve_setup_ms)
                .saturating_add(solve_ms)
                .saturating_add(adjoint_solve_ms)
                .saturating_add(derivative_elapsed)
                .saturating_add(source_derivative_elapsed)
                .saturating_add(source_y_derivative_elapsed)
                .saturating_add(cpu_field_elapsed)
                .saturating_add(cuda_field_elapsed)
                .saturating_add(sphere_elapsed)
                .saturating_add(radiation_elapsed),
        );
    result = record_capabilities(result, backend);
    result.environment.insert(
        "physical_gate.bem.3d_rigid_sphere_mie".into(),
        "validated".into(),
    );
    result
        .environment
        .insert("physical_gate.bem.3d_radiation".into(), "validated".into());
    result.environment.insert(
        "physical_gate.bem.3d_wavenumber_derivative".into(),
        "validated".into(),
    );
    result.environment.insert(
        "physical_gate.bem.3d_source_derivative".into(),
        "validated".into(),
    );
    result.environment.insert(
        "physical_gate.bem.3d_source_y_derivative".into(),
        "validated".into(),
    );
    result.environment.insert(
        "physical_gate.bem.3d_mesh_convergence".into(),
        "validated".into(),
    );
    let operation_count = cuda.matrix.as_slice().len().saturating_mul(2).max(1);
    let complex_flop_count = operation_count.saturating_mul(8);
    let timed_cuda_ms = cuda_elapsed
        .saturating_add(products_elapsed)
        .saturating_add(solve_setup_ms)
        .saturating_add(solve_ms)
        .saturating_add(adjoint_solve_ms)
        .saturating_add(derivative_elapsed)
        .saturating_add(cuda_field_elapsed)
        .saturating_add(sphere_elapsed)
        .saturating_add(radiation_elapsed)
        .max(1);
    let throughput = complex_flop_count as f64 / (timed_cuda_ms as f64 / 1_000.0);
    // Normalize the measured workload against an explicitly provisioned
    // vendor peak.  The fraction is derived here from the timed FLOP count;
    // callers cannot manufacture a passing fraction independently of the
    // workload evidence.
    let peak_flops = std::env::var("SONIUM_QA_CUDA_PEAK_FLOPS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0);
    let roofline_fraction = peak_flops
        .map(|peak| throughput / peak)
        .filter(|value| value.is_finite() && *value >= 0.0);
    result = result
        .metric("operation_count", operation_count as f64)
        .metric("complex_csr_flop_count", complex_flop_count as f64)
        .metric("throughput_flops_per_second", throughput)
        .metric("timed_cuda_workload_ms", timed_cuda_ms as f64)
        .metric("cuda_library_operations", stats.library_operations as f64)
        .metric("cuda_d2d_operations", stats.transfer.d2d_operations as f64)
        .metric(
            "cuda_live_allocation_bytes",
            stats.live_allocation_bytes as f64,
        );
    result.environment.insert(
        "vendor_baseline.cublas".into(),
        "exercised for dense Complex64 solve and BLAS dispatch".into(),
    );
    result.environment.insert(
        "vendor_baseline.cusparse".into(),
        "exercised for CSR forward/transpose/Hermitian products".into(),
    );
    result.environment.insert(
        "vendor_baseline.cufft".into(),
        "not_applicable_to_bem_workload".into(),
    );
    result.environment.insert(
        "performance.roofline_fraction".into(),
        roofline_fraction.map_or_else(|| "unavailable".into(), |value| value.to_string()),
    );
    if let Some(roofline_fraction) = roofline_fraction {
        result = result.metric("roofline_fraction", roofline_fraction);
    }
    let runtime = cuda_runtime_evidence(
        stats,
        QaExecutionCounts {
            executed: 1,
            passed: 1,
            ..QaExecutionCounts::default()
        },
    );
    let report = serde_json::to_vec_pretty(&serde_json::json!({
        "schema": "sonium-compute-cuda-bem-evidence/v1",
        "status": "passed",
        "case_id": CASE_ID,
        "workload_elapsed_ms": timed_cuda_ms,
        "metrics": result.metrics,
        "thresholds": result.thresholds,
        "threshold_directions": result.threshold_directions,
        "environment": result.environment,
        "hardware": cuda_hardware_provenance(),
        "runtime": {
            "kernel_launches": stats.kernel_launches,
            "library_operations": stats.library_operations,
            "synchronization_calls": stats.synchronization_calls,
            "transfers": {
                "h2d_operations": stats.transfer.h2d_operations,
                "h2d_bytes": stats.transfer.h2d_bytes,
                "d2h_operations": stats.transfer.d2h_operations,
                "d2h_bytes": stats.transfer.d2h_bytes,
                "d2d_operations": stats.transfer.d2d_operations,
                "d2d_bytes": stats.transfer.d2d_bytes,
            },
            "live_allocation_bytes": stats.live_allocation_bytes,
            "allocation_high_water_bytes": stats.allocation_high_water_bytes,
        },
    }))
    .map_err(|error| format!("serialize CUDA BEM QA artifact: {error}"))?;
    let (artifact, checksum) = write_qa_artifact("bem_cuda_matrix_report.json", &report)?;
    result = result
        .artifact_with_provenance(
            artifact,
            "crates/sonium-qa/bin/bem_cuda_matrix.rs",
            "runtime capability/assembly report",
            "v1",
            checksum,
        )
        .runtime_evidence(runtime)
        .physical_residual(solve_residual_error)
        .physical_reference_error(sphere_mie_error);
    Ok(result)
}

fn main() {
    let started = Instant::now();
    let result = match required_cuda_backend() {
        Ok(backend) => match run(&backend) {
            Ok(result) => result,
            Err(error) => failed_cuda(error, "cuda", QaFailureStage::Execution),
        },
        Err(error) => failed_cuda(
            format!("required NVIDIA CUDA backend unavailable: {error}"),
            "unavailable",
            QaFailureStage::Initialization,
        ),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);

    let json = match result.to_json_line() {
        Ok(json) => json,
        Err(error) => {
            eprintln!("CUDA BEM QA result serialization failed: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    };
    println!("QA_RESULT: {json}");
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
