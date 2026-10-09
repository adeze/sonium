//! Checked contract evidence for the broader preconditioner family.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::preconditioners::{
    AdditiveSchwarzPreconditioner, AmgConfig, AmgCycle, AmgPreconditioner, AmgRestriction,
    AmgSmoother, BlockJacobiPreconditioner, ChebyshevConfig, ChebyshevPreconditioner, IlutConfig,
};
use sonium_solvers::{AdjointCapability, CsrMatrix, Preconditioner};
use std::time::Instant;

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            checked_constructions,
            checked_identity_applies,
            dimension_rejections,
            non_finite_rejections,
            malformed_rejections,
            amg_levels,
            exact_adjoint_capabilities,
            amg_adjoint_contract_checks,
            amg_exact_hermitian_actions,
            amg_hermitian_residual,
        )) => {
            let record = QaResult::passed("solvers.preconditioner-contracts.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("checked_constructions", checked_constructions)
                .metric("checked_identity_applies", checked_identity_applies)
                .metric("dimension_rejections", dimension_rejections)
                .metric("non_finite_rejections", non_finite_rejections)
                .metric("malformed_rejections", malformed_rejections)
                .metric("amg_levels", amg_levels)
                .metric("exact_adjoint_capabilities", exact_adjoint_capabilities)
                .metric("amg_adjoint_contract_checks", amg_adjoint_contract_checks)
                .metric("amg_exact_hermitian_actions", amg_exact_hermitian_actions)
                .metric("amg_hermitian_residual", amg_hermitian_residual)
                .minimum("checked_constructions", 3.0)
                .minimum("checked_identity_applies", 3.0)
                .minimum("dimension_rejections", 3.0)
                .minimum("non_finite_rejections", 3.0)
                .minimum("malformed_rejections", 3.0)
                .minimum("amg_levels", 1.0)
                .minimum("exact_adjoint_capabilities", 1.0)
                .minimum("amg_adjoint_contract_checks", 2.0)
                .minimum("amg_exact_hermitian_actions", 1.0)
                .maximum("amg_hermitian_residual", 1.0e-10)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed("solvers.preconditioner-contracts.v1", error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("PRECONDITIONER CONTRACT FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64), String> {
    let matrix: CsrMatrix<f64> = CsrMatrix::identity(4);
    let amg = AmgPreconditioner::try_from_csr(&matrix, AmgConfig::default())
        .map_err(|error| format!("checked AMG setup failed: {error}"))?;
    let schwarz = AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 2, 0)
        .map_err(|error| format!("checked Schwarz setup failed: {error}"))?;
    let block = BlockJacobiPreconditioner::try_from_csr(&matrix, 2, IlutConfig::default())
        .map_err(|error| format!("checked Block-Jacobi setup failed: {error}"))?;
    let chebyshev = ChebyshevPreconditioner::try_from_csr(&matrix, ChebyshevConfig::default())
        .map_err(|error| format!("checked Chebyshev setup failed: {error}"))?;
    let identity_preconditioners: [&dyn Preconditioner<f64>; 3] = [&amg, &schwarz, &block];
    let preconditioners: [&dyn Preconditioner<f64>; 4] = [&amg, &schwarz, &block, &chebyshev];

    let checked_constructions = preconditioners.len() as f64;
    let rhs = [1.0, -2.0, 3.0, -4.0];
    let transpose = amg.try_apply_transpose(&rhs);
    let hermitian = amg.try_apply_hermitian(&rhs);
    let exact_identity = |result: &Result<Vec<f64>, _>| {
        result.as_ref().is_ok_and(|values| {
            values
                .iter()
                .zip(rhs)
                .all(|(got, expected)| (*got - expected).abs() <= 1.0e-12)
        })
    };
    let amg_adjoint_contract_checks = [
        match amg.transpose_capability() {
            AdjointCapability::Exact => exact_identity(&transpose),
            AdjointCapability::Unsupported => transpose.is_err(),
            _ => false,
        },
        match amg.hermitian_capability() {
            AdjointCapability::Exact => exact_identity(&hermitian),
            AdjointCapability::Unsupported => hermitian.is_err(),
            _ => false,
        },
    ]
    .into_iter()
    .filter(|rejected| *rejected)
    .count() as f64;
    if amg_adjoint_contract_checks < 2.0 {
        return Err("AMG adjoint dispatch violated its advertised capability".into());
    }
    let checked_identity_applies =
        identity_preconditioners
            .iter()
            .try_fold(0.0, |count, preconditioner| {
                let result = preconditioner
                    .try_apply(&rhs)
                    .map_err(|error| format!("checked identity application failed: {error}"))?;
                if result
                    .iter()
                    .zip(rhs)
                    .any(|(got, expected)| (*got - expected).abs() > 1.0e-12)
                {
                    return Err(format!("identity application returned {result:?}"));
                }
                Ok::<_, String>(count + 1.0)
            })?;

    let dimension_rejections = identity_preconditioners
        .iter()
        .filter(|preconditioner| preconditioner.try_apply(&[1.0]).is_err())
        .count() as f64;
    if dimension_rejections != identity_preconditioners.len() as f64 {
        return Err("at least one preconditioner accepted a short vector".into());
    }
    let non_finite_rejections = identity_preconditioners
        .iter()
        .filter(|preconditioner| {
            preconditioner
                .try_apply(&[1.0, f64::NAN, 3.0, 4.0])
                .is_err()
        })
        .count() as f64;
    if non_finite_rejections != identity_preconditioners.len() as f64 {
        return Err("at least one preconditioner accepted a non-finite vector".into());
    }

    let exact_adjoint_capabilities = preconditioners
        .iter()
        .filter(|preconditioner| {
            preconditioner.transpose_capability() == AdjointCapability::Exact
                && preconditioner.hermitian_capability() == AdjointCapability::Exact
        })
        .count() as f64;

    let complex_matrix = CsrMatrix::from_triplets(
        8,
        8,
        (0..8)
            .flat_map(|row| {
                let mut entries = vec![(row, row, Complex64::new(4.0, 0.0))];
                if row + 1 < 8 {
                    let upper = Complex64::new(-0.8, 0.2);
                    entries.push((row, row + 1, upper));
                    entries.push((row + 1, row, upper.conj()));
                }
                entries
            })
            .collect(),
    );
    let hermitian_amg = AmgPreconditioner::try_from_csr(
        &complex_matrix,
        AmgConfig {
            restriction: AmgRestriction::ConjugateTranspose,
            smoother: AmgSmoother::SymmetricGaussSeidel,
            cycle: AmgCycle::VCycle,
            num_pre_smooth: 1,
            num_post_smooth: 1,
            k_cycle_iterations: 0,
            f_point_smoothing: false,
            ..AmgConfig::default()
        },
    )
    .map_err(|error| format!("checked Hermitian AMG setup failed: {error}"))?;
    if hermitian_amg.transpose_capability() != AdjointCapability::Unsupported
        || hermitian_amg.hermitian_capability() != AdjointCapability::Exact
    {
        return Err(
            "Hermitian AMG capability contract did not match the bounded exact slice".into(),
        );
    }
    let r = [
        Complex64::new(0.25, -0.5),
        Complex64::new(-0.75, 0.125),
        Complex64::new(0.5, 0.75),
        Complex64::new(-0.25, -0.625),
        Complex64::new(0.9, 0.2),
        Complex64::new(-0.1, 0.8),
        Complex64::new(0.4, -0.3),
        Complex64::new(-0.6, 0.45),
    ];
    let s = [
        Complex64::new(-0.4, 0.6),
        Complex64::new(0.2, -0.8),
        Complex64::new(0.7, 0.1),
        Complex64::new(-0.9, -0.2),
        Complex64::new(0.35, 0.5),
        Complex64::new(-0.55, 0.3),
        Complex64::new(0.8, -0.7),
        Complex64::new(-0.15, 0.9),
    ];
    let forward = hermitian_amg
        .try_apply(&r)
        .map_err(|error| format!("Hermitian AMG forward action failed: {error}"))?;
    let hermitian = hermitian_amg
        .try_apply_hermitian(&r)
        .map_err(|error| format!("Hermitian AMG adjoint action failed: {error}"))?;
    let output_residual = forward
        .iter()
        .zip(&hermitian)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
        .fold(0.0_f64, f64::max);
    let adjoint_s = hermitian_amg
        .try_apply_hermitian(&s)
        .map_err(|error| format!("Hermitian AMG second adjoint action failed: {error}"))?;
    let lhs: Complex64 = forward.iter().zip(&s).map(|(a, b)| a.conj() * b).sum();
    let rhs: Complex64 = r.iter().zip(adjoint_s).map(|(a, b)| a.conj() * b).sum();
    let inner_product_residual = (lhs - rhs).norm();
    let amg_hermitian_residual = output_residual.max(inner_product_residual);
    let amg_exact_hermitian_actions = if amg_hermitian_residual <= 1.0e-10 {
        1.0
    } else {
        0.0
    };

    let mut malformed = matrix.clone();
    malformed.row_ptrs[4] = 3;
    let malformed_rejections = [
        AmgPreconditioner::try_from_csr(&malformed, AmgConfig::default()).is_err(),
        AdditiveSchwarzPreconditioner::try_from_csr(&malformed, 2, 0).is_err(),
        BlockJacobiPreconditioner::try_from_csr(&malformed, 2, IlutConfig::default()).is_err(),
    ]
    .into_iter()
    .filter(|rejected| *rejected)
    .count() as f64;
    if malformed_rejections != identity_preconditioners.len() as f64 {
        return Err("at least one preconditioner accepted malformed CSR".into());
    }

    let mut invalid_amg_config = AmgConfig::default();
    invalid_amg_config.max_levels = 0;
    if AmgPreconditioner::try_from_csr(&matrix, invalid_amg_config).is_ok() {
        return Err("checked AMG accepted max_levels=0".into());
    }

    Ok((
        checked_constructions,
        checked_identity_applies,
        dimension_rejections,
        non_finite_rejections,
        malformed_rejections,
        amg.num_levels() as f64,
        exact_adjoint_capabilities,
        amg_adjoint_contract_checks,
        amg_exact_hermitian_actions,
        amg_hermitian_residual,
    ))
}
