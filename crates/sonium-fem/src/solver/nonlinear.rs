//! FEM nonlinear assembly and solver interface
//!
//! Bridges [`NonlinearForm`](crate::forms::NonlinearForm) with the generic
//! Newton-Raphson solver in [`sonium_solvers::nonlinear`].
//!
//! The FEM-specific work (looping over elements, tabulating basis functions,
//! assembling residual/tangent) lives here. The actual Newton iteration
//! loop lives in `sonium-solvers`.

use crate::assembly::TripletMatrix;
use crate::forms::{LocalMatrix, LocalVector, NonlinearElementForm, NonlinearForm};
use crate::mesh::Mesh;
use crate::parallelism::{ELEMENTS_PER_THREAD, collect_indexed, merge_element_triplets};
use crate::physics::solid::contact::augmented_lagrangian::{ContactManager, ContactResult};
use crate::quadrature::QuadratureRule;
use crate::spaces::FunctionSpace;
use sonium_solvers::traits::{AccuracyStatus, SolverStatus};
use sonium_solvers::{LinearSolveConfig, LinearSolverType, solve_real};

use std::panic::{AssertUnwindSafe, catch_unwind};

/// Per-element nonlinear assembly output: (residual_contribs, tangent_triplets).
type ElementNonlinearContrib = (Vec<(usize, f64)>, Vec<(usize, usize, f64)>);

/// FEM-specific configuration layered on top of Newton-Raphson.
#[derive(Debug, Clone)]
pub struct FemNewtonConfig {
    /// Newton-Raphson solver configuration (tolerances, damping, etc.)
    pub newton: sonium_solvers::NewtonConfig,
    /// Quadrature order for residual/tangent assembly.
    pub quad_order: usize,
    /// Linear solver configuration for the inner tangent solve.
    pub inner_solver: LinearSolveConfig,
}

impl FemNewtonConfig {
    /// Validate the FEM-specific inputs before any assembly starts.
    ///
    /// The generic Newton solver validates its own configuration, but a FEM
    /// call can still fail earlier through an out-of-range Dirichlet index,
    /// an empty quadrature rule, or a non-finite initial state.  Returning a
    /// typed `NewtonError` here keeps malformed user input from becoming an
    /// assertion panic deep in the element loop.
    pub fn validate(&self) -> Result<(), sonium_solvers::NewtonError> {
        if self.quad_order == 0 {
            return Err(sonium_solvers::NewtonError::InvalidConfig(
                "FEM quadrature order must be non-zero".into(),
            ));
        }
        if self.inner_solver.max_iterations == 0 {
            return Err(sonium_solvers::NewtonError::InvalidConfig(
                "inner linear solver max_iterations must be non-zero".into(),
            ));
        }
        if !self.inner_solver.tolerance.is_finite() || self.inner_solver.tolerance <= 0.0 {
            return Err(sonium_solvers::NewtonError::InvalidConfig(
                "inner linear solver tolerance must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

fn validate_newton_inputs(
    n_dofs: usize,
    initial_guess: &[f64],
    config: &FemNewtonConfig,
) -> Result<(), sonium_solvers::NewtonError> {
    config.validate()?;
    if n_dofs == 0 {
        return Err(sonium_solvers::NewtonError::InvalidConfig(
            "nonlinear FEM system must contain at least one DOF".into(),
        ));
    }
    if initial_guess.len() != n_dofs {
        return Err(sonium_solvers::NewtonError::InvalidConfig(format!(
            "initial guess has length {}, expected {n_dofs}",
            initial_guess.len()
        )));
    }
    if initial_guess.iter().any(|value| !value.is_finite()) {
        return Err(sonium_solvers::NewtonError::InvalidConfig(
            "initial guess contains NaN or infinity".into(),
        ));
    }
    Ok(())
}

fn validate_dirichlet_dofs(
    n_dofs: usize,
    dirichlet_dofs: &[(usize, f64)],
) -> Result<(), sonium_solvers::NewtonError> {
    for &(dof, value) in dirichlet_dofs {
        if dof >= n_dofs {
            return Err(sonium_solvers::NewtonError::InvalidConfig(format!(
                "Dirichlet DOF {dof} is outside the {n_dofs}-DOF system"
            )));
        }
        if !value.is_finite() {
            return Err(sonium_solvers::NewtonError::InvalidConfig(format!(
                "Dirichlet value at DOF {dof} is not finite"
            )));
        }
    }
    Ok(())
}

/// Validate the mesh and function-space contract before a nonlinear closure
/// can enter element indexing or basis tabulation.
///
/// Function-space implementations historically exposed infallible methods and
/// a few reject unsupported element/degree combinations with `panic!`.  The
/// nonlinear entry points are public solver boundaries, so convert both
/// structural errors and those legacy panics into the same typed Newton error
/// rather than allowing malformed input to abort a process halfway through an
/// iteration.
fn validate_external_load(
    n_dofs: usize,
    external_load: &[f64],
) -> Result<(), sonium_solvers::NewtonError> {
    if external_load.is_empty() {
        return Ok(());
    }
    if external_load.len() != n_dofs {
        return Err(sonium_solvers::NewtonError::InvalidConfig(format!(
            "external load has {} entries; expected {n_dofs}",
            external_load.len()
        )));
    }
    if external_load.iter().any(|value| !value.is_finite()) {
        return Err(sonium_solvers::NewtonError::InvalidConfig(
            "external load contains NaN or infinity".into(),
        ));
    }
    Ok(())
}

fn validate_space_and_mesh<S: FunctionSpace>(
    space: &S,
    mesh: &Mesh,
) -> Result<(), sonium_solvers::NewtonError> {
    mesh.validate_connectivity().map_err(|errors| {
        sonium_solvers::NewtonError::InvalidConfig(format!(
            "invalid nonlinear FEM mesh connectivity: {errors:?}"
        ))
    })?;

    let validation = catch_unwind(AssertUnwindSafe(|| space.validate(mesh))).map_err(|_| {
        sonium_solvers::NewtonError::InvalidConfig(
            "function-space validation panicked for the supplied mesh".into(),
        )
    })?;
    validation.map_err(|error| {
        sonium_solvers::NewtonError::InvalidConfig(format!(
            "invalid nonlinear FEM function space: {error:?}"
        ))
    })
}

impl Default for FemNewtonConfig {
    fn default() -> Self {
        Self {
            newton: sonium_solvers::NewtonConfig::default(),
            quad_order: 2,
            inner_solver: LinearSolveConfig {
                solver: LinearSolverType::GmresIlu,
                max_iterations: 1000,
                tolerance: 1e-11,
                ..Default::default()
            },
        }
    }
}

/// Solve a nonlinear FEM problem using Newton-Raphson.
///
/// Assembles residual and tangent from the [`NonlinearForm`] at each Newton step,
/// then delegates to [`sonium_solvers::newton_raphson`] for the iteration.
pub fn fem_newton_solve<F, S>(
    form: &F,
    space: &S,
    mesh: &Mesh,
    initial_guess: &[f64],
    config: &FemNewtonConfig,
) -> Result<sonium_solvers::NewtonSolution<f64>, sonium_solvers::NewtonError>
where
    F: NonlinearForm<f64>,
    S: FunctionSpace,
{
    validate_space_and_mesh(space, mesh)?;
    let n_dofs = space.num_global_dofs(mesh);
    validate_newton_inputs(n_dofs, initial_guess, config)?;

    let x0 = initial_guess.to_vec();
    let quad_order = config.quad_order;
    let inner_config = config.inner_solver.clone();

    sonium_solvers::newton_raphson(
        &x0,
        |x| assemble_residual(form, space, mesh, x, quad_order),
        |x, rhs| {
            let (_, tangent) = assemble_nonlinear(form, space, mesh, x, quad_order);
            let csr = tangent.to_csr();
            solve_real(&csr, rhs, &inner_config)
                .map(|r| r.x)
                .map_err(|e| e.to_string())
        },
        &config.newton,
    )
}

/// Solve a nonlinear FEM problem using Newton-Raphson with Dirichlet BCs.
///
/// Like [`fem_newton_solve`], but enforces Dirichlet boundary conditions at each
/// Newton step by zeroing constrained residual entries and replacing constrained
/// tangent rows/columns with identity.
pub fn fem_newton_solve_with_bcs<F, S>(
    form: &F,
    space: &S,
    mesh: &Mesh,
    dirichlet_dofs: &[(usize, f64)],
    initial_guess: &[f64],
    config: &FemNewtonConfig,
) -> Result<sonium_solvers::NewtonSolution<f64>, sonium_solvers::NewtonError>
where
    F: NonlinearForm<f64>,
    S: FunctionSpace,
{
    fem_newton_solve_with_bcs_and_load(
        form,
        space,
        mesh,
        dirichlet_dofs,
        initial_guess,
        &[],
        config,
    )
}

/// Solve a nonlinear FEM problem with Dirichlet BCs and a fixed external load.
///
/// The supplied load vector is subtracted from the assembled internal
/// residual, so positive entries represent forces applied to the structure.
/// An empty vector is accepted only as the compatibility path used by
/// [`fem_newton_solve_with_bcs`].
pub fn fem_newton_solve_with_bcs_and_load<F, S>(
    form: &F,
    space: &S,
    mesh: &Mesh,
    dirichlet_dofs: &[(usize, f64)],
    initial_guess: &[f64],
    external_load: &[f64],
    config: &FemNewtonConfig,
) -> Result<sonium_solvers::NewtonSolution<f64>, sonium_solvers::NewtonError>
where
    F: NonlinearForm<f64>,
    S: FunctionSpace,
{
    validate_space_and_mesh(space, mesh)?;
    let n_dofs = space.num_global_dofs(mesh);
    validate_newton_inputs(n_dofs, initial_guess, config)?;
    validate_dirichlet_dofs(n_dofs, dirichlet_dofs)?;
    validate_external_load(n_dofs, external_load)?;

    let is_constrained = {
        let mut v = vec![false; n_dofs];
        for &(dof, _) in dirichlet_dofs {
            v[dof] = true;
        }
        v
    };

    // Apply Dirichlet values to initial guess
    let mut u0 = initial_guess.to_vec();
    for &(dof, val) in dirichlet_dofs {
        u0[dof] = val;
    }

    let x0 = u0;
    let quad_order = config.quad_order;
    let inner_config = config.inner_solver.clone();

    sonium_solvers::newton_raphson(
        &x0,
        |x| {
            let mut residual = assemble_residual(form, space, mesh, x, quad_order);

            if !external_load.is_empty() {
                for (internal, applied) in residual.iter_mut().zip(external_load) {
                    *internal -= applied;
                }
            }

            for &(dof, _) in dirichlet_dofs {
                residual[dof] = 0.0;
            }

            residual
        },
        |x, rhs| {
            let (_, mut tangent) = assemble_nonlinear(form, space, mesh, x, quad_order);

            apply_dirichlet_to_triplet(&mut tangent, &is_constrained);

            let mut rhs_mod = rhs.to_vec();
            for &(dof, _) in dirichlet_dofs {
                rhs_mod[dof] = 0.0;
            }

            let csr = tangent.to_csr();
            solve_real(&csr, &rhs_mod, &inner_config)
                .map(|r| r.x)
                .map_err(|e| e.to_string())
        },
        &config.newton,
    )
}

/// Assemble both residual and tangent for a nonlinear form.
pub(crate) fn assemble_nonlinear<F, S>(
    form: &F,
    space: &S,
    mesh: &Mesh,
    solution: &[f64],
    quad_order: usize,
) -> (Vec<f64>, TripletMatrix<f64>)
where
    F: NonlinearForm<f64>,
    S: FunctionSpace,
{
    let n_dofs = space.num_global_dofs(mesh);
    let n_elems = mesh.num_elements();

    let element_results: Vec<_> = collect_indexed(n_elems, ELEMENTS_PER_THREAD, |elem_idx| {
        assemble_element_nonlinear(form, space, mesh, solution, elem_idx, quad_order)
    });

    let mut residual = vec![0.0f64; n_dofs];
    let (elem_residuals, elem_tangents): (Vec<_>, Vec<_>) = element_results.into_iter().unzip();
    for (dof, val) in elem_residuals.into_iter().flatten() {
        residual[dof] += val;
    }
    let total_triplets: usize = elem_tangents.iter().map(Vec::len).sum();
    let tangent = merge_element_triplets(elem_tangents, n_dofs, n_dofs, total_triplets);

    (residual, tangent)
}

/// Assemble only the residual (used by the Newton residual closure).
fn assemble_residual<F, S>(
    form: &F,
    space: &S,
    mesh: &Mesh,
    solution: &[f64],
    quad_order: usize,
) -> Vec<f64>
where
    F: NonlinearForm<f64>,
    S: FunctionSpace,
{
    let n_dofs = space.num_global_dofs(mesh);
    let n_elems = mesh.num_elements();

    let element_residuals: Vec<_> = collect_indexed(n_elems, ELEMENTS_PER_THREAD, |elem_idx| {
        assemble_element_residual(form, space, mesh, solution, elem_idx, quad_order)
    });

    let mut residual = vec![0.0f64; n_dofs];
    for contribs in element_residuals {
        for (dof, val) in contribs {
            residual[dof] += val;
        }
    }

    residual
}

/// Single element residual assembly.
fn assemble_element_residual<F, S>(
    form: &F,
    space: &S,
    mesh: &Mesh,
    solution: &[f64],
    elem_idx: usize,
    quad_order: usize,
) -> Vec<(usize, f64)>
where
    F: NonlinearForm<f64>,
    S: FunctionSpace,
{
    let elem = &mesh.elements[elem_idx];
    let dofs = space.local_to_global(mesh, elem_idx);
    let local_sol: Vec<f64> = dofs.iter().map(|&d| solution[d]).collect();

    let quad = QuadratureRule::new(elem.element_type, quad_order);
    let mut local_vec = LocalVector::zeros(dofs.len());

    for qp in quad.iter() {
        let tab = space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
        form.element_residual(&tab, &local_sol, qp.weight, &mut local_vec);
    }

    dofs.iter()
        .zip(local_vec.values.iter())
        .map(|(&dof, &val)| (dof, val))
        .collect()
}

/// Solve a nonlinear FEM problem using Newton-Raphson with element-level forms.
///
/// This variant works with element-based formulations (beam, shell) that
/// bypass the standard FunctionSpace/BasisTabulation pathway. The form itself
/// handles element assembly — no mesh or function space needed.
///
/// # Arguments
///
/// - `form`: implements [`NonlinearElementForm<f64>`]
/// - `initial_guess`: global DOF vector
/// - `config`: Newton-Raphson + linear solver configuration
pub fn fem_newton_solve_element<F>(
    form: &F,
    initial_guess: &[f64],
    config: &FemNewtonConfig,
) -> Result<sonium_solvers::NewtonSolution<f64>, sonium_solvers::NewtonError>
where
    F: NonlinearElementForm<f64>,
{
    let n_dofs = form.num_global_dofs();
    validate_newton_inputs(n_dofs, initial_guess, config)?;

    let x0 = initial_guess.to_vec();
    let inner_config = config.inner_solver.clone();

    sonium_solvers::newton_raphson(
        &x0,
        |x| {
            let (residual, _) = crate::assembly::assemble_nonlinear_element_form(form, x);
            residual
        },
        |x, rhs| {
            let (_, tangent) = crate::assembly::assemble_nonlinear_element_form(form, x);
            let csr = tangent.to_csr();
            solve_real(&csr, rhs, &inner_config)
                .map(|r| r.x)
                .map_err(|e| e.to_string())
        },
        &config.newton,
    )
}

/// Solve a nonlinear FEM problem with augmented Lagrangian contact.
///
/// Wraps Newton-Raphson with an outer Uzawa iteration that updates Lagrange
/// multipliers for the contact constraints. At each Newton step, the bulk
/// residual/tangent from the [`NonlinearForm`] is augmented with contact
/// contributions from the [`ContactManager`].
///
/// Dirichlet BCs are enforced at each Newton iteration by zeroing constrained
/// residual entries and replacing tangent rows/columns with identity.
///
/// # Arguments
///
/// - `form`: bulk material form (e.g. [`HyperelasticForm`](crate::physics::solid::nonlinear_solid::HyperelasticForm))
/// - `space`: function space (e.g. `VectorH1Space`)
/// - `mesh`: the FEM mesh
/// - `contact`: mutable contact manager (owns BVH, multipliers, active set)
/// - `dirichlet_dofs`: prescribed DOF indices and their values `(dof_index, value)`
/// - `initial_guess`: global DOF vector (3 * n_nodes for 3D).
///   Dirichlet values are enforced on top of this.
/// - `config`: Newton-Raphson + linear solver configuration
///
/// # Returns
///
/// The converged displacement vector and contact diagnostics.
pub fn fem_newton_solve_contact<F, S>(
    form: &F,
    space: &S,
    mesh: &Mesh,
    contact: &mut ContactManager,
    dirichlet_dofs: &[(usize, f64)],
    initial_guess: &[f64],
    config: &FemNewtonConfig,
) -> Result<(sonium_solvers::NewtonSolution<f64>, ContactResult), sonium_solvers::NewtonError>
where
    F: NonlinearForm<f64>,
    S: FunctionSpace,
{
    fem_newton_solve_contact_with_load(
        form,
        space,
        mesh,
        contact,
        dirichlet_dofs,
        initial_guess,
        &[],
        config,
    )
}

/// Solve nonlinear FEM contact with a fixed external load vector.
///
/// The external vector is subtracted from the assembled internal residual,
/// so positive entries represent applied forces. An empty vector preserves
/// the displacement-controlled compatibility path.
pub fn fem_newton_solve_contact_with_load<F, S>(
    form: &F,
    space: &S,
    mesh: &Mesh,
    contact: &mut ContactManager,
    dirichlet_dofs: &[(usize, f64)],
    initial_guess: &[f64],
    external_load: &[f64],
    config: &FemNewtonConfig,
) -> Result<(sonium_solvers::NewtonSolution<f64>, ContactResult), sonium_solvers::NewtonError>
where
    F: NonlinearForm<f64>,
    S: FunctionSpace,
{
    validate_space_and_mesh(space, mesh)?;
    let n_dofs = space.num_global_dofs(mesh);
    validate_newton_inputs(n_dofs, initial_guess, config)?;
    validate_dirichlet_dofs(n_dofs, dirichlet_dofs)?;
    validate_external_load(n_dofs, external_load)?;

    // Build a fast lookup: is_constrained[dof] = true if Dirichlet
    let is_constrained = {
        let mut v = vec![false; n_dofs];
        for &(dof, _) in dirichlet_dofs {
            v[dof] = true;
        }
        v
    };

    let contact_config = contact.config().clone();

    // Apply Dirichlet values to initial guess
    let mut u = initial_guess.to_vec();
    for &(dof, val) in dirichlet_dofs {
        u[dof] = val;
    }

    let mut uzawa_converged = false;
    let mut uzawa_iter = 0;
    let mut multiplier_update_norm = f64::MAX;
    // Accumulated inner-Newton work counters across Uzawa iterations.
    let mut total_residual_evaluations = 0usize;
    let mut total_jacobian_evaluations = 0usize;
    let mut total_inner_iterations = 0usize;

    for k in 0..contact_config.max_uzawa_iterations {
        uzawa_iter = k + 1;

        // Step 1: Detect contact pairs with current displacement
        contact.detect_contact(&u);

        // Step 2: Inner Newton-Raphson with combined bulk + contact
        let x0 = u.clone();
        let quad_order = config.quad_order;
        let inner_config = config.inner_solver.clone();

        let newton_result = sonium_solvers::newton_raphson(
            &x0,
            |x| {
                // Bulk residual
                let mut residual = assemble_residual(form, space, mesh, x, quad_order);
                if !external_load.is_empty() {
                    for (internal, applied) in residual.iter_mut().zip(external_load) {
                        *internal -= applied;
                    }
                }

                // Add contact residual
                contact.contact_residual(x, &mut residual);

                // Enforce Dirichlet: zero residual at constrained DOFs
                for &(dof, _) in dirichlet_dofs {
                    residual[dof] = 0.0;
                }

                residual
            },
            |x, rhs| {
                // Bulk tangent
                let (_, mut tangent) = assemble_nonlinear(form, space, mesh, x, quad_order);

                // Add contact tangent
                for (row, col, val) in contact.contact_tangent(x) {
                    tangent.push(row, col, val);
                }

                // Enforce Dirichlet on tangent: replace constrained rows/cols
                // with identity. We do this on the triplet matrix by zeroing
                // entries in constrained rows/cols and adding 1.0 on diagonal.
                apply_dirichlet_to_triplet(&mut tangent, &is_constrained);

                // Also zero the RHS at constrained DOFs
                let mut rhs_mod = rhs.to_vec();
                for &(dof, _) in dirichlet_dofs {
                    rhs_mod[dof] = 0.0;
                }

                let csr = tangent.to_csr();
                solve_real(&csr, &rhs_mod, &inner_config)
                    .map(|r| r.x)
                    .map_err(|e| e.to_string())
            },
            &config.newton,
        )?;

        total_residual_evaluations += newton_result.residual_evaluations;
        total_jacobian_evaluations += newton_result.jacobian_evaluations;
        total_inner_iterations += newton_result.inner_iterations;
        u = newton_result.x.to_vec();

        // Re-impose Dirichlet exactly (Newton update may perturb them)
        for &(dof, val) in dirichlet_dofs {
            u[dof] = val;
        }

        // Step 3: Update Lagrange multipliers (Uzawa step)
        multiplier_update_norm = contact.update_multipliers(&u);

        // Step 4: Check outer convergence
        if multiplier_update_norm < contact_config.uzawa_tolerance {
            uzawa_converged = true;
            break;
        }
    }

    // Build results
    let max_pen = contact.max_penetration(&u);
    let newton_solution = sonium_solvers::NewtonSolution {
        x: u,
        iterations: uzawa_iter,
        residual: multiplier_update_norm,
        converged: uzawa_converged,
        status: if uzawa_converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        },
        attainable_accuracy: if uzawa_converged {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        },
        residual_evaluations: total_residual_evaluations,
        jacobian_evaluations: total_jacobian_evaluations,
        inner_iterations: total_inner_iterations,
    };

    let contact_result = ContactResult {
        uzawa_iterations: uzawa_iter,
        multiplier_update_norm,
        converged: uzawa_converged,
        active_pairs: contact.count_active_pairs(&newton_solution.x),
        max_penetration: max_pen,
    };

    Ok((newton_solution, contact_result))
}

/// Solve a nonlinear FEM problem with augmented Lagrangian contact,
/// using an element-level form (shell, beam).
///
/// Same Uzawa + Newton algorithm as [`fem_newton_solve_contact`], but works
/// with [`NonlinearElementForm`] instead of [`NonlinearForm`]. This is the
/// entry point for contact with corotational shells and beams.
///
/// The form owns its own mesh/connectivity (no `Mesh` or `FunctionSpace` needed).
/// Contact detection uses displacement DOFs only (first 3 DOFs per node);
/// the caller specifies `dofs_per_node` so the solver knows the stride.
pub fn fem_newton_solve_element_contact<F>(
    form: &F,
    contact: &mut ContactManager,
    dofs_per_node: usize,
    dirichlet_dofs: &[(usize, f64)],
    initial_guess: &[f64],
    config: &FemNewtonConfig,
) -> Result<(sonium_solvers::NewtonSolution<f64>, ContactResult), sonium_solvers::NewtonError>
where
    F: NonlinearElementForm<f64>,
{
    let n_dofs = form.num_global_dofs();
    validate_newton_inputs(n_dofs, initial_guess, config)?;
    validate_dirichlet_dofs(n_dofs, dirichlet_dofs)?;
    if !(3..=n_dofs).contains(&dofs_per_node) {
        return Err(sonium_solvers::NewtonError::InvalidConfig(format!(
            "dofs_per_node must be between 3 and {n_dofs}, got {dofs_per_node}"
        )));
    }
    if n_dofs % dofs_per_node != 0 {
        return Err(sonium_solvers::NewtonError::InvalidConfig(format!(
            "system DOF count {n_dofs} is not divisible by dofs_per_node {dofs_per_node}"
        )));
    }

    let is_constrained = {
        let mut v = vec![false; n_dofs];
        for &(dof, _) in dirichlet_dofs {
            v[dof] = true;
        }
        v
    };

    let contact_config = contact.config().clone();

    let mut u = initial_guess.to_vec();
    for &(dof, val) in dirichlet_dofs {
        u[dof] = val;
    }

    // Build the translational-DOF extraction for contact:
    // contact expects a flat [x0,y0,z0, x1,y1,z1, ...] displacement array
    // but shells have 6 DOFs per node. Extract translation DOFs only.
    let n_nodes = n_dofs / dofs_per_node;
    let extract_translation_disp = |full: &[f64]| -> Vec<f64> {
        let mut t = vec![0.0; 3 * n_nodes];
        for i in 0..n_nodes {
            t[3 * i] = full[dofs_per_node * i];
            t[3 * i + 1] = full[dofs_per_node * i + 1];
            t[3 * i + 2] = full[dofs_per_node * i + 2];
        }
        t
    };

    let mut uzawa_converged = false;
    let mut uzawa_iter = 0;
    let mut multiplier_update_norm = f64::MAX;
    // Accumulated inner-Newton work counters across Uzawa iterations.
    let mut total_residual_evaluations = 0usize;
    let mut total_jacobian_evaluations = 0usize;
    let mut total_inner_iterations = 0usize;

    for k in 0..contact_config.max_uzawa_iterations {
        uzawa_iter = k + 1;

        let u_trans = extract_translation_disp(&u);
        contact.detect_contact(&u_trans);

        let x0 = u.clone();
        let inner_config = config.inner_solver.clone();

        let newton_result = sonium_solvers::newton_raphson(
            &x0,
            |x| {
                let (mut residual, _) = crate::assembly::assemble_nonlinear_element_form(form, x);

                // Add contact residual (operates on translation DOFs)
                let x_trans = extract_translation_disp(x);
                let mut contact_res = vec![0.0; 3 * n_nodes];
                contact.contact_residual(&x_trans, &mut contact_res);
                // Scatter back to full DOF layout
                for i in 0..n_nodes {
                    for d in 0..3 {
                        residual[dofs_per_node * i + d] += contact_res[3 * i + d];
                    }
                }

                for &(dof, _) in dirichlet_dofs {
                    residual[dof] = 0.0;
                }

                residual
            },
            |x, rhs| {
                let (_, mut tangent) = crate::assembly::assemble_nonlinear_element_form(form, x);

                // Add contact tangent (maps from 3-DOF to full-DOF indices)
                let x_trans = extract_translation_disp(x);
                for (row3, col3, val) in contact.contact_tangent(&x_trans) {
                    let node_r = row3 / 3;
                    let d_r = row3 % 3;
                    let node_c = col3 / 3;
                    let d_c = col3 % 3;
                    let full_row = dofs_per_node * node_r + d_r;
                    let full_col = dofs_per_node * node_c + d_c;
                    tangent.push(full_row, full_col, val);
                }

                apply_dirichlet_to_triplet(&mut tangent, &is_constrained);

                let mut rhs_mod = rhs.to_vec();
                for &(dof, _) in dirichlet_dofs {
                    rhs_mod[dof] = 0.0;
                }

                let csr = tangent.to_csr();
                solve_real(&csr, &rhs_mod, &inner_config)
                    .map(|r| r.x)
                    .map_err(|e| e.to_string())
            },
            &config.newton,
        )?;

        total_residual_evaluations += newton_result.residual_evaluations;
        total_jacobian_evaluations += newton_result.jacobian_evaluations;
        total_inner_iterations += newton_result.inner_iterations;
        u = newton_result.x.to_vec();
        for &(dof, val) in dirichlet_dofs {
            u[dof] = val;
        }

        let u_trans = extract_translation_disp(&u);
        multiplier_update_norm = contact.update_multipliers(&u_trans);

        if multiplier_update_norm < contact_config.uzawa_tolerance {
            uzawa_converged = true;
            break;
        }
    }

    let u_trans = extract_translation_disp(&u);
    let max_pen = contact.max_penetration(&u_trans);
    let newton_solution = sonium_solvers::NewtonSolution {
        x: u,
        iterations: uzawa_iter,
        residual: multiplier_update_norm,
        converged: uzawa_converged,
        status: if uzawa_converged {
            SolverStatus::Converged
        } else {
            SolverStatus::MaxIterationsReached
        },
        attainable_accuracy: if uzawa_converged {
            AccuracyStatus::Attained
        } else {
            AccuracyStatus::NotAttained
        },
        residual_evaluations: total_residual_evaluations,
        jacobian_evaluations: total_jacobian_evaluations,
        inner_iterations: total_inner_iterations,
    };

    let contact_result = ContactResult {
        uzawa_iterations: uzawa_iter,
        multiplier_update_norm,
        converged: uzawa_converged,
        active_pairs: contact.count_active_pairs(&newton_solution.x),
        max_penetration: max_pen,
    };

    Ok((newton_solution, contact_result))
}

/// Apply Dirichlet BCs to a triplet matrix by zeroing constrained rows/columns
/// and placing 1.0 on the diagonal.
fn apply_dirichlet_to_triplet(triplet: &mut TripletMatrix<f64>, is_constrained: &[bool]) {
    // Zero out entries in constrained rows or columns
    for k in 0..triplet.nnz() {
        if is_constrained[triplet.rows[k]] || is_constrained[triplet.cols[k]] {
            triplet.values[k] = 0.0;
        }
    }
    // Add identity on diagonal for constrained DOFs
    for (dof, &constrained) in is_constrained.iter().enumerate() {
        if constrained {
            triplet.push(dof, dof, 1.0);
        }
    }
}

/// Single element nonlinear assembly (residual + tangent).
fn assemble_element_nonlinear<F, S>(
    form: &F,
    space: &S,
    mesh: &Mesh,
    solution: &[f64],
    elem_idx: usize,
    quad_order: usize,
) -> ElementNonlinearContrib
where
    F: NonlinearForm<f64>,
    S: FunctionSpace,
{
    let elem = &mesh.elements[elem_idx];
    let dofs = space.local_to_global(mesh, elem_idx);
    let n_local = dofs.len();
    let local_sol: Vec<f64> = dofs.iter().map(|&d| solution[d]).collect();

    let quad = QuadratureRule::new(elem.element_type, quad_order);
    let mut local_vec = LocalVector::zeros(n_local);
    let mut local_mat = LocalMatrix::zeros(n_local);

    for qp in quad.iter() {
        let tab = space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
        form.element_residual(&tab, &local_sol, qp.weight, &mut local_vec);
        form.element_tangent(&tab, &tab, &local_sol, qp.weight, &mut local_mat);
    }

    let residual_contribs: Vec<(usize, f64)> = dofs
        .iter()
        .enumerate()
        .map(|(i, &dof)| (dof, local_vec.values[i]))
        .collect();

    let mut tangent_contribs = Vec::with_capacity(n_local * n_local);
    for (i, &row_dof) in dofs.iter().enumerate() {
        for (j, &col_dof) in dofs.iter().enumerate() {
            tangent_contribs.push((row_dof, col_dof, local_mat.get(i, j)));
        }
    }

    (residual_contribs, tangent_contribs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::{unit_cube_hexahedra, unit_square_triangles};
    use crate::spaces::H1Space;

    #[test]
    fn malformed_fem_newton_inputs_are_typed_errors() {
        let config = FemNewtonConfig::default();
        assert!(matches!(
            validate_newton_inputs(2, &[0.0], &config),
            Err(sonium_solvers::NewtonError::InvalidConfig(_))
        ));
        assert!(matches!(
            validate_newton_inputs(1, &[f64::NAN], &config),
            Err(sonium_solvers::NewtonError::InvalidConfig(_))
        ));
        assert!(matches!(
            validate_dirichlet_dofs(1, &[(1, 0.0)]),
            Err(sonium_solvers::NewtonError::InvalidConfig(_))
        ));
        assert!(matches!(
            validate_dirichlet_dofs(1, &[(0, f64::INFINITY)]),
            Err(sonium_solvers::NewtonError::InvalidConfig(_))
        ));
    }

    #[test]
    fn external_load_validation_is_typed_and_empty_is_compatible() {
        assert!(validate_external_load(3, &[]).is_ok());
        assert!(matches!(
            validate_external_load(3, &[0.0, 0.0]),
            Err(sonium_solvers::NewtonError::InvalidConfig(message))
                if message.contains("expected 3")
        ));
        assert!(matches!(
            validate_external_load(3, &[0.0, f64::NAN, 0.0]),
            Err(sonium_solvers::NewtonError::InvalidConfig(message))
                if message.contains("external load")
        ));
    }

    #[test]
    fn invalid_quadrature_and_inner_solver_settings_are_rejected() {
        let mut config = FemNewtonConfig::default();
        config.quad_order = 0;
        assert!(matches!(
            config.validate(),
            Err(sonium_solvers::NewtonError::InvalidConfig(_))
        ));

        config.quad_order = 2;
        config.inner_solver.max_iterations = 0;
        assert!(matches!(
            config.validate(),
            Err(sonium_solvers::NewtonError::InvalidConfig(_))
        ));
    }

    #[test]
    fn nonlinear_space_gate_rejects_bad_connectivity_before_dof_lookup() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].nodes[0] = mesh.nodes.len();
        let space = H1Space::new(PolynomialDegree::P1);
        let error = validate_space_and_mesh(&space, &mesh)
            .expect_err("invalid connectivity must fail before nonlinear assembly");
        assert!(matches!(
            error,
            sonium_solvers::NewtonError::InvalidConfig(message)
                if message.contains("connectivity")
        ));
    }

    #[test]
    fn assembled_hyperelastic_tangent_matches_residual_jacobian() {
        use crate::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
        use crate::spaces::VectorH1Space;

        // Global-level production contract: the assembled tangent triplets
        // must equal the Jacobian of the assembled residual, i.e. the pair
        // `fem_newton_solve` iterates on is a true Newton pair.
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let form = HyperelasticForm::new(NeoHookean::from_young(1000.0, 0.3, 2));
        let quad_order = 2;
        let n_dofs = space.num_global_dofs(&mesh);

        // Smooth deformation with nonzero gradient everywhere.
        let state: Vec<f64> = mesh
            .nodes
            .iter()
            .flat_map(|point| {
                [
                    0.05 * point.x * (1.0 - point.y),
                    -0.03 * point.y * (1.0 - point.x),
                ]
            })
            .collect();
        assert_eq!(state.len(), n_dofs);
        let direction: Vec<f64> = (0..n_dofs)
            .map(|i| 0.1 * ((i as f64 * 1.7).sin() + 0.5))
            .collect();

        let triplet_matvec = |tangent: &TripletMatrix<f64>, v: &[f64]| {
            let mut y = vec![0.0; n_dofs];
            for ((&row, &col), &val) in tangent.rows.iter().zip(&tangent.cols).zip(&tangent.values)
            {
                y[row] += val * v[col];
            }
            y
        };

        let report = crate::validation::check_tangent_consistency(
            |current| assemble_residual(&form, &space, &mesh, current, quad_order),
            |current, tangent_direction| {
                let (_, tangent) = assemble_nonlinear(&form, &space, &mesh, current, quad_order);
                triplet_matvec(&tangent, tangent_direction)
            },
            &state,
            &direction,
            &[1.0e-3, 5.0e-4, 2.5e-4],
        )
        .expect("assembled residual and tangent dimensions must agree");

        assert!(
            report.observed_order > 1.8,
            "assembled tangent is not the residual Jacobian: {report:?}"
        );
    }

    #[test]
    fn nonlinear_space_gate_converts_unsupported_basis_panics_to_typed_error() {
        // P3 basis on explicit P2 geometry is the unsupported combination;
        // affine P1 hexahedra now have a supported P3 path.
        let mesh = unit_cube_hexahedra(1).to_p2();
        let space = H1Space::new(PolynomialDegree::P3);
        let error = validate_space_and_mesh(&space, &mesh)
            .expect_err("unsupported basis order must not panic at a solver boundary");
        assert!(matches!(
            error,
            sonium_solvers::NewtonError::InvalidConfig(message)
                if message.contains("function space") || message.contains("panicked")
        ));
    }
}
