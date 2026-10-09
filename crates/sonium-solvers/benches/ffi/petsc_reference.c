/*
 * petsc_reference.c — PETSc reference solver for benchmarking
 *
 * Usage:
 *   ./petsc_reference <matrix.mtx> [solver] [precond] [tol] [max_iter]
 *
 * solver:  gmres (default), fgmres, bicgstab, cg
 * precond: none (default), jacobi, ilu, amg
 * tol:     1e-6 (default)
 * max_iter: 500 (default)
 *
 * Output: JSON to stdout with timing and convergence info.
 *
 * Build:
 *   source $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables
 *   mpicc -O2 -o petsc_reference petsc_reference.c $PETSC_CC_INCLUDES $PETSC_LIB
 *
 * Or use build_petsc.sh.
 */

static char help[] = "PETSc reference solver for sonium-solvers benchmarking.\n";

#include <petscksp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Minimal Matrix Market reader.
 * Handles: %%MatrixMarket matrix coordinate real general/symmetric
 * Returns 0 on success.
 */
static PetscErrorCode ReadMatrixMarket(const char *path, Mat *A, PetscInt *nrows_out, PetscInt *nnz_out) {
    FILE *f;
    char line[1024];
    PetscInt nrows, ncols, nnz;
    PetscInt is_symmetric = 0;

    PetscFunctionBeginUser;
    f = fopen(path, "r");
    if (!f) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "Cannot open %s", path);

    /* Read header */
    if (!fgets(line, sizeof(line), f)) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "Empty file");
    if (strstr(line, "symmetric")) is_symmetric = 1;

    /* Skip comments */
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%') break;
    }

    /* Parse size line */
    if (sscanf(line, "%d %d %d", (int*)&nrows, (int*)&ncols, (int*)&nnz) != 3) {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "Cannot parse size line");
    }

    PetscCall(MatCreate(PETSC_COMM_WORLD, A));
    PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols));
    PetscCall(MatSetFromOptions(*A));
    PetscCall(MatSetUp(*A));

    /* Read triplets (1-indexed in .mtx) */
    PetscInt total_nnz = 0;
    for (PetscInt k = 0; k < nnz; k++) {
        int i, j;
        double v;
        if (fscanf(f, "%d %d %lf", &i, &j, &v) != 3) {
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "Cannot parse triplet %d", (int)k);
        }
        i--; j--;  /* Convert to 0-indexed */
        PetscCall(MatSetValue(*A, i, j, v, ADD_VALUES));
        total_nnz++;
        if (is_symmetric && i != j) {
            PetscCall(MatSetValue(*A, j, i, v, ADD_VALUES));
            total_nnz++;
        }
    }

    PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));

    fclose(f);
    *nrows_out = nrows;
    *nnz_out = total_nnz;
    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
    KSP        ksp;
    PC         pc;
    Mat        A;
    Vec        b, x;
    PetscInt   nrows, nnz, its;
    PetscReal  rnorm;
    PetscLogDouble t_setup_start, t_setup_end, t_solve_start, t_solve_end;

    const char *matrix_path;
    const char *solver_name  = "gmres";
    const char *precond_name = "none";
    double      tol          = 1e-6;
    int         max_iter     = 500;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    /* Parse arguments */
    if (argc < 2) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "Usage: %s <matrix.mtx> [solver] [precond] [tol] [max_iter]\n", argv[0]));
        PetscCall(PetscFinalize());
        return 1;
    }
    matrix_path = argv[1];
    if (argc > 2) solver_name  = argv[2];
    if (argc > 3) precond_name = argv[3];
    if (argc > 4) tol          = atof(argv[4]);
    if (argc > 5) max_iter     = atoi(argv[5]);

    /* Read matrix */
    PetscCall(ReadMatrixMarket(matrix_path, &A, &nrows, &nnz));

    /* Create RHS = ones, initial guess = zeros */
    PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
    PetscCall(VecSetSizes(b, PETSC_DECIDE, nrows));
    PetscCall(VecSetFromOptions(b));
    PetscCall(VecSet(b, 1.0));

    PetscCall(VecDuplicate(b, &x));
    PetscCall(VecSet(x, 0.0));

    /* Configure KSP */
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));

    /* Solver type */
    if      (strcmp(solver_name, "gmres")    == 0) PetscCall(KSPSetType(ksp, KSPGMRES));
    else if (strcmp(solver_name, "fgmres")   == 0) PetscCall(KSPSetType(ksp, KSPFGMRES));
    else if (strcmp(solver_name, "bicgstab") == 0) PetscCall(KSPSetType(ksp, KSPBCGS));
    else if (strcmp(solver_name, "cg")       == 0) PetscCall(KSPSetType(ksp, KSPCG));
    else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Unknown solver: %s\n", solver_name));
        PetscCall(PetscFinalize());
        return 1;
    }

    /* PETSC_DETERMINE is rejected for KSP tolerances by PETSc 3.19.  Keep
       the library defaults explicitly; PETSC_DEFAULT is portable across the
       supported PETSc 3.x releases. */
    PetscCall(KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_iter));
    if (strcmp(solver_name, "gmres") == 0 || strcmp(solver_name, "fgmres") == 0) {
        PetscCall(KSPGMRESSetRestart(ksp, 30));
    }

    /* Preconditioner type */
    PetscCall(KSPGetPC(ksp, &pc));
    if      (strcmp(precond_name, "none")   == 0) PetscCall(PCSetType(pc, PCNONE));
    else if (strcmp(precond_name, "jacobi") == 0) PetscCall(PCSetType(pc, PCJACOBI));
    else if (strcmp(precond_name, "ilu")    == 0) PetscCall(PCSetType(pc, PCILU));
    else if (strcmp(precond_name, "amg")    == 0) {
        PetscCall(PCSetType(pc, PCGAMG));
    }
    else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Unknown preconditioner: %s\n", precond_name));
        PetscCall(PetscFinalize());
        return 1;
    }

    /* Setup (preconditioner construction) */
    PetscCall(PetscTime(&t_setup_start));
    PetscCall(KSPSetUp(ksp));
    PetscCall(PetscTime(&t_setup_end));

    /* Solve */
    PetscCall(PetscTime(&t_solve_start));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(PetscTime(&t_solve_end));

    /* Results */
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));

    KSPConvergedReason reason;
    PetscCall(KSPGetConvergedReason(ksp, &reason));

    double setup_ms = (t_setup_end - t_setup_start) * 1000.0;
    double solve_ms = (t_solve_end - t_solve_start) * 1000.0;

    /* JSON output */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "{\"solver\":\"%s\",\"precond\":\"%s\",\"n\":%d,\"nnz\":%d,"
        "\"setup_ms\":%.3f,\"solve_ms\":%.3f,\"total_ms\":%.3f,"
        "\"iterations\":%d,\"residual\":%.6e,\"converged\":%s}\n",
        solver_name, precond_name, (int)nrows, (int)nnz,
        setup_ms, solve_ms, setup_ms + solve_ms,
        (int)its, (double)rnorm,
        reason > 0 ? "true" : "false"));

    /* Cleanup */
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return 0;
}
