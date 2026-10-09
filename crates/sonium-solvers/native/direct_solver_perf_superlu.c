#include <stdlib.h>
#include <string.h>

#include "slu_ddefs.h"

typedef struct {
    int n;
    int_t nnz;
    int_t *row_ptrs;
    int_t *column_indices;
    double *values;
} SoniumSuperluMatrix;

void *sonium_superlu_prepare(
    int n,
    int nnz,
    const int *row_ptrs,
    const int *column_indices,
    const double *values
) {
    SoniumSuperluMatrix *matrix = malloc(sizeof(SoniumSuperluMatrix));
    if (matrix == NULL) {
        return NULL;
    }
    matrix->n = n;
    matrix->nnz = (int_t)nnz;
    matrix->row_ptrs = malloc((size_t)(n + 1) * sizeof(int_t));
    matrix->column_indices = malloc((size_t)nnz * sizeof(int_t));
    matrix->values = malloc((size_t)nnz * sizeof(double));
    if (
        matrix->row_ptrs == NULL ||
        matrix->column_indices == NULL ||
        matrix->values == NULL
    ) {
        free(matrix->row_ptrs);
        free(matrix->column_indices);
        free(matrix->values);
        free(matrix);
        return NULL;
    }
    for (int i = 0; i <= n; ++i) {
        matrix->row_ptrs[i] = (int_t)row_ptrs[i];
    }
    for (int i = 0; i < nnz; ++i) {
        matrix->column_indices[i] = (int_t)column_indices[i];
        matrix->values[i] = values[i];
    }
    return matrix;
}

void sonium_superlu_destroy(void *opaque_matrix) {
    SoniumSuperluMatrix *matrix = opaque_matrix;
    if (matrix == NULL) {
        return;
    }
    free(matrix->row_ptrs);
    free(matrix->column_indices);
    free(matrix->values);
    free(matrix);
}

int sonium_superlu_solve(
    const void *opaque_matrix,
    const double *rhs,
    double *solution,
    long long *factor_nonzeros
) {
    const SoniumSuperluMatrix *matrix = opaque_matrix;
    const int n = matrix->n;
    SuperMatrix a;
    SuperMatrix l;
    SuperMatrix u;
    SuperMatrix b;
    SuperLUStat_t statistics;
    superlu_options_t options;
    int *column_permutation = NULL;
    int *row_permutation = NULL;
    int_t info = 0;

    memcpy(solution, rhs, (size_t)n * sizeof(double));
    column_permutation = int32Malloc(n);
    row_permutation = int32Malloc(n);
    if (column_permutation == NULL || row_permutation == NULL) {
        SUPERLU_FREE(column_permutation);
        SUPERLU_FREE(row_permutation);
        return -1;
    }

    dCreate_CompRow_Matrix(
        &a,
        n,
        n,
        matrix->nnz,
        matrix->values,
        matrix->column_indices,
        matrix->row_ptrs,
        SLU_NR,
        SLU_D,
        SLU_GE
    );
    dCreate_Dense_Matrix(&b, n, 1, solution, n, SLU_DN, SLU_D, SLU_GE);

    set_default_options(&options);
    options.PrintStat = NO;
    options.IterRefine = NOREFINE;
    StatInit(&statistics);
    dgssv(
        &options,
        &a,
        column_permutation,
        row_permutation,
        &l,
        &u,
        &b,
        &statistics,
        &info
    );
    StatFree(&statistics);

    if (info == 0) {
        SCformat *l_store = (SCformat *)l.Store;
        NCformat *u_store = (NCformat *)u.Store;
        *factor_nonzeros =
            (long long)l_store->nnz + (long long)u_store->nnz - (long long)n;
        Destroy_SuperNode_Matrix(&l);
        Destroy_CompCol_Matrix(&u);
    }

    Destroy_SuperMatrix_Store(&a);
    Destroy_SuperMatrix_Store(&b);
    SUPERLU_FREE(column_permutation);
    SUPERLU_FREE(row_permutation);
    return (int)info;
}
