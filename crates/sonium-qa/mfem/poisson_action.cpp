// MFEM side of the wide-comparison harness, slice 1: H1 Poisson stiffness
// operator action on the unit cube with Cartesian hexahedra.
//
// Builds the Qp H1 stiffness matrix with the default (exact for affine
// diffusion) integration rule, then reports permutation-invariant evidence:
// assembled-matrix sum / trace / Frobenius norm squared, and the action on
// the all-ones vector (ordering-independent input) with sum / sum-of-squares.
// Timings: one assembly, best-of-21 matrix-vector actions.
//
// The sonium side (sonium-qa binary `mfem_poisson_action`) assembles the
// same Galerkin discretization via `LaplacianForm` + `MatrixFreeOperator`
// on `unit_cube_hexahedra(nx)` and compares every scalar at 1e-9 relative.
// Same mesh, same order, exact quadrature on both sides: the discretization
// error is identical by construction, so agreement cross-checks correctness
// and the timings compare time-to-action.
//
// Compile (MFEM 4.5.2 dev install):
//   g++ -O2 -I/usr/include -I/usr/lib/x86_64-linux-gnu/openmpi/include \
//     poisson_action.cpp -o poisson_action \
//     -L/usr/lib/x86_64-linux-gnu -lmfem -lmpi
// Run:
//   ./poisson_action <nx> <order>
//
// Part of the D5 wide-MFEM harness (decision record
// reviews/audit-sonium-fem-extraction-20260926.md).

#include "mfem.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>

using namespace mfem;

static double wall_seconds()
{
   using clock = std::chrono::steady_clock;
   return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int main(int argc, const char *argv[])
{
   if (argc != 3) { std::printf("usage: %s <nx> <order>\n", argv[0]); return 2; }
   const int nx = std::atoi(argv[1]);
   const int order = std::atoi(argv[2]);
   if (nx < 1 || order < 1) { std::printf("nx and order must be positive\n"); return 2; }

   Mesh mesh = Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON);
   H1_FECollection fec(order, 3);
   FiniteElementSpace fes(&mesh, &fec);
   const int ndofs = fes.GetTrueVSize();

   BilinearForm a(&fes);
   // Explicit tensor Gauss rule of exactness 2p: the degree-2(p-1)
   // stiffness integrand on affine hexes is integrated exactly, matching
   // the sonium side (quadrature exactness 2p). MFEM's default rule is left
   // out so a default-rule change can never silently move the reference.
   const int exactness = 2 * order;
   const IntegrationRule *ir = &IntRules.Get(Geometry::CUBE, exactness);
   const int quad_points = ir->GetNPoints();
   a.AddDomainIntegrator(new DiffusionIntegrator(ir));
   const double t0 = wall_seconds();
   a.Assemble();
   a.Finalize();
   const double assemble_s = wall_seconds() - t0;

   SparseMatrix &S = a.SpMat();
   const int nnz = S.NumNonZeroElems();

   double sum = 0.0, trace = 0.0, frob2 = 0.0, maxabs = 0.0;
   long nnz_nonzero = 0, nnz_significant = 0;
   Array<int> cols;
   Vector vals;
   for (int i = 0; i < S.Height(); i++)
   {
      S.GetRow(i, cols, vals);
      for (int k = 0; k < cols.Size(); k++)
      {
         const double v = vals[k];
         const double a = v >= 0.0 ? v : -v;
         sum += v;
         frob2 += v * v;
         if (a > maxabs) { maxabs = a; }
         if (v != 0.0) { nnz_nonzero++; }
         if (cols[k] == i) { trace += v; }
      }
   }

   Vector x(ndofs), y(ndofs);
   x = 1.0;
   double action_s_best = 0.0;
   double action_sum = 0.0, action_sumsq = 0.0;
   for (int rep = 0; rep < 21; rep++)
   {
      const double s0 = wall_seconds();
      S.Mult(x, y);
      const double dt = wall_seconds() - s0;
      if (rep == 0 || dt < action_s_best) { action_s_best = dt; }
   }
   for (int i = 0; i < ndofs; i++) { action_sum += y[i]; action_sumsq += y[i] * y[i]; }

   std::printf("{");
   std::printf("\"nx\":%d,", nx);
   std::printf("\"order\":%d,", order);
   std::printf("\"quad_exactness\":%d,", exactness);
   std::printf("\"quad_points\":%d,", quad_points);
   std::printf("\"ndofs\":%d,", ndofs);
   std::printf("\"nnz\":%d,", nnz);
   std::printf("\"nnz_nonzero\":%ld,", nnz_nonzero);
   // Second pass: significance threshold needs the global max first.
   // (entries below 1e-12 of max are analytic-zero roundoff residue)
   for (int i = 0; i < S.Height(); i++)
   {
      S.GetRow(i, cols, vals);
      for (int k = 0; k < cols.Size(); k++)
      {
         const double v = vals[k];
         if ((v >= 0.0 ? v : -v) > 1e-12 * maxabs) { nnz_significant++; }
      }
   }
   std::printf("\"nnz_significant\":%ld,", nnz_significant);
   std::printf("\"maxabs\":%.17e,", maxabs);
   std::printf("\"mfem_version\":\"%s\",", mfem::GetVersionStr());
   std::printf("\"matrix_sum\":%.17e,", sum);
   std::printf("\"matrix_trace\":%.17e,", trace);
   std::printf("\"matrix_frob2\":%.17e,", frob2);
   std::printf("\"action_sum\":%.17e,", action_sum);
   std::printf("\"action_sumsq\":%.17e,", action_sumsq);
   std::printf("\"assemble_s\":%.6f,", assemble_s);
   std::printf("\"action_s_best\":%.9f", action_s_best);
   std::printf("}\n");
   return 0;
}
