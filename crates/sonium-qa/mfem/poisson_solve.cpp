// MFEM side of the wide-comparison harness, slice 2: manufactured Poisson
// solve with homogeneous Dirichlet BCs on the unit cube.
//
// Solves -Δu = f with u = sin(πx)·sin(πy)·sin(πz) (f = 3π²u, zero on the
// whole boundary) with unpreconditioned CG to 1e-12, reporting the L2 error
// against the analytic solution. The sonium side assembles the same Galerkin
// discretization; the Galerkin solution function is basis-independent, so
// both L2 errors must agree to solver tolerance despite different nodal
// bases (uniform vs Lobatto) — this is the equal-error anchor of the D5
// harness, complementing the operator-action comparison in
// poisson_action.cpp.
//
// Compile (MFEM 4.5.2 dev install):
//   g++ -O2 -I/usr/include -I/usr/include/hypre -I/usr/lib/x86_64-linux-gnu/openmpi/include \
//     poisson_solve.cpp -o poisson_solve \
//     -L/usr/lib/x86_64-linux-gnu -lmfem -lmpi_cxx -lmpi
// Run:
//   ./poisson_solve <nx> <order>

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

static double u_exact_fn(const Vector &x)
{
   return std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]) * std::sin(M_PI * x[2]);
}

static double f_rhs_fn(const Vector &x)
{
   return 3.0 * M_PI * M_PI * u_exact_fn(x);
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

   const int exactness = 2 * order;
   const IntegrationRule *ir = &IntRules.Get(Geometry::CUBE, exactness);
   BilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(ir));
   FunctionCoefficient f_coeff(f_rhs_fn);
   LinearForm b(&fes);
   // Over-integrated load (exactness 8): the transcendental source is resolved
   // far below discretization error so both sides assemble the same RHS.
   b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff, 8));
   FunctionCoefficient u_coeff(u_exact_fn);

   const double t0 = wall_seconds();
   a.Assemble();
   b.Assemble();
   GridFunction x(&fes);
   x.ProjectCoefficient(u_coeff);
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   const double assemble_s = wall_seconds() - t0;

   // Zero initial guess: iteration counts must reflect the solve, not the
   // projected exact solution used for boundary values above.
   X = 0.0;
   CGSolver cg;
   cg.SetOperator(*A);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(10000);
   cg.SetPrintLevel(0);
   const double s0 = wall_seconds();
   cg.Mult(B, X);
   const double solve_s = wall_seconds() - s0;
   a.RecoverFEMSolution(X, b, x);

   // Explicit high-order L2 (exactness 12): the built-in ComputeL2Error
   // uses a coarse default rule that under-resolves (u_h - u)^2 here.
   const IntegrationRule *err_ir = &IntRules.Get(Geometry::CUBE, 12);
   double err2 = 0.0;
   Array<int> edofs;
   Vector euh;
   for (int el = 0; el < fes.GetNE(); el++)
   {
      fes.GetElementDofs(el, edofs);
      x.GetSubVector(edofs, euh);
      const FiniteElement *fe = fes.GetFE(el);
      ElementTransformation *T = fes.GetElementTransformation(el);
      for (int q = 0; q < err_ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = err_ir->IntPoint(q);
         Vector shape(fe->GetDof());
         fe->CalcShape(ip, shape);
         double uh = 0.0;
         for (int k = 0; k < shape.Size(); k++) { uh += shape[k] * euh[k]; }
         T->SetIntPoint(&ip);
         Vector px;
         T->Transform(ip, px);
         const double ue = u_exact_fn(px);
         err2 += (uh - ue) * (uh - ue) * ip.weight * T->Weight();
      }
   }
   const double l2_error = std::sqrt(err2);

   std::printf("{");
   std::printf("\"nx\":%d,", nx);
   std::printf("\"order\":%d,", order);
   std::printf("\"ndofs\":%d,", ndofs);
   std::printf("\"mfem_version\":\"%s\",", mfem::GetVersionStr());
   std::printf("\"l2_error\":%.17e,", l2_error);
   std::printf("\"cg_iters\":%d,", cg.GetNumIterations());
   std::printf("\"cg_final_res\":%.17e,", cg.GetFinalNorm());
   std::printf("\"assemble_s\":%.6f,", assemble_s);
   std::printf("\"solve_s\":%.6f", solve_s);
   std::printf("}\n");
   return 0;
}
