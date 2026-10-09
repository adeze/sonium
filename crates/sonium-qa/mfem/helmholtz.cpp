// MFEM side of the wide-comparison harness, slice 3: complex Helmholtz
// operator action + manufactured solve at fixed kh on the unit cube.
//
// Solves -Δu - k²u = f with u = (1+i)·sin(πx)·sin(πy)·sin(πz) (zero on the
// whole boundary, so homogeneous Dirichlet applies) and complex wavenumber
// k = kr + i·ki. The complex system A = S - k²M is solved in real 2x2 block
// form [[Ar,-Ai],[Ai,Ar]] with unpreconditioned GMRES to 1e-12; the sonium
// side solves the native complex system with unpreconditioned complex GMRES
// at the same restart and tolerance. The action leg (no Dirichlet
// elimination) gates basis-independent evidence: matrix_sum = -k²·volume
// (sum(S) = a(1,1) = 0, sum(M) = volume by partition of unity) and the
// ones-action signature. The solve leg gates the true complex L2 error
// against the analytic solution — the equal-error anchor, exactly as in
// slice 2 (poisson_solve.cpp).
//
// Compile (MFEM 4.5.2 dev install):
//   g++ -O2 -I/usr/include -I/usr/lib/x86_64-linux-gnu/openmpi/include \
//     helmholtz.cpp -o helmholtz \
//     -L/usr/lib/x86_64-linux-gnu -lmfem -lmpi_cxx -lmpi
// Run:
//   ./helmholtz <nx> <order> <kr> <ki>

#include "mfem.hpp"
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>

using namespace mfem;

static double wall_seconds()
{
   using clock = std::chrono::steady_clock;
   return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Manufactured solution and source share the real envelope
// s(x) = sin(πx)·sin(πy)·sin(πz); u = (1+i)s, f = (3π²-k²)u.
static double s_env(const Vector &x)
{
   return std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]) * std::sin(M_PI * x[2]);
}

static std::complex<double> g_k2(0.0, 0.0);

static double f_rhs_re(const Vector &x)
{
   std::complex<double> u(1.0, 1.0);
   std::complex<double> f =
      (3.0 * M_PI * M_PI - g_k2) * u * s_env(x);
   return f.real();
}

static double f_rhs_im(const Vector &x)
{
   std::complex<double> u(1.0, 1.0);
   std::complex<double> f =
      (3.0 * M_PI * M_PI - g_k2) * u * s_env(x);
   return f.imag();
}

static std::complex<double> u_exact(const Vector &x)
{
   return std::complex<double>(s_env(x), s_env(x));
}

static double sparse_entry_sum(const SparseMatrix &A, double &maxabs)
{
   const double *data = A.GetData();
   double sum = 0.0;
   maxabs = 0.0;
   for (int i = 0; i < A.NumNonZeroElems(); i++)
   {
      sum += data[i];
      maxabs = std::max(maxabs, std::abs(data[i]));
   }
   return sum;
}

int main(int argc, const char *argv[])
{
   if (argc != 5)
   {
      std::printf("usage: %s <nx> <order> <kr> <ki>\n", argv[0]);
      return 2;
   }
   const int nx = std::atoi(argv[1]);
   const int order = std::atoi(argv[2]);
   const double kr = std::atof(argv[3]);
   const double ki = std::atof(argv[4]);
   if (nx < 1 || order < 1) { std::printf("nx and order must be positive\n"); return 2; }
   const std::complex<double> k(kr, ki);
   g_k2 = k * k;
   const double reck2 = g_k2.real();
   const double imck2 = g_k2.imag();

   Mesh mesh = Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON);
   H1_FECollection fec(order, 3);
   FiniteElementSpace fes(&mesh, &fec);
   const int ndofs = fes.GetTrueVSize();

   const int exactness = 2 * order;
   const IntegrationRule *ir = &IntRules.Get(Geometry::CUBE, exactness);
   BilinearForm a_s(&fes);
   a_s.AddDomainIntegrator(new DiffusionIntegrator(ir));
   BilinearForm a_m(&fes);
   a_m.AddDomainIntegrator(new MassIntegrator(ir));

   const double t0 = wall_seconds();
   a_s.Assemble();
   a_s.Finalize();
   a_m.Assemble();
   a_m.Finalize();
   const SparseMatrix &S = a_s.SpMat();
   const SparseMatrix &M = a_m.SpMat();
   // Ar = S - Re(k²)M, Ai = -Im(k²)M (S, M real symmetric).
   SparseMatrix *Ar_full = Add(1.0, S, -reck2, M);
   SparseMatrix *Ai_full = Add(0.0, S, -imck2, M);
   const double assemble_s = wall_seconds() - t0;

   // ---- Action leg: complex A applied to the real ones vector ----
   double maxabs_r = 0.0, maxabs_i = 0.0;
   const double msum_r = sparse_entry_sum(*Ar_full, maxabs_r);
   const double msum_i = sparse_entry_sum(*Ai_full, maxabs_i);
   const double maxabs = std::max(maxabs_r, maxabs_i);
   int nnz_nonzero = 0, nnz_significant = 0;
   {
      // Union pattern of Ar/Ai: an entry is structurally present if either
      // part is significant.
      const int *Ar_i = Ar_full->GetI(), *Ar_j = Ar_full->GetJ();
      const double *Ar_d = Ar_full->GetData();
      const int *Ai_i = Ai_full->GetI(), *Ai_j = Ai_full->GetJ();
      const double *Ai_d = Ai_full->GetData();
      for (int row = 0; row < ndofs; row++)
      {
         for (int t = Ar_i[row]; t < Ar_i[row + 1]; t++)
         {
            const double vr = Ar_d[t];
            // Locate the matching Ai entry (same pattern, same ordering).
            double vi = 0.0;
            for (int u = Ai_i[row]; u < Ai_i[row + 1]; u++)
            {
               if (Ai_j[u] == Ar_j[t]) { vi = Ai_d[u]; break; }
            }
            const double mod = std::sqrt(vr * vr + vi * vi);
            if (mod != 0.0) { nnz_nonzero++; }
            if (mod > 1e-12 * maxabs) { nnz_significant++; }
         }
      }
   }
   Array<int> blk_off(3);
   blk_off[0] = 0;
   blk_off[1] = ndofs;
   blk_off[2] = 2 * ndofs;
   BlockOperator A_full(blk_off);
   SparseMatrix *negAi_full = Add(-1.0, *Ai_full, 0.0, *Ai_full);
   A_full.SetBlock(0, 0, Ar_full);
   A_full.SetBlock(0, 1, negAi_full);
   A_full.SetBlock(1, 0, Ai_full);
   A_full.SetBlock(1, 1, Ar_full);
   Vector x_ones(2 * ndofs);
   for (int i = 0; i < ndofs; i++) { x_ones[i] = 1.0; x_ones[ndofs + i] = 0.0; }
   Vector y_act(2 * ndofs);
   double action_s_best = 1e300;
   for (int rep = 0; rep < 21; rep++)
   {
      const double r0 = wall_seconds();
      A_full.Mult(x_ones, y_act);
      action_s_best = std::min(action_s_best, wall_seconds() - r0);
   }
   double action_sum_r = 0.0, action_sum_i = 0.0, action_sumsq = 0.0;
   for (int i = 0; i < ndofs; i++)
   {
      action_sum_r += y_act[i];
      action_sum_i += y_act[ndofs + i];
      action_sumsq += y_act[i] * y_act[i] + y_act[ndofs + i] * y_act[ndofs + i];
   }

   // ---- Solve leg: homogeneous Dirichlet + real-block GMRES ----
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   SparseMatrix *Ar = new SparseMatrix(*Ar_full);
   SparseMatrix *Ai = new SparseMatrix(*Ai_full);
   FunctionCoefficient fr_coeff(f_rhs_re);
   FunctionCoefficient fi_coeff(f_rhs_im);
   LinearForm br(&fes), bi(&fes);
   br.AddDomainIntegrator(new DomainLFIntegrator(fr_coeff, 8));
   bi.AddDomainIntegrator(new DomainLFIntegrator(fi_coeff, 8));
   br.Assemble();
   bi.Assemble();
   Vector Br(br.Size()), Bi(bi.Size());
   Br = br;
   Bi = bi;
   // Homogeneous Dirichlet: Ar gets identity boundary rows/cols with zeroed
   // boundary load; Ai gets zero (DIAG_ZERO) boundary rows/cols so the
   // eliminated block rows stay identity.
   for (int t = 0; t < ess_tdof_list.Size(); t++)
   {
      const int dof = ess_tdof_list[t];
      Ar->EliminateRowCol(dof, 0.0, Br);
      Ai->EliminateRowCol(dof, 0.0, Bi, Operator::DIAG_ZERO);
   }
   BlockOperator A(blk_off);
   SparseMatrix *negAi = Add(-1.0, *Ai, 0.0, *Ai);
   A.SetBlock(0, 0, Ar);
   A.SetBlock(0, 1, negAi);
   A.SetBlock(1, 0, Ai);
   A.SetBlock(1, 1, Ar);
   Vector B(2 * ndofs), X(2 * ndofs);
   for (int i = 0; i < ndofs; i++) { B[i] = Br[i]; B[ndofs + i] = Bi[i]; }
   X = 0.0;
   GMRESSolver gmres;
   gmres.SetOperator(A);
   gmres.SetRelTol(1e-12);
   gmres.SetAbsTol(0.0);
   gmres.SetMaxIter(10000);
   gmres.SetKDim(50);
   gmres.SetPrintLevel(0);
   const double s0 = wall_seconds();
   gmres.Mult(B, X);
   const double solve_s = wall_seconds() - s0;

   // True complex L2 error by explicit high-order element quadrature
   // (exactness 12), mirroring poisson_solve.cpp.
   const IntegrationRule *err_ir = &IntRules.Get(Geometry::CUBE, 12);
   double err2 = 0.0;
   Array<int> edofs;
   Vector exr, exi;
   for (int el = 0; el < fes.GetNE(); el++)
   {
      fes.GetElementDofs(el, edofs);
      exr.SetSize(edofs.Size());
      exi.SetSize(edofs.Size());
      for (int k = 0; k < edofs.Size(); k++)
      {
         exr[k] = X[edofs[k]];
         exi[k] = X[ndofs + edofs[k]];
      }
      const FiniteElement *fe = fes.GetFE(el);
      ElementTransformation *T = fes.GetElementTransformation(el);
      for (int q = 0; q < err_ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = err_ir->IntPoint(q);
         Vector shape(fe->GetDof());
         fe->CalcShape(ip, shape);
         std::complex<double> uh(0.0, 0.0);
         for (int k = 0; k < shape.Size(); k++)
         {
            uh += shape[k] * std::complex<double>(exr[k], exi[k]);
         }
         T->SetIntPoint(&ip);
         Vector px;
         T->Transform(ip, px);
         const std::complex<double> ue = u_exact(px);
         const std::complex<double> d = uh - ue;
         err2 += (d.real() * d.real() + d.imag() * d.imag()) * ip.weight * T->Weight();
      }
   }
   const double l2_error = std::sqrt(err2);

   std::printf("{");
   std::printf("\"nx\":%d,", nx);
   std::printf("\"order\":%d,", order);
   std::printf("\"kr\":%.17e,", kr);
   std::printf("\"ki\":%.17e,", ki);
   std::printf("\"ndofs\":%d,", ndofs);
   std::printf("\"mfem_version\":\"%s\",", mfem::GetVersionStr());
   std::printf("\"matrix_sum_re\":%.17e,", msum_r);
   std::printf("\"matrix_sum_im\":%.17e,", msum_i);
   std::printf("\"nnz_nonzero\":%d,", nnz_nonzero);
   std::printf("\"nnz_significant\":%d,", nnz_significant);
   std::printf("\"action_sum_re\":%.17e,", action_sum_r);
   std::printf("\"action_sum_im\":%.17e,", action_sum_i);
   std::printf("\"action_sumsq\":%.17e,", action_sumsq);
   std::printf("\"assemble_s\":%.6f,", assemble_s);
   std::printf("\"action_s_best\":%.9f,", action_s_best);
   std::printf("\"l2_error\":%.17e,", l2_error);
   std::printf("\"gmres_iters\":%d,", gmres.GetNumIterations());
   std::printf("\"gmres_final_res\":%.17e,", gmres.GetFinalNorm());
   std::printf("\"solve_s\":%.6f", solve_s);
   std::printf("}\n");
   delete Ar_full;
   delete Ai_full;
   delete negAi_full;
   delete Ar;
   delete Ai;
   delete negAi;
   return 0;
}
