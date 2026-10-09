// MFEM side of the wide-comparison harness, slice 4: P1 interior stiffness
// + mass dump on the coupled-aperture fixture mesh.
//
// Reads the Gmsh v2.2 mesh exported by the sonium harness
// (`qa-mfem-aperture --export-mesh`), assembles the P1 H1 stiffness and mass
// matrices with exact quadrature, and dumps both CSRs plus vertex
// coordinates as JSON. The harness maps MFEM vertices onto interior FEM
// nodes by coordinates, forms A = S - k²M, and swaps that block into the
// monolithic aperture system as the MFEM-interior reference coupling.
//
// Compile (MFEM 4.5.2 dev install):
//   g++ -O2 -I/usr/include -I/usr/include/hypre -I/usr/lib/x86_64-linux-gnu/openmpi/include \
//     aperture_interior.cpp -o aperture_interior \
//     -L/usr/lib/x86_64-linux-gnu -lmfem -lmpi_cxx -lmpi
// Run:
//   ./aperture_interior <mesh.msh>

#include "mfem.hpp"
#include <cmath>
#include <cstdio>

using namespace mfem;

static void print_csr(const SparseMatrix &A, const char *name)
{
   const int n = A.Height();
   const int *I = A.GetI(), *J = A.GetJ();
   const double *D = A.GetData();
   std::printf("\"%s_rowptr\":[", name);
   for (int i = 0; i <= n; i++) { std::printf("%s%d", i ? "," : "", I[i]); }
   std::printf("],\"%s_colind\":[", name);
   for (int t = 0; t < I[n]; t++) { std::printf("%s%d", t ? "," : "", J[t]); }
   std::printf("],\"%s_data\":[", name);
   for (int t = 0; t < I[n]; t++)
   {
      std::printf("%s%.17e", t ? "," : "", D[t]);
   }
   std::printf("],");
}

int main(int argc, const char *argv[])
{
   if (argc != 2) { std::printf("usage: %s <mesh.msh>\n", argv[0]); return 2; }

   Mesh mesh(argv[1], 1, 1);
   const int nv = mesh.GetNV();
   const int ne = mesh.GetNE();
   if (nv < 4 || ne < 1) { std::printf("bad mesh: nv=%d ne=%d\n", nv, ne); return 2; }
   // Every volume element must be a positively oriented tetrahedron; a
   // flipped tet would silently flip the assembled signs.
   double min_vol = 1e300, total_vol = 0.0;
   for (int el = 0; el < ne; el++)
   {
      if (mesh.GetElementType(el) != Element::TETRAHEDRON)
      {
         std::printf("element %d is not a tetrahedron\n", el);
         return 2;
      }
      const double v = mesh.GetElementVolume(el);
      min_vol = std::min(min_vol, v);
      total_vol += v;
   }
   if (!(min_vol > 0.0)) { std::printf("inverted tet: min_vol=%.6e\n", min_vol); return 2; }

   H1_FECollection fec(1, 3);
   FiniteElementSpace fes(&mesh, &fec);
   const int ndofs = fes.GetTrueVSize();
   if (ndofs != nv) { std::printf("P1 dofs %d != vertices %d\n", ndofs, nv); return 2; }

   // Exact on straight tets: stiffness integrand is constant, mass is
   // quadratic.
   const IntegrationRule *ir = &IntRules.Get(Geometry::TETRAHEDRON, 2);
   BilinearForm a_s(&fes);
   a_s.AddDomainIntegrator(new DiffusionIntegrator(ir));
   BilinearForm a_m(&fes);
   a_m.AddDomainIntegrator(new MassIntegrator(ir));
   a_s.Assemble();
   a_s.Finalize();
   a_m.Assemble();
   a_m.Finalize();

   std::printf("{");
   std::printf("\"nv\":%d,", nv);
   std::printf("\"ne\":%d,", ne);
   std::printf("\"mfem_version\":\"%s\",", mfem::GetVersionStr());
   std::printf("\"min_tet_volume\":%.17e,", min_vol);
   std::printf("\"total_volume\":%.17e,", total_vol);
   print_csr(a_s.SpMat(), "stiffness");
   print_csr(a_m.SpMat(), "mass");
   std::printf("\"coords\":[");
   for (int i = 0; i < nv; i++)
   {
      const double *v = mesh.GetVertex(i);
      std::printf("%s[%.17e,%.17e,%.17e]", i ? "," : "", v[0], v[1], v[2]);
   }
   std::printf("]}\n");
   return 0;
}
