(* Wolfram goldens for sonium-fem Lagrange reference-element tabulation.
   Run: wolframscript -file crates/sonium-fem/fixtures/basis_golden.wls
   Output: crates/sonium-fem/fixtures/wolfram-golden-v1.json
   Independent evaluation of the P1/P2 tetrahedron and Q1 hexahedron shape
   functions and gradients at an interior point, with the crate's exact node
   ordering (P2 tet: 4 vertices then edge midpoints (0,1),(0,2),(0,3),
   (1,2),(1,3),(2,3); Q1 hex: standard [-1,1]^3 tensor ordering). Any basis
   transcription or node-ordering slip breaks assembly everywhere downstream,
   so the replay pins values and every gradient component. *)

out = FileNameJoin[{Directory[],
    "crates", "sonium-fem", "fixtures", "wolfram-golden-v1.json"}];

dec[x_] := StringReplace[
  First[StringSplit[ToString[N[x, 17], InputForm], "`"]], "*^" -> "e"];
decList[l_] := Map[dec, l];

(* evaluation point in each reference element *)
xi = 0.2; eta = 0.3; zeta = 0.1;
l = {1 - xi - eta - zeta, xi, eta, zeta};
dl = {{-1., -1., -1.}, {1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};

(* P1 tet: values + constant gradients *)
p1Vals = l;
p1Grads = dl;

(* P2 tet: vertex + edge-midpoint ordering *)
edges = {{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};
p2Vals = Join[
  Table[l[[a]]*(2*l[[a]] - 1), {a, 4}],
  Table[4*l[[e[[1]]]]*l[[e[[2]]]], {e, edges}]];
p2Grads = Join[
  Table[(4*l[[a]] - 1)*dl[[a]], {a, 4}],
  Table[4*(dl[[e[[1]]]]*l[[e[[2]]]] + l[[e[[1]]]]*dl[[e[[2]]]]),
    {e, edges}]];

(* Q1 hex on [-1,1]^3 at (0.1, 0.2, 0.3) *)
hx = 0.1; hy = 0.2; hz = 0.3;
signs = {{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
         {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
q1Vals = Table[(1 + s[[1]]*hx)*(1 + s[[2]]*hy)*(1 + s[[3]]*hz)/8, {s, signs}];
q1Grads = Table[
  {s[[1]]*(1 + s[[2]]*hy)*(1 + s[[3]]*hz)/8,
   s[[2]]*(1 + s[[1]]*hx)*(1 + s[[3]]*hz)/8,
   s[[3]]*(1 + s[[1]]*hx)*(1 + s[[2]]*hy)/8}, {s, signs}];

data = <|
  "meta" -> <|
    "engine" -> "Wolfram Engine 15.0.0",
    "protocol" -> "audit-oracle-hermetic-v1"|>,
  "values" -> Association[{
    "p1_tet_vals" -> <|"values" -> decList[p1Vals],
      "exact" -> "{1-xi-eta-zeta, xi, eta, zeta} at (0.2, 0.3, 0.1)"|>,
    "p1_tet_grads" -> <|"values" -> decList[Flatten[p1Grads]],
      "exact" -> "constant P1 tet gradients, row-major"|>,
    "p2_tet_vals" -> <|"values" -> decList[p2Vals],
      "exact" -> "vertex l(2l-1) then 4 l_a l_b on canonical edges"|>,
    "p2_tet_grads" -> <|"values" -> decList[Flatten[p2Grads]],
      "exact" -> "P2 tet gradients, row-major, canonical edge order"|>,
    "q1_hex_vals" -> <|"values" -> decList[q1Vals],
      "exact" -> "tensor Q1 on [-1,1]^3 at (0.1, 0.2, 0.3)"|>,
    "q1_hex_grads" -> <|"values" -> decList[Flatten[q1Grads]],
      "exact" -> "Q1 hex gradients, row-major"|>}]|>;

Export[out, data, "JSON", "Compact" -> True];
Print["wrote ", out];