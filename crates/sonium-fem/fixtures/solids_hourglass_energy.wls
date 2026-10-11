(* Independent Wolfram reference for Flanagan-Belytschko HEXA8 hourglass energy.
   Builds the 4 hourglass base vectors of the unit cube in closed form (exact
   integers), verifies orthogonality to the constant and linear coordinate
   fields (residuals exactly 0), and evaluates the viscous dissipation
   P = v.f with f = kappa G G^T v at 50-digit precision for 3 sample
   velocities: a rigid rotation (P = 0, zero work on rigid modes), a pure
   hourglass mode, and a mixed mode with rigid contamination. The Rust side
   consumes solids_hourglass_energy_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_hourglass_energy.wls > fixtures/solids_hourglass_energy_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Lane node ordering: natural coords (xi, eta, ze) in {-1, +1}^3. *)
xi = {-1, 1, 1, -1, -1, 1, 1, -1};
eta = {-1, -1, 1, 1, -1, -1, 1, 1};
ze = {-1, -1, -1, -1, 1, 1, 1, 1};

(* Hourglass base modes (exact integers). *)
h1 = xi*eta;
h2 = eta*ze;
h3 = ze*xi;
h4 = xi*eta*ze;
gam = {h1, h2, h3, h4};

(* Unit-cube physical coords and the constant field. *)
xc = (xi + 1)/2;
yc = (eta + 1)/2;
zc = (ze + 1)/2;
one = Table[1, {8}];

(* Orthogonality residuals Gamma.{1, x, y, z}: exactly 0. *)
res = Flatten[Table[{g.one, g.xc, g.yc, g.zc}, {g, gam}]];

(* Sample velocities (exact rationals). Per-component blocks feed the
   dissipation derivation; node-major interleave is exported to match the
   Phase 1 P1C1 slice layout ([x0, y0, z0, ...]). *)
kappa = 1/10;
w = {1/4, -1/8, 1/2};
spinX = w[[2]]*zc - w[[3]]*yc;
spinY = w[[3]]*xc - w[[1]]*zc;
spinZ = w[[1]]*yc - w[[2]]*xc;
t = {1/8, -1/4, 1/2};
mixX = h2/2 + t[[1]] + spinX;
mixY = h3/4 + t[[2]] + spinY;
mixZ = h4/8 + t[[3]] + spinZ;
vRigid = Join[spinX, spinY, spinZ];
vHg = Join[h1, 0*one, 0*one];
vMix = Join[mixX, mixY, mixZ];
vRigidOut = Flatten[Transpose[{spinX, spinY, spinZ}]];
vHgOut = Flatten[Transpose[{h1, 0*one, 0*one}]];
vMixOut = Flatten[Transpose[{mixX, mixY, mixZ}]];

(* Dissipation P = kappa Sum_a Sum_i (Gamma_a . v_i)^2. *)
diss[v_] := kappa*Total[Flatten[Table[
  {(g.Take[v, {1, 8}])^2, (g.Take[v, {9, 16}])^2, (g.Take[v, {17, 24}])^2},
  {g, gam}]]];
pRigid = diss[vRigid];
pHg = diss[vHg];
pMix = diss[vMix];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "hourglass_energy",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "node_xi" -> xi,
    "node_eta" -> eta,
    "node_zeta" -> ze,
    "cube_coords" -> Flatten[Transpose[{xc, yc, zc}]],
    "kappa" -> 0.1,
    "vel_rigid" -> dec /@ vRigidOut,
    "vel_hg" -> dec /@ vHgOut,
    "vel_mixed" -> dec /@ vMixOut
  |>,
  "values" -> <|
    "gamma" -> <|
      "values" -> dec /@ Flatten[gam],
      "exact" -> "Flanagan-Belytschko h1..h4 of the unit cube, mode-major"
    |>,
    "orthogonality_residuals" -> <|
      "values" -> dec /@ res,
      "exact" -> "Gamma_a.{1,x,y,z} = 0 for each mode a"
    |>,
    "dissipation" -> <|
      "values" -> dec /@ {pRigid, pHg, pMix},
      "exact" -> "P = kappa Sum_a,i (Gamma_a.v_i)^2"
    |>
  |>,
  "tolerances" -> <|
    "gamma_abs" -> 0,
    "gamma_rel" -> 0,
    "orthogonality_residuals_abs" -> 0,
    "orthogonality_residuals_rel" -> 0,
    "dissipation_abs" -> 1.0*^-15,
    "dissipation_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
