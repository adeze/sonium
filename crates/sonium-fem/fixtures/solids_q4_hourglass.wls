(* Independent Wolfram reference for Flanagan-Belytschko Q4 hourglass energy.
   Builds the hourglass base vector of the unit square in closed form (exact
   integers), verifies orthogonality to the constant and linear coordinate
   fields (residuals exactly 0), and evaluates the viscous dissipation
   P = v.f with f = kappa G G^T v at 50-digit precision for 3 sample
   velocities: a rigid rotation (P = 0, zero work on rigid modes), a pure
   hourglass mode, and a mixed mode with rigid contamination. The Rust side
   consumes solids_q4_hourglass_golden.json; this script is the derivation.
   Regen: wolframscript -script fixtures/solids_q4_hourglass.wls > fixtures/solids_q4_hourglass_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

(* Lane node ordering: natural coords (xi, eta) in {-1, +1}^2. *)
xi = {-1, 1, 1, -1};
eta = {-1, -1, 1, 1};

(* Hourglass base mode (exact integers). *)
h = xi*eta;
gam = h;

(* Unit-square physical coords and the constant field. *)
xc = (xi + 1)/2;
yc = (eta + 1)/2;
one = Table[1, {4}];

(* Orthogonality residuals Gamma.{1, x, y}: exactly 0. *)
res = {gam.one, gam.xc, gam.yc};

(* Sample velocities (exact rationals). Per-component blocks feed the
   dissipation derivation; node-major interleave is exported to match the
   Phase 1 P1C1 slice layout ([x0, y0, ...]). *)
kappa = 1/10;
w = 1/4;
spinX = -w*yc;
spinY = w*xc;
t = {1/8, -1/4};
mixX = h/2 + t[[1]] + spinX;
mixY = h/4 + t[[2]] + spinY;
vRigid = Join[spinX, spinY];
vHg = Join[h, 0*one];
vMix = Join[mixX, mixY];
vRigidOut = Flatten[Transpose[{spinX, spinY}]];
vHgOut = Flatten[Transpose[{h, 0*one}]];
vMixOut = Flatten[Transpose[{mixX, mixY}]];

(* Dissipation P = kappa Sum_i (Gamma . v_i)^2. *)
diss[v_] := kappa*Total[{(gam.Take[v, {1, 4}])^2, (gam.Take[v, {5, 8}])^2}];
pRigid = diss[vRigid];
pHg = diss[vHg];
pMix = diss[vMix];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "q4_hourglass",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "node_xi" -> xi,
    "node_eta" -> eta,
    "square_coords" -> Flatten[Transpose[{xc, yc}]],
    "kappa" -> 0.1,
    "vel_rigid" -> dec /@ vRigidOut,
    "vel_hg" -> dec /@ vHgOut,
    "vel_mixed" -> dec /@ vMixOut
  |>,
  "values" -> <|
    "gamma" -> <|
      "values" -> dec /@ gam,
      "exact" -> "Flanagan-Belytschko xi*eta mode of the unit square"
    |>,
    "orthogonality_residuals" -> <|
      "values" -> dec /@ res,
      "exact" -> "Gamma.{1,x,y} = 0"
    |>,
    "dissipation" -> <|
      "values" -> dec /@ {pRigid, pHg, pMix},
      "exact" -> "P = kappa Sum_i (Gamma.v_i)^2"
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
