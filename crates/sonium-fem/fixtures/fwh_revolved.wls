(* Independent Wolfram reference for the spinning revolved FW-H strip.
   Symbolically differentiates the exp(-ikR) Green function (no hand
   gradient), spins one meridional strip with modal phase exp(+im phi),
   and integrates thickness and loading with nested Gauss-Kronrod (s)
   x Clenshaw-Curtis (phi) quadrature at 40-digit working precision.
   The Rust side consumes fwh_revolved_golden.json through
   stationary_fwh_ring_pressure. Cases: (A) m = 1 combined, oblique
   observer; (B) m = 1 combined, on-axis observer (spinning null).
   Run: wolframscript -script fwh_revolved.wls > fwh_revolved_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86-64. *)

om = 1000; rho = SetPrecision[12/10, 45]; c = 340;
k = SetPrecision[om/c, 45];

(* Symbolic gradient of the free-space Green function, once. *)
gSym = D[Exp[-I kk Sqrt[xx^2 + yy^2 + zz^2]]/(4 Pi Sqrt[xx^2 + yy^2 + zz^2]),
  {{xx, yy, zz}}] /. kk -> k;
gVal[dx_, dy_, dz_] := Exp[-I k Sqrt[dx^2 + dy^2 + dz^2]]/(4 Pi Sqrt[dx^2 + dy^2 + dz^2]);

(* Strip geometry: meridional segment nodeA -> nodeB, unit normal nRZ. *)
nodeA = SetPrecision[{2/5, 0}, 45];
nodeB = SetPrecision[{3/5, 1/10}, 45];
stripLen = Norm[nodeB - nodeA];
strand[s_] := nodeA + s (nodeB - nodeA)/stripLen;
nRZ = SetPrecision[{-1/Sqrt[5], 2/Sqrt[5]}, 45];
nCart[phi_] := {nRZ[[1]] Cos[phi], nRZ[[1]] Sin[phi], nRZ[[2]]};
surf[s_, phi_] := Module[{q = strand[s]}, {q[[1]] Cos[phi], q[[1]] Sin[phi], q[[2]]}];

pModal = SetPrecision[1 + I/2, 45];
vnModal = SetPrecision[1/50 - I/100, 45];
mOrder = 1;

thickDensity[s_, phi_, obs_] := Module[{y, d, r},
  y = surf[s, phi];
  d = obs - y;
  r = strand[s][[1]];
  I om rho vnModal gVal[d[[1]], d[[2]], d[[3]]] r Exp[I mOrder phi]
];

(* FW-H loading uses the SOURCE-side gradient dG/dn_y = -dG/dn_x;
   gSym differentiates w.r.t. displacement (observer side), hence the
   explicit minus. Validated against the 3-D single-patch golden. *)
loadDensity[s_, phi_, obs_] := Module[{y, d, r, g, n},
  y = surf[s, phi];
  d = obs - y;
  r = strand[s][[1]];
  g = gSym /. {xx -> d[[1]], yy -> d[[2]], zz -> d[[3]]};
  n = nCart[phi];
  -pModal (g . n) r Exp[I mOrder phi]
];

innerThick[s_?NumericQ, obs_, pg_: 25, ag_: Automatic] := NIntegrate[
  thickDensity[s, phi, obs], {phi, 0, 2 Pi},
  Method -> "ClenshawCurtisRule", WorkingPrecision -> 40,
  PrecisionGoal -> pg, AccuracyGoal -> ag, MaxRecursion -> 20
];
innerLoad[s_?NumericQ, obs_, pg_: 25, ag_: Automatic] := NIntegrate[
  loadDensity[s, phi, obs], {phi, 0, 2 Pi},
  Method -> "ClenshawCurtisRule", WorkingPrecision -> 40,
  PrecisionGoal -> pg, AccuracyGoal -> ag, MaxRecursion -> 20
];
thick[obs_, pg_: 25, ag_: Automatic] := NIntegrate[
  innerThick[s, obs, pg, ag], {s, 0, stripLen},
  Method -> "GaussKronrodRule", WorkingPrecision -> 40,
  PrecisionGoal -> pg, AccuracyGoal -> ag, MaxRecursion -> 20
];
load[obs_, pg_: 25, ag_: Automatic] := NIntegrate[
  innerLoad[s, obs, pg, ag], {s, 0, stripLen},
  Method -> "GaussKronrodRule", WorkingPrecision -> 40,
  PrecisionGoal -> pg, AccuracyGoal -> ag, MaxRecursion -> 20
];

(* --- Case A: oblique observer. --- *)
obsA = SetPrecision[{2, 0, 1}, 45];
thickA = thick[obsA]; loadA = load[obsA]; totalA = thickA + loadA;

(* --- Case B: on-axis observer (spinning null for m = 1).
   Exact zero by symmetry: accuracy goal replaces the relative one. --- *)
obsB = SetPrecision[{0, 0, 3}, 45];
thickB = thick[obsB, 0, 30]; loadB = load[obsB, 0, 30]; totalB = thickB + loadB;

prec = 50;
cPair[z_] := {N[Re[z], prec], N[Im[z], prec]};
golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.fwh-revolved.v1",
  "convention" -> "SI units; exp(i wt) harmonic; exp(-i k r) propagation; meridional (r,z) strip revolved with modal phase exp(+i m phi), surface element r ds dphi",
  "generator" -> "crates/sonium-fem/fixtures/fwh_revolved.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> 40,
  "angular_frequency" -> N[om, prec],
  "density" -> N[rho, prec],
  "speed_of_sound" -> N[c, prec],
  "azimuthal_order" -> mOrder,
  "node_a" -> N[nodeA, prec],
  "node_b" -> N[nodeB, prec],
  "outward_normal" -> N[nRZ, prec],
  "pressure" -> cPair[pModal],
  "normal_velocity" -> cPair[vnModal],
  "case_a" -> <|
    "observer" -> N[obsA, prec],
    "thickness" -> cPair[thickA],
    "loading" -> cPair[loadA],
    "total" -> cPair[totalA]
  |>,
  "case_b" -> <|
    "observer" -> N[obsB, prec],
    "thickness" -> cPair[thickB],
    "loading" -> cPair[loadB],
    "total" -> cPair[totalB]
  |>,
  "tolerances" -> <|
    "component_abs" -> 1.0*^-9
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
