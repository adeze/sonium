(* Independent Wolfram reference for the spinning Lighthill ring.
   Symbolically differentiates the exp(-ikR) Green function (no hand
   Hessian), rotates the cylindrical moment per hoop angle, and
   integrates with Clenshaw-Curtis quadrature at 40-digit working
   precision. The Rust side consumes lighthill_ring_golden.json
   through lighthill_ring_pressure. Cases: (A) axisymmetric axial
   moment, in-plane observer; (B) m = 2 radial moment (rotation
   active), oblique observer; (C) m = 2 axial moment on the axis
   (spinning null).
   Run: wolframscript -script lighthill_ring.wls > lighthill_ring_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86-64. *)

om = 1000; c = 340;
k = SetPrecision[om/c, 45];

(* Symbolic Hessian of the free-space Green function, once. *)
hSym = D[Exp[-I kk Sqrt[xx^2 + yy^2 + zz^2]]/(4 Pi Sqrt[xx^2 + yy^2 + zz^2]),
  {{xx, yy, zz}, 2}] /. kk -> k;

rot[phi_] := {{Cos[phi], -Sin[phi], 0}, {Sin[phi], Cos[phi], 0}, {0, 0, 1}};
tCart[tcyl_, phi_] := rot[phi] . tcyl . Transpose[rot[phi]];
hoop[R_, z0_, phi_] := {R Cos[phi], R Sin[phi], z0};

sample[tcyl_, R_, z0_, obs_, m_, phi_] := Module[{y, d, h, t},
  y = hoop[R, z0, phi];
  d = obs - y;
  h = hSym /. {xx -> d[[1]], yy -> d[[2]], zz -> d[[3]]};
  t = tCart[tcyl, phi];
  Sum[t[[i, j]] h[[i, j]], {i, 3}, {j, 3}] Exp[I m phi]
];

ringPressure[tcyl_, R_, z0_, obs_, m_, pg_: 25, ag_: Automatic] := NIntegrate[
  sample[tcyl, R, z0, obs, m, phi], {phi, 0, 2 Pi},
  Method -> "ClenshawCurtisRule", WorkingPrecision -> 40,
  PrecisionGoal -> pg, AccuracyGoal -> ag, MaxRecursion -> 20
];

(* --- Case A: m = 0, axial moment, in-plane observer. --- *)
rA = SetPrecision[1/2, 45]; zA = SetPrecision[1/10, 45];
tA = SetPrecision[{{0, 0, 0}, {0, 0, 0}, {0, 0, 2 + I/2}}, 45];
obsA = SetPrecision[{3, 0, 0}, 45];
pA = ringPressure[tA, rA, zA, obsA, 0];

(* --- Case B: m = 2, radial moment, oblique observer. --- *)
tB = SetPrecision[{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}}, 45];
obsB = SetPrecision[{2, 0, 1}, 45];
pB = ringPressure[tB, rA, zA, obsB, 2];

(* --- Case C: m = 2, axial moment, on-axis observer (null).
   Exact zero by symmetry: accuracy goal replaces the relative one. --- *)
obsC = SetPrecision[{0, 0, 3}, 45];
pC = ringPressure[tA, rA, zA, obsC, 2, 0, 30];

prec = 50;
cPair[z_] := {N[Re[z], prec], N[Im[z], prec]};
golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.lighthill-ring.v1",
  "convention" -> "SI units; exp(i wt) harmonic; exp(-i k r) propagation; cylindrical (r,phi,z) moment per radian, modal phase exp(+i m phi)",
  "generator" -> "crates/sonium-fem/fixtures/lighthill_ring.wls",
  "wolfram_version" -> $Version,
  "working_precision_digits" -> 40,
  "angular_frequency" -> N[om, prec],
  "speed_of_sound" -> N[c, prec],
  "ring_radius_m" -> N[rA, prec],
  "axial_position_m" -> N[zA, prec],
  "case_a" -> <|
    "azimuthal_order" -> 0,
    "observer" -> N[obsA, prec],
    "stress_moment" -> Map[cPair, tA, {2}],
    "pressure" -> cPair[pA]
  |>,
  "case_b" -> <|
    "azimuthal_order" -> 2,
    "observer" -> N[obsB, prec],
    "stress_moment" -> Map[cPair, tB, {2}],
    "pressure" -> cPair[pB]
  |>,
  "case_c" -> <|
    "azimuthal_order" -> 2,
    "observer" -> N[obsC, prec],
    "stress_moment" -> Map[cPair, tA, {2}],
    "pressure" -> cPair[pC]
  |>,
  "tolerances" -> <|
    "pressure_abs" -> 1.0*^-9
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
