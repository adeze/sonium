(* sonium-qa fem case fem.airbag-jetting.v1.
   Jetting overpressure geometry of the uniform-pressure airbag
   (Theory Manual equation 1126, engine VOLPRE): segment projection
   with end clamping, jet angle in degrees, jet distance, facing
   factor, and shaped overpressure. Conical (N3 = N1) and dihedral
   jets, on/off-axis elements, back-facing and degenerate centers,
   constant deck-shape laws plus sloped laws and non-unit scales.
   Catches degree/radian, clamping, and degenerate-jet slips. *)

Off[General::munfl];

(* Engine FINTER: piecewise linear with end-slope extrapolation. *)
finter[x_, xs_, ys_] := Module[{n, lo, x0, x1, y0, y1, slope},
  n = Length[xs];
  If[n == 1, Return[ys[[1]]]];
  lo = 1;
  While[lo + 1 <= n && x > xs[[lo + 1]], lo++];
  lo = Min[lo, n - 1];
  x0 = xs[[lo]]; x1 = xs[[lo + 1]];
  y0 = ys[[lo]]; y1 = ys[[lo + 1]];
  slope = (y1 - y0)/(x1 - x0);
  If[x <= x0, y0 + (x - x0) slope, y1 - (x1 - x) slope]
];

(* Engine VOLPRE per-element chain. *)
jetSample[n1_, n2_, n3_, center_, normal_, t_, laws_] := Module[
  {mid, off, axis, span, ps1, ps2, ps3, an2, an3, det, mu, facr,
   radial, len1, lenN, facing, cosT, theta, pt, pa, pz},
  mid = (n1 + n3)/2;
  off = center - mid;
  axis = n2 - mid;
  span = n3 - mid;
  ps1 = off . axis; ps2 = axis . span; ps3 = off . span;
  an2 = axis . axis; an3 = span . span;
  det = ps2^2 - an2 an3;
  mu = If[det == 0, 0, (ps2 ps1 - ps3 an2)/det];
  facr = Clip[mu, {-1, 1}];
  radial = off - facr span;
  len1 = Sqrt[radial . radial];
  lenN = Sqrt[normal . normal];
  facing = If[len1 == 0, 0, Max[0, (radial . normal)/(lenN len1)]];
  cosT = If[len1 == 0, 0, Clip[(radial . axis)/((Sqrt[an2]) len1), {-1, 1}]];
  theta = ArcCos[cosT] 180/Pi;
  pt = laws["fpt"] finter[(t - laws["t0"]) laws["sct"], laws["tx"], laws["ty"]];
  pa = laws["fpa"] finter[theta laws["sca"], laws["ax"], laws["ay"]];
  pz = laws["fpz"] finter[len1 laws["scd"], laws["dx"], laws["dy"]];
  <|
   "theta_deg" -> N[theta, 50],
   "dist" -> N[len1, 50],
   "facing" -> N[facing, 50],
   "overpressure" -> N[pt facing pa pz, 50]
  |>
];

deckLaws = <|
  "fpt" -> 1, "sct" -> 1, "tx" -> {0}, "ty" -> {1},
  "fpa" -> 1, "sca" -> 1, "ax" -> {0}, "ay" -> {1/100},
  "fpz" -> 1, "scd" -> 1, "dx" -> {0}, "dy" -> {1/100},
  "t0" -> 0
|>;
slopeLaws = <|
  "fpt" -> 2, "sct" -> 1, "tx" -> {0, 10}, "ty" -> {1, 3},
  "fpa" -> 1/2, "sca" -> 2, "ax" -> {0, 90, 180}, "ay" -> {1, 1/2, 0},
  "fpz" -> 3, "scd" -> 1/10, "dx" -> {0, 1}, "dy" -> {1, 1/2},
  "t0" -> 0
|>;

n1c = {0, 0, 0}; n2c = {0, 0, 10}; n3c = n1c;
n1d = {-10, 0, 0}; n2d = {0, 5, 10}; n3d = {10, 0, 0};

cases = {
  <|"id" -> "conical-axis", "jet" -> "conical", "laws" -> "deck",
    "center" -> {0, 0, 5}, "normal" -> {0, 0, 1}, "t" -> 1|>,
  <|"id" -> "conical-side", "jet" -> "conical", "laws" -> "deck",
    "center" -> {5, 0, 0}, "normal" -> {1, 0, 0}, "t" -> 1|>,
  <|"id" -> "conical-back", "jet" -> "conical", "laws" -> "deck",
    "center" -> {0, 0, 5}, "normal" -> {0, 0, -1}, "t" -> 1|>,
  <|"id" -> "conical-345", "jet" -> "conical", "laws" -> "slope",
    "center" -> {3, 4, 0}, "normal" -> {3/5, 4/5, 0}, "t" -> 5|>,
  <|"id" -> "dihedral-mid", "jet" -> "dihedral", "laws" -> "slope",
    "center" -> {0, 3, 4}, "normal" -> {0, 3/5, 4/5}, "t" -> 5|>,
  <|"id" -> "dihedral-clamp", "jet" -> "dihedral", "laws" -> "deck",
    "center" -> {30, 0, 0}, "normal" -> {1, 0, 0}, "t" -> 1|>,
  <|"id" -> "dihedral-zero", "jet" -> "dihedral", "laws" -> "deck",
    "center" -> {0, 0, 0}, "normal" -> {0, 0, 1}, "t" -> 1|>
};

jets = <|"conical" -> {n1c, n2c, n3c}, "dihedral" -> {n1d, n2d, n3d}|>;
lawSets = <|"deck" -> deckLaws, "slope" -> slopeLaws|>;

results = Table[Module[{c, nodes, laws, sample},
   c = cases[[i]];
   nodes = jets[c["jet"]];
   laws = lawSets[c["laws"]];
   sample = jetSample[nodes[[1]], nodes[[2]], nodes[[3]],
     c["center"], c["normal"], c["t"], laws];
   Join[<|"id" -> c["id"], "jet" -> c["jet"], "laws" -> c["laws"],
     "center" -> c["center"], "normal" -> c["normal"], "t" -> c["t"]|>, sample]
  ], {i, Length[cases]}];

payload = <|
  "schema_version" -> 1,
  "case" -> "fem.airbag-jetting.v1",
  "angle_unit" -> "degrees",
  "deck_laws" -> <|
    "time" -> 1, "angle" -> N[1/100, 50], "dist" -> N[1/100, 50]|>,
  "slope_laws" -> <|
    "time_x" -> {0, 10}, "time_y" -> {1, 3}, "fpt" -> 2,
    "angle_x" -> {0, 90, 180}, "angle_y" -> {1, 1/2, 0}, "fpa" -> 1/2,
    "sca" -> 2, "dist_x" -> {0, 1}, "dist_y" -> {1, 1/2},
    "fpz" -> 3, "scd" -> 1/10|>,
  "conical" -> <|"n1" -> n1c, "n2" -> n2c, "n3" -> n3c|>,
  "dihedral" -> <|"n1" -> n1d, "n2" -> n2d, "n3" -> n3d|>,
  "cases" -> results
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
