(* Independent Wolfram reference for the sonium-fem SPH kernel checks.
   Derives, at 50-digit working precision, cubic B-spline values and radial
   gradient factors, the partition-of-unity sum on a cubic lattice, and the
   SPH difference-form gradient of a linear field. The Rust side consumes
   sph_kernel_consistency_golden.json through the sph module kernel path.
   Run: wolframscript -script sph_kernel_consistency.wls (bounded: < 60 s).
   Checked-in JSON was cross-computed with mpmath (see computed_with) since
   wolframscript was unavailable in the lane environment; this script is the
   canonical regen path. *)

prec = 50;
h = SetPrecision[13/10, prec];
lh = SetPrecision[6/5, prec];
sigma = SetPrecision[1/Pi, prec];

shape[q_] := Piecewise[{
  {1 - 3/2 q^2 + 3/4 q^3, q < 1},
  {1/4 (2 - q)^3, q < 2}}, 0];

kval[r_, hh_] := sigma/hh^3 shape[r/hh];

gradfactor[r_, hh_] := Module[{q = r/hh},
  Which[
    q < 0 || q >= 2, 0,
    q == 0, sigma/hh^5 (-3),
    q < 1, sigma/hh^5 (-3 + 9/4 q),
    True, sigma/hh^5 (-3/4 (2 - q)^2/q)]];

qs = {0, 1/2, 1, 3/2, 1999/1000, 2};
kvalRows = Table[<|
   "q" -> N[qs[[k]], prec],
   "W" -> kval[SetPrecision[qs[[k]], prec] h, h],
   "grad_factor" -> gradfactor[SetPrecision[qs[[k]], prec] h, h]|>,
  {k, Length[qs]}];

pts = Flatten[Table[
  {SetPrecision[i, prec], SetPrecision[j, prec], SetPrecision[k, prec]},
  {i, -3, 3}, {j, -3, 3}, {k, -3, 3}], 2];
center = {0, 0, 0};
support = 2 lh;
dist[a_, b_] := Sqrt[Total[(a - b)^2]];

pou = Total[Table[
  If[dist[center, p] < support || dist[center, p] == 0,
    kval[dist[center, p], lh], 0], {p, pts}]];

ffield[p_] := 2 p[[1]] - 3 p[[2]] + p[[3]] + 5;
fc = ffield[center];
sphgrad = {0, 0, 0};
Do[
  If[p != center && dist[center, p] < support,
    off = center - p;
    gf = gradfactor[dist[center, p], lh];
    sphgrad += (ffield[p] - fc) gf off],
  {p, pts}];

golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.sph-kernel-consistency.v1",
  "convention" -> "dimensionless kernel checks; 3D cubic B-spline, support 2h",
  "generator" -> "crates/sonium-fem/fixtures/sph_kernel_consistency.wls",
  "computed_with" -> "Wolfram Language regen script; checked-in JSON cross-computed with mpmath",
  "working_precision_digits" -> prec,
  "smoothing_length" -> N[h, prec],
  "dim" -> 3,
  "kernel_values" -> kvalRows,
  "partition_of_unity" -> <|
    "lattice_spacing" -> 1,
    "lattice_half_width" -> 3,
    "smoothing_length" -> N[lh, prec],
    "sum" -> pou|>,
  "gradient_accuracy" -> <|
    "field" -> "2x - 3y + z + 5",
    "sph_gradient" -> sphgrad,
    "exact_gradient" -> {2, -3, 1}|>,
  "tolerances" -> <|
    "abs" -> 1.0*^-12,
    "partition_vs_one" -> 0.05,
    "gradient_vs_exact" -> 0.06|> |;

outdir = If[$InputFileName =!= "", DirectoryName[$InputFileName], "."];
Export[FileNameJoin[{outdir, "sph_kernel_consistency_golden.json"}], golden, "JSON"];
Print["wrote sph_kernel_consistency_golden.json"];
