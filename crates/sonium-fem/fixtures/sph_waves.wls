(* Independent Wolfram reference for the sonium-fem one-step SPH wave check.
   Derives, at 50-digit working precision, summation densities, polynomial-EOS
   pressures, Monaghan artificial-viscosity accelerations (deck qa = 2,
   qb = 1), energy and continuity rates, and one symplectic Euler step for an
   8-particle 1D line with a velocity discontinuity. The Rust side consumes
   sph_waves_golden.json through the sph operator path.
   Run: wolframscript -script sph_waves.wls (bounded: < 60 s).
   Gate note: the lane checked in an mpmath twin (wolframscript was
   unavailable in its environment); the coordinator executed this script
   with Wolfram 15.0.0, fixed the `0.e-N` invalid-JSON emission below,
   and the checked-in JSON is genuine Wolfram output (worst rel dev
   vs the mpmath twin 1.1e-46). *)

prec = 50;
dim = 1;
h = SetPrecision[3/5, prec];
dx = SetPrecision[1/2, prec];
n = 8;
mass = SetPrecision[1, prec];
dt = SetPrecision[1/100, prec];
qa = SetPrecision[2, prec];
qb = SetPrecision[1, prec];
c1 = SetPrecision[13/10, prec];
rho0 = SetPrecision[2, prec];
eps = SetPrecision[1/100, prec];
sigma = SetPrecision[2/3, prec];

shape[q_] := Piecewise[{
  {1 - 3/2 q^2 + 3/4 q^3, q < 1},
  {1/4 (2 - q)^3, q < 2}}, 0];

kval[r_] := sigma/h^dim shape[r/h];

gradfactor[r_] := Module[{q = r/h},
  Which[
    q < 0 || q >= 2, 0,
    q == 0, sigma/h^(dim + 2) (-3),
    q < 1, sigma/h^(dim + 2) (-3 + 9/4 q),
    True, sigma/h^(dim + 2) (-3/4 (2 - q)^2/q)]];

dist[a_, b_] := Sqrt[Total[(a - b)^2]];
pos = Table[{dx (i - 1), 0, 0}, {i, n}];
vel = Table[If[i <= 4, {SetPrecision[1/5, prec], 0, 0}, {0, 0, 0}], {i, n}];
support = 2 h;
nbrs = Table[
  Select[Range[n], # != i && dist[pos[[i]], pos[[#]]] < support &], {i, n}];

rho = Table[
  mass kval[0] + Total[Table[mass kval[dist[pos[[i]], pos[[j]]]], {j, nbrs[[i]]}]],
  {i, n}];
pres = Table[c1 (rho[[i]]/rho0 - 1), {i, n}];
snd = Table[Sqrt[c1/rho0], {i, n}];

pivisc[i_, j_] := Module[{dxv, dvv, appr, dsq, mu, cbar, rbar},
  dxv = pos[[i]] - pos[[j]];
  dvv = vel[[i]] - vel[[j]];
  appr = dxv . dvv;
  If[appr >= 0, 0,
    dsq = dxv . dxv;
    mu = h appr/(dsq + eps h^2);
    cbar = (snd[[i]] + snd[[j]])/2;
    rbar = (rho[[i]] + rho[[j]])/2;
    (-qb cbar mu + qa mu^2)/rbar]];

accel = Table[{0, 0, 0}, {i, n}];
erate = Table[0, {i, n}];
drate = Table[0, {i, n}];
Do[
  hi = pres[[i]]/rho[[i]]^2;
  Do[
    off = pos[[i]] - pos[[j]];
    g = gradfactor[dist[pos[[i]], pos[[j]]]] off;
    dvv = vel[[i]] - vel[[j]];
    pi = pivisc[i, j];
    s = -mass (hi + pres[[j]]/rho[[j]]^2 + pi);
    accel[[i]] += s g;
    se = 1/2 mass (hi + pres[[j]]/rho[[j]]^2 + pi);
    erate[[i]] += se (dvv . g);
    drate[[i]] += mass (dvv . g),
    {j, nbrs[[i]]}],
  {i, n}];

vel1 = Table[vel[[i]] + accel[[i]] dt, {i, n}];
pos1 = Table[pos[[i]] + vel1[[i]] dt, {i, n}];
ene1 = Table[erate[[i]] dt, {i, n}];

golden = <|
  "schema_version" -> 1,
  "case_id" -> "fem.sph-waves.v1",
  "convention" -> "1D particle line in dim=1 kernel; deck viscosity qa=2 qb=1; linear polynomial EOS C1=1.3; one symplectic Euler step",
  "generator" -> "crates/sonium-fem/fixtures/sph_waves.wls",
  "computed_with" -> "Wolfram Language 15 N[_, 50] regen script (mpmath twin agreed to 1.1e-46 rel)",
  "working_precision_digits" -> prec,
  "dim" -> dim,
  "smoothing_length" -> N[h, prec],
  "spacing" -> N[dx, prec],
  "mass" -> 1,
  "dt" -> N[dt, prec],
  "qa" -> 2,
  "qb" -> 1,
  "eos" -> <|"c1" -> N[c1, prec], "c2" -> 0, "c3" -> 0, "rho0" -> N[rho0, prec]|>,
  "initial_velocity_left" -> N[1/5, prec],
  "count" -> n,
  "neighbors" -> (nbrs - 1),
  "density" -> rho,
  "pressure" -> pres,
  "sound_speed" -> snd,
  "acceleration" -> accel,
  "energy_rate" -> erate,
  "density_rate" -> drate,
  "velocity_after_step" -> vel1,
  "position_after_step" -> pos1,
  "energy_after_step" -> ene1,
  "tolerances" -> <|"abs" -> 1.0*^-12|> |>;

outdir = If[$InputFileName =!= "", DirectoryName[$InputFileName], "."];
(* Arbitrary-precision zeros export as invalid JSON `0.e-N` (gate fix:
   the lane's .wls was never executed). Normalize standalone zero
   mantissas after export; the word boundary keeps `10.e-N`-style
   substrings (none occur) intact. *)
json = ExportString[golden, "JSON"];
json = StringReplace[json, RegularExpression["(?i)\\b0\\.e[+-]?[0-9]+"] -> "0"];
Export[FileNameJoin[{outdir, "sph_waves_golden.json"}], json, "String"];
Print["wrote sph_waves_golden.json"];
