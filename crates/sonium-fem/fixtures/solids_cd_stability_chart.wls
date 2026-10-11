(* Independent Wolfram reference for the damped central-difference stability chart.
   Derives, at 50-digit working precision, the spectral radius |A_max| and the
   margin 1-|A_max| of the velocity-Verlet (leapfrog) amplification matrix on
   the state {u_n, dt v_{n-1/2}}, i.e. A = {{1-O^2, 1-2 xi O}, {-O^2, 1-2 xi O}}
   with O = w dt and xi the damping ratio, on the 5x3 grid
   O = {0.1, 0.5, 1.0, 1.5, 2.0} times xi = {0, 0.02, 0.1}, plus the boundary
   row Oc(xi) = 2(Sqrt[1+xi^2]-xi) where |A_max| = 1 exactly (lambda = -1 is a
   root of the characteristic polynomial there). All inputs are exact rationals
   or radicals; N[..., 50] is applied once per exported value. Stored values
   are row-major over {omega, xi}: index = i_omega*3 + i_xi (0-based). The Rust
   side consumes solids_cd_stability_chart_golden.json through
   sonium_solvers::time_integration::damped_stability_margin; this script is the
   derivation.
   Regen: wolframscript -script fixtures/solids_cd_stability_chart.wls > fixtures/solids_cd_stability_chart_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := If[x == 0, "0.0", ToString[CForm[N[x, prec]]]];

omegas = {1/10, 1/2, 1, 3/2, 2};
xis = {0, 1/50, 1/10};

ampRoots[om_, xi_] := Module[{tr = 2 - om^2 - 2*xi*om, det = 1 - 2*xi*om},
  {(tr + Sqrt[tr^2 - 4*det])/2, (tr - Sqrt[tr^2 - 4*det])/2}];

absMax[om_, xi_] := Max[Abs[#] & /@ ampRoots[om, xi]];
margin[om_, xi_] := 1 - absMax[om, xi];
omegaCrit[xi_] := 2*(Sqrt[1 + xi^2] - xi);

gridAbs = Flatten[Table[absMax[om, xi], {om, omegas}, {xi, xis}]];
gridMargin = Flatten[Table[margin[om, xi], {om, omegas}, {xi, xis}]];
crits = omegaCrit /@ xis;
boundaryAbs = absMax[omegaCrit[#], #] & /@ xis;

(* Self-check: the boundary row must sit exactly on |A| = 1. *)
If[Max[Abs[N[#, 60] - 1] & /@ boundaryAbs] > 10^-50, Quit[1]];

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "cd_stability_chart",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "omega_dt" -> N[omegas],
    "xi" -> N[xis]
  |>,
  "values" -> <|
    "abs_Amax" -> <|
      "values" -> dec /@ gridAbs,
      "exact" -> "Max[Abs[eig({{1-O^2,1-2 xi O},{-O^2,1-2 xi O}}))], row-major over {omega, xi}"
    |>,
    "margin" -> <|
      "values" -> dec /@ gridMargin,
      "exact" -> "1 - |A_max| (>0 stable, 0 boundary, <0 unstable)"
    |>,
    "boundary_omega_crit" -> <|
      "values" -> dec /@ crits,
      "exact" -> "Oc(xi) = 2*(Sqrt[1+xi^2]-xi)"
    |>,
    "boundary_abs_Amax" -> <|
      "values" -> dec /@ boundaryAbs,
      "exact" -> "|A_max| on the boundary (= 1 exactly)"
    |>
  |>,
  "tolerances" -> <|
    "abs_Amax_abs" -> 1.0*^-15,
    "abs_Amax_rel" -> 1.0*^-12,
    "margin_abs" -> 1.0*^-15,
    "margin_rel" -> 1.0*^-12,
    "boundary_omega_crit_abs" -> 1.0*^-15,
    "boundary_omega_crit_rel" -> 1.0*^-12,
    "boundary_abs_Amax_abs" -> 1.0*^-12,
    "boundary_abs_Amax_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
