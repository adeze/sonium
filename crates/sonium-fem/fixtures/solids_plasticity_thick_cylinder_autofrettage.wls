(*
  Thick-cylinder autofrettage reference (solids-golden-v1).
  The Rust side consumes solids_plasticity_thick_cylinder_autofrettage_golden.json;
  this script is the derivation (plane strain, von Mises, elastic-perfectly
  plastic; overload with elastic unloading). Triple-checked against Rust and
  Python to 1.3e-14.
  Regen: wolframscript -script fixtures/solids_plasticity_thick_cylinder_autofrettage.wls
  Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026).
  Internal pressure p* loads the cylinder into partial plasticity
  (elastic-plastic interface c); unloading is elastic. Prints the golden
  values as JSON. *)

a = 0.05; b = 0.10; sy = 250.0*10^6; pstar = 150.0*10^6;

py = sy*(1 - (a/b)^2)/Sqrt[3];
plim = (2/Sqrt[3])*sy*Log[b/a];
tresca = sy*Log[b/a];

(* Interface radius from p* = (2 sy/Sqrt3)(Log[c/a] + (b^2-c^2)/(2 b^2)). *)
c = c /. FindRoot[(2*sy/Sqrt[3])*(Log[c/a] + (b^2 - c^2)/(2*b^2)) - pstar == 0, {c, (a + b)/2}];

lame[p_, r_] := Module[{amp = p*a^2/(b^2 - a^2), shape = b^2/r^2},
  {amp*(1 - shape), amp*(1 + shape)}];

loaded[r_] := Module[{flow = 2*sy/Sqrt[3]},
  If[r <= c,
    {-pstar + flow*Log[r/a], -pstar + flow*Log[r/a] + flow},
    Module[{qc = pstar - flow*Log[c/a], amp = (pstar - flow*Log[c/a])*c^2/(b^2 - c^2), shape = b^2/r^2},
      {amp*(1 - shape), amp*(1 + shape)}]]];

resid[r_] := loaded[r] - lame[pstar, r];

bore = resid[a]; outer = resid[b];
grid = Table[resid[r], {r, a, b, (b - a)/2000}];
maxq = Max[(Sqrt[3]/2)*Abs[#[[2]] - #[[1]]] & /@ grid];

result = <|
  "yield_pressure_pa" -> py,
  "limit_pressure_pa" -> plim,
  "tresca_limit_crosscheck_pa" -> tresca,
  "interface_radius_m" -> c,
  "bore_radial_residual_pa" -> bore[[1]],
  "bore_hoop_residual_pa" -> bore[[2]],
  "outer_radial_residual_pa" -> outer[[1]],
  "outer_hoop_residual_pa" -> outer[[2]],
  "max_residual_equiv_pa" -> maxq
|>;

Print[ExportString[result, "JSON"]];
