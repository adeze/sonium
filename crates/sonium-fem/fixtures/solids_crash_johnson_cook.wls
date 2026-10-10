(* Independent Wolfram reference for the Johnson-Cook crash response.
   Derives, at 50-digit working precision, the 4340 steel flow stress at
   quasi-static and fast rates, the thermally softened stress, and the
   Johnson-Cook fracture strain at uniaxial triaxiality. All inputs are
   exact rationals; powers, logs, and exps stay exact until N[..., 50].
   The Rust side consumes solids_crash_johnson_cook_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_crash_johnson_cook.wls > fixtures/solids_crash_johnson_cook_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

a = 792*10^6;
b = 510*10^6;
nn = 26/100;
cc = 14/1000;
mm = 103/100;
ep = 1/100;
tRoom = 293;
tMelt = 1793;

hardening = a + b*ep^nn;
flowQuasi = hardening*(1 + cc*Log[1])*(1 - 0^mm);
flowFast = hardening*(1 + cc*Log[1000])*(1 - 0^mm);
tStar = (900 - tRoom)/(tMelt - tRoom);
flowHot = hardening*(1 + cc*Log[1])*(1 - tStar^mm);

d1 = 5/100;
d2 = 344/100;
d3 = -212/100;
triax = 1/3;
efUni = (d1 + d2*Exp[d3*triax])*(1 + 2/1000*Log[1])*(1 + 61/100*0);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "crash_johnson_cook",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "a_pa" -> 792*10^6,
    "b_pa" -> 510*10^6,
    "n" -> 0.26,
    "c" -> 0.014,
    "m" -> 1.03,
    "ep" -> 0.01,
    "fast_rate" -> 1000,
    "hot_temperature_k" -> 900,
    "t_melt_k" -> 1793
  |>,
  "values" -> <|
    "flow_quasistatic" -> <|
      "values" -> {dec[flowQuasi]},
      "exact" -> "(A+B ep^n)(1+C ln 1)(1-0^m)"
    |>,
    "flow_fast" -> <|
      "values" -> {dec[flowFast]},
      "exact" -> "(A+B ep^n)(1+C ln 1000)(1-0^m)"
    |>,
    "flow_hot" -> <|
      "values" -> {dec[flowHot]},
      "exact" -> "(A+B ep^n)(1-T*^m), T*=(900-293)/(1793-293)"
    |>,
    "fracture_strain_uniaxial" -> <|
      "values" -> {dec[efUni]},
      "exact" -> "(D1+D2 exp(D3/3))(1+D4 ln 1)(1+D5 0)"
    |>
  |>,
  "tolerances" -> <|
    "flow_quasistatic_abs" -> 1.0*^-6,
    "flow_quasistatic_rel" -> 1.0*^-12,
    "flow_fast_abs" -> 1.0*^-6,
    "flow_fast_rel" -> 1.0*^-12,
    "flow_hot_abs" -> 1.0*^-6,
    "flow_hot_rel" -> 1.0*^-12,
    "fracture_strain_uniaxial_abs" -> 1.0*^-15,
    "fracture_strain_uniaxial_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
