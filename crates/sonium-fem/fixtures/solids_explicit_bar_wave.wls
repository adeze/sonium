(* Independent Wolfram reference for the explicit bar wave QA.
   Derives, at 50-digit working precision, the steel bar wave speed and
   the d'Alembert Gaussian pulse displacement at three space-time points.
   All inputs are exact rationals; Sqrt and Exp stay exact until N[..., 50].
   The Rust side consumes solids_explicit_bar_wave_golden.json; this
   script is the derivation.
   Regen: wolframscript -script fixtures/solids_explicit_bar_wave.wls > fixtures/solids_explicit_bar_wave_golden.json
   Engine: Wolfram Language 15.0.0 for Linux x86 (64-bit) (May 26, 2026). *)

prec = 50;
dec[x_] := ToString[CForm[N[x, prec]]];

young = 200*10^9;
rho = 7800;
c = Sqrt[young/rho];
center = 10;
width = 4/10;
amp = 1/1000;
t = 8/10000;

u[x_] := amp/2*(Exp[-((x - center - c*t)/width)^2] + Exp[-((x - center + c*t)/width)^2]);
uCenter = u[center];
uRight = u[center + c*t];
uLeft = u[center - c*t];
area = 1/100;
pulseEnergy = young*area*amp^2*Sqrt[Pi/2]/(2*width);

golden = <|
  "meta" -> <|
    "engine" -> "Wolfram Language " <> $Version,
    "protocol" -> "solids-golden-v1",
    "case" -> "explicit_bar_wave",
    "units" -> "SI"
  |>,
  "inputs" -> <|
    "youngs_modulus_pa" -> 200*10^9,
    "density" -> 7800,
    "pulse_center_m" -> 10,
    "pulse_width_m" -> 0.4,
    "pulse_amplitude_m" -> 0.001,
    "time_s" -> 0.0008
  |>,
  "values" -> <|
    "wave_speed" -> <|
      "values" -> {dec[c]},
      "exact" -> "Sqrt[E/rho]"
    |>,
    "u_center" -> <|
      "values" -> {dec[uCenter]},
      "exact" -> "A/2 (Exp[-((c t)/w)^2] + Exp[-((c t)/w)^2])"
    |>,
    "u_right" -> <|
      "values" -> {dec[uRight]},
      "exact" -> "d'Alembert at x0 + c t"
    |>,
    "u_left" -> <|
      "values" -> {dec[uLeft]},
      "exact" -> "d'Alembert at x0 - c t"
    |>,
    "pulse_strain_energy" -> <|
      "values" -> {dec[pulseEnergy]},
      "exact" -> "E A A^2 Sqrt[Pi/2]/(2 w)"
    |>
  |>,
  "tolerances" -> <|
    "wave_speed_abs" -> 1.0*^-9,
    "wave_speed_rel" -> 1.0*^-12,
    "u_center_abs" -> 1.0*^-15,
    "u_center_rel" -> 1.0*^-12,
    "u_right_abs" -> 1.0*^-15,
    "u_right_rel" -> 1.0*^-12,
    "u_left_abs" -> 1.0*^-15,
    "u_left_rel" -> 1.0*^-12,
    "pulse_strain_energy_abs" -> 1.0*^-9,
    "pulse_strain_energy_rel" -> 1.0*^-12
  |>
|>;

WriteString[$Output, ExportString[golden, "JSON"]];
