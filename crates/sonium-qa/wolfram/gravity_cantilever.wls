(* Wolfram verification case SolidMechanics-FEM-Stationary-2D-PlaneStress-0004. *)
youngModulus = 200.*10^9; poissonRatio = 0.3; massDensity = 7800.;
gravity = -9.8; length = 1.; height = 0.2;
distributedLoad = massDensity gravity height;
secondMoment = height^3/12.;
verticalDisplacement[x_] := distributedLoad x^2 *
  (x^2 + 6 length^2 - 4 length x)/(24 secondMoment youngModulus);
payload = <|
  "schema_version" -> 1, "dimensions_m" -> {length, height},
  "young_modulus_pa" -> youngModulus, "poisson_ratio" -> poissonRatio,
  "mass_density_kg_m3" -> massDensity, "gravity_m_s2" -> gravity,
  "tip_vertical_displacement_m" -> N[verticalDisplacement[length], 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
