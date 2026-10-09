(* Published Wolfram verification case
   SolidMechanics-FEM-Stationary-2D-PlaneStress-0003:
   https://reference.wolfram.com/language/PDEModels/tutorial/StructuralMechanics/SolidMechanicsVerificationTests.html?view=all

   This evaluates the Euler-Bernoulli displacement expression printed by the
   official example at the same eleven x coordinates used by its tests.
*)
Off[General::munfl];

length = 1.;
height = 0.2;
thickness = 0.1;
youngModulus = 200. * 10^9;
poissonRatio = 0.3;
massDensity = 7800.;
topTraction = -10.^6;
sampleX = N[Range[0, 1, 1/10], 16];
secondMomentPerDepth = height^3/12;
verticalDisplacement[x_] :=
  topTraction x^2 (x^2 + 6 length^2 - 4 length x)/
    (24 secondMomentPerDepth youngModulus);

payload = <|
  "schema_version" -> 1,
  "case_id" -> "SolidMechanics-FEM-Stationary-2D-PlaneStress-0003",
  "dimensions_m" -> {length, height, thickness},
  "young_modulus_pa" -> youngModulus,
  "poisson_ratio" -> poissonRatio,
  "mass_density_kg_m3" -> massDensity,
  "top_traction_pa" -> topTraction,
  "sample_x_m" -> sampleX,
  "vertical_displacement_m" -> (N[verticalDisplacement[#], 16] & /@ sampleX)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
