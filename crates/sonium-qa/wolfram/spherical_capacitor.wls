(* Radially symmetric form of Wolfram's published spherical-capacitor PDE:
   https://reference.wolfram.com/language/PDEModels/tutorial/Electromagnetics/ModelCollection/SphericalCapacitor.html?view=all
   DSolve evaluates the Laplace boundary-value problem independently. *)

Off[General::munfl];
innerRadius = 0.01;
outerRadius = 0.05;
innerPotential = 100.;
vacuumPermittivity = 8.8541878128*10^-12;
sampleRadii = {0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05};
potential = DSolveValue[
  {
    1/r^2 D[r^2 D[v[r], r], r] == 0,
    v[innerRadius] == innerPotential,
    v[outerRadius] == 0
  },
  v,
  r
];
fieldRadii = Rest[Most[sampleRadii]];
radialField[r_] :=
  innerPotential innerRadius outerRadius/
    ((outerRadius - innerRadius) r^2);
capacitance =
  4 Pi vacuumPermittivity innerRadius outerRadius/
    (outerRadius - innerRadius);
innerCharge = capacitance innerPotential;
electrostaticEnergy = capacitance innerPotential^2/2;

payload = <|
  "schema_version" -> 2,
  "inner_radius_m" -> innerRadius,
  "outer_radius_m" -> outerRadius,
  "inner_potential_v" -> innerPotential,
  "sample_radii_m" -> sampleRadii,
  "potential_v" -> (N[potential[#], 16] & /@ sampleRadii),
  "field_radii_m" -> fieldRadii,
  "radial_field_v_m" -> (N[radialField[#], 16] & /@ fieldRadii),
  "vacuum_capacitance_f" -> N[capacitance, 16],
  "inner_charge_c" -> N[innerCharge, 16],
  "electrostatic_energy_j" -> N[electrostaticEnergy, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
