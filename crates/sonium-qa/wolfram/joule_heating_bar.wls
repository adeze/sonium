(* Homogeneous conducting-bar reduction of Wolfram's Joule-heating model:
   https://reference.wolfram.com/language/PDEModels/tutorial/Multiphysics/ModelCollection/JouleHeating.html?view=all

   Insulation on the horizontal faces makes the unit-square solution
   one-dimensional. DSolve independently evaluates the coupled electrical and
   thermal boundary-value problems used by the Sonium FEM test.
*)
Off[General::munfl];

voltage = 2.;
electricalConductivity = 5.;
thermalConductivity = 10.;
endTemperature = 300.;
sampleX = {0., 0.25, 0.5, 0.75, 1.};

potential = DSolveValue[
  {
    D[electricalConductivity D[v[x], x], x] == 0,
    v[0] == 0,
    v[1] == voltage
  },
  v,
  x
];
temperature = DSolveValue[
  {
    -D[thermalConductivity D[t[x], x], x] ==
      electricalConductivity D[potential[x], x]^2,
    t[0] == endTemperature,
    t[1] == endTemperature
  },
  t,
  x
];
current = electricalConductivity D[potential[x], x];
power = voltage current;

payload = <|
  "schema_version" -> 1,
  "sample_x_m" -> sampleX,
  "potential_v" -> (N[potential[#], 16] & /@ sampleX),
  "temperature_k" -> (N[temperature[#], 16] & /@ sampleX),
  "current_a_per_m" -> N[current, 16],
  "electrical_power_w_per_m" -> N[power, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
