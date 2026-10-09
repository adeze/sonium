(* Literal first current-carrying wire model from Wolfram's
   "Quasistatic Magnetic Fields" monograph.  The page publishes plots rather
   than a numerical table; this optional extractor emits deterministic samples
   for the exact published geometry and parameters. *)
Needs["NDSolve`FEM`"];

mu0 = 4 Pi 10^-7;
wireRadius = 0.2;
outerRadius = 1.;
currentDensity = 1.;

wireRegion = Disk[{0, 0}, wireRadius];
airRegion = Disk[{0, 0}, outerRadius];
domain = RegionDifference[airRegion, wireRegion];
vars = {{0, 0, Az[x, y]}, {x, y}};
pars = <|"RelativePermeability" -> 1|>;
wireMember = RegionMember[wireRegion];
pars["ExternalCurrentSource"] = {
  0,
  0,
  Piecewise[{{currentDensity, wireMember[{x, y}]}}, 0]
};
outerCondition = MagneticPotentialCondition[
  x^2 + y^2 >= outerRadius^2,
  vars,
  pars
];
mesh = ToElementMesh[domain, "RegionHoles" -> None];
operator = MagneticPDEComponent[vars, pars];
AzFun = NDSolveValue[
  {operator == 0, outerCondition},
  Az,
  Element[{x, y}, mesh]
];

sampleRadii = {0., 0.1, 0.2, 0.4, 0.6, 0.8};
bField = Curl[{0, 0, AzFun[x, y]}, {x, y, z}];
bThetaAt[r_] := If[
  r == 0.,
  0.,
  N[(bField[[2]] /. {x -> r, y -> 0, z -> 0}), 16]
];

payload = <|
  "schema_version" -> 1,
  "wire_radius_m" -> wireRadius,
  "outer_radius_m" -> outerRadius,
  "current_density_a_m2" -> currentDensity,
  "sample_radii_m" -> sampleRadii,
  "vector_potential_t_m" -> (N[AzFun[#, 0], 16] & /@ sampleRadii),
  "flux_density_t" -> (bThetaAt /@ sampleRadii)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
