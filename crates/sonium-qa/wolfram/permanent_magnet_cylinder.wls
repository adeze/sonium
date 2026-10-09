(* Literal first model from Wolfram's "Magnetostatics for Permanent Magnets":
   https://reference.wolfram.com/language/PDEModels/tutorial/Electromagnetics/MagnetostaticsForPermanentMagnets.html?view=all

   The page publishes field plots rather than a numerical table. This optional
   extractor therefore archives deterministic exterior-axis samples. It also
   emits the infinite-domain finite-cylinder formula as a useful limit check;
   the Wolfram FEM values differ slightly because Vm == 0 is imposed at r = 2 m.
*)

Needs["NDSolve`FEM`"];

mu0 = 4 Pi 10^-7;
magnetRadius = 0.1;
magnetHalfLength = 0.1;
magnetization = 1.2 10^6;
outerRadius = 2.;

magnet = Cylinder[
  {{0, 0, -magnetHalfLength}, {0, 0, magnetHalfLength}},
  magnetRadius
];
sphere = Ball[{0, 0, 0}, outerRadius];
vars = {Vm[x, y, z], {x, y, z}};
pars = <|
  "Magnetization" -> {
    0,
    0,
    Piecewise[{{magnetization, RegionMember[magnet][{x, y, z}]}}, 0]
  }
|>;

outerCondition = MagneticPotentialCondition[
  x^2 + y^2 + z^2 >= outerRadius^2,
  vars,
  pars
];
domain = RegionDifference[sphere, magnet];
mesh = ToElementMesh[domain, "RegionHoles" -> None];
operator = MagnetostaticPDEComponent[vars, pars];
VmFun = NDSolveValue[
  {operator == 0, outerCondition},
  Vm,
  Element[{x, y, z}, mesh]
];

samplePoints = {
  {0., 0., 0.15},
  {0., 0., 0.2},
  {0., 0., 0.3},
  {0., 0., 0.5},
  {0., 0., 1.0}
};
gradVm = Grad[VmFun[x, y, z], {x, y, z}];
bAt[point_] := N[-mu0 (gradVm /. Thread[{x, y, z} -> point]), 16];

(* Axial field outside a uniformly magnetized finite cylinder in an unbounded
   vacuum. This is not substituted for the finite-domain Wolfram solution. *)
infiniteDomainBz[z_] := mu0 magnetization/2 (
  (z + magnetHalfLength)/Sqrt[magnetRadius^2 + (z + magnetHalfLength)^2] -
  (z - magnetHalfLength)/Sqrt[magnetRadius^2 + (z - magnetHalfLength)^2]
);

payload = <|
  "schema_version" -> 1,
  "magnet_radius_m" -> magnetRadius,
  "magnet_length_m" -> 2 magnetHalfLength,
  "outer_sphere_radius_m" -> outerRadius,
  "magnetization_a_m" -> magnetization,
  "remanent_flux_density_t" -> N[mu0 magnetization, 16],
  "sample_points_m" -> samplePoints,
  "magnetic_potential_a" -> (N[VmFun @@ #, 16] & /@ samplePoints),
  "flux_density_t" -> (bAt /@ samplePoints),
  "infinite_domain_axial_flux_density_t" ->
    (N[infiniteDomainBz[#[[3]]], 16] & /@ samplePoints)
|>;

WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
