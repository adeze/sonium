(* Published Wolfram example:
   https://reference.wolfram.com/language/ref/NDEigensystem.html
   "Compute first 6 eigenfunctions of a circular membrane with edges clamped."
*)
Off[General::munfl];
{values, functions} = NDEigensystem[
  {-Laplacian[u[x, y], {x, y}], DirichletCondition[u[x, y] == 0, True]},
  u[x, y], {x, y} \[Element] Disk[], 6,
  Method -> {"SpatialDiscretization" -> {"FiniteElement", {"MeshOptions" -> {"MaxCellMeasure" -> 0.001}}}}
];
exact = Sort[N[{BesselJZero[0, 1]^2, BesselJZero[1, 1]^2, BesselJZero[1, 1]^2,
  BesselJZero[2, 1]^2, BesselJZero[2, 1]^2, BesselJZero[0, 2]^2}, 16]];
payload = <|"schema_version" -> 1, "ndeigensystem" -> Sort[N[values, 16]], "exact" -> exact|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
