(* Published NDEigensystem example families:
   https://reference.wolfram.com/language/ref/NDEigensystem.html

   These exercise a Schrodinger potential, a variable-coefficient
   Sturm-Liouville operator, and a homogeneous Robin boundary condition.
*)
Off[General::munfl];
region = Rectangle[{0, 0}, {1, 0.8}];
method = {"SpatialDiscretization" -> {"FiniteElement",
  {"MeshOptions" -> {"MaxCellMeasure" -> 0.001}}}};

schrodinger = First@NDEigensystem[
  {-Laplacian[u[x, y], {x, y}] + 5 u[x, y],
   DirichletCondition[u[x, y] == 0, True]},
  u[x, y], {x, y} \[Element] region, 4, Method -> method];

sturmLiouville = First@NDEigensystem[
  {-Div[(1 + x) Grad[u[x, y], {x, y}], {x, y}],
   DirichletCondition[u[x, y] == 0, True]},
  u[x, y], {x, y} \[Element] region, 4, Method -> method];

robin = First@NDEigensystem[
  {-Laplacian[u[x, y], {x, y}] + NeumannValue[2 u[x, y], True]},
  u[x, y], {x, y} \[Element] region, 4, Method -> method];

periodic = First@NDEigensystem[
  {-Laplacian[u[x, y], {x, y}] + 5 u[x, y],
   PeriodicBoundaryCondition[u[x, y], x == 1,
    Function[p, p - {1, 0}]]},
  u[x, y], {x, y} \[Element] region, 3, Method -> method];

blochPhase = Pi/3;
bloch = First@NDEigensystem[
  {-Laplacian[u[x, y], {x, y}] + 5 u[x, y],
   PeriodicBoundaryCondition[Exp[-I blochPhase] u[x, y], x == 1,
    Function[p, p - {1, 0}]]},
  u[x, y], {x, y} \[Element] region, 3, Method -> method];

bandPhases = {0, Pi/2, Pi};
quantumBands = Table[
  First@NDEigensystem[
    {-Laplacian[u[x, y], {x, y}] +
      10 (1 + Cos[2 Pi x]) u[x, y],
     PeriodicBoundaryCondition[Exp[-I phase] u[x, y], x == 1,
      Function[p, p - {1, 0}]]},
    u[x, y], {x, y} \[Element] region, 3, Method -> method],
  {phase, bandPhases}];

(* The published interval form is
     Method -> {"Eigensystem" -> {"FEAST", "Interval" -> {70, 112}}}
   but the free macOS Wolfram Engine distribution does not ship libFEAST.
   Compute the same reference spectrum with the available Arnoldi backend and
   select the requested inclusive interval. Sonium uses sparse shift-invert
   B-Lanczos directly for this case. *)
intervalCandidates = First@NDEigensystem[
  {-Laplacian[u[x, y], {x, y}] + 5 u[x, y],
   DirichletCondition[u[x, y] == 0, True]},
  u[x, y], {x, y} \[Element] region, 8,
  Method -> {"SpatialDiscretization" -> {"FiniteElement",
    {"MeshOptions" -> {"MaxCellMeasure" -> 0.001}}},
    "Eigensystem" -> {"Arnoldi", "Shift" -> 91}}];
interval = Select[intervalCandidates, 70 <= Re[#] <= 112 &];

(* Published radial hydrogen family. The reduced field u(r)=r R(r) obeys a
   one-dimensional Coulomb Hamiltonian. Starting at 10^-3 avoids evaluating
   the singular coefficient at the endpoint; the Arnoldi shift selects the
   negative bound spectrum rather than continuum modes near zero. *)
radialHydrogen = First@NDEigensystem[
  {-1/2 D[u[r], {r, 2}] - u[r]/r,
   DirichletCondition[u[r] == 0, True]},
  u[r], {r, 0.001, 60}, 4,
  Method -> {"SpatialDiscretization" -> {"FiniteElement",
    {"MeshOptions" -> {"MaxCellMeasure" -> 0.05}}},
    "Eigensystem" -> {"Arnoldi", "Shift" -> -0.5,
      "MaxIterations" -> 5000}}];

(* Published damped generalized-wave family. With spatially uniform damping,
   C = 0.2 M, so every NDEigensystem spatial mode reduces exactly to the
   quadratic temporal pencil s^2 + 0.2 s + lambda = 0. *)
dampedSpatial = Sort@First@NDEigensystem[
  {-Laplacian[u[x, y], {x, y}],
   DirichletCondition[u[x, y] == 0, True]},
  u[x, y], {x, y} \[Element] region, 1, Method -> method];
dampedRoots = Flatten[Table[
  {(-0.2 + Sqrt[0.2^2 - 4 lambda])/2,
   (-0.2 - Sqrt[0.2^2 - 4 lambda])/2},
  {lambda, dampedSpatial}]];

(* Published coupled-system family with a genuinely nonsymmetric operator and
   a complex spectrum. The complex Arnoldi shift selects the first conjugate
   branch rather than the nearby real higher-mode eigenvalues. *)
coupledSystem = First@First@NDEigensystem[
  {{-Laplacian[u[x, y], {x, y}] + 20 v[x, y],
    -2 Laplacian[v[x, y], {x, y}] - 20 u[x, y]},
   DirichletCondition[u[x, y] == 0, True],
   DirichletCondition[v[x, y] == 0, True]},
  {u[x, y], v[x, y]}, {x, y} \[Element] region, 1,
  Method -> {"SpatialDiscretization" -> {"FiniteElement",
    {"MeshOptions" -> {"MaxCellMeasure" -> 0.001}}},
    "Eigensystem" -> {"Arnoldi", "Shift" -> 45 + 13 I}}];

(* Published irregular/concave-region family. *)
lShapeRegion = Polygon[{{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}}];
lShape = Sort@First@NDEigensystem[
  {-Laplacian[u[x, y], {x, y}],
   DirichletCondition[u[x, y] == 0, True]},
  u[x, y], {x, y} \[Element] lShapeRegion, 4, Method -> method];

(* Published moving-domain/avoided-crossing family. Unequal overlapping disks
   remove the reflection degeneracy; the modes exchange spatial character as
   the center separation changes. Sonium remeshes every parameter point and
   tracks the fields by overlap on the common physical domain. *)
twoDiskSeparations = {0.65, 0.8, 0.95};
avoidedCrossing = Table[
  twoDiskRegion = RegionUnion[
    Disk[{-separation/2, 0}, 0.72],
    Disk[{separation/2, 0}, 0.58]];
  Sort@First@NDEigensystem[
    {-Laplacian[u[x, y], {x, y}],
     DirichletCondition[u[x, y] == 0, True]},
    u[x, y], {x, y} \[Element] twoDiskRegion, 4,
    Method -> method],
  {separation, twoDiskSeparations}];

(* Published image-derived acoustic-domain family. Use an explicit car-like
   binary silhouette so the fixture is self-contained and does not download
   ExampleData. ImageMesh supplies the region topology; the scaling maps pixel
   coordinates to the same 2-by-1 physical domain used by Sonium. *)
maskData = Table[
  Boole[(9 <= column <= 40 && 7 <= row <= 16) ||
    (17 <= column <= 32 && 17 <= row <= 21)],
  {row, 1, 24}, {column, 1, 48}];
imageRegion = TransformedRegion[
  ImageMesh[Image[maskData]], ScalingTransform[{2/48, 1/24}]];
imageMask = Sort@First@NDEigensystem[
  {-Laplacian[u[x, y], {x, y}],
   DirichletCondition[u[x, y] == 0, True]},
  u[x, y], {x, y} \[Element] imageRegion, 4];

(* Published homogeneous-Neumann family. The omitted boundary condition is
   the natural zero-flux condition and therefore retains the constant mode. *)
neumann = Sort@First@NDEigensystem[
  {-Laplacian[u[x, y], {x, y}]},
  u[x, y], {x, y} \[Element] region, 4, Method -> method];

(* Published one-dimensional interval family. *)
interval1D = Sort@First@NDEigensystem[
  {-D[q[x], {x, 2}], DirichletCondition[q[x] == 0, True]},
  q[x], {x, 0, 1}, 4];

(* Published partial-constraint family: fixed on the vertical edges and
   natural homogeneous Neumann data on the horizontal edges. *)
partiallyConstrained = Sort@First@NDEigensystem[
  {-Laplacian[u[x, y], {x, y}],
   DirichletCondition[u[x, y] == 0, x == 0 || x == 1]},
  u[x, y], {x, y} \[Element] region, 4, Method -> method];

(* Published wave/heat eigenfunction-expansion family. Project one initial
   field onto the numerically computed Dirichlet modes, evolve the same modal
   coordinates under the wave and heat propagators, and sample both fields. *)
{modalEigenvalues, modalEigenfunctions} = NDEigensystem[
  {-Laplacian[u[x, y], {x, y}],
   DirichletCondition[u[x, y] == 0, True]},
  u[x, y], {x, y} \[Element] region, 4, Method -> method];
initialField[x_, y_] := Sin[Pi x] Sin[Pi y/0.8] +
  1/4 Sin[2 Pi x] Sin[Pi y/0.8];
modalCoefficients = Table[
  With[{phi = modalEigenfunctions[[mode]]},
    Quiet[
      NIntegrate[initialField[x, y] phi, {x, 0, 1}, {y, 0, 0.8}] /
      NIntegrate[phi^2, {x, 0, 1}, {y, 0, 0.8}],
      {NIntegrate::slwcon, NIntegrate::eincr}]],
  {mode, Length[modalEigenvalues]}];
waveSample = Sum[
  modalCoefficients[[mode]] Cos[Sqrt[modalEigenvalues[[mode]]] 0.07]
    (modalEigenfunctions[[mode]] /. {x -> 0.375, y -> 0.3}),
  {mode, Length[modalEigenvalues]}];
heatSample = Sum[
  modalCoefficients[[mode]] Exp[-0.2 modalEigenvalues[[mode]] 0.03]
    (modalEigenfunctions[[mode]] /. {x -> 0.375, y -> 0.3}),
  {mode, Length[modalEigenvalues]}];

exactSchrodinger = Sort[N[
  {(1 + 1/0.8^2) Pi^2 + 5, (4 + 1/0.8^2) Pi^2 + 5,
   (1 + 4/0.8^2) Pi^2 + 5, (4 + 4/0.8^2) Pi^2 + 5}, 16]];
payload = <|
  "schema_version" -> 11,
  "schrodinger" -> Sort[N[schrodinger, 16]],
  "schrodinger_exact" -> exactSchrodinger,
  "sturm_liouville" -> Sort[N[sturmLiouville, 16]],
  "robin" -> Sort[N[robin, 16]],
  "periodic" -> Sort[N[periodic, 16]],
  "bloch" -> Sort[Re[N[bloch, 16]]],
  "quantum_bands" -> Map[Sort, Re[N[quantumBands, 16]]],
  "interval" -> Sort[Re[N[interval, 16]]],
  "radial_hydrogen" -> Sort[Re[N[radialHydrogen, 16]]],
  "damped_wave" -> ({Re[#], Im[#]} & /@ N[dampedRoots, 16]),
  "coupled_system" -> {Re[N[coupledSystem, 16]], Im[N[coupledSystem, 16]]},
  "l_shape" -> N[lShape, 16],
  "avoided_crossing" -> Map[Sort, Re[N[avoidedCrossing, 16]]],
  "image_mask" -> N[imageMask, 16],
  "neumann" -> N[neumann, 16],
  "wave_heat_samples" -> N[{waveSample, heatSample}, 16],
  "interval_1d" -> N[interval1D, 16],
  "partially_constrained" -> N[partiallyConstrained, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
