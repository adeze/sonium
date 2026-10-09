(* Published NDEigensystem structural-mode families:
   constrained plane-stress elasticity and Euler-Bernoulli beam resonance.
   https://reference.wolfram.com/language/ref/NDEigensystem.html *)
Off[General::munfl];

(* Plane-stress cantilever. The operator is written explicitly so its material
   convention is identical to Sonium's vector H1 plane-stress form. *)
e = 10.;
nu = 0.25;
rho = 2.;
c11 = e/(1 - nu^2);
c12 = e nu/(1 - nu^2);
mu = e/(2 (1 + nu));
structuralRegion = Rectangle[{0, 0}, {1, 0.2}];
structuralMethod = {"SpatialDiscretization" -> {"FiniteElement",
  {"MeshOptions" -> {"MaxCellMeasure" -> 0.0002}}}};
structural = Sort@First@NDEigensystem[
  {{-(c11 D[ux[x, y], {x, 2}] + mu D[ux[x, y], {y, 2}] +
        (c12 + mu) D[uy[x, y], x, y])/rho,
    -(mu D[uy[x, y], {x, 2}] + c11 D[uy[x, y], {y, 2}] +
        (c12 + mu) D[ux[x, y], x, y])/rho},
   DirichletCondition[ux[x, y] == 0, True],
   DirichletCondition[uy[x, y] == 0, True]},
  {ux[x, y], uy[x, y]}, {x, y} \[Element] structuralRegion, 3,
  Method -> structuralMethod];

(* Cantilever Euler-Bernoulli beam. Sonium uses consistent-mass frame
   elements; the slender geometry keeps the first three frame modes on the
   bending branch represented by this published fourth-order equation. *)
length = 1;
width = 0.1;
height = 0.05;
beamE = 2.*10^7;
beamRho = 1000.;
area = width height;
secondMoment = width height^3/12;
(* The free Engine's NDEigensystem backend does not accept the four endpoint
   constraints of this fourth-order ODE directly. Evaluate the same published
   cantilever characteristic equation, cos(beta) cosh(beta) + 1 == 0. *)
beamRoots = beta /. (FindRoot[
      Cos[beta] Cosh[beta] + 1 == 0, {beta, #}] & /@ {1.9, 4.7, 7.9});
beamFrequencies = beamRoots^2/(2 Pi length^2) Sqrt[
  beamE secondMoment/(beamRho area)];

payload = <|
  "schema_version" -> 1,
  "structural" -> N[structural, 16],
  "beam_frequencies_hz" -> N[beamFrequencies, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
