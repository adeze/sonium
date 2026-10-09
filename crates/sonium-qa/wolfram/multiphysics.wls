(* Wolfram Engine references for Sonium's ten multiphysics model families.
   The formulas and case names follow reviews/20260806-multiphysics.md in the
   all_of_sonium integration repository. Keep stdout machine-readable. *)

ClearAll[toVector];
toVector[value_] := N[Flatten[{value} /. z_Complex :> {Re[z], Im[z]}], 16];

(* Truncated diffusion series legitimately underflow once their terms are far
   below machine precision. Keep those diagnostics out of the JSON protocol. *)
Off[General::munfl];

(* Fluid-structure interaction. *)
fsiAddedMass[] := Module[{rho = 1000., area = 0.01, length = 1., mass = 1., stiffness = 100.},
  {rho area length, Sqrt[stiffness/(mass + rho area length)]}
];
fsiMoensKorteweg[] := Module[{e = 1.*^6, h = 1.*^-3, rho = 1000., radius = 0.01, nu = 0.3},
  Sqrt[e h/(2 rho radius (1 - nu^2))]
];
fsiOscillatingSphere[] := Module[{a = 0.01, rho = 1000., mu = 1.*^-3, omega = 100., lambda},
  lambda = (1 + I) a Sqrt[omega rho/(2 mu)];
  6 Pi mu a (1 + lambda + lambda^2/9)
];

(* Conjugate heat transfer. *)
chtPlaneWall[] := Module[{th = 400., tc = 300., hh = 100., hc = 50., length = 0.02, k = 10., u, q},
  u = 1/(1/hh + length/k + 1/hc);
  q = u (th - tc);
  {q, th - q/hh, tc + q/hc, u}
];
chtGraetz[] := Module[{x = 1., inlet = 300., wall = 400., h = 10., perimeter = 2., mdot = 1., cp = 1000.},
  {3.66, 48/11, wall + (inlet - wall) Exp[-h perimeter x/(mdot cp)]}
];
chtAdiabaticFin[] := Module[{x = 0.04, length = 0.1, thetaB = 100., h = 20., perimeter = 0.04, k = 200., area = 1.*^-4, m},
  m = Sqrt[h perimeter/(k area)];
  {thetaB Cosh[m (length - x)]/Cosh[m length], k area m thetaB Tanh[m length]}
];

(* Electrothermal and induction heating. *)
electrothermalDcBar[] := Module[{x = 0.5, length = 1., voltage = 2., t0 = 300., sigma = 5., k = 10., q},
  q = sigma (voltage/length)^2;
  {voltage x/length, t0 + q x (length - x)/(2 k), q}
];
electrothermalRoundConductor[] := Module[{omega = 1.*^6, radius = 1.*^-3, sigma = 5.8*^7, mu = 4.*^-7 Pi, kappa},
  kappa = Sqrt[-I omega mu sigma];
  kappa BesselJ[0, kappa radius]/(2 Pi radius sigma BesselJ[1, kappa radius])
];
electrothermalDoddDeeds[] := Module[
  {omega = 10000, radius = 1/2000, lift = 1/200, sigma = 58000000, mu = 4 Pi/10000000, integral},
  integral = NIntegrate[
    BesselJ[1, radius alpha]^2 Exp[-2 alpha lift]
      (alpha - Sqrt[alpha^2 + I omega mu sigma])/(alpha + Sqrt[alpha^2 + I omega mu sigma]),
    {alpha, 0, Infinity}, WorkingPrecision -> 30, AccuracyGoal -> 16, PrecisionGoal -> 16
  ];
  I omega mu Pi radius^2 integral
];

(* Thermoelectric transport. *)
thermoelectricLeg[] := Module[
  {x = 0.01, length = 0.02, th = 500., tc = 300., j = 2.*^5, rho = 1.*^-5, k = 1.5, alpha = 200.*^-6,
   slope, curvature, temperature, gradient, heatFlux, potential},
  slope = (tc - th)/length;
  curvature = rho j^2/k;
  temperature = th + slope x + curvature x (length - x)/2;
  gradient = slope + curvature (length - 2 x)/2;
  heatFlux = -k gradient + alpha temperature j;
  potential = -rho j x - alpha (temperature - th);
  {temperature, heatFlux, potential}
];
thermoelectricIoffe[] := Module[{th = 500., tc = 300., zt = 1., root},
  root = Sqrt[1 + zt];
  (th - tc)/th (root - 1)/(root + tc/th)
];
thermoelectricCompatibility[] := (Sqrt[1 + 1.] - 1)/(200.*^-6 400.);

(* Battery porous-electrode references. *)
tanXEqualsXRoot[index_Integer] := x /. FindRoot[
  Tan[x] == x, {x, index Pi + 1/100, (index + 1/2) Pi - 1/100}, WorkingPrecision -> 40
];
batteryParticle[] := Module[
  {c0 = 30000., flux = 1.*^-5, radius = 1.*^-3, diffusivity = 1.*^-8, time = 1., tau, roots, series},
  tau = diffusivity time/radius^2;
  roots = tanXEqualsXRoot /@ Range[100];
  series = Total[Exp[-roots^2 tau]/roots^2];
  {c0 - flux radius/diffusivity (3 tau + 1/5 - 2 series), c0 - 3 flux time/radius}
];
batterySand[] := Pi 1.*^-9 1000.^2/(4 1.^2);
batteryDeLevie[] := Module[{omega = 1.*^6, resistance = 10., capacitance = 1.*^-2, length = 1., propagation, characteristic},
  propagation = Sqrt[I omega resistance capacitance];
  characteristic = Sqrt[resistance/(I omega capacitance)];
  characteristic Coth[propagation length]
];

(* Inductionless magnetohydrodynamics. *)
mhdHartmann[] := Module[{y = 0., a = 0.1, dpdx = -100., sigma = 1.*^6, b = 0.5, ha = 10., scale},
  scale = -dpdx/(sigma b^2);
  {scale (1 - Cosh[ha y/a]/Cosh[ha]), 1 - Tanh[ha]/ha}
];
mhdShercliff[] := Module[{distance = 0.05, halfWidth = 1., ha = 100., core = 2., delta},
  delta = halfWidth/Sqrt[ha];
  {delta, core (1 - Exp[-distance/delta])}
];
mhdHunt[] := Module[{ratio}, ratio = 2. 0.1/(1. 0.2); {ratio, ratio/(1 + ratio)}];

(* Thin-film lubrication. *)
sliderPressure[x_, length_, h0_, h1_, mu_, speed_] := Module[{slope, h, harmonicHalf},
  slope = (h1 - h0)/length;
  h = h0 + slope x;
  harmonicHalf = h0 h1/(h0 + h1);
  (6 mu speed/slope) (-1/h + 1/h0 + harmonicHalf (1/h^2 - 1/h0^2))
];
lubricationSlider[] := Module[{length = 1/10, h0 = 1/10000, h1 = 1/20000, mu = 2/25, speed = 3, load},
  load = NIntegrate[sliderPressure[x, length, h0, h1, mu, speed], {x, 0, length},
    WorkingPrecision -> 30, AccuracyGoal -> 18, PrecisionGoal -> 18];
  {sliderPressure[length/2, length, h0, h1, mu, speed], load}
];
lubricationSommerfeld[] := Module[{theta = 0.7, radius = 0.05, clearance = 50.*^-6, e = 0.4, mu = 0.02, speed = 2.},
  6 mu speed radius/clearance^2 e Sin[theta] (2 + e Cos[theta])/((2 + e^2) (1 + e Cos[theta])^2)
];
lubricationSqueeze[] := Module[{r = 0.01, radius = 0.02, gap = 100.*^-6, rate = -2.*^-4, mu = 0.1},
  {-3 mu rate (radius^2 - r^2)/gap^3, -3 Pi mu radius^4 rate/(2 gap^3)}
];

(* Diffuse-interface two-phase flow. *)
phaseFieldInterface[] := Module[{epsilon = 1/100, gamma = 7/100, beta, kappa, energy},
  beta = 3 gamma/(2 Sqrt[2] epsilon);
  kappa = beta epsilon^2;
  energy = NIntegrate[
    beta (Tanh[x/(Sqrt[2] epsilon)]^2 - 1)^2/4
      + kappa Sech[x/(Sqrt[2] epsilon)]^4/(4 epsilon^2),
    {x, -12 epsilon, 12 epsilon}, WorkingPrecision -> 30
  ];
  {Tanh[10/Sqrt[2]], energy}
];
phaseFieldLaplace[] := {0.1/0.5, 2 0.1/0.5};
phaseFieldCapillary[] := Sqrt[0.07 2.^3/(1000. + 1.)];

(* Nonlinear reacting flow. *)
reactingFisherKpp[] := 2 Sqrt[2. 8.];
reactingBurkeSchumann[] := Module[{length = 1., zst = 0.25, x, z, fuel, oxidizer, progress, temperature},
  x = length (1 - zst);
  z = 1 - x/length;
  {fuel, oxidizer, progress} = If[z >= zst,
    {(z - zst)/(1 - zst), 0., (1 - z)/(1 - zst)},
    {0., (zst - z)/zst, z/zst}
  ];
  temperature = 300. + progress (2200. - 300.);
  {z, fuel, oxidizer, temperature, x}
];
reactingFrankKamenetskii[] := {0.878, 2., 3.32};

(* Participating-media radiation transport. *)
radiationSlab[] := Module[{tau = 0.2, mu = 0.5, incident = 3., source = 0., attenuation},
  attenuation = Exp[-tau/Abs[mu]];
  incident attenuation + source (1 - attenuation)
];
(* High-precision conservative isotropic H(1) benchmark tabulated by Chandrasekhar. *)
radiationChandrasekhar[] := 2.9078105291;
radiationMarshak[] := Module[{x = 1., time = 1., diffusion = 1., boundary = 4.},
  boundary Erfc[x/(2 Sqrt[diffusion time])]
];

cases = <|
  "fsi.added_mass_piston" -> toVector[fsiAddedMass[]],
  "fsi.moens_korteweg" -> toVector[fsiMoensKorteweg[]],
  "fsi.oscillating_sphere" -> toVector[fsiOscillatingSphere[]],
  "conjugate_heat_transfer.plane_wall" -> toVector[chtPlaneWall[]],
  "conjugate_heat_transfer.graetz" -> toVector[chtGraetz[]],
  "conjugate_heat_transfer.adiabatic_fin" -> toVector[chtAdiabaticFin[]],
  "electrothermal.dc_heated_bar" -> toVector[electrothermalDcBar[]],
  "electrothermal.round_conductor" -> toVector[electrothermalRoundConductor[]],
  "electrothermal.dodd_deeds" -> toVector[electrothermalDoddDeeds[]],
  "thermoelectric.constant_leg" -> toVector[thermoelectricLeg[]],
  "thermoelectric.ioffe_efficiency" -> toVector[thermoelectricIoffe[]],
  "thermoelectric.compatibility" -> toVector[thermoelectricCompatibility[]],
  "battery.spherical_particle" -> toVector[batteryParticle[]],
  "battery.sand_depletion" -> toVector[batterySand[]],
  "battery.de_levie" -> toVector[batteryDeLevie[]],
  "mhd.hartmann" -> toVector[mhdHartmann[]],
  "mhd.shercliff" -> toVector[mhdShercliff[]],
  "mhd.hunt" -> toVector[mhdHunt[]],
  "lubrication.linear_slider" -> toVector[lubricationSlider[]],
  "lubrication.sommerfeld" -> toVector[lubricationSommerfeld[]],
  "lubrication.squeeze_film" -> toVector[lubricationSqueeze[]],
  "phase_field_flow.planar_interface" -> toVector[phaseFieldInterface[]],
  "phase_field_flow.laplace_pressure" -> toVector[phaseFieldLaplace[]],
  "phase_field_flow.capillary_wave" -> toVector[phaseFieldCapillary[]],
  "reacting_flow.fisher_kpp" -> toVector[reactingFisherKpp[]],
  "reacting_flow.burke_schumann" -> toVector[reactingBurkeSchumann[]],
  "reacting_flow.frank_kamenetskii" -> toVector[reactingFrankKamenetskii[]],
  "radiation_transport.absorbing_slab" -> toVector[radiationSlab[]],
  "radiation_transport.chandrasekhar_h" -> toVector[radiationChandrasekhar[]],
  "radiation_transport.marshak_wave" -> toVector[radiationMarshak[]]
|>;

Print[ExportString[<|"schema_version" -> 1, "cases" -> cases|>, "RawJSON", "Compact" -> True]];
