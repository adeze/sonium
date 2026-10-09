# BEM acoustic convention contract

The public BEM/XEM boundary uses one harmonic convention:

* pressure fields use `exp(-i omega t)`;
* `k = omega / c` and the outgoing Green function is
  `G(r) = exp(i k r) / (4 pi r)`;
* boundary normals point out of the modeled acoustic domain;
* pressure is in pascals and normal particle velocity is in metres/second;
* source amplitudes and solved complex fields are peak phasors unless a
  consumer explicitly declares RMS values in the XEM convention registry;
* Burton–Miller coupling is dimensioned in the same length convention as the
  operator and must be finite and non-zero for a checked problem;
* transpose means `A^T` (no conjugation), while Hermitian means `A^H` and is
  the adjoint used by gradient and reciprocity checks.

The sign and orientation are carried by `sonium_xem::ConventionRegistry` and
are admitted by `BemProblem::from_xem`. A checked constructor rejects a
problem whose harmonic sign, coordinate frame, normal orientation, field
units, or scalar layout is not understood. Legacy constructors retain the
canonical convention for compatibility with analytical fixtures; they are not
an authorization to mix conventions silently.

Every independent reference should report the convention alongside frequency,
speed of sound, density, boundary condition, and whether amplitudes are peak
or RMS. A field comparison is invalid when any of those declarations differ.
