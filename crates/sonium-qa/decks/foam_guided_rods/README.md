# Solved guided H8 foam rods

Three quasistatic cases run the production H8 strain/force kernel and actual
native FOAM_PLAS LAW33 or quadratic isochoric LAW76 material. The 100 mm rod
has a 10 mm square cross-section. Transverse nodal coordinates are fixed;
each cross-section ties its four axial DOFs. Only x=0 is fixed axially.
Every interior plane and the loaded end are free, with displacements solved
by global Newton iterations and an assembled tridiagonal tangent. Material
histories commit only after force convergence.

The H8 operator receives an equivalent axial nodal increment producing the
exact logarithmic strain log(l_new/l_old). Current deformed H8 geometry books
volume ratio J and Cauchy traction forces. Hourglass and shock viscosity are
zero for this quasistatic guided mode. This is a spatial rod benchmark, not
unconstrained 3D dynamics, rate/failure accuracy, or the original foam fall-pad.

Reference traction magnitude is T(x)=peak[0.6+0.4(x/L)^2]. The right-end
traction is -peak and distributed reference-length force density is
b(x)=0.8 peak A x/L². The independent continuum axial stress is -T(x) and
the fixed-end reaction is 0.6 peak A. Consistent nodal body forces integrate
linear H8 axial shape functions exactly. Element axial stress equals the
cell-average reference traction; its midpoint difference is O(h²).

For monotone LAW33 compression, until capping J=exp(-T/E). In the capped
branch, T=500000+2261180(1-J)+100000(1-J)/(J-0.1). A stable quadratic root
in z=1-J gives J directly. The elastic/capped transition solves equality
with the uncapped exponential. This reference never calls FoamPlas33.

For LAW76, the elastic branch ends at T=900000. Thereafter P=-E log(J)/3,
ep=(3P-T)/E, q=1.5(T-P), and H=1+20ep. Substitution into
q²=3 tau² H² + 9 tau²(C-T0)/(CT0) H P + 9(CT0-3 tau²)/(CT0) P²
is a quadratic in P. Its admissible root gives J=exp(-3P/E).
T0=500000, C=900000, tau=420000 Pa. This inverse does not call Samp76Quadratic.

Continuum displacement is integral_0^x[J(s)-1]ds. Integration is split at
the final yield front. External work is computed independently by integration
by parts over the load multiplier a: integral A[-T(x)]{J(1,x)-1 -
integral_0^1[J(a,x)-1]da}dx. Doubling both work quadratures from 1024 to 2048
points must change the reference by <=1e-5. No native/engine outputs feed the
reference or calibrate its inputs.

Meshes have 16/32/64 elements and 128 accepted monotone load steps. A second
64-element solve doubles load resolution; a third evolves an elastic material
counterfactual under identical loads. Free force residuals, nonuniform plastic
strain, all-load material/volume references, reactions, stress fields,
external work, and displacement L2 errors are gated.

LAW33 mixed: peak=650 kPa, with elastic and yielded final regions. Midpoint
error cancellation near the yield front gives an oscillating finest-pair
rate; that value remains visible without a bound. Its 16-to-64 displacement
rate must be >=1.5 and fine displacement error <=0.5%. It is accompanied by
LAW33 smooth: peak=900 kPa, fully plastic at the final load. Both full-range
and finest-pair displacement rates must be >=1.5. LAW76 peak=1300 kPa has
elastic/plastic regions and gates both rates too. These separate cases retain
the mixed-region accuracy check while providing a smooth convergence gate.

Reference parameters and scope are checked against reference.json, with
SHA-256 provenance in each QA_RESULT. Run:

```sh
cargo run -p sonium-qa --features qa-binaries --bin qa-fem-foam-laws
```

All three cases require numerical_solve_executed=1. The five original
integrity/material replay cases remain separately reported, including
numerical_solve_executed=0 for the prescribed material-point paths.
