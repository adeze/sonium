//! Two-dimensional, revolved, and uniform-flow FW-H evaluators.
//!
//! The planar evaluator integrates a closed contour with the Hankel kernel.
//! The revolved evaluator spins meridional strips around the axis with a
//! modal `exp(imφ)` weight, so axisymmetric and spinning surfaces share one
//! path. The convected evaluators replace the free kernel with the Garrick
//! uniform-flow Green function plus the linearized permeable-surface mass
//! flux `Q' = ρu_n + (U·n)p/c²`, recovering the quiescent forms at zero
//! Mach. Thickness/loading splits are preserved in every dimension.

use num_complex::Complex64;
use sonium_wave::analytical::Point;
use sonium_wave::analytical::azimuthal::AzimuthalMode;
use sonium_wave::special::helmholtz::{
    HelmholtzFunctionError, convected_greens_function_gradient_2d, flip_time_convention,
    try_convected_greens_function_2d, try_convected_greens_function_3d,
    try_convected_greens_function_gradient_3d, try_greens_function_gradient_2d,
};
use sonium_wave::special::ring::{
    azimuthal_ring_integral, azimuthal_ring_integral_adaptive, gauss_legendre_rule,
};

use super::error::{AeroacousticsError, require_finite, require_positive};
use super::fwh::{FwhComponents, FwhSurfacePatch, fwh_patch_bare};

fn map_kernel_error(error: HelmholtzFunctionError) -> AeroacousticsError {
    match error {
        HelmholtzFunctionError::InvalidDistance(_) => AeroacousticsError::SingularObserver,
        HelmholtzFunctionError::InvalidWaveNumber(value) => {
            AeroacousticsError::NonFiniteParameter {
                name: "angular_frequency",
                value,
            }
        }
        HelmholtzFunctionError::NonFiniteInput(name) => AeroacousticsError::NonFiniteField { name },
        HelmholtzFunctionError::NonFiniteResult(name) => {
            AeroacousticsError::NonFiniteField { name }
        }
        HelmholtzFunctionError::InvalidMach(magnitude) => {
            AeroacousticsError::InvalidMach { magnitude }
        }
    }
}

/// One straight segment of a closed planar FW-H contour.
#[derive(Debug, Clone, Copy)]
pub struct FwhContourSegment {
    /// Segment midpoint `[m]` in the plane.
    pub position: [f64; 2],
    /// Unit normal pointing to the observer region.
    pub outward_normal: [f64; 2],
    /// Segment quadrature length `[m]`.
    pub length: f64,
    /// Complex acoustic pressure `[Pa]`.
    pub pressure: Complex64,
    /// Complex outward particle velocity `[m/s]`.
    pub normal_velocity: Complex64,
}

impl FwhContourSegment {
    /// Builds and validates one contour quadrature segment.
    pub fn new(
        position: [f64; 2],
        outward_normal: [f64; 2],
        length: f64,
        pressure: Complex64,
        normal_velocity: Complex64,
    ) -> Result<Self, AeroacousticsError> {
        require_positive("contour segment length", length)?;
        if position.iter().any(|x| !x.is_finite()) {
            return Err(AeroacousticsError::NonFiniteField { name: "position" });
        }
        if outward_normal.iter().any(|x| !x.is_finite()) {
            return Err(AeroacousticsError::NonFiniteField {
                name: "outward_normal",
            });
        }
        let norm =
            (outward_normal[0] * outward_normal[0] + outward_normal[1] * outward_normal[1]).sqrt();
        if (norm - 1.0).abs() > 1e-8 {
            return Err(AeroacousticsError::NonFiniteField {
                name: "unit outward_normal",
            });
        }
        if !pressure.re.is_finite() || !pressure.im.is_finite() {
            return Err(AeroacousticsError::NonFiniteField { name: "pressure" });
        }
        if !normal_velocity.re.is_finite() || !normal_velocity.im.is_finite() {
            return Err(AeroacousticsError::NonFiniteField {
                name: "normal_velocity",
            });
        }
        Ok(Self {
            position,
            outward_normal,
            length,
            pressure,
            normal_velocity,
        })
    }
}

fn validate_segment(segment: &FwhContourSegment) -> Result<(), AeroacousticsError> {
    require_positive("contour segment length", segment.length)?;
    if segment.position.iter().any(|x| !x.is_finite())
        || segment.outward_normal.iter().any(|x| !x.is_finite())
        || !segment.pressure.re.is_finite()
        || !segment.pressure.im.is_finite()
        || !segment.normal_velocity.re.is_finite()
        || !segment.normal_velocity.im.is_finite()
    {
        return Err(AeroacousticsError::NonFiniteField {
            name: "FW-H contour segment",
        });
    }
    let norm = (segment.outward_normal[0] * segment.outward_normal[0]
        + segment.outward_normal[1] * segment.outward_normal[1])
        .sqrt();
    if (norm - 1.0).abs() > 1e-8 {
        return Err(AeroacousticsError::NonFiniteField {
            name: "unit outward_normal",
        });
    }
    Ok(())
}

/// Evaluates a stationary planar FW-H contour at one exterior observer.
///
/// Thickness comes from normal velocity through the Hankel kernel;
/// loading comes from surface pressure through its source-normal
/// derivative (the negated, conjugated field gradient).
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid parameters, segments, or a
/// coincident observer.
pub fn stationary_fwh_pressure_2d(
    segments: &[FwhContourSegment],
    observer: [f64; 2],
    angular_frequency: f64,
    density: f64,
    speed_of_sound: f64,
) -> Result<FwhComponents, AeroacousticsError> {
    require_finite("angular_frequency", angular_frequency)?;
    require_positive("density", density)?;
    require_positive("speed_of_sound", speed_of_sound)?;
    if observer.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField { name: "observer" });
    }
    let wavenumber = angular_frequency / speed_of_sound;
    let mut thickness = Complex64::new(0.0, 0.0);
    let mut loading = Complex64::new(0.0, 0.0);
    for segment in segments {
        validate_segment(segment)?;
        let r = ((observer[0] - segment.position[0]).powi(2)
            + (observer[1] - segment.position[1]).powi(2))
        .sqrt();
        if r <= f64::EPSILON {
            return Err(AeroacousticsError::SingularObserver);
        }
        let green = flip_time_convention(sonium_wave::special::helmholtz::greens_function_2d(
            r, wavenumber,
        ));
        let gradient = try_greens_function_gradient_2d(&segment.position, &observer, wavenumber)
            .map_err(map_kernel_error)?;
        // Source-normal derivative: negate the field gradient, then flip
        // the wave-crate time convention to the aero convention.
        let source_derivative = -flip_time_convention(
            gradient[0] * segment.outward_normal[0] + gradient[1] * segment.outward_normal[1],
        );
        thickness += Complex64::new(0.0, angular_frequency * density)
            * segment.normal_velocity
            * green
            * segment.length;
        loading += segment.pressure * source_derivative * segment.length;
    }
    Ok(FwhComponents {
        thickness,
        loading,
        total: thickness + loading,
    })
}

/// One meridional strip revolved into a spinning FW-H surface band.
///
/// The strip runs from `node_a` to `node_b` in `(r, z)` with `r ≥ 0`;
/// pressure and velocity are modal amplitudes multiplied by `exp(imφ)`
/// around the hoop. Strips touching the axis are allowed: interior
/// meridional nodes keep `r > 0`.
#[derive(Debug, Clone, Copy)]
pub struct FwhRevolvedPatch {
    /// First meridional endpoint `(r, z)` `[m]`.
    pub node_a: [f64; 2],
    /// Second meridional endpoint `(r, z)` `[m]`.
    pub node_b: [f64; 2],
    /// Unit meridional outward normal `(n_r, n_z)`.
    pub outward_normal: [f64; 2],
    /// Azimuthal spinning order of the surface data.
    pub azimuthal: AzimuthalMode,
    /// Modal pressure amplitude `[Pa]`.
    pub pressure: Complex64,
    /// Modal outward-velocity amplitude `[m/s]`.
    pub normal_velocity: Complex64,
}

impl FwhRevolvedPatch {
    /// Builds and validates one revolved patch.
    pub fn new(
        node_a: [f64; 2],
        node_b: [f64; 2],
        outward_normal: [f64; 2],
        azimuthal: AzimuthalMode,
        pressure: Complex64,
        normal_velocity: Complex64,
    ) -> Result<Self, AeroacousticsError> {
        validate_revolved_nodes(node_a, node_b, outward_normal)?;
        if !pressure.re.is_finite() || !pressure.im.is_finite() {
            return Err(AeroacousticsError::NonFiniteField { name: "pressure" });
        }
        if !normal_velocity.re.is_finite() || !normal_velocity.im.is_finite() {
            return Err(AeroacousticsError::NonFiniteField {
                name: "normal_velocity",
            });
        }
        Ok(Self {
            node_a,
            node_b,
            outward_normal,
            azimuthal,
            pressure,
            normal_velocity,
        })
    }
}

fn validate_revolved_nodes(
    node_a: [f64; 2],
    node_b: [f64; 2],
    outward_normal: [f64; 2],
) -> Result<f64, AeroacousticsError> {
    if node_a.iter().any(|x| !x.is_finite()) || node_b.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "revolved patch nodes",
        });
    }
    if node_a[0] < 0.0 || node_b[0] < 0.0 {
        return Err(AeroacousticsError::NonFiniteParameter {
            name: "revolved patch radius",
            value: node_a[0].min(node_b[0]),
        });
    }
    if outward_normal.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "outward_normal",
        });
    }
    let norm =
        (outward_normal[0] * outward_normal[0] + outward_normal[1] * outward_normal[1]).sqrt();
    if (norm - 1.0).abs() > 1e-8 {
        return Err(AeroacousticsError::NonFiniteField {
            name: "unit outward_normal",
        });
    }
    let length = ((node_b[0] - node_a[0]).powi(2) + (node_b[1] - node_a[1]).powi(2)).sqrt();
    if !length.is_finite() || length <= 0.0 {
        return Err(AeroacousticsError::NonPositiveParameter {
            name: "revolved strip length",
            value: length,
        });
    }
    Ok(length)
}

/// Revolved-surface quadrature controls.
#[derive(Debug, Clone, Copy)]
pub struct RevolvedQuadrature {
    /// Relative tolerance of the adaptive azimuthal rule.
    pub tolerance: f64,
    /// Fixed Gauss-Legendre order along each meridional strip.
    pub meridional_order: usize,
}

impl RevolvedQuadrature {
    /// Builds checked quadrature controls.
    ///
    /// # Errors
    ///
    /// Returns [`AeroacousticsError`] for a non-positive tolerance.
    pub fn new(tolerance: f64, meridional_order: usize) -> Result<Self, AeroacousticsError> {
        require_positive("ring tolerance", tolerance)?;
        Ok(Self {
            tolerance,
            meridional_order,
        })
    }
}

/// Evaluates stationary revolved FW-H patches at one exterior observer.
///
/// Each meridional strip uses a fixed Gauss-Legendre rule while the hoop
/// integrates adaptively to `tolerance`; thickness and loading accumulate
/// through independent adaptive passes so each meets the tolerance.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid parameters, patches, a
/// singular observer, or a missed quadrature tolerance.
pub fn stationary_fwh_ring_pressure(
    patches: &[FwhRevolvedPatch],
    observer: [f64; 3],
    angular_frequency: f64,
    density: f64,
    speed_of_sound: f64,
    quadrature: RevolvedQuadrature,
) -> Result<FwhComponents, AeroacousticsError> {
    revolved_pressure(
        patches,
        observer,
        angular_frequency,
        density,
        speed_of_sound,
        quadrature,
    )
}

/// Fixed-azimuthal-order variant for observer sweeps.
///
/// Replaces the adaptive hoop rule with one caller-validated order shared
/// by every observer.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid parameters, patches, a
/// singular observer, or an invalid quadrature order.
pub fn stationary_fwh_ring_pressure_fixed(
    patches: &[FwhRevolvedPatch],
    observer: [f64; 3],
    angular_frequency: f64,
    density: f64,
    speed_of_sound: f64,
    meridional_order: usize,
    azimuthal_order_points: usize,
) -> Result<FwhComponents, AeroacousticsError> {
    revolved_pressure_fixed(
        patches,
        observer,
        angular_frequency,
        density,
        speed_of_sound,
        meridional_order,
        azimuthal_order_points,
    )
}

struct RevolvedContext {
    observer: [f64; 3],
    wavenumber: f64,
    omega_times_density: f64,
    meridional_nodes: Vec<f64>,
    meridional_weights: Vec<f64>,
}

fn revolved_context(
    observer: [f64; 3],
    angular_frequency: f64,
    density: f64,
    speed_of_sound: f64,
    meridional_order: usize,
) -> Result<RevolvedContext, AeroacousticsError> {
    require_finite("angular_frequency", angular_frequency)?;
    require_positive("density", density)?;
    require_positive("speed_of_sound", speed_of_sound)?;
    if observer.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField { name: "observer" });
    }
    let (nodes, weights) = gauss_legendre_rule(meridional_order)?;
    Ok(RevolvedContext {
        observer,
        wavenumber: angular_frequency / speed_of_sound,
        omega_times_density: angular_frequency * density,
        meridional_nodes: nodes,
        meridional_weights: weights,
    })
}

fn revolved_sample(
    patch: &FwhRevolvedPatch,
    strip_length: f64,
    context: &RevolvedContext,
    azimuth: f64,
    select: fn(&FwhComponents) -> Complex64,
) -> Option<Complex64> {
    let (sin, cos) = azimuth.sin_cos();
    let mut sum = Complex64::new(0.0, 0.0);
    for (node, weight) in context
        .meridional_nodes
        .iter()
        .zip(context.meridional_weights.iter())
    {
        let blend = 0.5 * (node + 1.0);
        let radius = patch.node_a[0] + (patch.node_b[0] - patch.node_a[0]) * blend;
        let axial = patch.node_a[1] + (patch.node_b[1] - patch.node_a[1]) * blend;
        let area = 0.5 * strip_length * weight * radius;
        let sample = fwh_patch_bare(
            [radius * cos, radius * sin, axial],
            [
                patch.outward_normal[0] * cos,
                patch.outward_normal[0] * sin,
                patch.outward_normal[1],
            ],
            area,
            patch.pressure,
            patch.normal_velocity,
            context.observer,
            context.wavenumber,
            context.omega_times_density,
        )?;
        sum += select(&sample);
    }
    Some(sum)
}

fn revolved_patch_components(
    patch: &FwhRevolvedPatch,
    context: &RevolvedContext,
    tolerance: f64,
) -> Result<FwhComponents, AeroacousticsError> {
    let strip_length = validate_revolved_nodes(patch.node_a, patch.node_b, patch.outward_normal)?;
    if !patch.pressure.re.is_finite()
        || !patch.pressure.im.is_finite()
        || !patch.normal_velocity.re.is_finite()
        || !patch.normal_velocity.im.is_finite()
    {
        return Err(AeroacousticsError::NonFiniteField {
            name: "revolved patch data",
        });
    }
    let order = patch.azimuthal.order();
    let thickness = azimuthal_ring_integral_adaptive(order, tolerance, |azimuth| {
        revolved_sample(patch, strip_length, context, azimuth, |sample| {
            sample.thickness
        })
        .unwrap_or(Complex64::new(f64::NAN, 0.0))
    })?;
    let loading = azimuthal_ring_integral_adaptive(order, tolerance, |azimuth| {
        revolved_sample(patch, strip_length, context, azimuth, |sample| {
            sample.loading
        })
        .unwrap_or(Complex64::new(f64::NAN, 0.0))
    })?;
    if !thickness.value.re.is_finite()
        || !thickness.value.im.is_finite()
        || !loading.value.re.is_finite()
        || !loading.value.im.is_finite()
    {
        return Err(AeroacousticsError::SingularObserver);
    }
    Ok(FwhComponents {
        thickness: thickness.value,
        loading: loading.value,
        total: thickness.value + loading.value,
    })
}

fn revolved_pressure(
    patches: &[FwhRevolvedPatch],
    observer: [f64; 3],
    angular_frequency: f64,
    density: f64,
    speed_of_sound: f64,
    quadrature: RevolvedQuadrature,
) -> Result<FwhComponents, AeroacousticsError> {
    require_positive("ring tolerance", quadrature.tolerance)?;
    let context = revolved_context(
        observer,
        angular_frequency,
        density,
        speed_of_sound,
        quadrature.meridional_order,
    )?;
    let mut total = FwhComponents::default();
    for patch in patches {
        let sample = revolved_patch_components(patch, &context, quadrature.tolerance)?;
        total.thickness += sample.thickness;
        total.loading += sample.loading;
        total.total += sample.total;
    }
    Ok(total)
}

fn revolved_pressure_fixed(
    patches: &[FwhRevolvedPatch],
    observer: [f64; 3],
    angular_frequency: f64,
    density: f64,
    speed_of_sound: f64,
    meridional_order: usize,
    azimuthal_points: usize,
) -> Result<FwhComponents, AeroacousticsError> {
    let context = revolved_context(
        observer,
        angular_frequency,
        density,
        speed_of_sound,
        meridional_order,
    )?;
    let mut total = FwhComponents::default();
    for patch in patches {
        let strip_length =
            validate_revolved_nodes(patch.node_a, patch.node_b, patch.outward_normal)?;
        let order = patch.azimuthal.order();
        let thickness = azimuthal_ring_integral(order, azimuthal_points, |azimuth| {
            revolved_sample(patch, strip_length, &context, azimuth, |sample| {
                sample.thickness
            })
            .unwrap_or(Complex64::new(f64::NAN, 0.0))
        })?;
        let loading = azimuthal_ring_integral(order, azimuthal_points, |azimuth| {
            revolved_sample(patch, strip_length, &context, azimuth, |sample| {
                sample.loading
            })
            .unwrap_or(Complex64::new(f64::NAN, 0.0))
        })?;
        if !thickness.re.is_finite()
            || !thickness.im.is_finite()
            || !loading.re.is_finite()
            || !loading.im.is_finite()
        {
            return Err(AeroacousticsError::SingularObserver);
        }
        total.thickness += thickness;
        total.loading += loading;
        total.total += thickness + loading;
    }
    Ok(total)
}

fn validate_mach_vector(mach: &[f64]) -> Result<(), AeroacousticsError> {
    if mach.iter().any(|component| !component.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField { name: "mach" });
    }
    let magnitude = mach
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    if magnitude >= 1.0 {
        return Err(AeroacousticsError::InvalidMach { magnitude });
    }
    Ok(())
}

/// One quadrature patch of a convected Kirchhoff surface.
///
/// Uniform mean flow breaks the `(p, u_n)` closure: the convected
/// representation needs the pressure gradient, so patches carry
/// `(p, ∇p)`. The gradient is available from any volume flow solution
/// restricted to the surface.
#[derive(Debug, Clone, Copy)]
pub struct FwhKirchhoffPatch {
    /// Patch centroid `[m]`.
    pub position: [f64; 3],
    /// Unit normal pointing to the observer region.
    pub outward_normal: [f64; 3],
    /// Patch quadrature area `[m²]`.
    pub area: f64,
    /// Complex acoustic pressure `[Pa]`.
    pub pressure: Complex64,
    /// Complex acoustic pressure gradient `[Pa/m]`.
    pub pressure_gradient: [Complex64; 3],
}

impl FwhKirchhoffPatch {
    /// Builds and validates one convected-surface quadrature patch.
    pub fn new(
        position: [f64; 3],
        outward_normal: [f64; 3],
        area: f64,
        pressure: Complex64,
        pressure_gradient: [Complex64; 3],
    ) -> Result<Self, AeroacousticsError> {
        FwhSurfacePatch::new(
            position,
            outward_normal,
            area,
            pressure,
            Complex64::new(0.0, 0.0),
        )?;
        if pressure_gradient
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(AeroacousticsError::NonFiniteField {
                name: "pressure_gradient",
            });
        }
        Ok(Self {
            position,
            outward_normal,
            area,
            pressure,
            pressure_gradient,
        })
    }
}

/// One straight segment of a convected planar Kirchhoff contour.
#[derive(Debug, Clone, Copy)]
pub struct FwhKirchhoffSegment {
    /// Segment midpoint `[m]` in the plane.
    pub position: [f64; 2],
    /// Unit normal pointing to the observer region.
    pub outward_normal: [f64; 2],
    /// Segment quadrature length `[m]`.
    pub length: f64,
    /// Complex acoustic pressure `[Pa]`.
    pub pressure: Complex64,
    /// Complex acoustic pressure gradient `[Pa/m]`.
    pub pressure_gradient: [Complex64; 2],
}

impl FwhKirchhoffSegment {
    /// Builds and validates one convected-contour quadrature segment.
    pub fn new(
        position: [f64; 2],
        outward_normal: [f64; 2],
        length: f64,
        pressure: Complex64,
        pressure_gradient: [Complex64; 2],
    ) -> Result<Self, AeroacousticsError> {
        FwhContourSegment::new(
            position,
            outward_normal,
            length,
            pressure,
            Complex64::new(0.0, 0.0),
        )?;
        if pressure_gradient
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(AeroacousticsError::NonFiniteField {
                name: "pressure_gradient",
            });
        }
        Ok(Self {
            position,
            outward_normal,
            length,
            pressure,
            pressure_gradient,
        })
    }
}

/// Evaluates a stationary Kirchhoff surface in uniform mean flow.
///
/// Green's-identity representation with the adjoint (flow-reversed)
/// Garrick Green function: each patch contributes `p∂_n G* − G*∂_n p +
/// (U·n)c⁻²[2iωG*p + G*(U·∇)p − p(U·∇G*)]`. The `total` field carries the
/// result; `thickness`/`loading` report the standard Kirchhoff split
/// (`−G*∂_n p`, and `p∂_n G*` plus the convective share). Recovers the
/// quiescent Kirchhoff integral at zero Mach.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid parameters, patches, a
/// non-subsonic Mach vector, or a coincident observer.
pub fn stationary_kirchhoff_pressure_convected(
    patches: &[FwhKirchhoffPatch],
    observer: [f64; 3],
    angular_frequency: f64,
    speed_of_sound: f64,
    mach: [f64; 3],
) -> Result<FwhComponents, AeroacousticsError> {
    require_finite("angular_frequency", angular_frequency)?;
    require_positive("speed_of_sound", speed_of_sound)?;
    validate_mach_vector(&mach)?;
    if observer.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField { name: "observer" });
    }
    let wavenumber = angular_frequency / speed_of_sound;
    let neg_mach = mach.map(|component| -component);
    let mean_velocity = [
        mach[0] * speed_of_sound,
        mach[1] * speed_of_sound,
        mach[2] * speed_of_sound,
    ];
    let observer_point = Point::new_3d(observer[0], observer[1], observer[2]);
    let mut thickness = Complex64::new(0.0, 0.0);
    let mut loading = Complex64::new(0.0, 0.0);
    for patch in patches {
        FwhKirchhoffPatch::new(
            patch.position,
            patch.outward_normal,
            patch.area,
            patch.pressure,
            patch.pressure_gradient,
        )?;
        let surface_point = Point::new_3d(patch.position[0], patch.position[1], patch.position[2]);
        let adjoint = flip_time_convention(
            try_convected_greens_function_3d(
                &observer_point,
                &surface_point,
                wavenumber,
                &neg_mach,
            )
            .map_err(map_kernel_error)?,
        );
        let adjoint_gradient = try_convected_greens_function_gradient_3d(
            &observer_point,
            &surface_point,
            wavenumber,
            &neg_mach,
        )
        .map_err(map_kernel_error)?
        .map(flip_time_convention);
        let normal_gradient: Complex64 = patch
            .pressure_gradient
            .iter()
            .zip(patch.outward_normal.iter())
            .map(|(gradient, normal)| gradient * normal)
            .sum();
        let adjoint_normal: Complex64 = adjoint_gradient
            .iter()
            .zip(patch.outward_normal.iter())
            .map(|(gradient, normal)| gradient * normal)
            .sum();
        let advection_pressure: Complex64 = patch
            .pressure_gradient
            .iter()
            .zip(mean_velocity.iter())
            .map(|(gradient, velocity)| gradient * velocity)
            .sum();
        let advection_adjoint: Complex64 = adjoint_gradient
            .iter()
            .zip(mean_velocity.iter())
            .map(|(gradient, velocity)| gradient * velocity)
            .sum();
        let mean_normal_flux = mean_velocity[0] * patch.outward_normal[0]
            + mean_velocity[1] * patch.outward_normal[1]
            + mean_velocity[2] * patch.outward_normal[2];
        thickness += -adjoint * normal_gradient * patch.area;
        loading += (patch.pressure * adjoint_normal
            + mean_normal_flux / speed_of_sound.powi(2)
                * (Complex64::new(0.0, 2.0 * angular_frequency) * adjoint * patch.pressure
                    + adjoint * advection_pressure
                    - patch.pressure * advection_adjoint))
            * patch.area;
    }
    Ok(FwhComponents {
        thickness,
        loading,
        total: thickness + loading,
    })
}

/// Evaluates a planar Kirchhoff contour in uniform mean flow.
///
/// The 2-D counterpart of [`stationary_kirchhoff_pressure_convected`]:
/// the integration-by-parts concomitant is dimension-independent, so the
/// same adjoint-based surface integrand applies with Hankel kernels.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid parameters, segments, a
/// non-subsonic Mach vector, or a coincident observer.
pub fn stationary_kirchhoff_pressure_2d_convected(
    segments: &[FwhKirchhoffSegment],
    observer: [f64; 2],
    angular_frequency: f64,
    speed_of_sound: f64,
    mach: [f64; 2],
) -> Result<FwhComponents, AeroacousticsError> {
    require_finite("angular_frequency", angular_frequency)?;
    require_positive("speed_of_sound", speed_of_sound)?;
    validate_mach_vector(&mach)?;
    if observer.iter().any(|x| !x.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField { name: "observer" });
    }
    let wavenumber = angular_frequency / speed_of_sound;
    let neg_mach = mach.map(|component| -component);
    let mean_velocity = [mach[0] * speed_of_sound, mach[1] * speed_of_sound];
    let mut thickness = Complex64::new(0.0, 0.0);
    let mut loading = Complex64::new(0.0, 0.0);
    for segment in segments {
        FwhKirchhoffSegment::new(
            segment.position,
            segment.outward_normal,
            segment.length,
            segment.pressure,
            segment.pressure_gradient,
        )?;
        let adjoint = flip_time_convention(
            try_convected_greens_function_2d(&observer, &segment.position, wavenumber, &neg_mach)
                .map_err(map_kernel_error)?,
        );
        let adjoint_gradient = convected_greens_function_gradient_2d(
            &observer,
            &segment.position,
            wavenumber,
            &neg_mach,
        )
        .map(flip_time_convention);
        if adjoint_gradient
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(AeroacousticsError::NonFiniteField {
                name: "convected gradient",
            });
        }
        let normal_gradient = segment.pressure_gradient[0] * segment.outward_normal[0]
            + segment.pressure_gradient[1] * segment.outward_normal[1];
        let adjoint_normal = adjoint_gradient[0] * segment.outward_normal[0]
            + adjoint_gradient[1] * segment.outward_normal[1];
        let advection_pressure = segment.pressure_gradient[0] * mean_velocity[0]
            + segment.pressure_gradient[1] * mean_velocity[1];
        let advection_adjoint =
            adjoint_gradient[0] * mean_velocity[0] + adjoint_gradient[1] * mean_velocity[1];
        let mean_normal_flux = mean_velocity[0] * segment.outward_normal[0]
            + mean_velocity[1] * segment.outward_normal[1];
        thickness += -adjoint * normal_gradient * segment.length;
        loading += (segment.pressure * adjoint_normal
            + mean_normal_flux / speed_of_sound.powi(2)
                * (Complex64::new(0.0, 2.0 * angular_frequency) * adjoint * segment.pressure
                    + adjoint * advection_pressure
                    - segment.pressure * advection_adjoint))
            * segment.length;
    }
    Ok(FwhComponents {
        thickness,
        loading,
        total: thickness + loading,
    })
}

#[cfg(test)]
mod tests {
    use super::super::fwh::stationary_fwh_pressure;
    use super::*;
    use sonium_wave::special::helmholtz::greens_function_2d;
    use std::f64::consts::PI;

    fn monopole_2d(
        radius: f64,
        segments: usize,
        omega: f64,
        rho: f64,
        c: f64,
    ) -> Vec<FwhContourSegment> {
        let k = omega / c;
        let length = 2.0 * PI * radius / segments as f64;
        (0..segments)
            .map(|index| {
                let phi = 2.0 * PI * (index as f64 + 0.5) / segments as f64;
                let (sin, cos) = phi.sin_cos();
                let position = [radius * cos, radius * sin];
                // Unit outgoing cylindrical wave with aero-convention phase.
                let pressure = flip_time_convention(greens_function_2d(radius, k)) * (4.0 * omega);
                let gradient = try_greens_function_gradient_2d(&[0.0, 0.0], &position, k).unwrap();
                let radial_gradient = flip_time_convention(gradient[0] * cos + gradient[1] * sin);
                // Momentum: iωρu + ∇p = 0.
                let normal_velocity =
                    radial_gradient * Complex64::new(0.0, 1.0) / (omega * rho) * (4.0 * omega);
                FwhContourSegment::new(position, [cos, sin], length, pressure, normal_velocity)
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn closed_contour_reconstructs_planar_monopole() {
        let (omega, rho, c) = (4.0, 1.2, 3.0);
        let contour = monopole_2d(0.5, 256, omega, rho, c);
        let observer = [2.0, 0.3];
        let actual = stationary_fwh_pressure_2d(&contour, observer, omega, rho, c)
            .unwrap()
            .total;
        let expected = flip_time_convention(greens_function_2d(
            observer[0].hypot(observer[1]),
            omega / c,
        )) * (4.0 * omega);
        assert!((actual - expected).norm() / expected.norm() < 5e-3);
    }

    fn revolved_sphere(
        radius: f64,
        strips: usize,
        omega: f64,
        rho: f64,
        c: f64,
    ) -> Vec<FwhRevolvedPatch> {
        let k = omega / c;
        (0..strips)
            .map(|index| {
                let polar_a = PI * index as f64 / strips as f64;
                let polar_b = PI * (index as f64 + 1.0) / strips as f64;
                let node_a = [radius * polar_a.sin(), radius * polar_a.cos()];
                let node_b = [radius * polar_b.sin(), radius * polar_b.cos()];
                let mid = 0.5 * (polar_a + polar_b);
                let outward_normal = [mid.sin(), mid.cos()];
                let green = Complex64::from_polar(1.0 / (4.0 * PI * radius), -k * radius);
                let normal_gradient = green * Complex64::new(-1.0 / radius, -k);
                let normal_velocity = normal_gradient * Complex64::new(0.0, 1.0) / (omega * rho);
                FwhRevolvedPatch::new(
                    node_a,
                    node_b,
                    outward_normal,
                    AzimuthalMode::axisymmetric(),
                    green,
                    normal_velocity,
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn revolved_sphere_reconstructs_monopole() {
        let (omega, rho, c) = (2.0, 1.2, 3.0);
        let surface = revolved_sphere(0.5, 40, omega, rho, c);
        let quadrature = RevolvedQuadrature::new(1e-9, 4).unwrap();
        let observer = [2.0, 0.3, -0.1];
        let actual =
            stationary_fwh_ring_pressure(&surface, observer, omega, rho, c, quadrature).unwrap();
        let distance =
            (observer[0] * observer[0] + observer[1] * observer[1] + observer[2] * observer[2])
                .sqrt();
        let expected = Complex64::from_polar(1.0 / (4.0 * PI * distance), -omega / c * distance);
        assert!(
            (actual.total - expected).norm() / expected.norm() < 2e-2,
            "relative error {}",
            (actual.total - expected).norm() / expected.norm()
        );
        assert!((actual.total - actual.thickness - actual.loading).norm() < 1e-12);
    }

    #[test]
    fn spinning_ring_matches_discrete_patch_summation() {
        let patch = FwhRevolvedPatch::new(
            [0.5, -0.1],
            [0.5, 0.1],
            [1.0, 0.0],
            AzimuthalMode::new(1),
            Complex64::new(1.0, -0.5),
            Complex64::new(0.2, 0.1),
        )
        .unwrap();
        let quadrature = RevolvedQuadrature::new(1e-10, 4).unwrap();
        let observer = [1.3, -0.4, 0.9];
        let ring = stationary_fwh_ring_pressure(
            std::slice::from_ref(&patch),
            observer,
            8.0,
            1.2,
            1.0,
            quadrature,
        )
        .unwrap()
        .total;
        // Independent check: midpoint-rule discrete spinning patches. The
        // azimuthal direction is periodic (spectral midpoint accuracy);
        // the axial direction uses 16 midpoint subdivisions.
        let samples = 720usize;
        let axial_cuts = 16usize;
        let mut patches = Vec::with_capacity(samples * axial_cuts);
        for sample in 0..samples {
            let phi = 2.0 * PI * (sample as f64 + 0.5) / samples as f64;
            let phase = Complex64::from_polar(1.0, phi);
            for cut in 0..axial_cuts {
                let axial = -0.1 + 0.2 * (cut as f64 + 0.5) / axial_cuts as f64;
                patches.push(
                    FwhSurfacePatch::new(
                        [0.5 * phi.cos(), 0.5 * phi.sin(), axial],
                        [phi.cos(), phi.sin(), 0.0],
                        0.5 * (2.0 * PI / samples as f64) * (0.2 / axial_cuts as f64),
                        patch.pressure * phase,
                        patch.normal_velocity * phase,
                    )
                    .unwrap(),
                );
            }
        }
        let discrete = stationary_fwh_pressure(&patches, observer, 8.0, 1.2, 1.0)
            .unwrap()
            .total;
        assert!(
            (ring - discrete).norm() / discrete.norm() < 1e-3,
            "relative mismatch {}",
            (ring - discrete).norm() / discrete.norm()
        );
    }

    /// Exact outgoing `(p, u)` data from a velocity potential.
    ///
    /// For any potential satisfying the convected equation, `u = ∇φ` and
    /// `p = −ρ(iω + U·∇)φ` satisfy the linearized equations exactly and
    /// radiate outward when `φ` is an interior convected Green function.
    /// Unlike plane waves, this data carries a vanishing sphere at
    /// infinity, so the FW-H integral must reproduce it end to end.
    fn potential_pressure_3d(
        position: [f64; 3],
        omega: f64,
        rho: f64,
        c: f64,
        mach: [f64; 3],
    ) -> Complex64 {
        let k = omega / c;
        let origin = Point::new_3d(0.0, 0.0, 0.0);
        let point = Point::new_3d(position[0], position[1], position[2]);
        let potential = flip_time_convention(
            try_convected_greens_function_3d(&origin, &point, k, &mach).unwrap(),
        );
        let velocity = try_convected_greens_function_gradient_3d(&origin, &point, k, &mach)
            .unwrap()
            .map(flip_time_convention);
        let advection =
            velocity[0] * mach[0] * c + velocity[1] * mach[1] * c + velocity[2] * mach[2] * c;
        -(Complex64::new(0.0, omega * rho) * potential + rho * advection)
    }

    fn potential_gradient_3d(
        position: [f64; 3],
        omega: f64,
        rho: f64,
        c: f64,
        mach: [f64; 3],
    ) -> [Complex64; 3] {
        // Central differences of the analytic potential pressure; the
        // h² error sits six orders below quadrature error.
        let h = 1e-5;
        std::array::from_fn(|axis| {
            let mut plus = position;
            let mut minus = position;
            plus[axis] += h;
            minus[axis] -= h;
            (potential_pressure_3d(plus, omega, rho, c, mach)
                - potential_pressure_3d(minus, omega, rho, c, mach))
                / (2.0 * h)
        })
    }

    fn potential_surface(
        radius: f64,
        n_mu: usize,
        n_phi: usize,
        omega: f64,
        rho: f64,
        c: f64,
        mach: [f64; 3],
    ) -> Vec<FwhKirchhoffPatch> {
        let area = 4.0 * PI * radius * radius / (n_mu * n_phi) as f64;
        let mut patches = Vec::with_capacity(n_mu * n_phi);
        for m in 0..n_mu {
            let mu = -1.0 + 2.0 * (m as f64 + 0.5) / n_mu as f64;
            let radial = (1.0 - mu * mu).sqrt();
            for p in 0..n_phi {
                let phi = 2.0 * PI * (p as f64 + 0.5) / n_phi as f64;
                let normal = [radial * phi.cos(), radial * phi.sin(), mu];
                let position = normal.map(|x| radius * x);
                let pressure = potential_pressure_3d(position, omega, rho, c, mach);
                let gradient = potential_gradient_3d(position, omega, rho, c, mach);
                patches.push(
                    FwhKirchhoffPatch::new(position, normal, area, pressure, gradient).unwrap(),
                );
            }
        }
        patches
    }

    #[test]
    fn kirchhoff_recovers_quiescent_monopole_at_zero_mach() {
        let (omega, rho, c) = (2.0, 1.2, 3.0);
        let k = omega / c;
        let radius = 0.5;
        let (n_mu, n_phi) = (48, 96);
        let area = 4.0 * PI * radius * radius / (n_mu * n_phi) as f64;
        let mut patches = Vec::with_capacity(n_mu * n_phi);
        for m in 0..n_mu {
            let mu = -1.0 + 2.0 * (m as f64 + 0.5) / n_mu as f64;
            let radial = (1.0 - mu * mu).sqrt();
            for p in 0..n_phi {
                let phi = 2.0 * PI * (p as f64 + 0.5) / n_phi as f64;
                let normal = [radial * phi.cos(), radial * phi.sin(), mu];
                let position = normal.map(|x| radius * x);
                let pressure = Complex64::from_polar(1.0 / (4.0 * PI * radius), -k * radius);
                let radial_derivative = pressure * Complex64::new(-1.0 / radius, -k);
                patches.push(
                    FwhKirchhoffPatch::new(
                        position,
                        normal,
                        area,
                        pressure,
                        normal.map(|component| radial_derivative * component),
                    )
                    .unwrap(),
                );
            }
        }
        let observer = [2.0, 0.3, -0.1];
        let actual =
            stationary_kirchhoff_pressure_convected(&patches, observer, omega, c, [0.0, 0.0, 0.0])
                .unwrap()
                .total;
        let distance =
            (observer[0] * observer[0] + observer[1] * observer[1] + observer[2] * observer[2])
                .sqrt();
        let expected = Complex64::from_polar(1.0 / (4.0 * PI * distance), -k * distance);
        assert!((actual - expected).norm() / expected.norm() < 2e-3);
        let _ = rho;
    }

    #[test]
    fn kirchhoff_reproduces_outgoing_potential_field() {
        let (omega, rho, c) = (6.0, 1.2, 3.0);
        let mach = [0.25, 0.1, -0.15];
        let surface = potential_surface(0.5, 64, 128, omega, rho, c, mach);
        let observer = [2.0, 0.3, -0.4];
        let actual = stationary_kirchhoff_pressure_convected(&surface, observer, omega, c, mach)
            .unwrap()
            .total;
        let expected = potential_pressure_3d(observer, omega, rho, c, mach);
        assert!(
            (actual - expected).norm() / expected.norm() < 5e-3,
            "relative error {}",
            (actual - expected).norm() / expected.norm()
        );
    }

    fn potential_pressure_2d(
        position: [f64; 2],
        omega: f64,
        rho: f64,
        c: f64,
        mach: [f64; 2],
    ) -> Complex64 {
        let k = omega / c;
        let origin = [0.0, 0.0];
        let potential = flip_time_convention(
            try_convected_greens_function_2d(&origin, &position, k, &mach).unwrap(),
        );
        let velocity = convected_greens_function_gradient_2d(&origin, &position, k, &mach)
            .map(flip_time_convention);
        let advection = velocity[0] * mach[0] * c + velocity[1] * mach[1] * c;
        -(Complex64::new(0.0, omega * rho) * potential + rho * advection)
    }

    fn potential_gradient_2d(
        position: [f64; 2],
        omega: f64,
        rho: f64,
        c: f64,
        mach: [f64; 2],
    ) -> [Complex64; 2] {
        let h = 1e-5;
        std::array::from_fn(|axis| {
            let mut plus = position;
            let mut minus = position;
            plus[axis] += h;
            minus[axis] -= h;
            (potential_pressure_2d(plus, omega, rho, c, mach)
                - potential_pressure_2d(minus, omega, rho, c, mach))
                / (2.0 * h)
        })
    }

    #[test]
    fn kirchhoff_planar_reproduces_outgoing_potential_field() {
        let (omega, rho, c) = (5.0, 1.2, 3.0);
        let mach = [0.2, -0.15];
        let segments = 512usize;
        let radius = 0.5;
        let contour: Vec<FwhKirchhoffSegment> = (0..segments)
            .map(|index| {
                let phi = 2.0 * PI * (index as f64 + 0.5) / segments as f64;
                let (sin, cos) = phi.sin_cos();
                let position = [radius * cos, radius * sin];
                let pressure = potential_pressure_2d(position, omega, rho, c, mach);
                let gradient = potential_gradient_2d(position, omega, rho, c, mach);
                FwhKirchhoffSegment::new(
                    position,
                    [cos, sin],
                    2.0 * PI * radius / segments as f64,
                    pressure,
                    gradient,
                )
                .unwrap()
            })
            .collect();
        let observer = [1.8, -0.5];
        let actual = stationary_kirchhoff_pressure_2d_convected(&contour, observer, omega, c, mach)
            .unwrap()
            .total;
        let expected = potential_pressure_2d(observer, omega, rho, c, mach);
        assert!(
            (actual - expected).norm() / expected.norm() < 5e-3,
            "relative error {}",
            (actual - expected).norm() / expected.norm()
        );
    }

    #[test]
    fn kirchhoff_recovers_quiescent_at_zero_mach() {
        // Quiescent momentum maps normal velocity to pressure gradient:
        // ∇p = −iωρu_n n.
        let (omega, rho, c) = (4.0, 1.2, 3.0);
        let contour = monopole_2d(0.5, 64, omega, rho, c);
        let still = stationary_fwh_pressure_2d(&contour, [2.0, 0.3], omega, rho, c).unwrap();
        let kirchhoff_contour: Vec<FwhKirchhoffSegment> = contour
            .iter()
            .map(|segment| {
                let factor = Complex64::new(0.0, -omega * rho) * segment.normal_velocity;
                FwhKirchhoffSegment::new(
                    segment.position,
                    segment.outward_normal,
                    segment.length,
                    segment.pressure,
                    segment.outward_normal.map(|component| factor * component),
                )
                .unwrap()
            })
            .collect();
        let kirchhoff = stationary_kirchhoff_pressure_2d_convected(
            &kirchhoff_contour,
            [2.0, 0.3],
            omega,
            c,
            [0.0, 0.0],
        )
        .unwrap();
        assert!((still.total - kirchhoff.total).norm() / still.total.norm() < 1e-12);

        let patch = FwhSurfacePatch::new(
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.1,
            Complex64::new(1.0, 0.5),
            Complex64::new(0.2, -0.1),
        )
        .unwrap();
        let still_3d =
            stationary_fwh_pressure(std::slice::from_ref(&patch), [2.0, 0.0, 0.0], omega, rho, c)
                .unwrap();
        let factor = Complex64::new(0.0, -omega * rho) * patch.normal_velocity;
        let kirchhoff_patch = FwhKirchhoffPatch::new(
            patch.position,
            patch.outward_normal,
            patch.area,
            patch.pressure,
            patch.outward_normal.map(|component| factor * component),
        )
        .unwrap();
        let kirchhoff_3d = stationary_kirchhoff_pressure_convected(
            std::slice::from_ref(&kirchhoff_patch),
            [2.0, 0.0, 0.0],
            omega,
            c,
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        assert!((still_3d.total - kirchhoff_3d.total).norm() / still_3d.total.norm() < 1e-12);
    }

    #[test]
    fn kirchhoff_rejects_sonic_mach() {
        let patch = FwhKirchhoffPatch::new(
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.1,
            Complex64::new(1.0, 0.0),
            [Complex64::new(0.0, 0.0); 3],
        )
        .unwrap();
        assert!(matches!(
            stationary_kirchhoff_pressure_convected(
                std::slice::from_ref(&patch),
                [2.0, 0.0, 0.0],
                4.0,
                3.0,
                [1.0, 0.0, 0.0]
            ),
            Err(AeroacousticsError::InvalidMach { .. })
        ));
    }

    #[test]
    fn revolved_patches_reject_degenerate_geometry() {
        assert!(
            FwhRevolvedPatch::new(
                [0.5, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                AzimuthalMode::axisymmetric(),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            )
            .is_err()
        );
        assert!(
            FwhRevolvedPatch::new(
                [-0.1, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                AzimuthalMode::axisymmetric(),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            )
            .is_err()
        );
    }
}
