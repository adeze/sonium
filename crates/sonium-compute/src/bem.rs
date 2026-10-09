//! Shared BEM GPU kernel ABI.
//!
//! These POD structs are consumed directly by Metal and CUDA kernels, so their
//! layout is part of the cross-backend contract.

use num_complex::{Complex32, Complex64};

use crate::scalar::BufferElement;

/// Per-pair input descriptor for regular-pair Helmholtz GPU kernels.
///
/// Binary layout must match `SoniumBemRegularPairIn` in the Metal and CUDA
/// kernels exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BemRegularPairIn {
    /// Source-element collocation point, typically the source centroid.
    pub source_point: [f32; 3],
    _pad0: f32,
    /// Source-element unit normal at the collocation point.
    pub source_normal: [f32; 3],
    _pad1: f32,
    /// Field-element vertex 0 in global coordinates.
    pub v0: [f32; 3],
    _pad2: f32,
    /// Field-element vertex 1.
    pub v1: [f32; 3],
    _pad3: f32,
    /// Field-element vertex 2.
    pub v2: [f32; 3],
    _pad4: f32,
}

impl BemRegularPairIn {
    /// Construct a pair descriptor from a source point + normal and the three
    /// field-element vertices, populating padding bytes safely.
    pub fn from_triangle(
        source_point: [f32; 3],
        source_normal: [f32; 3],
        v0: [f32; 3],
        v1: [f32; 3],
        v2: [f32; 3],
    ) -> Self {
        Self {
            source_point,
            _pad0: 0.0,
            source_normal,
            _pad1: 0.0,
            v0,
            _pad2: 0.0,
            v1,
            _pad3: 0.0,
            v2,
            _pad4: 0.0,
        }
    }
}

/// Per-pair regular Helmholtz output:
/// `(G, H, H^T, E)` over the field triangle in single precision.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BemRegularPairOut {
    /// G = ∫ exp(ikr)/(4πr) dS_y.
    pub g: Complex32,
    /// H = ∫ ∂G/∂n_y dS_y.
    pub h: Complex32,
    /// H^T = ∫ ∂G/∂n_x dS_y.
    pub ht: Complex32,
    /// E = ∫ ∂²G/∂n_x∂n_y dS_y.
    pub e: Complex32,
}

/// Per-pair input descriptor for the checked double-precision BEM kernel.
///
/// This is deliberately a separate ABI from [`BemRegularPairIn`].  CUDA and
/// Metal can support different scalar families, and keeping the element
/// width in the type prevents a caller from accidentally advertising an f32
/// computation as a Complex64 reference.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BemRegularPairInF64 {
    /// Source-element collocation point.
    pub source_point: [f64; 3],
    _pad0: f64,
    /// Source-element unit normal.
    pub source_normal: [f64; 3],
    _pad1: f64,
    /// Field-element vertices.
    pub v0: [f64; 3],
    _pad2: f64,
    pub v1: [f64; 3],
    _pad3: f64,
    pub v2: [f64; 3],
    _pad4: f64,
}

impl BemRegularPairInF64 {
    /// Construct a padded double-precision triangle descriptor.
    #[must_use]
    pub fn from_triangle(
        source_point: [f64; 3],
        source_normal: [f64; 3],
        v0: [f64; 3],
        v1: [f64; 3],
        v2: [f64; 3],
    ) -> Self {
        Self {
            source_point,
            _pad0: 0.0,
            source_normal,
            _pad1: 0.0,
            v0,
            _pad2: 0.0,
            v1,
            _pad3: 0.0,
            v2,
            _pad4: 0.0,
        }
    }
}

/// A constant-basis triangular boundary element for the CUDA field kernel.
///
/// The pressure and normal-velocity traces are already reduced to one
/// Complex64 value per element.  Keeping them beside the geometry makes the
/// repeated field-evaluation path independent of host-side surface data.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BemFieldElementC64 {
    /// Triangle vertex 0 in global coordinates.
    pub v0: [f64; 3],
    /// Triangle vertex 1 in global coordinates.
    pub v1: [f64; 3],
    /// Triangle vertex 2 in global coordinates.
    pub v2: [f64; 3],
    /// Constant pressure trace on the element.
    pub pressure: Complex64,
    /// Constant outward normal velocity trace on the element.
    pub velocity: Complex64,
}

impl BemFieldElementC64 {
    /// Construct one constant-basis triangular field element.
    #[must_use]
    pub const fn from_triangle(
        v0: [f64; 3],
        v1: [f64; 3],
        v2: [f64; 3],
        pressure: Complex64,
        velocity: Complex64,
    ) -> Self {
        Self {
            v0,
            v1,
            v2,
            pressure,
            velocity,
        }
    }
}

/// One exterior field evaluation point for the CUDA field kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BemFieldPointC64 {
    /// Evaluation point in global coordinates.
    pub position: [f64; 3],
}

impl BemFieldPointC64 {
    /// Construct one field evaluation point.
    #[must_use]
    pub const fn new(position: [f64; 3]) -> Self {
        Self { position }
    }
}

/// One precomputed quadrature sample for a device-resident Complex64 BEM
/// pair.  The sample contains geometry and a scalar basis weight; the CUDA
/// kernel performs all Green-function and normal-derivative arithmetic.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BemQuadratureSampleC64 {
    /// Physical integration point on the field element.
    pub position: [f64; 3],
    /// Unit normal of the field element at `position`.
    pub field_normal: [f64; 3],
    /// Tangent used by the regularised edge hypersingular contribution.
    pub edge_direction: [f64; 3],
    /// Geometric quadrature weight, including surface/Jacobian factors.
    pub weight: f64,
    /// Field-basis value at the quadrature point.
    pub basis_weight: f64,
    /// Kernel path selected for this sample.
    pub kind: u32,
    _pad: u32,
}

impl BemQuadratureSampleC64 {
    /// Ordinary sample evaluating all four `(G, H, Hᵀ, E)` channels.
    pub const KIND_FULL: u32 = 0;
    /// Regularised bulk sample evaluating `G`, `H`, `Hᵀ`, and the reduced
    /// hypersingular bulk term used by the CPU Duffy path.
    pub const KIND_SINGULAR_BULK: u32 = 1;
    /// Edge sample evaluating only the regularised hypersingular edge term.
    pub const KIND_HYPERSINGULAR_EDGE: u32 = 2;
    /// Sample evaluating `G`, `H`, and `Hᵀ` while omitting `E`.
    pub const KIND_GH_ONLY: u32 = 3;

    /// Construct a full ordinary quadrature sample.
    #[must_use]
    pub const fn full(
        position: [f64; 3],
        field_normal: [f64; 3],
        weight: f64,
        basis_weight: f64,
    ) -> Self {
        Self {
            position,
            field_normal,
            edge_direction: [0.0; 3],
            weight,
            basis_weight,
            kind: Self::KIND_FULL,
            _pad: 0,
        }
    }

    /// Construct a regularised bulk sample.
    #[must_use]
    pub const fn singular_bulk(
        position: [f64; 3],
        field_normal: [f64; 3],
        weight: f64,
        basis_weight: f64,
    ) -> Self {
        Self {
            position,
            field_normal,
            edge_direction: [0.0; 3],
            weight,
            basis_weight,
            kind: Self::KIND_SINGULAR_BULK,
            _pad: 0,
        }
    }

    /// Construct a regularised hypersingular edge sample.
    #[must_use]
    pub const fn hypersingular_edge(
        position: [f64; 3],
        edge_direction: [f64; 3],
        weight: f64,
        basis_weight: f64,
    ) -> Self {
        Self {
            position,
            field_normal: [0.0; 3],
            edge_direction,
            weight,
            basis_weight,
            kind: Self::KIND_HYPERSINGULAR_EDGE,
            _pad: 0,
        }
    }

    /// Construct a sample for the weakly singular G/H channels only.
    #[must_use]
    pub const fn gh_only(
        position: [f64; 3],
        field_normal: [f64; 3],
        weight: f64,
        basis_weight: f64,
    ) -> Self {
        Self {
            position,
            field_normal,
            edge_direction: [0.0; 3],
            weight,
            basis_weight,
            kind: Self::KIND_GH_ONLY,
            _pad: 0,
        }
    }
}

/// A range of precomputed quadrature samples belonging to one collocation
/// point and source normal.  `sample_offset..sample_offset+sample_count` is
/// validated before upload and consumed entirely on the device.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BemQuadraturePairC64 {
    /// Collocation/source point.
    pub source_point: [f64; 3],
    /// Unit source normal.
    pub source_normal: [f64; 3],
    /// First sample in the flattened sample buffer.
    pub sample_offset: u64,
    /// Number of samples in this pair.
    pub sample_count: u64,
}

impl BemQuadraturePairC64 {
    /// Construct one pair descriptor over a flattened sample buffer.
    #[must_use]
    pub const fn new(
        source_point: [f64; 3],
        source_normal: [f64; 3],
        sample_offset: u64,
        sample_count: u64,
    ) -> Self {
        Self {
            source_point,
            source_normal,
            sample_offset,
            sample_count,
        }
    }
}

/// Double-precision regular-pair output `(G, H, Hᵀ, E)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BemRegularPairOutF64 {
    /// `G = ∫ exp(ikr)/(4πr) dS_y`.
    pub g: Complex64,
    /// `H = ∫ ∂G/∂n_y dS_y`.
    pub h: Complex64,
    /// `Hᵀ = ∫ ∂G/∂n_x dS_y`.
    pub ht: Complex64,
    /// `E = ∫ ∂²G/∂n_x∂n_y dS_y`.
    pub e: Complex64,
}

/// Real-valued Laplace regular-pair output `(G, H, Hᵀ, E)`.
///
/// This ABI is intentionally separate from [`BemRegularPairOutF64`]: the
/// CUDA kernel performs only `f64` arithmetic and publishes four scalar
/// channels, rather than computing Complex64 values and discarding their
/// imaginary parts at the host boundary.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BemRegularPairOutRealF64 {
    /// `G = ∫ 1/(4πr) dS_y`.
    pub g: f64,
    /// `H = ∫ ∂G/∂n_y dS_y`.
    pub h: f64,
    /// `Hᵀ = ∫ ∂G/∂n_x dS_y`.
    pub ht: f64,
    /// `E = ∫ ∂²G/∂n_x∂n_y dS_y`.
    pub e: f64,
}

impl BufferElement for BemRegularPairIn {}
impl BufferElement for BemRegularPairOut {}
impl BufferElement for BemRegularPairInF64 {}
impl BufferElement for BemRegularPairOutF64 {}
impl BufferElement for BemRegularPairOutRealF64 {}
impl BufferElement for BemFieldElementC64 {}
impl BufferElement for BemFieldPointC64 {}
impl BufferElement for BemQuadratureSampleC64 {}
impl BufferElement for BemQuadraturePairC64 {}
