//! Runtime detection of ARM SME2 capabilities and cache geometry.

/// SME2 capability flags for this machine.
#[derive(Debug, Clone, Copy)]
pub struct Sme2Caps {
    /// SME2 is available
    pub has_sme2: bool,
    /// f64 outer product (FEAT_SME_F64F64)
    pub has_f64: bool,
    /// f32 outer product (always true if SME2)
    pub has_f32: bool,
    /// Streaming Vector Length in bits (128, 256, 512, etc.)
    pub svl_bits: usize,
    /// ZA tile size in bytes: (svl_bits/8)^2
    pub za_bytes: usize,
}

static SME2_CAPS: std::sync::OnceLock<Sme2Caps> = std::sync::OnceLock::new();

impl Sme2Caps {
    /// Detect SME2 capabilities at runtime (cached after first call).
    ///
    /// On macOS aarch64: checks sysctl `hw.optional.arm.FEAT_SME2` and
    /// `hw.optional.arm.FEAT_SME_F64F64`.
    ///
    /// On Linux aarch64: checks `getauxval(AT_HWCAP2)` for SME2 flags.
    ///
    /// On all other platforms: returns all-false (no SME2 support).
    pub fn detect() -> Self {
        *SME2_CAPS.get_or_init(Self::detect_inner)
    }

    fn detect_inner() -> Self {
        #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
        {
            detect_aarch64()
        }
        #[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux"))))]
        {
            Self {
                has_sme2: false,
                has_f64: false,
                has_f32: false,
                svl_bits: 0,
                za_bytes: 0,
            }
        }
    }
}

// ---- SME2 assembly helper (linked from sme2_kernels.S) ----

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
unsafe extern "C" {
    /// Read the Streaming Vector Length in bytes via SMSTART SM + RDSVL + SMSTOP SM.
    /// Only safe to call when SME2 is confirmed present.
    fn sonium_read_svl_bytes() -> usize;
}

// ---- macOS aarch64 detection via sysctl ----

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn detect_aarch64() -> Sme2Caps {
    let has_sme2 = sysctl_bool(c"hw.optional.arm.FEAT_SME2");
    let has_f64 = sysctl_bool(c"hw.optional.arm.FEAT_SME_F64F64");
    let has_f32 = has_sme2; // SME2 implies f32 outer product

    // Read actual SVL by entering streaming mode and executing RDSVL.
    // Only safe to call after confirming SME2 is present.
    let svl_bits = if has_sme2 {
        let svl_bytes = unsafe { sonium_read_svl_bytes() };
        svl_bytes * 8
    } else {
        0
    };
    let svl_bytes = svl_bits / 8;
    let za_bytes = svl_bytes * svl_bytes;

    Sme2Caps {
        has_sme2,
        has_f64,
        has_f32,
        svl_bits,
        za_bytes,
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn sysctl_bool(name: &core::ffi::CStr) -> bool {
    let mut val: i32 = 0;
    let mut size = core::mem::size_of::<i32>();
    let ret = unsafe {
        libc_sysctlbyname(
            name.as_ptr(),
            &mut val as *mut i32 as *mut core::ffi::c_void,
            &mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    ret == 0 && val != 0
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
unsafe extern "C" {
    #[link_name = "sysctlbyname"]
    fn libc_sysctlbyname(
        name: *const core::ffi::c_char,
        oldp: *mut core::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut core::ffi::c_void,
        newlen: usize,
    ) -> core::ffi::c_int;
}

// ---- Linux aarch64 detection via getauxval ----

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn detect_aarch64() -> Sme2Caps {
    // HWCAP2 bits for SME2 (from Linux kernel headers)
    const HWCAP2_SME2: u64 = 1 << 37;
    const HWCAP2_SME_F64F64: u64 = 1 << 35;

    let hwcap2 = unsafe { libc_getauxval(26) }; // AT_HWCAP2 = 26

    let has_sme2 = (hwcap2 & HWCAP2_SME2) != 0;
    let has_f64 = (hwcap2 & HWCAP2_SME_F64F64) != 0;
    let has_f32 = has_sme2;

    let svl_bits = if has_sme2 {
        let svl_bytes = unsafe { sonium_read_svl_bytes() };
        svl_bytes * 8
    } else {
        0
    };
    let svl_bytes = svl_bits / 8;
    let za_bytes = svl_bytes * svl_bytes;

    Sme2Caps {
        has_sme2,
        has_f64,
        has_f32,
        svl_bits,
        za_bytes,
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
unsafe extern "C" {
    #[link_name = "getauxval"]
    fn libc_getauxval(type_: u64) -> u64;
}

// ----------------------------------------------------------------------------
// L2 working-set budget detection
// ----------------------------------------------------------------------------
//
// Used by the GEMM cache-blocking decision. We want a byte count below which
// the unblocked B band fits comfortably in shared L2 alongside packed_A, the
// C row band, and the still-cold A/B rows being streamed.
//
// Detection priority:
//   1. `SONIUM_L2_BUDGET_BYTES` env var — explicit override.
//   2. macOS: `hw.perflevel0.l2cachesize` (P-cluster L2) / 2.
//      `hw.l2cachesize` only reports the E-cluster on Apple silicon, so we
//      prefer the perflevel0 sysctl; / 2 keeps headroom when sibling P-cores
//      are active.
//   3. Linux: `sysconf(_SC_LEVEL2_CACHE_SIZE)` / 2.
//   4. Fallback: 8 MB (matches the prior hard-coded constant).

const L2_BUDGET_FALLBACK: usize = 8 * 1024 * 1024;
const L2_BUDGET_MIN: usize = 256 * 1024; // 256 KB — guard against bogus tiny values

static L2_BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Per-thread L2 working-set budget for GEMM cache-blocking decisions, in bytes.
///
/// Cached on first call. Honours the `SONIUM_L2_BUDGET_BYTES` env var (must
/// parse as a positive `usize`). On macOS/Linux falls back to a platform
/// probe; elsewhere returns 8 MB.
pub fn l2_budget_bytes() -> usize {
    *L2_BUDGET.get_or_init(detect_l2_budget)
}

fn detect_l2_budget() -> usize {
    if let Ok(s) = std::env::var("SONIUM_L2_BUDGET_BYTES")
        && let Ok(n) = s.parse::<usize>()
        && n >= L2_BUDGET_MIN
    {
        return n;
    }
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        if let Some(p) = sysctl_usize(c"hw.perflevel0.l2cachesize") {
            return (p / 2).max(L2_BUDGET_MIN);
        }
        if let Some(p) = sysctl_usize(c"hw.l2cachesize") {
            return (p / 2).max(L2_BUDGET_MIN);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        // _SC_LEVEL2_CACHE_SIZE = 188 on Linux/aarch64 glibc.
        const SC_LEVEL2_CACHE_SIZE: core::ffi::c_int = 188;
        let sz = unsafe { libc_sysconf(SC_LEVEL2_CACHE_SIZE) };
        if sz > 0 {
            return ((sz as usize) / 2).max(L2_BUDGET_MIN);
        }
    }
    L2_BUDGET_FALLBACK
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn sysctl_usize(name: &core::ffi::CStr) -> Option<usize> {
    let mut val: u64 = 0;
    let mut size = core::mem::size_of::<u64>();
    let ret = unsafe {
        libc_sysctlbyname(
            name.as_ptr(),
            &mut val as *mut u64 as *mut core::ffi::c_void,
            &mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 && val > 0 {
        Some(val as usize)
    } else {
        None
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
unsafe extern "C" {
    #[link_name = "sysconf"]
    fn libc_sysconf(name: core::ffi::c_int) -> core::ffi::c_long;
}
