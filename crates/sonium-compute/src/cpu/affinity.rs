//! Best-effort worker-thread CPU affinity (Linux).
//!
//! Vendor BLAS pins its workers by default (OpenBLAS unless
//! `OPENBLAS_SET_AFFINITY=0`): without it, pool threads scatter across SMT
//! siblings and CCDs and FMA-bound bands share physical cores (dgetrf-1024
//! 8T on 64c Threadripper: 15.8ms unpinned vs 10.1ms pinned to CPUs 0-7).
//! Sonium pins the workers of every pool it owns, plus the ambient-sized
//! mirror pool behind top-level scopes, to distinct inherited-mask CPUs in
//! order. Packing won over CCD-spreading 2.3x for factorizations
//! (dpotrf-1024 8T: packed 4.4-4.5 vs spread 9.9-11.0, interleaved x3):
//! cross-panel L3 reuse of the trailing matrix beats conflict avoidance,
//! even though spreading wins cold single-SYRK calls (0.43 -> 0.26ms).
//! At 2048/16T scheduler placement only ties packing (dgetrf full 46.5
//! vs 45.2, dpotrf 16.0 vs 16.4 med) — the working sets exceed per-CCX
//! L3 either way, so placement stops mattering; packing stands.
//! `SONIUM_SET_AFFINITY=0` restores scheduler placement (diagnostic/A-B
//! fallback).
//!
//! Factorizations whose working set fits in L3 go further: a compact pool
//! of at most [`COMPACT_MAX_CPUS`] workers pinned to one L3 domain (first
//! CPUs of the sysfs L3 group containing CPU 0) keeps the trailing matrix
//! L3-resident across panels — dgetrf-1024 ~12.5ms vs ~14.6 spread, any
//! single CCX reproduces it under taskset. Drivers enter via
//! `run_compact` (clone inside for local NUMA first-touch) with an L3-fit
//! guard; `SONIUM_COMPACT=0` forces the mirror-pool funnel.
//!
//! Everything is best-effort: pin failures are ignored (a restricted
//! sandbox may refuse), non-Linux is a no-op, and the inherited mask
//! (containers/cgroups/taskset) is always respected — workers cycle within
//! it when it holds fewer CPUs than the pool. Results never change: only
//! thread placement is affected.

use std::sync::LazyLock;

/// Inherited CPU list, or `None` when affinity is disabled, unavailable,
/// or the mask read failed. Read once: the mask is fixed for the process.
static CPU_LIST: LazyLock<Option<Vec<usize>>> = LazyLock::new(|| {
    if std::env::var("SONIUM_SET_AFFINITY")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return None;
    }
    inherited_cpus()
});

#[cfg(target_os = "linux")]
fn inherited_cpus() -> Option<Vec<usize>> {
    // `sched_getaffinity` on pid 0 queries the calling thread; pool
    // builders run on threads that inherit the process mask, so this is
    // the mask the workers would get without pinning.
    // SAFETY: `set` is a valid zeroed `cpu_set_t`; `CPU_ISSET` only reads
    // indices below `CPU_SETSIZE`.
    unsafe {
        let mut set = std::mem::zeroed::<libc::cpu_set_t>();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return None;
        }
        let mut out = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &set) {
                out.push(cpu);
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

#[cfg(not(target_os = "linux"))]
fn inherited_cpus() -> Option<Vec<usize>> {
    None
}

/// Number of CPUs in the inherited affinity mask, or `None` when pinning
/// is disabled or the mask is unreadable. Lets pool sizing respect
/// cpusets/masks (containers, numactl, taskset) instead of oversubscribing
/// the allowed CPUs with an ambient-sized pool.
pub(crate) fn mask_cpu_count() -> Option<usize> {
    CPU_LIST.as_ref().map(|list| list.len())
}

/// The CPU `pin_worker(index)` will pin to, or `None` when pinning is a
/// no-op (disabled, unavailable, or the mask read failed). Test-only.
#[cfg(test)]
pub(crate) fn pinned_cpu(index: usize) -> Option<usize> {
    CPU_LIST.as_ref().map(|list| list[index % list.len()])
}

/// Current thread's affinity mask, for tests. Empty when unreadable.
#[cfg(test)]
pub(crate) fn current_mask() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    // SAFETY: as in `inherited_cpus`.
    unsafe {
        let mut set = std::mem::zeroed::<libc::cpu_set_t>();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &set) {
                out.push(cpu);
            }
        }
        out
    }
    #[cfg(not(target_os = "linux"))]
    Vec::new()
}

/// CPUs sharing one L3 domain, for the compact factorization pool:
/// the first [`COMPACT_MAX_CPUS`] CPUs of the L3 group containing the
/// first inherited CPU. One CCX keeps the trailing matrix L3-resident
/// across panels (dgetrf-1024: any single 4-core CCX ~11.7ms vs ~14.6
/// spread over the box). `None` when affinity is disabled, the topology
/// is unreadable, or a group cannot be formed — callers fall back to the
/// mirror pool. Read once: topology is fixed for the process.
pub(crate) const COMPACT_MAX_CPUS: usize = 4;

static COMPACT_CPUS: LazyLock<Option<Vec<usize>>> = LazyLock::new(|| {
    // Manual override first (diagnostic/A-B and odd topologies):
    // `SONIUM_COMPACT_CPUS=0-7,64-67` pins the compact pool exactly.
    if let Ok(text) = std::env::var("SONIUM_COMPACT_CPUS") {
        let mut cpus = parse_cpu_list(&text);
        if let Some(list) = CPU_LIST.as_ref() {
            cpus.retain(|c| list.contains(c));
        }
        if !cpus.is_empty() {
            return Some(cpus);
        }
    }
    let list = CPU_LIST.as_ref()?;
    if list.len() <= COMPACT_MAX_CPUS {
        return Some(list.clone());
    }
    let group = l3_group(list[0])?;
    if group.is_empty() {
        return None;
    }
    let take = group.len().min(COMPACT_MAX_CPUS);
    Some(group[..take].to_vec())
});

/// L3-group count for the wide compact pool: the first
/// [`COMPACT_MAX_CPUS`] CPUs of each of the first
/// [`COMPACT_WIDE_GROUPS`] L3 groups. Four stands (2026-10-02,
/// `cholphases` process-interleaved A/B, RAYON=16): 16 workers on four
/// CCXs beat 8-on-two 14/14 rounds at dpotrf-1024 (~-6%, SYRK phase
/// -11%) and tie 4/4 at dpotrf-512, while the same 16 CPUs via the
/// mirror-spawn funnel lose — the win needs install-funneling (see
/// `run_mirror`), not just width. The 8 MB factor stays L3-resident
/// across four 16 MB slices. Masks with fewer groups degrade
/// gracefully (`take` what exists; one group collapses to narrow via
/// the equality check below). `SONIUM_COMPACT_CPUS` still pins both
/// sets exactly (diagnostic/A-B and kill-switch). History: two groups
/// stood before (compute-dense rank-64 SYRK wants two L3 domains:
/// spotrf-1024 7.3 -> 5.9ms vs one CCX, while SMT twins on one CCX
/// lose); four extends the same L3-domain mechanism with width.
pub(crate) const COMPACT_WIDE_GROUPS: usize = 4;

/// CPUs for the wide compact pool: the first [`COMPACT_MAX_CPUS`] CPUs
/// of each of the first [`COMPACT_WIDE_GROUPS`] L3 groups.
/// `None` when the wide set would equal the narrow set (single group or
/// small mask: callers share the narrow pool), so no duplicate pool is
/// ever built. The manual override pins both sets exactly (narrow and
/// wide coincide there by construction).
static COMPACT_CPUS_WIDE: LazyLock<Option<Vec<usize>>> = LazyLock::new(|| {
    if std::env::var("SONIUM_COMPACT_CPUS").is_ok() {
        return None;
    }
    let groups = l3_groups()?;
    if groups.len() < 2 {
        return None;
    }
    let mut out = Vec::new();
    for group in groups.iter().take(COMPACT_WIDE_GROUPS) {
        out.extend_from_slice(&group[..group.len().min(COMPACT_MAX_CPUS)]);
    }
    let narrow = COMPACT_CPUS.clone()?;
    if out == narrow {
        return None;
    }
    Some(out)
});

/// Worker count for the spread pool (streaming kernels): one worker per
/// L3 group on a 16-group box, so each worker owns a distinct L3 slice.
/// Sized down by the mask when fewer CPUs exist (see [`spread_cpus`]).
pub(crate) const SPREAD_THREADS: usize = 16;

/// CPUs for the spread pool, if topology allowed one to be formed: one CPU
/// per L3 group round-robin (second round takes the next CPU of each
/// group), so workers never share an L3 slice until groups run out.
/// Test-visible so pool wiring can be asserted without running a solve.
pub(crate) fn spread_cpus() -> Option<Vec<usize>> {
    SPREAD_CPUS.clone()
}

static SPREAD_CPUS: LazyLock<Option<Vec<usize>>> = LazyLock::new(|| {
    // Manual override first (diagnostic/A-B and odd topologies).
    if let Ok(text) = std::env::var("SONIUM_SPREAD_CPUS") {
        let mut cpus = parse_cpu_list(&text);
        if let Some(list) = CPU_LIST.as_ref() {
            cpus.retain(|c| list.contains(c));
        }
        if !cpus.is_empty() {
            return Some(cpus);
        }
    }
    spread_cpu_set(SPREAD_THREADS)
});

/// Round-robin one CPU per L3 group until `width` CPUs are collected (or
/// groups run dry): each worker lands on a distinct L3 slice first, and
/// only shares once every group contributed. `None` when affinity is
/// disabled, the topology is unreadable, or no CPU could be chosen.
fn spread_cpu_set(width: usize) -> Option<Vec<usize>> {
    let groups = l3_groups()?;
    spread_from_groups(&groups, width)
}

/// Pure round-robin core of [`spread_cpu_set`], over caller-supplied
/// groups so the schedule is unit-testable without touching sysfs.
fn spread_from_groups(groups: &[Vec<usize>], width: usize) -> Option<Vec<usize>> {
    if groups.is_empty() || width == 0 {
        return None;
    }
    let mut out = Vec::new();
    let mut round = 0;
    while out.len() < width {
        let mut progressed = false;
        for g in groups.iter() {
            if out.len() >= width {
                break;
            }
            if g.is_empty() {
                continue;
            }
            let cpu = g[round % g.len()];
            if !out.contains(&cpu) {
                out.push(cpu);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
        round += 1;
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Distinct L3 groups covering the inherited mask, in CPU order.
/// `None` when affinity is disabled or no group forms.
fn l3_groups() -> Option<Vec<Vec<usize>>> {
    let list = CPU_LIST.as_ref()?;
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for cpu in list {
        if groups.iter().any(|g| g.contains(cpu)) {
            continue;
        }
        let group = l3_group(*cpu)?;
        if group.is_empty() {
            continue;
        }
        groups.push(group);
    }
    if groups.is_empty() {
        return None;
    }
    Some(groups)
}

/// L3-sharing CPUs of `cpu`, from sysfs (`shared_cpu_list` of the
/// level-3 cache; deepest level when no level 3 exists). Intersected with
/// the inherited mask. `None` when unreadable.
fn l3_group(cpu: usize) -> Option<Vec<usize>> {
    let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
    let mut best: Option<(u32, Vec<usize>)> = None;
    for index in 0..8 {
        let Ok(level_text) = std::fs::read_to_string(format!("{base}/index{index}/level")) else {
            continue;
        };
        let Ok(level) = level_text.trim().parse::<u32>() else {
            continue;
        };
        let Ok(shared) = std::fs::read_to_string(format!("{base}/index{index}/shared_cpu_list"))
        else {
            continue;
        };
        let mut cpus = parse_cpu_list(&shared);
        let list = CPU_LIST.as_ref()?;
        cpus.retain(|c| list.contains(c));
        if cpus.is_empty() {
            continue;
        }
        // Prefer level 3; otherwise the deepest level seen.
        if level == 3 {
            return Some(cpus);
        }
        if best.as_ref().map(|b| level > b.0).unwrap_or(true) {
            best = Some((level, cpus));
        }
    }
    best.map(|b| b.1)
}

/// Level-3 slice size in bytes for `cpu`'s slice, from sysfs (`size` of
/// the level-3 index). `None` when unreadable or unparsable.
fn l3_slice_bytes(cpu: usize) -> Option<usize> {
    let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
    for index in 0..8 {
        let Ok(level_text) = std::fs::read_to_string(format!("{base}/index{index}/level")) else {
            continue;
        };
        if level_text.trim() != "3" {
            continue;
        }
        let Ok(size_text) = std::fs::read_to_string(format!("{base}/index{index}/size")) else {
            continue;
        };
        let s = size_text.trim();
        let (num, mult) = match s.chars().last()? {
            'K' => (s[..s.len() - 1].parse::<usize>().ok()?, 1024usize),
            'M' => (s[..s.len() - 1].parse::<usize>().ok()?, 1024 * 1024),
            'G' => (s[..s.len() - 1].parse::<usize>().ok()?, 1024 * 1024 * 1024),
            _ => (s.parse::<usize>().ok()?, 1),
        };
        return num.checked_mul(mult);
    }
    None
}

/// Total level-3 bytes across the machine's unique L3 groups (one slice
/// size per group, so asymmetric topologies sum correctly). Used to size
/// N-split GEMM bands (bands × A footprint past L3 self-evicts). Cached;
/// `None` when topology or sizes are unreadable (callers keep defaults).
pub(crate) fn l3_total_bytes() -> Option<usize> {
    static TOTAL: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
        let groups = l3_groups()?;
        let mut total = 0usize;
        for g in &groups {
            total = total.checked_add(l3_slice_bytes(*g.first()?)?)?;
        }
        Some(total)
    });
    *TOTAL
}

/// Parse a sysfs cpu list (`0-3,8-11`) into sorted CPU ids.
fn parse_cpu_list(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in text.trim().split(',') {
        let part = part.trim();
        if let Some((lo, hi)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (lo.trim().parse::<usize>(), hi.trim().parse::<usize>()) {
                out.extend(lo..=hi);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            out.push(cpu);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The compact CPU set for the factorization pool, if topology allowed
/// one to be formed. Test-visible so pool wiring can be asserted without
/// running a solve.
pub(crate) fn compact_cpus() -> Option<Vec<usize>> {
    COMPACT_CPUS.clone()
}

/// The wide compact set ([`COMPACT_WIDE_GROUPS`] L3 domains), if it
/// differs from the narrow set. Test-visible; see [`compact_cpus`].
pub(crate) fn compact_cpus_wide() -> Option<Vec<usize>> {
    COMPACT_CPUS_WIDE.clone()
}

/// L3 size in bytes (representative CPU 0 entry), for the compact-pool
/// fit guard. `None` when unreadable.
///
/// Cached process-wide: the value is hardware-invariant, and the serial
/// GEMM gate evaluates this per call — an uncached sysfs read there cost
/// ~40% on thin serial Schur updates (2026-09-28 merge regression).
pub(crate) fn l3_bytes() -> Option<u64> {
    static CACHED: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CACHED.get_or_init(l3_bytes_uncached)
}

fn l3_bytes_uncached() -> Option<u64> {
    let text = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size").ok()?;
    let text = text.trim();
    let (digits, mult) = if let Some(k) = text.strip_suffix('K') {
        (k, 1024u64)
    } else if let Some(m) = text.strip_suffix('M') {
        (m, 1024u64 * 1024)
    } else {
        (text, 1u64)
    };
    digits.trim().parse::<u64>().ok().map(|v| v * mult)
}

/// Pin the calling worker to the compact set. No-op when the set is
/// empty (pool is never built then, so this is unreachable in practice).
pub(crate) fn pin_compact_worker(index: usize) {
    let Some(set) = COMPACT_CPUS.as_ref() else {
        return;
    };
    if set.is_empty() {
        return;
    }
    let cpu = set[index % set.len()];
    #[cfg(target_os = "linux")]
    // SAFETY: as in `pin_worker`.
    unsafe {
        let mut mask = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_SET(cpu, &mut mask);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mask);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

/// Pin the calling worker to the spread set (one L3 slice per worker).
/// Unreachable when the set is empty (no spread pool is built then).
pub(crate) fn pin_spread_worker(index: usize) {
    let Some(set) = SPREAD_CPUS.as_ref() else {
        return;
    };
    if set.is_empty() {
        return;
    }
    let cpu = set[index % set.len()];
    #[cfg(target_os = "linux")]
    // SAFETY: as in `pin_worker`.
    unsafe {
        let mut mask = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_SET(cpu, &mut mask);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mask);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

/// Pin the calling worker to the wide compact set. Unreachable when the
/// set is empty (no wide pool is built then).
pub(crate) fn pin_compact_wide_worker(index: usize) {
    let Some(set) = COMPACT_CPUS_WIDE.as_ref() else {
        return;
    };
    if set.is_empty() {
        return;
    }
    let cpu = set[index % set.len()];
    #[cfg(target_os = "linux")]
    // SAFETY: as in `pin_worker`.
    unsafe {
        let mut mask = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_SET(cpu, &mut mask);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mask);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

/// Pin the calling worker to `CPU_LIST[index % len]`. No-op when affinity
/// is disabled, unavailable, or the mask read failed.
pub(crate) fn pin_worker(index: usize) {
    let Some(list) = CPU_LIST.as_ref() else {
        return;
    };
    let cpu = list[index % list.len()];
    #[cfg(target_os = "linux")]
    // SAFETY: `set` is a valid `cpu_set_t` with exactly `cpu` set; a
    // nonzero return (sandbox refusal) is ignored by design.
    unsafe {
        let mut set = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_list_handles_ranges_singles_and_empty() {
        assert_eq!(
            parse_cpu_list("0-3,64-67"),
            vec![0, 1, 2, 3, 64, 65, 66, 67]
        );
        assert_eq!(parse_cpu_list("5"), vec![5]);
        assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
        assert_eq!(parse_cpu_list("3-1"), Vec::<usize>::new());
        assert_eq!(parse_cpu_list("0-1,1-2"), vec![0, 1, 2]);
    }

    #[test]
    fn compact_set_is_a_small_valid_subset_when_present() {
        if let Some(set) = compact_cpus() {
            assert!(!set.is_empty());
            assert!(set.len() <= COMPACT_MAX_CPUS);
            let mut sorted = set.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted, set);
            if let Some(list) = CPU_LIST.as_ref() {
                assert!(set.iter().all(|c| list.contains(c)));
            }
        }
    }

    #[test]
    fn l3_size_is_sane_when_readable() {
        if let Some(bytes) = l3_bytes() {
            assert!(bytes >= 1024);
        }
    }

    #[test]
    fn spread_schedule_hits_distinct_slices_first() {
        // Four L3 slices of four CPUs: the first four workers must own
        // distinct slices (one CPU from each group), the second four the
        // next CPU of each group. Synthetic groups keep this sysfs-free.
        let groups = vec![
            vec![0, 1, 2, 3],
            vec![8, 9, 10, 11],
            vec![16, 17, 18, 19],
            vec![24, 25, 26, 27],
        ];
        assert_eq!(
            spread_from_groups(&groups, 8),
            Some(vec![0, 8, 16, 24, 1, 9, 17, 25])
        );
        // Width beyond the total yields every CPU exactly once and the
        // loop still terminates (no duplicate endlessly re-added).
        assert_eq!(
            spread_from_groups(&groups, 100),
            Some((0..4).flat_map(|r| [r, 8 + r, 16 + r, 24 + r]).collect())
        );
        // Uneven + empty groups: empties skipped, sharing only by round.
        let ragged = vec![vec![], vec![0], vec![4, 5, 6]];
        assert_eq!(spread_from_groups(&ragged, 4), Some(vec![0, 4, 5, 6]));
        // Degenerate inputs yield None rather than an empty pool.
        assert_eq!(spread_from_groups(&groups, 0), None);
        assert_eq!(spread_from_groups(&[], 8), None);
        assert_eq!(spread_from_groups(&[vec![], vec![]], 8), None);
    }

    #[test]
    fn spread_set_is_unique_and_masked_when_present() {
        if std::env::var("SONIUM_SPREAD_CPUS").is_ok() {
            return;
        }
        if let Some(set) = spread_cpus() {
            assert!(!set.is_empty());
            assert!(set.len() <= SPREAD_THREADS);
            let mut unique = set.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), set.len());
            if let Some(list) = CPU_LIST.as_ref() {
                assert!(set.iter().all(|c| list.contains(c)));
            }
        }
    }

    #[test]
    fn wide_set_covers_four_domains_when_present() {
        // Without the manual override (tests never set it), a wide set
        // must hold at most `COMPACT_WIDE_GROUPS` narrow-sized groups
        // from distinct L3 domains and differ from the narrow set
        // (2026-10-02: four groups; dpotrf-1024 -6% vs two, ties 512).
        if std::env::var("SONIUM_COMPACT_CPUS").is_ok() {
            return;
        }
        if let Some(wide) = compact_cpus_wide() {
            assert!(!wide.is_empty());
            assert!(wide.len() <= COMPACT_WIDE_GROUPS * COMPACT_MAX_CPUS);
            let mut unique = wide.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), wide.len());
            if let Some(narrow) = compact_cpus() {
                assert_ne!(wide, narrow);
                assert!(wide.len() > narrow.len());
            }
            if let Some(list) = CPU_LIST.as_ref() {
                assert!(wide.iter().all(|c| list.contains(c)));
            }
        }
    }
}
