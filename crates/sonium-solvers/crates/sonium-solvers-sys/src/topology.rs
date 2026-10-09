//! Machine topology detection for bandwidth-bound solver kernels.
//!
//! The Linux path uses scheduler affinity to keep first-touch allocation and
//! repeated partition applies on the CPUs associated with a NUMA node. macOS
//! and other platforms report topology but leave affinity as an explicit no-op.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineTopology {
    pub total_logical_cpus: usize,
    pub numa_nodes: Vec<NumaNode>,
    pub sockets: Vec<SocketInfo>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaNode {
    pub id: usize,
    pub logical_cpus: Vec<usize>,
    pub memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocketInfo {
    pub id: usize,
    pub logical_cpus: Vec<usize>,
    pub numa_node_ids: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerPlacement {
    pub worker_index: usize,
    pub numa_node_id: usize,
    pub socket_id: usize,
    pub logical_cpus: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementPlan {
    pub workers: Vec<WorkerPlacement>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffinityStatus {
    Applied { cpus: Vec<usize> },
    Unsupported { reason: String },
    EmptyCpuList,
    Failed { cpus: Vec<usize>, error: String },
}

impl AffinityStatus {
    pub fn is_applied(&self) -> bool {
        matches!(self, Self::Applied { .. })
    }
}

impl fmt::Display for AffinityStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Applied { cpus } => write!(f, "applied cpus={}", format_cpu_list(cpus)),
            Self::Unsupported { reason } => write!(f, "unsupported: {reason}"),
            Self::EmptyCpuList => write!(f, "not applied: empty CPU list"),
            Self::Failed { cpus, error } => {
                write!(f, "failed cpus={}: {error}", format_cpu_list(cpus))
            }
        }
    }
}

impl MachineTopology {
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        if let Some(topology) = detect_linux() {
            return topology;
        }

        #[cfg(target_os = "macos")]
        if let Some(topology) = detect_macos() {
            return topology;
        }

        detect_generic()
    }

    pub fn placement_plan(&self, worker_count: usize) -> PlacementPlan {
        if worker_count == 0 {
            return PlacementPlan {
                workers: Vec::new(),
                notes: vec!["no workers requested".to_string()],
            };
        }

        let mut workers = Vec::with_capacity(worker_count);
        let nodes = if self.numa_nodes.is_empty() {
            vec![NumaNode {
                id: 0,
                logical_cpus: (0..self.total_logical_cpus.max(1)).collect(),
                memory_bytes: None,
            }]
        } else {
            self.numa_nodes.clone()
        };

        // Advisory split: each worker gets a disjoint strided subset of its
        // node's CPUs instead of the whole node list (which double-booked
        // cores when two workers shared a node). Pinning itself is still
        // Linux-only; treat the lists as guidance, not affinity.
        for worker_index in 0..worker_count {
            let node = &nodes[worker_index % nodes.len()];
            let socket_id = self.socket_for_node(node).unwrap_or(0);
            let lane = worker_index / nodes.len().max(1);
            let stride = (worker_count + nodes.len() - 1) / nodes.len().max(1);
            let logical_cpus: Vec<usize> = node
                .logical_cpus
                .iter()
                .skip(lane)
                .step_by(stride.max(1))
                .copied()
                .collect();
            workers.push(WorkerPlacement {
                worker_index,
                numa_node_id: node.id,
                socket_id,
                logical_cpus,
            });
        }

        let mut notes = Vec::new();
        notes.push(
            "placement is advisory (Linux affinity only; macOS/Windows ignore pinning)".to_string(),
        );
        if worker_count > nodes.len() {
            notes.push(format!(
                "workers exceed NUMA nodes; round-robin placement over {} node(s)",
                nodes.len()
            ));
        }
        if self.sockets.len() > 1 {
            notes.push(format!("{} sockets detected", self.sockets.len()));
        }

        PlacementPlan { workers, notes }
    }

    pub fn short_summary(&self) -> String {
        let numa = self
            .numa_nodes
            .iter()
            .map(|node| {
                let mem = node
                    .memory_bytes
                    .map(format_bytes)
                    .unwrap_or_else(|| "unknown memory".to_string());
                format!(
                    "node{} cpus={} {mem}",
                    node.id,
                    format_cpu_list(&node.logical_cpus)
                )
            })
            .collect::<Vec<_>>()
            .join("; ");

        format!(
            "{} logical CPUs, {} socket(s), {} NUMA node(s) [{}]",
            self.total_logical_cpus,
            self.sockets.len().max(1),
            self.numa_nodes.len().max(1),
            numa
        )
    }

    fn socket_for_node(&self, node: &NumaNode) -> Option<usize> {
        self.sockets
            .iter()
            .find(|socket| socket.numa_node_ids.contains(&node.id))
            .map(|socket| socket.id)
            .or_else(|| {
                self.sockets
                    .iter()
                    .find(|socket| {
                        node.logical_cpus
                            .iter()
                            .any(|cpu| socket.logical_cpus.contains(cpu))
                    })
                    .map(|socket| socket.id)
            })
    }
}

impl PlacementPlan {
    pub fn worker_for_index(&self, index: usize) -> Option<&WorkerPlacement> {
        if self.workers.is_empty() {
            None
        } else {
            self.workers.get(index % self.workers.len())
        }
    }

    pub fn short_summary(&self) -> String {
        if self.workers.is_empty() {
            return "no workers".to_string();
        }

        self.workers
            .iter()
            .map(|worker| {
                format!(
                    "w{}:node{}/socket{}/cpus={}",
                    worker.worker_index,
                    worker.numa_node_id,
                    worker.socket_id,
                    format_cpu_list(&worker.logical_cpus)
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

pub fn pin_current_thread_to(cpus: &[usize]) -> AffinityStatus {
    if cpus.is_empty() {
        return AffinityStatus::EmptyCpuList;
    }

    #[cfg(target_os = "linux")]
    {
        pin_current_thread_to_linux(cpus)
    }

    #[cfg(not(target_os = "linux"))]
    {
        AffinityStatus::Unsupported {
            reason: format!(
                "thread affinity is only implemented on Linux; requested cpus={}",
                format_cpu_list(cpus)
            ),
        }
    }
}

pub fn current_thread_affinity() -> AffinityStatus {
    #[cfg(target_os = "linux")]
    {
        current_thread_affinity_linux()
    }

    #[cfg(not(target_os = "linux"))]
    {
        AffinityStatus::Unsupported {
            reason: "thread affinity inspection is only implemented on Linux".to_string(),
        }
    }
}

#[cfg(target_os = "linux")]
fn pin_current_thread_to_linux(cpus: &[usize]) -> AffinityStatus {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    unsafe {
        libc::CPU_ZERO(&mut set);
    }
    for &cpu in cpus {
        unsafe {
            libc::CPU_SET(cpu, &mut set);
        }
    }

    let rc = unsafe {
        libc::pthread_setaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        )
    };
    if rc == 0 {
        AffinityStatus::Applied {
            cpus: cpus.to_vec(),
        }
    } else {
        AffinityStatus::Failed {
            cpus: cpus.to_vec(),
            error: std::io::Error::from_raw_os_error(rc).to_string(),
        }
    }
}

#[cfg(target_os = "linux")]
fn current_thread_affinity_linux() -> AffinityStatus {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let rc = unsafe {
        libc::pthread_getaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            &mut set,
        )
    };
    if rc != 0 {
        return AffinityStatus::Failed {
            cpus: Vec::new(),
            error: std::io::Error::from_raw_os_error(rc).to_string(),
        };
    }

    let cpus = (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) })
        .collect::<Vec<_>>();
    AffinityStatus::Applied { cpus }
}

pub fn parse_cpu_list(input: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in input
        .trim()
        .split(',')
        .filter(|part| !part.trim().is_empty())
    {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            let Some(start) = start.trim().parse::<usize>().ok() else {
                continue;
            };
            let Some(end) = end.trim().parse::<usize>().ok() else {
                continue;
            };
            if start <= end {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

pub fn format_cpu_list(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return "none".to_string();
    }

    let mut ranges = Vec::new();
    let mut start = cpus[0];
    let mut prev = cpus[0];
    for &cpu in &cpus[1..] {
        if cpu == prev + 1 {
            prev = cpu;
            continue;
        }
        push_cpu_range(&mut ranges, start, prev);
        start = cpu;
        prev = cpu;
    }
    push_cpu_range(&mut ranges, start, prev);
    ranges.join(",")
}

fn push_cpu_range(out: &mut Vec<String>, start: usize, end: usize) {
    if start == end {
        out.push(start.to_string());
    } else {
        out.push(format!("{start}-{end}"));
    }
}

#[cfg(target_os = "linux")]
fn detect_linux() -> Option<MachineTopology> {
    use std::fs;
    use std::path::PathBuf;

    let mut nodes = Vec::new();
    for entry in fs::read_dir("/sys/devices/system/node").ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name = name.to_str()?;
        let Some(id) = name.strip_prefix("node").and_then(|s| s.parse().ok()) else {
            continue;
        };

        let path = PathBuf::from("/sys/devices/system/node").join(name);
        let cpulist = fs::read_to_string(path.join("cpulist")).unwrap_or_default();
        nodes.push(NumaNode {
            id,
            logical_cpus: parse_cpu_list(&cpulist),
            memory_bytes: read_linux_node_memory_bytes(path.join("meminfo")),
        });
    }

    nodes.sort_by_key(|node| node.id);
    nodes.retain(|node| !node.logical_cpus.is_empty());
    if nodes.is_empty() {
        return None;
    }

    let total_logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or_else(|_| nodes.iter().map(|node| node.logical_cpus.len()).sum());
    let sockets = detect_linux_sockets(&nodes, total_logical_cpus);

    Some(MachineTopology {
        total_logical_cpus,
        numa_nodes: nodes,
        sockets,
        notes: vec!["topology from Linux sysfs".to_string()],
    })
}

#[cfg(target_os = "linux")]
fn read_linux_node_memory_bytes(path: impl AsRef<std::path::Path>) -> Option<u64> {
    let content = std::fs::read_to_string(path).ok()?;
    content.lines().find_map(|line| {
        if !line.contains("MemTotal:") {
            return None;
        }
        let kb = line
            .split_whitespace()
            .find_map(|part| part.parse::<u64>().ok())?;
        Some(kb * 1024)
    })
}

#[cfg(target_os = "linux")]
fn detect_linux_sockets(nodes: &[NumaNode], total_logical_cpus: usize) -> Vec<SocketInfo> {
    use std::collections::{BTreeMap, BTreeSet};

    let all_cpus = nodes
        .iter()
        .flat_map(|node| node.logical_cpus.iter().copied())
        .collect::<BTreeSet<_>>();
    let cpus = if all_cpus.is_empty() {
        (0..total_logical_cpus).collect::<Vec<_>>()
    } else {
        all_cpus.into_iter().collect::<Vec<_>>()
    };

    let mut by_socket: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for cpu in cpus {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id");
        let socket_id = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0);
        by_socket.entry(socket_id).or_default().push(cpu);
    }

    if by_socket.is_empty() {
        by_socket.insert(0, (0..total_logical_cpus.max(1)).collect());
    }

    by_socket
        .into_iter()
        .map(|(id, mut logical_cpus)| {
            logical_cpus.sort_unstable();
            let numa_node_ids = nodes
                .iter()
                .filter(|node| {
                    node.logical_cpus
                        .iter()
                        .any(|cpu| logical_cpus.binary_search(cpu).is_ok())
                })
                .map(|node| node.id)
                .collect();
            SocketInfo {
                id,
                logical_cpus,
                numa_node_ids,
            }
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn detect_macos() -> Option<MachineTopology> {
    let rt = crate::runtime::runtime();
    let cpus = (0..rt.total_cores.max(1)).collect::<Vec<_>>();
    Some(MachineTopology {
        total_logical_cpus: rt.total_cores.max(1),
        numa_nodes: vec![NumaNode {
            id: 0,
            logical_cpus: cpus.clone(),
            memory_bytes: None,
        }],
        sockets: vec![SocketInfo {
            id: 0,
            logical_cpus: cpus,
            numa_node_ids: vec![0],
        }],
        notes: vec![format!(
            "macOS treated as UMA; runtime reports {} performance core(s)",
            rt.perf_cores
        )],
    })
}

fn detect_generic() -> MachineTopology {
    let total = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let cpus = (0..total).collect::<Vec<_>>();
    MachineTopology {
        total_logical_cpus: total,
        numa_nodes: vec![NumaNode {
            id: 0,
            logical_cpus: cpus.clone(),
            memory_bytes: None,
        }],
        sockets: vec![SocketInfo {
            id: 0,
            logical_cpus: cpus,
            numa_node_ids: vec![0],
        }],
        notes: vec!["generic UMA fallback".to_string()],
    }
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    format!("{:.1}GiB", bytes as f64 / GIB)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_cpu_lists() {
        assert_eq!(parse_cpu_list("0-3,8,10-11"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpu_list(" 2,1,1,4-5 "), vec![1, 2, 4, 5]);
        assert_eq!(parse_cpu_list("bogus,7-5,9"), vec![9]);
    }

    #[test]
    fn formats_cpu_lists_as_ranges() {
        assert_eq!(format_cpu_list(&[]), "none");
        assert_eq!(format_cpu_list(&[0]), "0");
        assert_eq!(format_cpu_list(&[0, 1, 2, 5, 7, 8]), "0-2,5,7-8");
    }

    #[test]
    fn placement_round_robins_over_nodes() {
        let topology = MachineTopology {
            total_logical_cpus: 8,
            numa_nodes: vec![
                NumaNode {
                    id: 0,
                    logical_cpus: vec![0, 1, 2, 3],
                    memory_bytes: None,
                },
                NumaNode {
                    id: 1,
                    logical_cpus: vec![4, 5, 6, 7],
                    memory_bytes: None,
                },
            ],
            sockets: vec![
                SocketInfo {
                    id: 0,
                    logical_cpus: vec![0, 1, 2, 3],
                    numa_node_ids: vec![0],
                },
                SocketInfo {
                    id: 1,
                    logical_cpus: vec![4, 5, 6, 7],
                    numa_node_ids: vec![1],
                },
            ],
            notes: Vec::new(),
        };

        let plan = topology.placement_plan(3);
        assert_eq!(plan.workers[0].numa_node_id, 0);
        assert_eq!(plan.workers[1].numa_node_id, 1);
        assert_eq!(plan.workers[2].numa_node_id, 0);
        assert_eq!(plan.workers[1].socket_id, 1);
    }

    #[test]
    fn placement_handles_zero_workers() {
        let topology = detect_generic();
        let plan = topology.placement_plan(0);
        assert!(plan.workers.is_empty());
        assert!(!plan.notes.is_empty());
    }

    #[test]
    fn affinity_status_formats_cpu_ranges() {
        let status = AffinityStatus::Applied {
            cpus: vec![0, 1, 2, 6],
        };
        assert_eq!(status.to_string(), "applied cpus=0-2,6");
        assert!(status.is_applied());
    }
}
