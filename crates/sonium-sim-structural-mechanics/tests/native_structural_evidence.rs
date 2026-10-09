use sonium_sim_structural_mechanics::{
    CrossArchitectureTolerances, NativeStructuralReferencePacket,
    compare_structural_architecture_packets, run_native_structural_reference_packet,
};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn native_structural_packet_is_deterministic_and_machine_readable() {
    let first = run_native_structural_reference_packet().expect("native structural packet");
    let replay = run_native_structural_reference_packet().expect("native structural replay");
    assert_eq!(first.metadata.target_arch, std::env::consts::ARCH);
    assert_eq!(first.metadata.threads, 1);
    assert!(!first.metadata.compiler.is_empty());
    assert!(!first.metadata.cpu_model.is_empty());
    assert!(!first.metadata.scalar_dispatch.is_empty());
    assert!(!first.metadata.simd_dispatch.is_empty());
    assert!(!first.metadata.vendor_dispatch.is_empty());
    assert_eq!(
        first.canonical_payload_sha256,
        replay.canonical_payload_sha256
    );

    let comparison = compare_structural_architecture_packets(
        &first,
        &replay,
        CrossArchitectureTolerances::default(),
    )
    .expect("same-host deterministic comparison");
    assert!(comparison.accepted, "{comparison:#?}");
    assert!(comparison.topology_equal);
    assert!(comparison.canonical_payload_sha256_equal);
    if let Ok(path) = std::env::var("SONIUM_STRUCTURAL_EVIDENCE_OUTPUT") {
        let json = serde_json::to_vec_pretty(&first).expect("serialize native evidence artifact");
        std::fs::write(path, json).expect("write native evidence artifact");
    }
    println!(
        "NATIVE_STRUCTURAL_EVIDENCE {}",
        serde_json::to_string(&first).expect("serialize native structural evidence")
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "requires an accepted native-aarch64 packet from the architecture evidence job"]
fn native_x86_compares_with_supplied_accepted_aarch64_packet() {
    let path = std::env::var("SONIUM_AARCH64_STRUCTURAL_REFERENCE")
        .expect("SONIUM_AARCH64_STRUCTURAL_REFERENCE must name the accepted aarch64 packet");
    let bytes = std::fs::read(&path).expect("read accepted aarch64 structural packet");
    let aarch64: NativeStructuralReferencePacket =
        serde_json::from_slice(&bytes).expect("decode accepted aarch64 structural packet");
    assert_eq!(aarch64.metadata.target_arch, "aarch64");
    let x86 = run_native_structural_reference_packet().expect("native x86 structural packet");
    let comparison = compare_structural_architecture_packets(
        &aarch64,
        &x86,
        CrossArchitectureTolerances::default(),
    )
    .expect("cross-architecture comparison");
    assert!(comparison.accepted, "{comparison:#?}");
    println!(
        "AARCH64_X86_STRUCTURAL_COMPARISON {}",
        serde_json::to_string(&comparison).expect("serialize cross-architecture comparison")
    );
}
