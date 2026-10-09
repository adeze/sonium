use std::path::Path;

use sonium_sim_structural_mechanics::{
    ExternalBenchmarkDefinition, ExternalBenchmarkStatus, ExternalComparisonStatus,
    ExternalDisplacementObservation, ExternalDisplacementSource, ExternalExecutionStage,
    ExternalInputFile, ExternalRunOptions, ExternalSolverKind, StructuralMechanicsError,
    compare_external_observation, external_benchmark_status,
    parse_external_displacement_for_parity, render_external_decks, run_external_benchmark,
};

const SOLVERS: [ExternalSolverKind; 3] = [
    ExternalSolverKind::Calculix,
    ExternalSolverKind::CodeAster,
    ExternalSolverKind::Fenicsx,
];

#[test]
fn public_external_decks_round_trip_and_write_safely_without_execution() {
    let definition = ExternalBenchmarkDefinition::default();
    definition.validate().expect("default benchmark is valid");

    let encoded = serde_json::to_string(&definition).expect("definition should serialize");
    let decoded: ExternalBenchmarkDefinition =
        serde_json::from_str(&encoded).expect("definition should deserialize");
    assert_eq!(decoded, definition);

    let decks = render_external_decks(&definition).expect("all adapters should render");
    assert_eq!(decks.len(), SOLVERS.len());
    for (deck, solver) in decks.iter().zip(SOLVERS) {
        assert_eq!(deck.solver, solver);
        assert_eq!(deck.schema_version, 1);
        assert_eq!(deck.benchmark_name, definition.name);
        assert!(!deck.primary_file.is_empty());
        assert!(!deck.files.is_empty());
        assert!(deck.files.iter().all(|file| !file.contents.is_empty()));

        let directory = tempfile::tempdir().expect("temporary deck directory");
        let written = deck
            .write_to(directory.path())
            .expect("deck should write below caller directory");
        assert_eq!(written.len(), deck.files.len());
        assert!(
            written
                .iter()
                .all(|path| path.starts_with(directory.path()))
        );
        assert!(written.iter().all(|path| path.is_file()));
    }

    let mut traversal =
        render_external_decks(&definition).expect("all adapters should render")[0].clone();
    traversal.files[0] = ExternalInputFile {
        relative_path: "../escape.inp".to_string(),
        contents: "must not write".to_string(),
    };
    let directory = tempfile::tempdir().expect("temporary traversal directory");
    assert!(matches!(
        traversal.write_to(directory.path()),
        Err(StructuralMechanicsError::ExternalBenchmarkInvalid {
            field: "deck.relative_path",
            ..
        })
    ));
    assert!(!Path::new(&directory.path().join("..\u{2f}escape.inp")).exists());
}

#[test]
fn public_external_parity_requires_explicit_consistent_markers() {
    let definition = ExternalBenchmarkDefinition::default();
    let expected = definition.expected_displacement_m();

    for solver in SOLVERS {
        let output = format!("solver output displacement_m={expected:.17e}");
        let measured = parse_external_displacement_for_parity(solver, &output)
            .expect("explicit marker should be sufficient evidence");
        let comparison = compare_external_observation(
            solver,
            &definition,
            ExternalDisplacementObservation {
                displacement_m: measured,
                source: ExternalDisplacementSource::CommonMarker,
            },
            1.0e-12,
        )
        .expect("marked displacement should compare");
        assert!(comparison.evidence_sufficient);
        assert!(comparison.parity_passed());
    }

    assert!(
        parse_external_displacement_for_parity(
            ExternalSolverKind::Calculix,
            &format!("1 {expected:.17e} 0"),
        )
        .is_err()
    );
    assert!(
        parse_external_displacement_for_parity(
            ExternalSolverKind::Fenicsx,
            &format!("displacement_m={expected:.17e} U1={:.17e}", expected * 1.1),
        )
        .is_err()
    );
    assert!(
        parse_external_displacement_for_parity(
            ExternalSolverKind::Fenicsx,
            &format!("nondisplacement_m={expected:.17e}"),
        )
        .is_err()
    );
}

#[test]
fn public_external_status_is_fail_closed_until_a_solver_run_completes() {
    let status: ExternalBenchmarkStatus = external_benchmark_status();
    assert_eq!(status.schema_version, 1);
    assert_eq!(status.status, ExternalComparisonStatus::NotRun);
    assert_eq!(status.adapters_registered.len(), SOLVERS.len());
    assert!(!status.parity_passed());
    assert!(!status.evidence_sufficient);
}

#[cfg(unix)]
#[test]
fn public_external_fixture_run_records_provenance_without_a_real_solver() {
    use std::os::unix::fs::PermissionsExt;

    let definition = ExternalBenchmarkDefinition::default();
    let directory = tempfile::tempdir().expect("temporary fixture directory");
    let executable = directory.path().join("fixture-solver.sh");
    std::fs::write(
        &executable,
        format!(
            "#!/bin/sh\nprintf 'displacement_m={:.17e}\\n'\n",
            definition.expected_displacement_m()
        ),
    )
    .expect("fixture executable should be written");
    let mut permissions = std::fs::metadata(&executable)
        .expect("fixture metadata should be available")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&executable, permissions)
        .expect("fixture executable should be executable");

    for solver in SOLVERS {
        let workdir = directory.path().join(solver.to_string());
        let mut options = ExternalRunOptions::new(&workdir);
        options.executable = Some(executable.clone());
        options.tolerance = 1.0e-12;
        let result = run_external_benchmark(&definition, solver, &options)
            .expect("caller-supplied fixture should complete");

        assert_eq!(result.exit_code, Some(0));
        assert_eq!(result.executable, executable);
        assert!(result.parity_passed());
        assert!(result.provenance.parity_passed());
        assert_eq!(result.provenance.completed_stages.len(), 6);
        assert_eq!(
            result.provenance.observation_source,
            Some(ExternalDisplacementSource::CommonMarker)
        );
        for stage in [
            ExternalExecutionStage::DeckRendered,
            ExternalExecutionStage::ExecutableResolved,
            ExternalExecutionStage::DeckWritten,
            ExternalExecutionStage::ProcessCompleted,
            ExternalExecutionStage::OutputObserved,
            ExternalExecutionStage::NumericCompared,
        ] {
            assert!(result.provenance.has_stage(stage));
        }
    }
}
