use sonium_solvers_time_integration::{NEWMARK_CHECKPOINT_FORMAT_VERSION, NewmarkResult};

fn state(with_history: bool) -> NewmarkResult<f64> {
    NewmarkResult {
        displacement: vec![1.0, -2.0],
        velocity: vec![0.25, 0.5],
        acceleration: vec![-0.75, 1.5],
        displacement_history: with_history.then(|| vec![vec![0.0, 0.0], vec![1.0, -2.0]]),
        velocity_history: with_history.then(|| vec![vec![0.0, 0.0], vec![0.25, 0.5]]),
        acceleration_history: with_history.then(|| vec![vec![-1.0, 1.0], vec![-0.75, 1.5]]),
        steps_completed: 1,
        final_time: 0.1,
    }
}

#[test]
fn native_checkpoint_round_trips_state_and_history() {
    let original = state(true);
    let encoded = original.to_checkpoint_bytes().expect("encode");
    let restored = NewmarkResult::from_checkpoint_bytes(&encoded).expect("decode");
    assert_eq!(restored.displacement, original.displacement);
    assert_eq!(restored.velocity, original.velocity);
    assert_eq!(restored.acceleration, original.acceleration);
    assert_eq!(restored.displacement_history, original.displacement_history);
    assert_eq!(restored.velocity_history, original.velocity_history);
    assert_eq!(restored.acceleration_history, original.acceleration_history);
    assert_eq!(restored.steps_completed, original.steps_completed);
    assert_eq!(restored.final_time, original.final_time);
}

#[test]
fn native_checkpoint_round_trips_absent_history() {
    let original = state(false);
    let encoded = original.to_checkpoint_bytes().expect("encode");
    let restored = NewmarkResult::from_checkpoint_bytes(&encoded).expect("decode");
    assert_eq!(restored.displacement_history, None);
    assert_eq!(restored.velocity_history, None);
    assert_eq!(restored.acceleration_history, None);
}

#[test]
fn native_checkpoint_rejects_corruption_and_trailing_bytes() {
    let encoded = state(true).to_checkpoint_bytes().expect("encode");

    let mut truncated = encoded.clone();
    truncated.truncate(truncated.len() - 1);
    assert!(NewmarkResult::from_checkpoint_bytes(&truncated).is_err());

    let mut trailing = encoded.clone();
    trailing.push(0xA5);
    let error = NewmarkResult::from_checkpoint_bytes(&trailing).expect_err("trailing bytes");
    assert!(error.to_string().contains("trailing bytes"));

    let mut bad_magic = encoded.clone();
    bad_magic[0] ^= 0xFF;
    assert!(
        NewmarkResult::from_checkpoint_bytes(&bad_magic)
            .expect_err("bad magic")
            .to_string()
            .contains("invalid magic")
    );

    let mut bad_version = encoded;
    bad_version[8..10].copy_from_slice(&(NEWMARK_CHECKPOINT_FORMAT_VERSION + 1).to_le_bytes());
    assert!(
        NewmarkResult::from_checkpoint_bytes(&bad_version)
            .expect_err("bad version")
            .to_string()
            .contains("unsupported checkpoint version")
    );
}

#[test]
fn native_checkpoint_rejects_invalid_source_state() {
    let mut mismatched = state(false);
    mismatched.velocity.pop();
    assert!(
        mismatched
            .to_checkpoint_bytes()
            .expect_err("mismatched vectors")
            .to_string()
            .contains("dimensions differ")
    );

    let mut non_finite = state(false);
    non_finite.final_time = f64::NAN;
    assert!(
        non_finite
            .to_checkpoint_bytes()
            .expect_err("non-finite state")
            .to_string()
            .contains("non-finite")
    );

    let mut malformed_history = state(true);
    malformed_history.displacement_history.as_mut().unwrap()[0].pop();
    assert!(
        malformed_history
            .to_checkpoint_bytes()
            .expect_err("malformed history")
            .to_string()
            .contains("history shape")
    );
}

#[test]
fn native_checkpoint_rejects_incomplete_histories() {
    let mut truncated = state(true);
    truncated
        .displacement_history
        .as_mut()
        .expect("history present")
        .pop();
    let error = truncated
        .to_checkpoint_bytes()
        .expect_err("truncated history must be rejected");
    assert!(error.to_string().contains("history lengths"));

    let mut partial = state(true);
    partial.velocity_history = None;
    let error = partial
        .to_checkpoint_bytes()
        .expect_err("partial history must be rejected");
    assert!(error.to_string().contains("history presence"));
}
