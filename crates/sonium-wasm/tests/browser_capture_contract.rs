use sonium_roomsim::{
    BROWSER_CAPTURE_CONTRACT_SCHEMA_V1, BrowserCaptureContract, BrowserCaptureScreenshot,
    BrowserCaptureViewport,
};

fn valid_contract() -> BrowserCaptureContract {
    BrowserCaptureContract {
        schema: BROWSER_CAPTURE_CONTRACT_SCHEMA_V1.into(),
        entrypoint: "index.html".into(),
        viewport: BrowserCaptureViewport {
            width: 1920,
            height: 1080,
        },
        device_scale: 1,
        ready_selector: "#integrity[data-status=\"ok\"]".into(),
        failure_selector: "#integrity[data-status=\"failed\"]".into(),
        required_figures: vec!["room_plan.svg".into(), "pressure_slice.png".into()],
        screenshots: vec![
            BrowserCaptureScreenshot {
                name: "completed".into(),
                wait_for: Some("#integrity[data-status=\"ok\"]".into()),
                selector: None,
            },
            BrowserCaptureScreenshot {
                name: "figures".into(),
                wait_for: None,
                selector: Some("#figures".into()),
            },
        ],
        note: "Capture only after integrity is ready.".into(),
    }
}

#[test]
fn browser_capture_contract_round_trips_and_validates() {
    let contract = valid_contract();
    let json = contract.to_json().expect("valid contract should serialize");
    let restored = BrowserCaptureContract::from_json(
        std::str::from_utf8(&json).expect("contract JSON should be UTF-8"),
    )
    .expect("serialized contract should validate");
    assert_eq!(restored, contract);
}

#[test]
fn browser_capture_contract_rejects_unknown_fields() {
    let contract = valid_contract();
    let mut document: serde_json::Value =
        serde_json::from_slice(&contract.to_json().unwrap()).unwrap();
    document["unexpected"] = serde_json::Value::Bool(true);
    let json = serde_json::to_string(&document).unwrap();
    let error = BrowserCaptureContract::from_json(&json).unwrap_err();
    assert!(error.contains("invalid browser capture contract JSON"));
}

#[test]
fn browser_capture_contract_rejects_unsafe_or_duplicate_figures() {
    let mut unsafe_contract = valid_contract();
    unsafe_contract.required_figures[0] = "../outside.svg".into();
    assert!(unsafe_contract.validate().unwrap_err().contains("unsafe"));

    let mut duplicate_contract = valid_contract();
    duplicate_contract.required_figures[1] = duplicate_contract.required_figures[0].clone();
    assert!(
        duplicate_contract
            .validate()
            .unwrap_err()
            .contains("duplicate")
    );
}

#[test]
fn browser_capture_contract_rejects_manifest_figure_drift() {
    let contract = valid_contract();
    let error = contract
        .validate_against_figure_names(["room_plan.svg"])
        .unwrap_err();
    assert!(error.contains("missing gallery figure"));

    contract
        .validate_against_figure_names(["room_plan.svg", "pressure_slice.png", "extra.svg"])
        .expect("extra manifest figures are allowed");
}

#[test]
fn browser_capture_contract_requires_exactly_one_screenshot_trigger() {
    let mut both = valid_contract();
    both.screenshots[0].selector = Some("#figures".into());
    assert!(both.validate().unwrap_err().contains("exactly one"));

    let mut neither = valid_contract();
    neither.screenshots[0].wait_for = None;
    assert!(neither.validate().unwrap_err().contains("exactly one"));
}
