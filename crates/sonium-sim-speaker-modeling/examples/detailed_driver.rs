use sonium_sim_speaker_modeling::SpeakerDriverStudy;

fn main() {
    let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
    let report = study.configured_detailed_example_report();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
