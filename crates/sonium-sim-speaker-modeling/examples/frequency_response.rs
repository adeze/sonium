use sonium_sim_speaker_modeling::SpeakerDriverStudy;

fn main() {
    let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
    println!(
        "{:>9} {:>11} {:>11} {:>11} {:>13}",
        "freq(Hz)", "Re(Z)", "Im(Z)", "|I|(A)", "|x|(um)"
    );

    for response in study.configured_small_signal_response_sweep() {
        println!(
            "{:>9.1} {:>11.3} {:>11.3} {:>11.4} {:>13.3}",
            response.frequency_hz,
            response.impedance.re,
            response.impedance.im,
            response.current.norm(),
            response.diaphragm_displacement.norm() * 1.0e6
        );
    }
}
