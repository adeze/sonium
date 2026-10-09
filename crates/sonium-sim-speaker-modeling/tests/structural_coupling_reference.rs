use num_complex::Complex64;
use sonium_fem::physics::acoustics::electroacoustics::{
    CoupledIterativeConfig, CoupledPreconditioner,
};
use sonium_sim_speaker_modeling::{
    AxisymmetricLinearConfig, AxisymmetricLinearModel, ResolvedAcousticConfig,
    ResolvedAcousticModel, ResolvedCouplingConfig, ResolvedLoadCoupler,
};

fn coarse_linear_model() -> AxisymmetricLinearModel {
    let mut model = AxisymmetricLinearModel::build(AxisymmetricLinearConfig {
        magnetic_radial_cells: 8,
        magnetic_axial_cells: 8,
        structural_radial_cells: 6,
        structural_axial_cells: 1,
        acoustic_axial_cells: 4,
        iterative_solver: CoupledIterativeConfig {
            preconditioner: CoupledPreconditioner::Monolithic,
            ..CoupledIterativeConfig::default()
        },
        ..AxisymmetricLinearConfig::default()
    })
    .expect("coarse structural model should build");
    model
        .install_interface_load()
        .expect("structural-acoustic interface load should install");
    model
}

fn coarse_acoustic_model() -> ResolvedAcousticModel {
    ResolvedAcousticModel::build(ResolvedAcousticConfig {
        cells_per_physical_interval: 1,
        pml_cells: 1,
        pml_sigma_max: 2.0,
        ..ResolvedAcousticConfig::default()
    })
    .expect("coarse resolved acoustic model should build")
}

#[test]
fn public_structural_coupling_closes_fixed_point_and_monolithic_agreement() {
    let linear = coarse_linear_model();
    let acoustic = coarse_acoustic_model();
    let mut coupler =
        ResolvedLoadCoupler::new(&linear, &acoustic, ResolvedCouplingConfig::default())
            .expect("resolved load coupler should build");

    let point = coupler
        .solve_frequency(100.0, Complex64::new(2.83, 0.0))
        .expect("bounded structural-acoustic coupling should converge");

    assert!(point.converged);
    assert!(point.iterations <= ResolvedCouplingConfig::default().max_iterations);
    assert!(point.maximum_contraction_ratio < 1.0);
    assert!(
        point.monolithic_agreement_relative_error
            <= ResolvedCouplingConfig::default().monolithic_agreement_tolerance
    );
    assert!(point.acoustic_load_absorbed_power_w() >= 0.0);

    let load_defect = (point.total_acoustic_load_n_s_per_m() - point.z_resolved_n_s_per_m).norm()
        / point.z_resolved_n_s_per_m.norm().max(f64::MIN_POSITIVE);
    assert!(
        load_defect < 1.0e-6,
        "total/resolved load defect = {load_defect}"
    );
    assert!(
        point
            .velocity_change_history
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn public_structural_coupling_rejects_noncontractive_policy_before_solving() {
    let config = ResolvedCouplingConfig {
        max_contraction_ratio: 1.0,
        ..ResolvedCouplingConfig::default()
    };
    assert!(config.validate().is_err());
}
