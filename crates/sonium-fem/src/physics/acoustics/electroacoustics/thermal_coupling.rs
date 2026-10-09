//! Voice coil thermal model: lumped thermal network
//!
//! Models the thermal behavior of a loudspeaker motor as a lumped
//! resistance-capacitance network. Heat is generated in the voice coil
//! (I²R dissipation) and flows through conduction/convection paths to
//! the ambient.
//!
//! # Thermal network
//!
//! ```text
//!   P_in = I²R(T)
//!       ↓
//!   [Coil]  ──R_cf──  [Former]  ──R_fm──  [Magnet]  ──R_ma──  [Ambient]
//!    C_c                C_f                 C_m
//! ```
//!
//! - Coil: voice coil winding (heat source)
//! - Former: voice coil former / bobbin
//! - Magnet: magnet assembly + pole piece (large thermal mass)
//! - Ambient: fixed temperature boundary
//!
//! # Temperature-dependent resistance
//!
//! ```text
//! R_e(T) = R_e0 · (1 + α · (T - T_ref))
//! ```
//!
//! where α ≈ 0.00393 /K for copper.

/// Thermal resistance coefficient for copper wire (1/K).
pub const ALPHA_COPPER: f64 = 0.00393;

/// Thermal resistance coefficient for aluminum wire (1/K).
pub const ALPHA_ALUMINUM: f64 = 0.00403;

/// A node in the lumped thermal network.
#[derive(Debug, Clone)]
pub struct ThermalNode {
    /// Thermal capacitance C_th = mass × specific_heat (J/K).
    pub capacitance: f64,
    /// Current temperature (K).
    pub temperature: f64,
}

/// A thermal resistance link between two nodes.
#[derive(Debug, Clone)]
pub struct ThermalLink {
    /// Source node index.
    pub from: usize,
    /// Destination node index.
    pub to: usize,
    /// Thermal resistance R_th (K/W).
    pub resistance: f64,
}

/// Lumped thermal model for a voice coil motor.
///
/// Tracks temperature at each node and computes the temperature-dependent
/// voice coil resistance for coupling back to the electrical domain.
#[derive(Debug, Clone)]
pub struct VoiceCoilThermalModel {
    /// Thermal nodes.
    pub nodes: Vec<ThermalNode>,
    /// Thermal links between nodes.
    pub links: Vec<ThermalLink>,
    /// Index of the voice coil node (where I²R heat is injected).
    pub coil_node: usize,
    /// Index of the ambient node (fixed temperature boundary).
    pub ambient_node: usize,
    /// Ambient temperature (K).
    pub t_ambient: f64,
    /// Voice coil DC resistance at reference temperature (Ω).
    pub re_ref: f64,
    /// Reference temperature for re_ref (K).
    pub t_ref: f64,
    /// Temperature coefficient of resistance (1/K).
    pub alpha: f64,
}

impl VoiceCoilThermalModel {
    /// Create a standard 3-node motor thermal model.
    ///
    /// Nodes: 0=coil, 1=former, 2=magnet, 3=ambient (fixed).
    ///
    /// # Arguments
    /// - `re_ref`: DC resistance at `t_ref` (Ω)
    /// - `t_ambient`: ambient temperature (K), typically 293.15 (20°C)
    /// - `resistances`: `[R_coil_former, R_former_magnet, R_magnet_ambient]` (K/W)
    /// - `capacitances`: `[C_coil, C_former, C_magnet]` (J/K)
    pub fn standard(
        re_ref: f64,
        t_ambient: f64,
        resistances: [f64; 3],
        capacitances: [f64; 3],
    ) -> Self {
        let [r_coil_former, r_former_magnet, r_magnet_ambient] = resistances;
        let [c_coil, c_former, c_magnet] = capacitances;
        Self {
            nodes: vec![
                ThermalNode {
                    capacitance: c_coil,
                    temperature: t_ambient,
                },
                ThermalNode {
                    capacitance: c_former,
                    temperature: t_ambient,
                },
                ThermalNode {
                    capacitance: c_magnet,
                    temperature: t_ambient,
                },
                ThermalNode {
                    capacitance: 0.0,
                    temperature: t_ambient,
                }, // ambient (infinite capacity)
            ],
            links: vec![
                ThermalLink {
                    from: 0,
                    to: 1,
                    resistance: r_coil_former,
                },
                ThermalLink {
                    from: 1,
                    to: 2,
                    resistance: r_former_magnet,
                },
                ThermalLink {
                    from: 2,
                    to: 3,
                    resistance: r_magnet_ambient,
                },
            ],
            coil_node: 0,
            ambient_node: 3,
            t_ambient,
            re_ref,
            t_ref: t_ambient,
            alpha: ALPHA_COPPER,
        }
    }

    /// Create a simple 1-node model (coil → ambient via single resistance).
    ///
    /// Useful for quick estimates or when detailed thermal data is unavailable.
    pub fn simple(re_ref: f64, t_ambient: f64, r_total: f64, c_coil: f64) -> Self {
        Self {
            nodes: vec![
                ThermalNode {
                    capacitance: c_coil,
                    temperature: t_ambient,
                },
                ThermalNode {
                    capacitance: 0.0,
                    temperature: t_ambient,
                }, // ambient
            ],
            links: vec![ThermalLink {
                from: 0,
                to: 1,
                resistance: r_total,
            }],
            coil_node: 0,
            ambient_node: 1,
            t_ambient,
            re_ref,
            t_ref: t_ambient,
            alpha: ALPHA_COPPER,
        }
    }

    /// Voice coil resistance at the current coil temperature.
    pub fn resistance(&self) -> f64 {
        let t_coil = self.nodes[self.coil_node].temperature;
        self.re_ref * (1.0 + self.alpha * (t_coil - self.t_ref))
    }

    /// Voice coil resistance at an arbitrary temperature.
    pub fn resistance_at(&self, temperature: f64) -> f64 {
        self.re_ref * (1.0 + self.alpha * (temperature - self.t_ref))
    }

    /// Current voice coil temperature (K).
    pub fn coil_temperature(&self) -> f64 {
        self.nodes[self.coil_node].temperature
    }

    /// Reset all nodes to ambient temperature.
    pub fn reset(&mut self) {
        for node in &mut self.nodes {
            node.temperature = self.t_ambient;
        }
    }

    /// Advance the thermal network by one time step using forward Euler.
    ///
    /// `power_input`: electrical power dissipated in the voice coil (W) = I²·R(T).
    /// `dt`: time step (s).
    pub fn step(&mut self, power_input: f64, dt: f64) {
        let n = self.nodes.len();
        let mut dq = vec![0.0; n]; // net heat flow into each node (W)

        // Heat input to coil node
        dq[self.coil_node] += power_input;

        // Heat flow through links: Q = (T_from - T_to) / R_th
        for link in &self.links {
            let t_from = self.nodes[link.from].temperature;
            let t_to = self.nodes[link.to].temperature;
            let q = (t_from - t_to) / link.resistance;
            dq[link.from] -= q;
            dq[link.to] += q;
        }

        // Update temperatures (skip ambient node — fixed temperature)
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if i == self.ambient_node {
                continue;
            }
            if node.capacitance > 0.0 {
                node.temperature += dt * dq[i] / node.capacitance;
            }
        }
    }

    /// Compute steady-state temperatures for a given constant power input.
    ///
    /// Solves the linear thermal resistance network analytically.
    /// For a serial chain (standard model): T_coil = T_amb + P·(R_cf + R_fm + R_ma).
    pub fn steady_state(&mut self, power_input: f64) {
        // For a serial chain, the steady-state temperature rise at each node
        // is the sum of resistances from that node to ambient, times the power.
        // This works for any tree topology (no loops).

        // Reset to ambient
        self.reset();

        // Build the resistance-to-ambient for each node via BFS from ambient
        let n = self.nodes.len();
        let mut r_to_ambient = vec![f64::INFINITY; n];
        r_to_ambient[self.ambient_node] = 0.0;

        // For a serial chain, iterate from ambient inward
        let mut changed = true;
        while changed {
            changed = false;
            for link in &self.links {
                // Check both directions
                if r_to_ambient[link.to].is_finite() && r_to_ambient[link.from].is_infinite() {
                    r_to_ambient[link.from] = r_to_ambient[link.to] + link.resistance;
                    changed = true;
                }
                if r_to_ambient[link.from].is_finite() && r_to_ambient[link.to].is_infinite() {
                    r_to_ambient[link.to] = r_to_ambient[link.from] + link.resistance;
                    changed = true;
                }
            }
        }

        // For a single heat source at the coil node in a serial chain:
        // All links carry the same power P. Each node's temperature rise
        // equals P × (total resistance from that node to ambient).
        //
        // For the general case with a single source, this simplifies to:
        // T_i = T_amb + P × R_i_to_ambient  (for nodes between source and ambient)
        // But nodes "behind" the source (further from ambient than the source) don't see the heat.
        //
        // For the standard serial chain 0→1→2→3(ambient) with source at 0:
        // All of P flows through every link, so T_i = T_amb + P × R(i→ambient)
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if i == self.ambient_node {
                continue;
            }
            if r_to_ambient[i].is_finite() {
                // Only nodes between the coil and ambient get heated
                // In a serial chain with source at coil, all nodes are on the path
                let r_effective = r_to_ambient[i].min(r_to_ambient[self.coil_node]);
                node.temperature = self.t_ambient + power_input * r_effective;
            }
        }
        // Coil gets the full temperature rise
        self.nodes[self.coil_node].temperature =
            self.t_ambient + power_input * r_to_ambient[self.coil_node];
    }

    /// Total thermal resistance from coil to ambient (K/W).
    pub fn total_resistance(&self) -> f64 {
        self.links.iter().map(|l| l.resistance).sum()
    }

    /// Thermal time constant of the coil node (s).
    ///
    /// τ = R_total × C_coil (for the simple model).
    pub fn coil_time_constant(&self) -> f64 {
        self.total_resistance() * self.nodes[self.coil_node].capacitance
    }

    /// Compute steady-state coil temperature with self-consistent Re(T).
    ///
    /// Iterates: P = I_rms² · Re(T) → T = T_amb + P · R_total → update Re(T).
    /// Converges in a few iterations for typical parameters.
    ///
    /// Returns (coil_temperature, final_resistance, actual_power).
    pub fn steady_state_self_consistent(
        &mut self,
        voltage_rms: f64,
        z_mech_at_coil_mag: f64,
        bl: f64,
        le: f64,
        frequency: f64,
    ) -> (f64, f64, f64) {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let r_total = self.total_resistance();

        let mut re = self.re_ref;
        for _ in 0..20 {
            // Total electrical impedance
            let z_mot_re = bl * bl * z_mech_at_coil_mag / (z_mech_at_coil_mag * z_mech_at_coil_mag);
            // Simplified: assume Z_mech is real at the test frequency
            let z_total_sq = (re + z_mot_re) * (re + z_mot_re) + (omega * le) * (omega * le);
            let i_rms_sq = voltage_rms * voltage_rms / z_total_sq;
            let power = i_rms_sq * re;
            let t_coil = self.t_ambient + power * r_total;
            let re_new = self.resistance_at(t_coil);
            if (re_new - re).abs() < 1e-6 * re {
                break;
            }
            re = re_new;
        }

        let z_mot_re = bl * bl * z_mech_at_coil_mag / (z_mech_at_coil_mag * z_mech_at_coil_mag);
        let z_total_sq = (re + z_mot_re) * (re + z_mot_re) + (omega * le) * (omega * le);
        let i_rms_sq = voltage_rms * voltage_rms / z_total_sq;
        let power = i_rms_sq * re;
        let t_coil = self.t_ambient + power * r_total;

        self.nodes[self.coil_node].temperature = t_coil;

        (t_coil, re, power)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const T_AMB: f64 = 293.15; // 20°C

    fn simple_model() -> VoiceCoilThermalModel {
        // Simple: Re=6Ω, R_th=10 K/W, C=5 J/K → τ=50s
        VoiceCoilThermalModel::simple(6.0, T_AMB, 10.0, 5.0)
    }

    fn standard_model() -> VoiceCoilThermalModel {
        // Standard 3-node: R_cf=5, R_fm=3, R_ma=2 K/W (total=10 K/W)
        VoiceCoilThermalModel::standard(
            6.0,
            T_AMB,
            [5.0, 3.0, 2.0],  // thermal resistances
            [2.0, 5.0, 50.0], // thermal capacitances
        )
    }

    #[test]
    fn test_initial_resistance() {
        let model = simple_model();
        assert!((model.resistance() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_resistance_at_temperature() {
        let model = simple_model();
        // At T_amb + 100K: Re = 6 × (1 + 0.00393 × 100) = 6 × 1.393 = 8.358
        let re = model.resistance_at(T_AMB + 100.0);
        let expected = 6.0 * (1.0 + ALPHA_COPPER * 100.0);
        assert!(
            (re - expected).abs() < 1e-10,
            "Re({:.0}K) = {re:.3}Ω, expected {expected:.3}Ω",
            T_AMB + 100.0
        );
    }

    #[test]
    fn test_steady_state_simple() {
        let mut model = simple_model();
        let power = 10.0; // 10 W
        model.steady_state(power);

        // T_coil = T_amb + P × R_total = 293.15 + 10 × 10 = 393.15 K
        let expected = T_AMB + power * 10.0;
        assert!(
            (model.coil_temperature() - expected).abs() < 0.01,
            "T_coil = {:.2} K, expected {expected:.2} K",
            model.coil_temperature()
        );
    }

    #[test]
    fn test_steady_state_standard_chain() {
        let mut model = standard_model();
        let power = 5.0;
        model.steady_state(power);

        // Total R = 5+3+2 = 10 K/W
        // T_coil = T_amb + 5 × 10 = T_amb + 50
        let expected_coil = T_AMB + power * 10.0;
        assert!(
            (model.coil_temperature() - expected_coil).abs() < 0.01,
            "T_coil = {:.2} K, expected {expected_coil:.2} K",
            model.coil_temperature()
        );

        // Former: R_former_to_ambient = 3+2 = 5 K/W
        let expected_former = T_AMB + power * 5.0;
        assert!(
            (model.nodes[1].temperature - expected_former).abs() < 0.01,
            "T_former = {:.2} K, expected {expected_former:.2} K",
            model.nodes[1].temperature
        );

        // Magnet: R_magnet_to_ambient = 2 K/W
        let expected_magnet = T_AMB + power * 2.0;
        assert!(
            (model.nodes[2].temperature - expected_magnet).abs() < 0.01,
            "T_magnet = {:.2} K, expected {expected_magnet:.2} K",
            model.nodes[2].temperature
        );
    }

    #[test]
    fn test_transient_time_constant() {
        let mut model = simple_model();
        let power = 10.0;
        let tau = model.coil_time_constant(); // 10 × 5 = 50 s
        assert!((tau - 50.0).abs() < 1e-10);

        let dt = 0.1; // 100 ms steps
        let n_steps = (5.0 * tau / dt) as usize; // 5 time constants

        for _ in 0..n_steps {
            model.step(power, dt);
        }

        // After 5τ, should be ~99.3% of steady state
        let t_ss = T_AMB + power * 10.0; // 393.15 K
        let expected_frac = 1.0 - (-5.0_f64).exp(); // 0.9933
        let expected_t = T_AMB + (t_ss - T_AMB) * expected_frac;

        assert!(
            (model.coil_temperature() - expected_t).abs() < 1.0,
            "After 5τ: T={:.2}K, expected ~{expected_t:.2}K",
            model.coil_temperature()
        );
    }

    #[test]
    fn test_transient_at_one_tau() {
        let mut model = simple_model();
        let power = 10.0;
        let tau = model.coil_time_constant();
        let dt = 0.01;
        let n_steps = (tau / dt) as usize;

        for _ in 0..n_steps {
            model.step(power, dt);
        }

        // At t=τ: T = T_amb + ΔT_ss × (1 - e^(-1)) ≈ 63.2% of steady state
        let delta_t_ss = power * 10.0;
        let expected = T_AMB + delta_t_ss * (1.0 - (-1.0_f64).exp());

        let error = (model.coil_temperature() - expected).abs();
        assert!(
            error < 2.0,
            "At τ: T={:.2}K, expected ~{expected:.2}K (error={error:.2}K)",
            model.coil_temperature()
        );
    }

    #[test]
    fn test_re_increases_with_power() {
        let mut model = simple_model();
        model.steady_state(10.0);
        let re_10w = model.resistance();

        model.reset();
        model.steady_state(50.0);
        let re_50w = model.resistance();

        assert!(
            re_50w > re_10w,
            "Re should increase with power: Re(10W)={re_10w:.3}Ω, Re(50W)={re_50w:.3}Ω"
        );
    }

    #[test]
    fn test_self_consistent_converges() {
        let mut model = standard_model();
        // 2.83 Vrms into 8Ω ≈ 1W nominal
        let (t_coil, re_final, power) = model.steady_state_self_consistent(
            2.83,   // V_rms
            2.0,    // |Z_mech| at test freq
            5.0,    // BL
            0.5e-3, // Le
            1000.0, // frequency
        );

        assert!(t_coil > T_AMB, "Coil should heat up");
        assert!(re_final > 6.0, "Re should increase from nominal 6Ω");
        assert!(power > 0.0, "Power should be positive");
        assert!(power < 10.0, "Power should be reasonable for 2.83V");
    }

    #[test]
    fn test_reset_returns_to_ambient() {
        let mut model = simple_model();
        model.steady_state(50.0);
        assert!(model.coil_temperature() > T_AMB + 100.0);
        model.reset();
        assert!((model.coil_temperature() - T_AMB).abs() < 1e-10);
    }
}
