//! General circuit network solver using Modified Nodal Analysis (MNA)
//!
//! Assembles and solves arbitrary RLC circuits in the frequency domain.
//! The MNA system at angular frequency ω:
//!
//! ```text
//! (G + jωC) x = s
//! ```
//!
//! where `x = [node_voltages; branch_currents]` for voltage sources.
//!
//! # Applications
//!
//! - Passive crossover networks (Butterworth, Linkwitz-Riley, Zobel)
//! - Voice coil circuits coupled to mechanical/acoustic domains
//! - Arbitrary filter topologies with R, L, C, transformers
//!
//! # Example
//!
//! ```ignore
//! use sonium_fem::physics::coupled::lumped_circuit::network::*;
//!
//! let mut net = CircuitNetwork::new();
//! // RC lowpass: Vin -> R -> node 1 -> C -> ground
//! let vin = net.add_voltage_source(0, 1); // ground to node 1
//! net.add_resistor(1, 2, 1000.0);         // 1 kΩ
//! net.add_capacitor(2, 0, 1e-6);          // 1 µF
//! // f_c = 1/(2π·1000·1e-6) ≈ 159.15 Hz
//!
//! let omega = 2.0 * std::f64::consts::PI * 159.15;
//! let excitations = vec![(vin, Complex64::new(1.0, 0.0))];
//! let sol = net.solve(omega, &excitations).unwrap();
//! // sol.node_voltage(2) is the output voltage at -3 dB
//! ```

use num_complex::Complex64;
use sonium_compute::HostMatrix;
use sonium_solvers::dense::DenseMatrix;

/// Index of a voltage source branch in the MNA system.
///
/// Used to specify excitations and read branch currents from the solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VSourceIndex(pub usize);

/// A component in the circuit network.
#[derive(Debug, Clone)]
pub enum CircuitComponent {
    /// Resistor between two nodes. Resistance in ohms.
    Resistor {
        node_p: usize,
        node_n: usize,
        resistance: f64,
    },
    /// Capacitor between two nodes. Capacitance in farads.
    Capacitor {
        node_p: usize,
        node_n: usize,
        capacitance: f64,
    },
    /// Inductor between two nodes. Inductance in henries.
    Inductor {
        node_p: usize,
        node_n: usize,
        inductance: f64,
    },
    /// Ideal voltage source between two nodes.
    /// Voltage is defined as V(node_p) - V(node_n) = V_src.
    /// The source index is stored for excitation lookup.
    VoltageSource {
        node_p: usize,
        node_n: usize,
        index: VSourceIndex,
    },
    /// Frequency-dependent impedance between two nodes.
    /// Provided as a closure: omega -> Complex64 impedance.
    /// Used for voice coil ports, acoustic loads, etc.
    ImpedancePort {
        node_p: usize,
        node_n: usize,
        impedance_fn: ImpedanceFn,
    },
}

/// Wrapper for a frequency-dependent impedance function.
#[derive(Clone)]
pub struct ImpedanceFn {
    f: std::sync::Arc<dyn Fn(f64) -> Complex64 + Send + Sync>,
}

impl ImpedanceFn {
    pub fn new<F: Fn(f64) -> Complex64 + Send + Sync + 'static>(f: F) -> Self {
        Self {
            f: std::sync::Arc::new(f),
        }
    }

    pub fn eval(&self, omega: f64) -> Complex64 {
        (self.f)(omega)
    }
}

impl std::fmt::Debug for ImpedanceFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ImpedanceFn(..)")
    }
}

/// A circuit network for MNA analysis.
///
/// Node 0 is ground (reference). All other node indices are 1-based.
/// The network tracks the highest node index seen and the number of
/// voltage source branches.
#[derive(Debug)]
pub struct CircuitNetwork {
    components: Vec<CircuitComponent>,
    n_nodes: usize,    // highest node index (excluding ground)
    n_vsources: usize, // number of voltage sources
}

impl Default for CircuitNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitNetwork {
    /// Create an empty circuit network.
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            n_nodes: 0,
            n_vsources: 0,
        }
    }

    fn update_max_node(&mut self, node: usize) {
        if node > self.n_nodes {
            self.n_nodes = node;
        }
    }

    /// Add a resistor. Returns self for chaining.
    pub fn add_resistor(&mut self, node_p: usize, node_n: usize, resistance: f64) {
        self.update_max_node(node_p);
        self.update_max_node(node_n);
        self.components.push(CircuitComponent::Resistor {
            node_p,
            node_n,
            resistance,
        });
    }

    /// Add a capacitor. Returns self for chaining.
    pub fn add_capacitor(&mut self, node_p: usize, node_n: usize, capacitance: f64) {
        self.update_max_node(node_p);
        self.update_max_node(node_n);
        self.components.push(CircuitComponent::Capacitor {
            node_p,
            node_n,
            capacitance,
        });
    }

    /// Add an inductor.
    pub fn add_inductor(&mut self, node_p: usize, node_n: usize, inductance: f64) {
        self.update_max_node(node_p);
        self.update_max_node(node_n);
        self.components.push(CircuitComponent::Inductor {
            node_p,
            node_n,
            inductance,
        });
    }

    /// Add an ideal voltage source. Returns the source index for use in excitations.
    ///
    /// The voltage is defined as V(node_p) - V(node_n) = V_src.
    pub fn add_voltage_source(&mut self, node_p: usize, node_n: usize) -> VSourceIndex {
        self.update_max_node(node_p);
        self.update_max_node(node_n);
        let idx = VSourceIndex(self.n_vsources);
        self.n_vsources += 1;
        self.components.push(CircuitComponent::VoltageSource {
            node_p,
            node_n,
            index: idx,
        });
        idx
    }

    /// Add a frequency-dependent impedance port.
    ///
    /// The impedance function takes angular frequency ω and returns a Complex64 impedance.
    /// At each solve frequency, the admittance Y = 1/Z(ω) is stamped into the G matrix.
    ///
    /// **Note**: If the impedance returns zero (or near-zero), the port is skipped
    /// for that frequency. To model a short circuit, use a small resistance instead.
    ///
    /// Use this for voice coil subcircuits, acoustic loads, or any frequency-dependent element.
    pub fn add_impedance_port(&mut self, node_p: usize, node_n: usize, impedance_fn: ImpedanceFn) {
        self.update_max_node(node_p);
        self.update_max_node(node_n);
        self.components.push(CircuitComponent::ImpedancePort {
            node_p,
            node_n,
            impedance_fn,
        });
    }

    /// Total system size: n_nodes + n_vsources.
    fn system_size(&self) -> usize {
        self.n_nodes + self.n_vsources
    }

    /// Assemble and solve the MNA system at angular frequency ω.
    ///
    /// `excitations` specifies the complex voltage for each voltage source.
    /// Sources not listed default to 0V.
    pub fn solve(
        &self,
        omega: f64,
        excitations: &[(VSourceIndex, Complex64)],
    ) -> Result<CircuitSolution, MnaError> {
        let n = self.system_size();
        if n == 0 {
            return Err(MnaError::EmptyNetwork);
        }

        let mut matrix = HostMatrix::<Complex64>::from_elem(n, n, Complex64::new(0.0, 0.0));
        let mut rhs: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
        let jw = Complex64::new(0.0, omega);

        for comp in &self.components {
            match comp {
                CircuitComponent::Resistor {
                    node_p,
                    node_n,
                    resistance,
                } => {
                    let g = Complex64::new(1.0 / resistance, 0.0);
                    stamp_admittance(&mut matrix, *node_p, *node_n, g);
                }
                CircuitComponent::Capacitor {
                    node_p,
                    node_n,
                    capacitance,
                } => {
                    let y = jw * capacitance;
                    stamp_admittance(&mut matrix, *node_p, *node_n, y);
                }
                CircuitComponent::Inductor {
                    node_p,
                    node_n,
                    inductance,
                } => {
                    // Admittance of inductor: Y = 1/(jωL)
                    if omega.abs() < 1e-30 {
                        // At DC, inductor is a short — use very large admittance
                        let g = Complex64::new(1e15, 0.0);
                        stamp_admittance(&mut matrix, *node_p, *node_n, g);
                    } else {
                        let y = Complex64::new(1.0, 0.0) / (jw * inductance);
                        stamp_admittance(&mut matrix, *node_p, *node_n, y);
                    }
                }
                CircuitComponent::VoltageSource {
                    node_p,
                    node_n,
                    index,
                } => {
                    let branch = self.n_nodes + index.0;
                    // Stamp: the branch current row/column
                    // V(node_p) - V(node_n) = V_src
                    if *node_p > 0 {
                        let np = node_p - 1;
                        matrix[[np, branch]] += Complex64::new(1.0, 0.0);
                        matrix[[branch, np]] += Complex64::new(1.0, 0.0);
                    }
                    if *node_n > 0 {
                        let nn = node_n - 1;
                        matrix[[nn, branch]] -= Complex64::new(1.0, 0.0);
                        matrix[[branch, nn]] -= Complex64::new(1.0, 0.0);
                    }
                }
                CircuitComponent::ImpedancePort {
                    node_p,
                    node_n,
                    impedance_fn,
                } => {
                    let z = impedance_fn.eval(omega);
                    if z.norm() > 1e-30 {
                        let y = Complex64::new(1.0, 0.0) / z;
                        stamp_admittance(&mut matrix, *node_p, *node_n, y);
                    }
                }
            }
        }

        // Apply voltage source excitations to RHS
        for &(VSourceIndex(idx), voltage) in excitations {
            let branch = self.n_nodes + idx;
            rhs[branch] = voltage;
        }

        // Solve via LU decomposition
        let x = lu_solve_complex(&matrix, &rhs)?;

        Ok(CircuitSolution {
            values: x,
            n_nodes: self.n_nodes,
            n_vsources: self.n_vsources,
        })
    }

    /// Solve at multiple frequencies, returning a solution per frequency.
    pub fn solve_sweep(
        &self,
        frequencies_hz: &[f64],
        excitations: &[(VSourceIndex, Complex64)],
    ) -> Result<Vec<CircuitSolution>, MnaError> {
        frequencies_hz
            .iter()
            .map(|&f| {
                let omega = 2.0 * std::f64::consts::PI * f;
                self.solve(omega, excitations)
            })
            .collect()
    }

    /// Number of nodes (excluding ground).
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of voltage sources.
    pub fn n_vsources(&self) -> usize {
        self.n_vsources
    }
}

/// Stamp an admittance Y between node_p and node_n into the MNA matrix.
///
/// Node 0 is ground and not represented in the matrix.
fn stamp_admittance(
    matrix: &mut HostMatrix<Complex64>,
    node_p: usize,
    node_n: usize,
    y: Complex64,
) {
    if node_p > 0 {
        let np = node_p - 1;
        matrix[[np, np]] += y;
        if node_n > 0 {
            let nn = node_n - 1;
            matrix[[np, nn]] -= y;
        }
    }
    if node_n > 0 {
        let nn = node_n - 1;
        matrix[[nn, nn]] += y;
        if node_p > 0 {
            let np = node_p - 1;
            matrix[[nn, np]] -= y;
        }
    }
}

/// Dense complex LU solve for small MNA systems.
fn lu_solve_complex(
    a: &HostMatrix<Complex64>,
    b: &[Complex64],
) -> Result<Vec<Complex64>, MnaError> {
    let dense = DenseMatrix::from_row_major_vec(a.nrows(), a.ncols(), a.as_slice().to_vec());
    sonium_solvers::direct::lu_solve(&dense, b).map_err(|_| MnaError::SingularMatrix)
}

/// Solution of an MNA system at one frequency.
#[derive(Debug, Clone)]
pub struct CircuitSolution {
    values: Vec<Complex64>,
    n_nodes: usize,
    n_vsources: usize,
}

impl CircuitSolution {
    /// Node voltage (complex phasor). Node 0 (ground) returns 0.
    pub fn node_voltage(&self, node: usize) -> Complex64 {
        if node == 0 {
            Complex64::new(0.0, 0.0)
        } else {
            self.values[node - 1]
        }
    }

    /// Voltage across two nodes: V(node_p) - V(node_n).
    pub fn voltage_across(&self, node_p: usize, node_n: usize) -> Complex64 {
        self.node_voltage(node_p) - self.node_voltage(node_n)
    }

    /// Current delivered by a voltage source (flowing out of node_p through the load).
    ///
    /// The MNA branch variable represents current flowing from node_n to node_p
    /// through the source. This method negates it to return the conventional
    /// delivered current: from node_p through the external circuit to node_n.
    pub fn vsource_current(&self, idx: VSourceIndex) -> Complex64 {
        -self.values[self.n_nodes + idx.0]
    }

    /// Voltage transfer function: V(output_node) / V(input_node).
    /// Returns magnitude and phase (radians).
    pub fn transfer_function(&self, input_node: usize, output_node: usize) -> (f64, f64) {
        let v_in = self.node_voltage(input_node);
        let v_out = self.node_voltage(output_node);
        let h = v_out / v_in;
        (h.norm(), h.arg())
    }

    /// Transfer function magnitude in dB: 20·log10(|V_out/V_in|).
    pub fn transfer_db(&self, input_node: usize, output_node: usize) -> f64 {
        let (mag, _) = self.transfer_function(input_node, output_node);
        20.0 * mag.log10()
    }

    /// Number of voltage sources in the circuit.
    pub fn n_vsources(&self) -> usize {
        self.n_vsources
    }
}

/// Errors from MNA assembly or solve.
#[derive(Debug)]
pub enum MnaError {
    EmptyNetwork,
    SingularMatrix,
}

impl std::fmt::Display for MnaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyNetwork => write!(f, "Circuit network has no components"),
            Self::SingularMatrix => write!(f, "MNA matrix is singular"),
        }
    }
}

impl std::error::Error for MnaError {}

// ===========================================================================
// Standard crossover topologies
// ===========================================================================

/// Build a 2nd-order Butterworth lowpass crossover.
///
/// ```text
/// Input (node_in) ---[L]--- node_mid ---+--- Output (node_out)
///                                       |
///                                      [C]
///                                       |
///                                      GND
/// ```
///
/// Returns (node_mid, node_out) where node_out = node_mid for this topology.
/// The component values are computed for the given crossover frequency and load impedance.
pub fn butterworth_lowpass_2nd(
    net: &mut CircuitNetwork,
    node_in: usize,
    node_out: usize,
    crossover_freq_hz: f64,
    load_impedance: f64,
) {
    let omega_c = 2.0 * std::f64::consts::PI * crossover_freq_hz;
    // Butterworth Q = 1/√2, normalized L = √2, C = √2 for 2nd order
    let l = load_impedance * std::f64::consts::SQRT_2 / omega_c;
    let c = std::f64::consts::SQRT_2 / (omega_c * load_impedance);
    net.add_inductor(node_in, node_out, l);
    net.add_capacitor(node_out, 0, c);
}

/// Build a 2nd-order Butterworth highpass crossover.
///
/// ```text
/// Input (node_in) ---[C]--- node_mid ---+--- Output (node_out)
///                                       |
///                                      [L]
///                                       |
///                                      GND
/// ```
pub fn butterworth_highpass_2nd(
    net: &mut CircuitNetwork,
    node_in: usize,
    node_out: usize,
    crossover_freq_hz: f64,
    load_impedance: f64,
) {
    let omega_c = 2.0 * std::f64::consts::PI * crossover_freq_hz;
    let c = 1.0 / (load_impedance * std::f64::consts::SQRT_2 * omega_c);
    let l = load_impedance / (std::f64::consts::SQRT_2 * omega_c);
    net.add_capacitor(node_in, node_out, c);
    net.add_inductor(node_out, 0, l);
}

/// Build a 4th-order Linkwitz-Riley lowpass (two cascaded Butterworth 2nd-order).
///
/// Uses intermediate node between the two stages.
/// Returns the intermediate node index used.
pub fn linkwitz_riley_lowpass_4th(
    net: &mut CircuitNetwork,
    node_in: usize,
    node_out: usize,
    crossover_freq_hz: f64,
    load_impedance: f64,
) -> usize {
    // LR4 = two cascaded Butterworth 2nd-order at the same frequency.
    // In practice, the intermediate node needs a buffer (active crossover)
    // or the component values must be recalculated for the loaded case.
    // For the idealized (buffered) case:
    let node_mid = net.n_nodes() + 1;
    butterworth_lowpass_2nd(net, node_in, node_mid, crossover_freq_hz, load_impedance);
    butterworth_lowpass_2nd(net, node_mid, node_out, crossover_freq_hz, load_impedance);
    node_mid
}

/// Build a 4th-order Linkwitz-Riley highpass (two cascaded Butterworth 2nd-order).
pub fn linkwitz_riley_highpass_4th(
    net: &mut CircuitNetwork,
    node_in: usize,
    node_out: usize,
    crossover_freq_hz: f64,
    load_impedance: f64,
) -> usize {
    let node_mid = net.n_nodes() + 1;
    butterworth_highpass_2nd(net, node_in, node_mid, crossover_freq_hz, load_impedance);
    butterworth_highpass_2nd(net, node_mid, node_out, crossover_freq_hz, load_impedance);
    node_mid
}

/// Add a Zobel (impedance equalization) network across the load.
///
/// ```text
/// node_p ---[R_z]---[C_z]--- node_n
/// ```
///
/// Compensates the rising impedance of a voice coil inductance.
/// R_z = R_e (DC resistance), C_z = L_e / R_e²
///
/// Uses an internal node for the R-C junction. The node index is
/// allocated above the network's current highest node.
pub fn zobel_network(
    net: &mut CircuitNetwork,
    node_p: usize,
    node_n: usize,
    dc_resistance: f64,
    inductance: f64,
) {
    let r_z = dc_resistance;
    let c_z = inductance / (dc_resistance * dc_resistance);
    // Series R-C across the terminals: need an intermediate node
    let node_mid = net.n_nodes() + 1;
    net.add_resistor(node_p, node_mid, r_z);
    net.add_capacitor(node_mid, node_n, c_z);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const ONE: Complex64 = Complex64 { re: 1.0, im: 0.0 };

    /// Helper: magnitude in dB
    fn db(v: Complex64) -> f64 {
        20.0 * v.norm().log10()
    }

    #[test]
    fn test_resistor_divider() {
        // V_src (1V) -> R1 (1kΩ) -> node 2 -> R2 (1kΩ) -> GND
        let mut net = CircuitNetwork::new();
        let vs = net.add_voltage_source(1, 0); // V(1) = 1V
        net.add_resistor(1, 2, 1000.0);
        net.add_resistor(2, 0, 1000.0);

        let sol = net.solve(0.0, &[(vs, ONE)]).unwrap();

        // V(1) = 1V, V(2) = 0.5V (voltage divider)
        let v1 = sol.node_voltage(1);
        let v2 = sol.node_voltage(2);
        assert!((v1.re - 1.0).abs() < 1e-10, "V1 = {v1}");
        assert!((v2.re - 0.5).abs() < 1e-10, "V2 = {v2}");
    }

    #[test]
    fn test_rc_lowpass_cutoff() {
        // RC lowpass: V_src -> R(1kΩ) -> node 2 -> C(1µF) -> GND
        // f_c = 1/(2π·R·C) = 1/(2π·1000·1e-6) ≈ 159.15 Hz
        let mut net = CircuitNetwork::new();
        let vs = net.add_voltage_source(1, 0);
        net.add_resistor(1, 2, 1000.0);
        net.add_capacitor(2, 0, 1e-6);

        let f_c = 1.0 / (2.0 * PI * 1000.0 * 1e-6);

        // At cutoff: should be -3.01 dB
        let sol = net.solve(2.0 * PI * f_c, &[(vs, ONE)]).unwrap();
        let gain_db = db(sol.node_voltage(2));
        assert!(
            (gain_db - (-3.0103)).abs() < 0.05,
            "Gain at f_c = {gain_db:.2} dB, expected -3.01 dB"
        );

        // Well below cutoff: ~0 dB
        let sol_low = net.solve(2.0 * PI * 1.0, &[(vs, ONE)]).unwrap();
        let gain_low = db(sol_low.node_voltage(2));
        assert!(
            gain_low > -0.1,
            "Gain at 1 Hz = {gain_low:.2} dB, expected ~0 dB"
        );

        // Well above cutoff: should roll off at -20 dB/decade
        let sol_high = net.solve(2.0 * PI * f_c * 10.0, &[(vs, ONE)]).unwrap();
        let gain_high = db(sol_high.node_voltage(2));
        assert!(
            gain_high < -15.0,
            "Gain at 10×f_c = {gain_high:.2} dB, expected < -15 dB"
        );
    }

    #[test]
    fn test_rl_highpass_cutoff() {
        // RL highpass: V_src -> C -> node 2 -> R -> GND
        // Actually let's do: V_src -> R -> node 2 -> L -> GND → this is a lowpass
        // Highpass: V_src -> L -> node 2, R to ground
        // f_c = R/(2πL)
        let r = 1000.0;
        let l = 0.1; // 100 mH
        let f_c = r / (2.0 * PI * l); // ≈ 1591.5 Hz

        let mut net = CircuitNetwork::new();
        let vs = net.add_voltage_source(1, 0);
        // Highpass: series capacitor or series inductor in voltage divider
        // RL lowpass: V_src -> node 1, R from 1->2, L from 2->GND, output at 2
        // V_out/V_in = jωL / (R + jωL)
        // |H| at f_c = 1/√2 → -3 dB
        net.add_resistor(1, 2, r);
        net.add_inductor(2, 0, l);

        // This is actually a lowpass (output across inductor)
        // H(jω) = jωL/(R + jωL), at ω=R/L: H = j/(1+j) → |H| = 1/√2
        // Wait, that's a highpass. Let me verify:
        // At DC (ω=0): jωL → 0, so V_out = 0 → highpass behavior on the inductor
        // At ω→∞: jωL >> R, V_out → V_in → passes high frequencies
        // So RL with output across L is actually a highpass. Good.

        let sol = net.solve(2.0 * PI * f_c, &[(vs, ONE)]).unwrap();
        let gain_db = db(sol.node_voltage(2));
        assert!(
            (gain_db - (-3.01)).abs() < 0.1,
            "Gain at f_c = {gain_db:.2} dB, expected -3.01 dB"
        );
    }

    #[test]
    fn test_rlc_series_resonance() {
        // Series RLC: V_src -> R -> L -> C -> GND, output across C
        // Resonance at f_0 = 1/(2π√(LC))
        // Q = (1/R)√(L/C)
        let r = 10.0_f64;
        let l = 0.01_f64; // 10 mH
        let c = 1e-6_f64; // 1 µF
        let f0 = 1.0 / (2.0 * PI * (l * c).sqrt()); // ≈ 1591.5 Hz
        let q = (l / c).sqrt() / r; // = 100/10 = 10

        let mut net = CircuitNetwork::new();
        let vs = net.add_voltage_source(1, 0);
        net.add_resistor(1, 2, r);
        net.add_inductor(2, 3, l);
        net.add_capacitor(3, 0, c);

        // At resonance, the LC cancels and current = V/R
        // Voltage across C at resonance: V_C = I/(jωC) = (V/R)/(jω₀C)
        // |V_C|/|V_in| = 1/(ω₀RC) = Q
        let sol_res = net.solve(2.0 * PI * f0, &[(vs, ONE)]).unwrap();
        let v_c = sol_res.node_voltage(3);
        let gain_at_resonance = v_c.norm();

        assert!(
            (gain_at_resonance - q).abs() / q < 0.02,
            "Gain at resonance = {gain_at_resonance:.2}, expected Q = {q:.2}"
        );

        // Well below resonance: capacitor dominates, gain → 1 (series RLC across C is a bandpass though)
        // Actually for series RLC with output across C:
        // H(s) = 1/(s²LC + sRC + 1) — this is a lowpass with resonance peak = Q
        // At very low freq: H → 1
        let sol_low = net.solve(2.0 * PI * 10.0, &[(vs, ONE)]).unwrap();
        let gain_low = sol_low.node_voltage(3).norm();
        assert!(
            (gain_low - 1.0).abs() < 0.01,
            "Low-freq gain = {gain_low:.4}, expected ~1.0"
        );
    }

    #[test]
    fn test_butterworth_lowpass_2nd_shape() {
        // 2nd-order Butterworth lowpass at 2 kHz, 8Ω load
        let f_c = 2000.0;
        let r_load = 8.0;

        let mut net = CircuitNetwork::new();
        let vs = net.add_voltage_source(1, 0);
        butterworth_lowpass_2nd(&mut net, 1, 2, f_c, r_load);
        net.add_resistor(2, 0, r_load); // load

        // At cutoff: -3 dB (Butterworth maximally flat → -3.01 dB at f_c)
        let sol = net.solve(2.0 * PI * f_c, &[(vs, ONE)]).unwrap();
        let gain_db = db(sol.node_voltage(2));
        // Note: with actual load impedance, the filter response may shift slightly.
        // The Butterworth design assumes a matched load. With R_load in the circuit,
        // the -3dB point should be close to f_c.
        assert!(
            gain_db < -1.0 && gain_db > -6.0,
            "Gain at f_c = {gain_db:.2} dB, expected near -3 dB"
        );

        // Well below: ~0 dB relative to DC gain
        let sol_dc = net.solve(2.0 * PI * 10.0, &[(vs, ONE)]).unwrap();
        let gain_dc = sol_dc.node_voltage(2).norm();

        // Far above: -12 dB/octave slope (2nd order)
        let sol_4fc = net.solve(2.0 * PI * f_c * 4.0, &[(vs, ONE)]).unwrap();
        let sol_8fc = net.solve(2.0 * PI * f_c * 8.0, &[(vs, ONE)]).unwrap();
        let db_4fc = 20.0 * (sol_4fc.node_voltage(2).norm() / gain_dc).log10();
        let db_8fc = 20.0 * (sol_8fc.node_voltage(2).norm() / gain_dc).log10();

        // One octave from 4fc to 8fc should drop ~12 dB
        let slope = db_8fc - db_4fc;
        assert!(
            slope < -8.0 && slope > -16.0,
            "Slope = {slope:.1} dB/octave, expected ~-12"
        );
    }

    #[test]
    fn test_impedance_port() {
        // Voltage divider using an impedance port as the bottom element
        let mut net = CircuitNetwork::new();
        let vs = net.add_voltage_source(1, 0);
        net.add_resistor(1, 2, 1000.0);

        // Impedance port: constant 1kΩ at all frequencies
        net.add_impedance_port(2, 0, ImpedanceFn::new(|_omega| Complex64::new(1000.0, 0.0)));

        let sol = net.solve(2.0 * PI * 1000.0, &[(vs, ONE)]).unwrap();
        let v2 = sol.node_voltage(2);
        assert!(
            (v2.re - 0.5).abs() < 1e-10,
            "V2 = {v2}, expected 0.5 (divider)"
        );
    }

    #[test]
    fn test_voice_coil_impedance_via_port() {
        // Model a voice coil as an impedance port and verify impedance curve
        let re = 6.5;
        let le = 0.5e-3;

        let mut net = CircuitNetwork::new();
        let vs = net.add_voltage_source(1, 0);
        net.add_impedance_port(
            1,
            0,
            ImpedanceFn::new(move |omega| Complex64::new(re, omega * le)),
        );

        // At 1 kHz: Z = 6.5 + j·2π·1000·0.5e-3 = 6.5 + j3.14
        let sol = net.solve(2.0 * PI * 1000.0, &[(vs, ONE)]).unwrap();
        let i = sol.vsource_current(vs);
        let z = ONE / i; // Z = V/I

        let expected_z = Complex64::new(re, 2.0 * PI * 1000.0 * le);
        assert!(
            (z - expected_z).norm() < 0.01,
            "Z = {z}, expected {expected_z}"
        );
    }

    #[test]
    fn test_frequency_sweep() {
        // Simple RC lowpass sweep
        let mut net = CircuitNetwork::new();
        let vs = net.add_voltage_source(1, 0);
        net.add_resistor(1, 2, 1000.0);
        net.add_capacitor(2, 0, 1e-6);

        let freqs: Vec<f64> = (1..=5).map(|i| 10.0_f64.powi(i)).collect(); // 10, 100, 1k, 10k, 100k
        let solutions = net.solve_sweep(&freqs, &[(vs, ONE)]).unwrap();

        // Gain should monotonically decrease
        let gains: Vec<f64> = solutions.iter().map(|s| s.node_voltage(2).norm()).collect();

        for i in 1..gains.len() {
            assert!(
                gains[i] < gains[i - 1],
                "Gain should decrease: gains = {gains:?}"
            );
        }
    }

    #[test]
    fn test_current_conservation() {
        // KCL: current into a node equals current out
        // V_src -> R1(100Ω) -> node 2 -> R2(200Ω) -> GND
        //                           |-> R3(300Ω) -> GND
        let mut net = CircuitNetwork::new();
        let vs = net.add_voltage_source(1, 0);
        net.add_resistor(1, 2, 100.0);
        net.add_resistor(2, 0, 200.0);
        net.add_resistor(2, 0, 300.0);

        let sol = net.solve(0.0, &[(vs, ONE)]).unwrap();
        let v2 = sol.node_voltage(2);
        let i_source = sol.vsource_current(vs);

        // I through R1 = (V1 - V2)/R1
        let i_r1 = (ONE - v2) / Complex64::new(100.0, 0.0);
        // I through R2 = V2/R2
        let i_r2 = v2 / Complex64::new(200.0, 0.0);
        // I through R3 = V2/R3
        let i_r3 = v2 / Complex64::new(300.0, 0.0);

        // KCL at node 2: I_R1 = I_R2 + I_R3
        assert!(
            (i_r1 - i_r2 - i_r3).norm() < 1e-10,
            "KCL violated at node 2"
        );

        // Source current = I_R1
        assert!(
            (i_source.norm() - i_r1.norm()).abs() < 1e-10,
            "Source current mismatch"
        );
    }

    #[test]
    fn test_zobel_flattens_impedance() {
        // Voice coil: R=8Ω, L=1mH — impedance rises with frequency
        // Zobel: R_z=8Ω, C_z=L/R²=1e-3/64=15.625µF
        // Combined impedance should be approximately flat at 8Ω
        let re = 8.0;
        let le = 1e-3;

        // Network: V_src -> node 1, voice coil + zobel in parallel to ground
        let mut net = CircuitNetwork::new();
        let vs = net.add_voltage_source(1, 0);

        // Voice coil as R + L in series (R: 1->2, L: 2->0)
        net.add_resistor(1, 2, re);
        net.add_inductor(2, 0, le);

        // Zobel: R_z in series with C_z, across nodes 1-0
        zobel_network(&mut net, 1, 0, re, le);

        // At high frequency (10 kHz), impedance should stay near 8Ω
        let f_test = 10000.0;
        let sol = net.solve(2.0 * PI * f_test, &[(vs, ONE)]).unwrap();
        let i = sol.vsource_current(vs);
        let z = ONE / i;

        // Without Zobel: |Z| = sqrt(8² + (2π·10k·1e-3)²) = sqrt(64+3948) ≈ 63.3Ω
        // With Zobel: should be close to 8Ω
        assert!(
            (z.norm() - re).abs() < 1.0,
            "Impedance with Zobel at {f_test} Hz: |Z| = {:.2}Ω, expected ~{re}Ω",
            z.norm()
        );
    }
}
