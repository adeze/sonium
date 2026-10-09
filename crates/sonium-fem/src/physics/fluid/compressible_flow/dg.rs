//! Explicit P1 discontinuous-Galerkin compressible Navier--Stokes solver.

use std::collections::HashMap;

use crate::basis::PolynomialDegree;
use crate::mesh::{ElementType, Mesh};
use crate::quadrature::QuadratureRule;
use crate::spaces::dg::{DGSpace, build_interior_facets, element_diameter};
use crate::spaces::{BasisTabulation, FunctionSpace};

use super::{ConservativeState, IdealGas, ViscosityModel, rusanov_flux};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressibleBoundaryCondition {
    SlipWall,
    NoSlipAdiabatic {
        wall_velocity: [f64; 3],
    },
    NoSlipIsothermal {
        wall_velocity: [f64; 3],
        temperature: f64,
    },
    FarField(ConservativeState),
    SupersonicState(ConservativeState),
    SubsonicTotalInlet {
        total_pressure: f64,
        total_temperature: f64,
        direction: [f64; 3],
    },
    StaticPressureOutlet {
        pressure: f64,
    },
}

#[derive(Debug, Clone)]
pub struct CompressibleDgConfig {
    pub gas: IdealGas,
    pub viscosity: ViscosityModel,
    pub prandtl: f64,
    pub cfl: f64,
    pub sipg_penalty: f64,
    pub density_floor: f64,
    pub pressure_floor: f64,
    pub troubled_cell_relaxation: f64,
    pub maximum_step_retries: usize,
    pub boundaries: HashMap<i32, CompressibleBoundaryCondition>,
}

impl Default for CompressibleDgConfig {
    fn default() -> Self {
        Self {
            gas: IdealGas::AIR,
            viscosity: ViscosityModel::AIR,
            prandtl: 0.72,
            cfl: 0.25,
            sipg_penalty: 4.0,
            density_floor: 1.0e-12,
            pressure_floor: 1.0e-12,
            troubled_cell_relaxation: 0.05,
            maximum_step_retries: 8,
            boundaries: HashMap::new(),
        }
    }
}

impl CompressibleDgConfig {
    pub fn validate(&self) -> Result<(), CompressibleDgError> {
        if !self.gas.gamma.is_finite()
            || self.gas.gamma <= 1.0
            || !self.gas.gas_constant.is_finite()
            || self.gas.gas_constant <= 0.0
            || !self.viscosity.is_valid()
            || !self.prandtl.is_finite()
            || self.prandtl <= 0.0
            || !self.cfl.is_finite()
            || self.cfl <= 0.0
            || !self.sipg_penalty.is_finite()
            || self.sipg_penalty <= 0.0
            || !self.density_floor.is_finite()
            || self.density_floor <= 0.0
            || !self.pressure_floor.is_finite()
            || self.pressure_floor <= 0.0
            || !self.troubled_cell_relaxation.is_finite()
            || self.troubled_cell_relaxation < 0.0
            || self.maximum_step_retries == 0
        {
            return Err(CompressibleDgError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressibleDgError {
    InvalidConfiguration,
    UnsupportedMesh,
    StateSizeMismatch,
    MissingBoundaryCondition,
    NonPhysicalCellAverage,
    NonPhysicalQuadratureState,
    StepRetriesExhausted,
}

/// Element-local P1-DG conservative state. Values are stored for every local
/// vertex of every element; neighboring elements never share entries.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleDgState {
    pub dimension: usize,
    pub values: Vec<ConservativeState>,
    element_offsets: Vec<usize>,
}

impl CompressibleDgState {
    pub fn from_function<F>(mesh: &Mesh, mut initial: F) -> Result<Self, CompressibleDgError>
    where
        F: FnMut([f64; 3]) -> ConservativeState,
    {
        validate_simplex_mesh(mesh)?;
        let mut values = Vec::new();
        let mut element_offsets = Vec::with_capacity(mesh.num_elements() + 1);
        element_offsets.push(0);
        for element in &mesh.elements {
            for &node in element.vertices() {
                let point = &mesh.nodes[node];
                let mut state = initial([point.x, point.y, point.z]);
                if mesh.dimension == 2 {
                    state.momentum[2] = 0.0;
                }
                values.push(state);
            }
            element_offsets.push(values.len());
        }
        Ok(Self {
            dimension: mesh.dimension,
            values,
            element_offsets,
        })
    }

    pub fn num_equations(&self) -> usize {
        self.dimension + 2
    }

    pub fn element(&self, element: usize) -> &[ConservativeState] {
        &self.values[self.element_offsets[element]..self.element_offsets[element + 1]]
    }

    fn element_mut(&mut self, element: usize) -> &mut [ConservativeState] {
        let start = self.element_offsets[element];
        let end = self.element_offsets[element + 1];
        &mut self.values[start..end]
    }

    pub fn cell_average(&self, element: usize) -> ConservativeState {
        average_states(self.element(element))
    }

    pub fn validate_layout(&self, mesh: &Mesh) -> Result<(), CompressibleDgError> {
        if self.dimension != mesh.dimension
            || self.element_offsets.len() != mesh.num_elements() + 1
            || self.element_offsets.last().copied() != Some(self.values.len())
            || mesh.elements.iter().enumerate().any(|(element, cell)| {
                self.element_offsets[element + 1] - self.element_offsets[element]
                    != cell.vertices().len()
            })
        {
            return Err(CompressibleDgError::StateSizeMismatch);
        }
        Ok(())
    }

    fn add_scaled(&self, derivative: &[ConservativeState], scale: f64) -> Self {
        let mut result = self.clone();
        for (state, rate) in result.values.iter_mut().zip(derivative) {
            *state = combine_states(*state, 1.0, *rate, scale);
            if self.dimension == 2 {
                state.momentum[2] = 0.0;
            }
        }
        result
    }

    fn convex_combination(&self, a: f64, other: &Self, b: f64) -> Self {
        let mut result = self.clone();
        for ((out, left), right) in result
            .values
            .iter_mut()
            .zip(&self.values)
            .zip(&other.values)
        {
            *out = combine_states(*left, a, *right, b);
            if self.dimension == 2 {
                out.momentum[2] = 0.0;
            }
        }
        result
    }
}

pub struct CompressibleDgOperator<'a> {
    mesh: &'a Mesh,
    space: DGSpace,
    pub config: CompressibleDgConfig,
}

impl<'a> CompressibleDgOperator<'a> {
    pub fn new(mesh: &'a Mesh, config: CompressibleDgConfig) -> Result<Self, CompressibleDgError> {
        validate_simplex_mesh(mesh)?;
        config.validate()?;
        Ok(Self {
            mesh,
            space: DGSpace::new(PolynomialDegree::P1),
            config,
        })
    }

    /// Evaluate the semi-discrete `dU/dt` including Euler volume/facet fluxes,
    /// Newtonian stress, Fourier heat flux, and SIPG jump stabilization.
    pub fn spatial_operator(
        &self,
        state: &CompressibleDgState,
    ) -> Result<Vec<ConservativeState>, CompressibleDgError> {
        state.validate_layout(self.mesh)?;
        let mut integrated = vec![[0.0; 5]; state.values.len()];
        self.add_volume_terms(state, &mut integrated)?;
        self.add_interior_facet_terms(state, &mut integrated)?;
        self.add_boundary_terms(state, &mut integrated)?;
        Ok(self.apply_inverse_mass(&integrated))
    }

    pub fn stable_time_step(
        &self,
        state: &CompressibleDgState,
    ) -> Result<f64, CompressibleDgError> {
        state.validate_layout(self.mesh)?;
        let mut dt = f64::INFINITY;
        for element in 0..self.mesh.num_elements() {
            let average = state.cell_average(element);
            if !average.is_physical(
                self.config.gas,
                self.config.density_floor,
                self.config.pressure_floor,
            ) {
                return Err(CompressibleDgError::NonPhysicalCellAverage);
            }
            let h = element_diameter(self.mesh, element);
            let velocity = average.velocity();
            let speed = norm3(velocity)
                + self
                    .config
                    .gas
                    .speed_of_sound(average.pressure(self.config.gas), average.density);
            let temperature = self
                .config
                .gas
                .temperature(average.pressure(self.config.gas), average.density);
            let mu = self.config.viscosity.dynamic_viscosity(temperature);
            let conductivity = self.config.viscosity.thermal_conductivity(
                temperature,
                self.config.gas,
                self.config.prandtl,
            );
            let diffusivity = (mu / average.density)
                .max(conductivity / (average.density * self.config.gas.specific_heat_cp()));
            let convective = h / speed.max(f64::MIN_POSITIVE);
            let viscous = 0.5 * h * h / diffusivity.max(f64::MIN_POSITIVE);
            dt = dt.min(self.config.cfl * convective.min(viscous));
        }
        Ok(dt)
    }

    pub fn conserved_totals(&self, state: &CompressibleDgState) -> [f64; 5] {
        let mut totals = [0.0; 5];
        for element in 0..self.mesh.num_elements() {
            let volume = cell_volume(self.mesh, element);
            let average = state.cell_average(element).as_array();
            for equation in 0..5 {
                totals[equation] += volume * average[equation];
            }
        }
        totals
    }

    fn add_volume_terms(
        &self,
        state: &CompressibleDgState,
        residual: &mut [[f64; 5]],
    ) -> Result<(), CompressibleDgError> {
        for element in 0..self.mesh.num_elements() {
            let cell = state.element(element);
            let gradients = primitive_gradients(
                cell,
                &self.space.tabulate(self.mesh, element, 0.25, 0.25, 0.25),
                self.config.gas,
            );
            let rule = QuadratureRule::new(self.mesh.elements[element].element_type, 3);
            let offset = state.element_offsets[element];
            for point in rule.iter() {
                let tab = self.space.tabulate(
                    self.mesh,
                    element,
                    point.coords[0],
                    point.coords[1],
                    point.coords[2],
                );
                let value = interpolate_with_basis(cell, &tab.values);
                if !value.is_physical(
                    self.config.gas,
                    self.config.density_floor,
                    self.config.pressure_floor,
                ) {
                    return Err(CompressibleDgError::NonPhysicalQuadratureState);
                }
                let viscous = viscous_flux_tensor(
                    value,
                    gradients,
                    self.config.gas,
                    self.config.viscosity,
                    self.config.prandtl,
                );
                let weight = point.weight * tab.det_j;
                for local in 0..tab.n_dofs {
                    for direction in 0..self.mesh.dimension {
                        let mut normal = [0.0; 3];
                        normal[direction] = 1.0;
                        let inviscid = value.euler_flux(self.config.gas, normal);
                        for equation in 0..5 {
                            residual[offset + local][equation] += tab.grad(local, direction)
                                * (inviscid[equation] - viscous[direction][equation])
                                * weight;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn add_interior_facet_terms(
        &self,
        state: &CompressibleDgState,
        residual: &mut [[f64; 5]],
    ) -> Result<(), CompressibleDgError> {
        for facet in build_interior_facets(self.mesh) {
            let plus_centroid = element_centroid(self.mesh, facet.elem_plus);
            let minus_centroid = element_centroid(self.mesh, facet.elem_minus);
            let mut normal = facet.normal;
            if dot3(normal, sub3(minus_centroid, plus_centroid)) < 0.0 {
                normal = scale3(normal, -1.0);
            }
            let plus = state.element(facet.elem_plus);
            let minus = state.element(facet.elem_minus);
            let plus_gradients = primitive_gradients(
                plus,
                &self
                    .space
                    .tabulate(self.mesh, facet.elem_plus, 0.25, 0.25, 0.25),
                self.config.gas,
            );
            let minus_gradients = primitive_gradients(
                minus,
                &self
                    .space
                    .tabulate(self.mesh, facet.elem_minus, 0.25, 0.25, 0.25),
                self.config.gas,
            );
            let h = 0.5
                * (element_diameter(self.mesh, facet.elem_plus)
                    + element_diameter(self.mesh, facet.elem_minus));
            for (point, weight) in facet_quadrature(self.mesh, &facet.shared_nodes) {
                let plus_basis = barycentric_coordinates(self.mesh, facet.elem_plus, point);
                let minus_basis = barycentric_coordinates(self.mesh, facet.elem_minus, point);
                let left = interpolate_with_basis(plus, &plus_basis);
                let right = interpolate_with_basis(minus, &minus_basis);
                let flux =
                    self.numerical_flux(left, right, plus_gradients, minus_gradients, normal, h)?;
                let plus_offset = state.element_offsets[facet.elem_plus];
                let minus_offset = state.element_offsets[facet.elem_minus];
                for local in 0..plus.len() {
                    for equation in 0..5 {
                        residual[plus_offset + local][equation] -=
                            plus_basis[local] * flux[equation] * weight;
                    }
                }
                for local in 0..minus.len() {
                    for equation in 0..5 {
                        residual[minus_offset + local][equation] +=
                            minus_basis[local] * flux[equation] * weight;
                    }
                }
            }
        }
        Ok(())
    }

    fn add_boundary_terms(
        &self,
        state: &CompressibleDgState,
        residual: &mut [[f64; 5]],
    ) -> Result<(), CompressibleDgError> {
        for face in &self.mesh.boundaries {
            let condition = self
                .config
                .boundaries
                .get(&face.marker)
                .copied()
                .ok_or(CompressibleDgError::MissingBoundaryCondition)?;
            let element = face.element_idx;
            let cell = state.element(element);
            let gradients = primitive_gradients(
                cell,
                &self.space.tabulate(self.mesh, element, 0.25, 0.25, 0.25),
                self.config.gas,
            );
            let mut normal = facet_normal(self.mesh, &face.nodes);
            let face_centroid = points_centroid(self.mesh, &face.nodes);
            if dot3(
                normal,
                sub3(face_centroid, element_centroid(self.mesh, element)),
            ) < 0.0
            {
                normal = scale3(normal, -1.0);
            }
            let h = element_diameter(self.mesh, element);
            let offset = state.element_offsets[element];
            for (point, weight) in facet_quadrature(self.mesh, &face.nodes) {
                let basis = barycentric_coordinates(self.mesh, element, point);
                let interior = interpolate_with_basis(cell, &basis);
                let exterior = boundary_state(condition, interior, normal, self.config.gas)?;
                let flux =
                    self.numerical_flux(interior, exterior, gradients, gradients, normal, h)?;
                for local in 0..cell.len() {
                    for equation in 0..5 {
                        residual[offset + local][equation] -=
                            basis[local] * flux[equation] * weight;
                    }
                }
            }
        }
        Ok(())
    }

    fn numerical_flux(
        &self,
        left: ConservativeState,
        right: ConservativeState,
        left_gradients: PrimitiveGradients,
        right_gradients: PrimitiveGradients,
        normal: [f64; 3],
        h: f64,
    ) -> Result<[f64; 5], CompressibleDgError> {
        for state in [left, right] {
            if !state.is_physical(
                self.config.gas,
                self.config.density_floor,
                self.config.pressure_floor,
            ) {
                return Err(CompressibleDgError::NonPhysicalQuadratureState);
            }
        }
        let inviscid = rusanov_flux(left, right, self.config.gas, normal);
        let left_viscous = viscous_flux_tensor(
            left,
            left_gradients,
            self.config.gas,
            self.config.viscosity,
            self.config.prandtl,
        );
        let right_viscous = viscous_flux_tensor(
            right,
            right_gradients,
            self.config.gas,
            self.config.viscosity,
            self.config.prandtl,
        );
        let left_temperature = self
            .config
            .gas
            .temperature(left.pressure(self.config.gas), left.density);
        let right_temperature = self
            .config
            .gas
            .temperature(right.pressure(self.config.gas), right.density);
        let maximum_diffusivity = (self.config.viscosity.dynamic_viscosity(left_temperature)
            / left.density)
            .max(self.config.viscosity.dynamic_viscosity(right_temperature) / right.density);
        let penalty = self.config.sipg_penalty * maximum_diffusivity / h.max(f64::MIN_POSITIVE);
        let left_array = left.as_array();
        let right_array = right.as_array();
        let mut flux = inviscid;
        for equation in 0..5 {
            let viscous_normal = (0..self.mesh.dimension)
                .map(|direction| {
                    0.5 * (left_viscous[direction][equation] + right_viscous[direction][equation])
                        * normal[direction]
                })
                .sum::<f64>();
            flux[equation] = inviscid[equation] - viscous_normal
                + penalty * (left_array[equation] - right_array[equation]);
        }
        Ok(flux)
    }

    fn apply_inverse_mass(&self, integrated: &[[f64; 5]]) -> Vec<ConservativeState> {
        let mut derivative = vec![ConservativeState::from_array([0.0; 5]); integrated.len()];
        let mut offset = 0;
        for element in 0..self.mesh.num_elements() {
            let n = self.mesh.elements[element].vertices().len();
            let volume = cell_volume(self.mesh, element);
            let scale = (n * (n + 1)) as f64 / volume;
            for equation in 0..5 {
                let sum = (0..n)
                    .map(|local| integrated[offset + local][equation])
                    .sum::<f64>();
                for local in 0..n {
                    let value =
                        scale * (integrated[offset + local][equation] - sum / (n + 1) as f64);
                    let mut array = derivative[offset + local].as_array();
                    array[equation] = value;
                    derivative[offset + local] = ConservativeState::from_array(array);
                }
            }
            offset += n;
        }
        derivative
    }
}

pub struct CompressibleDgSolver<'a> {
    pub operator: CompressibleDgOperator<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleStepReport {
    pub accepted_time_step: f64,
    pub retries: usize,
    pub minimum_density: f64,
    pub minimum_pressure: f64,
    pub conserved_before: [f64; 5],
    pub conserved_after: [f64; 5],
}

impl<'a> CompressibleDgSolver<'a> {
    pub fn new(mesh: &'a Mesh, config: CompressibleDgConfig) -> Result<Self, CompressibleDgError> {
        Ok(Self {
            operator: CompressibleDgOperator::new(mesh, config)?,
        })
    }

    /// Advance one SSPRK3 step. A nonphysical cell average rejects the trial
    /// and halves `dt`, up to the configured retry limit.
    pub fn step(
        &self,
        state: &mut CompressibleDgState,
        requested_time_step: f64,
    ) -> Result<CompressibleStepReport, CompressibleDgError> {
        state.validate_layout(self.operator.mesh)?;
        let before = self.operator.conserved_totals(state);
        let original = state.clone();
        let stable = self.operator.stable_time_step(state)?;
        let mut dt = requested_time_step.min(stable);
        for retry in 0..self.operator.config.maximum_step_retries {
            match self.ssprk3_trial(&original, dt) {
                Ok(candidate) => {
                    let (minimum_density, minimum_pressure) =
                        minimum_thermodynamics(&candidate, self.operator.config.gas);
                    *state = candidate;
                    return Ok(CompressibleStepReport {
                        accepted_time_step: dt,
                        retries: retry,
                        minimum_density,
                        minimum_pressure,
                        conserved_before: before,
                        conserved_after: self.operator.conserved_totals(state),
                    });
                }
                Err(CompressibleDgError::NonPhysicalCellAverage)
                | Err(CompressibleDgError::NonPhysicalQuadratureState) => dt *= 0.5,
                Err(error) => return Err(error),
            }
        }
        Err(CompressibleDgError::StepRetriesExhausted)
    }

    fn ssprk3_trial(
        &self,
        initial: &CompressibleDgState,
        dt: f64,
    ) -> Result<CompressibleDgState, CompressibleDgError> {
        let rate0 = self.operator.spatial_operator(initial)?;
        let mut stage1 = initial.add_scaled(&rate0, dt);
        self.limit(&mut stage1)?;

        let rate1 = self.operator.spatial_operator(&stage1)?;
        let euler1 = stage1.add_scaled(&rate1, dt);
        let mut stage2 = initial.convex_combination(0.75, &euler1, 0.25);
        self.limit(&mut stage2)?;

        let rate2 = self.operator.spatial_operator(&stage2)?;
        let euler2 = stage2.add_scaled(&rate2, dt);
        let mut stage3 = initial.convex_combination(1.0 / 3.0, &euler2, 2.0 / 3.0);
        self.limit(&mut stage3)?;
        Ok(stage3)
    }

    fn limit(&self, state: &mut CompressibleDgState) -> Result<(), CompressibleDgError> {
        apply_characteristic_vertex_limiter(
            self.operator.mesh,
            state,
            self.operator.config.gas,
            self.operator.config.troubled_cell_relaxation,
        );
        apply_zhang_shu_positivity(
            state,
            self.operator.config.gas,
            self.operator.config.density_floor,
            self.operator.config.pressure_floor,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct PrimitiveGradients {
    velocity: [[f64; 3]; 3],
    temperature: [f64; 3],
}

fn primitive_gradients(
    cell: &[ConservativeState],
    tab: &BasisTabulation,
    gas: IdealGas,
) -> PrimitiveGradients {
    let mut velocity = [[0.0; 3]; 3];
    let mut temperature = [0.0; 3];
    for (local, state) in cell.iter().enumerate() {
        let nodal_velocity = state.velocity();
        let nodal_temperature = gas.temperature(state.pressure(gas), state.density);
        for direction in 0..tab.spatial_dim {
            for component in 0..3 {
                velocity[component][direction] +=
                    nodal_velocity[component] * tab.grad(local, direction);
            }
            temperature[direction] += nodal_temperature * tab.grad(local, direction);
        }
    }
    PrimitiveGradients {
        velocity,
        temperature,
    }
}

fn viscous_flux_tensor(
    state: ConservativeState,
    gradients: PrimitiveGradients,
    gas: IdealGas,
    viscosity: ViscosityModel,
    prandtl: f64,
) -> [[f64; 5]; 3] {
    let pressure = state.pressure(gas);
    let temperature = gas.temperature(pressure, state.density);
    let mu = viscosity.dynamic_viscosity(temperature);
    let conductivity = viscosity.thermal_conductivity(temperature, gas, prandtl);
    let velocity = state.velocity();
    let divergence = gradients.velocity[0][0] + gradients.velocity[1][1] + gradients.velocity[2][2];
    let mut stress = [[0.0; 3]; 3];
    for (component, row) in stress.iter_mut().enumerate() {
        for (direction, slot) in row.iter_mut().enumerate() {
            *slot = mu
                * (gradients.velocity[component][direction]
                    + gradients.velocity[direction][component]
                    - if component == direction {
                        2.0 * divergence / 3.0
                    } else {
                        0.0
                    });
        }
    }
    let mut flux = [[0.0; 5]; 3];
    for direction in 0..3 {
        flux[direction][1] = stress[0][direction];
        flux[direction][2] = stress[1][direction];
        flux[direction][3] = stress[2][direction];
        flux[direction][4] = velocity
            .iter()
            .enumerate()
            .map(|(component, value)| value * stress[component][direction])
            .sum::<f64>()
            + conductivity * gradients.temperature[direction];
    }
    flux
}

fn boundary_state(
    condition: CompressibleBoundaryCondition,
    interior: ConservativeState,
    normal: [f64; 3],
    gas: IdealGas,
) -> Result<ConservativeState, CompressibleDgError> {
    let pressure = interior.pressure(gas);
    let velocity = interior.velocity();
    let state = match condition {
        CompressibleBoundaryCondition::SlipWall => {
            let normal_velocity = dot3(velocity, normal);
            let reflected = sub3(velocity, scale3(normal, 2.0 * normal_velocity));
            ConservativeState::from_primitive(interior.density, reflected, pressure, gas)
        }
        CompressibleBoundaryCondition::NoSlipAdiabatic { wall_velocity } => {
            let reflected = sub3(scale3(wall_velocity, 2.0), velocity);
            ConservativeState::from_primitive(interior.density, reflected, pressure, gas)
        }
        CompressibleBoundaryCondition::NoSlipIsothermal {
            wall_velocity,
            temperature,
        } => {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(CompressibleDgError::InvalidConfiguration);
            }
            let interior_temperature = gas.temperature(pressure, interior.density);
            let ghost_temperature = (2.0 * temperature - interior_temperature).max(1.0e-12);
            let density = pressure / (gas.gas_constant * ghost_temperature);
            let reflected = sub3(scale3(wall_velocity, 2.0), velocity);
            ConservativeState::from_primitive(density, reflected, pressure, gas)
        }
        CompressibleBoundaryCondition::FarField(far_field) => {
            characteristic_far_field_state(interior, far_field, normal, gas)
        }
        CompressibleBoundaryCondition::SupersonicState(prescribed) => prescribed,
        CompressibleBoundaryCondition::SubsonicTotalInlet {
            total_pressure,
            total_temperature,
            direction,
        } => {
            let direction_norm = norm3(direction);
            if total_pressure <= 0.0 || total_temperature <= 0.0 || direction_norm <= f64::EPSILON {
                return Err(CompressibleDgError::InvalidConfiguration);
            }
            let unit_direction = scale3(direction, 1.0 / direction_norm);
            let speed = norm3(velocity);
            let sound = gas.speed_of_sound(pressure, interior.density);
            let mach = (speed / sound).clamp(0.0, 0.999);
            let factor = 1.0 + 0.5 * (gas.gamma - 1.0) * mach * mach;
            let static_temperature = total_temperature / factor;
            let static_pressure = total_pressure / factor.powf(gas.gamma / (gas.gamma - 1.0));
            let density = static_pressure / (gas.gas_constant * static_temperature);
            let inlet_speed = mach * (gas.gamma * gas.gas_constant * static_temperature).sqrt();
            ConservativeState::from_primitive(
                density,
                scale3(unit_direction, inlet_speed),
                static_pressure,
                gas,
            )
        }
        CompressibleBoundaryCondition::StaticPressureOutlet { pressure } => {
            if pressure <= 0.0 || !pressure.is_finite() {
                return Err(CompressibleDgError::InvalidConfiguration);
            }
            ConservativeState::from_primitive(interior.density, velocity, pressure, gas)
        }
    };
    Ok(state)
}

fn characteristic_far_field_state(
    interior: ConservativeState,
    far_field: ConservativeState,
    normal: [f64; 3],
    gas: IdealGas,
) -> ConservativeState {
    let ui = interior.velocity();
    let uf = far_field.velocity();
    let pi = interior.pressure(gas);
    let pf = far_field.pressure(gas);
    let ci = gas.speed_of_sound(pi, interior.density);
    let cf = gas.speed_of_sound(pf, far_field.density);
    let uni = dot3(ui, normal);
    if uni >= ci {
        return interior;
    }
    if uni <= -ci {
        return far_field;
    }
    let unf = dot3(uf, normal);
    let r_plus = uni + 2.0 * ci / (gas.gamma - 1.0);
    let r_minus = unf - 2.0 * cf / (gas.gamma - 1.0);
    let normal_velocity = 0.5 * (r_plus + r_minus);
    let sound_speed = 0.25 * (gas.gamma - 1.0) * (r_plus - r_minus);
    let upstream = if normal_velocity < 0.0 {
        far_field
    } else {
        interior
    };
    let entropy = upstream.pressure(gas) / upstream.density.powf(gas.gamma);
    let density = (sound_speed * sound_speed / (gas.gamma * entropy))
        .max(f64::MIN_POSITIVE)
        .powf(1.0 / (gas.gamma - 1.0));
    let pressure = entropy * density.powf(gas.gamma);
    let upstream_velocity = upstream.velocity();
    let tangential = sub3(
        upstream_velocity,
        scale3(normal, dot3(upstream_velocity, normal)),
    );
    ConservativeState::from_primitive(
        density,
        add3(tangential, scale3(normal, normal_velocity)),
        pressure,
        gas,
    )
}

fn apply_characteristic_vertex_limiter(
    mesh: &Mesh,
    state: &mut CompressibleDgState,
    gas: IdealGas,
    relaxation: f64,
) {
    let original = state.clone();
    let averages = (0..mesh.num_elements())
        .map(|element| original.cell_average(element))
        .collect::<Vec<_>>();
    let mut neighbors: Vec<Vec<(usize, [f64; 3])>> = vec![Vec::new(); mesh.num_elements()];
    for facet in build_interior_facets(mesh) {
        let mut normal = facet.normal;
        if dot3(
            normal,
            sub3(
                element_centroid(mesh, facet.elem_minus),
                element_centroid(mesh, facet.elem_plus),
            ),
        ) < 0.0
        {
            normal = scale3(normal, -1.0);
        }
        neighbors[facet.elem_plus].push((facet.elem_minus, normal));
        neighbors[facet.elem_minus].push((facet.elem_plus, scale3(normal, -1.0)));
    }
    for element in 0..mesh.num_elements() {
        if neighbors[element].is_empty() {
            continue;
        }
        let average = averages[element];
        let mut theta = 1.0_f64;
        for &(neighbor, normal) in &neighbors[element] {
            let center_characteristics = characteristic_variables(average, normal, gas);
            let neighbor_characteristics =
                characteristic_variables(averages[neighbor], normal, gas);
            for nodal in original.element(element) {
                let nodal_characteristics = characteristic_variables(*nodal, normal, gas);
                for wave in 0..5 {
                    let delta_neighbor =
                        neighbor_characteristics[wave] - center_characteristics[wave];
                    let margin = relaxation * delta_neighbor.abs().max(1.0e-12);
                    let lower =
                        center_characteristics[wave].min(neighbor_characteristics[wave]) - margin;
                    let upper =
                        center_characteristics[wave].max(neighbor_characteristics[wave]) + margin;
                    let delta = nodal_characteristics[wave] - center_characteristics[wave];
                    if delta > 0.0 {
                        theta = theta.min((upper - center_characteristics[wave]) / delta);
                    } else if delta < 0.0 {
                        theta = theta.min((lower - center_characteristics[wave]) / delta);
                    }
                }
            }
        }
        let average_array = average.as_array();
        for nodal in state.element_mut(element) {
            let values = nodal.as_array();
            let mut limited = [0.0; 5];
            for equation in 0..5 {
                limited[equation] = average_array[equation]
                    + theta.clamp(0.0, 1.0) * (values[equation] - average_array[equation]);
            }
            *nodal = ConservativeState::from_array(limited);
        }
    }
}

fn apply_zhang_shu_positivity(
    state: &mut CompressibleDgState,
    gas: IdealGas,
    density_floor: f64,
    pressure_floor: f64,
) -> Result<(), CompressibleDgError> {
    for element in 0..state.element_offsets.len() - 1 {
        let average = state.cell_average(element);
        if !average.is_physical(gas, density_floor, pressure_floor) {
            return Err(CompressibleDgError::NonPhysicalCellAverage);
        }
        let average_array = average.as_array();
        let mut theta = 1.0_f64;
        for nodal in state.element(element) {
            if nodal.density < density_floor {
                theta = theta
                    .min((average.density - density_floor) / (average.density - nodal.density));
            }
        }
        for nodal in state.element(element) {
            let nodal_array = nodal.as_array();
            let candidate = |factor: f64| {
                let mut values = [0.0; 5];
                for equation in 0..5 {
                    values[equation] = average_array[equation]
                        + factor * (nodal_array[equation] - average_array[equation]);
                }
                ConservativeState::from_array(values)
            };
            if candidate(theta).pressure(gas) < pressure_floor {
                let mut lower = 0.0;
                let mut upper = theta;
                for _ in 0..60 {
                    let midpoint = 0.5 * (lower + upper);
                    if candidate(midpoint).pressure(gas) >= pressure_floor {
                        lower = midpoint;
                    } else {
                        upper = midpoint;
                    }
                }
                theta = theta.min(lower);
            }
        }
        for nodal in state.element_mut(element) {
            let values = nodal.as_array();
            let mut limited = [0.0; 5];
            for equation in 0..5 {
                limited[equation] = average_array[equation]
                    + theta.clamp(0.0, 1.0) * (values[equation] - average_array[equation]);
            }
            *nodal = ConservativeState::from_array(limited);
        }
    }
    Ok(())
}

fn characteristic_variables(state: ConservativeState, normal: [f64; 3], gas: IdealGas) -> [f64; 5] {
    let velocity = state.velocity();
    let pressure = state.pressure(gas);
    let sound = gas.speed_of_sound(pressure, state.density);
    let normal_velocity = dot3(velocity, normal);
    let tangent1 = normalized_tangent(normal);
    let tangent2 = cross3(normal, tangent1);
    [
        pressure - state.density * sound * normal_velocity,
        state.density - pressure / (sound * sound),
        pressure + state.density * sound * normal_velocity,
        dot3(velocity, tangent1),
        dot3(velocity, tangent2),
    ]
}

fn normalized_tangent(normal: [f64; 3]) -> [f64; 3] {
    let seed = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let tangent = cross3(normal, seed);
    scale3(tangent, 1.0 / norm3(tangent))
}

fn validate_simplex_mesh(mesh: &Mesh) -> Result<(), CompressibleDgError> {
    let expected = match mesh.dimension {
        2 => ElementType::Triangle,
        3 => ElementType::Tetrahedron,
        _ => return Err(CompressibleDgError::UnsupportedMesh),
    };
    if mesh.elements.is_empty()
        || mesh
            .elements
            .iter()
            .any(|element| element.element_type != expected)
    {
        return Err(CompressibleDgError::UnsupportedMesh);
    }
    Ok(())
}

fn average_states(states: &[ConservativeState]) -> ConservativeState {
    let mut sum = [0.0; 5];
    for state in states {
        for (target, value) in sum.iter_mut().zip(state.as_array()) {
            *target += value;
        }
    }
    for value in &mut sum {
        *value /= states.len() as f64;
    }
    ConservativeState::from_array(sum)
}

fn combine_states(
    left: ConservativeState,
    left_scale: f64,
    right: ConservativeState,
    right_scale: f64,
) -> ConservativeState {
    let left = left.as_array();
    let right = right.as_array();
    let mut result = [0.0; 5];
    for equation in 0..5 {
        result[equation] = left_scale * left[equation] + right_scale * right[equation];
    }
    ConservativeState::from_array(result)
}

fn interpolate_with_basis(states: &[ConservativeState], basis: &[f64]) -> ConservativeState {
    let mut result = [0.0; 5];
    for (state, shape) in states.iter().zip(basis) {
        for (target, value) in result.iter_mut().zip(state.as_array()) {
            *target += shape * value;
        }
    }
    ConservativeState::from_array(result)
}

fn minimum_thermodynamics(state: &CompressibleDgState, gas: IdealGas) -> (f64, f64) {
    state.values.iter().fold(
        (f64::INFINITY, f64::INFINITY),
        |(minimum_density, minimum_pressure), value| {
            (
                minimum_density.min(value.density),
                minimum_pressure.min(value.pressure(gas)),
            )
        },
    )
}

fn cell_volume(mesh: &Mesh, element: usize) -> f64 {
    let rule = QuadratureRule::new(mesh.elements[element].element_type, 1);
    let space = DGSpace::new(PolynomialDegree::P1);
    rule.iter()
        .map(|point| {
            point.weight
                * space
                    .tabulate(
                        mesh,
                        element,
                        point.coords[0],
                        point.coords[1],
                        point.coords[2],
                    )
                    .det_j
        })
        .sum()
}

fn element_centroid(mesh: &Mesh, element: usize) -> [f64; 3] {
    points_centroid(mesh, mesh.elements[element].vertices())
}

fn points_centroid(mesh: &Mesh, nodes: &[usize]) -> [f64; 3] {
    let mut result = [0.0; 3];
    for &node in nodes {
        result[0] += mesh.nodes[node].x;
        result[1] += mesh.nodes[node].y;
        result[2] += mesh.nodes[node].z;
    }
    scale3(result, 1.0 / nodes.len() as f64)
}

fn barycentric_coordinates(mesh: &Mesh, element: usize, point: [f64; 3]) -> Vec<f64> {
    let vertices = mesh.elements[element].vertices();
    if mesh.dimension == 2 {
        let a = point3(mesh, vertices[0]);
        let b = point3(mesh, vertices[1]);
        let c = point3(mesh, vertices[2]);
        let denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
        let l0 =
            ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) / denominator;
        let l1 =
            ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) / denominator;
        vec![l0, l1, 1.0 - l0 - l1]
    } else {
        let a = point3(mesh, vertices[0]);
        let matrix = [
            sub3(point3(mesh, vertices[1]), a),
            sub3(point3(mesh, vertices[2]), a),
            sub3(point3(mesh, vertices[3]), a),
        ];
        let rhs = sub3(point, a);
        let determinant = dot3(matrix[0], cross3(matrix[1], matrix[2]));
        let l1 = dot3(rhs, cross3(matrix[1], matrix[2])) / determinant;
        let l2 = dot3(matrix[0], cross3(rhs, matrix[2])) / determinant;
        let l3 = dot3(matrix[0], cross3(matrix[1], rhs)) / determinant;
        vec![1.0 - l1 - l2 - l3, l1, l2, l3]
    }
}

fn facet_quadrature(mesh: &Mesh, nodes: &[usize]) -> Vec<([f64; 3], f64)> {
    let area = facet_measure(mesh, nodes);
    if nodes.len() == 2 {
        let a = point3(mesh, nodes[0]);
        let b = point3(mesh, nodes[1]);
        let offset = 1.0 / 3.0_f64.sqrt();
        [0.5 * (1.0 - offset), 0.5 * (1.0 + offset)]
            .into_iter()
            .map(|s| (add3(scale3(a, 1.0 - s), scale3(b, s)), 0.5 * area))
            .collect()
    } else {
        const RULE: [([f64; 3], f64); 7] = [
            ([1.0 / 3.0; 3], 0.225),
            (
                [
                    0.059_715_871_789_770,
                    0.470_142_064_105_115,
                    0.470_142_064_105_115,
                ],
                0.132_394_152_788_506,
            ),
            (
                [
                    0.470_142_064_105_115,
                    0.059_715_871_789_770,
                    0.470_142_064_105_115,
                ],
                0.132_394_152_788_506,
            ),
            (
                [
                    0.470_142_064_105_115,
                    0.470_142_064_105_115,
                    0.059_715_871_789_770,
                ],
                0.132_394_152_788_506,
            ),
            (
                [
                    0.797_426_985_353_087,
                    0.101_286_507_323_456,
                    0.101_286_507_323_456,
                ],
                0.125_939_180_544_827,
            ),
            (
                [
                    0.101_286_507_323_456,
                    0.797_426_985_353_087,
                    0.101_286_507_323_456,
                ],
                0.125_939_180_544_827,
            ),
            (
                [
                    0.101_286_507_323_456,
                    0.101_286_507_323_456,
                    0.797_426_985_353_087,
                ],
                0.125_939_180_544_827,
            ),
        ];
        let points = [
            point3(mesh, nodes[0]),
            point3(mesh, nodes[1]),
            point3(mesh, nodes[2]),
        ];
        RULE.into_iter()
            .map(|(basis, weight)| {
                let point = add3(
                    add3(scale3(points[0], basis[0]), scale3(points[1], basis[1])),
                    scale3(points[2], basis[2]),
                );
                (point, weight * area)
            })
            .collect()
    }
}

fn facet_measure(mesh: &Mesh, nodes: &[usize]) -> f64 {
    if nodes.len() == 2 {
        norm3(sub3(point3(mesh, nodes[1]), point3(mesh, nodes[0])))
    } else {
        0.5 * norm3(cross3(
            sub3(point3(mesh, nodes[1]), point3(mesh, nodes[0])),
            sub3(point3(mesh, nodes[2]), point3(mesh, nodes[0])),
        ))
    }
}

fn facet_normal(mesh: &Mesh, nodes: &[usize]) -> [f64; 3] {
    let raw = if nodes.len() == 2 {
        let edge = sub3(point3(mesh, nodes[1]), point3(mesh, nodes[0]));
        [edge[1], -edge[0], 0.0]
    } else {
        cross3(
            sub3(point3(mesh, nodes[1]), point3(mesh, nodes[0])),
            sub3(point3(mesh, nodes[2]), point3(mesh, nodes[0])),
        )
    };
    scale3(raw, 1.0 / norm3(raw))
}

fn point3(mesh: &Mesh, node: usize) -> [f64; 3] {
    [mesh.nodes[node].x, mesh.nodes[node].y, mesh.nodes[node].z]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(a: [f64; 3], scale: f64) -> [f64; 3] {
    [scale * a[0], scale * a[1], scale * a[2]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{unit_cube_tetrahedra, unit_square_triangles};

    fn uniform_state() -> ConservativeState {
        ConservativeState::from_primitive(1.2, [30.0, 5.0, 0.0], 101_325.0, IdealGas::AIR)
    }

    fn config_with_far_field(mesh: &Mesh, state: ConservativeState) -> CompressibleDgConfig {
        let mut config = CompressibleDgConfig::default();
        config.viscosity = ViscosityModel::Constant(1.8e-5);
        for face in &mesh.boundaries {
            config
                .boundaries
                .insert(face.marker, CompressibleBoundaryCondition::FarField(state));
        }
        config
    }

    #[test]
    fn uniform_flow_residual_vanishes_on_triangles() {
        let mesh = unit_square_triangles(2);
        let reference = uniform_state();
        let state = CompressibleDgState::from_function(&mesh, |_| reference).unwrap();
        let operator =
            CompressibleDgOperator::new(&mesh, config_with_far_field(&mesh, reference)).unwrap();
        let residual = operator.spatial_operator(&state).unwrap();
        let maximum = residual
            .iter()
            .flat_map(|value| value.as_array())
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(maximum < 1.0e-5, "maximum residual={maximum:e}");
    }

    #[test]
    fn uniform_flow_residual_vanishes_on_tetrahedra() {
        let mesh = unit_cube_tetrahedra(1);
        let reference = uniform_state();
        let state = CompressibleDgState::from_function(&mesh, |_| reference).unwrap();
        let operator =
            CompressibleDgOperator::new(&mesh, config_with_far_field(&mesh, reference)).unwrap();
        let residual = operator.spatial_operator(&state).unwrap();
        let maximum = residual
            .iter()
            .flat_map(|value| value.as_array())
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(maximum < 1.0e-5, "maximum residual={maximum:e}");
    }

    #[test]
    fn positivity_scaling_preserves_cell_average() {
        let mesh = unit_square_triangles(1);
        let gas = IdealGas::AIR;
        let reference = ConservativeState::from_primitive(1.0, [0.0; 3], 1.0, gas);
        let mut state = CompressibleDgState::from_function(&mesh, |_| reference).unwrap();
        let before = state.cell_average(0);
        state.element_mut(0)[0].density = -0.1;
        // Compensate another node so the average remains physical.
        state.element_mut(0)[1].density += 1.1;
        let perturbed_average = state.cell_average(0);
        apply_zhang_shu_positivity(&mut state, gas, 1.0e-10, 1.0e-10).unwrap();
        let after = state.cell_average(0);
        assert!((after.density - perturbed_average.density).abs() < 1.0e-12);
        assert!(state.element(0).iter().all(|value| value.density > 0.0));
        assert!((before.total_energy - after.total_energy).abs() < 1.0e-12);
    }

    #[test]
    fn ssprk3_advances_physical_uniform_state() {
        let mesh = unit_square_triangles(1);
        let reference = uniform_state();
        let mut state = CompressibleDgState::from_function(&mesh, |_| reference).unwrap();
        let solver =
            CompressibleDgSolver::new(&mesh, config_with_far_field(&mesh, reference)).unwrap();
        let report = solver.step(&mut state, 1.0e-5).unwrap();
        assert!(report.minimum_density > 0.0);
        assert!(report.minimum_pressure > 0.0);
        assert_eq!(report.retries, 0);
    }

    #[test]
    fn wall_and_characteristic_boundary_states_are_physical() {
        let gas = IdealGas::AIR;
        let interior = uniform_state();
        let normal = [1.0, 0.0, 0.0];
        let slip = boundary_state(
            CompressibleBoundaryCondition::SlipWall,
            interior,
            normal,
            gas,
        )
        .unwrap();
        assert!((slip.velocity()[0] + interior.velocity()[0]).abs() < 1.0e-12);
        let far = boundary_state(
            CompressibleBoundaryCondition::FarField(interior),
            interior,
            normal,
            gas,
        )
        .unwrap();
        assert!(far.is_physical(gas, 1.0e-12, 1.0e-12));
    }
}
