//! Species, Arrhenius reactions, and mixture properties.

pub const UNIVERSAL_GAS_CONSTANT: f64 = 8.314_462_618;

#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalSpecies {
    pub name: String,
    pub molar_mass: f64,
    pub diffusivity: f64,
    pub specific_heat: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrheniusReaction {
    /// Stoichiometric coefficients: negative reactants, positive products.
    pub stoichiometry: Vec<f64>,
    /// Nonnegative mass-action orders for each species.
    pub orders: Vec<f64>,
    pub pre_exponential_factor: f64,
    pub activation_energy: f64,
    /// Positive for exothermic heat release, J per reaction extent.
    pub heat_release: f64,
}

impl ArrheniusReaction {
    pub fn validate(&self, species_count: usize) -> Result<(), ReactingFlowError> {
        if self.stoichiometry.len() != species_count || self.orders.len() != species_count {
            return Err(ReactingFlowError::SpeciesCountMismatch);
        }
        if self.pre_exponential_factor < 0.0
            || !self.pre_exponential_factor.is_finite()
            || self.activation_energy < 0.0
            || !self.activation_energy.is_finite()
            || !self.heat_release.is_finite()
        {
            return Err(ReactingFlowError::InvalidReactionCoefficient);
        }
        if self
            .orders
            .iter()
            .any(|order| !order.is_finite() || *order < 0.0)
        {
            return Err(ReactingFlowError::InvalidReactionOrder);
        }
        Ok(())
    }

    pub fn rate_constant(&self, temperature: f64) -> f64 {
        assert!(temperature > 0.0);
        self.pre_exponential_factor
            * (-self.activation_energy / (UNIVERSAL_GAS_CONSTANT * temperature)).exp()
    }

    pub fn progress_rate(&self, concentrations: &[f64], temperature: f64) -> f64 {
        assert_eq!(concentrations.len(), self.orders.len());
        self.rate_constant(temperature)
            * concentrations
                .iter()
                .zip(&self.orders)
                .map(|(concentration, order)| concentration.max(0.0).powf(*order))
                .product::<f64>()
    }

    pub fn species_source(&self, species: usize, concentrations: &[f64], temperature: f64) -> f64 {
        self.stoichiometry[species] * self.progress_rate(concentrations, temperature)
    }

    pub fn progress_rate_derivative(
        &self,
        with_respect_to: usize,
        concentrations: &[f64],
        temperature: f64,
    ) -> f64 {
        let concentration = concentrations[with_respect_to].max(f64::MIN_POSITIVE);
        self.progress_rate(concentrations, temperature) * self.orders[with_respect_to]
            / concentration
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReactionMechanism {
    pub species: Vec<ChemicalSpecies>,
    pub reactions: Vec<ArrheniusReaction>,
}

impl ReactionMechanism {
    pub fn validate(&self) -> Result<(), ReactingFlowError> {
        if self.species.is_empty() {
            return Err(ReactingFlowError::EmptyMechanism);
        }
        for species in &self.species {
            if species.molar_mass <= 0.0
                || species.diffusivity <= 0.0
                || species.specific_heat <= 0.0
                || !species.molar_mass.is_finite()
                || !species.diffusivity.is_finite()
                || !species.specific_heat.is_finite()
            {
                return Err(ReactingFlowError::InvalidSpeciesCoefficient);
            }
        }
        for reaction in &self.reactions {
            reaction.validate(self.species.len())?;
        }
        Ok(())
    }

    pub fn sources(&self, concentrations: &[f64], temperature: f64) -> Vec<f64> {
        let mut sources = vec![0.0; self.species.len()];
        for reaction in &self.reactions {
            for (index, source) in sources.iter_mut().enumerate() {
                *source += reaction.species_source(index, concentrations, temperature);
            }
        }
        sources
    }

    pub fn heat_release_rate(&self, concentrations: &[f64], temperature: f64) -> f64 {
        self.reactions
            .iter()
            .map(|reaction| {
                reaction.heat_release * reaction.progress_rate(concentrations, temperature)
            })
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReactingFlowError {
    EmptyMechanism,
    SpeciesCountMismatch,
    InvalidSpeciesCoefficient,
    InvalidReactionCoefficient,
    InvalidReactionOrder,
}
