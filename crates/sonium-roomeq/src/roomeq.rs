//! RoomEQ configuration types.
//!
//! Minimal re-implementation of the autoeq roomeq config types, sufficient for
//! JSON generation by `sonium-sim-room` and round-trip validation in tests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

/// RoomEQ configuration version.
pub fn default_config_version() -> String {
    CURRENT_CONFIG_SCHEMA_VERSION.to_string()
}

/// The newest RoomEQ configuration schema accepted by direct validation.
pub const CURRENT_CONFIG_SCHEMA_VERSION: &str = "1.2.0";

/// Typed validation failures for RoomEQ configuration data.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValidationError {
    /// The version field is absent or blank.
    EmptyVersion,
    /// A configuration version requires an explicit migration before use.
    UnsupportedVersion { version: String },
    /// A room configuration contains no speaker entries.
    EmptySpeakers,
    /// A numeric option contains NaN or infinity.
    NonFinite { field: &'static str },
    /// A numeric option must be strictly positive.
    NonPositive { field: &'static str, value: f64 },
    /// A lower/upper pair is not ordered.
    InvalidRange {
        field: &'static str,
        min: f64,
        max: f64,
    },
    /// A count option is zero where at least one item is required.
    ZeroCount { field: &'static str },
    /// An audio sample rate is outside the supported positive range.
    InvalidSampleRate { field: &'static str, value: u32 },
    /// A required string/path value is blank.
    EmptyValue { field: &'static str },
    /// Mutually exclusive configuration fields were supplied together.
    InvalidCombination { field: &'static str },
}

impl fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVersion => write!(f, "RoomEQ version must not be empty"),
            Self::UnsupportedVersion { version } => write!(
                f,
                "RoomEQ configuration version {version:?} is unsupported; migrate it explicitly"
            ),
            Self::EmptySpeakers => write!(f, "RoomEQ configuration must contain a speaker"),
            Self::NonFinite { field } => write!(f, "RoomEQ field {field} must be finite"),
            Self::NonPositive { field, value } => {
                write!(f, "RoomEQ field {field} must be positive, got {value}")
            }
            Self::InvalidRange { field, min, max } => {
                write!(f, "RoomEQ range {field} is invalid: min={min}, max={max}")
            }
            Self::ZeroCount { field } => write!(f, "RoomEQ count {field} must be non-zero"),
            Self::InvalidSampleRate { field, value } => {
                write!(
                    f,
                    "RoomEQ sample rate {field} must be positive, got {value}"
                )
            }
            Self::EmptyValue { field } => write!(f, "RoomEQ field {field} must not be empty"),
            Self::InvalidCombination { field } => {
                write!(f, "RoomEQ fields in {field} are mutually incompatible")
            }
        }
    }
}

impl std::error::Error for ConfigValidationError {}

/// Records the intentional schema changes applied while reading a RoomEQ
/// configuration document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomConfigMigrationReport {
    /// Version declared by the input document, or the implicit earliest
    /// version for legacy documents that had no version field.
    pub from_version: String,
    /// Version of the returned configuration.
    pub to_version: String,
    /// Ordered migration descriptions.
    pub steps: Vec<String>,
}

/// Failure while reading a versioned RoomEQ configuration document.
#[derive(Debug)]
pub enum RoomConfigReadError {
    /// The document is not valid JSON.
    Json(serde_json::Error),
    /// The JSON root is not an object.
    InvalidRoot,
    /// The version field is present but not a string.
    InvalidVersionField,
    /// No declared migration leads to the current schema.
    UnsupportedVersion { version: String },
    /// The migrated configuration does not satisfy its semantic contract.
    Validation(ConfigValidationError),
}

impl fmt::Display for RoomConfigReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "invalid RoomEQ configuration JSON: {error}"),
            Self::InvalidRoot => write!(f, "RoomEQ configuration JSON root must be an object"),
            Self::InvalidVersionField => {
                write!(
                    f,
                    "RoomEQ configuration version must be a string when present"
                )
            }
            Self::UnsupportedVersion { version } => write!(
                f,
                "RoomEQ configuration version {version:?} has no migration to {CURRENT_CONFIG_SCHEMA_VERSION}"
            ),
            Self::Validation(error) => write!(f, "invalid RoomEQ configuration: {error}"),
        }
    }
}

impl std::error::Error for RoomConfigReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Validation(error) => Some(error),
            Self::InvalidRoot | Self::InvalidVersionField | Self::UnsupportedVersion { .. } => None,
        }
    }
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), ConfigValidationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ConfigValidationError::NonFinite { field })
    }
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), ConfigValidationError> {
    validate_finite(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ConfigValidationError::NonPositive { field, value })
    }
}

fn validate_non_negative(field: &'static str, value: f64) -> Result<(), ConfigValidationError> {
    validate_finite(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ConfigValidationError::InvalidRange {
            field,
            min: 0.0,
            max: value,
        })
    }
}

fn validate_range(field: &'static str, min: f64, max: f64) -> Result<(), ConfigValidationError> {
    validate_finite(field, min)?;
    validate_finite(field, max)?;
    if max >= min {
        Ok(())
    } else {
        Err(ConfigValidationError::InvalidRange { field, min, max })
    }
}

fn validate_non_empty(field: &'static str, value: &str) -> Result<(), ConfigValidationError> {
    if value.trim().is_empty() {
        Err(ConfigValidationError::EmptyValue { field })
    } else {
        Ok(())
    }
}

fn validate_unit_interval(field: &'static str, value: f64) -> Result<(), ConfigValidationError> {
    validate_finite(field, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ConfigValidationError::InvalidRange {
            field,
            min: 0.0,
            max: value,
        })
    }
}

fn validate_weights(field: &'static str, weights: &[f64]) -> Result<(), ConfigValidationError> {
    if weights.is_empty() {
        return Err(ConfigValidationError::ZeroCount { field });
    }
    let mut sum = 0.0;
    for &weight in weights {
        validate_non_negative(field, weight)?;
        sum += weight;
    }
    validate_finite(field, sum)?;
    if sum > 0.0 {
        Ok(())
    } else {
        Err(ConfigValidationError::InvalidRange {
            field,
            min: 0.0,
            max: sum,
        })
    }
}

fn validate_frequency_list(
    field: &'static str,
    values: &[f64],
) -> Result<(), ConfigValidationError> {
    if values.is_empty() {
        return Err(ConfigValidationError::ZeroCount { field });
    }
    let mut previous = 0.0;
    for (index, &value) in values.iter().enumerate() {
        validate_positive(field, value)?;
        if index > 0 && value <= previous {
            return Err(ConfigValidationError::InvalidRange {
                field,
                min: previous,
                max: value,
            });
        }
        previous = value;
    }
    Ok(())
}

impl MeasurementRef {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        match self {
            Self::Path(path) => {
                if path.as_os_str().is_empty() {
                    Err(ConfigValidationError::EmptyValue {
                        field: "measurement.path",
                    })
                } else {
                    Ok(())
                }
            }
            Self::Named { path, name } => {
                if path.as_os_str().is_empty() {
                    return Err(ConfigValidationError::EmptyValue {
                        field: "measurement.path",
                    });
                }
                if let Some(name) = name {
                    validate_non_empty("measurement.name", name)?;
                }
                Ok(())
            }
        }
    }
}

impl MeasurementSource {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        match self {
            Self::Single(single) => {
                single.measurement.validate()?;
                if let Some(name) = &single.speaker_name {
                    validate_non_empty("measurement.speaker_name", name)?;
                }
            }
            Self::Multiple(multiple) => {
                if multiple.measurements.is_empty() {
                    return Err(ConfigValidationError::ZeroCount {
                        field: "measurement.measurements",
                    });
                }
                for measurement in &multiple.measurements {
                    measurement.validate()?;
                }
                if let Some(name) = &multiple.speaker_name {
                    validate_non_empty("measurement.speaker_name", name)?;
                }
            }
        }
        Ok(())
    }
}

impl SpeakerConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        match self {
            Self::Single(source) => source.validate(),
            Self::Group(group) => {
                validate_non_empty("speaker.name", &group.name)?;
                if group.measurements.is_empty() {
                    return Err(ConfigValidationError::ZeroCount {
                        field: "speaker.measurements",
                    });
                }
                for source in &group.measurements {
                    source.validate()?;
                }
                if let Some(name) = &group.speaker_name {
                    validate_non_empty("speaker.speaker_name", name)?;
                }
                if let Some(crossover) = &group.crossover {
                    validate_non_empty("speaker.crossover", crossover)?;
                }
                Ok(())
            }
            Self::MultiSub(group) => {
                validate_non_empty("speaker.name", &group.name)?;
                if group.subwoofers.is_empty() {
                    return Err(ConfigValidationError::ZeroCount {
                        field: "speaker.subwoofers",
                    });
                }
                for source in &group.subwoofers {
                    source.validate()?;
                }
                if let Some(name) = &group.speaker_name {
                    validate_non_empty("speaker.speaker_name", name)?;
                }
                Ok(())
            }
        }
    }
}

impl CrossoverConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_non_empty("crossover.type", &self.crossover_type)?;
        let definitions = usize::from(self.frequency.is_some())
            + usize::from(self.frequencies.is_some())
            + usize::from(self.frequency_range.is_some());
        if definitions == 0 {
            return Err(ConfigValidationError::EmptyValue {
                field: "crossover.frequency",
            });
        }
        if definitions > 1 {
            return Err(ConfigValidationError::InvalidCombination { field: "crossover" });
        }
        if let Some(frequency) = self.frequency {
            validate_positive("crossover.frequency", frequency)?;
        }
        if let Some(frequencies) = &self.frequencies {
            validate_frequency_list("crossover.frequencies", frequencies)?;
        }
        if let Some((min, max)) = self.frequency_range {
            validate_positive("crossover.frequency_range.min", min)?;
            validate_positive("crossover.frequency_range.max", max)?;
            validate_range("crossover.frequency_range", min, max)?;
        }
        Ok(())
    }
}

impl RoomDimensions {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_positive("room_dimensions.length", self.length)?;
        validate_positive("room_dimensions.width", self.width)?;
        validate_positive("room_dimensions.height", self.height)
    }
}

impl BassManagementConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_non_empty("bass_management.lfe_channel", &self.lfe_channel)?;
        validate_finite(
            "bass_management.lfe_playback_gain_db",
            self.lfe_playback_gain_db,
        )?;
        validate_finite("bass_management.sub_trim_db", self.sub_trim_db)?;
        validate_non_negative("bass_management.max_sub_boost_db", self.max_sub_boost_db)?;
        validate_non_negative(
            "bass_management.headroom_margin_db",
            self.headroom_margin_db,
        )?;
        for (group, crossover) in &self.group_crossovers {
            validate_non_empty("bass_management.group_crossovers.key", group)?;
            validate_non_empty("bass_management.group_crossovers.value", crossover)?;
        }
        Ok(())
    }
}

impl SubwooferSystemConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if let Some(crossover) = &self.crossover {
            validate_non_empty("system.subwoofers.crossover", crossover)?;
        }
        if self.mapping.is_empty() {
            return Err(ConfigValidationError::ZeroCount {
                field: "system.subwoofers.mapping",
            });
        }
        for (subwoofer, speaker) in &self.mapping {
            validate_non_empty("system.subwoofers.mapping.key", subwoofer)?;
            validate_non_empty("system.subwoofers.mapping.value", speaker)?;
        }
        Ok(())
    }
}

impl SystemConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.speakers.is_empty() {
            return Err(ConfigValidationError::ZeroCount {
                field: "system.speakers",
            });
        }
        for (role, speaker) in &self.speakers {
            validate_non_empty("system.speakers.key", role)?;
            validate_non_empty("system.speakers.value", speaker)?;
        }
        if let Some(subwoofers) = &self.subwoofers {
            subwoofers.validate()?;
        }
        if let Some(bass_management) = &self.bass_management {
            bass_management.validate()?;
        }
        Ok(())
    }
}

// ============================================================================
// Measurement references
// ============================================================================

/// Reference to a measurement file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MeasurementRef {
    /// Path to CSV file (freq, spl, phase columns).
    Path(PathBuf),
    /// Named measurement with optional metadata.
    Named {
        /// Path to the CSV measurement file.
        path: PathBuf,
        /// Optional display name for the measurement.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}

/// Source of measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MeasurementSource {
    /// A single measurement file with optional speaker name.
    Single(MeasurementSingle),
    /// Multiple measurement files to be averaged.
    Multiple(MeasurementMultiple),
}

/// Single measurement with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementSingle {
    pub measurement: MeasurementRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
}

/// Multiple measurements with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementMultiple {
    pub measurements: Vec<MeasurementRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
}

// ============================================================================
// Speaker configuration
// ============================================================================

/// Speaker configuration (single, group, or multi-sub).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SpeakerConfig {
    /// Group of measurements (multi-driver case).
    Group(SpeakerGroup),
    /// Multiple subwoofers optimization.
    MultiSub(MultiSubGroup),
    /// Single channel.
    Single(MeasurementSource),
}

/// Group of measurements for a single speaker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerGroup {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
    pub measurements: Vec<MeasurementSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crossover: Option<String>,
}

/// Multiple subwoofers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSubGroup {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
    pub subwoofers: Vec<MeasurementSource>,
    #[serde(default)]
    pub allpass_optimization: bool,
}

// ============================================================================
// System configuration
// ============================================================================

/// System topology model.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SystemModel {
    #[default]
    Stereo,
    HomeCinema,
}

/// Subwoofer optimization strategy.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SubwooferStrategy {
    #[default]
    Single,
    Mso,
}

/// Processing mode for the optimization engine.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingMode {
    #[default]
    LowLatency,
    PhaseLinear,
    Hybrid,
}

/// Strategy for handling multiple measurements per speaker.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MultiMeasurementStrategy {
    #[default]
    Average,
    WeightedSum,
    Minimax,
    VariancePenalized,
    SpatialRobustness,
    MinimaxUncertainty,
}

/// Subwoofer system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubwooferSystemConfig {
    #[serde(default)]
    pub config: SubwooferStrategy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crossover: Option<String>,
    #[serde(flatten)]
    pub mapping: HashMap<String, String>,
}

/// Explicit system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    #[serde(default)]
    pub model: SystemModel,
    pub speakers: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subwoofers: Option<SubwooferSystemConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bass_management: Option<BassManagementConfig>,
}

/// Bass management configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BassManagementConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub redirect_bass: bool,
    #[serde(default = "default_lfe_channel")]
    pub lfe_channel: String,
    #[serde(default = "default_lfe_playback_gain_db")]
    pub lfe_playback_gain_db: f64,
    #[serde(default)]
    pub apply_lfe_gain_to_chain: bool,
    #[serde(default)]
    pub sub_trim_db: f64,
    #[serde(default = "default_max_sub_boost_db")]
    pub max_sub_boost_db: f64,
    #[serde(default = "default_headroom_margin_db")]
    pub headroom_margin_db: f64,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub group_crossovers: HashMap<String, String>,
    #[serde(default = "default_true")]
    pub optimize_groups: bool,
}

fn default_true() -> bool {
    true
}
fn default_lfe_channel() -> String {
    "LFE".to_string()
}
fn default_lfe_playback_gain_db() -> f64 {
    10.0
}
fn default_max_sub_boost_db() -> f64 {
    6.0
}
fn default_headroom_margin_db() -> f64 {
    6.0
}

impl Default for BassManagementConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            redirect_bass: true,
            lfe_channel: default_lfe_channel(),
            lfe_playback_gain_db: default_lfe_playback_gain_db(),
            apply_lfe_gain_to_chain: false,
            sub_trim_db: 0.0,
            max_sub_boost_db: default_max_sub_boost_db(),
            headroom_margin_db: default_headroom_margin_db(),
            group_crossovers: HashMap::new(),
            optimize_groups: true,
        }
    }
}

// ============================================================================
// Crossover & target
// ============================================================================

/// Crossover configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossoverConfig {
    #[serde(rename = "type")]
    pub crossover_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequencies: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_range: Option<(f64, f64)>,
}

// ============================================================================
// Optimizer configuration
// ============================================================================

/// Multi-measurement configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiMeasurementConfig {
    #[serde(default)]
    pub strategy: MultiMeasurementStrategy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<Vec<f64>>,
    #[serde(default = "default_variance_lambda")]
    pub variance_lambda: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_robustness: Option<SpatialRobustnessConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_uncertainty: Option<BootstrapUncertaintyConfig>,
}

fn default_variance_lambda() -> f64 {
    1.0
}

impl Default for MultiMeasurementConfig {
    fn default() -> Self {
        Self {
            strategy: MultiMeasurementStrategy::default(),
            weights: None,
            variance_lambda: default_variance_lambda(),
            spatial_robustness: None,
            bootstrap_uncertainty: None,
        }
    }
}

/// Spatial robustness configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialRobustnessConfig {
    #[serde(default = "default_variance_threshold")]
    pub variance_threshold_db: f64,
    #[serde(default = "default_transition_width")]
    pub transition_width_db: f64,
    #[serde(default = "default_min_correction_depth")]
    pub min_correction_depth: f64,
    #[serde(default = "default_mask_smoothing_octaves")]
    pub mask_smoothing_octaves: f64,
}

fn default_variance_threshold() -> f64 {
    3.0
}
fn default_transition_width() -> f64 {
    2.0
}
fn default_min_correction_depth() -> f64 {
    0.1
}
fn default_mask_smoothing_octaves() -> f64 {
    1.0 / 6.0
}

impl Default for SpatialRobustnessConfig {
    fn default() -> Self {
        Self {
            variance_threshold_db: default_variance_threshold(),
            transition_width_db: default_transition_width(),
            min_correction_depth: default_min_correction_depth(),
            mask_smoothing_octaves: default_mask_smoothing_octaves(),
        }
    }
}

/// Bootstrap scalarisation strategy.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapScalarisation {
    #[default]
    WorstCase,
    Cvar,
}

/// Bootstrap uncertainty configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BootstrapUncertaintyConfig {
    #[serde(default = "default_bootstrap_num_resamples")]
    pub num_resamples: usize,
    #[serde(default = "default_bootstrap_alpha")]
    pub alpha: f64,
    #[serde(default = "default_bootstrap_seed")]
    pub seed: u64,
    #[serde(default)]
    pub scalarisation: BootstrapScalarisation,
    #[serde(default = "default_bootstrap_cvar_alpha")]
    pub cvar_alpha: f64,
}

fn default_bootstrap_num_resamples() -> usize {
    400
}
fn default_bootstrap_alpha() -> f64 {
    0.10
}
fn default_bootstrap_seed() -> u64 {
    0xC0FFEE
}
fn default_bootstrap_cvar_alpha() -> f64 {
    0.20
}

impl Default for BootstrapUncertaintyConfig {
    fn default() -> Self {
        Self {
            num_resamples: default_bootstrap_num_resamples(),
            alpha: default_bootstrap_alpha(),
            seed: default_bootstrap_seed(),
            scalarisation: BootstrapScalarisation::default(),
            cvar_alpha: default_bootstrap_cvar_alpha(),
        }
    }
}

/// CEA2034 correction mode.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Cea2034CorrectionMode {
    Flat,
    Score,
    #[default]
    Auto,
}

/// CEA2034 speaker pre-correction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cea2034CorrectionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
    #[serde(default = "default_cea2034_version")]
    pub version: String,
    #[serde(default)]
    pub correction_mode: Cea2034CorrectionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listening_distance_m: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_latency_ms: Option<f64>,
    #[serde(default = "default_nearfield_threshold")]
    pub nearfield_threshold_m: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_freq: Option<f64>,
    #[serde(default = "default_cea2034_num_filters")]
    pub num_filters: usize,
    #[serde(default = "default_cea2034_max_q")]
    pub max_q: f64,
    #[serde(default = "default_cea2034_max_db")]
    pub max_db: f64,
    #[serde(default = "default_cea2034_min_db")]
    pub min_db: f64,
}

fn default_cea2034_version() -> String {
    "asr".to_string()
}
fn default_nearfield_threshold() -> f64 {
    2.0
}
fn default_cea2034_num_filters() -> usize {
    5
}
fn default_cea2034_max_q() -> f64 {
    3.0
}
fn default_cea2034_max_db() -> f64 {
    3.0
}
fn default_cea2034_min_db() -> f64 {
    -12.0
}

impl Default for Cea2034CorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            speaker_name: None,
            version: default_cea2034_version(),
            correction_mode: Cea2034CorrectionMode::default(),
            listening_distance_m: None,
            system_latency_ms: None,
            nearfield_threshold_m: default_nearfield_threshold(),
            min_freq: None,
            num_filters: default_cea2034_num_filters(),
            max_q: default_cea2034_max_q(),
            max_db: default_cea2034_max_db(),
            min_db: default_cea2034_min_db(),
        }
    }
}

/// Target shape preset.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TargetShape {
    #[default]
    Flat,
    Harman,
    Custom,
    File,
    FromMeasurement,
}

/// User preference adjustments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreference {
    #[serde(default)]
    pub bass_shelf_db: f64,
    #[serde(default = "default_bass_shelf_freq")]
    pub bass_shelf_freq: f64,
    #[serde(default)]
    pub treble_shelf_db: f64,
    #[serde(default = "default_treble_shelf_freq")]
    pub treble_shelf_freq: f64,
}

fn default_bass_shelf_freq() -> f64 {
    200.0
}
fn default_treble_shelf_freq() -> f64 {
    8000.0
}

impl Default for UserPreference {
    fn default() -> Self {
        Self {
            bass_shelf_db: 0.0,
            bass_shelf_freq: default_bass_shelf_freq(),
            treble_shelf_db: 0.0,
            treble_shelf_freq: default_treble_shelf_freq(),
        }
    }
}

/// Target response configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetResponseConfig {
    #[serde(default)]
    pub shape: TargetShape,
    #[serde(default = "default_tilt_slope")]
    pub slope_db_per_octave: f64,
    #[serde(default = "default_tilt_reference_freq")]
    pub reference_freq: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curve_path: Option<PathBuf>,
    #[serde(default)]
    pub preference: UserPreference,
    #[serde(default)]
    pub broadband_precorrection: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role_targets: Option<RoleTargetConfig>,
}

fn default_tilt_slope() -> f64 {
    -0.8
}
fn default_tilt_reference_freq() -> f64 {
    1000.0
}

impl Default for TargetResponseConfig {
    fn default() -> Self {
        Self {
            shape: TargetShape::Flat,
            slope_db_per_octave: default_tilt_slope(),
            reference_freq: default_tilt_reference_freq(),
            curve_path: None,
            preference: UserPreference::default(),
            broadband_precorrection: false,
            role_targets: None,
        }
    }
}

/// Role-aware target adjustments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleTargetConfig {
    #[serde(default = "default_role_targets_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub front_slope_offset_db_per_octave: f64,
    #[serde(default)]
    pub center_slope_offset_db_per_octave: f64,
    #[serde(default)]
    pub surround_slope_offset_db_per_octave: f64,
    #[serde(default)]
    pub height_slope_offset_db_per_octave: f64,
    #[serde(default)]
    pub subwoofer_slope_offset_db_per_octave: f64,
    #[serde(default)]
    pub lfe_slope_offset_db_per_octave: f64,
    #[serde(default)]
    pub center_treble_shelf_db: f64,
    #[serde(default)]
    pub surround_treble_shelf_db: f64,
    #[serde(default)]
    pub height_treble_shelf_db: f64,
    #[serde(default)]
    pub subwoofer_bass_shelf_db: f64,
    #[serde(default)]
    pub lfe_bass_shelf_db: f64,
    #[serde(default)]
    pub center_dialog_boost_db: f64,
    #[serde(default = "default_center_dialog_low_hz")]
    pub center_dialog_low_hz: f64,
    #[serde(default = "default_center_dialog_high_hz")]
    pub center_dialog_high_hz: f64,
    #[serde(default)]
    pub cinema_x_curve_enabled: bool,
    #[serde(default)]
    pub cinema_x_curve_db_per_octave: f64,
    #[serde(default = "default_cinema_x_curve_start_hz")]
    pub cinema_x_curve_start_hz: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listening_distance_m: Option<f64>,
    #[serde(default = "default_cinema_reference_distance_m")]
    pub cinema_reference_distance_m: f64,
    #[serde(default)]
    pub distance_treble_rolloff_db_per_doubling: f64,
}

fn default_role_targets_enabled() -> bool {
    true
}
fn default_center_dialog_low_hz() -> f64 {
    300.0
}
fn default_center_dialog_high_hz() -> f64 {
    4000.0
}
fn default_cinema_x_curve_start_hz() -> f64 {
    2000.0
}
fn default_cinema_reference_distance_m() -> f64 {
    3.0
}

impl Default for RoleTargetConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            front_slope_offset_db_per_octave: 0.0,
            center_slope_offset_db_per_octave: 0.0,
            surround_slope_offset_db_per_octave: 0.0,
            height_slope_offset_db_per_octave: 0.0,
            subwoofer_slope_offset_db_per_octave: 0.0,
            lfe_slope_offset_db_per_octave: 0.0,
            center_treble_shelf_db: 0.0,
            surround_treble_shelf_db: 0.0,
            height_treble_shelf_db: 0.0,
            subwoofer_bass_shelf_db: 0.0,
            lfe_bass_shelf_db: 0.0,
            center_dialog_boost_db: 0.0,
            center_dialog_low_hz: default_center_dialog_low_hz(),
            center_dialog_high_hz: default_center_dialog_high_hz(),
            cinema_x_curve_enabled: false,
            cinema_x_curve_db_per_octave: 0.0,
            cinema_x_curve_start_hz: default_cinema_x_curve_start_hz(),
            listening_distance_m: None,
            cinema_reference_distance_m: default_cinema_reference_distance_m(),
            distance_treble_rolloff_db_per_doubling: 0.0,
        }
    }
}

/// FIR filter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirConfig {
    #[serde(default = "default_fir_taps")]
    pub taps: usize,
    #[serde(default = "default_fir_phase")]
    pub phase: String,
    #[serde(default)]
    pub correct_excess_phase: bool,
    #[serde(default = "default_phase_smoothing")]
    pub phase_smoothing: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_ringing: Option<PreRingingConfig>,
}

fn default_fir_taps() -> usize {
    4096
}
fn default_fir_phase() -> String {
    "kirkeby".to_string()
}
fn default_phase_smoothing() -> f64 {
    0.167
}

impl Default for FirConfig {
    fn default() -> Self {
        Self {
            taps: default_fir_taps(),
            phase: default_fir_phase(),
            correct_excess_phase: false,
            phase_smoothing: default_phase_smoothing(),
            pre_ringing: None,
        }
    }
}

/// Pre-ringing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreRingingConfig {
    #[serde(default = "default_pre_ringing_threshold")]
    pub threshold_db: f64,
    #[serde(default = "default_pre_ringing_time")]
    pub max_time_s: f64,
}

fn default_pre_ringing_threshold() -> f64 {
    -30.0
}
fn default_pre_ringing_time() -> f64 {
    0.005
}

impl Default for PreRingingConfig {
    fn default() -> Self {
        Self {
            threshold_db: default_pre_ringing_threshold(),
            max_time_s: default_pre_ringing_time(),
        }
    }
}

/// Mixed-phase correction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPhaseConfig {
    #[serde(default = "default_mixed_phase_fir_length")]
    pub max_fir_length_ms: f64,
    #[serde(default = "default_pre_ringing_threshold")]
    pub pre_ringing_threshold_db: f64,
    #[serde(default = "default_mixed_phase_spatial_depth")]
    pub min_spatial_depth: f64,
    #[serde(default = "default_mask_smoothing")]
    pub phase_smoothing_octaves: f64,
}

fn default_mixed_phase_fir_length() -> f64 {
    10.0
}
fn default_mixed_phase_spatial_depth() -> f64 {
    0.5
}
fn default_mask_smoothing() -> f64 {
    1.0 / 6.0
}

impl Default for MixedPhaseConfig {
    fn default() -> Self {
        Self {
            max_fir_length_ms: default_mixed_phase_fir_length(),
            pre_ringing_threshold_db: default_pre_ringing_threshold(),
            min_spatial_depth: default_mixed_phase_spatial_depth(),
            phase_smoothing_octaves: default_mask_smoothing(),
        }
    }
}

/// Mixed mode configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedModeConfig {
    #[serde(default = "default_crossover_freq")]
    pub crossover_freq: f64,
    #[serde(default = "default_crossover_type")]
    pub crossover_type: String,
    #[serde(default = "default_fir_band")]
    pub fir_band: String,
}

fn default_crossover_freq() -> f64 {
    300.0
}
fn default_crossover_type() -> String {
    "LR24".to_string()
}
fn default_fir_band() -> String {
    "low".to_string()
}

impl Default for MixedModeConfig {
    fn default() -> Self {
        Self {
            crossover_freq: default_crossover_freq(),
            crossover_type: default_crossover_type(),
            fir_band: default_fir_band(),
        }
    }
}

/// Schroeder split configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchroederSplitConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_schroeder_freq")]
    pub schroeder_freq: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub room_dimensions: Option<RoomDimensions>,
    #[serde(default)]
    pub low_freq_config: LowFreqFilterConfig,
    #[serde(default)]
    pub high_freq_config: HighFreqFilterConfig,
}

fn default_schroeder_freq() -> f64 {
    300.0
}

impl Default for SchroederSplitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            schroeder_freq: default_schroeder_freq(),
            room_dimensions: None,
            low_freq_config: LowFreqFilterConfig::default(),
            high_freq_config: HighFreqFilterConfig::default(),
        }
    }
}

/// Room dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomDimensions {
    pub length: f64,
    pub width: f64,
    pub height: f64,
}

/// Low frequency filter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowFreqFilterConfig {
    #[serde(default = "default_low_freq_max_q")]
    pub max_q: f64,
    #[serde(default = "default_min_q")]
    pub min_q: f64,
    #[serde(default)]
    pub allow_boost: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_db: Option<f64>,
}

fn default_low_freq_max_q() -> f64 {
    5.0
}
fn default_min_q() -> f64 {
    0.5
}

impl Default for LowFreqFilterConfig {
    fn default() -> Self {
        Self {
            max_q: default_low_freq_max_q(),
            min_q: default_min_q(),
            allow_boost: false,
            max_db: None,
        }
    }
}

/// High frequency filter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighFreqFilterConfig {
    #[serde(default = "default_high_freq_max_q")]
    pub max_q: f64,
    #[serde(default)]
    pub shelving_only: bool,
}

fn default_high_freq_max_q() -> f64 {
    1.0
}

impl Default for HighFreqFilterConfig {
    fn default() -> Self {
        Self {
            max_q: default_high_freq_max_q(),
            shelving_only: false,
        }
    }
}

/// Auto-optimizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoOptimizerConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub filter_count: bool,
    #[serde(default = "default_true")]
    pub q_bounds: bool,
    #[serde(default = "default_true")]
    pub gain_bounds: bool,
    #[serde(default = "default_auto_min_filters")]
    pub min_filters: usize,
    #[serde(default = "default_auto_max_filters")]
    pub max_filters: usize,
}

fn default_auto_min_filters() -> usize {
    1
}
fn default_auto_max_filters() -> usize {
    12
}

impl Default for AutoOptimizerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            filter_count: true,
            q_bounds: true,
            gain_bounds: true,
            min_filters: default_auto_min_filters(),
            max_filters: default_auto_max_filters(),
        }
    }
}

/// Phase alignment configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseAlignmentConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_phase_min_freq")]
    pub min_freq: f64,
    #[serde(default = "default_phase_max_freq")]
    pub max_freq: f64,
    #[serde(default = "default_true")]
    pub optimize_polarity: bool,
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: f64,
}

fn default_phase_min_freq() -> f64 {
    60.0
}
fn default_phase_max_freq() -> f64 {
    100.0
}
fn default_max_delay_ms() -> f64 {
    3.0
}

impl Default for PhaseAlignmentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_freq: default_phase_min_freq(),
            max_freq: default_phase_max_freq(),
            optimize_polarity: true,
            max_delay_ms: default_max_delay_ms(),
        }
    }
}

/// Multi-seat configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSeatConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub strategy: MultiSeatStrategy,
    #[serde(default)]
    pub primary_seat: usize,
    #[serde(default = "default_max_deviation_db")]
    pub max_deviation_db: f64,
    #[serde(default)]
    pub optimize_polarity: bool,
    #[serde(default)]
    pub allpass_filters_per_sub: usize,
    #[serde(default = "default_true")]
    pub per_sub_peq: bool,
    #[serde(default = "default_true")]
    pub global_eq: bool,
    #[serde(default = "default_all_channel_multiseat_enabled")]
    pub all_channel_enabled: bool,
    #[serde(default = "default_all_channel_multiseat_strategy")]
    pub all_channel_strategy: MultiMeasurementStrategy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seat_weights: Option<Vec<f64>>,
    #[serde(default = "default_primary_seat_weight")]
    pub primary_seat_weight: f64,
}

fn default_max_deviation_db() -> f64 {
    6.0
}
fn default_all_channel_multiseat_enabled() -> bool {
    true
}
fn default_all_channel_multiseat_strategy() -> MultiMeasurementStrategy {
    MultiMeasurementStrategy::SpatialRobustness
}
fn default_primary_seat_weight() -> f64 {
    2.0
}

impl Default for MultiSeatConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: MultiSeatStrategy::MinimizeVariance,
            primary_seat: 0,
            max_deviation_db: default_max_deviation_db(),
            optimize_polarity: false,
            allpass_filters_per_sub: 0,
            per_sub_peq: true,
            global_eq: true,
            all_channel_enabled: true,
            all_channel_strategy: default_all_channel_multiseat_strategy(),
            seat_weights: None,
            primary_seat_weight: default_primary_seat_weight(),
        }
    }
}

/// Multi-seat strategy.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MultiSeatStrategy {
    #[default]
    MinimizeVariance,
    PrimaryWithConstraints,
    Average,
    ModalBasis,
    ContinuousArea,
}

/// Channel matching configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMatchingConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_channel_matching_threshold")]
    pub threshold_db: f64,
    #[serde(default = "default_channel_matching_max_filters")]
    pub max_filters: usize,
}

fn default_channel_matching_threshold() -> f64 {
    0.75
}
fn default_channel_matching_max_filters() -> usize {
    5
}

impl Default for ChannelMatchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_db: default_channel_matching_threshold(),
            max_filters: default_channel_matching_max_filters(),
        }
    }
}

/// Group delay optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupDelayOptimizationConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_gd_max_delay_ms")]
    pub max_delay_ms: f64,
    #[serde(default = "default_gd_ap_per_channel")]
    pub ap_per_channel: usize,
    #[serde(default = "default_gd_ap_min_q")]
    pub ap_min_q: f64,
    #[serde(default = "default_gd_ap_max_q")]
    pub ap_max_q: f64,
    #[serde(default = "default_true")]
    pub optimize_polarity: bool,
    #[serde(default = "default_gd_coherence_threshold")]
    pub coherence_threshold: f64,
    #[serde(default = "default_gd_min_improvement_db")]
    pub min_improvement_db: f64,
    #[serde(default = "default_gd_max_iter")]
    pub max_iter: usize,
    #[serde(default = "default_gd_popsize")]
    pub popsize: usize,
    #[serde(default = "default_gd_tol")]
    pub tol: f64,
    #[serde(default = "default_true")]
    pub adaptive_allpass: bool,
}

fn default_gd_max_delay_ms() -> f64 {
    20.0
}
fn default_gd_ap_per_channel() -> usize {
    2
}
fn default_gd_ap_min_q() -> f64 {
    0.3
}
fn default_gd_ap_max_q() -> f64 {
    10.0
}
fn default_gd_coherence_threshold() -> f64 {
    0.8
}
fn default_gd_min_improvement_db() -> f64 {
    1.0
}
fn default_gd_max_iter() -> usize {
    2000
}
fn default_gd_popsize() -> usize {
    20
}
fn default_gd_tol() -> f64 {
    1e-8
}

impl Default for GroupDelayOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_delay_ms: default_gd_max_delay_ms(),
            ap_per_channel: default_gd_ap_per_channel(),
            ap_min_q: default_gd_ap_min_q(),
            ap_max_q: default_gd_ap_max_q(),
            optimize_polarity: true,
            coherence_threshold: default_gd_coherence_threshold(),
            min_improvement_db: default_gd_min_improvement_db(),
            max_iter: default_gd_max_iter(),
            popsize: default_gd_popsize(),
            tol: default_gd_tol(),
            adaptive_allpass: true,
        }
    }
}

/// Subwoofer-specific optimizer overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubOptimizerConfig {
    #[serde(default = "default_sub_num_filters")]
    pub num_filters: usize,
    #[serde(default = "default_sub_max_db")]
    pub max_db: f64,
    #[serde(default = "default_sub_min_db")]
    pub min_db: f64,
    #[serde(default = "default_min_q")]
    pub min_q: f64,
    #[serde(default = "default_sub_max_q")]
    pub max_q: f64,
}

fn default_sub_num_filters() -> usize {
    10
}
fn default_sub_max_db() -> f64 {
    18.0
}
fn default_sub_min_db() -> f64 {
    -18.0
}
fn default_sub_max_q() -> f64 {
    10.0
}

impl Default for SubOptimizerConfig {
    fn default() -> Self {
        Self {
            num_filters: default_sub_num_filters(),
            max_db: default_sub_max_db(),
            min_db: default_sub_min_db(),
            min_q: default_min_q(),
            max_q: default_sub_max_q(),
        }
    }
}

impl MultiMeasurementConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_non_negative("multi_measurement.variance_lambda", self.variance_lambda)?;
        if let Some(weights) = &self.weights {
            validate_weights("multi_measurement.weights", weights)?;
        }
        if let Some(spatial) = &self.spatial_robustness {
            spatial.validate()?;
        }
        if let Some(bootstrap) = &self.bootstrap_uncertainty {
            bootstrap.validate()?;
        }
        Ok(())
    }
}

impl SpatialRobustnessConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_non_negative(
            "spatial_robustness.variance_threshold_db",
            self.variance_threshold_db,
        )?;
        validate_positive(
            "spatial_robustness.transition_width_db",
            self.transition_width_db,
        )?;
        validate_non_negative(
            "spatial_robustness.min_correction_depth",
            self.min_correction_depth,
        )?;
        validate_positive(
            "spatial_robustness.mask_smoothing_octaves",
            self.mask_smoothing_octaves,
        )
    }
}

impl BootstrapUncertaintyConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.num_resamples == 0 {
            return Err(ConfigValidationError::ZeroCount {
                field: "bootstrap.num_resamples",
            });
        }
        validate_unit_interval("bootstrap.alpha", self.alpha)?;
        validate_unit_interval("bootstrap.cvar_alpha", self.cvar_alpha)
    }
}

impl Cea2034CorrectionConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_non_empty("cea2034.version", &self.version)?;
        if let Some(name) = &self.speaker_name {
            validate_non_empty("cea2034.speaker_name", name)?;
        }
        if let Some(distance) = self.listening_distance_m {
            validate_positive("cea2034.listening_distance_m", distance)?;
        }
        if let Some(latency) = self.system_latency_ms {
            validate_non_negative("cea2034.system_latency_ms", latency)?;
        }
        validate_positive("cea2034.nearfield_threshold_m", self.nearfield_threshold_m)?;
        if let Some(min_freq) = self.min_freq {
            validate_positive("cea2034.min_freq", min_freq)?;
        }
        if self.num_filters == 0 {
            return Err(ConfigValidationError::ZeroCount {
                field: "cea2034.num_filters",
            });
        }
        validate_positive("cea2034.max_q", self.max_q)?;
        validate_finite("cea2034.max_db", self.max_db)?;
        validate_finite("cea2034.min_db", self.min_db)?;
        validate_range("cea2034.db", self.min_db, self.max_db)
    }
}

impl UserPreference {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_finite("target.preference.bass_shelf_db", self.bass_shelf_db)?;
        validate_positive("target.preference.bass_shelf_freq", self.bass_shelf_freq)?;
        validate_finite("target.preference.treble_shelf_db", self.treble_shelf_db)?;
        validate_positive(
            "target.preference.treble_shelf_freq",
            self.treble_shelf_freq,
        )
    }
}

impl RoleTargetConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        let finite_fields = [
            (
                "target.role.front_slope_offset_db_per_octave",
                self.front_slope_offset_db_per_octave,
            ),
            (
                "target.role.center_slope_offset_db_per_octave",
                self.center_slope_offset_db_per_octave,
            ),
            (
                "target.role.surround_slope_offset_db_per_octave",
                self.surround_slope_offset_db_per_octave,
            ),
            (
                "target.role.height_slope_offset_db_per_octave",
                self.height_slope_offset_db_per_octave,
            ),
            (
                "target.role.subwoofer_slope_offset_db_per_octave",
                self.subwoofer_slope_offset_db_per_octave,
            ),
            (
                "target.role.lfe_slope_offset_db_per_octave",
                self.lfe_slope_offset_db_per_octave,
            ),
            (
                "target.role.center_treble_shelf_db",
                self.center_treble_shelf_db,
            ),
            (
                "target.role.surround_treble_shelf_db",
                self.surround_treble_shelf_db,
            ),
            (
                "target.role.height_treble_shelf_db",
                self.height_treble_shelf_db,
            ),
            (
                "target.role.subwoofer_bass_shelf_db",
                self.subwoofer_bass_shelf_db,
            ),
            ("target.role.lfe_bass_shelf_db", self.lfe_bass_shelf_db),
            (
                "target.role.center_dialog_boost_db",
                self.center_dialog_boost_db,
            ),
            (
                "target.role.cinema_x_curve_db_per_octave",
                self.cinema_x_curve_db_per_octave,
            ),
            (
                "target.role.distance_treble_rolloff_db_per_doubling",
                self.distance_treble_rolloff_db_per_doubling,
            ),
        ];
        for (field, value) in finite_fields {
            validate_finite(field, value)?;
        }
        validate_positive(
            "target.role.center_dialog_low_hz",
            self.center_dialog_low_hz,
        )?;
        validate_positive(
            "target.role.center_dialog_high_hz",
            self.center_dialog_high_hz,
        )?;
        validate_range(
            "target.role.center_dialog_hz",
            self.center_dialog_low_hz,
            self.center_dialog_high_hz,
        )?;
        validate_positive(
            "target.role.cinema_x_curve_start_hz",
            self.cinema_x_curve_start_hz,
        )?;
        if let Some(distance) = self.listening_distance_m {
            validate_positive("target.role.listening_distance_m", distance)?;
        }
        validate_positive(
            "target.role.cinema_reference_distance_m",
            self.cinema_reference_distance_m,
        )
    }
}

impl TargetResponseConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_finite("target.slope_db_per_octave", self.slope_db_per_octave)?;
        validate_positive("target.reference_freq", self.reference_freq)?;
        self.preference.validate()?;
        if let Some(role_targets) = &self.role_targets {
            role_targets.validate()?;
        }
        Ok(())
    }
}

impl PreRingingConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_finite("fir.pre_ringing.threshold_db", self.threshold_db)?;
        validate_positive("fir.pre_ringing.max_time_s", self.max_time_s)
    }
}

impl FirConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.taps == 0 {
            return Err(ConfigValidationError::ZeroCount { field: "fir.taps" });
        }
        validate_non_empty("fir.phase", &self.phase)?;
        validate_non_negative("fir.phase_smoothing", self.phase_smoothing)?;
        if let Some(pre_ringing) = &self.pre_ringing {
            pre_ringing.validate()?;
        }
        Ok(())
    }
}

impl MixedPhaseConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_positive("mixed_phase.max_fir_length_ms", self.max_fir_length_ms)?;
        validate_finite(
            "mixed_phase.pre_ringing_threshold_db",
            self.pre_ringing_threshold_db,
        )?;
        validate_non_negative("mixed_phase.min_spatial_depth", self.min_spatial_depth)?;
        validate_positive(
            "mixed_phase.phase_smoothing_octaves",
            self.phase_smoothing_octaves,
        )
    }
}

impl MixedModeConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_positive("mixed_mode.crossover_freq", self.crossover_freq)?;
        validate_non_empty("mixed_mode.crossover_type", &self.crossover_type)?;
        validate_non_empty("mixed_mode.fir_band", &self.fir_band)
    }
}

impl LowFreqFilterConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_positive("schroeder.low_freq.min_q", self.min_q)?;
        validate_positive("schroeder.low_freq.max_q", self.max_q)?;
        validate_range("schroeder.low_freq.q", self.min_q, self.max_q)?;
        if let Some(max_db) = self.max_db {
            validate_non_negative("schroeder.low_freq.max_db", max_db)?;
        }
        Ok(())
    }
}

impl HighFreqFilterConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_positive("schroeder.high_freq.max_q", self.max_q)
    }
}

impl SchroederSplitConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_positive("schroeder.schroeder_freq", self.schroeder_freq)?;
        if let Some(dimensions) = &self.room_dimensions {
            dimensions.validate()?;
        }
        self.low_freq_config.validate()?;
        self.high_freq_config.validate()
    }
}

impl AutoOptimizerConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.min_filters == 0 || self.max_filters == 0 {
            return Err(ConfigValidationError::ZeroCount {
                field: "auto_optimizer.filters",
            });
        }
        if self.max_filters < self.min_filters {
            return Err(ConfigValidationError::InvalidRange {
                field: "auto_optimizer.filters",
                min: self.min_filters as f64,
                max: self.max_filters as f64,
            });
        }
        Ok(())
    }
}

impl PhaseAlignmentConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_positive("phase_alignment.min_freq", self.min_freq)?;
        validate_positive("phase_alignment.max_freq", self.max_freq)?;
        validate_range("phase_alignment.frequency", self.min_freq, self.max_freq)?;
        validate_non_negative("phase_alignment.max_delay_ms", self.max_delay_ms)
    }
}

impl MultiSeatConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_non_negative("multi_seat.max_deviation_db", self.max_deviation_db)?;
        validate_positive("multi_seat.primary_seat_weight", self.primary_seat_weight)?;
        if let Some(weights) = &self.seat_weights {
            validate_weights("multi_seat.seat_weights", weights)?;
        }
        Ok(())
    }
}

impl ChannelMatchingConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_non_negative("channel_matching.threshold_db", self.threshold_db)?;
        if self.max_filters == 0 {
            return Err(ConfigValidationError::ZeroCount {
                field: "channel_matching.max_filters",
            });
        }
        Ok(())
    }
}

impl GroupDelayOptimizationConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_non_negative("group_delay.max_delay_ms", self.max_delay_ms)?;
        if self.ap_per_channel == 0 || self.max_iter == 0 || self.popsize == 0 {
            return Err(ConfigValidationError::ZeroCount {
                field: "group_delay.count",
            });
        }
        validate_positive("group_delay.ap_min_q", self.ap_min_q)?;
        validate_positive("group_delay.ap_max_q", self.ap_max_q)?;
        validate_range("group_delay.q", self.ap_min_q, self.ap_max_q)?;
        validate_unit_interval("group_delay.coherence_threshold", self.coherence_threshold)?;
        validate_finite("group_delay.min_improvement_db", self.min_improvement_db)?;
        validate_positive("group_delay.tol", self.tol)
    }
}

impl SubOptimizerConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.num_filters == 0 {
            return Err(ConfigValidationError::ZeroCount {
                field: "sub_optimizer.num_filters",
            });
        }
        validate_finite("sub_optimizer.max_db", self.max_db)?;
        validate_finite("sub_optimizer.min_db", self.min_db)?;
        validate_range("sub_optimizer.db", self.min_db, self.max_db)?;
        validate_positive("sub_optimizer.min_q", self.min_q)?;
        validate_positive("sub_optimizer.max_q", self.max_q)?;
        validate_range("sub_optimizer.q", self.min_q, self.max_q)
    }
}

// ============================================================================
// Main OptimizerConfig
// ============================================================================

fn default_loss_type() -> String {
    "flat".to_string()
}
fn default_algorithm() -> String {
    "cobyla".to_string()
}
fn default_strategy() -> String {
    "currenttobest1bin".to_string()
}
fn default_num_filters() -> usize {
    7
}
fn default_min_filter_improvement() -> f64 {
    0.01
}
fn default_elimination_threshold() -> f64 {
    0.01
}
fn default_max_q() -> f64 {
    10.0
}
fn default_max_db() -> f64 {
    12.0
}
fn default_min_db() -> f64 {
    -12.0
}
fn default_min_freq() -> f64 {
    20.0
}
fn default_max_freq() -> f64 {
    16000.0
}
fn default_max_iter() -> usize {
    5000
}
fn default_population() -> usize {
    30
}
fn default_peq_model() -> String {
    "pk".to_string()
}
fn default_smooth_n() -> usize {
    3
}
fn default_refine() -> bool {
    true
}
fn default_asymmetric_loss() -> bool {
    true
}

/// Optimizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    #[serde(default)]
    pub processing_mode: ProcessingMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fir: Option<FirConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixed_config: Option<MixedModeConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixed_phase: Option<MixedPhaseConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_correction: Option<MixedPhaseConfig>,
    #[serde(default = "default_loss_type")]
    pub loss_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epa_config: Option<serde_json::Value>,
    #[serde(default = "default_algorithm")]
    pub algorithm: String,
    #[serde(default = "default_strategy")]
    pub strategy: String,
    #[serde(default = "default_num_filters")]
    pub num_filters: usize,
    #[serde(default = "default_min_filter_improvement")]
    pub min_filter_improvement: f64,
    #[serde(default = "default_elimination_threshold")]
    pub elimination_threshold: f64,
    #[serde(default = "default_min_q")]
    pub min_q: f64,
    #[serde(default = "default_max_q")]
    pub max_q: f64,
    #[serde(default = "default_min_db")]
    pub min_db: f64,
    #[serde(default = "default_max_db")]
    pub max_db: f64,
    #[serde(default = "default_min_freq")]
    pub min_freq: f64,
    #[serde(default = "default_max_freq")]
    pub max_freq: f64,
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    #[serde(default = "default_population")]
    pub population: usize,
    #[serde(default = "default_peq_model")]
    pub peq_model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default = "default_refine")]
    pub refine: bool,
    #[serde(default = "default_asymmetric_loss")]
    pub asymmetric_loss: bool,
    #[serde(default = "default_smooth_n")]
    pub smooth_n: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_measurement: Option<MultiMeasurementConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_seat: Option<MultiSeatConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schroeder_split: Option<SchroederSplitConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_alignment: Option<PhaseAlignmentConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_matching: Option<ChannelMatchingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_delay_optimization: Option<GroupDelayOptimizationConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sub_optimizer: Option<SubOptimizerConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_response: Option<TargetResponseConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cea2034_correction: Option<Cea2034CorrectionConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_optimizer: Option<AutoOptimizerConfig>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            processing_mode: ProcessingMode::default(),
            fir: None,
            mixed_config: None,
            mixed_phase: None,
            phase_correction: None,
            loss_type: default_loss_type(),
            epa_config: None,
            algorithm: default_algorithm(),
            strategy: default_strategy(),
            num_filters: default_num_filters(),
            min_filter_improvement: default_min_filter_improvement(),
            elimination_threshold: default_elimination_threshold(),
            min_q: default_min_q(),
            max_q: default_max_q(),
            min_db: default_min_db(),
            max_db: default_max_db(),
            min_freq: default_min_freq(),
            max_freq: default_max_freq(),
            max_iter: default_max_iter(),
            population: default_population(),
            peq_model: default_peq_model(),
            seed: None,
            refine: default_refine(),
            asymmetric_loss: default_asymmetric_loss(),
            smooth_n: default_smooth_n(),
            multi_measurement: None,
            multi_seat: None,
            schroeder_split: None,
            phase_alignment: None,
            channel_matching: None,
            group_delay_optimization: None,
            sub_optimizer: None,
            target_response: None,
            cea2034_correction: None,
            auto_optimizer: None,
        }
    }
}

impl OptimizerConfig {
    /// Validate the numeric optimizer contract before a candidate loop starts.
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        let finite = [
            ("min_filter_improvement", self.min_filter_improvement),
            ("elimination_threshold", self.elimination_threshold),
            ("min_q", self.min_q),
            ("max_q", self.max_q),
            ("min_db", self.min_db),
            ("max_db", self.max_db),
            ("min_freq", self.min_freq),
            ("max_freq", self.max_freq),
        ];
        for (field, value) in finite {
            if !value.is_finite() {
                return Err(ConfigValidationError::NonFinite { field });
            }
        }
        if self.num_filters == 0 {
            return Err(ConfigValidationError::ZeroCount {
                field: "num_filters",
            });
        }
        if self.max_iter == 0 {
            return Err(ConfigValidationError::ZeroCount { field: "max_iter" });
        }
        if self.population == 0 {
            return Err(ConfigValidationError::ZeroCount {
                field: "population",
            });
        }
        if self.smooth_n == 0 {
            return Err(ConfigValidationError::ZeroCount { field: "smooth_n" });
        }
        validate_non_negative("min_filter_improvement", self.min_filter_improvement)?;
        validate_non_negative("elimination_threshold", self.elimination_threshold)?;
        if self.min_q <= 0.0 {
            return Err(ConfigValidationError::NonPositive {
                field: "min_q",
                value: self.min_q,
            });
        }
        if self.max_q < self.min_q {
            return Err(ConfigValidationError::InvalidRange {
                field: "q",
                min: self.min_q,
                max: self.max_q,
            });
        }
        if self.max_db < self.min_db {
            return Err(ConfigValidationError::InvalidRange {
                field: "db",
                min: self.min_db,
                max: self.max_db,
            });
        }
        if self.min_freq <= 0.0 {
            return Err(ConfigValidationError::NonPositive {
                field: "min_freq",
                value: self.min_freq,
            });
        }
        if self.max_freq <= self.min_freq {
            return Err(ConfigValidationError::InvalidRange {
                field: "frequency",
                min: self.min_freq,
                max: self.max_freq,
            });
        }
        validate_non_empty("algorithm", &self.algorithm)?;
        validate_non_empty("strategy", &self.strategy)?;
        validate_non_empty("loss_type", &self.loss_type)?;
        validate_non_empty("peq_model", &self.peq_model)?;
        if let Some(fir) = &self.fir {
            fir.validate()?;
        }
        if let Some(mixed_config) = &self.mixed_config {
            mixed_config.validate()?;
        }
        if let Some(mixed_phase) = &self.mixed_phase {
            mixed_phase.validate()?;
        }
        if let Some(phase_correction) = &self.phase_correction {
            phase_correction.validate()?;
        }
        if let Some(multi_measurement) = &self.multi_measurement {
            multi_measurement.validate()?;
        }
        if let Some(multi_seat) = &self.multi_seat {
            multi_seat.validate()?;
        }
        if let Some(schroeder_split) = &self.schroeder_split {
            schroeder_split.validate()?;
        }
        if let Some(phase_alignment) = &self.phase_alignment {
            phase_alignment.validate()?;
        }
        if let Some(channel_matching) = &self.channel_matching {
            channel_matching.validate()?;
        }
        if let Some(group_delay) = &self.group_delay_optimization {
            group_delay.validate()?;
        }
        if let Some(sub_optimizer) = &self.sub_optimizer {
            sub_optimizer.validate()?;
        }
        if let Some(target_response) = &self.target_response {
            target_response.validate()?;
        }
        if let Some(cea2034) = &self.cea2034_correction {
            cea2034.validate()?;
        }
        if let Some(auto_optimizer) = &self.auto_optimizer {
            auto_optimizer.validate()?;
        }
        Ok(())
    }
}

// ============================================================================
// Recording config (minimal stub)
// ============================================================================

/// Recording configuration (minimal fields needed for roundtrip).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecordingConfiguration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub playback_device_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub playback_sample_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recording_device_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recording_sample_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub room_dimensions: Option<RoomDimensions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup_description: Option<String>,
}

// ============================================================================
// RoomConfig
// ============================================================================

/// Top-level room configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomConfig {
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemConfig>,
    pub speakers: HashMap<String, SpeakerConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossovers: Option<HashMap<String, CrossoverConfig>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_curve: Option<String>,
    pub optimizer: OptimizerConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recording_config: Option<RecordingConfiguration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cea2034_cache: Option<HashMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctc: Option<serde_json::Value>,
}

impl RoomConfig {
    /// Read, migrate, and validate a RoomEQ configuration document.
    ///
    /// Direct `serde_json` deserialization is retained for compatibility, but
    /// consumers that promote a document into scoring or optimization should
    /// use this boundary so legacy defaults and unsupported future versions
    /// cannot be mistaken for the current schema.
    pub fn from_json_migrated(
        json: &str,
    ) -> Result<(Self, RoomConfigMigrationReport), RoomConfigReadError> {
        let mut document: serde_json::Value =
            serde_json::from_str(json).map_err(RoomConfigReadError::Json)?;
        let object = document
            .as_object_mut()
            .ok_or(RoomConfigReadError::InvalidRoot)?;
        let from_version = match object.get("version") {
            Some(serde_json::Value::String(version)) => version.clone(),
            Some(_) => return Err(RoomConfigReadError::InvalidVersionField),
            None => "1.0.0".to_string(),
        };
        let mut steps = Vec::new();
        match from_version.as_str() {
            "1.0.0" => {
                steps.push("1.0.0 -> 1.2.0: stamp the current schema version".to_string());
            }
            "1.1.0" => {
                steps.push("1.1.0 -> 1.2.0: stamp the current schema version".to_string());
            }
            CURRENT_CONFIG_SCHEMA_VERSION => {}
            _ => {
                return Err(RoomConfigReadError::UnsupportedVersion {
                    version: from_version,
                });
            }
        }
        object.insert(
            "version".to_string(),
            serde_json::Value::String(CURRENT_CONFIG_SCHEMA_VERSION.to_string()),
        );
        let config: Self = serde_json::from_value(document).map_err(RoomConfigReadError::Json)?;
        config.validate().map_err(RoomConfigReadError::Validation)?;
        Ok((
            config,
            RoomConfigMigrationReport {
                from_version,
                to_version: CURRENT_CONFIG_SCHEMA_VERSION.to_string(),
                steps,
            },
        ))
    }

    /// Validate the stable configuration envelope before serialization or
    /// optimization. Measurement-file contents remain the responsibility of
    /// the reader that owns them, but all nested serde configuration values are
    /// admitted here so malformed data cannot reach scoring or optimization.
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.version.trim().is_empty() {
            return Err(ConfigValidationError::EmptyVersion);
        }
        if self.version != CURRENT_CONFIG_SCHEMA_VERSION {
            return Err(ConfigValidationError::UnsupportedVersion {
                version: self.version.clone(),
            });
        }
        if self.speakers.is_empty() {
            return Err(ConfigValidationError::EmptySpeakers);
        }
        if let Some(system) = &self.system {
            system.validate()?;
        }
        for (name, speaker) in &self.speakers {
            validate_non_empty("speakers.key", name)?;
            speaker.validate()?;
        }
        if let Some(crossovers) = &self.crossovers {
            if crossovers.is_empty() {
                return Err(ConfigValidationError::ZeroCount {
                    field: "crossovers",
                });
            }
            for (name, crossover) in crossovers {
                validate_non_empty("crossovers.key", name)?;
                crossover.validate()?;
            }
        }
        if let Some(target_curve) = &self.target_curve {
            validate_non_empty("target_curve", target_curve)?;
        }
        self.optimizer.validate()?;
        if let Some(recording) = &self.recording_config {
            if recording.playback_sample_rate == Some(0) {
                return Err(ConfigValidationError::InvalidSampleRate {
                    field: "playback_sample_rate",
                    value: 0,
                });
            }
            if recording.recording_sample_rate == Some(0) {
                return Err(ConfigValidationError::InvalidSampleRate {
                    field: "recording_sample_rate",
                    value: 0,
                });
            }
            if let Some(dimensions) = &recording.room_dimensions {
                dimensions.validate()?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config() -> RoomConfig {
        RoomConfig {
            version: default_config_version(),
            system: None,
            speakers: HashMap::from([(
                "left".to_string(),
                SpeakerConfig::Single(MeasurementSource::Single(MeasurementSingle {
                    measurement: MeasurementRef::Path(PathBuf::from("left.csv")),
                    speaker_name: None,
                })),
            )]),
            crossovers: None,
            target_curve: None,
            optimizer: OptimizerConfig::default(),
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        }
    }

    #[test]
    fn config_validation_accepts_generated_defaults() {
        assert!(minimal_config().validate().is_ok());
    }

    #[test]
    fn versioned_reader_migrates_legacy_documents_and_rejects_future_versions() {
        let mut legacy = serde_json::to_value(minimal_config()).expect("serialize fixture");
        legacy["version"] = serde_json::Value::String("1.0.0".to_string());
        let (migrated, report) = RoomConfig::from_json_migrated(
            &serde_json::to_string(&legacy).expect("serialize legacy fixture"),
        )
        .expect("known legacy version migrates");
        assert_eq!(migrated.version, CURRENT_CONFIG_SCHEMA_VERSION);
        assert_eq!(report.from_version, "1.0.0");
        assert_eq!(report.to_version, CURRENT_CONFIG_SCHEMA_VERSION);
        assert_eq!(report.steps.len(), 1);

        legacy["version"] = serde_json::Value::String("2.0.0".to_string());
        assert!(matches!(
            RoomConfig::from_json_migrated(
                &serde_json::to_string(&legacy).expect("serialize future fixture"),
            ),
            Err(RoomConfigReadError::UnsupportedVersion { version }) if version == "2.0.0"
        ));
    }

    #[test]
    fn direct_validation_rejects_legacy_versions_until_migrated() {
        let mut config = minimal_config();
        config.version = "1.1.0".to_string();
        assert_eq!(
            config.validate(),
            Err(ConfigValidationError::UnsupportedVersion {
                version: "1.1.0".to_string(),
            })
        );
    }

    #[test]
    fn config_validation_rejects_nonfinite_and_unordered_ranges() {
        let mut config = minimal_config();
        config.optimizer.max_freq = f64::NAN;
        assert_eq!(
            config.validate(),
            Err(ConfigValidationError::NonFinite { field: "max_freq" })
        );

        config.optimizer.max_freq = 10.0;
        config.optimizer.min_freq = 20.0;
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::InvalidRange {
                field: "frequency",
                ..
            })
        ));
    }

    #[test]
    fn config_validation_rejects_empty_speakers_and_zero_rate() {
        let mut config = minimal_config();
        config.speakers.clear();
        assert_eq!(config.validate(), Err(ConfigValidationError::EmptySpeakers));

        let mut config = minimal_config();
        config.recording_config = Some(RecordingConfiguration {
            playback_sample_rate: Some(0),
            ..RecordingConfiguration::default()
        });
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::InvalidSampleRate {
                field: "playback_sample_rate",
                value: 0
            })
        ));
    }

    #[test]
    fn config_validation_rejects_nested_crossover_and_target_errors() {
        let mut config = minimal_config();
        config.crossovers = Some(HashMap::from([(
            "main".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: None,
                frequencies: Some(vec![80.0, 70.0]),
                frequency_range: None,
            },
        )]));
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::InvalidRange {
                field: "crossover.frequencies",
                ..
            })
        ));

        config.crossovers = None;
        config.optimizer.target_response = Some(TargetResponseConfig {
            reference_freq: 0.0,
            ..TargetResponseConfig::default()
        });
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::NonPositive {
                field: "target.reference_freq",
                ..
            })
        ));
    }

    #[test]
    fn target_response_config_serde_matches_default() {
        // The review's serde-vs-Default mismatch: `slope_db_per_octave`
        // deserialized to 0.0 while `Default` said 0.0 too — both must now
        // agree on `default_tilt_slope()` so omitted fields behave like a
        // default-constructed config.
        let default = TargetResponseConfig::default();
        let round_tripped: TargetResponseConfig =
            serde_json::from_value(serde_json::to_value(&default).unwrap()).unwrap();
        assert_eq!(
            round_tripped.slope_db_per_octave,
            default.slope_db_per_octave
        );
        assert_eq!(round_tripped.slope_db_per_octave, default_tilt_slope());
        let partial: TargetResponseConfig = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(partial.slope_db_per_octave, default_tilt_slope());
        assert_eq!(partial.reference_freq, default_tilt_reference_freq());
    }

    #[test]
    fn config_validation_rejects_nested_speaker_and_room_errors() {
        let mut config = minimal_config();
        config.speakers.insert(
            "group".to_string(),
            SpeakerConfig::Group(SpeakerGroup {
                name: "group".to_string(),
                speaker_name: None,
                measurements: Vec::new(),
                crossover: None,
            }),
        );
        assert_eq!(
            config.validate(),
            Err(ConfigValidationError::ZeroCount {
                field: "speaker.measurements"
            })
        );

        config.speakers.remove("group");
        config.recording_config = Some(RecordingConfiguration {
            room_dimensions: Some(RoomDimensions {
                length: 0.0,
                width: 4.0,
                height: 2.5,
            }),
            ..RecordingConfiguration::default()
        });
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::NonPositive {
                field: "room_dimensions.length",
                ..
            })
        ));
    }

    #[test]
    fn config_validation_rejects_nested_system_and_weight_errors() {
        let mut config = minimal_config();
        config.system = Some(SystemConfig {
            model: SystemModel::Stereo,
            speakers: HashMap::from([("L".to_string(), "left".to_string())]),
            subwoofers: None,
            bass_management: Some(BassManagementConfig {
                max_sub_boost_db: f64::NAN,
                ..BassManagementConfig::default()
            }),
        });
        assert_eq!(
            config.validate(),
            Err(ConfigValidationError::NonFinite {
                field: "bass_management.max_sub_boost_db"
            })
        );

        config.system = None;
        config.optimizer.multi_seat = Some(MultiSeatConfig {
            seat_weights: Some(vec![1.0, -0.5]),
            ..MultiSeatConfig::default()
        });
        assert!(matches!(
            config.validate(),
            Err(ConfigValidationError::InvalidRange {
                field: "multi_seat.seat_weights",
                ..
            })
        ));
    }
}
