//! FUNCT_PYTHON deck survey and evaluation verdict.
//!
//! Parses `/FUNCT_PYTHON` blocks from decks read by
//! [`RadDeck`](crate::rad_parse::RadDeck), classifies each block as pure,
//! sibling-calling, or engine-coupled, and records the Phase-5 lane 5E
//! evaluation verdict (dated deferral, see below).
//!
//! # Spike survey (`PYTHON/FUNCT_PYTHON`)
//!
//! The deck (`test_torsion_orthstrain_shell_0000.rad`) carries three
//! blocks; both `/IMPDISP` cards drive function id 3 with `Scale_x =
//! 0.05` (`Scale_y` 5 and 1.57):
//!
//! | card | name | body | calls | accessors |
//! |---|---|---|---|
//! | `/FUNCT_PYTHON/1` | `myfun2` | `return x` | none | 7 nodal (`CX`,
//! `DX`, `DRX`, `VX`, `VRX`, `AX`, `ARX`) plus `TIME`/`DT` scalars |
//! | `/FUNCT_PYTHON/21` | `myfun4` | `return x` | none | 89 element
//! fields (`ALPHA` .. `WPLA`, incl. `DT`) |
//! | `/FUNCT_PYTHON/3` | `myfun` | smoothstep pieces | `myfun2` | none |
//!
//! The `/1` and `/21` bodies are identity functions whose comments
//! enumerate the engine callback surface the namespace must provide:
//! nodal position/displacement/rotation/velocity/acceleration getters
//! keyed by node id, element field getters keyed by element id, and the
//! `TIME`/`DT` scalars. `/3` is the live load curve: zero below 0, one
//! above 1, and `d*d*d*(10-15*d+6*d*d)` between, calling `myfun2` for
//! `d`. The starter runs with `-python` (see the golden
//! `starter_options`); plain `/FUNCT/1000,2000,3000,/2` siblings serve
//! the failure model and mesh-size scaling, not the motion.
//!
//! # What evaluation needs
//!
//! Calling a block per engine cycle with a float abscissa and float
//! return; full Python call semantics (cross-block calls like
//! `myfun2(x)`); the callback namespace above bound to engine state
//! (sensors, nodal and element fields at the current cycle, passed as
//! a per-call snapshot); deterministic results (same abscissa and
//! state give bitwise-equal returns). Phase 7 lane 7D ships all four
//! behind the `funct-python` feature ([`evaluate_python`]).
//!
//! # Verdict (P5C3, closed by Phase 7 lane 7D)
//!
//! Evaluation is implemented; without the `funct-python` feature it
//! stays deferred to [`DEFER_UNTIL`] with [`DEFER_MISSING`] missing
//! toward [`DEFER_TARGET`]. Rationale: a native subset evaluator
//! cannot reproduce Python semantics and would silently diverge on
//! decks outside its grammar, so Phase 7 evaluates real Python behind
//! an explicit trust boundary instead of pretending a subset suffices.
//! Block parsing, call/accessor extraction, purity tagging, and the
//! transcribed deck smoothstep closed form remain as cross-checks.
//!
//! # Trust boundary (read before running decks)
//!
//! The evaluator embeds CPython and executes deck `def` bodies with
//! the FULL interpreter: imports, builtins, file and network access,
//! subprocesses, everything the host Python process can do. There is
//! NO sandbox, NO restricted subset, NO import allowlist, and NO
//! resource limit. Treat every `/FUNCT_PYTHON` deck as executable code
//! from its author: run only decks you trust exactly as far as you
//! trust their source, in an environment where arbitrary Python
//! execution is acceptable. Sensor accessors expose only the snapshot
//! values the caller places in [`PythonContext`]; they do not widen
//! deck access beyond the interpreter itself. Determinism holds for
//! pure decks (same blocks, abscissa, and context give bit-identical
//! results); decks that read clocks, random state, files, or the
//! network can vary run to run like any Python program.
//!
//! # Examples
//!
//! ```rust
//! use sonium_qa::rad_parse::RadDeck;
//! use sonium_qa::funct_python::{Purity, classify, collect_blocks};
//!
//! let deck = RadDeck::parse_str(
//!     "/FUNCT_PYTHON/3\ndef myfun(x):\n    if x <= 0:\n        return 0\n    return x\n/END\n",
//! )
//! .expect("deck");
//! let blocks = collect_blocks(&deck).expect("collect");
//! assert_eq!(blocks[0].name, "myfun");
//! assert_eq!(classify(&blocks[0], &blocks), Purity::Pure);
//! ```

// Rust guideline compliant 2026-02-21

use crate::rad_parse::{CardKind, DeckCard, RadDeck};
use std::backtrace::Backtrace;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
#[cfg(feature = "funct-python")]
use std::sync::Arc;

/// Condition that lifts the evaluation deferral.
///
/// Without the `funct-python` feature, `/FUNCT_PYTHON` evaluation
/// stays deferred until this condition holds; the verdict above names
/// the missing piece and replay target.
pub const DEFER_UNTIL: &str = "build enables the funct-python feature";

/// Missing piece blocking evaluation without the feature.
pub const DEFER_MISSING: &str =
    "embedded CPython interpreter (pyo3) with TIME/DT scalars and nodal/element sensor snapshots";

/// Replay target the deferred evaluation would hit.
pub const DEFER_TARGET: &str = "trusted-deck FUNCT_PYTHON evaluation returning abscissa-mapped sensor values (deterministic for pure decks; no sandbox)";

/// Private failure modes behind [`FunctPythonError`].
///
/// Each configuration constructs a different subset: the
/// interpreter variants only with the feature on, the unavailable
/// variant only with the feature off.
#[derive(Debug)]
#[allow(dead_code)]
enum ErrorKind {
    /// Block without its header identifier.
    MissingId,
    /// Block without a `def` line.
    MissingDef,
    /// Malformed `def` line.
    BadDef { line: String },
    /// Block with no body lines.
    EmptyBody,
    /// Evaluation target id absent from the blocks.
    UnknownBlock(i64),
    /// Python raised, or the return is not a float.
    EvaluationFailed(String),
    /// Python returned NaN or an infinity.
    NonFiniteResult(f64),
    /// Evaluation without the `funct-python` feature.
    EvaluationUnavailable,
}

/// `/FUNCT_PYTHON` block collection failure.
#[derive(Debug)]
pub struct FunctPythonError {
    kind: ErrorKind,
    backtrace: Backtrace,
}

impl FunctPythonError {
    /// Reports a missing id, `def` line, or body.
    #[must_use]
    pub const fn is_missing(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::MissingId | ErrorKind::MissingDef | ErrorKind::EmptyBody
        )
    }

    /// Reports a malformed `def` line.
    #[must_use]
    pub const fn is_malformed(&self) -> bool {
        matches!(self.kind, ErrorKind::BadDef { .. })
    }

    /// Reports an evaluation failure (unknown block, Python exception,
    /// bad return, or non-finite result).
    #[must_use]
    pub const fn is_evaluation(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::UnknownBlock(_)
                | ErrorKind::EvaluationFailed(_)
                | ErrorKind::NonFiniteResult(_)
        )
    }

    /// Reports evaluation without the `funct-python` feature.
    #[must_use]
    pub const fn is_unavailable(&self) -> bool {
        matches!(self.kind, ErrorKind::EvaluationUnavailable)
    }

    fn from_kind(kind: ErrorKind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }
}

impl Display for FunctPythonError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ErrorKind::MissingId => {
                write!(f, "block lacks its header id\n{}", self.backtrace)
            }
            ErrorKind::MissingDef => {
                write!(f, "block lacks its def line\n{}", self.backtrace)
            }
            ErrorKind::BadDef { line } => {
                write!(f, "malformed def line {line:?}\n{}", self.backtrace)
            }
            ErrorKind::EmptyBody => {
                write!(f, "block has no body lines\n{}", self.backtrace)
            }
            ErrorKind::UnknownBlock(id) => {
                write!(f, "unknown FUNCT_PYTHON block {id}\n{}", self.backtrace)
            }
            ErrorKind::EvaluationFailed(detail) => {
                write!(
                    f,
                    "FUNCT_PYTHON evaluation failed: {detail}\n{}",
                    self.backtrace
                )
            }
            ErrorKind::NonFiniteResult(value) => {
                write!(
                    f,
                    "FUNCT_PYTHON returned non-finite {value}\n{}",
                    self.backtrace
                )
            }
            ErrorKind::EvaluationUnavailable => {
                write!(
                    f,
                    "FUNCT_PYTHON evaluation needs the funct-python feature \
                     (see DEFER_UNTIL, DEFER_MISSING, DEFER_TARGET)\n{}",
                    self.backtrace
                )
            }
        }
    }
}

impl std::error::Error for FunctPythonError {}

/// One `/FUNCT_PYTHON` block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctPythonBlock {
    /// Function identifier from the card header.
    pub id: i64,
    /// Function name from the `def` line.
    pub name: String,
    /// Single argument name from the `def` line.
    pub param: String,
    /// Verbatim body lines after the `def` line.
    pub body: Vec<String>,
}

/// Purity tag of a block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Purity {
    /// No sibling calls and no engine accessors.
    Pure,
    /// Calls sibling blocks, no engine accessors.
    SiblingCalls(Vec<String>),
    /// References engine accessors (sorted names).
    EngineCoupled(Vec<String>),
}

/// Reports whether a name is a Python identifier.
fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Parses a `def name(param):` line.
fn parse_def(line: &str) -> Result<(String, String), FunctPythonError> {
    let trimmed = line.trim();
    let bad = || {
        FunctPythonError::from_kind(ErrorKind::BadDef {
            line: line.to_string(),
        })
    };
    let Some(rest) = trimmed.strip_prefix("def ") else {
        return Err(bad());
    };
    let Some(open) = rest.find('(') else {
        return Err(bad());
    };
    let Some(close) = rest.find(')') else {
        return Err(bad());
    };
    if close < open || !rest[close + 1..].trim().starts_with(':') {
        return Err(bad());
    }
    let name = rest[..open].trim().to_string();
    let param = rest[open + 1..close].trim().to_string();
    if !is_identifier(&name) || !is_identifier(&param) {
        return Err(bad());
    }
    Ok((name, param))
}

/// Collects deck blocks in deck order.
///
/// The reader keeps the `def` line as the card title and every later
/// line (comments and blanks included) as verbatim body rows; this
/// function joins each row back to its line.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_parse::RadDeck;
/// use sonium_qa::funct_python::collect_blocks;
///
/// let deck = RadDeck::parse_str(
///     "/FUNCT_PYTHON/1\ndef myfun2(x):\n    return x\n/END\n",
/// )
/// .expect("deck");
/// let blocks = collect_blocks(&deck).expect("collect");
/// assert_eq!(blocks.len(), 1);
/// assert_eq!(blocks[0].param, "x");
/// ```
///
/// # Errors
///
/// Returns [`FunctPythonError`] for missing ids, missing or malformed
/// `def` lines, multi-parameter definitions, or empty bodies.
pub fn collect_blocks(deck: &RadDeck) -> Result<Vec<FunctPythonBlock>, FunctPythonError> {
    let mut blocks = Vec::new();
    for card in &deck.cards {
        if card.kind != CardKind::FunctPython {
            continue;
        }
        blocks.push(decode_block(card)?);
    }
    Ok(blocks)
}

/// Decodes one block card.
fn decode_block(card: &DeckCard) -> Result<FunctPythonBlock, FunctPythonError> {
    let Some(id) = card.id else {
        return Err(FunctPythonError::from_kind(ErrorKind::MissingId));
    };
    if card.title.trim().is_empty() {
        return Err(FunctPythonError::from_kind(ErrorKind::MissingDef));
    }
    let (name, param) = parse_def(&card.title)?;
    if card.fields.is_empty() {
        return Err(FunctPythonError::from_kind(ErrorKind::EmptyBody));
    }
    let body = card.fields.iter().map(|row| row.concat()).collect();
    Ok(FunctPythonBlock {
        id,
        name,
        param,
        body,
    })
}

/// Extracts called names from body lines.
///
/// Reports every identifier immediately followed by `(`, scanning
/// comments too (the surveyed accessor inventory lives in comments).
/// Results are sorted and deduplicated.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::funct_python::called_names;
///
/// let body = vec!["    d = myfun2(x)".to_string(), "    return d".to_string()];
/// assert_eq!(called_names(&body), vec!["myfun2".to_string()]);
/// ```
#[must_use]
pub fn called_names(body: &[String]) -> Vec<String> {
    let mut names = BTreeSet::new();
    for line in body {
        let bytes = line.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            let byte = bytes[index];
            let starts = byte.is_ascii_alphabetic() || byte == b'_';
            if !starts {
                index += 1;
                continue;
            }
            let mut end = index + 1;
            while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
                end += 1;
            }
            if end < bytes.len()
                && bytes[end] == b'('
                && let Ok(name) = std::str::from_utf8(&bytes[index..end])
            {
                names.insert(name.to_string());
            }
            index = end;
        }
    }
    names.into_iter().collect()
}

/// Reports whether a called name is an engine accessor.
///
/// Accessors are all-caps words (letters, digits, underscores) with at
/// least one letter: `CX`, `K`, `HC_DSSE_F`. Lowercase helpers such as
/// `print` and sibling names never match.
fn is_accessor(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && name.chars().any(|c| c.is_ascii_alphabetic())
}

/// Extracts engine accessor references.
///
/// Filters [`called_names`] to all-caps callback names; results are
/// sorted. Bare scalar globals (`TIME`, `DT` without call syntax) are
/// outside this scan and stay surveyed documentation.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::funct_python::accessor_refs;
///
/// let body = vec!["    #print(f\"{CX(100)}\")".to_string(), "    return x".to_string()];
/// assert_eq!(accessor_refs(&body), vec!["CX".to_string()]);
/// ```
#[must_use]
pub fn accessor_refs(body: &[String]) -> Vec<String> {
    called_names(body)
        .into_iter()
        .filter(|name| is_accessor(name))
        .collect()
}

/// Extracts sibling block calls.
///
/// Filters [`called_names`] to `def` names of other blocks in
/// `blocks`, excluding the block itself; results are sorted.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::funct_python::{FunctPythonBlock, sibling_calls};
///
/// let caller = FunctPythonBlock {
///     id: 3,
///     name: "myfun".to_string(),
///     param: "x".to_string(),
///     body: vec!["    d = myfun2(x)".to_string()],
/// };
/// let callee = FunctPythonBlock {
///     id: 1,
///     name: "myfun2".to_string(),
///     param: "x".to_string(),
///     body: vec!["    return x".to_string()],
/// };
/// let blocks = vec![caller.clone(), callee];
/// assert_eq!(sibling_calls(&caller, &blocks), vec!["myfun2".to_string()]);
/// ```
#[must_use]
pub fn sibling_calls(block: &FunctPythonBlock, blocks: &[FunctPythonBlock]) -> Vec<String> {
    let known: BTreeSet<&str> = blocks
        .iter()
        .filter(|peer| peer.name != block.name)
        .map(|peer| peer.name.as_str())
        .collect();
    called_names(&block.body)
        .into_iter()
        .filter(|name| known.contains(name.as_str()))
        .collect()
}

/// Tags a block by purity.
///
/// Engine-coupled wins over sibling-calling: a block touching any
/// accessor is coupled however many siblings it also calls.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::funct_python::{FunctPythonBlock, Purity, classify};
///
/// let block = FunctPythonBlock {
///     id: 1,
///     name: "myfun2".to_string(),
///     param: "x".to_string(),
///     body: vec!["    return x".to_string()],
/// };
/// let blocks = vec![block.clone()];
/// assert_eq!(classify(&block, &blocks), Purity::Pure);
/// ```
#[must_use]
pub fn classify(block: &FunctPythonBlock, blocks: &[FunctPythonBlock]) -> Purity {
    let accessors = accessor_refs(&block.body);
    if !accessors.is_empty() {
        return Purity::EngineCoupled(accessors);
    }
    let siblings = sibling_calls(block, blocks);
    if siblings.is_empty() {
        Purity::Pure
    } else {
        Purity::SiblingCalls(siblings)
    }
}

/// Evaluates the deck smoothstep closed form.
///
/// Transcribes `/FUNCT_PYTHON/3` (`myfun`) with `myfun2` as the
/// identity: zero at or below 0, one at or above 1, and
/// `d*d*d*(10-15*d+6*d*d)` between. This cross-checks the deck
/// formula; it is not Python evaluation (see the verdict). Note the
/// deck comment beside the code reads `10-14*d`, contradicting its
/// own code line; the transcription follows the code (the `15`
/// quintic is the `C2` smoothstep).
/// Non-finite inputs propagate (comparisons fail through to the
/// polynomial, which preserves NaN).
///
/// # Examples
///
/// ```rust
/// use sonium_qa::funct_python::smoothstep_unit;
///
/// assert_eq!(smoothstep_unit(0.0), 0.0);
/// assert_eq!(smoothstep_unit(0.5), 0.5);
/// assert_eq!(smoothstep_unit(1.0), 1.0);
/// ```
#[must_use]
pub fn smoothstep_unit(abscissa: f64) -> f64 {
    if abscissa <= 0.0 {
        0.0
    } else if abscissa >= 1.0 {
        1.0
    } else {
        let poly = 10.0 - 15.0 * abscissa + 6.0 * abscissa * abscissa;
        abscissa * abscissa * abscissa * poly
    }
}

/// Sensor snapshot bound into one Python evaluation.
///
/// Carries the `TIME`/`DT` scalars plus the nodal and element values
/// the deck accessors read: displacement, velocity, and acceleration
/// per `(node id, dof)` with dof `X`, `Y`, or `Z`, and element fields
/// per `(element id, field name)`. Snapshots are per-call by design:
/// an explicit driver rebuilds one per cycle from the current state,
/// which keeps evaluation deterministic (same blocks, abscissa, and
/// snapshot give bit-identical results for pure decks).
///
/// Missing entries fail closed: a deck call with no snapshot entry
/// raises inside Python and surfaces as an evaluation error, never a
/// silent zero.
#[derive(Debug, Clone, Default)]
pub struct PythonContext {
    /// Current cycle time, bound as `TIME`.
    pub time: f64,
    /// Current step length, bound as `DT`.
    pub dt: f64,
    /// Nodal displacements keyed by `(node id, dof)`.
    pub nodal_disp: BTreeMap<(i64, char), f64>,
    /// Nodal velocities keyed by `(node id, dof)`.
    pub nodal_vel: BTreeMap<(i64, char), f64>,
    /// Nodal accelerations keyed by `(node id, dof)`.
    pub nodal_acce: BTreeMap<(i64, char), f64>,
    /// Element fields keyed by `(element id, field name)`.
    pub element_var: BTreeMap<(i64, String), f64>,
}

/// Evaluates one `/FUNCT_PYTHON` block at an abscissa.
///
/// Reconstructs each block `def` (name, parameter, verbatim body) in
/// deck order in one namespace (so sibling calls resolve), then calls
/// block `id` with `abscissa`.
/// The namespace also binds `TIME`/`DT` from `context` plus the
/// `node_disp`, `node_vel`, `node_acce` (each `(id, dof)`, dof a
/// `X`/`Y`/`Z` string or `1`/`2`/`3`), and `element_var`
/// (`(id, name)`) sensor stubs backed by the context snapshot.
/// Requires the `funct-python` feature; without it every call fails
/// with [`FunctPythonError::is_unavailable`].
///
/// Decks sharing a function name resolve last-wins in deck order;
/// well-formed decks keep names unique.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::funct_python::{PythonContext, collect_blocks, evaluate_python};
/// use sonium_qa::rad_parse::RadDeck;
///
/// let deck = RadDeck::parse_str(
///     "/FUNCT_PYTHON/1\ndef linear(x):\n    return 2.0 * x + TIME\n/END\n",
/// )
/// .expect("deck");
/// let blocks = collect_blocks(&deck).expect("collect");
/// let context = PythonContext {
///     time: 1.0,
///     ..Default::default()
/// };
/// match evaluate_python(&blocks, 1, 3.0, &context) {
///     Ok(value) => assert_eq!(value, 7.0),
///     Err(err) => assert!(err.is_unavailable(), "unexpected {err}"),
/// }
/// ```
///
/// # Errors
///
/// Returns [`FunctPythonError`] for an unknown block id, any Python
/// exception or non-float return, a non-finite return, or a missing
/// feature build.
pub fn evaluate_python(
    blocks: &[FunctPythonBlock],
    id: i64,
    abscissa: f64,
    context: &PythonContext,
) -> Result<f64, FunctPythonError> {
    #[cfg(not(feature = "funct-python"))]
    {
        let _ = (blocks, id, abscissa, context);
        Err(FunctPythonError::from_kind(
            ErrorKind::EvaluationUnavailable,
        ))
    }
    #[cfg(feature = "funct-python")]
    {
        evaluate_python_inner(blocks, id, abscissa, context)
    }
}

/// Real evaluator, compiled only with the `funct-python` feature.
#[cfg(feature = "funct-python")]
fn evaluate_python_inner(
    blocks: &[FunctPythonBlock],
    id: i64,
    abscissa: f64,
    context: &PythonContext,
) -> Result<f64, FunctPythonError> {
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyCFunction, PyDict, PyTuple};

    /// Normalizes a nodal dof string (`X`/`Y`/`Z`, case-insensitive).
    fn normalize_dof_str(text: &str) -> Option<char> {
        let axis = text.trim().to_ascii_uppercase();
        if axis.len() == 1 {
            let axis = axis.chars().next().unwrap_or('?');
            if matches!(axis, 'X' | 'Y' | 'Z') {
                return Some(axis);
            }
        }
        None
    }

    /// Normalizes a nodal dof number (`1`/`2`/`3`).
    fn normalize_dof_int(number: i64) -> Option<char> {
        match number {
            1 => Some('X'),
            2 => Some('Y'),
            3 => Some('Z'),
            _ => None,
        }
    }

    let target = blocks
        .iter()
        .find(|block| block.id == id)
        .ok_or_else(|| FunctPythonError::from_kind(ErrorKind::UnknownBlock(id)))?;
    let name = target.name.as_str();
    let failed = |detail: String| FunctPythonError::from_kind(ErrorKind::EvaluationFailed(detail));
    let lift = |err: PyErr| failed(err.to_string());

    let snapshot = Arc::new(context.clone());
    let nodal = |table: BTreeMap<(i64, char), f64>, label: &'static str| {
        let table = Arc::new(table);
        move |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| -> PyResult<f64> {
            let node: i64 = args.get_item(0)?.extract()?;
            let dof = args.get_item(1)?;
            let axis = dof
                .extract::<String>()
                .ok()
                .and_then(|text| normalize_dof_str(&text))
                .or_else(|| dof.extract::<i64>().ok().and_then(normalize_dof_int))
                .ok_or_else(|| PyValueError::new_err("dof must be an X/Y/Z string or 1/2/3"))?;
            table.get(&(node, axis)).copied().ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "{label} has no snapshot for node {node} dof {axis}"
                ))
            })
        }
    };
    let disp = nodal(snapshot.nodal_disp.clone(), "node_disp");
    let vel = nodal(snapshot.nodal_vel.clone(), "node_vel");
    let acce = nodal(snapshot.nodal_acce.clone(), "node_acce");
    let elements = Arc::new(snapshot.element_var.clone());
    let element =
        move |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| -> PyResult<f64> {
            let pair: (i64, String) = args.extract()?;
            elements.get(&pair).copied().ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "element_var has no snapshot for element {} field {}",
                    pair.0, pair.1
                ))
            })
        };

    Python::with_gil(|py| {
        let namespace = PyDict::new(py);
        namespace.set_item("TIME", snapshot.time).map_err(lift)?;
        namespace.set_item("DT", snapshot.dt).map_err(lift)?;
        let disp = PyCFunction::new_closure(py, None, None, disp).map_err(lift)?;
        namespace.set_item("node_disp", disp).map_err(lift)?;
        let vel = PyCFunction::new_closure(py, None, None, vel).map_err(lift)?;
        namespace.set_item("node_vel", vel).map_err(lift)?;
        let acce = PyCFunction::new_closure(py, None, None, acce).map_err(lift)?;
        namespace.set_item("node_acce", acce).map_err(lift)?;
        let element = PyCFunction::new_closure(py, None, None, element).map_err(lift)?;
        namespace.set_item("element_var", element).map_err(lift)?;
        for block in blocks {
            let source = format!(
                "def {}({}):\n{}\n",
                block.name,
                block.param,
                block.body.join("\n")
            );
            let code = std::ffi::CString::new(source.as_str())
                .map_err(|_| failed(format!("block {} source holds a NUL byte", block.id)))?;
            py.run(&code, Some(&namespace), None)
                .map_err(|err| failed(format!("block {} exec: {err}", block.id)))?;
        }
        let function = namespace
            .get_item(name)
            .map_err(lift)?
            .ok_or_else(|| failed(format!("function {name:?} missing after exec")))?;
        let value: f64 = function
            .call1((abscissa,))
            .map_err(lift)?
            .extract()
            .map_err(|err| failed(format!("block {id} call: {err}")))?;
        if !value.is_finite() {
            return Err(FunctPythonError::from_kind(ErrorKind::NonFiniteResult(
                value,
            )));
        }
        Ok(value)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rad_parse::RadDeck;

    /// `/FUNCT_PYTHON/1` body, deck-verbatim.
    ///
    /// Right-trimmed like the reader keeps it; the gated real-deck
    /// test asserts line-for-line equality with the vendor deck.
    const BODY_ONE: &str = r#"def myfun2(x):
    #print(f"TIME: {TIME:.16f}, DT: {DT:.16f}")
    #print(f"CX_100: {CX(100):16f}")
    #print(f"DX_100: {DX(100):16f}")
    #print(f"DRX_100: {DRX(100):16f}")
    #print(f"VX_100: {VX(100):16f}")
    #print(f"VRX_100: {VRX(100):16f}")
    #print(f"AX_100: {AX(100):16f}")
    #print(f"ARX_100: {ARX(100):16f}")
    #  CX_101,  CY_201,  CZ_301
    #  DX_102,  DY_202,  DZ_302
    # DRX_103, DRY_203, DRZ_303
    #  VX_104,  VY_204 , VZ_304
    # VRX_105, VRY_205, VRZ_305
    #  AX_106,  AY_206,  AZ_306
    # ARX_107, ARY_207, ARZ_307

    return x

"#;

    /// `/FUNCT_PYTHON/3` body, deck-verbatim.
    ///
    /// Ends with the ruler comment the reader keeps as a verbatim body
    /// row (the vendor skips `#` lines; the classifier ignores it).
    const BODY_THREE: &str = r#"def myfun(x):
    #if x < 0 return 0
    #else if x > 1 return 1
    #else return y = d*d*d*(10-14*d+6*d*d) where d = x
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    else:
        d = myfun2(x) # it is possible to call other functions within a function
        return d*d*d*(10-15*d+6*d*d)

#---1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|---10----|
"#;

    /// `/FUNCT_PYTHON/21` body, deck-verbatim.
    const BODY_TWENTYONE: &str = r#"def myfun4(x):
#    print(f" ALPHA         : {ALPHA(1):16f}")
#    print(f" AMS           : {AMS(2):16f}")
#    print(f" BFRAC         : {BFRAC(3):16f}")
#    print(f" BULK          : {BULK(4):16f}")
#    print(f" COLOR         : {COLOR(5):16f}")
#    print(f" DAM1          : {DAM1(6):16f}")
#    print(f" DAM2          : {DAM2(7):16f}")
#    print(f" DAM3          : {DAM3(8):16f}")
#    print(f" DAMA          : {DAMA(9):16f}")
#    print(f" DAMG          : {DAMG(10):16f}")
#    print(f" DAMINI        : {DAMINI(11):16f}")
#    print(f" DENS          : {DENS(12):16f}")
#    print(f" DOMAIN        : {DOMAIN(13):16f}")
#    print(f" DT            : {DT(14):16f}")
#    print(f" EINT          : {EINT(15):16f}")
#    print(f" EINTM         : {EINTM(16):16f}")
#    print(f" EINTV         : {EINTV(17):16f}")
#    print(f" ENER          : {ENER(18):16f}")
#    print(f" ENTH          : {ENTH(19):16f}")
#    print(f" ENTHM         : {ENTHM(20):16f}")
#    print(f" ENTHV         : {ENTHV(21):16f}")
#    print(f" EPSD          : {EPSD(22):16f}")
#    print(f" EPSP          : {EPSP(23):16f}")
#    print(f" FAIL          : {FAIL(24):16f}")
#    print(f" FAILURE       : {FAILURE(25):16f}")
#    print(f" FILL          : {FILL(26):16f}")
#    print(f" FLAY          : {FLAY(27):16f}")
#    print(f" FLDF          : {FLDF(28):16f}")
#    print(f" FLDZ          : {FLDZ(29):16f}")
#    print(f" GROUP         : {GROUP(30):16f}")
#    print(f" HC_DSSE_F     : {HC_DSSE_F(31):16f}")
#    print(f" HC_DSSE_Z     : {HC_DSSE_Z(32):16f}")
#    print(f" HOURGLASS     : {HOURGLASS(33):16f}")
#    print(f" K             : {K(34):16f}")
#    print(f" M151DENS      : {M151DENS(35):16f}")
#    print(f" M151ENER      : {M151ENER(36):16f}")
#    print(f" M151PRES      : {M151PRES(37):16f}")
#    print(f" M151VFRAC     : {M151VFRAC(38):16f}")
#    print(f" MACH          : {MACH(39):16f}")
#    print(f" MASS          : {MASS(40):16f}")
#    print(f" MOMX          : {MOMX(41):16f}")
#    print(f" MOMXY         : {MOMXY(42):16f}")
#    print(f" MOMXZ         : {MOMXZ(43):16f}")
#    print(f" MOMY          : {MOMY(44):16f}")
#    print(f" MOMYZ         : {MOMYZ(45):16f}")
#    print(f" MOMZ          : {MOMZ(46):16f}")
#    print(f" NL_EPSD       : {NL_EPSD(47):16f}")
#    print(f" NL_EPSP       : {NL_EPSP(48):16f}")
#    print(f" NXTF          : {NXTF(49):16f}")
#    print(f" OFF           : {OFF(50):16f}")
#    print(f" P             : {P(51):16f}")
#    print(f" PHI           : {PHI(52):16f}")
#    print(f" SCHLIEREN     : {SCHLIEREN(53):16f}")
#    print(f" SIGEQ         : {SIGEQ(54):16f}")
#    print(f" SIGX          : {SIGX(55):16f}")
#    print(f" SIGXY         : {SIGXY(56):16f}")
#    print(f" SIGY          : {SIGY(57):16f}")
#    print(f" SIGYZ         : {SIGYZ(58):16f}")
#    print(f" SIGZ          : {SIGZ(59):16f}")
#    print(f" SIGZX         : {SIGZX(60):16f}")
#    print(f" SSP           : {SSP(61):16f}")
#    print(f" TDEL          : {TDEL(62):16f}")
#    print(f" TDET          : {TDET(63):16f}")
#    print(f" TEMP          : {TEMP(64):16f}")
#    print(f" THICK         : {THICK(65):16f}")
#    print(f" THIN          : {THIN(66):16f}")
#    print(f" TILLOTSON     : {TILLOTSON(67):16f}")
#    print(f" TSAIWU        : {TSAIWU(68):16f}")
#    print(f" TVIS          : {TVIS(69):16f}")
#    print(f" VDAM1         : {VDAM1(70):16f}")
#    print(f" VDAM2         : {VDAM2(71):16f}")
#    print(f" VDAM3         : {VDAM3(72):16f}")
#    print(f" VELX          : {VELX(73):16f}")
#    print(f" VELXY         : {VELXY(74):16f}")
#    print(f" VELXZ         : {VELXZ(75):16f}")
#    print(f" VELY          : {VELY(76):16f}")
#    print(f" VELYZ         : {VELYZ(77):16f}")
#    print(f" VELZ          : {VELZ(78):16f}")
#    print(f" VFRAC1        : {VFRAC1(79):16f}")
#    print(f" VFRAC2        : {VFRAC2(80):16f}")
#    print(f" VFRAC3        : {VFRAC3(81):16f}")
#    print(f" VFRAC4        : {VFRAC4(82):16f}")
#    print(f" VOLU          : {VOLU(83):16f}")
#    print(f" VONM          : {VONM(84):16f}")
#    print(f" VORT          : {VORT(85):16f}")
#    print(f" VORTX         : {VORTX(86):16f}")
#    print(f" VORTY         : {VORTY(87):16f}")
#    print(f" VORTZ         : {VORTZ(88):16f}")
#    print(f" WPLA          : {WPLA(89):16f}")
    return x



"#;

    /// Builds the three-block deck from transcribed bodies.
    fn python_deck_text() -> String {
        format!(
            "/FUNCT_PYTHON/1\n{BODY_ONE}/FUNCT_PYTHON/21\n{BODY_TWENTYONE}/FUNCT_PYTHON/3\n{BODY_THREE}/END\n"
        )
    }

    #[test]
    fn deck_blocks_collect_with_names() {
        let deck = RadDeck::parse_str(&python_deck_text()).expect("deck");
        let blocks = collect_blocks(&deck).expect("collect");
        assert_eq!(blocks.len(), 3);
        assert_eq!(
            blocks.iter().map(|b| b.id).collect::<Vec<_>>(),
            vec![1, 21, 3]
        );
        assert_eq!(blocks[0].name, "myfun2");
        assert_eq!(blocks[1].name, "myfun4");
        assert_eq!(blocks[2].name, "myfun");
        assert!(blocks.iter().all(|b| b.param == "x"));
        assert_eq!(blocks[0].body.len(), 18);
        assert_eq!(blocks[1].body.len(), 93);
        assert_eq!(blocks[2].body.len(), 12);
    }

    #[test]
    fn sibling_call_links_myfun_to_myfun2() {
        let deck = RadDeck::parse_str(&python_deck_text()).expect("deck");
        let blocks = collect_blocks(&deck).expect("collect");
        assert!(sibling_calls(&blocks[0], &blocks).is_empty());
        assert!(sibling_calls(&blocks[1], &blocks).is_empty());
        assert_eq!(
            sibling_calls(&blocks[2], &blocks),
            vec!["myfun2".to_string()]
        );
        assert_eq!(
            classify(&blocks[2], &blocks),
            Purity::SiblingCalls(vec!["myfun2".to_string()])
        );
    }

    #[test]
    fn nodal_accessors_match_survey() {
        let deck = RadDeck::parse_str(&python_deck_text()).expect("deck");
        let blocks = collect_blocks(&deck).expect("collect");
        let refs = accessor_refs(&blocks[0].body);
        let expected = ["ARX", "AX", "CX", "DRX", "DX", "VRX", "VX"];
        assert_eq!(
            refs,
            expected.iter().map(ToString::to_string).collect::<Vec<_>>()
        );
        assert_eq!(classify(&blocks[0], &blocks), Purity::EngineCoupled(refs));
    }

    /// Element accessor inventory of `/FUNCT_PYTHON/21`, sorted.
    const ELEMENT_ACCESSORS: [&str; 89] = [
        "ALPHA",
        "AMS",
        "BFRAC",
        "BULK",
        "COLOR",
        "DAM1",
        "DAM2",
        "DAM3",
        "DAMA",
        "DAMG",
        "DAMINI",
        "DENS",
        "DOMAIN",
        "DT",
        "EINT",
        "EINTM",
        "EINTV",
        "ENER",
        "ENTH",
        "ENTHM",
        "ENTHV",
        "EPSD",
        "EPSP",
        "FAIL",
        "FAILURE",
        "FILL",
        "FLAY",
        "FLDF",
        "FLDZ",
        "GROUP",
        "HC_DSSE_F",
        "HC_DSSE_Z",
        "HOURGLASS",
        "K",
        "M151DENS",
        "M151ENER",
        "M151PRES",
        "M151VFRAC",
        "MACH",
        "MASS",
        "MOMX",
        "MOMXY",
        "MOMXZ",
        "MOMY",
        "MOMYZ",
        "MOMZ",
        "NL_EPSD",
        "NL_EPSP",
        "NXTF",
        "OFF",
        "P",
        "PHI",
        "SCHLIEREN",
        "SIGEQ",
        "SIGX",
        "SIGXY",
        "SIGY",
        "SIGYZ",
        "SIGZ",
        "SIGZX",
        "SSP",
        "TDEL",
        "TDET",
        "TEMP",
        "THICK",
        "THIN",
        "TILLOTSON",
        "TSAIWU",
        "TVIS",
        "VDAM1",
        "VDAM2",
        "VDAM3",
        "VELX",
        "VELXY",
        "VELXZ",
        "VELY",
        "VELYZ",
        "VELZ",
        "VFRAC1",
        "VFRAC2",
        "VFRAC3",
        "VFRAC4",
        "VOLU",
        "VONM",
        "VORT",
        "VORTX",
        "VORTY",
        "VORTZ",
        "WPLA",
    ];

    #[test]
    fn element_accessors_match_survey() {
        let deck = RadDeck::parse_str(&python_deck_text()).expect("deck");
        let blocks = collect_blocks(&deck).expect("collect");
        let refs = accessor_refs(&blocks[1].body);
        let expected = ELEMENT_ACCESSORS
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert_eq!(refs.len(), 89);
        assert_eq!(refs, expected);
        assert_eq!(classify(&blocks[1], &blocks), Purity::EngineCoupled(refs));
        assert!(accessor_refs(&blocks[2].body).is_empty());
    }

    #[test]
    fn smoothstep_matches_quintic() {
        assert_eq!(smoothstep_unit(-2.0), 0.0);
        assert_eq!(smoothstep_unit(2.0), 1.0);
        assert!((smoothstep_unit(0.25) - 0.103515625).abs() < 1e-15);
        assert!((smoothstep_unit(0.75) - 0.896484375).abs() < 1e-15);
        let mut prev = 0.0;
        for step in 0..=20 {
            let x = f64::from(step) / 20.0;
            let value = smoothstep_unit(x);
            assert!(value >= prev, "monotone at {x}");
            assert!(
                (smoothstep_unit(1.0 - x) - (1.0 - value)).abs() < 1e-15,
                "symmetric at {x}"
            );
            prev = value;
        }
        assert!(smoothstep_unit(f64::NAN).is_nan());
    }

    #[test]
    fn malformed_blocks_fail_closed() {
        for (text, missing) in [
            ("/FUNCT_PYTHON\ndef f(x):\n    return x\n/END\n", true),
            ("/FUNCT_PYTHON/9\nhello\n    return x\n/END\n", false),
            ("/FUNCT_PYTHON/9\ndef f(x, y):\n    return x\n/END\n", false),
            ("/FUNCT_PYTHON/9\ndef f(x):\n/END\n", true),
            ("/FUNCT_PYTHON/9\ndef 9x(x):\n    return x\n/END\n", false),
        ] {
            let deck = RadDeck::parse_str(text).expect("deck parses");
            let error = collect_blocks(&deck).expect_err("must fail");
            assert_eq!(error.is_missing(), missing, "{text}");
            assert_eq!(error.is_malformed(), !missing, "{text}");
        }
        let missing = RadDeck::parse_str("/FUNCT_PYTHON\ndef f(x):\n    return x\n/END\n")
            .expect("deck parses");
        let error = collect_blocks(&missing).expect_err("missing id");
        assert!(error.to_string().contains("header id"));
        assert!(!DEFER_UNTIL.is_empty());
        assert!(!DEFER_MISSING.is_empty());
        assert!(!DEFER_TARGET.is_empty());
    }

    #[test]
    fn vendor_deck_matches_transcriptions() {
        let Ok(root) = std::env::var(crate::rad::oc_golden::MINIQA_ENV) else {
            eprintln!("skipped: set OPENCOURANT_MINIQA to the miniqa root");
            return;
        };
        let deck = RadDeck::parse_file(
            std::path::PathBuf::from(root)
                .join("PYTHON/FUNCT_PYTHON/data/test_torsion_orthstrain_shell_0000.rad"),
        )
        .expect("deck parses");
        let blocks = collect_blocks(&deck).expect("collect");
        assert_eq!(blocks.len(), 3);
        for (block, body) in [
            (&blocks[0], BODY_ONE),
            (&blocks[1], BODY_TWENTYONE),
            (&blocks[2], BODY_THREE),
        ] {
            let expected = body.lines().map(str::to_string).collect::<Vec<_>>();
            let mut actual = vec![format!("def {}({}):", block.name, block.param)];
            actual.extend(block.body.clone());
            assert_eq!(actual, expected, "block {}", block.id);
        }
        assert_eq!(
            sibling_calls(&blocks[2], &blocks),
            vec!["myfun2".to_string()]
        );
        assert_eq!(accessor_refs(&blocks[0].body).len(), 7);
        assert_eq!(accessor_refs(&blocks[1].body).len(), 89);
    }

    /// Builds a single-parameter block from raw body lines.
    #[cfg(feature = "funct-python")]
    fn eval_block(id: i64, name: &str, body: &[&str]) -> FunctPythonBlock {
        FunctPythonBlock {
            id,
            name: name.to_string(),
            param: "x".to_string(),
            body: body.iter().map(|line| line.to_string()).collect(),
        }
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_pure_math() {
        let blocks = [eval_block(1, "linear", &["    return 2 * x + 1"])];
        let context = PythonContext::default();
        let value = evaluate_python(&blocks, 1, 3.0, &context).expect("eval");
        assert_eq!(value, 7.0);
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_body_import() {
        let blocks = [eval_block(
            1,
            "floored",
            &["    import math", "    return math.floor(x)"],
        )];
        let context = PythonContext::default();
        let value = evaluate_python(&blocks, 1, 2.7, &context).expect("eval");
        assert_eq!(value, 2.0);
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_sibling_call() {
        let blocks = [
            eval_block(1, "double", &["    return 2 * x"]),
            eval_block(2, "quad", &["    return 2 * double(x)"]),
        ];
        let context = PythonContext::default();
        let value = evaluate_python(&blocks, 2, 3.0, &context).expect("eval");
        assert_eq!(value, 12.0);
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_time_scalars() {
        let blocks = [eval_block(1, "clock", &["    return TIME + DT"])];
        let context = PythonContext {
            time: 1.5,
            dt: 0.001,
            ..Default::default()
        };
        let value = evaluate_python(&blocks, 1, 0.0, &context).expect("eval");
        assert!((value - 1.501).abs() < 1e-12, "value {value}");
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_node_disp_str_dof() {
        let blocks = [eval_block(1, "sense", &["    return node_disp(4, 'X')"])];
        let context = PythonContext {
            nodal_disp: BTreeMap::from([((4, 'X'), 2.5)]),
            ..Default::default()
        };
        let value = evaluate_python(&blocks, 1, 0.0, &context).expect("eval");
        assert_eq!(value, 2.5);
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_node_vel_int_dof() {
        let blocks = [eval_block(1, "sense", &["    return node_vel(4, 2)"])];
        let context = PythonContext {
            nodal_vel: BTreeMap::from([((4, 'Y'), -1.0)]),
            ..Default::default()
        };
        let value = evaluate_python(&blocks, 1, 0.0, &context).expect("eval");
        assert_eq!(value, -1.0);
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_node_acce_lower_dof() {
        let blocks = [eval_block(1, "sense", &["    return node_acce(7, 'z')"])];
        let context = PythonContext {
            nodal_acce: BTreeMap::from([((7, 'Z'), 3.0)]),
            ..Default::default()
        };
        let value = evaluate_python(&blocks, 1, 0.0, &context).expect("eval");
        assert_eq!(value, 3.0);
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_element_var() {
        let blocks = [eval_block(
            1,
            "sense",
            &["    return element_var(9, 'HC_DSSE_F')"],
        )];
        let context = PythonContext {
            element_var: BTreeMap::from([((9, "HC_DSSE_F".to_string()), 0.25)]),
            ..Default::default()
        };
        let value = evaluate_python(&blocks, 1, 0.0, &context).expect("eval");
        assert_eq!(value, 0.25);
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_missing_snapshot_fails_closed() {
        let blocks = [eval_block(1, "sense", &["    return node_disp(4, 'X')"])];
        let context = PythonContext::default();
        let err = evaluate_python(&blocks, 1, 0.0, &context).expect_err("must fail");
        assert!(err.is_evaluation(), "kind {err}");
        assert!(!err.is_unavailable(), "kind {err}");
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_bad_dof_fails_closed() {
        let blocks = [eval_block(1, "sense", &["    return node_disp(4, 'W')"])];
        let context = PythonContext {
            nodal_disp: BTreeMap::from([((4, 'X'), 2.5)]),
            ..Default::default()
        };
        let err = evaluate_python(&blocks, 1, 0.0, &context).expect_err("must fail");
        assert!(err.is_evaluation(), "kind {err}");
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_unknown_id_fails_closed() {
        let blocks = [eval_block(1, "linear", &["    return x"])];
        let context = PythonContext::default();
        let err = evaluate_python(&blocks, 999, 1.0, &context).expect_err("must fail");
        assert!(err.is_evaluation(), "kind {err}");
        assert!(!err.is_unavailable(), "kind {err}");
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_exception_fails_closed() {
        let blocks = [eval_block(1, "boom", &["    raise ValueError('boom')"])];
        let context = PythonContext::default();
        let err = evaluate_python(&blocks, 1, 1.0, &context).expect_err("must fail");
        assert!(err.is_evaluation(), "kind {err}");
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_syntax_error_fails_closed() {
        let blocks = [eval_block(1, "broken", &["    return x +"])];
        let context = PythonContext::default();
        let err = evaluate_python(&blocks, 1, 1.0, &context).expect_err("must fail");
        assert!(err.is_evaluation(), "kind {err}");
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_non_float_fails_closed() {
        let blocks = [eval_block(1, "words", &["    return 'nope'"])];
        let context = PythonContext::default();
        let err = evaluate_python(&blocks, 1, 1.0, &context).expect_err("must fail");
        assert!(err.is_evaluation(), "kind {err}");
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_non_finite_fails_closed() {
        let blocks = [eval_block(1, "huge", &["    return float('inf')"])];
        let context = PythonContext::default();
        let err = evaluate_python(&blocks, 1, 1.0, &context).expect_err("must fail");
        assert!(err.is_evaluation(), "kind {err}");
    }

    #[cfg(feature = "funct-python")]
    #[test]
    fn eval_deck_roundtrip() {
        let deck = RadDeck::parse_str("/FUNCT_PYTHON/1\ndef scaled(x):\n    return 3 * x\n/END\n")
            .expect("deck");
        let blocks = collect_blocks(&deck).expect("collect");
        let context = PythonContext::default();
        let value = evaluate_python(&blocks, 1, 2.0, &context).expect("eval");
        assert_eq!(value, 6.0);
    }

    #[cfg(not(feature = "funct-python"))]
    #[test]
    fn eval_off_feature_is_unavailable() {
        let context = PythonContext::default();
        let err = evaluate_python(&[], 1, 0.0, &context).expect_err("must defer");
        assert!(err.is_unavailable(), "kind {err}");
        assert!(!err.is_evaluation(), "kind {err}");
    }
}
