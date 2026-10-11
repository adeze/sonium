//! SUBMODEL sections and TRANSF transforms for Radioss decks.
//!
//! Applies `//SUBMODEL` section splits and merges plus `/TRANSFORM`
//! node-group maps onto decks parsed by [`RadDeck`](crate::rad_parse::RadDeck).
//! Everything here borrows the parsed deck; the reader stays untouched.
//!
//! # Deck survey (miniqa, Phase-5 lane 5E)
//!
//! `SUBMODEL/TRANSF` (`BOULE1V5_0000.rad`, `/BEGIN` 2019) carries four
//! node-group transforms on `/GRNOD/NODE/18` (nodes 1000-1039), no
//! `//SUBMODEL` section:
//!
//! | card | title | group | parameters (deck-transcribed) |
//! |---|---|---|---|
//! | `/TRANSFORM/SYM/1` | sym_back | 18 | nodes 1089, 1086 |
//! | `/TRANSFORM/SCA/2` | scale_back | 18 | factors 5, 2.5, 1.25, center 1114 |
//! | `/TRANSFORM/ROT/3` | rot_back | 18 | nodes 1095, 1115, angle -90 |
//! | `/TRANSFORM/TRA/4` | tra_back | 18 | shift X -5 |
//!
//! The three `SUBMODEL/update_v_100` decks wrap their whole model in one
//! `//SUBMODEL/1` with empty offset rows (all seven offsets zero) and no
//! nesting: ACTIV (`HYPE_SUB`, 41 cards inside), PLOAD (`BOXBEAM_SUB`,
//! 77 inside), FAIL (`SPECT_SUB`, 38 inside). Each closes with `//ENDSUB`
//! before `/END`. ACTIV nests `/INTER/TYPE3` inside its submodel although
//! the Reference Guide lists TYPE3 as submodel-incompatible; the vendor
//! starter accepts and runs the deck, so this module accepts it too.
//!
//! # Semantics and verification
//!
//! Field layouts come from the Reference Guide (`//SUBMODEL` pp. 198-202,
//! `/TRANSFORM/*` pp. 2188-2204) and the `hm_cfg_files` card formats (note
//! the 20-character real fields on transform cards). Application order
//! follows `lectrans.F`: transforms run in deck order, cumulatively, after
//! group building, against current coordinates. Rotation nets out to a
//! standard right-hand rotation by the deck angle about the N1-N0 axis
//! (`euler_mrot.F` builds `R(-a)` and `euler_vrot.F` applies it
//! transposed); scale maps a zero factor to one; translation adds its
//! vector. A fresh double-precision starter run on the TRANSF deck
//! (2026-10-11, dev-time, scratch) reproduces every stage of all 40 nodes
//! within 5e-13 of this module; the committed tests pin that agreement.
//!
//! # Coverage boundaries (fail closed, never silent)
//!
//! * Node-group (`grnd_ID`) transforms only. Submodel-targeted transforms
//!   have no deck coverage and need full-option rewriting, so decks that
//!   require them fail group resolution instead of misreading.
//! * `TRA` skew-frame rows and nonzero submodel id offsets are decoded
//!   but rejected with an explicit error naming the blocker.
//! * The reader keeps whitespace tokens, so sparse rows with leading gaps
//!   share token shapes with dense rows; decoding assumes left-dense rows
//!   (true of every surveyed deck row) and notes the sparse-row ambiguity
//!   for the coordinator. Node-vs-point axis disambiguation resolves
//!   through the deck node ids.
//!
//! # Examples
//!
//! ```rust
//! use sonium_qa::rad_parse::RadDeck;
//! use sonium_qa::rad_submodel::apply_transforms;
//!
//! let deck = RadDeck::parse_str(
//!     "/NODE\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n/GRNOD/NODE/7\n1 2\n/TRANSFORM/TRA/4\ntra\n7 1.0 2.0 3.0\n/END\n",
//! )
//! .expect("deck");
//! let applied = apply_transforms(&deck).expect("apply");
//! assert_eq!(applied.final_nodes[&1], [1.0, 2.0, 3.0]);
//! assert_eq!(applied.final_nodes[&2], [2.0, 2.0, 3.0]);
//! ```

// Rust guideline compliant 2026-02-21

use crate::rad_parse::{CardKind, DeckCard, RadDeck, TypedCard};
use std::backtrace::Backtrace;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};

/// Private failure modes behind [`SubmodelError`].
#[derive(Debug)]
enum ErrorKind {
    /// Transform card without its header identifier.
    MissingTransformId,
    /// Wrong data-row count on a transform card.
    BadRowCount { kind: &'static str, rows: usize },
    /// Wrong token count on a transform data row.
    BadFieldCount {
        kind: &'static str,
        row: usize,
        fields: usize,
    },
    /// Token that is not a number where one is required.
    BadNumber { kind: &'static str, token: String },
    /// Single axis node where the vendor demands a pair.
    SingleNodeRef { kind: &'static str },
    /// Node id with no `/NODE` entry.
    UnknownNode { id: i64 },
    /// Group id with no `/GRNOD` entry.
    UnknownGroup { id: i64 },
    /// Group flavor outside node-list resolution.
    UnsupportedGroup { kind: &'static str, id: i64 },
    /// Cyclic `/GRNOD/GRNOD` nesting.
    GroupCycle { id: i64 },
    /// Duplicate group definition.
    DuplicateGroup { id: i64 },
    /// Duplicate node id across `/NODE` cards.
    DuplicateNode { id: i64 },
    /// Transforms with no node table to apply to.
    EmptyNodeTable,
    /// Zero-length rotation or symmetry axis.
    DegenerateAxis { kind: &'static str },
    /// Translation in a skew frame (not deck-covered).
    SkewTranslation { id: i64 },
    /// Card that is not `//SUBMODEL` passed to the offset decoder.
    NotSubmodelCard,
    /// `//SUBMODEL` marker without its header identifier.
    MissingSubmodelId,
    /// `//ENDSUB` with no open `//SUBMODEL`.
    UnbalancedEndsub { index: usize },
    /// `//SUBMODEL` never closed by `//ENDSUB`.
    UnterminatedSubmodel { id: i64 },
    /// Malformed submodel offset row.
    BadOffset { detail: String },
    /// Merge with nonzero effective id offsets.
    NonzeroOffsets { id: i64, offsets: String },
}

/// SUBMODEL split, merge, or transform failure.
#[derive(Debug)]
pub struct SubmodelError {
    kind: ErrorKind,
    backtrace: Backtrace,
}

impl SubmodelError {
    /// Reports a malformed transform or offset row.
    #[must_use]
    pub const fn is_decode(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::MissingTransformId
                | ErrorKind::BadRowCount { .. }
                | ErrorKind::BadFieldCount { .. }
                | ErrorKind::BadNumber { .. }
                | ErrorKind::SingleNodeRef { .. }
                | ErrorKind::NotSubmodelCard
                | ErrorKind::BadOffset { .. }
        )
    }

    /// Reports a missing or unresolvable deck entity.
    #[must_use]
    pub const fn is_lookup(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::UnknownNode { .. }
                | ErrorKind::UnknownGroup { .. }
                | ErrorKind::UnsupportedGroup { .. }
                | ErrorKind::GroupCycle { .. }
                | ErrorKind::DuplicateGroup { .. }
                | ErrorKind::DuplicateNode { .. }
                | ErrorKind::EmptyNodeTable
        )
    }

    /// Reports a zero-length rotation or symmetry axis.
    #[must_use]
    pub const fn is_degenerate(&self) -> bool {
        matches!(self.kind, ErrorKind::DegenerateAxis { .. })
    }

    /// Reports a decoded-but-unapplied construct (skew, offsets).
    #[must_use]
    pub const fn is_unsupported(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::SkewTranslation { .. } | ErrorKind::NonzeroOffsets { .. }
        )
    }

    /// Reports unbalanced `//SUBMODEL` / `//ENDSUB` markers.
    #[must_use]
    pub const fn is_layout(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::MissingSubmodelId
                | ErrorKind::UnbalancedEndsub { .. }
                | ErrorKind::UnterminatedSubmodel { .. }
        )
    }

    fn from_kind(kind: ErrorKind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }
}

impl Display for SubmodelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ErrorKind::MissingTransformId => {
                write!(f, "transform card lacks its header id\n{}", self.backtrace)
            }
            ErrorKind::BadRowCount { kind, rows } => {
                write!(f, "{kind} carries {rows} data rows\n{}", self.backtrace)
            }
            ErrorKind::BadFieldCount { kind, row, fields } => write!(
                f,
                "{kind} row {row} carries {fields} fields\n{}",
                self.backtrace
            ),
            ErrorKind::BadNumber { kind, token } => write!(
                f,
                "{kind} token {token:?} is not a number\n{}",
                self.backtrace
            ),
            ErrorKind::SingleNodeRef { kind } => {
                write!(f, "{kind} names a single axis node\n{}", self.backtrace)
            }
            ErrorKind::UnknownNode { id } => {
                write!(f, "node {id} has no /NODE entry\n{}", self.backtrace)
            }
            ErrorKind::UnknownGroup { id } => {
                write!(f, "group {id} has no /GRNOD entry\n{}", self.backtrace)
            }
            ErrorKind::UnsupportedGroup { kind, id } => {
                write!(f, "group {id} uses unsupported {kind}\n{}", self.backtrace)
            }
            ErrorKind::GroupCycle { id } => {
                write!(f, "group {id} nests cyclically\n{}", self.backtrace)
            }
            ErrorKind::DuplicateGroup { id } => {
                write!(f, "group {id} repeats in deck\n{}", self.backtrace)
            }
            ErrorKind::DuplicateNode { id } => {
                write!(f, "node {id} repeats across /NODE\n{}", self.backtrace)
            }
            ErrorKind::EmptyNodeTable => {
                write!(f, "transforms with no node table\n{}", self.backtrace)
            }
            ErrorKind::DegenerateAxis { kind } => {
                write!(f, "{kind} axis has zero length\n{}", self.backtrace)
            }
            ErrorKind::SkewTranslation { id } => {
                write!(f, "translation {id} needs a skew frame\n{}", self.backtrace)
            }
            ErrorKind::NotSubmodelCard => {
                write!(f, "offset decode needs //SUBMODEL\n{}", self.backtrace)
            }
            ErrorKind::MissingSubmodelId => {
                write!(f, "//SUBMODEL lacks its header id\n{}", self.backtrace)
            }
            ErrorKind::UnbalancedEndsub { index } => write!(
                f,
                "//ENDSUB at card {index} opens nothing\n{}",
                self.backtrace
            ),
            ErrorKind::UnterminatedSubmodel { id } => {
                write!(f, "//SUBMODEL/{id} never closes\n{}", self.backtrace)
            }
            ErrorKind::BadOffset { detail } => {
                write!(f, "bad submodel offsets: {detail}\n{}", self.backtrace)
            }
            ErrorKind::NonzeroOffsets { id, offsets } => write!(
                f,
                "//SUBMODEL/{id} offsets [{offsets}] need rewriting\n{}",
                self.backtrace
            ),
        }
    }
}

impl std::error::Error for SubmodelError {}

/// Degrees-to-radians factor for rotation angles.
///
/// The vendor multiplies `ANGLE * PI / 180` (`lectrans.F`); this constant
/// keeps that exact sequence. Changing it rescales every rotation.
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// Axis-length floor below which an axis is degenerate.
///
/// The vendor divides by `MAX(EM20, S)` with `EM20 = 1e-20`
/// (`constant_mod.F`, `lectrans.F`); shorter axes would spin numerical
/// garbage, so this module fails closed instead (documented deviation).
const AXIS_FLOOR: f64 = 1e-20;

/// Reflects a point across a plane.
///
/// The plane passes through `center` with the given normal `direction`,
/// matching `lectrans.F`: `p - 2*t*((p-c).t)` with a normalized axis.
/// Returns `None` when the direction is shorter than [`AXIS_FLOOR`].
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_submodel::reflect_point;
///
/// let mirrored = reflect_point([3.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
/// assert_eq!(mirrored, Some([-3.0, 1.0, 0.0]));
/// assert_eq!(reflect_point([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), None);
/// ```
#[must_use]
pub fn reflect_point(point: [f64; 3], center: [f64; 3], direction: [f64; 3]) -> Option<[f64; 3]> {
    let length =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    if length < AXIS_FLOOR {
        return None;
    }
    let axis = [
        direction[0] / length,
        direction[1] / length,
        direction[2] / length,
    ];
    let along = (point[0] - center[0]) * axis[0]
        + (point[1] - center[1]) * axis[1]
        + (point[2] - center[2]) * axis[2];
    Some([
        point[0] - 2.0 * axis[0] * along,
        point[1] - 2.0 * axis[1] * along,
        point[2] - 2.0 * axis[2] * along,
    ])
}

/// Scales a point about a center.
///
/// A zero factor scales by one, matching the vendor `IF (TX == ZERO)
/// TX = ONE` mapping (`lectrans.F`); omitted deck factors are zero.
/// Total: every input maps to exactly one output.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_submodel::scale_point;
///
/// assert_eq!(scale_point([3.0, 4.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]), [5.0, 4.0, 0.0]);
/// ```
#[must_use]
pub fn scale_point(point: [f64; 3], center: [f64; 3], factors: [f64; 3]) -> [f64; 3] {
    let applied = [
        if factors[0] == 0.0 { 1.0 } else { factors[0] },
        if factors[1] == 0.0 { 1.0 } else { factors[1] },
        if factors[2] == 0.0 { 1.0 } else { factors[2] },
    ];
    [
        center[0] + (point[0] - center[0]) * applied[0],
        center[1] + (point[1] - center[1]) * applied[1],
        center[2] + (point[2] - center[2]) * applied[2],
    ]
}

/// Builds a right-hand rotation matrix.
///
/// The matrix rotates by `angle_deg` (signed degrees) about `axis`,
/// matching the net vendor rotation (`euler_mrot.F` plus `euler_vrot.F`).
/// Returns `None` when the axis is shorter than [`AXIS_FLOOR`].
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_submodel::rotation_matrix;
///
/// let rot = rotation_matrix([0.0, 0.0, 1.0], -90.0).expect("axis");
/// assert!((rot[0][1] - 1.0).abs() < 1e-12);
/// assert!((rot[1][0] + 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn rotation_matrix(axis: [f64; 3], angle_deg: f64) -> Option<[[f64; 3]; 3]> {
    let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if length < AXIS_FLOOR {
        return None;
    }
    let unit = [axis[0] / length, axis[1] / length, axis[2] / length];
    let angle = angle_deg * DEG_TO_RAD;
    let (sin, cos) = angle.sin_cos();
    let xx = unit[0] * unit[0] * (1.0 - cos) + cos;
    let yy = unit[1] * unit[1] * (1.0 - cos) + cos;
    let zz = unit[2] * unit[2] * (1.0 - cos) + cos;
    let xy = unit[0] * unit[1] * (1.0 - cos) - sin * unit[2];
    let yx = unit[1] * unit[0] * (1.0 - cos) + sin * unit[2];
    let xz = unit[0] * unit[2] * (1.0 - cos) + sin * unit[1];
    let zx = unit[2] * unit[0] * (1.0 - cos) - sin * unit[1];
    let yz = unit[1] * unit[2] * (1.0 - cos) - sin * unit[0];
    let zy = unit[2] * unit[1] * (1.0 - cos) + sin * unit[0];
    Some([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])
}

/// Rotates a point about a center.
///
/// Combines [`rotation_matrix`] with the center offset, matching
/// `euler_vrot.F`: `X0 + R*(X - X0)`. Returns `None` on a degenerate
/// axis; callers skip zero angles before reaching this function.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_submodel::rotate_point;
///
/// let turned = rotate_point([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 90.0).expect("axis");
/// assert!((turned[0]).abs() < 1e-12);
/// assert!((turned[1] - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn rotate_point(
    point: [f64; 3],
    center: [f64; 3],
    axis: [f64; 3],
    angle_deg: f64,
) -> Option<[f64; 3]> {
    let matrix = rotation_matrix(axis, angle_deg)?;
    let relative = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    Some([
        center[0]
            + matrix[0][0] * relative[0]
            + matrix[0][1] * relative[1]
            + matrix[0][2] * relative[2],
        center[1]
            + matrix[1][0] * relative[0]
            + matrix[1][1] * relative[1]
            + matrix[1][2] * relative[2],
        center[2]
            + matrix[2][0] * relative[0]
            + matrix[2][1] * relative[1]
            + matrix[2][2] * relative[2],
    ])
}

/// Translates a point by a shift vector.
///
/// Total: every input maps to exactly one output.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_submodel::translate_point;
///
/// assert_eq!(translate_point([1.0, 2.0, 3.0], [-5.0, 0.0, 0.0]), [-4.0, 2.0, 3.0]);
/// ```
#[must_use]
pub fn translate_point(point: [f64; 3], shift: [f64; 3]) -> [f64; 3] {
    [
        point[0] + shift[0],
        point[1] + shift[1],
        point[2] + shift[2],
    ]
}

/// Node-group transform family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformKind {
    /// Plane symmetry (`/TRANSFORM/SYM`).
    Sym,
    /// Anisotropic scaling (`/TRANSFORM/SCA`).
    Sca,
    /// Rotation about an axis (`/TRANSFORM/ROT`).
    Rot,
    /// Translation by a vector (`/TRANSFORM/TRA`).
    Tra,
}

/// Axis defined by deck nodes or raw points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxisSpec {
    /// Axis from node `0` through node `1` (center at node `0`).
    Nodes(i64, i64),
    /// Axis from point `0` through point `1` (center at point `0`).
    Points([f64; 3], [f64; 3]),
}

/// Translation vector from values or deck nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShiftSpec {
    /// Explicit shift in deck length units.
    Values([f64; 3]),
    /// Shift from node `0` to node `1`.
    Nodes(i64, i64),
}

/// Decoded transform parameters by family.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformParams {
    /// Symmetry about the plane through the axis origin.
    Sym { axis: AxisSpec },
    /// Scaling with raw factors and optional center node.
    Sca {
        factors: [f64; 3],
        center: Option<i64>,
    },
    /// Rotation about the axis by signed degrees.
    Rot { axis: AxisSpec, angle_deg: f64 },
    /// Translation by values or node span.
    Tra { shift: ShiftSpec },
}

/// One decoded node-group transform.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    /// Transform identifier from the card header.
    pub id: i64,
    /// Target node group identifier.
    pub group: i64,
    /// Family-specific parameters.
    pub params: TransformParams,
}

impl Transform {
    /// Reports the transform family.
    #[must_use]
    pub const fn kind(&self) -> TransformKind {
        match self.params {
            TransformParams::Sym { .. } => TransformKind::Sym,
            TransformParams::Sca { .. } => TransformKind::Sca,
            TransformParams::Rot { .. } => TransformKind::Rot,
            TransformParams::Tra { .. } => TransformKind::Tra,
        }
    }
}

/// Parses a deck integer token.
fn parse_int(token: &str) -> Option<i64> {
    token.parse::<i64>().ok()
}

/// Parses a deck real token.
///
/// Accepts Fortran `D` exponents like the reader and rejects
/// non-finite values, which are malformed data, not parameters.
fn parse_real(token: &str) -> Option<f64> {
    let normalized = token.replace('D', "E").replace('d', "e");
    normalized.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Decodes the group id heading a transform data row.
fn decode_group(kind: &'static str, row: &[String]) -> Result<i64, SubmodelError> {
    let Some(first) = row.first() else {
        return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
            kind,
            row: 0,
            fields: 0,
        }));
    };
    parse_int(first).ok_or_else(|| {
        SubmodelError::from_kind(ErrorKind::BadNumber {
            kind,
            token: first.clone(),
        })
    })
}

/// Decodes three consecutive reals starting at `offset`.
fn decode_triple(
    kind: &'static str,
    row_index: usize,
    row: &[String],
    offset: usize,
) -> Result<[f64; 3], SubmodelError> {
    let mut out = [0.0; 3];
    for (slot, token) in out.iter_mut().zip(row.iter().skip(offset)) {
        let Some(value) = parse_real(token) else {
            return Err(SubmodelError::from_kind(ErrorKind::BadNumber {
                kind,
                token: token.clone(),
            }));
        };
        *slot = value;
    }
    if row.len() < offset + 3 {
        return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
            kind,
            row: row_index,
            fields: row.len(),
        }));
    }
    Ok(out)
}

/// Reports whether both tokens name deck nodes.
fn are_node_pair(first: &str, second: &str, nodes: &BTreeSet<i64>) -> bool {
    match (parse_int(first), parse_int(second)) {
        (Some(a), Some(b)) => nodes.contains(&a) && nodes.contains(&b),
        _ => false,
    }
}

/// Decodes a `/TRANSFORM/SYM` card.
fn decode_sym(id: i64, card: &DeckCard, nodes: &BTreeSet<i64>) -> Result<Transform, SubmodelError> {
    const KIND: &str = "SYM";
    if card.fields.len() > 2 {
        return Err(SubmodelError::from_kind(ErrorKind::BadRowCount {
            kind: KIND,
            rows: card.fields.len(),
        }));
    }
    let Some(head) = card.fields.first() else {
        return Err(SubmodelError::from_kind(ErrorKind::BadRowCount {
            kind: KIND,
            rows: 0,
        }));
    };
    let group = decode_group(KIND, head)?;
    let rest = &head[1..];
    if rest.len() == 1 {
        return Err(SubmodelError::from_kind(ErrorKind::SingleNodeRef {
            kind: KIND,
        }));
    }
    if rest.len() == 2 && are_node_pair(&rest[0], &rest[1], nodes) {
        if let Some(second) = card.fields.get(1) {
            for token in second {
                if parse_real(token).is_none() {
                    return Err(SubmodelError::from_kind(ErrorKind::BadNumber {
                        kind: KIND,
                        token: token.clone(),
                    }));
                }
            }
        }
        let axis = AxisSpec::Nodes(
            parse_int(&rest[0]).expect("node pair"),
            parse_int(&rest[1]).expect("node pair"),
        );
        return Ok(Transform {
            id,
            group,
            params: TransformParams::Sym { axis },
        });
    }
    if rest.len() == 2 {
        let missing = [parse_int(&rest[0]), parse_int(&rest[1])]
            .into_iter()
            .flatten()
            .find(|node| !nodes.contains(node));
        if let Some(node) = missing {
            return Err(SubmodelError::from_kind(ErrorKind::UnknownNode {
                id: node,
            }));
        }
    }
    if rest.len() != 3 {
        return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
            kind: KIND,
            row: 0,
            fields: head.len(),
        }));
    }
    let first = decode_triple(KIND, 0, head, 1)?;
    let second = match card.fields.get(1) {
        None => [0.0; 3],
        Some(row) if row.len() == 3 => decode_triple(KIND, 1, row, 0)?,
        Some(row) => {
            return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
                kind: KIND,
                row: 1,
                fields: row.len(),
            }));
        }
    };
    Ok(Transform {
        id,
        group,
        params: TransformParams::Sym {
            axis: AxisSpec::Points(first, second),
        },
    })
}

/// Decodes a `/TRANSFORM/SCA` card.
fn decode_sca(id: i64, card: &DeckCard, nodes: &BTreeSet<i64>) -> Result<Transform, SubmodelError> {
    const KIND: &str = "SCA";
    if card.fields.len() != 1 {
        return Err(SubmodelError::from_kind(ErrorKind::BadRowCount {
            kind: KIND,
            rows: card.fields.len(),
        }));
    }
    let head = &card.fields[0];
    let group = decode_group(KIND, head)?;
    if head.len() < 4 || head.len() > 5 {
        return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
            kind: KIND,
            row: 0,
            fields: head.len(),
        }));
    }
    let factors = decode_triple(KIND, 0, head, 1)?;
    let center = match head.get(4) {
        None => None,
        Some(token) => {
            let Some(node) = parse_int(token) else {
                return Err(SubmodelError::from_kind(ErrorKind::BadNumber {
                    kind: KIND,
                    token: token.clone(),
                }));
            };
            if node != 0 && !nodes.contains(&node) {
                return Err(SubmodelError::from_kind(ErrorKind::UnknownNode {
                    id: node,
                }));
            }
            if node == 0 { None } else { Some(node) }
        }
    };
    Ok(Transform {
        id,
        group,
        params: TransformParams::Sca { factors, center },
    })
}

/// Decodes a `/TRANSFORM/ROT` card.
fn decode_rot(id: i64, card: &DeckCard, nodes: &BTreeSet<i64>) -> Result<Transform, SubmodelError> {
    const KIND: &str = "ROT";
    if card.fields.len() > 2 {
        return Err(SubmodelError::from_kind(ErrorKind::BadRowCount {
            kind: KIND,
            rows: card.fields.len(),
        }));
    }
    let Some(head) = card.fields.first() else {
        return Err(SubmodelError::from_kind(ErrorKind::BadRowCount {
            kind: KIND,
            rows: 0,
        }));
    };
    let group = decode_group(KIND, head)?;
    let rest = &head[1..];
    if rest.len() == 1 {
        return Err(SubmodelError::from_kind(ErrorKind::SingleNodeRef {
            kind: KIND,
        }));
    }
    if rest.len() == 2 {
        if !are_node_pair(&rest[0], &rest[1], nodes) {
            let missing = [parse_int(&rest[0]), parse_int(&rest[1])]
                .into_iter()
                .flatten()
                .find(|node| !nodes.contains(node));
            if let Some(node) = missing {
                return Err(SubmodelError::from_kind(ErrorKind::UnknownNode {
                    id: node,
                }));
            }
            return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
                kind: KIND,
                row: 0,
                fields: head.len(),
            }));
        }
        let angle = match card.fields.get(1) {
            None => 0.0,
            Some(row) if row.len() == 1 => parse_real(&row[0]).ok_or_else(|| {
                SubmodelError::from_kind(ErrorKind::BadNumber {
                    kind: KIND,
                    token: row[0].clone(),
                })
            })?,
            Some(row) => {
                return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
                    kind: KIND,
                    row: 1,
                    fields: row.len(),
                }));
            }
        };
        let axis = AxisSpec::Nodes(
            parse_int(&rest[0]).expect("node pair"),
            parse_int(&rest[1]).expect("node pair"),
        );
        return Ok(Transform {
            id,
            group,
            params: TransformParams::Rot {
                axis,
                angle_deg: angle,
            },
        });
    }
    if rest.len() != 3 {
        return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
            kind: KIND,
            row: 0,
            fields: head.len(),
        }));
    }
    let first = decode_triple(KIND, 0, head, 1)?;
    let (second, angle) = match card.fields.get(1) {
        None => ([0.0; 3], 0.0),
        Some(row) if row.len() == 3 => (decode_triple(KIND, 1, row, 0)?, 0.0),
        Some(row) if row.len() == 4 => {
            let point = decode_triple(KIND, 1, row, 0)?;
            let angle = parse_real(&row[3]).ok_or_else(|| {
                SubmodelError::from_kind(ErrorKind::BadNumber {
                    kind: KIND,
                    token: row[3].clone(),
                })
            })?;
            (point, angle)
        }
        Some(row) => {
            return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
                kind: KIND,
                row: 1,
                fields: row.len(),
            }));
        }
    };
    Ok(Transform {
        id,
        group,
        params: TransformParams::Rot {
            axis: AxisSpec::Points(first, second),
            angle_deg: angle,
        },
    })
}

/// Decodes a `/TRANSFORM/TRA` card.
fn decode_tra(id: i64, card: &DeckCard, nodes: &BTreeSet<i64>) -> Result<Transform, SubmodelError> {
    const KIND: &str = "TRA";
    if card.fields.len() > 2 {
        return Err(SubmodelError::from_kind(ErrorKind::BadRowCount {
            kind: KIND,
            rows: card.fields.len(),
        }));
    }
    let Some(head) = card.fields.first() else {
        return Err(SubmodelError::from_kind(ErrorKind::BadRowCount {
            kind: KIND,
            rows: 0,
        }));
    };
    let group = decode_group(KIND, head)?;
    if let Some(second) = card.fields.get(1) {
        if second.len() != 1 {
            return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
                kind: KIND,
                row: 1,
                fields: second.len(),
            }));
        }
        let Some(skew) = parse_int(&second[0]) else {
            return Err(SubmodelError::from_kind(ErrorKind::BadNumber {
                kind: KIND,
                token: second[0].clone(),
            }));
        };
        if skew != 0 {
            return Err(SubmodelError::from_kind(ErrorKind::SkewTranslation { id }));
        }
    }
    let rest = &head[1..];
    if rest.len() == 2 && are_node_pair(&rest[0], &rest[1], nodes) {
        let shift = ShiftSpec::Nodes(
            parse_int(&rest[0]).expect("node pair"),
            parse_int(&rest[1]).expect("node pair"),
        );
        return Ok(Transform {
            id,
            group,
            params: TransformParams::Tra { shift },
        });
    }
    if rest.len() == 2 {
        let missing = [parse_int(&rest[0]), parse_int(&rest[1])]
            .into_iter()
            .flatten()
            .find(|node| !nodes.contains(node));
        if let Some(node) = missing {
            return Err(SubmodelError::from_kind(ErrorKind::UnknownNode {
                id: node,
            }));
        }
    }
    if rest.len() > 3 {
        return Err(SubmodelError::from_kind(ErrorKind::BadFieldCount {
            kind: KIND,
            row: 0,
            fields: head.len(),
        }));
    }
    let mut shift = [0.0; 3];
    for (slot, token) in shift.iter_mut().zip(rest.iter()) {
        let Some(value) = parse_real(token) else {
            return Err(SubmodelError::from_kind(ErrorKind::BadNumber {
                kind: KIND,
                token: token.clone(),
            }));
        };
        *slot = value;
    }
    Ok(Transform {
        id,
        group,
        params: TransformParams::Tra {
            shift: ShiftSpec::Values(shift),
        },
    })
}

/// Decodes one transform card.
///
/// `nodes` holds the deck node ids and disambiguates node-form axes
/// from point-form coordinates: two trailing integers that both name
/// deck nodes decode as a node pair. Parameterless axis transforms are
/// rejected (the vendor would silently no-op or worse); a group-only
/// translation decodes as a zero shift.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_parse::RadDeck;
/// use sonium_qa::rad_submodel::{AxisSpec, TransformParams, decode_transform_card};
/// use std::collections::BTreeSet;
///
/// let deck = RadDeck::parse_str("/TRANSFORM/SYM/1\nsym\n18 1089 1086\n/END\n").expect("deck");
/// let nodes = BTreeSet::from([1086, 1089]);
/// let parsed = decode_transform_card(&deck.cards[0], &nodes).expect("decode");
/// assert_eq!(parsed.id, 1);
/// assert_eq!(parsed.group, 18);
/// assert!(matches!(parsed.params, TransformParams::Sym { axis: AxisSpec::Nodes(1089, 1086) }));
/// ```
///
/// # Errors
///
/// Returns [`SubmodelError`] when the card is not a transform, lacks its
/// header id, carries malformed rows, names one axis node, or references
/// nodes outside `nodes`.
pub fn decode_transform_card(
    card: &DeckCard,
    nodes: &BTreeSet<i64>,
) -> Result<Transform, SubmodelError> {
    let Some(id) = card.id else {
        return Err(SubmodelError::from_kind(ErrorKind::MissingTransformId));
    };
    match card.kind {
        CardKind::TransformSym => decode_sym(id, card, nodes),
        CardKind::TransformSca => decode_sca(id, card, nodes),
        CardKind::TransformRot => decode_rot(id, card, nodes),
        CardKind::TransformTra => decode_tra(id, card, nodes),
        _ => Err(SubmodelError::from_kind(ErrorKind::BadRowCount {
            kind: "transform",
            rows: card.fields.len(),
        })),
    }
}

/// Collects deck transforms in deck order.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_parse::RadDeck;
/// use sonium_qa::rad_submodel::collect_transforms;
///
/// let deck = RadDeck::parse_str("/NODE\n1 0.0 0.0 0.0\n/TRANSFORM/TRA/4\ntra\n7 1.0\n/END\n")
///     .expect("deck");
/// let parsed = collect_transforms(&deck).expect("collect");
/// assert_eq!(parsed.len(), 1);
/// assert_eq!(parsed[0].group, 7);
/// ```
///
/// # Errors
///
/// Returns [`SubmodelError`] when any transform card is malformed (see
/// [`decode_transform_card`]).
pub fn collect_transforms(deck: &RadDeck) -> Result<Vec<Transform>, SubmodelError> {
    let mut ids = BTreeSet::new();
    for card in &deck.cards {
        if let Some(TypedCard::Nodes(nodes)) = &card.typed {
            for node in nodes {
                ids.insert(node.id);
            }
        }
    }
    let mut out = Vec::new();
    for card in &deck.cards {
        match card.kind {
            CardKind::TransformSym
            | CardKind::TransformSca
            | CardKind::TransformRot
            | CardKind::TransformTra => out.push(decode_transform_card(card, &ids)?),
            _ => {}
        }
    }
    Ok(out)
}

/// Builds the deck node table.
///
/// Merges every `/NODE` card into an id-to-coordinate map.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_parse::RadDeck;
/// use sonium_qa::rad_submodel::node_table;
///
/// let deck = RadDeck::parse_str("/NODE\n1 1.0 2.0 3.0\n/END\n").expect("deck");
/// let table = node_table(&deck).expect("table");
/// assert_eq!(table[&1], [1.0, 2.0, 3.0]);
/// ```
///
/// # Errors
///
/// Returns [`SubmodelError`] when a node id repeats across cards.
pub fn node_table(deck: &RadDeck) -> Result<BTreeMap<i64, [f64; 3]>, SubmodelError> {
    let mut table = BTreeMap::new();
    for card in &deck.cards {
        if let Some(TypedCard::Nodes(nodes)) = &card.typed {
            for node in nodes {
                if table.insert(node.id, [node.x, node.y, node.z]).is_some() {
                    return Err(SubmodelError::from_kind(ErrorKind::DuplicateNode {
                        id: node.id,
                    }));
                }
            }
        }
    }
    Ok(table)
}

/// Resolves a node group to member ids.
///
/// Supports `/GRNOD/NODE` flat lists and recursive `/GRNOD/GRNOD`
/// nesting with cycle detection; every other group flavor fails
/// closed. Duplicate group definitions are rejected.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_parse::RadDeck;
/// use sonium_qa::rad_submodel::resolve_group_nodes;
///
/// let deck = RadDeck::parse_str("/GRNOD/NODE/7\nseven\n1 2 3\n/END\n").expect("deck");
/// assert_eq!(resolve_group_nodes(&deck, 7).expect("group"), vec![1, 2, 3]);
/// ```
///
/// # Errors
///
/// Returns [`SubmodelError`] for unknown groups, unsupported flavors,
/// non-numeric members, duplicate definitions, or nesting cycles.
pub fn resolve_group_nodes(deck: &RadDeck, group: i64) -> Result<Vec<i64>, SubmodelError> {
    let mut visiting = BTreeSet::new();
    resolve_group_inner(deck, group, &mut visiting)
}

/// Recurses through nested node groups.
fn resolve_group_inner(
    deck: &RadDeck,
    group: i64,
    visiting: &mut BTreeSet<i64>,
) -> Result<Vec<i64>, SubmodelError> {
    if !visiting.insert(group) {
        return Err(SubmodelError::from_kind(ErrorKind::GroupCycle {
            id: group,
        }));
    }
    let mut found: Option<&DeckCard> = None;
    for card in &deck.cards {
        let is_group = matches!(
            card.kind,
            CardKind::GrnodNode
                | CardKind::GrnodGrnod
                | CardKind::GrnodPart
                | CardKind::GrnodSurf
                | CardKind::GrnodSubset
                | CardKind::GrnodGrshel
        );
        if is_group && card.id == Some(group) {
            if found.is_some() {
                return Err(SubmodelError::from_kind(ErrorKind::DuplicateGroup {
                    id: group,
                }));
            }
            found = Some(card);
        }
    }
    let result = match found {
        None => Err(SubmodelError::from_kind(ErrorKind::UnknownGroup {
            id: group,
        })),
        Some(card) if card.kind == CardKind::GrnodNode => {
            let mut members = Vec::new();
            for row in &card.fields {
                for token in row {
                    let Some(id) = parse_int(token) else {
                        return Err(SubmodelError::from_kind(ErrorKind::BadNumber {
                            kind: "GRNOD",
                            token: token.clone(),
                        }));
                    };
                    members.push(id);
                }
            }
            Ok(members)
        }
        Some(card) if card.kind == CardKind::GrnodGrnod => {
            let mut members = Vec::new();
            for row in &card.fields {
                for token in row {
                    let Some(inner) = parse_int(token) else {
                        return Err(SubmodelError::from_kind(ErrorKind::BadNumber {
                            kind: "GRNOD",
                            token: token.clone(),
                        }));
                    };
                    members.extend(resolve_group_inner(deck, inner, visiting)?);
                }
            }
            Ok(members)
        }
        Some(card) => Err(SubmodelError::from_kind(ErrorKind::UnsupportedGroup {
            kind: match card.kind {
                CardKind::GrnodPart => "GRNOD/PART",
                CardKind::GrnodSurf => "GRNOD/SURF",
                CardKind::GrnodSubset => "GRNOD/SUBSET",
                _ => "GRNOD/GRSHEL",
            },
            id: group,
        })),
    };
    visiting.remove(&group);
    result
}

/// Node coordinates after one transform stage.
#[derive(Debug, Clone, PartialEq)]
pub struct StageResult {
    /// Transform identifier.
    pub id: i64,
    /// Target group identifier.
    pub group: i64,
    /// Member coordinates after this stage.
    pub nodes: BTreeMap<i64, [f64; 3]>,
}

/// Full transform chain applied to a deck.
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedTransforms {
    /// Per-transform member tables in deck order.
    pub stages: Vec<StageResult>,
    /// Whole node table after the chain.
    pub final_nodes: BTreeMap<i64, [f64; 3]>,
}

/// Applies every deck transform in deck order.
///
/// Reference nodes and centers resolve against current coordinates at
/// each stage, matching the vendor in-place update; each stage feeds
/// the next. Zero-angle rotations record an unchanged stage.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_parse::RadDeck;
/// use sonium_qa::rad_submodel::apply_transforms;
///
/// let deck = RadDeck::parse_str(
///     "/NODE\n1 0.0 0.0 0.0\n/GRNOD/NODE/7\nseven\n1\n/TRANSFORM/TRA/4\ntra\n7 -5.0\n/END\n",
/// )
/// .expect("deck");
/// let applied = apply_transforms(&deck).expect("apply");
/// assert_eq!(applied.stages.len(), 1);
/// assert_eq!(applied.final_nodes[&1], [-5.0, 0.0, 0.0]);
/// ```
///
/// # Errors
///
/// Returns [`SubmodelError`] for malformed transforms, unresolvable
/// groups or reference nodes, and degenerate axes with nonzero angles.
pub fn apply_transforms(deck: &RadDeck) -> Result<AppliedTransforms, SubmodelError> {
    let mut table = node_table(deck)?;
    let transforms = collect_transforms(deck)?;
    if !transforms.is_empty() && table.is_empty() {
        return Err(SubmodelError::from_kind(ErrorKind::EmptyNodeTable));
    }
    let mut stages = Vec::with_capacity(transforms.len());
    for transform in &transforms {
        let members = resolve_group_nodes(deck, transform.group)?;
        for member in &members {
            if !table.contains_key(member) {
                return Err(SubmodelError::from_kind(ErrorKind::UnknownNode {
                    id: *member,
                }));
            }
        }
        apply_one(&mut table, transform, &members)?;
        let mut stage = BTreeMap::new();
        for member in &members {
            if let Some(position) = table.get(member) {
                stage.insert(*member, *position);
            }
        }
        stages.push(StageResult {
            id: transform.id,
            group: transform.group,
            nodes: stage,
        });
    }
    Ok(AppliedTransforms {
        stages,
        final_nodes: table,
    })
}

/// Looks up a reference node position.
fn lookup(table: &BTreeMap<i64, [f64; 3]>, id: i64) -> Result<[f64; 3], SubmodelError> {
    table
        .get(&id)
        .copied()
        .ok_or_else(|| SubmodelError::from_kind(ErrorKind::UnknownNode { id }))
}

/// Applies one transform to group members in place.
fn apply_one(
    table: &mut BTreeMap<i64, [f64; 3]>,
    transform: &Transform,
    members: &[i64],
) -> Result<(), SubmodelError> {
    match &transform.params {
        TransformParams::Sym { axis } => {
            let (center, direction) = axis_geometry(table, axis)?;
            let mut mapped = BTreeMap::new();
            for member in members {
                let position = lookup(table, *member)?;
                let Some(next) = reflect_point(position, center, direction) else {
                    return Err(SubmodelError::from_kind(ErrorKind::DegenerateAxis {
                        kind: "SYM",
                    }));
                };
                mapped.insert(*member, next);
            }
            table.extend(mapped);
            Ok(())
        }
        TransformParams::Sca { factors, center } => {
            let origin = match center {
                Some(id) => lookup(table, *id)?,
                None => [0.0; 3],
            };
            for member in members {
                let position = lookup(table, *member)?;
                table.insert(*member, scale_point(position, origin, *factors));
            }
            Ok(())
        }
        TransformParams::Rot { axis, angle_deg } => {
            if *angle_deg == 0.0 {
                return Ok(());
            }
            let (center, direction) = axis_geometry(table, axis)?;
            let mut mapped = BTreeMap::new();
            for member in members {
                let position = lookup(table, *member)?;
                let Some(next) = rotate_point(position, center, direction, *angle_deg) else {
                    return Err(SubmodelError::from_kind(ErrorKind::DegenerateAxis {
                        kind: "ROT",
                    }));
                };
                mapped.insert(*member, next);
            }
            table.extend(mapped);
            Ok(())
        }
        TransformParams::Tra { shift } => {
            let vector = match shift {
                ShiftSpec::Values(values) => *values,
                ShiftSpec::Nodes(first, second) => {
                    let origin = lookup(table, *first)?;
                    let tip = lookup(table, *second)?;
                    [tip[0] - origin[0], tip[1] - origin[1], tip[2] - origin[2]]
                }
            };
            for member in members {
                let position = lookup(table, *member)?;
                table.insert(*member, translate_point(position, vector));
            }
            Ok(())
        }
    }
}

/// Resolves axis center and direction from current coordinates.
fn axis_geometry(
    table: &BTreeMap<i64, [f64; 3]>,
    axis: &AxisSpec,
) -> Result<([f64; 3], [f64; 3]), SubmodelError> {
    match axis {
        AxisSpec::Nodes(first, second) => {
            let origin = lookup(table, *first)?;
            let tip = lookup(table, *second)?;
            Ok((
                origin,
                [tip[0] - origin[0], tip[1] - origin[1], tip[2] - origin[2]],
            ))
        }
        AxisSpec::Points(first, second) => Ok((
            *first,
            [
                second[0] - first[0],
                second[1] - first[1],
                second[2] - first[2],
            ],
        )),
    }
}

/// Submodel id offsets in deck order.
///
/// Order matches the single offset card (`%10d` seven times): default,
/// nodes, elements, parts, materials, properties, submodels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubmodelOffsets {
    /// Default offset for every option.
    pub def: i64,
    /// Node offset, defaulting to [`SubmodelOffsets::def`].
    pub nod: i64,
    /// Element offset, defaulting to [`SubmodelOffsets::def`].
    pub ele: i64,
    /// Part offset, defaulting to [`SubmodelOffsets::def`].
    pub part: i64,
    /// Material offset, defaulting to [`SubmodelOffsets::def`].
    pub mat: i64,
    /// Property offset, defaulting to [`SubmodelOffsets::def`].
    pub prop: i64,
    /// Submodel offset, defaulting to [`SubmodelOffsets::def`].
    pub sub: i64,
}

impl SubmodelOffsets {
    /// Resolves zero entries against the default.
    ///
    /// The vendor `WHILE_ZERO` drawable substitutes the default for
    /// every zero specific offset (`submodel.cfg`).
    #[must_use]
    pub const fn effective(&self) -> [i64; 7] {
        [
            self.def,
            if self.nod == 0 { self.def } else { self.nod },
            if self.ele == 0 { self.def } else { self.ele },
            if self.part == 0 { self.def } else { self.part },
            if self.mat == 0 { self.def } else { self.mat },
            if self.prop == 0 { self.def } else { self.prop },
            if self.sub == 0 { self.def } else { self.sub },
        ]
    }

    /// Reports whether every effective offset is zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        let applied = self.effective();
        applied[0] == 0
            && applied[1] == 0
            && applied[2] == 0
            && applied[3] == 0
            && applied[4] == 0
            && applied[5] == 0
            && applied[6] == 0
    }
}

/// Decodes the offset card of a `//SUBMODEL` marker.
///
/// Missing rows and trailing entries default to zero like the vendor
/// fixed-format read; entries fill left-dense in deck order.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_parse::RadDeck;
/// use sonium_qa::rad_submodel::decode_offsets;
///
/// let deck = RadDeck::parse_str("//SUBMODEL/1\nmodel\n100 0 5\n//ENDSUB\n/END\n").expect("deck");
/// let offsets = decode_offsets(&deck.cards[0]).expect("offsets");
/// assert_eq!(offsets.effective(), [100, 100, 5, 100, 100, 100, 100]);
/// ```
///
/// # Errors
///
/// Returns [`SubmodelError`] for non-submodel cards, extra rows, more
/// than seven entries, or non-integer entries.
pub fn decode_offsets(card: &DeckCard) -> Result<SubmodelOffsets, SubmodelError> {
    if card.kind != CardKind::Submodel {
        return Err(SubmodelError::from_kind(ErrorKind::NotSubmodelCard));
    }
    if card.fields.len() > 1 {
        return Err(SubmodelError::from_kind(ErrorKind::BadOffset {
            detail: format!("{} data rows", card.fields.len()),
        }));
    }
    let mut values = [0_i64; 7];
    if let Some(row) = card.fields.first() {
        if row.len() > 7 {
            return Err(SubmodelError::from_kind(ErrorKind::BadOffset {
                detail: format!("{} entries", row.len()),
            }));
        }
        for (slot, token) in values.iter_mut().zip(row.iter()) {
            let Some(value) = parse_int(token) else {
                return Err(SubmodelError::from_kind(ErrorKind::BadOffset {
                    detail: format!("token {token:?}"),
                }));
            };
            *slot = value;
        }
    }
    Ok(SubmodelOffsets {
        def: values[0],
        nod: values[1],
        ele: values[2],
        part: values[3],
        mat: values[4],
        prop: values[5],
        sub: values[6],
    })
}

/// One `//SUBMODEL` section with its card range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmodelSection {
    /// Submodel identifier from the marker header.
    pub id: i64,
    /// Nesting depth with the outermost section at one.
    pub level: usize,
    /// Enclosing section identifier, if nested.
    pub father: Option<i64>,
    /// First content card index (marker excluded).
    pub start: usize,
    /// One past the last content card index.
    pub end: usize,
    /// Decoded id offsets.
    pub offsets: SubmodelOffsets,
}

/// Splits a deck into its submodel sections.
///
/// Supports vendor hierarchy (sections inside sections) with levels
/// and fathers like `hm_read_submodel.F`; sections report in closing
/// order (innermost first).
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_parse::RadDeck;
/// use sonium_qa::rad_submodel::split_submodels;
///
/// let deck = RadDeck::parse_str("//SUBMODEL/1\nmodel\n/NODE\n1 0.0 0.0 0.0\n//ENDSUB\n/END\n")
///     .expect("deck");
/// let sections = split_submodels(&deck).expect("split");
/// assert_eq!(sections.len(), 1);
/// assert_eq!(sections[0].id, 1);
/// assert!(sections[0].offsets.is_zero());
/// ```
///
/// # Errors
///
/// Returns [`SubmodelError`] for missing marker ids, malformed offset
/// rows, `//ENDSUB` without an open section, or unclosed sections.
pub fn split_submodels(deck: &RadDeck) -> Result<Vec<SubmodelSection>, SubmodelError> {
    let mut sections = Vec::new();
    let mut stack: Vec<(i64, usize, SubmodelOffsets)> = Vec::new();
    for (index, card) in deck.cards.iter().enumerate() {
        match card.kind {
            CardKind::Submodel => {
                let Some(id) = card.id else {
                    return Err(SubmodelError::from_kind(ErrorKind::MissingSubmodelId));
                };
                let offsets = decode_offsets(card)?;
                stack.push((id, index + 1, offsets));
            }
            CardKind::Endsub => {
                let Some((id, start, offsets)) = stack.pop() else {
                    return Err(SubmodelError::from_kind(ErrorKind::UnbalancedEndsub {
                        index,
                    }));
                };
                sections.push(SubmodelSection {
                    id,
                    level: stack.len() + 1,
                    father: stack.last().map(|entry| entry.0),
                    start,
                    end: index,
                    offsets,
                });
            }
            _ => {}
        }
    }
    if let Some((id, _, _)) = stack.pop() {
        return Err(SubmodelError::from_kind(ErrorKind::UnterminatedSubmodel {
            id,
        }));
    }
    Ok(sections)
}

/// Merges zero-offset submodels into one card stream.
///
/// Drops the `//SUBMODEL` / `//ENDSUB` markers and splices section
/// content at the marker position, preserving card order. Sections
/// with nonzero effective offsets are rejected: shifting ids needs
/// per-family reference rewriting with no deck coverage.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_parse::RadDeck;
/// use sonium_qa::rad_submodel::merge_submodels;
///
/// let deck = RadDeck::parse_str("//SUBMODEL/1\nmodel\n/NODE\n1 0.0 0.0 0.0\n//ENDSUB\n/END\n")
///     .expect("deck");
/// let merged = merge_submodels(&deck).expect("merge");
/// assert_eq!(merged.cards.len(), 2);
/// ```
///
/// # Errors
///
/// Returns [`SubmodelError`] for unbalanced markers (see
/// [`split_submodels`]) or nonzero effective offsets.
pub fn merge_submodels(deck: &RadDeck) -> Result<RadDeck, SubmodelError> {
    for section in split_submodels(deck)? {
        if !section.offsets.is_zero() {
            let applied = section.offsets.effective();
            let rendered = applied
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(SubmodelError::from_kind(ErrorKind::NonzeroOffsets {
                id: section.id,
                offsets: rendered,
            }));
        }
    }
    let cards = deck
        .cards
        .iter()
        .filter(|card| !matches!(card.kind, CardKind::Submodel | CardKind::Endsub))
        .cloned()
        .collect();
    Ok(RadDeck { cards })
}

/// Counts card families inside a section.
///
/// Keys are canonical header families (`/NODE`, `/SHELL`, ...);
/// nested markers count under `//SUBMODEL` / `//ENDSUB`.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad_parse::RadDeck;
/// use sonium_qa::rad_submodel::{section_inventory, split_submodels};
///
/// let deck = RadDeck::parse_str("//SUBMODEL/1\nmodel\n/NODE\n1 0.0 0.0 0.0\n//ENDSUB\n/END\n")
///     .expect("deck");
/// let sections = split_submodels(&deck).expect("split");
/// let counts = section_inventory(&deck, &sections[0]);
/// assert_eq!(counts["/NODE"], 1);
/// ```
///
/// # Panics
///
/// Panics when `section` addresses cards outside `deck`, which means
/// the section came from another deck (caller contract violation).
#[must_use]
pub fn section_inventory(deck: &RadDeck, section: &SubmodelSection) -> BTreeMap<String, usize> {
    assert!(
        section.start <= section.end && section.end <= deck.cards.len(),
        "foreign submodel section"
    );
    let mut counts = BTreeMap::new();
    for card in &deck.cards[section.start..section.end] {
        *counts.entry(card.kind.to_string()).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rad_parse::RadDeck;

    /// Vendor agreement band for transcribed starter echoes.
    ///
    /// The fresh double-precision starter prints `E20.13` and matched
    /// this module within 5e-13 on all 40 TRANSF nodes at author time;
    /// the 1e-9 band absorbs print rounding plus `libm` drift across
    /// platforms while staying far below any modeling tolerance.
    const VENDOR_BAND: f64 = 1e-9;

    /// TRANSF group and reference nodes, deck-verbatim.
    ///
    /// Transcribed from `BOULE1V5_0000.rad` `/NODE` (group 1000-1039,
    /// references 1086, 1089, 1095, 1114, 1115); the gated real-deck
    /// test replays these exact rows through the reader.
    const TRANSF_INPUT: &str = "1000 -10.015957411335 -1.0753791459204 1.2391039848
1001 -9.9618378126856 -1.0305449929418 1.2391039848
1002 -9.9077183244864 -1.0753790788868 1.2391039848
1003 -9.8853012873122 -1.1836180436887 1.2391039848
1004 -9.9077183697413 -1.2918572291079 1.2391039848
1005 -9.9618379440661 -1.3366913571964 1.2391039848
1006 -10.01595743071 -1.2918572913331 1.2391039848
1007 -10.03837446944 -1.1836183064494 1.2391039848
1008 -10.079425558147 -.94844285229668 .9444560048
1009 -9.9618379104077 -.85103016211667 .9444560048
1010 -9.8442502636587 -.94844295723159 .9444560048
1011 -9.795543845171 -1.1836182925904 .9444560048
1012 -9.8442501889883 -1.4187935906139 .9444560048
1013 -9.9618378493138 -1.5162063008757 .9444560048
1014 -10.07942552661 -1.4187934831334 .9444560048
1015 -10.128131914551 -1.183618170402 .9444560048
1016 -10.103259229593 -.90077550940476 .49999998143592
1017 -9.9618379395404 -.78361792047473 .49999998143592
1018 -9.8204165200878 -.90077547008994 .49999998143592
1019 -9.7618377237843 -1.1836183519874 .49999998143592
1020 -9.8204165171181 -1.4664609343543 .49999998143592
1021 -9.9618378153727 -1.5836185351637 .49999998143592
1022 -10.103259266362 -1.4664609626383 .49999998143592
1023 -10.161838031129 -1.1836181036511 .49999998143592
1024 -10.079425558147 -.94844285229668 .0555439952
1025 -9.9618379104077 -.85103016211667 .0555439952
1026 -9.8442502636587 -.94844295723159 .0555439952
1027 -9.795543845171 -1.1836182925904 .0555439952
1028 -9.8442501889883 -1.4187935906139 .0555439952
1029 -9.9618378493138 -1.5162063008757 .0555439952
1030 -10.07942552661 -1.4187934831334 .0555439952
1031 -10.128131914551 -1.183618170402 .0555439952
1032 -10.015957411335 -1.0753791459204 -.2391039848
1033 -9.9618378126856 -1.0305449929418 -.2391039848
1034 -9.9077183244864 -1.0753790788868 -.2391039848
1035 -9.8853012873122 -1.1836180436887 -.2391039848
1036 -9.9077183697413 -1.2918572291079 -.2391039848
1037 -9.9618379440661 -1.3366913571964 -.2391039848
1038 -10.01595743071 -1.2918572913331 -.2391039848
1039 -10.03837446944 -1.1836183064494 -.2391039848
1086 -.899999976 .500000119 -.5
1089 -3.9000001 .500000358 -.5
1095 .50000002 -.1 .5
1114 2.10000002 -2.5 2.5
1115 .50000002 -.1 2.5
";

    /// TRANSF final coordinates after the full chain.
    ///
    /// Transcribed from the fresh starter `NEW NODE COORDINATES` table
    /// of `/TRANSFORM/TRA/4` (dev-time run, 2026-10-11).
    const TRANSF_FINAL: &str = "1000 -0.3338450280991E+01 -0.2279784701622E+01 0.9238799810000E+00
1001 -0.3226364876986E+01 -0.2009186744093E+01 0.9238799810000E+00
1002 -0.3338450070291E+01 -0.1738589267379E+01 0.9238799810000E+00
1003 -0.3609047473367E+01 -0.1626503995278E+01 0.9238799810000E+00
1004 -0.3879645445844E+01 -0.1738589321193E+01 0.9238799810000E+00
1005 -0.3991730787623E+01 -0.2009187157099E+01 0.9238799810000E+00
1006 -0.3879645644522E+01 -0.2279784626036E+01 0.9238799810000E+00
1007 -0.3609048191242E+01 -0.2391869905917E+01 0.9238799810000E+00
1008 -0.3021109572213E+01 -0.2597125536808E+01 0.5555700060000E+00
1009 -0.2777577799924E+01 -0.2009187375717E+01 0.5555700060000E+00
1010 -0.3021109740872E+01 -0.1421249064367E+01 0.5555700060000E+00
1011 -0.3609048059867E+01 -0.1177716784572E+01 0.5555700060000E+00
1012 -0.4196986324328E+01 -0.1421248316302E+01 0.5555700060000E+00
1013 -0.4440518146821E+01 -0.2009186540324E+01 0.5555700060000E+00
1014 -0.4196986149304E+01 -0.2597125004411E+01 0.5555700060000E+00
1015 -0.3609047886877E+01 -0.2840657131472E+01 0.5555700060000E+00
1016 -0.2901941224477E+01 -0.2716293932013E+01 -0.2320510006371E-07
1017 -0.2609047195819E+01 -0.2009187575086E+01 -0.2320510006371E-07
1018 -0.2901941013524E+01 -0.1302080384487E+01 -0.2320510006371E-07
1019 -0.3609048194934E+01 -0.1009186177638E+01 -0.2320510006371E-07
1020 -0.4316154674185E+01 -0.1302079918976E+01 -0.2320510006371E-07
1021 -0.4609048732541E+01 -0.2009186316913E+01 -0.2320510006371E-07
1022 -0.4316154857560E+01 -0.2716293665195E+01 -0.2320510006371E-07
1023 -0.3609047733426E+01 -0.3009187714362E+01 -0.2320510006371E-07
1024 -0.3021109572213E+01 -0.2597125536808E+01 -0.5555700060000E+00
1025 -0.2777577799924E+01 -0.2009187375717E+01 -0.5555700060000E+00
1026 -0.3021109740872E+01 -0.1421249064367E+01 -0.5555700060000E+00
1027 -0.3609048059867E+01 -0.1177716784572E+01 -0.5555700060000E+00
1028 -0.4196986324328E+01 -0.1421248316302E+01 -0.5555700060000E+00
1029 -0.4440518146821E+01 -0.2009186540324E+01 -0.5555700060000E+00
1030 -0.4196986149304E+01 -0.2597125004411E+01 -0.5555700060000E+00
1031 -0.3609047886877E+01 -0.2840657131472E+01 -0.5555700060000E+00
1032 -0.3338450280991E+01 -0.2279784701622E+01 -0.9238799810000E+00
1033 -0.3226364876986E+01 -0.2009186744093E+01 -0.9238799810000E+00
1034 -0.3338450070291E+01 -0.1738589267379E+01 -0.9238799810000E+00
1035 -0.3609047473367E+01 -0.1626503995278E+01 -0.9238799810000E+00
1036 -0.3879645445844E+01 -0.1738589321193E+01 -0.9238799810000E+00
1037 -0.3991730787623E+01 -0.2009187157099E+01 -0.9238799810000E+00
1038 -0.3879645644522E+01 -0.2279784626036E+01 -0.9238799810000E+00
1039 -0.3609048191242E+01 -0.2391869905917E+01 -0.9238799810000E+00
";

    /// Per-stage probe coordinates (SYM, SCA, ROT tables).
    ///
    /// Fresh-starter echoes for nodes 1000, 1016, 1039 after each of
    /// the first three stages; the TRA stage equals [`TRANSF_FINAL`].
    /// Each row reads `STAGE id x y z`.
    const TRANSF_PROBES: &str =
        "SYM 1000 0.2215956960324E+01 -0.1075380120396E+01 0.1239103984800E+01
SYM 1016 0.2303258806403E+01 -0.9007764977907E+00 0.4999999814359E+00
SYM 1039 0.2238374001183E+01 -0.1183619284497E+01 -0.2391039848000E+00
SCA 1000 0.2679784721622E+01 0.1061549699009E+01 0.9238799810000E+00
SCA 1016 0.3116293952013E+01 0.1498058755523E+01 -0.2320510006371E-07
SCA 1039 0.2791869925917E+01 0.7909517887575E+00 -0.9238799810000E+00
ROT 1000 0.1661549719009E+01 -0.2279784701622E+01 0.9238799810000E+00
ROT 1016 0.2098058775523E+01 -0.2716293932013E+01 -0.2320510006371E-07
ROT 1039 0.1390951808758E+01 -0.2391869905917E+01 -0.9238799810000E+00
";

    /// Parses `id x y z` rows into a coordinate map.
    fn parse_table(text: &str) -> BTreeMap<i64, [f64; 3]> {
        text.lines()
            .map(|line| {
                let mut parts = line.split_whitespace();
                let id: i64 = parts.next().expect("id").parse().expect("int");
                let x: f64 = parts.next().expect("x").parse().expect("real");
                let y: f64 = parts.next().expect("y").parse().expect("real");
                let z: f64 = parts.next().expect("z").parse().expect("real");
                (id, [x, y, z])
            })
            .collect()
    }

    /// Builds the minimal TRANSF deck from transcribed rows.
    ///
    /// Node rows are verbatim; group 18 spans 1000-1039 like the deck
    /// (the gated real-deck test asserts the deck list matches); the
    /// transform rows are token-identical to the deck rows.
    fn transf_deck_text() -> String {
        let mut deck = String::from("/NODE\n");
        deck.push_str(TRANSF_INPUT);
        deck.push_str("/GRNOD/NODE/18\ngroup\n");
        for row in 0..4 {
            for col in 0..10 {
                if col > 0 {
                    deck.push(' ');
                }
                deck.push_str(&(1000 + row * 10 + col).to_string());
            }
            deck.push('\n');
        }
        deck.push_str("/TRANSFORM/SYM/1\nsym_back\n18 1089 1086\n");
        deck.push_str("/TRANSFORM/SCA/2\nscale_back\n18 5 2.5 1.25 1114\n");
        deck.push_str("/TRANSFORM/ROT/3\nrot_back\n18 1095 1115\n-90\n");
        deck.push_str("/TRANSFORM/TRA/4\ntra_back\n18 -5\n/END\n");
        deck
    }

    /// Reports the worst absolute component deviation.
    fn worst_deviation(first: &BTreeMap<i64, [f64; 3]>, second: &BTreeMap<i64, [f64; 3]>) -> f64 {
        let mut worst = 0.0_f64;
        for (id, left) in first {
            let right = second.get(id).expect("shared node");
            for axis in 0..3 {
                worst = worst.max((left[axis] - right[axis]).abs());
            }
        }
        worst
    }

    #[test]
    fn reflection_is_involution() {
        let center = [1.0, -2.0, 0.5];
        let direction = [0.0, 1.0, 1.0];
        let point = [3.0, 4.0, -1.0];
        let once = reflect_point(point, center, direction).expect("axis");
        let twice = reflect_point(once, center, direction).expect("axis");
        for axis in 0..3 {
            assert!((twice[axis] - point[axis]).abs() < 1e-12, "axis {axis}");
        }
        let planar = reflect_point(center, center, direction).expect("axis");
        assert_eq!(planar, center);
    }

    #[test]
    fn reflection_matches_plane_formula() {
        let mirrored =
            reflect_point([2.0, 3.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]).expect("axis");
        assert_eq!(mirrored, [2.0, 3.0, -4.0]);
        assert_eq!(
            reflect_point([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1e-21, 0.0, 0.0]),
            None
        );
    }

    #[test]
    fn scale_maps_zero_factor_to_one() {
        assert_eq!(
            scale_point([4.0, 4.0, 4.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
            [4.0, 4.0, 4.0]
        );
        assert_eq!(
            scale_point([3.0, 4.0, 5.0], [1.0, 0.0, -1.0], [5.0, 2.5, 1.25]),
            [11.0, 10.0, 6.5]
        );
    }

    #[test]
    fn rotation_matches_right_hand_rule() {
        let quarter =
            rotate_point([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 90.0).expect("axis");
        assert!(quarter[0].abs() < 1e-12);
        assert!((quarter[1] - 1.0).abs() < 1e-12);
        assert!(quarter[2].abs() < 1e-12);
        let back =
            rotate_point([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], -90.0).expect("axis");
        assert!(back[0].abs() < 1e-12);
        assert!((back[1] - 1.0).abs() < 1e-12);
        let full =
            rotate_point([1.0, 2.0, 3.0], [0.5, -0.5, 0.0], [1.0, 1.0, 1.0], 360.0).expect("axis");
        for axis in 0..3 {
            assert!((full[axis] - [1.0, 2.0, 3.0][axis]).abs() < 1e-12);
        }
        assert_eq!(
            rotate_point([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 45.0),
            None
        );
    }

    #[test]
    fn deck_sym_row_decodes_to_nodes() {
        let deck =
            RadDeck::parse_str("/TRANSFORM/SYM/1\nsym_back\n18 1089 1086\n/END\n").expect("deck");
        let nodes = BTreeSet::from([1086, 1089]);
        let parsed = decode_transform_card(&deck.cards[0], &nodes).expect("decode");
        assert_eq!(parsed.kind(), TransformKind::Sym);
        assert_eq!(
            parsed.params,
            TransformParams::Sym {
                axis: AxisSpec::Nodes(1089, 1086)
            }
        );
    }

    #[test]
    fn deck_sca_rot_tra_rows_decode() {
        let deck = RadDeck::parse_str(
            "/NODE\n1114 0.0 0.0 0.0\n1095 0.0 0.0 0.0\n1115 0.0 0.0 1.0\n\
             /TRANSFORM/SCA/2\nscale_back\n18 5 2.5 1.25 1114\n\
             /TRANSFORM/ROT/3\nrot_back\n18 1095 1115\n-90\n\
             /TRANSFORM/TRA/4\ntra_back\n18 -5\n/END\n",
        )
        .expect("deck");
        let parsed = collect_transforms(&deck).expect("collect");
        assert_eq!(parsed.len(), 3);
        assert_eq!(
            parsed[0].params,
            TransformParams::Sca {
                factors: [5.0, 2.5, 1.25],
                center: Some(1114)
            }
        );
        assert_eq!(
            parsed[1].params,
            TransformParams::Rot {
                axis: AxisSpec::Nodes(1095, 1115),
                angle_deg: -90.0
            }
        );
        assert_eq!(
            parsed[2].params,
            TransformParams::Tra {
                shift: ShiftSpec::Values([-5.0, 0.0, 0.0])
            }
        );
    }

    #[test]
    fn point_form_and_fortran_exponents_decode() {
        let deck = RadDeck::parse_str(
            "/TRANSFORM/SYM/1\nsym\n7 1.0D+00 0.0 0.0\n2.0 0.0 0.0\n\
             /TRANSFORM/ROT/2\nrot\n7 0.0 0.0 0.0\n0.0 0.0 1.0 45.0\n/END\n",
        )
        .expect("deck");
        let nodes = BTreeSet::new();
        let sym = decode_transform_card(&deck.cards[0], &nodes).expect("sym");
        assert_eq!(
            sym.params,
            TransformParams::Sym {
                axis: AxisSpec::Points([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])
            }
        );
        let rot = decode_transform_card(&deck.cards[1], &nodes).expect("rot");
        assert_eq!(
            rot.params,
            TransformParams::Rot {
                axis: AxisSpec::Points([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                angle_deg: 45.0
            }
        );
    }

    #[test]
    fn malformed_transforms_fail_closed() {
        let nodes: BTreeSet<i64> = BTreeSet::from([11, 12]);
        for (text, check) in [
            ("/TRANSFORM/ROT/3\nrot\n18 11\n/END\n", "single node"),
            ("/TRANSFORM/SYM/1\nsym\n18 11 99\n/END\n", "unknown node"),
            ("/TRANSFORM/SCA/2\nsca\n18 5 2.5\n/END\n", "short factors"),
            ("/TRANSFORM/TRA/4\ntra\n18 -5\n3\n/END\n", "skew frame"),
            (
                "/TRANSFORM/TRA/4\ntra\n18 1.0 2.0 3.0 4.0\n/END\n",
                "long row",
            ),
        ] {
            let deck = RadDeck::parse_str(text).expect("deck parses");
            let error = decode_transform_card(&deck.cards[0], &nodes).expect_err(check);
            assert!(
                error.is_decode() || error.is_lookup() || error.is_unsupported(),
                "{check}: {error}"
            );
        }
        let skew = RadDeck::parse_str("/TRANSFORM/TRA/4\ntra\n18 -5\n3\n/END\n").expect("deck");
        let error = decode_transform_card(&skew.cards[0], &nodes).expect_err("skew must fail");
        assert!(error.is_unsupported());
        assert!(error.to_string().contains("skew frame"));
        let zero_skew =
            RadDeck::parse_str("/TRANSFORM/TRA/4\ntra\n18 -5\n0\n/END\n").expect("deck");
        assert!(decode_transform_card(&zero_skew.cards[0], &nodes).is_ok());
    }

    #[test]
    fn group_resolution_covers_nesting_and_cycles() {
        let deck = RadDeck::parse_str(
            "/GRNOD/NODE/1\none\n1 2\n/GRNOD/NODE/2\ntwo\n3\n/GRNOD/GRNOD/3\nthree\n1 2\n/END\n",
        )
        .expect("deck");
        assert_eq!(
            resolve_group_nodes(&deck, 3).expect("nested"),
            vec![1, 2, 3]
        );
        assert!(
            resolve_group_nodes(&deck, 9)
                .expect_err("unknown")
                .is_lookup()
        );
        let cyclic = RadDeck::parse_str("/GRNOD/GRNOD/4\nfour\n4\n/END\n").expect("deck");
        let error = resolve_group_nodes(&cyclic, 4).expect_err("cycle");
        assert!(error.is_lookup());
        assert!(error.to_string().contains("cyclically"));
        let part = RadDeck::parse_str("/GRNOD/PART/5\nfive\n1\n/END\n").expect("deck");
        let error = resolve_group_nodes(&part, 5).expect_err("part flavor");
        assert!(error.is_lookup());
        assert!(error.to_string().contains("GRNOD/PART"));
    }

    #[test]
    fn error_predicates_partition() {
        let deck = RadDeck::parse_str("/TRANSFORM/SYM/1\nsym\n18 11\n/END\n").expect("deck");
        let nodes = BTreeSet::from([11]);
        let decode = decode_transform_card(&deck.cards[0], &nodes).expect_err("single");
        assert!(decode.is_decode());
        assert!(!decode.is_lookup());
        assert!(!decode.is_degenerate());
        assert!(!decode.is_unsupported());
        assert!(!decode.is_layout());
        let unbalanced = RadDeck::parse_str("//ENDSUB\n/END\n").expect("deck parses");
        let layout = split_submodels(&unbalanced).expect_err("unbalanced");
        assert!(layout.is_layout());
    }

    #[test]
    fn transf_chain_matches_starter() {
        let deck = RadDeck::parse_str(&transf_deck_text()).expect("deck");
        let transforms = collect_transforms(&deck).expect("collect");
        assert_eq!(transforms.len(), 4);
        assert_eq!(
            transforms.iter().map(Transform::kind).collect::<Vec<_>>(),
            vec![
                TransformKind::Sym,
                TransformKind::Sca,
                TransformKind::Rot,
                TransformKind::Tra
            ]
        );
        assert_eq!(
            transforms.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        let applied = apply_transforms(&deck).expect("apply");
        assert_eq!(applied.stages.len(), 4);
        let mut probes: BTreeMap<&str, BTreeMap<i64, [f64; 3]>> = BTreeMap::new();
        for line in TRANSF_PROBES.lines() {
            let mut parts = line.split_whitespace();
            let stage = parts.next().expect("stage");
            let id: i64 = parts.next().expect("id").parse().expect("int");
            let x: f64 = parts.next().expect("x").parse().expect("real");
            let y: f64 = parts.next().expect("y").parse().expect("real");
            let z: f64 = parts.next().expect("z").parse().expect("real");
            probes.entry(stage).or_default().insert(id, [x, y, z]);
        }
        for (stage, key) in [
            (&applied.stages[0], "SYM"),
            (&applied.stages[1], "SCA"),
            (&applied.stages[2], "ROT"),
        ] {
            let expected = probes.get(key).expect("probe stage");
            for (id, want) in expected {
                let got = stage.nodes.get(id).expect("probe node");
                for axis in 0..3 {
                    assert!(
                        (got[axis] - want[axis]).abs() < VENDOR_BAND,
                        "{key} node {id} axis {axis}"
                    );
                }
            }
        }
        let expected = parse_table(TRANSF_FINAL);
        let dev = worst_deviation(&expected, &applied.final_nodes);
        assert!(dev < VENDOR_BAND, "final chain dev {dev:.3e}");
        for (id, want) in &expected {
            let got = applied.final_nodes.get(id).expect("final node");
            for axis in 0..3 {
                assert!(
                    (got[axis] - want[axis]).abs() < VENDOR_BAND,
                    "final node {id} axis {axis}"
                );
            }
        }
        let refs = parse_table(TRANSF_INPUT);
        for id in [1086, 1089, 1095, 1114, 1115] {
            assert_eq!(applied.final_nodes.get(&id), refs.get(&id));
        }
    }

    #[test]
    fn offsets_default_and_merge_identity() {
        let deck =
            RadDeck::parse_str("//SUBMODEL/1\nmodel\n/NODE\n1 0.0 0.0 0.0\n//ENDSUB\n/END\n")
                .expect("deck");
        let sections = split_submodels(&deck).expect("split");
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].level, 1);
        assert_eq!(sections[0].father, None);
        assert!(sections[0].offsets.is_zero());
        let merged = merge_submodels(&deck).expect("merge");
        let kept: Vec<_> = deck
            .cards
            .iter()
            .filter(|card| !matches!(card.kind, CardKind::Submodel | CardKind::Endsub))
            .cloned()
            .collect();
        assert_eq!(merged.cards, kept);
        let counts = section_inventory(&deck, &sections[0]);
        assert_eq!(counts.get("/NODE"), Some(&1));
    }

    #[test]
    fn nesting_reports_levels_and_fathers() {
        let deck = RadDeck::parse_str(
            "//SUBMODEL/1\nouter\n/NODE\n1 0.0 0.0 0.0\n//SUBMODEL/2\ninner\n/SHELL/1\n1 1 1 1 1\n//ENDSUB\n//ENDSUB\n/END\n",
        )
        .expect("deck");
        let sections = split_submodels(&deck).expect("split");
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].id, 2);
        assert_eq!(sections[0].level, 2);
        assert_eq!(sections[0].father, Some(1));
        assert_eq!(sections[1].id, 1);
        assert_eq!(sections[1].level, 1);
        assert_eq!(sections[1].father, None);
        let merged = merge_submodels(&deck).expect("merge");
        assert_eq!(merged.cards.len(), 3);
    }

    #[test]
    fn nonzero_offsets_reject_merge() {
        let deck = RadDeck::parse_str(
            "//SUBMODEL/1\nmodel\n100 0 0 0 0 0 0\n/NODE\n1 0.0 0.0 0.0\n//ENDSUB\n/END\n",
        )
        .expect("deck");
        let sections = split_submodels(&deck).expect("split");
        assert!(!sections[0].offsets.is_zero());
        assert_eq!(
            sections[0].offsets.effective(),
            [100, 100, 100, 100, 100, 100, 100]
        );
        let error = merge_submodels(&deck).expect_err("nonzero must fail");
        assert!(error.is_unsupported());
        assert!(error.to_string().contains("//SUBMODEL/1"));
        let sparse =
            RadDeck::parse_str("//SUBMODEL/2\nmodel\n0 0 7\n//ENDSUB\n/END\n").expect("deck");
        let sections = split_submodels(&sparse).expect("split");
        assert_eq!(sections[0].offsets.effective(), [0, 0, 7, 0, 0, 0, 0]);
    }

    #[test]
    fn unbalanced_markers_fail_closed() {
        let open = RadDeck::parse_str("//SUBMODEL/1\nmodel\n/NODE\n1 0.0 0.0 0.0\n/END\n")
            .expect("deck parses");
        let error = split_submodels(&open).expect_err("unterminated");
        assert!(error.is_layout());
        assert!(error.to_string().contains("//SUBMODEL/1"));
        let bare = RadDeck::parse_str("//SUBMODEL\nmodel\n//ENDSUB\n/END\n").expect("deck parses");
        let error = split_submodels(&bare).expect_err("missing id");
        assert!(error.is_layout());
    }

    #[test]
    fn duplicate_nodes_reject_table() {
        let deck =
            RadDeck::parse_str("/NODE\n1 0.0 0.0 0.0\n1 1.0 1.0 1.0\n/END\n").expect("deck parses");
        let error = node_table(&deck).expect_err("duplicate");
        assert!(error.is_lookup());
        let deck = RadDeck::parse_str(
            "/NODE\n1 0.0 0.0 0.0\n/GRNOD/NODE/7\nseven\n2\n/TRANSFORM/TRA/1\ntra\n7 1.0\n/END\n",
        )
        .expect("deck parses");
        let error = apply_transforms(&deck).expect_err("missing member");
        assert!(error.is_lookup());
        let deck = RadDeck::parse_str(
            "/NODE\n1 0.0 0.0 0.0\n2 0.0 0.0 0.0\n/GRNOD/NODE/7\nseven\n1\n/TRANSFORM/ROT/1\nrot\n7 1 2\n45.0\n/END\n",
        )
        .expect("deck parses");
        let error = apply_transforms(&deck).expect_err("degenerate");
        assert!(error.is_degenerate());
    }

    #[test]
    fn vendor_decks_replay_when_available() {
        let Ok(root) = std::env::var(crate::rad::oc_golden::MINIQA_ENV) else {
            eprintln!("skipped: set OPENCOURANT_MINIQA to the miniqa root");
            return;
        };
        let root = std::path::PathBuf::from(root);
        let starter = root.join("SUBMODEL/TRANSF/data/BOULE1V5_0000.rad");
        let deck = RadDeck::parse_file(&starter).expect("TRANSF parses");
        let transforms = collect_transforms(&deck).expect("collect");
        assert_eq!(transforms.len(), 4);
        assert_eq!(
            transforms.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert!(transforms.iter().all(|t| t.group == 18));
        let table = node_table(&deck).expect("table");
        let transcribed = parse_table(TRANSF_INPUT);
        for (id, want) in &transcribed {
            assert_eq!(table.get(id), Some(want), "node {id}");
        }
        let members = resolve_group_nodes(&deck, 18).expect("group 18");
        let expected: Vec<i64> = (1000..1040).collect();
        assert_eq!(members, expected);
        let applied = apply_transforms(&deck).expect("apply");
        let finals = parse_table(TRANSF_FINAL);
        let dev = worst_deviation(&finals, &applied.final_nodes);
        assert!(dev < VENDOR_BAND, "vendor chain dev {dev:.3e}");
        for (case, file, inside, functs) in [
            (
                "ACTIV",
                "SUBMODEL/update_v_100/ACTIV/data/HYPE_SUB_0000.rad",
                41,
                0,
            ),
            (
                "PLOAD",
                "SUBMODEL/update_v_100/PLOAD/data/BOXBEAM_SUB_0000.rad",
                77,
                2,
            ),
            (
                "FAIL",
                "SUBMODEL/update_v_100/FAIL/Jonhson_Cook/data/SPECT_SUB_0000.rad",
                38,
                10,
            ),
        ] {
            let deck = RadDeck::parse_file(root.join(file)).expect("starter parses");
            let sections = split_submodels(&deck).expect("split");
            assert_eq!(sections.len(), 1, "{case}");
            assert_eq!(sections[0].id, 1, "{case}");
            assert_eq!(sections[0].level, 1, "{case}");
            assert!(sections[0].offsets.is_zero(), "{case}");
            assert_eq!(sections[0].end - sections[0].start, inside, "{case}");
            let merged = merge_submodels(&deck).expect("merge");
            assert_eq!(merged.cards.len(), deck.cards.len() - 2, "{case}");
            let counts = section_inventory(&deck, &sections[0]);
            assert_eq!(counts.get("/FUNCT").copied().unwrap_or(0), functs, "{case}");
        }
    }
}
