//! Universal Radioss deck reader with typed card decoders.
//!
//! This module parses starter and engine (`.rad`) decks into a [`RadDeck`]:
//! a flat card list with frozen [`DeckCard`] shapes (family, identifier,
//! title, token fields, optional [`TypedCard`]). It resolves `#include`
//! files relative to the deck directory, fails closed on unknown headers,
//! and emits canonical text that re-parses to an identical model.
//!
//! Line rules: a `/` at column zero opens a card; `#`, `$`, and bare
//! `Copyright>` license debris lines are comments; anything else is card
//! data. Titles are the first data line when it is non-numeric, except
//! [`CardKind::force_title`] families which always take it. `/FUNCT_PYTHON`
//! keeps every line verbatim because Python is whitespace sensitive, while
//! `/BCS`, `/ALE/BCS`, and `/RWALL/*` rows use fixed 10-character columns
//! so omitted leading fields keep their positions.
//!
//! # Examples
//!
//! ```rust
//! use sonium_qa::rad_parse::RadDeck;
//!
//! let deck = RadDeck::parse_str("/NODE\n1 0.0 0.0 0.0\n/END\n").expect("deck");
//! assert_eq!(deck.cards.len(), 2);
//! let again = deck.round_trip().expect("round trip");
//! assert_eq!(deck, again);
//! ```

// Rust guideline compliant 2026-02-21

#[doc(inline)]
pub use crate::rad_cards::CardKind;
use crate::rad_cards::parse_header;
use std::fmt;
use std::path::{Path, PathBuf};

/// Parsed deck as a flat card list in file order.
#[derive(Debug, Clone, PartialEq)]
pub struct RadDeck {
    /// Cards in file order with includes expanded inline.
    pub cards: Vec<DeckCard>,
}

/// One deck card with raw fields and an optional typed decoding.
#[derive(Debug, Clone, PartialEq)]
pub struct DeckCard {
    /// Card family from the normalized header.
    pub kind: CardKind,
    /// Trailing numeric header identifier when present.
    pub id: Option<i64>,
    /// Second trailing numeric qualifier (`/GRNOD/NODE/1/1`), if any.
    ///
    /// The vendor starter accepts double-numeric headers and our writer
    /// emits them, so the qualifier is preserved opaquely and re-emitted
    /// by [`RadDeck::emit`]. No miniqa deck uses one (gate amendment).
    pub sub_id: Option<i64>,
    /// Title line below the header, empty when the block has none.
    pub title: String,
    /// Data rows as whitespace tokens (see module docs for exceptions).
    pub fields: Vec<Vec<String>>,
    /// Typed decoding for structural and flag families, if any.
    pub typed: Option<TypedCard>,
}

/// Typed decoding of a card block by structural family.
///
/// Mesh, material, property, part, boundary, load, output, interface,
/// wall, and spring families carry their leading structure here; the raw
/// [`DeckCard::fields`] always remain available for the full block. Pure
/// flag families share one decoder each: [`TypedCard::Anim`] covers
/// `/ANIM/*` and `/OUTP/*`, [`TypedCard::Th`] covers `/TH/*`, and
/// [`TypedCard::Print`] covers `/PRINT/*`.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedCard {
    /// `/NODE` coordinate table.
    Nodes(Vec<Node>),
    /// Element table for one element family.
    Elems(ElemBlock),
    /// `/SPHCEL` supporting node list.
    Sphcel(SphcelCard),
    /// `/PART` property/material/subset triple.
    Part(PartCard),
    /// `/MAT/*` law name with initial density when present.
    Mat(MatCard),
    /// `/PROP/*` property family marker.
    Prop(PropCard),
    /// `/BCS` or `/ALE/BCS` boundary codes.
    Bcs(BcsCard),
    /// Gravity, imposed, initial, and concentrated loads.
    Load(LoadCard),
    /// `/INTER/*` leading identifier row.
    Inter(InterCard),
    /// `/RWALL/*` rigid wall header row.
    Rwall(RwallCard),
    /// `/TH/*` variable names with entity entries.
    Th(ThCard),
    /// `/ANIM/*` and `/OUTP/*` flags with numeric parameters.
    Anim(AnimCard),
    /// `/PRINT/*` printout request marker.
    Print(PrintCard),
    /// `/FUNCT` abscissa/value pairs.
    Funct(FunctCard),
    /// `/SENSOR/*` trigger delay with dependencies.
    Sensor(SensorCard),
    /// `/BEGIN` run name, version, and unit systems.
    Begin(BeginCard),
}

/// Mesh node with an identifier and coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Node identifier.
    pub id: i64,
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

/// Element family of an [`ElemBlock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    /// Four-node shells (`/SHELL`).
    Shell,
    /// Three-node shells (`/SH3N`).
    Tri,
    /// Eight-node bricks (`/BRICK`).
    Brick,
    /// Four-node quads (`/QUAD`).
    Quad,
    /// Two-node trusses (`/TRUSS`).
    Truss,
    /// Three-node beams (`/BEAM`).
    Beam,
    /// Two-node springs (`/SPRING`).
    Spring,
    /// Four-node tetrahedra (`/TETRA4`).
    Tetra,
}

/// One finite element with connectivity and shell extras.
#[derive(Debug, Clone, PartialEq)]
pub struct Elem {
    /// Element identifier.
    pub id: i64,
    /// Node identifiers in deck order.
    pub nodes: Vec<i64>,
    /// Orthotropy angle for shells when the row carries one.
    pub angle: Option<f64>,
    /// Per-element thickness for shells when the row carries one.
    pub thick: Option<f64>,
}

/// Element table of a single family with its type tag.
#[derive(Debug, Clone, PartialEq)]
pub struct ElemBlock {
    /// Element family decoded from the card header.
    pub etype: ElemType,
    /// Elements in deck order.
    pub elems: Vec<Elem>,
}

/// SPH cell block listing supporting nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct SphcelCard {
    /// Supporting node identifiers in deck order.
    pub nodes: Vec<i64>,
}

/// Part definition combining property and material references.
///
/// Reference Guide `/PART` (p.194): title, then `prop_ID mat_ID
/// subset_ID Thick` with the subset and thickness optional.
#[derive(Debug, Clone, PartialEq)]
pub struct PartCard {
    /// Property identifier of the part elements.
    pub prop: i64,
    /// Material identifier of the part elements.
    pub mat: i64,
    /// Subset identifier when the row carries one.
    pub subset: Option<i64>,
    /// Part thickness override when the row carries one.
    pub thick: Option<f64>,
}

/// Material law name with its initial density when present.
///
/// Reference Guide `/MAT/LAW01` (p.1008): row zero opens with `RHO_I` for
/// single-material laws. Multi-material laws reuse row zero for flags
/// (`/MAT/LAW51` carries `Iform`), so `rho_init` stays `None` there and
/// for gas tables or empty void blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct MatCard {
    /// Law name such as `LAW1` or `PLAS_JOHNS`.
    pub law: String,
    /// First scalar of row zero, the initial density for most laws.
    pub rho_init: Option<f64>,
}

/// Property family marker.
///
/// Reference Guide `/PROP/SHELL` (p.1615): property layouts vary too much
/// by type for a uniform scalar decoding, so v1 carries the family tag
/// while the raw fields keep the full block.
#[derive(Debug, Clone, PartialEq)]
pub struct PropCard {
    /// Property type such as `SHELL` or `TYPE1`.
    pub ptype: String,
}

/// Boundary condition codes with skew and group references.
///
/// Reference Guide `/BCS` (p.1885) and `/ALE/BCS` (p.2179): a six-boolean
/// code pair plus skew and group identifiers. For `/ALE/BCS` the pair
/// carries grid/Lagrange codes instead of translation/rotation codes.
#[derive(Debug, Clone, PartialEq)]
pub struct BcsCard {
    /// First three-boolean code (translation, grid for ALE).
    pub tra: String,
    /// Second three-boolean code (rotation, Lagrange for ALE).
    pub rot: String,
    /// Skew system identifier when the row carries one.
    pub skew: Option<i64>,
    /// Node group identifier when the row carries one.
    pub grnod: Option<i64>,
}

/// Interface leading row with the interface type tag.
///
/// Per-type semantics differ (Reference Guide `/INTER/TYPE7`, p.252), so
/// v1 decodes the leading integer row uniformly without naming fields.
#[derive(Debug, Clone, PartialEq)]
pub struct InterCard {
    /// Interface type number from the card header.
    pub itype: i64,
    /// First data row as integers in deck order.
    pub head: Vec<i64>,
}

/// Rigid wall header row with master node and groups.
///
/// Reference Guide `/RWALL` (p.1922): `node_ID Slide grnd_ID1 grnd_ID2`
/// with the master node omitted for fixed walls (decoded as zero).
#[derive(Debug, Clone, PartialEq)]
pub struct RwallCard {
    /// Master node identifier, zero for fixed walls.
    pub node: i64,
    /// Sliding flag.
    pub slide: i64,
    /// Node group identifiers in deck order.
    pub grnds: Vec<i64>,
}

/// Gravity and imposed-motion load block by load kind.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadCard {
    /// `/GRAV` gravity load on a node group.
    Grav(GravCard),
    /// `/IMPVEL` imposed velocity block.
    Impvel(ImpCard),
    /// `/IMPDISP` imposed displacement block.
    Impdisp(ImpCard),
    /// `/INIVEL/*` initial velocity block.
    Inivel(InivelCard),
    /// `/CLOAD` concentrated force or moment.
    Cload(CloadCard),
    /// `/PLOAD` pressure load on a surface.
    Pload(PloadCard),
    /// `/ACCEL` accelerometer definition.
    Accel(AccelCard),
    /// `/VEL/*` engine imposed-velocity entity list.
    Vel(VelCard),
}

/// Gravity load with direction, group, and scale factors.
///
/// Reference Guide `/GRAV` (p.2100): `fct_IDT Dir skew_ID sens_ID
/// grnd_ID Ascalex FscaleY` with the two scales optional.
#[derive(Debug, Clone, PartialEq)]
pub struct GravCard {
    /// Time function identifier.
    pub funct: i64,
    /// Direction flag such as `X`, `Y`, or `Z`.
    pub dir: String,
    /// Skew system identifier.
    pub skew: i64,
    /// Sensor identifier.
    pub sens: i64,
    /// Node group identifier.
    pub grnod: i64,
    /// Abscissa scale factor when the row carries one.
    pub ax: Option<f64>,
    /// Ordinate scale factor when the row carries one.
    pub fy: Option<f64>,
    /// Eighth token when present, preserved opaquely.
    ///
    /// Same writer convention as [`CloadCard::extra_scale`]: our decks
    /// carry the gravity magnitude here and the starter accepts the row.
    pub extra_scale: Option<f64>,
}

/// Imposed velocity or displacement block with scale row.
///
/// Reference Guide `/IMPDISP` (p.1897); `/IMPVEL` (p.1904) shares the
/// layout: an identifier row plus `Ascalex FscaleY Tstart Tstop`. Scales
/// default per the manual (abscissa and ordinate 1.0, stop 1E30) when
/// absent; the decoded options stay `None` so consumers apply them.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpCard {
    /// Time function identifier.
    pub funct: i64,
    /// Direction flag such as `X`, `YY`, or `ZZ`.
    pub dir: String,
    /// Skew system identifier.
    pub skew: i64,
    /// Sensor identifier.
    pub sens: i64,
    /// Node group identifier.
    pub grnod: i64,
    /// Coordinate flag when the row carries one.
    pub icoor: Option<i64>,
    /// Undocumented seventh identifier token, always zero in the corpus.
    pub extra: Option<i64>,
    /// Abscissa scale factor when the scale row carries one.
    pub ax: Option<f64>,
    /// Ordinate scale factor when the scale row carries one.
    pub fy: Option<f64>,
    /// Start time when the scale row carries one.
    pub tstart: Option<f64>,
    /// Stop time when the scale row carries one.
    pub tstop: Option<f64>,
}

/// Initial velocity vector with group and skew references.
///
/// Reference Guide `/INIVEL` (p.2070): `VX VY VZ grnd_ID skew_ID`.
/// The header kind keeps the type (`TRA`, `ROT`, `T+G`).
#[derive(Debug, Clone, PartialEq)]
pub struct InivelCard {
    /// Initial velocity components.
    pub v: [f64; 3],
    /// Node group identifier.
    pub grnod: i64,
    /// Skew system identifier.
    pub skew: i64,
}

/// Concentrated load with direction, group, and scale factors.
///
/// Reference Guide `/CLOAD` (p.2098): `fct_IDT Dir skew_ID sens_ID
/// grnd_ID Ascalex Fscaley` with the two scales optional.
#[derive(Debug, Clone, PartialEq)]
pub struct CloadCard {
    /// Time function identifier.
    pub funct: i64,
    /// Direction flag such as `X`, `Y`, or `Z`.
    pub dir: String,
    /// Skew system identifier.
    pub skew: i64,
    /// Sensor identifier.
    pub sens: i64,
    /// Node group identifier.
    pub grnod: i64,
    /// Abscissa scale factor when the row carries one.
    pub ax: Option<f64>,
    /// Ordinate scale factor when the row carries one.
    pub fy: Option<f64>,
    /// Eighth token when present, preserved opaquely.
    ///
    /// The guide documents seven `/CLOAD` fields, but our writer emits
    /// the load magnitude as an eighth token and the vendor starter
    /// accepts the row (cantilever/plate legs green), so the reader
    /// keeps it rather than failing (gate amendment).
    pub extra_scale: Option<f64>,
}

/// Pressure load on a surface with optional sensor and scales.
///
/// Reference Guide `/PLOAD` (p.2117): `surf_ID fct_IDT sens_ID Ipinch
/// Idel Ascalex Fscaley` with trailing fields optional.
#[derive(Debug, Clone, PartialEq)]
pub struct PloadCard {
    /// Surface identifier.
    pub surf: i64,
    /// Time function identifier.
    pub funct: i64,
    /// Sensor identifier when the row carries one.
    pub sens: Option<i64>,
    /// Pinch flag when the row carries one.
    pub pinch: Option<i64>,
    /// Deletion flag when the row carries one.
    pub del: Option<i64>,
    /// Abscissa scale factor when the row carries one.
    pub ax: Option<f64>,
    /// Ordinate scale factor when the row carries one.
    pub fy: Option<f64>,
}

/// Accelerometer definition with node, skew, and cutoff.
///
/// Reference Guide `/ACCEL` (p.2620): `node_ID Skew_ID Fcut`.
#[derive(Debug, Clone, PartialEq)]
pub struct AccelCard {
    /// Node identifier.
    pub node: i64,
    /// Skew system identifier.
    pub skew: i64,
    /// Cutoff frequency.
    pub fcut: f64,
}

/// Engine imposed-velocity entity list.
#[derive(Debug, Clone, PartialEq)]
pub struct VelCard {
    /// Entity identifiers flattened in deck order.
    pub ids: Vec<i64>,
}

/// Time history request with variable names and entities.
///
/// Reference Guide `/TH/NODE` (p.2557): title, one variable-name row,
/// then entity rows of `id skew_ID name`. Blocks without a variable
/// row (a 44SLD8 `/TH/PART` case) decode with empty `vars`.
#[derive(Debug, Clone, PartialEq)]
pub struct ThCard {
    /// Requested variable names such as `DEF` or `VX`.
    pub vars: Vec<String>,
    /// Requested entities in deck order.
    pub ents: Vec<ThEnt>,
}

/// One time history entity with optional skew and name.
#[derive(Debug, Clone, PartialEq)]
pub struct ThEnt {
    /// Entity identifier.
    pub id: i64,
    /// Skew or frame reference when the row carries an integer second token.
    pub skew: Option<i64>,
    /// Entity name for the history curves, empty when absent.
    pub name: String,
}

/// Animation or ASCII output flag with numeric parameters.
///
/// Reference Guide `/ANIM/DT` (p.2876) and `/OUTP` (p.2933): most flag
/// cards carry no data (presence enables the output); `/ANIM/DT`,
/// `/OUTP/DT`, and version stamps carry scalar parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimCard {
    /// Flag key such as `ANIM/ELEM/VONM` or `OUTP/DT`.
    pub key: String,
    /// Numeric parameters flattened in deck order, empty for pure flags.
    pub params: Vec<f64>,
}

/// Printout frequency request marker.
///
/// Reference Guide `/PRINT` (p.2775): parameters live in the header and
/// the block carries no data, so presence alone is the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrintCard;

/// Function abscissa and value pairs.
///
/// Reference Guide `/FUNCT` (p.2239): title, then strictly increasing
/// `X Y` pairs in fixed 20-character columns.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctCard {
    /// Points in deck order as `(abscissa, value)` pairs.
    pub points: Vec<[f64; 2]>,
}

/// Sensor trigger delay with sensor dependencies.
///
/// Reference Guide `/SENSOR` (p.2270): `/SENSOR/TIME` carries a delay
/// alone while `/SENSOR/SENS` adds two sensor identifiers.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorCard {
    /// Time delay before the sensor fires.
    pub delay: f64,
    /// Sensor dependencies, empty for time sensors.
    pub deps: Vec<i64>,
}

/// Deck header with run name, version, and unit systems.
///
/// Reference Guide `/BEGIN` (p.47): run name, an input-version row, and
/// input and work unit rows; short engine-style headers may omit units.
#[derive(Debug, Clone, PartialEq)]
pub struct BeginCard {
    /// Input manual version such as 2019.
    pub version: i64,
    /// Remaining version-row integers (run and submodel flags).
    pub flags: Vec<i64>,
    /// Input mass, length, and time units when the block carries them.
    pub input: Option<[String; 3]>,
    /// Work mass, length, and time units when the block carries them.
    pub work: Option<[String; 3]>,
}

/// Failure of a deck read, decode, or round trip.
#[derive(Debug)]
pub enum ParseError {
    /// Filesystem or encoding failure with the deck path.
    Io {
        /// Deck or include path that failed.
        path: PathBuf,
        /// Underlying I/O error.
        error: std::io::Error,
    },
    /// Header line matching no known card family.
    UnknownHeader {
        /// File holding the header.
        file: String,
        /// One-based line number of the header.
        line: usize,
        /// Raw header text.
        raw: String,
    },
    /// Data line outside any card block.
    DanglingData {
        /// File holding the line.
        file: String,
        /// One-based line number.
        line: usize,
        /// Line preview.
        preview: String,
    },
    /// `#include` that cannot be resolved or read.
    Include {
        /// File holding the directive.
        file: String,
        /// One-based line number.
        line: usize,
        /// Requested include path as written.
        target: String,
        /// Why resolution failed.
        message: String,
    },
    /// Card block whose data cannot build its typed value.
    Decode {
        /// File holding the card.
        file: String,
        /// One-based line number of the header.
        line: usize,
        /// Canonical family header.
        kind: &'static str,
        /// Block identifier when present.
        id: Option<i64>,
        /// Why decoding failed.
        message: String,
    },
    /// `#include` nesting deeper than [`MAX_INCLUDE_DEPTH`].
    TooDeep {
        /// File holding the directive.
        file: String,
        /// One-based line number.
        line: usize,
    },
}

impl fmt::Display for ParseError {
    /// Renders the failure with file and line context.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => {
                write!(f, "cannot read {}: {error}", path.display())
            }
            Self::UnknownHeader { file, line, raw } => {
                write!(f, "{file}:{line}: unknown card header {raw:?}")
            }
            Self::DanglingData {
                file,
                line,
                preview,
            } => {
                write!(f, "{file}:{line}: data outside any card: {preview:?}")
            }
            Self::Include {
                file,
                line,
                target,
                message,
            } => {
                write!(f, "{file}:{line}: cannot include {target:?}: {message}")
            }
            Self::Decode {
                file,
                line,
                kind,
                id,
                message,
            } => {
                write!(
                    f,
                    "{file}:{line}: cannot decode {kind} id {id:?}: {message}"
                )
            }
            Self::TooDeep { file, line } => {
                write!(
                    f,
                    "{file}:{line}: include nesting exceeds {MAX_INCLUDE_DEPTH}"
                )
            }
        }
    }
}

impl std::error::Error for ParseError {
    /// Returns the I/O source for read failures.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Maximum `#include` nesting depth.
///
/// Ten would already cover every realistic deck (the corpus peaks at one
/// level); sixteen backstops hostile or cyclic include graphs alongside
/// the cycle detector, at negligible cost to legitimate decks.
const MAX_INCLUDE_DEPTH: usize = 16;

/// Fixed width of Radioss real fields in characters.
///
/// The Reference Guide block format lays out ten 10-character fields per
/// row; reals span two fields. The 20-column fallback repair and the
/// column-aware cards rely on this width.
const REAL_FIELD_WIDTH: usize = 20;

/// Fixed width of Radioss integer fields in characters.
///
/// Boundary and wall header rows pack codes and identifiers into
/// consecutive 10-character fields; omitted leading fields keep their
/// positions only when tokenized by column.
const INT_FIELD_WIDTH: usize = 10;

/// Character column where shell thickness extras start.
///
/// Shell element rows span the 100-character block format with thickness
/// in the trailing 20-character real field; a lone extra starting before
/// this column is an orthotropy angle instead and must not misdecode.
const SHELL_THICK_COLUMN: usize = 80;

/// One input line with provenance for error messages.
struct SpannedLine {
    /// File the line was read from.
    file: String,
    /// One-based line number within that file.
    line: usize,
    /// Raw line text without the line break.
    text: String,
}

/// Open card block accumulating data lines.
struct OpenCard {
    /// Card family from the header.
    kind: CardKind,
    /// Block identifier from the header.
    id: Option<i64>,
    /// Second header qualifier, if any.
    sub_id: Option<i64>,
    /// File holding the header.
    file: String,
    /// One-based header line number.
    line: usize,
    /// Kept data lines in deck order.
    lines: Vec<String>,
}

impl RadDeck {
    /// Parses a deck file, resolving `#include` directives.
    ///
    /// Includes resolve relative to the file holding the directive, nest
    /// to [`MAX_INCLUDE_DEPTH`], and fail on cycles or missing targets.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use sonium_qa::rad_parse::RadDeck;
    ///
    /// let deck = RadDeck::parse_file("model_0000.rad").expect("deck");
    /// assert!(!deck.cards.is_empty());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ParseError`] for unreadable files, unknown headers, data
    /// outside any card, unresolvable includes, or undecodable blocks.
    pub fn parse_file(path: impl AsRef<Path>) -> Result<Self, ParseError> {
        let root = path.as_ref();
        let text = std::fs::read_to_string(root).map_err(|error| ParseError::Io {
            path: root.to_path_buf(),
            error,
        })?;
        let lines: Vec<String> = text.lines().map(str::to_string).collect();
        let display = root.to_string_lossy().into_owned();
        let key = std::fs::canonicalize(root)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| display.clone());
        let mut frames = vec![IncludeFrame {
            dir: root
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from(".")),
            file: display,
            lines,
            index: 0,
        }];
        let mut stack = vec![key];
        Self::lex(&mut frames, &mut stack, true)
    }

    /// Parses deck text without touching the filesystem.
    ///
    /// Canonical output from [`RadDeck::emit`] carries no directives, so
    /// round trips use this constructor. A `#include` line here errors
    /// because there is no directory to resolve it against.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_qa::rad_parse::RadDeck;
    ///
    /// let deck = RadDeck::parse_str("/TITLE\nbeam\n/END\n").expect("deck");
    /// assert_eq!(deck.cards[0].title, "beam");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ParseError`] for unknown headers, data outside any card,
    /// include directives, or undecodable blocks.
    pub fn parse_str(text: &str) -> Result<Self, ParseError> {
        let mut frames = vec![IncludeFrame {
            dir: PathBuf::from("."),
            file: "<str>".to_string(),
            lines: text.lines().map(str::to_string).collect(),
            index: 0,
        }];
        let mut stack = vec![frames[0].file.clone()];
        Self::lex(&mut frames, &mut stack, false)
    }

    /// Splits expanded lines into cards with file context.
    fn lex(
        frames: &mut Vec<IncludeFrame>,
        stack: &mut Vec<String>,
        allow_include: bool,
    ) -> Result<Self, ParseError> {
        let mut cards = Vec::new();
        let mut open: Option<OpenCard> = None;
        while let Some(spanned) = next_line(frames, stack, allow_include, &open)? {
            if spanned.text.starts_with('/') {
                if let Some(card) = open.take() {
                    cards.push(finish_card(card)?);
                }
                let (kind, id, sub_id) =
                    parse_header(&spanned.text).ok_or_else(|| ParseError::UnknownHeader {
                        file: spanned.file.clone(),
                        line: spanned.line,
                        raw: spanned.text.clone(),
                    })?;
                open = Some(OpenCard {
                    kind,
                    id,
                    sub_id,
                    file: spanned.file,
                    line: spanned.line,
                    lines: Vec::new(),
                });
                continue;
            }
            let Some(card) = open.as_mut() else {
                if spanned.text.trim().is_empty() {
                    continue;
                }
                let trimmed = spanned.text.trim_start();
                if is_comment(trimmed) {
                    continue;
                }
                return Err(ParseError::DanglingData {
                    file: spanned.file,
                    line: spanned.line,
                    preview: preview(&spanned.text),
                });
            };
            if card.kind == CardKind::FunctPython {
                card.lines.push(trim_end(&spanned.text).to_string());
                continue;
            }
            let trimmed = spanned.text.trim();
            if trimmed.is_empty() || is_comment(spanned.text.trim_start()) {
                continue;
            }
            card.lines.push(trim_end(&spanned.text).to_string());
        }
        if let Some(card) = open.take() {
            cards.push(finish_card(card)?);
        }
        Ok(Self { cards })
    }

    /// Emits canonical text that re-parses to an identical model.
    ///
    /// Headers use canonical families with identifiers, titles print
    /// verbatim, and rows print space-joined except for the column-aware
    /// head rows of boundary and wall cards and lone shell thickness
    /// values, which pad to their thickness column. Comments and include
    /// directives never survive; the model predates them.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_qa::rad_parse::RadDeck;
    ///
    /// let deck = RadDeck::parse_str("/NODE\n1 0.0 0.0 0.0\n").expect("deck");
    /// assert_eq!(deck.emit(), "/NODE\n1 0.0 0.0 0.0\n");
    /// ```
    #[must_use]
    pub fn emit(&self) -> String {
        let mut out = String::new();
        for card in &self.cards {
            out.push_str(card.kind.as_str());
            if let Some(id) = card.id {
                out.push('/');
                out.push_str(&id.to_string());
            }
            if let Some(sub) = card.sub_id {
                out.push('/');
                out.push_str(&sub.to_string());
            }
            out.push('\n');
            if !card.title.is_empty() {
                out.push_str(&card.title);
                out.push('\n');
            }
            if let Some(widths) = column_widths(card.kind) {
                for (index, row) in card.fields.iter().enumerate() {
                    if index == 0 {
                        out.push_str(&pad_columns(row, widths));
                    } else {
                        out.push_str(&row.join(" "));
                    }
                    out.push('\n');
                }
            } else if matches!(card.kind, CardKind::Shell | CardKind::Sh3n) {
                for row in &card.fields {
                    out.push_str(&emit_shell_row(card.kind, row));
                    out.push('\n');
                }
            } else {
                for row in &card.fields {
                    out.push_str(&row.join(" "));
                    out.push('\n');
                }
            }
        }
        out
    }

    /// Emits canonical text and parses it back into a model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_qa::rad_parse::RadDeck;
    ///
    /// let deck = RadDeck::parse_str("/PART/1\np\n1 1 0\n").expect("deck");
    /// assert_eq!(deck.round_trip().expect("round trip"), deck);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ParseError`] when the canonical text fails to parse,
    /// which indicates a lexer or emitter defect rather than bad input.
    pub fn round_trip(&self) -> Result<Self, ParseError> {
        Self::parse_str(&self.emit())
    }
}

/// One open file in the include stack.
struct IncludeFrame {
    /// Directory for resolving nested includes.
    dir: PathBuf,
    /// Display path of this file.
    file: String,
    /// All lines of this file.
    lines: Vec<String>,
    /// Next line index.
    index: usize,
}

/// Yields the next non-directive line, expanding includes inline.
fn next_line(
    frames: &mut Vec<IncludeFrame>,
    stack: &mut Vec<String>,
    allow_include: bool,
    open: &Option<OpenCard>,
) -> Result<Option<SpannedLine>, ParseError> {
    loop {
        let Some(frame) = frames.last_mut() else {
            return Ok(None);
        };
        if frame.index >= frame.lines.len() {
            frames.pop();
            stack.pop();
            continue;
        }
        let text = frame.lines[frame.index].clone();
        let file = frame.file.clone();
        let line = frame.index + 1;
        let dir = frame.dir.clone();
        frame.index += 1;
        if let Some(target) = include_target(&text) {
            let python = open
                .as_ref()
                .is_some_and(|c| c.kind == CardKind::FunctPython);
            if python {
                return Ok(Some(SpannedLine { file, line, text }));
            }
            if !allow_include {
                return Err(ParseError::Include {
                    file,
                    line,
                    target,
                    message: "includes need parse_file".to_string(),
                });
            }
            if target.is_empty() {
                return Err(ParseError::Include {
                    file,
                    line,
                    target,
                    message: "missing include path".to_string(),
                });
            }
            push_include(frames, stack, &dir, &file, line, &target)?;
            continue;
        }
        return Ok(Some(SpannedLine { file, line, text }));
    }
}

/// Extracts the include target from a directive line, if any.
///
/// A bare directive with no path yields an empty target so the caller
/// can fail closed on the malformed line instead of dropping it.
fn include_target(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    if !trimmed
        .get(..8)
        .is_some_and(|p| p.eq_ignore_ascii_case("#include"))
    {
        return None;
    }
    let rest = &trimmed[8..];
    if !rest.is_empty() && !rest.starts_with([' ', '\t']) {
        return None;
    }
    Some(rest.split_whitespace().next().unwrap_or("").to_string())
}

/// Pushes an include file onto the reading stack.
fn push_include(
    frames: &mut Vec<IncludeFrame>,
    stack: &mut Vec<String>,
    dir: &Path,
    file: &str,
    line: usize,
    target: &str,
) -> Result<(), ParseError> {
    if frames.len() >= MAX_INCLUDE_DEPTH {
        return Err(ParseError::TooDeep {
            file: file.to_string(),
            line,
        });
    }
    let path = dir.join(target);
    let canonical = std::fs::canonicalize(&path).map_err(|_| ParseError::Include {
        file: file.to_string(),
        line,
        target: target.to_string(),
        message: "cannot resolve path".to_string(),
    })?;
    let key = canonical.to_string_lossy().into_owned();
    if stack.contains(&key) {
        return Err(ParseError::Include {
            file: file.to_string(),
            line,
            target: target.to_string(),
            message: "include cycle".to_string(),
        });
    }
    let text = std::fs::read_to_string(&canonical).map_err(|error| ParseError::Io {
        path: canonical.clone(),
        error,
    })?;
    let parent = canonical
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    stack.push(key);
    frames.push(IncludeFrame {
        dir: parent,
        file: canonical.to_string_lossy().into_owned(),
        lines: text.lines().map(str::to_string).collect(),
        index: 0,
    });
    Ok(())
}

/// Reports comment lines dropped outside `/FUNCT_PYTHON`.
///
/// `#` marks Radioss comments and `$` marks converter comments (a
/// tstpeel `/TH/RBODY` case); bare `Copyright>` lines are license debris
/// from unpatched vendor `qadiags.inc` files (see miniqa `PATCHES.md`).
/// The caller passes the trimmed-start line.
fn is_comment(trimmed_start: &str) -> bool {
    trimmed_start.starts_with('#')
        || trimmed_start.starts_with('$')
        || trimmed_start.starts_with("Copyright>")
}

/// Strips trailing whitespace, including carriage returns.
fn trim_end(text: &str) -> &str {
    text.trim_end_matches([' ', '\t', '\r'])
}

/// Shortens a line for error previews.
fn preview(text: &str) -> String {
    text.chars().take(60).collect()
}

/// Builds a deck card from accumulated lines with typed decoding.
fn finish_card(open: OpenCard) -> Result<DeckCard, ParseError> {
    let fail = |message: String| ParseError::Decode {
        file: open.file.clone(),
        line: open.line,
        kind: open.kind.as_str(),
        id: open.id,
        message,
    };
    let (title, rows) = split_title(open.kind, &open.lines);
    let mut fields = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        fields.push(tokenize_row(open.kind, index, row).map_err(fail)?);
    }
    let typed = decode_typed(open.kind, open.id, &title, &fields).map_err(fail)?;
    Ok(DeckCard {
        kind: open.kind,
        id: open.id,
        sub_id: open.sub_id,
        title,
        fields,
        typed,
    })
}

/// Splits the title line from data rows for one card.
fn split_title(kind: CardKind, lines: &[String]) -> (String, &[String]) {
    if lines.is_empty() {
        return (String::new(), lines);
    }
    if kind.force_title() || !is_data_line(&lines[0]) {
        (lines[0].trim().to_string(), &lines[1..])
    } else {
        (String::new(), lines)
    }
}

/// Reports first lines that decode as data rows rather than titles.
///
/// Uses whitespace tokens with column repair and sign splitting, so
/// fixed-format rows that tokenize dirty still count as data. Column
/// layouts stay out of the check: a long title must never trip the
/// head-row column bounds. A single-token title shaped exactly like a
/// fused number would misread as data; no miniqa title does.
fn is_data_line(raw: &str) -> bool {
    let tokens: Vec<String> = raw.split_whitespace().map(str::to_string).collect();
    let tokens =
        if tokens.iter().any(|t| needs_repair(t)) && !tokens.iter().any(|t| is_sign_fused(t)) {
            repair_columns(raw).unwrap_or(tokens)
        } else {
            tokens
        };
    let mut count = 0;
    for token in &tokens {
        for piece in split_token(token) {
            count += 1;
            if !is_numeric_token(&piece) {
                return false;
            }
        }
    }
    count > 0
}

/// Reports tokens that parse as deck integers or reals.
fn is_numeric_token(token: &str) -> bool {
    let Some(first) = token.as_bytes().first() else {
        return false;
    };
    if !matches!(first, b'+' | b'-' | b'.' | b'0'..=b'9') {
        return false;
    }
    parse_i64(token).is_some() || parse_f64(token).is_some()
}

/// Parses a deck integer, tolerating leading zeros and signs.
fn parse_i64(token: &str) -> Option<i64> {
    token.parse::<i64>().ok()
}

/// Parses a deck real, accepting Fortran `D` exponents.
///
/// Returns `None` for non-numeric text and non-finite values, which are
/// malformed data rather than usable coordinates or parameters.
fn parse_f64(token: &str) -> Option<f64> {
    let normalized = token.replace('D', "E").replace('d', "e");
    normalized.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Tokenizes one data row following the card layout rules.
///
/// Column layouts apply to the head row only; wall body rows below it
/// follow generic whitespace tokenization like every other card.
fn tokenize_row(kind: CardKind, index: usize, raw: &str) -> Result<Vec<String>, String> {
    if kind == CardKind::FunctPython {
        return Ok(vec![raw.to_string()]);
    }
    if index == 0
        && let Some(widths) = column_widths(kind)
    {
        let total: usize = widths.iter().sum();
        if raw.chars().count() > total {
            return Err("row overruns its fixed columns".to_string());
        }
        return Ok(tokenize_columns(raw, kind));
    }
    let tokens: Vec<String> = raw.split_whitespace().map(str::to_string).collect();
    check_shell_extra(kind, raw, &tokens)?;
    let damaged =
        tokens.iter().any(|t| needs_repair(t)) && !tokens.iter().any(|t| is_sign_fused(t));
    if damaged && let Some(fixed) = repair_columns(raw) {
        return Ok(fixed);
    }
    if kind == CardKind::Node
        && let Some(fixed) = repair_fused_id(raw, &tokens)
    {
        return Ok(fixed);
    }
    Ok(tokens)
}

/// Splits node identifiers fused with coordinates into valid reals.
///
/// A mapSimple `/NODE` row fuses id `1062` with `1.52E-05` into the
/// valid real `10621.52587890625000E-05`, invisible to damage checks.
/// When the first token straddles the 10-character identifier column
/// without parsing as an integer while the column itself holds one,
/// the identifier-led layout recovers both fields. Other families
/// keep whitespace tokens so free-format real rows never split.
fn repair_fused_id(raw: &str, tokens: &[String]) -> Option<Vec<String>> {
    let first = tokens.first()?;
    if parse_i64(first).is_some() {
        return None;
    }
    let leading = raw.chars().take_while(|c| c.is_whitespace()).count();
    let end = leading + first.chars().count();
    if leading >= INT_FIELD_WIDTH || end <= INT_FIELD_WIDTH {
        return None;
    }
    repair_id_led(raw)
}

/// Rejects ambiguous lone shell extras outside the thickness column.
///
/// A six-token `/SHELL` row carries per-element thickness only when the
/// extra trails in the last 20 columns; an angle-only row would sit
/// mid-line and must fail closed rather than misdecode as thickness.
/// Seven-token rows disambiguate by order (angle then thickness).
fn check_shell_extra(kind: CardKind, raw: &str, tokens: &[String]) -> Result<(), String> {
    let lone = match kind {
        CardKind::Shell => tokens.len() == 6,
        CardKind::Sh3n => tokens.len() == 5,
        _ => false,
    };
    if !lone {
        return Ok(());
    }
    let Some(extra) = tokens.last() else {
        return Ok(());
    };
    let start = raw.rfind(extra.as_str()).map(|b| raw[..b].chars().count());
    if start.is_some_and(|c| c >= SHELL_THICK_COLUMN) {
        Ok(())
    } else {
        Err("lone shell extra outside the thickness column".to_string())
    }
}

/// Reports tokens mixing digits with unparseable content.
fn needs_repair(token: &str) -> bool {
    token.bytes().any(|b| b.is_ascii_digit()) && !is_numeric_token(token)
}

/// Reports numeric-character tokens with an internal sign boundary.
///
/// `160.0-1.5E-05` fuses two free-format numbers at the minus; exponent
/// signs after `E` or `D` never count because the preceding letter fails
/// the digit-or-dot test. Sign-fused rows skip column repair (which would
/// misalign them) and are split by numeric decoders instead.
fn is_sign_fused(token: &str) -> bool {
    if !token.bytes().all(|b| {
        matches!(
            b,
            b'0'..=b'9' | b'.' | b'E' | b'e' | b'D' | b'd' | b'+' | b'-'
        )
    }) {
        return false;
    }
    let chars: Vec<char> = token.chars().collect();
    chars.windows(3).any(|w| {
        matches!(w[1], '+' | '-')
            && matches!(w[0], '0'..='9' | '.')
            && matches!(w[2], '0'..='9' | '.')
    })
}

/// Splits sign-fused numbers inside one token.
fn split_token(token: &str) -> Vec<String> {
    if !is_sign_fused(token) {
        return vec![token.to_string()];
    }
    let mut out = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = token.chars().collect();
    for (i, window) in chars.windows(3).enumerate() {
        if matches!(window[1], '+' | '-')
            && matches!(window[0], '0'..='9' | '.')
            && matches!(window[2], '0'..='9' | '.')
        {
            let at = i + 1;
            out.push(chars[start..at].iter().collect());
            start = at;
        }
    }
    out.push(chars[start..].iter().collect());
    out
}

/// Splits sign-fused numbers inside decoder value lists.
///
/// Only all-numeric tokens split, so names such as `R1_master` (which
/// carry letters) pass through untouched; decoders for numeric rows call
/// this before parsing while name-carrying decoders never do.
fn split_numbers(row: &[String]) -> Vec<String> {
    row.iter().flat_map(|t| split_token(t)).collect()
}

/// Repairs dot-fused fixed-format rows via column re-chunking.
///
/// Writers filling fixed-width real fields omit the separator between
/// adjacent values (`0.0125` + `3.85E-04`); re-chunking the raw line
/// recovers the fields exactly when every chunk validates. Pure real
/// rows use 20-character chunks while identifier-led rows (nodes, whose
/// identifier may itself fuse with the first real) use a 10-character
/// identifier plus 20-character reals. The layout follows the leading
/// column content rather than the damaged tokens. Returns `None`
/// unless the matching layout validates, leaving damaged rows to fail
/// closed in their decoders.
fn repair_columns(raw: &str) -> Option<Vec<String>> {
    let head: String = raw.chars().take(INT_FIELD_WIDTH).collect();
    if !head.trim().is_empty() && parse_i64(head.trim()).is_some() {
        repair_id_led(raw)
    } else {
        repair_reals(raw)
    }
}

/// Repairs all-real rows via 20-character chunks.
fn repair_reals(raw: &str) -> Option<Vec<String>> {
    let chunks = chunk_chars(raw, REAL_FIELD_WIDTH);
    if chunks.is_empty() {
        return None;
    }
    let mut fixed = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.split_whitespace().count() != 1 || parse_f64(trimmed).is_none() {
            return None;
        }
        fixed.push(trimmed.to_string());
    }
    if fixed.is_empty() {
        return None;
    }
    Some(fixed)
}

/// Repairs identifier-led rows via a 10-character head.
///
/// The head chunk must parse as an integer; the tail then parses as
/// whitespace tokens (covering values that overflow their 20-character
/// fields) or, failing that, as 20-character chunks in the manner of
/// [`repair_reals`].
fn repair_id_led(raw: &str) -> Option<Vec<String>> {
    let chars: Vec<char> = raw.chars().collect();
    if chars.is_empty() {
        return None;
    }
    let head: String = chars.iter().take(INT_FIELD_WIDTH).collect();
    let head = head.trim();
    if head.is_empty() || parse_i64(head).is_none() {
        return None;
    }
    let tail: String = chars.iter().skip(INT_FIELD_WIDTH).collect();
    let words: Vec<String> = tail.split_whitespace().map(str::to_string).collect();
    if !words.is_empty() && words.iter().all(|w| parse_f64(w).is_some()) {
        let mut fixed = vec![head.to_string()];
        fixed.extend(words);
        return Some(fixed);
    }
    let mut fixed = vec![head.to_string()];
    for chunk in chunk_chars(&tail, REAL_FIELD_WIDTH) {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.split_whitespace().count() != 1 || parse_f64(trimmed).is_none() {
            return None;
        }
        fixed.push(trimmed.to_string());
    }
    Some(fixed)
}

/// Splits text into fixed-width character chunks.
fn chunk_chars(text: &str, width: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars.chunks(width).map(|c| c.iter().collect()).collect()
}

/// Reports the column layout for position-sensitive cards.
fn column_widths(kind: CardKind) -> Option<&'static [usize]> {
    match kind {
        CardKind::Bcs | CardKind::AleBcs => {
            Some(&[INT_FIELD_WIDTH, INT_FIELD_WIDTH, INT_FIELD_WIDTH])
        }
        CardKind::RwallCyl | CardKind::RwallParal | CardKind::RwallPlane | CardKind::RwallSpher => {
            Some(&[
                INT_FIELD_WIDTH,
                INT_FIELD_WIDTH,
                INT_FIELD_WIDTH,
                INT_FIELD_WIDTH,
            ])
        }
        _ => None,
    }
}

/// Tokenizes one row by fixed column widths.
fn tokenize_columns(raw: &str, kind: CardKind) -> Vec<String> {
    let Some(widths) = column_widths(kind) else {
        return raw.split_whitespace().map(str::to_string).collect();
    };
    let chars: Vec<char> = raw.chars().collect();
    let mut fields = Vec::with_capacity(widths.len());
    let mut offset = 0;
    for width in widths {
        let start = offset.min(chars.len());
        let end = (offset + width).min(chars.len());
        let field: String = chars[start..end].iter().collect();
        fields.push(field.trim().to_string());
        offset += width;
    }
    while fields.last().is_some_and(String::is_empty) {
        fields.pop();
    }
    fields
}

/// Emits one shell row, padding lone thickness to its column.
///
/// Rows without extras print space-joined; a lone extra prints from
/// [`SHELL_THICK_COLUMN`] so re-parsing sees trailing thickness rather
/// than a mid-line angle.
fn emit_shell_row(kind: CardKind, row: &[String]) -> String {
    let nodes = match kind {
        CardKind::Shell => 4,
        CardKind::Sh3n => 3,
        _ => return row.join(" "),
    };
    if row.len() != nodes + 2 {
        return row.join(" ");
    }
    let mut out = String::new();
    for token in row.iter().take(nodes + 1) {
        for _ in 0..INT_FIELD_WIDTH.saturating_sub(token.chars().count()) {
            out.push(' ');
        }
        out.push_str(token);
    }
    while out.chars().count() < SHELL_THICK_COLUMN {
        out.push(' ');
    }
    if let Some(thick) = row.get(nodes + 1) {
        out.push_str(thick);
    }
    out
}

/// Pads fields back into fixed columns for canonical output.
fn pad_columns(row: &[String], widths: &[usize]) -> String {
    let mut out = String::new();
    for (index, width) in widths.iter().enumerate() {
        let field = row.get(index).map(String::as_str).unwrap_or("");
        let pad = width.saturating_sub(field.chars().count());
        for _ in 0..pad {
            out.push(' ');
        }
        out.push_str(field);
    }
    out.trim_end().to_string()
}

/// Decodes a card block into its typed family value, if any.
fn decode_typed(
    kind: CardKind,
    id: Option<i64>,
    title: &str,
    fields: &[Vec<String>],
) -> Result<Option<TypedCard>, String> {
    let _ = (id, title);
    match kind {
        CardKind::Node => Ok(Some(TypedCard::Nodes(decode_nodes(fields)?))),
        CardKind::Shell
        | CardKind::Sh3n
        | CardKind::Brick
        | CardKind::Quad
        | CardKind::Truss
        | CardKind::Beam
        | CardKind::Spring
        | CardKind::Tetra4 => Ok(Some(TypedCard::Elems(decode_elems(kind, fields)?))),
        CardKind::Sphcel => Ok(Some(TypedCard::Sphcel(decode_sphcel(fields)?))),
        CardKind::Part => Ok(Some(TypedCard::Part(decode_part(fields)?))),
        CardKind::Bcs | CardKind::AleBcs => Ok(Some(TypedCard::Bcs(decode_bcs(fields)?))),
        CardKind::Grav => Ok(Some(TypedCard::Load(LoadCard::Grav(decode_grav(fields)?)))),
        CardKind::Impvel => Ok(Some(TypedCard::Load(LoadCard::Impvel(decode_imp(fields)?)))),
        CardKind::Impdisp => Ok(Some(TypedCard::Load(LoadCard::Impdisp(decode_imp(
            fields,
        )?)))),
        CardKind::InivelTra | CardKind::InivelRot | CardKind::InivelT | CardKind::InivelTG => Ok(
            Some(TypedCard::Load(LoadCard::Inivel(decode_inivel(fields)?))),
        ),
        CardKind::Cload => Ok(Some(TypedCard::Load(LoadCard::Cload(decode_cload(
            fields,
        )?)))),
        CardKind::Pload => Ok(Some(TypedCard::Load(LoadCard::Pload(decode_pload(
            fields,
        )?)))),
        CardKind::Accel => Ok(Some(TypedCard::Load(LoadCard::Accel(decode_accel(
            fields,
        )?)))),
        CardKind::VelTraXyz | CardKind::VelTraZ => {
            Ok(Some(TypedCard::Load(LoadCard::Vel(decode_vel(fields)?))))
        }
        CardKind::InterType2
        | CardKind::InterType3
        | CardKind::InterType5
        | CardKind::InterType7
        | CardKind::InterType11
        | CardKind::InterType12
        | CardKind::InterType19
        | CardKind::InterType22
        | CardKind::InterType25 => Ok(Some(TypedCard::Inter(decode_inter(kind, fields)?))),
        CardKind::RwallCyl | CardKind::RwallParal | CardKind::RwallPlane | CardKind::RwallSpher => {
            Ok(Some(TypedCard::Rwall(decode_rwall(fields)?)))
        }
        CardKind::Funct => Ok(Some(TypedCard::Funct(decode_funct(fields)?))),
        CardKind::SensorSens | CardKind::SensorTime => {
            Ok(Some(TypedCard::Sensor(decode_sensor(kind, fields)?)))
        }
        CardKind::Begin => Ok(Some(TypedCard::Begin(decode_begin(fields)?))),
        CardKind::Print | CardKind::Print100 => Ok(Some(TypedCard::Print(decode_print(fields)?))),
        _ => decode_family(kind, fields),
    }
}

/// Decodes prefix-dispatched families (materials, props, flags, TH).
fn decode_family(kind: CardKind, fields: &[Vec<String>]) -> Result<Option<TypedCard>, String> {
    let text = kind.as_str();
    if let Some(law) = text.strip_prefix("/MAT/") {
        return Ok(Some(TypedCard::Mat(decode_mat(law, fields)?)));
    }
    if let Some(ptype) = text.strip_prefix("/PROP/") {
        return Ok(Some(TypedCard::Prop(PropCard {
            ptype: ptype.to_string(),
        })));
    }
    if text.starts_with("/TH/") {
        return Ok(Some(TypedCard::Th(decode_th(fields)?)));
    }
    if text.starts_with("/ANIM/") || text.starts_with("/OUTP/") {
        return Ok(Some(TypedCard::Anim(decode_anim(text, fields)?)));
    }
    Ok(None)
}

/// Requires an integer token with row context.
fn need_i64(row: &[String], index: usize, what: &str) -> Result<i64, String> {
    row.get(index)
        .and_then(|t| parse_i64(t))
        .ok_or_else(|| format!("{what} needs integer token {}", index + 1))
}

/// Requires a real token with row context.
fn need_f64(row: &[String], index: usize, what: &str) -> Result<f64, String> {
    row.get(index)
        .and_then(|t| parse_f64(t))
        .ok_or_else(|| format!("{what} needs real token {}", index + 1))
}

/// Decodes `/NODE` coordinate rows.
fn decode_nodes(fields: &[Vec<String>]) -> Result<Vec<Node>, String> {
    let mut nodes = Vec::with_capacity(fields.len());
    for (row_no, row) in fields.iter().enumerate() {
        let values = split_numbers(row);
        if values.len() != 4 {
            return Err(format!("node row {} needs 4 tokens", row_no + 1));
        }
        nodes.push(Node {
            id: need_i64(&values, 0, "node id")?,
            x: need_f64(&values, 1, "node x")?,
            y: need_f64(&values, 2, "node y")?,
            z: need_f64(&values, 3, "node z")?,
        });
    }
    Ok(nodes)
}

/// Decodes one element-family table.
fn decode_elems(kind: CardKind, fields: &[Vec<String>]) -> Result<ElemBlock, String> {
    let etype = match kind {
        CardKind::Shell => ElemType::Shell,
        CardKind::Sh3n => ElemType::Tri,
        CardKind::Brick => ElemType::Brick,
        CardKind::Quad => ElemType::Quad,
        CardKind::Truss => ElemType::Truss,
        CardKind::Beam => ElemType::Beam,
        CardKind::Spring => ElemType::Spring,
        CardKind::Tetra4 => ElemType::Tetra,
        _ => return Err("not an element family".to_string()),
    };
    let mut elems = Vec::with_capacity(fields.len());
    for (row_no, row) in fields.iter().enumerate() {
        let values = split_numbers(row);
        elems.push(decode_elem(etype, &values).map_err(|m| format!("row {}: {m}", row_no + 1))?);
    }
    Ok(ElemBlock { etype, elems })
}

/// Decodes one element row by family arity.
fn decode_elem(etype: ElemType, values: &[String]) -> Result<Elem, String> {
    let nodes = match etype {
        ElemType::Shell => 4,
        ElemType::Tri => 3,
        ElemType::Brick => 8,
        ElemType::Quad => 4,
        ElemType::Truss => 2,
        ElemType::Beam => 3,
        ElemType::Spring => 2,
        ElemType::Tetra => 4,
    };
    if values.len() < nodes + 1 {
        return Err(format!("needs {} node tokens", nodes + 1));
    }
    let id = need_i64(values, 0, "element id")?;
    let mut ids = Vec::with_capacity(nodes);
    for index in 1..=nodes {
        ids.push(need_i64(values, index, "node id")?);
    }
    let extras = &values[nodes + 1..];
    let (angle, thick) = match (etype, extras.len()) {
        (ElemType::Shell | ElemType::Tri, 0) => (None, None),
        (ElemType::Shell | ElemType::Tri, 1) => (None, Some(need_f64(values, nodes + 1, "thick")?)),
        (ElemType::Shell | ElemType::Tri, 2) => (
            Some(need_f64(values, nodes + 1, "angle")?),
            Some(need_f64(values, nodes + 2, "thick")?),
        ),
        (_, 0) => (None, None),
        _ => return Err("unexpected trailing tokens".to_string()),
    };
    Ok(Elem {
        id,
        nodes: ids,
        angle,
        thick,
    })
}

/// Decodes `/SPHCEL` supporting node rows.
fn decode_sphcel(fields: &[Vec<String>]) -> Result<SphcelCard, String> {
    let mut nodes = Vec::with_capacity(fields.len());
    for (row_no, row) in fields.iter().enumerate() {
        let values = split_numbers(row);
        if values.len() != 1 {
            return Err(format!("sphcel row {} needs 1 token", row_no + 1));
        }
        nodes.push(need_i64(&values, 0, "support node")?);
    }
    Ok(SphcelCard { nodes })
}

/// Decodes `/PART` property and material references.
fn decode_part(fields: &[Vec<String>]) -> Result<PartCard, String> {
    let Some(row) = fields.first() else {
        return Err("part needs one data row".to_string());
    };
    if fields.len() != 1 {
        return Err("part needs exactly one data row".to_string());
    }
    let values = split_numbers(row);
    if values.len() < 2 || values.len() > 4 {
        return Err("part row needs 2 to 4 tokens".to_string());
    }
    let opt_i64 = |index: usize| match values.get(index) {
        None => Ok(None),
        Some(token) => parse_i64(token)
            .map(Some)
            .ok_or_else(|| format!("part needs integer token {}", index + 1)),
    };
    let opt_f64 = |index: usize| match values.get(index) {
        None => Ok(None),
        Some(token) => parse_f64(token)
            .map(Some)
            .ok_or_else(|| format!("part needs real token {}", index + 1)),
    };
    Ok(PartCard {
        prop: need_i64(&values, 0, "prop id")?,
        mat: need_i64(&values, 1, "mat id")?,
        subset: opt_i64(2)?,
        thick: opt_f64(3)?,
    })
}

/// Decodes `/MAT/*` law tags with initial density.
fn decode_mat(law: &str, fields: &[Vec<String>]) -> Result<MatCard, String> {
    // `/MAT/LAW51` row zero carries the `Iform` flag, not a density.
    let rho_init = if law == "LAW51" {
        None
    } else {
        fields.first().and_then(|row| {
            let values = split_numbers(row);
            values.first().and_then(|t| parse_f64(t))
        })
    };
    Ok(MatCard {
        law: law.to_string(),
        rho_init,
    })
}

/// Decodes `/BCS` and `/ALE/BCS` boundary code rows.
fn decode_bcs(fields: &[Vec<String>]) -> Result<BcsCard, String> {
    if fields.len() != 1 {
        return Err("boundary block needs exactly one data row".to_string());
    }
    let row = &fields[0];
    let codes = row.first().map(String::as_str).unwrap_or("");
    if codes.is_empty() {
        return Err("boundary block needs code tokens".to_string());
    }
    let parts: Vec<&str> = codes.split_whitespace().collect();
    let (tra, rot) = match parts.as_slice() {
        [tra, rot] => (tra.to_string(), rot.to_string()),
        [code] if code.len() == 6 && code.bytes().all(|b| b == b'0' || b == b'1') => {
            (code[..3].to_string(), code[3..].to_string())
        }
        [code] => (code.to_string(), String::new()),
        _ => return Err("code field needs one or two tokens".to_string()),
    };
    for code in [&tra, &rot] {
        if !code.bytes().all(|b| b == b'0' || b == b'1') {
            return Err(format!("code {code:?} needs 0/1 characters"));
        }
    }
    let opt_id = |index: usize| match row.get(index).map(String::as_str) {
        None | Some("") => Ok(None),
        Some(token) => parse_i64(token)
            .map(Some)
            .ok_or_else(|| format!("boundary needs integer field {}", index + 1)),
    };
    Ok(BcsCard {
        tra,
        rot,
        skew: opt_id(1)?,
        grnod: opt_id(2)?,
    })
}

/// Requires exactly one data row for single-row definitions.
fn single_row<'a>(fields: &'a [Vec<String>], what: &str) -> Result<&'a [String], String> {
    if fields.len() != 1 {
        return Err(format!("{what} needs exactly one data row"));
    }
    fields
        .first()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{what} needs one data row"))
}

/// Decodes `/GRAV` gravity loads.
fn decode_grav(fields: &[Vec<String>]) -> Result<GravCard, String> {
    let values = split_numbers(single_row(fields, "gravity")?);
    if values.len() < 5 || values.len() > 8 {
        return Err("gravity row needs 5 to 8 tokens".to_string());
    }
    Ok(GravCard {
        funct: need_i64(&values, 0, "function id")?,
        dir: need_dir(&values)?,
        skew: need_i64(&values, 2, "skew id")?,
        sens: need_i64(&values, 3, "sensor id")?,
        grnod: need_i64(&values, 4, "group id")?,
        ax: opt_f64(&values, 5, "abscissa scale")?,
        fy: opt_f64(&values, 6, "ordinate scale")?,
        extra_scale: opt_f64(&values, 7, "extra scale")?,
    })
}

/// Decodes `/CLOAD` concentrated loads.
fn decode_cload(fields: &[Vec<String>]) -> Result<CloadCard, String> {
    let values = split_numbers(single_row(fields, "concentrated load")?);
    if values.len() < 5 || values.len() > 8 {
        return Err("concentrated load row needs 5 to 8 tokens".to_string());
    }
    Ok(CloadCard {
        funct: need_i64(&values, 0, "function id")?,
        dir: need_dir(&values)?,
        skew: need_i64(&values, 2, "skew id")?,
        sens: need_i64(&values, 3, "sensor id")?,
        grnod: need_i64(&values, 4, "group id")?,
        ax: opt_f64(&values, 5, "abscissa scale")?,
        fy: opt_f64(&values, 6, "ordinate scale")?,
        extra_scale: opt_f64(&values, 7, "extra scale")?,
    })
}

/// Requires the direction token of a load row.
fn need_dir(values: &[String]) -> Result<String, String> {
    values
        .get(1)
        .cloned()
        .ok_or_else(|| "load row needs a direction token".to_string())
}

/// Parses an optional real token by position.
fn opt_f64(values: &[String], index: usize, what: &str) -> Result<Option<f64>, String> {
    match values.get(index) {
        None => Ok(None),
        Some(token) => parse_f64(token)
            .map(Some)
            .ok_or_else(|| format!("{what} needs real token {}", index + 1)),
    }
}

/// Parses an optional integer token by position.
fn opt_i64(values: &[String], index: usize, what: &str) -> Result<Option<i64>, String> {
    match values.get(index) {
        None => Ok(None),
        Some(token) => parse_i64(token)
            .map(Some)
            .ok_or_else(|| format!("{what} needs integer token {}", index + 1)),
    }
}

/// Decodes `/IMPVEL` and `/IMPDISP` imposed-motion blocks.
fn decode_imp(fields: &[Vec<String>]) -> Result<ImpCard, String> {
    if fields.len() > 2 {
        return Err("imposed motion needs one or two data rows".to_string());
    }
    let Some(head_row) = fields.first() else {
        return Err("imposed motion needs an identifier row".to_string());
    };
    let head = split_numbers(head_row);
    if head.len() < 5 || head.len() > 7 {
        return Err("imposed identifier row needs 5 to 7 tokens".to_string());
    }
    let scales = match fields.get(1) {
        None => Vec::new(),
        Some(row) => split_numbers(row),
    };
    if scales.len() > 4 {
        return Err("imposed scale row needs at most 4 tokens".to_string());
    }
    for (index, token) in scales.iter().enumerate() {
        if parse_f64(token).is_none() {
            return Err(format!("imposed scale token {} needs a real", index + 1));
        }
    }
    let scale = |index: usize| scales.get(index).and_then(|t| parse_f64(t));
    Ok(ImpCard {
        funct: need_i64(&head, 0, "function id")?,
        dir: need_dir(&head)?,
        skew: need_i64(&head, 2, "skew id")?,
        sens: need_i64(&head, 3, "sensor id")?,
        grnod: need_i64(&head, 4, "group id")?,
        icoor: opt_i64(&head, 5, "coordinate flag")?,
        extra: opt_i64(&head, 6, "seventh token")?,
        ax: scale(0),
        fy: scale(1),
        tstart: scale(2),
        tstop: scale(3),
    })
}

/// Decodes `/INIVEL/*` initial velocity blocks.
fn decode_inivel(fields: &[Vec<String>]) -> Result<InivelCard, String> {
    let values = split_numbers(single_row(fields, "initial velocity")?);
    if values.len() != 5 {
        return Err("initial velocity row needs 5 tokens".to_string());
    }
    Ok(InivelCard {
        v: [
            need_f64(&values, 0, "vx")?,
            need_f64(&values, 1, "vy")?,
            need_f64(&values, 2, "vz")?,
        ],
        grnod: need_i64(&values, 3, "group id")?,
        skew: need_i64(&values, 4, "skew id")?,
    })
}

/// Decodes `/PLOAD` pressure loads.
fn decode_pload(fields: &[Vec<String>]) -> Result<PloadCard, String> {
    let values = split_numbers(single_row(fields, "pressure load")?);
    if values.len() < 2 || values.len() > 7 {
        return Err("pressure load row needs 2 to 7 tokens".to_string());
    }
    Ok(PloadCard {
        surf: need_i64(&values, 0, "surface id")?,
        funct: need_i64(&values, 1, "function id")?,
        sens: opt_i64(&values, 2, "sensor id")?,
        pinch: opt_i64(&values, 3, "pinch flag")?,
        del: opt_i64(&values, 4, "deletion flag")?,
        ax: opt_f64(&values, 5, "abscissa scale")?,
        fy: opt_f64(&values, 6, "ordinate scale")?,
    })
}

/// Decodes `/ACCEL` accelerometer definitions.
fn decode_accel(fields: &[Vec<String>]) -> Result<AccelCard, String> {
    let values = split_numbers(single_row(fields, "accelerometer")?);
    if values.len() != 3 {
        return Err("accelerometer row needs 3 tokens".to_string());
    }
    Ok(AccelCard {
        node: need_i64(&values, 0, "node id")?,
        skew: need_i64(&values, 1, "skew id")?,
        fcut: need_f64(&values, 2, "cutoff")?,
    })
}

/// Decodes `/VEL/*` engine entity lists.
fn decode_vel(fields: &[Vec<String>]) -> Result<VelCard, String> {
    let mut ids = Vec::new();
    for (row_no, row) in fields.iter().enumerate() {
        for token in split_numbers(row) {
            ids.push(
                parse_i64(&token)
                    .ok_or_else(|| format!("velocity row {} needs integer tokens", row_no + 1))?,
            );
        }
    }
    Ok(VelCard { ids })
}

/// Decodes `/INTER/*` leading identifier rows.
fn decode_inter(kind: CardKind, fields: &[Vec<String>]) -> Result<InterCard, String> {
    let itype = match kind {
        CardKind::InterType2 => 2,
        CardKind::InterType3 => 3,
        CardKind::InterType5 => 5,
        CardKind::InterType7 => 7,
        CardKind::InterType11 => 11,
        CardKind::InterType12 => 12,
        CardKind::InterType19 => 19,
        CardKind::InterType22 => 22,
        CardKind::InterType25 => 25,
        _ => return Err("not an interface family".to_string()),
    };
    let Some(head) = fields.first() else {
        return Err("interface needs a leading data row".to_string());
    };
    let mut ids = Vec::with_capacity(head.len());
    for (index, token) in split_numbers(head).iter().enumerate() {
        ids.push(
            parse_i64(token)
                .ok_or_else(|| format!("interface leading token {} needs an integer", index + 1))?,
        );
    }
    Ok(InterCard { itype, head: ids })
}

/// Decodes `/RWALL/*` rigid wall header rows.
fn decode_rwall(fields: &[Vec<String>]) -> Result<RwallCard, String> {
    let Some(row) = fields.first() else {
        return Err("wall needs a header data row".to_string());
    };
    let node = match row.first().map(String::as_str) {
        None | Some("") => 0,
        Some(token) => parse_i64(token).ok_or_else(|| "wall needs an integer node".to_string())?,
    };
    let slide = match row.get(1).map(String::as_str) {
        None | Some("") => return Err("wall needs a slide flag".to_string()),
        Some(token) => {
            parse_i64(token).ok_or_else(|| "wall needs an integer slide flag".to_string())?
        }
    };
    let mut grnds = Vec::new();
    for (index, field) in row.iter().skip(2).enumerate() {
        if field.is_empty() {
            return Err("wall group gap before trailing groups".to_string());
        }
        grnds.push(
            parse_i64(field).ok_or_else(|| format!("wall group {} needs an integer", index + 1))?,
        );
    }
    Ok(RwallCard { node, slide, grnds })
}

/// Decodes `/TH/*` variable names with entity entries.
fn decode_th(fields: &[Vec<String>]) -> Result<ThCard, String> {
    let Some(head) = fields.first() else {
        return Ok(ThCard {
            vars: Vec::new(),
            ents: Vec::new(),
        });
    };
    let numeric_first = head.first().is_some_and(|t| parse_f64(t).is_some());
    let (vars, entities) = if numeric_first {
        (Vec::new(), fields)
    } else {
        (head.clone(), &fields[1..])
    };
    let mut ents = Vec::with_capacity(entities.len());
    for (row_no, row) in entities.iter().enumerate() {
        let Some(first) = row.first() else {
            return Err(format!("history row {} needs an entity id", row_no + 1));
        };
        let id = parse_i64(first)
            .ok_or_else(|| format!("history row {} needs an integer id", row_no + 1))?;
        let skew = row.get(1).and_then(|t| parse_i64(t));
        let name = if skew.is_some() {
            row.get(2..).unwrap_or(&[]).join(" ")
        } else {
            row.get(1..).unwrap_or(&[]).join(" ")
        };
        ents.push(ThEnt { id, skew, name });
    }
    Ok(ThCard { vars, ents })
}

/// Decodes `/ANIM/*` and `/OUTP/*` output flags.
fn decode_anim(key: &str, fields: &[Vec<String>]) -> Result<AnimCard, String> {
    let mut params = Vec::new();
    for (row_no, row) in fields.iter().enumerate() {
        for (col_no, token) in split_numbers(row).iter().enumerate() {
            params.push(parse_f64(token).ok_or_else(|| {
                format!(
                    "output row {} token {} needs a real",
                    row_no + 1,
                    col_no + 1
                )
            })?);
        }
    }
    Ok(AnimCard {
        key: key.trim_start_matches('/').to_string(),
        params,
    })
}

/// Decodes `/PRINT/*` printout markers.
fn decode_print(fields: &[Vec<String>]) -> Result<PrintCard, String> {
    if fields.is_empty() {
        Ok(PrintCard)
    } else {
        Err("print block needs no data rows".to_string())
    }
}

/// Decodes `/FUNCT` abscissa and value pairs.
fn decode_funct(fields: &[Vec<String>]) -> Result<FunctCard, String> {
    let mut points = Vec::with_capacity(fields.len());
    for (row_no, row) in fields.iter().enumerate() {
        let values = split_numbers(row);
        if values.len() != 2 {
            return Err(format!("function row {} needs 2 tokens", row_no + 1));
        }
        points.push([
            need_f64(&values, 0, "abscissa")?,
            need_f64(&values, 1, "value")?,
        ]);
    }
    Ok(FunctCard { points })
}

/// Decodes `/SENSOR/*` trigger delays with dependencies.
fn decode_sensor(kind: CardKind, fields: &[Vec<String>]) -> Result<SensorCard, String> {
    let rows = match kind {
        CardKind::SensorTime => 1,
        CardKind::SensorSens => 2,
        _ => return Err("not a sensor family".to_string()),
    };
    if fields.len() != rows {
        return Err(format!("sensor needs exactly {rows} data rows"));
    }
    let Some(first_row) = fields.first() else {
        return Err("sensor needs a delay row".to_string());
    };
    let delay_row = split_numbers(first_row);
    if delay_row.len() != 1 {
        return Err("sensor delay row needs 1 token".to_string());
    }
    let delay = need_f64(&delay_row, 0, "delay")?;
    let mut deps = Vec::new();
    if let Some(row) = fields.get(1) {
        for token in split_numbers(row) {
            deps.push(
                parse_i64(&token)
                    .ok_or_else(|| "sensor dependency needs an integer".to_string())?,
            );
        }
    }
    Ok(SensorCard { delay, deps })
}

/// Decodes `/BEGIN` versions with unit systems.
fn decode_begin(fields: &[Vec<String>]) -> Result<BeginCard, String> {
    if fields.is_empty() || fields.len() > 3 {
        return Err("begin block needs 1 to 3 data rows".to_string());
    }
    let Some(version_row) = fields.first() else {
        return Err("begin block needs a version row".to_string());
    };
    let version = split_numbers(version_row);
    let Some(first) = version.first() else {
        return Err("begin block needs a version token".to_string());
    };
    let version_id =
        parse_i64(first).ok_or_else(|| "begin version needs an integer".to_string())?;
    let mut flags = Vec::with_capacity(version.len().saturating_sub(1));
    for token in version.iter().skip(1) {
        flags.push(parse_i64(token).ok_or_else(|| "begin flag needs an integer".to_string())?);
    }
    let units = |index: usize, what: &str| match fields.get(index) {
        None => Ok(None),
        Some(row) => match row.as_slice() {
            [mass, length, time] => Ok(Some([mass.clone(), length.clone(), time.clone()])),
            _ => Err(format!("begin {what} units need 3 tokens")),
        },
    };
    Ok(BeginCard {
        version: version_id,
        flags,
        input: units(1, "input")?,
        work: units(2, "work")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses text or panics with the failure message.
    fn parse(text: &str) -> RadDeck {
        RadDeck::parse_str(text).expect("deck parses")
    }

    /// Fetches the single card of a one-card deck.
    fn sole(text: &str) -> DeckCard {
        let deck = parse(text);
        assert_eq!(deck.cards.len(), 1);
        deck.cards.into_iter().next().expect("one card")
    }

    #[test]
    fn starter_snippet_decodes() {
        let deck = parse(
            "#RADIOSS STARTER\n/BEGIN\nROOT\n2019 0\nkg m s\nkg m s\n\
             /NODE\n1 0.0 0.0 1.D0\n/PART/1\nsteel\n1 1 0\n\
             /MAT/LAW1/1\nsteel\n7.8e-6\n/BCS/1\nfix\n   111 111         0         2\n/END\n",
        );
        assert_eq!(deck.cards.len(), 6);
        assert_eq!(deck.cards[0].title, "ROOT");
        assert!(matches!(
            deck.cards[0].typed,
            Some(TypedCard::Begin(BeginCard { version: 2019, .. }))
        ));
        assert!(matches!(deck.cards[1].typed, Some(TypedCard::Nodes(_))));
        let nodes = match &deck.cards[1].typed {
            Some(TypedCard::Nodes(nodes)) => nodes,
            other => panic!("nodes typed, got {other:?}"),
        };
        assert_eq!(nodes.len(), 1);
        assert_eq!((nodes[0].id, nodes[0].z), (1, 1.0));
        assert!(matches!(deck.cards[2].typed, Some(TypedCard::Part(_))));
        assert!(matches!(deck.cards[3].typed, Some(TypedCard::Mat(_))));
        assert!(matches!(deck.cards[4].typed, Some(TypedCard::Bcs(_))));
        assert_eq!(deck.cards[5].kind, CardKind::End);
        assert_eq!(deck.round_trip().expect("round trip"), deck);
    }

    #[test]
    fn engine_snippet_decodes() {
        let deck = parse(
            "/RUN/CT3AV4/1\n2000\n/ANIM/DT\n0.0 200.0\n/ANIM/GZIP\n\
             /PRINT/-100/55\n/DT\n0.5 0.0\n/TFILE/4\n20.0\n",
        );
        assert_eq!(deck.cards.len(), 6);
        assert_eq!(deck.cards[0].id, Some(1));
        assert!(matches!(
            deck.cards[1].typed,
            Some(TypedCard::Anim(AnimCard { .. }))
        ));
        if let Some(TypedCard::Anim(anim)) = &deck.cards[1].typed {
            assert_eq!(anim.params, vec![0.0, 200.0]);
        }
        assert!(matches!(deck.cards[3].typed, Some(TypedCard::Print(_))));
        assert_eq!(deck.round_trip().expect("round trip"), deck);
    }

    #[test]
    fn title_rules() {
        let part = sole("/PART/3\n1\n1 1\n");
        assert_eq!(part.title, "1");
        let th = sole("/TH/RWALL/3\n1\nDEF\n5 0\n");
        assert_eq!(th.title, "1");
        let skew = sole("/SKEW/MOV/1\n11\n3643 3637 3655\n");
        assert_eq!(skew.title, "11");
        assert!(skew.typed.is_none());
        let node = sole("/NODE\n1 0 0 0\n");
        assert_eq!(node.title, "");
        let damp_starter = sole("/DAMP/1\ndamping\n0.01 0.02\n");
        assert_eq!(damp_starter.title, "damping");
        let damp_engine = sole("/DAMP\n40 0\n");
        assert_eq!(damp_engine.title, "");
        let title = sole("/TITLE\n12345\n");
        assert_eq!(title.title, "12345");
    }

    #[test]
    fn comments_and_debris_skip() {
        let deck = parse(
            "# Copyright (C) 2026\n#---1----|----2----|\n/TITLE\nbeam\n\
             #/UNIT/LENGTH/m\nCopyright>        OpenRadioss\n$ converter note\n/NODE\n1 0 0 0\n/END\n",
        );
        assert_eq!(deck.cards.len(), 3);
        assert_eq!(deck.cards[0].title, "beam");
        assert!(deck.cards[0].fields.is_empty());
        assert_eq!(deck.round_trip().expect("round trip"), deck);
    }

    #[test]
    fn funct_python_verbatim() {
        let deck = parse(
            "/FUNCT_PYTHON/3\ndef myfun2(x):\n    # comment\n    return x\n\n/FUNCT/2\nsize\n0.0 0.0\n",
        );
        assert_eq!(deck.cards.len(), 2);
        let python = &deck.cards[0];
        assert_eq!(python.title, "def myfun2(x):");
        assert_eq!(
            python.fields,
            vec![
                vec!["    # comment".to_string()],
                vec!["    return x".to_string()],
                vec![String::new()],
            ]
        );
        assert_eq!(deck.round_trip().expect("round trip"), deck);
    }

    #[test]
    fn funct_dot_fusion_repairs() {
        let row = "              0.01253.85482000000000E-04";
        assert_eq!(row.len(), 40);
        let card = sole(&format!("/FUNCT/1\ncurve1\n0.0 0.0\n{row}\n"));
        match &card.typed {
            Some(TypedCard::Funct(funct)) => {
                assert_eq!(funct.points.len(), 2);
                assert!((funct.points[1][0] - 0.0125).abs() < 1e-12);
                assert!((funct.points[1][1] - 3.85482e-4).abs() < 1e-12);
            }
            other => panic!("funct typed, got {other:?}"),
        }
        let deck = parse(&format!("/FUNCT/1\ncurve1\n0.0 0.0\n{row}\n"));
        assert_eq!(deck.round_trip().expect("round trip"), deck);
    }

    #[test]
    fn impdisp_scale_fusion_repairs() {
        let row =
            "1.00000000000000    1.00000000000000                     0.01.00000000000000E+31";
        assert_eq!(row.len(), 80);
        let card = sole(&format!("/IMPDISP/1\nload\n1 X 0 0 2\n{row}\n"));
        match &card.typed {
            Some(TypedCard::Load(LoadCard::Impdisp(imp))) => {
                assert_eq!(imp.tstop, Some(1.0e31));
                assert_eq!(imp.ax, Some(1.0));
            }
            other => panic!("impdisp typed, got {other:?}"),
        }
    }

    #[test]
    fn node_sign_fusion_splits() {
        let row = "         9               160.0-1.5258789062500E-05                 0.0";
        let card = sole(&format!("/NODE\n{row}\n"));
        match &card.typed {
            Some(TypedCard::Nodes(nodes)) => {
                assert_eq!(nodes.len(), 1);
                assert_eq!(nodes[0].id, 9);
                assert!((nodes[0].x - 160.0).abs() < 1e-9);
                assert!((nodes[0].y + 1.52587890625e-5).abs() < 1e-18);
            }
            other => panic!("nodes typed, got {other:?}"),
        }
        // The raw fused tokens survive in fields for faithful consumers.
        assert_eq!(card.fields[0].len(), 3);
    }

    #[test]
    fn node_dot_fusion_repairs() {
        let row = "        35    499.999999999997    177.6315789473681.52587890625000E-05";
        assert_eq!(row.len(), 70);
        let card = sole(&format!("/NODE\n{row}\n"));
        match &card.typed {
            Some(TypedCard::Nodes(nodes)) => {
                assert_eq!(nodes.len(), 1);
                assert!((nodes[0].x - 499.999999999997).abs() < 1e-9);
                assert!((nodes[0].y - 177.631578947368).abs() < 1e-9);
                assert!((nodes[0].z - 1.52587890625e-5).abs() < 1e-18);
            }
            other => panic!("nodes typed, got {other:?}"),
        }
    }

    #[test]
    fn bcs_columns() {
        let full = format!("{:>10}{:>10}{:>10}", "111 111", "0", "2");
        let card = sole(&format!("/BCS/1\nfix\n{full}\n"));
        match &card.typed {
            Some(TypedCard::Bcs(bcs)) => {
                assert_eq!((bcs.tra.as_str(), bcs.rot.as_str()), ("111", "111"));
                assert_eq!((bcs.skew, bcs.grnod), (Some(0), Some(2)));
            }
            other => panic!("bcs typed, got {other:?}"),
        }
        let skew_gap = format!("{:>10}{:>10}{:>10}", "011 111", "", "1");
        let card = sole(&format!("/BCS/4\nguide\n{skew_gap}\n"));
        match &card.typed {
            Some(TypedCard::Bcs(bcs)) => {
                assert_eq!((bcs.skew, bcs.grnod), (None, Some(1)));
            }
            other => panic!("bcs typed, got {other:?}"),
        }
        let deck = parse(&format!("/BCS/1\nfix\n{full}\n"));
        assert_eq!(deck.round_trip().expect("round trip"), deck);
    }

    #[test]
    fn rwall_columns() {
        let fixed = format!("{:>10}{:>10}{:>10}{:>10}", "", "1", "0", "0");
        let card = sole(&format!(
            "/RWALL/PLANE/501\nrw y=0\n{fixed}\n1 0 0\n0 0 0\n1 0 0\n"
        ));
        match &card.typed {
            Some(TypedCard::Rwall(wall)) => {
                assert_eq!((wall.node, wall.slide), (0, 1));
                assert_eq!(wall.grnds, vec![0, 0]);
            }
            other => panic!("rwall typed, got {other:?}"),
        }
        let deck = parse(&format!(
            "/RWALL/PLANE/501\nrw y=0\n{fixed}\n1 0 0\n0 0 0\n1 0 0\n"
        ));
        assert_eq!(deck.round_trip().expect("round trip"), deck);
    }

    #[test]
    fn shell_extras() {
        let thick = format!("{:>10}{:>10}{:>10}{:>10}{:>10}{:>50}", 1, 2, 3, 4, 5, "2.0");
        let card = sole(&format!("/SHELL/1\n{thick}\n"));
        match &card.typed {
            Some(TypedCard::Elems(block)) => {
                assert_eq!(block.elems[0].thick, Some(2.0));
                assert_eq!(block.elems[0].angle, None);
            }
            other => panic!("shell typed, got {other:?}"),
        }
        let both = format!(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>30}{:>20}",
            1, 2, 3, 4, 5, "0.5", "2.0"
        );
        let card = sole(&format!("/SHELL/1\n{both}\n"));
        match &card.typed {
            Some(TypedCard::Elems(block)) => {
                assert_eq!(block.elems[0].angle, Some(0.5));
                assert_eq!(block.elems[0].thick, Some(2.0));
            }
            other => panic!("shell typed, got {other:?}"),
        }
        let mid = format!("{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}", 1, 2, 3, 4, 5, "0.5");
        assert!(RadDeck::parse_str(&format!("/SHELL/1\n{mid}\n")).is_err());
        let deck = parse(&format!("/SHELL/1\n{thick}\n{both}\n1 2 3 4 5\n"));
        assert_eq!(deck.round_trip().expect("round trip"), deck);
        let fused_id =
            sole("/NODE\n      10621.52587890625000E-05                 0.0    179.999992370605\n");
        match &fused_id.typed {
            Some(TypedCard::Nodes(nodes)) => {
                assert_eq!(nodes[0].id, 1062);
                assert!((nodes[0].x - 1.52587890625e-5).abs() < 1e-18);
            }
            other => panic!("nodes typed, got {other:?}"),
        }
    }

    #[test]
    fn th_vars_and_entities() {
        let card = sole("/TH/NODE/2\nnodes\nDEF\n210 0\n954 Cuivre_Loi49\n");
        match &card.typed {
            Some(TypedCard::Th(th)) => {
                assert_eq!(th.vars, vec!["DEF".to_string()]);
                assert_eq!(th.ents.len(), 2);
                assert_eq!(th.ents[0].id, 210);
                assert_eq!(th.ents[0].skew, Some(0));
                assert_eq!(th.ents[1].name, "Cuivre_Loi49");
                assert_eq!(th.ents[1].skew, None);
            }
            other => panic!("th typed, got {other:?}"),
        }
        let bare = sole("/TH/PART/9\np\n1\n");
        match &bare.typed {
            Some(TypedCard::Th(th)) => {
                assert!(th.vars.is_empty());
                assert_eq!(th.ents.len(), 1);
            }
            other => panic!("th typed, got {other:?}"),
        }
        let deck = parse("/TH/NODE/2\nnodes\nDEF\n210 0\n954 Cuivre_Loi49\n");
        assert_eq!(deck.round_trip().expect("round trip"), deck);
    }

    #[test]
    fn loads_decode() {
        let grav = sole("/GRAV/1\ng\n1 Z 0 0 3 1.0 1.0\n");
        assert!(matches!(
            grav.typed,
            Some(TypedCard::Load(LoadCard::Grav(_)))
        ));
        let imp = sole("/IMPVEL/1\nv\n1 X 0 0 2 0 0\n1.0 2.0 0.0\n");
        match &imp.typed {
            Some(TypedCard::Load(LoadCard::Impvel(imp))) => {
                assert_eq!(imp.extra, Some(0));
                assert_eq!(imp.tstop, None);
                assert_eq!(imp.fy, Some(2.0));
            }
            other => panic!("impvel typed, got {other:?}"),
        }
        let bare_imp = sole("/IMPVEL/1\nv\n1 Z 0 0 3\n");
        assert!(matches!(
            bare_imp.typed,
            Some(TypedCard::Load(LoadCard::Impvel(_)))
        ));
        let inivel = sole("/INIVEL/TRA/1\nin\n0 0 -1.5 2 0\n");
        assert!(matches!(
            inivel.typed,
            Some(TypedCard::Load(LoadCard::Inivel(_)))
        ));
        let pload = sole("/PLOAD/1\np\n999 999\n");
        match &pload.typed {
            Some(TypedCard::Load(LoadCard::Pload(pload))) => {
                assert_eq!(pload.sens, None);
            }
            other => panic!("pload typed, got {other:?}"),
        }
        let accel = sole("/ACCEL/1\na\n208 0 1.65\n");
        assert!(matches!(
            accel.typed,
            Some(TypedCard::Load(LoadCard::Accel(_)))
        ));
        let vel = sole("/VEL/TRA/XYZ/1\nv\n1 2 3\n");
        assert!(matches!(vel.typed, Some(TypedCard::Load(LoadCard::Vel(_)))));
    }

    #[test]
    fn interface_sensor_begin_decode() {
        let inter = sole("/INTER/TYPE7/1\ni\n1 2 0 0 0 0 0 0 0\n1.0 2.0\n");
        match &inter.typed {
            Some(TypedCard::Inter(inter)) => {
                assert_eq!(inter.itype, 7);
                assert_eq!(inter.head.len(), 9);
            }
            other => panic!("inter typed, got {other:?}"),
        }
        let time = sole("/SENSOR/TIME/1\ns\n0.5\n");
        match &time.typed {
            Some(TypedCard::Sensor(sensor)) => {
                assert_eq!(sensor.delay, 0.5);
                assert!(sensor.deps.is_empty());
            }
            other => panic!("sensor typed, got {other:?}"),
        }
        let sens = sole("/SENSOR/SENS/1\ns\n0.5\n2 0\n");
        match &sens.typed {
            Some(TypedCard::Sensor(sensor)) => {
                assert_eq!(sensor.deps, vec![2, 0]);
            }
            other => panic!("sensor typed, got {other:?}"),
        }
        let begin = sole("/BEGIN\nROOT\n2019 0\nkg m s\nkg m s\n");
        match &begin.typed {
            Some(TypedCard::Begin(begin)) => {
                assert_eq!(begin.version, 2019);
                assert!(begin.work.is_some());
            }
            other => panic!("begin typed, got {other:?}"),
        }
        let short = sole("/BEGIN\nROOT\n90 0\n");
        match &short.typed {
            Some(TypedCard::Begin(begin)) => {
                assert_eq!(begin.version, 90);
                assert!(begin.input.is_none());
            }
            other => panic!("begin typed, got {other:?}"),
        }
    }

    #[test]
    fn part_mat_prop_shapes() {
        let bare = sole("/PART/1\np\n5 601\n");
        match &bare.typed {
            Some(TypedCard::Part(part)) => {
                assert_eq!(part.subset, None);
            }
            other => panic!("part typed, got {other:?}"),
        }
        let thick = sole("/PART/1\np\n5 601 0 2.0\n");
        match &thick.typed {
            Some(TypedCard::Part(part)) => {
                assert_eq!(part.thick, Some(2.0));
            }
            other => panic!("part typed, got {other:?}"),
        }
        let mat = sole("/MAT/PLAS_JOHNS/4\nSPECT\n7.8e-6\n");
        match &mat.typed {
            Some(TypedCard::Mat(mat)) => {
                assert_eq!(mat.law, "PLAS_JOHNS");
                assert!(mat.rho_init.is_some());
            }
            other => panic!("mat typed, got {other:?}"),
        }
        let law51 = sole("/MAT/LAW51/1\nm\n1 0\n");
        match &law51.typed {
            Some(TypedCard::Mat(mat)) => assert_eq!(mat.rho_init, None),
            other => panic!("mat typed, got {other:?}"),
        }
        let prop = sole("/PROP/SHELL/1\nShell\n1.0\n");
        assert!(matches!(prop.typed, Some(TypedCard::Prop(_))));
    }

    #[test]
    fn failures_are_loud() {
        assert!(matches!(
            RadDeck::parse_str("/NOPE/1\n"),
            Err(ParseError::UnknownHeader { .. })
        ));
        assert!(matches!(
            RadDeck::parse_str("hello\n/TITLE\nt\n"),
            Err(ParseError::DanglingData { .. })
        ));
        assert!(matches!(
            RadDeck::parse_str("/TITLE\nt\n#include x.inc\n"),
            Err(ParseError::Include { .. })
        ));
        assert!(matches!(
            RadDeck::parse_str("/PART/1\np\n"),
            Err(ParseError::Decode { .. })
        ));
        assert!(matches!(
            RadDeck::parse_str("/FUNCT/1\nc\na b\n"),
            Err(ParseError::Decode { .. })
        ));
        assert!(matches!(
            RadDeck::parse_str("/TH/NODE/1\nn\nDEF\nXX\n"),
            Err(ParseError::Decode { .. })
        ));
        assert!(matches!(
            RadDeck::parse_str("/PRINT\n1.0\n"),
            Err(ParseError::Decode { .. })
        ));
        assert!(matches!(
            RadDeck::parse_file("/nonexistent/deck.rad"),
            Err(ParseError::Io { .. })
        ));
    }

    #[test]
    fn include_round_trip_through_files() {
        let root = std::env::temp_dir().join(format!("rad_parse_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("scratch dir");
        std::fs::write(root.join("part.inc"), "/NODE\n7 1.0 2.0 3.0\n").expect("inc");
        std::fs::write(
            root.join("deck.rad"),
            "/TITLE\nd\n#include part.inc\n/END\n",
        )
        .expect("deck");
        let deck = RadDeck::parse_file(root.join("deck.rad")).expect("deck parses");
        assert_eq!(deck.cards.len(), 3);
        assert_eq!(deck.cards[1].kind, CardKind::Node);
        assert_eq!(deck.round_trip().expect("round trip"), deck);
        let again = RadDeck::parse_file(root.join("deck.rad")).expect("deck parses");
        assert_eq!(again.cards.len(), 3);
        std::fs::write(root.join("bad.rad"), "/TITLE\nd\n#include nope.inc\n").expect("bad");
        assert!(matches!(
            RadDeck::parse_file(root.join("bad.rad")),
            Err(ParseError::Include { .. })
        ));
        std::fs::write(root.join("a.inc"), "#include b.inc\n").expect("a");
        std::fs::write(root.join("b.inc"), "#include a.inc\n").expect("b");
        std::fs::write(root.join("cyc.rad"), "/TITLE\nd\n#include a.inc\n").expect("cyc");
        assert!(matches!(
            RadDeck::parse_file(root.join("cyc.rad")),
            Err(ParseError::Include { .. })
        ));
        let _ = std::fs::remove_dir_all(&root);
    }
}
