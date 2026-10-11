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
//! `myfun2(x)`); the callback namespace above bound to live engine
//! state (sensors, nodal and element fields at the current cycle);
//! deterministic results (same abscissa and state give bitwise-equal
//! returns). None of these exist in-tree: there is no Python runtime
//! dependency, no engine-state binding layer, and no approved sandbox
//! for executing deck-supplied code.
//!
//! # Verdict (P5C3)
//!
//! Dated deferral. Full evaluation is deferred to [`DEFER_UNTIL`]:
//! the missing piece is [`DEFER_MISSING`], and the replay target it
//! would hit is [`DEFER_TARGET`]. Rationale: a native subset evaluator
//! cannot reproduce Python semantics and would silently diverge on
//! decks outside its grammar, while embedding CPython needs a new
//! dependency plus a sandbox decision this lane cannot take (no
//! `Cargo.toml` edits, offline gate). What ships instead is real and
//! tested: block parsing, call/accessor extraction, purity tagging,
//! and a transcribed closed form of the deck smoothstep for
//! cross-checks (not evaluation).
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
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};

/// Deferral revisit date (`YYYY-MM-DD`).
///
/// Full `/FUNCT_PYTHON` evaluation lands on or before this date; the
/// verdict above names the missing piece and replay target.
pub const DEFER_UNTIL: &str = "2026-11-30";

/// Missing piece blocking evaluation.
pub const DEFER_MISSING: &str = "approved Python execution sandbox (embedded CPython vs restricted evaluator) with live engine-state accessor bindings (TIME/DT, nodal and element sensor callbacks)";

/// Replay target the deferred evaluation would hit.
pub const DEFER_TARGET: &str = "PYTHON/FUNCT_PYTHON golden TH series (test_torsion_orthstrain_shell) through IMPDISP-driven motion";

/// Private failure modes behind [`FunctPythonError`].
#[derive(Debug)]
enum ErrorKind {
    /// Block without its header identifier.
    MissingId,
    /// Block without a `def` line.
    MissingDef,
    /// Malformed `def` line.
    BadDef { line: String },
    /// Block with no body lines.
    EmptyBody,
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
}
