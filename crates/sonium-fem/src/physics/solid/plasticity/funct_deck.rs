//! Parses Radioss function and table deck cards.
//!
//! Reads `/FUNCT`, `/TABLE/0`, and `/TABLE/1` deck text into the frozen
//! [`Funct1D`](super::funct::Funct1D) and [`Table2D`](super::funct::Table2D)
//! types without modifying them. Card layouts follow Reference Guide
//! `/FUNCT/fct_ID` (p.2229), `/TABLE/0/table_ID` (p.2247), and
//! `/TABLE/1/table_ID` (p.2251). Only two-dimensional tables convert to
//! `Table2D`; higher dimensions parse but report unsupported conversion.
//! Evaluation clamps out-of-range inputs per the frozen types, while the
//! manual extrapolates `/FUNCT` linearly; keep curve ranges wide enough.
//!
//! # Examples
//!
//! ```rust
//! use sonium_fem::physics::solid::plasticity::funct_deck::parse_funct_deck;
//!
//! let deck = "/FUNCT/7\ntitle\n0.0 1.0\n1.0 3.0\n/END\n";
//! let table = parse_funct_deck(deck).unwrap();
//! assert_eq!(table.get(7).unwrap().eval(0.5), 2.0);
//! ```
//!
//! # Errors
//!
//! Every parser returns [`DeckError`] naming the card and line that
//! failed, wrapping frozen-constructor rejections verbatim.
//!
//! # Panics
//!
//! No panics on any input; malformed text yields [`DeckError`].
//!
//! # Safety
//!
//! Pure safe Rust with no unsafe code.

// Rust guideline compliant 2026-02-21

use super::funct::{Funct1D, Table2D};
use std::collections::BTreeMap;

/// Failure to parse one deck card.
#[derive(Debug, Clone, PartialEq)]
pub struct DeckError {
    /// Human-readable cause with card identity and line number.
    pub message: String,
}

impl DeckError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DeckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "deck error: {}", self.message)
    }
}

impl std::error::Error for DeckError {}

/// Parsed `/FUNCT` cards indexed by function id.
#[derive(Debug, Clone, Default)]
pub struct FunctDeck {
    functs: BTreeMap<u32, Funct1D>,
    points: BTreeMap<u32, Vec<[f64; 2]>>,
}

impl FunctDeck {
    /// Looks up one function by id.
    pub fn get(&self, id: u32) -> Option<&Funct1D> {
        self.functs.get(&id)
    }

    /// Returns the raw nodes of one function by id.
    ///
    /// Retained because the frozen type exposes evaluation only;
    /// table resampling builds union grids from these nodes.
    pub fn nodes(&self, id: u32) -> Option<&[[f64; 2]]> {
        self.points.get(&id).map(Vec::as_slice)
    }

    /// Iterates `(id, function)` pairs in id order.
    pub fn iter(&self) -> impl Iterator<Item = (&u32, &Funct1D)> {
        self.functs.iter()
    }

    /// Counts parsed functions.
    pub fn len(&self) -> usize {
        self.functs.len()
    }

    /// Reports whether no function parsed.
    pub fn is_empty(&self) -> bool {
        self.functs.is_empty()
    }
}

/// One raw card: header line plus body lines with numbers.
struct Card {
    header: String,
    line_no: usize,
    body: Vec<(usize, String)>,
}

/// Splits deck text into slash cards.
///
/// A card starts at a line whose first non-blank character is `/` and
/// runs to the next such line. `/END` and `#ENDDATA` end the deck.
/// Comment lines starting with `#` are dropped; inline `#` tails are
/// cut because `#` never occurs inside a number.
fn split_cards(text: &str) -> Vec<Card> {
    let mut cards = Vec::new();
    let mut current: Option<Card> = None;
    for (index, raw) in text.lines().enumerate() {
        let line_no = index + 1;
        let cut = raw.find('#').map_or(raw, |at| &raw[..at]);
        let trimmed = cut.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('/') {
            if let Some(card) = current.take() {
                cards.push(card);
            }
            let head = trimmed.to_ascii_uppercase();
            if head == "/END" || head == "#ENDDATA" {
                break;
            }
            current = Some(Card {
                header: trimmed.to_string(),
                line_no,
                body: Vec::new(),
            });
        } else if let Some(card) = current.as_mut() {
            card.body.push((line_no, trimmed.to_string()));
        }
    }
    if let Some(card) = current.take() {
        cards.push(card);
    }
    cards
}

/// Parses one card header of the form `/NAME/id[/sub]`.
fn header_parts(header: &str) -> Vec<String> {
    header
        .split('/')
        .filter(|part| !part.is_empty())
        .map(|part| part.to_ascii_uppercase())
        .collect()
}

/// Parses every whitespace-separated float on one line.
fn parse_floats(line: &str, line_no: usize, what: &str) -> Result<Vec<f64>, DeckError> {
    let mut out = Vec::new();
    for token in line.split_whitespace() {
        match token.parse::<f64>() {
            Ok(value) => out.push(value),
            Err(_) => {
                return Err(DeckError::new(format!(
                    "{what} line {line_no}: '{token}' is not a number"
                )));
            }
        }
    }
    Ok(out)
}

/// Parses `/FUNCT` cards from deck text.
///
/// Skips the first body line unless it holds only floats, then reads
/// `(x, y)` pairs in free format (one or more pairs per line).
/// Duplicate ids keep the last card. Cards of other types are ignored.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct_deck::parse_funct_deck;
///
/// let deck = "/FUNCT/500006\nshear yield\n-1000.0 5.86e-4\n1000.0 5.86e-4\n";
/// let table = parse_funct_deck(deck).unwrap();
/// assert_eq!(table.get(500006).unwrap().eval(0.75), 5.86e-4);
/// ```
///
/// # Errors
///
/// Returns [`DeckError`] for a bad card id, an odd float count, or a
/// frozen [`Funct1D`] rejection (fewer than two nodes, non-finite
/// values, or non-increasing abscissae).
pub fn parse_funct_deck(text: &str) -> Result<FunctDeck, DeckError> {
    let mut deck = FunctDeck::default();
    for card in split_cards(text) {
        let parts = header_parts(&card.header);
        if parts.first().map(String::as_str) != Some("FUNCT") || parts.len() != 2 {
            continue;
        }
        let id: u32 = parts[1].parse().map_err(|_| {
            DeckError::new(format!(
                "card {} line {}: bad function id '{}'",
                card.header, card.line_no, parts[1]
            ))
        })?;
        let mut floats = Vec::new();
        let mut lines = card.body.iter().peekable();
        let skip_title = lines
            .peek()
            .is_some_and(|(_, first)| !first.split_whitespace().all(|t| t.parse::<f64>().is_ok()));
        if skip_title {
            lines.next();
        }
        for (line_no, line) in lines {
            floats.extend(parse_floats(line, *line_no, &card.header)?);
        }
        if floats.len() % 2 != 0 {
            return Err(DeckError::new(format!(
                "card {}: odd float count {}",
                card.header,
                floats.len()
            )));
        }
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for pair in floats.as_chunks::<2>().0 {
            x.push(pair[0]);
            y.push(pair[1]);
        }
        let funct = Funct1D::new(x, y)
            .map_err(|cause| DeckError::new(format!("card {}: {cause}", card.header)))?;
        let nodes = floats
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| [pair[0], pair[1]])
            .collect();
        deck.functs.insert(id, funct);
        deck.points.insert(id, nodes);
    }
    Ok(deck)
}

/// Parsed `/TABLE/0` data-point card.
#[derive(Debug, Clone)]
pub struct Table0 {
    /// Table identifier from the card header.
    pub id: u32,
    /// Entry counts per dimension (1 to 4 used).
    pub shape: Vec<usize>,
    /// Grid values per dimension in card order.
    pub grids: Vec<Vec<f64>>,
    /// Table values, first entry varying fastest.
    pub values: Vec<f64>,
}

/// Parses `/TABLE/0` data-point cards from deck text.
///
/// Layout per card: title line, `dimension n1 n2 n3 n4`, the `n1`
/// first-entry values, the `n2` second-entry values, and so on, then
/// the `n1*n2*...` table values with the first entry varying fastest.
/// Cards of other types are ignored.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct_deck::parse_table0_deck;
///
/// let deck = "/TABLE/0/9\ntitle\n2 2 2 0 0\n0.0 1.0\n0.0 10.0\n1.0 2.0 3.0 4.0\n";
/// let tables = parse_table0_deck(deck).unwrap();
/// assert_eq!(tables[0].values, vec![1.0, 2.0, 3.0, 4.0]);
/// ```
///
/// # Errors
///
/// Returns [`DeckError`] for a bad header, dimension outside 1..=4,
/// wrong value counts, or non-numeric fields.
pub fn parse_table0_deck(text: &str) -> Result<Vec<Table0>, DeckError> {
    let mut tables = Vec::new();
    for card in split_cards(text) {
        let parts = header_parts(&card.header);
        if parts.len() != 3 || parts[0] != "TABLE" || parts[1] != "0" {
            continue;
        }
        let id: u32 = parts[2].parse().map_err(|_| {
            DeckError::new(format!(
                "card {} line {}: bad table id '{}'",
                card.header, card.line_no, parts[2]
            ))
        })?;
        let mut lines = card.body.iter();
        if lines.next().is_none() {
            return Err(DeckError::new(format!(
                "card {}: missing title line",
                card.header
            )));
        }
        let (shape_no, shape_line) = lines.next().ok_or_else(|| {
            DeckError::new(format!("card {}: missing dimension line", card.header))
        })?;
        let shape_floats = parse_floats(shape_line, *shape_no, &card.header)?;
        if shape_floats.len() != 5 {
            return Err(DeckError::new(format!(
                "card {}: dimension line needs 5 numbers",
                card.header
            )));
        }
        let dim = shape_floats[0] as usize;
        if !(1..=4).contains(&dim) {
            return Err(DeckError::new(format!(
                "card {}: dimension {dim} outside 1..=4",
                card.header
            )));
        }
        let shape: Vec<usize> = shape_floats[1..=dim].iter().map(|v| *v as usize).collect();
        if shape.contains(&0) {
            return Err(DeckError::new(format!(
                "card {}: grid counts must be positive",
                card.header
            )));
        }
        let mut rest = Vec::new();
        for (line_no, line) in lines {
            rest.extend(parse_floats(line, *line_no, &card.header)?);
        }
        let mut grids = Vec::new();
        let mut at = 0;
        for count in &shape {
            if at + count > rest.len() {
                return Err(DeckError::new(format!(
                    "card {}: missing grid values",
                    card.header
                )));
            }
            grids.push(rest[at..at + count].to_vec());
            at += count;
        }
        let need: usize = shape.iter().product();
        if rest.len() - at != need {
            return Err(DeckError::new(format!(
                "card {}: needs {need} table values, found {}",
                card.header,
                rest.len() - at
            )));
        }
        tables.push(Table0 {
            id,
            shape,
            grids,
            values: rest[at..].to_vec(),
        });
    }
    Ok(tables)
}

/// Converts a two-dimensional [`Table0`] into a frozen [`Table2D`].
///
/// The card stores values with the first entry varying fastest, which
/// is exactly the row-major `(y, x)` layout `Table2D::new` expects.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct_deck::{
///     parse_table0_deck, table0_to_table2d,
/// };
///
/// let deck = "/TABLE/0/9\ntitle\n2 2 2 0 0\n0.0 1.0\n0.0 10.0\n1.0 2.0 3.0 4.0\n";
/// let parsed = parse_table0_deck(deck).unwrap();
/// let table = table0_to_table2d(&parsed[0]).unwrap();
/// assert_eq!(table.eval(1.0, 10.0), 4.0);
/// ```
///
/// # Errors
///
/// Returns [`DeckError`] unless the dimension is exactly 2, or when
/// the frozen constructor rejects the grids.
pub fn table0_to_table2d(parsed: &Table0) -> Result<Table2D, DeckError> {
    if parsed.shape.len() != 2 {
        return Err(DeckError::new(format!(
            "table {}: needs dimension 2 for Table2D, found {}",
            parsed.id,
            parsed.shape.len()
        )));
    }
    Table2D::new(
        parsed.grids[0].clone(),
        parsed.grids[1].clone(),
        parsed.values.clone(),
    )
    .map_err(|cause| DeckError::new(format!("table {}: {cause}", parsed.id)))
}

/// Parsed `/TABLE/1` function-list card.
#[derive(Debug, Clone)]
pub struct Table1 {
    /// Table identifier from the card header.
    pub id: u32,
    /// Table dimension: 1 holds pairs, 2 holds function rows.
    pub dim: usize,
    /// Raw `(x, y)` pairs for dimension 1.
    pub pairs: Vec<[f64; 2]>,
    /// `(function id, abscissa, scale)` rows for dimension 2.
    pub rows: Vec<(u32, f64, f64)>,
}

/// Parses `/TABLE/1` function-list cards from deck text.
///
/// Dimension 1 behaves like `/FUNCT` with inline pairs. Dimension 2
/// holds `(function id, abscissa)` rows with an optional third scale
/// column multiplying the referenced function. Higher dimensions
/// parse the header only and fail conversion.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct_deck::parse_table1_deck;
///
/// let deck = "/TABLE/1/3\ntitle\n2\n10 1.0e-4\n20 1.0\n";
/// let tables = parse_table1_deck(deck).unwrap();
/// assert_eq!(tables[0].rows.len(), 2);
/// ```
///
/// # Errors
///
/// Returns [`DeckError`] for a bad header, a missing dimension, or
/// malformed rows.
pub fn parse_table1_deck(text: &str) -> Result<Vec<Table1>, DeckError> {
    let mut tables = Vec::new();
    for card in split_cards(text) {
        let parts = header_parts(&card.header);
        if parts.len() != 3 || parts[0] != "TABLE" || parts[1] != "1" {
            continue;
        }
        let id: u32 = parts[2].parse().map_err(|_| {
            DeckError::new(format!(
                "card {} line {}: bad table id '{}'",
                card.header, card.line_no, parts[2]
            ))
        })?;
        let mut lines = card.body.iter();
        if lines.next().is_none() {
            return Err(DeckError::new(format!(
                "card {}: missing title line",
                card.header
            )));
        }
        let (dim_no, dim_line) = lines.next().ok_or_else(|| {
            DeckError::new(format!("card {}: missing dimension line", card.header))
        })?;
        let dim_floats = parse_floats(dim_line, *dim_no, &card.header)?;
        if dim_floats.len() != 1 {
            return Err(DeckError::new(format!(
                "card {}: dimension line needs 1 number",
                card.header
            )));
        }
        let dim = dim_floats[0] as usize;
        if dim == 1 {
            let mut floats = Vec::new();
            for (line_no, line) in lines {
                floats.extend(parse_floats(line, *line_no, &card.header)?);
            }
            if floats.len() % 2 != 0 {
                return Err(DeckError::new(format!(
                    "card {}: odd float count {}",
                    card.header,
                    floats.len()
                )));
            }
            let pairs = floats
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| [pair[0], pair[1]])
                .collect();
            tables.push(Table1 {
                id,
                dim,
                pairs,
                rows: Vec::new(),
            });
            continue;
        }
        if dim != 2 {
            tables.push(Table1 {
                id,
                dim,
                pairs: Vec::new(),
                rows: Vec::new(),
            });
            continue;
        }
        let mut rows = Vec::new();
        for (line_no, line) in lines {
            let row = parse_floats(line, *line_no, &card.header)?;
            if row.len() != 2 && row.len() != 3 {
                return Err(DeckError::new(format!(
                    "card {} line {line_no}: row needs 2 or 3 numbers",
                    card.header
                )));
            }
            if row[0] < 0.0 || row[0].fract() != 0.0 {
                return Err(DeckError::new(format!(
                    "card {} line {line_no}: bad function id",
                    card.header
                )));
            }
            rows.push((row[0] as u32, row[1], row.get(2).copied().unwrap_or(1.0)));
        }
        tables.push(Table1 {
            id,
            dim,
            pairs: Vec::new(),
            rows,
        });
    }
    Ok(tables)
}

/// Builds a [`Funct1D`] from a dimension-1 [`Table1`].
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct_deck::{
///     parse_table1_deck, table1_to_funct,
/// };
///
/// let deck = "/TABLE/1/7\ntitle\n1\n0.0 0.0\n0.5 2.0\n";
/// let parsed = parse_table1_deck(deck).unwrap();
/// let funct = table1_to_funct(&parsed[0]).unwrap();
/// assert_eq!(funct.eval(0.25), 1.0);
/// ```
///
/// # Errors
///
/// Returns [`DeckError`] unless the dimension is 1, or when the
/// frozen constructor rejects the pairs.
pub fn table1_to_funct(parsed: &Table1) -> Result<Funct1D, DeckError> {
    if parsed.dim != 1 {
        return Err(DeckError::new(format!(
            "table {}: needs dimension 1 for Funct1D, found {}",
            parsed.id, parsed.dim
        )));
    }
    let (x, y) = (
        parsed.pairs.iter().map(|pair| pair[0]).collect(),
        parsed.pairs.iter().map(|pair| pair[1]).collect(),
    );
    Funct1D::new(x, y).map_err(|cause| DeckError::new(format!("table {}: {cause}", parsed.id)))
}

/// Builds a [`Table2D`] from a dimension-2 [`Table1`].
///
/// Resamples every referenced function on the union of all their
/// abscissae (sorted, duplicates removed), scales rows by the row
/// scale, and stores values row-major with the row abscissa as the
/// second entry. Queries on the union grid reproduce each function
/// exactly; off-grid queries blend neighbors linearly.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::plasticity::funct_deck::{
///     parse_funct_deck, parse_table1_deck, table1_to_table2d,
/// };
///
/// let deck = "/FUNCT/10\nt\n0.0 1.0\n1.0 2.0\n/FUNCT/20\nt\n0.0 1.0\n1.0 4.0\n\
///     /TABLE/1/3\ntitle\n2\n10 1.0e-4\n20 1.0\n";
/// let functs = parse_funct_deck(deck).unwrap();
/// let parsed = parse_table1_deck(deck).unwrap();
/// let table = table1_to_table2d(&parsed[0], &functs).unwrap();
/// assert_eq!(table.eval(1.0, 1.0e-4), 2.0);
/// assert_eq!(table.eval(1.0, 1.0), 4.0);
/// ```
///
/// # Errors
///
/// Returns [`DeckError`] unless the dimension is 2, when a row
/// references a missing function, or when grids are degenerate.
pub fn table1_to_table2d(parsed: &Table1, functs: &FunctDeck) -> Result<Table2D, DeckError> {
    if parsed.dim != 2 {
        return Err(DeckError::new(format!(
            "table {}: needs dimension 2 for Table2D, found {}",
            parsed.id, parsed.dim
        )));
    }
    if parsed.rows.is_empty() {
        return Err(DeckError::new(format!(
            "table {}: needs at least one function row",
            parsed.id
        )));
    }
    let mut grid: Vec<f64> = Vec::new();
    for (id, _, _) in &parsed.rows {
        let nodes = functs
            .nodes(*id)
            .ok_or_else(|| DeckError::new(format!("table {}: missing function {id}", parsed.id)))?;
        grid.extend(nodes.iter().map(|pair| pair[0]));
    }
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    grid.dedup();
    if grid.len() < 2 {
        return Err(DeckError::new(format!(
            "table {}: union grid needs two abscissae",
            parsed.id
        )));
    }
    let mut y = Vec::with_capacity(parsed.rows.len());
    let mut values = Vec::with_capacity(parsed.rows.len() * grid.len());
    for (id, abscissa, scale) in &parsed.rows {
        let funct = functs
            .get(*id)
            .ok_or_else(|| DeckError::new(format!("table {}: missing function {id}", parsed.id)))?;
        y.push(*abscissa);
        for gx in &grid {
            values.push(scale * funct.eval(*gx));
        }
    }
    Table2D::new(grid, y, values)
        .map_err(|cause| DeckError::new(format!("table {}: {cause}", parsed.id)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FUNCTS: &str = "#---1----|----2----|\n\
        /FUNCT/500006\n\
        shear yield dir 12\n\
        # X Y\n\
        -1000.0 5.86e-4\n\
        1000.0 5.86e-4\n\
        /FUNCT/500007\n\
        shear yield dir 23\n\
        -1000.0 1.45e-4\n\
        1000.0 1.45e-4\n\
        /END\n";

    #[test]
    fn funct_deck_parses_two_cards() {
        let deck = parse_funct_deck(FUNCTS).unwrap();
        assert_eq!(deck.len(), 2);
        assert_eq!(deck.get(500006).unwrap().eval(0.75), 5.86e-4);
        assert_eq!(deck.get(500007).unwrap().eval(-3.0), 1.45e-4);
        assert_eq!(
            deck.nodes(500006).unwrap(),
            &[[-1000.0, 5.86e-4], [1000.0, 5.86e-4]]
        );
    }

    #[test]
    fn funct_deck_clamps_ends() {
        let deck = parse_funct_deck(FUNCTS).unwrap();
        let funct = deck.get(500006).unwrap();
        assert_eq!(funct.eval(-1.0e9), 5.86e-4);
        assert_eq!(funct.eval(1.0e9), 5.86e-4);
    }

    #[test]
    fn funct_deck_skips_title_with_digits() {
        let deck = "/FUNCT/11\ncurve 1 rate 10\n0.0 0.0\n1.0 2.0\n";
        let parsed = parse_funct_deck(deck).unwrap();
        assert_eq!(parsed.get(11).unwrap().eval(0.5), 1.0);
    }

    #[test]
    fn funct_deck_rejects_bad_pair_count() {
        let bad = "/FUNCT/1\ntitle\n0.0 1.0\n2.0\n";
        assert!(parse_funct_deck(bad).is_err());
    }

    #[test]
    fn funct_deck_rejects_unordered_abscissae() {
        let bad = "/FUNCT/1\ntitle\n1.0 1.0\n0.0 2.0\n";
        let err = parse_funct_deck(bad).unwrap_err();
        assert!(err.message.contains("strictly increase"), "{err}");
    }

    #[test]
    fn table0_round_trip_to_table2d() {
        let deck = "/TABLE/0/17\ntitle\n2 3 2 0 0\n\
            0.0 0.5 1.0\n0.0 100.0\n\
            1.0 2.0 3.0 4.0 5.0 6.0\n";
        let parsed = parse_table0_deck(deck).unwrap();
        assert_eq!(parsed[0].id, 17);
        let table = table0_to_table2d(&parsed[0]).unwrap();
        assert_eq!(table.eval(0.0, 0.0), 1.0);
        assert_eq!(table.eval(1.0, 0.0), 3.0);
        assert_eq!(table.eval(0.0, 100.0), 4.0);
        assert_eq!(table.eval(0.5, 50.0), 3.5);
    }

    #[test]
    fn table0_rejects_wrong_value_count() {
        let deck = "/TABLE/0/17\ntitle\n2 2 2 0 0\n0.0 1.0\n0.0 1.0\n1.0 2.0 3.0\n";
        assert!(parse_table0_deck(deck).is_err());
    }

    #[test]
    fn table1_dim1_converts_to_funct() {
        let deck = "/TABLE/1/7\ntitle\n1\n0.0 0.0\n0.5 2.0\n";
        let parsed = parse_table1_deck(deck).unwrap();
        let funct = table1_to_funct(&parsed[0]).unwrap();
        assert_eq!(funct.eval(0.25), 1.0);
    }

    #[test]
    fn table1_dim2_resamples_union_grid() {
        let deck = format!("/TABLE/1/3\nrate table\n2\n500006 0.0\n500007 100.0\n{FUNCTS}");
        let functs = parse_funct_deck(&deck).unwrap();
        let parsed = parse_table1_deck(&deck).unwrap();
        let table = table1_to_table2d(&parsed[0], &functs).unwrap();
        assert_eq!(table.eval(0.0, 0.0), 5.86e-4);
        assert_eq!(table.eval(0.0, 100.0), 1.45e-4);
        let mid = table.eval(0.0, 50.0);
        assert!((mid - 0.5 * (5.86e-4 + 1.45e-4)).abs() < 1.0e-18);
    }

    #[test]
    fn table1_missing_funct_errors() {
        let deck = "/TABLE/1/3\ntitle\n2\n99 0.0\n100 1.0\n";
        let functs = parse_funct_deck(deck).unwrap();
        let parsed = parse_table1_deck(deck).unwrap();
        let err = table1_to_table2d(&parsed[0], &functs).unwrap_err();
        assert!(err.message.contains("missing function 99"), "{err}");
    }
}
