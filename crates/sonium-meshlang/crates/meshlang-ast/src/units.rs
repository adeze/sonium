//! Physical units and dimensional algebra.
//!
//! Parsing is deliberately total: unsupported symbols and malformed compound
//! expressions return a [`UnitParseError`] instead of being mistaken for the
//! absence of a unit by parser clients.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Exponents of the seven SI base dimensions plus plane angle.
///
/// Angle is retained as a pseudo-dimension because MeshLang must distinguish
/// angular parameters from ordinary dimensionless scalars at the syntax/type
/// boundary even though radians are dimensionless in SI.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Dimension {
    pub length: i8,
    pub time: i8,
    pub mass: i8,
    #[serde(default)]
    pub electric_current: i8,
    #[serde(default)]
    pub temperature: i8,
    #[serde(default)]
    pub amount: i8,
    #[serde(default)]
    pub luminous_intensity: i8,
    pub angle: i8,
}

impl Dimension {
    const fn new(length: i8, time: i8, mass: i8, angle: i8) -> Self {
        Self {
            length,
            time,
            mass,
            electric_current: 0,
            temperature: 0,
            amount: 0,
            luminous_intensity: 0,
            angle,
        }
    }

    pub const DIMENSIONLESS: Self = Self::new(0, 0, 0, 0);
    pub const LENGTH: Self = Self::new(1, 0, 0, 0);
    pub const TIME: Self = Self::new(0, 1, 0, 0);
    pub const MASS: Self = Self::new(0, 0, 1, 0);
    pub const FREQUENCY: Self = Self::new(0, -1, 0, 0);
    pub const VELOCITY: Self = Self::new(1, -1, 0, 0);
    pub const ACCELERATION: Self = Self::new(1, -2, 0, 0);
    pub const ANGLE: Self = Self::new(0, 0, 0, 1);
    pub const FORCE: Self = Self::new(1, -2, 1, 0);
    pub const PRESSURE: Self = Self::new(-1, -2, 1, 0);
    pub const ENERGY: Self = Self::new(2, -2, 1, 0);
    pub const POWER: Self = Self::new(2, -3, 1, 0);

    pub fn checked_mul(self, other: Self) -> Result<Self, UnitParseErrorKind> {
        self.checked_combine(other, false)
    }

    pub fn checked_div(self, other: Self) -> Result<Self, UnitParseErrorKind> {
        self.checked_combine(other, true)
    }

    fn checked_combine(self, other: Self, subtract: bool) -> Result<Self, UnitParseErrorKind> {
        let combine = |left: i8, right: i8| {
            if subtract {
                left.checked_sub(right)
            } else {
                left.checked_add(right)
            }
            .ok_or(UnitParseErrorKind::ExponentOutOfRange)
        };
        Ok(Self {
            length: combine(self.length, other.length)?,
            time: combine(self.time, other.time)?,
            mass: combine(self.mass, other.mass)?,
            electric_current: combine(self.electric_current, other.electric_current)?,
            temperature: combine(self.temperature, other.temperature)?,
            amount: combine(self.amount, other.amount)?,
            luminous_intensity: combine(self.luminous_intensity, other.luminous_intensity)?,
            angle: combine(self.angle, other.angle)?,
        })
    }

    /// Raise every dimension exponent to an integer power.
    pub fn powi(self, exponent: i8) -> Result<Self, UnitParseErrorKind> {
        let scale = |value: i8| {
            value
                .checked_mul(exponent)
                .ok_or(UnitParseErrorKind::ExponentOutOfRange)
        };
        Ok(Self {
            length: scale(self.length)?,
            time: scale(self.time)?,
            mass: scale(self.mass)?,
            electric_current: scale(self.electric_current)?,
            temperature: scale(self.temperature)?,
            amount: scale(self.amount)?,
            luminous_intensity: scale(self.luminous_intensity)?,
            angle: scale(self.angle)?,
        })
    }

    pub fn is_dimensionless(&self) -> bool {
        *self == Self::DIMENSIONLESS
    }

    pub fn is_compatible(&self, other: &Self) -> bool {
        self == other
    }

    pub fn display_name(&self) -> &'static str {
        match *self {
            Self::DIMENSIONLESS => "dimensionless",
            Self::LENGTH => "length",
            Self::TIME => "time",
            Self::MASS => "mass",
            Self::FREQUENCY => "frequency",
            Self::VELOCITY => "velocity",
            Self::ACCELERATION => "acceleration",
            Self::ANGLE => "angle",
            Self::FORCE => "force",
            Self::PRESSURE => "pressure",
            Self::ENERGY => "energy",
            Self::POWER => "power",
            _ => "compound dimension",
        }
    }
}

/// A parsed unit expression, preserving the exact source spelling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Unit {
    pub text: String,
    pub dimension: Dimension,
    /// Multiplier that converts a value in this unit to coherent SI units.
    pub to_si_factor: f64,
}

impl Unit {
    /// Parse a complete unit expression.
    ///
    /// Supported grammar: `factor ((* | /) factor)*`, where a factor is a
    /// symbol or parenthesized expression followed by an optional signed
    /// integer exponent (`m^2`, `s^-1`, `kg*m/s^2`). `1` and `%` are
    /// explicit dimensionless units; `dB` is rejected with
    /// [`UnitParseErrorKind::LogarithmicUnit`] because decibels are
    /// logarithmic and have no linear `to_si_factor`.
    pub fn parse(text: &str) -> Result<Self, UnitParseError> {
        let mut parser = UnitParser::new(text);
        parser.skip_whitespace();
        if parser.is_eof() {
            return Err(parser.error(UnitParseErrorKind::Empty));
        }
        let parsed = parser.parse_product()?;
        parser.skip_whitespace();
        if !parser.is_eof() {
            return Err(parser.error(UnitParseErrorKind::UnexpectedToken));
        }
        Ok(Self {
            text: text.to_owned(),
            dimension: parsed.dimension,
            to_si_factor: parsed.factor,
        })
    }

    pub fn dimensionless() -> Self {
        Self {
            text: "1".into(),
            dimension: Dimension::DIMENSIONLESS,
            to_si_factor: 1.0,
        }
    }
}

/// A precise failure from [`Unit::parse`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnitParseError {
    pub input: String,
    pub offset: usize,
    pub kind: UnitParseErrorKind,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum UnitParseErrorKind {
    Empty,
    UnknownSymbol,
    UnexpectedToken,
    MissingFactor,
    MissingClosingParenthesis,
    MissingExponent,
    ExponentOutOfRange,
    NonFiniteScale,
    /// A logarithmic unit (`dB`) that has no linear `to_si_factor`.
    LogarithmicUnit,
}

impl fmt::Display for UnitParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self.kind {
            UnitParseErrorKind::Empty => "unit expression is empty",
            UnitParseErrorKind::UnknownSymbol => "unknown unit symbol",
            UnitParseErrorKind::UnexpectedToken => "unexpected token in unit expression",
            UnitParseErrorKind::MissingFactor => "expected a unit factor",
            UnitParseErrorKind::MissingClosingParenthesis => "missing closing parenthesis",
            UnitParseErrorKind::MissingExponent => "expected an integer exponent",
            UnitParseErrorKind::ExponentOutOfRange => "unit exponent is out of range",
            UnitParseErrorKind::NonFiniteScale => "unit scale is not finite",
            UnitParseErrorKind::LogarithmicUnit => {
                "logarithmic units (dB) have no linear SI conversion factor"
            }
        };
        write!(f, "{message} at byte {} in `{}`", self.offset, self.input)
    }
}

impl std::error::Error for UnitParseError {}

#[derive(Clone, Copy)]
struct ParsedUnit {
    dimension: Dimension,
    factor: f64,
}

struct UnitParser<'a> {
    input: &'a str,
    offset: usize,
}

impl<'a> UnitParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, offset: 0 }
    }

    fn parse_product(&mut self) -> Result<ParsedUnit, UnitParseError> {
        let mut value = self.parse_factor()?;
        loop {
            self.skip_whitespace();
            let divide = match self.peek_char() {
                Some('*' | '·') => false,
                Some('/') => true,
                _ => break,
            };
            self.bump_char();
            let rhs = self.parse_factor()?;
            if divide {
                value.dimension = value
                    .dimension
                    .checked_div(rhs.dimension)
                    .map_err(|kind| self.error(kind))?;
                value.factor /= rhs.factor;
            } else {
                value.dimension = value
                    .dimension
                    .checked_mul(rhs.dimension)
                    .map_err(|kind| self.error(kind))?;
                value.factor *= rhs.factor;
            }
            if !value.factor.is_finite() {
                return Err(self.error(UnitParseErrorKind::NonFiniteScale));
            }
        }
        Ok(value)
    }

    fn parse_factor(&mut self) -> Result<ParsedUnit, UnitParseError> {
        self.skip_whitespace();
        let mut value = if self.peek_char() == Some('(') {
            self.bump_char();
            let nested = self.parse_product()?;
            self.skip_whitespace();
            if self.peek_char() != Some(')') {
                return Err(self.error(UnitParseErrorKind::MissingClosingParenthesis));
            }
            self.bump_char();
            nested
        } else {
            self.parse_symbol()?
        };

        self.skip_whitespace();
        if self.peek_char() == Some('^') {
            self.bump_char();
            let exponent = self.parse_exponent()?;
            value.dimension = value
                .dimension
                .powi(exponent)
                .map_err(|kind| self.error(kind))?;
            value.factor = value.factor.powi(i32::from(exponent));
            if !value.factor.is_finite() {
                return Err(self.error(UnitParseErrorKind::NonFiniteScale));
            }
        }
        Ok(value)
    }

    fn parse_symbol(&mut self) -> Result<ParsedUnit, UnitParseError> {
        let start = self.offset;
        while let Some(character) = self.peek_char() {
            if character.is_alphanumeric() || matches!(character, '%' | 'µ' | 'μ') {
                self.bump_char();
            } else {
                break;
            }
        }
        if start == self.offset {
            return Err(self.error(UnitParseErrorKind::MissingFactor));
        }
        let symbol = &self.input[start..self.offset];
        // Decibels are logarithmic, not a linear scale factor: `3 dB` must
        // not silently convert with `to_si_factor = 1.0`.
        if symbol == "dB" {
            return Err(UnitParseError {
                input: self.input.to_owned(),
                offset: start,
                kind: UnitParseErrorKind::LogarithmicUnit,
            });
        }
        symbol_value(symbol).ok_or_else(|| UnitParseError {
            input: self.input.to_owned(),
            offset: start,
            kind: UnitParseErrorKind::UnknownSymbol,
        })
    }

    fn parse_exponent(&mut self) -> Result<i8, UnitParseError> {
        self.skip_whitespace();
        let start = self.offset;
        if matches!(self.peek_char(), Some('+' | '-')) {
            self.bump_char();
        }
        let digits_start = self.offset;
        while self.peek_char().is_some_and(|c| c.is_ascii_digit()) {
            self.bump_char();
        }
        if digits_start == self.offset {
            return Err(self.error(UnitParseErrorKind::MissingExponent));
        }
        self.input[start..self.offset]
            .parse::<i8>()
            .map_err(|_| UnitParseError {
                input: self.input.to_owned(),
                offset: start,
                kind: UnitParseErrorKind::ExponentOutOfRange,
            })
    }

    fn skip_whitespace(&mut self) {
        while self.peek_char().is_some_and(char::is_whitespace) {
            self.bump_char();
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.offset..].chars().next()
    }

    fn bump_char(&mut self) {
        if let Some(character) = self.peek_char() {
            self.offset += character.len_utf8();
        }
    }

    fn is_eof(&self) -> bool {
        self.offset == self.input.len()
    }

    fn error(&self, kind: UnitParseErrorKind) -> UnitParseError {
        UnitParseError {
            input: self.input.to_owned(),
            offset: self.offset,
            kind,
        }
    }
}

fn symbol_value(symbol: &str) -> Option<ParsedUnit> {
    let value = match symbol {
        // Explicit dimensionless forms (`dB` is rejected in
        // `parse_symbol`: it is logarithmic, not a linear scale-1 unit).
        "1" => (Dimension::DIMENSIONLESS, 1.0),
        "%" => (Dimension::DIMENSIONLESS, 0.01),

        // Length.
        "m" => (Dimension::LENGTH, 1.0),
        "km" => (Dimension::LENGTH, 1e3),
        "cm" => (Dimension::LENGTH, 1e-2),
        "mm" => (Dimension::LENGTH, 1e-3),
        "um" | "µm" | "μm" => (Dimension::LENGTH, 1e-6),
        "in" => (Dimension::LENGTH, 0.0254),

        // Time.
        "s" => (Dimension::TIME, 1.0),
        "ms" => (Dimension::TIME, 1e-3),
        "us" | "µs" | "μs" => (Dimension::TIME, 1e-6),

        // Remaining SI base units.
        "kg" => (Dimension::MASS, 1.0),
        "g" => (Dimension::MASS, 1e-3),
        "A" => (
            Dimension {
                electric_current: 1,
                ..Dimension::DIMENSIONLESS
            },
            1.0,
        ),
        "K" => (
            Dimension {
                temperature: 1,
                ..Dimension::DIMENSIONLESS
            },
            1.0,
        ),
        "mol" => (
            Dimension {
                amount: 1,
                ..Dimension::DIMENSIONLESS
            },
            1.0,
        ),
        "cd" => (
            Dimension {
                luminous_intensity: 1,
                ..Dimension::DIMENSIONLESS
            },
            1.0,
        ),

        // Angle and common derived units.
        "rad" => (Dimension::ANGLE, 1.0),
        "deg" => (Dimension::ANGLE, std::f64::consts::PI / 180.0),
        "Hz" => (Dimension::FREQUENCY, 1.0),
        "kHz" => (Dimension::FREQUENCY, 1e3),
        "MHz" => (Dimension::FREQUENCY, 1e6),
        "N" => (Dimension::FORCE, 1.0),
        "Pa" => (Dimension::PRESSURE, 1.0),
        "kPa" => (Dimension::PRESSURE, 1e3),
        "MPa" => (Dimension::PRESSURE, 1e6),
        "J" => (Dimension::ENERGY, 1.0),
        "W" => (Dimension::POWER, 1.0),
        _ => return None,
    };
    Some(ParsedUnit {
        dimension: value.0,
        factor: value.1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_derived_and_dimensionless_units() {
        assert_eq!(Unit::parse("mm").unwrap().dimension, Dimension::LENGTH);
        assert_eq!(Unit::parse("Hz").unwrap().dimension, Dimension::FREQUENCY);
        assert!(Unit::parse("1").unwrap().dimension.is_dimensionless());
        assert_eq!(Unit::parse("%").unwrap().to_si_factor, 0.01);
    }

    #[test]
    fn parses_compound_exponent_and_parenthesized_units() {
        let acceleration = Unit::parse("m/s^2").unwrap();
        assert_eq!(acceleration.dimension, Dimension::ACCELERATION);

        let pressure = Unit::parse("kg/(m*s^2)").unwrap();
        assert_eq!(pressure.dimension, Dimension::PRESSURE);
        assert_eq!(
            pressure.to_si_factor,
            Unit::parse("Pa").unwrap().to_si_factor
        );

        let inverse_area = Unit::parse("cm^-2").unwrap();
        assert_eq!(inverse_area.dimension.length, -2);
        assert!((inverse_area.to_si_factor - 1e4).abs() < 1e-9);
    }

    #[test]
    fn preserves_exact_unit_spelling() {
        let unit = Unit::parse(" kg * m / s^2 ").unwrap();
        assert_eq!(unit.text, " kg * m / s^2 ");
        assert_eq!(unit.dimension, Dimension::FORCE);
    }

    #[test]
    fn reports_malformed_or_unknown_units_explicitly() {
        assert_eq!(Unit::parse("").unwrap_err().kind, UnitParseErrorKind::Empty);
        assert_eq!(
            Unit::parse("furlong").unwrap_err().kind,
            UnitParseErrorKind::UnknownSymbol
        );
        assert_eq!(
            Unit::parse("m/").unwrap_err().kind,
            UnitParseErrorKind::MissingFactor
        );
        assert_eq!(
            Unit::parse("m^").unwrap_err().kind,
            UnitParseErrorKind::MissingExponent
        );
        assert_eq!(
            Unit::parse("(m/s").unwrap_err().kind,
            UnitParseErrorKind::MissingClosingParenthesis
        );
        assert_eq!(
            Unit::parse("m^127*m").unwrap_err().kind,
            UnitParseErrorKind::ExponentOutOfRange
        );
        // Decibels are logarithmic: reject instead of converting with a
        // bogus linear factor of 1.0.
        assert_eq!(
            Unit::parse("dB").unwrap_err().kind,
            UnitParseErrorKind::LogarithmicUnit
        );
    }

    #[test]
    fn dimensional_algebra_is_closed_for_supported_exponents() {
        assert_eq!(
            Dimension::VELOCITY
                .checked_div(Dimension::FREQUENCY)
                .unwrap(),
            Dimension::LENGTH
        );
        assert_eq!(
            Dimension::LENGTH.checked_mul(Dimension::FREQUENCY).unwrap(),
            Dimension::VELOCITY
        );
        assert_eq!(Dimension::LENGTH.powi(2).unwrap().length, 2);
    }

    #[test]
    fn dimensional_algebra_reports_exponent_overflow_without_panicking() {
        let maximum = Dimension {
            length: i8::MAX,
            ..Dimension::DIMENSIONLESS
        };
        let minimum = Dimension {
            length: i8::MIN,
            ..Dimension::DIMENSIONLESS
        };

        assert_eq!(
            maximum.checked_mul(Dimension::LENGTH),
            Err(UnitParseErrorKind::ExponentOutOfRange)
        );
        assert_eq!(
            minimum.checked_div(Dimension::LENGTH),
            Err(UnitParseErrorKind::ExponentOutOfRange)
        );
    }
}
