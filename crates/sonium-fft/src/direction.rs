//! Forward / inverse direction enum, ABI-compatible with `rustfft::FftDirection`.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftDirection {
    Forward,
    Inverse,
}

impl FftDirection {
    pub fn opposite(self) -> Self {
        match self {
            FftDirection::Forward => FftDirection::Inverse,
            FftDirection::Inverse => FftDirection::Forward,
        }
    }
}
