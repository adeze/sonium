//! Internal `array!` macro used by sonium-bem tests and a few residual call
//! sites inside the crate. After the migration off ndarray, it is just an
//! alias for `vec!` producing a `Vec<T>`.

macro_rules! array {
    ($($value:expr),* $(,)?) => {
        vec![$($value),*]
    };
}

pub(crate) use array;
