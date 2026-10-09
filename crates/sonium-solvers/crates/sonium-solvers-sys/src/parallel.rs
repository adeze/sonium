//! Parallel utilities using rayon
//!
//! Provides parallel abstractions that work across native and WASM environments.
//! Every helper falls back to a serial iterator below [`SERIAL_THRESHOLD`]
//! items (or when the rayon pool has a single thread), since thread dispatch
//! exceeds the work for the crate's small vectors (<1000 DOFs per guidance).

use rayon::prelude::*;

/// Minimum length before a `parallel_*` helper uses rayon; below this the
/// serial path runs. Thread-pool dispatch + work-stealing dominate tiny
/// payloads, so this is a correctness-of-performance floor, not a tuning knob.
pub const SERIAL_THRESHOLD: usize = 1024;

/// Check if parallel processing is available.
pub fn is_parallel_available() -> bool {
    rayon::current_num_threads() > 1
}

fn use_parallel(len: usize) -> bool {
    len >= SERIAL_THRESHOLD && is_parallel_available()
}

/// Parallel map over a slice
pub fn parallel_map<T, U, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    if !use_parallel(data.len()) {
        return data.iter().map(f).collect();
    }
    data.par_iter().map(f).collect()
}

/// Parallel map with index
pub fn parallel_map_indexed<U, F>(count: usize, f: F) -> Vec<U>
where
    U: Send,
    F: Fn(usize) -> U + Sync + Send,
{
    if !use_parallel(count) {
        return (0..count).map(f).collect();
    }
    (0..count).into_par_iter().map(f).collect()
}

/// Parallel for_each over a slice
pub fn parallel_for_each<T, F>(data: &[T], f: F)
where
    T: Sync,
    F: Fn(&T) + Sync + Send,
{
    if !use_parallel(data.len()) {
        return data.iter().for_each(f);
    }
    data.par_iter().for_each(f);
}

/// Parallel enumerate and map
pub fn parallel_enumerate_map<T, U, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(usize, &T) -> U + Sync + Send,
{
    if !use_parallel(data.len()) {
        return data.iter().enumerate().map(|(i, x)| f(i, x)).collect();
    }
    data.par_iter().enumerate().map(|(i, x)| f(i, x)).collect()
}

/// Parallel filter_map
pub fn parallel_filter_map<T, U, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Option<U> + Sync + Send,
{
    if !use_parallel(data.len()) {
        return data.iter().filter_map(f).collect();
    }
    data.par_iter().filter_map(f).collect()
}

/// Parallel flat_map
pub fn parallel_flat_map<T, U, I, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    I: rayon::iter::IntoParallelIterator<Item = U> + IntoIterator<Item = U>,
    F: Fn(&T) -> I + Sync + Send,
{
    if !use_parallel(data.len()) {
        return data.iter().flat_map(f).collect();
    }
    data.par_iter().flat_map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_map_small_input_never_enters_parallel_inner_iterator() {
        struct SerialOnly(Vec<usize>);
        impl IntoIterator for SerialOnly {
            type Item = usize;
            type IntoIter = std::vec::IntoIter<usize>;
            fn into_iter(self) -> Self::IntoIter {
                self.0.into_iter()
            }
        }
        impl IntoParallelIterator for SerialOnly {
            type Item = usize;
            type Iter = rayon::vec::IntoIter<usize>;
            fn into_par_iter(self) -> Self::Iter {
                panic!("serial fallback dispatched to Rayon")
            }
        }
        assert_eq!(
            parallel_flat_map(&[1, 2], |&n| SerialOnly(vec![n, n + 1])),
            vec![1, 2, 2, 3]
        );
    }

    #[test]
    fn test_parallel_map() {
        let data = vec![1, 2, 3, 4, 5];
        let result = parallel_map(&data, |x| x * 2);
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_map_indexed() {
        let result = parallel_map_indexed(5, |i| i * 2);
        assert_eq!(result, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_parallel_enumerate_map() {
        let data = vec![10, 20, 30];
        let result = parallel_enumerate_map(&data, |i, x| (i, *x));
        assert_eq!(result, vec![(0, 10), (1, 20), (2, 30)]);
    }

    #[test]
    fn test_small_inputs_stay_serial_but_correct() {
        let data: Vec<usize> = (0..16).collect();
        assert_eq!(parallel_map(&data, |x| x + 1)[15], 16);
        assert!(!use_parallel(data.len()));
    }

    #[test]
    fn test_parallel_availability_matches_pool() {
        assert_eq!(is_parallel_available(), rayon::current_num_threads() > 1);
    }
}
