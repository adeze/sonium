use libloading::{Library, Symbol};
use num_complex::Complex64;
use sonium_fft::{FftDirection, FftPlanner};
use std::ffi::c_void;

type FftwComplex = [f64; 2];
type FftwPlan = *mut c_void;
type FftwPlanDft1d =
    unsafe extern "C" fn(i32, *mut FftwComplex, *mut FftwComplex, i32, u32) -> FftwPlan;
type FftwExecuteDft = unsafe extern "C" fn(FftwPlan, *mut FftwComplex, *mut FftwComplex);
type FftwDestroyPlan = unsafe extern "C" fn(FftwPlan);

const FFTW_FORWARD: i32 = -1;
const FFTW_BACKWARD: i32 = 1;
const FFTW_ESTIMATE: u32 = 1 << 6;

struct FftwApi {
    _library: Library,
    plan_dft_1d: FftwPlanDft1d,
    execute_dft: FftwExecuteDft,
    destroy_plan: FftwDestroyPlan,
}

impl FftwApi {
    fn load() -> Option<Self> {
        #[cfg(target_os = "windows")]
        const LIBRARIES: &[&str] = &["libfftw3-3.dll", "fftw3.dll"];
        #[cfg(target_os = "macos")]
        const LIBRARIES: &[&str] = &["libfftw3.3.dylib", "libfftw3.dylib"];
        #[cfg(all(unix, not(target_os = "macos")))]
        const LIBRARIES: &[&str] = &["libfftw3.so.3", "libfftw3.so"];

        for name in LIBRARIES {
            // SAFETY: the library is retained until after copied symbols are used.
            let Ok(library) = (unsafe { Library::new(*name) }) else {
                continue;
            };
            // SAFETY: these names are the stable FFTW C ABI.
            let symbols = unsafe {
                (
                    load_symbol::<FftwPlanDft1d>(&library, b"fftw_plan_dft_1d\0"),
                    load_symbol::<FftwExecuteDft>(&library, b"fftw_execute_dft\0"),
                    load_symbol::<FftwDestroyPlan>(&library, b"fftw_destroy_plan\0"),
                )
            };
            if let (Ok(plan_dft_1d), Ok(execute_dft), Ok(destroy_plan)) = symbols {
                return Some(Self {
                    _library: library,
                    plan_dft_1d,
                    execute_dft,
                    destroy_plan,
                });
            }
        }
        None
    }

    fn execute(&self, input: &[Complex64], direction: i32) -> Vec<Complex64> {
        let mut input = input.iter().map(|x| [x.re, x.im]).collect::<Vec<_>>();
        let mut output = vec![[0.0, 0.0]; input.len()];
        let n = i32::try_from(input.len()).expect("FFTW test length fits i32");
        // SAFETY: FFTW owns the plan; buffers remain valid for the invocation.
        unsafe {
            let plan = (self.plan_dft_1d)(
                n,
                input.as_mut_ptr(),
                output.as_mut_ptr(),
                direction,
                FFTW_ESTIMATE,
            );
            assert!(!plan.is_null(), "FFTW failed to create a plan for n={n}");
            (self.execute_dft)(plan, input.as_mut_ptr(), output.as_mut_ptr());
            (self.destroy_plan)(plan);
        }
        output
            .into_iter()
            .map(|[re, im]| Complex64::new(re, im))
            .collect()
    }
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, libloading::Error> {
    // SAFETY: the requested type matches the documented FFTW C ABI.
    unsafe { library.get::<T>(name).map(|symbol: Symbol<'_, T>| *symbol) }
}

#[test]
fn fftw_matches_sonium_when_available() {
    let Some(fftw) = FftwApi::load() else {
        eprintln!("FFTW not available; skipping optional differential test");
        return;
    };
    for &n in &[1usize, 2, 3, 4, 5, 7, 8, 11, 16, 23, 32, 45, 64] {
        let samples = (0..n)
            .map(|i| {
                Complex64::new(
                    (0.17 * i as f64).sin() + 0.03 * i as f64,
                    (0.11 * i as f64).cos() - 0.02 * i as f64,
                )
            })
            .collect::<Vec<_>>();
        for &(direction, sign) in &[
            (FftDirection::Forward, FFTW_FORWARD),
            (FftDirection::Inverse, FFTW_BACKWARD),
        ] {
            let expected = fftw.execute(&samples, sign);
            let mut actual = samples.clone();
            FftPlanner::<f64>::new()
                .plan(n, direction)
                .process(&mut actual);
            let tolerance = 2.0e-11 * (n as f64).max(1.0);
            for (got, want) in actual.iter().zip(expected) {
                assert!((got - want).norm() < tolerance, "FFTW mismatch at n={n}");
            }
        }
    }
}
