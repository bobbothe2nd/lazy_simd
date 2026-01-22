//! Core traits and types that abstract usage of SIMD.

pub mod backend;

mod aligned;
pub use aligned::{Simd, SimdSlice};

mod unaligned;
pub use self::unaligned::{
    add_arrays, add_arrays_double, add_scalar_array, add_scalar_array_double, add_scalar_slice,
    add_scalar_slice_double, add_slices, add_slices_double, div_arrays, div_arrays_double,
    div_scalar_array, div_scalar_array_double, div_scalar_slice, div_scalar_slice_double,
    div_slices, div_slices_double, mul_add_scalar_slice, mul_add_scalar_slice_double, mul_arrays,
    mul_arrays_double, mul_scalar_array, mul_scalar_array_double, mul_scalar_slice,
    mul_scalar_slice_double, mul_slices, mul_slices_double, sub_arrays, sub_arrays_double,
    sub_scalar_array, sub_scalar_array_double, sub_scalar_slice, sub_scalar_slice_double,
    sub_slices, sub_slices_double,
};

/// A SIMD vector which has a total size equal to it's lane count.
pub type SimdRegister<T, const LANES: usize = { crate::MAX_SIMD_SINGLE_PRECISION_LANES }> =
    Simd<T, LANES, LANES>;

/// A SIMD vector which has a lane count of [`crate::MAX_SIMD_SINGLE_PRECISION_LANES`].
pub type AutoSimd<T, const N: usize> = Simd<T, N, { crate::MAX_SIMD_SINGLE_PRECISION_LANES }>;

/// A marker trait for scalars that are feasible for use in SIMD.
pub trait SimdScalar:
    num_traits::NumAssign + num_traits::NumAssignOps + num_traits::NumCast
{
}

impl SimdScalar for u8 {}
impl SimdScalar for u16 {}
impl SimdScalar for u32 {}
impl SimdScalar for u64 {}
impl SimdScalar for u128 {}
impl SimdScalar for usize {}
impl SimdScalar for i8 {}
impl SimdScalar for i16 {}
impl SimdScalar for i32 {}
impl SimdScalar for i64 {}
impl SimdScalar for i128 {}
impl SimdScalar for isize {}
impl SimdScalar for f32 {}
impl SimdScalar for f64 {}

/// Sealed marker trait for whether `Self` represents a supported lane count for a SIMD register.
pub trait SupportedLaneCount: crate::Internal {}

/// A type that implements [`SupportedLaneCount`] based off the value of `N`.
pub struct LaneCount<const N: usize>;

impl<const N: usize> crate::Internal for LaneCount<N> {}

impl SupportedLaneCount for LaneCount<1> {}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse"),
    all(target_arch = "aarch64", target_feature = "neon"),
))]
impl SupportedLaneCount for LaneCount<2> {}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse"),
    all(target_arch = "aarch64", target_feature = "neon"),
))]
impl SupportedLaneCount for LaneCount<4> {}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
impl SupportedLaneCount for LaneCount<8> {}

/// Represents a scalar that can be copied and is expected for a SIMD register.
pub trait SimdElement: SimdScalar + Copy + Default {}

impl<T: SimdScalar + Copy + Default> SimdElement for T {}

impl backend::NonAssociativeSimd<[f32; 1], f32, 1> for [f32; 1] {
    type Backend = backend::SinglePrecisionNoSimd<1>;
}

impl backend::NonAssociativeSimd<[f64; 1], f64, 1> for [f64; 1] {
    type Backend = backend::DoublePrecisionNoSimd<1>;
}
