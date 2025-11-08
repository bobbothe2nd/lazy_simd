//! The internal low-level, primitive backend for all SIMD-focused operations.

use crate::{
    scalar::{
        AcceleratedScalar, ImplicitSupersetOf, SimdAdd, SimdDiv, SimdMul, SimdSub, SupersetOf,
    },
    simd::SimdScalar,
};

/// Generic functions that define `Self` as a backend, operating on chunks of `T`.
pub trait UnalignedSimdOps<
    T: SupersetOf<E>,
    E: AcceleratedScalar<F, LANES> + SupersetOf<F> + Copy,
    F: SimdScalar + Copy,
    const LANES: usize,
>: SimdAdd<Set = T> + SimdSub<Set = T> + SimdMul<Set = T> + SimdDiv<Set = T>
{
    /// Computes `self + a * b` and copies them into the output.
    #[must_use]
    #[inline(always)]
    fn simd_mul_add(a: &T, b: &T, rhs: &T) -> T {
        Self::simd_add(&Self::simd_mul(a, b), rhs)
    }

    /// Computes the multiplicative inverse of `self` and copies that into the output.
    ///
    /// Formula: `1 / self`
    #[must_use]
    #[inline(always)]
    fn simd_inv(a: &T) -> T {
        Self::simd_div(&T::from_subset(E::from_subset(F::one())), a)
    }

    /// Computes the negation of `self` and copies that into the output.
    ///
    /// Formula: `0 - self`
    #[must_use]
    #[inline(always)]
    fn simd_neg(a: &T) -> T {
        Self::simd_sub(&T::from_subset(E::from_subset(F::zero())), a)
    }
}

impl<
        T: SupersetOf<E>,
        E: AcceleratedScalar<F, LANES> + SupersetOf<F> + Copy,
        F: SimdScalar + Copy,
        const LANES: usize,
        U: SimdAdd<Set = T> + SimdSub<Set = T> + SimdMul<Set = T> + SimdDiv<Set = T>,
    > UnalignedSimdOps<T, E, F, LANES> for U
{
}

/// Variant of [`UnalignedSimdOps`] that requires alignment.
///
/// These operations are more efficient than their unaligned counterpart, but
/// they often aren't as safe. If unaligned, these operations will cause a data
/// load fault.
pub trait AlignedSimdOps<
    T: SupersetOf<E>,
    E: AcceleratedScalar<F, LANES> + SupersetOf<F> + Copy,
    F: SimdScalar + Copy,
    const LANES: usize,
>
{
    /// Adds `self + rhs` and copies them into the output.
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    unsafe fn simd_add(a: &T, b: &T) -> T;

    /// Subtracts `self - rhs` and copies them into the output.
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    unsafe fn simd_sub(a: &T, b: &T) -> T;

    /// Multiplies `self * rhs` and copies them into the output.
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    unsafe fn simd_mul(a: &T, b: &T) -> T;

    /// Divides `self / rhs` and copies them into the output.
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    unsafe fn simd_div(a: &T, b: &T) -> T;

    /// Computes `self + a * b` and copies them into the output.
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    #[inline(always)]
    unsafe fn simd_mul_add(a: &T, b: &T, rhs: &T) -> T {
        unsafe { Self::simd_add(&Self::simd_mul(a, b), rhs) }
    }

    /// Computes the multiplicative inverse of `self` and copies that into the output.
    ///
    /// Formula: `1 / self`
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    #[inline(always)]
    unsafe fn simd_inv(a: &T) -> T {
        unsafe { Self::simd_div(&T::from_subset(E::from_subset(F::one())), a) }
    }

    /// Computes the negation of `self` and copies that into the output.
    ///
    /// Formula: `0 - self`
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    #[inline(always)]
    unsafe fn simd_neg(a: &T) -> T {
        unsafe { Self::simd_sub(&T::from_subset(E::from_subset(F::zero())), a) }
    }
}

/// A wrapping trait around [`UnalignedSimdOps`] that removes redundant associated functions.
pub trait NonAssociativeSimd<
    E: AcceleratedScalar<F, LANES> + SupersetOf<F> + Copy,
    F: SimdScalar + Copy,
    const LANES: usize,
>: SupersetOf<E> + Sized
{
    /// The low level backend that makes no assumption that `Self` is aligned.
    type Backend: UnalignedSimdOps<Self, E, F, LANES>;

    /// Adds `self + rhs` and copies them into the output.
    #[must_use]
    #[inline(always)]
    fn simd_add(&self, rhs: &Self) -> Self {
        Self::Backend::simd_add(self, rhs)
    }

    /// Subtracts `self - rhs` and copies them into the output.
    #[must_use]
    #[inline(always)]
    fn simd_sub(&self, rhs: &Self) -> Self {
        Self::Backend::simd_sub(self, rhs)
    }

    /// Multiplies `self 8 rhs` and copies them into the output.
    #[must_use]
    #[inline(always)]
    fn simd_mul(&self, rhs: &Self) -> Self {
        Self::Backend::simd_mul(self, rhs)
    }

    /// Divides `self / rhs` and copies them into the output.
    #[must_use]
    #[inline(always)]
    fn simd_div(&self, rhs: &Self) -> Self {
        Self::Backend::simd_div(self, rhs)
    }

    /// Computes the multiplicative inverse of `self` and copies that into the output.
    ///
    /// Formula: `1 / self`
    #[must_use]
    #[inline(always)]
    fn simd_inv(&self) -> Self {
        Self::Backend::simd_inv(self)
    }

    /// Computes the negation of `self` and copies that into the output.
    ///
    /// Formula: `0 - self`
    #[must_use]
    #[inline(always)]
    fn simd_neg(&self) -> Self {
        Self::Backend::simd_neg(self)
    }

    /// Computes `self + a * b` and copies them into the output.
    #[must_use]
    #[inline(always)]
    fn simd_mul_add(&self, a: &Self, b: &Self) -> Self {
        Self::Backend::simd_mul_add(self, a, b)
    }
}

/// Variant of [`NonAssociativeSimd`] that requires alignment.
///
/// These operations are more efficient than their unaligned counterpart, but
/// they often aren't as safe. If unaligned, these operations will cause a data
/// load fault.
pub trait AlignedSimd<
    E: AcceleratedScalar<F, LANES> + SupersetOf<F> + Copy,
    F: SimdScalar + Copy,
    const LANES: usize,
>: SupersetOf<E> + Sized
{
    /// The low level backend used on type `Self`.
    type Backend: AlignedSimdOps<Self, E, F, LANES>;

    /// Adds `self + rhs` and copies them into the output.
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    #[inline(always)]
    unsafe fn simd_add(&self, rhs: &Self) -> Self {
        unsafe { Self::Backend::simd_add(self, rhs) }
    }

    /// Subtracts `self - rhs` and copies them into the output.
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    #[inline(always)]
    unsafe fn simd_sub(&self, rhs: &Self) -> Self {
        unsafe { Self::Backend::simd_sub(self, rhs) }
    }

    /// Multipliess `self * rhs` and copies them into the output.
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    #[inline(always)]
    unsafe fn simd_mul(&self, rhs: &Self) -> Self {
        unsafe { Self::Backend::simd_mul(self, rhs) }
    }

    /// Divides `self / rhs` and copies them into the output.
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    #[inline(always)]
    unsafe fn simd_div(&self, rhs: &Self) -> Self {
        unsafe { Self::Backend::simd_div(self, rhs) }
    }

    /// Computes the multiplicative inverse of `self` and copies that into the output.
    ///
    /// Formula: `1 / self`
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    #[inline(always)]
    unsafe fn simd_inv(&self) -> Self {
        unsafe { Self::Backend::simd_inv(self) }
    }

    /// Computes the negation of `self` and copies that into the output.
    ///
    /// Formula: `0 - self`
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    #[inline(always)]
    unsafe fn simd_neg(&self) -> Self {
        unsafe { Self::Backend::simd_neg(self) }
    }

    /// Computes `self + a * b` and copies them into the output.
    ///
    /// # Safety
    ///
    /// This function assumes proper alignment; it an immediate
    /// data load fault at the CPU level if called on unaligned
    /// input.
    #[must_use]
    #[inline(always)]
    unsafe fn simd_mul_add(&self, a: &Self, b: &Self) -> Self {
        unsafe { Self::Backend::simd_mul_add(self, a, b) }
    }
}

/// Plain backend for single precision floats.
pub struct SinglePrecisionNoSimd<const N: usize>;

impl<const N: usize> ImplicitSupersetOf<f32> for SinglePrecisionNoSimd<N> {}
impl<const N: usize> AcceleratedScalar<f32, N> for SinglePrecisionNoSimd<N> {}

impl<const N: usize> SinglePrecisionNoSimd<N> {
    /// Lane count of the vector.
    pub const SIZE: usize = N;
}

impl<const N: usize> SimdAdd for SinglePrecisionNoSimd<N> {
    type Set = [f32; N];

    fn simd_add(a: &Self::Set, b: &Self::Set) -> Self::Set {
        let mut out = unsafe { core::mem::zeroed::<Self::Set>() };
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] + b[i];
        }
        out
    }

    #[inline(always)]
    fn simd_add_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] + b[i];
        }
    }
}

impl<const N: usize> SimdSub for SinglePrecisionNoSimd<N> {
    type Set = [f32; N];

    fn simd_sub(a: &Self::Set, b: &Self::Set) -> Self::Set {
        let mut out = unsafe { core::mem::zeroed::<Self::Set>() };
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] - b[i];
        }
        out
    }

    #[inline(always)]
    fn simd_sub_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] - b[i];
        }
    }
}

impl<const N: usize> SimdMul for SinglePrecisionNoSimd<N> {
    type Set = [f32; N];

    fn simd_mul(a: &Self::Set, b: &Self::Set) -> Self::Set {
        let mut out = unsafe { core::mem::zeroed::<Self::Set>() };
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] * b[i];
        }
        out
    }

    #[inline(always)]
    fn simd_mul_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] * b[i];
        }
    }
}

impl<const N: usize> SimdDiv for SinglePrecisionNoSimd<N> {
    type Set = [f32; N];

    fn simd_div(a: &Self::Set, b: &Self::Set) -> Self::Set {
        let mut out = unsafe { core::mem::zeroed::<Self::Set>() };
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] / b[i];
        }
        out
    }

    #[inline(always)]
    fn simd_div_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] / b[i];
        }
    }
}

impl<const N: usize> AlignedSimdOps<[f32; N], [f32; N], f32, N> for SinglePrecisionNoSimd<N> {
    unsafe fn simd_add(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        <Self as SimdAdd>::simd_add(a, b)
    }

    unsafe fn simd_sub(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        <Self as SimdSub>::simd_sub(a, b)
    }

    unsafe fn simd_mul(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        <Self as SimdMul>::simd_mul(a, b)
    }

    unsafe fn simd_div(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
        <Self as SimdDiv>::simd_div(a, b)
    }
}

/// Plain backend for double precision floats.
pub struct DoublePrecisionNoSimd<const N: usize>;

impl<const N: usize> ImplicitSupersetOf<f64> for DoublePrecisionNoSimd<N> {}
impl<const N: usize> AcceleratedScalar<f64, N> for DoublePrecisionNoSimd<N> {}

impl<const N: usize> DoublePrecisionNoSimd<N> {
    /// Lane count of the vector.
    pub const SIZE: usize = N;
}

impl<const N: usize> SimdAdd for DoublePrecisionNoSimd<N> {
    type Set = [f64; N];

    fn simd_add(a: &Self::Set, b: &Self::Set) -> Self::Set {
        let mut out = unsafe { core::mem::zeroed::<Self::Set>() };
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] + b[i];
        }
        out
    }

    #[inline(always)]
    fn simd_add_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] + b[i];
        }
    }
}

impl<const N: usize> SimdSub for DoublePrecisionNoSimd<N> {
    type Set = [f64; N];

    fn simd_sub(a: &Self::Set, b: &Self::Set) -> Self::Set {
        let mut out = unsafe { core::mem::zeroed::<Self::Set>() };
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] - b[i];
        }
        out
    }

    #[inline(always)]
    fn simd_sub_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] - b[i];
        }
    }
}

impl<const N: usize> SimdMul for DoublePrecisionNoSimd<N> {
    type Set = [f64; N];

    fn simd_mul(a: &Self::Set, b: &Self::Set) -> Self::Set {
        let mut out = unsafe { core::mem::zeroed::<Self::Set>() };
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] * b[i];
        }
        out
    }

    #[inline(always)]
    fn simd_mul_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] * b[i];
        }
    }
}

impl<const N: usize> SimdDiv for DoublePrecisionNoSimd<N> {
    type Set = [f64; N];

    fn simd_div(a: &Self::Set, b: &Self::Set) -> Self::Set {
        let mut out = unsafe { core::mem::zeroed::<Self::Set>() };
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] / b[i];
        }
        out
    }

    #[inline(always)]
    fn simd_div_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
        for (i, val) in out.iter_mut().enumerate() {
            *val = a[i] / b[i];
        }
    }
}

impl<const N: usize> AlignedSimdOps<[f64; N], [f64; N], f64, N> for DoublePrecisionNoSimd<N> {
    unsafe fn simd_add(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
        <Self as SimdAdd>::simd_add(a, b)
    }

    unsafe fn simd_sub(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
        <Self as SimdSub>::simd_sub(a, b)
    }

    unsafe fn simd_mul(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
        <Self as SimdMul>::simd_mul(a, b)
    }

    unsafe fn simd_div(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
        <Self as SimdDiv>::simd_div(a, b)
    }
}
