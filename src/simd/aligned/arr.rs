use core::{
    array::TryFromSliceError,
    mem::MaybeUninit,
    ops::{
        Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
        RangeToInclusive,
    },
    ptr::from_ref,
};
use num_traits::{Inv, MulAdd, One, Zero};
use crate::scalar::{AddByRef, DivByRef, MulByRef, Primitive, SubByRef, SupersetOf};
use crate::simd::{
    backend::{self, AlignedSimd, NonAssociativeSimd},
    LaneCount, SimdElement, SupportedLaneCount,
};
use super::{SimdAligned, SimdSlice};

pub struct TryFromUnalignedError;

/// A statically sized vector with `N` elements of type `T`, batched into chunks of size `LANES`.
///
/// Every operation performed on a `dyn Any` when wrapped in a `Simd` is accelerated and vectorized
/// using aligned SIMD loads and mathematical operators where `Self` is the operand. This massively
/// improves performance over naive plain arithmetic, but it comes at a cost.
///
/// There are two main downsides:
///
/// 1. It is wasteful; the alignment is usually between 16 and 64 bytes, so it must copy all it's
///    vectors before use and can waste some room near the tail.
/// 2. It has strict trait bounds. `LANES` must be a [`SupportedLaneCount`], `[T; LANES]` must implement
///    [`AlignedSimd`], and `T` must be a primitive scalar.
///
/// Beyond that, the use of this abstraction is very convenient. It defeats [`crate::simd::add_arrays`]
/// and similar in raw performance, power efficiency, and flexibility. The only place where such functions
/// can defeat this abstraction, is in *memory efficiency*. The caveat explained above, padding and copying
/// arrays at construction, can be noticeable as the vectors get larger. This is an unavoidable pitfall of
/// using aligned SIMD acceleration.
#[repr(transparent)]
#[derive(Debug, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Simd<T, const N: usize, const LANES: usize = { crate::MAX_SIMD_SINGLE_PRECISION_LANES }>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    vector: [T; N],
}

impl<T, const N: usize, const LANES: usize> PartialEq<SimdSlice<T, LANES>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES> + NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn eq(&self, other: &SimdSlice<T, LANES>) -> bool {
        other.eq(self)
    }
}

impl<T, const N: usize, const LANES: usize> PartialEq<SimdSlice<T, LANES>> for &Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES> + NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn eq(&self, other: &SimdSlice<T, LANES>) -> bool {
        other.eq(*self)
    }
}

impl<T, const N: usize, const LANES: usize> PartialEq<SimdSlice<T, LANES>> for &mut Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES> + NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn eq(&self, other: &SimdSlice<T, LANES>) -> bool {
        other.eq(*self)
    }
}

impl<T, const N: usize, const LANES: usize> PartialEq<&mut SimdSlice<T, LANES>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES> + NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn eq(&self, other: &&mut SimdSlice<T, LANES>) -> bool {
        other.eq(self)
    }
}

impl<T, const N: usize, const LANES: usize> Clone for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        Self::new(*self.as_array())
    }

    #[inline(always)]
    fn clone_from(&mut self, source: &Self) {
        self.vector = *source.as_array();
    }
}

impl<T, const N: usize, const LANES: usize> Index<Range<usize>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = SimdSlice<T, LANES>;

    #[inline(always)]
    fn index(&self, index: Range<usize>) -> &Self::Output {
        SimdSlice::from_slice(&self.vector[index])
    }
}

impl<T, const N: usize, const LANES: usize> IndexMut<Range<usize>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        SimdSlice::from_slice_mut(&mut self.vector[index])
    }
}

impl<T, const N: usize, const LANES: usize> Index<RangeFrom<usize>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = SimdSlice<T, LANES>;

    #[inline(always)]
    fn index(&self, index: RangeFrom<usize>) -> &Self::Output {
        SimdSlice::from_slice(&self.vector[index])
    }
}

impl<T, const N: usize, const LANES: usize> IndexMut<RangeFrom<usize>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut Self::Output {
        SimdSlice::from_slice_mut(&mut self.vector[index])
    }
}

impl<T, const N: usize, const LANES: usize> Index<RangeFull> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = SimdSlice<T, LANES>;

    #[inline(always)]
    fn index(&self, _: RangeFull) -> &Self::Output {
        SimdSlice::from_slice(self.as_array())
    }
}

impl<T, const N: usize, const LANES: usize> IndexMut<RangeFull> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn index_mut(&mut self, _: RangeFull) -> &mut Self::Output {
        SimdSlice::from_slice_mut(self.as_mut_array())
    }
}

impl<T, const N: usize, const LANES: usize> Index<RangeInclusive<usize>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = SimdSlice<T, LANES>;

    #[inline(always)]
    fn index(&self, index: RangeInclusive<usize>) -> &Self::Output {
        SimdSlice::from_slice(&self.vector[index])
    }
}

impl<T, const N: usize, const LANES: usize> IndexMut<RangeInclusive<usize>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: RangeInclusive<usize>) -> &mut Self::Output {
        SimdSlice::from_slice_mut(&mut self.vector[index])
    }
}

impl<T, const N: usize, const LANES: usize> Index<RangeTo<usize>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = SimdSlice<T, LANES>;

    #[inline(always)]
    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        SimdSlice::from_slice(&self.vector[index])
    }
}

impl<T, const N: usize, const LANES: usize> IndexMut<RangeTo<usize>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut Self::Output {
        SimdSlice::from_slice_mut(&mut self.vector[index])
    }
}

impl<T, const N: usize, const LANES: usize> Index<RangeToInclusive<usize>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = SimdSlice<T, LANES>;

    #[inline(always)]
    fn index(&self, index: RangeToInclusive<usize>) -> &Self::Output {
        SimdSlice::from_slice(&self.vector[index])
    }
}

impl<T, const N: usize, const LANES: usize> IndexMut<RangeToInclusive<usize>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: RangeToInclusive<usize>) -> &mut Self::Output {
        SimdSlice::from_slice_mut(&mut self.vector[index])
    }
}

impl<T, const N: usize, const LANES: usize> Index<usize> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.as_array().index(index)
    }
}

impl<T, const N: usize, const LANES: usize> IndexMut<usize> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.as_mut_array().index_mut(index)
    }
}

impl<T, const N: usize, const LANES: usize> SupersetOf<[T; N]> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn from_subset(scalar: [T; N]) -> Self {
        Self::new(scalar)
    }

    #[inline(always)]
    fn into_subset(self) -> [T; N] {
        self.to_array()
    }
}

impl<T, const N: usize, const LANES: usize> SupersetOf<T> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn from_subset(scalar: T) -> Self {
        Self::splat(scalar)
    }

    #[inline(always)]
    fn into_subset(self) -> T {
        self.vector[0]
    }
}

impl<T, const N: usize, const LANES: usize> Default for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn default() -> Self {
        Self::splat(T::default())
    }
}

impl<T, const N: usize, const LANES: usize> Zero for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Zero + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn zero() -> Self {
        Self::splat(T::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.as_array().iter().all(|val| val.is_zero())
    }
}

impl<T, const N: usize, const LANES: usize> One for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + One + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn one() -> Self {
        Self::splat(T::one())
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.as_array().iter().all(|val| val.is_one())
    }
}

impl<T, const N: usize, const LANES: usize> Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    /// The amount of SIMD lanes used when an operation is performed on `Self`.
    pub const LANES: usize = LANES;

    /// The total length of the vector, where all aligned data is stored.
    pub const LEN: usize = N;

    /// Generates a vector from a function to convert an index to an element.
    #[inline(always)]
    pub fn from_fn<F: FnMut(usize) -> T>(f: F) -> Self {
        Self {
            vector: core::array::from_fn(f),
        }
    }

    /// Constructs a new vector from an array and copies it to ensure alignment.
    #[inline(always)]
    pub const fn new(vector: [T; N]) -> Self {
        Self {
            vector: vector,
        }
    }

    /// Constructs a new vector by copying `scalar` `N` times.
    #[inline(always)]
    pub const fn splat(scalar: T) -> Self {
        Self {
            vector: [scalar; N],
        }
    }

    /// Returns `&self` coerced as an array `&[T; N]`.
    #[inline(always)]
    pub const fn as_array(&self) -> &[T; N] {
        &self.vector
    }

    /// Returns `&mut self` coerced as an array `&mut [T; N]`.
    #[inline(always)]
    pub const fn as_mut_array(&mut self) -> &mut [T; N] {
        &mut self.vector
    }

    /// Copies the contents of `self` into an array and moves out of the vector.
    #[inline(always)]
    pub const fn to_array(self) -> [T; N] {
        self.vector
    }

    /// Copies exactly [`Self::LEN`] elements into the slice.
    ///
    /// # Panics
    ///
    /// Panics if the length of the slice is less than that of the
    /// SIMD vector on an instance of `self`.
    #[inline(always)]
    pub const fn copy_to_slice(self, slice: &mut [T]) {
        assert!(
            slice.len() >= Self::LEN,
            "slice length must be at least the number of elements"
        );
        unsafe {
            self.copy_to_slice_unchecked(slice);
        }
    }

    /// Variant of [`Self::copy_to_slice`] stripped of all safety checks.
    ///
    /// # Safety
    ///
    /// This is unsafe anywhere the safe variant panics.
    #[inline(always)]
    pub const unsafe fn copy_to_slice_unchecked(self, slice: &mut [T]) {
        let tmp = self;
        unsafe {
            core::ptr::copy_nonoverlapping(from_ref(tmp.as_array()), slice.as_mut_ptr().cast(), 1)
        }
    }

    /// Copies at most [`Self::LEN`] elements into the slice until it ends.
    #[inline(always)]
    pub fn copy_to_end(self, slice: &mut [T]) {
        for (a, b) in self.to_array().into_iter().zip(slice) {
            *b = a;
        }
    }

    /// Loads from the slice until it ends, falling back to `or`.
    #[inline(always)]
    pub fn load_from_either(slice: &[T], or: Self) -> Self {
        let array = {
            let mut buf = MaybeUninit::<[T; N]>::uninit();
            let backup = or.to_array();
            let ptr = buf.as_mut_ptr().cast::<T>();
            #[allow(clippy::needless_range_loop)] // it allows more for optimizations as a range
            for i in 0..Self::LEN {
                unsafe {
                    ptr.add(i).write({
                        if let Some(&val) = slice.get(i) {
                            val
                        } else {
                            backup[i]
                        }
                    });
                }
            }
            unsafe { buf.assume_init() }
        };
        Self::new(array)
    }

    /// Loads from the slice until it ends, falling back to `default`.
    #[inline(always)]
    pub fn load_or(slice: &[T], default: T) -> Self {
        let array = {
            let mut buf = [default; N];
            slice
                .iter()
                .take(Self::LEN)
                .enumerate()
                .for_each(|(i, &val)| {
                    buf[i] = val;
                });
            buf
        };
        Self::new(array)
    }

    /// Loads from the slice until it ends, falling back to the default value of `T`.
    #[inline(always)]
    pub fn load_or_default(self, slice: &[T]) -> Self
    where
        T: Default,
    {
        Self::new(<[T; N] as crate::scalar::ArrayOf<T>>::pad_to(slice))
    }
}

impl<T, const N: usize, const LANES: usize> AddByRef<Self> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn as_add(&self, rhs: &Self) -> Self::Output {
        <Self as crate::scalar::SimdAdd>::simd_add(self.as_array(), rhs.as_array()).into()
    }

    #[inline(always)]
    fn add_into(&self, rhs: &Self, out: &mut Self::Output) {
        *out = self.as_add(rhs);
    }
}

impl<T, const N: usize, const LANES: usize> AddByRef<T> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn as_add(&self, rhs: &T) -> Self::Output {
        self.as_add(&Self::new([*rhs; N]))
    }

    #[inline(always)]
    fn add_into(&self, rhs: &T, out: &mut Self::Output) {
        *out = self.as_add(rhs);
    }
}

impl<T, const N: usize, const LANES: usize> SubByRef<Self> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn as_sub(&self, rhs: &Self) -> Self::Output {
        <Self as crate::scalar::SimdSub>::simd_sub(self.as_array(), rhs.as_array()).into()
    }

    #[inline(always)]
    fn sub_into(&self, rhs: &Self, out: &mut Self::Output) {
        *out = self.as_sub(rhs);
    }
}

impl<T, const N: usize, const LANES: usize> SubByRef<T> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn as_sub(&self, rhs: &T) -> Self::Output {
        self.as_sub(&Self::new([*rhs; N]))
    }

    #[inline(always)]
    fn sub_into(&self, rhs: &T, out: &mut Self::Output) {
        *out = self.as_sub(rhs);
    }
}

impl<T, const N: usize, const LANES: usize> MulByRef<Self> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn as_mul(&self, rhs: &Self) -> Self::Output {
        <Self as crate::scalar::SimdMul>::simd_mul(self.as_array(), rhs.as_array()).into()
    }

    #[inline(always)]
    fn mul_into(&self, rhs: &Self, out: &mut Self::Output) {
        *out = self.as_mul(rhs);
    }
}

impl<T, const N: usize, const LANES: usize> MulByRef<T> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn as_mul(&self, rhs: &T) -> Self::Output {
        self.as_mul(&Self::new([*rhs; N]))
    }

    #[inline(always)]
    fn mul_into(&self, rhs: &T, out: &mut Self::Output) {
        *out = self.as_mul(rhs);
    }
}

impl<T, const N: usize, const LANES: usize> DivByRef<Self> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn as_div(&self, rhs: &Self) -> Self::Output {
        <Self as crate::scalar::SimdDiv>::simd_div(self.as_array(), rhs.as_array()).into()
    }

    #[inline(always)]
    fn div_into(&self, rhs: &Self, out: &mut Self::Output) {
        *out = self.as_div(rhs);
    }
}

impl<T, const N: usize, const LANES: usize> DivByRef<T> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn as_div(&self, rhs: &T) -> Self::Output {
        self.as_div(&Self::new([*rhs; N]))
    }

    #[inline(always)]
    fn div_into(&self, rhs: &T, out: &mut Self::Output) {
        *out = self.as_div(rhs);
    }
}

impl<T, const N: usize, const LANES: usize> From<[T; N]> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn from(value: [T; N]) -> Self {
        Self::new(value)
    }
}

impl<T, const N: usize, const LANES: usize> From<Simd<T, N, LANES>> for [T; N]
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn from(value: Simd<T, N, LANES>) -> Self {
        value.to_array()
    }
}

impl<T, const N: usize, const LANES: usize> TryFrom<&[T]> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Error = TryFromSliceError;

    #[inline(always)]
    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        Ok(Self::new(value.try_into()?))
    }
}

impl<T, const N: usize, const LANES: usize> TryFrom<&mut [T]> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Error = TryFromSliceError;

    #[inline(always)]
    fn try_from(value: &mut [T]) -> Result<Self, Self::Error> {
        Ok(Self::new(value.try_into()?))
    }
}

impl<T, const N: usize, const LANES: usize> AsRef<[T; N]> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn as_ref(&self) -> &[T; N] {
        self.as_array()
    }
}

impl<T, const N: usize, const LANES: usize> AsMut<[T; N]> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T; N] {
        self.as_mut_array()
    }
}

impl<T, const N: usize, const LANES: usize> AsRef<[T]> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self.as_array()
    }
}

impl<T, const N: usize, const LANES: usize> AsMut<[T]> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_array()
    }
}

impl<T, const N: usize, const LANES: usize> TryFrom<&SimdSlice<T, LANES>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    type Error = TryFromUnalignedError;

    #[inline(always)]
    fn try_from(value: &SimdSlice<T, LANES>) -> Result<Self, Self::Error> {
        if value.len() != N {
            return Err(TryFromUnalignedError);
        }

        let array = {
            let mut tmp = MaybeUninit::<[T; N]>::uninit();
            let ptr = tmp.as_mut_ptr().cast::<T>();
            unsafe {
                core::ptr::copy_nonoverlapping(value.as_ptr(), ptr, N);
            };
            unsafe { tmp.assume_init() }
        };

        Ok(Self::new(array))
    }
}

impl<T, const N: usize, const LANES: usize> TryFrom<&mut SimdSlice<T, LANES>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    type Error = TryFromUnalignedError;

    #[inline(always)]
    fn try_from(value: &mut SimdSlice<T, LANES>) -> Result<Self, Self::Error> {
        if value.len() != N {
            return Err(TryFromUnalignedError);
        }

        let array = {
            let mut tmp = MaybeUninit::<[T; N]>::uninit();
            let ptr = tmp.as_mut_ptr().cast::<T>();
            unsafe {
                core::ptr::copy_nonoverlapping(value.as_ptr(), ptr, N);
            };
            unsafe { tmp.assume_init() }
        };

        Ok(Self::new(array))
    }
}

impl<T, const N: usize, const LANES: usize> AsRef<SimdSlice<T, LANES>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn as_ref(&self) -> &SimdSlice<T, LANES> {
        SimdSlice::from_slice(self.as_array())
    }
}

impl<T, const N: usize, const LANES: usize> AsMut<SimdSlice<T, LANES>> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn as_mut(&mut self) -> &mut SimdSlice<T, LANES> {
        SimdSlice::from_slice_mut(self.as_mut_array())
    }
}

impl<T, const N: usize, const LANES: usize> Deref for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Target = [T; N];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T, const N: usize, const LANES: usize> DerefMut for Simd<T, N, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

macro_rules! impl_core_ops {
    (
        impl ($(
            {
                @intrinsic $op:ident

                $simd_trait:ident {
                    fn $simd_fn:ident {}
                    fn $simd_unsize_fn:ident {}
                }

                $core_trait:ident {
                    fn $core_fn:ident {}
                }

                $assign_trait:ident {
                    fn $assign_fn:ident {}
                }
            }$(,)?
        )*) for Simd {}
    ) => {
        $(
            impl<T, const N: usize, const LANES: usize> $crate::scalar::$simd_trait for Simd<T, N, LANES>
            where
                T: SimdElement + Primitive + Default,
                LaneCount<LANES>: SupportedLaneCount,
                [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
            {
                type Set = [T; N];

                #[inline(always)]
                fn $simd_fn (a: &Self::Set, b: &Self::Set) -> Self::Set {
                    const {
                        core::assert!(size_of::<[T; LANES]>() % align_of::<Self>() == 0, "invalid SIMD lane size");
                    }

                    let mut out = MaybeUninit::<SimdAligned<[T; N]>>::uninit();
                    for i in (const { 0..(N - (N % LANES)) }).step_by(LANES) {
                        let a_chunk = unsafe { SimdAligned(a.as_ptr().add(i).cast::<[T; LANES]>().read()) };
                        let b_chunk = unsafe { SimdAligned(b.as_ptr().add(i).cast::<[T; LANES]>().read()) };

                        // SAFETY: SIMD guarantees alignment
                        let val = unsafe { <[T; LANES] as AlignedSimd<[T; LANES], T, LANES>>::$op(&(&a_chunk).0, &(&b_chunk).0) };
                        unsafe { out.as_mut_ptr().cast::<T>().add(i).cast::<[T; LANES]>().write(val) };
                    }
                    if const { (N - (N % LANES)) != N } {
                        let a_chunk = SimdAligned(<[T; LANES] as $crate::scalar::ArrayOf<T>>::pad_to(&a[const { N - (N % LANES).. }]));
                        let b_chunk = SimdAligned(<[T; LANES] as $crate::scalar::ArrayOf<T>>::pad_to(&b[const { N - (N % LANES).. }]));
                        let val = unsafe { &<[T; LANES] as AlignedSimd<[T; LANES], T, LANES>>::$op(&(&a_chunk).0, &(&b_chunk).0)[const { ..N % LANES }] };
                        let ptr = unsafe { out.as_mut_ptr().cast::<T>().add(const { N - (N % LANES) }) };
                        for offset in const { 0..(N % LANES) } {
                            unsafe {
                                ptr.add(offset).write(val[offset]);
                            }
                        }
                    }
                    unsafe { out.assume_init().0 }
                }

                #[inline(always)]
                fn $simd_unsize_fn (a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
                    const {
                        core::assert!(size_of::<[T; LANES]>() % align_of::<Self>() == 0, "invalid SIMD lane size");
                    }

                    for i in (const { 0..(N - (N % LANES)) }).step_by(LANES) {
                        let a_chunk = unsafe { SimdAligned(a.as_ptr().add(i).cast::<[T; LANES]>().read()) };
                        let b_chunk = unsafe { SimdAligned(b.as_ptr().add(i).cast::<[T; LANES]>().read()) };

                        // SAFETY: SIMD guarantees alignment
                        let val = unsafe { <[T; LANES] as AlignedSimd<[T; LANES], T, LANES>>::$op(&(&a_chunk).0, &(&b_chunk).0) };
                        out[i..(i + LANES)].copy_from_slice(&val);
                    }

                    if const { (N - (N % LANES)) != N } {
                        let a_chunk = SimdAligned(<[T; LANES] as $crate::scalar::ArrayOf<T>>::pad_to(&a[const { N - (N % LANES).. }]));
                        let b_chunk = SimdAligned(<[T; LANES] as $crate::scalar::ArrayOf<T>>::pad_to(&b[const { N - (N % LANES).. }]));
                        let val = unsafe { &<[T; LANES] as AlignedSimd<[T; LANES], T, LANES>>::$op(&(&a_chunk).0, &(&b_chunk).0)[const { ..N % LANES }] };
                        for offset in const { 0..(N % LANES) } {
                            out[const { N - (N % LANES) } + offset] = val[offset];
                        }
                    }
                }
            }

            impl<T, const N: usize, const LANES: usize> core::ops::$core_trait<Self> for Simd<T, N, LANES>
            where
                T: SimdElement + Primitive + Default,
                LaneCount<LANES>: SupportedLaneCount,
                [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
            {
                type Output = Self;

                #[inline(always)]
                fn $core_fn (self, rhs: Self) -> Self::Output {
                    <Self as crate::scalar::$simd_trait>::$simd_fn(self.as_array(), rhs.as_array()).into()
                }
            }

            impl<T, const N: usize, const LANES: usize> core::ops::$core_trait<T> for Simd<T, N, LANES>
            where
                T: SimdElement + Primitive + Default,
                LaneCount<LANES>: SupportedLaneCount,
                [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
            {
                type Output = Self;

                #[inline(always)]
                fn $core_fn (self, rhs: T) -> Self::Output {
                    <Self as crate::scalar::$simd_trait>::$simd_fn(self.as_array(), &[rhs; N]).into()
                }
            }

            impl<T, const N: usize, const LANES: usize> core::ops::$assign_trait<T> for Simd<T, N, LANES>
            where
                T: SimdElement + Primitive + Default,
                LaneCount<LANES>: SupportedLaneCount,
                [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
            {
                #[inline(always)]
                fn $assign_fn (&mut self, rhs: T) {
                    *self = <Self as crate::scalar::$simd_trait>::$simd_fn(self.as_array(), &[rhs; N]).into();
                }
            }

            impl<T, const N: usize, const LANES: usize> core::ops::$assign_trait<Self> for Simd<T, N, LANES>
            where
                T: SimdElement + Primitive + Default,
                LaneCount<LANES>: SupportedLaneCount,
                [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
            {
                #[inline(always)]
                fn $assign_fn (&mut self, rhs: Self) {
                    *self = <Self as crate::scalar::$simd_trait>::$simd_fn(self.as_array(), rhs.as_array()).into();
                }
            }
        )*
    };
}

impl_core_ops!(
    impl (
        {
            @intrinsic simd_add
            SimdAdd {
                fn simd_add {}
                fn simd_add_into {}
            }
            Add {
                fn add {}
            }
            AddAssign {
                fn add_assign {}
            }
        },
        {
            @intrinsic simd_mul
            SimdMul {
                fn simd_mul {}
                fn simd_mul_into {}
            }
            Mul {
                fn mul {}
            }
            MulAssign {
                fn mul_assign {}
            }
        },
        {
            @intrinsic simd_sub
            SimdSub {
                fn simd_sub {}
                fn simd_sub_into {}
            }
            Sub {
                fn sub {}
            }
            SubAssign {
                fn sub_assign {}
            }
        },
        {
            @intrinsic simd_div
            SimdDiv {
                fn simd_div {}
                fn simd_div_into {}
            }
            Div {
                fn div {}
            }
            DivAssign {
                fn div_assign {}
            }
        },
    ) for Simd {}
);

impl<T, const N: usize, const LANES: usize>
    crate::simd::backend::AlignedSimdOps<[T; N], [T; N], T, N> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    #[inline(always)]
    unsafe fn simd_add(a: &[T; N], b: &[T; N]) -> [T; N] {
        <Self as crate::scalar::SimdAdd>::simd_add(a, b)
    }

    #[inline(always)]
    unsafe fn simd_sub(a: &[T; N], b: &[T; N]) -> [T; N] {
        <Self as crate::scalar::SimdSub>::simd_sub(a, b)
    }

    #[inline(always)]
    unsafe fn simd_mul(a: &[T; N], b: &[T; N]) -> [T; N] {
        <Self as crate::scalar::SimdMul>::simd_mul(a, b)
    }

    #[inline(always)]
    unsafe fn simd_div(a: &[T; N], b: &[T; N]) -> [T; N] {
        <Self as crate::scalar::SimdDiv>::simd_div(a, b)
    }
}

impl<T, const N: usize, const LANES: usize> core::ops::Neg for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe {
            <Self as backend::AlignedSimdOps<[T; N], [T; N], T, N>>::simd_neg(self.as_array())
                .into()
        }
    }
}

impl<T, const N: usize, const LANES: usize> Inv for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn inv(self) -> Self::Output {
        unsafe {
            <Self as backend::AlignedSimdOps<[T; N], [T; N], T, N>>::simd_inv(self.as_array())
                .into()
        }
    }
}

impl<T, const N: usize, const LANES: usize> MulAdd<Self, Self> for Simd<T, N, LANES>
where
    T: SimdElement + Primitive + Default,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: AlignedSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self::Output {
        // SAFETY: `Self` guarantees alignment through `SimdAligned`
        unsafe {
            <Self as backend::AlignedSimdOps<[T; N], [T; N], T, N>>::simd_mul_add(
                self.as_array(),
                a.as_array(),
                b.as_array(),
            )
            .into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{iter::*, marker::*, ops::*, *};

    fn ops_on<T, const N: usize>(arr1: [T; N], arr2: [T; N], op: fn(T, T) -> T) -> [T; N] {
        let mut out: MaybeUninit<[T; N]> = MaybeUninit::uninit();
        for (i, (val1, val2)) in arr1.into_iter().zip(arr2).enumerate() {
            unsafe {
                out.as_mut_ptr().cast::<T>().add(i).write(op(val1, val2));
            }
        }
        unsafe { out.assume_init() }
    }

    fn mul_add<T: MulAdd<Output = T> + Copy>(a: T, b: T) -> T {
        a.mul_add(b, b)
    }

    fn inv<T: Inv<Output = T>>(a: T, _: T) -> T {
        a.inv()
    }

    fn neg<T: Neg<Output = T>>(a: T, _: T) -> T {
        -a
    }

    #[test]
    fn big_buffer_simd() {
        for i in 0..1000 {
            let val = i as f32;
            let arr1 = Simd::new([
                val,
                val + 1.0,
                val + 2.0,
                val + 3.0,
                val + 4.0,
                val + 5.0,
                val + 6.0,
                val + 7.0,
                val + 8.0,
                val + 9.0,
                val + 10.0,
                val + 11.0,
                val + 12.0,
            ]);
            let arr2 = Simd::new([
                val + 12.0,
                val + 11.0,
                val + 10.0,
                val + 9.0,
                val + 8.0,
                val + 7.0,
                val + 6.0,
                val + 5.0,
                val + 4.0,
                val + 3.0,
                val + 2.0,
                val + 1.0,
                val,
            ]);

            let simd_add = *(arr1.clone() + arr2.clone());
            let norm_add = ops_on(*arr1, *arr2, f32::add);
            assert_eq!(simd_add, norm_add);

            let simd_sub = *(arr1.clone() - arr2.clone());
            let norm_sub = ops_on(*arr1, *arr2, f32::sub);
            assert_eq!(simd_sub, norm_sub);

            let simd_mul = *(arr1.clone() * arr2.clone());
            let norm_mul = ops_on(*arr1, *arr2, f32::mul);
            assert_eq!(simd_mul, norm_mul);

            let simd_div = *(arr1.clone() / arr2.clone());
            let norm_div = ops_on(*arr1, *arr2, f32::div);
            assert_eq!(simd_div, norm_div);

            let simd_mul_add = *(arr1.clone().mul_add(arr2.clone(), arr2.clone()));
            let norm_mul_add = ops_on(*arr1, *arr2, mul_add);
            assert_eq!(simd_mul_add, norm_mul_add);

            let simd_inv = *(arr1.clone().inv());
            let norm_inv = ops_on(*arr1, *arr2, inv);
            assert_eq!(simd_inv, norm_inv);
            let simd_inv_div = *(Simd::one() / arr1.clone());
            assert_eq!(simd_inv, simd_inv_div);

            let simd_neg = *(-arr1.clone());
            let norm_neg = ops_on(*arr1, *arr2, neg);
            assert_eq!(simd_neg, norm_neg);
            let simd_neg_sub = *(Simd::zero() - arr1);
            assert_eq!(simd_neg, simd_neg_sub);
        }
    }
}
