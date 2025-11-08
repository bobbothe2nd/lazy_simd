use crate::{
    scalar::{
        AddByRef, DivByRef, MulByRef, Primitive, SimdAdd, SimdDiv, SimdMul, SimdSub, SubByRef,
    },
    simd::{
        backend::{AlignedSimd, NonAssociativeSimd},
        LaneCount, Simd, SimdElement, SupportedLaneCount,
    },
};
use core::{
    ops::{
        Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
        RangeToInclusive,
    },
    ptr::{slice_from_raw_parts, slice_from_raw_parts_mut},
};

/// An unsized vector that implements SIMD-bound operations.
///
/// Details on such types can be found in the main type [`Simd`].
///
/// # Layout
///
/// This type is transparent over a slice of type `[T]` with the additional compile-time
/// metadata of `const LANES`.
#[repr(transparent)]
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct SimdSlice<T, const LANES: usize = { crate::MAX_SIMD_SINGLE_PRECISION_LANES }>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    vector: [T],
}

impl<T, const N: usize, const LANES: usize> PartialEq<Simd<T, N, LANES>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES> + AlignedSimd<[T; LANES], T, LANES>,
{
    fn eq(&self, other: &Simd<T, N, LANES>) -> bool {
        self.eq(&other[..])
    }
}

impl<T, const LANES: usize> Index<usize> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, const LANES: usize> IndexMut<usize> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<T, const LANES: usize> Index<Range<usize>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    fn index(&self, index: Range<usize>) -> &Self::Output {
        let len = index.end.wrapping_sub(index.start);
        let start = index.start % self.len();
        let len = len % self.len();

        let ptr = unsafe { self.as_ptr().add(start) };

        unsafe { Self::from_raw_parts(ptr, len) }
    }
}

impl<T, const LANES: usize> IndexMut<Range<usize>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        let len = index.end.wrapping_sub(index.start);
        let start = index.start % self.len();
        let len = len % self.len();

        let ptr = unsafe { self.as_mut_ptr().add(start) };

        unsafe { Self::from_raw_parts_mut(ptr, len) }
    }
}

impl<T, const LANES: usize> Index<RangeFrom<usize>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    fn index(&self, index: RangeFrom<usize>) -> &Self::Output {
        let start = index.start % self.len();
        let len = self.len().wrapping_sub(start) % self.len();

        let ptr = unsafe { self.as_ptr().add(start) };

        unsafe { Self::from_raw_parts(ptr, len) }
    }
}

impl<T, const LANES: usize> IndexMut<RangeFrom<usize>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut Self::Output {
        let start = index.start % self.len();
        let len = self.len().wrapping_sub(start) % self.len();

        let ptr = unsafe { self.as_mut_ptr().add(start) };

        unsafe { Self::from_raw_parts_mut(ptr, len) }
    }
}

impl<T, const LANES: usize> Index<RangeFull> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    fn index(&self, _: RangeFull) -> &Self::Output {
        self
    }
}

impl<T, const LANES: usize> IndexMut<RangeFull> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn index_mut(&mut self, _: RangeFull) -> &mut Self::Output {
        self
    }
}

impl<T, const LANES: usize> Index<RangeInclusive<usize>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    fn index(&self, index: RangeInclusive<usize>) -> &Self::Output {
        let len = (index.end().wrapping_add(1)).wrapping_sub(*index.start());
        let start = *index.start() % self.len();
        let len = len % self.len();

        let ptr = unsafe { self.as_ptr().add(start) };

        unsafe { Self::from_raw_parts(ptr, len) }
    }
}

impl<T, const LANES: usize> IndexMut<RangeInclusive<usize>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn index_mut(&mut self, index: RangeInclusive<usize>) -> &mut Self::Output {
        let len = (index.end().wrapping_add(1)).wrapping_sub(*index.start());
        let start = *index.start() % self.len();
        let len = len % self.len();

        let ptr = unsafe { self.as_mut_ptr().add(start) };

        unsafe { Self::from_raw_parts_mut(ptr, len) }
    }
}

impl<T, const LANES: usize> Index<RangeTo<usize>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        let len = index.end % self.len();

        let ptr = self.as_ptr();

        unsafe { Self::from_raw_parts(ptr, len) }
    }
}

impl<T, const LANES: usize> IndexMut<RangeTo<usize>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut Self::Output {
        let len = index.end % self.len();

        let ptr = self.as_mut_ptr();

        unsafe { Self::from_raw_parts_mut(ptr, len) }
    }
}

impl<T, const LANES: usize> Index<RangeToInclusive<usize>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Output = Self;

    fn index(&self, index: RangeToInclusive<usize>) -> &Self::Output {
        let len = (index.end.wrapping_add(1)) % self.len();

        let ptr = self.as_ptr();

        unsafe { Self::from_raw_parts(ptr, len) }
    }
}

impl<T, const LANES: usize> IndexMut<RangeToInclusive<usize>> for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn index_mut(&mut self, index: RangeToInclusive<usize>) -> &mut Self::Output {
        let len = (index.end.wrapping_add(1)) % self.len();

        let ptr = self.as_mut_ptr();

        unsafe { Self::from_raw_parts_mut(ptr, len) }
    }
}

impl<T, const LANES: usize> SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    /// The amount of SIMD lanes used when an operation is performed on `Self`.
    pub const LANES: usize = LANES;

    /// Transmutes the immutable slice to an immutable vector.
    pub const fn from_slice(slice: &[T]) -> &Self {
        unsafe { &*(slice as *const [T] as *const Self) }
    }

    /// Transmutes the mutable slice to a mutable vector.
    pub const fn from_slice_mut(slice: &mut [T]) -> &mut Self {
        unsafe { &mut *(slice as *mut [T] as *mut Self) }
    }

    /// Creates a vector from the core components.
    ///
    /// # Safety
    ///
    /// This has the same safety preconditions as [`core::slice::from_raw_parts`].
    ///
    /// To summarize, `ptr` must point to a valid array of `len` fully initialized values
    /// of type `T` for the lifetime of `'a` and the total size must not exceed that of
    /// [`isize::MAX`] (`size_of::<T>() * len`).
    pub const unsafe fn from_raw_parts<'a>(ptr: *const T, len: usize) -> &'a Self {
        let slice = slice_from_raw_parts(ptr, len);
        unsafe { &*(slice as *const Self) }
    }

    /// Variant of [`Self::from_raw_parts`] that operates on a mutable pointer.
    ///
    /// # Safety
    ///
    /// Details are included in the immutable counterpart.
    pub const unsafe fn from_raw_parts_mut<'a>(ptr: *mut T, len: usize) -> &'a mut Self {
        let slice = slice_from_raw_parts_mut(ptr, len);
        unsafe { &mut *(slice as *mut Self) }
    }

    /// Returns `&Self` coerced to a slice.
    ///
    /// This method is automatically performed on immutable dereference.
    pub const fn as_slice(&self) -> &[T] {
        &self.vector
    }

    /// Returns `&mut Self` coerced to a slice.
    ///
    /// This method is automatically performed on mutable dereference.
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.vector
    }
}

impl<T, const LANES: usize> Deref for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const LANES: usize> DerefMut for SimdSlice<T, LANES>
where
    T: SimdElement + Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

macro_rules! impl_core_ops {
    (
        impl ($(
            {
                @intrinsic $op:ident, $plain:ident, $scalar:ident

                $simd_trait:ident {
                    fn $simd_fn:ident {}
                    fn $simd_unsize_fn:ident {}
                }

                $by_ref_tr:ident {
                    fn $unreachable:ident {}
                    fn $op_fn:ident {}
                }
            }$(,)?
        )*) for Simd {}
    ) => {
        $(
            impl<T, const LANES: usize> $crate::scalar::$simd_trait for SimdSlice<T, LANES>
            where
                T: SimdElement + Primitive + Default,
                LaneCount<LANES>: SupportedLaneCount,
                [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
            {
                type Set = [T];

                fn $simd_fn (_: &Self::Set, _: &Self::Set) -> Self::Set
                where
                    Self::Set: Sized,
                {
                    unreachable!();
                }

                #[inline(always)]
                fn $simd_unsize_fn (a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
                    let len = a.len();
                    assert_eq!(len, b.len(), "input vectors must be of equal size");
                    assert!(len <= out.len(), "output vector must be of equal or greater size than input vectors");

                    for i in (0..(len - (len % LANES))).step_by(LANES) {
                        let a_chunk = unsafe { &*a.as_ptr().add(i).cast::<[T; LANES]>() };
                        let b_chunk = unsafe { &*b.as_ptr().add(i).cast::<[T; LANES]>() };

                        // SAFETY: SIMD guarantees alignment
                        let val = a_chunk.$op(b_chunk);
                        out[i..(i + LANES)].copy_from_slice(&val);
                    }

                    if (len - (len % LANES)) != len {
                        let a_chunk = &<[T; LANES] as $crate::scalar::ArrayOf<T>>::pad_to(&a[len - (len % LANES)..]);
                        let b_chunk = &<[T; LANES] as $crate::scalar::ArrayOf<T>>::pad_to(&b[len - (len % LANES)..]);
                        let val = &a_chunk.$op(&(&b_chunk))[..len % LANES];
                        for offset in 0..(len % LANES) {
                            out[len - (len % LANES) + offset] = val[offset];
                        }
                    }
                }
            }

            impl<T, const LANES: usize> $by_ref_tr <Self> for SimdSlice<T, LANES>
            where
                T: SimdElement + Primitive + Default,
                LaneCount<LANES>: SupportedLaneCount,
                [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
            {
                type Output = Self;

                fn $unreachable (&self, _: &Self) -> <Self as $by_ref_tr <T>>::Output
                    where
                        <Self as $by_ref_tr <T>>::Output: Sized
                {
                    unreachable!();
                }

                fn $op_fn(&self, rhs: &Self, out: &mut <Self as $by_ref_tr <T>>::Output) {
                    Self::$simd_unsize_fn(self.as_slice(), rhs.as_slice(), out.as_mut_slice());
                }
            }

            impl<T, const LANES: usize> $by_ref_tr <T> for SimdSlice<T, LANES>
            where
                T: SimdElement + Primitive + Default,
                LaneCount<LANES>: SupportedLaneCount,
                [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
            {
                type Output = Self;

                fn $unreachable (&self, _: &T) -> <Self as $by_ref_tr <T>>::Output
                    where
                        <Self as $by_ref_tr <T>>::Output: Sized
                {
                    unreachable!();
                }

                fn $op_fn(&self, rhs: &T, out: &mut <Self as $by_ref_tr <T>>::Output) {
                    let len = self.len();
                    assert!(len <= out.len(), "output vector must be of equal or greater size than input vector");

                    let b_chunk = [*rhs; LANES];

                    for i in (0..(len - (len % LANES))).step_by(LANES) {
                        let a_chunk = unsafe { &*self.as_ptr().add(i).cast::<[T; LANES]>() };

                        // SAFETY: SIMD guarantees alignment
                        let val = a_chunk.$op(&b_chunk);
                        out[i..(i + LANES)].copy_from_slice(&val);
                    }

                    if (len - (len % LANES)) != len {
                        let a_chunk = &<[T; LANES] as $crate::scalar::ArrayOf<T>>::pad_to(&self[len - (len % LANES)..]);
                        let val = &a_chunk.$op(&b_chunk)[..len % LANES];
                        for offset in 0..(len % LANES) {
                            out[len - (len % LANES) + offset] = val[offset];
                        }
                    }
                }
            }

            impl<T, const LANES: usize> SimdSlice<T, LANES>
            where
                T: SimdElement + Primitive + Default,
                LaneCount<LANES>: SupportedLaneCount,
                [T; LANES]: NonAssociativeSimd<[T; LANES], T, LANES>,
            {
                /// Performs an operation on vectors `self` and `rhs`, reusing `self` as the output.
                #[inline(always)]
                pub fn $plain(&mut self, rhs: &Self) {
                    let len = self.len();
                    assert_eq!(len, rhs.len(), "input vectors must be of equal size");

                    for i in (0..(len - (len % LANES))).step_by(LANES) {
                        let a_chunk = unsafe { &*self.as_ptr().add(i).cast::<[T; LANES]>() };
                        let b_chunk = unsafe { &*rhs.as_ptr().add(i).cast::<[T; LANES]>() };

                        // SAFETY: SIMD guarantees alignment
                        let val = a_chunk.$op(b_chunk);
                        self.index_mut(i..(i + LANES)).copy_from_slice(&val);
                    }

                    if (len - (len % LANES)) != len {
                        let a_chunk = <[T; LANES] as $crate::scalar::ArrayOf<T>>::pad_to(&self[len - (len % LANES)..]);
                        let b_chunk = <[T; LANES] as $crate::scalar::ArrayOf<T>>::pad_to(&rhs[len - (len % LANES)..]);
                        let val = &a_chunk.$op(&(&b_chunk))[..len % LANES];
                        for (offset, &elem) in val.into_iter().enumerate() {
                            *self.index_mut(len - (len % LANES) + offset) = elem;
                        }
                    }
                }

                /// Performs an operation on vector `self` by `rhs`, reusing `self` as the output.
                #[inline(always)]
                pub fn $scalar(&mut self, rhs: T) {
                    let len = self.len();

                    let b_chunk = [rhs; LANES];

                    for i in (0..(len - (len % LANES))).step_by(LANES) {
                        let a_chunk = unsafe { &*self.as_ptr().add(i).cast::<[T; LANES]>() };

                        // SAFETY: SIMD guarantees alignment
                        let val = a_chunk.$op(&b_chunk);
                        self[i..(i + LANES)].copy_from_slice(&val);
                    }

                    if (len - (len % LANES)) != len {
                        let a_chunk = <[T; LANES] as $crate::scalar::ArrayOf<T>>::pad_to(&self[len - (len % LANES)..]);
                        let val = &a_chunk.$op(&b_chunk)[..len % LANES];
                        for (offset, &elem) in val.into_iter().enumerate() {
                            *self.index_mut(len - (len % LANES) + offset) = elem;
                        }
                    }
                }
            }
        )*
    };
}

impl_core_ops!(
    impl (
        {
            @intrinsic simd_add, add, add_scalar
            SimdAdd {
                fn simd_add {}
                fn simd_add_into {}
            }
            AddByRef {
                fn as_add {}
                fn add_into {}
            }
        },
        {
            @intrinsic simd_mul, mul, mul_scalar
            SimdMul {
                fn simd_mul {}
                fn simd_mul_into {}
            }
            MulByRef {
                fn as_mul {}
                fn mul_into {}
            }
        },
        {
            @intrinsic simd_sub, sub, sub_scalar
            SimdSub {
                fn simd_sub {}
                fn simd_sub_into {}
            }
            SubByRef {
                fn as_sub {}
                fn sub_into {}
            }
        },
        {
            @intrinsic simd_div, div, div_scalar
            SimdDiv {
                fn simd_div {}
                fn simd_div_into {}
            }
            DivByRef {
                fn as_div {}
                fn div_into {}
            }
        },
    ) for Simd {}
);

#[cfg(test)]
mod tests {
    use crate::{scalar::Flatten, simd::Simd};

    use super::*;

    #[test]
    fn slicing_safety() {
        let simd = Simd::new([1.0; 128]);
        let a = &simd[50..][..];
        let b = SimdSlice::<f64, 4>::from_slice(&[1.0; 128]);
        let mut out = [0.0; 128];
        let out = SimdSlice::<f64, 4>::from_slice_mut(&mut out);
        a[..=15].add_into(&b[5..21], out);
        assert_eq!(out[..16].as_slice(), [2.0; 16]);
    }

    #[test]
    fn self_as_output() {
        let mut simd = Simd::new([5.0; 128]);
        let expected: Simd<f64, 128, 4> = Simd::new([[15.0; 64], [5.0; 64]].flatten());
        let data = {
            let a = &Simd::new([2.0; 64])[..];
            let b = &mut Simd::new([0.0; 128])[..];
            let out = &mut simd[64..];

            a.mul_into(&5.0, &mut b[..64]);
            out.add(&b);

            out
        };
        assert_eq!(data, &expected[..]);
    }
}
