//! Extensive coverage of generic mathematic operations in traits.

use core::{
    mem::{ManuallyDrop, MaybeUninit},
    num::{
        NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize, NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize, Wrapping
    }, ptr::copy_nonoverlapping,
};

use crate::simd::SimdScalar;

/// Enables addiion without consumption of data.
///
/// Provides a trait method to add unsized types and avoid overhead of
/// moving out of a value.
pub trait AddByRef<Rhs: ?Sized = Self> {
    /// The dynamically sized output type of `Self + Rhs`.
    type Output: ?Sized;

    /// Adds `self` and `rhs` without consuming them.
    #[must_use]
    fn as_add(&self, rhs: &Rhs) -> Self::Output
    where
        Self::Output: Sized;

    /// Like [`AddByRef::as_add`], only it writes output to a pre-existing buffer.
    fn add_into(&self, rhs: &Rhs, out: &mut Self::Output);
}

/// Enables subtraction without consumption of data.
///
/// Provides a trait method to add unsized types and avoid overhead of
/// moving out of a value.
pub trait SubByRef<Rhs: ?Sized = Self> {
    /// The dynamically sized output type of `Self - Rhs`.
    type Output: ?Sized;

    /// Subtracts `self` and `rhs` without consuming them.
    #[must_use]
    fn as_sub(&self, rhs: &Rhs) -> Self::Output
    where
        Self::Output: Sized;

    /// Like [`SubByRef::as_sub`], only it writes output to a pre-existing buffer.
    fn sub_into(&self, rhs: &Rhs, out: &mut Self::Output);
}

/// Enables multiplication without consumption of data.
///
/// Provides a trait method to add unsized types and avoid overhead of
/// moving out of a value.
pub trait MulByRef<Rhs: ?Sized = Self> {
    /// The dynamically sized output type of `Self * Rhs`.
    type Output: ?Sized;

    /// Multiplies `self` and `rhs` without consuming them.
    #[must_use]
    fn as_mul(&self, rhs: &Rhs) -> Self::Output
    where
        Self::Output: Sized;

    /// Like [`MulByRef::as_mul`], only it writes output to a pre-existing buffer.
    fn mul_into(&self, rhs: &Rhs, out: &mut Self::Output);
}

/// Enables division without consumption of data.
///
/// Provides a trait method to add unsized types and avoid overhead of
/// moving out of a value.
pub trait DivByRef<Rhs: ?Sized = Self> {
    /// The dynamically sized output type of `Self / Rhs`.
    type Output: ?Sized;

    /// Divides `self` and `rhs` without consuming them.
    #[must_use]
    fn as_div(&self, rhs: &Rhs) -> Self::Output
    where
        Self::Output: Sized;

    /// Like [`DivByRef::as_div`], only it writes output to a pre-existing buffer.
    fn div_into(&self, rhs: &Rhs, out: &mut Self::Output);
}

/// Represents an indirect SIMD-accelerated scalar.
pub trait AcceleratedScalar<T: SimdScalar, const LANES: usize>: ImplicitSupersetOf<T> {}

impl<F: SimdScalar> AcceleratedScalar<F, 1> for F {}

impl<F: SimdScalar + Copy + Primitive, const N: usize> AcceleratedScalar<F, N> for [F; N] {}

/// A marker trait implying that `Self` wraps `T` indirectly.
pub trait ImplicitSupersetOf<T: ?Sized> {}

impl<S: SupersetOf<T>, T> ImplicitSupersetOf<T> for S {}

impl<T, const X: usize, const Y: usize, const N: usize> ImplicitSupersetOf<[T; N]> for [[T; X]; Y] {}

impl<T, const X: usize, const Y: usize, const Z: usize, const N: usize> ImplicitSupersetOf<[T; N]>
    for [[[T; X]; Y]; Z]
{
}

/// A marker trait implying that `T` wraps `Self` indirectly.
pub trait ImplicitSubsetOf<T: ?Sized> {}

impl<S, T: ImplicitSupersetOf<S>> ImplicitSubsetOf<T> for S {}

/// A marker trait for types that are not nested and use little orq no indirection.
pub trait Primitive {}

macro_rules! transparent_impl {
    ($tr:path, $($ty:ident),*) => {
        $(
            impl<T: $tr> $tr for $ty <T> {}
        )*
    };
}

transparent_impl!(Primitive, MaybeUninit, ManuallyDrop, Wrapping);

impl Primitive for () {}
impl Primitive for bool {}

impl Primitive for u8 {}
impl Primitive for u16 {}
impl Primitive for u32 {}
impl Primitive for u64 {}
impl Primitive for u128 {}
impl Primitive for usize {}

impl Primitive for i8 {}
impl Primitive for i16 {}
impl Primitive for i32 {}
impl Primitive for i64 {}
impl Primitive for i128 {}
impl Primitive for isize {}

impl Primitive for f32 {}
impl Primitive for f64 {}

impl Primitive for NonZeroU8 {}
impl Primitive for NonZeroU16 {}
impl Primitive for NonZeroU32 {}
impl Primitive for NonZeroU64 {}
impl Primitive for NonZeroU128 {}
impl Primitive for NonZeroUsize {}

impl Primitive for NonZeroI8 {}
impl Primitive for NonZeroI16 {}
impl Primitive for NonZeroI32 {}
impl Primitive for NonZeroI64 {}
impl Primitive for NonZeroI128 {}
impl Primitive for NonZeroIsize {}

#[cfg(target_has_atomic = "8")]
impl Primitive for core::sync::atomic::AtomicBool {}

#[cfg(target_has_atomic = "8")]
impl Primitive for core::sync::atomic::AtomicU8 {}
#[cfg(target_has_atomic = "16")]
impl Primitive for core::sync::atomic::AtomicU16 {}
#[cfg(target_has_atomic = "32")]
impl Primitive for core::sync::atomic::AtomicU32 {}
#[cfg(target_has_atomic = "64")]
impl Primitive for core::sync::atomic::AtomicU64 {}
#[cfg(target_has_atomic = "ptr")]
impl Primitive for core::sync::atomic::AtomicUsize {}

#[cfg(target_has_atomic = "8")]
impl Primitive for core::sync::atomic::AtomicI8 {}
#[cfg(target_has_atomic = "16")]
impl Primitive for core::sync::atomic::AtomicI16 {}
#[cfg(target_has_atomic = "32")]
impl Primitive for core::sync::atomic::AtomicI32 {}
#[cfg(target_has_atomic = "64")]
impl Primitive for core::sync::atomic::AtomicI64 {}
#[cfg(target_has_atomic = "ptr")]
impl Primitive for core::sync::atomic::AtomicIsize {}

/// Trait that states `Self` is a superset of `T`.
pub trait SupersetOf<T: ?Sized> {
    /// Copies `scalar` for all slots in an instance `Self`.
    fn from_subset(scalar: T) -> Self
    where
        Self: Sized,
        T: Copy;

    /// Moves out a unit `T` from `self`.
    fn into_subset(self) -> T
    where
        Self: Sized;
}

impl<T> SupersetOf<T> for T {
    fn from_subset(scalar: T) -> Self {
        scalar
    }

    fn into_subset(self) -> T {
        self
    }
}

impl<T: Copy + Primitive, const N: usize> SupersetOf<T> for [T; N] {
    fn from_subset(scalar: T) -> Self {
        [scalar; N]
    }

    fn into_subset(self) -> T {
        self[0]
    }
}

impl<T: Copy + Primitive, const X: usize, const Y: usize> SupersetOf<T> for [[T; X]; Y] {
    fn from_subset(scalar: T) -> Self {
        [[scalar; X]; Y]
    }

    fn into_subset(self) -> T {
        self[0][0]
    }
}

impl<T: Copy + Primitive, const X: usize, const Y: usize, const Z: usize> SupersetOf<T>
    for [[[T; X]; Y]; Z]
{
    fn from_subset(scalar: T) -> Self {
        [[[scalar; X]; Y]; Z]
    }

    fn into_subset(self) -> T {
        self[0][0][0]
    }
}

#[cfg(feature = "alloc")]
impl<T: Clone> SupersetOf<T> for alloc::vec::Vec<T> {
    fn from_subset(scalar: T) -> Self {
        alloc::vec![scalar]
    }

    fn into_subset(self) -> T {
        self[0].clone()
    }
}

#[cfg(feature = "std")]
impl<T: Clone> SupersetOf<T> for std::collections::VecDeque<T> {
    fn from_subset(scalar: T) -> Self {
        let mut deque = std::collections::VecDeque::new();
        deque.push_front(scalar);
        deque
    }

    fn into_subset(self) -> T {
        self[0].clone()
    }
}

#[cfg(feature = "std")]
impl<T: Clone> SupersetOf<T> for std::collections::LinkedList<T> {
    fn from_subset(scalar: T) -> Self {
        let mut deque = std::collections::LinkedList::new();
        deque.push_front(scalar);
        deque
    }

    fn into_subset(self) -> T {
        self.front().unwrap().clone()
    }
}

#[cfg(feature = "std")]
impl<T: Ord> SupersetOf<T> for std::collections::BTreeSet<T> {
    fn from_subset(scalar: T) -> Self {
        let mut map = std::collections::BTreeSet::new();
        map.insert(scalar);
        map
    }

    fn into_subset(mut self) -> T {
        self.pop_first().unwrap()
    }
}

#[cfg(feature = "std")]
impl<K: Ord + Default, V> SupersetOf<V> for std::collections::BTreeMap<K, V> {
    fn from_subset(scalar: V) -> Self {
        let mut map = std::collections::BTreeMap::new();
        map.insert(K::default(), scalar);
        map
    }

    fn into_subset(mut self) -> V {
        self.pop_first().unwrap().1
    }
}

#[cfg(feature = "std")]
impl<T: core::hash::Hash + Eq + Clone + Default> SupersetOf<T> for std::collections::HashSet<T> {
    fn from_subset(scalar: T) -> Self {
        let mut map = std::collections::HashSet::new();
        map.insert(scalar);
        map
    }

    fn into_subset(self) -> T {
        self.get(&T::default()).unwrap().clone()
    }
}

#[cfg(feature = "std")]
impl<K: core::hash::Hash + Eq + Default, V: Clone> SupersetOf<V>
    for std::collections::HashMap<K, V>
{
    fn from_subset(scalar: V) -> Self {
        let mut map = std::collections::HashMap::new();
        map.insert(K::default(), scalar);
        map
    }

    fn into_subset(self) -> V {
        self.get(&K::default()).unwrap().clone()
    }
}

#[cfg(feature = "std")]
impl<T: Ord + Clone> SupersetOf<T> for std::collections::BinaryHeap<T> {
    fn from_subset(scalar: T) -> Self {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(scalar);
        heap
    }

    fn into_subset(self) -> T {
        self.as_slice()[0].clone()
    }
}

/// Trait that states `Self` is a subset of `T`.
pub trait SubsetOf<T: ?Sized> {
    /// Copies `self` for all slots in the superset.
    fn into_superset(self) -> T
    where
        Self: Sized;

    /// Gets a unit value `Self` from `T`.
    fn from_superset(superset: T) -> Self
    where
        Self: Sized;
}

impl<T: Copy, S: SupersetOf<T>> SubsetOf<S> for T {
    fn from_superset(superset: S) -> Self {
        superset.into_subset()
    }

    fn into_superset(self) -> S {
        S::from_subset(self)
    }
}

/// Trait that states `Self` is a statically sized superset of `T`.
///
/// This is required for a well behaved superset. For example, `Vec<u8>` is
/// less convenient to work with than a `[u8; 16]` when copying elements or
/// fetching a unit.
pub trait ArrayOf<T>: SupersetOf<T> {
    /// Pads or trims `slice` to the length of `Self`.
    fn pad_to(slice: &[T]) -> Self
    where
        Self: Sized;
}

impl<T: Copy + Default + Primitive, const N: usize> ArrayOf<T> for [T; N] {
    fn pad_to(slice: &[T]) -> Self {
        let len = slice.len();
        if len >= N {
            unsafe { slice.try_into().unwrap_unchecked() }
        } else {
            let mut buffer = [T::default(); N];
            unsafe {
                copy_nonoverlapping(slice.as_ptr(), (&raw mut buffer).cast::<T>(), slice.len());
            }
            buffer
        }
    }
}

impl<T: Copy + Default + Primitive, const X: usize, const Y: usize> ArrayOf<T> for [[T; X]; Y] {
    fn pad_to(slice: &[T]) -> Self {
        let len = slice.len();
        if len >= (X * Y) {
            unsafe { slice.as_ptr().cast::<[[T; X]; Y]>().read() }
        } else {
            let mut buffer = [[T::default(); X]; Y];
            unsafe {
                copy_nonoverlapping(slice.as_ptr(), (&raw mut buffer).cast::<T>(), slice.len());
            }
            buffer
        }
    }
}

impl<T: Copy + Default + Primitive, const X: usize, const Y: usize, const Z: usize> ArrayOf<T>
    for [[[T; X]; Y]; Z]
{
    fn pad_to(slice: &[T]) -> Self {
        let len = slice.len();
        if len >= (X * Y * Z) {
            unsafe { slice.as_ptr().cast::<[[[T; X]; Y]; Z]>().read() }
        } else {
            let mut buffer = [[[T::default(); X]; Y]; Z];
            unsafe {
                copy_nonoverlapping(slice.as_ptr(), (&raw mut buffer).cast::<T>(), slice.len());
            }
            buffer
        }
    }
}

/// Trait that allows for flattenning deeply nested types into simple arrays.
pub trait Flatten<T, A: SupersetOf<T> + ImplicitSubsetOf<Self>>: ArrayOf<T> {
    /// Performs the flattenning operation.
    fn flatten(self) -> A;
}

impl<T: Copy + Default + Primitive, const N: usize> Flatten<T, [T; N]> for [T; N] {
    fn flatten(self) -> [T; N] {
        self
    }
}

impl<T: Copy + Default + Primitive, const X: usize, const Y: usize, const N: usize>
    Flatten<T, [T; N]> for [[T; X]; Y]
{
    fn flatten(self) -> [T; N] {
        const {
            assert!((X * Y) == N, "input arrays have invalid size for output");
        }

        let mut out = [T::default(); N];
        for (i, val) in self.into_iter().enumerate() {
            let idx = i * X;
            let extended_idx = (i + 1) * X;
            out[idx..extended_idx].copy_from_slice(&val);
        }
        out
    }
}

impl<
        T: Copy + Default + Primitive,
        const X: usize,
        const Y: usize,
        const Z: usize,
        const N: usize,
    > Flatten<T, [T; N]> for [[[T; X]; Y]; Z]
{
    fn flatten(self) -> [T; N] {
        const {
            assert!((X * Y * Z) == N, "input arrays have invalid size for output");
        }

        let mut out = [T::default(); N];
        for (i, arr) in self.into_iter().enumerate() {
            for (j, val) in arr.into_iter().enumerate() {
                let idx = (i * (Y * X)) + (j * X);
                let extended_idx = (i * (Y * X)) + ((j + 1) * X);
                out[idx..extended_idx].copy_from_slice(&val);
            }
        }
        out
    }
}

/// Trait for associated SIMD-accelerated addition.
pub trait SimdAdd {
    /// The value that each addition operation will associate with.
    type Set: ?Sized;

    /// Adds `a + b` and returns the output.
    #[must_use]
    fn simd_add(a: &Self::Set, b: &Self::Set) -> Self::Set
    where
        Self::Set: Sized;

    /// Adds `a + b` and writes the output to the specified buffer.
    fn simd_add_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set);
}

/// Trait for associated SIMD-accelerated subtraction.
pub trait SimdSub {
    /// The value that each subtraction operation will associate with.
    type Set: ?Sized;

    /// Subtracts `a + b` and returns the output.
    #[must_use]
    fn simd_sub(a: &Self::Set, b: &Self::Set) -> Self::Set
    where
        Self::Set: Sized;

    /// Subtracts `a - b` and writes the output to the specified buffer.
    fn simd_sub_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set);
}

/// Trait for associated SIMD-accelerated multiplication.
pub trait SimdMul {
    /// The value that each multiplication operation will associate with.
    type Set: ?Sized;

    /// Multiplies `a * b` and returns the output.
    #[must_use]
    fn simd_mul(a: &Self::Set, b: &Self::Set) -> Self::Set
    where
        Self::Set: Sized;

    /// Multiplies `a * b` and writes the output to the specified buffer.
    fn simd_mul_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set);
}

/// Trait for associated SIMD-accelerated division.
pub trait SimdDiv {
    /// The value that each division operation will associate with.
    type Set: ?Sized;

    /// Divides `a / b` and returns the output.
    #[must_use]
    fn simd_div(a: &Self::Set, b: &Self::Set) -> Self::Set
    where
        Self::Set: Sized;

    /// Divides `a / b` and writes the output to the specified buffer.
    fn simd_div_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set);
}
