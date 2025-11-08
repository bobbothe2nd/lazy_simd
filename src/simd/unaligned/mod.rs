use core::mem::MaybeUninit;

use crate::{
    scalar::{SimdAdd, SimdDiv, SimdMul, SimdSub},
    simd::backend::{DoublePrecisionNoSimd, NonAssociativeSimd},
    MAX_SIMD_SINGLE_PRECISION_LANES,
};

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse"),
    all(target_arch = "aarch64", target_feature = "neon")
))]
impl NonAssociativeSimd<[f32; 4], f32, 4> for [f32; 4] {
    type Backend = F32x4;
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
impl NonAssociativeSimd<[f32; 8], f32, 8> for [f32; 8] {
    type Backend = F32x8;
}

impl
    NonAssociativeSimd<[f64; MAX_SIMD_SINGLE_PRECISION_LANES], f64, MAX_SIMD_SINGLE_PRECISION_LANES>
    for [f64; MAX_SIMD_SINGLE_PRECISION_LANES]
{
    type Backend = DoublePrecisionNoSimd<MAX_SIMD_SINGLE_PRECISION_LANES>;
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
impl NonAssociativeSimd<[f64; 4], f64, 4> for [f64; 4] {
    type Backend = F64x4;
}

#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "sse",
        not(target_feature = "avx")
    ),
    all(target_arch = "aarch64", target_feature = "neon"),
))]
pub type SinglePrecisionSimdBackend = F32x4;
#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
pub type SinglePrecisionSimdBackend = F32x8;
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        any(target_feature = "avx", target_feature = "sse")
    ),
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
pub type SinglePrecisionSimdBackend = super::backend::SinglePrecisionNoSimd;

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
pub type DoublePrecisionSimdBackend = F64x4;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx")))]
pub type DoublePrecisionSimdBackend = super::backend::DoublePrecisionNoSimd<1>;

pub const SINGLE_PRECISION_BACKEND_SIZE: usize = SinglePrecisionSimdBackend::SIZE;
pub const DOUBLE_PRECISION_BACKEND_SIZE: usize = DoublePrecisionSimdBackend::SIZE;

macro_rules! impl_closed_ops {
    ($array_variant_name:ident, $slice_variant_name:ident, $scalar_variant_name:ident, $scalar_array_name:ident, $scalar:ty, $op:path, $size:expr) => {
        /// Computes the value of an operation performed on slices `a` and `b`, writing the output to the specified buffer.
        #[inline(always)]
        pub fn $slice_variant_name(a: &[$scalar], b: &[$scalar], out: &mut [$scalar]) {
            let len_cached = a.len();

            assert_eq!(
                len_cached,
                b.len(),
                "input slices must be the same length to add"
            );
            assert_eq!(
                len_cached,
                out.len(),
                "output must be the same size as input"
            );

            for i in (0..len_cached).step_by($size) {
                if i > (len_cached - $size) {
                    break;
                }

                let normalized_idx = i / $size;
                let a_batch = &{
                    let mut buf = unsafe { core::mem::zeroed::<[$scalar; $size]>() };
                    buf.copy_from_slice(&a[i..(i + $size)]);
                    buf
                };
                let b_batch = &{
                    let mut buf = unsafe { core::mem::zeroed::<[$scalar; $size]>() };
                    buf.copy_from_slice(&b[i..(i + $size)]);
                    buf
                };
                let val = $op(a_batch, b_batch);
                unsafe {
                    out.as_mut_ptr()
                        .cast::<[$scalar; $size]>()
                        .add(normalized_idx)
                        .write(val);
                }
            }
            let i = len_cached - (len_cached % $size);
            if i != len_cached {
                let a_batch = &{
                    let mut buf = unsafe { core::mem::zeroed::<[$scalar; $size]>() };
                    buf.copy_from_slice(&a[i..]);
                    buf
                };
                let b_batch = &{
                    let mut buf = unsafe { core::mem::zeroed::<[$scalar; $size]>() };
                    buf.copy_from_slice(&b[i..]);
                    buf
                };
                let val = &$op(a_batch, b_batch)[..(len_cached % $size)];
                let ptr = unsafe { out.as_mut_ptr().cast::<$scalar>().add(i) };
                for offset in 0..(len_cached % $size) {
                    unsafe {
                        ptr.add(offset).write(val[offset]);
                    }
                }
            }
        }

        /// Computes the value of an operation performed on a slice `a` by `scalar`, writing the output to the specified buffer.
        #[inline(always)]
        pub fn $scalar_variant_name(slice: &[$scalar], scalar: $scalar, out: &mut [$scalar]) {
            let len_cached = slice.len();

            assert_eq!(
                len_cached,
                out.len(),
                "output must be the same size as input"
            );

            let scalar_batch = &[scalar; $size];

            let mut batch;

            for i in (0..len_cached).step_by($size) {
                if i > (len_cached - $size) {
                    break;
                }

                let normalized_idx = i / $size;
                batch = {
                    let mut buf = unsafe { core::mem::zeroed::<[$scalar; $size]>() };
                    buf.copy_from_slice(&slice[i..(i + $size)]);
                    buf
                };
                let val = $op(&batch, scalar_batch);
                unsafe {
                    out.as_mut_ptr()
                        .cast::<[$scalar; $size]>()
                        .add(normalized_idx)
                        .write(val);
                }
            }
            let i = len_cached - (len_cached % $size);
            if i != len_cached {
                batch = {
                    let mut buf = unsafe { core::mem::zeroed::<[$scalar; $size]>() };
                    buf.copy_from_slice(&slice[i..]);
                    buf
                };
                let val = &$op(&batch, scalar_batch)[..(len_cached % $size)];
                let ptr = unsafe { out.as_mut_ptr().cast::<$scalar>().add(i) };
                for offset in 0..(len_cached % $size) {
                    unsafe {
                        ptr.add(offset).write(val[offset]);
                    }
                }
            }
        }

        /// Computes the value of an operation performed on array `a` by `scalar`, returning an array of the output.
        #[must_use]
        #[inline(always)]
        pub fn $scalar_array_name<const N: usize>(
            a: &[$scalar; N],
            scalar: $scalar,
        ) -> [$scalar; N] {
            let mut out = [scalar; N];
            $scalar_variant_name(a, scalar, &mut out);
            out
        }

        /// Computes the value of an operation performed on arrays `a` and `b`, returning an array of the output.
        #[must_use]
        #[inline(always)]
        pub fn $array_variant_name<const N: usize>(
            a: &[$scalar; N],
            b: &[$scalar; N],
        ) -> [$scalar; N] {
            let mut out = MaybeUninit::<[$scalar; N]>::uninit();
            for i in (0..N).step_by($size) {
                if i > (N - $size) {
                    break;
                }

                let normalized_idx = i / $size;
                let a_batch = &{
                    let mut buf = unsafe { core::mem::zeroed::<[$scalar; $size]>() };
                    buf.copy_from_slice(&a[i..(i + $size)]);
                    buf
                };
                let b_batch = &{
                    let mut buf = unsafe { core::mem::zeroed::<[$scalar; $size]>() };
                    buf.copy_from_slice(&b[i..(i + $size)]);
                    buf
                };
                let val = $op(a_batch, b_batch);
                unsafe {
                    out.as_mut_ptr()
                        .cast::<[$scalar; $size]>()
                        .add(normalized_idx)
                        .write(val);
                }
            }
            let i = N - (N % $size);
            if i != N {
                let a_batch = &{
                    let mut buf = unsafe { core::mem::zeroed::<[$scalar; $size]>() };
                    buf.copy_from_slice(
                        &<[$scalar; $size] as $crate::scalar::ArrayOf<$scalar>>::pad_to(&a[i..]),
                    );
                    buf
                };
                let b_batch = &{
                    let mut buf = unsafe { core::mem::zeroed::<[$scalar; $size]>() };
                    buf.copy_from_slice(
                        &<[$scalar; $size] as $crate::scalar::ArrayOf<$scalar>>::pad_to(&b[i..]),
                    );
                    buf
                };
                let val = &$op(a_batch, b_batch)[..(N % $size)];
                let ptr = unsafe { out.as_mut_ptr().cast::<$scalar>().add(i) };
                for offset in 0..(N % $size) {
                    unsafe {
                        ptr.add(offset).write(val[offset]);
                    }
                }
            }
            unsafe { out.assume_init() }
        }
    };
}

impl_closed_ops!(
    add_arrays,
    add_slices,
    add_scalar_slice,
    add_scalar_array,
    f32,
    SinglePrecisionSimdBackend::simd_add,
    SINGLE_PRECISION_BACKEND_SIZE
);
impl_closed_ops!(
    sub_arrays,
    sub_slices,
    sub_scalar_slice,
    sub_scalar_array,
    f32,
    SinglePrecisionSimdBackend::simd_sub,
    SINGLE_PRECISION_BACKEND_SIZE
);
impl_closed_ops!(
    mul_arrays,
    mul_slices,
    mul_scalar_slice,
    mul_scalar_array,
    f32,
    SinglePrecisionSimdBackend::simd_mul,
    SINGLE_PRECISION_BACKEND_SIZE
);
impl_closed_ops!(
    div_arrays,
    div_slices,
    div_scalar_slice,
    div_scalar_array,
    f32,
    SinglePrecisionSimdBackend::simd_div,
    SINGLE_PRECISION_BACKEND_SIZE
);

impl_closed_ops!(
    add_arrays_double,
    add_slices_double,
    add_scalar_slice_double,
    add_scalar_array_double,
    f64,
    DoublePrecisionSimdBackend::simd_add,
    DOUBLE_PRECISION_BACKEND_SIZE
);
impl_closed_ops!(
    sub_arrays_double,
    sub_slices_double,
    sub_scalar_slice_double,
    sub_scalar_array_double,
    f64,
    DoublePrecisionSimdBackend::simd_sub,
    DOUBLE_PRECISION_BACKEND_SIZE
);
impl_closed_ops!(
    mul_arrays_double,
    mul_slices_double,
    mul_scalar_slice_double,
    mul_scalar_array_double,
    f64,
    DoublePrecisionSimdBackend::simd_mul,
    DOUBLE_PRECISION_BACKEND_SIZE
);
impl_closed_ops!(
    div_arrays_double,
    div_slices_double,
    div_scalar_slice_double,
    div_scalar_array_double,
    f64,
    DoublePrecisionSimdBackend::simd_div,
    DOUBLE_PRECISION_BACKEND_SIZE
);

/// A function specifically designed for SIMD-accelerated matrix multiplication.
///
/// It will add the product of each element in `rhs` and `scalar` to each element in `out`.
#[inline(always)]
pub fn mul_add_scalar_slice(scalar: f32, rhs: &[f32], out: &mut [f32]) {
    let len = rhs.len();
    assert_eq!(len, out.len(), "rhs and out must have equal length");

    let scalar_batch = &[scalar; SINGLE_PRECISION_BACKEND_SIZE];

    let batchs = len / SINGLE_PRECISION_BACKEND_SIZE;
    let remainder = len % SINGLE_PRECISION_BACKEND_SIZE;

    // process full SIMD-width batchs
    for batch_idx in 0..batchs {
        let i = batch_idx * SINGLE_PRECISION_BACKEND_SIZE;

        // form stack arrays for the batch (no heap)
        let rhs_batch = &{
            let mut tmp = unsafe { core::mem::zeroed::<[f32; SINGLE_PRECISION_BACKEND_SIZE]>() };
            tmp.copy_from_slice(&rhs[i..i + SINGLE_PRECISION_BACKEND_SIZE]);
            tmp
        };

        let out_batch_in = &{
            let mut tmp = unsafe { core::mem::zeroed::<[f32; SINGLE_PRECISION_BACKEND_SIZE]>() };
            tmp.copy_from_slice(&out[i..i + SINGLE_PRECISION_BACKEND_SIZE]);
            tmp
        };

        // mul and add using the backend (pure functional: return values)
        let mul_val = SinglePrecisionSimdBackend::simd_mul(scalar_batch, rhs_batch);
        let new_batch = SinglePrecisionSimdBackend::simd_add(out_batch_in, &mul_val);

        // write result back to out slice (single write per element)
        out[i..i + SINGLE_PRECISION_BACKEND_SIZE].copy_from_slice(&new_batch);
    }

    // tail (scalar)
    if remainder != 0 {
        let start = batchs * SINGLE_PRECISION_BACKEND_SIZE;
        for j in 0..remainder {
            out[start + j] += scalar * rhs[start + j];
        }
    }
}

/// Similar to [`crate::simd::mul_add_scalar_slice`], only for double precision floats (`f64`).
#[inline(always)]
pub fn mul_add_scalar_slice_double(scalar: f64, rhs: &[f64], out: &mut [f64]) {
    let len = rhs.len();
    assert_eq!(len, out.len(), "rhs and out must have equal length");

    let scalar_batch = &[scalar; DOUBLE_PRECISION_BACKEND_SIZE];

    let batchs = len / DOUBLE_PRECISION_BACKEND_SIZE;
    let remainder = len % DOUBLE_PRECISION_BACKEND_SIZE;

    // process full SIMD-width batchs
    for batch_idx in 0..batchs {
        let i = batch_idx * DOUBLE_PRECISION_BACKEND_SIZE;

        // form stack arrays for the batch
        let rhs_batch = &{
            let mut tmp = unsafe { core::mem::zeroed::<[f64; DOUBLE_PRECISION_BACKEND_SIZE]>() };
            tmp.copy_from_slice(&rhs[i..i + DOUBLE_PRECISION_BACKEND_SIZE]);
            tmp
        };

        let out_batch_in = &{
            let mut tmp = unsafe { core::mem::zeroed::<[f64; DOUBLE_PRECISION_BACKEND_SIZE]>() };
            tmp.copy_from_slice(&out[i..i + DOUBLE_PRECISION_BACKEND_SIZE]);
            tmp
        };

        // mul and add using the backend (pure functional: return values)
        let mul_val = DoublePrecisionSimdBackend::simd_mul(scalar_batch, rhs_batch);
        let new_batch = DoublePrecisionSimdBackend::simd_add(out_batch_in, &mul_val);

        // write result back to out slice (single write per element)
        out[i..i + DOUBLE_PRECISION_BACKEND_SIZE].copy_from_slice(&new_batch);
    }

    // tail (scalar)
    if remainder != 0 {
        let start = batchs * SINGLE_PRECISION_BACKEND_SIZE;
        for j in 0..remainder {
            out[start + j] += scalar * rhs[start + j];
        }
    }
}

macro_rules! simd_ops_internal {
    (@add [f32; 8], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm256_add_ps | vaddq_f32) [f32; 8], $a, $b)
    };

    (@add [f32; 4], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm_add_ps | vaddq_f32) [f32; 4], $a, $b)
    };

    (@add [f64; 4], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm256_add_pd | vaddq_f64) [f64; 4], $a, $b)
    };

    (@sub [f32; 8], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm256_sub_ps | vsubq_f32) [f32; 8], $a, $b)
    };

    (@sub [f32; 4], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm_sub_ps | vsubq_f32) [f32; 4], $a, $b)
    };

    (@sub [f64; 4], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm256_sub_pd | vsubq_f64) [f64; 4], $a, $b)
    };

    (@mul [f32; 8], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm256_mul_ps | vmulq_f32) [f32; 8], $a, $b)
    };

    (@mul [f32; 4], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm_mul_ps | vmulq_f32) [f32; 4], $a, $b)
    };

    (@mul [f64; 4], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm256_mul_pd | vmulq_f64) [f64; 4], $a, $b)
    };

    (@div [f32; 8], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm256_div_ps | vdivq_f32) [f32; 8], $a, $b)
    };

    (@div [f32; 4], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm_div_ps | vdivq_f32) [f32; 4], $a, $b)
    };

    (@div [f64; 4], $a:ident, $b:ident) => {
        simd_ops_internal!(@(_mm256_div_pd | vdivq_f64) [f64; 4], $a, $b)
    };

    // f32x8
    (@($op1:path | $op2:path) [f32; 8], $a:ident, $b:ident) => {{
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        unsafe {
            use core::arch::x86_64::*;
            let va = _mm256_loadu_ps($a.as_ptr());
            let vb = _mm256_loadu_ps($b.as_ptr());
            let mut out = [0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), $op1(va, vb));
            out
        }
        #[cfg(not(
            all(target_arch = "x86_64", target_feature = "avx")
        ))]
        {
            compile_error!("adding f32x8 requires AVX (x86_64)");
        }
    }};

    // f32x4
    (@($op1:path | $op2:path) [f32; 4], $a:ident, $b:ident) => {{
        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        unsafe {
            use core::arch::x86_64::*;
            let va = _mm_loadu_ps($a.as_ptr());
            let vb = _mm_loadu_ps($b.as_ptr());
            let mut out = [0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), $op1(va, vb));
            out
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            use core::arch::aarch64::*;
            let va = vld1q_f32($a.as_ptr());
            let vb = vld1q_f32($b.as_ptr());
            let mut out = [0f32; 4];
            vst1q_f32(out.as_mut_ptr(), $op2(va, vb));
            out
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "sse"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            compile_error!("adding f32x4 requires SSE (x86_64) or NEON (aarch64)");
        }
    }};

    // f64x4
    (@($op1:path | $op2:path) [f64; 4], $a:ident, $b:ident) => {{
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        unsafe {
            use core::arch::x86_64::*;
            let va = _mm256_loadu_pd($a.as_ptr());
            let vb = _mm256_loadu_pd($b.as_ptr());
            let mut out = [0f64; 4];
            _mm256_storeu_pd(out.as_mut_ptr(), $op1(va, vb));
            out
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            use core::arch::aarch64::*;
            let va = vld1q_f64($a.as_ptr());
            let vb = vld1q_f64($b.as_ptr());
            let mut out = [0f64; 4];
            vst1q_f64(out.as_mut_ptr(), $op2(va, vb));
            out
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            compile_error!("adding f64x4 requires AVX (x86_64) or NEON (aarch64)");
        }
    }};
}

macro_rules! impl_simd_ops {
    ($name:ident, [$type:ident; $len:tt], $doc:expr) => {
        #[doc = $doc]
        pub struct $name;

        impl $name {
            const SIZE: usize = $len;
            const _ASSERT_SIZE: () = {
                core::assert!(core::mem::size_of::<[$type; $len]>() <= $crate::MAX_SIMD_SIZE, "array too large for SIMD on target");
                core::assert!($len <= $crate::MAX_SIMD_SINGLE_PRECISION_LANES, "too many lanes for SIMD on target");
            };
        }

        impl $crate::scalar::ImplicitSupersetOf<$type> for $name {}

        impl $crate::scalar::AcceleratedScalar<$type, $len> for $name {}

        impl $crate::scalar::SimdAdd for $name {
            type Set = [$type; $len];

            #[inline(always)]
            fn simd_add(a: &Self::Set, b: &Self::Set) -> Self::Set {
                const {
                    let _ = Self::_ASSERT_SIZE;
                }

                simd_ops_internal!(@add [$type; $len], a, b)
            }

            #[inline(always)]
            fn simd_add_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
                *out = Self::simd_add(a, b);
            }
        }

        impl $crate::scalar::SimdSub for $name {
            type Set = [$type; $len];

            #[inline(always)]
            fn simd_sub(a: &Self::Set, b: &Self::Set) -> Self::Set {
                const {
                    let _ = Self::_ASSERT_SIZE;
                }

                simd_ops_internal!(@sub [$type; $len], a, b)
            }

            #[inline(always)]
            fn simd_sub_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
                *out = Self::simd_sub(a, b);
            }
        }

        impl $crate::scalar::SimdMul for $name {
            type Set = [$type; $len];

            #[inline(always)]
            fn simd_mul(a: &Self::Set, b: &Self::Set) -> Self::Set {
                const {
                    let _ = Self::_ASSERT_SIZE;
                }

                simd_ops_internal!(@mul [$type; $len], a, b)
            }

            #[inline(always)]
            fn simd_mul_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
                *out = Self::simd_mul(a, b);
            }
        }

        impl $crate::scalar::SimdDiv for $name {
            type Set = [$type; $len];

            #[inline(always)]
            fn simd_div(a: &Self::Set, b: &Self::Set) -> Self::Set {
                const {
                    let _ = Self::_ASSERT_SIZE;
                }

                simd_ops_internal!(@div [$type; $len], a, b)
            }

            #[inline(always)]
            fn simd_div_into(a: &Self::Set, b: &Self::Set, out: &mut Self::Set) {
                *out = Self::simd_div(a, b);
            }
        }
    };
}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse"),
    all(target_arch = "aarch64", target_feature = "neon")
))]
impl_simd_ops!(
    F32x4,
    [f32; 4],
    "
A SIMD wrapper around an array of 4 32-bit floating point values.

Only available on platforms that support SSE or NEON and consumes 128 bits (or 16 bytes) of memory.
"
);

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
impl_simd_ops!(
    F32x8,
    [f32; 8],
    "
A SIMD wrapper around an array of 8 32-bit floating point values.

Only available on platforms that support AVX and consumes 256 bits (or 32 bytes) of memory.
"
);

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
impl_simd_ops!(
    F64x4,
    [f64; 4],
    "
A SIMD wrapper around an array of 4 64-bit floating point values.

Only available on platforms that support AVX and consumes 256 bits (or 32 bytes) of memory.
"
);

#[cfg(test)]
mod tests {
    use super::*;
    use core::{iter::*, marker::*, ops::*, *};
    use num_traits::*;

    #[allow(dead_code)]
    fn ops_on<T, const N: usize>(arr1: [T; N], arr2: [T; N], op: fn(T, T) -> T) -> [T; N] {
        let mut out: MaybeUninit<[T; N]> = MaybeUninit::uninit();
        for (i, (val1, val2)) in arr1.into_iter().zip(arr2).enumerate() {
            unsafe {
                out.as_mut_ptr().cast::<T>().add(i).write(op(val1, val2));
            }
        }
        unsafe { out.assume_init() }
    }

    #[allow(dead_code)]
    fn mul_add<T: MulAdd<Output = T> + Copy>(a: T, b: T) -> T {
        a.mul_add(b, b)
    }

    #[allow(dead_code)]
    fn inv<T: Inv<Output = T> + Copy>(a: T, _: T) -> T {
        a.inv()
    }

    #[test]
    fn big_buffer() {
        for i in 0..1000 {
            let val = i as f32;
            let arr1 = [
                val,
                val + 1.0,
                val + 2.0,
                val + 3.0,
                val + 4.0,
                val + 5.0,
                val + 6.0,
            ];
            let arr2 = [
                val + 6.0,
                val + 5.0,
                val + 4.0,
                val + 3.0,
                val + 2.0,
                val + 1.0,
                val,
            ];

            let simd_add = add_arrays(&arr1, &arr2);
            let norm_add = ops_on(arr1, arr2, f32::add);
            assert_eq!(simd_add, norm_add);

            let simd_sub = sub_arrays(&arr1, &arr2);
            let norm_sub = ops_on(arr1, arr2, f32::sub);
            assert_eq!(simd_sub, norm_sub);

            let simd_mul = mul_arrays(&arr1, &arr2);
            let norm_mul = ops_on(arr1, arr2, f32::mul);
            assert_eq!(simd_mul, norm_mul);

            let simd_div = div_arrays(&arr1, &arr2);
            let norm_div = ops_on(arr1, arr2, f32::div);
            assert_eq!(simd_div, norm_div);
        }
    }

    #[test]
    fn big_buffer_double() {
        for i in 0..1000 {
            let val = i as f64;
            let arr1 = [
                val,
                val + 1.0,
                val + 2.0,
                val + 3.0,
                val + 4.0,
                val + 5.0,
                val + 6.0,
                val + 7.0,
            ];
            let arr2 = [
                val + 7.0,
                val + 6.0,
                val + 5.0,
                val + 4.0,
                val + 3.0,
                val + 2.0,
                val + 1.0,
                val,
            ];

            let simd_add = add_arrays_double(&arr1, &arr2);
            let norm_add = ops_on(arr1, arr2, f64::add);
            assert_eq!(simd_add, norm_add);

            let simd_sub = sub_arrays_double(&arr1, &arr2);
            let norm_sub = ops_on(arr1, arr2, f64::sub);
            assert_eq!(simd_sub, norm_sub);

            let simd_mul = mul_arrays_double(&arr1, &arr2);
            let norm_mul = ops_on(arr1, arr2, f64::mul);
            assert_eq!(simd_mul, norm_mul);

            let simd_div = div_arrays_double(&arr1, &arr2);
            let norm_div = ops_on(arr1, arr2, f64::div);
            assert_eq!(simd_div, norm_div);
        }
    }

    #[test]
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "sse"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    fn f32x4() {
        for i in 0..1000 {
            let val = i as f32;
            let arr1 = [val, val + 1.0, val + 2.0, val + 3.0];
            let arr2 = [val + 3.0, val + 2.0, val + 1.0, val];

            let simd_add = arr1.simd_add(&arr2);
            let norm_add = ops_on(arr1, arr2, f32::add);
            assert_eq!(simd_add, norm_add);

            let simd_sub = arr1.simd_sub(&arr2);
            let norm_sub = ops_on(arr1, arr2, f32::sub);
            assert_eq!(simd_sub, norm_sub);

            let simd_mul = arr1.simd_mul(&arr2);
            let norm_mul = ops_on(arr1, arr2, f32::mul);
            assert_eq!(simd_mul, norm_mul);

            let simd_div = arr1.simd_div(&arr2);
            let norm_div = ops_on(arr1, arr2, f32::div);
            assert_eq!(simd_div, norm_div);

            // these will differ slightly in some cases
            let simd_mul_add = arr1.simd_mul_add(&arr2, &arr2);
            let norm_mul_add = ops_on(arr1, arr2, mul_add);
            assert_eq!(simd_mul_add, norm_mul_add);

            let simd_inv = arr1.simd_inv();
            let norm_inv = ops_on(arr1, arr2, inv);
            assert_eq!(simd_inv, norm_inv);
        }
    }

    #[test]
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "avx"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    fn f32x8() {
        for i in 0..1000 {
            let val = i as f32;
            let arr1 = [
                val,
                val + 1.0,
                val + 2.0,
                val + 3.0,
                val + 4.0,
                val + 5.0,
                val + 6.0,
                val + 7.0,
            ];
            let arr2 = [
                val + 7.0,
                val + 6.0,
                val + 5.0,
                val + 4.0,
                val + 3.0,
                val + 2.0,
                val + 1.0,
                val,
            ];

            let simd_add = arr1.simd_add(&arr2);
            let norm_add = ops_on(arr1, arr2, f32::add);
            assert_eq!(simd_add, norm_add);

            let simd_sub = arr1.simd_sub(&arr2);
            let norm_sub = ops_on(arr1, arr2, f32::sub);
            assert_eq!(simd_sub, norm_sub);

            let simd_mul = arr1.simd_mul(&arr2);
            let norm_mul = ops_on(arr1, arr2, f32::mul);
            assert_eq!(simd_mul, norm_mul);

            let simd_div = arr1.simd_div(&arr2);
            let norm_div = ops_on(arr1, arr2, f32::div);
            assert_eq!(simd_div, norm_div);

            let simd_mul_add = arr1.simd_mul_add(&arr2, &arr2);
            let norm_mul_add = ops_on(arr1, arr2, mul_add);
            assert_eq!(simd_mul_add, norm_mul_add);

            let simd_inv = arr1.simd_inv();
            let norm_inv = ops_on(arr1, arr2, inv);
            assert_eq!(simd_inv, norm_inv);
        }
    }

    #[test]
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "avx"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    fn f64x4() {
        for i in 0..1000 {
            let val = i as f64;
            let arr1 = [val, val + 1.0, val + 2.0, val + 3.0];
            let arr2 = [val + 3.0, val + 2.0, val + 1.0, val];

            let simd_add = arr1.simd_add(&arr2);
            let norm_add = ops_on(arr1, arr2, f64::add);
            assert_eq!(simd_add, norm_add);

            let simd_sub = arr1.simd_sub(&arr2);
            let norm_sub = ops_on(arr1, arr2, f64::sub);
            assert_eq!(simd_sub, norm_sub);

            let simd_mul = arr1.simd_mul(&arr2);
            let norm_mul = ops_on(arr1, arr2, f64::mul);
            assert_eq!(simd_mul, norm_mul);

            let simd_div = arr1.simd_div(&arr2);
            let norm_div = ops_on(arr1, arr2, f64::div);
            assert_eq!(simd_div, norm_div);

            let simd_mul_add = arr1.simd_mul_add(&arr2, &arr2);
            let norm_mul_add = ops_on(arr1, arr2, mul_add);
            assert_eq!(simd_mul_add, norm_mul_add);

            let simd_inv = arr1.simd_inv();
            let norm_inv = ops_on(arr1, arr2, inv);
            assert_eq!(simd_inv, norm_inv);
        }
    }
}
