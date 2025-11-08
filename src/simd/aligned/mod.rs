mod arr;
pub use arr::Simd;

mod slice;
pub use slice::SimdSlice;

use super::backend::AlignedSimd;

#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    repr(align(64))
)]
#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx"), repr(align(32)))]
#[cfg_attr(
    any(
        all(target_arch = "x86_64", target_feature = "sse"),
        all(target_arch = "aarch64", target_feature = "neon")
    ),
    repr(align(16))
)]
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct SimdAligned<T: ?Sized>(pub T);

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse"),
    all(target_arch = "aarch64", target_feature = "neon")
))]
impl AlignedSimd<[f32; 4], f32, 4> for [f32; 4] {
    type Backend = F32x4;
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
impl AlignedSimd<[f32; 8], f32, 8> for [f32; 8] {
    type Backend = F32x8;
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
impl AlignedSimd<[f64; 4], f64, 4> for [f64; 4] {
    type Backend = F64x4;
}

impl
    AlignedSimd<
        [f64; crate::MAX_SIMD_SINGLE_PRECISION_LANES],
        f64,
        { crate::MAX_SIMD_SINGLE_PRECISION_LANES },
    > for [f64; crate::MAX_SIMD_SINGLE_PRECISION_LANES]
{
    type Backend =
        super::backend::DoublePrecisionNoSimd<{ crate::MAX_SIMD_SINGLE_PRECISION_LANES }>;
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
            let va = _mm256_load_ps($a.as_ptr());
            let vb = _mm256_load_ps($b.as_ptr());
            let mut out = SimdAligned([0f32; 8]);
            _mm256_store_ps(out.0.as_mut_ptr(), $op1(va, vb));
            out.0
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
            let va = _mm_load_ps($a.as_ptr());
            let vb = _mm_load_ps($b.as_ptr());
            let mut out = SimdAligned([0f32; 4]);
            _mm_store_ps(out.0.as_mut_ptr(), $op1(va, vb));
            out.0
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            use core::arch::aarch64::*;
            let va = vld1q_f32($a.as_ptr());
            let vb = vld1q_f32($b.as_ptr());
            let mut out = SimdAligned([0f32; 4]);
            vst1q_f32(out.0.as_mut_ptr(), $op2(va, vb));
            out.0
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
            let va = _mm256_load_pd($a.as_ptr());
            let vb = _mm256_load_pd($b.as_ptr());
            let mut out = SimdAligned([0f64; 4]);
            _mm256_store_pd(out.0.as_mut_ptr(), $op1(va, vb));
            out.0
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            use core::arch::aarch64::*;
            let va = vld1q_f64($a.as_ptr());
            let vb = vld1q_f64($b.as_ptr());
            let mut out = SimdAligned([0f64; 4]);
            vst1q_f64(out.0.as_mut_ptr(), $op2(va, vb));
            out.0
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
            const _ASSERT_SIZE: () = {
                core::assert!(core::mem::size_of::<[$type; $len]>() <= $crate::MAX_SIMD_SIZE, "array too large for SIMD on target");
                core::assert!($len <= $crate::MAX_SIMD_SINGLE_PRECISION_LANES, "too many lanes for SIMD on target");
            };
        }

        impl $crate::scalar::ImplicitSupersetOf<$type> for $name {}

        impl $crate::scalar::AcceleratedScalar<$type, $len> for $name {}

        impl $crate::simd::backend::AlignedSimdOps<[$type; $len], [$type; $len], $type, $len> for $name {
            unsafe fn simd_add(a: &[$type; $len], b: &[$type; $len]) -> [$type; $len] {
                const {
                    let _ = Self::_ASSERT_SIZE;
                }

                simd_ops_internal!(@add [$type; $len], a, b)
            }

            unsafe fn simd_sub(a: &[$type; $len], b: &[$type; $len]) -> [$type; $len] {
                const {
                    let _ = Self::_ASSERT_SIZE;
                }

                simd_ops_internal!(@sub [$type; $len], a, b)
            }

            unsafe fn simd_mul(a: &[$type; $len], b: &[$type; $len]) -> [$type; $len] {
                const {
                    let _ = Self::_ASSERT_SIZE;
                }

                simd_ops_internal!(@mul [$type; $len], a, b)
            }

            unsafe fn simd_div(a: &[$type; $len], b: &[$type; $len]) -> [$type; $len] {
                const {
                    let _ = Self::_ASSERT_SIZE;
                }

                simd_ops_internal!(@div [$type; $len], a, b)
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
