//! With nothing but `core`, and `num_traits`, the entire crate is build to be
//! portable, yet powerful. It supports addition, multiplication, division, and
//! subtraction as the four main arithmetic operations performed - all of which
//! are improved by SIMD.

#![forbid(missing_docs)]
#![forbid(unsafe_op_in_unsafe_fn)]
#![allow(clippy::modulo_one)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod private {
    pub trait Internal {}
}

use crate::private::Internal;

pub mod scalar;
pub mod simd;

/// The largest size (in bytes) a SIMD type can have on the target.
///
/// A reflection of the largest contiguous array of bytes that can be stored in a
/// localized, singular, specialized SIMD register.
pub const MAX_SIMD_SIZE: usize = if cfg!(all(target_arch = "x86_64", target_feature = "avx512f")) {
    64 // 512 bits
} else if cfg!(all(target_arch = "x86_64", target_feature = "avx")) {
    32 // 256 bits
} else if cfg!(any(
    all(target_arch = "x86_64", target_feature = "sse"),
    all(target_arch = "aarch64", target_feature = "neon")
)) {
    16 // 128 bits
} else {
    0 // no simd supported
};

/// A representation of the total lanes in a SIMD register for single precision floats.
pub const MAX_SIMD_SINGLE_PRECISION_LANES: usize =
    if cfg!(all(target_arch = "x86_64", target_feature = "avx512f")) {
        16 // f32x16
    } else if cfg!(all(target_arch = "x86_64", target_feature = "avx")) {
        8 // f32x8
    } else if cfg!(any(
        all(target_arch = "x86_64", target_feature = "sse"),
        all(target_arch = "aarch64", target_feature = "neon")
    )) {
        4 // f32x4
    } else {
        1 // no simd supported
    };

/// A representation of the total lanes in a SIMD register for double precision floats.
pub const MAX_SIMD_DOUBLE_PRECISION_LANES: usize =
    if cfg!(all(target_arch = "x86_64", target_feature = "avx512f")) {
        8 // f64x8
    } else if cfg!(all(target_arch = "x86_64", target_feature = "avx")) {
        4 // f32x4
    } else if cfg!(any(
        all(target_arch = "x86_64", target_feature = "sse"),
        all(target_arch = "aarch64", target_feature = "neon")
    )) {
        2 // f64x2
    } else {
        1 // no simd supported
    };
