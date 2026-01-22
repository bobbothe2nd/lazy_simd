// right now, the `plain arithmetic` test greatly outperforms `Simd`...

use core::array::from_fn;
use lazy_simd::{
    scalar::AddByRef,
    simd::{add_arrays, Simd},
};
use std::time::Instant;

const LENGTH: usize = 1 << 14;
const ITERS: usize = 1 << 4;

static mut ARR_POOL: [[f32; LENGTH]; 4] = unsafe { core::mem::zeroed() };
static mut SIMD_POOL: [Simd<f32, LENGTH>; 2] = unsafe { core::mem::zeroed() };

fn main() {
    let arr_pool = unsafe { &mut *(&raw mut ARR_POOL) };
    let simd_pool = unsafe { &mut *(&raw mut SIMD_POOL) };

    let add_rhs = from_fn(|i| (i as f32).sqrt());
    let by_idx = from_fn(|i| i as _);

    for _ in 0..ITERS {
        let start = Instant::now();
        {
            arr_pool[0] = add_arrays(&by_idx, &add_rhs);
        }
        let elapsed_simd_unaligned = start.elapsed();

        let simd_by_idx = Simd::new(by_idx);
        let simd_add_rhs = Simd::new(add_rhs);

        let start = Instant::now();
        {
            simd_by_idx.add_into(&simd_add_rhs, &mut simd_pool[0]);
        }
        let elapsed_simd = start.elapsed();

        let start = Instant::now();
        {
            arr_pool[3] = {
                let out = &mut arr_pool[2];
                for (i, val) in by_idx.into_iter().enumerate() {
                    (*out)[i] = val + add_rhs[i];
                }
                arr_pool[2]
            };
        }
        let elapsed_generic = start.elapsed();

        println!(
            "\nSIMD-accelerated time: {:?}\nSIMD unaligned time: {:?}\nplain arithmetic time: {:?}",
            elapsed_simd, elapsed_simd_unaligned, elapsed_generic
        );

        let out0 = &simd_pool[0];
        let out1 = &arr_pool[0];
        let out2 = &arr_pool[3];

        assert_eq!(out0.as_array(), out1);
        assert_eq!(out1, out2);
    }
}
