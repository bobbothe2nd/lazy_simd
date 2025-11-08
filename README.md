# Lazy SIMD

This library contains utilities to batch arrays into smaller chunks and perform light mathematical operations on each element. Every operation performed has the opportunity to be run with SIMD acceleration, but only when the target allows.

Here are some examples:

```rust
let c = add_arrays(&a, &b);
let c_simd = Simd::new(c);
let y = c_simd + x;
```
