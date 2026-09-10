// Tiled matmul is adapted from NVlabs/cuda-oxide's Apache-2.0 tiled_gemm example.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::too_many_arguments)]
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;
use serde::Deserialize;
use std::{error::Error, fs, path::Path};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn tiled_matmul(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut out: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        static mut TA: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TB: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * 16 + ty;
        let col = thread::blockIdx_x() as usize * 16 + tx;
        let mut sum = 0.0f32;
        let mut tile = 0usize;
        while tile < (k as usize).div_ceil(16) {
            let ak = tile * 16 + tx;
            let bk = tile * 16 + ty;
            // Each of the 256 threads owns one tile cell. All threads reach
            // both barriers, including those outside the matrix at the edges.
            unsafe {
                TA[ty * 16 + tx] = if row < m as usize && ak < k as usize {
                    a[row * k as usize + ak]
                } else {
                    0.0
                };
                TB[ty * 16 + tx] = if bk < k as usize && col < n as usize {
                    b[bk * n as usize + col]
                } else {
                    0.0
                };
            }
            thread::sync_threads();
            let mut i = 0usize;
            while i < 16 {
                unsafe {
                    sum += TA[ty * 16 + i] * TB[i * 16 + tx];
                }
                i += 1;
            }
            thread::sync_threads();
            tile += 1;
        }
        if let Some(index) = thread::index_2d_runtime(&out) {
            if row < m as usize {
                if let Some(cell) = out.get_mut(index) {
                    *cell = sum;
                }
            }
        }
    }

    /// Register-tiled matmul: 64x64 block tile, 4x4 per thread, 32-deep K steps.
    ///
    /// Each thread reuses four A values against four B values, so shared-memory
    /// traffic per multiply-add is a quarter of the one-output-per-thread kernel
    /// above. Global loads move four floats per instruction into shared memory.
    /// The host launches this only when `m` and `n` are multiples of 64 and `k` of
    /// 32; `tiled_matmul` covers every other shape, including the edges.
    #[kernel]
    pub fn tiled_matmul_registers(
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        mut out: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        use cuda_device::vector::{self, F32x4};

        // A is stored transposed (k-major, row stride 68) so each thread reads the
        // four rows it owns with one 128-bit load instead of four scalar loads.
        static mut AT: SharedArray<f32, 4352, 16> = SharedArray::UNINIT; // 64 k-rows x 68 stride
        static mut BS: SharedArray<f32, 4096, 16> = SharedArray::UNINIT; // 64 rows x 64 columns
        let tx = thread::threadIdx_x() as usize; // 0..16, column group
        let ty = thread::threadIdx_y() as usize; // 0..16, row group
        let tid = ty * 16 + tx;
        let kk = k as usize;
        let nn = n as usize;
        let row0 = thread::blockIdx_y() as usize * 64;
        let col0 = thread::blockIdx_x() as usize * 64;
        let mut acc = [[0.0f32; 4]; 4];
        let mut step = 0usize;
        while step < kk {
            // Four quad passes per tile: A (64x64, stored transposed) and B (64x64).
            let b_quads = unsafe { core::ptr::addr_of_mut!(BS) }.cast::<F32x4>();
            let mut pass = 0usize;
            while pass < 4 {
                let q = tid + pass * 256;
                let row = q / 16;
                let col = (q % 16) * 4;
                let a_start = (row0 + row) * kk + step + col;
                if let Some(quad) = vector::as_vectors::<F32x4>(&a[a_start..a_start + 4]) {
                    let v = quad[0].as_slice();
                    unsafe {
                        AT[col * 68 + row] = v[0];
                        AT[(col + 1) * 68 + row] = v[1];
                        AT[(col + 2) * 68 + row] = v[2];
                        AT[(col + 3) * 68 + row] = v[3];
                    }
                }
                let b_start = (step + row) * nn + col0 + col;
                if let Some(quad) = vector::as_vectors::<F32x4>(&b[b_start..b_start + 4]) {
                    unsafe {
                        *b_quads.add(row * 16 + col / 4) = quad[0];
                    }
                }
                pass += 1;
            }
            thread::sync_threads();
            let mut i = 0usize;
            while i < 64 {
                let mut av = [0.0f32; 4];
                // One 128-bit shared read for the four rows this thread owns.
                let a_base = core::ptr::addr_of!(AT).cast::<F32x4>();
                let a_lanes = unsafe { (*a_base.add(i * 17 + ty)).0 };
                av[0] = a_lanes[0];
                av[1] = a_lanes[1];
                av[2] = a_lanes[2];
                av[3] = a_lanes[3];
                let mut bv = [0.0f32; 4];
                // One 128-bit shared read for the four columns this thread owns.
                let base = core::ptr::addr_of!(BS).cast::<F32x4>();
                let lanes = unsafe { (*base.add(i * 16 + tx)).0 };
                bv[0] = lanes[0];
                bv[1] = lanes[1];
                bv[2] = lanes[2];
                bv[3] = lanes[3];
                let mut c = 0usize;
                let mut r = 0usize;
                r = 0;
                while r < 4 {
                    c = 0;
                    while c < 4 {
                        acc[r][c] += av[r] * bv[c];
                        c += 1;
                    }
                    r += 1;
                }
                i += 1;
            }
            thread::sync_threads();
            step += 64;
        }
        // SAFETY: the host validated `m * n` output elements and launched exactly
        // one 64x64 tile per block, so every written cell is inside the buffer and
        // belongs to this block alone.
        let out_ptr = out.as_mut_ptr();
        let mut r = 0usize;
        while r < 4 {
            let mut c = 0usize;
            while c < 4 {
                unsafe {
                    *out_ptr.add((row0 + ty * 4 + r) * nn + col0 + tx * 4 + c) = acc[r][c];
                }
                c += 1;
            }
            r += 1;
        }
    }

    #[kernel]
    pub fn bias_gelu(width: u32, x: &[f32], bias: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(cell) = out.get_mut(index) {
            let z = x[i] + bias[i % width as usize];
            let t = cuda_device::float::tanh_approx_f32(0.7978845608 * (z + 0.044715 * z * z * z));
            *cell = 0.5 * z * (1.0 + t);
        }
    }

    #[kernel]
    pub fn layer_norm(
        width: u32,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        static mut REDUCE: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize;
        let d = width as usize;
        let base = row * d;
        let mut sum = 0.0f32;
        let mut j = tid;
        while j < d {
            sum += x[base + j];
            j += 256;
        }
        unsafe {
            REDUCE[tid] = sum;
        }
        thread::sync_threads();
        let mut stride = 128usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    REDUCE[tid] += REDUCE[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        let mean = unsafe { REDUCE[0] } / d as f32;
        // Preserve every thread's mean before reusing shared memory.
        thread::sync_threads();
        let mut variance = 0.0f32;
        j = tid;
        while j < d {
            let centered = x[base + j] - mean;
            variance += centered * centered;
            j += 256;
        }
        unsafe {
            REDUCE[tid] = variance;
        }
        thread::sync_threads();
        stride = 128;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    REDUCE[tid] += REDUCE[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        let inv = 1.0 / cuda_device::float::sqrt_rn_f32(unsafe { REDUCE[0] } / d as f32 + 1e-5);
        j = tid;
        while j < d {
            // Host validates rows*width elements and exactly 256 threads/block.
            // Row blocks are disjoint; thread t owns columns t + 256*q.
            unsafe {
                *out.as_mut_ptr().add(base + j) = (x[base + j] - mean) * inv * gamma[j] + beta[j];
            }
            j += 256;
        }
    }

    /// One warp per row: 128-bit quad access and shuffle reductions, no barriers.
    ///
    /// The host launches this only when `width` is a multiple of four, so the quad
    /// views exist; the scalar `layer_norm` handles every other width.
    #[kernel]
    pub fn layer_norm_warp(
        width: u32,
        rows: u32,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        use cuda_device::vector::{self, F32x4};
        use cuda_device::warp;

        let d = width as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        // The host fixes 256 threads per block, so each block owns eight rows.
        let row = thread::blockIdx_x() as usize * 8 + (tid >> 5);
        if row >= rows as usize {
            return;
        }
        let base = row * d;
        let Some(quads) = vector::as_vectors::<F32x4>(&x[base..base + d]) else {
            return;
        };
        let Some(gamma_quads) = vector::as_vectors::<F32x4>(gamma) else {
            return;
        };
        let Some(beta_quads) = vector::as_vectors::<F32x4>(beta) else {
            return;
        };

        let mut sum = 0.0f32;
        let mut q = lane;
        while q < quads.len() {
            let v = quads[q].as_slice();
            sum += v[0] + v[1] + v[2] + v[3];
            q += 32;
        }
        let mut offset = 16u32;
        while offset > 0 {
            sum += warp::shuffle_down_f32(sum, offset);
            offset >>= 1;
        }
        let mean = warp::shuffle_f32(sum, 0) / d as f32;

        let mut variance = 0.0f32;
        q = lane;
        while q < quads.len() {
            let v = quads[q].as_slice();
            let (c0, c1, c2, c3) = (v[0] - mean, v[1] - mean, v[2] - mean, v[3] - mean);
            variance += c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3;
            q += 32;
        }
        offset = 16;
        while offset > 0 {
            variance += warp::shuffle_down_f32(variance, offset);
            offset >>= 1;
        }
        let inv = 1.0
            / cuda_device::float::sqrt_rn_f32(warp::shuffle_f32(variance, 0) / d as f32 + 1e-5);

        // SAFETY: the host validated `rows * width` output elements and each warp
        // owns the disjoint row at `base`, so this view aliases no other warp.
        let out_row = unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr().add(base), d) };
        let Some(out_quads) = vector::as_vectors_mut::<F32x4>(out_row) else {
            return;
        };
        q = lane;
        while q < quads.len() {
            let xv = quads[q].as_slice();
            let gv = gamma_quads[q].as_slice();
            let bv = beta_quads[q].as_slice();
            out_quads[q] = F32x4::new([
                (xv[0] - mean) * inv * gv[0] + bv[0],
                (xv[1] - mean) * inv * gv[1] + bv[1],
                (xv[2] - mean) * inv * gv[2] + bv[2],
                (xv[3] - mean) * inv * gv[3] + bv[3],
            ]);
            q += 32;
        }
    }

    /// Two warps per row, half a row each, combined through one shared exchange.
    ///
    /// Doubling the warps per row doubles the threads the grid can keep resident,
    /// which is what the single-warp version is short of. Each warp reduces its half
    /// with shuffles; the two partial sums meet once in shared memory behind a single
    /// barrier. The host picks this for widths up to 2048 that are multiples of four.
    #[kernel]
    pub fn layer_norm_pair(
        width: u32,
        rows: u32,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        use cuda_device::vector::{self, F32x4};
        use cuda_device::warp;

        static mut PARTIAL: SharedArray<f32, 16> = SharedArray::UNINIT;
        let d = width as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp = tid >> 5;
        // 256 threads per block: four rows, two warps each.
        let half = warp & 1;
        let row = thread::blockIdx_x() as usize * 4 + (warp >> 1);
        if row >= rows as usize {
            return;
        }
        let base = row * d;
        let Some(quads) = vector::as_vectors::<F32x4>(&x[base..base + d]) else {
            return;
        };
        let Some(gamma_quads) = vector::as_vectors::<F32x4>(gamma) else {
            return;
        };
        let Some(beta_quads) = vector::as_vectors::<F32x4>(beta) else {
            return;
        };
        let span = quads.len().div_ceil(2);

        let mut sum = 0.0f32;
        let mut squares = 0.0f32;
        let mut i = 0usize;
        while i * 32 < span {
            let q = half * span + lane + 32 * i;
            if q < quads.len() {
                let v = quads[q].as_slice();
                sum += v[0] + v[1] + v[2] + v[3];
                squares += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
            }
            i += 1;
        }
        let mut offset = 16u32;
        while offset > 0 {
            sum += warp::shuffle_down_f32(sum, offset);
            squares += warp::shuffle_down_f32(squares, offset);
            offset >>= 1;
        }
        if lane == 0 {
            unsafe {
                PARTIAL[warp * 2] = sum;
                PARTIAL[warp * 2 + 1] = squares;
            }
        }
        thread::sync_threads();
        let pair = (warp >> 1) * 4;
        let total = unsafe { PARTIAL[pair] + PARTIAL[pair + 2] };
        let total_squares = unsafe { PARTIAL[pair + 1] + PARTIAL[pair + 3] };
        let mean = total / d as f32;
        let variance = total_squares / d as f32 - mean * mean;
        let inv = 1.0 / cuda_device::float::sqrt_rn_f32(variance + 1e-5);

        // SAFETY: the host validated `rows * width` output elements and each warp
        // owns the disjoint half-row at `base`, so this view aliases no other warp.
        let out_row = unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr().add(base), d) };
        let Some(out_quads) = vector::as_vectors_mut::<F32x4>(out_row) else {
            return;
        };
        i = 0;
        while i * 32 < span {
            let q = half * span + lane + 32 * i;
            if q < quads.len() {
                let v = quads[q].as_slice();
                let gv = gamma_quads[q].as_slice();
                let bv = beta_quads[q].as_slice();
                out_quads[q] = F32x4::new([
                    (v[0] - mean) * inv * gv[0] + bv[0],
                    (v[1] - mean) * inv * gv[1] + bv[1],
                    (v[2] - mean) * inv * gv[2] + bv[2],
                    (v[3] - mean) * inv * gv[3] + bv[3],
                ]);
            }
            i += 1;
        }
    }

    #[kernel]
    pub fn triangle(n: u32, channels: u32, a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let flat = index.get();
        if let Some(cell) = out.get_mut(index) {
            let c = flat % channels as usize;
            let pair = flat / channels as usize;
            let i = pair / n as usize;
            let j = pair % n as usize;
            let mut sum = 0.0f32;
            let mut k = 0usize;
            while k < n as usize {
                sum += a[(i * n as usize + k) * channels as usize + c]
                    * b[(j * n as usize + k) * channels as usize + c];
                k += 1;
            }
            *cell = sum;
        }
    }

    #[kernel]
    pub fn neighbor(
        width: u32,
        x: &[f32],
        weights: &[f32],
        rowptr: &[u32],
        indices: &[u32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let flat = index.get();
        if let Some(cell) = out.get_mut(index) {
            let row = flat / width as usize;
            let feature = flat % width as usize;
            let mut edge = rowptr[row] as usize;
            let end = rowptr[row + 1] as usize;
            let mut sum = 0.0f32;
            while edge < end {
                sum += weights[edge] * x[indices[edge] as usize * width as usize + feature];
                edge += 1;
            }
            *cell = sum;
        }
    }
}

// Rendered per measurement by scripts/evolve.py. `include!` keeps the
// `#[cuda_module]` body inline, which the macro requires of a module.
include!("candidates.rs");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    op: String,
    dims: Vec<usize>,
    warmup: usize,
    iterations: usize,
    /// `best` or `new` runs a generated variant; absent runs the committed kernels.
    #[serde(default)]
    variant: Option<String>,
}
fn read_f32(path: &Path, count: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() != count.checked_mul(4).ok_or("input size overflow")? {
        return Err(format!("{}: incorrect byte length", path.display()).into());
    }
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err("inputs must be finite FP32".into());
    }
    Ok(values)
}
fn read_u32(path: &Path, count: usize) -> Result<Vec<u32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() != count.checked_mul(4).ok_or("input size overflow")? {
        return Err("invalid CSR byte length".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn product(dims: &[usize]) -> Result<usize, Box<dyn Error>> {
    let size = dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or("shape overflow")?;
    if size == 0 || size > i32::MAX as usize {
        return Err("shape product must be in 1..=i32::MAX".into());
    }
    Ok(size)
}
struct Capture {
    library: libloading::Library,
    active: bool,
}
impl Capture {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            library: unsafe { libloading::Library::new("libcuda.so.1")? },
            active: false,
        })
    }
    fn call(&self, name: &[u8]) -> Result<(), Box<dyn Error>> {
        let function: libloading::Symbol<unsafe extern "C" fn() -> i32> =
            unsafe { self.library.get(name)? };
        let status = unsafe { function() };
        if status != 0 {
            return Err(format!("CUDA profiler API error {status}").into());
        }
        Ok(())
    }
    fn start(&mut self) -> Result<(), Box<dyn Error>> {
        self.call(b"cuProfilerStart\0")?;
        self.active = true;
        Ok(())
    }
    fn stop(&mut self) -> Result<(), Box<dyn Error>> {
        self.call(b"cuProfilerStop\0")?;
        self.active = false;
        Ok(())
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        if self.active {
            let _ = self.stop();
        }
    }
}

/// Which generated role a run asked for: the loop's incumbent or its proposal.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CandidateRole {
    Best,
    New,
}

impl CandidateRole {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "best" => Ok(Self::Best),
            "new" => Ok(Self::New),
            other => Err(format!("unknown variant {other:?}; expected \"best\" or \"new\"").into()),
        }
    }
}

/// A generated variant together with the launch geometry baked into its render.
enum CandidateLaunch {
    Matmul {
        role: CandidateRole,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
    },
    Layernorm {
        role: CandidateRole,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
    },
}

impl CandidateLaunch {
    fn geometry(&self) -> ((u32, u32, u32), (u32, u32, u32)) {
        match *self {
            CandidateLaunch::Matmul {
                grid_dim, block_dim, ..
            }
            | CandidateLaunch::Layernorm {
                grid_dim, block_dim, ..
            } => (grid_dim, block_dim),
        }
    }
}

/// Resolve the generated variant's launch geometry, rejecting a shape its tile
/// does not divide. A generated variant is shape-specialised on purpose: falling
/// back to a different kernel would make the measurement describe something the
/// manifest did not ask for.
fn resolve_candidate(
    op: &str,
    dims: &[usize],
    role: CandidateRole,
) -> Result<CandidateLaunch, Box<dyn Error>> {
    match op {
        "matmul" => {
            let (block_m, block_n, k_step) = match role {
                CandidateRole::Best => candidates::MATMUL_TILE_BEST,
                CandidateRole::New => candidates::MATMUL_TILE_NEW,
            };
            let (threads_x, threads_y) = match role {
                CandidateRole::Best => candidates::MATMUL_BLOCK_BEST,
                CandidateRole::New => candidates::MATMUL_BLOCK_NEW,
            };
            let (m, n, k) = (dims[0], dims[1], dims[2]);
            if block_m == 0
                || block_n == 0
                || k_step == 0
                || m % block_m as usize != 0
                || n % block_n as usize != 0
                || k % k_step as usize != 0
            {
                return Err(format!(
                    "generated variant: shape [{m}, {n}, {k}] is not a multiple of the \
                     {block_m}x{block_n} tile with k step {k_step}"
                )
                .into());
            }
            Ok(CandidateLaunch::Matmul {
                role,
                grid_dim: (n as u32 / block_n, m as u32 / block_m, 1),
                block_dim: (threads_x, threads_y, 1),
            })
        }
        "layernorm" => {
            let (rows_per_block, threads) = match role {
                CandidateRole::Best => candidates::LAYERNORM_CFG_BEST,
                CandidateRole::New => candidates::LAYERNORM_CFG_NEW,
            };
            let (rows, width) = (dims[0], dims[1]);
            if rows_per_block == 0
                || threads == 0
                || rows % rows_per_block as usize != 0
                || width % 4 != 0
            {
                return Err(format!(
                    "generated variant: shape [{rows}, {width}] needs rows divisible by \
                     {rows_per_block} and width divisible by 4"
                )
                .into());
            }
            Ok(CandidateLaunch::Layernorm {
                role,
                grid_dim: (rows as u32 / rows_per_block, 1, 1),
                block_dim: (threads, 1, 1),
            })
        }
        other => Err(format!(
            "generated variants exist for matmul and layernorm, not {other:?}"
        )
        .into()),
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory = args
        .next()
        .ok_or("usage: mage-oxide INPUT_DIR [--iterations N] [--capture]")?;
    let dir = Path::new(&directory);
    let mut manifest: Manifest = serde_json::from_slice(&fs::read(dir.join("input.json"))?)?;
    let mut capture_requested = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iterations" => {
                manifest.iterations = args.next().ok_or("missing iteration count")?.parse()?
            }
            "--capture" => capture_requested = true,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if manifest.iterations == 0 || manifest.iterations > 100000 || manifest.warmup > 10000 {
        return Err("invalid iteration count".into());
    }
    let dims = &manifest.dims;
    let expected_dims = match manifest.op.as_str() {
        "matmul" | "neighbor" => 3,
        "gelu" | "layernorm" | "triangle" => 2,
        _ => return Err("unknown operation".into()),
    };
    if dims.len() != expected_dims || dims.iter().any(|&d| d > i32::MAX as usize) {
        return Err("invalid dimensions".into());
    }
    let (a_len, b_len, c_len, out_len) = match manifest.op.as_str() {
        "matmul" => (
            product(&[dims[0], dims[2]])?,
            product(&[dims[2], dims[1]])?,
            0,
            product(&dims[..2])?,
        ),
        "gelu" => (product(dims)?, dims[1], 0, product(dims)?),
        "layernorm" => (product(dims)?, dims[1], dims[1], product(dims)?),
        "triangle" => {
            let len = product(&[dims[0], dims[0], dims[1]])?;
            (len, len, 0, len)
        }
        "neighbor" => (product(&dims[..2])?, dims[2], 0, product(&dims[..2])?),
        _ => unreachable!(),
    };
    let a = read_f32(&dir.join("a.bin"), a_len)?;
    let mut b = read_f32(&dir.join("b.bin"), b_len)?;
    let c = if c_len > 0 {
        read_f32(&dir.join("c.bin"), c_len)?
    } else {
        vec![0.0]
    };
    let (rowptr, mut indices) = if manifest.op == "neighbor" {
        let ptr = read_u32(&dir.join("rowptr.bin"), dims[0] + 1)?;
        let idx = read_u32(&dir.join("indices.bin"), dims[2])?;
        if ptr[0] != 0
            || ptr[dims[0]] as usize != dims[2]
            || ptr.windows(2).any(|w| w[0] > w[1])
            || idx.iter().any(|&i| i as usize >= dims[0])
        {
            return Err("invalid CSR adjacency".into());
        }
        (ptr, idx)
    } else {
        (vec![0u32], vec![0u32])
    };
    if b.is_empty() {
        b.push(0.0);
    }
    if indices.is_empty() {
        indices.push(0);
    }
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let c_dev = DeviceBuffer::from_host(&stream, &c)?;
    let ptr_dev = DeviceBuffer::from_host(&stream, &rowptr)?;
    let idx_dev = DeviceBuffer::from_host(&stream, &indices)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, out_len)?;
    let module = kernels::load(&ctx)?;
    let candidate_role = match manifest.variant.as_deref() {
        Some(value) => Some(CandidateRole::parse(value)?),
        None => None,
    };
    let candidate_module = match candidate_role {
        Some(_) => Some(candidates::load(&ctx)?),
        None => None,
    };
    let candidate_launch = match candidate_role {
        Some(role) => Some(resolve_candidate(&manifest.op, dims, role)?),
        None => None,
    };
    let linear = LaunchConfig {
        grid_dim: ((out_len as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = || -> Result<(), cuda_core::DriverError> {
        // SAFETY: inputs and CSR bounds were validated before device allocation;
        // each kernel receives the required launch shape and distinct output.
        unsafe {
            if let (Some(candidate_module), Some(plan)) =
                (candidate_module.as_ref(), candidate_launch.as_ref())
            {
                let (grid_dim, block_dim) = plan.geometry();
                let config = LaunchConfig {
                    grid_dim,
                    block_dim,
                    shared_mem_bytes: 0,
                };
                let width = dims[1] as u32;
                return match plan {
                    CandidateLaunch::Matmul {
                        role: CandidateRole::Best,
                        ..
                    } => candidate_module.matmul_best(
                        &stream,
                        config,
                        width,
                        dims[2] as u32,
                        &a_dev,
                        &b_dev,
                        cuda_host::RowWidth::new(&mut out, width),
                    ),
                    CandidateLaunch::Matmul {
                        role: CandidateRole::New,
                        ..
                    } => candidate_module.matmul_new(
                        &stream,
                        config,
                        width,
                        dims[2] as u32,
                        &a_dev,
                        &b_dev,
                        cuda_host::RowWidth::new(&mut out, width),
                    ),
                    CandidateLaunch::Layernorm {
                        role: CandidateRole::Best,
                        ..
                    } => candidate_module.layernorm_best(
                        &stream,
                        config,
                        width,
                        dims[0] as u32,
                        &a_dev,
                        &b_dev,
                        &c_dev,
                        &mut out,
                    ),
                    CandidateLaunch::Layernorm {
                        role: CandidateRole::New,
                        ..
                    } => candidate_module.layernorm_new(
                        &stream,
                        config,
                        width,
                        dims[0] as u32,
                        &a_dev,
                        &b_dev,
                        &c_dev,
                        &mut out,
                    ),
                };
            }
            match manifest.op.as_str() {
                "matmul" => {
                    let (m, n, k) = (dims[0], dims[1], dims[2]);
                    if m % 64 == 0 && n % 64 == 0 && k % 64 == 0 {
                        // Square 64x64 tiles, no edge masking needed.
                        module.tiled_matmul_registers(
                            &stream,
                            LaunchConfig {
                                grid_dim: ((n / 64) as u32, (m / 64) as u32, 1),
                                block_dim: (16, 16, 1),
                                shared_mem_bytes: 0,
                            },
                            n as u32,
                            k as u32,
                            &a_dev,
                            &b_dev,
                            cuda_host::RowWidth::new(&mut out, n as u32),
                        )
                    } else {
                        module.tiled_matmul(
                            &stream,
                            LaunchConfig {
                                grid_dim: (
                                    (dims[1] as u32).div_ceil(16),
                                    (dims[0] as u32).div_ceil(16),
                                    1,
                                ),
                                block_dim: (16, 16, 1),
                                shared_mem_bytes: 0,
                            },
                            dims[0] as u32,
                            dims[1] as u32,
                            dims[2] as u32,
                            &a_dev,
                            &b_dev,
                            cuda_host::RowWidth::new(&mut out, dims[1] as u32),
                        )
                    }
                }
                "gelu" => {
                    module.bias_gelu(&stream, linear, dims[1] as u32, &a_dev, &b_dev, &mut out)
                }
                "layernorm" => {
                    let rows = dims[0] as u32;
                    let width = dims[1] as u32;
                    // Eight rows per block, one warp each.
                    let block = LaunchConfig {
                        grid_dim: (rows.div_ceil(8), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    if width % 4 == 0 && width <= 2048 {
                        // Two warps per row: 1024 blocks instead of 512, so more warps stay resident.
                        module.layer_norm_pair(
                            &stream,
                            LaunchConfig {
                                grid_dim: (rows.div_ceil(4), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            width,
                            rows,
                            &a_dev,
                            &b_dev,
                            &c_dev,
                            &mut out,
                        )
                    } else if width % 4 == 0 {
                        module.layer_norm_warp(
                            &stream, block, width, rows,
                            &a_dev, &b_dev, &c_dev, &mut out,
                        )
                    } else {
                        module.layer_norm(
                            &stream,
                            LaunchConfig {
                                grid_dim: (rows, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            width,
                            &a_dev,
                            &b_dev,
                            &c_dev,
                            &mut out,
                        )
                    }
                }
                "triangle" => module.triangle(
                    &stream,
                    linear,
                    dims[0] as u32,
                    dims[1] as u32,
                    &a_dev,
                    &b_dev,
                    &mut out,
                ),
                "neighbor" => module.neighbor(
                    &stream,
                    linear,
                    dims[1] as u32,
                    &a_dev,
                    &b_dev,
                    &ptr_dev,
                    &idx_dev,
                    &mut out,
                ),
                _ => unreachable!(),
            }
        }
    };
    let mut capture = Capture::new()?;
    for _ in 0..manifest.warmup {
        launch()?;
    }
    stream.synchronize()?;
    let start = ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end = ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let mut samples = Vec::with_capacity(manifest.iterations);
    if capture_requested {
        capture.start()?;
    }
    for _ in 0..manifest.iterations {
        start.record(&stream)?;
        launch()?;
        end.record(&stream)?;
        samples.push(start.elapsed_ms(&end)? as f64 * 1000.0);
    }
    if capture_requested {
        capture.stop()?;
    }
    let result = out.to_host_vec(&stream)?;
    let run_id = format!(
        "{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
        std::process::id()
    );
    let run_dir = dir.join("rust-runs").join(&run_id);
    fs::create_dir_all(&run_dir)?;
    let output_bytes: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(run_dir.join("output.bin"), &output_bytes)?;
    fs::write(dir.join("rust-output.bin"), &output_bytes)?;
    // Provenance: a timing record has to say which kernel produced it, including
    // which committed kernel the shape selected when no variant was requested.
    let variant = manifest
        .variant
        .clone()
        .unwrap_or_else(|| "committed".to_string());
    let (variant_kernel, variant_params) = match (candidate_role, manifest.op.as_str()) {
        (Some(CandidateRole::Best), "matmul") => {
            ("matmul_best", candidates::BEST_MATMUL_PARAMS_JSON)
        }
        (Some(CandidateRole::New), "matmul") => ("matmul_new", candidates::NEW_MATMUL_PARAMS_JSON),
        (Some(CandidateRole::Best), _) => {
            ("layernorm_best", candidates::BEST_LAYERNORM_PARAMS_JSON)
        }
        (Some(CandidateRole::New), _) => {
            ("layernorm_new", candidates::NEW_LAYERNORM_PARAMS_JSON)
        }
        (None, "matmul") => (
            if dims[0] % 64 == 0 && dims[1] % 64 == 0 && dims[2] % 64 == 0 {
                "tiled_matmul_registers"
            } else {
                "tiled_matmul"
            },
            "",
        ),
        (None, "layernorm") => (
            if dims[1] % 4 == 0 && dims[1] <= 2048 {
                "layer_norm_pair"
            } else if dims[1] % 4 == 0 {
                "layer_norm_warp"
            } else {
                "layer_norm"
            },
            "",
        ),
        (None, "gelu") => ("bias_gelu", ""),
        (None, "triangle") => ("triangle", ""),
        (None, _) => ("neighbor", ""),
    };
    let timing = serde_json::json!({"op": manifest.op, "dims": dims, "warmup": manifest.warmup,
        "iterations": manifest.iterations, "capture": capture_requested, "samples_us": samples,
        "event_mean_us": samples.iter().sum::<f64>() / samples.len() as f64,
        "precision": "FP32 scalar arithmetic; no tensor cores", "run_id": run_id,
        "variant": variant, "variant_kernel": variant_kernel, "variant_params": variant_params});
    fs::write(
        run_dir.join("timing.json"),
        serde_json::to_string_pretty(&timing)? + "\n",
    )?;
    fs::write(
        dir.join("rust-timing.json"),
        serde_json::to_string_pretty(&timing)? + "\n",
    )?;
    println!("{}", serde_json::to_string(&timing)?);
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("mage-oxide: {error}");
        std::process::exit(1);
    }
}
