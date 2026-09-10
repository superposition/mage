// cuTile Rust kernels for the Mage comparison harness.
//
// A peer of `examples/oxide`: same input directory, same flags, same output
// files, so one Python harness drives both. The kernels here are tile programs
// written in stable Rust and lowered to cubins through CUDA Tile IR; the
// comparison keeps strict FP32 arithmetic and checks every output against
// PyTorch.
//
// SPDX-License-Identifier: Apache-2.0

use cuda_core::Stream;
use cutile::api;
use cutile::prelude::*;
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::TileKernel;
use serde::Deserialize;
use std::{error::Error, fs, path::Path, sync::Arc};

/// Tile shapes. A partition shape is a compile-time tile shape in cuTile and
/// part of the JIT specialization key, so each run keeps to one shape.
const MATMUL_M: usize = 16;
const MATMUL_N: usize = 16;
const MATMUL_K: usize = 8;
const GELU_ROWS: usize = 8;
const GELU_COLS: usize = 128;
const TRIANGLE_TILE: usize = 32;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// Strict FP32 tiled matmul, `z = x @ y`.
    ///
    /// The tile is the unit of work, not the thread: each program owns one
    /// [BM, BN] output tile and walks the contraction axis in BK steps, the
    /// shape the upstream GEMM tutorial uses. The harness measures what `mma`
    /// produces on this hardware against the strict-FP32 PyTorch reference
    /// instead of assuming a precision mode.
    #[cutile::entry()]
    fn matmul<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        // The accumulator starts at zero here rather than loading `z`: the
        // harness launches the same kernel many times over one output buffer,
        // and `z = x @ y` must hold after every launch, not `n * (x @ y)`.
        let mut acc = constant(0.0f32, shape![BM, BN]);
        for step in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, step]);
            let tile_y = part_y.load([step, pid.1]);
            acc = mma(tile_x, tile_y, acc);
        }
        z.store(acc);
    }

    /// Bias + GELU, matching `F.gelu(x + bias, approximate="tanh")`.
    ///
    /// One program owns a [BM, BN] tile; the bias segment is taken from the
    /// tile's column block, so the whole tile shares one broadcast bias vector.
    #[cutile::entry()]
    fn bias_gelu<const BM: i32, const BN: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        bias: &Tensor<f32, { [-1] }>,
        cubic: f32,
        scale: f32,
        half: f32,
        one: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_bias = bias.partition(shape![BN]);
        let tile_x: Tile<f32, { [BM, BN] }> = x.load_like(y);
        let tile_bias: Tile<f32, { [BM, BN] }> = part_bias
            .load([pid.1])
            .reshape(shape![1, BN])
            .broadcast(y.shape());
        let cubic_tile = cubic.broadcast(y.shape());
        let scale_tile = scale.broadcast(y.shape());
        let half_tile = half.broadcast(y.shape());
        let one_tile = one.broadcast(y.shape());
        let t = tile_x + tile_bias;
        y.store(half_tile * t * (one_tile + tanh(scale_tile * (t + cubic_tile * t * t * t))));
    }

    /// Layer normalization, one row per program, biased variance, `eps = 1e-5`.
    ///
    /// The tile is a whole row, so both reductions cover it exactly and the
    /// affine parameters broadcast along it. Tile dimensions must be powers of
    /// two, so a row of arbitrary width is zero-padded up to `BW`; the padded
    /// lanes contribute nothing to either sum because `sumsq` is taken over the
    /// raw row, and the true width is the divisor.
    #[cutile::entry()]
    fn layer_norm<const BW: i32>(
        y: &mut Tensor<f32, { [1, BW] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        gamma: &Tensor<f32, { [-1] }>,
        beta: &Tensor<f32, { [-1] }>,
        w_true: f32,
        eps: f32,
    ) {
        let tile_x: Tile<f32, { [1, BW] }> = x.load_like(y);
        let inv_n = 1.0f32 / w_true;
        let inv_n_tile: Tile<f32, { [1, BW] }> = inv_n.broadcast(y.shape());
        let eps_tile: Tile<f32, { [1, BW] }> = eps.broadcast(y.shape());
        let sum_row: Tile<f32, { [1] }> = reduce_sum(tile_x, 1i32);
        let sumsq_row: Tile<f32, { [1] }> = reduce_sum(tile_x * tile_x, 1i32);
        let mean: Tile<f32, { [1, BW] }> = sum_row
            .reshape(shape![1, 1])
            .broadcast(y.shape())
            * inv_n_tile;
        let mean_square: Tile<f32, { [1, BW] }> = sumsq_row
            .reshape(shape![1, 1])
            .broadcast(y.shape())
            * inv_n_tile;
        let variance = mean_square - mean * mean;
        let inv_std: Tile<f32, { [1, BW] }> = rsqrt(variance + eps_tile, ftz::Disabled);
        let centered = tile_x - mean;
        let part_gamma = gamma.partition(shape![BW]);
        let part_beta = beta.partition(shape![BW]);
        let scale: Tile<f32, { [1, BW] }> = part_gamma
            .load([0])
            .reshape(shape![1, BW])
            .broadcast(y.shape());
        let shift: Tile<f32, { [1, BW] }> = part_beta
            .load([0])
            .reshape(shape![1, BW])
            .broadcast(y.shape());
        y.store(centered * inv_std * scale + shift);
    }

    /// Triangle contraction, `z[i, j, c] = sum_k a[i, k, c] * b[j, k, c]`.
    ///
    /// Each channel is one matrix multiply: the channel axis is the third grid
    /// axis, and the program multiplies a [BI, N] slab of `a` by the matching
    /// [N, BJ] slab of `b`. This is the batched-GEMM idiom the upstream
    /// `batch_matmul` example uses, with the channel grid axis carried into the
    /// load index.
    #[cutile::entry()]
    fn triangle<const BI: i32, const BJ: i32, const N: i32>(
        y: &mut Tensor<f32, { [BI, BJ, 1] }>,
        a: &Tensor<f32, { [-1, -1, -1] }>,
        b: &Tensor<f32, { [-1, -1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_a = a.partition(shape![BI, N, 1]);
        let part_b = b.partition(shape![BJ, N, 1]);
        let tile_a: Tile<f32, { [BI, N] }> =
            part_a.load([pid.0, 0, pid.2]).reshape(shape![BI, N]);
        let tile_b: Tile<f32, { [N, BJ] }> = part_b
            .load([pid.1, 0, pid.2])
            .reshape(shape![BJ, N])
            .transpose();
        let zero: Tile<f32, { [BI, BJ] }> = constant(0.0f32, shape![BI, BJ]);
        let acc = mma(tile_a, tile_b, zero);
        y.store(acc.reshape(shape![BI, BJ, 1]));
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    op: String,
    dims: Vec<usize>,
    warmup: usize,
    iterations: usize,
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

/// Zero-extend a row-major matrix on the right and the bottom, and a vector to
/// `len`. Tile dimensions must be powers of two and partitions must divide the
/// tensor, so shapes that are not multiples of a tile are padded before the
/// timed region; padded lanes are exact zeros, which contribute nothing to a
/// sum of products or to the reductions that consume them.
fn pad_rows(values: &[f32], rows: usize, cols: usize, rows_pad: usize, cols_pad: usize) -> Vec<f32> {
    let mut padded = vec![0.0f32; rows_pad * cols_pad];
    for row in 0..rows {
        padded[row * cols_pad..row * cols_pad + cols]
            .copy_from_slice(&values[row * cols..row * cols + cols]);
    }
    padded
}

fn pad_vector(values: &[f32], len: usize) -> Vec<f32> {
    let mut padded = vec![0.0f32; len];
    padded[..values.len()].copy_from_slice(values);
    padded
}

/// Zero-extend a channel-last [n, n, channels] stack in the two leading
/// dimensions.
fn pad_square(values: &[f32], n: usize, channels: usize, n_pad: usize) -> Vec<f32> {
    let mut padded = vec![0.0f32; n_pad * n_pad * channels];
    for i in 0..n {
        for j in 0..n {
            let source = (i * n + j) * channels;
            let target = (i * n_pad + j) * channels;
            padded[target..target + channels].copy_from_slice(&values[source..source + channels]);
        }
    }
    padded
}

/// CUDA profiler start/stop, so an external Nsight capture brackets exactly the
/// measured region. Same mechanism as `examples/oxide`: the driver's
/// `libcuda.so.1` is loaded lazily and the two entry points are called by name.
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

struct Harness {
    device: Arc<Device>,
    stream: Arc<Stream>,
}

impl Harness {
    fn new() -> Result<Self, Box<dyn Error>> {
        let device = Device::new(0)?;
        let stream = device.new_stream()?;
        Ok(Self { device, stream })
    }

    /// Warm up outside the measured region, then price each launch with one
    /// event pair. An event pair only prices a launch once the end event has
    /// completed; the PyTorch reference synchronizes the same way per sample.
    fn time(
        &self,
        warmup: usize,
        iterations: usize,
        capture_requested: bool,
        launch: &mut dyn FnMut() -> Result<(), Box<dyn Error>>,
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        for _ in 0..warmup {
            launch()?;
        }
        unsafe { self.stream.synchronize() }?;
        let start = self.device.new_event()?;
        let end = self.device.new_event()?;
        let mut samples = Vec::with_capacity(iterations);
        let mut capture = Capture::new()?;
        if capture_requested {
            capture.start()?;
        }
        for _ in 0..iterations {
            start.record(&self.stream)?;
            launch()?;
            end.record(&self.stream)?;
            end.synchronize()?;
            samples.push(start.elapsed_time(&end)? as f64 * 1000.0);
        }
        if capture_requested {
            capture.stop()?;
        }
        Ok(samples)
    }
}

/// Retain the output and the event samples the way the oxide binary does, so
/// `experiment.py`, `comparison.py` and `profile_suite.py` read either one.
fn write_run(
    dir: &Path,
    manifest: &Manifest,
    capture_requested: bool,
    samples: &[f64],
    result: &[f32],
    detail: serde_json::Value,
) -> Result<(), Box<dyn Error>> {
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
    let mut timing = serde_json::json!({
        "op": manifest.op, "dims": manifest.dims, "warmup": manifest.warmup,
        "iterations": manifest.iterations, "capture": capture_requested,
        "samples_us": samples,
        "event_mean_us": samples.iter().sum::<f64>() / samples.len() as f64,
        "precision": "FP32; every output checked against the PyTorch reference",
        "run_id": run_id,
    });
    if let (Some(target), Some(extra)) = (timing.as_object_mut(), detail.as_object()) {
        for (key, value) in extra {
            target.insert(key.clone(), value.clone());
        }
    }
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

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory = args
        .next()
        .ok_or("usage: mage-cutile INPUT_DIR [--iterations N] [--capture]")?;
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
    let expected_dims = match manifest.op.as_str() {
        "matmul" | "neighbor" => 3,
        "gelu" | "layernorm" | "triangle" => 2,
        _ => return Err("unknown operation".into()),
    };
    if manifest.dims.len() != expected_dims || manifest.dims.iter().any(|&d| d > i32::MAX as usize) {
        return Err("invalid dimensions".into());
    }
    match manifest.op.as_str() {
        "matmul" => run_matmul(dir, &manifest, capture_requested),
        "gelu" => run_bias_gelu(dir, &manifest, capture_requested),
        "layernorm" => run_layer_norm(dir, &manifest, capture_requested),
        "triangle" => run_triangle(dir, &manifest, capture_requested),
        other => Err(format!(
            "operation {other} is not implemented in the cuTile track yet; \
             the cuda-oxide binary in examples/oxide covers all five operations"
        )
        .into()),
    }
}

fn run_matmul(dir: &Path, manifest: &Manifest, capture_requested: bool) -> Result<(), Box<dyn Error>> {
    let (m, n, k) = (manifest.dims[0], manifest.dims[1], manifest.dims[2]);
    let a = read_f32(&dir.join("a.bin"), product(&[m, k])?)?;
    let b = read_f32(&dir.join("b.bin"), product(&[k, n])?)?;
    let (m_pad, n_pad, k_pad) = (
        m.div_ceil(MATMUL_M) * MATMUL_M,
        n.div_ceil(MATMUL_N) * MATMUL_N,
        k.div_ceil(MATMUL_K) * MATMUL_K,
    );
    let a_pad = pad_rows(&a, m, k, m_pad, k_pad);
    let b_pad = pad_rows(&b, k, n, k_pad, n_pad);

    let harness = Harness::new()?;
    let stream = &harness.stream;
    let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(a_pad))
        .reshape(&[m_pad, k_pad])
        .sync_on(stream)?
        .into();
    let y: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(b_pad))
        .reshape(&[k_pad, n_pad])
        .sync_on(stream)?
        .into();
    let mut z: Tensor<f32> = api::zeros::<f32>(&[m_pad, n_pad]).sync_on(stream)?;
    let generics = vec![
        MATMUL_M.to_string(),
        MATMUL_N.to_string(),
        MATMUL_K.to_string(),
        k_pad.to_string(),
    ];
    let samples = harness.time(
        manifest.warmup,
        manifest.iterations,
        capture_requested,
        &mut || {
            let _ = kernels::matmul((&mut z).partition([MATMUL_M, MATMUL_N]), &x, &y)
                .generics(generics.clone())
                .sync_on(stream)?;
            Ok(())
        },
    )?;

    let host: Vec<f32> = z.to_host_vec().sync_on(stream)?;
    let mut result = Vec::with_capacity(product(&[m, n])?);
    for row in 0..m {
        result.extend_from_slice(&host[row * n_pad..row * n_pad + n]);
    }
    write_run(
        dir,
        manifest,
        capture_requested,
        &samples,
        &result,
        serde_json::json!({"tiles": {"m": MATMUL_M, "n": MATMUL_N, "k": MATMUL_K},
                           "padded": [m_pad, n_pad, k_pad]}),
    )
}

/// Bias + GELU over rows: the column tile is fixed and the tensor is padded up
/// to it, so every row uses one specialization.
fn run_bias_gelu(dir: &Path, manifest: &Manifest, capture_requested: bool) -> Result<(), Box<dyn Error>> {
    let (rows, width) = (manifest.dims[0], manifest.dims[1]);
    let a = read_f32(&dir.join("a.bin"), product(&[rows, width])?)?;
    let bias = read_f32(&dir.join("b.bin"), width)?;
    let rows_pad = rows.div_ceil(GELU_ROWS) * GELU_ROWS;
    let width_pad = width.div_ceil(GELU_COLS) * GELU_COLS;
    let a_pad = pad_rows(&a, rows, width, rows_pad, width_pad);
    let bias_pad = pad_vector(&bias, width_pad);

    let harness = Harness::new()?;
    let stream = &harness.stream;
    let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(a_pad))
        .reshape(&[rows_pad, width_pad])
        .sync_on(stream)?
        .into();
    let bias_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(bias_pad))
        .sync_on(stream)?
        .into();
    let mut y: Tensor<f32> = api::zeros::<f32>(&[rows_pad, width_pad]).sync_on(stream)?;
    let generics = vec![GELU_ROWS.to_string(), GELU_COLS.to_string()];
    let samples = harness.time(
        manifest.warmup,
        manifest.iterations,
        capture_requested,
        &mut || {
            let _ = kernels::bias_gelu(
                (&mut y).partition([GELU_ROWS, GELU_COLS]),
                &x,
                &bias_dev,
                0.044715f32,
                0.7978845608f32,
                0.5f32,
                1.0f32,
            )
            .generics(generics.clone())
            .sync_on(stream)?;
            Ok(())
        },
    )?;

    let host: Vec<f32> = y.to_host_vec().sync_on(stream)?;
    let mut result = Vec::with_capacity(product(&[rows, width])?);
    for row in 0..rows {
        result.extend_from_slice(&host[row * width_pad..row * width_pad + width]);
    }
    write_run(
        dir,
        manifest,
        capture_requested,
        &samples,
        &result,
        serde_json::json!({"tiles": {"rows": GELU_ROWS, "cols": GELU_COLS},
                           "padded": [rows_pad, width_pad]}),
    )
}

/// Layer normalization, one row per program. The row must be one tile and tile
/// dimensions must be powers of two, so the row is padded to the next power of
/// two and the kernel divides by the true width.
fn run_layer_norm(dir: &Path, manifest: &Manifest, capture_requested: bool) -> Result<(), Box<dyn Error>> {
    let (rows, width) = (manifest.dims[0], manifest.dims[1]);
    let a = read_f32(&dir.join("a.bin"), product(&[rows, width])?)?;
    let gamma = read_f32(&dir.join("b.bin"), width)?;
    let beta = read_f32(&dir.join("c.bin"), width)?;
    let width_pad = width.next_power_of_two();
    let a_pad = pad_rows(&a, rows, width, rows, width_pad);
    let gamma_pad = pad_vector(&gamma, width_pad);
    let beta_pad = pad_vector(&beta, width_pad);

    let harness = Harness::new()?;
    let stream = &harness.stream;
    let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(a_pad))
        .reshape(&[rows, width_pad])
        .sync_on(stream)?
        .into();
    let gamma_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(gamma_pad))
        .sync_on(stream)?
        .into();
    let beta_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(beta_pad))
        .sync_on(stream)?
        .into();
    let mut y: Tensor<f32> = api::zeros::<f32>(&[rows, width_pad]).sync_on(stream)?;
    let generics = vec![width_pad.to_string()];
    let samples = harness.time(
        manifest.warmup,
        manifest.iterations,
        capture_requested,
        &mut || {
            let _ = kernels::layer_norm(
                (&mut y).partition([1, width_pad]),
                &x,
                &gamma_dev,
                &beta_dev,
                width as f32,
                1e-5f32,
            )
            .generics(generics.clone())
            .sync_on(stream)?;
            Ok(())
        },
    )?;

    let host: Vec<f32> = y.to_host_vec().sync_on(stream)?;
    let mut result = Vec::with_capacity(product(&[rows, width])?);
    for row in 0..rows {
        result.extend_from_slice(&host[row * width_pad..row * width_pad + width]);
    }
    write_run(
        dir,
        manifest,
        capture_requested,
        &samples,
        &result,
        serde_json::json!({"tiles": {"rows": 1, "cols": width_pad},
                           "padded": [rows, width_pad]}),
    )
}

fn run_triangle(dir: &Path, manifest: &Manifest, capture_requested: bool) -> Result<(), Box<dyn Error>> {
    let (n, channels) = (manifest.dims[0], manifest.dims[1]);
    let a = read_f32(&dir.join("a.bin"), product(&[n, n, channels])?)?;
    let b = read_f32(&dir.join("b.bin"), product(&[n, n, channels])?)?;
    let tile = n.next_power_of_two().min(TRIANGLE_TILE);
    let n_pad = n.div_ceil(tile) * tile;
    let a_pad = pad_square(&a, n, channels, n_pad);
    let b_pad = pad_square(&b, n, channels, n_pad);

    let harness = Harness::new()?;
    let stream = &harness.stream;
    let a_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(a_pad))
        .reshape(&[n_pad, n_pad, channels])
        .sync_on(stream)?
        .into();
    let b_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(b_pad))
        .reshape(&[n_pad, n_pad, channels])
        .sync_on(stream)?
        .into();
    let mut y: Tensor<f32> = api::zeros::<f32>(&[n_pad, n_pad, channels]).sync_on(stream)?;
    let generics = vec![tile.to_string(), tile.to_string(), n_pad.to_string()];
    let samples = harness.time(
        manifest.warmup,
        manifest.iterations,
        capture_requested,
        &mut || {
            let _ = kernels::triangle((&mut y).partition([tile, tile, 1]), &a_dev, &b_dev)
                .generics(generics.clone())
                .sync_on(stream)?;
            Ok(())
        },
    )?;

    let host: Vec<f32> = y.to_host_vec().sync_on(stream)?;
    let mut result = Vec::with_capacity(product(&[n, n, channels])?);
    for i in 0..n {
        for j in 0..n {
            let source = (i * n_pad + j) * channels;
            result.extend_from_slice(&host[source..source + channels]);
        }
    }
    write_run(
        dir,
        manifest,
        capture_requested,
        &samples,
        &result,
        serde_json::json!({"tiles": {"i": tile, "j": tile, "channels": channels},
                           "padded": [n_pad, n_pad, channels]}),
    )
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mage-cutile: {error}");
        std::process::exit(1);
    }
}
