"""FP32 Triton counterparts for the five learning kernels; fixed, untuned tiles.

Keep this separate from Mage's general operators: the experiment requires IEEE
FP32 dot products (no TF32), contiguous inputs, and forward-only operations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def matrix_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  CHANNELS: tl.constexpr, TRIANGLE: tl.constexpr,
                  TILE: tl.constexpr):
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    channel = tl.program_id(2)
    kk = tl.arange(0, TILE)
    acc = tl.zeros((TILE, TILE), tl.float32)
    for base in range(tl.cdiv(K, TILE)):
        k = base * TILE + kk
        if TRIANGLE:
            ai = (rows[:, None] * K + k[None, :]) * CHANNELS + channel
            bi = (cols[None, :] * K + k[:, None]) * CHANNELS + channel
        else:
            ai = rows[:, None] * K + k[None, :]
            bi = k[:, None] * N + cols[None, :]
        a = tl.load(A + ai, (rows[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + bi, (k[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    out = (rows[:, None] * N + cols[None, :]) * CHANNELS + channel
    tl.store(C + out, acc, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def gelu_kernel(A, B, C, TOTAL: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(A + i, i < TOTAL, other=0) + tl.load(B + i % WIDTH)
    y = 0.5 * z * (1.0 + libdevice.tanh(0.7978845608028654 * (z + 0.044715 * z * z * z)))
    tl.store(C + i, y, i < TOTAL)


@triton.jit
def norm_kernel(A, SCALE, BIAS, C, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(A + row * WIDTH + col, col < WIDTH, other=0)
    mean = tl.sum(x, 0) / WIDTH
    centered = tl.where(col < WIDTH, x - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / WIDTH
    scale = tl.load(SCALE + col, col < WIDTH, other=0)
    bias = tl.load(BIAS + col, col < WIDTH, other=0)
    y = centered * tl.rsqrt(variance + 1e-5) * scale + bias
    tl.store(C + row * WIDTH + col, y, col < WIDTH)


@triton.jit
def neighbor_kernel(A, WEIGHT, PTR, INDEX, C, WIDTH: tl.constexpr,
                    FEATURES: tl.constexpr, EDGES: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * FEATURES + tl.arange(0, FEATURES)
    start = tl.load(PTR + row)
    end = tl.load(PTR + row + 1)
    lane = tl.arange(0, EDGES)
    acc = tl.zeros((EDGES, FEATURES), tl.float32)
    for base in range(start, end, EDGES):
        edge = base + lane
        src = tl.load(INDEX + edge, edge < end, other=0)
        weight = tl.load(WEIGHT + edge, edge < end, other=0)
        x = tl.load(A + src[:, None] * WIDTH + col[None, :],
                    (edge[:, None] < end) & (col[None, :] < WIDTH), other=0)
        acc += weight[:, None] * x
    tl.store(C + row * WIDTH + col, tl.sum(acc, 0), col < WIDTH)


def implementation(directory):
    """Load the shared bytes once and return a launch callable with resident output."""
    manifest = json.loads((directory / "input.json").read_text())
    op, dims = manifest["op"], manifest["dims"]
    def read(name, shape, dtype="<f4"):
        return torch.from_numpy(np.fromfile(directory / name, dtype=dtype).reshape(shape)).cuda()
    if op == "matmul":
        m, n, k = dims
        a, b = read("a.bin", (m, k)), read("b.bin", (k, n))
        out = torch.empty((m, n), device="cuda")
        def launch():
            matrix_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32), 1)](
                a, b, out, m, n, k, 1, False, 32, num_warps=4)
    elif op == "triangle":
        n, c = dims
        a, b = read("a.bin", (n, n, c)), read("b.bin", (n, n, c))
        out = torch.empty_like(a)
        def launch():
            matrix_kernel[(triton.cdiv(n, 32), triton.cdiv(n, 32), c)](
                a, b, out, n, n, n, c, True, 32, num_warps=4)
    elif op in {"gelu", "layernorm"}:
        rows, width = dims
        a, b = read("a.bin", (rows, width)), read("b.bin", (width,))
        out = torch.empty_like(a)
        if op == "gelu":
            def launch():
                gelu_kernel[(triton.cdiv(rows * width, 256),)](
                    a, b, out, rows * width, width, 256)
        else:
            c = read("c.bin", (width,))
            def launch():
                norm_kernel[(rows,)](a, b, c, out, width, triton.next_power_of_2(width))
    elif op == "neighbor":
        n, width, edges = dims
        a, weight = read("a.bin", (n, width)), read("b.bin", (edges,))
        ptr = read("rowptr.bin", (n + 1,), "<u4")
        index = read("indices.bin", (edges,), "<u4")
        out = torch.empty_like(a)
        def launch():
            neighbor_kernel[(n, triton.cdiv(width, 64))](a, weight, ptr, index, out, width, 64, 32)
    else:
        raise ValueError(f"Unknown operation: {op}")
    def run():
        launch()
        return out
    return manifest, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    manifest, fn = implementation(args.directory)
    for _ in range(manifest["warmup"]):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        for _ in range(args.iterations):
            fn()
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
