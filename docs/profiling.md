# Profile Python and Rust

| Path | Target | What it records |
| --- | --- | --- |
| Triton events | Python calling Triton JIT kernels | In-process CUDA-event timings; no hardware counters |
| Nsight Systems | Python or a native executable | Individual CUDA kernel launches from the SQLite trace |
| Nsight Compute | Python or a native executable | Instrumented kernel metrics, subject to counter permissions and available metrics |

## Existing Python workflow

```bash
mage profile script.py
mage profile --backend nsys --output-dir artifacts/python-trace script.py
mage profile --backend ncu --output-dir artifacts/python-counters script.py
```

Use the active virtual environment. Nsight invokes that environment's Python interpreter. Triton profiling remains in process and keeps the existing context-manager API.

## Python function workflow

A function can be profiled without a wrapper script:

```bash
mage profile -m package.module:function
mage profile -m package.module:function --call-args '[1, 2]' --call-kwargs '{"scale": 4}'
mage profile -m package.module:function --warmup 5 --iterations 20
mage profile --backend nsys --capture-range cuda \
  --output-dir artifacts/function-trace -m package.module:function
```

The target is imported as `module:function` from the directory where `mage` is
invoked; dotted attributes such as `module:Class.method` are accepted. Positional
and keyword arguments are JSON literals, so they carry scalars, strings, lists,
and objects, not tensors. A function that needs tensors either builds them from
its JSON arguments or takes no arguments and constructs its own inputs.

`--warmup` calls run before `--iterations` measured calls. With the Triton
backend the warmup launches are discarded from the reported metrics, so compile
and first-touch costs stay out of the table; that backend records `@triton.jit`
launches only, and it reports an error instead of an empty session when a target
launches none. With `--capture-range cuda`, the generated driver calls the CUDA
profiler start and stop APIs around the measured calls, so a native capture
contains only the measured region. The driver is a temporary file and is deleted
after the run.

## Native executable workflow

Build the Rust binary and generate input directories using the [setup guide](https://github.com/superposition/mage/blob/master/docs/guide.md), then:

```bash
mage profile-exec --backend nsys --capture-range cuda \
  --output-dir artifacts/rust-trace -- \
  examples/oxide/target/release/mage-oxide artifacts/mage-001/matmul \
  --iterations 100 --capture

mage profile-exec --backend ncu --capture-range cuda --launch-count 1 \
  --output-dir artifacts/rust-counters -- \
  examples/oxide/target/release/mage-oxide artifacts/mage-001/matmul \
  --iterations 1 --capture
```

Arguments after `--` are passed as an argument array, without a shell. A native executable needs Nsight; the Triton backend observes Python JIT calls.

For the equivalent PyTorch reference:

```bash
mage profile --backend nsys --capture-range cuda \
  --output-dir artifacts/python-trace \
  examples/oxide/python_target.py -- artifacts/mage-001/matmul --iterations 100

mage profile --backend ncu --capture-range cuda --launch-count 1 \
  --output-dir artifacts/python-counters \
  examples/oxide/python_target.py -- artifacts/mage-001/matmul --iterations 1
```

The target loads data, initializes CUDA, and warms up before calling the CUDA profiler start API. It synchronizes before stopping capture. Python operations may launch several kernels; an NCU limit of one records only the first. Inspect the Systems trace before deciding which launches to collect.

## Retained evidence

To capture all five operations and both languages sequentially:

```bash
python examples/oxide/profile_suite.py --backend nsys \
  --inputs artifacts/mage-001 --output artifacts/mage-001-nsys
python examples/oxide/profile_suite.py --backend ncu \
  --inputs artifacts/mage-001 --output artifacts/mage-001-ncu
```

The Compute suite captures the first kernel per operation. Python operations
that launch several kernels need a follow-up capture to examine the rest.
Run GPU experiments serially; concurrent profilers can interfere with collection.

Each invocation with `--output-dir` creates a unique subdirectory:

- `capture.nsys-rep` and `capture.sqlite`, or `capture.ncu-rep` and `metrics.csv`.
- `process.log` for target and profiler diagnostics.
- `capture.json` for exact arguments, backend, capture range, exit status, and capture outcome.
- `kernels.json` and `kernels.csv` for the normalized per-launch metrics after a successful capture.

SQLite history remains enabled unless `--no-persist` is set. Use `--db PATH` for an experiment-specific database. Schema migration adds missing metric columns and retains old rows.

Every Rust execution also retains its output and event samples in a unique
`rust-runs/<run-id>/` directory beneath its input directory. The files at the
input-directory root are conveniences for the latest run; use the run ID in
the harness result to retrieve the original samples after later profiling.

Systems reads individual CUPTI kernel launches once, resolving string IDs. Compute reads each metric's unit; unavailable counters remain missing. An aggregate grid count is not substituted for a three-dimensional launch shape. Capture errors, interruption, and empty captures return a nonzero status.

## Performance-counter access in WSL

If Compute reports `ERR_NVGPUCTRPERM`, open **Windows NVIDIA Control Panel → Developer → Manage GPU Performance Counters** and enable access for the user running the experiment. If the Developer menu is hidden, enable Developer Settings from the Desktop menu.

This is a host permission setting; installing a Linux display driver in WSL is not the remedy. NVIDIA documents it on the [performance-counter permission page](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

## Reading a number honestly

Event timings cover device work between events with inputs already on the GPU. They exclude compilation, input preparation, and host-to-device copies. Very short kernels can still be influenced by host submission gaps. Systems trace durations and Compute replay durations come from different instrumentation; do not mix them into a single speedup table.

Record precision, shapes, warmup, repetitions, software versions, and GPU state. A utilization percentage is not a FLOP count and cannot by itself establish arithmetic intensity.
