"""Nsight target: allocation and warmup precede the CUDA capture range."""
import argparse
from pathlib import Path
import torch
from experiment import reference

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--iterations", type=int, default=100)
args = parser.parse_args()
if args.iterations <= 0:
    parser.error("iterations must be positive")
manifest, fn = reference(args.directory)
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
