"""Simple test script for profiling."""
import torch
from mage import add, relu, fma

device = torch.device("cuda")

# Test add
x = torch.randn(1_000_000, device=device)
y = torch.randn(1_000_000, device=device)

for _ in range(20):
    out = add(x, y)

torch.cuda.synchronize()

# Test relu
for _ in range(20):
    out = relu(x)

torch.cuda.synchronize()

# Test fma
a = torch.randn(1_000_000, device=device)
for _ in range(20):
    out = fma(a, x, y)

torch.cuda.synchronize()
print("Done!")
