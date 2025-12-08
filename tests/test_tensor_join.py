import pytest
import torch

from mage.tensor_logic import TensorJoinPlan, tensor_join


@pytest.mark.parametrize(
    "equation,shapes",
    [
        ("abc,cd->abd", ((4, 8, 16), (16, 5))),
        ("bij,bjk->bik", ((2, 7, 5), (2, 5, 3))),
        ("ik,kj->ij", ((9, 11), (11, 13))),
        ("abcd,be->aecd", ((3, 5, 7, 2), (5, 4))),
        ("ab,bc,cd->ad", ((6, 3), (3, 8), (8, 2))),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tensor_join_matches_einsum(device, equation, shapes, dtype):
    tensors = [torch.randn(shape, device=device, dtype=dtype) for shape in shapes]
    plan = TensorJoinPlan.from_equation(equation, tensors)

    expected = torch.einsum(equation, *tensors)
    result = tensor_join(equation, *tensors, plan=plan)

    if dtype in {torch.float16, torch.bfloat16}:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 1e-5, 1e-4

    assert result.dtype == expected.dtype
    assert torch.allclose(result, expected, atol=atol, rtol=rtol)


def test_tensor_join_plan_reuse_requires_matching_shapes(device):
    a = torch.randn(3, 4, device=device)
    b = torch.randn(4, 5, device=device)
    plan = TensorJoinPlan.from_equation("ab,bc->ac", (a, b))

    # Alter shape so validation should fail.
    b_bad = torch.randn(3, 5, device=device)

    with pytest.raises(ValueError, match="dimension mismatch"):
        tensor_join("ab,bc->ac", a, b_bad, plan=plan)


def test_tensor_join_dtype_override(device):
    a = torch.randn(5, 4, device=device, dtype=torch.float16)
    b = torch.randn(4, 6, device=device, dtype=torch.float16)

    result = tensor_join("ab,bc->ac", a, b, out_dtype=torch.float32)
    assert result.dtype == torch.float32

    expected = torch.einsum("ab,bc->ac", a, b).to(torch.float32)
    assert torch.allclose(result, expected, atol=1e-3, rtol=5e-2)


@pytest.mark.parametrize(
    "bad_equation",
    ["abc,cd", "abc->", "a...,b->ab"],
)
def test_tensor_join_rejects_invalid_equations(bad_equation, device):
    a = torch.randn(2, 3, device=device)
    b = torch.randn(3, 4, device=device)

    with pytest.raises(ValueError):
        tensor_join(bad_equation, a, b)


def test_tensor_join_requires_operands(device):
    with pytest.raises(ValueError, match="requires at least one operand"):
        tensor_join("ab->ab")


def test_tensor_join_disallows_ellipsis(device):
    a = torch.randn(2, 3, device=device)
    b = torch.randn(3, 4, device=device)

    with pytest.raises(ValueError, match="does not yet support ellipsis"):
        TensorJoinPlan.from_equation("a...,b->ab", (a, b))


def test_tensor_join_plan_equation_mismatch(device):
    a = torch.randn(3, 4, device=device)
    b = torch.randn(4, 5, device=device)
    plan = TensorJoinPlan.from_equation("ab,bc->ac", (a, b))

    with pytest.raises(ValueError, match="plan equation does not match"):
        tensor_join("ab,bc->ab", a, b, plan=plan)


def test_tensor_join_gradients_match_einsum(device):
    a = torch.randn(4, 6, device=device, requires_grad=True)
    b = torch.randn(6, 3, device=device, requires_grad=True)

    out_join = tensor_join("ik,kj->ij", a, b)
    loss_join = out_join.pow(2).sum()
    loss_join.backward()
    grad_a_join = a.grad.detach().clone()
    grad_b_join = b.grad.detach().clone()

    a.grad.zero_()
    b.grad.zero_()

    out_einsum = torch.einsum("ik,kj->ij", a, b)
    loss_einsum = out_einsum.pow(2).sum()
    loss_einsum.backward()

    assert torch.allclose(a.grad, grad_a_join, atol=1e-5, rtol=1e-4)
    assert torch.allclose(b.grad, grad_b_join, atol=1e-5, rtol=1e-4)


def test_tensor_join_outer_product_matches_einsum(device):
    a = torch.randn(2, 3, device=device)
    b = torch.randn(4, 5, device=device)

    expected = torch.einsum("ab,cd->abcd", a, b)
    result = tensor_join("ab,cd->abcd", a, b)

    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-5)
