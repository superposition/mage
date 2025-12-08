"""Progressively richer tensor_join syntax exercises."""

import pytest
import torch

from mage.tensor_logic import TensorJoinPlan, tensor_join


SYNTAX_CASES = [
    ("Matrix multiply", "ab,bc->ac", ((3, 5), (5, 2))),
    ("Axis permutation", "abc->cba", ((2, 3, 4),)),
    ("Outer product", "a,b->ab", ((4,), (6,))),
    ("Diagonal extraction", "abb->a", ((3, 3, 3),)),
    ("Mixed contraction", "abcd,be,cf->adf", ((2, 4, 3, 5), (4, 6), (3, 7))),
    ("Batch matmul", "bij,bjk->bik", ((2, 3, 4), (2, 4, 5))),
    ("Broadcast add", "abc,c->abc", ((3, 4, 5), (5,))),
    ("Tri contraction", "abc,ad,be->ce", ((2, 3, 4), (2, 5), (3, 6))),
    ("Two step transpose", "abcd->badc", ((2, 3, 4, 5),)),
    ("Batch outer", "ba,c->bac", ((2, 3), (4,))),
    ("5D contraction", "abcde,cf,bd->aef", ((2, 3, 4, 5, 6), (4, 7), (3, 5))),
    ("Column sum", "ab->a", ((6, 7),)),
    ("Hadamard", "abc,abc->abc", ((2, 3, 4), (2, 3, 4))),
    ("Batched trace", "bii->b", ((5, 4, 4),)),
    ("Symmetric contraction", "abc,abd->cd", ((2, 3, 4), (2, 3, 5))),
    ("Vector-matrix product", "i,ij->j", ((5,), (5, 4))),
    ("Matrix-vector product", "ij,j->i", ((4, 6), (6,))),
    ("Vector hadamard", "i,i->i", ((7,), (7,))),
    ("Trace diagonal", "ij,ji->i", ((3, 4), (4, 3))),
    ("Batch trace contraction", "bij,bji->b", ((2, 3, 4), (2, 4, 3))),
    ("Row scaling", "ab,a->ab", ((5, 6), (5,))),
    ("Column scaling", "ab,b->ab", ((5, 6), (6,))),
    ("Outer rank-3", "i,j,k->ijk", ((3,), (4,), (5,))),
    ("Outer rank-4", "i,j,kl->ijkl", ((2,), (3,), (4, 5))),
    ("Permutation four axes", "abcd->dcba", ((2, 3, 4, 5),)),
    ("Permutation five axes", "abcde->eadcb", ((2, 3, 4, 5, 6),)),
    ("Double shared contraction", "abcd,abce->de", ((2, 3, 4, 5), (2, 3, 4, 6))),
    ("Triple shared contraction", "abc,acd,aef->bdef", ((2, 3, 4), (2, 4, 5), (2, 6, 7))),
    ("Broadcast sum reduce", "abc,abc->a", ((3, 4, 5), (3, 4, 5))),
    ("Symmetric outer", "ab,ac->abc", ((3, 4), (3, 5))),
    ("Batch matvec", "bij,bj->bi", ((2, 3, 4), (2, 4))),
    ("Matmul column broadcast", "bij,jk->bik", ((2, 3, 4), (4, 5))),
    ("Matmul row broadcast", "ij,bjk->bik", ((3, 4), (2, 4, 5))),
    ("Quadratic form", "ij,jk,kl->il", ((3, 3), (3, 4), (4, 3))),
    ("Batch quadratic form", "bij,bjk,bkl->bil", ((2, 3, 3), (2, 3, 4), (2, 4, 3))),
    ("Diagonal extraction 4D", "aabb->ab", ((3, 3, 4, 4),)),
    ("Column gather contraction", "abc,bd->acd", ((3, 4, 5), (4, 6))),
    ("Lower contraction", "abc,bc->a", ((3, 4, 5), (4, 5))),
    ("Raise contraction", "abc,a->bc", ((3, 4, 5), (3,))),
    ("Batched outer expansion", "bi,j->bij", ((2, 3), (4,))),
    ("Batched diagonal sum", "bii,bjj->b", ((2, 5, 5), (2, 5, 5))),
    ("Permutation with projection", "abcd,be->ecad", ((2, 3, 4, 5), (3, 6))),
    ("Reduce keep last", "abcd->d", ((2, 3, 4, 5),)),
    ("Sum first axis", "abcd->bcd", ((2, 3, 4, 5),)),
    ("Sum middle axis", "abcd->acd", ((2, 3, 4, 5),)),
    ("Sum last axis", "abcd->abc", ((2, 3, 4, 5),)),
    ("Skew contraction", "abc,abd,acd->d", ((2, 3, 4), (2, 3, 5), (2, 4, 5))),
    ("Triple broadcast outer", "ab,c,d->abcd", ((3, 4), (5,), (6,))),
    ("Tied-axis contraction", "abca->bc", ((3, 4, 5, 3),)),
    ("Matrix chain", "ab,bc,cd,de->ae", ((3, 4), (4, 5), (5, 6), (6, 2))),
    ("Batch matrix chain", "bai,bij,bjk->bak", ((2, 3, 4), (2, 4, 5), (2, 5, 6))),
    ("Weighted diagonal extraction", "aib,ajb->ij", ((3, 4, 5), (3, 6, 5))),
    ("Cross contraction matrix", "abc,def,ad,be,cf->ad", ((2, 3, 4), (5, 6, 7), (2, 5), (3, 6), (4, 7))),
    ("Batch cross contraction", "abcd,abef,cd,ef->ab", ((2, 3, 4, 5), (2, 3, 6, 7), (4, 5), (6, 7))),
    ("Permutation with broadcast", "abc,bd,ce->ade", ((3, 4, 5), (4, 6), (5, 7))),
    ("Partial transpose contraction", "abcd,acbe->de", ((2, 3, 4, 5), (2, 4, 3, 6))),
    ("Singleton bridge", "ab,cb->ac", ((3, 1), (4, 1))),
    ("Sum last two axes", "abc->a", ((3, 4, 5),)),
    ("Sum first two axes", "abc->c", ((3, 4, 5),)),
    ("Sum middle axis solo", "abc->ac", ((3, 4, 5),)),
    ("Mixed free reorder", "abc,ade->bcde", ((2, 3, 4), (2, 5, 6))),
    ("Batch outer permutation", "bij,bk->bikj", ((2, 3, 4), (2, 5))),
    ("Masked contraction", "abc,abd,b->cd", ((3, 4, 5), (3, 4, 6), (4,))),
    ("Outer reuse sum", "ab,ab->b", ((4, 5), (4, 5))),
    ("Tensor inverse weighting", "abc,ade,bdf->cef", ((3, 4, 5), (3, 6, 7), (4, 6, 8))),
    ("Batch bilinear form", "bij,bjk,blk->bil", ((2, 3, 4), (2, 4, 5), (2, 6, 5))),
    ("Triangular aggregator", "abc,abd,abe->cde", ((3, 4, 5), (3, 4, 6), (3, 4, 7))),
    ("Three-way broadcast", "ab,c,ad->bcd", ((4, 5), (6,), (4, 7))),
    ("Rank-5 permutation", "abcde->deabc", ((2, 3, 4, 5, 6),)),
    ("Batch permuted matmul", "bji,bjk->bik", ((2, 5, 4), (2, 5, 6))),
    ("Double broadcast add", "abc,c,bd->abd", ((3, 4, 5), (5,), (4, 6))),
    ("Batch sum mixing", "bcd,bd->bc", ((2, 3, 5), (2, 5))),
    ("Chain with broadcast", "ab,bc,d->ad", ((3, 4), (4, 5), (6,))),
    ("Four-way contraction", "abc,ade,bf,ce->df", ((2, 3, 4), (2, 5, 8), (3, 7), (4, 8))),
    ("Batch outer matching", "bi,bj->bij", ((2, 3), (2, 4))),
    ("Axis rotation", "abcd->cadb", ((2, 3, 4, 5),)),
    ("Multi reduction", "abcd,ad->bc", ((3, 4, 5, 6), (3, 6))),
    ("Masked bilinear", "ab,cb,dc->ad", ((3, 4), (5, 4), (6, 5))),
    ("Tri reduction", "abc,abc,abc->c", ((3, 4, 5), (3, 4, 5), (3, 4, 5))),
    ("Batch left contraction", "bij,ai->baj", ((2, 3, 4), (5, 3))),
    ("Symmetric broadcast gather", "ab,ac,d->bcd", ((4, 5), (4, 6), (7,))),
    ("Permutation contraction mix", "abcd,bcf->adf", ((2, 3, 4, 5), (3, 4, 6))),
    ("Batch swap contraction", "bij,bki->bk", ((2, 3, 4), (2, 5, 3))),
    ("Rank-6 shuffle", "abcdef->bdfaec", ((2, 3, 4, 5, 6, 7),)),
    ("Broadcasted chain", "ab,cd,ae->bcde", ((3, 4), (5, 6), (3, 7))),
]


@pytest.mark.parametrize("description,equation,shapes", SYNTAX_CASES, ids=[case[0] for case in SYNTAX_CASES])
def test_tensor_join_syntax_examples(device, description, equation, shapes):
    tensors = [torch.randn(shape, device=device) for shape in shapes]

    expected = torch.einsum(equation, *tensors)
    result = tensor_join(equation, *tensors)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-5)


def test_tensor_join_multiple_path_planning(device):
    """Ensure combining sequential joins matches a fused equation."""

    a = torch.randn(3, 4, device=device)
    b = torch.randn(4, 5, device=device)
    c = torch.randn(5, 6, device=device)

    fused = tensor_join("ab,bc,cd->ad", a, b, c)
    sequential = tensor_join("ab,bc->ac", a, b)
    sequential = tensor_join("ac,cd->ad", sequential, c)

    assert torch.allclose(fused, sequential, atol=1e-6, rtol=1e-5)


def test_tensor_join_plan_reuse_round_trip(device):
    equation = "abc,cd->abd"
    shapes = ((3, 4, 5), (5, 6))
    tensors = [torch.randn(shape, device=device) for shape in shapes]

    plan = TensorJoinPlan.from_equation(equation, tensors)
    first = tensor_join(equation, *tensors, plan=plan)

    tensors_reshaped = [t.clone() for t in tensors]
    second = tensor_join(equation, *tensors_reshaped, plan=plan)

    assert torch.allclose(first, second, atol=1e-6, rtol=1e-5)


def test_tensor_join_dtype_override_in_syntax_playground(device):
    a = torch.randn(3, 4, device=device, dtype=torch.float16)
    b = torch.randn(4, 5, device=device, dtype=torch.float16)

    result = tensor_join("ab,bc->ac", a, b, out_dtype=torch.float32)
    expected = torch.einsum("ab,bc->ac", a, b).to(torch.float32)

    assert result.dtype == torch.float32
    assert torch.allclose(result, expected, atol=1e-3, rtol=5e-2)


INVALID_SYNTAX_CASES = [
    ("Missing operand", "ab->ac", ((3, 4),)),
    ("Undefined output label", "ab,bc->ad", ((2, 3), (3, 4))),
    ("Dimension mismatch", "ab,a->b", ((2, 3), (4,))),
    ("Missing output", "abc", ((2, 3, 4),)),
    ("Ellipsis unsupported", "a...,b->ab", ((2, 3), (3, 4))),
    ("Duplicate output label", "ab,bc->cc", ((2, 3), (3, 4))),
    ("Empty output", "ab,ab->", ((3, 4), (3, 4))),
    ("Trace to scalar", "aa->", ((5, 5),)),
    ("Vector inner product", "a,a->", ((9,), (9,))),
]


@pytest.mark.parametrize("description,equation,shapes", INVALID_SYNTAX_CASES, ids=[case[0] for case in INVALID_SYNTAX_CASES])
def test_tensor_join_syntax_invalid_cases(device, description, equation, shapes):
    tensors = [torch.randn(shape, device=device) for shape in shapes]

    with pytest.raises((ValueError, RuntimeError)):
        tensor_join(equation, *tensors)
