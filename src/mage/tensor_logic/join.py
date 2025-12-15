"""Join (generalised tensor contraction) primitives for tensor logic.

The long term goal is to back :func:`tensor_join` with a Triton kernel so
that any tensor logic rule written as an Einstein summation can execute
Without bouncing through PyTorch.  For now we implement a small amount of
shape/semantic validation and lower suitable equations to one or more
matrix multiplications powered by PyTorch.  This lets us lock down tests
and public API surface while we iterate on the Triton kernel in follow-up
changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch


class _ShapeProxy:
    """Lightweight stand-in exposing ``shape``/``ndim`` for cache planning."""

    __slots__ = ("shape",)

    def __init__(self, shape: Sequence[int]):
        self.shape = tuple(int(dim) for dim in shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)


@dataclass(slots=True)
class TensorJoinPlan:
    """Pre-parsed description of a tensor join.

    The plan makes it straightforward to (eventually) share shape
    metadata between forward and backward kernels without re-parsing the
    Einstein equation.  For now it is mostly a validation artefact.
    """

    equation: str
    input_labels: tuple[tuple[str, ...], ...]
    output_labels: tuple[str, ...]
    contracted_labels: tuple[str, ...]
    label_dims: dict[str, int]

    @staticmethod
    def from_equation(equation: str, operands: Sequence[torch.Tensor]) -> "TensorJoinPlan":
        """Parse an einsum-style equation and validate operand shapes."""
        cleaned = equation.replace(" ", "")
        if "..." in cleaned:
            raise ValueError("tensor_join does not yet support ellipsis notation")
        if "->" not in cleaned:
            raise ValueError("equation must contain '->' to specify output indices")

        input_part, output_part = cleaned.split("->", maxsplit=1)
        input_tokens = input_part.split(",")
        if len(input_tokens) != len(operands):
            raise ValueError(
                f"expected {len(input_tokens)} operands for equation '{equation}', got {len(operands)}"
            )
        if not output_part:
            raise ValueError("equation must specify at least one output axis")

        input_labels: list[tuple[str, ...]] = []
        label_dims: dict[str, int] = {}

        for token, tensor in zip(input_tokens, operands):
            labels = tuple(token)
            if len(labels) != tensor.ndim:
                raise ValueError(
                    f"operand shape mismatch for token '{token}': "
                    f"expected {len(labels)} dims, got {tensor.ndim}"
                )
            for axis_char, dim in zip(labels, tensor.shape):
                previous = label_dims.get(axis_char)
                if previous is None:
                    label_dims[axis_char] = dim
                elif previous != dim:
                    raise ValueError(
                        f"dimension mismatch for axis '{axis_char}': "
                        f"{previous} vs {dim}"
                    )
            input_labels.append(labels)

        output_labels = tuple(output_part)
        for label in output_labels:
            if label not in label_dims:
                raise ValueError(f"output label '{label}' not present in any input operand")

        contracted_labels = tuple(sorted(set(label_dims) - set(output_labels)))

        return TensorJoinPlan(
            equation=cleaned,
            input_labels=tuple(input_labels),
            output_labels=output_labels,
            contracted_labels=contracted_labels,
            label_dims=label_dims,
        )


def tensor_join(
    equation: str,
    *operands: torch.Tensor,
    plan: TensorJoinPlan | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute a tensor join described by an einsum-style equation.

    Parameters
    ----------
    equation:
        Einstein summation string.  An explicit output ("->") is required
        so that the free variables are unambiguous.
    *operands:
        Input tensors matching the left-hand side of the equation.
    plan:
        Optional :class:`TensorJoinPlan`.  If provided the function will
        skip re-parsing and reuse metadata.
    out_dtype:
        Optional dtype override for the output.  Useful when the inputs
        are low precision but a higher precision accumulator is desired.

    Returns
    -------
    torch.Tensor
        The contracted tensor.  The default implementation delegates to
        :func:`torch.einsum` while we bootstrap the Triton kernel.
    """

    if not operands:
        raise ValueError("tensor_join requires at least one operand")

    cleaned_equation = equation.replace(" ", "")

    if plan is None:
        plan = _get_or_create_plan(cleaned_equation, operands)
    else:
        if plan.equation != cleaned_equation:
            raise ValueError("plan equation does not match provided equation")
        _validate_against_plan(plan, operands)

    if _can_use_binary_matmul(plan, operands):
        result = _tensor_join_binary_matmul(plan, operands[0], operands[1], out_dtype)
    else:
        result = torch.einsum(plan.equation, *operands)
        if out_dtype is not None and result.dtype != out_dtype:
            result = result.to(out_dtype)
    return result


def tensor_join_cache_clear() -> None:
    """Clear the internal plan cache used by :func:`tensor_join`."""

    _cached_plan.cache_clear()


def tensor_join_cache_info() -> "functools._CacheInfo":
    """Return cache statistics for the internal plan cache."""

    return _cached_plan.cache_info()


def _validate_against_plan(plan: TensorJoinPlan, operands: Sequence[torch.Tensor]) -> None:
    if len(plan.input_labels) != len(operands):
        raise ValueError(
            f"plan expects {len(plan.input_labels)} operands, got {len(operands)}"
        )

    for labels, tensor in zip(plan.input_labels, operands):
        if len(labels) != tensor.ndim:
            raise ValueError(
                f"operand rank mismatch for plan labelled '{''.join(labels)}': {tensor.ndim}"
            )
        for axis_char, dim in zip(labels, tensor.shape):
            expected = plan.label_dims[axis_char]
            if expected != dim:
                raise ValueError(
                    f"dimension mismatch for axis '{axis_char}': {expected} vs {dim}"
                )


def _can_use_binary_matmul(plan: TensorJoinPlan, operands: Sequence[torch.Tensor]) -> bool:
    if len(operands) != 2:
        return False
    lhs, rhs = operands
    if lhs.device != rhs.device:
        return False
    if lhs.dtype != rhs.dtype:
        return False

    lhs_labels = plan.input_labels[0]
    rhs_labels = plan.input_labels[1]

    # All contracted labels must be present in both operands to use matmul reduction.
    contracted = [label for label in lhs_labels if label in plan.contracted_labels]
    if not contracted:
        # No reduction axis: treat as outer product, defer to einsum for now.
        return False
    for label in contracted:
        if label not in rhs_labels:
            return False

    return True


def _tensor_join_binary_matmul(
    plan: TensorJoinPlan,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out_dtype: torch.dtype | None,
) -> torch.Tensor:
    lhs_labels = plan.input_labels[0]
    rhs_labels = plan.input_labels[1]

    contracted = [label for label in lhs_labels if label in plan.contracted_labels]
    # Preserve operand-specific ordering for deterministic reshapes.
    rhs_contracted = [label for label in rhs_labels if label in plan.contracted_labels]
    if set(rhs_contracted) != set(contracted):
        fallback = torch.einsum(plan.equation, lhs, rhs)
        if out_dtype is not None and fallback.dtype != out_dtype:
            fallback = fallback.to(out_dtype)
        return fallback
    output_labels = plan.output_labels
    batch_labels = [label for label in output_labels if label in lhs_labels and label in rhs_labels]
    lhs_output_unique = [label for label in lhs_labels if label in output_labels and label not in batch_labels]
    rhs_output_unique = [label for label in rhs_labels if label in output_labels and label not in batch_labels]

    lhs_perm_indices = [lhs_labels.index(label) for label in batch_labels]
    lhs_perm_indices += [lhs_labels.index(label) for label in lhs_output_unique]
    lhs_perm_indices += [lhs_labels.index(label) for label in contracted]

    rhs_perm_indices = [rhs_labels.index(label) for label in batch_labels]
    rhs_perm_indices += [rhs_labels.index(label) for label in contracted]
    rhs_perm_indices += [rhs_labels.index(label) for label in rhs_output_unique]

    lhs_prepped = lhs.permute(lhs_perm_indices).contiguous()
    rhs_prepped = rhs.permute(rhs_perm_indices).contiguous()

    batch_dims = [plan.label_dims[label] for label in batch_labels]
    lhs_unique_dims = [plan.label_dims[label] for label in lhs_output_unique]
    rhs_unique_dims = [plan.label_dims[label] for label in rhs_output_unique]
    contracted_dims = [plan.label_dims[label] for label in contracted]

    batch_size = math.prod(batch_dims) if batch_dims else 1
    lhs_m = math.prod(lhs_unique_dims) if lhs_unique_dims else 1
    lhs_k = math.prod(contracted_dims)
    rhs_n = math.prod(rhs_unique_dims) if rhs_unique_dims else 1

    lhs_matrix = lhs_prepped.reshape(batch_size, lhs_m, lhs_k)
    rhs_matrix = rhs_prepped.reshape(batch_size, lhs_k, rhs_n)

    target_dtype = out_dtype or lhs.dtype
    compute_dtype = torch.float32

    if not batch_labels:
        lhs_flat = lhs_matrix.reshape(lhs_m, lhs_k).to(compute_dtype)
        rhs_flat = rhs_matrix.reshape(lhs_k, rhs_n).to(compute_dtype)
        matmul_result = torch.matmul(lhs_flat, rhs_flat)
        if target_dtype != compute_dtype:
            matmul_result = matmul_result.to(target_dtype)
        if lhs_unique_dims or rhs_unique_dims:
            reshaped = matmul_result.reshape(*lhs_unique_dims, *rhs_unique_dims)
        else:
            reshaped = matmul_result.reshape(())
        current_order = lhs_output_unique + rhs_output_unique
    else:
        lhs_batched = lhs_matrix.to(compute_dtype)
        rhs_batched = rhs_matrix.to(compute_dtype)
        result_batched = torch.bmm(lhs_batched, rhs_batched)
        if target_dtype != compute_dtype:
            result_batched = result_batched.to(target_dtype)
        reshaped_dims: list[int] = []
        if batch_dims:
            reshaped_dims.extend(batch_dims)
        if lhs_unique_dims:
            reshaped_dims.extend(lhs_unique_dims)
        if rhs_unique_dims:
            reshaped_dims.extend(rhs_unique_dims)
        reshaped = result_batched.reshape(*reshaped_dims) if reshaped_dims else result_batched.reshape(())
        current_order = batch_labels + lhs_output_unique + rhs_output_unique

    if list(current_order) == list(output_labels) or not current_order:
        result = reshaped
    else:
        perm = [current_order.index(label) for label in output_labels]
        result = reshaped.permute(perm).contiguous()

    if out_dtype is not None and result.dtype != out_dtype:
        result = result.to(out_dtype)
    return result


def _get_or_create_plan(equation: str, operands: Sequence[torch.Tensor]) -> TensorJoinPlan:
    shapes = tuple(tuple(int(dim) for dim in tensor.shape) for tensor in operands)
    plan = _cached_plan(equation, shapes)
    _validate_against_plan(plan, operands)
    return plan


@lru_cache(maxsize=512)
def _cached_plan(equation: str, shapes: tuple[tuple[int, ...], ...]) -> TensorJoinPlan:
    proxies = [_ShapeProxy(shape) for shape in shapes]
    return TensorJoinPlan.from_equation(equation, proxies)  # type: ignore[arg-type]
