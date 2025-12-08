"""Inductive logic programming style tensor_join checks."""

import torch

from mage.tensor_logic import TensorJoinPlan, tensor_join


def _relation_tensor(pairs: list[tuple[int, int]], size: int, device: torch.device) -> torch.Tensor:
    relation = torch.zeros(size, size, device=device)
    for src, dst in pairs:
        relation[src, dst] = 1.0
    return relation


def test_tensor_join_grandparent_relation_matches_matmul(device: torch.device) -> None:
    size = 6
    parent_pairs = [(0, 2), (0, 3), (1, 4), (2, 5), (3, 5)]
    parent = _relation_tensor(parent_pairs, size, device)

    grandparent = tensor_join("ij,jk->ik", parent, parent)
    expected = torch.matmul(parent, parent)

    assert torch.allclose(grandparent, expected, atol=1e-6, rtol=0.0)


def test_tensor_join_three_step_ancestor_chain(device: torch.device) -> None:
    size = 5
    parent_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    parent = _relation_tensor(parent_pairs, size, device)

    plan = TensorJoinPlan.from_equation("ij,jk,kl->il", (parent, parent, parent))
    three_step = tensor_join("ij,jk,kl->il", parent, parent, parent, plan=plan)
    expected = torch.einsum("ij,jk,kl->il", parent, parent, parent)

    assert torch.allclose(three_step, expected, atol=1e-6, rtol=0.0)


def test_tensor_join_sibling_parent_rule(device: torch.device) -> None:
    size = 6
    parent_pairs = [(0, 2), (1, 3), (2, 4), (2, 5)]
    sibling_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
    parent = _relation_tensor(parent_pairs, size, device)
    sibling = _relation_tensor(sibling_pairs, size, device)

    child_to_parent = parent.transpose(0, 1)
    aunt_matrix = tensor_join("ij,jk->ik", child_to_parent, sibling)
    expected = torch.matmul(child_to_parent, sibling)

    assert torch.allclose(aunt_matrix, expected, atol=1e-6, rtol=0.0)


def test_tensor_join_descendant_support_counts(device: torch.device) -> None:
    size = 6
    parent_pairs = [(0, 2), (0, 3), (2, 4), (3, 5)]
    parent = _relation_tensor(parent_pairs, size, device)

    grandparent = tensor_join("ij,jk->ik", parent, parent)
    support = tensor_join("ij->i", grandparent)
    expected = grandparent.sum(dim=1)

    assert support.shape == (size,)
    assert torch.allclose(support, expected, atol=1e-6, rtol=0.0)


def test_tensor_join_mlp_with_padded_layers(device: torch.device) -> None:
    torch.manual_seed(0)
    layer_sizes = [5, 3, 4, 2]
    max_units = max(layer_sizes)
    num_layers = len(layer_sizes)

    weights = torch.zeros(num_layers, max_units, max_units, device=device)
    for layer in range(1, num_layers):
        rows = layer_sizes[layer]
        cols = layer_sizes[layer - 1]
        weights[layer, :rows, :cols] = torch.randn(rows, cols, device=device)

    activations = torch.zeros(num_layers, max_units, device=device)
    activations[0, : layer_sizes[0]] = torch.randn(layer_sizes[0], device=device)

    for layer in range(1, num_layers):
        rows = layer_sizes[layer]
        cols = layer_sizes[layer - 1]
        w_slice = weights[layer, :rows, :cols]
        prev = activations[layer - 1, :cols]
        logits = tensor_join("jk,k->j", w_slice, prev)
        activations[layer, :rows] = torch.sigmoid(logits)

    expected = torch.zeros_like(activations)
    expected[0, : layer_sizes[0]] = activations[0, : layer_sizes[0]]
    for layer in range(1, num_layers):
        rows = layer_sizes[layer]
        cols = layer_sizes[layer - 1]
        w_slice = weights[layer, :rows, :cols]
        prev = expected[layer - 1, :cols]
        expected[layer, :rows] = torch.sigmoid(torch.matmul(w_slice, prev))

    assert torch.allclose(activations, expected, atol=1e-6, rtol=1e-5)


def test_tensor_join_rnn_unroll_matches_reference(device: torch.device) -> None:
    torch.manual_seed(1)
    hidden = 6
    inputs = 4
    steps = 5

    w = torch.randn(hidden, hidden, device=device)
    v = torch.randn(hidden, inputs, device=device)
    state = torch.randn(hidden, device=device)
    seq = torch.randn(inputs, steps, device=device)

    plan_state = TensorJoinPlan.from_equation("ij,j->i", (w, state))
    plan_input = TensorJoinPlan.from_equation("ij,j->i", (v, seq[:, 0]))

    join_states = []
    h = state.clone()
    for t in range(steps):
        recurrent = tensor_join("ij,j->i", w, h, plan=plan_state)
        driven = tensor_join("ij,j->i", v, seq[:, t], plan=plan_input)
        h = torch.sigmoid(recurrent + driven)
        join_states.append(h.clone())

    ref_states = []
    h_ref = state.clone()
    for t in range(steps):
        recurrent = torch.matmul(w, h_ref)
        driven = torch.matmul(v, seq[:, t])
        h_ref = torch.sigmoid(recurrent + driven)
        ref_states.append(h_ref.clone())

    stacked_join = torch.stack(join_states, dim=0)
    stacked_ref = torch.stack(ref_states, dim=0)

    assert torch.allclose(stacked_join, stacked_ref, atol=1e-6, rtol=1e-5)
