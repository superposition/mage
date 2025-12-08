import math

import pytest
import torch
import torch.nn.functional as F

from mage.tensor_logic import TensorJoinPlan, tensor_join


def _one_hot_tokens(seq_len: int, vocab: int, device: torch.device) -> torch.Tensor:
    indices = torch.randint(0, vocab, (seq_len,), device=device)
    tokens = torch.zeros(seq_len, vocab, device=device)
    tokens.scatter_(1, indices.unsqueeze(-1), 1.0)
    return tokens


def _layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(var + eps)
    return normalized * weight + bias


def _clone_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    clone = tensor.detach().clone()
    clone.requires_grad_(True)
    return clone


def _build_reasoning_vocab(device: torch.device) -> tuple[list[str], torch.Tensor]:
    shapes = ["box", "circle", "triangle", "hexagon"]
    colors = ["blue", "red", "green", "yellow"]
    sizes = ["small", "large"]
    materials = ["metal", "wood"]
    patterns = ["striped", "dotted"]

    features = (
        [f"shape:{shape}" for shape in shapes]
        + [f"color:{color}" for color in colors]
        + [f"size:{size}" for size in sizes]
        + [f"material:{material}" for material in materials]
        + [f"pattern:{pattern}" for pattern in patterns]
    )
    feature_index = {feature: idx for idx, feature in enumerate(features)}

    def vector(selected: list[str]) -> torch.Tensor:
        vec = torch.zeros(len(features), device=device)
        for feature in selected:
            vec[feature_index[feature]] = 1.0
        return vec

    tokens: list[tuple[str, torch.Tensor]] = []

    def add_token(name: str, selected: list[str]) -> None:
        tokens.append((name, vector(selected)))

    for shape in shapes:
        add_token(shape, [f"shape:{shape}"])
    for shape in shapes:
        for color in colors:
            add_token(f"{color}_{shape}", [f"shape:{shape}", f"color:{color}"])
    for shape in shapes:
        for size in sizes:
            add_token(f"{size}_{shape}", [f"shape:{shape}", f"size:{size}"])
    for shape in shapes:
        for material in materials:
            add_token(f"{material}_{shape}", [f"shape:{shape}", f"material:{material}"])
    for shape in shapes:
        for pattern in patterns:
            add_token(f"{pattern}_{shape}", [f"shape:{shape}", f"pattern:{pattern}"])

    for name, selected in [
        ("blue_small_box", ["shape:box", "color:blue", "size:small"]),
        ("blue_small_circle", ["shape:circle", "color:blue", "size:small"]),
        ("red_large_box", ["shape:box", "color:red", "size:large"]),
        ("red_large_triangle", ["shape:triangle", "color:red", "size:large"]),
        ("green_small_circle", ["shape:circle", "color:green", "size:small"]),
        ("green_small_hexagon", ["shape:hexagon", "color:green", "size:small"]),
        ("yellow_large_circle", ["shape:circle", "color:yellow", "size:large"]),
        ("metal_blue_box", ["shape:box", "material:metal", "color:blue"]),
        ("metal_blue_circle", ["shape:circle", "material:metal", "color:blue"]),
        ("wood_red_triangle", ["shape:triangle", "material:wood", "color:red"]),
        ("wood_red_hexagon", ["shape:hexagon", "material:wood", "color:red"]),
        ("striped_blue_box", ["shape:box", "pattern:striped", "color:blue"]),
        ("striped_blue_circle", ["shape:circle", "pattern:striped", "color:blue"]),
        ("dotted_green_triangle", ["shape:triangle", "pattern:dotted", "color:green"]),
        ("dotted_green_hexagon", ["shape:hexagon", "pattern:dotted", "color:green"]),
    ]:
        add_token(name, selected)

    names = [name for name, _ in tokens]
    embeddings = torch.stack([vec for _, vec in tokens], dim=0)
    identity = torch.eye(len(names), device=device)
    plan = TensorJoinPlan.from_equation("pv,vd->pd", (identity, embeddings))
    joined = tensor_join("pv,vd->pd", identity, embeddings, plan=plan)
    return names, joined


ANALOGY_CASES: list[tuple[str, str, str, str]] = [
    ("blue_box", "box", "circle", "blue_circle"),
    ("red_box", "box", "triangle", "red_triangle"),
    ("green_box", "box", "hexagon", "green_hexagon"),
    ("yellow_circle", "circle", "box", "yellow_box"),
    ("large_box", "box", "circle", "large_circle"),
    ("small_triangle", "triangle", "hexagon", "small_hexagon"),
    ("metal_box", "box", "circle", "metal_circle"),
    ("wood_circle", "circle", "triangle", "wood_triangle"),
    ("blue_small_box", "small_box", "small_circle", "blue_small_circle"),
    ("red_large_box", "large_box", "large_triangle", "red_large_triangle"),
    ("metal_blue_box", "metal_box", "metal_circle", "metal_blue_circle"),
    ("wood_red_triangle", "wood_triangle", "wood_hexagon", "wood_red_hexagon"),
    ("striped_box", "box", "circle", "striped_circle"),
    ("dotted_triangle", "triangle", "hexagon", "dotted_hexagon"),
    ("striped_blue_box", "striped_box", "striped_circle", "striped_blue_circle"),
    ("dotted_green_triangle", "dotted_triangle", "dotted_hexagon", "dotted_green_hexagon"),
    ("blue_circle", "blue_box", "red_box", "red_circle"),
    ("small_circle", "small_box", "large_box", "large_circle"),
    ("metal_circle", "metal_box", "wood_box", "wood_circle"),
    ("triangle", "triangle", "hexagon", "hexagon"),
]


def test_tensor_join_token_embedding_matches_matmul(device: torch.device) -> None:
    seq_len, vocab, model_dim = 5, 7, 4
    tokens = _one_hot_tokens(seq_len, vocab, device)
    embedding = torch.randn(vocab, model_dim, device=device)

    plan = TensorJoinPlan.from_equation("pv,vd->pd", (tokens, embedding))

    embedded = tensor_join("pv,vd->pd", tokens, embedding, plan=plan)
    expected = torch.matmul(tokens, embedding)

    assert torch.allclose(embedded, expected, atol=1e-6, rtol=1e-6)


def test_tensor_join_attention_projections_match_einsum(device: torch.device) -> None:
    seq_len, model_dim, num_heads, head_dim, value_dim = 4, 6, 3, 2, 3
    stream = torch.randn(seq_len, model_dim, device=device)
    w_q = torch.randn(model_dim, num_heads, head_dim, device=device)
    w_k = torch.randn(model_dim, num_heads, head_dim, device=device)
    w_v = torch.randn(model_dim, num_heads, value_dim, device=device)

    query = tensor_join("pd,dhk->phk", stream, w_q)
    key = tensor_join("pd,dhk->phk", stream, w_k)
    value = tensor_join("pd,dhv->phv", stream, w_v)

    expected_q = torch.einsum("pd,dhk->phk", stream, w_q)
    expected_k = torch.einsum("pd,dhk->phk", stream, w_k)
    expected_v = torch.einsum("pd,dhv->phv", stream, w_v)

    assert torch.allclose(query, expected_q, atol=1e-6, rtol=1e-5)
    assert torch.allclose(key, expected_k, atol=1e-6, rtol=1e-5)
    assert torch.allclose(value, expected_v, atol=1e-6, rtol=1e-5)

    scores = tensor_join("phk,qhk->pqh", query, key)
    expected_scores = torch.einsum("phk,qhk->pqh", expected_q, expected_k)
    assert torch.allclose(scores, expected_scores, atol=1e-6, rtol=1e-5)

    attn_probs = torch.softmax(scores / math.sqrt(head_dim), dim=1)
    expected_probs = torch.softmax(expected_scores / math.sqrt(head_dim), dim=1)
    assert torch.allclose(attn_probs, expected_probs, atol=1e-6, rtol=1e-6)

    attn_output = tensor_join("pqh,qhv->phv", attn_probs, value)
    expected_output = torch.einsum("pqh,qhv->phv", expected_probs, expected_v)
    assert torch.allclose(attn_output, expected_output, atol=1e-6, rtol=1e-5)


def test_tensor_join_batched_attention_matches_einsum(device: torch.device) -> None:
    batch, seq_len = 2, 5
    model_dim, num_heads, head_dim = 8, 2, 4
    value_dim = 4

    stream = torch.randn(batch, seq_len, model_dim, device=device)
    w_q = torch.randn(model_dim, num_heads, head_dim, device=device)
    w_k = torch.randn(model_dim, num_heads, head_dim, device=device)
    w_v = torch.randn(model_dim, num_heads, value_dim, device=device)

    query = tensor_join("bpd,dhk->bphk", stream, w_q)
    key = tensor_join("bpd,dhk->bphk", stream, w_k)
    value = tensor_join("bpd,dhv->bphv", stream, w_v)

    expected_q = torch.einsum("bpd,dhk->bphk", stream, w_q)
    expected_k = torch.einsum("bpd,dhk->bphk", stream, w_k)
    expected_v = torch.einsum("bpd,dhv->bphv", stream, w_v)

    scores = tensor_join("bphk,bqhk->bpqh", query, key)
    expected_scores = torch.einsum("bphk,bqhk->bpqh", expected_q, expected_k)

    weights = torch.softmax(scores / math.sqrt(head_dim), dim=2)
    expected_weights = torch.softmax(expected_scores / math.sqrt(head_dim), dim=2)

    attn = tensor_join("bpqh,bqhv->bphv", weights, value)
    expected_attn = torch.einsum("bpqh,bqhv->bphv", expected_weights, expected_v)

    assert torch.allclose(query, expected_q, atol=1e-6, rtol=1e-5)
    assert torch.allclose(key, expected_k, atol=1e-6, rtol=1e-5)
    assert torch.allclose(value, expected_v, atol=1e-6, rtol=1e-5)
    assert torch.allclose(scores, expected_scores, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weights, expected_weights, atol=1e-6, rtol=1e-5)
    assert torch.allclose(attn, expected_attn, atol=1e-6, rtol=1e-5)


def test_tensor_join_transformer_block_matches_reference(device: torch.device) -> None:
    seq_len, model_dim, num_heads, head_dim, value_dim = 3, 8, 2, 4, 4
    ff_hidden = 10
    vocab = 11

    stream = torch.randn(seq_len, model_dim, device=device)
    w_q = torch.randn(model_dim, num_heads, head_dim, device=device)
    w_k = torch.randn(model_dim, num_heads, head_dim, device=device)
    w_v = torch.randn(model_dim, num_heads, value_dim, device=device)
    w_o = torch.randn(num_heads, value_dim, model_dim, device=device)
    ff_w1 = torch.randn(model_dim, ff_hidden, device=device)
    ff_w2 = torch.randn(ff_hidden, model_dim, device=device)
    token_out = torch.randn(vocab, model_dim, device=device)

    # Tensor-logic path
    q = tensor_join("pd,dhk->phk", stream, w_q)
    k = tensor_join("pd,dhk->phk", stream, w_k)
    v = tensor_join("pd,dhv->phv", stream, w_v)
    scores = tensor_join("phk,qhk->pqh", q, k) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=1)
    attn = tensor_join("pqh,qhv->phv", weights, v)
    projected = tensor_join("phv,hvd->pd", attn, w_o)
    stream_with_attn = stream + projected
    ff_hidden_act = torch.relu(tensor_join("pd,df->pf", stream_with_attn, ff_w1))
    ff_output = tensor_join("pf,fd->pd", ff_hidden_act, ff_w2)
    updated_stream = stream_with_attn + ff_output
    logits = tensor_join("pd,td->pt", updated_stream, token_out)

    # Reference einsum path
    q_ref = torch.einsum("pd,dhk->phk", stream, w_q)
    k_ref = torch.einsum("pd,dhk->phk", stream, w_k)
    v_ref = torch.einsum("pd,dhv->phv", stream, w_v)
    scores_ref = torch.einsum("phk,qhk->pqh", q_ref, k_ref) / math.sqrt(head_dim)
    weights_ref = torch.softmax(scores_ref, dim=1)
    attn_ref = torch.einsum("pqh,qhv->phv", weights_ref, v_ref)
    projected_ref = torch.einsum("phv,hvd->pd", attn_ref, w_o)
    stream_with_attn_ref = stream + projected_ref
    ff_hidden_ref = torch.relu(torch.einsum("pd,df->pf", stream_with_attn_ref, ff_w1))
    ff_output_ref = torch.einsum("pf,fd->pd", ff_hidden_ref, ff_w2)
    updated_stream_ref = stream_with_attn_ref + ff_output_ref
    logits_ref = torch.einsum("pd,td->pt", updated_stream_ref, token_out)

    assert torch.allclose(q, q_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(k, k_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(v, v_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weights, weights_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(attn, attn_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(projected, projected_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ff_hidden_act, ff_hidden_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ff_output, ff_output_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(updated_stream, updated_stream_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(logits, logits_ref, atol=1e-6, rtol=1e-5)


def test_tensor_join_transformer_block_grads_match_einsum(device: torch.device) -> None:
    seq_len, model_dim, num_heads, head_dim, value_dim = 3, 8, 2, 4, 4
    ff_hidden = 10

    base_stream = torch.randn(seq_len, model_dim, device=device)
    base_w_q = torch.randn(model_dim, num_heads, head_dim, device=device)
    base_w_k = torch.randn(model_dim, num_heads, head_dim, device=device)
    base_w_v = torch.randn(model_dim, num_heads, value_dim, device=device)
    base_w_o = torch.randn(num_heads, value_dim, model_dim, device=device)
    base_ff_w1 = torch.randn(model_dim, ff_hidden, device=device)
    base_ff_w2 = torch.randn(ff_hidden, model_dim, device=device)

    # Tensor-logic path with gradients
    stream = _clone_with_grad(base_stream)
    w_q = _clone_with_grad(base_w_q)
    w_k = _clone_with_grad(base_w_k)
    w_v = _clone_with_grad(base_w_v)
    w_o = _clone_with_grad(base_w_o)
    ff_w1 = _clone_with_grad(base_ff_w1)
    ff_w2 = _clone_with_grad(base_ff_w2)

    q = tensor_join("pd,dhk->phk", stream, w_q)
    k = tensor_join("pd,dhk->phk", stream, w_k)
    v = tensor_join("pd,dhv->phv", stream, w_v)
    scores = tensor_join("phk,qhk->pqh", q, k) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=1)
    attn = tensor_join("pqh,qhv->phv", weights, v)
    projected = tensor_join("phv,hvd->pd", attn, w_o)
    stream_with_attn = stream + projected
    ff_hidden_act = torch.relu(tensor_join("pd,df->pf", stream_with_attn, ff_w1))
    ff_output = tensor_join("pf,fd->pd", ff_hidden_act, ff_w2)
    updated_stream = stream_with_attn + ff_output
    loss = updated_stream.pow(2).sum()
    loss.backward()

    grads_join = {
        "stream": stream.grad.detach().clone(),
        "w_q": w_q.grad.detach().clone(),
        "w_k": w_k.grad.detach().clone(),
        "w_v": w_v.grad.detach().clone(),
        "w_o": w_o.grad.detach().clone(),
        "ff_w1": ff_w1.grad.detach().clone(),
        "ff_w2": ff_w2.grad.detach().clone(),
    }

    # Reference einsum path with gradients
    stream_ref = _clone_with_grad(base_stream)
    w_q_ref = _clone_with_grad(base_w_q)
    w_k_ref = _clone_with_grad(base_w_k)
    w_v_ref = _clone_with_grad(base_w_v)
    w_o_ref = _clone_with_grad(base_w_o)
    ff_w1_ref = _clone_with_grad(base_ff_w1)
    ff_w2_ref = _clone_with_grad(base_ff_w2)

    q_ref = torch.einsum("pd,dhk->phk", stream_ref, w_q_ref)
    k_ref = torch.einsum("pd,dhk->phk", stream_ref, w_k_ref)
    v_ref = torch.einsum("pd,dhv->phv", stream_ref, w_v_ref)
    scores_ref = torch.einsum("phk,qhk->pqh", q_ref, k_ref) / math.sqrt(head_dim)
    weights_ref = torch.softmax(scores_ref, dim=1)
    attn_ref = torch.einsum("pqh,qhv->phv", weights_ref, v_ref)
    projected_ref = torch.einsum("phv,hvd->pd", attn_ref, w_o_ref)
    stream_with_attn_ref = stream_ref + projected_ref
    ff_hidden_ref = torch.relu(torch.einsum("pd,df->pf", stream_with_attn_ref, ff_w1_ref))
    ff_output_ref = torch.einsum("pf,fd->pd", ff_hidden_ref, ff_w2_ref)
    updated_stream_ref = stream_with_attn_ref + ff_output_ref
    loss_ref = updated_stream_ref.pow(2).sum()
    loss_ref.backward()

    grads_ref = {
        "stream": stream_ref.grad,
        "w_q": w_q_ref.grad,
        "w_k": w_k_ref.grad,
        "w_v": w_v_ref.grad,
        "w_o": w_o_ref.grad,
        "ff_w1": ff_w1_ref.grad,
        "ff_w2": ff_w2_ref.grad,
    }

    for name in grads_join:
        assert torch.allclose(grads_join[name], grads_ref[name], atol=1e-6, rtol=1e-5), f"gradient mismatch for {name}"


def test_tensor_join_multi_block_stack_matches_reference(device: torch.device) -> None:
    seq_len, model_dim, num_heads, head_dim, value_dim = 4, 6, 2, 3, 3
    ff_hidden = 9

    stream = torch.randn(seq_len, model_dim, device=device)

    params = []
    for _ in range(2):
        params.append(
            (
                torch.randn(model_dim, num_heads, head_dim, device=device),  # w_q
                torch.randn(model_dim, num_heads, head_dim, device=device),  # w_k
                torch.randn(model_dim, num_heads, value_dim, device=device),  # w_v
                torch.randn(num_heads, value_dim, model_dim, device=device),  # w_o
                torch.randn(model_dim, ff_hidden, device=device),  # ff_w1
                torch.randn(ff_hidden, model_dim, device=device),  # ff_w2
            )
        )

    def run_block(inp: torch.Tensor, block_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        w_q, w_k, w_v, w_o, ff_w1, ff_w2 = block_params
        q = tensor_join("pd,dhk->phk", inp, w_q)
        k = tensor_join("pd,dhk->phk", inp, w_k)
        v = tensor_join("pd,dhv->phv", inp, w_v)
        scores = tensor_join("phk,qhk->pqh", q, k) / math.sqrt(head_dim)
        weights = torch.softmax(scores, dim=1)
        attn = tensor_join("pqh,qhv->phv", weights, v)
        projected = tensor_join("phv,hvd->pd", attn, w_o)
        residual = inp + projected
        hidden = torch.relu(tensor_join("pd,df->pf", residual, ff_w1))
        ff_out = tensor_join("pf,fd->pd", hidden, ff_w2)
        return residual + ff_out

    def run_block_ref(inp: torch.Tensor, block_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        w_q, w_k, w_v, w_o, ff_w1, ff_w2 = block_params
        q = torch.einsum("pd,dhk->phk", inp, w_q)
        k = torch.einsum("pd,dhk->phk", inp, w_k)
        v = torch.einsum("pd,dhv->phv", inp, w_v)
        scores = torch.einsum("phk,qhk->pqh", q, k) / math.sqrt(head_dim)
        weights = torch.softmax(scores, dim=1)
        attn = torch.einsum("pqh,qhv->phv", weights, v)
        projected = torch.einsum("phv,hvd->pd", attn, w_o)
        residual = inp + projected
        hidden = torch.relu(torch.einsum("pd,df->pf", residual, ff_w1))
        ff_out = torch.einsum("pf,fd->pd", hidden, ff_w2)
        return residual + ff_out

    output = stream
    output_ref = stream
    for block in params:
        output = run_block(output, block)
        output_ref = run_block_ref(output_ref, block)

    assert torch.allclose(output, output_ref, atol=1e-6, rtol=1e-5)


def test_tensor_join_transformer_block_with_norm_and_dropout(device: torch.device) -> None:
    seq_len, model_dim, num_heads, head_dim, value_dim = 5, 12, 3, 4, 4
    ff_hidden = 16
    dropout_p = 0.25

    stream = torch.randn(seq_len, model_dim, device=device)

    w_q = torch.randn(model_dim, num_heads, head_dim, device=device)
    w_k = torch.randn(model_dim, num_heads, head_dim, device=device)
    w_v = torch.randn(model_dim, num_heads, value_dim, device=device)
    w_o = torch.randn(num_heads, value_dim, model_dim, device=device)
    ff_w1 = torch.randn(model_dim, ff_hidden, device=device)
    ff_w2 = torch.randn(ff_hidden, model_dim, device=device)

    ln1_weight = torch.randn(model_dim, device=device)
    ln1_bias = torch.randn(model_dim, device=device)
    ln2_weight = torch.randn(model_dim, device=device)
    ln2_bias = torch.randn(model_dim, device=device)

    # Tensor-logic style computation
    normed_stream = _layer_norm(stream, ln1_weight, ln1_bias)
    q = tensor_join("pd,dhk->phk", normed_stream, w_q)
    k = tensor_join("pd,dhk->phk", normed_stream, w_k)
    v = tensor_join("pd,dhv->phv", normed_stream, w_v)
    scores = tensor_join("phk,qhk->pqh", q, k) / math.sqrt(head_dim)
    attn_weights = torch.softmax(scores, dim=1)
    attn = tensor_join("pqh,qhv->phv", attn_weights, v)
    projected = tensor_join("phv,hvd->pd", attn, w_o)
    torch.manual_seed(42)
    dropped_proj = F.dropout(projected, p=dropout_p, training=True)
    stream_residual = stream + dropped_proj

    normed_residual = _layer_norm(stream_residual, ln2_weight, ln2_bias)
    ff_hidden_act = torch.relu(tensor_join("pd,df->pf", normed_residual, ff_w1))
    ff_output = tensor_join("pf,fd->pd", ff_hidden_act, ff_w2)
    torch.manual_seed(42)
    dropped_ff = F.dropout(ff_output, p=dropout_p, training=True)
    final_stream = stream_residual + dropped_ff

    # Reference einsum path
    normed_stream_ref = _layer_norm(stream, ln1_weight, ln1_bias)
    q_ref = torch.einsum("pd,dhk->phk", normed_stream_ref, w_q)
    k_ref = torch.einsum("pd,dhk->phk", normed_stream_ref, w_k)
    v_ref = torch.einsum("pd,dhv->phv", normed_stream_ref, w_v)
    scores_ref = torch.einsum("phk,qhk->pqh", q_ref, k_ref) / math.sqrt(head_dim)
    attn_weights_ref = torch.softmax(scores_ref, dim=1)
    attn_ref = torch.einsum("pqh,qhv->phv", attn_weights_ref, v_ref)
    projected_ref = torch.einsum("phv,hvd->pd", attn_ref, w_o)
    torch.manual_seed(42)
    dropped_proj_ref = F.dropout(projected_ref, p=dropout_p, training=True)
    stream_residual_ref = stream + dropped_proj_ref

    normed_residual_ref = _layer_norm(stream_residual_ref, ln2_weight, ln2_bias)
    ff_hidden_ref = torch.relu(torch.einsum("pd,df->pf", normed_residual_ref, ff_w1))
    ff_output_ref = torch.einsum("pf,fd->pd", ff_hidden_ref, ff_w2)
    torch.manual_seed(42)
    dropped_ff_ref = F.dropout(ff_output_ref, p=dropout_p, training=True)
    final_stream_ref = stream_residual_ref + dropped_ff_ref

    assert torch.allclose(q, q_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(k, k_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(v, v_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(attn_weights, attn_weights_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(projected, projected_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stream_residual, stream_residual_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ff_hidden_act, ff_hidden_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(dropped_ff, dropped_ff_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(final_stream, final_stream_ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("source,target,query,expected", ANALOGY_CASES)
def test_tensor_join_object_reasoning_analogies(
    source: str, target: str, query: str, expected: str, device: torch.device
) -> None:
    names, embeddings = _build_reasoning_vocab(device)
    name_to_index = {name: idx for idx, name in enumerate(names)}

    analog_vector = (
        embeddings[name_to_index[source]]
        - embeddings[name_to_index[target]]
        + embeddings[name_to_index[query]]
    )
    expected_vector = embeddings[name_to_index[expected]]

    assert torch.allclose(analog_vector, expected_vector, atol=1e-6, rtol=1e-6)

    similarities = F.cosine_similarity(analog_vector.unsqueeze(0), embeddings, dim=1)
    best_idx = torch.argmax(similarities).item()
    assert names[best_idx] == expected
    assert similarities[best_idx] > 0.99


def test_tensor_join_reasoning_temperature_controls_confidence(device: torch.device) -> None:
    names, embeddings = _build_reasoning_vocab(device)
    name_to_index = {name: idx for idx, name in enumerate(names)}

    base = embeddings[name_to_index["metal_blue_box"]]
    reference = embeddings[name_to_index["metal_box"]]
    query = embeddings[name_to_index["metal_circle"]]
    target_idx = name_to_index["metal_blue_circle"]

    analogy = base - reference + query
    logits = F.cosine_similarity(analogy.unsqueeze(0), embeddings, dim=1)

    low_temp = 0.25
    high_temp = 5.0
    probs_low = torch.softmax(logits / low_temp, dim=0)
    probs_high = torch.softmax(logits / high_temp, dim=0)

    assert torch.argmax(probs_low).item() == target_idx
    assert torch.argmax(probs_high).item() == target_idx
    assert probs_low[target_idx] > probs_high[target_idx]

    entropy_low = -(probs_low * probs_low.clamp_min(1e-9).log()).sum()
    entropy_high = -(probs_high * probs_high.clamp_min(1e-9).log()).sum()
    assert entropy_high > entropy_low
