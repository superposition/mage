import math

import torch

from mage.tensor_logic import TensorJoinPlan, tensor_join


def test_tensor_join_cosine_kernel_attention(device: torch.device) -> None:
    """Simulate attention with a cosine kernel and ensure tensor_join matches einsum."""

    torch.manual_seed(0)
    num_queries, num_keys, dim, value_dim = 4, 5, 6, 3

    queries = torch.randn(num_queries, dim, device=device)
    keys = torch.randn(num_keys, dim, device=device)
    values = torch.randn(num_keys, value_dim, device=device)

    # Cosine kernel construction mirrors normalized dot products in attention systems.
    q_norm = torch.nn.functional.normalize(queries, dim=-1)
    k_norm = torch.nn.functional.normalize(keys, dim=-1)

    plan = TensorJoinPlan.from_equation("qd,kd->qk", (q_norm, k_norm))
    affinity = tensor_join("qd,kd->qk", q_norm, k_norm, plan=plan)
    expected_affinity = torch.einsum("qd,kd->qk", q_norm, k_norm)
    assert torch.allclose(affinity, expected_affinity, atol=1e-6, rtol=1e-5)

    # Softmax weights over the affinity emulate standard scaled-dot attention.
    scaled = affinity / math.sqrt(dim)
    weights = torch.softmax(scaled, dim=-1)

    # Weighted value readout should line up with einsum-based attention.
    output = tensor_join("qk,kv->qv", weights, values)
    expected_output = torch.einsum("qk,kv->qv", weights, values)
    assert torch.allclose(output, expected_output, atol=1e-6, rtol=1e-5)


def test_tensor_join_rbf_kernel_regression(device: torch.device) -> None:
    """Build the Nadaraya-Watson/RBF estimator and cross-check every intermediate contraction."""

    torch.manual_seed(1)
    num_points, num_queries, dim, value_dim = 6, 3, 5, 2
    sigma = 0.7

    keys = torch.randn(num_points, dim, device=device)
    queries = torch.randn(num_queries, dim, device=device)
    targets = torch.randn(num_points, value_dim, device=device)

    # Squared-norm helpers let us expand ||x - y||^2 via tensor_join contractions.
    key_sq = tensor_join("id,id->i", keys, keys)
    query_sq = tensor_join("qd,qd->q", queries, queries)
    cross = tensor_join("qd,id->qi", queries, keys)

    # Gaussian kernel evaluated from the expanded distance matrix.
    dist = query_sq.unsqueeze(-1) + key_sq.unsqueeze(0) - 2.0 * cross
    kernel = torch.exp(-dist / (2.0 * sigma ** 2))

    # Kernel-weighted targets produce the regression prediction vector.
    predictions = tensor_join("qi,iv->qv", kernel, targets)

    # Mirror each stage with explicit einsum/elementwise computations for clarity.
    expected_key_sq = (keys ** 2).sum(dim=-1)
    expected_query_sq = (queries ** 2).sum(dim=-1)
    expected_cross = torch.einsum("qd,id->qi", queries, keys)
    expected_dist = expected_query_sq.unsqueeze(-1) + expected_key_sq.unsqueeze(0) - 2.0 * expected_cross
    expected_kernel = torch.exp(-expected_dist / (2.0 * sigma ** 2))
    expected_predictions = torch.einsum("qi,iv->qv", expected_kernel, targets)

    assert torch.allclose(key_sq, expected_key_sq, atol=1e-6, rtol=1e-5)
    assert torch.allclose(query_sq, expected_query_sq, atol=1e-6, rtol=1e-5)
    assert torch.allclose(cross, expected_cross, atol=1e-6, rtol=1e-5)
    assert torch.allclose(kernel, expected_kernel, atol=1e-6, rtol=1e-5)
    assert torch.allclose(predictions, expected_predictions, atol=1e-6, rtol=1e-5)


def test_tensor_join_projection_marginalizes_axes(device: torch.device) -> None:
    """Projection acts as marginalization: summing out the specified axes should match reshape/sum logic."""

    torch.manual_seed(2)
    tensor = torch.randn(3, 4, 5, device=device)

    # Project over the third axis (einstein sum collapses 'c'), yielding a 3x4 marginal.
    projected = tensor_join("abc->ab", tensor)
    expected = tensor.sum(dim=2)
    assert torch.allclose(projected, expected, atol=1e-6, rtol=1e-5)


def test_tensor_join_selective_projection_keeps_axes(device: torch.device) -> None:
    """Selective projection keeps chosen axes via einsum output labels; verifies we can aggregate conditionally."""

    torch.manual_seed(3)
    tensor = torch.randn(2, 3, 4, device=device)

    # Keep axes 'a' and 'c' by projecting out 'b'.
    projected = tensor_join("abc->ac", tensor)
    expected = tensor.sum(dim=1)
    assert torch.allclose(projected, expected, atol=1e-6, rtol=1e-5)


def test_tensor_join_projection_with_conditioning(device: torch.device) -> None:
    """Marginalize conditioned tensors: join with a mask and project to compute conditional probabilities."""

    torch.manual_seed(4)
    tensor = torch.randn(2, 3, 4, device=device).abs()
    mask = torch.tensor([True, False, True], device=device)

    # Apply mask (selective projection) before marginalizing axis 'b'.
    masked = tensor * mask.unsqueeze(0).unsqueeze(-1)
    projected = tensor_join("abc->ac", masked)
    expected = masked.sum(dim=1)
    assert torch.allclose(projected, expected, atol=1e-6, rtol=1e-5)


def test_tensor_join_embedding_superposition_membership(device: torch.device) -> None:
    """Superpose random unit embeddings and recover membership via dot products, Bloom-filter style."""

    torch.manual_seed(5)
    num_objects, embed_dim = 10, 512
    embeddings = torch.nn.functional.normalize(torch.randn(num_objects, embed_dim, device=device), dim=-1)

    # Encode a set with a multi-hot selector; tensor_join swings embeddings into the superposed vector.
    members = torch.tensor([1, 3, 7], device=device)
    selector = torch.zeros(num_objects, device=device)
    selector[members] = 1.0
    superposed = tensor_join("x,xd->d", selector, embeddings)

    # Scores for present objects hover near 1, absent objects near 0 with high-dimensional embeddings.
    scores = tensor_join("d,xd->x", superposed, embeddings)
    member_scores = scores[members]
    nonmember_scores = scores[torch.tensor([i for i in range(num_objects) if i not in members], device=device)]

    assert torch.all(member_scores > 0.6)
    assert torch.all(nonmember_scores < 0.4)


def test_tensor_join_relation_superposition(device: torch.device) -> None:
    """Tensor product embeddings of relations reconstruct adjacency when projected back."""

    torch.manual_seed(6)
    num_objects, embed_dim = 6, 128
    embeddings = torch.nn.functional.normalize(torch.randn(num_objects, embed_dim, device=device), dim=-1)

    relation_pairs = [(0, 1), (2, 3), (4, 5), (1, 4)]
    relation = torch.zeros(num_objects, num_objects, device=device)
    for x, y in relation_pairs:
        relation[x, y] = 1.0

    # Build superposed relation embedding via tensor products.
    emb_relation = tensor_join("xy,xi,yj->ij", relation, embeddings, embeddings)

    # Project back to an adjacency estimate by contracting with the basis embeddings again.
    reconstructed = tensor_join("ij,xi,yj->xy", emb_relation, embeddings, embeddings)

    for x, y in relation_pairs:
        assert reconstructed[x, y] > 0.5

    non_pairs = [(0, 2), (3, 3), (5, 0)]
    for x, y in non_pairs:
        assert reconstructed[x, y] < 0.5


def test_tensor_join_relation_superposition_tensor_decomposition(device: torch.device) -> None:
    """Confirm EmbR with factor matrices Emb constitutes a Tucker decomposition of the relation."""

    torch.manual_seed(7)
    num_objects, embed_dim = 5, 64
    embeddings = torch.eye(embed_dim, device=device)[:num_objects]

    relation = torch.zeros(num_objects, num_objects, device=device)
    relation[0, 2] = 1.0
    relation[1, 3] = 1.0
    relation[4, 0] = 1.0

    core = tensor_join("xy,xi,yj->ij", relation, embeddings, embeddings)
    reconstructed = tensor_join("ij,xi,yj->xy", core, embeddings, embeddings)

    assert torch.allclose(reconstructed, relation, atol=1e-3, rtol=1e-3)


def test_tensor_join_rule_embedding_forward_chain(device: torch.device) -> None:
    """Embed a simple rule and perform forward chaining: Cons(a,c) <- R(a,b) & S(b,c)."""

    torch.manual_seed(8)
    objects, embed_dim = 6, 64
    embeddings = torch.nn.functional.normalize(torch.randn(objects, embed_dim, device=device), dim=-1)

    rel_r = torch.zeros(objects, objects, device=device)
    rel_s = torch.zeros(objects, objects, device=device)
    rel_r[0, 1] = 1.0
    rel_r[1, 2] = 1.0
    rel_s[1, 3] = 1.0
    rel_s[2, 4] = 1.0

    emb_r = tensor_join("xy,xi,yj->ij", rel_r, embeddings, embeddings)
    emb_s = tensor_join("xy,xi,yj->ij", rel_s, embeddings, embeddings)

    inferred_core = tensor_join("ij,jk->ik", emb_r, emb_s)

    inferred_relation = tensor_join("ik,xi,yk->xy", inferred_core, embeddings, embeddings)

    assert inferred_relation[0, 3] > 0.5
    assert inferred_relation[1, 4] > 0.5
    assert inferred_relation[0, 4] < 0.3


def test_tensor_join_similarity_temperature_control(device: torch.device) -> None:
    """Softmax temperature modulates analogical borrowing across embeddings."""

    torch.manual_seed(9)
    objects = 5
    embeddings = torch.eye(objects, device=device)

    sim_matrix = tensor_join("xd,yd->xy", embeddings, embeddings)

    low_temp = 0.05
    high_temp = 0.8
    attn_low = torch.softmax(sim_matrix / low_temp, dim=-1)
    attn_high = torch.softmax(sim_matrix / high_temp, dim=-1)

    diag_low = torch.diagonal(attn_low)
    diag_high = torch.diagonal(attn_high)

    entropy_low = -(attn_low * attn_low.clamp_min(1e-9).log()).sum(dim=-1)
    entropy_high = -(attn_high * attn_high.clamp_min(1e-9).log()).sum(dim=-1)

    assert torch.all(diag_low > diag_high)
    assert torch.all(entropy_low < entropy_high)
