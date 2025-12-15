import torch

from mage.tensor_logic import tensor_join


def _orthonormal_basis(dim: int, subspace_dim: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    matrix = torch.randn(dim, subspace_dim, device=device, generator=generator)
    # QR with reduced mode gives the orthonormal columns we need for the subspace basis.
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def _projection_matrix(basis: torch.Tensor) -> torch.Tensor:
    # Projector onto the span of the basis columns: P = B B^T.
    return tensor_join("dk,fk->df", basis, basis)


def test_tensor_join_projection_matrix_idempotent(device: torch.device) -> None:
    basis = _orthonormal_basis(dim=8, subspace_dim=3, device=device)
    projector = _projection_matrix(basis)

    squared = tensor_join("df,fg->dg", projector, projector)
    assert torch.allclose(squared, projector, atol=1e-6, rtol=1e-5)


def test_tensor_join_projection_preserves_subspace_vectors(device: torch.device) -> None:
    basis = _orthonormal_basis(dim=10, subspace_dim=4, device=device)
    projector = _projection_matrix(basis)

    coeffs = torch.randn(basis.shape[1], device=device)
    vector_in_subspace = tensor_join("dk,k->d", basis, coeffs)

    projected = tensor_join("df,f->d", projector, vector_in_subspace)
    assert torch.allclose(projected, vector_in_subspace, atol=1e-6, rtol=1e-5)


def test_tensor_join_projection_annihilates_orthogonal_components(device: torch.device) -> None:
    basis = _orthonormal_basis(dim=7, subspace_dim=3, device=device)
    projector = _projection_matrix(basis)

    vector = torch.randn(7, device=device)
    projected = tensor_join("df,f->d", projector, vector)
    residual = vector - projected

    # Coordinates of the residual in the subspace basis should vanish.
    residual_coords = tensor_join("d,dk->k", residual, basis)
    assert torch.allclose(residual_coords, torch.zeros_like(residual_coords), atol=1e-6, rtol=1e-5)
