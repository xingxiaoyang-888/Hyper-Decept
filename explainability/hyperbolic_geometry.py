"""Differentiable Poincare-ball geometry helpers for white-box explanations."""

from __future__ import annotations

import torch


def project_to_ball(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Project points to the open unit Poincare ball without changing direction."""
    max_norm = 1.0 - eps
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def poincare_distance(
    u: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute pairwise Poincare distance for equally shaped point tensors."""
    u = project_to_ball(u, eps=eps)
    v = project_to_ball(v, eps=eps)
    u_sq = torch.sum(u * u, dim=-1)
    v_sq = torch.sum(v * v, dim=-1)
    diff_sq = torch.sum((u - v) ** 2, dim=-1)
    denominator = ((1.0 - u_sq) * (1.0 - v_sq)).clamp_min(eps)
    argument = 1.0 + 2.0 * diff_sq / denominator
    return torch.acosh(argument.clamp_min(1.0 + eps))


def poincare_radius(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Return geodesic distance from each point to the origin."""
    x = project_to_ball(x, eps=eps)
    norm = torch.linalg.vector_norm(x, dim=-1).clamp(max=1.0 - eps)
    return 2.0 * torch.atanh(norm)


def normalized_distance_distortion(
    reference: torch.Tensor,
    explained: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean relative distortion between aligned distance vectors."""
    if reference.shape != explained.shape:
        raise ValueError("reference and explained distances must have equal shape")
    if reference.numel() == 0:
        return reference.new_tensor(0.0)
    return torch.mean(torch.abs(explained - reference) / reference.abs().clamp_min(eps))


def radial_order_loss(
    reference_radius: torch.Tensor,
    explained_radius: torch.Tensor,
    margin: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize inversions of the reference center-to-periphery ordering.

    Pairwise comparisons whose reference radii are effectively tied are ignored.
    The returned hinge loss is differentiable with respect to ``explained_radius``.
    """
    if reference_radius.shape != explained_radius.shape:
        raise ValueError("reference and explained radii must have equal shape")
    if reference_radius.numel() < 2:
        return reference_radius.new_tensor(0.0)

    ref_delta = reference_radius[:, None] - reference_radius[None, :]
    exp_delta = explained_radius[:, None] - explained_radius[None, :]
    valid = torch.triu(ref_delta.abs() > eps, diagonal=1)
    if not torch.any(valid):
        return reference_radius.new_tensor(0.0)

    expected_sign = torch.sign(ref_delta[valid])
    signed_delta = expected_sign * exp_delta[valid]
    return torch.relu(float(margin) - signed_delta).mean()


def radial_order_agreement(
    reference_radius: torch.Tensor,
    explained_radius: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Fraction of non-tied node pairs retaining their radial ordering."""
    if reference_radius.shape != explained_radius.shape:
        raise ValueError("reference and explained radii must have equal shape")
    if reference_radius.numel() < 2:
        return reference_radius.new_tensor(1.0)

    ref_delta = reference_radius[:, None] - reference_radius[None, :]
    exp_delta = explained_radius[:, None] - explained_radius[None, :]
    valid = torch.triu(ref_delta.abs() > eps, diagonal=1)
    if not torch.any(valid):
        return reference_radius.new_tensor(1.0)
    agreement = torch.sign(ref_delta[valid]) == torch.sign(exp_delta[valid])
    return agreement.to(reference_radius.dtype).mean()
