import torch

from explainability.hyperbolic_geometry import (
    normalized_distance_distortion,
    poincare_distance,
    poincare_radius,
    radial_order_agreement,
    radial_order_loss,
)


def test_poincare_distance_is_symmetric_and_zero_on_diagonal():
    u = torch.tensor([[0.1, 0.2], [0.4, -0.1]])
    v = torch.tensor([[0.2, 0.1], [-0.2, 0.3]])
    assert torch.allclose(poincare_distance(u, v), poincare_distance(v, u))
    assert torch.all(poincare_distance(u, u) < 0.01)


def test_poincare_radius_increases_toward_boundary():
    points = torch.tensor([[0.1, 0.0], [0.5, 0.0], [0.9, 0.0]])
    radii = poincare_radius(points)
    assert torch.all(radii[1:] > radii[:-1])


def test_distance_distortion_is_zero_for_equal_vectors():
    values = torch.tensor([0.5, 1.0, 2.0])
    assert normalized_distance_distortion(values, values).item() == 0.0


def test_radial_order_detects_inversion():
    reference = torch.tensor([0.1, 0.5, 0.9])
    retained = torch.tensor([0.2, 0.4, 0.8])
    inverted = torch.tensor([0.9, 0.5, 0.1])
    assert radial_order_loss(reference, retained).item() == 0.0
    assert radial_order_agreement(reference, retained).item() == 1.0
    assert radial_order_loss(reference, inverted).item() > 0.0
    assert radial_order_agreement(reference, inverted).item() == 0.0


def test_geometry_losses_are_differentiable():
    reference = torch.tensor([0.1, 0.5, 0.9])
    explained = torch.tensor([0.8, 0.4, 0.2], requires_grad=True)
    loss = radial_order_loss(reference, explained)
    loss.backward()
    assert explained.grad is not None
