import torch

from explainability.geo_pgexplainer import GeoPGExplainer, GeoPGExplainerConfig


EDGE_TYPE = ("user", "follows", "user")


def _toy_problem():
    node_embeddings = {
        "user": torch.tensor([
            [0.10, 0.00],
            [0.25, 0.05],
            [0.55, 0.10],
            [0.80, 0.05],
        ])
    }
    edge_index = {
        EDGE_TYPE: torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    }
    base = node_embeddings["user"]

    def model_forward(mask_dict):
        mask = mask_dict[EDGE_TYPE]
        src, dst = edge_index[EDGE_TYPE]
        messages = torch.zeros_like(base)
        messages.index_add_(0, dst, base[src] * mask[:, None])
        degree = torch.zeros(base.shape[0])
        degree.index_add_(0, dst, mask)
        explained = base + 0.3 * messages / degree.clamp_min(1.0)[:, None]
        explained = explained / (1.0 + torch.linalg.vector_norm(
            explained, dim=-1, keepdim=True
        ))
        prediction = explained[:, 0].mean().reshape(1)
        return prediction, explained

    return node_embeddings, edge_index, model_forward


def test_zero_geometry_coefficients_select_pgexplainer_mode():
    node_embeddings, edge_index, model_forward = _toy_problem()
    explainer = GeoPGExplainer(GeoPGExplainerConfig(epochs=3, hidden_dim=8))
    result = explainer.fit(node_embeddings, edge_index, model_forward)
    assert result.mode == "pgexplainer"
    assert EDGE_TYPE in result.edge_masks
    assert result.edge_masks[EDGE_TYPE].shape == (4,)
    assert 0.0 <= result.metrics["radial_order_agreement"] <= 1.0


def test_positive_geometry_coefficients_select_geo_mode():
    node_embeddings, edge_index, model_forward = _toy_problem()
    explainer = GeoPGExplainer(GeoPGExplainerConfig(
        epochs=3,
        hidden_dim=8,
        geodesic_coefficient=1.0,
        radial_coefficient=1.0,
    ))
    result = explainer.fit(node_embeddings, edge_index, model_forward)
    assert result.mode == "geo_pgexplainer"
    assert "geodesic" in result.losses
    assert "radial" in result.losses


def test_relation_specific_masks_are_supported():
    node_embeddings, edge_index, model_forward = _toy_problem()
    second_type = ("user", "similar", "user")
    edge_index[second_type] = torch.tensor([[0, 1], [2, 3]])

    def two_relation_forward(mask_dict):
        prediction, embedding = model_forward({EDGE_TYPE: mask_dict[EDGE_TYPE]})
        adjustment = mask_dict[second_type].mean() * 0.01
        return prediction + adjustment, embedding

    result = GeoPGExplainer(GeoPGExplainerConfig(epochs=2, hidden_dim=8)).fit(
        node_embeddings, edge_index, two_relation_forward
    )
    assert set(result.edge_masks) == {EDGE_TYPE, second_type}
